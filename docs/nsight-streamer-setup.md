# Nsight Systems in-browser viewer on Brev — setup, findings, open issues

**Status:** infrastructure wired up and verified; **blocked** on access to the
NVIDIA Nsight Streamer container image (private on NGC). Browser-side streaming
test still pending. See [Open issues](#open-issues--what-still-needs-deciding).

This documents the work to make the `jupyterlab-nvidia-nsight` extension's
in-browser report viewer work on the single-node Brev cloud instances used for
the Profiling AI Software Bootcamp.

---

## How the extension actually works

`jupyterlab-nvidia-nsight` does **not** render `.nsys-rep` / `.ncu-rep` reports
inside JupyterLab. When you open a report it:

1. Uses the **Docker Python SDK** (inside the Jupyter container) to talk to the
   host Docker daemon and spawn a **separate "Nsight Streamer" container**
   (`nvcr.io/nvidia/devtools/nsight-streamer-systems:<tag>`).
2. That sibling container runs the real Nsight Systems GUI and streams it to the
   browser over WebRTC (selkies-gstreamer).
3. The report file is copied into the streamer container over the Docker API
   (not bind-mounted), so host-path mapping is a non-issue.

Consequences: the Jupyter container must (a) reach the Docker daemon, and
(b) tell the browser a reachable address/URL for the streamed UI.

### There is no "Nsight" menu bar item

v1.0.0 does **not** add a top-level menu. Its UI surfaces are:
- a profiling toggle **button in the notebook toolbar**, right after Run (▶);
- **command-palette** entries under "NVIDIA Nsight" (Cmd/Ctrl+Shift+C);
- right-click **context-menu** entries on report tabs / report files;
- the `.nsys-rep` / `.ncu-rep` file openers.

So "no Nsight menu" is expected, not a bug.

---

## The original error

> Failed to start docker client. Is the docker socket mounted? …

Root cause: the Jupyter container did not mount `/var/run/docker.sock`, so
`docker.DockerClient()` could not reach the daemon. Fixed by mounting the socket
(see compose changes below). Jupyter runs as root in the container
(`--allow-root`), so there is no socket-permission issue once it is mounted.

---

## The Brev constraint

Attendees reach JupyterLab at `https://notebooks-<id>.brevlab.com/lab`. This is
fronted by **Cloudflare Access** (the URL 302-redirects to
`brevlab.cloudflareaccess.com`). Key implications:

- We **do not control** the reverse proxy, so the official NVIDIA "nginx
  location block" recipe for exposing the streamer is not available to us.
- Only the **single JupyterLab port** is exposed at that hostname. The streamer's
  dynamically-chosen TCP ports are not reachable from the outside, so the
  "pin ports + open the firewall" approach does **not** apply here.
- Cloudflare proxies HTTP/WebSocket fine but will **not** pass raw WebRTC/UDP
  media. → streaming viability hinges on whether the streamer carries media over
  a **WebSocket data channel** (works through Cloudflare) vs **raw WebRTC**
  (won't). This is the remaining unknown that needs a browser test.

---

## The approach implemented on this branch

Route the streamer **through the JupyterLab URL Brev already exposes** — no extra
ports, no hardcoded IPs — using `jupyter-server-proxy`.

The extension builds the viewer URL as:

```
<JupyterLab base URL>/<proxyDomain>/<port>/
```

We set `proxyDomain = proxy` so this becomes `…/proxy/<port>/`, which is exactly
the route `jupyter-server-proxy` serves. The frontend confirms this path:
`insideDocker ? a = dockerHost || window.location.hostname : …` and
`url = proxyDomain ? `${baseUrl}/${proxyDomain}/${port}/` : …`. Because
`dockerHost` is left empty, when proxyDomain is *not* used the address would fall
back to `window.location.hostname` automatically — so nothing is instance-specific.

`jupyter-server-proxy`'s `/proxy/<port>/` route always targets `127.0.0.1:<port>`.
The streamer's ports are published on the **host**, so we run the Jupyter
container with **`network_mode: host`** so that the container's `127.0.0.1` is the
host — letting `/proxy/<port>/` reach the streamer. JupyterLab still listens on
host `:8888` for Brev. (With host networking, an explicit compose `ports:` block
is invalid and was removed.)

### Compose / config changes (this branch)

`docker-compose.yaml`:
- mount `/var/run/docker.sock` → lets the extension reach the Docker daemon;
- `network_mode: host` (removed the `ports:` block);
- mount `config/nsight-overrides.json` → JupyterLab settings overrides;
- add `jupyter-server-proxy` to the startup `pip install`.

`config/nsight-overrides.json` (baked JupyterLab setting, keyed by the plugin id
`jupyterlab-nvidia-nsight:plugin`):
- `ui.proxyDomain = "proxy"`, `ui.enabled = true`, `ui.suppressServerAddressWarning = true`;
- everything else left at schema defaults; **no IPs**, so it is portable across instances.

---

## What is verified working

On a live instance with these changes:

- ✅ Docker socket reachable from the container via the Python SDK (original
  error gone): `docker.DockerClient().version()` succeeds.
- ✅ `jupyter-server-proxy` 4.5.0 installed and enabled; `jupyterlab-nvidia-nsight`
  enabled; `overrides.json` parsed with `proxyDomain = proxy`.
- ✅ **Brev URL still returns 200 under host networking** — access not broken.
- ✅ **Proxy plumbing proven end-to-end**: served a dummy HTTP server on a host
  port and fetched it through `…:8888/proxy/<port>/` successfully — the exact
  path the streamer will use.

The only thing *not* yet exercised is the streamer itself (blocked below) and the
browser-side WebRTC/WebSocket media test.

---

## Open issues / what still needs deciding

### 1. The Nsight Streamer image is private on NGC (hard blocker)

`nvcr.io/nvidia/devtools/nsight-streamer-systems` is **not anonymously
accessible**:

| Check (anonymous) | `nvidia/pytorch` (public) | `…/nsight-streamer-systems` |
|---|---|---|
| NGC catalog API (`api.ngc.nvidia.com/v2/repos/…`) | returns `latestTag` | **404 NOT_FOUND** |
| Registry pull token (`nvcr.io/proxy_auth`) | token issued | **DENIED / Access Denied** |

Two distinct problems result:
- The image **cannot be pulled** without NGC credentials entitled to it
  (`docker login nvcr.io` with an NGC API key whose org has access).
- The extension resolves the image tag with an **unauthenticated** call
  (`urllib.urlopen(.../v2/repos/nvidia/devtools/nsight-streamer-systems)` →
  reads `latestTag`). For a private repo this **404s**, so the viewer fails at
  version resolution **even if the image is already pulled locally**.

**Needs an NVIDIA/NGC decision**, e.g. one of:
- confirm whether this repo is meant to be public (and get it made public); or
- provision an NGC API key with entitlement onto the instances *and* address the
  anonymous version-lookup 404 (e.g. a pinned tag + a patched/forked extension,
  or a local stand-in for the catalog endpoint); or
- accept that the in-browser viewer is not available for this audience and rely
  on the fallback below.

### 2. WebRTC media transport through Cloudflare (needs a browser test)

Even with the image resolved, confirm the GUI actually displays through Brev:
open a report and watch for **"Waiting for stream"**. If it streams, media is
going over WebSocket and we are done. If it hangs on "Waiting for stream", media
needs raw WebRTC and a **TURN server** would be required (`ui.turnHost` /
`turnPort` / `turnUsername` / `turnPassword`) — unrealistic to stand up per
instance for the workshop.

---

## Recommended fallback for the workshop (works today, no NGC dependency)

The extension's **profiling** features (generating reports from notebook cells /
the toolbar button / `nsys` from a cell) work regardless — only *viewing*
in-browser needs the streamer. So for attendees:

- View timelines by **downloading the `.nsys-rep` and opening it in a local
  Nsight Systems GUI** (free: <https://developer.nvidia.com/nsight-systems>). The
  intro notebooks already link reports this way.
- Use `!nsys stats <report>.nsys-rep` for a quick in-notebook statistical summary.
- Use the existing **export workflows** the intro notebooks teach (text reports,
  Perfetto traces → ui.perfetto.dev, profiler HTML / flame graphs).

---

## How to test a fresh deploy

1. Deploy a Brev instance from this branch.
2. Confirm JupyterLab loads at the brevlab URL.
3. Run a notebook cell that produces a report (or use an existing
   `workspace/reports/*.nsys-rep`).
4. Click the report / right-click → open with Nsight UI.
   - "Failed to start docker client" → socket mount didn't take.
   - Error mentioning the image / `latestTag` / 404 → the NGC access blocker (#1).
   - Tab opens but stuck on "Waiting for stream" → media/TURN issue (#2).
   - GUI renders → success; merge to main.

## How to revert (if abandoning the streamer for tomorrow)

In `docker-compose.yaml`: drop `network_mode: host`, restore the `ports:` block,
remove the `docker.sock` and `overrides.json` mounts, and drop
`jupyter-server-proxy` from the pip install. (Keeping the socket mount alone is
harmless and re-enables the viewer on any non-proxied / direct-IP instance.)
