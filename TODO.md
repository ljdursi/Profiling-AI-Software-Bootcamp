# Workshop TODO List

## Tomorrow's Priority Tasks

### 1. New Introductory Notebooks: PyTorch Profiler → Nsight Systems via a Two-Bug Demo
**Branch**: `introductory-notebooks`

- **Pedagogical goal**: Two short notebooks at the *start* of the workshop that teach when to reach for each tool by solving two bugs in one training script:
  - **Notebook 1 — PyTorch Profiler**: A `DataLoader` stall (caused by `num_workers=0, pin_memory=False`). The profiler's step view shows the `DataLoader` section dominating each step; fix is 3 lines. Pedagogical message: "profiler told you the section, you fix the section."
  - **Notebook 2 — Nsight Systems**: H2D copies not overlapping with compute (caused by `.to(device)` without `non_blocking=True`). PyTorch Profiler shows time in `Memcpy HtoD` but doesn't make the *serialization* obvious. Nsys' timeline shows a sawtooth where H2D sits between kernels instead of underneath them. Fix: `non_blocking=True`. Pedagogical message: "you need the timeline view to see *concurrency* problems."

- **Training task**: CIFAR-10 + ResNet18 at native 32x32, single GPU, fixed seed, short enough that the profile-fix-reprofile loop is snappy.

- **Script structure** (revised 2026-05-27 — see `workspace/source_code/intro/NOTES.md`): **two training tasks, four files**, in `workspace/source_code/intro/`:
  - **(a) `train_v1.py`** — original training task (CIFAR-10 + ResNet18 32×32). Has Bug 1 (DataLoader stall).
  - **(b) `train_v1_fixed.py`** — Bug 1 fixed (`num_workers=N`, `pin_memory=True`).
  - **(c) `train_v2.py`** — *new, more ambitious training task* enabled by the v1 fix (heavier model / images / mixed precision / etc., TBD). Has Bug 2.
  - **(d) `train_v2_fixed.py`** — Bug 2 fixed.

  Pedagogical arc: fixing the data-loading stall unlocks a more ambitious training problem, which in turn surfaces a new class of bug that needs the timeline view to diagnose. Frees Bug 2 from having to live inside the GPU-bound CIFAR/ResNet18 32×32 workload where CPU-side bugs were shown to be invisible.

- **Substeps (do in order)**:
  - **(a) v1 pair done** (currently named `train_v0_baseline.py` and `train_v1_dataloader_fixed.py`; rename to `train_v1.py` / `train_v1_fixed.py`). Verified on L4: 256.8 → 93.7 ms/step. DataLoader stall reproduces clearly.
  - **(a.1) Design the v2 training task**: pick a more ambitious training scenario (larger images / bigger model / mixed precision / etc.) that the v1 fix makes feasible AND that creates headroom for Bug 2 to be visible.
  - **(a.2) Pick Bug 2** for the v2 task — see task #6. Candidates: leftover `autograd.set_detect_anomaly(True)`, per-step eval pass, unused aux head, cudnn-benchmark issues. With a heavier workload, also re-test `non_blocking` / `.item()` before discarding.
  - **(a.3) Write `train_v2.py` and `train_v2_fixed.py`**; delete the now-misleading `train_v2_fully_fixed.py`.
  - **(a.4) Re-verify** all four scripts show the expected progression on this single L4.
  - **(b) Notebook 1 — `pytorch-profiler-intro.ipynb`**: Run v0 under `torch.profiler.profile()` with `tensorboard_trace_handler`, point to TensorBoard at `:8889`, walk through the section breakdown, derive the fix, rerun on v1 to show the win.
  - **(c) Notebook 2 — `nsight-systems-intro.ipynb`**: Try profiler on v1 — looks fine but GPU is still idle. Run nsys, open the timeline, see the H2D sawtooth, derive the fix, rerun on v2.
  - **(d) Wire into TOC**: Place both ahead of the existing `nsys-introduction.ipynb`. Decide whether `nsys-introduction.ipynb` becomes a deeper-dive or is folded into Notebook 2.
  - **(e) Decide fate of `emit_nvtx()` in `baseline.py`**: defer until after intros are written — its role depends on how Notebook 2 lands.

- **Components reference**:
  - `torch_tb_profiler`: pre-installed; TensorBoard already auto-starts in the container on port 8889.
  - `torch.profiler.profile()` + `tensorboard_trace_handler`: macro-level per-step timing, view in TensorBoard.
  - `nsys profile --pytorch=autograd-nvtx`: nsys flag that auto-emits NVTX ranges for PyTorch ops; no code changes required.

- **Workflow**: PyTorch Profiler (which section is slow?) → Nsight Systems (concurrency, kernel launch, sync issues) → Nsight Compute (kernel internals)

- **Resources**:
  - [PyTorch Profiler with TensorBoard Tutorial](https://docs.pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html)
  - [Automatic NVTX annotations with --pytorch flag](https://the-dsvolk.github.io/ai-perf/ai-infra/Tale_of_two_profilers.html)
  - [Speed Up PyTorch Training with Nsight](https://arikpoz.github.io/posts/2025-05-25-speed-up-pytorch-training-by-3x-with-nvidia-nsight-and-pytorch-2-tricks/)

### 2. Nsight Systems Analysis Recipes (multi-report / cross-rank analysis)
- **Goal**: Introduce the `nsys recipe` system — multi-report statistical analysis that complements opening individual `.nsys-rep` files in the GUI. Cover a built-in recipe, a diagnostic pair, and a custom recipe, all anchored to the existing 4-GPU DDP reports the lab already produces.
- **Why early**: The recipe system is the natural next step after students have learned to read individual nsys timelines. It also gives concrete numbers (overlap %, idle gaps, per-rank stats) to back up qualitative timeline observations.
- **Notebook home**: Extend `nsys-application.ipynb` with a new section at the end. Reuses the `baseline_nvtx.nsys-rep` and `firstOptim.nsys-rep` reports the lab already generates, so the before/after pairing lives in one place.

- **Substeps (do in order)**:
  - **(a) Built-in recipe headline demo — `nccl_gpu_overlap_trace`**: Run the recipe on both `baseline_nvtx.nsys-rep` and `firstOptim.nsys-rep`. Compare the resulting communication-vs-compute overlap percentages per rank. Directly quantifies the DDP optimization the lab teaches qualitatively ("overlap went from X% to Y%"). One CLI invocation per report; the recipe emits CSV + a Jupyter notebook in an `.nsys-analysis` bundle. Decide whether to show the auto-generated notebook inline or pull the CSV into our own cells.
  - **(b) Diagnostic recipe pair — `gpu_gaps` + `cuda_gpu_kern_pace`**: `gpu_gaps` (default threshold 500ms) flags idle periods — surfaces data-loader stalls and sync waits. `cuda_gpu_kern_pace` shows whether kernel cadence is consistent across all 4 ranks; divergence indicates stragglers. Run both on `firstOptim.nsys-rep` (the more interesting one) feeding all 4 GPU traces together. Demonstrates the multi-report value-add that's hard to see in a single timeline.
  - **(c) Custom recipe — per-rank time in NVTX ranges**: Write a bespoke recipe that extracts the manual NVTX ranges already in `ddp_optimize.py` (`"Train"`, `"Data loading"`, `"Copy to device"`, `"Forward pass"`, `"Backward pass"`) and produces a per-rank summary (stacked bar or table). Will likely require copying a built-in recipe (e.g., `nccl_sum.py`) as a template since the public docs are light on the user-defined-recipe API. Side benefit: ties student-written NVTX instrumentation from earlier labs into programmatic downstream analysis.

- **Operational notes**:
  - Reports already exist at `workspace/reports/baseline_nvtx.nsys-rep` and `workspace/reports/firstOptim.nsys-rep` (regenerated whenever the user re-runs `nsys-application.ipynb`).
  - `nsys recipe` is available in the compose container alongside `nsys` itself (`nvcr.io/nvidia/pytorch:26.02-py3`); no new installs required.
  - Built-in recipes live at `<nsys-target-dir>/python/packages/nsys-recipe/recipes/` inside the container — locate via `nsys --version` then `find` from `/usr/local/cuda`. Read one before writing (c).
  - Output `.nsys-analysis` bundles include both raw CSV/Parquet and a generated Jupyter notebook; we can either embed those notebooks or load the CSVs into our own cells.

- **Resources**:
  - [Nsight Systems Analysis Guide](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)
  - "Available Advanced Analysis Recipes" section of the above for the recipe catalog
  - "Tutorial: Create a User-Defined Recipe" section (referenced but light) — fall back to reading built-in recipes as templates

### 3. Add JupyterLab Nsight Extension
- **Goal**: Enable in-browser Nsight Systems profiling
- **Resource**: https://developer.nvidia.com/tools-overview/nsight-jupyterlab
- **Why**: Students can view profiling results directly in JupyterLab without downloading .nsys-rep files
- **Implementation**: Add to docker-compose pip install and configure JupyterLab extension
- **Benefit**: First couple of notebooks can use local Nsight analysis before needing downloads

### 4. Add Nsight Compute Coverage (Future)
- **Goal**: Cover Nsight Compute for detailed kernel analysis
- **Use Case**: Micro-level Python kernel profiling (warp stalls, cache efficiency, occupancy)
- **Note**: Third level after PyTorch Profiler and Nsight Systems
- **When**: After establishing PyTorch Profiler workflow
- **Resources**:
  - [Fix GPU Bottlenecks: PyTorch Profiler + Nsight](https://acecloud.ai/blog/gpu-bottlenecks-pytorch-profiler-nsight/)
  - [Profiling PyTorch with Nsight Compute](https://dev-discuss.pytorch.org/t/how-profiling-pytorch-using-nsight-compute/2530/2)

## Completed Today ✓
- ✓ Single-node migration of all notebooks
- ✓ Removed Slurm commands
- ✓ Fixed hardcoded IPs to localhost
- ✓ Created docker-compose.yaml for Brev
- ✓ Added automated data download and setup
- ✓ Added SYS_ADMIN capability for nsys profiling
- ✓ Tested nsys profiling - working perfectly!

## Implementation Notes

### PyTorch + Nsight Integration Points
1. **Automatic NVTX Annotations**:
   - Update notebook cells to use: `!nsys profile --pytorch=autograd-nvtx --trace=cuda,osrt,nvtx ...`
   - No code changes needed - automatically annotates PyTorch ops

2. **PyTorch Profiler TensorBoard**:
   - Already have `torch_tb_profiler` installed
   - Can use `torch.profiler.profile()` with `tensorboard_trace_handler`
   - View results on TensorBoard (port 8889)

3. **Profiling Workflow** (from research):
   - Level 1: PyTorch Profiler - "Which operator is slow?"
   - Level 2: Nsight Systems - "Why? (I/O bound, sync bound, etc.)"
   - Level 3: Nsight Compute - "Kernel-level details (warp stalls, cache)"

## Notes
- Target hardware: 4× NVIDIA L4 GPUs (23GB each, Ada Lovelace, FP8 supported)
- Base image: nvcr.io/nvidia/pytorch:26.02-py3
- All changes on main branch, pushed to origin
- torch_tb_profiler already installed for TensorBoard integration

## Sources
- [Interactive Guide: Nsys vs PyTorch Profiler](https://the-dsvolk.github.io/ai-perf/ai-infra/Tale_of_two_profilers.html)
- [PyTorch Profiler Tutorial](https://docs.pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html)
- [Speed Up PyTorch Training 3x with Nsight](https://arikpoz.github.io/posts/2025-05-25-speed-up-pytorch-training-by-3x-with-nvidia-nsight-and-pytorch-2-tricks/)
- [Fix GPU Bottlenecks Guide](https://acecloud.ai/blog/gpu-bottlenecks-pytorch-profiler-nsight/)
- [NASA AI Profiler Guide](https://www.nas.nasa.gov/hackathon/assets/pdf/AI_Profiler.pdf)
