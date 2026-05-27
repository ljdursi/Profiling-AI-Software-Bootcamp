# Intro-Notebook Scripts: Design Notes & Pause State

## What's here

Three scripts implementing the original "DataLoader → non_blocking" two-bug plan:
- `train_v0_baseline.py` — both bugs present
- `train_v1_dataloader_fixed.py` — DataLoader fixed, `.to(device)` still synchronous
- `train_v2_fully_fixed.py` — both fixed (uses `non_blocking=True`)

All three use CIFAR-10 + adapted ResNet18 (3×3 stride-1 stem, no maxpool) at native 32×32, batch 256, 60 iterations w/ 5 warmup, NVTX ranges, optional `--profile` flag for `torch.profiler` traces to `/workspace/logs/<script>/`.

## Measured behavior on this instance (1× L4)

| script | mean step | throughput |
| --- | --- | --- |
| v0 | 256.8 ms | 997 img/s |
| v1 | 93.7 ms  | 2727 img/s |
| v2 | 93.9 ms  | 2727 img/s |

- **v0 → v1**: dramatic (2.7×), DataLoader stall reproduces as planned.
- **v1 → v2**: **invisible**. `non_blocking=True` does not move wallclock at this scale.

## Why Bug 2 (non_blocking) doesn't show

Empirically tested at:
- 32×32 batch 256 (3 MB H2D): no change
- 128×128 batch 256 (50 MB H2D): no change
- 224×224 batch 128 (77 MB H2D): no change

Root cause: on L4, ResNet18 is GPU-bound enough that the CPU prep (data load + H2D + enqueue) happily fits inside the GPU compute window. Saving the few ms of CPU-side blocking on H2D has zero throughput effect because the CPU wasn't on the critical path anyway.

Also tested (all within ~3 ms of clean):
- `loss.item()` per step
- `torch.cuda.synchronize()` per step
- `logits.cpu() + y.cpu()` per step for metric
- per-sample Python loop on GPU
- Same on TinyCNN (where CPU prep becomes the bottleneck and CPU sync still doesn't matter)

## Revised structure (decided 2026-05-27)

Drop the single-script-three-versions plan. New structure is **two training scripts, four files total**:

1. **`train_v1.py`** — original training task. Has Bug 1 (DataLoader stall).
2. **`train_v1_fixed.py`** — Bug 1 fixed.
3. **`train_v2.py`** — *new, more ambitious training task* enabled by the v1 fix. Has Bug 2.
4. **`train_v2_fixed.py`** — Bug 2 fixed.

Pedagogical arc: "Once we fix the data-loading stall, we can take on a more ambitious training problem — and that bigger problem surfaces a new class of bug that PyTorch Profiler can hint at but Nsight Systems makes obvious."

This frees Bug 2 from having to live inside the simple CIFAR-10 + ResNet18 + 32×32 setup that was empirically shown to hide CPU-side sync bugs. The v2 task can scale up image size, model, batch size, or introduce mixed precision / autograd-anomaly / per-step eval — whatever combination produces a clean profiler-vs-nsys contrast.

## What to do next (paused here)

1. **Decide the v2 training task**: how to scale up from CIFAR-10 + ResNet18 + 32×32. Options to weigh: larger images (128/224), bigger model (ResNet50d, ViT-S), mixed precision, longer schedule, larger batch. Pick something that opens up a Bug 2 with clear timeline-vs-profiler contrast.
2. **Pick Bug 2** to live in the v2 task. Candidates worth trying:
   - `torch.autograd.set_detect_anomaly(True)` left enabled (~30% overhead, very real bug)
   - Per-step eval forward on a held-out batch (extra GPU work between train steps)
   - Unused auxiliary head computed every forward (extra GPU kernels on critical path)
   - cudnn benchmark off with varying input shapes (algo search per shape change)
   - Now that the workload is heavier, `non_blocking` / `.item()` could become visible — re-test before discarding
3. **Rename / restructure** existing files:
   - `train_v0_baseline.py` → `train_v1.py`
   - `train_v1_dataloader_fixed.py` → `train_v1_fixed.py`
   - Delete `train_v2_fully_fixed.py` (replaced by the two new v2 files once designed)
4. **Re-verify** the v1 → v1_fixed and v2 → v2_fixed deltas show on this hardware.
5. **Then** write the two intro notebooks (PyTorch Profiler intro on the v1 pair, Nsight Systems intro on the v2 pair).
