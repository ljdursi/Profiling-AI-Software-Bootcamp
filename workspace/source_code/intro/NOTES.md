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

## What to do next (paused here)

Bug 2 needs to be **GPU-cost on the critical path**, not a CPU-side sync. Candidates listed in task #6:
1. `torch.autograd.set_detect_anomaly(True)` left enabled (~30% overhead)
2. Per-step eval forward on a held-out batch
3. Unused auxiliary head computed every forward
4. cudnn benchmark off with varying input shapes

Once Bug 2 is chosen and verified, update v1 and v2 accordingly, remove the misleading `non_blocking` framing from script docstrings, and proceed to writing the two intro notebooks.
