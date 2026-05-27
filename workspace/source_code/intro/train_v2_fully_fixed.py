"""Intro lab v2: both bugs fixed.

Diff vs v1_dataloader_fixed.py:
    x = x.to(device)               -> x = x.to(device, non_blocking=True)
    y = y.to(device)               -> y = y.to(device, non_blocking=True)

Expected behavior:
    - Nsight Systems timeline shows the gap before each forward is closed:
      the CPU returns from .to() immediately, queues the next forward, and the
      GPU stays busy across step boundaries.
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.cuda import nvtx
from torch.utils.data import DataLoader

SEED = 0
BATCH_SIZE = 256
NUM_ITERS = 60
WARMUP_ITERS = 5
DATA_ROOT = "/workspace/data"
LOG_ROOT = "/workspace/logs"

NUM_WORKERS = 4
PIN_MEMORY = True


def build_loader():
    transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=False, transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
        persistent_workers=NUM_WORKERS > 0,
    )


def build_model(device):
    model = torchvision.models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model.to(device)


def train(profile: bool):
    torch.manual_seed(SEED)
    device = torch.device("cuda")
    loader = build_loader()
    model = build_model(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    profiler_ctx = None
    if profile:
        script_name = Path(__file__).stem
        logdir = Path(LOG_ROOT) / script_name
        logdir.mkdir(parents=True, exist_ok=True)
        profiler_ctx = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1, warmup=WARMUP_ITERS, active=10, repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(logdir)),
            record_shapes=True,
            with_stack=False,
        )

    step_times = []
    loader_iter = iter(loader)
    if profiler_ctx is not None:
        profiler_ctx.__enter__()

    for step in range(NUM_ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        nvtx.range_push("step")

        nvtx.range_push("data_load")
        x, y = next(loader_iter)
        nvtx.range_pop()

        nvtx.range_push("h2d")
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        nvtx.range_pop()

        nvtx.range_push("forward")
        logits = model(x)
        loss = loss_fn(logits, y)
        nvtx.range_pop()

        nvtx.range_push("backward")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nvtx.range_pop()

        nvtx.range_push("step_opt")
        optimizer.step()
        nvtx.range_pop()

        nvtx.range_pop()  # step

        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)

        if profiler_ctx is not None:
            profiler_ctx.step()

    if profiler_ctx is not None:
        profiler_ctx.__exit__(None, None, None)

    timed = step_times[WARMUP_ITERS:]
    mean_ms = 1000 * sum(timed) / len(timed)
    print(f"steps timed: {len(timed)}  mean step: {mean_ms:.1f} ms  "
          f"throughput: {BATCH_SIZE / (mean_ms / 1000):.0f} img/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    train(profile=args.profile)
