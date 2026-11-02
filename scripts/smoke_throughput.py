#!/usr/bin/env python3
"""Repeated synthetic CIFAR throughput benchmark; two-loss mode uses raw-gradient PCD proxy."""
import argparse
import statistics
import time

import torch
import torch.nn.functional as F

from pcdkd.models import resnet8, resnet32


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()


def train_step(model, optimizer, x, y, *, two_backward, tau=0.1):
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    if not two_backward:
        F.cross_entropy(logits, y).backward()
    else:
        params = tuple(p for p in model.parameters() if p.requires_grad)
        primary = torch.autograd.grad(F.cross_entropy(logits, y), params, retain_graph=True)
        secondary = torch.autograd.grad(logits.square().mean(), params)
        flat_primary = torch.cat([grad.reshape(-1) for grad in primary])
        flat_secondary = torch.cat([grad.reshape(-1) for grad in secondary])
        gram12 = torch.dot(flat_primary, flat_secondary)
        gram22 = torch.dot(flat_secondary, flat_secondary)
        coefficient = torch.clamp(tau - gram12 / gram22.clamp_min(1e-20), min=0)
        combined = flat_primary + coefficient * flat_secondary
        offset = 0
        for parameter in params:
            size = parameter.numel()
            parameter.grad = combined[offset : offset + size].view_as(parameter).clone()
            offset += size
    optimizer.step()


def bench(
    model_fn, device, *, num_classes=10, batch=128, warmup=10, steps=50, repeats=3,
    two_backward=False,
):
    measurements = []
    for repeat in range(repeats):
        torch.manual_seed(1000 + repeat)
        model = model_fn(num_classes=num_classes).to(device).train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.randn(batch, 3, 32, 32, device=device)
        y = torch.randint(0, num_classes, (batch,), device=device)
        for _ in range(warmup):
            train_step(model, optimizer, x, y, two_backward=two_backward)
        synchronize(device)
        start = time.perf_counter()
        for _ in range(steps):
            train_step(model, optimizer, x, y, two_backward=two_backward)
        synchronize(device)
        measurements.append(batch * steps / (time.perf_counter() - start))
    return statistics.median(measurements), min(measurements), max(measurements)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="one CPU ResNet-8 smoke case with 2 warm-ups and 3 steps",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    devices = ["cpu"] if args.quick else ["cpu"] + (
        ["mps"] if torch.backends.mps.is_available() else []
    )
    cases = [("resnet8", resnet8, 10)] if args.quick else [
        (name, model_fn, classes)
        for classes in (10, 100)
        for name, model_fn in (("resnet8", resnet8), ("resnet32", resnet32))
    ]
    for device in devices:
        for name, model_fn, classes in cases:
            for two in (False, True):
                mode = "two-objective-pcd-proxy" if two else "forward-backward"
                median, low, high = bench(
                    model_fn,
                    device,
                    num_classes=classes,
                    warmup=2 if args.quick else 10,
                    steps=3 if args.quick else 50,
                    repeats=1 if args.quick else 3,
                    two_backward=two,
                )
                print(
                    f"{name} classes={classes} device={device} mode={mode}: "
                    f"median={median:.1f} img/s range=[{low:.1f}, {high:.1f}] "
                    f"(synthetic, threads={torch.get_num_threads()})",
                    flush=True,
                )
