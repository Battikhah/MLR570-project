#!/usr/bin/env python3
"""Bounded synthetic throughput benchmark for CIFAR ResNets."""
import time

import torch

from pcdkd.models import resnet8, resnet32


def bench(model_fn, device, batch=128, steps=3, two_backward=False):
    model = model_fn().to(device).train()
    x = torch.randn(batch, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (batch,), device=device)
    for _ in range(1):
        model.zero_grad(set_to_none=True)
        model(x).sum().backward()
    if device == "mps":
        torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        model.zero_grad(set_to_none=True)
        logits = model(x)
        if two_backward:
            params = tuple(p for p in model.parameters() if p.requires_grad)
            torch.autograd.grad(
                torch.nn.functional.cross_entropy(logits, y), params, retain_graph=True
            )
            torch.autograd.grad(logits.square().mean(), params)
        else:
            torch.nn.functional.cross_entropy(logits, y).backward()
    if device == "mps":
        torch.mps.synchronize()
    return batch * steps / (time.perf_counter() - start)


if __name__ == "__main__":
    for device in ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []):
        for model_name, fn in (("resnet8", resnet8), ("resnet32", resnet32)):
            for two in (False, True):
                mode = "two-backward" if two else "forward-backward"
                rate = bench(fn, device, two_backward=two)
                print(f"{model_name} {device} {mode}: {rate:.1f} images/s (synthetic)")
