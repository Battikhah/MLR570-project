"""Flatten per-objective gradients into one vector and write a direction back into ``.grad``."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def flat_grad(
    loss: torch.Tensor, params: Sequence[torch.Tensor], retain_graph: bool
) -> torch.Tensor:
    """Gradient of ``loss`` w.r.t. ``params`` as one 1-D tensor (unused parameters give 0).

    If ``loss`` does not require grad (e.g. a constant), the zero vector is returned.
    """
    params = list(params)
    if not params:
        raise ValueError("params is empty")
    if not loss.requires_grad:
        return torch.cat([torch.zeros_like(p).reshape(-1) for p in params])
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    return torch.cat(
        [
            (torch.zeros_like(p) if g is None else g).reshape(-1)
            for p, g in zip(params, grads, strict=True)
        ]
    )


@torch.no_grad()
def unflatten_to(params: Sequence[torch.Tensor], vec: torch.Tensor) -> None:
    """Write consecutive slices of ``vec`` into each ``p.grad`` (overwrites ``.grad``)."""
    params = list(params)
    total = sum(p.numel() for p in params)
    if vec.dim() != 1 or vec.numel() != total:
        raise ValueError(f"vec must be 1-D with {total} elements, got shape {tuple(vec.shape)}")
    offset = 0
    for p in params:
        n = p.numel()
        p.grad = vec[offset : offset + n].view_as(p).to(dtype=p.dtype, device=p.device).clone()
        offset += n
