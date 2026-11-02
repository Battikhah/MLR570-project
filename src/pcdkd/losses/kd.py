"""Hinton-style knowledge-distillation loss.

    L_KD = T^2 * KL(p_T^T || p_S^T),   p^T = softmax(z / T)

``teacher_probs`` are the teacher's probabilities at temperature 1 (the same tensor the
reliability gate and the corruption map operate on). They are softened here as
``softmax(log p / T)``, which equals ``softmax(z / T)`` whenever ``p = softmax(z)``.
Pass ``pretempered=True`` if the caller has already applied the temperature.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_REDUCTIONS = ("mean", "sum", "none")


def soften_probs(probs: torch.Tensor, T: float) -> torch.Tensor:
    """Temperature-soften a probability tensor: softmax(log p / T) along the last dim (zeros stay
    zero)."""
    if T <= 0:
        raise ValueError(f"temperature must be positive, got {T}")
    if T == 1.0:
        return probs
    logp = torch.log(probs.clamp_min(0.0))  # log 0 = -inf -> softmax weight 0
    return torch.softmax(logp / T, dim=-1)


def kd_loss(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    T: float,
    mask: torch.Tensor | None = None,
    reduction: str = "mean",
    *,
    pretempered: bool = False,
) -> torch.Tensor:
    """T^2 * KL(p_T^T || p_S^T) per example, reduced over the masked-in examples.

    Args:
        student_logits: (B, C) raw student logits.
        teacher_probs: (B, C) teacher probabilities at T = 1 (or at T if ``pretempered``);
            treated as constants (detached).
        T: distillation temperature (> 0).
        mask: optional (B,) bool tensor; only True rows contribute.
        reduction: "mean" (over masked-in rows), "sum", or "none" (per-row, masked-out rows = 0).
        pretempered: ``teacher_probs`` already carry the temperature.

    Returns a scalar (or (B,) for "none"). If ``mask`` selects no rows the result is an exact
    zero that is still connected to ``student_logits`` (its gradient is zero), so
    ``backward``/``autograd.grad`` work.
    """
    if reduction not in _REDUCTIONS:
        raise ValueError(f"reduction must be one of {_REDUCTIONS}, got {reduction!r}")
    if student_logits.shape != teacher_probs.shape or student_logits.dim() != 2:
        raise ValueError(
            f"expected matching (B, C) shapes, got {tuple(student_logits.shape)} "
            f"and {tuple(teacher_probs.shape)}"
        )
    p_t = teacher_probs.detach().to(student_logits.dtype)
    if not pretempered:
        p_t = soften_probs(p_t, T)
    log_p_s = F.log_softmax(student_logits / T, dim=-1)
    # xlogy(0, 0) = 0, so teacher zeros contribute nothing.
    per_row = (torch.xlogy(p_t, p_t) - p_t * log_p_s).sum(dim=-1) * (T * T)

    if mask is not None:
        if mask.shape != per_row.shape:
            msg = f"mask must have shape {tuple(per_row.shape)}, got {tuple(mask.shape)}"
            raise ValueError(msg)
        mask = mask.to(device=per_row.device, dtype=torch.bool)
        per_row = torch.where(mask, per_row, torch.zeros_like(per_row))
        n = int(mask.sum().item())
    else:
        n = per_row.shape[0]

    if reduction == "none":
        return per_row
    if reduction == "sum":
        return per_row.sum()
    if n == 0:
        return student_logits.sum() * 0.0
    return per_row.sum() / n
