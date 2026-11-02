"""Hinton-style knowledge-distillation loss.

    L_KD = T^2 * KL(p_T^T || p_S^T),   p^T = softmax(z / T)

``teacher_probs`` are the teacher's probabilities at temperature 1 (the same tensor the
reliability gate and the corruption map operate on). They are softened here as
``softmax(log p / T)``, which equals ``softmax(z / T)`` whenever ``p = softmax(z)``.
Pass ``pretempered=True`` if the caller has already applied the temperature.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

_REDUCTIONS = ("mean", "sum", "none")
TEACHER_KINDS = ("probs", "logprobs")


def _check_temperature(T: float) -> float:
    T = float(T)
    if not math.isfinite(T) or T <= 0:
        raise ValueError(f"temperature must be finite and positive, got {T}")
    return T


def soften_probs(probs: torch.Tensor, T: float) -> torch.Tensor:
    """Temperature-soften a probability tensor: softmax(log p / T) along the last dim (zeros stay
    zero)."""
    T = _check_temperature(T)
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
    kind: str = "probs",
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
        kind: "probs" (default) or "logprobs". With "logprobs", ``teacher_probs`` holds teacher
            log-probabilities (any additive per-row constant is removed by renormalization), and
            softening is done as log_softmax(log p / T) without leaving log space, so rows built
            in log space (e.g. float64 corrupted rows) do not underflow to exact zeros first.
            ``-inf`` entries are allowed (probability 0).

    Returns a scalar (or (B,) for "none"). If ``mask`` selects no rows the result is an exact
    zero that is still connected to ``student_logits`` (its gradient is zero), so
    ``backward``/``autograd.grad`` work.
    """
    T = _check_temperature(T)  # validated before any branch (incl. pretempered)
    if kind not in TEACHER_KINDS:
        raise ValueError(f"kind must be one of {TEACHER_KINDS}, got {kind!r}")
    if reduction not in _REDUCTIONS:
        raise ValueError(f"reduction must be one of {_REDUCTIONS}, got {reduction!r}")
    if student_logits.shape != teacher_probs.shape or student_logits.dim() != 2:
        raise ValueError(
            f"expected matching (B, C) shapes, got {tuple(student_logits.shape)} "
            f"and {tuple(teacher_probs.shape)}"
        )
    log_p_s = F.log_softmax(student_logits / T, dim=-1)
    if kind == "probs":
        p_t = teacher_probs.detach().to(student_logits.dtype)
        if not pretempered:
            p_t = soften_probs(p_t, T)
        # xlogy(0, 0) = 0, so teacher zeros contribute nothing.
        per_row = (torch.xlogy(p_t, p_t) - p_t * log_p_s).sum(dim=-1) * (T * T)
    else:
        lp = teacher_probs.detach()  # softened in its own precision (float64 is CPU-only)
        log_p_t = F.log_softmax(lp if pretempered else lp / T, dim=-1)
        log_p_t = log_p_t.to(device=student_logits.device, dtype=student_logits.dtype)
        p_t = log_p_t.exp()
        # entries with log p = -inf have p = 0 and contribute nothing
        safe = torch.where(torch.isfinite(log_p_t), log_p_t, torch.zeros_like(log_p_t))
        per_row = (p_t * (safe - log_p_s)).sum(dim=-1) * (T * T)

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
