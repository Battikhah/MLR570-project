"""Teacher reliability gate (proposal p. 2): which examples receive the KD loss."""

from __future__ import annotations

import torch

GATE_MODES = ("none", "conf_only", "correct_and_conf")


def reliability_mask(
    teacher_probs: torch.Tensor, labels: torch.Tensor, q: float, mode: str
) -> torch.Tensor:
    """Boolean (B,) mask of examples whose teacher output is trusted.

    Modes:
        "none":             every example (no gate).
        "conf_only":        max_c p_T(c) > q.
        "correct_and_conf": argmax_c p_T(c) == label AND max_c p_T(c) > q.
    The threshold is strict: an example with max probability exactly q is excluded.
    ``teacher_probs`` are temperature-1 probabilities; ``labels`` are the (training) labels.
    """
    if mode not in GATE_MODES:
        raise ValueError(f"mode must be one of {GATE_MODES}, got {mode!r}")
    if teacher_probs.dim() != 2:
        raise ValueError(f"teacher_probs must be (B, C), got {tuple(teacher_probs.shape)}")
    if mode == "none":
        return torch.ones(teacher_probs.shape[0], dtype=torch.bool, device=teacher_probs.device)
    conf, pred = teacher_probs.max(dim=-1)
    keep = conf > q
    if mode == "correct_and_conf":
        if labels.shape != pred.shape:
            msg = f"labels must have shape {tuple(pred.shape)}, got {tuple(labels.shape)}"
            raise ValueError(msg)
        keep = keep & (pred == labels.to(pred.device))
    return keep
