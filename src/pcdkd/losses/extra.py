"""Extra distillation losses used as separate PCD objectives.

* ``dkd_parts``: DKD's target-class (TCKD) and non-target-class (NCKD) terms (Zhao et al.,
  CVPR 2022). Per example, KL(p_t || p_s) = TCKD + (1 - p_t[target]) * NCKD (chain rule of KL);
  both are returned *unweighted* (times T^2) so a combiner can treat them as two objectives.
* ``at_loss``: attention transfer (Zagoruyko & Komodakis, ICLR 2017), summed over feature pairs.

Teacher inputs are logits or log-probabilities (a per-row additive constant is irrelevant),
so corrupted teacher rows built in log space can be passed directly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_logsumexp(logp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(logp.masked_fill(~mask, float("-inf")), dim=-1)


def dkd_parts(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, target: torch.Tensor, T: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (TCKD, NCKD), each a batch-mean scalar times T^2."""
    log_ps = F.log_softmax(student_logits / T, dim=-1)
    log_pt = F.log_softmax(teacher_logits.detach().to(student_logits.dtype) / T, dim=-1)
    gt = F.one_hot(target, student_logits.shape[-1]).bool()
    # binary (target, rest) log-probabilities
    s_t, t_t = log_ps[gt], log_pt[gt]
    s_r, t_r = _masked_logsumexp(log_ps, ~gt), _masked_logsumexp(log_pt, ~gt)
    tckd = t_t.exp() * (t_t - s_t) + t_r.exp() * (t_r - s_r)
    # distribution among non-target classes
    ns, nt = log_ps - s_r[:, None], log_pt - t_r[:, None]
    nckd = (nt.exp() * (nt - ns)).masked_fill(gt, 0.0).sum(-1)
    return tckd.mean() * T * T, nckd.mean() * T * T


def _attention(f: torch.Tensor) -> torch.Tensor:
    return F.normalize(f.pow(2).mean(1).flatten(1), dim=1)


def at_loss(student_feats, teacher_feats) -> torch.Tensor:
    """Sum over layers of mean squared difference of normalized spatial attention maps."""
    if len(student_feats) != len(teacher_feats):
        raise ValueError("student and teacher must give the same number of feature maps")
    return sum(
        (_attention(s) - _attention(t.detach())).pow(2).mean()
        for s, t in zip(student_feats, teacher_feats, strict=True)
    )
