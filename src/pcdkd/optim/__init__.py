"""pcdkd.optim: gradient utilities, EMA normalization, the PCD QP and gradient combiners."""

from pcdkd.optim.combiners import (
    COMBINERS,
    PCD,
    AdaptiveLambda,
    Combiner,
    NormalizedWeightedSum,
    PCGrad,
    WeightedSum,
)
from pcdkd.optim.grads import flat_grad, unflatten_to
from pcdkd.optim.normalize import EMANormalizer
from pcdkd.optim.pcd import (
    pcd_k2_closed_form,
    pcd_kkt_enumeration,
    primary_margin,
    solve_pcd_qp,
)

__all__ = [
    "COMBINERS",
    "PCD",
    "AdaptiveLambda",
    "Combiner",
    "EMANormalizer",
    "NormalizedWeightedSum",
    "PCGrad",
    "WeightedSum",
    "flat_grad",
    "pcd_k2_closed_form",
    "pcd_kkt_enumeration",
    "primary_margin",
    "solve_pcd_qp",
    "unflatten_to",
]
