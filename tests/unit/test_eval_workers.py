"""Evaluation metrics must not depend on DataLoader worker count."""

import pytest
import torch

from pcdkd.data.build import SyntheticCifar
from pcdkd.data.cifar import IndexedDataset
from pcdkd.experiments.suite import build_specs
from pcdkd.models import get_model
from pcdkd.train.trainer import _loader, evaluate


def test_synthetic_metrics_match_with_zero_and_two_workers():
    specs, _ = build_specs("smoke", families=["core-p0"])
    spec = specs[0]
    torch.manual_seed(0)
    model = get_model(spec["arch"], 100).eval()
    dataset = IndexedDataset(SyntheticCifar(16, 100, seed=5))
    baseline = evaluate(model, _loader(dataset, 8, False, 0, 0, "cpu"), 100, "cpu")
    try:
        parallel = evaluate(model, _loader(dataset, 8, False, 0, 2, "cpu"), 100, "cpu")
    except RuntimeError as exc:
        if "torch_shm_manager" in str(exc):
            pytest.skip("sandbox blocks PyTorch worker shared-memory startup")
        raise
    assert baseline == parallel
