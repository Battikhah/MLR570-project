import pytest
import torch

from pcdkd.losses.gate import reliability_mask

P = torch.tensor(
    [
        [0.90, 0.05, 0.05],  # correct (label 0), confident
        [0.10, 0.80, 0.10],  # wrong (label 0), confident
        [0.50, 0.30, 0.20],  # correct, max == q exactly -> excluded
        [0.40, 0.35, 0.25],  # correct, not confident
    ]
)
Y = torch.tensor([0, 0, 0, 0])


def test_none_mode_all_true():
    assert reliability_mask(P, Y, 0.5, "none").tolist() == [True] * 4


def test_conf_only_strict_threshold():
    assert reliability_mask(P, Y, 0.5, "conf_only").tolist() == [True, True, False, False]


def test_correct_and_conf():
    assert reliability_mask(P, Y, 0.5, "correct_and_conf").tolist() == [True, False, False, False]
    assert reliability_mask(P, Y, 0.0, "correct_and_conf").tolist() == [True, False, True, True]


def test_mask_dtype_and_bad_mode():
    m = reliability_mask(P, Y, 0.5, "conf_only")
    assert m.dtype == torch.bool and m.shape == (4,)
    with pytest.raises(ValueError):
        reliability_mask(P, Y, 0.5, "bogus")
