import pytest
import torch

from pcdkd.models import resnet8, resnet14, resnet20, resnet32


@pytest.mark.parametrize("factory", [resnet8, resnet14, resnet20, resnet32])
@pytest.mark.parametrize("classes", [10, 100])
def test_cpu_output_shape(factory, classes):
    assert factory(classes)(torch.randn(2, 3, 32, 32)).shape == (2, classes)


def test_parameter_counts():
    counts = [sum(p.numel() for p in f().parameters()) for f in (resnet8, resnet14, resnet20, resnet32)]
    assert counts == [75_802, 170_906, 266_010, 455_568]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_forward():
    assert resnet8().to("mps")(torch.randn(1, 3, 32, 32, device="mps")).shape == (1, 10)
