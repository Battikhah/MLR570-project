import pytest
import torch

from pcdkd.models import resnet8, resnet14, resnet20, resnet32


def analytic_parameter_count(depth, num_classes=10, shortcut_type="A"):
    blocks_per_stage = (depth - 2) // 6

    def block_params(in_channels, channels, projection):
        params = 9 * in_channels * channels + 2 * channels
        params += 9 * channels**2 + 2 * channels
        if projection:
            params += in_channels * channels + 2 * channels
        return params

    total = 3 * 16 * 3**2 + 2 * 16
    for in_channels, channels in ((16, 16), (16, 32), (32, 64)):
        projection = shortcut_type == "B" and in_channels != channels
        total += block_params(in_channels, channels, projection)
        total += (blocks_per_stage - 1) * block_params(channels, channels, False)
    return total + 64 * num_classes + num_classes


@pytest.mark.parametrize("factory", [resnet8, resnet14, resnet20, resnet32])
@pytest.mark.parametrize("classes", [10, 100])
def test_cpu_output_shape(factory, classes):
    assert factory(classes)(torch.randn(2, 3, 32, 32)).shape == (2, classes)


def test_parameter_counts():
    factories = (resnet8, resnet14, resnet20, resnet32)
    counts = [sum(p.numel() for p in factory().parameters()) for factory in factories]
    analytic = [analytic_parameter_count(depth) for depth in (8, 14, 20, 32)]
    assert analytic == [75_290, 172_506, 269_722, 464_154]
    assert counts == analytic


def test_option_b_projection_shortcuts_remain_selectable():
    for depth, factory in zip(
        (8, 14, 20, 32), (resnet8, resnet14, resnet20, resnet32), strict=True
    ):
        model = factory(shortcut_type="B")
        actual = sum(parameter.numel() for parameter in model.parameters())
        assert actual == analytic_parameter_count(depth, shortcut_type="B")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_forward():
    assert resnet8().to("mps")(torch.randn(1, 3, 32, 32, device="mps")).shape == (1, 10)
