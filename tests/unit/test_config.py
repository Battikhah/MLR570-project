import pytest

from pcdkd.utils.config import load_config


def test_load_and_reject_unknown(tmp_path):
    config = load_config("configs/mve_cifar10.yaml")
    assert config.epochs == 20
    assert config.method == "pcd" and config.tau_grid == [0.05, 0.1, 0.3]
    assert config.lambda_grid == [0.0, 0.1, 0.5, 1.0]
    assert config.gate_mode == "none" and config.select_temperature_on_val
    path = tmp_path / "bad.yaml"
    path.write_text("typo: 1\n")
    with pytest.raises(ValueError, match="unknown"):
        load_config(path)


@pytest.mark.parametrize(
    "yaml_value", ["temperature: '4'", "batch_size: 128.0", "select_temperature_on_val: 1"]
)
def test_config_rejects_wrong_types_as_value_error(tmp_path, yaml_value):
    path = tmp_path / "bad-type.yaml"
    path.write_text(yaml_value)
    with pytest.raises(ValueError):
        load_config(path)
