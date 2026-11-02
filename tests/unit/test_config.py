from pathlib import Path

import pytest

from pcdkd.utils.config import load_config


def test_load_and_reject_unknown(tmp_path):
    assert load_config("configs/mve_cifar10.yaml").epochs == 20
    path = tmp_path / "bad.yaml"
    path.write_text("typo: 1\n")
    with pytest.raises(ValueError, match="unknown"):
        load_config(path)
