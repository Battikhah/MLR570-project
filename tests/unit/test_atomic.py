from pathlib import Path

import pytest

from pcdkd.utils import atomic


def test_remove_leaves_a_busy_file(tmp_path, monkeypatch):
    f = tmp_path / "checkpoint.pt"
    f.write_text("x")

    def busy(self, missing_ok=False):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(Path, "unlink", busy)
    atomic.remove(f)  # no exception
    assert f.exists()


def test_save_torch_cleans_up_when_the_target_stays_busy(tmp_path, monkeypatch):
    def busy(tmp, dst, tries=60, wait=1.0):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(atomic, "replace_with_retry", busy)
    with pytest.raises(PermissionError):
        atomic.save_torch({"a": 1}, tmp_path / "c.pt")
    assert list(tmp_path.iterdir()) == []
