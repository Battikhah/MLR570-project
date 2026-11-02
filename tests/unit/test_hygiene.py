import subprocess
from pathlib import Path

from scripts.check_repo_hygiene import check


def test_detects_oversize_dataset_and_secret(tmp_path):
    (tmp_path / "plain.txt").write_text("AKIA" + "1" * 16)
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "latest.bin").write_bytes(b"x")
    (tmp_path / "model.npz").write_bytes(b"x")
    (tmp_path / "large.txt").write_bytes(b"x" * (5 * 1024 * 1024 + 1))
    paths = ["plain.txt", "checkpoints/latest.bin", "model.npz", "large.txt"]
    assert len(check(paths, tmp_path)) == 4


def test_hygiene_clean_repo_has_no_findings():
    root = Path(__file__).resolve().parents[2]
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    assert check(filter(None, tracked), root) == []


def test_missing_tracked_file_is_ignored(tmp_path):
    assert check(["deleted.txt"], tmp_path) == []
