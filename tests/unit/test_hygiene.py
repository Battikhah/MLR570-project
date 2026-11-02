from scripts.check_repo_hygiene import check


def test_detects_oversize_dataset_and_secret(tmp_path):
    (tmp_path / "plain.txt").write_text("AKIA1234567890123456")
    (tmp_path / "weights.bin").write_bytes(b"x")
    (tmp_path / "large.txt").write_bytes(b"x" * (5 * 1024 * 1024 + 1))
    assert len(check(["plain.txt", "weights.bin", "large.txt"], tmp_path)) == 3
