import hashlib
import subprocess
import sys

from scripts.fetch_cifar import md5


def test_md5_and_dry_run(tmp_path):
    sample = tmp_path / "tiny"
    sample.write_bytes(b"test")
    assert md5(sample) == hashlib.md5(b"test").hexdigest()
    result = subprocess.run([sys.executable, "scripts/fetch_cifar.py", "cifar10", "--root", str(tmp_path), "--dry-run"], capture_output=True, text=True, check=True)
    assert "expected md5 c58f30108f718f92721af3b95e74349a" in result.stdout
    assert not (tmp_path / "cifar-10-python.tar.gz").exists()
