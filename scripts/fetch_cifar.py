#!/usr/bin/env python3
"""Opt-in download of official CIFAR archives with checksum verification."""
import argparse
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

ARCHIVES = {
    "cifar10": (
        "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        "c58f30108f718f92721af3b95e74349a",
        "cifar-10-python.tar.gz",
    ),
    "cifar100": (
        "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
        "eb9058c3a382ffc7106e4002c42a8d85",
        "cifar-100-python.tar.gz",
    ),
}


def md5(path):
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=ARCHIVES)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    url, expected, name = ARCHIVES[args.dataset]
    target = args.root / name
    print(f"{url} -> {target} (expected md5 {expected})")
    if args.dry_run:
        return
    if not args.yes:
        parser.error("download requires --yes")
    if target.exists():
        parser.error(f"refusing to overwrite {target}")
    args.root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.root, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        urllib.request.urlretrieve(url, temporary_path)  # noqa: S310 - URL comes from the fixed archive allowlist.
        actual = md5(temporary_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    if actual != expected:
        temporary_path.unlink()
        raise SystemExit(f"md5 mismatch: expected {expected}, got {actual}")
    os.replace(temporary_path, target)
    print(f"verified {target.stat().st_size} bytes")


if __name__ == "__main__":
    main()
