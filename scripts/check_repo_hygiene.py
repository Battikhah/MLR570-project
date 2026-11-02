#!/usr/bin/env python3
"""Reject oversized tracked files, data artifacts, and obvious credentials."""
import re
import subprocess
import sys
from pathlib import Path

PATTERNS = re.compile(
    r"(^|/)(?:datasets?|checkpoints?)(/|$)|(^|/)(?:cifar|checkpoint|ckpt|weights?)[^/]*(?:$|[._-])|\.(?:pt|pth|ckpt|safetensors)$",
    re.I,
)
SECRETS = re.compile(
    r"AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----|"
    r"\bgh[po]_[A-Za-z0-9]{20,}\b"
)


def check(paths, root=Path(".")):
    errors = []
    for name in paths:
        path = root / name
        if "Project Files/" not in name and path.stat().st_size > 5 * 1024 * 1024:
            errors.append(f"oversized tracked file: {name}")
        if PATTERNS.search(name):
            errors.append(f"dataset/checkpoint path: {name}")
        if path.is_file() and path.stat().st_size < 5 * 1024 * 1024:
            try:
                if SECRETS.search(path.read_text(encoding="utf-8", errors="ignore")):
                    errors.append(f"possible secret: {name}")
            except OSError:
                pass
    return errors


def main():
    files = subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0")  # noqa: S603, S607
    errors = check(filter(None, files))
    print("\n".join(errors) if errors else "repository hygiene OK")
    return bool(errors)


if __name__ == "__main__":
    sys.exit(main())
