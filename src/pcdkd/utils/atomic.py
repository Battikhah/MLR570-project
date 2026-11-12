"""Atomic file replacement that tolerates an uploader (rclone, a sync client) holding the file.

On Windows, ``os.replace`` fails with PermissionError while another process has the target open,
which an uploader does while it copies the file to the cloud. A large teacher checkpoint on a slow
link can take minutes, so the default wait is long.
"""

from __future__ import annotations

import os
import time
from pathlib import Path


def replace_with_retry(tmp: Path, dst: Path, tries: int = 300, wait: float = 1.0) -> None:
    for i in range(tries):
        try:
            os.replace(tmp, dst)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(wait)


def save_torch(obj, path) -> None:
    """``torch.save`` to a temporary file next to ``path``, then replace ``path`` atomically."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    replace_with_retry(tmp, path)
