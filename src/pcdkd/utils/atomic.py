"""Atomic file replacement that tolerates an uploader (rclone, a sync client) holding the file.

On Windows, ``os.replace`` fails with PermissionError while another process has the target open,
which an uploader does while it copies the file to the cloud. Callers that overwrite a file
periodically (checkpoints) should treat a final PermissionError as "try again next time".
"""

from __future__ import annotations

import os
import time
from pathlib import Path


def replace_with_retry(tmp: Path, dst: Path, tries: int = 60, wait: float = 1.0) -> None:
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
    try:
        replace_with_retry(tmp, path)
    except PermissionError:
        tmp.unlink(missing_ok=True)
        raise


def remove(path) -> None:
    """Delete ``path`` if it exists; a file another process holds open is left for later."""
    try:
        Path(path).unlink(missing_ok=True)
    except PermissionError:
        pass
