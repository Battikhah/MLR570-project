"""scripts/colab/run_colab.sh: the Drive restore skips session files, and the copy back sends only
what this session wrote."""

import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "colab" / "run_colab.sh"
pytestmark = pytest.mark.skipif(
    not (shutil.which("bash") and shutil.which("rsync")), reason="needs bash and rsync"
)


def _write(path, text="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _bash(tmp_path, body):
    env = f"export DRIVE={tmp_path}/drive WORK={tmp_path}/work; source {SCRIPT}; set +e; "
    subprocess.run(["bash", "-c", env + 'mkdir -p "$LOGS"; ' + body], check=True)


def test_restore_then_push(tmp_path):
    drive, run = tmp_path / "drive", "outputs/screen/fam/c/a"
    _write(tmp_path / "snap" / "outputs/screen/teachers/t.pt")
    (drive / "snapshot").mkdir(parents=True)
    with tarfile.open(drive / "snapshot/outputs.tgz", "w:gz") as tar:
        tar.add(tmp_path / "snap" / "outputs", arcname="outputs")
    _write(drive / run / "s0/result.json")
    for name in ("claim", "error.json", "checkpoint.pt.tmp", "checkpoint.pt"):
        _write(drive / run / "s1" / name)

    _bash(tmp_path, "restore")
    work = tmp_path / "work"
    got = {p.relative_to(work).as_posix() for p in work.rglob("*") if p.is_file()}
    assert got == {
        ".restored",
        "outputs/screen/teachers/t.pt",
        f"{run}/s0/result.json",
        f"{run}/s1/checkpoint.pt",
    }

    _bash(
        tmp_path,
        f'sleep 1; mkdir -p "$WORK/{run}/s2"; echo r > "$WORK/{run}/s2/result.json"; '
        f'echo c > "$WORK/{run}/s2/claim"; push',
    )
    assert (drive / run / "s2/result.json").exists()
    assert not (drive / run / "s2/claim").exists()
