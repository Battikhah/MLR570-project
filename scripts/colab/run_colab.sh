#!/usr/bin/env bash
# Run the PCD suite queue on a Google Colab GPU, continuing where the AUS workstation stopped.
#
# Colab's counterpart of scripts/workstation/run_weekend.ps1, and it follows the same rules:
#   * Results live in Google Drive (My Drive/pcd-suite, the folder the workstation's rclone writes).
#     Colab mounts it at /content/drive, so no rclone is needed.
#   * Runs write to the local disk (fast). Every SYNC_MIN minutes, results, checkpoints and logs are
#     copied to Drive. The copy never deletes anything. A disconnect loses at most that much, and
#     the next session resumes from the last checkpoint.
#   * At the start, the earlier results come back from Drive, so the queue skips finished runs.
#     With FAMILY set (quick restore): only the teachers, a tiny marker per finished run of those
#     families (never copied back) and the checkpoints of unfinished ones; nothing large is read.
#     With FAMILY empty: the snapshot archive (1.2 GB, one file), then the files newer than it.
#   * torch is the same build the workstation ran (2.14.1, CUDA 12.6), so results stay comparable.
#
# Usage: run it from the notebook scripts/colab/pcd_colab.ipynb, or in a Colab terminal:
#   bash scripts/colab/run_colab.sh                       # whole queue, 2 processes
#   PROCS=3 FAMILY="teacher-switch" bash scripts/colab/run_colab.sh
#   SETUP_ONLY=1 bash scripts/colab/run_colab.sh          # restore + install + tests, no runs
# Keep the cell running: it does the Drive copies. Stopping it stops the workers.
set -euo pipefail

DRIVE="${DRIVE:-/content/drive/MyDrive/pcd-suite}"   # the workstation's Drive folder
WORK="${WORK:-/content/pcd-work}"                     # local disk, lost when the VM goes away
PROFILE="${PROFILE:-screen}"
PROCS="${PROCS:-2}"          # processes sharing the GPU (ResNet-8 students barely load one)
WORKERS="${WORKERS:-2}"      # data-loading workers per process (free Colab has 2 CPU cores)
SYNC_MIN="${SYNC_MIN:-5}"
FAMILY="${FAMILY:-}"         # optional subset, space-separated; default: every family
OUT="$WORK/outputs"
LOGS="$WORK/logs"
DATA="$WORK/data"
CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Claims belong to one session; *.tmp are half-written; failures of old sessions are retried.
EXCLUDE=(--exclude claim --exclude '*.claim' --exclude '*.tmp')

restore() {
  command -v rsync >/dev/null || { apt-get -qq update && apt-get -qq install -y rsync; }
  # Only when the local results are gone (a fresh VM). --ignore-existing: never overwrite local.
  if find "$OUT/$PROFILE" -name result.json -print -quit 2>/dev/null | grep -q .; then
    echo "Local results present: skipping the Drive restore."
    [[ -f "$WORK/.restored" ]] || touch "$WORK/.restored"
    return
  fi
  mkdir -p "$OUT"
  if [[ -n "$FAMILY" ]]; then quick_restore; touch "$WORK/.restored"; return; fi
  if [[ -f "$DRIVE/snapshot/outputs.tgz" ]]; then
    echo "Extracting the results snapshot from Drive (one file) ..."
    tar -xzf "$DRIVE/snapshot/outputs.tgz" -C "$WORK" || echo "Snapshot did not extract; copying file by file."
  fi
  echo "Copying results newer than the snapshot (all of them if there is none) ..."
  rsync -rt --ignore-existing "${EXCLUDE[@]}" --exclude error.json --exclude '*.error.json' \
    "$DRIVE/outputs/" "$OUT/"
  find "$OUT" \( -name claim -o -name '*.claim' -o -name '*.tmp' -o -name error.json \
    -o -name '*.error.json' \) -delete
  touch "$WORK/.restored"
}

quick_restore() {
  echo "Quick restore for: $FAMILY (teachers, finished-run markers, checkpoints) ..."
  mkdir -p "$OUT/$PROFILE"
  rsync -rt "${EXCLUDE[@]}" --exclude '*.error.json' --exclude '*.ckpt' \
    "$DRIVE/outputs/$PROFILE/teachers" "$OUT/$PROFILE/"
  # shellcheck disable=SC2086  # FAMILY is a space-separated list on purpose
  python3 - "$DRIVE/outputs/$PROFILE" "$OUT/$PROFILE" $FAMILY <<'PY'
import os, shutil, sys
src, dst, fams = sys.argv[1], sys.argv[2], sys.argv[3:]
done = resumed = 0
for fam in fams:
    for root, dirs, files in os.walk(os.path.join(src, fam)):
        rel = os.path.relpath(root, src)
        if "result.json" in files:  # finished: a marker so the queue skips it (push never sends it)
            os.makedirs(os.path.join(dst, rel), exist_ok=True)
            with open(os.path.join(dst, rel, "result.json"), "w") as f:
                f.write('{"placeholder": "finished; the real result is on Drive"}\n')
            done += 1
        elif "checkpoint.pt" in files:  # unfinished: resume from its checkpoint
            os.makedirs(os.path.join(dst, rel), exist_ok=True)
            shutil.copy2(os.path.join(root, "checkpoint.pt"), os.path.join(dst, rel, "checkpoint.pt"))
            resumed += 1
print(f"{done} finished runs marked, {resumed} checkpoints copied")
PY
}

push() {
  # Only files written since the restore, so each copy does not re-check thousands of files on the
  # Drive mount. Copy, never delete: Drive keeps everything any session wrote.
  # A result.json under 2 KB is a quick-restore marker: never send it, so it can't replace a result.
  (cd "$OUT" && find . -type f -newer "$WORK/.restored" ! -name claim ! -name '*.claim' \
    ! -name '*.tmp' ! \( -name result.json -size -2k \) -print0) |
    rsync -t --from0 --files-from=- "$OUT/" "$DRIVE/outputs/"
  rsync -rt "$LOGS/" "$DRIVE/logs/"
}

alive() {
  local p
  for p in "$@"; do kill -0 "$p" 2>/dev/null && return 0; done
  return 1
}

main() {
  [[ -d "$DRIVE" ]] || { echo "No $DRIVE: mount Drive with the account that holds pcd-suite (set DRIVE= if it is elsewhere)."; exit 1; }
  mkdir -p "$OUT" "$LOGS" "$DATA"
  local stamp; stamp="colab-$(date +%Y%m%d-%H%M)"
  exec > >(tee -a "$LOGS/session-$stamp.log") 2>&1
  export PYTHONUNBUFFERED=1

  restore

  # environment: the locked packages, then the workstation's CUDA 12.6 build of torch
  cd "$CODE"
  echo "Code: $(git rev-parse --short HEAD) $(git log -1 --format=%s)"
  command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
  uv sync --frozen
  # --python: into the project's .venv. Colab's own uv settings otherwise send it to the system Python,
  # and the runs would use the locked torch 2.15.0+cu132 instead of the workstation's build.
  uv pip install --python .venv/bin/python --reinstall "torch==2.14.1" "torchvision==0.29.1" \
    --index-url https://download.pytorch.org/whl/cu126
  uv run --no-sync python -c "import sys, torch; ok = torch.cuda.is_available(); print('torch', torch.__version__, 'CUDA:', ok, torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok else 1)" \
    || { echo "No GPU: Runtime > Change runtime type > GPU, then run again."; exit 1; }
  uv run --no-sync python -c "import sys, torch; sys.exit(0 if torch.__version__.startswith('2.14.1') else 1)" \
    || { echo "torch is not the workstation's 2.14.1+cu126 build; stopping so results stay comparable."; exit 1; }

  [[ "${SKIP_TESTS:-0}" == 1 ]] || uv run --no-sync pytest -q -m "not slow and not requires_data"

  uv run --no-sync python -c "from torchvision.datasets import CIFAR10, CIFAR100; import sys; [c(sys.argv[1], train=t, download=True) for c in (CIFAR10, CIFAR100) for t in (True, False)]" "$DATA"
  # CRD's pre-trained CIFAR-100 teachers (mdistiller release), only needed by lit-p1
  local md="$DATA/mdistiller"
  if [[ -z "$FAMILY" || " $FAMILY " == *" lit-p1 "* ]] &&
    [[ ! -f "$md/cifar_teachers/resnet32x4_vanilla/ckpt_epoch_240.pth" ]]; then
    mkdir -p "$md"
    curl -L --fail -o "$md/cifar_teachers.tar" \
      https://github.com/megvii-research/mdistiller/releases/download/checkpoints/cifar_teachers.tar
    echo "e7c8b2f7d449af08e1222663e845ad23ca4a2044735b980740d86a96088de52c  $md/cifar_teachers.tar" | sha256sum -c - \
      || { rm -f "$md/cifar_teachers.tar"; echo "cifar_teachers.tar has the wrong SHA-256; deleted, run again."; exit 1; }
    tar -xf "$md/cifar_teachers.tar" -C "$md" && rm "$md/cifar_teachers.tar"
  fi
  if [[ "${SETUP_ONLY:-0}" == 1 ]]; then push; echo "Setup OK."; return; fi

  export PCDKD_OUT="$OUT"
  local select=(--profile "$PROFILE")
  # shellcheck disable=SC2206  # FAMILY is a space-separated list on purpose
  [[ -n "$FAMILY" ]] && select+=(--family $FAMILY)
  uv run --no-sync python scripts/run_suite.py clear-claims "${select[@]}"
  local pids=()
  for i in $(seq 1 "$PROCS"); do
    uv run --no-sync python scripts/run_suite.py run "${select[@]}" --queue --window none \
      --workers "$WORKERS" --data-root "$DATA" > "$LOGS/worker$i-$stamp.log" 2>&1 &
    pids+=($!)
    sleep 20   # stagger start-up so the first teachers are claimed in order
  done
  trap 'pkill -f scripts/run_suite.py; push' EXIT   # uv and each worker's child process
  echo "$PROCS workers running; results copy to $DRIVE/outputs every $SYNC_MIN min."
  while alive "${pids[@]}"; do
    sleep $((SYNC_MIN * 60)) & wait $! || true
    push
    echo "$(date +%H:%M) $(find "$OUT/$PROFILE" -name result.json -newer "$WORK/.restored" | wc -l) new results this session; $(tail -qn1 "$LOGS"/worker*-"$stamp".log | cut -c1-110 | tr '\n' '|')"
  done
  echo "Queue finished. Results: $DRIVE/outputs"
}

# sourced (tests): only define the functions
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi
