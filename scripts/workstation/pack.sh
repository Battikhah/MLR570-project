#!/usr/bin/env bash
# Zip the code the AUS workstation needs (the repo has no git remote to clone from).
# Includes uncommitted and untracked files, skips git-ignored ones (data, outputs, .venv).
#   scripts/workstation/pack.sh [out.zip]     then copy the zip to OneDrive/pcd-suite/pcd-code.zip
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
out="${1:-$HOME/Downloads/pcd-code.zip}"
rm -f "$out"
git ls-files -co --exclude-standard -z -- \
  pyproject.toml uv.lock .python-version README.md AGENTS.md src scripts tests configs docs/experiment-suite.md \
  | xargs -0 zip -q "$out"
echo "wrote $out ($(du -h "$out" | cut -f1)); upload it to OneDrive as pcd-suite/pcd-code.zip"
