# AML-Code-and-Paper

MLR570 (Advanced Machine Learning) project: a survey paper on knowledge distillation and
gradient-conflict / priority-constrained optimization, and an ML research project built around
Priority-Constrained Descent (PCD).

## Layout

| Path | Contents |
|---|---|
| `Project Files/` | Original course materials and the survey draft — **read-only** |
| `docs/requirements/` | Assignment requirements and project objectives extracted from `Project Files/` |
| `docs/decisions/` | Research and engineering decision records |
| `docs/reviews/` | Cross-agent review evidence and unresolved disagreements |
| `literature/` | Source registry (`sources.yaml`), reading notes; local PDFs (git-ignored) |
| `research/` | Research notes, candidate directions, experiment plans |
| `paper/` | Survey paper revisions (built from the preserved draft) |
| `src/`, `tests/`, `configs/`, `scripts/` | Code, tests, experiment configs, utility scripts |
| `data/`, `outputs/` | Local datasets and run outputs — git-ignored |

## Working here

- Agents: read `AGENTS.md` (shared rules; `CLAUDE.md` imports it).
- Orca worktree setup is defined in `orca.yaml` (runs `uv sync --frozen` once a Python project exists).
- Python environment: `uv sync`, tests: `uv run pytest` (once `pyproject.toml` exists).
- Experiments: the full PCD-in-KD suite (families, arms, GPU run order, analysis) is in
  `docs/experiment-suite.md`; `uv run python scripts/run_suite.py list` shows its size.
