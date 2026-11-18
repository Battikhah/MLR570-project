# Decision 0005 — Extended access (18–23 Nov 2026): nightly windows and the follow-up runs

- **Status:** Adopted. Amends decision 0004: its blocks keep their order and now run after the blocks below.
- **Date:** 2026-11-18
- **Scope:**
  - The access-window guard (`experiments/window.py`) and the dashboard's "next window".
  - Eight new arms appended to five families.
  - A `core-p1` seed override.
  - `suite.EXTENDED_PLAN`, which now opens `suite.WEEKEND_PLAN`.

  Existing spec IDs and contents do not change: 0 of the 1,837 screen specs and 0 of the 2,074 full specs differ, checked against the previous commit.

## Context

- **The lab extended remote access (email, 2026-11-18).** Until Monday 23 November 08:00, the workstation may also be used every night from 20:00 to 08:00, in addition to the usual Thursday 18:00 – Monday 08:00 window. Use outside these hours is monitored and can cost us the access. The only extra night still ahead is Wednesday 18 → Thursday 19 November. Every other night until Monday falls inside the weekly window.
- **GPU time available:** about 11 h tonight plus about 85 h from Thursday 18:00 to Monday 07:40, roughly 96 h. That is about 1,250 runs at the measured 13 runs/h (3 processes, 60-epoch ResNet-8 runs).
- **The first weekend's results** (655 runs; summary in `research/results/screen-2026-11-16/`):
  - **VERIFIED, 5 seeds:** PCD stays at the CE level under teacher corruption while WKD and DOT lose 15–17 pp.
  - **INTERPRETATION:** PCD's effective KD weight (λ_eff 0.01–0.75) explains the result as well as its adaptivity does. NWKD at a fixed λ = 0.5 is as robust (PCD τ0.5 − NWKD 0.5 = +0.30 [−0.20, +0.79] at ρ = 0.7). We have no fixed-weight arm at the other λ values PCD actually uses.
  - **Too few seeds:** PCGrad, the strongest clean-teacher competitor, has 1 seed at ρ = 0.5 and 0.7.
  - **PCDLM** (decision 0004) was +1.05 [+0.29, +1.80] over PCD on the clean teacher, but was only run in `teacher-corruption`.

## Decision

### Access window

`window_close(now)` returns the weekly close (Monday 08:00) inside the weekly window. Between `NIGHTLY = (2026-11-18 20:00, 2026-11-23 08:00)` it also returns 08:00 the next morning for any time from 20:00 to 08:00. Outside both it returns None, so the runner refuses to start.

`next_open(now)` gives the dashboard the start of the next window. The stop margin (20 min plus 1.5 epochs) is unchanged, so a night session pauses with a checkpoint by about 07:40 and the last Drive upload finishes before 08:00.

### Arms (appended last to each family's list)

| Family | New arms | Why |
|---|---|---|
| `teacher-corruption` | `nwkd_l0.25`, `nwkd_l0.75` | With the existing 0.5 and 1.0, a fixed-weight curve over the λ_eff range PCD uses (0.01–0.75). It tests whether PCD's per-step weight beats the fixed weight it matches. |
| `core-p0`, `core-p1` | `pcdlm_t0.5`, `pcdlm_t1.0` | PCDLM on the clean benchmarks, next to the tuned weighted sums |
| `lit-p1` | `pcdlm_t1.0` | PCDLM on the literature-matched pair, where DOT reproduces 74.87 vs the published 75.12 |
| `label-noise` | `pcdlm_t1.0` | PCDLM uses more KD; label noise is where more KD helps |

`PROFILES["screen"]["seed_overrides"]` adds `core-p1: 3`.

### Queue: `EXTENDED_PLAN` first, then decision 0004's blocks unchanged

| Block | Selection | Runs left | ~GPU h |
|---|---|---|---|
| 1 | teacher-corruption s0–4, all ρ, `nwkd_l0.25`, `nwkd_l0.75` | 50 | 3.8 |
| 2 | teacher-corruption s0–4, all ρ, `pcgrad1s` | 14 | 1.1 |
| 3 | core-p0 s0–2, `pcdlm_t0.5`, `pcdlm_t1.0` | 6 | 0.5 |
| 4 | lit-p1 s0–1, `pcdlm_t1.0`, `pcd_t0.5`, `pcd_t1.0`, `mgda` | 5 | 1.7 |
| 5 | label-noise s0–1, `pcdlm_t1.0` | 14 | 1.1 |
| 6 | teacher-corruption s3–4, ρ0.1/0.3, `TC_CORE_ARMS` | 32 | 2.2 |
| 7 | core-p1 s0–2, every arm except DTO-KD | 32 | 2.2+ |
| 0004 (g) | label-noise s1 rest, sensitivity s0, mve-cifar10 s1, label-noise-noisy-teacher s0, label-noise s2 | 33, 84, 12, 24, 91 | 16.9 |
| tail | everything else, in the old order | 887 | ~64 |

Runs left are counted against the 655 results on Drive. GPU hours use the measured pace of a 60-epoch ResNet-8 run. The ResNet-8x4 families (`core-p1`, `lit-p1`) are slower than this estimate. In total, about 94 h are queued against about 96 h available, so most of the tail may also run. What goes unrun comes from the end of the tail.

## Consequences

- The matched-weight comparison (block 1) is the run a reviewer will ask for. If NWKD at a matched λ equals PCD, the paper reports PCD's robustness as coming from its bounded weight and not from conflict detection.
- `dtokd` stays out of every block. Its results sit below CE in every family (for example 50.1 vs 55.4 on core-p0), which points to an implementation problem. It still runs in the tail, and nothing from it is reported until the rule (`MinNorm(log_loss=True)`, `optim/combiners_extra.py`) has been checked against the paper's Eqs. 11–12 and its step size.
- After 23 November only the weekly window applies again. `NIGHTLY` is a single fixed period, so a further extension needs a new entry.
