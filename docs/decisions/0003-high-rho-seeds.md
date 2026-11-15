# Decision 0003 — Seeds for teacher corruption at high ρ

- **Status:** Adopted for the remaining workstation window; block order amended by decision 0004.
- **Date:** 2026-11-14
- **Scope:** Scheduling only; it amends the block order of decision 0002. Spec IDs and contents do not change.

## Context

The first 291 screen runs (seed 0, plus seed 1 for part of teacher corruption) showed three things:

- Five CE runs in teacher-corruption share one spec and seed (CE does not use the teacher), yet
  their test top-1 spans 54.82–56.15 (sd 0.46) from GPU nondeterminism. Single-seed gaps below
  about 1.5 pp are not interpretable.
- The largest effect is at high teacher corruption: from ρ = 0.0 to 0.7, DOT loses 17.1 pp and
  WKD 0.9 loses 15.6 pp, while PCD τ = 1.0 and NWKD λ = 0.5 lose 1.9 pp each (seed 0).
  Decision 0002 gave extra seeds only to ρ = 0.0, 0.1 and 0.3, where arms differ by less than the noise.
- DTO-KD is below CE in almost every cell (effective weight 50–180), which needs an implementation
  audit; more seeds of it would not change a conclusion.

At about 15 runs per hour, the 0002 plan would not finish before the window closes.

## Decision

Changes to `suite.WEEKEND_PLAN`:

1. New block right after teacher-corruption seed 0: seeds 1–2 at ρ = 0.5 and 0.7 for
   `TC_CORE_ARMS` (CE, PCD τ = 0.5 and 1.0, NWKD λ = 0.5 and 1.0, WKD 0.9, DOT, PCDrev τ = 0.5).
2. Teacher-corruption seeds 3–4 now cover ρ = 0.0, 0.5 and 0.7 for `TC_CORE_ARMS` only (48 runs
   instead of 84), and move ahead of `core-p0` seeds 1–2. With seeds 0–4 this gives five paired
   seeds for the robustness and PCD-vs-NWKD comparison.
3. `core-p0` seeds 1–2 and `label-noise` seed 1 move ahead of `sensitivity`.
4. Blocks that select every arm skip `DEFERRED_ARMS` (`dtokd`); those runs go to the tail.

Everything not selected still runs in the tail, in the legacy order.

## Consequences

- Teacher-corruption ρ = 0.1 and 0.3 get three seeds (0–2), not five; the other TC arms at
  ρ = 0.5/0.7 stay at one seed unless the tail reaches them.
- DTO-KD results outside `lit-p1` stay single-seed. Audit it before reporting it.
- This is a further deviation from the pre-registered plan and must be reported with 0002.
