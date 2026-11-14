# Decision 0002 — Weekend queue revision

- **Status:** Adopted for the remaining workstation window.
- **Date:** 2026-11-14
- **Scope:** Scheduling and replication allocation only; existing experiment IDs and spec contents remain stable.

## Context

One weekend remains on the GPU workstation, and all results collected so far use a single seed. The
proposal promised five seeds and paired differences for teacher corruption. The survey's evidence
bar is a tuned weighted-sum sweep, a plug-in comparison on a strong logit loss, and one rule from
each method camp. Completing one seed of every family would leave the paper's central comparisons
under-replicated.

The `screen` profile therefore gives five seeds to teacher corruption, three to label noise and
`core-p0`, two to `lit-p1`, and otherwise retains its existing cap. No family's declared `seeds`
value changes. `lit-p1` gains `pcd_t2.0`: core-p0 validation selected τ = 2.0 (val top-1 55.66,
versus 55.00 at τ = 0.5 and 54.62 at τ = 1.0; τ = 0.1 was approximately CE). `lit-p1` has no
validation set, so it cannot tune τ itself.

## Decision

Use the ordered blocks in `suite.WEEKEND_PLAN`:

1. `lit-p1`, seed 0, its original nine arms.
2. `pilot`, seed 0.
3. `mve-cifar10`, seed 0.
4. `core-p0`, seed 0.
5. `label-noise`, seed 0.
6. `teacher-corruption`, seed 0, all conditions.
7. `teacher-corruption`, seeds 1–2, conditions ρ = 0.0, 0.1, 0.3.
8. `lit-p1`, seeds 0–1, arms CE, WKD 0.9, NWKD 1.0, PCD τ = 2.0, PCGrad, and DOT.
9. Seed 0 of `temperature`, `multi-loss-dkd`, `pcd-ablations`, and `core-p1`.
10. `sensitivity`, seed 0.
11. `teacher-corruption`, seeds 3–4, conditions ρ = 0.0, 0.1, 0.3.
12. `label-noise`, seed 1.
13. `core-p0`, seeds 1–2, then `mve-cifar10`, seed 1.
14. `label-noise-noisy-teacher`, seed 0.
15. `label-noise`, seed 2.

Within each block, runs follow design order after sorting by seed. Unmatched tasks are retained once
in the legacy `(seed, PRIORITY rank, design position)` order. The first matching block owns any task
matched more than once.

## Consequences

- Deferred after the listed blocks: `loss-scale`, `tau-sweep`, `low-data`, `multi-teacher`,
  `multi-loss-at`, the remaining `capacity-gap` and `long-tail` tasks, teacher-corruption ρ = 0.5
  and 0.7 beyond seed 0, and other seed-1 work. The deferred tail still runs if earlier blocks
  finish.
- This schedule deviates from the pre-registered plan. Report the deviations explicitly; the
  resulting subset and replication are not a completed five-seed confirmatory study.
- The PCD τ = 2.0 choice comes from core-p0's validation split. `lit-p1` has no validation split,
  and its results are not used to tune that value.
- `screen` seed overrides and the added `lit-p1` arm produce new spec IDs for added runs. Existing
  IDs and spec contents must remain unchanged so checkpoint resumption remains valid.
