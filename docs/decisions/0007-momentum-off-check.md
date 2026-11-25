# Decision 0007: PCD against matched fixed weights with optimizer momentum off

Date: 25 November 2026

## Context

A three-part independent review (code, statistics, claims) of the final screen set found no bug, and found that the main result holds. The open risk it named is the optimizer:
- The PCD paper trains with Adam. Its appendix G reports that SGD with momentum flips the sign of the realised secondary progress on many steps (`literature/notes/varam2026pcd.md`).
- Every run in this project uses SGD with momentum 0.9, which averages the update over roughly 10 steps. That could smooth away PCD's per-step adaptivity.
- A PCD author would therefore say that the tie with a fixed weight is partly built in. No run answers this: the only momentum-off arms are `pcd_t0.5_mom0` and `nwkd_l1.0_mom0` in `pcd-ablations`, with 1 seed and no matched pair.

The lab workstation's access window has closed. The team has Colab compute units, so the check runs on Colab (`scripts/colab/`).

## Decision

- **Three new arms with optimizer momentum 0:** `pcd_t1.0_mom0`, `nwkd_l0.5_mom0` and `nwkd_l0.75_mom0`. They are appended last to `teacher-corruption` and `teacher-switch`, so no existing spec ID changes.
  - The fixed weights 0.5 and 0.75 bracket PCD τ1.0's average weight (0.58 in the switch, 0.42–0.75 under constant corruption).
  - The learning rate stays at 0.05, as in the existing `_mom0` ablation, so all three arms share the same training conditions.
- **Queue order,** first in `EXTENDED_PLAN`:
  1. teacher-switch at ρ0.7, seeds 0–2 (9 runs);
  2. the clean teacher, seeds 0–2 (9 runs);
  3. the rest of teacher-switch, including the 23 missing seeds and the momentum-off arms at ρ0.5 and seeds 3–4;
  4. constant ρ0.7, seeds 0–2 (9 runs).
- **Adam is not added.** It would need a change to the training loop. Momentum off already tests the paper's main warning.

## What the result will mean (HYPOTHESIS)

- **If PCD τ1.0 still ties the bracketing fixed weights without momentum:** "PCD acts like a fixed weight" does not depend on momentum, and the objection is answered.
- **If PCD beats them only without momentum:** momentum was hiding PCD's adaptivity. That is a new, positive finding, and the paper must report it.

## Consequences

- The momentum-off runs will be less accurate overall. The `_mom0` ablation lost 3–4 pp at the same learning rate, so only within-regime comparisons are meaningful.
- **Not yet done:** a cross-agent review of this change.
