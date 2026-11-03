# Phase 7 review summary and coordinator dispositions

Run `run_8e2818695e01`. The target is the idea report, proposal and plan on `Battikhah/research-kd-sota`.

| Cycle | Reviewer | Score | Verdict |
|---|---|---|---|
| 1 | Codex, `task_e32b7cee9b3a` | 4/10 | ABANDON in current form, with 6 minimum fixes |
| 2 | Codex, `task_71c07d07ef05` | 7/10 | PROCEED, conditional on fix 5 |

## Unresolved after cycle 2 (maximum two cycles, AGENTS.md §5): fix 5, statistical protocol
- **Codex position:**
  - The report said "95% CI" while the plan said "90% TOST".
  - A pilot-variance gate built on three runs is unreliable.
- **Coordinator disposition, 2026-11-03: AGREE. Both changes are applied in this commit.**
  - (a) All documents now specify a 90% TOST interval (two one-sided tests at α = 0.05 each), δ = 0.5 points.
  - (b) The pilot-variance gate is removed. Equivalence is an always-reported *secondary* analysis, read against the realised SD of the paired differences from the 10 confirmatory pairs. The pilot SD is descriptive only.
  - The confirmatory family stays the three Holm-corrected superiority contrasts.

## Open items before freezing the preregistration (from fix 4, PARTIAL)
- Pin the mdistiller commit SHA and the teacher-checkpoint hash.
- Pin the KD loss weights and temperature, and the EMA β.
- Check RLD's licence before the I4 follow-up.
