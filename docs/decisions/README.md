# Decision records

One file per decision: context, evidence, decision, consequences. The story that connects them is in [`docs/project-story.md`](../project-story.md).

| # | Date | Decision | Triggered by |
|---|---|---|---|
| [0001](0001-research-objective.md) | 2 Nov | Test PCD against its own fixed-weight version (NWKD), under a degrading teacher; pre-register hypotheses H0–H4 | PCD's closed form for two losses is a weighted sum whose KD weight grows with conflict |
| [0002](0002-weekend-queue-revision.md) | 14 Nov | Extra seeds for the central families (teacher corruption 5, label noise 3, core-p0 3, lit-p1 2); queue them first | All results single-seed, one weekend of GPU time left |
| [0003](0003-high-rho-seeds.md) | 14 Nov | Move the extra seeds to high corruption (ρ0.5/0.7); defer DTO-KD | CE noise sd 0.46 pp; methods differ by 15 pp only at high ρ; DTO-KD below CE |
| [0004](0004-layerwise-pcd-and-priority-queue.md) | 15 Nov | Add per-layer PCD (`pcdl`) and per-layer PCD on momentum buffers (`pcdlm`) | PCD behaving like NWKD at a matched weight |
| [0005](0005-extended-access-plan.md) | 18 Nov (reranked 21 Nov) | Use the extra access nights for matched fixed weights (NWKD 0.25/0.75), PCGrad seeds and PCDLM on clean benchmarks | Lab extended remote access; no fixed weight at the values PCD uses |
| [0006](0006-why-pcd-ties-a-fixed-weight.md) | 22 Nov | Gated seeds at high ρ and a `teacher-switch` family, run first | Matched weights tie PCD; PCD's weight barely moves within a run |
