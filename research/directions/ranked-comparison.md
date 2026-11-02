# Ranked comparison of candidate research objectives

Inputs: `research/directions/candidates-claude.md`, `research/notes/experimental-feasibility.md`,
and the two adversarial reviews in `docs/reviews/` (Orca Run `run_f9e2e8953f54`, two rounds).
Scores are **reviewer judgements (INTERPRETATION)**, 1–5, 5 = best, on seven criteria:
(a) theoretical compatibility with PCD/FedPCD, (b) potential measurable advantage, (c) novelty,
(d) benchmarks, (e) reproducibility, (f) feasibility on one Apple M4 (16 GB, MPS, ~13 GiB free) in one semester,
(g) suitability for the MLR570 assignment. Totals are unweighted.

## Merged candidate list

| ID | Objective | Merged from |
|---|---|---|
| **S** | KD with CE primary / KD secondary; teacher corruption as a stressor; PCD vs budget-matched tuned normalized weighted KD | Coordinator synthesis of M1's mechanism question + M3 |
| M1 | Compression/sparsity: PCD vs low-weight, EMA-normalized, rescale-matched weighted sum (independent replication, ± prox step) | Claude D2 + Codex objective 1 |
| M2 | FedPCD client-progress fairness with drift-aware baselines (FedProx, SCAFFOLD, FedNova) | Claude D3 + Codex objective 2 |
| M3 | Submitted proposal as written: reliability-gated PCD-KD, full method matrix | Claude D1 + Codex objective 3 |
| M4 | Group fairness / worst-group robustness ("groups as constraints") | Claude D4 + Codex objective 5 |
| M5 | Clean vs adversarial/corruption robustness | Claude D7 + Codex objective 4 |
| M6 | PCD for quantization-aware training | Claude D5 |
| M7 | Continual learning (normalized-margin GEM) | Claude D6 |

## Final scores (after round 2)

| ID | Claude a–g | Claude total | Codex a–g | Codex total |
|---|---|---|---|---|
| **S** | 3 4 3 4 3 3 5 | **25** | 2 4 2 5 3 2–3 5 | **23–24** |
| M1 | 5 2 2 4 4 4 2 | 25 (revised from 27) | 5 4 2 4 4 3 3 (D2) | 25 |
| M2 | 4 4 4 4 2 4 2 | 24 | 3 4 3 4 2 3 2 | 21 |
| M3 | 2 3 2 4 3 3 5 | 22 | 2 3 2 4 3 2 5 | 21 |
| M4 | 3 3 4 2 3 4 1 | 20 | 3 3 3 4 3 5 2 | 23 |
| M7 | 3 2 1 4 4 5 1 | 20 | 3 2 1 4 4 5 2 | 21 |
| M6 | 2 3 3 3 3 3 1 | 18 | 2 3 3 4 3 3 2 | 20 |
| M5 | 1 3 2 3 3 1 2 | 15 | 1 4 3 5 3 1 2 | 19 |

Codex also scored compression without the mechanism framing (C1: 5 4 1 5 5 3 2 = 25).

## Ranking and rationale

1. **S — selected (top-1).** Ties M1 on Claude's total and is within one point on Codex's; it wins decisively
   on (g) because it keeps the submitted research question (unreliable teachers). It is also the only candidate
   in which the projection-vs-normalization question is **not already answered by the PCD authors' own data**:
   PCD App. E.3 Table 10 shows that in compression the constraint is active on 100% of steps with cosine ≈ 0, so PCD
   behaves as a fixed normalized weighted sum (VERIFIED by both agents). Both agents independently recommend S as top-1
   after amendment.
2. **M1 — backup.** Strongest theoretical fit and official MIT code, but its outcome is largely predictable from
   Table 10 and it departs from the submitted proposal.
3. **M2** — strongest *published* evidence of a gain, but FedPCD code is not publicly released (as of 2026-11-02),
   so it needs a full re-implementation; weak assignment fit.
4. M3 as written is dominated by S (same question, unfocused matrix, mis-specified control).
5. M4, M7, M6, M5 — low assignment fit; M7 has near-zero novelty (PCD paper states GEM/A-GEM are algorithmically
   equivalent to PCD at τ = 0); M5 is outside PCD's theory and too expensive on MPS.

Selection was **not** based on hypothetical improvement: theory (PCD Cor. 4.6 / Prop. 4.7) predicts that PCD will
*not* help, and may hurt, under corrupted teachers (see `docs/decisions/0001-research-objective.md`). S is chosen
because it yields a falsifiable, theory-grounded comparison that fits the course, not because PCD is expected to win.
