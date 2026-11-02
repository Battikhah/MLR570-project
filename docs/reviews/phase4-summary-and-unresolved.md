# Phase 4 cross-review summary and unresolved disagreements

Orca Run `run_f9e2e8953f54`. Two review rounds (the maximum). Full texts:
`docs/reviews/phase4-claude-reviews-codex.md`, `docs/reviews/phase4-codex-reviews-claude.md`.

| Task | Agent | Dispatch | Commit (worker branch) |
|---|---|---|---|
| Literature & directions | Claude | ctx_c6a4e1da51a1 | 1bec167 |
| Math, code audit, feasibility | Codex | ctx_b0909bf78f3b | d1cd580 |
| R1: Claude reviews Codex + PCD code audit | Claude | ctx_d90b9a741091 | 3bd15fe |
| R1: Codex reviews Claude + attacks D2 | Codex | ctx_5476083f8722 | bb005c4 |
| R2: Claude responds + attacks S | Claude | ctx_48436bd229a8 | 67c4359 |
| R2: Codex responds + attacks S | Codex | ctx_e3c6794e3bf1 | e893290 |

## Resolved by evidence (selection)

| Issue | Resolution |
|---|---|
| PCD venue | Published, TMLR 2026 (TMLR list, arXiv journal_ref; coordinator re-checked arXiv API) |
| FedPCD code | Not publicly released as of 2026-11-02 (`js/main.js` `code: ""`; README "Coming soon") |
| "Normalized-WS control untested" (Claude) | Narrowed: App. E.2 tested EMA-normalized baselines at w ≥ 0.1; the missing control is low-weight, rescale-matched |
| PCD ≈ weighted sum when c ≈ 0 | Exact active-case form `a + (τ − aᵀb/‖b‖²) b`; App. E.3 Table 10 confirms empirically in compression |
| Undisclosed proximal step | Repo README/scripts assert it; paper HTML has no disclosure for PCD (Codex keyword audit). Baseline treatment unknown |
| Codex compute estimates | Withdrawn/corrected (throughput unsupported; compression overhead < 2% on A10G; FedPCD 5–10 clients/round) |
| Table 3 reading | 44 wins / 4 losses / 54 ties of 102; one genuine compression-regime loss (53.8% vs 51.9%) |
| D5 scooping risk | Withdrawn to "overlap risk"; GAQAT registered as prior gradient-conflict QAT |
| Candidate S control | "λ matched to τ" replaced by budget-matched tuned normalized WKD with grid containing τ − c̄ |

## Unresolved after two rounds (escalated)

1. **S vs M1 as top-1.** Both total 25/35 on Claude's scale; Codex scores S 23–24 vs D2 25. Tie broken by assignment
   fit (coordinator decision 0001); instructor stance on scope unknown.
2. **Whether full-data CIFAR-100 CE–KD cosine is materially negative.** No source measures it — resolved only by the MVE.
3. **M4/MPS throughput.** Neither agent measured it — resolved by the smoke test.
4. **Whether PCD's paper baselines also received the proximal step.** Requires the authors.
5. **OpenReview reviews for PCD (HT01yGHLEt).** Blocked by bot challenge (HTTP 403).
6. **Dataset-content licences** (CIFAR: no licence statement, citation requested; Fashion-MNIST: MIT text covers software).

## Process deviations (recorded)

- Codex's first Phase-3 `worker_done` was a summary; full content was in its committed files (read by coordinator).
- Claude sent one Crossref request containing the user's email in a `mailto` parameter; self-reported, not repeated.
- Codex's environment could not resolve github.com (clone failed); the code audit was performed by Claude instead.
