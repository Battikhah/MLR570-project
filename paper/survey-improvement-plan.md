# Survey improvement plan (tied to `Survey_Guideline.pdf`)

The original draft `Project Files/Current_Survey.md` (commit `07af803`) is **preserved unchanged**. Revisions will be
made in `paper/` as a new manuscript that carries over every section, claim, and attribution of the draft, then
extends it. Requirement IDs G1–G11 refer to `docs/requirements/assignment-requirements.md`.
Status of each draft gap is VERIFIED from Orca Run `run_1a90ee37419e` unless marked otherwise.

## 0. Open questions for the instructor (blocking format decisions)

Length / page limit, deadline, grading rubric, whether IEEE style is mandatory ("such as IEEE"), whether per-claim
citations are expected, template (Markdown vs LaTeX IEEEtran — the sample is IEEE two-column).

## 1. Requirement-by-requirement plan

| Req | Draft status | Planned change | Sources (registry ids) |
|---|---|---|---|
| G1 Title | PARTIAL (generic, L1) | Make the focus explicit: teacher reliability, gradient conflict, priority-constrained optimization in KD | — |
| G2 Abstract | MISSING (L7 placeholder) | Write last: motivation, contributions, taxonomy, key trends | — |
| G3 Problem / trends | PARTIAL (L12–15) | Add recent trends: conflict-aware KD (2021–2026), priority/hierarchical MOO | hayder2026dtokd, chen2026scope, yuan2026tpkd, varam2026pcd |
| G4 Survey motivation | MISSING | State the gap: no survey links KD reliability, gradient-conflict MTL methods, and priority-constrained descent | gap-analysis §2 |
| G5 Scope / dimensions / outline | MISSING | Explicit taxonomy axes: (i) what is transferred, (ii) reliability signal, (iii) how conflict is resolved (weighting / projection / constraint / priority); roadmap paragraph | — |
| G6 Methods + comparisons | MET per method; comparisons mostly to the team's own proposal | Add method-to-method comparisons; expand PCGrad/CAGrad beyond abstract-level reading (L119, L126); add MGDA, Nash-MTL, DBGD, GEM lineage | yu2020pcgrad, liu2021cagrad, sener2018mgda, navon2022nashmtl, gong2021dbgd, lopezpaz2017gem |
| G7 Visual aids | MISSING | Fig. 1 taxonomy; Fig. 2 geometry of PCD vs PCGrad vs weighted sum (2-D gradient diagram); Table: method comparison built from the draft's per-method fields (reliability signal / supervision / update mechanism / strength / limitation) | — |
| G8 Critical analysis / open problems | MET, uneven | Add the verified critical points: PCD's scale-not-conflict finding, no convergence theorem, momentum sign flips, PCD ≡ adaptive weighted sum (λ_t = τ − c_t), FedPCD baseline/code gaps, scalarization counter-evidence | varam2026pcd, irshaid2026fedpcd, xin2022mtohelp, kurin2022unitary, royer2023scalarization |
| G9 Future directions | PARTIAL (caveats, L191–195) | Convert caveats into concrete directions grounded in gap-analysis §§1–2 | gap-analysis |
| G10 Conclusion | MISSING | Add | — |
| G11 References | MISSING (49 inline links, 28 unique URLs) | Build a numbered IEEE-style list from `literature/sources.yaml`; every draft URL registered and verified first | registry |

## 2. Draft-specific fixes (VERIFIED issues)

1. Register and verify **all 28 unique URLs** of the draft in `literature/sources.yaml` before citing; several draft
   sources (e.g. CCKD, Hard-Gate KD, CA-MKD, label revision, refined-logit KD, capacity-gap and fidelity papers) are
   not yet in the registry. Each gets `status: unverified` until checked.
2. Resolve the draft's verification notes with registry evidence: PCD TMLR status now VERIFIED (L185); FedPCD (L195)
   must be introduced and cited (`irshaid2026fedpcd`, preprint, code not publicly released as of 2026-11-02).
3. Remove research-process wording (L57, L77, L89, L139, L149, L185) after resolving each item; keep uncertainty
   labels where verification is incomplete.
4. Resolve the SCKD threshold issue (L139) — note the paper's Table 6 reports thresholds > 1 (zhu2021sckd).
5. Use "negative transfer" (proposal §2) and connect the survey to the project's research question.
6. Use one name for SCOPE (draft says "Medical SCOPE"; registry id chen2026scope).
7. Convert bullets to prose; keep every original claim and attribution.

## 3. Workflow

Survey work is split by section across Orca workers with exclusive file ownership (`paper/sections/*.md`), each
section cross-reviewed by the other agent against the guideline row above and the registry. Reference list generated
from the registry. No new citation is added without a registry record; nothing is cited as published unless VERIFIED.
