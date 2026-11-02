# PCD / FedPCD — literature synthesis and research gaps

Date: 2026-11-02. Author: Claude literature worker (Orca task_94c59c478ef4). Source ids refer to
`literature/sources.yaml`; detailed reading notes are in `literature/notes/varam2026pcd.md`,
`irshaid2026fedpcd.md` and `lopezpaz2017gem.md`.

Labels: **VERIFIED** (checked against a primary source, cited), **INTERPRETATION**, **HYPOTHESIS**,
**RECOMMENDATION**. Evidence types: **[THEORY]** proven guarantee, **[EMPIRICAL]** reported experimental
evidence, **[UNTESTED]** hypothesis nobody has tested.

---

## 1. Status of the two core papers

| | PCD (`varam2026pcd`) | FedPCD (`irshaid2026fedpcd`) |
|---|---|---|
| Authors | Dara Varam, Mohamed I. AlHajri (equal), AUS (+MIT) | Yousef Irshaid, Dara Varam (equal), Mohamed I. AlHajri, AUS (+MIT) |
| Status | **Published, TMLR, Sept 2026** (VERIFIED: TMLR list + bib, arXiv journal_ref, PDF header) | **arXiv preprint**, v1 2026-10-09; no venue found (VERIFIED: arXiv) |
| arXiv | 2606.29521 (v1 2026-06-28, v2 2026-09-21) | 2610.13508 (v1) |
| OpenReview | forum HT01yGHLEt (listed by TMLR; page itself blocked by a bot challenge) | none found |
| Code | github.com/DaraVaram/priority-constrained-descent (MIT; author-linked by README and project page; **not linked from the paper PDF**). Static audit in literature/notes/varam2026pcd.md: a rewrite of the research code, PCD only (no baselines), no physical-pruning export, published runs used an undisclosed proximal step. | **none released**: project-page Code button is `code: ""`, "Coming soon" (VERIFIED in the cross-review) |

VERIFIED corrections to the survey draft:
- The PCD TMLR status (draft L185, "not directly verified") is now verified.
- FedPCD (draft L195) exists and should be cited as an arXiv preprint.

## 2. Theoretical motivation

### 2.1 What problem PCD solves (VERIFIED, varam2026pcd Sec. 1, 3–4)

PCD targets objective sets with a hierarchy: one primary deliverable (task loss) and secondaries that shape the
solution. Weighted sums encode importance through scale-dependent weights (Cor. B.10: WS is not scale invariant).
Symmetric MOO directions (MGDA, PCGrad, CAGrad) are permutation-invariant in the objectives and can stall at
Pareto "cancellation" equilibria. PCD instead projects the normalized primary gradient onto the half-spaces
{d : g̃_jᵀd ≥ τ‖g̃_j‖²}.

### 2.2 Guarantees and their conditions

- **[THEORY]** The direction gives per-step, tolerance-controlled first-order progress on every secondary
  (Prop. 4.4). Conditions: feasibility (always holds for K = 2) and τ > 0. Strict decrease needs locally smooth
  secondaries; in the convex nonsmooth case it holds up to a "kink tax".
- **[THEORY]** Inactive regime: when g̃1 already satisfies the constraints, PCD is plain GD on L1 (Cor. 4.3).
- **[THEORY]** The normalized direction vanishes only at composite multi-stationary (CMS) points (Thm 4.8, τ > 0).
  This is a pointwise endpoint result and explicitly not a stability or convergence result.
- **[THEORY]** Asymptotic scale invariance through EMA normalization (Thm B.9).
- **[THEORY], negative:** for K = 2 the primary descends to first order only if 1 + τc − c² > 0 (Prop. 4.7).
  Under strong conflict, PCD can increase the primary loss to first order.
- **Not guaranteed (VERIFIED, Sec. 4.7–4.8, 6):**
  - No convergence theorem or rate for Algorithm 1.
  - Guarantees cover the direction, not the realized Adam/momentum step.
  - The deployed rescale has non-CMS fixed points at g1 = 0.
  - On genuine trade-offs with no common stationary point, "the PCD direction with τ > 0 has no fixed point";
    the method "is best understood as targeting the regularizer-style hierarchy".
  - Assumption 3.1 requires **convex** secondaries; adversarial and other non-convex secondaries are outside the proofs.
- **Ancestry (VERIFIED, authors' own statement):**
  - For K = 2 the QP equals Gong et al.'s dynamic-barrier GD on raw gradients (`gong2021dbgd`, which *does* have
    convergence proofs).
  - GEM/A-GEM are "algorithmically equivalent to PCD with τ = 0".
  - ε-SOFT-GEM (`hu2020softgem`) already adds a unit-normalized margin at K = 2.
  - Pareto Navigation GD (`ye2022pngd`) is an anchored general-K program.

  INTERPRETATION: PCD's contribution is scale-normalized, per-objective margins with one interpretable τ, the
  general-K analysis, and the empirical study. It is not the projection itself.

### 2.3 What FedPCD adds (VERIFIED, irshaid2026fedpcd Sec. 3–4)

- The primary is the FedAvg aggregate; the secondaries are the individual client pseudo-gradients, plus a cache of
  recently seen clients (history H) to cope with partial participation.
- It uses instantaneous ℓ2 normalization, because EMA statistics go stale for rarely sampled clients.
- When FedAvg already satisfies every client constraint, the update is exactly FedAvg.
- **[THEORY]** O(1/T) rate on ‖∇F‖² and on the weighted client gradient norms, so accumulation points are CMS.
  Assumptions: full participation, deterministic local GD, H = 0, L-smooth clients, and an assumed **uniform cosine
  margin γ** between client gradients and the QP output with γ_eff = γ(1−r) − r > 0 under local drift.
  INTERPRETATION: this margin is an assumption about the algorithm's own trajectory. It is not implied by data
  heterogeneity and can fail exactly when heterogeneity is severe.
- Heterogeneity and drift enter the theory only through r. The algorithm has no drift correction (no control
  variates as in SCAFFOLD), and many-local-step regimes are explicitly untested (authors' limitation).
- Fairness here means client-level progress protection with the global objective as priority. The paper
  contrasts this with q-FFL/AFL re-weighting and with FedMGDA+'s symmetric common descent.

## 3. Related literature map

Benchmarks are listed only where I checked them; "n.c." = not checked in this pass.

### 3.1 Gradient-conflict / multi-task optimizers (all symmetric unless noted)

| id | Mechanism | Venue (status) | Benchmarks |
|---|---|---|---|
| `sener2018mgda` | min-norm convex combination (MGDA-UB) | NeurIPS 2018 | n.c. |
| `yu2020pcgrad` | project away pairwise conflicts | NeurIPS 2020 | n.c. |
| `liu2021cagrad` | average-anchored, worst-case improvement | NeurIPS 2021 | n.c. |
| `liu2021imtl` | equal-projection gradient + loss balancing | ICLR 2021 | Cityscapes, NYUv2, CelebA (from search snippet; re-check) |
| `navon2022nashmtl` | Nash bargaining | ICML 2022 | n.c. |
| `senushkin2023alignedmtl` | condition-number alignment | CVPR 2023 | n.c. |
| `liu2023famo` | O(1) adaptive loss weighting | venue unverified | n.c. |
| `chen2018gradnorm`, `kendall2018uncertainty` | gradient-norm / uncertainty weighting | ICML 2018 / CVPR 2018 | n.c. |
| `lin2019paretomtl` | preference-vector Pareto MTL | NeurIPS 2019 | n.c. |
| `du2018auxgradsim` | **asymmetric**: main-task cosine gating of auxiliary losses | arXiv | n.c. |
| AuxiNash (Shamsian et al., ICML 2023; in survey draft, not re-registered) | **asymmetric** bargaining | — | — |

### 3.2 Lexicographic / hierarchical / constrained

| id | Notes |
|---|---|
| `gong2021dbgd` | dynamic barrier; lexicographic and constrained; convergence proofs (raw gradients). K = 2 PCD ancestor. |
| `ye2022pngd` | Pareto navigation: anchored, general K, scheduled margin. |
| `skalse2022lmorl` | strict lexicographic multi-objective RL (IJCAI 2022). |
| `chamon2020pacc`, `cotter2019twoplayer`, `gallegoposada2025cooper` | primal–dual / Lagrangian constrained learning: constraints on *levels* (L_j ≤ c), not per-step progress. Cooper is a PCD baseline (Sec. 5.1.2). |
| `agarwal2018reductions` | fairness-constrained classification via reductions. |
| `sagawa2020groupdro` | worst-group robustness (min–max over groups). |
| `lopezpaz2017gem`, `chaudhry2019agem`, `hu2020softgem`, `farajtabar2019ogd` | continual-learning projections. GEM uses the same QP family (see §2.2). |

### 3.3 Federated heterogeneity and fairness

| id | Role | In FedPCD's comparison? |
|---|---|---|
| `mcmahan2017fedavg` | FedAvg | yes (primary direction) |
| `li2020fedprox` | proximal drift control | **no** |
| `karimireddy2020scaffold` | control-variate drift correction | **no** (cited only) |
| `wang2020fednova` | normalized averaging | **no** |
| `acar2021feddyn` | dynamic regularization; source of the Dirichlet split protocol | protocol only |
| `li2020qffl`, `mohri2019afl`, `li2021ditto` | fairness by re-weighting, minimax, personalization | yes |
| `hu2022fedmgda` | symmetric MOO aggregation | yes |
| `wang2021fedfv` | PCGrad-style client projection + history | yes (a strongest baseline) |

### 3.4 KD gradient-conflict work (from the survey draft, metadata re-verified)

- `zhu2021sckd`: ICCV 2021; supervised-primary gradient-cosine gating of KD terms.
- `hayder2026dtokd`: ICLR 2026; dynamic MOO trade-off.
- `chen2026scope`: J. Imaging 2026; classification-priority conflict removal plus confidence-weighted relations.
- `yuan2026tpkd`: arXiv 2026; task-margin-preserving projection.
- `sumit2026chad`: arXiv 2026; reliability-aware KD.
- `hinton2015kd`: the base KD loss.

INTERPRETATION: SCOPE and TPKD already implement "classification has priority, project away the conflicting KD
component". This is close to PCD-KD at τ = 0 or small τ.

## 4. Contradicting and cautionary evidence (actively sought)

1. **Scalarization is often enough [EMPIRICAL].** `xin2022mtohelp`, `kurin2022unitary` and `royer2023scalarization`
   report that well-tuned weighted sums match specialized MTL optimizers. `lin2022randomweighting` proposes random
   weighting as a litmus baseline. Any PCD claim must beat a *tuned* scalarization at equal search effort.
2. **PCD's own tuned-baseline study is moderate [EMPIRICAL, VERIFIED Table 3].** Over 102 matched comparisons:
   44 wins, 4 losses, 54 within noise. The dramatic Sec. 5.1.1 gaps rest on a WS grid the authors themselves say is
   2–4 orders of magnitude off.
3. **Mechanism is scale, not conflict [EMPIRICAL, VERIFIED Sec. 5.1.1].** In pruning, cos(task, reg) ≤ 1.2e-3 and the
   gain is attributed to normalization. INTERPRETATION: with c ≈ 0 and an active constraint, d̃* = g̃1 + τg̃2, which is
   a normalized weighted sum. **[UNTESTED]** the paper's normalized-baseline appendix (E.2) uses w ≥ 0.1, so a
   normalized WS at PCD's τ range (0.01–0.09) appears never to have been compared.
4. **Primary can be sacrificed [THEORY, Prop. 4.7]** under strong conflict, which is exactly the "unreliable
   teacher" regime the KD proposal targets.
5. **Realized steps differ from the guaranteed direction [EMPIRICAL, App. G].** Under SGD+momentum, the sign of
   secondary progress flips on 47–73% of steps. The KD proposal plans SGD without momentum, then with momentum.
6. **FedPCD evaluation gaps [INTERPRETATION].** Correction from the cross-review: at a fixed τ = 0.02 FedPCD still
   has the best global accuracy in 18/18 configurations and the best worst-10% accuracy in 15/18 (computed from
   Table 1), so τ selection alone does not create the headline. Remaining gaps:
   - No FedProx/SCAFFOLD/FedNova controls.
   - Larger hyperparameter grid for FedPCD (6τ × 11H) than for the baselines (≤ 3 values).
   - No validation split described for hyperparameter selection.
   - One local epoch only.
   - Low absolute CIFAR-100 accuracy.
   - No public code.
7. **KD-specific [EMPIRICAL, from the survey draft's verified sources]:** teacher accuracy is not transfer utility
   (Yuan et al. CVPR 2020), fidelity is not generalization (Stanton et al. NeurIPS 2021), and the recipe can dominate
   (Beyer et al. CVPR 2022). The draft also reports that CHAD found teacher-confidence-only gates gave little benefit.
8. **No independent replication** of PCD or FedPCD exists that I could find. Both come from the same group, and the
   only results available are the authors'. INTERPRETATION: an independent small-scale replication is itself a
   legitimate contribution.

## 5. Research gaps and unexplored applications

Criterion for "priority is the right formulation" (INTERPRETATION, from PCD Sec. 4.8 and Assumption 3.1): a natural
deliverable L1 exists; the secondary's *level* is not specified in advance (otherwise a Lagrangian level constraint
fits better); the secondary is regularizer-like with plausible joint stationarity; and the secondary's gradient
scale differs strongly from or drifts relative to L1's.

| Application | Primary | Secondary | Fit to PCD theory | Prior PCD-style work |
|---|---|---|---|---|
| Structured pruning / sparsity / low-rank | task CE | group lasso, ℓ1, nuclear norm | **good** (convex secondaries) | PCD paper itself, DBGD, Cooper/L0 |
| Quantization-aware training | task CE | distance-to-grid fidelity | partial (non-convex, periodic) | named as future work by PCD; no PCD paper found |
| Continual learning | new-task loss | replay/memory losses | partial (non-convex) | **GEM/A-GEM/ε-SOFT-GEM: essentially solved** |
| Federated fairness | FedAvg aggregate | client losses | partial (theory needs alignment assumption) | FedPCD (preprint) |
| Centralized group fairness / worst-group | average risk | per-group risks | partial (non-convex) | **none found** (novelty searches 2026-11-02) |
| Fairness surrogates (DP/EO gap) | task CE | fairness penalty | partial (non-convex) | none found; Lagrangian methods dominate |
| Adversarial robustness | clean CE (or robust loss) | TRADES/adv loss | **outside theory** (non-convex, stated) | CA-AT (`xue2024caat`) uses gradient-conflict projection, symmetric |
| KD | CE | gated KD-KL | partial; CE and KD typically have no common stationary point → "no fixed point" regime | SCKD, SCOPE, TPKD, DTO-KD are close |

Novelty searches run on 2026-11-02 (web search, arXiv API "priority-constrained", GitHub): I found no published
application of PCD to KD, fairness, robustness, continual learning or quantization. The only PCD-derived work found
was FedPCD. The authors' public posts (search-result snippet, unverified) name FL, continual learning, generative
modeling and quantization as planned directions, so **QAT and continual learning carry a high scooping risk from
the original group.**

## 6. What I could not verify

- RESOLVED in the cross-review: the FedPCD code link (empty, "Coming soon") and the PCD repo HEAD (e9afdbc, 2026-09-29).
- NEW (cross-review): whether the PCD *baselines* received the same proximal step that the published PCD runs used
  (the code README says the PCD runs did; the paper is silent).
- The OpenReview forum HT01yGHLEt (reviews, decision text, camera-ready supplementary). Every OpenReview request
  returned a bot challenge (HTTP 403).
- FedPCD venue and submission status (only arXiv v1 found). No FedPCD code.
- Whether the PCD GitHub code matches the paper (delegated to Codex). GitHub profile DaraVaram lists EPFL while the
  paper lists AUS/MIT; provenance rests on the README → arXiv link and the shared github.io project page.
- Venues for FAMO, FedNova, AFL, Chamon & Ribeiro, Cotter et al., Agarwal et al., OGD, GroupDRO, random weighting,
  CA-AT, Tiny Episodic Memories. DBLP and Semantic Scholar were rate-limited (HTTP 429), so these stay `unverified`.
- Benchmarks used by most MTL baselines (PCGrad, CAGrad, Nash-MTL, Aligned-MTL, FAMO). Not checked.
- The full IMTL author list (from search snippet only).
- The PCD v1 PDF (v1-vs-v2 differences inferred from the abs page and a mirror).
- Whether FedPCD tuned hyperparameters on test data (no validation split is described; needs code or authors).
- The PCD App. E.2 Table 9 rows were not transcribed. The claim that no normalized-WS point with w < 0.1 exists rests
  on the sentence "same hyperparameter grid as the original submission".
- DBGD, AuxiNash and Pareto Navigation GD full texts were not read in this pass (claims come from PCD's own
  related work and from the survey draft).
