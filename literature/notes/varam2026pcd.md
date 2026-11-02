# varam2026pcd — Priority-Constrained Descent (PCD)

Read 2026-11-02 by the Claude literature worker. Source: arXiv 2606.29521v2 PDF (52 pp.; header "Published in
Transactions on Machine Learning Research (09/2026)"), saved locally as `literature/pdfs/varam2026pcd.pdf`.
Section and page numbers refer to v2.

## Bibliographic status (VERIFIED)

- Title: "Not All Objectives Are Born Equal: Priority-Constrained Descent for Hierarchical Multi-Objective Optimization".
- Authors: Dara Varam, Mohamed I. AlHajri, equal contribution (AUS CSE; Varam also at MIT Senseable City Lab).
- Venue: **TMLR, September 2026**. The TMLR accepted-papers list (jmlr.org/tmlr/papers/) and TMLR bib
  `HT01yGHLEt.bib` both show it, with OpenReview forum `HT01yGHLEt`. The arXiv journal_ref reads "Transactions on
  Machine Learning Research 2026". The OpenReview page itself served a bot challenge, so I could not read the
  reviews or the decision text.
  → The survey draft's "TMLR status unverified" (L185) can now be resolved to VERIFIED.
- arXiv: v1 2026-06-28, v2 2026-09-21. The abstract on the arXiv abs page says "Pareto dominance" and "invariant to
  objective scaling"; the v2 PDF abstract says "stronger trade-offs" and "asymptotically invariant to objective
  scaling", and adds tuned application-specific baselines. A third-party mirror lists v1 at 33 pp.; the abs page
  comment says 52 pp. I did not download v1, so the v1-vs-v2 differences are inferred. **Cite v2 / the TMLR
  version, and quote the abstract from the PDF.**
- Code: `github.com/DaraVaram/priority-constrained-descent` (MIT license per the GitHub API; created 2026-09-21).
  The README links arXiv 2606.29521, and the project page daravaram.github.io/PCD is on the same account. The paper
  PDF has no code URL ("A more comprehensive reproducibility checklist will accompany this paper's code
  implementation", App. H). Codex is auditing the code; I did not.

## Problem (VERIFIED, Sec. 1, 3)

Deep-learning objectives are often hierarchical: a primary loss L1 (the task) plus secondaries L2..LK (sparsity,
low-rank, compression, robustness) that "constrain the solution". Weighted sums and symmetric MOO methods (MGDA,
PCGrad, CAGrad, FAMO, Nash-MTL) treat the objectives as interchangeable, which the authors call the "symmetry problem".

## Method (VERIFIED, Def. 4.1, 4.5, Alg. 1)

- EMA-normalize every gradient: g̃_i = s_i g_i.
- Solve d̃* = argmin ½‖d̃ − g̃1‖² s.t. g̃_jᵀ d̃ ≥ τ‖g̃_j‖² for all j ≥ 2. This is the Euclidean projection of g̃1
  onto an intersection of half-spaces.
- KKT form: d̃* = g̃1 + Σ μ_j g̃_j with μ_j ≥ 0. The primary coefficient is pinned at 1.
- Closed form for K = 2 and K = 3. For general K it uses working-set enumeration (≤ 2^{K−1} solves) or Goldfarb–Idnani.
- Deployed update: d* = d̃*·‖g1‖/‖d̃*‖, passed to the optimizer (Adam in the experiments).
- Infeasible QP (possible only for K > 2): fall back to g̃1.
- Cost: K backprops per step; the QP itself is negligible.
- **Ancestry (stated by the authors, Sec. 4.1):** "The construction of Eq. (4) is not itself new." For K = 2 it equals
  Gong et al. (2021) dynamic-barrier GD (`gong2021dbgd`) with barrier β‖∇g‖² on raw gradients, with β_j = τ s_j/s1.
  PCD's novelty is per-objective scale normalization, the τ interpretation, the general-K analysis, and the evaluation.
  The related-work section (Sec. 2, p. 4) also says GEM/A-GEM are "algorithmically equivalent to PCD with τ = 0",
  ε-SOFT-GEM adds a unit-normalized margin at K = 2, and Pareto Navigation GD (`ye2022pngd`) is an anchored
  general-K program with a shared, scheduled margin.

## Theory — what is guaranteed (THEORETICAL GUARANTEE, with scope)

| Result | Statement | Conditions / scope |
|---|---|---|
| Thm 4.2 | d̃* is the projection of g̃1 onto F_τ; KKT form above | F_τ ≠ ∅ |
| Cor 4.3 | If g̃1 already satisfies every constraint, d̃* = g̃1 (plain GD on L1) | — |
| Prop 4.4 | Per-step first-order secondary progress g̃_jᵀd̃* ≥ τ‖g̃_j‖²; strict decrease of smooth L_j for small η; nonsmooth case holds up to a "kink tax" | F_τ ≠ ∅, τ > 0; **about the direction d̃*, not the realized optimizer step** |
| Prop 4.7 | (K = 2, unit norms, c = cos) primary first-order descent iff 1 + τc − c² > 0, i.e. τ < (1−c²)/(−c) when c < 0 | **The primary is NOT guaranteed to descend under strong conflict** |
| Thm 4.8 | d̃* = 0 ⇔ all g̃_i = 0 (CMS), so it cannot stall at Pareto "cancellation" equilibria | τ > 0, feasible; pointwise endpoint result, "should not be read as a stability result" |
| Thm B.9 | Asymptotic scale invariance under EMA normalization (WS is not invariant) | ε → 0 |
| B.1/B.2 | Feasibility via Farkas; infeasible iff 0 ∈ conv of the nonzero secondaries | K > 2 |

**Assumption 3.1:** primary C¹; **secondaries proper, convex, lsc** (ℓ1, group lasso, nuclear norm, TV, hinge).

**Explicitly not guaranteed (VERIFIED, Sec. 4.7, 4.8, 6):**
- "We prove no convergence theorem for Algorithm 1": no iterate convergence and no rates. DBGD does have
  convergence proofs for the raw-gradient update.
- The deployed rescale has fixed points where g1 = 0 that are not CMS.
- Adam or momentum can flip the sign of the realized secondary progress. Measured in App. G: Adam keeps the sign on
  92.3–99.8% of steps; SGD+momentum flips it on 47–73% of steps.
- On "genuine trade-off problems with no common stationary point, no CMS point exists and the PCD direction with
  τ > 0 has no fixed point. Iterates need not converge in this regime, and the framework is best understood as
  targeting the regularizer-style hierarchy."
- Non-convex secondaries ("adversarial losses or hard combinatorial sparsity targets") fall outside the proofs (Sec. 6).

## Empirical evidence (EXISTING EMPIRICAL EVIDENCE, as reported by the authors)

- **Structured pruning (K = 2, CE + channel group lasso).** DenseNet-121, ResNet-34, Inception, MobileNetV2 on
  CIFAR-10/100; Adam 1e-3, 300 epochs, 5 seeds; no fine-tuning. Sec. 5.1.1 shows large gaps over MOO baselines,
  e.g. ResNet-34/CIFAR-100 at 80% reduction: PCD 73.0% vs WS 27.3%, AuxiNash 44.4%, PCGrad 7.4%.
  **Caveat stated by the authors (Sec. 5.1.2):** the WS grid w ≥ 0.1 "floors λ_eff at 0.111, 2 to 4 orders of
  magnitude above the range in which group-lasso scalarization operates best", so the Sec. 5.1.1 gaps overstate
  the advantage.
- **Tuned-baseline study (Sec. 5.1.2, Table 3; 330 runs, 3 seeds).** Baselines: tuned log-grid scalarization,
  DBGD (β and α branches), Cooper, proximal group lasso. Full range (102 matched comparisons): PCD better 44,
  worse 4, within noise 54. Compression regime ≥ 85% sparsity (47): better 28, worse 1, within noise 18.
  → INTERPRETATION: against tuned baselines the advantage is real but moderate, and more than half of the
  full-range comparisons are ties.
- **Mechanism (Sec. 5.1.1, App. E.3).** "The cause is objective scale rather than gradient conflict": the raw
  regularizer gradient norm is 1–2 orders of magnitude larger than the task gradient, and
  cos(g_task, g_reg) ≤ 1.2×10⁻³. The constraint is active at every step.
- **Normalization ablation (App. E.1/E.2).** Normalization is "fundamental": disabling it costs 11.2+ points.
  Baselines re-run with EMA normalization improve (MGDA 1.23% → 33.29%) but stay in a narrow, collapsed sparsity band.
- **K = 3 (ℓ1 + nuclear norm) and synthetic experiments up to K = 8.** 0/1785 configurations were infeasible.
- **No KD, no federated, no fairness, no robustness experiments** in this paper. KD does not appear anywhere in the text.

## Critical reading (INTERPRETATION)

1. **PCD ≈ normalized weighted sum in the paper's own main regime.** For K = 2 with unit-normalized gradients and
   an active constraint, d̃* = g̃1 + (τ − c)g̃2 (Cor. 4.6 / Prop. 4.7). With c ≈ 0, as measured (≤ 1.2e-3), this is
   **g̃1 + τ g̃2, a normalized weighted sum with weight τ**, followed by a rescale to ‖g1‖.
   VERIFIED (App. E.2, p. 37): the EMA-normalized baselines were run "over the same hyperparameter grid as the
   original submission", i.e. w ∈ {0.1..0.9}. A normalized WS with weight in [0.01, 0.1], the range of PCD's best
   τ, therefore does not appear in the paper (re-check against the Table 9 rows and code).
   → **HYPOTHESIS:** a GradNorm-style normalized WS with λ = τ matches PCD on pruning. This is a decisive and cheap test.
2. The projection's distinctive regimes (inactive vs. active-with-conflict) matter only when c varies in sign or
   magnitude across steps. Problems with real, varying gradient conflict are where PCD can differ from normalized WS.
3. Every guarantee is per-step and directional. None of them covers generalization or final accuracy.
4. Non-convex secondaries (KD-KL, adversarial loss, fairness surrogates, other clients' losses) are outside
   Assumption 3.1. Prop. 4.4(ii) only needs local smoothness, so the per-step progress statement still holds
   locally (first order). The convex-feasibility and kink-tax analysis does not carry over.

## Datasets / benchmarks used

CIFAR-10, CIFAR-100 (pruning, K = 3 sparsity/low-rank); synthetic problems.
