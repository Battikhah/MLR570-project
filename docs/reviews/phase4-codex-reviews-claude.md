# Phase 4 adversarial cross-review — Claude research

Checked 2026-11-02 against the PCD v2 HTML, FedPCD v1 HTML, author/project pages, the supplied assignment requirements, and the two agents’ committed notes. Evidence labels: **VERIFIED** means checked against a primary source; **INTERPRETATION** is this reviewer’s reasoning; **UNSUPPORTED** means the available primary evidence does not establish the claim. Evidence classes are **THEORETICAL GUARANTEE**, **EXISTING EMPIRICAL EVIDENCE**, and **UNTESTED HYPOTHESIS**. This is a bounded review, not a systematic literature review.

Primary sources: Varam & AlHajri, [PCD arXiv v2 HTML](https://arxiv.org/html/2606.29521), [arXiv record](https://arxiv.org/abs/2606.29521), [author project page](https://daravaram.github.io/priority-constrained-descent/); Irshaid, Varam & AlHajri, [FedPCD HTML](https://arxiv.org/html/2610.13508), [arXiv record](https://arxiv.org/abs/2610.13508). Assignment evidence: [requirements extracted from proposal and Survey Guideline](../requirements/assignment-requirements.md).

## 1. Adversarial verdicts on Claude’s PCD claims

| Claude claim | Verdict | Primary evidence and correction |
|---|---|---|
| Table 3 has 102 matched comparisons: 44 PCD wins, 4 losses, 54 ties. | **AGREE — VERIFIED; EXISTING EMPIRICAL EVIDENCE** | PCD Table 3, §5.1.2, reports 102 comparisons across the full range: 44 better, 4 worse, 54 within noise (HTML §5.1.2, Table 3). This is not a 44/102 decisive win rate: 54 are ties under the authors’ stated noise criterion. |
| ResNet-34/CIFAR-100 gives “73.0 vs 27.3.” | **AGREE WITH QUALIFICATION — VERIFIED; EXISTING EMPIRICAL EVIDENCE** | §5.1.1/Table 2 has this pair at 80% target sparsity (PCD 73.0, WS 27.3); at 90% the PCD entry is 70.9 while WS remains 27.3. The pair must be tied to its 80% operating point, not generalized to all sparsities. |
| The WS grid was 2–4 orders of magnitude off. | **AGREE WITH QUALIFICATION — VERIFIED; EXISTING EMPIRICAL EVIDENCE** | §5.1.2 describes the WS coefficient mapping `λ_eff=w/(1−w)` and says the tested `w≥0.1` floor corresponds to coefficients 2–4 orders above the best-performing group-lasso range. This is an authors’ explanation of grid mismatch, not proof that every PCD advantage is an artifact. App. E.4 separately tunes regularizer coefficients; see §2 below. |
| Raw task/regularizer cosine never exceeded 1.2×10⁻³; authors attribute gains to objective scale, not meaningful gradient conflict. | **AGREE — VERIFIED; EXISTING EMPIRICAL EVIDENCE** | §5.1.1 reports maximum absolute cosine ≤1.2×10⁻³ and explicitly attributes the result to objective scale rather than directional conflict. App. E.3 reports negative alignment on roughly 0.41–0.46 of steps but with tiny severity (about 10⁻⁵), and maximum absolute cosine about 0.0011. Thus “no conflict” is too strong; “measured negative signs were frequent but tiny in magnitude” is accurate. |
| Momentum reverses realized secondary-progress sign on 47–73% of SGD steps. | **AGREE WITH QUALIFICATION — VERIFIED; EXISTING EMPIRICAL EVIDENCE** | App. G reports 70.4%, 47.2%, 73.3%, and 54.3% for its tested SGD+momentum configurations; Adam results are much lower (0.2–7.7%). This is evidence for those settings only, not a universal rate. It also shows the QP’s direction guarantee is not a guarantee on the momentum/Adam realized update. |
| App. E.2 normalized baselines reuse the `w≥0.1` grid. | **AGREE — VERIFIED; EXISTING EMPIRICAL EVIDENCE** | App. E.2 says each symmetric baseline uses the PCD EMA normalization and the “same hyperparameter grid as the original submission”; its table tests `w={0.1,0.2,0.4,0.6,0.8,1}`. It does not test matched small normalized weights. |
| The normalized-WS control `λ∈[0.01,0.09]` is wholly untested. | **DISAGREE AS WORDED — VERIFIED; EXISTING EMPIRICAL EVIDENCE** | A normalized weighted-sum baseline was tested in App. E.2, so normalization itself is not untested. A **matched-low-weight EMA-normalized sum with the PCD rescale and active-constraint coefficient** appears absent from the reported tables. App. E.4’s tuned *raw* scalarization includes small regularizer coefficients (including `10⁻⁴` to `10⁻²`), but it is not the same normalization-matched control. State the gap narrowly; do not claim no relevant scalarization controls exist. |
| “Not itself new” / PCD is equivalent to DBGD for two objectives. | **AGREE WITH QUALIFICATION — VERIFIED; THEORETICAL GUARANTEE / method relation** | §3/related-work text says the two-objective projection construction is not itself new and relates PCD to Gong et al.’s dynamic-barrier GD. PCD’s `τ` corresponds to a dynamic-barrier coefficient after accounting for per-objective scales (`β_j=τ s_j/s_1` in the paper’s notation). This is an algorithmic/direction relation; PCD’s EMA normalization, final rescaling, multi-secondary framing and stated analysis/experiments are not thereby identical to every DBGD implementation. |
| GEM/A-GEM are algorithmically equivalent to PCD at `τ=0`. | **AGREE WITH QUALIFICATION — VERIFIED; method relation** | PCD §3.2 states this equivalence. It is the shared two-objective half-space projection geometry at zero margin. It does not establish equivalence of full training pipelines, normalization, optimizer momentum, or positive-margin variants; the paper separately notes ε-SOFT-GEM adds a unit-normalized positive margin. |
| Prop. 4.7 gives the primary-descent condition; Thm. 4.8 says the normalized direction vanishes only at composite multi-stationary points. | **AGREE WITH QUALIFICATION — VERIFIED; THEORETICAL GUARANTEE** | Prop. 4.7 gives `g̃₁ᵀd̃*=1+τc−c²` after unit normalization (and therefore descent iff positive); with `c<0`, require `τ<(1−c²)/(−c)`. Thm. 4.8 is a pointwise statement about zeros of the normalized PCD direction under its assumptions and `τ>0`; it is **not** an algorithm convergence theorem, stability result, or guarantee for the paper’s rescaled direction/optimizer trajectory. The paper explicitly distinguishes these. |

**Review correction:** Claude’s “D2 is the missing normalized-weight control” is directionally useful but overstates absence. The paper has a normalized-baseline ablation and separate tuned raw regularizer comparisons. The untested increment is the **low-weight, EMA-normalized, rescale-matched** scalarization at the effective PCD coefficient, with a tuning budget matched to PCD.

## 2. D2 mathematical challenge: PCD versus normalized weighted sum

**INTERPRETATION — derivation (two objectives).** Let `a=g̃₁=s₁g₁`, `b=g̃₂=s₂g₂`, and `c=aᵀb`. The unconstrained candidate is `a`. If `aᵀb≥τ||b||²`, the constraint is inactive and `d̃*=a`. Otherwise it is active, and Euclidean projection onto the boundary gives

```text
d̃* = a + α b,     α = τ − (aᵀb)/||b||².
```

Thus in the active case the *direction before PCD’s final rescale* is exactly the ray of a positive normalized weighted sum, `a+αb`. If writing the sum as `(1−w)a+wb`, its equivalent simplex weight is `w=α/(1+α)`, not `w=α`; under the common ratio convention `a+λb`, the coefficient is `λ=α`. For approximately unit EMA-normalized vectors and `c≈0`, `α≈τ` and `w≈τ/(1+τ)`. Hence for `τ=.01… .09`, the simplex weights are about `.0099… .0826`, below App. E.2’s minimum `.1`.

This is **not** an unconditional exact equality to “normalized WS, `λ=τ`” in the paper’s deployed update. First, `s_i` comes from an EMA of raw squared gradient norms, not current-batch unit normalization; in general `||a||` and `||b||` are not 1, and the exact coefficient is `α=τ−c/||b||²`, not simply `τ`. In raw-gradient coordinates the ray is `g₁+α(s₂/s₁)g₂`. Second, PCD rescales the projected direction to `||g₁||`:

```text
||d̃*||² = ||a||² + 2αc + α²||b||²
d* = (||g₁||/||d̃*||) (a+αb).
```

For unit `a,b` with active constraint, `α=τ−c`, so `||d̃*||²=1+τ²−c²`; at `c≈0`, the pre-rescale norm is `√(1+τ²)`, not 1. A normalized WS direction has the same ray if it uses the same `a,b,α`; matching actual SGD steps also requires matching PCD’s final magnitude factor and learning rate. Under ordinary SGD, omitting this factor changes the effective step by `√(1+τ²)` in the unit/orthogonal case. Momentum/Adam further mean direction equivalence at the current gradient does not imply realized-update equivalence.

The equivalence breaks or changes when the constraint is inactive (`α=0`), `c` is material or varies, EMA norms differ from one, there are more than two constraints (the solution is a combination of an active set, not one fixed scalar weight), feasibility/fallback changes the QP, or optimizer state changes the actual parameter update. With tiny cosine and unit-like EMA norms, the coefficient deviation from `τ` is small in absolute terms, but can be nontrivial relative to `τ=.01`.

**VERIFIED — author evidence:** App. E.2 tests normalized WS but only on coarse weights down to `.1`, and reports that normalization alone did not explain the gap on that grid. App. E.4 includes tuned raw scalarizations over small regularizer coefficients and shows scalarization is a credible competitor in some operating points. Therefore the decisive ablation is still open, but the authors have already tested adjacent mechanisms; presenting it as a cleanly untouched question or a new method is unsupported.

## 3. FedPCD claims and counterevidence

| Claude claim | Verdict | Primary evidence and correction |
|---|---|---|
| FedPCD wins global accuracy in 18/18 settings and worst-decile in 17/18. | **AGREE WITH QUALIFICATION — VERIFIED; EXISTING EMPIRICAL EVIDENCE** | FedPCD §5 and its tables cover 3 datasets × 2 participation rates × 3 Dirichlet α values; best-τ headline selection is 18/18 global and 17/18 worst-decile. Fashion-MNIST is the worst-tail exception: FedLF 65.73 versus FedPCD 63.22. A separate fixed `τ=.02` report has 18/18 global but only 15/18 tail wins. Do not conflate selected-best-τ with fixed-τ evidence. |
| CIFAR-100 FedAvg 9.08 versus FedPCD 18.33. | **AGREE WITH QUALIFICATION — VERIFIED; EXISTING EMPIRICAL EVIDENCE** | The values occur in the same reported CIFAR-100 configuration/table column (α=.01 and 5% participation, fixed FedPCD τ=.02); they are percentage accuracies. The striking relative gain starts from very low accuracy; it does not establish useful absolute performance. In this setting the reported worst-decile accuracy is much lower, and paper-level spread caveats remain. |
| FedPCD searched a 6τ×11H grid while most baselines had at most 3 values. | **AGREE WITH QUALIFICATION — VERIFIED; EXISTING EMPIRICAL EVIDENCE** | §5 hyperparameter table lists six `τ` candidates and `H∈{0,…,10}` (66 nominal combinations), while several baseline grids are fixed or at most three values. Selection is performed over reported validation/metric protocol after five seeds; the larger search is a fairness concern, although fixed-τ results reduce the likelihood that the entire outcome is tuning alone. Avoid describing this as proof of bias. |
| FedPCD omits FedProx, SCAFFOLD and FedNova. | **AGREE — VERIFIED; EXISTING EMPIRICAL EVIDENCE** | §5 baseline list and result tables do not include these three; SCAFFOLD/FedNova are cited in related work, not evaluated. Their absence leaves conventional drift-control comparisons unanswered. |
| The paper uses/does not use a validation split. | **UNSUPPORTED if stated categorically either way** | The paper’s searchable text does not describe a distinct validation partition or a clear held-out model-selection protocol; it does say hyperparameters are chosen after averaging five seeds and reports final-round metrics. This supports “validation procedure not described clearly” and a possible model-selection leakage risk, but does not prove that test data were used for selection. Verify the code/protocol or ask authors before alleging test leakage. |
| No official FedPCD code exists. | **UNSUPPORTED** | Claude’s own note says the author project page has a “Code” link whose destination did not resolve. The Codex registry correctly leaves the repo, ownership, license and contents unverified. Evidence establishes an unresolved advertised code link, not nonexistence. |

**Interpretation — counterevidence to “FedPCD is best”:** The reported setting is simulated FL, not deployment; a favorable best-τ grid can inflate selected comparisons; FedLF wins one worst-tail case; FedPCD’s CIFAR-100 cross-client spread is reported higher than competitors in some settings; low absolute CIFAR-100 accuracy limits practical interpretation; and the authors’ theory assumes an alignment margin that heterogeneity/local drift may violate. The reported wins are worth replicating, not sufficient to conclude FedPCD beats well-tuned drift-aware methods.

## 4. Challenge to D2 as the course project and novelty claims

**INTERPRETATION:** An independent reproduction with a carefully matched low-weight normalized-WS control could be a useful method audit, but the phrase “independent replication + mechanism audit is novel” needs restraint. The PCD authors already vary normalization, compare symmetric methods, log constraint activity/conflict, examine optimizer sign reversals, and tune raw group-lasso coefficients. The incremental contribution is a narrow control plus independent implementation and a smaller, preregistered replication. “Any outcome is publishable” is not evidence and should be removed.

**VERIFIED:** The submitted project proposal asks whether reliability-gated PCD-KD improves student accuracy and reduces negative transfer when teacher outputs are unreliable (CIFAR-100; proposal pp. 2–3). The survey guideline explicitly asks for critical analysis and grounded future directions (G8–G9), so D2 would support the survey’s critique. **INTERPRETATION:** D2 is only a partial match to the ML-project proposal and a larger scope change than D1; survey relevance does not itself satisfy the separate ML project deliverable. Instructor/team acceptance of a pivot is unknown. Conversely, the existing proposal’s large method × gate × corruption × seed matrix should be cut regardless; a D1 pilot can include the same matched normalized-WS control and preserve the proposal’s research question.

**Novelty and scooping audit:**

- **D5 QAT:** Claude says no PCD-QAT paper found, but its “high scooping risk” is not supported by evidence of an active competing PCD-QAT project. The authors’ project page lists quantization as a possible application, which establishes overlap risk, not a probability or timeline. GAQAT ([arXiv:2412.05551](https://arxiv.org/abs/2412.05551)) is prior gradient-adaptive conflict mitigation for quantization-aware training, so “gradient-conflict QAT is new” would be false; the narrower PCD-specific gap remains unverified by bounded search. Standard QAT/STE and post-training quantization are strong simpler controls.
- **D6 continual learning:** The PCD paper itself says GEM/A-GEM are algorithmically equivalent at `τ=0`, and ε-SOFT-GEM already adds a unit-normalized positive margin. This is strong prior-art evidence and makes a basic “PCD equals GEM with margin” contribution low novelty. It does **not** prove every EMA-normalized PCD/optimizer implementation is identical to ε-SOFT-GEM; compare exact margins, scales, memory sampling and optimizer updates before claiming equivalence. “Essentially identical” is an interpretation, not a complete literature conclusion.
- **D4 fairness:** Existing GroupDRO, reductions and Lagrangian constrained-learning approaches make the general objective crowded. The Claude search did not verify its noted lexicographic-fairness lead, and “no PCD/priority-projection fairness method found” is a search result, not proof of absence. A targeted novelty search and defensible group benchmark are required.
- **D1 KD:** SCKD, SCOPE, TPKD and reliability-aware KD make classification-priority and gating adjacent work. The novel component is a specific PCD × reliability-gate stress test; whether that is enough novelty is an **UNTESTED HYPOTHESIS**. The gate itself may reduce gradient conflict, reducing PCD’s marginal value.
- **D2 compression/method audit:** Compression is directly evaluated by PCD; that improves reproducibility but lowers novelty. A new low-weight normalized control would be a legitimate extension if it changes conclusions, but no result should be assumed.

## 5. Merged candidate scoring

**Scale:** 1 = weak/poor, 5 = strong/good. Each cell gives score, compact rationale, and evidence class; these are reviewer judgments, not measurements. Candidate union: Claude D1–D7 plus Codex objectives 1–5. Merges: Claude D1 = Codex 3 (KD); Claude D3 = Codex 2 (FedPCD); Claude D4 = Codex 5 (group fairness); Claude D7 = Codex 4 (robustness). Codex 1 (compression) is added as C1; Claude D2, D5 and D6 remain distinct. “Evidence class” labels whether the justification rests on a theorem, reported experiments, or an untested expectation; where both appear, both are stated.

### D1 / Codex 3 — Reliability-gated PCD knowledge distillation

- **(a) Theory 2/5:** PCD’s directional QP applies, but the KD secondary is nonconvex and primary descent can fail under conflict (**THEORETICAL GUARANTEE**, conditional/limited).
- **(b) Measurable advantage 3/5:** test accuracy, negative transfer and per-batch cosine are measurable; whether gating leaves useful conflict is unknown (**EXISTING EMPIRICAL EVIDENCE** for metrics; **UNTESTED HYPOTHESIS** for gains).
- **(c) Novelty 2/5:** related priority, projection and reliability gates exist; PCD × gate is incremental (**EXISTING EMPIRICAL EVIDENCE** from adjacent published methods; novelty itself **UNTESTED HYPOTHESIS**).
- **(d) Benchmarks 4/5:** CIFAR-100 and CIFAR-10 have established protocols and the course proposal supplies model/data choices (**EXISTING EMPIRICAL EVIDENCE**).
- **(e) Reproducibility 3/5:** proposed teacher/student protocol is specified, but recipe, teacher provenance and schedule need fixing; no local result (**UNTESTED HYPOTHESIS**).
- **(f) M4 feasibility 2/5:** plausible only after sharply reducing the planned method/corruption/τ matrix (**UNTESTED HYPOTHESIS**, cost estimate only).
- **(g) Assignment fit 5/5:** directly matches submitted ML proposal and KD survey (**VERIFIED** course documents).

### D2 — PCD independent replication and matched-normalization mechanism audit

- **(a) Theory 5/5:** the two-objective projection formula predicts the exact active-set scalarization relation and its limits (**THEORETICAL GUARANTEE**).
- **(b) Measurable advantage 4/5:** objective scales, cosine, active rate, update norm and matched-sparsity performance are directly measurable (**EXISTING EMPIRICAL EVIDENCE** motivates tests; outcome **UNTESTED HYPOTHESIS**).
- **(c) Novelty 2/5:** authors already did normalization and extensive baselines; only the narrow low-weight/rescale-matched control is clearly missing (**EXISTING EMPIRICAL EVIDENCE**; novelty of outcome **UNTESTED HYPOTHESIS**).
- **(d) Benchmarks 4/5:** paper’s CIFAR compression tasks and synthetic cases are defined, though licensing/use terms need checking (**EXISTING EMPIRICAL EVIDENCE**).
- **(e) Reproducibility 4/5:** official MIT PCD repository is author-linked and scripts exist, but exact configs/commit access and faithful reproduction remain unchecked (**VERIFIED** link/license; execution **UNTESTED HYPOTHESIS**).
- **(f) M4 feasibility 3/5:** reduced ResNet/CIFAR schedules are plausible, while full ResNet-34, many sparsity targets and five-seed grids are uncertain (**UNTESTED HYPOTHESIS**).
- **(g) Assignment fit 3/5:** excellent support for survey critical-analysis requirement, but it pivots away from the separate submitted KD project (**VERIFIED** requirements; suitability judgment **INTERPRETATION**).

### D3 / Codex 2 — FedPCD client fairness, drift and matched comparison

- **(a) Theory 3/5:** client progress constraints fit the hierarchy, but convergence needs full participation/smoothness and an assumed positive alignment margin (**THEORETICAL GUARANTEE**, conditional).
- **(b) Measurable advantage 4/5:** mean, worst-decile, spread, cosine and drift can be recorded; the paper reports favorable but qualified numbers (**EXISTING EMPIRICAL EVIDENCE**; robustness to controls **UNTESTED HYPOTHESIS**).
- **(c) Novelty 3/5:** a fair FedProx/SCAFFOLD comparison and local-epoch drift study address real omissions, but remain an evaluation extension (**EXISTING EMPIRICAL EVIDENCE**; contribution untested).
- **(d) Benchmarks 4/5:** Fashion-MNIST/CIFAR federated partitions are specified by the paper, though split generation/provenance should be replicated (**EXISTING EMPIRICAL EVIDENCE**).
- **(e) Reproducibility 2/5:** official code target and license remain unresolved; reimplementation is necessary (**UNSUPPORTED** code availability; outcome **UNTESTED HYPOTHESIS**).
- **(f) M4 feasibility 3/5:** a small simulation slice is plausible; full client × α × participation × method × seed matrix is not (**UNTESTED HYPOTHESIS**, estimates only).
- **(g) Assignment fit 2/5:** federated fairness is not the submitted KD project and is only adjacent to the survey (**VERIFIED** proposal scope; fit judgment **INTERPRETATION**).

### D4 / Codex 5 — Centralized group fairness / robustness (“groups as clients”)

- **(a) Theory 3/5:** the QP can protect group-loss directional progress, but group risks are generally nonconvex and multi-constraint feasibility is not assured (**THEORETICAL GUARANTEE**, limited).
- **(b) Measurable advantage 3/5:** group accuracy/worst-group loss and disparity are measurable, but per-step loss progress need not improve deployment fairness (**EXISTING EMPIRICAL EVIDENCE** from adjacent methods; transfer **UNTESTED HYPOTHESIS**).
- **(c) Novelty 3/5:** priority group constraints may be distinct, yet GroupDRO/reductions/Lagrangian work is close and search was not systematic (**EXISTING EMPIRICAL EVIDENCE**; exact gap **UNTESTED HYPOTHESIS**).
- **(d) Benchmarks 4/5:** tabular/synthetic or established group-shift datasets exist; licenses, group definitions and suitability must be checked (**EXISTING EMPIRICAL EVIDENCE**; choice unresolved).
- **(e) Reproducibility 3/5:** small data and standard models help, but surrogate and split choices are sensitive (**UNTESTED HYPOTHESIS**).
- **(f) M4 feasibility 5/5:** tabular/MNIST-scale work is computationally modest, subject to K backward passes (**UNTESTED HYPOTHESIS**, straightforward estimate).
- **(g) Assignment fit 2/5:** departs from KD proposal and broad KD survey (**VERIFIED** course scope; judgment **INTERPRETATION**).

### D5 — PCD for quantization-aware training

- **(a) Theory 2/5:** distance-to-grid/fake quantization is nonconvex and outside the main convex-secondary theory (**THEORETICAL GUARANTEE** does not cover it).
- **(b) Measurable advantage 3/5:** accuracy at fixed bit-width, quantization error and deployment cost are measurable, but gradient conflict benefit is unknown (**UNTESTED HYPOTHESIS**).
- **(c) Novelty 3/5:** PCD-specific QAT was not verified, but GAQAT already studies gradient conflict in QAT and authors list quantization as a possible use (**EXISTING EMPIRICAL EVIDENCE**; exact gap **UNTESTED HYPOTHESIS**).
- **(d) Benchmarks 4/5:** CIFAR small-model QAT is established, with hardware/device benchmarking needing careful control (**EXISTING EMPIRICAL EVIDENCE**).
- **(e) Reproducibility 3/5:** fake-quant and bit-width recipes are reproducible in principle but add implementation choices (**UNTESTED HYPOTHESIS**).
- **(f) M4 feasibility 3/5:** small CNN is plausible; actual quantized deployment speed on MPS is uncertain (**UNTESTED HYPOTHESIS**).
- **(g) Assignment fit 2/5:** does not directly address submitted KD proposal or survey focus (**VERIFIED** requirements; judgment **INTERPRETATION**).

### D6 — Continual learning / normalized-margin GEM

- **(a) Theory 3/5:** PCD’s half-space geometry directly matches episodic no-forgetting constraints, but positive-margin convergence is not inherited automatically (**THEORETICAL GUARANTEE**, narrow).
- **(b) Measurable advantage 2/5:** forgetting/average accuracy are measurable, but no clear advantage over GEM/ε-SOFT-GEM is established (**EXISTING EMPIRICAL EVIDENCE** for close prior art; benefit **UNTESTED HYPOTHESIS**).
- **(c) Novelty 1/5:** GEM/A-GEM τ=0 equivalence and ε-SOFT-GEM’s positive normalized margin are already acknowledged by PCD (**EXISTING EMPIRICAL EVIDENCE**).
- **(d) Benchmarks 4/5:** Permuted/Rotated MNIST and split CIFAR protocols are established and small (**EXISTING EMPIRICAL EVIDENCE**).
- **(e) Reproducibility 4/5:** benchmark protocols and baseline implementations are available, though memory sampling must be matched (**EXISTING EMPIRICAL EVIDENCE**).
- **(f) M4 feasibility 5/5:** small continual-learning models are tractable on this hardware (**UNTESTED HYPOTHESIS**, strong cost-based estimate).
- **(g) Assignment fit 2/5:** little direct KD proposal or survey alignment (**VERIFIED** documents; judgment **INTERPRETATION**).

### D7 / Codex 4 — Clean/adversarial robustness trade-off

- **(a) Theory 1/5:** adversarial loss lies outside the paper’s stated proof scope; strong trade-off can also violate primary-descent condition (**THEORETICAL GUARANTEE** is inapplicable/conditional).
- **(b) Measurable advantage 4/5:** clean and robust accuracy under fixed attacks are measurable, but attack-strength choices can dominate (**EXISTING EMPIRICAL EVIDENCE** for standard metrics; PCD gain **UNTESTED HYPOTHESIS**).
- **(c) Novelty 3/5:** projection-based clean/adversarial methods already exist; exact PCD application may be new but search is bounded (**EXISTING EMPIRICAL EVIDENCE**; gap **UNTESTED HYPOTHESIS**).
- **(d) Benchmarks 5/5:** CIFAR adversarial/robustness benchmarks and metrics are abundant (**EXISTING EMPIRICAL EVIDENCE**).
- **(e) Reproducibility 3/5:** attacks are standardized but require strict attack tuning, seeds and evaluation safeguards (**EXISTING EMPIRICAL EVIDENCE**; actual reproducibility **UNTESTED HYPOTHESIS**).
- **(f) M4 feasibility 1/5:** PGD inner loops multiply compute and are a poor fit for M4/MPS semester budget (**UNTESTED HYPOTHESIS**, cost reasoning).
- **(g) Assignment fit 2/5:** not the submitted KD objective; only broad gradient-optimization relevance (**VERIFIED** scope; judgment **INTERPRETATION**).

### C1 — Compression/sparsity/low-rank PCD (Codex objective 1)

- **(a) Theory 5/5:** convex group-lasso/ℓ1/nuclear secondaries fit the stated objective assumptions, subject to feasibility and kink handling (**THEORETICAL GUARANTEE**).
- **(b) Measurable advantage 4/5:** accuracy at matched sparsity/rank is direct; source paper reports wins but also 54 ties and tiny cosine (**EXISTING EMPIRICAL EVIDENCE**; transfer **UNTESTED HYPOTHESIS**).
- **(c) Novelty 1/5:** this is the PCD paper’s central tested application (**EXISTING EMPIRICAL EVIDENCE**).
- **(d) Benchmarks 5/5:** source paper defines CIFAR/model/compression protocols (**EXISTING EMPIRICAL EVIDENCE**).
- **(e) Reproducibility 5/5:** author-linked MIT implementation and experiment scripts offer the strongest available reproduction path; actual run not checked (**VERIFIED** provenance/license; execution **UNTESTED HYPOTHESIS**).
- **(f) M4 feasibility 3/5:** small ResNet and limited sparsity targets seem possible, but full paper-scale models/sweeps do not (**UNTESTED HYPOTHESIS**, estimate).
- **(g) Assignment fit 2/5:** weaker than KD for both submitted proposal and KD survey (**VERIFIED** documents; judgment **INTERPRETATION**).

## 6. Recommendation and falsification

**RECOMMENDATION — top-1: D2, narrowly scoped matched-normalization mechanism audit.** It has the cleanest falsifiable method question: at the reported near-orthogonal compression regime, does PCD still outperform an EMA-normalized weighted sum at the exact effective coefficient and matched update norm, when both receive equal tuning budget and are compared at matched sparsity? Run one small CIFAR-10 model/regularizer first, record raw and normalized cosine, active rate, direction/update norms, primary alignment and matched-sparsity accuracy, with paired seeds and a fixed validation protocol. Do not call this a new optimizer or assume it will favor PCD.

**Strongest argument against D2:** the strongest issue is assignment and novelty, not the algebra. The project proposal already commits to reliability-gated KD, and the PCD authors have already run a substantial normalization/mechanism study; a small matched-control audit may be an incremental result and may not satisfy the separate ML project brief without a deliberate scope decision. If the team must retain the proposal, **backup D1** is the best fit, but only as a reduced study with the matched normalized-weighted-KD control and a gate-only ablation; otherwise the same confound remains.

**RECOMMENDATION — falsification criteria:** reject “PCD helps” for the chosen setting if it does not beat the equally tuned normalized scalarization at the predeclared matched sparsity/primary metric, if paired uncertainty includes only negligible or negative effects, if advantage disappears under matched update magnitude, or if diagnostics show no meaningful conflict/constraint activity and no advantage beyond scalarization. Report this as a negative result rather than changing objective, metric or test protocol after seeing the outcomes.

## 7. What I could not verify

- **UNVERIFIED:** A complete systematic novelty search for PCD + QAT/fairness/continual learning, or whether authors are actively developing D5/D6; the QAT project-page listing shows only possible application overlap.
- **UNVERIFIED:** FedPCD repository destination, implementation, license and whether its reported model-selection protocol uses a separate validation set; the paper text is insufficient to allege test leakage.
- **UNVERIFIED:** Whether the matched low-weight normalized-WS control changes any empirical ranking; no code was run and no training was performed.
- **UNVERIFIED:** M4 wall-clock feasibility in practice; all time/compute statements in candidate notes are estimates, not benchmarks.
- **UNVERIFIED:** Whether the course instructor accepts replacing or substantially redirecting the submitted KD project.

## Round 2 — final adversarial review of Claude’s review and candidate S

Checked 2026-11-02 against PCD v2 HTML, FedPCD v1 HTML, the author-linked repository records, CIFAR owner page, and primary KD papers. **AGREE / DISAGREE / PARTLY AGREE / UNSUPPORTED** are claim verdicts; evidence is labeled **THEORETICAL GUARANTEE**, **EXISTING EMPIRICAL EVIDENCE**, or **UNTESTED HYPOTHESIS**. PDF retrieval of PCD v2 timed out through the available web reader; the exhaustive keyword check below is over the complete arXiv HTML text, not an independently extracted PDF.

### A. Response to Claude’s round-1 review, item by item

| # | Verdict | Response and evidence |
|---|---|---|
| 1 | **AGREE, with provenance limit** | The TMLR 2026 status is stronger than my prior “author page states” wording; accept Claude’s correction to published, supported by the TMLR listing/forum and arXiv record. **VERIFIED / EXISTING EMPIRICAL EVIDENCE:** [PCD v2 arXiv HTML](https://arxiv.org/html/2606.29521). |
| 2 | **AGREE** | FedPCD is marked under review in the author-linked project README, so replace the earlier vague “no venue/acceptance info” status. The author project’s code target is empty, as separately corrected below. **VERIFIED.** |
| 3 | **AGREE** | The FedPCD paper reports the Fashion-MNIST worst-tail exception (FedLF 65.73 vs best FedPCD 63.22 and σc 9.34 vs 10.43), higher CIFAR-100 client spread in reported results, and final communication-round models. These limit “FedPCD best” generalization. **VERIFIED / EXISTING EMPIRICAL EVIDENCE:** [FedPCD v1 HTML](https://arxiv.org/html/2610.13508), Tables 1–3 and App. A. |
| 4 | **AGREE** | Best-τ headline counts and fixed `τ=.02` must be separated: the latter gives global 18/18 and worst-decile 15/18, while per-setting τ selection produces 17/18 tail wins. This is positive fixed-τ evidence, but the 3 fixed-τ tail misses and model-selection protocol remain relevant. **VERIFIED / EXISTING EMPIRICAL EVIDENCE.** |
| 5 | **AGREE; correction accepted** | My round-1 phrasing “within reported seed noise at one headline compression point” understated both aggregate ties and the counterexample. Correct counts: all 102 comparisons are 44 wins, 4 losses, 54 within noise; ≥85% compression is 28 wins, 1 loss, 18 within noise among 47. The ResNet-34/CIFAR-100 crossing is 53.8% at 98.1% sparsity for tuned scalarization versus 51.9% at 97.9% for PCD (`τ=.09`). **EXISTING EMPIRICAL EVIDENCE:** PCD §5.1.2 / App. E.4 [HTML](https://arxiv.org/html/2606.29521). I corrected my own math note accordingly. |
| 6 | **AGREE** | Cite PCD §5.1.1: raw regularizer gradients are about 10–100× task-gradient norm and cosine magnitude is at most 1.2×10⁻³; the authors attribute the tested compression mechanism chiefly to scale imbalance. App. E.3 still reports frequent negative signs of tiny severity, so this does not mean mathematically zero conflict. **EXISTING EMPIRICAL EVIDENCE.** |
| 7 | **AGREE** | Claude’s clone audit resolves PCD code date/hash, license, dependencies, tests and entry points more strongly than my earlier registry. I updated the registry to include the clone HEAD/first-commit dates and qualification that the release is a rewrite; I did not independently rerun Claude’s shim tests. **VERIFIED from Claude’s read-only clone report; not independently rerun here.** |
| 8 | **AGREE** | “Paper-aligned scripts” was too generous: they launch PCD runs, but the release lacks competing baselines and physical-pruning export, and its purported `--prox` setting creates a material protocol ambiguity. I revised my own math and repository notes. Static script presence is not reproduction of the headline comparisons. **VERIFIED repository finding; reproduction outcome UNTESTED.** |
| 9 | **AGREE; superseded** | The FedPCD author project repository is now identified; its code URL is blank and README says code is not public. My registry previously marked a link target unresolved; it now states not released as of the check date, with `official: null`, `license: null`. **VERIFIED.** |
| 10 | **AGREE / INTERPRETATION** | FedPCD normalizes current pseudo-gradient vectors; it does not use PCD’s objective EMA. Stale history may make constraints less representative or infeasible, but absent empirical feasibility logs this remains a hypothesis, not a reported failure. **VERIFIED method detail; infeasibility effect UNTESTED HYPOTHESIS.** |
| 11 | **AGREE, with a scope correction** | The KD note needed closest prior work: SCKD, SCOPE, TPKD and DTO-KD. Also, CE/KD are often incompatible objectives, and PCD Prop. 4.7 permits primary CE ascent under sufficient conflict. However “CE and KD generally have no common stationary point” is not a universal theorem for every model/data/parameterization; phrase it conditionally as an incompatible-target trade-off. **THEORETICAL GUARANTEE is conditional; adjacent method evidence is EXISTING EMPIRICAL EVIDENCE.** |
| 12 | **DISAGREE as a ranking judgment; evidence accepted** | Claude’s ranking is reasonable if theoretical fit/novelty dominate; my project ranking gave heavy weight to the submitted KD proposal and survey fit. Neither ranking is a factual claim. S now deserves conditional priority because it directly fits the proposal, but it must pass a conflict pilot and is not proven to outperform fixed normalized KD. **INTERPRETATION; outcome UNTESTED HYPOTHESIS.** |
| 13 | **AGREE; correction accepted** | My `100–300 images/s` M4 estimate was unsupported and is withdrawn. PCD App. H’s 3.64 h versus 3.58 h on A10G for ResNet-34/CIFAR-100 is not a valid throughput proxy for ResNet-8 on M4/MPS; the architecture/FLOP argument does not bridge hardware/software differences. The M4 needs a smoke timing. **EXISTING EMPIRICAL EVIDENCE** for A10G only; M4 cost **UNTESTED HYPOTHESIS**. |
| 14 | **AGREE; correction accepted with application boundary** | I incorrectly modeled the compression penalty gradient as another full network backward and estimated ~2×. In the authors’ K=2 compression measurement, PCD including normalization/QP/rescale/instrumentation is 3.64±.03 h versus weighted sum 3.58±.03 h, under 2% overhead, because group-lasso is cheap weight-only work. This does not apply to CE+KD, where two objective gradients flow through the student network. I replaced the compression cost estimate and removed any inference that this A10G ratio is an M4 result. **EXISTING EMPIRICAL EVIDENCE.** |
| 15 | **AGREE; arithmetic corrected** | FedPCD samples 5–10 of 100 clients per round; 300 rounds × one local epoch is 1,500–3,000 client-epochs/seed, not 9,000. I replaced the 30-client arithmetic and withdrew the unsupported 2–12 hour M4 estimate; simulator runtime needs a measured smoke run. **VERIFIED** client count/rate; runtime **UNTESTED HYPOTHESIS**. |
| 16 | **AGREE; correction accepted** | CIFAR owner page states Python archive sizes of 163 MB (CIFAR-10) and 161 MB (CIFAR-100), requests citation of the tech report, and has no clear dataset license statement. I updated `repositories.yaml` and feasibility inventory. **VERIFIED:** [official CIFAR page](https://www.cs.toronto.edu/~kriz/cifar.html). |
| 17 | **AGREE** | Fashion-MNIST repository’s MIT notice does not by itself settle copyright/license scope for every dataset asset. Keep dataset-content rights unresolved. **VERIFIED repository text; license interpretation cautious.** |
| 18 | **AGREE** | CIFAR-C/CIFAR-100-C size, license/provenance and download feasibility were not verified in my feasibility note; do not list them as approved benchmark choices. **UNSUPPORTED in prior note.** |
| 19 | **AGREE** | Paired seeds, validation-only tuning, matched search budgets, preregistered metric and leakage checks are appropriate recommendations, but neither PCD’s App. H nor FedPCD’s appendix should be presented as following that recommended protocol. **RECOMMENDATION; existing paper protocol differs.** |
| 20 | **AGREE** | PCD’s anchored QP is not strict lexicographic optimization: its positive secondary margin may sacrifice primary descent (Prop. 4.7). DBGD’s formal lexicographic interpretation/proofs are a distinct lineage. **INTERPRETATION grounded in PCD theory.** |

#### A1. Undisclosed prox step: exact boundary of what is verified

**VERIFIED — PCD paper HTML audit:** I searched the PCD v2 HTML for `prox`, `proximal`, `soft-threshold`, and `threshold`, including the rendered App. E and App. H. The indexed HTML has no match for `prox`; `proximal` appears in §5.1.2 where proximal group lasso is described as a **baseline** (“Adam on cross-entropy followed by a per-group soft-threshold”), and in App. E.4 discussion/results for that baseline. “Threshold” also appears for post-hoc channel removal after training (§5.1), not as an in-loop PCD proximal update. I found no disclosure in §5 or App. H that PCD itself receives soft-thresholding.

**VERIFIED — repository statement, not paper disclosure:** The official repository’s `experiments/README.md` and `experiments/objectives.py` reportedly say the “paper’s runs” used a per-step proximal group soft-threshold with threshold `lambda*lr`; the sweep script passes `--prox` to PCD runs at `λ=1e-3` (5e-4 MobileNetV2). Thus it is accurate to call this an **undisclosed-in-the-paper protocol discrepancy asserted by the repo**, not verified from the paper that the historical runs in fact used it. Whether baselines also received an equivalent operation is **UNVERIFIED**. App. E.4’s zero-τ PCD and proximal-GL-only points show zero native sparsity in the listed configuration, which weakens a “prox alone explains compression” claim but does not identify interaction effects at positive τ. The ambiguity is material to reproducing native sparsity and should be resolved with a with/without-prox ablation; it is not evidence by itself that the reported result is invalid.

**UNVERIFIED:** direct PDF text search could not be completed because the web reader timed out fetching the arXiv PDF. The statements above describe the complete indexed v2 HTML only. No PDF was downloaded or bypassed.

#### A2. Compute corrections and registry changes

- **VERIFIED:** PCD App. H records A10G ResNet-34/CIFAR-100: full K=2 PCD 3.64±0.03 hours and weighted sum 3.58±0.03 hours, under 2% overhead in that group-lasso setup. It is not an M4 benchmark and does not apply to two expensive network-gradient objectives such as CE/KD.
- **VERIFIED:** FedPCD §5.1 uses N=100 and ρ∈{.05,.1}, hence 5–10 active clients per round. At the paper’s 300 rounds and one local epoch this is 1,500–3,000 participating client-epochs per seed. I make no wall-clock claim.
- **Corrections applied:** `research/notes/experimental-feasibility.md` removes 100–300 images/s, derived KD hours, ~2× compression cost, and 9,000 client-epochs / 2–12 h FL estimate; it replaces these with workload counts and calls for a measured smoke run. It adds the A10G-specific under-2% result and CIFAR archive size/citation guidance.
- **Corrections applied to `literature/repositories.yaml`:** PCD repo first date and HEAD/last-commit evidence; README/script prox claim and paper-disclosure gap; FedPCD title/status changed to code not released with blank official URL and null license; CIFAR Python archive sizes and tech-report citation requirement added. **Interpretation limit:** the proxy step is attributed to repository README/scripts, not to a paper statement.
- **Correction applied to `research/notes/pcd-math-and-implementation.md`:** Table 3 and ≥85% compression counts and the actual 53.8-vs-51.9 crossing now replace my earlier claim that the scalarization only tied one headline point. Code coverage and proximal disclosure are qualified as above.

### B. Adversarial review of coordinator candidate S

**Candidate statement:** CE-primary, KD-secondary; teacher output corruption 0/10/30% serves as a conflict dial; compare PCD to tuned and EMA-normalized scalarization with gated and ungated controls; ResNet-8 student, frozen ResNet-32 teacher, CIFAR-100 with CIFAR-10 pilot.

#### B1. Do KD papers show actual negative CE–KD gradients?

- **VERIFIED — EXISTING EMPIRICAL EVIDENCE, but not the full claim:** DTO-KD (ICLR 2026, [official proceedings paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/e546ce4230c15bc422a500bb47b341ea-Paper-Conference.pdf)) explicitly defines conflict as `⟨g_dist,g_task⟩<0` and plots measured gradient-conflict/dominance traces over 500 iterations for object detection (Fig. 1, p. 1), alongside classification and detection experiments. This is direct evidence that negative task/KD alignment occurs in at least a reported KD setting, but not a statistic for the proposal’s ResNet/CIFAR logit-KD setup.
- **VERIFIED — EXISTING EMPIRICAL EVIDENCE, qualified:** SCKD (ICCV 2021, [official CVF PDF](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhu_Student_Customized_Knowledge_Distillation_Bridging_the_Gap_Between_Student_and_ICCV_2021_paper.pdf), §§3.3–3.4) computes cosine between student-supervision and KD-loss gradients, gates KD based on that measure, and reports its relation to CKA and a threshold sensitivity study. It does not report a clear negative-cosine frequency or magnitude distribution in the text reviewed; its algorithm’s stated `cos≥φ` removal rule and prose about inverse directions are also not straightforwardly consistent. Treat it as evidence of gradient-similarity-based control and qualitative negative-transfer motivation, not a reliable quantitative estimate of conflict prevalence.
- **VERIFIED — EXISTING EMPIRICAL EVIDENCE, method definition not prevalence estimate:** SCOPE ([primary article page](https://pmc.ncbi.nlm.nih.gov/articles/PMC13613376/)) defines CE–KD cosine `ρ`, says `ρ<0` means conflict, and projects the conflicting KD component; it evaluates medical-image tasks. Its existence shows the topic is crowded. The accessible article text supports the criterion and framework, but I did not verify a quantitative corpus-wide negative-cosine rate from the reported experiments.
- **VERIFIED — THEORETICAL/EMPIRICAL, not cosine-rate evidence:** TPKD ([arXiv:2609.39338](https://arxiv.org/abs/2609.39338)) analyzes when full KL imitation is blocked by correct-label preservation and empirically evaluates task-preserving corrections; it is evidence that task/KD objectives can be structurally incompatible, not a measured cosine frequency.
- **VERIFIED — EXISTING EMPIRICAL EVIDENCE:** MoKD ([arXiv:2505.08170](https://arxiv.org/abs/2505.08170)) and DTO-KD frame KD as multi-objective with conflict/dominance; it further reduces the novelty of a generic PCD-for-KD application. These sources do not establish that CIFAR-100’s corrupted ResNet-32 teacher will yield a materially negative *parameter-gradient* cosine.

**Finding:** There is evidence for negative task/KD gradient alignment in adjacent KD experiments and methods that explicitly monitor it. I found no primary evidence quantifying how 0/10/30% high-confidence wrong-logit replacement changes CE–KD cosine on the proposal’s CIFAR/ResNet setting. “Corruption increases conflict” is a **HYPOTHESIS**, not a verified premise.

#### B2. Does the corruption dial behave as proposed?

**INTERPRETATION:** For softmax logits, CE logit gradient is proportional to `p_s−y`; KL teacher-to-student gradient is proportional to `p_s−p_T` (with temperature factors). If `p_T` is replaced by a high-confidence wrong target while the student still favors the correct label, these vectors can oppose. But parameter gradients are `Jᵀ` times those logit-space signals and batch aggregation can change or cancel cosine; initial predictions, temperature, student state, data order and corruption assignment all matter. Therefore 10/30% corruption need not monotonically lower parameter-space cosine or increase PCD constraint activity.

**INTERPRETATION:** A correctness gate that uses only training labels to exclude incorrect teacher outputs removes most deliberately corrupted examples by construction. The gate may therefore remove the very conflict that motivated the dial. It also changes KD’s sample distribution and effective batch size. A gated-Pcd versus gated-normalized-KD contrast is still meaningful as an optimizer comparison, but log (i) cosine on the full batch, (ii) cosine restricted to gated-in examples, (iii) gate rate by corruption level, and (iv) loss/accuracy by corrupted versus untouched examples. Do not claim gating isolates optimizer effects unless both methods use the identical mask and all gate-only controls are included.

**RECOMMENDATION:** Treat corruption as a controlled stressor, not an assumed conflict dial. A training-only, fixed-seed corruption map and validation/test segregation are required. Define whether corruptions replace all teacher logits for selected training samples or merely alter the top class; preserve the full soft distribution for uncorrupted samples. Use label correctness only on train data to form the gate.

#### B3. Is PCD necessarily a different optimizer than normalized weighted KD?

**INTERPRETATION — exact two-objective algebra:** With normalized gradients `a=s_CE g_CE`, `b=s_KD g_KD`, `c=aᵀb`, when the PCD constraint is active,

```text
d̃_PCD = a + αb,   α = τ − c/||b||².
```

Thus PCD in the active K=2 case is **exactly a per-step adaptive normalized weighted sum** before final rescaling. A fixed `λ=τ` normalized sum is the matched control only near `c=0` and `||b||≈1`; with materially negative `c`, PCD raises the coefficient to `α>τ` (for normalized unit b), so its direction differs from fixed-τ scalarization. If the constraint is inactive, PCD is just `a`. If all gradients are collinear, distinct coefficients may still give the same ray. After PCD’s final rescale, the update magnitude is `||g_CE||`; the scalarization control must use the identical EMA statistics and final norm to make the contrast about direction.

Consequently, S’s title/question should be sharpened from “does projection add value beyond normalization?” to **“does PCD’s priority-derived, gradient-dependent coefficient improve the primary/secondary trade-off over fixed EMA-normalized weights under measured conflict?”** It is not a test of projection versus all scalarization: a scalar baseline using per-step `λ_t=α_t` plus identical rescaling is algebraically identical in this active K=2 case and should be used as an implementation sanity check, not presented as an independent competitor. A static λ=tau test under conflict estimates the value of adaptivity over fixed weights.

**THEORETICAL GUARANTEE with caveat:** Corollary 4.6 establishes the active-set formula. Prop. 4.7’s normalized primary margin `1+τc−c²` can become nonpositive at strong conflict; PCD is not guaranteed to preserve CE descent. This does not make the experiment ill-posed. It means the finite-horizon trade-off itself is informative and CE ascent must be reported as a possible failure, not hidden as “secondary success.”

**VERIFIED, conditional:** §4.8 says that on genuine trade-off problems with no common stationary point, PCD with `τ>0` has no fixed point and iterates need not converge. It is not established here that every CE/KD pair, student architecture, and finite dataset has no common stationary point. Incompatible labels/teacher targets make that regime plausible; it remains **UNTESTED** for the proposed model. A nonstationary objective pair can still be compared over a predeclared finite schedule, though convergence claims must be withheld.

#### B4. M4 workload and minimum viable sequence

**VERIFIED counts / INTERPRETATION feasibility:** Six named arms do not equal six runs per corruption because CE-only does not depend on teacher corruption. If the remaining five arms (tuned raw weighted KD, matched EMA-normalized KD, ungated PCD, gated PCD, gated normalized KD) each run at all three corruption levels, that is 15 arm-condition cells plus one CE-only cell. This is 16 configurations × 3 pilot seeds = 48 pilot runs, then × 5 confirmatory seeds = 80 runs, or **128 runs before tuning overhead**. At 20 epochs and 50,000 training images, that is roughly 128 million training-example presentations; at 40 epochs, 256 million. PCD requires separately available CE/KD gradients and more backward work than a combined weighted-loss backward; frozen teacher inference is additional. No M4/MPS throughput measurement exists, so wall-clock feasibility cannot be responsibly asserted from the paper’s A10G number.

If weighted KD and τ each need a grid rather than one pilot-selected setting, even a modest four-value λ grid across 3 corruption levels and 3 pilot seeds adds 36 full runs; τ grid tuning adds further runs. The current arm/matrix description is therefore **not yet a realistic committed M4 plan**. **RECOMMENDATION:** first run a 3-seed × 3-corruption × 1-baseline conflict audit (9 short CIFAR-10 runs), log gradient cosines/constraint activity, and time the actual network. Proceed only if corruption yields meaningful negative cosine and PCD active steps; then narrow confirmation to clean + one corruption level, one λ/τ selected using validation, and 3–5 paired seeds. A full 0/10/30 × 6-arm × 5-seed design can follow only if the smoke timings and course calendar support it.

#### B5. Candidate S scores (1–5)

Scale: 1 weak/poor, 5 strong/good. These are judgments for S as presently specified; every score includes its rationale and evidence class.

- **(a) Theoretical compatibility — 2/5:** hierarchy semantics fit PCD, but KD is nonconvex, feasibility/realized-step guarantees are limited, and Prop. 4.7 allows primary ascent (**THEORETICAL GUARANTEE**, conditional/negative).
- **(b) Potential measurable advantage — 4/5:** parameter-gradient cosine, active rate, CE margin, accuracy and corruption-split outcomes are measurable; the effect on this CIFAR-100 recipe is unknown (**EXISTING EMPIRICAL EVIDENCE** in adjacent KD; **UNTESTED HYPOTHESIS** for S).
- **(c) Novelty — 2/5:** DTO-KD, MoKD, SCKD and SCOPE already address KD gradient conflicts/task priority; PCD plus explicit corruption/gating remains a narrow extension (**EXISTING EMPIRICAL EVIDENCE**; contribution **UNTESTED HYPOTHESIS**).
- **(d) Benchmarks — 5/5:** CIFAR-10/100 and ResNet teacher/student protocols are established and the proposal specifies splits/models (**EXISTING EMPIRICAL EVIDENCE**, course proposal).
- **(e) Reproducibility — 3/5:** seed-corruption maps and teacher/student recipe are implementable, but exact pretrained teacher checkpoints/recipe and MPS determinism need fixing (**VERIFIED** proposal-level protocol; exact run reproduction **UNTESTED HYPOTHESIS**).
- **(f) M4 feasibility — 2/5 as written; 3/5 if staged/reduced:** at least 128 base runs plus λ/τ tuning, with two network-gradient objectives and 20–40 epoch schedules; actual MPS time unknown (**UNTESTED HYPOTHESIS**, workload count only).
- **(g) MLR570 suitability — 5/5:** directly retains the submitted reliability-gated KD question and aligns with the survey’s KD/conflict scope (**VERIFIED** assignment documents).

#### B6. Decision: should S replace M1 as top-1?

**RECOMMENDATION:** **Yes, conditionally, for the course project shortlist; no unconditional approval of the full matrix.** S should replace compression audit M1 as the assignment-aligned top-1 candidate if the short pilot verifies nontrivial negative CE–KD cosine and meaningful PCD constraint activity under at least one controlled corruption level. If the correctness gate removes the conflict or PCD and fixed normalized KD produce indistinguishable trajectories, the project should report that null result or retain M1 only as a separate compression audit—not manufacture a conflict regime.

**Single strongest argument against S:** the proposed teacher-corruption × reliability-gate design may cancel its own mechanism: the gate rejects corrupted teacher outputs, leaving the gated PCD comparison in a low-conflict regime where the PCD update is nearly a fixed normalized weighted sum, while substantial prior KD conflict/projection work already makes the remaining PCD-specific novelty narrow. The unmeasured 128+ run matrix on one M4 adds a second practical risk, but the mechanism cancellation is the central scientific risk.

### Round-2 corrections to Codex-owned files

1. **`research/notes/experimental-feasibility.md`:** withdrew unsupported M4 throughput/runtimes; replaced ~2× compression cost with the paper-specific A10G <2% result; corrected FedPCD to 5–10 clients/round and 1,500–3,000 client-epochs/seed; added CIFAR archive sizes/citation requirement; clarified the repo-reported PCD prox step is absent from paper disclosure.
2. **`literature/repositories.yaml`:** recorded PCD repo first-commit date/HEAD and the repository’s claimed prox setting with explicit paper-disclosure caveat; changed FedPCD record to not released/blank code/null license; added CIFAR Python sizes and tech-report citation requirement.
3. **`research/notes/pcd-math-and-implementation.md`:** corrected compression outcome summary to 44/4/54 and 28/1/18; named the 53.8%-vs-51.9% crossing; replaced overstated “paper-aligned reproduction” with release limits and disclosed-protocol gap; corrected FedPCD repository status.
