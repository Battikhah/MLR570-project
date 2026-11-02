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
