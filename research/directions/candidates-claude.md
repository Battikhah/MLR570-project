# Candidate research directions for PCD / FedPCD (Claude worker)

Date: 2026-11-02. Orca task_94c59c478ef4. Evidence is in `research/notes/pcd-fedpcd-literature.md` and in
`literature/notes/*.md`; ids refer to `literature/sources.yaml`.

Labels: VERIFIED / INTERPRETATION / HYPOTHESIS / RECOMMENDATION; [THEORY] / [EMPIRICAL] / [UNTESTED].

Constraints assumed:
- Apple M4, 16 GB, MPS/CPU only, about 14 GiB disk; one semester; four students.
- No dataset download over ~200 MB without approval (AGENTS.md §2).

PCD cost is K backprops per step, plus a negligible QP for small K (VERIFIED, varam2026pcd Sec. 4.6). Every
"feasibility" statement below is an INTERPRETATION; nothing was run.

Scores are 1–5 (5 best) and are my judgement. **Fit** = theoretical fit of PCD/FedPCD. **Novelty** = after the
novelty search. **Feas.** = feasibility on this hardware and timeline. **Risk** = probability of an uninformative
outcome (5 = low risk).

| # | Direction | Fit | Novelty | Feas. | Risk | Notes |
|---|---|---|---|---|---|---|
| D1 | Current proposal: reliability-gated PCD-KD (CIFAR-100) | 2 | 2 | 2 | 2 | crowded; theory mismatch; large run matrix |
| D2 | "Projection or normalization?": independent replication + mechanism audit of PCD | 5 | 4 | 4 | 5 | any outcome is publishable as a finding |
| D3 | FedPCD audit/extension: drift controls, local epochs, fair search | 4 | 4 | 4 | 4 | no code: must re-implement |
| D4 | Centralized priority-constrained group robustness/fairness ("groups as clients") | 3 | 4 | 4 | 3 | dataset choice constrained by size |
| D5 | PCD for quantization-aware training | 3 | 3 | 3 | 3 | scooping risk from the PCD authors |
| D6 | PCD for continual learning (normalized-margin GEM) | 3 | 1 | 5 | 3 | prior art ≈ identical |
| D7 | PCD for accuracy–robustness (adversarial) training | 1 | 3 | 2 | 2 | outside theory; PGD cost |

---

## D1 — Reliability-gated PCD for knowledge distillation (current proposal)

- **Problem.** Does CE-primary, gated-KD-secondary PCD improve student accuracy and reduce negative transfer
  when teachers are unreliable (proposal pp. 2–3, `docs/requirements/assignment-requirements.md`)?
- **Theoretical fit.**
  - (+) There is a natural primary (labels).
  - (−) The KD-KL secondary is non-convex in θ, so it falls outside Assumption 3.1. Only the local smooth-case
    progress statement (Prop. 4.4(ii)) applies [THEORY scope, VERIFIED].
  - (−) CE (one-hot target) and KD (soft teacher target) generally have no common stationary point.
    INTERPRETATION: this is the "genuine trade-off" regime in which "the PCD direction with τ > 0 has no fixed
    point" (varam2026pcd Sec. 4.8).
  - (−) Under strong conflict (the unreliable-teacher case) Prop. 4.7 says PCD can stop descending CE.
  - (−) The gate already removes many conflicting examples, so the residual conflict that PCD resolves may be small.
- **Closest prior work / novelty.**
  - SCKD (`zhu2021sckd`): supervised-primary cosine gating.
  - SCOPE (`chen2026scope`): classification-priority conflict removal + confidence weighting.
  - TPKD (`yuan2026tpkd`): task-preserving projection.
  - DTO-KD (`hayder2026dtokd`): MOO KD.
  - CHAD (`sumit2026chad`): reliability-aware.
  - No paper applies PCD itself to KD (novelty search 2026-11-02). However, "classification has priority, project
    out conflicting KD gradient" is already published (SCOPE, TPKD).
  - INTERPRETATION: the remaining novelty is the specific gate × PCD combination and the corrupted-teacher stress
    test. That is incremental.
- **Contradicting evidence.**
  - Teacher accuracy is not utility (Yuan et al. CVPR 2020); fidelity is not generalization (Stanton et al. 2021).
  - Confidence-only gates gave little benefit in CHAD.
  - Tuned scalarization matches MTO methods (`xin2022mtohelp`, `kurin2022unitary`).
  - Momentum flips the secondary-progress sign often (PCD App. G).
- **Benchmarks.** CIFAR-100, CIFAR-10 (download size not checked here; Codex vets datasets). A ResNet-32 teacher must be trained or obtained.
- **Feasibility.**
  - Small students (ResNet-8/14) are cheap per run.
  - The planned matrix is large: about 11 methods × τ grid {0, 0.1, 0.3, 0.5} × 3 corruption levels × 5 seeds
    (+ momentum subset + ResNet-14). That is hundreds of runs at 2× backprop cost for gradient methods, plus
    re-implementing DTO-KD/SCKD/SCOPE.
  - HYPOTHESIS: this does not fit one semester on an M4 at standard schedules (e.g. 240 epochs) without cutting the
    matrix substantially.
- **RECOMMENDATION.** Keep D1 only as a reduced application study, ideally fused with D2's normalized-WS control:
  is PCD-KD better than EMA-normalized weighted KD with λ = τ? Do not keep it as the headline.
- **Citations:** varam2026pcd, zhu2021sckd, chen2026scope, yuan2026tpkd, hayder2026dtokd, sumit2026chad,
  hinton2015kd, yu2020pcgrad, liu2021cagrad, xin2022mtohelp, kurin2022unitary.

## D2 — "Projection or normalization?" A mechanism audit and independent replication of PCD

- **Problem.** PCD's headline gains are in structured pruning. There the authors measure cos(task, reg) ≤ 1.2e-3
  and attribute the gain to "objective scale rather than gradient conflict" [EMPIRICAL, VERIFIED].
  - With c ≈ 0 and an active constraint, the K = 2 closed form gives d̃* = g̃1 + τ g̃2 (INTERPRETATION from
    Cor. 4.6 / Prop. 4.7). That is an **EMA-normalized weighted sum with weight τ**, rescaled to ‖g1‖.
  - The paper's normalized-baseline appendix (E.2) used w ∈ {0.1..0.9}, i.e. λ_eff ≥ 0.111, so the matching
    λ ∈ [0.01, 0.09] control appears untested (VERIFIED wording, "same hyperparameter grid as the original
    submission").
  - Question: **does PCD's advantage come from the projection or from normalization?** And in which regimes
    (strong or varying conflict, K = 3) does the projection matter?
- **Theoretical fit.** Highest. Convex secondaries (group lasso, ℓ1, nuclear norm) are exactly Assumption 3.1. The
  analysis predicts precisely when PCD and normalized WS coincide (constraint always active and c ≈ 0) and when
  they differ (inactive regime, c < 0, K = 3 with competing secondaries).
- **Closest prior work / novelty.**
  - The PCD paper itself, plus DBGD (`gong2021dbgd`), GradNorm (`chen2018gradnorm`), Cooper.
  - No independent replication was found. The normalized-WS-at-matched-τ ablation is not in the paper.
  - Novelty is in evaluation and analysis, not a new method. INTERPRETATION: that suits a course project and also
    feeds directly into the survey's critical-analysis section (guideline G8).
- **Contradicting evidence.**
  - Authors' App. E.2: normalization alone did not rescue the baselines. That was on the w ≥ 0.1 grid, though.
  - If normalized WS matches PCD, PCD's value reduces to its τ interpretation. That is still a finding.
- **Benchmarks.** CIFAR-10/100 with small ResNets (ResNet-20/32) or the paper's models at reduced epochs. Synthetic
  conflict problems from PCD Sec. 5.3. Optional K = 3 (ℓ1 + nuclear norm).
- **Feasibility.**
  - K = 2 → 2 backprops/step. A τ sweep of about 6 points × 3 methods (PCD, normalized WS λ = τ, tuned raw WS)
    × 3 seeds on CIFAR-10 is about 54 runs.
  - The public MIT code lets the team check implementation details. Codex is auditing it.
  - Logging cos(g1, g2), constraint activity, and realized-step sign is cheap and directly tests the theory.
  - HYPOTHESIS: feasible on an M4 with ≤ 100-epoch schedules. A smoke test is needed to estimate per-epoch time.
- **Measurement plan sketch (RECOMMENDATION).**
  - Match on native sparsity as the paper does.
  - Use equal search budgets and report paired seed differences.
  - Add a regime where conflict is real: a primary plus a secondary with non-zero, sign-varying cosine.
- **Citations:** varam2026pcd, gong2021dbgd, chen2018gradnorm, gallegoposada2025cooper, xin2022mtohelp,
  kurin2022unitary, royer2023scalarization, lin2022randomweighting.

## D3 — FedPCD under client drift and fair comparison

- **Problem.** FedPCD reports the highest global and tail accuracy, but its evaluation has gaps (VERIFIED from the paper):
  - 1 local epoch only.
  - No FedProx/SCAFFOLD/FedNova controls.
  - A larger hyperparameter grid for FedPCD than for the baselines.
  - No validation split described.
  - Code not released.
  - Its theory needs an assumed uniform alignment margin γ_eff > 0 that local drift erodes.

  Questions:
  1. Does FedPCD's advantage survive matched search effort and drift-correcting baselines?
  2. How does it degrade with local epochs E ∈ {1, 2, 5}?
  3. Does a normalization-only control (aggregating unit-normalized client updates, rescaled to ‖ḡ‖) explain part
     of the gain?
  4. Does the empirical alignment γ_t measured during training predict when FedPCD helps? This is a direct test of
     the Thm 4.1 condition.
- **Theoretical fit.** Good. The aggregate-as-primary, clients-as-constraints hierarchy is natural. The theory is
  conditional, and its condition can be measured directly.
- **Closest prior work / novelty.** FedPCD itself (preprint, one month old); FedFV (`wang2021fedfv`, projection +
  history); FedMGDA+ (`hu2022fedmgda`); q-FFL/AFL/Ditto. No independent evaluation exists, and none of the four
  questions above is answered in the paper (VERIFIED by reading it).
- **Contradicting evidence.** FedFV/FedLF/CRAFT come close on several configurations. FedPCD's CIFAR-100 spread σ_c
  is higher than the baselines'. Absolute CIFAR-100 accuracy is low (~9–36%), so the regime is unusual.
- **Benchmarks.** Fashion-MNIST (MLP) and CIFAR-10 (3-layer CNN) with Dirichlet α ∈ {0.01, 0.1, 1}, 100 simulated
  clients, following `acar2021feddyn`. Both datasets are small (<200 MB).
- **Feasibility.**
  - Simulation on one machine is cheap with small models: 10% participation → about 10 client updates/round.
  - The QP has up to ~100 half-space constraints with history. A small dual projected-gradient / NNLS solver
    in torch is needed, because KKT enumeration is infeasible at that size. That is an engineering risk but a
    tractable one.
  - Baselines (FedAvg, FedProx, SCAFFOLD, q-FFL, FedFV) are simple to implement.
  - HYPOTHESIS: feasible within a semester at reduced rounds (e.g. 200–300).
- **Citations:** irshaid2026fedpcd, varam2026pcd, mcmahan2017fedavg, li2020fedprox, karimireddy2020scaffold,
  wang2020fednova, acar2021feddyn, li2020qffl, mohri2019afl, li2021ditto, hu2022fedmgda, wang2021fedfv.

## D4 — Centralized priority-constrained group robustness / fairness ("groups as clients")

- **Problem.** Average risk is the deliverable, while each protected group or subpopulation must keep making
  progress. This is the centralized analogue of FedPCD: the primary is the average-loss gradient, and the secondaries
  are per-group loss gradients (K = number of groups, typically 2–8).
- **Theoretical fit.**
  - Partial. Group losses are non-convex, so only the local progress statement applies.
  - K small means closed forms exist for K ≤ 3. Infeasibility for K > 2 is possible when groups conflict
    (Thm B.2), and the fallback is primary-only.
  - INTERPRETATION: priority is a better formulation than min–max (GroupDRO) when the average is the real
    deliverable and the group requirement is "do not regress" rather than "equalize".
- **Closest prior work / novelty.**
  - GroupDRO (`sagawa2020groupdro`), Lagrangian fairness constraints (`cotter2019twoplayer`,
    `agarwal2018reductions`, `chamon2020pacc`), FedMGDA+-style symmetric MOO.
  - Novelty search (2026-11-02) found no PCD or priority-projection fairness method. The "Lexicographically Fair
    Learning" line (Diana et al., arXiv 2102.08454, seen in search results only, NOT registered/verified) is related
    in spirit.
- **Contradicting evidence.**
  - Projection-style fairness fixes can satisfy a surrogate without reducing true disparity (arXiv 2101.12476,
    search snippet only, unverified).
  - Lagrangian methods target constraint *levels* directly, which may be what practitioners want.
- **Benchmarks.** The smallest options are tabular census-income data (e.g. UCI Adult, a few MB) and a synthetic
  Colored-MNIST-style split built from MNIST. Waterbirds/CelebA are standard but larger (size and licence not
  checked; Codex owns dataset vetting; approval needed if > 200 MB).
- **Feasibility.** High for tabular and MNIST-scale data (CPU-fast). K + 1 backprops per step.
- **Citations:** varam2026pcd, irshaid2026fedpcd, sagawa2020groupdro, cotter2019twoplayer, agarwal2018reductions,
  chamon2020pacc, hu2022fedmgda.

## D5 — PCD for quantization-aware training

- **Problem.** Task CE is primary, and a fidelity term pulling weights toward a low-bit grid is secondary. The PCD
  paper's intro lists QAT as a canonical hierarchy (VERIFIED, Sec. 1) and its conclusion names hardware-aware
  constraints as a frontier.
- **Theoretical fit.** Partial. Distance-to-grid is non-convex (periodic) and nonsmooth at grid midpoints, so it is
  outside Assumption 3.1. It is regularizer-like, though, and joint stationarity is plausible.
- **Closest prior work / novelty.** No PCD-QAT paper was found. Gradient-conflict QAT work exists in other forms
  (GAQAT, CAGE; search results only, not registered/verified). **High scooping risk:** the PCD authors have a
  quantization survey (`Abushahla et al. 2025`, arXiv 2508.15008, cited in PCD) and publicly list quantization as a
  next direction (search snippet).
- **Contradicting evidence.** Straight-through-estimator QAT baselines are strong and simple. Not checked in this pass.
- **Benchmarks.** CIFAR-10 with small CNNs/ResNets.
- **Feasibility.** Medium: QAT implementation details (fake-quant, bit-width schedules) add engineering risk.
- **Citations:** varam2026pcd; other QAT sources still need verification before use.

## D6 — PCD for continual learning (normalized-margin GEM)

- **Problem.** New-task loss is primary, and replay losses on episodic memory are secondary.
- **Theoretical fit.** Natural hierarchy; losses are non-convex.
- **Closest prior work / novelty. Very low novelty (VERIFIED).**
  - The PCD paper itself states that GEM and A-GEM "are algorithmically equivalent to PCD with τ = 0".
  - ε-SOFT-GEM (`hu2020softgem`) "adds a unit-normalized positive margin at K = 2".
  - GEM already reported a γ margin (`lopezpaz2017gem`, p. 5).
  - The only remaining delta is EMA normalization with per-task margins.
- **Contradicting evidence.** Simple experience replay is a strong baseline (`chaudhry2019tiny`; claim recalled,
  must be re-read before use).
- **Benchmarks.** Permuted/Rotated MNIST, Split CIFAR-100. Small and cheap.
- **Feasibility.** Very high, but the likely outcome is "≈ ε-SOFT-GEM".
- **RECOMMENDATION.** Do not pursue as the main project. At most, use it as a sanity case in D2.
- **Citations:** lopezpaz2017gem, chaudhry2019agem, hu2020softgem, farajtabar2019ogd, chaudhry2019tiny, varam2026pcd.

## D7 — PCD for the accuracy–robustness trade-off

- **Problem.** Clean CE is primary, and an adversarial or TRADES term is secondary. The reverse ordering is also
  possible.
- **Theoretical fit.** Poor. The PCD authors explicitly say adversarial losses fall outside their proofs (VERIFIED,
  Sec. 6). The accuracy–robustness relation is a genuine trade-off, i.e. the no-fixed-point regime.
- **Closest prior work.** CA-AT (`xue2024caat`) already projects the adversarial gradient based on its angle to the
  clean gradient. TRADES (`zhang2019trades`) is the standard trade-off objective.
- **Feasibility.** Low. PGD inner loops multiply cost (several forward/backward passes per step) on MPS.
- **Citations:** zhang2019trades, xue2024caat, varam2026pcd.

---

## Overall recommendation (RECOMMENDATION, INTERPRETATION)

1. **Primary pick: D2**, optionally with a reduced D1 or a D4 application as the "transfer" experiment.
   - It has the strongest theoretical fit, a sharp and falsifiable question, and public author code.
   - It is cheap at K = 2, and any outcome is informative.
   - It also strengthens the survey's critical-analysis section.
2. **Strong alternative: D3** if the team wants the federated angle.
   - Its novelty is high relative to a one-month-old preprint, and there are clear, verifiable gaps.
   - It needs a QP solver for many constraints and a full re-implementation, because no code exists.
3. **D1 as currently scoped is not recommended as the headline.**
   - The literature is crowded: SCOPE, TPKD and SCKD already give classification priority over KD.
   - The KD secondary falls in PCD's no-fixed-point / possible-primary-ascent regime.
   - The run matrix is too large for an M4 in one semester.
4. Avoid D6 (prior art is essentially identical) and D7 (outside theory, expensive). D5 only if the team accepts
   the scooping risk.

Whatever is chosen, include:
- a tuned scalarization at equal search budget;
- an EMA-normalized weighted sum at λ = τ;
- logged cos(g1, g2), constraint-activity rate, and realized-step sign;
- paired seed differences.
