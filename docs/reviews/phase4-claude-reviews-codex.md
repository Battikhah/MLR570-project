# Phase 4 cross-review, round 1 of max 2: Claude reviews Codex

- **Date:** 2026-11-02.
- **Reviewer:** Claude (Orca task_1c5651b8a557).
- **Reviewed:** branch `Battikhah/research-math-codex` @ `d1cd580`: `research/notes/pcd-math-and-implementation.md` (MATH),
  `research/notes/experimental-feasibility.md` (FEAS), `literature/repositories.yaml` (REPO).
- **Also re-examined:** my own branch `Battikhah/research-literature-claude` @ `1bec167`.

**Verdicts.** AGREE / DISAGREE / UNSUPPORTED.

**Labels.**
- Claim status: VERIFIED / INTERPRETATION / HYPOTHESIS / RECOMMENDATION.
- Evidence type: **[TG]** theoretical guarantee, **[EE]** existing empirical evidence, **[UH]** untested hypothesis.

**Primary sources.**
- PCD: arXiv 2606.29521v2 PDF, cited as "PCD §x / p.y".
- FedPCD: arXiv 2610.13508v1 PDF, cited as "FedPCD §x".
- PCD code: clone at /tmp/pcd-review/pcd-code, HEAD `e9afdbc`, cited as "code path:line".
- Numeric checks: the script described in §1.4.

---

## 1. Specific checks assigned to Claude

### 1.1 Static audit of `DaraVaram/priority-constrained-descent`

The clone succeeded (`git clone --depth 50` into /tmp/pcd-review/pcd-code; nothing installed).

| Item | Finding (VERIFIED unless labelled) |
|---|---|
| HEAD | `e9afdbc9f4cb09343934eadb31dc72ea5ebac9b0`, 2026-09-29 01:30:25 +0400, "Add tutorial notebook, PyPI publishing, fuller citation metadata". Only 3 commits: `8000bb8` and `713a4a9` on 2026-09-21, then `e9afdbc`. |
| License | `LICENSE` line 1–3: "MIT License / Copyright (c) 2026 Dara Varam and Mohamed I. AlHajri". `pyproject.toml`: `license = "MIT"`. |
| Dependencies | `requires-python = ">=3.9"`; `dependencies = ["numpy", "torch>=1.13"]`; extras `experiments=[torchvision]`, `examples=[matplotlib]`, `test=[pytest]`. No pins, no lock file. |
| Tests | `tests/test_qp.py`: closed form K=2, Cramer K=3, Dykstra-projection equivalence for general K, inactive, infeasible, zero-secondary, per-objective τ. `tests/test_optim.py`: Alg. 1 first step, rescale, scale invariance, halting, state dict, a training loop. `pcd/__init__.py` imports torch, which is not installed here, so the suite cannot run unmodified. **I ran `test_qp.py` against a numpy-only shim exposing `pcd/qp.py` (read in full first): 77 passed, exit 0.** `test_optim.py` was not run (needs torch). CI runs both on Ubuntu, CPU torch (`.github/workflows/tests.yml`). |
| QP | `pcd/qp.py:27-76`. Working-set KKT enumeration on the normalized Gram matrix (`optim.py:101` passes `gram * outer(scale, scale)`). `b = τ·diag(Gs) − G[1:,0]` (l.56); inactive shortcut (l.57, Cor. 4.3); zero secondaries skipped (l.61); linear-independence test (l.79-84); infeasible → `weights = e_0`, i.e. primary only (l.76, Alg. 1 l.13-14). **Matches PCD Def. 4.1, Thm 4.2, App. B.2.** |
| EMA | `pcd/normalize.py:34-38`: v ← βv+(1−β)‖g‖², v̂ = v/(1−βᵗ), s = 1/√(v̂+ε). **Matches PCD Def. 4.5.** The comment at l.36-37 says the research code used 1/(√v̂+ε). |
| Rescale | `pcd/optim.py:97-99`: g1 = 0 → zero update (PCD §4.7). `optim.py:107-110`: d* = d̃*·‖g1‖/‖d̃*‖, with d̃* = 0 → 0. **Matches Alg. 1 l.17-20.** |
| τ validation | `optim.py:54` rejects only τ < 0. **τ > 1 is accepted silently** (checked: `solve_qp(...,1.5)` returns feasible). This is outside the paper's τ ∈ [0,1]. |
| Seeds | `experiments/train.py:138-139` seeds `random`, `numpy`, `torch`. `cudnn.benchmark=True` (l.137); no `use_deterministic_algorithms`; DataLoader shuffle uses the global RNG with `num_workers=4` default. GPU runs are therefore not bit-reproducible. |
| Data path | `train.py:68-72`: torchvision CIFAR10/100 `download=True` into `--data-dir ./data`. |
| Device | `train.py:59`: default `cuda` else `cpu`. MPS must be passed explicitly. **HYPOTHESIS [UH]:** `optim.py:87` calls `.double()` on device tensors before `.cpu()`. MPS has no float64 (multiple PyTorch-forum / Lightning reports, e.g. discuss.pytorch.org/t/164655, Lightning #21261), so this likely raises on the M4. A one-line local reorder (`.cpu().double()`) would be needed. Not executed. |
| Reproduces paper tables? | **Partly.** `experiments/README.md` "Not included": the **physical-pruning export** behind Fig. 4, Tables 1, 2, 6, 7 and the FLOPs/latency/size numbers "is not part of this release yet", and **"This repository contains PCD only"** (no baselines). `sweeps.sh` reproduces PCD's native-sparsity runs: 800 K=2 + 180 K=3 runs × 300 epochs, about 3.6 h per ResNet-34/CIFAR-100 run on an A10G. The release is a **rewrite**; the authors report agreement with the research solvers to 1e-11 / 2e-10 relative. |
| **Undisclosed proximal step** | `experiments/objectives.py:31-32`: "The paper's runs applied one [proximal step] after every optimizer step with threshold lambda * lr". README: λ = 1e-3 (5e-4 MobileNetV2). `sweeps.sh` l.13-17 passes `--prox` to **every** PCD run, including τ = 0. **Paper App. H (p. 51) and §5 never say PCD runs used a proximal step**; "prox" appears only as the proximal-GL *baseline* (PCD §5.1.2, p. 17). Whether the baselines also received it is UNVERIFIED. INTERPRETATION: PCD's reported native group sparsity (‖θ_g‖ < 1e-4) is PCD + proximal snap. Mitigating evidence [EE]: in App. E.4 (ResNet-34/CIFAR-100 table) PCD's τ = 0 matched point shows 0.00% native sparsity, and proximal GL alone at λ = 0.01 shows 0.00%. So the λ = 1e-3 snap on its own creates no measurable sparsity. It may still interact with PCD's pressure at τ > 0, which is untested [UH]. |
| Model selection | VERIFIED, both in the README checklist and App. H: "No held-out validation split was used"; baselines at their highest mean test accuracy, PCD at fixed τ = 0.02. (Test-set selection favours the baselines here, but it is still test-set selection.) |
| Normalized-WS baseline / λ grid | **Not in the code** (no baselines). Only the paper text applies: App. E.2 (p. 37) ran EMA-normalized baselines "over the same hyperparameter grid as the original submission", i.e. w ∈ {0.1,…,0.9} → λ_eff = w/(1−w) ≥ 0.111. My D2 premise therefore still rests on the paper text and cannot be checked against code. The author account also holds forks of `LibMTL` and `TorchJD` (created 2026-09-28), which might host baselines; not inspected (UNVERIFIED). |
| Affiliation (EPFL vs AUS/MIT) | The GitHub profile (`api.github.com/users/DaraVaram`) shows company "EPFL", location Lausanne. The paper gives AUS + MIT Senseable City Lab; the PCD/FedPCD pages give varam@mit.edu. Same-account evidence: github.io project pages for both papers, `CITATION.cff` naming both authors, and AUS course repos `MLR503-Final`, `MLR510-Final`, `MLR555-Advanced-AI`, `COE49412` (same MLR course prefix as MLR570). INTERPRETATION: the same person, whose profile is newer than the paper's affiliation. Provenance is adequate but not author-confirmed. |

Registration: recorded in my notes (`literature/notes/varam2026pcd.md` §"Code audit"). Proposed corrections to
Codex's `literature/repositories.yaml` are in §3.

### 1.2 FedPCD "Code" link

**Resolved (VERIFIED).**
- daravaram.github.io/FedPCD/ wires its buttons from `js/main.js`, where
  `LINKS = { paper: "https://arxiv.org/pdf/2610.13508", arxiv: "https://arxiv.org/abs/2610.13508", code: "" }`
  with the comment `// code release, e.g. "https://github.com/DaraVaram/fedpcd"`.
- The repo `DaraVaram/FedPCD` README: an empty link "says 'Coming soon'"; "Code is empty until the code is public";
  the paper is a "preprint, under review".
- The repo contents are only `README.md, index.html, css/, js/, assets/, .nojekyll`.
- **FedPCD code is not publicly released as of 2026-11-02** (an announced "Coming soon" link; this does not mean no code exists privately).
- For comparison, the PCD page's `main.js` has `code: "https://github.com/DaraVaram/priority-constrained-descent"`.

### 1.3 Codex's math, checked against the papers and numerically

| Codex claim (MATH) | Verdict | Evidence |
|---|---|---|
| QP, KKT d̃* = g̃1 + Σμ_j g̃_j, μ ≥ 0, sign convention θ ← θ − ηd | AGREE | PCD Def. 4.1, Thm 4.2 (p. 6–7), Prop. 4.4 sign convention (p. 8); `qp.py` docstring l.3-11. |
| K=2: g̃1 if bᵀa ≥ τ‖b‖², else a + (τ − bᵀa/‖b‖²)b; b = 0 → a | AGREE | PCD Cor. 4.6 (p. 9). Numerically, 2000 random draws: max \|closed form − code\| = 4.4e-16. `qp.py:61` skips zero secondaries. |
| Gram system G_A μ_A = τ·diag(G_A) − G_{A,1} | AGREE | `qp.py:56,68`; PCD App. B.2. Codex's caveats on linear independence and degenerate sets match `qp.py:66` (`_independent`). |
| Farkas: τ > 0 infeasible iff 0 ∈ conv of the nonzero secondaries; τ = 0 always feasible; anti-parallel example | AGREE | PCD §4.1 (p. 6), Thm B.2 (p. 30). Numeric: secondaries (b, −b), τ = 0.1 → `feasible=False`; τ = 0 → `True`. |
| g̃1ᵀd̃* = 1 + τc − c²; with c < 0 the "necessary strict condition" τ < (1−c²)/(−c) | AGREE (wording) | PCD Prop. 4.7 (p. 10) states it as **iff** (necessary *and* sufficient) in the active regime. Identity confirmed numerically to 1e-12. |
| EMA definition, asymptotic scale invariance, rescale not a descent guarantee | AGREE | PCD Def. 4.5, Thm B.9; Prop. 4.7 shows the primary can ascend. |
| Convergence: no global theorem; sketch needs smoothness, boundedness, feasibility, uniform alignment, small η | AGREE | PCD §4.7 ("We prove no convergence theorem for Algorithm 1", p. 10) and App. C.3 (p. 36): L1 L-smooth and bounded below, feasible at every iterate, bounded iterates, α̃_t ≥ μ0, η ≤ μ0/L. The sketch certifies primary stationarity only, and the authors say they "do not claim it as a contribution". |
| FedPCD theorem assumptions: full participation, deterministic local GD, fixed K local steps, H = 0, fixed weights, L-smooth F_i, F bounded below, rate bounds, QP feasibility, uniform alignment γ; no convexity, no bounded gradients | AGREE | FedPCD §4 (Thm 4.1, Eq. 13–16) and App. B.1 ("We require neither convexity nor a uniform bound on the local gradients. Stochastic gradients, partial participation, and historical directions are outside this theorem's scope."). Codex should add the explicit drift condition γ_eff = γ(1−r) − r > 0 with r from Eq. 14. It is the only place client drift enters, and it shrinks as local steps or Lη_ℓ grow. |
| "For one secondary all τ ∈ [0,1] feasible; τ range is prescribed, not a feasibility limit" | AGREE + note | PCD §4.1. The code does not enforce τ ≤ 1 (`optim.py:54`). |
| "Primary zero → QP has nonzero solution, deployed step is zero" | AGREE | PCD §4.7 "deployed fixed point" (p. 11); `optim.py:97`. |
| DBGD equivalence is direction-only | AGREE | PCD §4.1 (p. 6). |

### 1.4 Numeric-check reproducibility

The scratch script is in my scratchpad, outside the repo. It uses numpy 2.2.6 and `pcd/qp.py` from `e9afdbc`, and checks:
- 2000 random K=2 draws: closed form vs code, plus the identity 1 + τc − c²;
- the anti-parallel Farkas example;
- with exactly orthogonal unit g̃1 ⊥ g̃2, ‖PCD − (g̃1 + τg̃2)‖ ≈ 1e-17 for τ ∈ {0.01, 0.05, 0.1}.

The last point confirms that in the c = 0 active regime the PCD direction **is** the normalized weighted sum with
λ = τ [TG, by Cor. 4.6]. macOS-Accelerate BLAS printed spurious matmul RuntimeWarnings; every result was finite.

---

## 2. Claim-by-claim verdicts on Codex's other substantive claims

| # | Codex claim | Verdict | Evidence / correction |
|---|---|---|---|
| 1 | PCD "arXiv v2 2026-09-21; the author project page states TMLR 2026" | AGREE, evidence can be stronger | TMLR accepted-papers list (jmlr.org/tmlr/papers/, "September 2026", forum HT01yGHLEt); TMLR bib; arXiv journal_ref. Status should be **published (VERIFIED)**, not only "page states". |
| 2 | FedPCD "marked under review on the author project page" | AGREE | `DaraVaram/FedPCD` README: "preprint, under review". (This corrects my own record; see §5.) |
| 3 | FedPCD caveats: FedLF beats it on worst-decile and spread in one Fashion-MNIST setting; CIFAR-100 spread higher; final-round results | AGREE | FedPCD Table 1 (FMNIST 10% α = 0.01: FedLF W10 65.73 / σc 9.34 vs best FedPCD 63.22 / 10.43); §5.2 text on CIFAR-100 σ_c; App. A "model obtained at the final communication round". |
| 4 | "FedPCD uses best τ per setting in some headline text; distinguish fixed τ = 0.02" | AGREE in method, but the effect is small | My count from Table 1 at fixed τ = 0.02: best global accuracy **18/18**, best W10% **15/18**. The losses are FMNIST 10% α = 0.01 and α = 0.1, and CIFAR-10 10% α = 0.01 (56.71 vs FedLF 56.93). The paper claims 17/18 for W10 with τ chosen per configuration. **[EE]** This is evidence *for* FedPCD that neither agent had stated. |
| 5 | "In PCD's own compression study tuned group-lasso scalarization comes within reported seed noise at one headline compression point" | **DISAGREE** (understates the contrary evidence) | PCD Table 3 (p. 19): full range, PCD better 44, **worse 4**, within noise **54** of 102; compression regime ≥ 85%: better 28, worse 1, noise 18 of 47. The one compression-regime loss is a genuine crossing: tuned scalarization 53.8% at 98.1% sparsity vs PCD 51.9% at 97.9%, ResNet-34/CIFAR-100 (p. 18). Ties are the majority of the full range, not a single point. |
| 6 | Compression advantage arose mainly from scale mismatch; cosine ≤ 1.2e-3 | AGREE (cite paper, not page) | PCD §5.1.1 p. 14: "The cause is objective scale rather than gradient conflict … cosine … never exceeds 1.2 × 10⁻³." |
| 7 | Repo: structure, MIT, README entry points, tests compare closed forms / KKT / independent projection | AGREE | Confirmed by clone (§1.1). Codex's UNVERIFIED items (HEAD, deps, configs, seeds, data path) are now resolved in §1.1. |
| 8 | PCD repo provides "paper-aligned experiment scripts" for the CIFAR studies | **PARTLY DISAGREE** | The scripts reproduce PCD training runs and native sparsity only. No baselines, no physical-pruning export, and the runs include an undisclosed proximal step. The headline Tables 1–2 and Fig. 4 cannot be reproduced from the release (`experiments/README.md` "Not included"). |
| 9 | FedPCD Code link destination unresolved | Superseded | Resolved: empty, "Coming soon" (§1.2). |
| 10 | FedPCD normalizes per vector instead of EMA; staleness can cause infeasibility | AGREE / INTERPRETATION | FedPCD Eq. 9 and §3.2 (EMA examined in App. D). Infeasibility with history is plausible (Thm B.2 logic) [UH]; FedPCD does not report feasibility rates (not found in main text). |
| 11 | FEAS objective 3 (KD): novelty "confounded with sample selection" | AGREE, incomplete | Codex's files do not cite the closest KD prior art: SCOPE (`chen2026scope`, classification-priority conflict removal), TPKD (`yuan2026tpkd`, task-preserving projection), SCKD (`zhu2021sckd`). Codex also omits the theory mismatch: CE and KD generally have no common stationary point, which is the "no fixed point" regime (PCD §4.8 p. 12), and Prop. 4.7 allows CE ascent under the strong-conflict / bad-teacher regime the proposal targets. |
| 12 | FEAS recommendation: the strongest direct experiment is small CIFAR-100 KD, *if* gates are isolated | **DISAGREE** on ranking | See §4 scores. KD has the best assignment fit but the weakest theoretical fit and the most crowded prior art. |
| 13 | FEAS compute: KD student throughput "100–300 images/s" on M4 | **UNSUPPORTED** | No measurement or source. Scale check: the PCD authors report about 3.64 h for 300 epochs of ResNet-34/CIFAR-100 on an A10G (App. H), i.e. roughly 50,000 × 300 / 13,100 s ≈ 1,100 img/s **including** PCD. ResNet-8 has about two orders of magnitude fewer FLOPs than ResNet-34. HYPOTHESIS: M4/MPS throughput for ResNet-8 is well above 300 img/s, so Codex's hours are likely pessimistic. It needs a smoke test. |
| 14 | FEAS compute: compression "if gradients are computed in separate passes" ≈ 1.0–3.5 h/run, i.e. ~2× | **DISAGREE** | PCD App. H: K=2 PCD costs 3.64 h vs WS 3.58 h (< 2% overhead). The group-lasso gradient is a cheap weight-only function, not a second network backward. (My own D2 text claimed "2 backprops/step"; corrected, §5.) |
| 15 | FEAS compute: FedPCD "30 sampled clients × 300 rounds = 9,000 client-epochs per seed" | **DISAGREE** (arithmetic) | FedPCD §5.1: N = 100, ρ ∈ {0.05, 0.1} → 5–10 clients/round, so 1,500–3,000 client-epochs per seed over 300 rounds. With about 600 FMNIST samples per client on average and a 784-256-128-10 MLP, HYPOTHESIS: minutes, not "2–12 hours", per seed on an M4 CPU. Needs a smoke test. |
| 16 | Datasets: CIFAR page has no license statement | AGREE | cs.toronto.edu/~kriz/cifar.html: no license; "please cite the tech report"; python archives CIFAR-10 163 MB, CIFAR-100 161 MB (sizes should be added to REPO). |
| 17 | Fashion-MNIST MIT may describe code, not data | AGREE | The README "License" section's MIT text covers "this software and associated documentation files"; the GitHub API reports MIT. The data's status is legitimately ambiguous. |
| 18 | FEAS objective 4: CIFAR-10-C/100-C usable | UNSUPPORTED | No size, licence or provenance given by Codex; not checked by me either. |
| 19 | FEAS statistics: paired seeds, validation-only τ selection, equal tuning budget, preregistered contrast, falsification criteria | AGREE | Note the irony: PCD itself (App. H) used no validation split. FedPCD describes none and selected baselines "after averaging the five seeds" on reported metrics (App. A). |
| 20 | MATH: "PCD is not strict lexicographic optimization" | AGREE (INTERPRETATION) | Consistent with Prop. 4.7. DBGD (`gong2021dbgd`) is the lexicographic method with convergence proofs. |

---

## 3. Proposed corrections to Codex's `literature/repositories.yaml` (Codex owns it; not edited)

1. `varam2026pcdcode`:
   - `date: 2026-09-21` (first commit `8000bb8`).
   - Add HEAD `e9afdbc9f4cb…` (2026-09-29) to `verified_by` with "shallow clone 2026-11-02 (Claude)".
   - Add to `relevance`: "rewrite of research code; PCD only, no baselines; physical-pruning export not released;
     published runs used `--prox` (λ = 1e-3) not disclosed in paper".
   - `license: MIT` is confirmed by the LICENSE text.
2. `irshaid2026fedpcdcode`:
   - Change the title to "FedPCD code (not released)" and keep `official: null`.
   - Set `license: null` (n/a).
   - `verified_by`: "project page js/main.js `code: ""`, button 'Coming soon'; FedPCD repo README 'Code is empty until
     the code is public' (checked 2026-11-02)".
   - Status stays `unverified`, or use a note "not released".
3. `cifar10` / `cifar100`: add archive sizes (163 MB / 161 MB python versions) and the citation requirement
   ("please cite the tech report", Krizhevsky 2009).

---

## 4. Merged candidate list and scores

Merging rules:
- **M1** = Claude D2 (projection-vs-normalization audit) + Codex objective 1 (compression primary task). Same
  application; D2 adds the mechanism question.
- **M2** = Claude D3 + Codex objective 2 (federated client progress).
- **M3** = Claude D1 + Codex objective 3 (KD; the submitted proposal).
- **M4** = Claude D4 + Codex objective 5. Group robustness / demographic fairness: per-group-loss secondaries (D4) or
  a fairness-surrogate secondary (Codex 5).
- **M5** = Claude D7 + Codex objective 4. Robustness: adversarial (D7) or corruption (Codex 4).
- **M6** = Claude D5 (QAT), Claude only.
- **M7** = Claude D6 (continual learning), Claude only.

Criteria:
- (a) theoretical compatibility with PCD/FedPCD;
- (b) potential measurable advantage;
- (c) novelty;
- (d) benchmarks;
- (e) reproducibility;
- (f) feasibility on M4/16 GB/MPS/14 GiB in one semester;
- (g) MLR570 suitability.

Scores are 1–5, 5 best. Each justification carries its evidence type.

### M1: Compression/sparsity, with the projection-vs-normalization audit

**Total: 27/35.**
- **(a) 5.** Convex secondaries (group lasso, ℓ1, nuclear) are exactly Assumption 3.1 [TG, PCD §3.1].
- **(b) 3.** Gains over *tuned* baselines are real but moderate (44 W / 4 L / 54 ties, Table 3) [EE]. The audit's
  likely result is "PCD ≈ normalized WS when c ≈ 0" [TG via Cor. 4.6]. Any measurable difference outside that regime
  is [UH].
- **(c) 3.** PCD already ablates normalization (App. E.1/E.2), but not at λ = τ; there is no independent
  replication; the proximal-step confound is new [UH].
- **(d) 4.** CIFAR-10/100 with native sparsity, FLOPs and latency are standard. The physical-pruning export is not
  released [VERIFIED].
- **(e) 4.** MIT code with QP tests passing (77/77 via shim), fixed seeds and sweeps listed. Minus one for the
  rewrite-vs-research-code gap, no baselines, the undisclosed prox step, and cuDNN non-determinism.
- **(f) 4.** Secondary cost is < 2% (App. H) and small ResNets suffice. Sparsity appears only at the end of the
  cosine schedule (README), so shortened runs must keep the schedule shape. A probable MPS float64 patch is needed [UH].
- **(g) 2.** Departs from the submitted KD proposal and from the survey's KD focus. It fits the survey's critical
  analysis (G8) only indirectly.

### M2: Federated client-progress fairness (FedPCD audit/extension)

**Total: 24/35.**
- **(a) 4.** Aggregate-as-primary is natural, and an O(1/T) rate exists under assumed alignment γ_eff > 0 [TG,
  conditional].
- **(b) 4.** Large reported gains that persist at fixed τ (18/18 accuracy, 15/18 W10) [EE, single group, no
  independent replication].
- **(c) 4.** There is no independent evaluation; drift controls and multi-epoch behaviour are untested (FedPCD §6) [UH].
- **(d) 4.** FMNIST/CIFAR-10 Dirichlet splits are standard (Acar et al. protocol); the FMNIST licence is ambiguous.
- **(e) 2.** Code not publicly released as of 2026-11-02 (§1.2); full re-implementation plus a many-constraint QP solver are needed;
  hyperparameter selection is not described.
- **(f) 4.** Small models; corrected arithmetic gives 1.5–3k client-epochs per seed (§2 #15) [UH timing].
- **(g) 2.** Topic change from the KD proposal; weak link to the KD survey.

### M3: KD with CE primary and gated KD secondary (submitted proposal)

**Total: 22/35.**
- **(a) 2.** The KD-KL secondary is non-convex (outside Assumption 3.1). CE and KD generally lack a common stationary
  point, the "no fixed point" regime (PCD §4.8) [TG negative]. Prop. 4.7 allows CE ascent under strong conflict [TG].
- **(b) 3.** Negative transfer under corrupted teachers is measurable. Whether PCD beats gated weighted KD at λ = τ
  is [UH]. Tuned scalarization often matches MTO methods (`xin2022mtohelp`) [EE, contrary].
- **(c) 2.** SCOPE, TPKD and SCKD already prioritise classification over KD [EE]. PCD-KD itself is unpublished, but
  the delta is incremental.
- **(d) 4.** CIFAR-100 (161 MB) with ResNet teacher/students is standard.
- **(e) 3.** The PCD library is reusable (`apply_gradients`, `optim.py:75`). DTO-KD/SCKD/SCOPE code is not verified;
  the teacher must be trained.
- **(f) 3.** About 2× cost (two network backwards). Feasible only after shrinking the proposal's matrix (about 11
  methods × τ × corruption × 5 seeds).
- **(g) 5.** It is the submitted proposal, and the survey is on KD gradient conflict.

### M4: Group robustness / demographic fairness

**Total: 20/35.**
- **(a) 3.** Groups-as-constraints mirrors FedPCD. The losses are non-convex; infeasibility is possible for K > 2
  (Thm B.2) [TG partial].
- **(b) 3.** Worst-group and average accuracy are measurable; the advantage over GroupDRO or Lagrangian methods is
  [UH].
- **(c) 4.** No PCD fairness work found (search 2026-11-02) [VERIFIED absence, search-limited].
- **(d) 2.** Small tabular sets (Adult) are weak benchmarks. Waterbirds/CelebA sizes and licences are unchecked.
  Codex flags legal and social validity.
- **(e) 3.** No reference code; the baselines (GroupDRO, reductions) are well known.
- **(f) 4.** Tabular or MNIST-scale data is cheap.
- **(g) 1.** Unrelated to the KD proposal or the survey.

### M6: Quantization-aware training

**Total: 18/35.**
- **(a) 2.** The distance-to-grid secondary is non-convex and periodic [TG does not apply].
- **(b) 3.** Accuracy at a fixed bit-width is measurable; any advantage is [UH].
- **(c) 3.** No PCD-QAT paper found, but the authors flag quantization as future work: scooping risk.
- **(d) 3.** CIFAR with small CNNs; deployment latency on an M4 is not representative.
- **(e) 3.** The PCD library is reusable; QAT pipeline choices add variance.
- **(f) 3.** Fake-quantization engineering on MPS is untested.
- **(g) 1.** Unrelated to KD.

### M7: Continual learning (normalized-margin GEM)

**Total: 20/35.**
- **(a) 3.** The current-task-primary hierarchy is natural; the losses are non-convex.
- **(b) 2.** The likely result is ≈ ε-SOFT-GEM; ER is a strong simple baseline [EE, recalled].
- **(c) 1.** The PCD paper itself says GEM/A-GEM are "algorithmically equivalent to PCD with τ = 0", and
  ε-SOFT-GEM adds the normalized margin (PCD §2, p. 4) [VERIFIED].
- **(d) 4.** Permuted MNIST and Split CIFAR-100 are standard and small.
- **(e) 4.** Simple to implement; GEM baselines are well known.
- **(f) 5.** Very cheap.
- **(g) 1.** Unrelated to KD.

### M5: Robustness (adversarial or corruption)

**Total: 15/35.**
- **(a) 1.** The PCD authors exclude adversarial losses from the proofs (PCD §6); it is a genuine trade-off.
- **(b) 3.** Clean and robust accuracy are measurable; any advantage is [UH].
- **(c) 2.** CA-AT already does gradient-conflict projection for AT (`xue2024caat`) [EE].
- **(d) 3.** CIFAR-10 (+ CIFAR-10-C, unverified size/licence).
- **(e) 3.** Standard attacks exist; no PCD-specific code.
- **(f) 1.** PGD inner loops multiply cost on MPS. The corruption-only variant is cheaper, but its secondary is ill-defined.
- **(g) 2.** Weak link to KD.

### Ranking

M1 (27) > M2 (24) > M3 (22) > M4 = M7 (20) > M6 (18) > M5 (15).

INTERPRETATION: the totals are unweighted sums of my scores. A coordinator who weights (g) heavily would move M3 up.

### Top-1 and backup (RECOMMENDATION)

- **Top-1: M1.** Projection-vs-normalization audit and independent replication on CIFAR-10 compression, with every
  arm run with and without the proximal step.
  - The decisive contrast: PCD(τ) vs an EMA-normalized WS with λ = τ vs a tuned raw WS, at matched native sparsity
    and equal tuning budget, selected on a held-out validation split.
  - Log cos(g1, g2), constraint-active rate, infeasible rate, and realized-step sign.
  - Add one engineered regime with sign-varying conflict, where the PCD theory says the two methods must differ.
- **Backup: M3, reduced** (the submitted KD proposal), not M2. M3 keeps the course proposal and the survey coherent,
  while M2 needs a full re-implementation with no reference code.
  - Reduction: CE-only, gated weighted KD, gated *normalized* weighted KD at λ = τ, ungated PCD, gated PCD; one
    student; 3 seeds; 10%/30% corrupted-teacher ablation.
  - Build in the M1 control so the result answers "does the projection matter here?".
  - M2 is the third choice. It has the strongest published evidence of a gain, but a weak (e) score.
  - This **revises my phase-3 file**, which named D3 (= M2) as the strong alternative. The reasons for the revision
    are §1.2 (no code) and criterion (g).

### Strongest argument AGAINST my top-1 (M1)

1. **The answer may be predictable and therefore low-value.** In the regime the paper itself measured (c ≤ 1.2e-3,
   constraint always active, App. E.3), Cor. 4.6 implies PCD *is* a normalized WS with λ = τ. My numeric check shows
   the difference is about 1e-17 at c = 0. An experiment confirming algebra is not research; the only informative
   part is the engineered-conflict regime, which is synthetic and may not convince a grader.
2. **The assignment risk is real.** The submitted MLR570 proposal is about reliability-gated KD, and the survey is
   about KD gradient conflict. Switching to compression could be judged off-proposal; no rubric or deadline is
   stated (`docs/requirements/assignment-requirements.md`).
3. **The replication may be confounded rather than clarified.** The release omits the baselines and the
   physical-pruning export, and the published runs used an undisclosed proximal step. Matching the paper's numbers
   on an M4 at shortened schedules may be impossible: sparsity "appears in the last few epochs" of a 300-epoch
   cosine schedule. A null result could then be blamed on the replication setup.

---

## 5. Corrections to my own previously committed files (Claude branch)

| # | File | Correction |
|---|---|---|
| 1 | `literature/sources.yaml` (`irshaid2026fedpcd.verified_by`) | Was "No venue/acceptance information found". Now: README "preprint, under review"; `js/main.js` `code: ""` ("Coming soon"); repo contents listed. |
| 2 | `literature/sources.yaml` (`varam2026pcd.verified_by`) | "code audit delegated to Codex" replaced by the clone HEAD `e9afdbc`, LICENSE copyright, and a pointer to the audit notes. |
| 3 | `literature/notes/varam2026pcd.md` | Added §"Code audit": QP, EMA and rescale file:lines; tests 77/77 (shim); seeds; data path; MPS float64 risk; release is a rewrite; no baselines; physical-pruning export missing; **undisclosed proximal step**; test-set selection (App. H); < 2% K=2 overhead; affiliation provenance. |
| 4 | `literature/notes/irshaid2026fedpcd.md` | Status "under review" per README; Code link resolved (none). Added a fixed-τ = 0.02 recount (18/18 accuracy, 15/18 W10) that **weakens my earlier τ-selection critique**. |
| 5 | `research/directions/candidates-claude.md` | D1: replaced "download size not checked" with VERIFIED sizes (161/163 MB) and the citation requirement. D2: corrected "K = 2 → 2 backprops/step" to "< 2% overhead (App. H)"; added the prox/export/baselines/schedule/MPS caveats. |
| 6 | `research/notes/pcd-fedpcd-literature.md` | Code row updated (audit findings; FedPCD code not released); §4 item 6 now notes the fixed-τ result; §6 marks the FedPCD link and PCD HEAD as resolved and adds the new open question (did the baselines get the prox step?). |

## 6. Still unverified after this round

- Whether the PCD baselines in the paper also used the proximal step (ask the authors).
- Whether the release reproduces the paper's numbers. No training was run; only `test_qp.py` was run, via the shim.
- The MPS float64 failure at `optim.py:87` (no torch installed; not executed).
- The OpenReview reviews and decision for HT01yGHLEt (bot challenge).
- The contents of the author's `LibMTL` / `TorchJD` forks.
- CIFAR-10-C and Waterbirds/CelebA sizes and licences.
- All M4 throughput numbers, from both agents.

---

## Round 2 (final), 2026-11-02

**Scope.** Responses to Codex's round-1 review of my work (`Battikhah/research-math-codex:docs/reviews/phase4-codex-reviews-claude.md`
@ `bb005c4`), and the attack on coordinator candidate S.

**New primary reads this round:**
- PCD v2 App. E.3 Table 10 (p. 44), App. E.4 Table 11, App. G Table 18 (p. 50–51);
- SCKD (CVF open-access PDF);
- DTO-KD (ICLR 2026 proceedings PDF);
- TPKD (arXiv 2609.39338);
- CHAD (arXiv 2607.19956);
- SCOPE (Europe PMC full text PMC13613376, CC BY; the MDPI site returned 403);
- DHO (arXiv 2505.07675, new);
- GAQAT arXiv metadata;
- PCD project page.

### R2-A. Item-by-item response to Codex's review of my work

| # | Codex point | My response | Evidence |
|---|---|---|---|
| A1a | "Normalized-WS control wholly untested" is wrong as worded | **ACCEPT.** | App. E.2 tested EMA-normalized baselines on w ∈ {0.1, 0.2, 0.4, 0.6, 0.8, 1} (Table 9 caption: "same as Sec. 5.1's w-grid"). App. E.4 Table 11 tuned *raw* scalarization at λ_eff ∈ {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}; e.g. λ = 3e-4 gives 72.71 ± 0.56% at 77.77% sparsity vs PCD τ = 0.01 at 73.03% / 77.26% (Δ +0.32, within noise), and λ = 1e-3 gives 70.49% / 88.51% vs PCD τ = 0.02 at 70.92% / 87.50%. The missing control is narrowly the **low-weight, EMA-normalized, rescale-matched** sum. All my files are reworded (corrections C1–C3). |
| A1b | Exact coefficient α = τ − aᵀb/‖b‖²; simplex weight w = α/(1+α); √(1+τ²) norm discrepancy | **ACCEPT; independently confirmed.** | The derivation follows from PCD Cor. 4.6. For unit a, b: ‖a+αb‖² = 1 + 2αc + α² = 1 + τ² − c² when α = τ − c, so √(1+τ²) at c = 0. **New [EE]:** PCD App. E.3 Table 10 (ResNet-34/CIFAR-100) shows CE efficiency cos(d*, g1) = 1.000 / 0.999 / 0.994 / 0.976 at τ = 0.01 / 0.05 / 0.10 / 0.20. The c = 0 prediction 1/√(1+τ²) gives 1.000 / 0.999 / 0.995 / 0.981. Direction deviation is 0.0168 / 0.0510 / 0.1067 / 0.2187 ≈ τ. The constraint is active on 100% of steps, conflict frequency is 0.41–0.47 and severity ~1e-5, and the authors write "There is little conflict to resolve" (p. 43). Codex's E.3 figures are VERIFIED. |
| A1c | "No conflict" too strong: negative signs frequent but tiny | **ACCEPT.** | Table 10, as above. |
| A1d | App. G sign flips 70.4 / 47.2 / 73.3 / 54.3% (SGD+m), Adam 0.2–7.7% | **ACCEPT, VERIFIED** (Table 18, p. 51). | Additional point from App. G: the SGD+momentum control "lands on the same frontier at matched sparsity" (67.81 ± 0.16 at 92.55% vs PCD-Adam 68.42 at 92.24%). The per-step guarantee does not determine the outcome [EE]. |
| A1e | 73.0 vs 27.3 is tied to the 80% target | **ACCEPT** | Table 2. |
| A2 | "No FedPCD code exists" is unsupported | **ACCEPT the wording correction.** | The facts were resolved in round 1 (`js/main.js` `code: ""`, "Coming soon"; README "Code is empty until the code is public"). All my files now say "**not publicly released as of 2026-11-02**" (C4). |
| A2b | "Validation split: unsupported if stated categorically" | **ACCEPT for FedPCD.** My notes already say "not described … may have been selected on test data (unverified)". For **PCD** it is categorical and VERIFIED. | PCD App. H p. 51: "No held-out validation split was used"; baselines at their highest mean test accuracy. |
| A3 | D5 "high scooping risk" unsupported; GAQAT is prior gradient-conflict QAT | **ACCEPT both.** | PCD project page (fetched this round) lists "Quantization-aware training" as an example application. That is overlap, not evidence of a competing project, so "high" is withdrawn. GAQAT registered as `jiang2024gaqat` (arXiv 2412.05551, v1 2024-12-07, Jiang, Meng, Tang, Yu, Li, Wang, Zhu; no journal_ref; preprint). D5 corrected (C5). |
| A4a | D1 novelty "incremental" | **AGREE** (my D1 already said "That is incremental"). | — |
| A4b | Remove "any outcome is publishable" | **ACCEPT.** | Removed from the D2 table row and the recommendation (C6). Replaced with "outcome largely predictable from App. E.3 Table 10". |
| A5 | D6 "essentially identical" is an interpretation | **ACCEPT.** | Kept as INTERPRETATION; exact margin, scale and optimizer differences from ε-SOFT-GEM are unverified (its full text is not read). |
| A6 | D4 "no PCD fairness method found" is a search result, not proof | **ACCEPT.** | Already labelled "search-limited". |

**Consequence for M1 (D2).** Table 10 already shows, in the authors' own data, that in compression PCD behaves as
the c ≈ 0 normalized sum predicts. M1's remaining question is therefore narrower and its outcome largely
predictable. I lower **M1 (b) 3 → 2 and (c) 3 → 2, so M1 totals 25/35** (was 27).

### R2-B. Attack on coordinator candidate S

#### (i) Is there existing evidence that the CE–KD gradient cosine is materially negative?

| Source | Measured CE–KD gradient statistic? | What it reports |
|---|---|---|
| **DHO** `kang2025dho` (arXiv 2505.07675; "Accepted at NeurIPS 2026" per arXiv comment) | **YES** | App. G.2 Table 26: single-head CE+KD, final-epoch cos(∇θ L_CE, ∇θ L_KD), 8-dataset mean. layer1 **−0.392**, layer2 −0.393, layer3 −0.405, layer4 **−0.453**, classifier head **−0.757**. Table 27, layer4: EuroSAT −0.80, DTD −0.67, FGVC-Aircraft −0.62. Fig. 3: the 10-dataset average cosine "turns negative" (graphical). **Setting:** 16-shot labels, CLIP RN50 zero-shot teacher, ResNet-18 student, not full-data CIFAR-100. **[TG] Prop. 1 / Cor. 1:** at a linear head the CE–KD inner product is ∝ ‖p̂−m‖² − r² (m = (y+p)/2, r = ‖y−p‖/2), so conflict is guaranteed whenever the student prediction lies in the ball with diameter [y, p]; for p̂ = (1−t)y + tp it equals −t(1−t)‖p−y‖². |
| **DTO-KD** `hayder2026dtokd` | Plot only | Fig. 1 (p. 2): "conflict score" ⟨g_dist, g_task⟩ (a **raw inner product**, not a cosine) over 500 iterations on the **COCO detection** task, for DTO-KD vs Roy Miles & Deng (2024). Values are graphical and not extractable from text; there is no CIFAR classification statistic. |
| **SCOPE** `chen2026scope` | Plot only | Eq. (10) detects conflict as ρ = cos(g_CE, g_KD) < δ = 0. Fig. 3 plots cos(g_CE, g_KD) against teacher entropy at epochs 1/50/100 on the Multiple Myeloma dataset for No-GCR / PCGrad / CAGrad / GCR. **No numeric summary** in the text; the figure image was not retrievable (Europe PMC / NCBI 403/404). |
| **SCKD** `zhu2021sckd` | No | Fig. 1 is a schematic. Table 6 sweeps the cosine threshold **ϕ ∈ {1.2, 1.1, 1.0, 0.9, 0.8}** with Top-1 75.2 / 75.1 / 75.4 / 75.0 / 74.9 (CIFAR-100). **A cosine threshold above 1 is impossible**, a second internal inconsistency to add to the one the survey draft already flags. A 0.5-point spread suggests the gating threshold barely matters [EE]. |
| **TPKD** `yuan2026tpkd` | Output-space only | Mechanism: once label learning makes the student "more discriminative" than the teacher, "moving its full prediction closer to the teacher then requires giving back some of that discrimination" (Intro, p. 1). Table 2 / §6: "unprojected learning loses at least one CE-step margin on every example" (2 states × 32 batches × 64, output-step protocol). No parameter-space CE–KD cosine. |
| **CHAD** `sumit2026chad` | Related (val-CE vs per-sample KD) | s(x) = cos(∇θ L_val, ∇θ L_KD(x)) on a 5,000-sample probe: **51.3%** of samples harmful or neutral (s ≤ 0). Useful mean **+0.162 ± 0.110**, harmful mean **−0.168 ± 0.112**. Teacher-confidence features correlate at \|r\| ≤ 0.09 (r² < 0.007) with usefulness (§5.2, Table 2). Bangla summarization, not vision. |

**Answer to (i).**
- Materially negative CE–KD cosines **have been measured**, but only in DHO's few-shot VLM-teacher setting:
  −0.39 to −0.45 in the backbone and −0.76 at the head [EE]. They are also **guaranteed at the head by geometry**
  whenever the student lies between label and teacher [TG, DHO Prop. 1].
- **No source found** reports CE–KD cosine statistics for full-data CIFAR-100 KD with a trained CNN teacher.
- **No source found** measures how teacher corruption changes the conflict.
- INTERPRETATION from DHO Cor. 1: head-level conflict magnitude scales with ‖p − y‖². A confidently *wrong* teacher
  has ‖p − y‖² ≈ 2, while a confidently correct one (p_y = 0.9) has ≈ 0.01. Corruption should therefore increase
  head conflict by orders of magnitude. This covers the head only and uses inner products, not cosines [UH].

#### (ii) Could the gate remove the conflict, so that PCD ≈ normalized WS again?

**Yes, largely by construction (VERIFIED from the proposal design).**
- The proposal's L2 admits only examples where the teacher is **correct and** max-prob > q (proposal pp. 2–3).
- S's corruption replaces outputs with **high-confidence wrong** predictions. Every corrupted output fails the
  correctness test, so **in the gated arms the 10/30% corruption dial is filtered out** and only reduces the
  number of KD examples.
- What remains is correct-teacher conflict, whose head magnitude scales with ‖p − y‖² (small for confident
  correct teachers, DHO Cor. 1). The survey draft's worked example (CE–KD logit inner product −0.0077 for a correct
  0.85-confidence teacher) is consistent with this: present but small.
- HYPOTHESIS [UH]: gated PCD and gated normalized WKD will be close.
- **Consequence:** the conflict dial only works in the **ungated** arms. S's "gated PCD vs gated normalized WS"
  contrast tests the gate, not the projection.

#### (iii) Do the missing common stationary point and Prop. 4.7 make the comparison ill-posed?

**Not ill-posed; informative, but the theory predicts against PCD in the corrupted regime.**

1. **No fixed point (PCD §4.8, p. 12).** With τ > 0 and no common stationary point, the PCD direction never
   vanishes. Training is still finite-horizon with a cosine schedule to zero, so final accuracy is well-defined
   [EE-type evaluation]. This is a test *outside* PCD's theory scope, which should be stated.
2. **Prop. 4.7 thresholds at DHO-like conflict.** The CE-ascent threshold is τ* = (1−c²)/(−c):
   - c = −0.45: τ* = 1.77 (no CE ascent for any τ ∈ [0,1]);
   - c = −0.76 (head-level): τ* = 0.556;
   - c = −0.80: τ* = 0.45.

   The proposal's grid τ ∈ {0, 0.1, 0.3, 0.5} reaches the boundary only if the whole-vector cosine approaches the
   head values [TG + UH].
3. **Key mechanism point [TG, Cor. 4.6]: S's control is mis-specified.**
   - In the active regime PCD = g̃1 + μ g̃2 with **μ = τ − c** (unit norms). At c = −0.45 and τ = 0.1, μ = 0.55:
     PCD is a normalized WS with λ = 0.55, not λ = τ = 0.1.
   - A constant-λ normalized WS reproduces PCD exactly whenever c is steady. **PCD differs from a tuned normalized
     WS only through the per-step adaptivity of λ_t = max(0, τ − c_t)**, i.e. when c *varies*, and through the
     inactive regime (λ_t = 0 when c_t ≥ τ).
   - The fair control is therefore a normalized WKD with λ **tuned on the same budget**, with a grid containing the
     pilot-measured τ − c̄. **"λ matched to τ" is a strawman** whenever c̄ is materially non-zero.
4. **Direction of the predicted effect [TG mechanism, outcome UH].** μ = τ − c *grows* as KD conflicts more with CE.
   With a corrupted teacher in the ungated arm, PCD therefore puts **more** weight on the harmful KD direction,
   to guarantee τ-progress on the KD loss. A constant-λ WKD keeps it fixed. So S's implicit hypothesis that PCD
   protects against unreliable teachers is opposite to PCD's mechanism. PCD protects the secondary, and the
   primary only through the anchor (Prop. 4.7). SCOPE-style projection, which removes the conflicting KD
   component, encodes the protection the proposal actually wants.

#### (iv) M4 budget

All counts are runs. Timings are [UH] until a smoke test is run.

| Stage | Design | Runs |
|---|---|---|
| Teacher | ResNet-32 on CIFAR-100 (the CIFAR-10 pilot needs its own teacher) | 2 |
| Pilot, CIFAR-10, 1 seed | CE-only 1; tuned WKD 3 λ × 3 corruption = 9; normalized WKD 4 λ (incl. τ − c̄) × 3 = 12; ungated PCD 3 τ × 3 = 9; gated PCD 3 τ × 3 = 9; gated normalized WKD 3 λ × 3 = 9 | 49 |
| Selection, CIFAR-100, 1 seed, validation split | Same as the pilot minus CE-only | 48 |
| Confirmatory, CIFAR-100, 5 seeds | CE-only 5 + 5 arms × 3 corruption × 5 seeds = 75 | 80 |
| Optional M1 anchor | 1 small ResNet, CE + group lasso, PCD vs normalized WS, ± prox, 3 seeds | 12 |
| **Total** | | **≈ 190** (≈ 180 without the anchor) |

Cost model:
- Every KD arm needs two full student backward passes plus a teacher forward. This is unlike compression, where
  PCD costs < 2% extra (PCD App. H).
- At an assumed 1,500 img/s effective throughput for ResNet-8 under KD and 60-epoch schedules (3 M images/run),
  that is ≈ 35 min/run, i.e. **≈ 110 h** total.
- At Codex's 300 img/s it is ≈ 2.8 h/run, i.e. **≈ 530 h**, which is infeasible on one M4.
- **The budget is realistic only if a 1-epoch smoke test shows ≳ 1,000 img/s** [UH]. Otherwise cut confirmatory
  seeds to 3 and corruption levels to {0, 30%}, giving ≈ 100 runs.
- Engineering prerequisites:
  - patch `pcd/optim.py:87` for MPS float64 [UH];
  - `pcd.PCD.backward` already takes per-loss gradients (`optim.py:59-72`).

#### (v) Scores for S, and whether S replaces M1

**S: 25/35**, equal to the revised M1 (25/35).
- **(a) 3.** PCD's QP applies and its KD behaviour is analytically predictable (μ = τ − c_t) [TG]. The KD secondary
  is non-convex and has no fixed point (§4.8) [TG negative]. Compatibility is as a *test bed*, not as a likely
  beneficiary.
- **(b) 4.** Conflict in KD is material where measured (DHO −0.39 … −0.76) [EE]. The corruption dial and per-step
  diagnostics make the PCD-vs-adaptive-λ difference directly observable. Whether accuracy differs is [UH].
- **(c) 3.** No paper compares PCD with a tuned normalized WKD or measures conflict versus teacher corruption
  (search 2026-11-02). Adjacent work exists: DTO-KD adaptive weights, SCOPE/TPKD projection, SCKD gating [EE].
- **(d) 4.** CIFAR-100/10 are standard (161/163 MB, citation required); ResNet-8/32 are the proposal's choices.
- **(e) 3.** The PCD library is reusable (MIT; 77/77 QP tests pass). The teacher must be trained; the corruption
  protocol is new; no reference KD-PCD code exists.
- **(f) 3.** About 180–190 runs; feasible only if the throughput smoke test passes [UH].
- **(g) 5.** It keeps the submitted MLR570 research question about unreliable teachers, and the KD survey
  (VERIFIED, proposal pp. 2–3).

**Recommendation (RECOMMENDATION): S should replace M1 as top-1, with two mandatory amendments.**
- With M1's (b)/(c) lowered by Table 10, S ties M1 on total and wins decisively on (g).
- S is also where the projection-vs-normalization question is not already answered by the authors' data.
- **Amendment 1:** replace "normalized WKD at λ matched to τ" with a **budget-matched tuned normalized WKD**,
  with a grid containing τ − c̄ measured in the pilot. Report the per-step λ_t = max(0, τ − c_t) that PCD
  implicitly uses.
- **Amendment 2:** apply the corruption dial to the **ungated** arms for the mechanism test. Treat the gated arms as
  a separate gate-effect question, since the correctness gate removes corrupted outputs by construction.
- The M1 anchor is optional. App. E.3 Table 10 already supplies the c ≈ 0 evidence, so a replication adds only
  the ± prox check.
- M1 becomes the backup if the instructor accepts a scope change or the KD budget smoke test fails.

**Single strongest argument against S.** PCD's own KKT structure (μ = τ − c) makes it **increase** the weight on the
KD direction exactly when a corrupted teacher conflicts with the labels, while the reliability gate removes the
corrupted outputs before PCD ever sees them. So in the ungated arm theory predicts PCD is *worse* than a fixed or
tuned weighted KD, and in the gated arm there is little conflict left for PCD to act on. S is therefore likely to
deliver a predictable negative or null answer to "does PCD add value?". That is scientifically legitimate, but it
means the course project's headline method is expected to lose to its own controls.

### R2-C. Corrections to my own files in this round

| # | File | Correction |
|---|---|---|
| C1 | `literature/notes/varam2026pcd.md` (critical reading #1) | Exact coefficient α = τ − aᵀb/‖b‖², w = α/(1+α), pre-rescale norm √(1+τ²−c²); added the App. E.3 Table 10 empirical confirmation; replaced "does not appear in the paper" with the narrowed missing control (low-weight, EMA-normalized, rescale-matched), citing App. E.2 and the App. E.4 small-λ raw rows. |
| C2 | `research/notes/pcd-fedpcd-literature.md` §4 item 3 | Same narrowing; "[UNTESTED] … appears never to have been compared" replaced. |
| C3 | `research/directions/candidates-claude.md` D2 | Same narrowing and exact form. |
| C4 | `literature/sources.yaml`, `literature/notes/irshaid2026fedpcd.md`, `research/notes/pcd-fedpcd-literature.md`, `research/directions/candidates-claude.md`, this review §1.2 / §4 | Every "no code exists / no released code / none released" now reads "**not publicly released as of 2026-11-02**". |
| C5 | `research/directions/candidates-claude.md` D5, synthesis §5 | "High scooping risk" withdrawn and replaced by "overlap risk" (project page lists QAT as an example application); GAQAT cited as `jiang2024gaqat`. |
| C6 | `research/directions/candidates-claude.md` | Removed "any outcome is publishable" (table row) and "any outcome is informative" (recommendation). |
| C7 | `literature/sources.yaml` | Registered `jiang2024gaqat` and `kang2025dho`; added `literature/notes/kang2025dho.md`. |
| C8 | This review §4 | M1 rescored (b) 3 → 2, (c) 3 → 2, total 27 → 25 (see R2-A). S added at 25; ranking S ≈ M1 > M2 (24) > M3 (22) > M4 = M7 (20) > M6 (18) > M5 (15). Top-1 is now S (amended), backup M1. |

### R2-D. Unresolved after two rounds (for `docs/reviews/`)

1. Whether S (KD) or M1 (compression audit) should be top-1. Both total 25/35 under my scores. The tie-break is
   criterion (g) and the instructor's stance on scope, which is outside the evidence. Codex's round-1 scores
   rated D2 above D1 on its own scale.
2. Whether a full-data CIFAR-100 CE–KD cosine is materially negative. No measurement exists in the sources I could
   find; DHO's few-shot VLM numbers may not transfer.
3. Throughput on the M4. Neither agent has measured it.
4. Whether the PCD baselines also used the proximal step (round 1).
