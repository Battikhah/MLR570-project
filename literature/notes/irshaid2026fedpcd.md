# irshaid2026fedpcd — FedPCD: Priority-Constrained Federated Aggregation

Read 2026-11-02 by the Claude literature worker. Source: arXiv 2610.13508v1 PDF (30 pp.), saved locally as
`literature/pdfs/irshaid2026fedpcd.pdf`.

## Bibliographic status (VERIFIED)

- Authors: Yousef Irshaid, Dara Varam (equal contribution), Mohamed I. AlHajri; AUS CSE (+ MIT for Varam).
- arXiv v1 2026-10-09, cs.LG. **Preprint**: no journal_ref. The project-page repo README says "preprint, under review"
  (venue not named). (Corrected in the cross-review, 2026-11-02.)
- Project page daravaram.github.io/FedPCD links the arXiv paper. Its "Code" button is wired in `js/main.js` as
  `code: ""` with the comment `// code release, e.g. "https://github.com/DaraVaram/fedpcd"`, and shows "Coming soon".
  The repo README says "Code is empty until the code is public". **No FedPCD code is released (VERIFIED 2026-11-02).**
- The survey draft mentions "FedPCD" at L195 without a citation; this record supplies it.
- Name collisions: unrelated GitHub repos `LiFan0304/FedPCD` and `StandWolf/FedPCD` ("Prototype-Cross Distillation
  for Multimodal Federated Learning") are not this work.

## Method (VERIFIED, Sec. 3.2–3.3, Alg. 1)

- Clients run local SGD and send Δ_i. The server forms pseudo-gradients g_i = Δ_i/η_ℓ.
- Primary: the FedAvg-weighted aggregate ḡ_t. Secondaries: every participating client's g_i, plus cached
  directions of clients seen in the last H rounds ("client history").
- QP: d̃* = argmin ½‖d̃ − ḡ̃‖² s.t. g̃_iᵀ d̃ ≥ τ‖g̃_i‖² over the participating and cached clients.
- Normalization is **instantaneous ℓ2**, not EMA, because EMA statistics go stale under partial participation (App. D).
- Rescale to ‖ḡ‖; server step θ ← θ − η_g d.
- If FedAvg already satisfies every constraint, the update is unchanged FedAvg.
- K is now the number of clients (tens of constraints). The QP is "projection onto half-spaces"; the solver
  implementation is not specified in the main text (INTERPRETATION: the 2^{K−1} enumeration of PCD is infeasible
  at this K, so a generic QP or active-set solver must be used; check against code once released).

## What it adds over PCD for federated settings

- Priority between objectives becomes priority between **the global aggregate and individual clients**. Fairness
  emerges from client-level progress constraints rather than from an explicit fairness surrogate.
- Handling of partial participation through history caching. Expected coverage is N(1−(1−ρ)^{H+1}) distinct clients.
- **Theory (THEORETICAL GUARANTEE, Thm 4.1 / Cor 4.2):** O(1/T) average squared-gradient-norm rate, and
  O(1/T) for the p-weighted client gradient norms (hence CMS accumulation points). Assumptions: full participation,
  deterministic full-batch local GD, H = 0, L-smooth F_i, F bounded below, **a feasible QP, and a uniform cosine
  alignment margin γ > 0** between every client gradient and d̃* at every round, with γ_eff = γ(1−r) − r > 0
  after local drift. Convexity is not required.
  → INTERPRETATION: the alignment-margin assumption is strong and is a condition on the algorithm's own iterates,
  not a checkable property of the problem. The authors themselves note heterogeneous clients "need not admit a
  common stationary point". The theorem covers no stochastic gradients, no partial participation, and no history.
- Client drift (heterogeneity) enters the theory only through r (local-step deviation). The algorithm has no drift
  correction mechanism such as SCAFFOLD control variates.

## Empirical evidence (EXISTING EMPIRICAL EVIDENCE, as reported)

- Datasets/models: CIFAR-10 (3-layer CNN, GroupNorm), CIFAR-100 (ResNet-18, GroupNorm), Fashion-MNIST (MLP).
  N = 100 clients, Dirichlet α ∈ {0.01, 0.1, 1} (Acar et al. protocol), participation 5%/10%, **1 local epoch**,
  500/500/300 rounds, plain SGD without momentum or weight decay, 5 paired seeds.
- Baselines (10): FedAvg, AFL, q-FFL, Ditto, FedMGDA+, FedFV, FedLF, FedHEAL, FedPW, CRAFT.
  **Not compared: FedProx, SCAFFOLD, FedNova, FedDyn** (drift/normalization methods).
  SCAFFOLD is cited but not run.
- Reported results: highest global accuracy in all 18 configurations and best worst-10% accuracy in 17/18.
  The CIFAR-100 gains are large in relative terms, but absolute accuracy is low: Table 1, α = 0.01, 5%
  participation: FedAvg 9.08, CRAFT 15.46, FedPCD (τ = 0.02) 18.33. CIFAR-10 α = 0.01, 5%: FedAvg 48.09,
  FedLF 65.59, FedPCD 71.90 (τ = 0.01).
  Rounds needed to reach FedAvg's final accuracy: 10%–44% of FedAvg's.
- On CIFAR-100, FedPCD's client-accuracy spread σ_c is higher than the baselines'. The authors attribute this to
  higher accuracy.

## Critical reading (INTERPRETATION / possible weaknesses to test)

0. **Cross-review correction (2026-11-02): τ cherry-picking does not explain the headline.** Computed from Table 1
   at a single fixed τ = 0.02: FedPCD has the highest global accuracy in **18/18** configurations and the best
   worst-10% accuracy in **15/18** (the paper reports 17/18 with τ chosen per configuration). The losses are
   Fashion-MNIST 10% at α = 0.01 and α = 0.1 (FedLF), plus CIFAR-10 at 10%, α = 0.01 (FedLF 56.93 vs 56.71). The grid-size concern below
   still applies to the margins, not to the existence of a gain.
1. **Search-effort asymmetry.** The FedPCD grid is τ ∈ {6 values} × H ∈ {0..10}; most baselines get 3 values
   (App. A, Table 2). The selection rule is "best performing hyperparameters per dataset after averaging the five
   seeds", and all metrics are measured on client test partitions at the final round. **No separate validation
   split is described**, so hyperparameters may have been selected on test data (unverified; ask the authors or
   check code). This is the same "comparable search effort" issue the course proposal raises.
2. **Missing drift and normalization controls.** FedPCD unit-normalizes each client direction before the QP.
   Normalized-update aggregation (FedNova-like) and drift correction (SCAFFOLD/FedProx) are the obvious cheaper
   alternatives, and none is compared. → HYPOTHESIS: part of the gain comes from normalization plus history
   (FedFV/FedLF also use H = 10 and are the strongest baselines).
3. Many local epochs (high drift) are untested. The authors state this limitation (Sec. 6).
4. Not compatible with secure aggregation: the server must see individual client pseudo-gradients (authors' limitation).
5. Low absolute CIFAR-100 accuracy (~9–36%) means conclusions may not transfer to well-tuned FL regimes.
6. There is no code, so independent replication is required before any claim is reused.

## Datasets / benchmarks used

CIFAR-10, CIFAR-100, Fashion-MNIST, all with Dirichlet-partitioned synthetic federation.
