# Independent Code Review — Claude Track B, Cycle 1

**Target:** `Battikhah/impl-optim-claude` at `503f3f68793965afc8ac6d09872615aab52cdcce`
**Review scope:** the target commit and its parent-relative changes; read-only archive under `/tmp/review-claude-503f3f6`.
**Verdict:** four findings (one MAJOR, three MINOR). The target test suite and Ruff pass; there is no trainer in this snapshot, so cross-track data flow and label-leakage claims remain unverified.

## Findings

### 1. MAJOR — `prop47_margin` is not the actual primary-progress margin under EMA normalization

**Evidence:** `src/pcdkd/optim/combiners.py:237-247` computes `c` as the raw-gradient cosine, then records `1 + tau*c - c*c`. Positive EMA scales preserve the angle, but they do not make either normalized vector unit length. PCD Proposition 4.7 explicitly assumes `||g̃1|| = ||g̃2|| = 1` and derives that formula only under those conditions ([PCD v2, §4.5, Prop. 4.7](https://arxiv.org/html/2606.29521v2#S4.SS5)). The target's own literature note states the same condition at `literature/notes/varam2026pcd.md:56`.

**Counterexample:** let `a=g̃1=(0.1,0)`, `b=g̃2=(-0.8,0.6)`, and `tau=0.1`. Then `cos(a,b)=-0.8`, the active multiplier is `0.18`, and `d̃=a+0.18b`; the logged unit-norm formula is `+0.28`, while the actual normalized primary margin `a·d̃/||a||²` is `-0.44`. Thus the sign used in `prop47_nonpositive_fraction` can be reversed for EMA-normalized steps. These are valid non-unit EMA-scaled vectors; the difference does not depend on final magnitude rescaling, which preserves the sign.

**Minimal fix:** compute and report the actual `g̃1·d̃/||g̃1||²` after solving the QP (or its generalized K=2 formula `1 + τ(a·b)/||a||² - (a·b)²/(||a||²||b||²)`). Keep `1+τc−c²` under an explicitly idealized/unit-norm key if that comparison is useful, and make the actual margin `None` at the primary-stationary halt. The experiment plan currently asks for this value as a per-step margin (`research/experiment-plans/pcd-kd-conflict.md:60-63`), so the current logged fraction should not be interpreted as measured CE-ascent frequency.

### 2. MINOR — `cos_norm` is an inner product, although the experiment plan asks for normalized cosine

**Evidence:** `src/pcdkd/optim/combiners.py:104-105` stores `scale1*scale2*(g1·g2)`, exactly `g̃1·g̃2`. It can lie outside `[-1,1]`; the histogram's `below`/`above` counters acknowledge that. The module docstring correctly explains the distinction, but the key name remains easy for analysis code to treat as an angle. The experiment plan asks for raw and normalized cosine (`research/experiment-plans/pcd-kd-conflict.md:60-61`). Since both scales are positive, `cos_raw` already equals the cosine between the normalized vectors.

**Minimal fix:** expose the dot product as `dot_norm` (or retain `cos_norm` only as a compatibility alias) and report a true `cos_tilde` if the plan needs a separate normalized-angle column. Keep the current explanatory docstring.

### 3. MINOR — `pretempered=True` bypasses the positive-temperature validation

**Evidence:** `src/pcdkd/losses/kd.py:19-27` validates `T`, but `kd_loss` calls `soften_probs` only when `pretempered` is false (`:61-64`). In an independent run, `kd_loss(logits, probs, 0.0, pretempered=True)` returned `nan`, and `T=-1.0` returned a finite but meaningless loss instead of raising. The public contract says `T > 0` (`:45`).

**Minimal fix:** validate `T` as finite and positive at the top of `kd_loss`, before the `pretempered` branch; retain the validation in `soften_probs` for direct callers.

### 4. MINOR — `AdaptiveLambda` reports a positive active coefficient when PCD halts on zero primary gradient

**Evidence:** `PCD._direction` returns zero with `active=False` and `lambda_eff=0` when `norms[0] == 0` (`src/pcdkd/optim/combiners.py:247-251`). `AdaptiveLambda._direction` computes its lambda and `active` fields first, then returns a zero direction without resetting them (`:281-294`). Independent input `g1=[0,0]`, `g2=[1,0]`, `tau=0.2` produced equal zero directions but PCD `(active=False, lambda_eff=0)` versus adaptive lambda `(active=True, lambda_eff=0.2)`. This violates the stated K=2 identity/diagnostic behavior on the paper's explicit primary-stationary halt branch ([PCD v2, Algorithm 1, steps 7–8](https://arxiv.org/html/2606.29521v2#S4.S6)).

**Minimal fix:** on the zero-primary branch, set `active=False`, `lambda_eff=0`, and `lambda_raw=0` before returning; add a boundary assertion to the identity test.

## Assumption and concern verdicts

- **KD softening and exact zeros — AGREE, with a finite-precision limit.** For valid probability rows, `softmax(log(p)/T)` is algebraically equivalent to `softmax(z/T)` when `p=softmax(z)`. The existing one-hot test passes with finite loss and student gradient. Exact zeros remain zero, which is appropriate for the probability-only input. But information lost to float32 underflow cannot be recovered: with `z=[-200,0]`, float32 `softmax(z)` is `[0,1]`, so softening those probabilities at `T=20` stays `[0,1]`, while direct `softmax(z/20)` is approximately `[4.54e-5,0.999955]`. This is a representation limitation, not a defect in the probability-only contract. The `pretempered` temperature-validation defect is Finding 3.
- **`cos_norm` interpretation — DISAGREE if it is analyzed as a cosine.** It is the normalized-vector dot product; the source docstring says so, but the field name and plan terminology invite misuse. See Finding 2.
- **Proposition 4.7 diagnostic — DISAGREE as a measure of actual EMA-step primary progress.** It is only the unit-norm idealization, and the counterexample in Finding 1 reverses its sign. Its current documentation is candid about the idealization, but no separate actual margin is logged.
- **PCGrad variant — AGREE with the experiment-plan definition; qualify any comparison to Yu et al.** The code implements one-sided “protect CE” projection (`src/pcdkd/optim/combiners.py:181-200`), exactly the baseline specified in `research/experiment-plans/pcd-kd-conflict.md:33`. It is not the symmetric, task-wise projection method described by Yu et al. ([*Gradient Surgery for Multi-Task Learning*, arXiv:2001.06782](https://arxiv.org/abs/2001.06782)). The class and module docs say “one-sided”; preserve that label in result tables/plots (the serialized `method` value is currently just `pcgrad`). This is a plan-matched baseline, not canonical PCGrad.
- **NWKD rescale / PCD K=2 identity — AGREE.** Both paths call the same `_rescale` helper; `test_nwkd_rescale_and_formula` checks its norm, while `test_pcd_equals_nwkd_with_alpha_in_active_case` clones the pre-step EMA state and checks the direction and resulting normalizer state for active cases.
- **Closed form, KKT, and infeasible fallback — AGREE.** The K=2 formula implements `alpha=max(0, tau-(g̃1·g̃2)/||g̃2||²)`; the independent KKT enumeration uses Gram constraints and returns the primary-only weights when no KKT-valid set exists. The zero-secondary and primary-stationary branches are covered. I independently recomputed four fixture cases from the QP/KKT equations: active K=2 weights `[1, 0.87655928622]` (maximum direction error `1.11e-16`), inactive K=2 `[1,0]` (error `0`), active feasible K=3 `[1,0,0.364703891797]` (error `0`), and infeasible K=4 primary-only `[1,0,0,0]` (error `0`); active-set and feasibility flags matched in all four.
- **Official fixture provenance — AGREE, with the standard fixture limitation.** `tests/fixtures/README.md:8-21, 23-57` identifies `pcd/qp.py`, official commit `e9afdbc9f4cb09343934eadb31dc72ea5ebac9b0`, generator procedure, seed, and license. The stated clone exists at `/tmp/pcd-review/pcd-code` and resolves to that exact commit; `tests/unit/test_pcd.py:122-140` checks metadata and compares all 51 stored cases to the implementation. CI compares against saved outputs rather than executing the official solver, but the independent equation checks above support the fixture values.
- **EMA bias correction, per-objective counters, and state round-trip — AGREE.** `EMANormalizer` uses each name's own `t`, applies `v/(1-beta**t)`, and serializes both `v` and `t`; `tests/unit/test_normalize.py` checks the first-step/manual recurrence, independent names, and a restored subsequent update.
- **Track A / Track B trainer integration and label leakage — UNSUPPORTED at this target commit.** This snapshot has no trainer or CIFAR/corruption pipeline to inspect. The gate and KD APIs both accept `(B,C)` `teacher_probs` tensors and the synthetic integration test passes softmax probability rows, so their local interfaces agree; that does not establish the planned indexed-data conversion, training-only corruption, or exclusion of validation/test labels in a real trainer.
- **MPS and security — UNSUPPORTED for runtime MPS; no code-level security finding.** `gram_matrix` performs dot products on the source device and transfers to CPU before float64 (`src/pcdkd/optim/pcd.py:92-103`), avoiding float64 MPS operations by inspection. No MPS device was available to execute KD/combiner kernels. This commit adds no network or subprocess behavior in runtime code; the fixture README's generator uses local `git rev-parse` only.

## Test integrity and independent checks

The target commit adds unit and synthetic integration tests; it does not remove or weaken existing assertions. The active-set, KKT, finite-gradient, zero-primary, rescale, state, and deterministic tiny-training assertions passed. The float32 identity tolerance is explicitly documented in `tests/unit/test_combiners.py:98-109`; fixture comparison uses `1e-8` against float64 stored values.

Commands run against the archived target commit (using the worktree's locked virtual environment):

```text
$ PYTHONPATH=src:. /Users/khalidhamdan/orca/workspaces/AML-Code-and-Paper/impl-data-codex/.venv/bin/pytest -q
........................................................................ [ 47%]
........................................................................ [ 94%]
.........                                                                [100%]
153 passed in 3.55s
exit code: 0

$ /Users/khalidhamdan/orca/workspaces/AML-Code-and-Paper/impl-data-codex/.venv/bin/ruff check .
All checks passed!
exit code: 0
```

Independent fixture recomputation output:

```text
random K=2: weights [1, 0.87655928622], max direction error 1.1102230246251565e-16; active/feasible matched
inactive K=2: weights [1, 0], max direction error 0; active/feasible matched
random K=3: weights [1, 0, 0.364703891797], max direction error 0; active/feasible matched
infeasible K=4: weights [1, 0, 0, 0], max direction error 0; active/feasible matched
```

**Review total:** 4 findings; worst severity MAJOR (Prop. 4.7 margin can have the opposite sign from actual EMA-normalized primary progress).
