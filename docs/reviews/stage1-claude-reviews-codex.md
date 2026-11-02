# Stage 1 code review, cycle 1 of max 2: Claude reviews Codex track A

- **Date:** 2026-11-02.
- **Reviewer:** Claude (Orca task_e2d6d295c333).
- **Target:** branch `Battikhah/impl-data-codex` @ `b09f1e6` (commits `3fe697c`, `b09f1e6`; 28 files, +945/−3 vs `0c73aa8`).
- **Method:** `git archive b09f1e6 | tar -x -C /tmp/review-claude`. Nothing on the Codex branch was edited.
  - Tests and lint were run from that copy with `PYTHONPATH=/tmp/review-claude/src:/tmp/review-claude uv run --no-sync --project <claude worktree> …`.
  - The copy's `pcdkd` was confirmed to shadow the editable install: `pcdkd.data.__file__` = `/tmp/review-claude/src/pcdkd/data/__init__.py`.
  - The hygiene script and the literature-docs CI step were run in a throw-away `git init` copy (`/tmp/review-claude-git`).
- **Primary sources:**
  - He et al. 2016, arXiv 1512.03385v1, fetched from arxiv.org and read with `pdftotext -layout`.
  - torchvision 0.29.2 `datasets/cifar.py`, for the archive md5 values.
  - `research/experiment-plans/pcd-kd-conflict.md` §4, the corruption specification.
- **Verdict labels:** AGREE / DISAGREE / UNSUPPORTED. Claims are VERIFIED (re-run here) unless labelled.

## 0. Reproduction of the author's checks

| Command (in /tmp/review-claude) | Output | Exit |
|---|---|---|
| `uv run --no-sync --project $P pytest -q -p no:cacheprovider` | `23 passed in 4.24s` (includes `test_mps_forward`; MPS is available) | 0 |
| `uv run --no-sync --project $P ruff check . --no-cache` | `All checks passed!` | 0 |
| `python scripts/check_repo_hygiene.py` (temp git repo of b09f1e6) | `dataset/checkpoint path: src/pcdkd/data/cifar.py` / `possible secret: tests/unit/test_hygiene.py` | **1** |
| `python scripts/build_literature_docs.py && git diff --exit-code -- docs/literature` | no diff | 0 |

## 1. Findings

| # | Sev. | Location | Defect |
|---|---|---|---|
| F1 | **BLOCKER** (CI) | `scripts/check_repo_hygiene.py:8-15`, `.github/workflows/ci.yml:12` | The hygiene check fails on the repository itself, so the CI is red on every push. |
| F2 | **MAJOR** | `src/pcdkd/data/corruption.py:34-36` | Saturated teachers produce exact one-hot corrupted rows, so the corruption strength jumps discontinuously under T > 1. |
| F3 | **MAJOR** | `scripts/smoke_throughput.py:10-33` | One warm-up step and three timed steps give unstable throughput numbers, and these numbers drive a budget decision. |
| F4 | MINOR | `corruption.py:20-27` | Logits vs probabilities are auto-detected by heuristic, and logits can be misread as probabilities. |
| F5 | MINOR | `corruption.py:20,37` | Output is float64 NumPy and cannot be moved to MPS as-is. |
| F6 | MINOR | `tests/unit/test_hygiene.py`, `tests/unit/test_corruption.py:21-22` | Tests miss a negative hygiene case and the spec's confidence-equality check. |
| F7 | MINOR | `src/pcdkd/data/cifar.py:27-28` | The validation split comes from `train=True` data, which is augmented by default. |
| F8 | MINOR | `src/pcdkd/utils/config.py:8-21` | The config schema cannot express the plan's arms. |
| F9 | MINOR | `ci.yml` | Supply-chain and robustness gaps. |
| F10 | NIT | `corruption.py:7-16` | Corruption maps are not nested across rates. |
| F11 | NIT | `corruption.py:34-36` | A uniform teacher row stays uniform, and its argmax ties at the label. |
| F12 | NIT | `check_repo_hygiene.py:22` | `stat()` on a tracked but deleted file raises `FileNotFoundError`. |

### F1: BLOCKER for CI. The hygiene check fails on its own repository

**Coordinator issue (1): CONFIRMED.** Exit code 1 on the b09f1e6 tree (§0).

Both hits are false positives:

- **`PATTERNS` matches any basename starting with `cifar`, `checkpoint`, `ckpt` or `weight`.**
  - The second alternative, `(^|/)(?:cifar|checkpoint|ckpt|weights?)[^/]*(?:$|[._-])`, flags `src/pcdkd/data/cifar.py`. It also flags `foo/weight_decay.md` (tested: `True`).
  - At the same time it misses real artefacts: `outputs/ckpt/x` → `False`, `model_weights.npz` → `False`.
- **`SECRETS` matches the literal test key in `tests/unit/test_hygiene.py:5`.**
  - `SECRETS.search(...)` gives `match='AKIA1234567890123456'`.

**Minimal fix:**

- Restrict `PATTERNS` to artefact extensions and data directories. For example: `\.(?:pt|pth|ckpt|safetensors|npz|npy|h5|onnx|tar\.gz|zip)$|(^|/)(?:data|datasets?|checkpoints?|outputs)/`.
  - `data/` and `outputs/` are git-ignored, so any tracked file there is an error.
- Build the test credential at runtime, e.g. `"AKIA" + "1" * 16`.
- Add a test that `check(git ls-files)` returns `[]` on the repository (see F6).

### F2: MAJOR. Corrupted rows from saturated teachers are exact one-hot, so T-softening is discontinuous

**Coordinator issue (4).**

- `out[i] = (1 - confidence) / (k - 1)`. When the teacher's float32 softmax max-prob rounds to exactly 1.0, every off-class entry is exactly 0.
  - This happens once the logit gap exceeds a threshold between 17 and 20 at k = 10 (measured below).
  - Claude's `kd_loss` softens probabilities as `softmax(log p / T)`. log 0 = −inf, so the corrupted row stays one-hot at any T.
  - A row just below saturation is softened substantially.
- Measured with a k = 10 teacher (logit gap g on class 0), the row corrupted to class 3, and T = 4:

| gap g | f32 max == 1.0 | zeros in corrupted row | softened wrong-class prob |
|---|---|---|---|
| 10 | False | 0 | 0.5751 |
| 15 | False | 0 | 0.8239 |
| 17 | False | 0 | 0.8873 |
| 20 | **True** | **9** | **1.0000** |
| 25 | True | 9 | 1.0000 |

- **Consequences:**
  - The corruption is strongest for the teacher's most confident training examples.
  - It is a step change set by float32 rounding, not by the plan's definition (plan §4: "fixed confidence p_wrong = teacher's own max-prob").
  - Numerically it is safe: `kd_loss` stays finite and its gradient is finite on these rows (tested: `kd_loss 16.94`, `grad finite True`).
  - Whether well-trained ResNet-32 training-set logit gaps exceed about 17–20 is a HYPOTHESIS. It is plausible for an overfit teacher, and it should be measured in the pilot.
- **Minimal fix:**
  - Build corrupted rows in log space from float64 teacher logits: `log p_wrong = log conf`, `log p_other = log1p(-conf) - log(k-1)`, with `log1p(-conf)` taken from a log-sum-exp of the non-max logits.
  - Store teacher outputs as float64 logits or log-probs, not float32 probabilities.
  - Alternatively, apply the plan's definition at temperature T rather than at T = 1.
- **Cross-track note:** Claude's `kd_loss` can take `pretempered=True`. A follow-up could also accept log-probs.

### F3: MAJOR. The throughput benchmark is too short to be stable

**Coordinator issue (2): partly CONFIRMED, partly REFUTED.**

- `torch.mps.synchronize()` **is present**, before and after the timed loop (lines 17-18 and 31-32). "Missing synchronize" is REFUTED.
- The instability is **CONFIRMED**. Its causes are one warm-up iteration, run only on the one-backward path, and `steps=3` (384 images timed). The same `bench()`, called 6 times in one process (ResNet-8, MPS):

| Measurement | Runs (img/s) |
|---|---|
| author `bench`, one backward | 3782, 6929, 5487, 2867, 4885, 6162 (2.4× spread) |
| author `bench`, two backward | 2754, 2622, 1775, 2738, 2920, 2777 |
| 10 warm-up + 50 timed steps, one backward | 4369, 4406, 4243, 4688 |
| 10 warm-up + 50 timed steps, two backward | 2280, 2709, 2536, 2635 |

- With adequate warm-up, two backward passes are consistently slower, as expected. The "two-backward faster" observation is a noise artefact (INTERPRETATION, consistent with the table).
- The number matters: plan §10 picks the confirmatory budget from a 1,000 img/s threshold.
- **The benchmark also omits real costs:**
  - gradient flattening, Gram matrix and combiner;
  - the optimizer step;
  - the dataloader and augmentation;
  - the 100-class head (`y` ∈ [0, 10)).

  So it is an upper bound.
- **Minimal fix:**
  - Use at least 10 warm-up steps per mode.
  - Use at least 50 timed steps, repeated 3 or more times, and report the median and range.
  - Include `flat_grad`, `PCD.combine`, `unflatten_to` and `opt.step()` in the two-backward mode.
  - Add a `num_classes=100` case.

### F4: MINOR. Ambiguous logits/probabilities auto-detection

- `corrupted_teacher_distribution([[1., 0., 0.]], [1], {})` returns `[[1, 0, 0]]`. Read as logits it should be `softmax = [0.576, 0.212, 0.212]`.
- Any non-negative logit row that sums to 1 is misread. This is rare for trained nets but silent.
- **Fix:** take an explicit `kind="logits"|"probs"` argument, or two functions.

### F5: MINOR. float64 output is not usable on MPS as-is

- `torch.from_numpy(out).to("mps")` raises `TypeError: Cannot convert a MPS Tensor to float64 dtype…` (tested).
- The trainer must cast with `.float()` on CPU before moving to the device. Claude's `kd_loss` also casts the teacher to the student's dtype.
- **Fix:** return float32, or document the cast.
- This interacts with F2: keep logits in float64 on CPU, and cast only the final rows.

### F6: MINOR. Test gaps (no test was weakened)

- **Hygiene test:** `test_hygiene.py` checks only positives. That is why F1 reached the branch. Add a negative test on `git ls-files`.
- **Corruption confidence:** `test_corruption.py:21-22` checks only that the wrong class is the argmax. It never checks the spec: `out[i, wrong] == probs[i].max()`, plus the uniform off-class mass. Add both.
- No existing test was modified or weakened; all test files are new.
- Expected values are independently derived:
  - analytic parameter counts, consistent with He et al. Table 6 (VERIFIED: 0.27M for ResNet-20, 0.46M for ResNet-32 at he2016.txt lines 454-455 vs 269,722 and 464,154);
  - exact per-class split counts;
  - the md5 against `hashlib`.

### F7: MINOR. Validation images are augmented by default

- `cifar_dataset(train=True)` defaults to `augment=train`. The plan's 5,000-image validation split is carved from the training set.
- A trainer that subsets one instance would therefore evaluate validation on augmented images.
- **Fix:** document, or provide `cifar_splits(root, val_size, split_seed)` that returns an augmented train subset and an un-augmented val subset. Both are built over the same base data, so the returned `index` stays the original CIFAR index.
- **Related integration point (INTERPRETATION):** teacher probabilities will be precomputed on un-augmented images while the student sees augmented crops. That is standard offline KD, but it should be stated in the plan.

### F8: MINOR. The config cannot express the experiment arms

- `ExperimentConfig` lacks fields the plan needs: method/combiner, λ, β, ε, gate mode, q, momentum, weight decay, LR schedule, val size and T-selection.
- Types are not validated: `temperature: "4"` raises `TypeError` rather than `ValueError`.
- **Fix:** extend the schema when the trainer lands. It should mirror the `pcdkd.optim.COMBINERS` keys (`wkd`, `nwkd`, `pcgrad`, `pcd`, `adaptive_lambda`) and `pcdkd.losses.GATE_MODES`.

### F9: MINOR. CI workflow

**Coordinator issue (5).**

| Item | Finding |
|---|---|
| `astral-sh/setup-uv@v5`, `actions/checkout@v4` | Pinned to mutable major tags, not commit SHAs. This is a supply-chain risk. **Fix:** pin SHAs. |
| uv version and Python version | Not pinned. `uv sync --frozen` honours `uv.lock` and `requires-python`, so it is acceptable. Pinning `version:` in setup-uv improves reproducibility. |
| `uv sync --frozen` | Correct: it fails if the lock is stale. |
| `pytest -m "not slow and not requires_data"` | Correct. Markers are registered and `--strict-markers` is on. |
| hygiene step | Fails (F1). |
| `build_literature_docs.py && git diff --exit-code -- docs/literature` | Passes today (exit 0). It is blind to newly generated, untracked files: I added `docs/literature/new.md` and `git diff` still exited 0. **Fix:** `test -z "$(git status --porcelain -- docs/literature)"`. |
| `permissions:` | Absent. **Fix:** add `permissions: contents: read`. |
| `on: [push, pull_request]` | Duplicate runs on PR branches (NIT). |

### F10: NIT. Maps at different rates are independent

- The 10% map is not a subset of the 30% map: 25 of the 100 overlap (tested, 1,000 labels, seed 0).
- The plan does not require nesting. Nested maps (one permutation, take a prefix) would make 10% vs 30% comparisons paired within a seed (RECOMMENDATION).

### F11: NIT. Uniform teacher rows

- conf = 1/k gives a uniform corrupted row whose argmax (0) can equal the label.
- `correct_and_conf` could then accept it, but only when q < 1/k. Harmless at the plan's q.
- **Coordinator issue (4), sub-case:** "teacher max-prob < 1/(C−1)" cannot occur below 1/C, because max-prob ≥ 1/C always.
  - The wrong class is the strict argmax whenever conf > 1/C, since (1 − conf)/(C − 1) < conf ⟺ conf > 1/C.
  - Rows always sum to 1 (tested, max error 0).
  - The wrong class never equals the label: `make_corruption_map` excludes it, and `corrupted_teacher_distribution` raises if it does.

### F12: NIT

- `check_repo_hygiene.py:22` calls `path.stat()` on every `git ls-files` entry. A tracked file deleted in the working tree raises `FileNotFoundError`.
- **Fix:** `if not path.exists(): continue`.

## 2. Coordinator issue (3): option-A shortcut. AGREE, no defect

- **He et al. §3.3** (he2016.txt lines 214-219): "(A) … zero entries padded for increasing dimensions … when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2."
- **He et al. §4.2** (line 503): "On this dataset we use identity shortcuts in all cases (i.e., option A)."
- **Implementation:** `x[:, :, ::2, ::2]` is exactly a stride-2 identity, and `F.pad` zero-fills the new channels (8 + 8 for 16 → 32). Verified with `torch.equal` against a manual concatenation.
- **Placement:** the split of padded channels is unspecified in the paper. Any placement is equivalent up to a channel permutation of the following layers at random initialisation (INTERPRETATION).
- **Parameter counts:** 269,722 and 464,154 for ResNet-20 and ResNet-32 match Table 6 (0.27M and 0.46M).
- **Option B:** stays selectable and is parameter-count-tested.

## 3. Determinism, leakage, device, security

- **Determinism (VERIFIED):**
  - `make_corruption_map` and `stratified_split` use a local `default_rng(seed)`, and the tests assert repeatability.
  - `seed_everything` seeds Python, NumPy (global), torch, CUDA and MPS.
  - `worker_init_fn` derives distinct per-worker seeds from `torch.initial_seed()`.
  - MPS kernel non-determinism is not controllable; the plan already records this.
- **Leakage (VERIFIED for track A):**
  - Corruption keys are restricted to `train_indices`: there is a range check, and the integration test asserts no overlap with val.
  - Labels are used only for training indices.
  - No code reads test labels.
  - Remaining trainer obligations:
    - never apply the map or the gate to val or test outputs;
    - run T and λ selection on val only.
- **Device:** models run on MPS (`test_mps_forward` passed). The only float64 hazard is F5.
- **Security:**
  - `ruff` S rules pass.
  - `subprocess` calls only `git ls-files` with a fixed argv (`noqa S603/S607` justified).
  - `fetch_cifar.py`:
    - downloads only from an HTTPS allowlist;
    - requires `--yes`;
    - refuses to overwrite;
    - writes through a temp file plus `os.replace`;
    - verifies md5. The values `c58f3010…` and `eb9058c3…` equal torchvision's `tgz_md5` (cifar.py lines 34 and 155). md5 is the publisher's integrity hash, not authentication: acceptable, and stated with `usedforsecurity=False`.
  - `save_corruption_map` silently overwrites its path (NIT, run directories are fresh).
  - `pyproject.toml` gains `pythonpath = ["."]`. It is needed so tests can import `scripts.*`. It does not conflict with track B, which did not touch `pyproject.toml`.

## 4. Track A ↔ track B integration contract

| Contract | Status |
|---|---|
| Data returns `(x, y, index)` with the original CIFAR index | VERIFIED (`IndexedDataset`), if the trainer wraps the full dataset and then subsets (F7) |
| Corruption produces teacher probability rows at T = 1 | VERIFIED. It matches Claude's `kd_loss(teacher_probs at T=1)` and `reliability_mask(teacher_probs, labels, q, mode)`. Cast to float32 first (F5); see F2 for saturated rows |
| The gate removes corrupted rows | VERIFIED by construction: the corrupted argmax ≠ label whenever conf > 1/C (F11) |
| Teacher rows with exact zeros in `kd_loss` | Safe: finite loss and gradient (tested). Semantics affected (F2) |

## 5. Author assumptions

The Codex commits (`3fe697c`, `b09f1e6`) record no explicit assumptions in their messages, and the worker report was not available to me. I classify the assumptions implicit in code and docstrings:

| Assumption | Verdict | Evidence |
|---|---|---|
| "option-A shortcuts follow He et al. (2016), §4.2" (resnet_cifar.py:1) | AGREE | §2 |
| Corrupted row: wrong class at the teacher's own max-prob, rest uniform (plan §4) | AGREE in exact arithmetic. DISAGREE that it holds for saturated float32 inputs | F2 |
| Logits vs probabilities can be auto-detected | DISAGREE | F4 counter-example |
| Wrong class drawn uniformly from non-true classes, training indices only | AGREE | code lines 13-15; integration test |
| The hygiene regexes flag only artefacts and secrets | DISAGREE | F1 |
| A 3-step synthetic benchmark measures throughput | DISAGREE | F3 table |
| CI literature-docs check detects stale generated docs | DISAGREE (partial) | F9: untracked files are missed |
| CIFAR archive md5 values | AGREE | match torchvision 0.29.2 |

## 6. Required before merge (RECOMMENDATION)

1. Fix F1 and add a negative hygiene test, or CI is permanently red.
2. Fix F2 (log-space or float64 corruption from logits) before any teacher outputs are cached.
3. Rework F3 before the throughput number is used for the go/no-go budget decision.
4. Fix F4–F9 in the same correction cycle. F10–F12 are optional.
