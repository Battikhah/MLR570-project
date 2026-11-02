# Test fixtures

## `pcd_reference_qp.json`

Reference outputs of the **official PCD QP solver** used by `tests/unit/test_pcd.py` to cross-check
`pcdkd.optim.pcd.solve_pcd_qp` (tolerance 1e-8, float64 on CPU).

- **Source:** `pcd/qp.py`, function `solve_qp(gram, tau)` (default `tol=1e-9`), from
  <https://github.com/DaraVaram/priority-constrained-descent> at commit
  `e9afdbc` (full hash in the JSON `provenance.commit`).
- **License / attribution:** MIT License, Copyright (c) 2026 Dara Varam and Mohamed I. AlHajri.
  Only *outputs* of the solver are stored here; no official source code is vendored.
- **How it was produced (2026-11-02):** read-only clone at `/tmp/pcd-review/pcd-code`; `pcd/qp.py` was
  loaded **by file path** with `importlib` (it imports NumPy only), so the `pcd` package and its torch
  front end were neither installed nor imported. Run with `uv run python -I <script> <clone> <out.json>`
  (NumPy 2.4.6, float64).
- **Contents:** 51 seeded problems (`rng_seed` 20261102, dimension 6): 30 random (K = 2, 3, 4, biased
  toward conflict; includes single-, partial- and fully-active sets), 9 inactive, 6 infeasible
  (K = 3, 4 with two exactly anti-parallel secondaries, τ > 0), 6 with one zero secondary.
  Each record stores the input `vectors` (row 0 = primary), `tau`, and the official `weights`, `active`,
  `feasible`, and `direction = weights @ vectors`.

Generator script (kept here so it is reproducible without adding a file outside this task's ownership):

```python
"""Generate tests/fixtures/pcd_reference_qp.json from the official pcd/qp.py (numpy-only shim).

Usage: python gen_pcd_reference_qp.py <path/to/pcd-code> <out.json>
"""
import importlib.util
import json
import subprocess
import sys

import numpy as np

repo, out = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("official_qp", f"{repo}/pcd/qp.py")
qp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qp)  # imports numpy only; the pcd package (torch front end) is not imported
commit = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True,
                        check=True).stdout.strip()

rng = np.random.default_rng(20261102)
DIM = 6
TAUS = [0.0, 0.05, 0.1, 0.3, 0.5, 1.0]
problems = []


def add(kind, V, tau):
    G = V @ V.T
    res = qp.solve_qp(G, tau)
    problems.append({
        "kind": kind, "K": int(V.shape[0]), "tau": float(tau), "vectors": V.tolist(),
        "weights": res.weights.tolist(), "active": list(res.active), "feasible": bool(res.feasible),
        "direction": (res.weights @ V).tolist(),
    })


for K in (2, 3, 4):                                   # generic random problems: 10 per K
    for _ in range(10):
        V = rng.standard_normal((K, DIM)) * rng.uniform(0.1, 3.0, size=(K, 1))
        V[1:] -= rng.uniform(0.0, 1.0) * V[0]          # bias toward conflict so constraints bind
        add("random", V, float(rng.choice(TAUS)))
for K in (2, 3, 4):                                   # inactive: secondaries aligned with primary
    for _ in range(3):
        g1 = rng.standard_normal(DIM)
        V = np.stack([g1] + [g1 + 0.1 * rng.standard_normal(DIM) for _ in range(K - 1)])
        add("inactive", V, float(rng.choice([0.0, 0.05, 0.1])))
for K in (3, 4):                                      # infeasible: two exactly anti-parallel secondaries
    for _ in range(3):
        V = rng.standard_normal((K, DIM))
        V[2] = -2.0 * V[1]
        add("infeasible", V, float(rng.choice([0.05, 0.1, 0.3])))
for K in (2, 3, 4):                                   # one zero secondary
    for _ in range(2):
        V = rng.standard_normal((K, DIM))
        V[1:] -= 0.8 * V[0]
        V[1] = 0.0
        add("zero_secondary", V, float(rng.choice(TAUS[1:])))

doc = {
    "provenance": {
        "source": "https://github.com/DaraVaram/priority-constrained-descent",
        "commit": commit, "file": "pcd/qp.py", "function": "solve_qp(gram, tau) with default tol=1e-9",
        "license": "MIT, Copyright (c) 2026 Dara Varam and Mohamed I. AlHajri",
        "generator": "tests/fixtures/README.md (script reproduced there)", "rng_seed": 20261102,
        "numpy": np.__version__, "dtype": "float64",
        "fields": "vectors: K x dim rows (row 0 = primary); weights/active/feasible: official output; "
                  "direction = weights @ vectors",
    },
    "problems": problems,
}
with open(out, "w") as f:
    json.dump(doc, f, indent=1)
from collections import Counter
print(len(problems), Counter((p["kind"], p["K"], tuple(p["active"]), p["feasible"]) for p in problems))
```
