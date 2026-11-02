"""Running aggregation of combiner ``info`` dicts into JSON-serialisable summaries.

Numeric keys get count/mean/std/min/max (Welford, float64); bool keys get the fraction of True;
``cos_raw`` and ``cos_norm`` also get fixed-edge histograms on [-1, 1] (values outside are
counted in ``below``/``above``); ``prop47_margin`` also gets the fraction of steps with a
non-positive margin. ``None`` values (not applicable for that method) are skipped and
non-finite values are counted per key but excluded from the statistics. Strings, lists and
dicts are ignored.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable

import numpy as np

HIST_KEYS = ("cos_raw", "cos_norm")
_SKIP = frozenset({"step"})


class _Stat:
    __slots__ = ("n", "mean", "m2", "lo", "hi", "nonfinite")

    def __init__(self) -> None:
        self.n, self.mean, self.m2 = 0, 0.0, 0.0
        self.lo, self.hi = math.inf, -math.inf
        self.nonfinite = 0

    def add(self, x: float) -> None:
        if not math.isfinite(x):
            self.nonfinite += 1
            return
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)
        self.lo, self.hi = min(self.lo, x), max(self.hi, x)

    def summary(self) -> dict:
        if self.n == 0:
            return {"count": 0, "nonfinite": self.nonfinite}
        return {
            "count": self.n,
            "mean": self.mean,
            "std": math.sqrt(self.m2 / self.n),
            "min": self.lo,
            "max": self.hi,
            "nonfinite": self.nonfinite,
        }


class DiagnosticsAggregator:
    """Accumulate per-step ``info`` dicts. ``summary()`` / ``to_json()`` give the aggregate."""

    def __init__(self, n_bins: int = 20, hist_keys: Iterable[str] = HIST_KEYS):
        if n_bins < 1:
            raise ValueError("n_bins must be >= 1")
        self.edges = np.linspace(-1.0, 1.0, n_bins + 1)
        self.hist_keys = tuple(hist_keys)
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.numeric: dict[str, _Stat] = {}
        self.bools: dict[str, list[int]] = {}  # key -> [n_true, n]
        self.hists = {
            k: {"counts": np.zeros(len(self.edges) - 1, dtype=np.int64), "below": 0, "above": 0}
            for k in self.hist_keys
        }
        self.margin_nonpos = [0, 0]

    def update(self, info: dict) -> None:
        self.steps += 1
        for key, val in info.items():
            if key in _SKIP or val is None:
                continue
            if isinstance(val, bool | np.bool_):
                b = self.bools.setdefault(key, [0, 0])
                b[0] += int(bool(val))
                b[1] += 1
            elif isinstance(val, int | float | np.integer | np.floating):
                x = float(val)
                self.numeric.setdefault(key, _Stat()).add(x)
                if key in self.hists and math.isfinite(x):
                    self._hist_add(key, x)
                if key == "prop47_margin" and math.isfinite(x):
                    self.margin_nonpos[0] += int(x <= 0.0)
                    self.margin_nonpos[1] += 1

    def _hist_add(self, key: str, x: float) -> None:
        h = self.hists[key]
        if x < self.edges[0]:
            h["below"] += 1
        elif x > self.edges[-1]:
            h["above"] += 1
        else:
            i = int(np.searchsorted(self.edges, x, side="right")) - 1
            h["counts"][min(i, len(h["counts"]) - 1)] += 1

    def fraction(self, key: str) -> float | None:
        b = self.bools.get(key)
        return None if not b or b[1] == 0 else b[0] / b[1]

    def summary(self) -> dict:
        out: dict = {
            "steps": self.steps,
            "numeric": {k: s.summary() for k, s in sorted(self.numeric.items())},
            "fractions": {k: b[0] / b[1] for k, b in sorted(self.bools.items()) if b[1]},
            "histograms": {
                k: {
                    "edges": self.edges.tolist(),
                    "counts": h["counts"].tolist(),
                    "below": h["below"],
                    "above": h["above"],
                }
                for k, h in self.hists.items()
            },
            "prop47_nonpositive_fraction": (
                self.margin_nonpos[0] / self.margin_nonpos[1] if self.margin_nonpos[1] else None
            ),
        }
        return out

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.summary(), allow_nan=False, **kwargs)
