"""Per-objective EMA gradient normalization (PCD Def. 4.5).

    v_t = beta v_{t-1} + (1 - beta) ||g_t||^2
    v_hat_t = v_t / (1 - beta^t),   s_t = 1 / sqrt(v_hat_t + eps)

One scalar state per objective name, each with its own step counter. Arithmetic is done in
Python float64 (only ||g||^2 is read from the device), so it works on MPS.
"""

from __future__ import annotations

import math

import torch


class EMANormalizer:
    def __init__(self, beta: float = 0.99, eps: float = 1e-8):
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta must be in [0, 1), got {beta}")
        if eps < 0:
            raise ValueError(f"eps must be non-negative, got {eps}")
        self.beta = float(beta)
        self.eps = float(eps)
        self.v: dict[str, float] = {}
        self.t: dict[str, int] = {}

    def update_sq(self, name: str, sq_norm: float) -> float:
        """Update objective ``name`` with a squared gradient norm and return its scale s_t."""
        sq_norm = float(sq_norm)
        if not math.isfinite(sq_norm) or sq_norm < 0:
            raise ValueError(f"squared norm must be finite and >= 0, got {sq_norm}")
        t = self.t.get(name, 0) + 1
        v = self.beta * self.v.get(name, 0.0) + (1.0 - self.beta) * sq_norm
        self.t[name], self.v[name] = t, v
        v_hat = v / (1.0 - self.beta**t)
        denom = math.sqrt(v_hat + self.eps)
        return 1.0 / denom if denom > 0 else 0.0

    def scale(self, name: str, grad_vec: torch.Tensor) -> float:
        """Update with ``grad_vec`` (any shape) and return s_t for objective ``name``."""
        sq = float(torch.linalg.vector_norm(grad_vec.detach()).item()) ** 2
        return self.update_sq(name, sq)

    def v_hat(self, name: str) -> float:
        t = self.t.get(name, 0)
        return 0.0 if t == 0 else self.v[name] / (1.0 - self.beta**t)

    def state_dict(self) -> dict:
        return {"beta": self.beta, "eps": self.eps, "v": dict(self.v), "t": dict(self.t)}

    def load_state_dict(self, state: dict) -> None:
        self.beta, self.eps = float(state["beta"]), float(state["eps"])
        self.v = {k: float(x) for k, x in state["v"].items()}
        self.t = {k: int(x) for k, x in state["t"].items()}
