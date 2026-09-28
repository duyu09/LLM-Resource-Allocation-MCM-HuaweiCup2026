from __future__ import annotations

import numpy as np
from scipy.linalg import helmert


def closure(p, eps: float = 1e-6):
    x = np.asarray(p, dtype=float)
    x = np.maximum(x, 0.0) + eps
    if x.ndim == 1:
        return x / x.sum()
    return x / x.sum(axis=1, keepdims=True)


def clr(p, eps: float = 1e-6):
    x = closure(p, eps)
    lx = np.log(x)
    if x.ndim == 1:
        return lx - lx.mean()
    return lx - lx.mean(axis=1, keepdims=True)


def ilr_basis(n_parts: int) -> np.ndarray:
    # Orthonormal rows perpendicular to the all-ones vector.
    return helmert(n_parts, full=False)


def ilr(p, eps: float = 1e-6):
    c = clr(p, eps)
    H = ilr_basis(c.shape[-1])
    return c @ H.T


def aitchison_center(p, eps: float = 1e-6):
    x = closure(p, eps)
    if x.ndim == 1:
        return x
    g = np.exp(np.mean(np.log(x), axis=0))
    return g / g.sum()


def softmax_logits_to_composition(logits16):
    s = np.asarray(logits16, dtype=float)
    s = np.concatenate([s, np.zeros(1, dtype=float)])
    s = s - np.max(s)
    e = np.exp(s)
    return e / e.sum()


def composition_entropy(p, eps: float = 1e-12):
    x = np.asarray(p, float)
    x = np.clip(x, eps, 1.0)
    return float(-np.sum(x*np.log(x)))
