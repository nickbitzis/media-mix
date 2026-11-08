"""Constrained ridge regression: non-negative media coefficients, optional Gaussian priors on single coefficients.

Minimizes ||y - X b||^2 + lam * sum_{j penalized} b_j^2 + sum_k w_k^2 (b_{j_k} - mu_k)^2
subject to lower bounds (0 for media, -inf for controls), by stacking the penalty rows under X and solving one
bounded least squares problem. A prior row with weight sigma / se is the posterior mode of a model with Gaussian
noise sigma and a Normal(mu, se) prior on that coefficient, which is how lift-test calibration enters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear


@dataclass(frozen=True)
class Prior:
    col: int
    mean: float
    weight: float  # sigma_noise / se_coefficient


def fit_constrained_ridge(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    nonneg: np.ndarray,
    penalize: np.ndarray,
    priors: tuple[Prior, ...] = (),
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    p = X.shape[1]
    rows = [X]
    rhs = [y]
    pen = np.flatnonzero(penalize)
    if lam > 0 and len(pen):
        R = np.zeros((len(pen), p))
        R[np.arange(len(pen)), pen] = np.sqrt(lam)
        rows.append(R)
        rhs.append(np.zeros(len(pen)))
    for pr in priors:
        r = np.zeros((1, p))
        r[0, pr.col] = pr.weight
        rows.append(r)
        rhs.append(np.array([pr.weight * pr.mean]))
    A = np.vstack(rows)
    b = np.concatenate(rhs)
    lo = np.where(nonneg, 0.0, -np.inf)
    hi = np.full(p, np.inf)
    if not np.any(nonneg):
        return np.linalg.lstsq(A, b, rcond=None)[0]
    return lsq_linear(A, b, bounds=(lo, hi), method="bvls", lsmr_tol="auto").x
