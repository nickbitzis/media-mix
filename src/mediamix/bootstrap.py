"""Moving block bootstrap over weeks.

Weekly residuals are autocorrelated (shocks last a few weeks), so resampling single weeks understates the
uncertainty. Instead, draw random blocks of consecutive training weeks (8 by default) until the sample is as long
as the training window, refit the ridge on those rows, and recompute contributions and ROI on the original weeks.
Hyperparameters stay fixed at the searched values, so the intervals cover coefficient uncertainty, not
hyperparameter uncertainty (they are narrower than the truth would be).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import FittedMMM, fit_mmm


def block_indices(rows: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    n = len(rows)
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
    return rows[idx]


def bootstrap(model: FittedMMM, df: pd.DataFrame, n_draws: int = 200, block: int = 8, seed: int = 0,
              priors_fn=None) -> list[FittedMMM]:
    rng = np.random.default_rng(seed)
    H = model.design(df)[:, 1 + model.n_ctrl :]
    draws = []
    for _ in range(n_draws):
        rows = block_indices(model.train_rows, block, rng)
        draws.append(fit_mmm(df, model.hyper, model.lam, rows, priors_fn=priors_fn, channels=model.channels, H=H))
    return draws


def interval(values, level: float = 0.9) -> tuple[float, float]:
    a = (1 - level) / 2
    lo, hi = np.quantile(np.asarray(values, dtype=float), [a, 1 - a])
    return float(lo), float(hi)


def roi_draws(draws: list[FittedMMM], df: pd.DataFrame, rows: np.ndarray) -> dict[str, np.ndarray]:
    per = [d.roi(df, rows) for d in draws]
    return {c: np.array([p[c] for p in per]) for c in draws[0].channels}
