"""Hyperparameter search: adstock shape, Hill half saturation and shape per channel, and the ridge penalty.

The score is out of sample error on rolling validation windows (fit on everything before a window, predict the
window), averaged over folds. A full grid over 7 channels is too large (hundreds of candidates each, to the 7th
power), so the search is coordinate descent: one channel at a time, alternating between its adstock and its Hill
parameters, holding the rest fixed, for a few passes. Transformed columns are cached, so each candidate is one
small bounded least squares solve.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .model import ChannelHyper, fit_mmm
from .transforms import Adstock, Hill, channel_scale, transform
from .world import CHANNELS

GEOMETRIC_DECAYS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
WEIBULL = ((1.5, 2.0), (1.5, 3.0), (2.0, 3.0), (2.0, 5.0), (3.0, 4.0))
HALF_SATS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
HILL_SHAPES = (1.0, 1.5, 2.0, 3.0)
LAMBDAS = (0.003, 0.01, 0.03, 0.1, 0.3)


@dataclass
class Grid:
    adstocks: tuple[Adstock, ...] = tuple(
        [Adstock("geometric", decay=d) for d in GEOMETRIC_DECAYS]
        + [Adstock("weibull", shape=k, scale=s) for k, s in WEIBULL]
    )
    hills: tuple[Hill, ...] = tuple(Hill(k, s) for k, s in itertools.product(HALF_SATS, HILL_SHAPES))
    lambdas: tuple[float, ...] = LAMBDAS

    @property
    def size(self) -> int:
        return len(self.adstocks) * len(self.hills)


DEFAULT_START = ChannelHyper(Adstock("geometric", decay=0.3), Hill(1.0, 1.5))


def rolling_folds(train_rows: np.ndarray, n_folds: int = 2, window: int = 26) -> list[tuple[np.ndarray, np.ndarray]]:
    end = len(train_rows)
    folds = []
    for k in range(n_folds, 0, -1):
        v0 = end - k * window
        folds.append((train_rows[:v0], train_rows[v0 : v0 + window]))
    return folds


@dataclass
class SearchResult:
    hyper: dict[str, ChannelHyper]
    lam: float
    score: float
    evaluations: int
    history: list[dict] = field(default_factory=list)


def search(
    df: pd.DataFrame,
    train_rows: np.ndarray,
    grid: Grid | None = None,
    channels: tuple[str, ...] = CHANNELS,
    passes: int = 2,
    priors_fn=None,
    start: dict[str, ChannelHyper] | None = None,
    lam: float = 0.03,
    n_folds: int = 2,
    window: int = 26,
) -> SearchResult:
    grid = grid or Grid()
    folds = rolling_folds(train_rows, n_folds, window)
    y = df["sales"].to_numpy()
    hyper = dict(start) if start else {c: DEFAULT_START for c in channels}
    cache: dict[tuple[str, Adstock, Hill], np.ndarray] = {}

    def column(c: str, h: ChannelHyper) -> np.ndarray:
        key = (c, h.adstock, h.hill)
        if key not in cache:
            x = df[c].to_numpy()
            cache[key] = transform(x, h.adstock, h.hill, channel_scale(x, h.adstock))
        return cache[key]

    evals = 0

    def score(hy: dict[str, ChannelHyper], lam_: float) -> float:
        nonlocal evals
        H = np.column_stack([column(c, hy[c]) for c in channels])
        err = 0.0
        for fit_rows, val_rows in folds:
            m = fit_mmm(df, hy, lam_, fit_rows, priors_fn=priors_fn, channels=channels, H=H)
            pred = m.design(df, H)[val_rows] @ m.coef * m.y_scale
            err += float(np.mean(((y[val_rows] - pred) / y[val_rows]) ** 2))
            evals += 1
        return err / len(folds)

    best = score(hyper, lam)
    history = [{"pass": 0, "channel": None, "score": best}]
    for p in range(1, passes + 1):
        for c in channels:
            for which in ("adstock", "hill"):
                options = grid.adstocks if which == "adstock" else grid.hills
                for opt in options:
                    cand = (ChannelHyper(opt, hyper[c].hill) if which == "adstock"
                            else ChannelHyper(hyper[c].adstock, opt))
                    if cand == hyper[c]:
                        continue
                    trial = {**hyper, c: cand}
                    s = score(trial, lam)
                    if s < best - 1e-12:
                        best, hyper = s, trial
            history.append({"pass": p, "channel": c, "score": best})
        for lam_ in grid.lambdas:
            s = score(hyper, lam_)
            if s < best - 1e-12:
                best, lam = s, lam_
        history.append({"pass": p, "channel": "lambda", "score": best, "lam": lam})
    return SearchResult(hyper, lam, best, evals, history)
