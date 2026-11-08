"""The marketing mix model: design matrix, fit, prediction, decomposition, ROI, and response curves.

sales[t] / y_scale = intercept + controls[t] @ gamma + sum_c beta_c * hill(adstock(spend_c)[t] / scale_c)

Controls (standardized on the training rows): linear trend, two yearly sine and cosine pairs, Black Friday and
Christmas dummies, price index, promo flag, competitor index. Media coefficients are constrained to be >= 0.
The whole spend series is transformed before rows are selected, so carryover flows from training into holdout.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .ridge import Prior, fit_constrained_ridge
from .transforms import Adstock, Hill, channel_scale, hill, hill_derivative, steady_state_multiplier, transform
from .world import CHANNELS, WEEKS_PER_YEAR, holidays

CONTROL_NAMES = ("trend", "sin1", "cos1", "sin2", "cos2", "black_friday", "christmas", "price_index", "promo",
                 "competitor_index")


def control_matrix(df: pd.DataFrame) -> np.ndarray:
    t = df["t"].to_numpy(dtype=float)
    woy = df["woy"].to_numpy()
    ang = 2 * np.pi * woy / WEEKS_PER_YEAR
    hol = holidays(woy)
    cols = [t / WEEKS_PER_YEAR, np.sin(ang), np.cos(ang), np.sin(2 * ang), np.cos(2 * ang), hol["black_friday"],
            hol["christmas"], df["price_index"].to_numpy(), df["promo"].to_numpy(), df["competitor_index"].to_numpy()]
    return np.column_stack(cols)


@dataclass(frozen=True)
class ChannelHyper:
    adstock: Adstock
    hill: Hill

    def to_dict(self) -> dict:
        return {"adstock": self.adstock.to_dict(), "adstock_label": self.adstock.label(), "hill": self.hill.to_dict()}


def media_columns(df: pd.DataFrame, hyper: dict[str, ChannelHyper], channels=CHANNELS) -> tuple[np.ndarray, dict]:
    scales = {c: channel_scale(df[c].to_numpy(), hyper[c].adstock) for c in channels}
    H = np.column_stack([transform(df[c].to_numpy(), hyper[c].adstock, hyper[c].hill, scales[c]) for c in channels])
    return H, scales


@dataclass
class FittedMMM:
    channels: tuple[str, ...]
    hyper: dict[str, ChannelHyper]
    scales: dict[str, float]
    lam: float
    ctrl_mean: np.ndarray
    ctrl_sd: np.ndarray
    y_scale: float
    coef: np.ndarray  # [intercept, controls..., media...] in y_scale units
    train_rows: np.ndarray
    sigma: float = 0.0  # residual sd on training rows, y_scale units
    priors: tuple[Prior, ...] = ()
    meta: dict = field(default_factory=dict)

    @property
    def n_ctrl(self) -> int:
        return len(self.ctrl_mean)

    @property
    def beta(self) -> dict[str, float]:
        """Media coefficients in revenue dollars per week at full saturation."""
        m = self.coef[1 + self.n_ctrl :]
        return {c: float(b * self.y_scale) for c, b in zip(self.channels, m, strict=True)}

    def design(self, df: pd.DataFrame, H: np.ndarray | None = None) -> np.ndarray:
        C = (control_matrix(df) - self.ctrl_mean) / self.ctrl_sd
        if H is None:
            H = np.column_stack([transform(df[c].to_numpy(), self.hyper[c].adstock, self.hyper[c].hill,
                                           self.scales[c]) for c in self.channels])
        return np.column_stack([np.ones(len(df)), C, H])

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.design(df) @ self.coef * self.y_scale

    def contributions(self, df: pd.DataFrame, coef: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Revenue per week: baseline (intercept and controls) and each channel. Sums to predict(df)."""
        coef = self.coef if coef is None else coef
        X = self.design(df)
        k = 1 + self.n_ctrl
        out = {"baseline": X[:, :k] @ coef[:k] * self.y_scale}
        for j, c in enumerate(self.channels):
            out[c] = X[:, k + j] * coef[k + j] * self.y_scale
        return out

    def roi(self, df: pd.DataFrame, rows: np.ndarray | None = None, coef: np.ndarray | None = None) -> dict:
        contrib = self.contributions(df, coef)
        rows = np.arange(len(df)) if rows is None else rows
        out = {}
        for c in self.channels:
            s = df[c].to_numpy()[rows].sum()
            out[c] = float(contrib[c][rows].sum() / s) if s > 0 else 0.0
        return out

    def response(self, c: str, weekly_spend, beta: float | None = None) -> np.ndarray:
        """Weekly revenue from a constant weekly spend once carryover settles."""
        h = self.hyper[c]
        z = np.asarray(weekly_spend, dtype=float) * steady_state_multiplier(h.adstock) / self.scales[c]
        b = self.beta[c] if beta is None else beta
        return b * hill(z, h.hill.half_sat, h.hill.shape)

    def marginal_roi(self, c: str, weekly_spend, beta: float | None = None) -> np.ndarray:
        """Extra weekly revenue per extra dollar of weekly spend, at that spend level."""
        h = self.hyper[c]
        m = steady_state_multiplier(h.adstock) / self.scales[c]
        z = np.asarray(weekly_spend, dtype=float) * m
        b = self.beta[c] if beta is None else beta
        return b * m * hill_derivative(z, h.hill.half_sat, h.hill.shape)

    def betas_from_coef(self, coef: np.ndarray) -> dict[str, float]:
        m = coef[1 + self.n_ctrl :]
        return {c: float(b * self.y_scale) for c, b in zip(self.channels, m, strict=True)}


def fit_mmm(
    df: pd.DataFrame,
    hyper: dict[str, ChannelHyper],
    lam: float,
    rows: np.ndarray,
    priors_fn=None,
    channels: tuple[str, ...] = CHANNELS,
    H: np.ndarray | None = None,
    sigma: float | None = None,
) -> FittedMMM:
    """Fit on the given rows. priors_fn(fitted_shell) returns Prior objects (used for lift-test calibration)."""
    C = control_matrix(df)
    mu, sd = C[rows].mean(axis=0), C[rows].std(axis=0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    if H is None:
        H, scales = media_columns(df, hyper, channels)
    else:
        scales = {c: channel_scale(df[c].to_numpy(), hyper[c].adstock) for c in channels}
    y = df["sales"].to_numpy()
    y_scale = float(y[rows].mean())
    shell = FittedMMM(tuple(channels), dict(hyper), scales, lam, mu, sd, y_scale, np.zeros(1 + C.shape[1] + H.shape[1]),
                      rows, sigma=sigma or 0.0)
    X = shell.design(df, H)
    p = X.shape[1]
    nonneg = np.zeros(p, dtype=bool)
    nonneg[1 + C.shape[1] :] = True
    penalize = np.ones(p, dtype=bool)
    penalize[0] = False
    priors = tuple(priors_fn(shell, H)) if priors_fn else ()
    coef = fit_constrained_ridge(X[rows], y[rows] / y_scale, lam, nonneg, penalize, priors)
    shell.coef = coef
    shell.priors = priors
    resid = y[rows] / y_scale - X[rows] @ coef
    shell.sigma = float(resid.std(ddof=min(p, len(rows) - 1)))
    return shell


def mape(actual: np.ndarray, pred: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    return float(np.mean(np.abs(actual - pred) / np.abs(actual)))
