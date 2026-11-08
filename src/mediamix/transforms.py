"""Media transforms: adstock (carryover) then Hill saturation (diminishing returns).

Adstock spreads a week's spend over the following weeks. Two shapes:
- geometric: a[t] = x[t] + decay * a[t-1]. The effect peaks in the week of spend and fades.
- weibull: a[t] = sum_l w[l] * x[t-l], with w[l] the Weibull probability mass in [l, l+1), normalized to sum to 1.
  With shape > 1 the effect peaks a few weeks later, which suits TV and podcasts.

Hill saturation works on adstocked spend divided by a channel scale (mean weekly spend in weeks the channel is on,
times the adstock's steady state multiplier), so the half saturation point K is "this many typical weeks".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

WEIBULL_LAGS = 13


@dataclass(frozen=True)
class Adstock:
    kind: str  # "geometric" or "weibull"
    decay: float = 0.0  # geometric
    shape: float = 1.0  # weibull
    scale: float = 1.0  # weibull

    def label(self) -> str:
        if self.kind == "geometric":
            return f"geometric(decay={self.decay:g})"
        return f"weibull(shape={self.shape:g}, scale={self.scale:g})"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Hill:
    half_sat: float  # K, in channel scale units
    shape: float  # s

    def to_dict(self) -> dict:
        return asdict(self)


def geometric_adstock(x: np.ndarray, decay: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    carry = 0.0
    for t, v in enumerate(x):
        carry = v + decay * carry
        out[t] = carry
    return out


def weibull_weights(shape: float, scale: float, lags: int = WEIBULL_LAGS) -> np.ndarray:
    edges = np.arange(lags + 1, dtype=float)
    surv = np.exp(-((edges / scale) ** shape))
    w = surv[:-1] - surv[1:]
    return w / w.sum()


def weibull_adstock(x: np.ndarray, shape: float, scale: float, lags: int = WEIBULL_LAGS) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    w = weibull_weights(shape, scale, lags)
    return np.convolve(x, w)[: len(x)]


def adstock(x: np.ndarray, a: Adstock) -> np.ndarray:
    if a.kind == "geometric":
        return geometric_adstock(x, a.decay)
    if a.kind == "weibull":
        return weibull_adstock(x, a.shape, a.scale)
    raise ValueError(f"unknown adstock kind {a.kind}")


def steady_state_multiplier(a: Adstock) -> float:
    """Adstocked level of a constant spend of 1 per week, after the carryover settles."""
    if a.kind == "geometric":
        return 1.0 / (1.0 - a.decay)
    return 1.0


def hill(z: np.ndarray | float, half_sat: float, shape: float) -> np.ndarray:
    z = np.maximum(np.asarray(z, dtype=float), 0.0)
    zs = z**shape
    return zs / (zs + half_sat**shape)


def hill_derivative(z: np.ndarray | float, half_sat: float, shape: float) -> np.ndarray:
    z = np.maximum(np.asarray(z, dtype=float), 1e-12)
    ks = half_sat**shape
    return shape * ks * z ** (shape - 1) / (z**shape + ks) ** 2


def channel_scale(spend: np.ndarray, a: Adstock) -> float:
    """Typical weekly spend while the channel is on, in adstock units. Fixed from spend only (no sales)."""
    spend = np.asarray(spend, dtype=float)
    on = spend > 0.01 * spend.max() if spend.max() > 0 else np.ones_like(spend, dtype=bool)
    return float(spend[on].mean() * steady_state_multiplier(a)) or 1.0


def transform(spend: np.ndarray, a: Adstock, h: Hill, scale: float) -> np.ndarray:
    return hill(adstock(spend, a) / scale, h.half_sat, h.shape)
