"""Budget optimizer: split a weekly budget across channels to maximize predicted media revenue.

Each channel's curve is steady state weekly revenue r_c(x) = beta_c * hill(x * m_c / scale_c), with x the weekly
spend. maximize sum_c r_c(x_c) subject to sum_c x_c = budget and lo_c <= x_c <= hi_c.

The main solver is scipy SLSQP from several starting points (Hill curves with shape > 1 are S shaped, so the
problem is not concave and one start can stall). A greedy marginal ROI allocator (start at the minimums, then hand
out the budget in small steps to whichever channel returns the most for the next step) is the fallback when scipy
is missing, and always runs as one more candidate. The best feasible answer wins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .transforms import hill, hill_derivative

try:  # scipy is a dependency, but the greedy path keeps working without it
    from scipy.optimize import minimize
except ImportError:  # pragma: no cover
    minimize = None


@dataclass(frozen=True)
class Curve:
    beta: float
    mult: float  # steady state multiplier / scale
    half_sat: float
    shape: float

    def revenue(self, x):
        return self.beta * hill(np.asarray(x, dtype=float) * self.mult, self.half_sat, self.shape)

    def marginal(self, x):
        return self.beta * self.mult * hill_derivative(np.asarray(x, dtype=float) * self.mult, self.half_sat,
                                                       self.shape)


def total_revenue(curves: list[Curve], x: np.ndarray) -> float:
    return float(sum(c.revenue(v) for c, v in zip(curves, x, strict=True)))


def check_bounds(budget: float, lo: np.ndarray, hi: np.ndarray) -> None:
    if np.any(lo < 0) or np.any(hi < lo):
        raise ValueError("each channel needs 0 <= min <= max")
    if lo.sum() > budget + 1e-6:
        raise ValueError(f"the minimums add up to {lo.sum():,.0f}, more than the budget {budget:,.0f}")
    if hi.sum() < budget - 1e-6:
        raise ValueError(f"the maximums add up to {hi.sum():,.0f}, less than the budget {budget:,.0f}")


def greedy(curves: list[Curve], budget: float, lo: np.ndarray, hi: np.ndarray, steps: int = 2000) -> np.ndarray:
    check_bounds(budget, lo, hi)
    x = lo.astype(float).copy()
    remaining = budget - x.sum()
    step = max(budget / steps, 1e-9)
    while remaining > 1e-9:
        d = min(step, remaining)
        gains = np.array([c.revenue(min(v + d, h)) - c.revenue(v) if v < h - 1e-9 else -np.inf
                          for c, v, h in zip(curves, x, hi, strict=True)])
        j = int(np.argmax(gains))
        add = min(d, hi[j] - x[j])
        x[j] += add
        remaining -= add
    return x


def _slsqp(curves, budget, lo, hi, x0):
    scale = max(budget, 1.0)

    def f(z):
        return -total_revenue(curves, z * scale) / scale

    def g(z):
        return -np.array([c.marginal(v * scale) for c, v in zip(curves, z, strict=True)], dtype=float).ravel()

    res = minimize(f, x0 / scale, jac=g, method="SLSQP", bounds=list(zip(lo / scale, hi / scale, strict=True)),
                   constraints=[{"type": "eq", "fun": lambda z: z.sum() - budget / scale,
                                 "jac": lambda z: np.ones_like(z)}], options={"maxiter": 300, "ftol": 1e-12})
    return np.clip(res.x * scale, lo, hi)


def optimize(curves: list[Curve], budget: float, lo, hi, start=None, solver: str = "auto", steps: int = 2000,
             starts: int = 3) -> dict:
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    check_bounds(budget, lo, hi)
    cands = {"greedy": greedy(curves, budget, lo, hi, steps)}
    if solver != "greedy" and minimize is not None:
        inits = [cands["greedy"], np.clip(np.full(len(curves), budget / len(curves)), lo, hi)]
        if start is not None:
            s = np.asarray(start, dtype=float)
            inits.append(np.clip(s * budget / max(s.sum(), 1e-9), lo, hi))
        for i, s0 in enumerate(inits[:starts]):
            cands[f"slsqp{i}"] = _slsqp(curves, budget, lo, hi, s0)
    best_name, best_x, best_v = None, None, -np.inf
    for name, x in cands.items():
        feasible = abs(x.sum() - budget) <= 1e-3 * max(budget, 1) and np.all(x >= lo - 1e-6) and np.all(x <= hi + 1e-6)
        v = total_revenue(curves, x)
        if feasible and v > best_v + 1e-9:
            best_name, best_x, best_v = name, x, v
    return {"x": best_x, "revenue": best_v, "solver": "greedy" if best_name == "greedy" else "slsqp"}
