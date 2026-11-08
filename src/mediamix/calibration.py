"""Lift-test calibration: a simulated geo experiment on one channel, turned into a prior on its coefficient.

The experiment: 80 regions, half chosen at random go dark on the channel for 6 weeks inside a flight. Regional
revenue is the region's share of national revenue plus regional noise; dark regions lose their share of the
channel's true effect, including carryover into the 4 weeks after the test. The readout is a difference in
differences against an 8 week pre-period, scaled to national revenue, divided by the national spend the dark
regions would have received. That is the incremental revenue per dollar (iROAS) for that flight.

The model's version of the same number is linear in the channel's coefficient beta:
    iROAS_model = beta * sum_window(h_full - h_dark) / saved_spend
so the experiment's estimate and standard error become a Normal prior on beta, added as one penalty row in the
ridge solve (weight sigma_noise / se_beta). That is exactly the posterior mode with that prior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .ridge import Prior
from .transforms import transform
from .world import World


@dataclass
class LiftTest:
    channel: str
    test_rows: np.ndarray  # weeks the treated regions are dark
    window_rows: np.ndarray  # test weeks plus the carryover weeks after
    pre_rows: np.ndarray
    saved_spend: float  # national spend the dark regions would have received, per unit of share
    iroas: float  # estimate
    se: float
    true_iroas: float
    n_regions: int
    treated_share: float

    def to_dict(self) -> dict:
        return {"channel": self.channel, "test_weeks": [int(self.test_rows[0]), int(self.test_rows[-1])],
                "window_weeks": [int(self.window_rows[0]), int(self.window_rows[-1])],
                "iroas": round(self.iroas, 3), "se": round(self.se, 3),
                "ci90": [round(self.iroas - 1.645 * self.se, 3), round(self.iroas + 1.645 * self.se, 3)],
                "true_iroas": round(self.true_iroas, 3), "regions": self.n_regions,
                "treated_share": round(self.treated_share, 3)}


def dark_spend(df: pd.DataFrame, channel: str, rows: np.ndarray) -> np.ndarray:
    x = df[channel].to_numpy().copy()
    x[rows] = 0.0
    return x


def simulate_geo_test(world: World, channel: str = "tv", start: int = 93, weeks: int = 6, carry: int = 4,
                      pre: int = 8, n_regions: int = 80, region_noise: float = 0.015, seed: int = 99) -> LiftTest:
    rng = np.random.default_rng(seed)
    df = world.data
    test_rows = np.arange(start, start + weeks)
    window = np.arange(start, start + weeks + carry)
    pre_rows = np.arange(start - pre, start)
    tr = world.truth[channel]
    full = tr["contribution"]
    dark = tr["beta"] * transform(dark_spend(df, channel, test_rows), tr["adstock"], tr["hill"], tr["scale"])
    loss = full - dark  # national revenue lost if every region went dark

    share = rng.dirichlet(np.full(n_regions, 4.0))
    treated = np.zeros(n_regions, dtype=bool)
    treated[rng.choice(n_regions, n_regions // 2, replace=False)] = True
    sales = df["sales"].to_numpy()
    rows = np.concatenate([pre_rows, window])
    # Regional revenue per unit of share, so regions of different size are comparable.
    per_share = np.empty((n_regions, len(rows)))
    for g in range(n_regions):
        y = sales[rows] - np.where(treated[g], loss[rows], 0.0)
        noise = rng.normal(0, region_noise * sales[rows] / np.sqrt(share[g] * n_regions))
        per_share[g] = y + noise
    pre_mean = per_share[:, : len(pre_rows)].mean(axis=1)
    win_sum = per_share[:, len(pre_rows) :].sum(axis=1)
    did = win_sum - len(window) * pre_mean
    lost = did[~treated].mean() - did[treated].mean()
    se_lost = np.sqrt(did[~treated].var(ddof=1) / (~treated).sum() + did[treated].var(ddof=1) / treated.sum())
    saved = float(df[channel].to_numpy()[test_rows].sum())
    return LiftTest(channel, test_rows, window, pre_rows, saved, float(lost / saved), float(se_lost / saved),
                    float(loss[window].sum() / saved), n_regions, float(share[treated].sum()))


def lift_prior_fn(df: pd.DataFrame, test: LiftTest, sigma: float, strength: float = 1.0):
    """Return a priors_fn for fit_mmm: a Normal(iROAS, se) prior mapped onto the channel's coefficient."""

    def fn(shell, H):
        c = test.channel
        j = shell.channels.index(c)
        h = shell.hyper[c]
        full = H[:, j]
        dark = transform(dark_spend(df, c, test.test_rows), h.adstock, h.hill, shell.scales[c])
        lift_per_beta = float((full - dark)[test.window_rows].sum()) * shell.y_scale / test.saved_spend
        if lift_per_beta <= 1e-12:
            return []
        mean = test.iroas / lift_per_beta
        se = test.se / lift_per_beta
        return [Prior(1 + shell.n_ctrl + j, mean, strength * sigma / se)]

    return fn
