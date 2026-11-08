"""Seeded synthetic world for Juniper Row, a made-up direct-to-consumer bedding brand.

Three years of weekly revenue driven by a baseline (trend, yearly seasonality, holidays, a peak season bump),
price and promotions, competitor activity, and seven media channels with known true adstock, saturation, and ROI.
ROI here is revenue per media dollar over the whole period (often called ROAS).

Built-in challenges, each a common reason real MMMs go wrong:
- social and display share a campaign calendar, so their spend is correlated.
- TV runs mostly in October to December, the same weeks as the peak season bump (confounding).
- podcast is a flat sponsorship contract: almost no week to week variation to learn from.
- paid search is always on and its budget follows demand, including demand TV, social, and influencer created.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .transforms import Adstock, Hill, channel_scale, transform

CHANNELS = ("tv", "search", "social", "display", "podcast", "influencer", "email")
LABELS = {
    "tv": "TV",
    "search": "Paid search",
    "social": "Paid social",
    "display": "Display",
    "podcast": "Podcast",
    "influencer": "Influencer",
    "email": "Email",
}
CONTROLS = ("price_index", "promo", "competitor_index")
WEEKS_PER_YEAR = 52


@dataclass(frozen=True)
class ChannelTruth:
    name: str
    adstock: Adstock
    hill: Hill
    roi: float  # true revenue per dollar over the whole period
    challenge: str = ""


TRUTH = (
    ChannelTruth("tv", Adstock("weibull", shape=2.0, scale=3.0), Hill(1.0, 2.0), 1.4,
                 "Mostly on during peak season, so it is confounded with seasonal demand."),
    ChannelTruth("search", Adstock("geometric", decay=0.0), Hill(1.5, 1.0), 1.1,
                 "Always on, and its budget follows demand that other channels created."),
    ChannelTruth("social", Adstock("geometric", decay=0.3), Hill(1.0, 1.5), 1.9,
                 "Shares a campaign calendar with display (correlated spend)."),
    ChannelTruth("display", Adstock("geometric", decay=0.2), Hill(1.5, 1.0), 0.8,
                 "Shares a campaign calendar with social (correlated spend)."),
    ChannelTruth("podcast", Adstock("weibull", shape=1.5, scale=3.0), Hill(1.0, 2.0), 1.5,
                 "Flat sponsorship contract with almost no spend variation."),
    ChannelTruth("influencer", Adstock("geometric", decay=0.5), Hill(0.75, 2.0), 2.4, ""),
    ChannelTruth("email", Adstock("geometric", decay=0.1), Hill(1.0, 1.0), 4.0, ""),
)


@dataclass
class WorldConfig:
    seed: int = 7
    weeks: int = 3 * WEEKS_PER_YEAR
    start: str = "2023-01-02"
    base_level: float = 900_000.0
    trend_per_year: float = 0.12
    noise_sd: float = 12_000.0
    truth: tuple[ChannelTruth, ...] = TRUTH
    search_demand_share: float = 0.05  # search budget per dollar of demand
    shock_sd: float = 0.01  # weekly unobserved demand shocks, share of base level
    peak_bump: float = 0.14  # extra Oct to Dec demand the model's seasonality terms cannot fully capture


@dataclass
class World:
    config: WorldConfig
    data: pd.DataFrame  # observed: week, controls, spend per channel, sales
    truth: dict  # per channel: params, beta, scale, contribution series
    baseline: np.ndarray  # true baseline revenue (everything except media and noise)
    extra: dict = field(default_factory=dict)

    @property
    def spend(self) -> pd.DataFrame:
        return self.data[list(CHANNELS)]

    def true_roi(self, idx: np.ndarray | slice | None = None) -> dict[str, float]:
        idx = slice(None) if idx is None else idx
        out = {}
        for c in CHANNELS:
            s = self.data[c].to_numpy()[idx].sum()
            out[c] = float(self.truth[c]["contribution"][idx].sum() / s) if s > 0 else 0.0
        return out

    def true_response(self, c: str, weekly_spend: np.ndarray | float) -> np.ndarray:
        """True weekly revenue from a constant weekly spend once carryover settles."""
        t = self.truth[c]
        a: Adstock = t["adstock"]
        h: Hill = t["hill"]
        from .transforms import hill, steady_state_multiplier

        z = np.asarray(weekly_spend, dtype=float) * steady_state_multiplier(a) / t["scale"]
        return t["beta"] * hill(z, h.half_sat, h.shape)


def holidays(woy: np.ndarray) -> dict[str, np.ndarray]:
    return {"black_friday": (woy == 47).astype(float), "christmas": (woy == 51).astype(float)}


def _ar1(rng, n, phi, sd):
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0, sd)
    return x


def simulate(cfg: WorldConfig | None = None) -> World:
    cfg = cfg or WorldConfig()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.weeks
    t = np.arange(n)
    woy = t % WEEKS_PER_YEAR
    year = t // WEEKS_PER_YEAR
    dates = pd.date_range(cfg.start, periods=n, freq="7D")

    # Baseline: trend, two yearly harmonics, holidays, and a peak season bump centered on week 45.
    ang = 2 * np.pi * woy / WEEKS_PER_YEAR
    season = 0.07 * np.sin(ang - 1.2) + 0.03 * np.cos(2 * ang)
    bump = cfg.peak_bump * np.exp(-0.5 * ((woy - 45) / 2.5) ** 2)
    hol = holidays(woy)
    level = cfg.base_level * (1 + cfg.trend_per_year * t / WEEKS_PER_YEAR)
    shock = _ar1(rng, n, 0.7, cfg.shock_sd)  # unobserved demand: word of mouth, weather, press
    demand = level + cfg.base_level * (season + bump + shock + 0.30 * hol["black_friday"] + 0.18 * hol["christmas"])

    # Promotions: Black Friday, Christmas, plus about 8 random promo weeks a year.
    promo = np.maximum(hol["black_friday"], hol["christmas"])
    for y in range(int(year.max()) + 1):
        weeks = rng.choice(np.setdiff1d(np.arange(2, 46), [47, 51]), 8, replace=False) + y * WEEKS_PER_YEAR
        promo[weeks[weeks < n]] = 1.0
    price_index = np.where(promo == 1, 0.85, 1.0) * (1 + 0.04 * (year >= 2)) + rng.normal(0, 0.01, n)
    competitor_index = np.clip(1 + _ar1(rng, n, 0.8, 0.12), 0.4, None)
    baseline = demand - 700_000 * (price_index - 1.0) + 60_000 * promo - 90_000 * (competitor_index - 1.0)

    # Media spend plans.
    spend: dict[str, np.ndarray] = {}
    peak = (woy >= 40) & (woy <= 51)
    spring = (woy >= 14) & (woy <= 16) & (year >= 1)
    spend["tv"] = np.where(peak, rng.uniform(140_000, 230_000, n), 0.0) + np.where(spring, 70_000, 0.0)
    flight = np.clip(1 + _ar1(rng, n, 0.85, 0.25), 0.2, None)
    spend["social"] = 45_000 * flight * rng.lognormal(0, 0.3, n)
    spend["display"] = 35_000 * flight * rng.lognormal(0, 0.3, n)
    spend["podcast"] = 18_000 * (1 + rng.normal(0, 0.02, n))
    infl = np.zeros(n)
    for start in rng.choice(n - 3, 14, replace=False):
        infl[start : start + rng.integers(2, 4)] += rng.uniform(25_000, 55_000)
    spend["influencer"] = infl
    spend["email"] = np.where(rng.random(n) < 0.55, 15_000 * rng.lognormal(0, 0.3, n), 0.0)

    truth: dict[str, dict] = {}
    contrib: dict[str, np.ndarray] = {}

    def add_channel(ct: ChannelTruth, x: np.ndarray):
        scale = channel_scale(x, ct.adstock)
        h = transform(x, ct.adstock, ct.hill, scale)
        beta = ct.roi * x.sum() / h.sum()
        contrib[ct.name] = beta * h
        truth[ct.name] = {"adstock": ct.adstock, "hill": ct.hill, "roi": ct.roi, "beta": float(beta),
                          "scale": scale, "contribution": beta * h, "challenge": ct.challenge}

    by_name = {ct.name: ct for ct in cfg.truth}
    for c in ("tv", "social", "display", "podcast", "influencer", "email"):
        add_channel(by_name[c], spend[c])
    # Search follows demand: base demand plus the demand TV, social, and influencer created.
    created = contrib["tv"] + contrib["social"] + contrib["influencer"]
    spend["search"] = cfg.search_demand_share * (demand + 0.6 * created) * rng.lognormal(0, 0.06, n)
    add_channel(by_name["search"], spend["search"])

    media = sum(contrib[c] for c in CHANNELS)
    sales = baseline + media + rng.normal(0, cfg.noise_sd, n)
    data = pd.DataFrame({"week": dates.strftime("%Y-%m-%d"), "t": t, "woy": woy, "price_index": price_index,
                         "promo": promo, "competitor_index": competitor_index,
                         **{c: spend[c] for c in CHANNELS}, "sales": sales})
    return World(cfg, data, truth, baseline, {"peak": peak, "demand": demand})
