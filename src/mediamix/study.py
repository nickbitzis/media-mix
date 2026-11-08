"""End to end study: simulate, search, fit, calibrate, bootstrap, validate against truth, plan the budget."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .bootstrap import bootstrap, interval, roi_draws
from .calibration import LiftTest, lift_prior_fn, simulate_geo_test
from .model import FittedMMM, fit_mmm, mape
from .optimizer import Curve, optimize, total_revenue
from .search import Grid, search
from .transforms import steady_state_multiplier
from .world import CHANNELS, LABELS, World, WorldConfig, simulate

QUARTER_WEEKS = 13
GATE_CHANNELS = ("influencer", "email")
GATE_PAIR = ("social", "display")  # correlated pair: the split is weakly identified, the combined ROI is not
GATE_TOLERANCE = 0.30
ROI_TOLERANCE = 0.25  # "recovered" in the ROI table when within 25% of truth


@dataclass
class StudyConfig:
    seed: int = 3
    holdout_weeks: int = 26
    passes: int = 2
    n_boot: int = 200
    n_boot_opt: int = 100
    block: int = 8
    lift_channel: str = "search"
    world: WorldConfig | None = None


@dataclass
class Study:
    config: StudyConfig = field(default_factory=StudyConfig)
    world: World | None = None
    base: FittedMMM | None = None
    calibrated: FittedMMM | None = None
    lift: LiftTest | None = None
    draws: dict = field(default_factory=dict)
    report: dict = field(default_factory=dict)
    current: np.ndarray | None = None

    def run(self) -> Study:
        cfg = self.config
        timings = {}
        t0 = time.perf_counter()
        self.world = simulate(cfg.world or WorldConfig(seed=cfg.seed))
        df = self.world.data
        n = len(df)
        self.train = np.arange(n - cfg.holdout_weeks)
        self.hold = np.arange(n - cfg.holdout_weeks, n)
        timings["simulate_s"] = time.perf_counter() - t0

        t = time.perf_counter()
        res = search(df, self.train, Grid(), passes=cfg.passes, n_folds=4, window=13)
        self.base = fit_mmm(df, res.hyper, res.lam, self.train)
        self.base.meta = {"search_score": res.score, "evaluations": res.evaluations}
        timings["search_s"] = time.perf_counter() - t

        t = time.perf_counter()
        self.lift = simulate_geo_test(self.world, cfg.lift_channel)
        self.prior_fn = lift_prior_fn(df, self.lift, self.base.sigma)
        res_c = search(df, self.train, Grid(), passes=1, n_folds=4, window=13, priors_fn=self.prior_fn,
                       start=res.hyper, lam=res.lam)
        self.calibrated = fit_mmm(df, res_c.hyper, res_c.lam, self.train, priors_fn=self.prior_fn)
        self.calibrated.meta = {"search_score": res_c.score, "evaluations": res_c.evaluations}
        timings["calibrate_s"] = time.perf_counter() - t

        t = time.perf_counter()
        self.draws["base"] = bootstrap(self.base, df, cfg.n_boot, cfg.block, seed=cfg.seed)
        self.draws["calibrated"] = bootstrap(self.calibrated, df, cfg.n_boot, cfg.block, seed=cfg.seed,
                                             priors_fn=self.prior_fn)
        timings["bootstrap_s"] = time.perf_counter() - t

        last_q = np.arange(n - QUARTER_WEEKS, n)
        self.current = df[list(CHANNELS)].to_numpy()[last_q].mean(axis=0)
        t = time.perf_counter()
        self.plan = self.optimize_budget(self.current.sum() * QUARTER_WEEKS)
        self.plan_base = self.optimize_budget(self.current.sum() * QUARTER_WEEKS, name="base")
        timings["optimize_s"] = time.perf_counter() - t
        timings["total_s"] = time.perf_counter() - t0
        self.report = self._report({k: round(v, 2) for k, v in timings.items()})
        return self

    # ---- helpers -------------------------------------------------------------------------------------------

    def model(self, name: str = "calibrated") -> FittedMMM:
        if name not in ("base", "calibrated"):
            raise ValueError("model must be base or calibrated")
        return self.base if name == "base" else self.calibrated

    def curves(self, m: FittedMMM, beta: dict | None = None) -> list[Curve]:
        beta = beta or m.beta
        return [Curve(beta[c], steady_state_multiplier(m.hyper[c].adstock) / m.scales[c], m.hyper[c].hill.half_sat,
                      m.hyper[c].hill.shape) for c in m.channels]

    def true_curves(self) -> list[Curve]:
        tr = self.world.truth
        return [Curve(tr[c]["beta"], steady_state_multiplier(tr[c]["adstock"]) / tr[c]["scale"],
                      tr[c]["hill"].half_sat, tr[c]["hill"].shape) for c in CHANNELS]

    def default_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return 0.5 * self.current, 2.0 * self.current

    def evaluate(self, weekly: np.ndarray, name: str = "calibrated") -> dict:
        """Predicted weekly media revenue for an allocation: point, 90% range over bootstrap draws, and truth."""
        weekly = np.asarray(weekly, dtype=float)
        m = self.model(name)
        point = total_revenue(self.curves(m), weekly)
        sims = np.array([total_revenue(self.curves(m, d.beta), weekly) for d in self.draws[name]])
        truth = total_revenue(self.true_curves(), weekly)
        spend = float(weekly.sum())
        lo, hi = interval(sims)
        q = QUARTER_WEEKS
        return {
            "weekly_spend": {c: round(float(v), 2) for c, v in zip(CHANNELS, weekly, strict=True)},
            "quarter_spend": round(spend * q, 2),
            "quarter_media_revenue": round(point * q, 2),
            "quarter_media_revenue_ci90": [round(lo * q, 2), round(hi * q, 2)],
            "quarter_true_media_revenue": round(truth * q, 2),
            "roi": round(point / spend, 4) if spend else 0.0,
            "roi_ci90": [round(lo / spend, 4), round(hi / spend, 4)] if spend else [0.0, 0.0],
            "true_roi": round(truth / spend, 4) if spend else 0.0,
        }

    def optimize_budget(self, quarter_budget: float, lo=None, hi=None, name: str = "calibrated",
                        uncertainty: bool = True) -> dict:
        weekly_budget = quarter_budget / QUARTER_WEEKS
        dlo, dhi = self.default_bounds()
        lo = dlo if lo is None else np.asarray(lo, dtype=float)
        hi = dhi if hi is None else np.asarray(hi, dtype=float)
        m = self.model(name)
        best = optimize(self.curves(m), weekly_budget, lo, hi, start=self.current)
        current_scaled = self.current * weekly_budget / self.current.sum()
        out = {
            "quarter_budget": round(quarter_budget, 2),
            "model": name,
            "solver": best["solver"],
            "bounds": {c: [round(float(a), 2), round(float(b), 2)] for c, a, b in zip(CHANNELS, lo, hi, strict=True)},
            "current": self.evaluate(current_scaled, name),
            "optimized": self.evaluate(best["x"], name),
        }
        cur, opt = out["current"], out["optimized"]
        out["lift_pct"] = round(100 * (opt["quarter_media_revenue"] / cur["quarter_media_revenue"] - 1), 2)
        true_ratio = opt["quarter_true_media_revenue"] / cur["quarter_true_media_revenue"]
        out["true_lift_pct"] = round(100 * (true_ratio - 1), 2)
        if uncertainty:
            draws = self.draws[name][: self.config.n_boot_opt]
            allocs, gains = [], []
            for d in draws:
                cv = self.curves(m, d.beta)
                r = optimize(cv, weekly_budget, lo, hi, start=best["x"], steps=400, starts=3)
                allocs.append(r["x"])
                gains.append(total_revenue(cv, best["x"]) / max(total_revenue(cv, current_scaled), 1e-9) - 1)
            allocs = np.array(allocs) * QUARTER_WEEKS
            out["allocation_ci90"] = {c: [round(float(v), 2) for v in interval(allocs[:, j])]
                                      for j, c in enumerate(CHANNELS)}
            out["lift_pct_ci90"] = [round(100 * v, 2) for v in interval(gains)]
            out["prob_beats_current"] = round(float(np.mean(np.array(gains) > 0)), 3)
        return out

    def scenarios(self, user_quarter: dict[str, float] | None = None, name: str = "calibrated") -> dict:
        rows = {"current": self.current, "optimized": self.plan["optimized"]["weekly_spend"]}
        rows["optimized"] = np.array([rows["optimized"][c] for c in CHANNELS])
        if user_quarter:
            unknown = set(user_quarter) - set(CHANNELS)
            if unknown:
                raise ValueError(f"unknown channels: {sorted(unknown)}")
            rows["user"] = np.array([float(user_quarter.get(c, self.current[i] * QUARTER_WEEKS)) / QUARTER_WEEKS
                                     for i, c in enumerate(CHANNELS)])
        return {k: self.evaluate(v, name) for k, v in rows.items()}

    def response_curves(self, name: str = "calibrated", points: int = 41) -> dict:
        m = self.model(name)
        df = self.world.data
        out = {}
        for j, c in enumerate(CHANNELS):
            cur = float(self.current[j])
            top = max(cur * 3, float(df[c].max()) * 1.2, 1.0)
            x = np.linspace(0, top, points)
            sims = np.array([d.response(c, x) for d in self.draws[name]])
            lo, hi = np.quantile(sims, [0.05, 0.95], axis=0)
            out[c] = {
                "label": LABELS[c], "spend": x.round(2).tolist(), "revenue": m.response(c, x).round(2).tolist(),
                "revenue_lo": lo.round(2).tolist(), "revenue_hi": hi.round(2).tolist(),
                "true_revenue": self.world.true_response(c, x).round(2).tolist(),
                "current_spend": round(cur, 2), "current_revenue": round(float(m.response(c, cur)), 2),
                "marginal_roi_at_current": round(float(m.marginal_roi(c, cur)), 4),
                "average_roi_at_current": round(float(m.response(c, cur)) / cur, 4) if cur > 0 else 0.0,
                "marginal_roi": m.marginal_roi(c, x).round(4).tolist(),
            }
        return out

    def decomposition(self, name: str = "calibrated") -> dict:
        m = self.model(name)
        df = self.world.data
        parts = m.contributions(df)
        return {"week": df["week"].tolist(), "actual": df["sales"].round(2).tolist(),
                "fitted": m.predict(df).round(2).tolist(), "holdout_start": int(self.hold[0]),
                "series": {k: np.asarray(v).round(2).tolist() for k, v in parts.items()},
                "true_series": {"baseline": self.world.baseline.round(2).tolist(),
                                **{c: self.world.truth[c]["contribution"].round(2).tolist() for c in CHANNELS}}}

    # ---- report --------------------------------------------------------------------------------------------

    def _roi_table(self) -> list[dict]:
        df = self.world.data
        truth = self.world.true_roi(self.train)
        rows = []
        point = {k: self.model(k).roi(df, self.train) for k in ("base", "calibrated")}
        draws = {k: roi_draws(self.draws[k], df, self.train) for k in ("base", "calibrated")}
        for c in CHANNELS:
            r = {"channel": c, "label": LABELS[c], "true_roi": round(truth[c], 3),
                 "spend": round(float(df[c].to_numpy()[self.train].sum()), 2),
                 "challenge": self.world.truth[c]["challenge"]}
            for k in ("base", "calibrated"):
                lo, hi = interval(draws[k][c])
                err = point[k][c] / truth[c] - 1
                r[k] = {"roi": round(point[k][c], 3), "ci90": [round(lo, 3), round(hi, 3)],
                        "error_pct": round(100 * err, 1), "covers_truth": bool(lo <= truth[c] <= hi),
                        "recovered": bool(abs(err) <= ROI_TOLERANCE)}
            rows.append(r)
        return rows

    def pair_roi(self, name: str) -> tuple[float, float]:
        df = self.world.data
        s = {c: df[c].to_numpy()[self.train].sum() for c in GATE_PAIR}
        est, tru = self.model(name).roi(df, self.train), self.world.true_roi(self.train)
        tot = sum(s.values())
        return (sum(est[c] * s[c] for c in GATE_PAIR) / tot, sum(tru[c] * s[c] for c in GATE_PAIR) / tot)

    def gate(self, name: str = "calibrated") -> tuple[bool, list[dict]]:
        roi = {r["channel"]: r for r in self.report["roi"]}
        checks = []
        for c in GATE_CHANNELS:
            est, tru = roi[c][name]["roi"], roi[c]["true_roi"]
            checks.append({"check": LABELS[c], "recovered": est, "true": tru})
        est, tru = self.pair_roi(name)
        checks.append({"check": "Paid social + display combined", "recovered": round(est, 3), "true": round(tru, 3)})
        for ch in checks:
            ch["error_pct"] = round(100 * (ch["recovered"] / ch["true"] - 1), 1)
            ch["pass"] = bool(abs(ch["error_pct"]) <= 100 * GATE_TOLERANCE)
        return all(ch["pass"] for ch in checks), checks

    def _report(self, timings: dict) -> dict:
        df = self.world.data
        y = df["sales"].to_numpy()
        fit = {}
        for k in ("base", "calibrated"):
            m = self.model(k)
            pred = m.predict(df)
            fit[k] = {"holdout_mape": round(mape(y[self.hold], pred[self.hold]), 4),
                      "train_mape": round(mape(y[self.train], pred[self.train]), 4), "lambda": m.lam,
                      "hyper": {c: m.hyper[c].to_dict() for c in CHANNELS},
                      "beta": {c: round(v, 2) for c, v in m.beta.items()},
                      "evaluations": m.meta.get("evaluations"), "prior": [
                          {"col": p.col, "mean": round(p.mean, 5), "weight": round(p.weight, 3)} for p in m.priors]}
        truth_h = {c: {"adstock": self.world.truth[c]["adstock"].to_dict(),
                       "adstock_label": self.world.truth[c]["adstock"].label(),
                       "hill": self.world.truth[c]["hill"].to_dict()} for c in CHANNELS}
        self.report = {
            "brand": "Juniper Row (made up)",
            "weeks": len(df), "train_weeks": len(self.train), "holdout_weeks": len(self.hold),
            "first_week": df["week"].iloc[0], "last_week": df["week"].iloc[-1],
            "channels": [{"channel": c, "label": LABELS[c]} for c in CHANNELS],
            "fit": fit, "truth_hyper": truth_h, "roi": self._roi_table(), "lift_test": self.lift.to_dict(),
            "bootstrap": {"draws": self.config.n_boot, "block_weeks": self.config.block},
            "current_weekly": {c: round(float(v), 2) for c, v in zip(CHANNELS, self.current, strict=True)},
            "plan": self.plan, "timings": timings,
            "plan_base": {k: self.plan_base[k] for k in ("lift_pct", "true_lift_pct", "lift_pct_ci90", "optimized")},
            "bayesian": "Skipped. The calibrated ridge is the posterior mode of a Gaussian model with a Normal "
                        "prior on the lift tested coefficient; a sampler would add hyperparameter uncertainty.",
        }
        ok, checks = self.gate()
        self.report["gate"] = {"pass": ok, "tolerance_pct": 100 * GATE_TOLERANCE, "checks": checks}
        return self.report
