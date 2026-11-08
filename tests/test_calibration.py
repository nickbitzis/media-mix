import numpy as np
import pytest

from mediamix.calibration import lift_prior_fn, simulate_geo_test
from mediamix.model import fit_mmm


def test_lift_test_estimate_is_near_truth(study):
    lt = study.lift
    assert lt.channel == "search"
    assert abs(lt.iroas - lt.true_iroas) < 3 * lt.se
    assert 0 < lt.se < 0.5 * lt.true_iroas


def test_lift_test_is_seeded(study):
    a = simulate_geo_test(study.world, "search")
    assert a.iroas == pytest.approx(study.lift.iroas)
    assert simulate_geo_test(study.world, "search", seed=1).iroas != pytest.approx(a.iroas)


def test_strong_prior_pins_model_iroas_to_the_experiment(study):
    df, lt = study.world.data, study.lift
    m0 = study.base
    fn = lift_prior_fn(df, lt, m0.sigma, strength=1e4)
    m = fit_mmm(df, m0.hyper, m0.lam, study.train, priors_fn=fn)
    H = m.design(df)[:, 1 + m.n_ctrl :]
    (prior,) = fn(m, H)
    j = m.channels.index("search")
    assert m.coef[1 + m.n_ctrl + j] == pytest.approx(prior.mean, rel=1e-3)


def test_calibration_moves_search_toward_truth(study):
    rows = {r["channel"]: r for r in study.report["roi"]}
    s = rows["search"]
    assert abs(s["calibrated"]["error_pct"]) < abs(s["base"]["error_pct"])
    assert s["base"]["roi"] > s["true_roi"]  # uncalibrated model over-credits always-on search


def test_calibrated_plan_does_better_in_truth(study):
    assert study.report["plan"]["true_lift_pct"] > study.report["plan_base"]["true_lift_pct"]


def test_calibration_prior_has_positive_weight(study):
    (p,) = study.calibrated.priors
    assert p.weight > 0 and np.isfinite(p.mean)
