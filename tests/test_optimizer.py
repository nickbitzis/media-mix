import numpy as np
import pytest

from mediamix import optimizer as opt
from mediamix.optimizer import Curve, check_bounds, greedy, optimize, total_revenue


def curves():
    # strong, concave channel; weaker concave channel; S-shaped channel
    return [Curve(100_000, 1 / 20_000, 1.0, 1.0), Curve(40_000, 1 / 10_000, 1.0, 1.0),
            Curve(80_000, 1 / 30_000, 1.0, 2.0)]


def test_respects_budget_and_bounds():
    lo, hi = np.array([5_000, 2_000, 0]), np.array([40_000, 10_000, 50_000])
    r = optimize(curves(), 60_000, lo, hi)
    assert r["x"].sum() == pytest.approx(60_000, rel=1e-6)
    assert np.all(r["x"] >= lo - 1e-6) and np.all(r["x"] <= hi + 1e-6)


def test_beats_equal_split_on_known_case():
    cv = curves()
    budget = 60_000
    r = optimize(cv, budget, np.zeros(3), np.full(3, budget))
    assert r["revenue"] > total_revenue(cv, np.full(3, budget / 3)) * 1.02


def test_identical_concave_curves_split_equally():
    c = Curve(50_000, 1 / 10_000, 1.0, 1.0)
    r = optimize([c, c], 20_000, np.zeros(2), np.full(2, 20_000))
    assert np.allclose(r["x"], [10_000, 10_000], rtol=1e-3)


def test_marginal_rois_equalize_at_interior_optimum():
    cv = curves()[:2]
    r = optimize(cv, 30_000, np.zeros(2), np.full(2, 30_000))
    m = [float(c.marginal(v)) for c, v in zip(cv, r["x"], strict=True)]
    assert m[0] == pytest.approx(m[1], rel=0.01)


def test_greedy_is_near_optimal_on_concave_curves():
    cv = curves()[:2]
    lo, hi = np.zeros(2), np.full(2, 60_000)
    g = greedy(cv, 60_000, lo, hi)
    assert g.sum() == pytest.approx(60_000)
    best = optimize(cv, 60_000, lo, hi)
    assert total_revenue(cv, g) <= best["revenue"] + 1e-6
    assert total_revenue(cv, g) >= 0.999 * best["revenue"]


def test_greedy_stalls_on_s_curves_so_slsqp_is_kept():
    # The S-shaped channel returns almost nothing for the first small step, so greedy never starts it.
    cv = curves()
    g = greedy(cv, 60_000, np.zeros(3), np.full(3, 60_000))
    best = optimize(cv, 60_000, np.zeros(3), np.full(3, 60_000))
    assert g[2] == 0 and best["x"][2] > 0 and best["revenue"] > total_revenue(cv, g)


def test_greedy_fallback_without_scipy(monkeypatch):
    monkeypatch.setattr(opt, "minimize", None)
    r = optimize(curves(), 60_000, np.zeros(3), np.full(3, 60_000))
    assert r["solver"] == "greedy" and r["x"].sum() == pytest.approx(60_000)


@pytest.mark.parametrize("lo,hi,budget", [([30_000, 30_000, 30_000], [50_000] * 3, 60_000),
                                          ([0, 0, 0], [10_000] * 3, 60_000), ([0, 10, 0], [5, 5, 5], 10)])
def test_infeasible_bounds_raise(lo, hi, budget):
    with pytest.raises(ValueError):
        check_bounds(budget, np.array(lo, float), np.array(hi, float))


def test_pinned_bounds_are_honored():
    lo = hi = np.array([10_000.0, 20_000.0, 30_000.0])
    r = optimize(curves(), 60_000, lo, hi)
    assert np.allclose(r["x"], lo)
