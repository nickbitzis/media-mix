import numpy as np
import pytest

from mediamix.world import CHANNELS, TRUTH, WorldConfig, simulate


@pytest.fixture(scope="module")
def world():
    return simulate(WorldConfig(seed=3))


def test_seeded_and_deterministic(world):
    again = simulate(WorldConfig(seed=3))
    assert np.allclose(again.data["sales"], world.data["sales"])
    assert not np.allclose(simulate(WorldConfig(seed=4)).data["sales"], world.data["sales"])


def test_shape_and_columns(world):
    assert len(world.data) == 156
    for c in CHANNELS + ("sales", "price_index", "promo", "competitor_index", "week"):
        assert c in world.data


def test_true_roi_matches_configured_roi(world):
    roi = world.true_roi()
    for ct in TRUTH:
        assert roi[ct.name] == pytest.approx(ct.roi, rel=1e-9)


def test_sales_is_baseline_plus_media_plus_noise(world):
    media = sum(world.truth[c]["contribution"] for c in CHANNELS)
    resid = world.data["sales"].to_numpy() - world.baseline - media
    assert abs(resid.mean()) < 5_000 and resid.std() == pytest.approx(world.config.noise_sd, rel=0.2)


def test_designed_challenges_are_present(world):
    d = world.data
    assert np.corrcoef(d["social"], d["display"])[0, 1] > 0.5  # correlated
    assert d["tv"][world.extra["peak"]].sum() / d["tv"].sum() > 0.8  # mostly peak season
    assert d["podcast"].std() / d["podcast"].mean() < 0.05  # almost no variation
    assert (d["search"] > 0).all()  # always on
    assert np.corrcoef(d["search"], world.extra["demand"])[0, 1] > 0.8  # follows demand
    assert (d["email"] == 0).mean() > 0.3 and (d["influencer"] == 0).mean() > 0.3  # on and off weeks


def test_true_response_is_increasing_and_zero_at_zero(world):
    x = np.linspace(0, 300_000, 50)
    for c in CHANNELS:
        r = world.true_response(c, x)
        assert r[0] == 0 and np.all(np.diff(r) >= 0)
