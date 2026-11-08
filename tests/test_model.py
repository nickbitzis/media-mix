import numpy as np
import pytest

from mediamix.model import ChannelHyper, fit_mmm, mape
from mediamix.world import CHANNELS


def test_decomposition_sums_to_fitted(study):
    df = study.world.data
    for m in (study.base, study.calibrated):
        parts = m.contributions(df)
        assert np.allclose(sum(parts.values()), m.predict(df))


def test_media_contributions_are_nonnegative(study):
    parts = study.calibrated.contributions(study.world.data)
    for c in CHANNELS:
        assert parts[c].min() >= -1e-9


def test_roi_is_contribution_over_spend(study):
    df, m = study.world.data, study.calibrated
    parts = m.contributions(df)
    roi = m.roi(df, study.train)
    assert roi["tv"] == pytest.approx(parts["tv"][study.train].sum() / df["tv"][study.train].sum())


def test_response_curve_zero_at_zero_and_increasing(study):
    m = study.calibrated
    x = np.linspace(0, 400_000, 60)
    for c in CHANNELS:
        r = m.response(c, x)
        assert r[0] == 0 and np.all(np.diff(r) >= -1e-9)


def test_marginal_roi_matches_finite_difference(study):
    m = study.calibrated
    for c in CHANNELS:
        x = 0.8 * float(study.current[CHANNELS.index(c)]) + 1000
        eps = 1.0
        fd = (m.response(c, x + eps) - m.response(c, x - eps)) / (2 * eps)
        assert float(m.marginal_roi(c, x)) == pytest.approx(float(fd), rel=1e-3, abs=1e-9)


def test_true_parameters_fit_well_on_clean_case():
    from tests.conftest import clean_world

    df, params = clean_world()
    hy = {c: ChannelHyper(a, h) for c, (a, h, _) in params.items()}
    m = fit_mmm(df, hy, 0.0, np.arange(156), channels=("a", "b"))
    assert mape(df["sales"], m.predict(df)) < 0.005


def test_holdout_mape_is_small(study):
    assert study.report["fit"]["calibrated"]["holdout_mape"] < 0.03
    assert mape(np.array([100.0, 200.0]), np.array([110.0, 180.0])) == pytest.approx(0.1)
