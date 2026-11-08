import numpy as np

from mediamix.model import fit_mmm
from mediamix.search import Grid, rolling_folds, search
from tests.conftest import clean_world


def test_rolling_folds_are_ordered_and_disjoint():
    rows = np.arange(130)
    folds = rolling_folds(rows, 4, 13)
    assert len(folds) == 4
    for fit, val in folds:
        assert fit.max() < val.min() and len(val) == 13
    assert folds[-1][1][-1] == 129


def test_default_grid_contains_world_truth():
    from mediamix.world import TRUTH

    g = Grid()
    for ct in TRUTH:
        assert ct.adstock in g.adstocks and ct.hill in g.hills


def test_search_recovers_known_parameters_on_clean_case():
    df, params = clean_world()
    rows = np.arange(130)
    res = search(df, rows, channels=("a", "b"), passes=2, n_folds=2, window=26)
    for c, (ad, h, _) in params.items():
        assert res.hyper[c].adstock == ad, c
        assert res.hyper[c].hill == h, c


def test_search_score_improves_or_holds():
    df, _ = clean_world(seed=1)
    res = search(df, np.arange(130), channels=("a", "b"), passes=1, n_folds=2, window=26)
    scores = [h["score"] for h in res.history]
    assert all(b <= a + 1e-15 for a, b in zip(scores, scores[1:], strict=False))
    assert res.evaluations > 0


def test_clean_case_recovers_coefficients():
    df, params = clean_world()
    rows = np.arange(130)
    res = search(df, rows, channels=("a", "b"), passes=2, n_folds=2, window=26)
    m = fit_mmm(df, res.hyper, 0.0, rows, channels=("a", "b"))
    for c, (_, _, beta) in params.items():
        assert abs(m.beta[c] / beta - 1) < 0.05
