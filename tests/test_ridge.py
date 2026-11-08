import numpy as np
import pytest

from mediamix.ridge import Prior, fit_constrained_ridge


def data(seed=0, n=200):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=(n, 3))])
    beta = np.array([2.0, 1.0, -0.5, 0.8])
    return X, X @ beta + rng.normal(0, 0.05, n), beta


def masks(p, nonneg_cols=()):
    nn = np.zeros(p, bool)
    nn[list(nonneg_cols)] = True
    pen = np.ones(p, bool)
    pen[0] = False
    return nn, pen


def test_matches_least_squares_without_penalty_or_bounds():
    X, y, _ = data()
    nn, pen = masks(4)
    b = fit_constrained_ridge(X, y, 0.0, nn, pen)
    assert np.allclose(b, np.linalg.lstsq(X, y, rcond=None)[0])


def test_nonnegative_columns_stay_nonnegative():
    X, y, _ = data()
    nn, pen = masks(4, (2,))  # true coefficient on column 2 is -0.5
    b = fit_constrained_ridge(X, y, 0.0, nn, pen)
    assert b[2] == pytest.approx(0.0, abs=1e-9)
    assert b[1] > 0.5


def test_unconstrained_recovers_truth():
    X, y, beta = data()
    nn, pen = masks(4, (1, 3))
    b = fit_constrained_ridge(X, y, 0.0, nn, pen)
    assert np.allclose(b, beta, atol=0.02)


def test_ridge_shrinks_penalized_but_not_intercept():
    X, y, _ = data()
    nn, pen = masks(4, (1, 3))
    b0 = fit_constrained_ridge(X, y, 0.0, nn, pen)
    b1 = fit_constrained_ridge(X, y, 500.0, nn, pen)
    assert np.all(np.abs(b1[1:]) < np.abs(b0[1:]))
    assert b1[0] == pytest.approx(b0[0], abs=0.1)


def test_ridge_matches_closed_form():
    X, y, _ = data()
    nn, pen = masks(4)
    lam = 3.0
    P = np.diag([0, lam, lam, lam])
    closed = np.linalg.solve(X.T @ X + P, X.T @ y)
    assert np.allclose(fit_constrained_ridge(X, y, lam, nn, pen), closed)


def test_prior_pulls_toward_mean():
    X, y, _ = data()
    nn, pen = masks(4, (1,))
    free = fit_constrained_ridge(X, y, 0.0, nn, pen)
    weak = fit_constrained_ridge(X, y, 0.0, nn, pen, (Prior(1, 3.0, 1.0),))
    strong = fit_constrained_ridge(X, y, 0.0, nn, pen, (Prior(1, 3.0, 1e4),))
    assert free[1] < weak[1] < strong[1]
    assert strong[1] == pytest.approx(3.0, abs=1e-3)
