import numpy as np
import pytest

from mediamix.bootstrap import block_indices, bootstrap, interval, roi_draws


def test_block_indices_length_and_membership():
    rows = np.arange(10, 140)
    idx = block_indices(rows, 8, np.random.default_rng(0))
    assert len(idx) == len(rows) and set(idx) <= set(rows)


def test_blocks_are_contiguous_runs():
    rows = np.arange(130)
    idx = block_indices(rows, 8, np.random.default_rng(1))
    for k in range(0, 128, 8):
        blk = idx[k : k + 8]
        assert np.all(np.diff(blk) == 1)


def test_bootstrap_is_reproducible(study):
    a = bootstrap(study.base, study.world.data, 3, seed=5)
    b = bootstrap(study.base, study.world.data, 3, seed=5)
    assert all(np.allclose(x.coef, y.coef) for x, y in zip(a, b, strict=True))


def test_draws_vary_and_keep_hyperparameters(study):
    draws = study.draws["base"]
    betas = np.array([[d.beta[c] for c in d.channels] for d in draws])
    assert betas.std(axis=0).max() > 0
    assert all(d.hyper == study.base.hyper for d in draws)


def test_interval_quantiles():
    lo, hi = interval(np.arange(101), 0.9)
    assert lo == pytest.approx(5) and hi == pytest.approx(95)


def test_hard_channels_have_wider_intervals_than_well_identified_ones(study):
    r = roi_draws(study.draws["base"], study.world.data, study.train)
    width = {c: (np.quantile(v, 0.95) - np.quantile(v, 0.05)) for c, v in r.items()}
    true = study.world.true_roi(study.train)
    rel = {c: width[c] / true[c] for c in width}
    assert max(rel[c] for c in ("display", "search")) > max(rel[c] for c in ("influencer", "email"))
