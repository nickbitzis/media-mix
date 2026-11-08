import numpy as np
import pytest

from mediamix.transforms import (
    Adstock,
    Hill,
    adstock,
    channel_scale,
    geometric_adstock,
    hill,
    hill_derivative,
    steady_state_multiplier,
    transform,
    weibull_adstock,
    weibull_weights,
)


def test_geometric_adstock_hand_values():
    # a0 = 100, a1 = 0 + 0.5*100 = 50, a2 = 20 + 0.5*50 = 45, a3 = 0 + 22.5
    assert np.allclose(geometric_adstock([100, 0, 20, 0], 0.5), [100, 50, 45, 22.5])


def test_geometric_zero_decay_is_identity():
    x = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    assert np.allclose(geometric_adstock(x, 0.0), x)


def test_geometric_steady_state():
    a = geometric_adstock(np.ones(200), 0.6)
    assert a[-1] == pytest.approx(1 / (1 - 0.6))
    assert steady_state_multiplier(Adstock("geometric", decay=0.6)) == pytest.approx(2.5)


def test_weibull_weights_hand_values():
    # shape 1, scale 1: survival e^-l, mass e^-l - e^-(l+1); with 2 lags normalized by 1 - e^-2
    w = weibull_weights(1.0, 1.0, lags=2)
    raw = np.array([1 - np.exp(-1), np.exp(-1) - np.exp(-2)])
    assert np.allclose(w, raw / raw.sum())
    assert w.sum() == pytest.approx(1.0)


def test_weibull_shape_above_one_delays_peak():
    w = weibull_weights(2.0, 3.0)
    assert int(np.argmax(w)) >= 1
    assert np.all(w >= 0)


def test_weibull_adstock_spreads_but_keeps_total():
    x = np.zeros(40)
    x[5] = 100.0
    a = weibull_adstock(x, 2.0, 3.0)
    assert a[:5].sum() == 0
    assert a.sum() == pytest.approx(100.0)
    assert steady_state_multiplier(Adstock("weibull", shape=2, scale=3)) == 1.0


def test_adstock_dispatch_and_unknown_kind():
    x = np.array([1.0, 2.0, 3.0])
    assert np.allclose(adstock(x, Adstock("geometric", decay=0.0)), x)
    with pytest.raises(ValueError):
        adstock(x, Adstock("bogus"))


@pytest.mark.parametrize("z,k,s,expected", [(1.0, 1.0, 1.0, 0.5), (2.0, 1.0, 1.0, 2 / 3), (2.0, 1.0, 2.0, 0.8),
                                             (0.5, 1.0, 2.0, 0.2), (0.0, 1.0, 2.0, 0.0), (3.0, 3.0, 3.0, 0.5)])
def test_hill_hand_values(z, k, s, expected):
    assert float(hill(z, k, s)) == pytest.approx(expected)


def test_hill_is_bounded_and_increasing():
    z = np.linspace(0, 20, 200)
    h = hill(z, 1.5, 2.0)
    assert np.all(np.diff(h) >= 0) and h.max() < 1 and h.min() == 0


@pytest.mark.parametrize("k,s", [(1.0, 1.0), (0.5, 2.0), (2.0, 3.0)])
def test_hill_derivative_matches_finite_difference(k, s):
    z = np.array([0.3, 1.0, 2.5])
    eps = 1e-6
    fd = (hill(z + eps, k, s) - hill(z - eps, k, s)) / (2 * eps)
    assert np.allclose(hill_derivative(z, k, s), fd, rtol=1e-4)


def test_channel_scale_ignores_off_weeks():
    x = np.array([0, 0, 100, 300, 0])
    assert channel_scale(x, Adstock("geometric", decay=0.5)) == pytest.approx(200 * 2)


def test_transform_chains_adstock_then_hill():
    x = np.array([100.0, 0.0, 0.0])
    out = transform(x, Adstock("geometric", decay=0.5), Hill(1.0, 1.0), 100.0)
    # adstock 100, 50, 25 -> z 1, .5, .25 -> hill z/(1+z)
    assert np.allclose(out, [0.5, 1 / 3, 0.2])
