import numpy as np
import pandas as pd
import pytest

from mediamix.study import Study, StudyConfig
from mediamix.transforms import Adstock, Hill, channel_scale, transform

SMALL = StudyConfig(seed=3, n_boot=40, n_boot_opt=20)


@pytest.fixture(scope="session")
def study():
    return Study(SMALL).run()


def clean_world(n=156, seed=0, noise=2_000.0):
    """Two channels with on/off spend, no confounding, tiny noise, known parameters that sit on the search grid."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    a = np.where(rng.random(n) < 0.6, rng.uniform(20_000, 60_000, n), 0.0)
    b = np.where(rng.random(n) < 0.5, rng.uniform(5_000, 25_000, n), 0.0)
    params = {"a": (Adstock("geometric", decay=0.5), Hill(1.0, 2.0), 150_000.0),
              "b": (Adstock("geometric", decay=0.0), Hill(0.5, 1.0), 60_000.0)}
    df = pd.DataFrame({"t": t, "woy": t % 52, "price_index": 1 + rng.normal(0, 0.02, n),
                       "promo": (rng.random(n) < 0.1).astype(float), "competitor_index": 1 + rng.normal(0, 0.1, n),
                       "a": a, "b": b})
    sales = 1_000_000 + 30_000 * np.sin(2 * np.pi * t / 52)
    for c, (ad, h, beta) in params.items():
        sales = sales + beta * transform(df[c].to_numpy(), ad, h, channel_scale(df[c].to_numpy(), ad))
    df["sales"] = sales + rng.normal(0, noise, n)
    return df, params
