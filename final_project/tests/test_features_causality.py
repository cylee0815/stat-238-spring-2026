"""Leakage smoke test: every feature at time t depends on data at time <= t only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from project.features import build_features, feature_warmup


def _synthetic_prices(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_p = np.cumsum(rng.normal(0.0, 0.01, size=n))
    p = np.exp(log_p) * 100.0
    idx = pd.date_range("2000-01-01", periods=n, freq="B")
    return pd.DataFrame({"Close": p}, index=idx)


def test_no_lookahead_under_future_scrambling() -> None:
    """Scrambling prices at indices >= k must leave features at indices < k untouched."""
    n = 800
    k = 500
    prices = _synthetic_prices(n)

    feat_truth = build_features(prices)

    rng = np.random.default_rng(99)
    perm = rng.permutation(n - k)
    prices_scrambled = prices.copy()
    prices_scrambled.iloc[k:] = prices.iloc[k:].to_numpy()[perm]
    feat_scrambled = build_features(prices_scrambled)

    pd.testing.assert_frame_equal(feat_truth.iloc[:k], feat_scrambled.iloc[:k])


def test_warmup_is_finite_and_features_eventually_clean() -> None:
    """After feature_warmup() rows, every feature should be finite."""
    n = 800
    prices = _synthetic_prices(n)
    feat = build_features(prices)
    warmup = feature_warmup()
    tail = feat.iloc[warmup:]
    assert tail.notna().all().all(), (
        f"NaN remains after warmup={warmup}; first NaN locations:\n"
        f"{tail.isna().any(axis=1).idxmax()}"
    )


def test_intercept_is_constant_one() -> None:
    n = 400
    prices = _synthetic_prices(n)
    feat = build_features(prices)
    assert (feat["x4_intercept"] == 1.0).all()
