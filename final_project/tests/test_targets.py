"""Monte Carlo return targets: correctness and right-edge truncation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from project.targets import GAMMA_DEFAULT, H_DEFAULT, mc_returns


def _const_rewards(n: int, value: float) -> pd.Series:
    return pd.Series(
        np.full(n, value), index=pd.date_range("2000-01-01", periods=n, freq="B")
    )


def test_mc_return_matches_definition_constant_rewards() -> None:
    """For R_t = c constant, y_t = c * sum_{k=0}^H gamma^k everywhere."""
    n = 200
    H = 60
    gamma = 0.95
    c = 0.001
    R = _const_rewards(n, c)

    y = mc_returns(R, gamma=gamma, H=H)
    expected = c * (1.0 - gamma ** (H + 1)) / (1.0 - gamma)

    assert len(y) == n - H
    np.testing.assert_allclose(y.to_numpy(), expected, rtol=1e-12)


def test_mc_return_matches_definition_random_rewards() -> None:
    """Match the slow but explicit reference implementation row-for-row."""
    rng = np.random.default_rng(0)
    n = 300
    H = 20
    gamma = 0.9
    R = pd.Series(
        rng.normal(0.0, 0.01, size=n),
        index=pd.date_range("2000-01-01", periods=n, freq="B"),
    )

    y = mc_returns(R, gamma=gamma, H=H)

    expected = []
    for t in range(n - H):
        s = 0.0
        for k in range(H + 1):
            s += (gamma ** k) * R.iloc[t + k]
        expected.append(s)
    np.testing.assert_allclose(y.to_numpy(), np.array(expected), rtol=1e-12)


def test_no_right_edge_leakage() -> None:
    """y_t is dropped whenever the window [t, t+H] runs past the data."""
    n = 100
    H = 60
    R = _const_rewards(n, 0.0)
    R.iloc[-1] = np.nan  # mimic step_reward's last-day NaN
    y = mc_returns(R, H=H)
    # Largest valid t has t + H = n - 2, so length is n - H - 1.
    assert len(y) == n - H - 1


def test_left_edge_nan_dropped() -> None:
    """Leading NaN (e.g., from cost warmup) propagates into dropped y_t rows."""
    n = 200
    H = 60
    R = _const_rewards(n, 0.001)
    R.iloc[:5] = np.nan  # leading warmup NaN
    y = mc_returns(R, H=H)
    # The first 5 windows touch a NaN, so they're dropped; afterward y_t is fine.
    assert R.index[5] == y.index[0]


def test_horizon_too_long_raises() -> None:
    R = _const_rewards(10, 0.0)
    with pytest.raises(ValueError):
        mc_returns(R, H=20)


def test_default_constants_match_proposal() -> None:
    assert GAMMA_DEFAULT == 0.95
    assert H_DEFAULT == 60
