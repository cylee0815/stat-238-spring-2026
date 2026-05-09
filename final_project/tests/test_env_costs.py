"""Volatility-dependent transaction-cost accounting in env.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from project.env import (
    C0_DEFAULT,
    LAMBDA_DEFAULT,
    VOL_WINDOW_DEFAULT,
    step_reward,
    transaction_cost,
)


def _synthetic_prices(n: int = 200, sigma: float = 0.01, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_p = np.cumsum(rng.normal(0.0, sigma, size=n))
    p = np.exp(log_p) * 100.0
    idx = pd.date_range("2000-01-01", periods=n, freq="B")
    return pd.DataFrame({"Close": p}, index=idx)


def test_transaction_cost_constant_prices_equals_c0() -> None:
    """Constant prices => sigma_t = 0 => c_t = c_0 once the rolling window fills."""
    n = 100
    p = pd.Series(np.full(n, 100.0), index=pd.date_range("2000-01-01", periods=n, freq="B"))
    prices = pd.DataFrame({"Close": p})
    c = transaction_cost(prices)
    np.testing.assert_allclose(c.iloc[VOL_WINDOW_DEFAULT + 1 :].to_numpy(), C0_DEFAULT)


def test_transaction_cost_scales_with_volatility() -> None:
    """Doubling daily vol must shift c_t by approximately lambda * delta_sigma."""
    n = 400
    low = _synthetic_prices(n, sigma=0.005, seed=1)
    high = _synthetic_prices(n, sigma=0.020, seed=1)
    c_low = transaction_cost(low).dropna()
    c_high = transaction_cost(high).dropna()
    assert c_high.mean() > c_low.mean() + 1e-4, "high-vol cost did not increase"


def test_step_reward_no_trade_no_cost() -> None:
    """Holding a constant position => cost component is zero, R_t = a * r_{t+1}."""
    n = 200
    prices = _synthetic_prices(n, sigma=0.01, seed=2)
    actions = np.full(n, 1)
    R = step_reward(prices, actions, initial_position=1.0)

    next_r = np.log(prices["Close"]).diff().shift(-1)
    mask = R.notna() & next_r.notna()
    np.testing.assert_allclose(R[mask].to_numpy(), next_r[mask].to_numpy(), atol=1e-12)


def test_step_reward_one_trade_cost() -> None:
    """Single flip 0 -> +1 at index k charges exactly c_k of cost."""
    n = 200
    prices = _synthetic_prices(n, sigma=0.01, seed=3)
    k = 100
    actions = np.zeros(n, dtype=int)
    actions[k:] = 1

    R = step_reward(prices, actions, initial_position=0.0)
    R_no_cost = step_reward(prices, actions, c0=0.0, lam=0.0, initial_position=0.0)

    diff = R_no_cost - R  # equals c_t * |a_t - a_{t-1}|, which is c_k at t=k, else 0
    expected_cost = transaction_cost(prices).iloc[k]
    np.testing.assert_allclose(diff.iloc[k], expected_cost, atol=1e-12)

    # Off-flip days: the cost-difference must be zero
    others = diff.drop(index=diff.index[k]).dropna()
    np.testing.assert_allclose(others.to_numpy(), 0.0, atol=1e-12)


def test_step_reward_uses_next_day_return() -> None:
    """R_t with a_t = +1 must equal r_{t+1}, not r_t."""
    n = 60
    prices = _synthetic_prices(n, sigma=0.01, seed=4)
    actions = np.ones(n, dtype=int)
    R = step_reward(prices, actions, c0=0.0, lam=0.0, initial_position=1.0)

    log_p = np.log(prices["Close"])
    r = log_p.diff()
    # R at index t should equal r at index t+1
    for t in range(n - 1):
        if not np.isnan(R.iloc[t]):
            np.testing.assert_allclose(R.iloc[t], r.iloc[t + 1], atol=1e-12)
