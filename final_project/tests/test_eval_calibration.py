"""Posterior-predictive evaluation: per-metric correctness, aggregation, coverage.

This file replaces the original ``@pytest.mark.skip`` placeholder. The
slow integration test at the bottom (``test_credible_interval_coverage``)
is a *light* version of the proposal's 100-rep ground-truth coverage
check: 30 replications using a small Gaussian regression, feeding into
the real :func:`project.eval.posterior_predictive_metrics` pipeline. The
gate is >= 25 / 30, the binomial-acceptance band for nominal 95% coverage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from project.env import ACTIONS, step_reward
from project.eval import (
    action_frequency,
    cumulative_log_return,
    max_drawdown,
    posterior_predictive_metrics,
    pseudo_regret_vs_best_fixed,
    sharpe,
    turnover,
)
from project.features import FEATURE_COLS, build_features
from project.priors import make_priors
from project.rollout import simulate_path
from project.sampler import gibbs_gaussian
from project.thompson import policy_from_trace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _toy_rewards(values: list[float]) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx, name="rewards")


def _synthetic_prices(n: int, sigma: float = 0.01, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_p = np.cumsum(rng.normal(0.0, sigma, size=n))
    p = np.exp(log_p) * 100.0
    idx = pd.date_range("2000-01-01", periods=n, freq="B")
    return pd.DataFrame({"Close": p}, index=idx)


# ---------------------------------------------------------------------------
# Per-metric unit tests
# ---------------------------------------------------------------------------

def test_cumulative_log_return_matches_cumsum() -> None:
    R = _toy_rewards([0.001, 0.002, -0.0005, 0.0, 0.003])
    pd.testing.assert_series_equal(
        cumulative_log_return(R),
        R.cumsum().rename("cum_log_return"),
        check_names=False,
    )


def test_sharpe_matches_formula_iid() -> None:
    rng = np.random.default_rng(0)
    n = 500
    r = rng.normal(loc=0.001, scale=0.01, size=n)
    R = pd.Series(r, index=pd.date_range("2020-01-01", periods=n, freq="B"))
    expected = (r.mean() / r.std(ddof=1)) * np.sqrt(252)
    np.testing.assert_allclose(sharpe(R), expected, rtol=1e-12)


def test_sharpe_zero_std_returns_zero() -> None:
    R = _toy_rewards([0.001] * 50)
    assert sharpe(R) == 0.0


def test_sharpe_skips_nan() -> None:
    R = _toy_rewards([np.nan, 0.001, np.nan, 0.002, -0.001])
    expected = (np.array([0.001, 0.002, -0.001]).mean()
                / np.array([0.001, 0.002, -0.001]).std(ddof=1)) * np.sqrt(252)
    np.testing.assert_allclose(sharpe(R), expected, rtol=1e-12)


def test_max_drawdown_monotone_zero() -> None:
    cum = pd.Series(np.linspace(0.0, 1.0, 50),
                    index=pd.date_range("2020-01-01", periods=50, freq="B"))
    assert max_drawdown(cum) == 0.0


def test_max_drawdown_handcomputed() -> None:
    """cum log returns 0 -> 0.10 -> -0.05 -> 0.02. MDD between 0.10 and -0.05 is 0.15 (log)."""
    cum = pd.Series(
        [0.0, 0.10, -0.05, 0.02],
        index=pd.date_range("2020-01-01", periods=4, freq="B"),
    )
    np.testing.assert_allclose(max_drawdown(cum), 0.15, rtol=1e-12)


def test_turnover_constant_action_zero() -> None:
    a = pd.Series([1] * 30, index=pd.date_range("2020-01-01", periods=30, freq="B"))
    assert turnover(a) == 0.0


def test_turnover_matches_mean_abs_diff() -> None:
    a = pd.Series([0, 1, 1, -1, 0, 0],
                  index=pd.date_range("2020-01-01", periods=6, freq="B"))
    expected = (1 + 0 + 2 + 1 + 0) / 5.0
    np.testing.assert_allclose(turnover(a), expected, rtol=1e-12)


def test_action_frequency_normalizes() -> None:
    a = pd.Series([-1, -1, 0, 0, 0, 1, 1, 1, 1, 1],
                  index=pd.date_range("2020-01-01", periods=10, freq="B"))
    f = action_frequency(a)
    assert f == {-1: 0.2, 0: 0.3, 1: 0.5}


def test_pseudo_regret_vs_best_fixed_handcomputed() -> None:
    idx = pd.date_range("2020-01-01", periods=4, freq="B")
    rpa = {
        -1: pd.Series([0.0, -0.01, 0.0, -0.005], index=idx),
        0: pd.Series([0.0, 0.0, 0.0, 0.0], index=idx),
        +1: pd.Series([0.01, 0.005, -0.001, 0.003], index=idx),
        "chosen": pd.Series([0.005, 0.0, 0.001, 0.002], index=idx),
    }
    # totals: -1 -> -0.015, 0 -> 0, +1 -> 0.017 ; best_fixed = +1
    expected = rpa[+1] - rpa["chosen"]
    pd.testing.assert_series_equal(
        pseudo_regret_vs_best_fixed(rpa), expected, check_names=False,
    )


def test_pseudo_regret_requires_chosen_key() -> None:
    idx = pd.date_range("2020-01-01", periods=3, freq="B")
    rpa = {0: pd.Series([0.0] * 3, index=idx), 1: pd.Series([0.0] * 3, index=idx)}
    with pytest.raises(ValueError):
        pseudo_regret_vs_best_fixed(rpa)


# ---------------------------------------------------------------------------
# posterior_predictive_metrics: aggregation correctness
# ---------------------------------------------------------------------------

def _flat_features(prices: pd.DataFrame, p: int = 4) -> pd.DataFrame:
    cols = list(FEATURE_COLS[:p])
    return pd.DataFrame(np.ones((len(prices), p)), index=prices.index, columns=cols)


def test_posterior_predictive_metrics_keys_and_finite() -> None:
    n = 60
    prices = _synthetic_prices(n, seed=1)
    feats = _flat_features(prices)
    rng = np.random.default_rng(0)
    beta = rng.standard_normal((1, 30, len(ACTIONS), 4))
    pol = policy_from_trace({"beta": beta})

    result = posterior_predictive_metrics(pol, prices, feats, n_paths=20, seed=42)
    assert set(result.keys()) == {"sharpe", "max_drawdown", "turnover"}
    for metric, summary in result.items():
        assert set(summary.keys()) == {"mean", "q025", "q975"}
        for v in summary.values():
            assert np.isfinite(v), f"{metric}.{summary} non-finite"


def test_posterior_predictive_metrics_determinism_same_seed() -> None:
    n = 60
    prices = _synthetic_prices(n, seed=2)
    feats = _flat_features(prices)
    rng = np.random.default_rng(0)
    beta = rng.standard_normal((1, 30, len(ACTIONS), 4))
    pol = policy_from_trace({"beta": beta})
    a = posterior_predictive_metrics(pol, prices, feats, n_paths=20, seed=42)
    b = posterior_predictive_metrics(pol, prices, feats, n_paths=20, seed=42)
    assert a == b


def test_posterior_predictive_metrics_degenerate_zero_width() -> None:
    """All posterior draws identical -> all paths identical -> CrI width = 0."""
    n = 60
    prices = _synthetic_prices(n, seed=3)
    feats = _flat_features(prices)
    # Every draw favors action +1 unanimously: identical paths.
    A, p, D = len(ACTIONS), 4, 50
    long_id = ACTIONS.index(+1)
    beta = np.zeros((1, D, A, p))
    beta[0, :, long_id, :] = 1000.0
    pol = policy_from_trace({"beta": beta})
    result = posterior_predictive_metrics(pol, prices, feats, n_paths=10, seed=7)
    for m, s in result.items():
        np.testing.assert_allclose(s["q025"], s["q975"], atol=1e-12, err_msg=f"{m} CrI not zero-width")
        np.testing.assert_allclose(s["mean"], s["q025"], atol=1e-12)


# ---------------------------------------------------------------------------
# Coverage: the headline calibration test (replaces the old skip).
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_credible_interval_coverage() -> None:
    """30 reps; for each: synthetic Gaussian regression, fit Gibbs, run
    posterior_predictive_metrics on synthetic prices, check 95% CrI on
    Sharpe contains the true-policy Sharpe in >= 25 / 30. The gate is the
    binomial-acceptance band for nominal 95% coverage.
    """
    n_reps = 30
    cov = 0
    base = np.random.SeedSequence(31415)
    seeds = base.spawn(n_reps)

    A = len(ACTIONS)        # 3
    p = 4

    for r in range(n_reps):
        rng = np.random.default_rng(seeds[r])

        # Synthetic prices and features.
        prices = _synthetic_prices(n=400, sigma=0.01, seed=int(rng.integers(1 << 30)))
        feats_full = build_features(prices).dropna()
        # Restrict prices to the rows where features are valid (so the rollout
        # window aligns with the regression training window).
        prices = prices.loc[feats_full.index]
        N = len(feats_full)

        # Synthetic regression: known beta_true, sigma_true. Behavior actions
        # uniform; the regression target y is purely synthetic (decoupled from
        # the actual log-return path -- this is testing the *evaluation*
        # pipeline, not the env model itself).
        sigma_true = 0.02
        beta_true = 0.05 * rng.standard_normal((A, p))
        behavior_a = rng.integers(0, A, size=N)
        X = feats_full.to_numpy()
        y = (X * beta_true[behavior_a]).sum(axis=1) + sigma_true * rng.standard_normal(N)

        # Fit a small Gibbs sampler.
        priors = make_priors(sigma_y_hat=float(np.std(y)), p=p, n_actions=A)
        trace = gibbs_gaussian(
            X, y, behavior_a, priors,
            n_draws=400, n_burn=200, n_chains=2,
            seed=int(rng.integers(1 << 30)),
        )
        policy = policy_from_trace(trace, feature_cols=tuple(feats_full.columns))

        # Posterior CrI on Sharpe.
        result = posterior_predictive_metrics(
            policy, prices, feats_full,
            n_paths=80,
            seed=int(rng.integers(1 << 30)),
        )
        lo = result["sharpe"]["q025"]
        hi = result["sharpe"]["q975"]

        # "True" Sharpe: deterministic argmax-policy under beta_true.
        true_actions = np.array(
            [ACTIONS[int(np.argmax(X[t] @ beta_true.T))] for t in range(N)],
            dtype=int,
        )
        true_rewards = step_reward(prices, true_actions, initial_position=0.0)
        true_sharpe = sharpe(true_rewards)

        if lo <= true_sharpe <= hi:
            cov += 1

    assert cov >= 25, (
        f"Sharpe credible-interval coverage: {cov} / {n_reps} (expected >= 25). "
        "Falling below this gate suggests the posterior over Sharpe is mis-calibrated "
        "or the sampler / rollout pipeline has regressed."
    )
