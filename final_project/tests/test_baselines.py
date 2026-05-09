"""Frequentist baselines: random, tabular MC-regression, linear FQI.

The shared-contract block parametrizes over all three policies. Tabular- and
linear-specific tests then probe the parts of the contract that only one
policy is responsible for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from project.baselines import (
    classical_baseline_distribution,
    linear_fqi_path,
    random_policy_path,
    tabular_q_path,
)
from project.env import ACTIONS, step_reward
from project.features import FEATURE_COLS, build_features


# ---------------------------------------------------------------------------
# Helpers (legacy synthetic + drift-vol synthetic)
# ---------------------------------------------------------------------------

def _synthetic_prices(n: int, sigma: float = 0.01, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_p = np.cumsum(rng.normal(0.0, sigma, size=n))
    p = np.exp(log_p) * 100.0
    idx = pd.date_range("2000-01-01", periods=n, freq="B")
    return pd.DataFrame({"Close": p}, index=idx)


def _flat_features(prices: pd.DataFrame, p: int = 4) -> pd.DataFrame:
    cols = list(FEATURE_COLS[:p])
    return pd.DataFrame(np.ones((len(prices), p)), index=prices.index, columns=cols)


def _drift_prices(n: int, mu: float = 0.0, sigma: float = 0.01, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_p = np.cumsum(rng.normal(loc=mu, scale=sigma, size=n))
    p = np.exp(log_p) * 100.0
    idx = pd.date_range("2000-01-01", periods=n, freq="B")
    return pd.DataFrame({"Close": p}, index=idx)


def _split(prices: pd.DataFrame, train_frac: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_train = int(len(prices) * train_frac)
    return prices.iloc[:n_train], prices.iloc[n_train:]


def _wrap_random(prices_train, features_train, prices_test, features_test, *, seed):
    """Adapter so `random_policy_path` matches the (train, test) signature
    that the contract tests parametrize over."""
    return random_policy_path(prices_test, features_test, n_actions=len(ACTIONS), seed=seed)


# ---------------------------------------------------------------------------
# Existing random-policy tests (preserved verbatim)
# ---------------------------------------------------------------------------

def test_random_policy_path_determinism_same_seed() -> None:
    n = 80
    prices = _synthetic_prices(n)
    feats = _flat_features(prices)
    a = random_policy_path(prices, feats, n_actions=len(ACTIONS), seed=42)
    b = random_policy_path(prices, feats, n_actions=len(ACTIONS), seed=42)
    pd.testing.assert_series_equal(a["actions"], b["actions"])
    pd.testing.assert_series_equal(a["rewards"], b["rewards"])


def test_random_policy_path_different_seeds_differ() -> None:
    n = 80
    prices = _synthetic_prices(n)
    feats = _flat_features(prices)
    a = random_policy_path(prices, feats, n_actions=len(ACTIONS), seed=1)
    b = random_policy_path(prices, feats, n_actions=len(ACTIONS), seed=2)
    assert not a["actions"].equals(b["actions"])


def test_random_policy_path_actions_in_action_set() -> None:
    n = 200
    prices = _synthetic_prices(n)
    feats = _flat_features(prices)
    out = random_policy_path(prices, feats, n_actions=len(ACTIONS), seed=7)
    unique_acts = set(out["actions"].unique().tolist())
    assert unique_acts.issubset(set(ACTIONS))


def test_random_policy_path_rewards_consistent_with_env() -> None:
    n = 80
    prices = _synthetic_prices(n, sigma=0.012, seed=3)
    feats = _flat_features(prices)
    out = random_policy_path(prices, feats, n_actions=len(ACTIONS), seed=99)
    expected = step_reward(prices, out["actions"].to_numpy(), initial_position=0.0)
    pd.testing.assert_series_equal(out["rewards"], expected.rename("rewards"), rtol=1e-12)


def test_random_policy_path_returns_required_keys() -> None:
    n = 40
    prices = _synthetic_prices(n)
    feats = _flat_features(prices)
    out = random_policy_path(prices, feats, n_actions=len(ACTIONS), seed=5)
    assert set(out.keys()) >= {"actions", "rewards"}


# ---------------------------------------------------------------------------
# Shared contract (parametrized over all three baselines)
# ---------------------------------------------------------------------------

POLICIES = [
    pytest.param(_wrap_random, id="random"),
    pytest.param(tabular_q_path, id="tabular"),
    pytest.param(linear_fqi_path, id="linear"),
]


@pytest.fixture
def shared_data():
    """A 1500-day drift-vol series, train/test split 70/30, full-history features."""
    n = 1500
    prices = _drift_prices(n, mu=0.0001, sigma=0.01, seed=0)
    features = build_features(prices)
    pt, pte = _split(prices)
    return pt, features.loc[pt.index], pte, features.loc[pte.index]


@pytest.mark.parametrize("policy_fn", POLICIES)
def test_baseline_returns_required_keys(policy_fn, shared_data) -> None:
    pt, ft, pte, fte = shared_data
    out = policy_fn(pt, ft, pte, fte, seed=42)
    assert set(out.keys()) == {"actions", "positions", "rewards", "cum_log_return"}
    for k, v in out.items():
        assert isinstance(v, pd.Series), f"key {k!r} is not a Series"
        assert len(v) == len(pte), f"key {k!r} has length {len(v)} != {len(pte)}"
    pd.testing.assert_index_equal(out["actions"].index, pte.index)


@pytest.mark.parametrize("policy_fn", POLICIES)
def test_baseline_determinism_same_seed(policy_fn, shared_data) -> None:
    pt, ft, pte, fte = shared_data
    a = policy_fn(pt, ft, pte, fte, seed=11)
    b = policy_fn(pt, ft, pte, fte, seed=11)
    pd.testing.assert_series_equal(a["actions"], b["actions"])
    pd.testing.assert_series_equal(a["rewards"], b["rewards"])


@pytest.mark.parametrize("policy_fn", POLICIES)
def test_baseline_action_validity(policy_fn, shared_data) -> None:
    pt, ft, pte, fte = shared_data
    out = policy_fn(pt, ft, pte, fte, seed=3)
    assert set(out["actions"].dropna().unique().tolist()).issubset(set(ACTIONS))


@pytest.mark.parametrize("policy_fn", POLICIES)
def test_baseline_cost_accounting_matches_step_reward(policy_fn, shared_data) -> None:
    pt, ft, pte, fte = shared_data
    out = policy_fn(pt, ft, pte, fte, seed=4)
    expected = step_reward(pte, out["actions"].to_numpy(), initial_position=0.0)
    pd.testing.assert_series_equal(out["rewards"], expected.rename("rewards"), rtol=1e-12)


@pytest.mark.parametrize("policy_fn", POLICIES)
def test_baseline_warmup_rows_are_neutral(policy_fn) -> None:
    """NaN-feature rows -> action 0, matching the random_policy_path convention."""
    n = 1500
    prices = _drift_prices(n, mu=0.0001, sigma=0.01, seed=0)
    full_feats = build_features(prices)
    pt, pte = _split(prices)
    ft = full_feats.loc[pt.index]
    fte_isolated = build_features(pte)              # warmup NaN at the head
    out = policy_fn(pt, ft, pte, fte_isolated, seed=42)
    nan_mask = fte_isolated.isna().any(axis=1).to_numpy()
    assert nan_mask.any(), "test setup error: expected NaN warmup rows"
    assert (out["actions"].to_numpy()[nan_mask] == 0).all()


@pytest.mark.parametrize("policy_fn", POLICIES)
def test_baseline_causal_to_test_scrambling(policy_fn) -> None:
    """Scrambling test prices at indices >= K_test cannot change the policy's
    actions earlier than K_test - z_window. Random ignores prices entirely;
    tabular/linear see different test features only at indices >= K_test - 252."""
    n, train_len, K_test = 2000, 1400, 400
    z_window = 252
    causal_k = K_test - z_window                   # 148

    prices_truth = _drift_prices(n, mu=0.0001, sigma=0.01, seed=11)
    rng = np.random.default_rng(99)
    perm = rng.permutation(n - train_len - K_test)
    prices_scram = prices_truth.copy()
    prices_scram.iloc[train_len + K_test:] = (
        prices_truth.iloc[train_len + K_test:].to_numpy()[perm]
    )

    feat_truth = build_features(prices_truth)
    feat_scram = build_features(prices_scram)
    pt = prices_truth.iloc[:train_len]
    ft = feat_truth.loc[pt.index]
    pte_truth = prices_truth.iloc[train_len:]
    pte_scram = prices_scram.iloc[train_len:]
    fte_truth = feat_truth.loc[pte_truth.index]
    fte_scram = feat_scram.loc[pte_scram.index]

    out_truth = policy_fn(pt, ft, pte_truth, fte_truth, seed=2024)
    out_scram = policy_fn(pt, ft, pte_scram, fte_scram, seed=2024)

    pd.testing.assert_series_equal(
        out_truth["actions"].iloc[:causal_k],
        out_scram["actions"].iloc[:causal_k],
        check_exact=True,
    )


# ---------------------------------------------------------------------------
# Tabular-specific
# ---------------------------------------------------------------------------

def test_tabular_clamps_test_features_outside_train_support() -> None:
    """Bin edges fit on train alone; test features beyond train support must
    clamp to the edge bins (no NaN actions, all in ACTIONS)."""
    n_train, n_test = 800, 700
    rng = np.random.default_rng(0)
    prices_train = _drift_prices(n_train, mu=0.0001, sigma=0.005, seed=0)
    log_p_test = np.cumsum(rng.normal(loc=0.005, scale=0.02, size=n_test))
    p_test = np.exp(log_p_test) * float(prices_train["Close"].iloc[-1])
    test_idx = pd.date_range(
        prices_train.index[-1] + pd.Timedelta(days=1), periods=n_test, freq="B",
    )
    prices_test = pd.DataFrame({"Close": p_test}, index=test_idx)
    full = pd.concat([prices_train, prices_test])
    feats = build_features(full)
    ft = feats.loc[prices_train.index]
    fte = feats.loc[prices_test.index]

    out = tabular_q_path(prices_train, ft, prices_test, fte, seed=42)

    assert not out["actions"].isna().any()
    assert set(out["actions"].unique().tolist()).issubset(set(ACTIONS))


def test_tabular_recovers_long_in_drift_dominated_env() -> None:
    """Drift mu=0.005, vol sigma=0.001 -> c_t ~= 0.001 << drift; +1 dominates.

    NB: env's initial_position=0 means even an all-+1 policy pays one transition
    cost on the first valid day, but mu >> c_t makes that cost a rounding error.
    The 80% gate is loose; trivial environment, trivial method should pass.

    n_bins=2 here, not the default 4: the synthetic series has ~735 valid
    training rows after feature-warmup and MC-target right-truncation; with
    2^3=8 binnable joint states across 3 actions there are ~30 train samples
    per cell, enough for the per-cell empirical mean to dominate noise. The
    default n_bins=4 is sized for ~6200 SPY train rows; documenting this is
    the discretization fairness story for nb-09's sweep over n_bins.
    """
    n = 1500
    prices = _drift_prices(n, mu=0.005, sigma=0.001, seed=42)
    feats = build_features(prices)
    pt, pte = _split(prices)
    ft, fte = feats.loc[pt.index], feats.loc[pte.index]
    out = tabular_q_path(pt, ft, pte, fte, seed=42, n_bins=2)
    valid_mask = fte.notna().all(axis=1).to_numpy()
    valid_actions = out["actions"].to_numpy()[valid_mask]
    frac_long = float((valid_actions == +1).mean())
    assert frac_long >= 0.80, f"only {frac_long:.2%} long, expected >= 80%"


def test_tabular_td_streaming_mode_is_reserved() -> None:
    """The headline is mc_regression; td_streaming is a nb-09 sweep stub."""
    n = 200
    prices = _drift_prices(n, mu=0.0, sigma=0.01, seed=0)
    feats = build_features(prices)
    pt, pte = _split(prices)
    ft, fte = feats.loc[pt.index], feats.loc[pte.index]
    with pytest.raises(NotImplementedError):
        tabular_q_path(pt, ft, pte, fte, seed=0, mode="td_streaming")


# ---------------------------------------------------------------------------
# Linear-FQI-specific
# ---------------------------------------------------------------------------

def test_linear_fqi_recovers_long_in_drift_dominated_env() -> None:
    """Same drift env as the tabular test; linear FQI must clear 80% long."""
    n = 1500
    prices = _drift_prices(n, mu=0.005, sigma=0.001, seed=42)
    feats = build_features(prices)
    pt, pte = _split(prices)
    ft, fte = feats.loc[pt.index], feats.loc[pte.index]
    out = linear_fqi_path(pt, ft, pte, fte, seed=42)
    valid_mask = fte.notna().all(axis=1).to_numpy()
    valid_actions = out["actions"].to_numpy()[valid_mask]
    frac_long = float((valid_actions == +1).mean())
    assert frac_long >= 0.80, f"only {frac_long:.2%} long, expected >= 80%"


def test_linear_fqi_rejects_misordered_features() -> None:
    """Column reordering must raise. Silent acceptance would point the FQI
    weights at the wrong semantic features and break the fairness comparison
    against the Bayesian model, where ``policy.feature_cols`` is enforced."""
    n = 1500
    prices = _drift_prices(n, mu=0.0001, sigma=0.01, seed=0)
    feats = build_features(prices)
    swapped = feats[[FEATURE_COLS[1], FEATURE_COLS[0], FEATURE_COLS[2], FEATURE_COLS[3]]]
    pt, pte = _split(prices)
    ft_bad, fte_bad = swapped.loc[pt.index], swapped.loc[pte.index]
    with pytest.raises(ValueError):
        linear_fqi_path(pt, ft_bad, pte, fte_bad, seed=1)


def test_linear_fqi_td_streaming_mode_is_reserved() -> None:
    n = 200
    prices = _drift_prices(n, mu=0.0, sigma=0.01, seed=0)
    feats = build_features(prices)
    pt, pte = _split(prices)
    ft, fte = feats.loc[pt.index], feats.loc[pte.index]
    with pytest.raises(NotImplementedError):
        linear_fqi_path(pt, ft, pte, fte, seed=0, mode="td_streaming")


# ---------------------------------------------------------------------------
# classical_baseline_distribution orchestrator
# ---------------------------------------------------------------------------

def test_classical_baseline_distribution_schema() -> None:
    """Returns one entry per metric, each with {mean, ci_lo, ci_hi} -- mirrors
    posterior_predictive_metrics' nested-dict shape so plotting code can stay
    branch-free across Bayesian and classical baselines."""
    n = 1200
    prices = _drift_prices(n, mu=0.0001, sigma=0.01, seed=0)
    feats = build_features(prices)
    pt, pte = _split(prices)
    ft, fte = feats.loc[pt.index], feats.loc[pte.index]
    out = classical_baseline_distribution(
        tabular_q_path, pt, ft, pte, fte, n_seeds=5, base_seed=0,
    )
    assert set(out.keys()) == {"sharpe", "max_drawdown", "turnover"}
    for m, summary in out.items():
        assert set(summary.keys()) == {"mean", "ci_lo", "ci_hi"}
        for v in summary.values():
            assert np.isfinite(v), f"{m} summary has non-finite value"


def test_classical_baseline_distribution_reproducibility() -> None:
    n = 1200
    prices = _drift_prices(n, mu=0.0001, sigma=0.01, seed=0)
    feats = build_features(prices)
    pt, pte = _split(prices)
    ft, fte = feats.loc[pt.index], feats.loc[pte.index]
    a = classical_baseline_distribution(
        linear_fqi_path, pt, ft, pte, fte, n_seeds=5, base_seed=7,
    )
    b = classical_baseline_distribution(
        linear_fqi_path, pt, ft, pte, fte, n_seeds=5, base_seed=7,
    )
    assert a == b


def test_classical_baseline_distribution_ci_brackets_mean() -> None:
    """Bootstrap-of-the-mean CI must contain the empirical mean by construction."""
    n = 1200
    prices = _drift_prices(n, mu=0.0001, sigma=0.01, seed=0)
    feats = build_features(prices)
    pt, pte = _split(prices)
    ft, fte = feats.loc[pt.index], feats.loc[pte.index]
    out = classical_baseline_distribution(
        _wrap_random, pt, ft, pte, fte, n_seeds=20, base_seed=0,
    )
    s = out["sharpe"]
    assert s["ci_lo"] <= s["mean"] <= s["ci_hi"]
