"""Rollout: does ``simulate_path`` wire ThompsonPolicy + features + env together correctly.

We avoid running the actual sampler in these tests; ``policy`` is built
from a hand-crafted trace so we can pin actions deterministically and
reason about the resulting reward path in closed form.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from project.env import ACTIONS, step_reward
from project.features import FEATURE_COLS, build_features
from project.rollout import simulate_path
from project.thompson import policy_from_trace


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------

def _synthetic_prices(n: int, sigma: float = 0.01, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_p = np.cumsum(rng.normal(0.0, sigma, size=n))
    p = np.exp(log_p) * 100.0
    idx = pd.date_range("2000-01-01", periods=n, freq="B")
    return pd.DataFrame({"Close": p}, index=idx)


def _flat_features(prices: pd.DataFrame, p: int = 4) -> pd.DataFrame:
    """All-ones feature matrix with no NaN rows -- the rollout consumes every day."""
    cols = FEATURE_COLS[:p]
    return pd.DataFrame(np.ones((len(prices), p)), index=prices.index, columns=list(cols))


def _trace_with_dominant_action(action_id: int, p: int = 4, A: int = 3, D: int = 50) -> dict:
    """A 'trace' whose every draw has action ``action_id`` overwhelming the others."""
    beta = np.zeros((1, D, A, p))
    beta[0, :, action_id, :] = 1000.0
    return {"beta": beta}


# ---------------------------------------------------------------------------
# Closed-form: dominant +1 action -> rewards match step_reward(all-long)
# ---------------------------------------------------------------------------

def test_simulate_path_matches_step_reward_when_action_is_pinned() -> None:
    """If Thompson always picks +1, simulate_path's rewards must equal
    ``step_reward(prices, all +1, initial_position=1)`` exactly.

    We use ``initial_position=+1`` so that the (deterministic) action
    sequence has zero transition costs throughout.
    """
    n = 80
    prices = _synthetic_prices(n, sigma=0.01, seed=1)
    feats = _flat_features(prices, p=4)

    long_id = ACTIONS.index(+1)
    trace = _trace_with_dominant_action(long_id, p=4, A=len(ACTIONS), D=20)
    pol = policy_from_trace(trace, feature_cols=FEATURE_COLS)

    out = simulate_path(
        pol, prices, feats,
        resample_every=None, seed=42, initial_position=1.0,
    )
    expected_R = step_reward(prices, np.full(n, +1, dtype=int), initial_position=1.0)

    pd.testing.assert_series_equal(
        out["rewards"], expected_R.rename("rewards"), check_exact=False, rtol=1e-12,
    )
    assert (out["actions"] == +1).all()
    pd.testing.assert_series_equal(out["actions"], out["positions"], check_names=False)


# ---------------------------------------------------------------------------
# Causality: scrambling prices at >= k cannot change actions at < k
# ---------------------------------------------------------------------------

def test_simulate_path_actions_are_strictly_causal() -> None:
    n, k = 800, 600
    prices_truth = _synthetic_prices(n, sigma=0.01, seed=11)
    rng = np.random.default_rng(99)
    perm = rng.permutation(n - k)
    prices_scrambled = prices_truth.copy()
    prices_scrambled.iloc[k:] = prices_truth.iloc[k:].to_numpy()[perm]

    feats_truth = build_features(prices_truth)
    feats_scrambled = build_features(prices_scrambled)

    rng_state = np.random.default_rng(7)
    beta = rng_state.standard_normal((1, 100, len(ACTIONS), 4))
    pol = policy_from_trace({"beta": beta})

    out_truth = simulate_path(pol, prices_truth, feats_truth, seed=2024, resample_every=None)
    out_scram = simulate_path(pol, prices_scrambled, feats_scrambled, seed=2024, resample_every=None)

    pd.testing.assert_series_equal(
        out_truth["actions"].iloc[:k],
        out_scram["actions"].iloc[:k],
        check_exact=True,
    )


# ---------------------------------------------------------------------------
# Cost: rollout's reward path equals step_reward applied to its own actions
# ---------------------------------------------------------------------------

def test_simulate_path_rewards_consistent_with_step_reward() -> None:
    """Wiring check: rewards returned by simulate_path must equal
    step_reward(prices, returned-actions, initial_position).
    """
    n = 60
    prices = _synthetic_prices(n, sigma=0.012, seed=5)
    feats = _flat_features(prices, p=4)

    rng = np.random.default_rng(0)
    beta = rng.standard_normal((1, 50, len(ACTIONS), 4))
    pol = policy_from_trace({"beta": beta})

    out = simulate_path(pol, prices, feats, seed=42, resample_every=1, initial_position=0.0)
    actions = out["actions"].to_numpy()

    expected = step_reward(prices, actions, initial_position=0.0)
    pd.testing.assert_series_equal(out["rewards"], expected.rename("rewards"), rtol=1e-12)


# ---------------------------------------------------------------------------
# Per-episode determinism + per-step differs from per-episode
# ---------------------------------------------------------------------------

def test_simulate_path_per_episode_determinism() -> None:
    """Same seed under the per-episode default reproduces the path exactly."""
    n = 50
    prices = _synthetic_prices(n, sigma=0.01, seed=2)
    feats = _flat_features(prices, p=4)

    rng = np.random.default_rng(0)
    beta = rng.standard_normal((1, 100, len(ACTIONS), 4))
    pol = policy_from_trace({"beta": beta})

    a = simulate_path(pol, prices, feats, seed=42, resample_every=None)
    b = simulate_path(pol, prices, feats, seed=42, resample_every=None)
    pd.testing.assert_series_equal(a["actions"], b["actions"])
    pd.testing.assert_series_equal(a["rewards"], b["rewards"])


def test_simulate_path_per_step_differs_from_per_episode_same_seed() -> None:
    """Per-step Thompson must produce a different action sequence than per-episode
    PSRL even when they share the seed -- they consume the rng differently.
    """
    n = 60
    prices = _synthetic_prices(n, sigma=0.01, seed=2)
    feats = _flat_features(prices, p=4)

    # Random beta with enough spread that the chosen draw really matters.
    rng = np.random.default_rng(0)
    beta = rng.standard_normal((1, 200, len(ACTIONS), 4)) * 5.0
    pol = policy_from_trace({"beta": beta})

    per_episode = simulate_path(pol, prices, feats, seed=42, resample_every=None)
    per_step = simulate_path(pol, prices, feats, seed=42, resample_every=1)
    assert not per_episode["actions"].equals(per_step["actions"])


def test_simulate_path_validates_action_labels_length() -> None:
    n = 30
    prices = _synthetic_prices(n)
    feats = _flat_features(prices, p=4)
    rng = np.random.default_rng(0)
    pol = policy_from_trace({"beta": rng.standard_normal((1, 10, len(ACTIONS), 4))})
    with pytest.raises(ValueError):
        simulate_path(pol, prices, feats, seed=1, action_labels=(0, 1))   # 2 != 3
