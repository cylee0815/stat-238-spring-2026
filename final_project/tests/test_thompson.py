"""Thompson-sampling policy: shape, determinism, posterior_q correctness.

The policy is a thin functional view onto the sampler's ``beta`` block, so
every test here works on a synthetic trace dict shaped like
:func:`project.sampler.gibbs_gaussian`'s output. No actual sampler runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from project.features import FEATURE_COLS
from project.thompson import (
    ThompsonPolicy,
    policy_from_trace,
    posterior_q,
    thompson_action,
)


def _toy_trace(
    *,
    n_chains: int = 2,
    n_draws: int = 20,
    A: int = 3,
    p: int = 4,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    beta = rng.standard_normal((n_chains, n_draws, A, p))
    return {"beta": beta}


def test_policy_from_trace_flatten_shape() -> None:
    """Chains and draws are flattened into a single posterior-sample axis."""
    n_chains, n_draws, A, p = 4, 30, 3, 4
    trace = _toy_trace(n_chains=n_chains, n_draws=n_draws, A=A, p=p)
    pol = policy_from_trace(trace)
    assert isinstance(pol, ThompsonPolicy)
    assert pol.beta_draws.shape == (n_chains * n_draws, A, p)
    assert pol.n_actions == A
    assert pol.feature_cols == FEATURE_COLS


def test_policy_from_trace_thin_halves_draws() -> None:
    n_chains, n_draws, A, p = 2, 100, 3, 4
    trace = _toy_trace(n_chains=n_chains, n_draws=n_draws, A=A, p=p)
    full = policy_from_trace(trace, thin=1)
    thinned = policy_from_trace(trace, thin=2)
    assert full.beta_draws.shape[0] == n_chains * n_draws
    assert thinned.beta_draws.shape[0] == (n_chains * n_draws) // 2


def test_policy_from_trace_rejects_mismatched_feature_cols() -> None:
    trace = _toy_trace(p=4)
    with pytest.raises(ValueError):
        policy_from_trace(trace, feature_cols=("a", "b"))


def test_policy_from_trace_rejects_nonpositive_thin() -> None:
    trace = _toy_trace()
    with pytest.raises(ValueError):
        policy_from_trace(trace, thin=0)


def test_thompson_action_determinism_same_seed() -> None:
    trace = _toy_trace(p=4)
    pol = policy_from_trace(trace)
    x = np.array([0.1, -0.2, 0.05, 1.0])
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    actions_a = [thompson_action(pol, x, rng_a) for _ in range(50)]
    actions_b = [thompson_action(pol, x, rng_b) for _ in range(50)]
    assert actions_a == actions_b


def test_thompson_action_different_seeds_differ() -> None:
    trace = _toy_trace(p=4, seed=1)
    pol = policy_from_trace(trace)
    x = np.array([0.1, -0.2, 0.05, 1.0])
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(99)
    actions_a = [thompson_action(pol, x, rng_a) for _ in range(100)]
    actions_b = [thompson_action(pol, x, rng_b) for _ in range(100)]
    assert actions_a != actions_b


def test_posterior_q_shape_and_mean() -> None:
    n_chains, n_draws, A, p = 2, 100, 3, 4
    trace = _toy_trace(n_chains=n_chains, n_draws=n_draws, A=A, p=p, seed=42)
    pol = policy_from_trace(trace)
    x = np.array([0.3, -0.1, 0.2, 1.0])
    q = posterior_q(pol, x)
    assert q.shape == (n_chains * n_draws, A)
    # Mean of x @ beta_draws[d, a] over d == x @ beta_draws.mean(0)[a]
    np.testing.assert_allclose(
        q.mean(axis=0), pol.beta_draws.mean(axis=0) @ x, atol=1e-12
    )


def test_thompson_action_pathological_dominant() -> None:
    """If one action's beta is 1000x larger, Thompson returns it ~always."""
    A, p, D = 3, 4, 1000
    beta = np.zeros((1, D, A, p))     # 1 chain, D draws
    beta[0, :, 1, :] = 1000.0          # action 1 dominates every draw
    pol = policy_from_trace({"beta": beta})
    rng = np.random.default_rng(0)
    x = np.ones(p)
    actions = [thompson_action(pol, x, rng) for _ in range(1000)]
    assert sum(a == 1 for a in actions) >= 999


def test_thompson_action_per_episode_deterministic() -> None:
    """``draw_idx`` fixed -> action is a function of (policy, x) only; rng ignored."""
    trace = _toy_trace(seed=1)
    pol = policy_from_trace(trace)
    x = np.array([0.1, -0.2, 0.05, 1.0])
    a1 = thompson_action(pol, x, np.random.default_rng(0), draw_idx=5)
    a2 = thompson_action(pol, x, np.random.default_rng(99999), draw_idx=5)
    a3 = thompson_action(pol, x, np.random.default_rng(7), draw_idx=5)
    assert a1 == a2 == a3
