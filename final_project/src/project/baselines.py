"""Frequentist baselines.

Three policies share a single dict-of-Series return shape so that
:mod:`project.eval` and :func:`classical_baseline_distribution` can consume
any of them without branching:

- :func:`random_policy_path`   -- uniform-random-action floor benchmark.
- :func:`tabular_q_path`       -- tabular MC-regression Q baseline.
- :func:`linear_fqi_path`      -- linear FQI on the same features as the
  Bayesian model (the fairness-controlled comparator).

Headline mode for both classical baselines is ``mc_regression``: one-shot
empirical Bellman regression on the same MC targets the Bayesian sampler
fits, with no TD bootstrap and no replay buffer. This makes the Bayesian-vs-
classical contrast turn purely on uncertainty handling rather than on data
flow or effective horizon. The TD-streaming variants (``mode="td_streaming"``)
are reserved for the nb-09 sweep and currently raise ``NotImplementedError``.

:func:`classical_baseline_distribution` runs any of the above ``n_seeds``
times and reports each path-level metric as a mean + 95% percentile-bootstrap
CI. This is the classical analogue of
:func:`project.eval.posterior_predictive_metrics`: without it, the Bayesian
posterior predictive distribution would be compared against a single
classical point estimate, biasing the contrast against the classical method.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from project.env import (
    ACTIONS,
    C0_DEFAULT,
    LAMBDA_DEFAULT,
    VOL_WINDOW_DEFAULT,
    step_reward,
)
from project.eval import max_drawdown, sharpe, turnover
from project.features import FEATURE_COLS
from project.targets import GAMMA_DEFAULT, H_DEFAULT, mc_returns


# ---------------------------------------------------------------------------
# Random-action benchmark
# ---------------------------------------------------------------------------

def random_policy_path(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    n_actions: int,
    *,
    seed: int,
    action_labels: tuple[int, ...] = ACTIONS,
    initial_position: float = 0.0,
    c0: float = C0_DEFAULT,
    lam: float = LAMBDA_DEFAULT,
    vol_window: int = VOL_WINDOW_DEFAULT,
) -> dict[str, pd.Series]:
    """Uniform-random action baseline with the same env model as the Bayesian
    rollout.

    Warmup days (any NaN feature row) get action 0 (neutral) so the cost
    accounting on the first valid day matches the rollout convention.

    Returns the same keyset as :func:`project.rollout.simulate_path`:
    ``actions``, ``positions``, ``rewards``, ``cum_log_return``.
    """
    if len(action_labels) != n_actions:
        raise ValueError(
            f"action_labels has {len(action_labels)} entries but n_actions={n_actions}"
        )
    rng = np.random.default_rng(seed)
    aligned = features.reindex(prices.index)
    valid_mask = aligned.notna().all(axis=1).to_numpy()
    n = len(prices)

    actions = np.zeros(n, dtype=int)
    n_valid = int(valid_mask.sum())
    rand_ids = rng.integers(0, n_actions, size=n_valid)
    labels = np.asarray(action_labels, dtype=int)
    actions[valid_mask] = labels[rand_ids]

    rewards = step_reward(
        prices, actions,
        c0=c0, lam=lam, vol_window=vol_window,
        initial_position=initial_position,
    ).rename("rewards")
    actions_s = pd.Series(actions, index=prices.index, name="actions")
    cum = rewards.cumsum().rename("cum_log_return")
    return {
        "actions": actions_s,
        "positions": actions_s.rename("positions"),
        "rewards": rewards,
        "cum_log_return": cum,
    }


# ---------------------------------------------------------------------------
# Internal: behaviour-policy training data + binning helpers
# ---------------------------------------------------------------------------

def _generate_training_data(
    prices_train: pd.DataFrame,
    *,
    rng: np.random.Generator,
    n_actions: int,
    action_labels: tuple[int, ...],
    gamma: float,
    H: int,
    initial_position: float,
    c0: float,
    lam: float,
    vol_window: int,
) -> tuple[np.ndarray, pd.Series]:
    """Generate behaviour-policy actions, run the env, compute MC targets.

    The behaviour policy is uniform-random over ``n_actions`` -- this matches
    the assumption ``test_eval_calibration.py`` makes for the Bayesian
    sampler's training data.
    """
    n_train = len(prices_train)
    behavior_a = rng.integers(0, n_actions, size=n_train)
    behavior_labels = np.asarray(action_labels, dtype=int)[behavior_a]
    train_rewards = step_reward(
        prices_train, behavior_labels,
        c0=c0, lam=lam, vol_window=vol_window,
        initial_position=initial_position,
    )
    y = mc_returns(train_rewards, gamma=gamma, H=H)
    return behavior_a, y


def _quantile_bin_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-frequency bin edges; the right edge is bumped past the max so
    that ``searchsorted`` lands the maximum value in the last bin."""
    v = values[~np.isnan(values)]
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(v, quantiles)
    edges[-1] = edges[-1] + 1e-12
    # Collapse degenerate ties so searchsorted doesn't pile everything into one
    # bin when many quantiles coincide.
    for k in range(1, len(edges)):
        if edges[k] <= edges[k - 1]:
            edges[k] = edges[k - 1] + 1e-12
    return edges


def _bin_features(features: pd.DataFrame, edges: dict[str, np.ndarray]) -> np.ndarray:
    """Bin features into integer bin indices. NaN rows -> -1 sentinel.

    Bin edges are fit on training features only. Test features beyond the
    training support are clamped to the edge bins via ``np.clip``.
    """
    n = len(features)
    binned = np.full((n, len(edges)), -1, dtype=int)
    for k, (col, e) in enumerate(edges.items()):
        v = features[col].to_numpy()
        valid = ~np.isnan(v)
        n_bins_k = len(e) - 1
        idx = np.searchsorted(e[1:-1], v[valid], side="right")
        binned[valid, k] = np.clip(idx, 0, n_bins_k - 1)
    return binned


def _wrap_path(
    actions: np.ndarray,
    prices_test: pd.DataFrame,
    *,
    initial_position: float,
    c0: float,
    lam: float,
    vol_window: int,
) -> dict[str, pd.Series]:
    """Compute rewards for the chosen test actions and assemble the return dict."""
    rewards = step_reward(
        prices_test, actions,
        c0=c0, lam=lam, vol_window=vol_window,
        initial_position=initial_position,
    ).rename("rewards")
    actions_s = pd.Series(actions, index=prices_test.index, name="actions")
    cum = rewards.cumsum().rename("cum_log_return")
    return {
        "actions": actions_s,
        "positions": actions_s.rename("positions"),
        "rewards": rewards,
        "cum_log_return": cum,
    }


# ---------------------------------------------------------------------------
# Tabular MC-regression Q baseline
# ---------------------------------------------------------------------------

def tabular_q_path(
    prices_train: pd.DataFrame,
    features_train: pd.DataFrame,
    prices_test: pd.DataFrame,
    features_test: pd.DataFrame,
    *,
    seed: int,
    n_actions: int = 3,
    n_bins: int = 4,
    gamma: float = GAMMA_DEFAULT,
    H: int = H_DEFAULT,
    mode: str = "mc_regression",
    action_labels: tuple[int, ...] = ACTIONS,
    initial_position: float = 0.0,
    c0: float = C0_DEFAULT,
    lam: float = LAMBDA_DEFAULT,
    vol_window: int = VOL_WINDOW_DEFAULT,
    # td_streaming-only knobs (reserved for nb-09 sweep)
    alpha: float = 0.05,
    alpha_schedule: str = "constant",
    epsilon_init: float = 1.0,
    epsilon_min: float = 0.05,
) -> dict[str, pd.Series]:
    """Tabular Q-baseline by empirical-mean Bellman regression on MC targets.

    Headline mode (``mode="mc_regression"``):
      1. Behaviour policy: uniform-random over ``n_actions`` on training
         features (matches the Bayesian sampler's training-data assumption).
      2. Run the env to get per-step rewards under the behaviour actions.
      3. Compute MC targets ``y_t`` via :func:`project.targets.mc_returns`.
      4. Bucket each binnable training feature into ``n_bins`` quantile bins
         (constant features such as the intercept are dropped from the joint
         state).
      5. For each ``(state-bin tuple, action)`` cell, ``Q(s, a) = mean(y_t)``
         over training rows in that cell. Empty cells fall back to the
         per-action marginal mean of ``y_t``.
      6. At test time, bucket each test feature row using the train-fit bin
         edges (clamped to the edge bins outside the training support), then
         pick ``argmax_a Q(s, a)``. NaN-feature warmup rows get action 0
         (neutral), matching :func:`random_policy_path`.

    Sweep mode (``mode="td_streaming"``): online TD-Q with epsilon-greedy
    exploration is reserved for the nb-09 sensitivity sweep and currently
    raises ``NotImplementedError`` to keep the headline contract clean.

    Parameters
    ----------
    prices_train, features_train
        Training inputs. Features are expected to share their index with
        ``prices_train``; rows with any NaN are skipped during fit.
    prices_test, features_test
        Test inputs. Returned actions are indexed by ``prices_test.index``.
    seed
        Required for reproducibility. Threads through the behaviour-policy
        RNG only (the regression itself is deterministic).
    n_actions
        Default 3 (matches :data:`project.env.ACTIONS`).
    n_bins
        Default 4 quantile bins per binnable feature; n_bins**(n_binnable)
        joint states.
    gamma, H
        MC-target discount and finite horizon. Default to
        :data:`project.targets.GAMMA_DEFAULT` and :data:`H_DEFAULT` so that
        the classical and Bayesian baselines see the same effective horizon.
    mode
        ``"mc_regression"`` (headline) or ``"td_streaming"`` (nb-09 sweep,
        currently raises).
    action_labels
        Mapping from action ids to env labels; default ``(-1, 0, +1)``.
    initial_position
        Position the agent enters the first day with. The headline default
        is 0 (neutral) to match :func:`random_policy_path` and
        :func:`project.rollout.simulate_path`.
    c0, lam, vol_window
        Forwarded to :func:`project.env.step_reward`.
    alpha, alpha_schedule, epsilon_init, epsilon_min
        Reserved for the ``td_streaming`` sweep; ignored in the headline mode.
    """
    if mode not in ("mc_regression", "td_streaming"):
        raise ValueError(f"mode must be 'mc_regression' or 'td_streaming', got {mode!r}")
    if len(action_labels) != n_actions:
        raise ValueError(
            f"action_labels has {len(action_labels)} entries but n_actions={n_actions}"
        )
    if mode == "td_streaming":
        raise NotImplementedError(
            "tabular_q_path mode='td_streaming' is reserved for the nb-09 "
            "TD-bootstrap sweep; use the default mode='mc_regression' for the "
            "headline comparison."
        )

    rng = np.random.default_rng(seed)
    behavior_a, y = _generate_training_data(
        prices_train,
        rng=rng,
        n_actions=n_actions,
        action_labels=action_labels,
        gamma=gamma, H=H,
        initial_position=initial_position,
        c0=c0, lam=lam, vol_window=vol_window,
    )

    # Identify binnable training features (non-constant). The intercept
    # column drops out of the joint state by design.
    binnable_cols = [
        c for c in features_train.columns
        if features_train[c].dropna().std(ddof=0) > 0
    ]
    if not binnable_cols:
        raise ValueError("No features have nonzero variance in training data")

    edges = {
        c: _quantile_bin_edges(features_train[c].to_numpy(), n_bins)
        for c in binnable_cols
    }
    train_bins = _bin_features(features_train[binnable_cols], edges)

    # Encode the joint state as a single integer for fast aggregation.
    n_binnable = len(binnable_cols)
    multipliers = (n_bins ** np.arange(n_binnable)).astype(np.int64)
    n_states = int(n_bins ** n_binnable)
    train_state_ids = (train_bins.astype(np.int64) * multipliers).sum(axis=1)

    # Common rows: features valid AND y_t valid.
    feat_valid = (train_bins >= 0).all(axis=1)
    y_full = y.reindex(prices_train.index)
    y_arr = y_full.to_numpy()
    y_valid = ~np.isnan(y_arr)
    common = feat_valid & y_valid

    sum_y = np.zeros((n_states, n_actions))
    cnt = np.zeros((n_states, n_actions), dtype=np.int64)
    np.add.at(sum_y, (train_state_ids[common], behavior_a[common]), y_arr[common])
    np.add.at(cnt, (train_state_ids[common], behavior_a[common]), 1)

    # Per-action marginal as fall-back for empty (state, action) cells.
    marginal = np.zeros(n_actions)
    for a in range(n_actions):
        mask_a = (behavior_a == a) & common
        if mask_a.any():
            marginal[a] = float(y_arr[mask_a].mean())

    q_table = np.where(cnt > 0, sum_y / np.maximum(cnt, 1), marginal[None, :])

    # Greedy on test.
    test_bins = _bin_features(features_test[binnable_cols], edges)
    n_test = len(prices_test)
    test_actions = np.zeros(n_test, dtype=int)              # warmup default: 0
    test_valid = (test_bins >= 0).all(axis=1)
    if test_valid.any():
        test_state_ids = (test_bins[test_valid].astype(np.int64) * multipliers).sum(axis=1)
        greedy_id = q_table[test_state_ids].argmax(axis=1)
        test_actions[test_valid] = np.asarray(action_labels, dtype=int)[greedy_id]

    return _wrap_path(
        test_actions, prices_test,
        initial_position=initial_position, c0=c0, lam=lam, vol_window=vol_window,
    )


# ---------------------------------------------------------------------------
# Linear FQI baseline
# ---------------------------------------------------------------------------

def linear_fqi_path(
    prices_train: pd.DataFrame,
    features_train: pd.DataFrame,
    prices_test: pd.DataFrame,
    features_test: pd.DataFrame,
    *,
    seed: int,
    n_actions: int = 3,
    gamma: float = GAMMA_DEFAULT,
    H: int = H_DEFAULT,
    mode: str = "mc_regression",
    feature_cols: tuple[str, ...] = FEATURE_COLS,
    action_labels: tuple[int, ...] = ACTIONS,
    initial_position: float = 0.0,
    c0: float = C0_DEFAULT,
    lam: float = LAMBDA_DEFAULT,
    vol_window: int = VOL_WINDOW_DEFAULT,
    # td_streaming-only knobs (reserved for nb-09 sweep)
    alpha: float = 0.05,
    alpha_schedule: str = "constant",
    epsilon_init: float = 1.0,
    epsilon_min: float = 0.05,
) -> dict[str, pd.Series]:
    """Linear FQI baseline: ``Q(s, a) = phi(s)^T w_a`` fit by OLS on MC targets.

    Headline mode (``mode="mc_regression"``):
      1. Same uniform-random behaviour policy on training data as the
         tabular baseline.
      2. Compute MC targets ``y_t``.
      3. For each action ``a``, fit ``w_a = argmin_w sum_t (y_t - phi_t^T w)^2``
         over training rows where the behaviour action equalled ``a``. Solved
         via :func:`numpy.linalg.lstsq` (handles rank-deficient design without
         silently returning NaNs). An action with no training samples gets
         ``w_a = 0``, so its Q-value is identically 0 (documented; should not
         occur for ``n_train > a few hundred`` with uniform behaviour).
      4. At test time, ``argmax_a phi_t^T w_a``. NaN-feature warmup rows get
         action 0, matching :func:`random_policy_path`.

    Feature-column alignment is enforced: if either ``features_train.columns``
    or ``features_test.columns`` differs from ``feature_cols`` (default
    :data:`project.features.FEATURE_COLS`), this raises ``ValueError``. This
    mirrors :func:`project.rollout.simulate_path`'s check on
    ``policy.feature_cols`` -- silent column re-ordering would point the FQI
    weights at different semantic features than the Bayesian model's
    ``beta_a`` and break the fairness comparison.

    Sweep mode (``mode="td_streaming"``): reserved for nb-09; raises
    ``NotImplementedError`` to keep the headline contract clean.
    """
    if mode not in ("mc_regression", "td_streaming"):
        raise ValueError(f"mode must be 'mc_regression' or 'td_streaming', got {mode!r}")
    if len(action_labels) != n_actions:
        raise ValueError(
            f"action_labels has {len(action_labels)} entries but n_actions={n_actions}"
        )
    expected = list(feature_cols)
    if list(features_train.columns) != expected:
        raise ValueError(
            f"features_train.columns {list(features_train.columns)} do not match "
            f"feature_cols {expected}"
        )
    if list(features_test.columns) != expected:
        raise ValueError(
            f"features_test.columns {list(features_test.columns)} do not match "
            f"feature_cols {expected}"
        )
    if mode == "td_streaming":
        raise NotImplementedError(
            "linear_fqi_path mode='td_streaming' is reserved for the nb-09 "
            "TD-bootstrap sweep; use the default mode='mc_regression' for the "
            "headline comparison."
        )

    rng = np.random.default_rng(seed)
    behavior_a, y = _generate_training_data(
        prices_train,
        rng=rng,
        n_actions=n_actions,
        action_labels=action_labels,
        gamma=gamma, H=H,
        initial_position=initial_position,
        c0=c0, lam=lam, vol_window=vol_window,
    )

    feat_arr = features_train.to_numpy()
    feat_valid = ~np.isnan(feat_arr).any(axis=1)
    y_full = y.reindex(prices_train.index)
    y_arr = y_full.to_numpy()
    y_valid = ~np.isnan(y_arr)
    common = feat_valid & y_valid

    X = feat_arr[common]
    y_c = y_arr[common]
    a_c = behavior_a[common]
    p = X.shape[1]

    weights = np.zeros((n_actions, p))
    for a in range(n_actions):
        mask = a_c == a
        # Need at least p observations to fit p coefficients; otherwise leave
        # weights at zero (Q == 0 for that action).
        if mask.sum() < p:
            continue
        Xa = X[mask]
        ya = y_c[mask]
        w_a, *_ = np.linalg.lstsq(Xa, ya, rcond=None)
        weights[a] = w_a

    test_arr = features_test.to_numpy()
    test_valid = ~np.isnan(test_arr).any(axis=1)
    n_test = len(prices_test)
    test_actions = np.zeros(n_test, dtype=int)              # warmup default: 0
    if test_valid.any():
        Q_test = test_arr[test_valid] @ weights.T           # (n_valid, n_actions)
        greedy_id = Q_test.argmax(axis=1)
        test_actions[test_valid] = np.asarray(action_labels, dtype=int)[greedy_id]

    return _wrap_path(
        test_actions, prices_test,
        initial_position=initial_position, c0=c0, lam=lam, vol_window=vol_window,
    )


# ---------------------------------------------------------------------------
# Multi-seed orchestration: classical analogue of posterior_predictive_metrics
# ---------------------------------------------------------------------------

_PolicyFn = Callable[..., dict[str, pd.Series]]


def _percentile_bootstrap_ci(
    values: np.ndarray,
    *,
    n_boot: int,
    rng: np.random.Generator,
    ci: float = 0.95,
) -> dict[str, float]:
    """Mean + percentile bootstrap CI on the *mean* of ``values``."""
    n = len(values)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = float(values[idx].mean())
    alpha = (1.0 - ci) / 2.0
    return {
        "mean": float(values.mean()),
        "ci_lo": float(np.quantile(boot_means, alpha)),
        "ci_hi": float(np.quantile(boot_means, 1.0 - alpha)),
    }


def classical_baseline_distribution(
    fn: _PolicyFn,
    prices_train: pd.DataFrame,
    features_train: pd.DataFrame,
    prices_test: pd.DataFrame,
    features_test: pd.DataFrame,
    *,
    n_seeds: int = 20,
    base_seed: int = 0,
    n_boot: int = 2000,
    **fn_kwargs,
) -> dict[str, dict[str, float]]:
    """Run a classical baseline ``n_seeds`` times and report mean + bootstrap CI
    on Sharpe, max drawdown, and turnover.

    Parameters
    ----------
    fn
        Any baseline with the signature
        ``fn(prices_train, features_train, prices_test, features_test, *, seed, **kw)``
        returning the standard ``{actions, positions, rewards, cum_log_return}``
        dict (see :func:`tabular_q_path`, :func:`linear_fqi_path`).
    prices_train, features_train, prices_test, features_test
        Forwarded to ``fn`` once per seed.
    n_seeds
        Number of independent runs. 20 is the default for sensitivity sweeps;
        bump to 100 for headline figures. Higher ``n_seeds`` tightens the CI
        on each metric.
    base_seed
        Seeds for ``fn`` are ``range(base_seed, base_seed + n_seeds)``. The
        bootstrap RNG is derived deterministically from ``base_seed`` so that
        identical inputs reproduce identical CIs.
    n_boot
        Number of bootstrap resamples for each metric's mean CI. Default
        2000; the 95% CI is the 2.5/97.5 percentile of the bootstrap mean.
    fn_kwargs
        Forwarded to ``fn`` on every call.

    Returns
    -------
    dict
        ``{"sharpe": {"mean", "ci_lo", "ci_hi"},
           "max_drawdown": {...}, "turnover": {...}}``. Mirrors the nested-dict
        shape of :func:`project.eval.posterior_predictive_metrics` so plotting
        and tabulation code can stay branch-free across Bayesian and classical
        baselines.
    """
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be >= 1, got {n_seeds}")

    seeds = list(range(int(base_seed), int(base_seed) + int(n_seeds)))
    sharpes = np.empty(n_seeds)
    mdds = np.empty(n_seeds)
    turns = np.empty(n_seeds)

    for i, s in enumerate(seeds):
        out = fn(
            prices_train, features_train, prices_test, features_test,
            seed=int(s), **fn_kwargs,
        )
        sharpes[i] = sharpe(out["rewards"])
        mdds[i] = max_drawdown(out["cum_log_return"])
        turns[i] = turnover(out["actions"])

    # Bootstrap RNG: deterministic from base_seed (separate stream from fn seeds).
    boot_ss = np.random.SeedSequence(int(base_seed))
    boot_rng = np.random.default_rng(boot_ss.spawn(1)[0])
    return {
        "sharpe": _percentile_bootstrap_ci(sharpes, n_boot=n_boot, rng=boot_rng),
        "max_drawdown": _percentile_bootstrap_ci(mdds, n_boot=n_boot, rng=boot_rng),
        "turnover": _percentile_bootstrap_ci(turns, n_boot=n_boot, rng=boot_rng),
    }
