"""Posterior-predictive evaluation metrics.

All scalar metrics here operate on a single realized path. The Bayesian
flavour comes from :func:`posterior_predictive_metrics`, which draws
``n_paths`` posterior rollouts via :func:`project.rollout.simulate_path`
and reports each scalar metric as a posterior predictive distribution
(``mean``, ``q025``, ``q975``).

Conventions:
- Sharpe is annualized via ``ann_factor=252`` (daily data).
- Max drawdown is reported as a non-negative magnitude in log-return
  space: the largest peak-to-trough drop of ``cum_log_return``.
- Pseudo-regret is computed against the best fixed action *in
  hindsight*: argmax over candidate actions of total realized reward.
  No oracle Q-function is consulted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from project.rollout import simulate_path
from project.thompson import ThompsonPolicy


# ---------------------------------------------------------------------------
# Per-path scalar metrics
# ---------------------------------------------------------------------------

def cumulative_log_return(rewards: pd.Series) -> pd.Series:
    """Running sum of per-step log-rewards. NaN entries are propagated as 0
    contributions but kept positionally; downstream code typically calls
    ``.dropna()`` before plotting."""
    return rewards.cumsum().rename("cum_log_return")


def sharpe(rewards: pd.Series, *, ann_factor: int = 252) -> float:
    """Annualized Sharpe ratio: ``mean / std * sqrt(ann_factor)``.

    NaN entries are dropped. Returns 0.0 if the std is zero (no
    variability) or fewer than two finite samples remain.
    """
    r = rewards.dropna().to_numpy()
    if r.size < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd == 0.0:
        return 0.0
    return float(r.mean() / sd * np.sqrt(ann_factor))


def max_drawdown(cum_log_return: pd.Series) -> float:
    """Magnitude of the worst peak-to-trough drop in cumulative log return.

    Returns a non-negative float. ``0.0`` for a monotonically non-decreasing
    path. NaN is dropped before the running-max comparison.
    """
    c = cum_log_return.dropna().to_numpy()
    if c.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(c)
    drawdown = running_max - c
    return float(drawdown.max())


def turnover(actions: pd.Series) -> float:
    """Mean ``|a_t - a_{t-1}|`` across the path. NaN-safe via ``.dropna()``."""
    return float(actions.diff().abs().dropna().mean())


def action_frequency(actions: pd.Series) -> dict[int, float]:
    """Proportion of time each action label was held. Keys are the unique
    action values (cast to ``int``); values sum to 1.0."""
    counts = actions.value_counts(normalize=True, sort=False)
    return {int(k): float(v) for k, v in counts.items()}


def pseudo_regret_vs_best_fixed(rewards_per_action: dict) -> pd.Series:
    """Per-step pseudo-regret of the chosen path vs the best fixed action.

    Parameters
    ----------
    rewards_per_action
        Dict containing:
        - ``"chosen"`` -> the realized reward path of the policy under test.
        - one or more candidate keys (action labels) -> the reward path of a
          fixed-action policy that always plays that label.

    Returns
    -------
    pandas.Series
        ``rewards_per_action[best_a] - rewards_per_action["chosen"]``, where
        ``best_a`` maximises the *total* realized reward across the
        candidate keys (best fixed action in hindsight).
    """
    if "chosen" not in rewards_per_action:
        raise ValueError("rewards_per_action must include a 'chosen' key")
    candidates = {k: v for k, v in rewards_per_action.items() if k != "chosen"}
    if not candidates:
        raise ValueError("rewards_per_action needs at least one candidate action key")
    totals = {a: float(np.nansum(r.to_numpy())) for a, r in candidates.items()}
    best_a = max(totals, key=totals.__getitem__)
    return candidates[best_a] - rewards_per_action["chosen"]


# ---------------------------------------------------------------------------
# Posterior predictive aggregation
# ---------------------------------------------------------------------------

def _summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "q025": float(np.quantile(values, 0.025)),
        "q975": float(np.quantile(values, 0.975)),
    }


def posterior_predictive_metrics(
    policy: ThompsonPolicy,
    prices: pd.DataFrame,
    features: pd.DataFrame,
    *,
    n_paths: int = 500,
    seed: int,
    resample_every: int | None = None,
    initial_position: float = 0.0,
) -> dict[str, dict[str, float]]:
    """Posterior predictive distribution of the path-level scalar metrics.

    Simulates ``n_paths`` independent rollouts (per-episode by default --
    one fresh posterior index per path), computes Sharpe, max drawdown,
    and turnover on each, and reports the mean and 95% CrI of each.

    Parameters
    ----------
    policy, prices, features
        Forwarded to :func:`project.rollout.simulate_path`.
    n_paths
        Number of independent rollouts. 500 is the default for sensitivity
        runs; bump to 2000 for headline figures.
    seed
        Required. A :class:`numpy.random.SeedSequence` is spawned from this
        seed to derive ``n_paths`` independent path seeds, so the result is
        reproducible without correlating paths through a shared RNG.
    resample_every
        Forwarded to :func:`project.rollout.simulate_path`. ``None``
        (default) is per-episode PSRL.
    initial_position
        Forwarded to :func:`project.rollout.simulate_path`.

    Returns
    -------
    dict
        ``{"sharpe": {"mean": ..., "q025": ..., "q975": ...}, "max_drawdown": {...}, "turnover": {...}}``.
    """
    if n_paths < 1:
        raise ValueError(f"n_paths must be >= 1, got {n_paths}")
    if seed is None:
        raise ValueError("seed is required")

    seed_seq = np.random.SeedSequence(seed)
    path_seed_seqs = seed_seq.spawn(n_paths)

    sharpes = np.empty(n_paths)
    mdds = np.empty(n_paths)
    turns = np.empty(n_paths)

    for i, ss in enumerate(path_seed_seqs):
        path_seed = int(ss.generate_state(1)[0])
        out = simulate_path(
            policy, prices, features,
            seed=path_seed,
            resample_every=resample_every,
            initial_position=initial_position,
        )
        sharpes[i] = sharpe(out["rewards"])
        mdds[i] = max_drawdown(out["cum_log_return"])
        turns[i] = turnover(out["actions"])

    return {
        "sharpe": _summarize(sharpes),
        "max_drawdown": _summarize(mdds),
        "turnover": _summarize(turns),
    }
