"""Posterior-sampled rollout of a :class:`ThompsonPolicy` on a price series.

A *rollout* applies the policy to a feature stream day by day, records the
chosen action, and returns the per-day reward path under
:func:`project.env.step_reward`. The default regime is per-episode posterior
sampling (Osband-style PSRL): a single posterior index is drawn once at
rollout start and held fixed for the whole window. ``resample_every=k``
redraws every k valid steps; ``resample_every=1`` recovers per-step
Thompson.

Decision-time contract:
    actions[t] is chosen using features.iloc[t]; the reward earned is
    a_t * r_{t+1} - c_t * |a_t - a_{t-1}|, computed by
    :func:`project.env.step_reward`. Warmup rows (any NaN in the feature
    row) default to action 0, the neutral position. The first valid step
    pays a transition cost off ``initial_position`` per the env model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from project.env import (
    ACTIONS,
    C0_DEFAULT,
    LAMBDA_DEFAULT,
    VOL_WINDOW_DEFAULT,
    step_reward,
)
from project.thompson import ThompsonPolicy, thompson_action


def simulate_path(
    policy: ThompsonPolicy,
    prices: pd.DataFrame,
    features: pd.DataFrame,
    *,
    seed: int,
    resample_every: int | None = None,
    action_labels: tuple[int, ...] = ACTIONS,
    initial_position: float = 0.0,
    c0: float = C0_DEFAULT,
    lam: float = LAMBDA_DEFAULT,
    vol_window: int = VOL_WINDOW_DEFAULT,
) -> dict[str, pd.Series]:
    """Roll the Thompson policy out over a price series.

    Parameters
    ----------
    policy
        Fitted :class:`ThompsonPolicy`.
    prices
        DataFrame with ``"Close"``, indexed by trading day.
    features
        DataFrame whose columns equal ``policy.feature_cols`` and whose
        index aligns with ``prices.index`` (or is a superset of the
        valid rollout window). Rows with any NaN are treated as warmup
        and skipped.
    seed
        Required. Seeds an :class:`numpy.random.Generator` used for both
        the posterior-index draw(s) and (in per-step mode) every
        decision.
    resample_every
        ``None``  -> per-episode (one posterior draw, fixed for whole rollout)
        ``1``     -> per-step Thompson (redraw every valid step)
        ``k > 1`` -> redraw every k valid steps
    action_labels
        Mapping from policy action ids ``range(n_actions)`` to env labels.
        Length must equal ``policy.n_actions``.
    initial_position
        Position the agent enters the first day with; per
        :func:`project.env.step_reward` this controls the day-0 cost.
    c0, lam, vol_window
        Cost-formula parameters; defaults match :mod:`project.env`.

    Returns
    -------
    dict of pandas.Series
        Keys ``"actions"``, ``"positions"`` (identical -- the action *is*
        the held position from t to t+1 by the env's convention),
        ``"rewards"``, ``"cum_log_return"``. All four are indexed by
        ``prices.index``. ``rewards`` and ``cum_log_return`` carry NaN at
        the right edge (no ``r_{t+1}``) and at the left edge (cost
        warmup), matching :func:`project.env.step_reward`'s contract.
    """
    if seed is None:
        raise ValueError("seed is required for reproducibility")
    if len(action_labels) != policy.n_actions:
        raise ValueError(
            f"action_labels has {len(action_labels)} entries but policy.n_actions={policy.n_actions}"
        )
    if resample_every is not None and resample_every < 1:
        raise ValueError(f"resample_every must be >= 1 or None, got {resample_every}")
    expected_cols = list(policy.feature_cols)
    if list(features.columns) != expected_cols:
        raise ValueError(
            f"features.columns {list(features.columns)} do not match "
            f"policy.feature_cols {expected_cols}"
        )

    rng = np.random.default_rng(seed)
    aligned = features.reindex(prices.index)
    valid_mask = aligned.notna().all(axis=1).to_numpy()
    feat_arr = aligned.to_numpy()
    n = len(prices)

    actions = np.zeros(n, dtype=int)         # warmup default: neutral position 0

    fixed_idx: int | None = None
    if resample_every is None:
        fixed_idx = int(rng.integers(0, policy.n_draws_total))

    valid_step = 0
    for i in range(n):
        if not valid_mask[i]:
            continue
        if resample_every is not None and valid_step % resample_every == 0:
            fixed_idx = int(rng.integers(0, policy.n_draws_total))
        a_id = thompson_action(policy, feat_arr[i], rng, draw_idx=fixed_idx)
        actions[i] = action_labels[a_id]
        valid_step += 1

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
