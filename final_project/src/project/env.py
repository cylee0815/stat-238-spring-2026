"""Trading environment: vol-dependent transaction cost and per-step reward.

Implements the proposal's section 5 reward model:

    c_t   = c_0 + lambda * sigma_t^(20)        (eq 21)
    R_t   = a_t * r_{t+1} - c_t * |a_t - a_{t-1}|   (eq 22)

with ``r_{t+1} = log(P_{t+1}/P_t)``. The cost is computed from a strictly
causal 20-day rolling realized vol of returns, matching the convention
in :mod:`project.features`.

The reward is the per-step quantity that feeds the Monte Carlo return in
:mod:`project.targets`. There is no Bellman bootstrap here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

C0_DEFAULT: float = 0.0005      # 5 basis points
LAMBDA_DEFAULT: float = 0.5
VOL_WINDOW_DEFAULT: int = 20

ACTIONS: tuple[int, ...] = (-1, 0, 1)


def transaction_cost(
    prices: pd.DataFrame,
    *,
    c0: float = C0_DEFAULT,
    lam: float = LAMBDA_DEFAULT,
    vol_window: int = VOL_WINDOW_DEFAULT,
) -> pd.Series:
    """Per-day cost ``c_t = c_0 + lambda * sigma_t^(20)``.

    ``sigma_t`` is the 20-day rolling std of log returns over the strictly
    past window ``r_{t-vol_window}..r_{t-1}``. The first ``vol_window``
    rows are NaN (insufficient history).
    """
    p = prices["Close"].astype(float)
    r = np.log(p).diff()
    sigma_t = r.shift(1).rolling(vol_window).std()
    return c0 + lam * sigma_t


def step_reward(
    prices: pd.DataFrame,
    actions: np.ndarray | pd.Series | list[int],
    *,
    c0: float = C0_DEFAULT,
    lam: float = LAMBDA_DEFAULT,
    vol_window: int = VOL_WINDOW_DEFAULT,
    initial_position: float = 0.0,
) -> pd.Series:
    """Per-step reward ``R_t = a_t * r_{t+1} - c_t * |a_t - a_{t-1}|``.

    Parameters
    ----------
    prices
        DataFrame with ``"Close"``, indexed by trading day.
    actions
        Integer action per row, taken from :data:`ACTIONS`. Length must
        match ``len(prices)``. Action at row ``t`` is the position the
        agent chose at the end of day ``t``, which earns the next-day
        log return ``r_{t+1}``.
    c0, lam, vol_window
        Cost-formula parameters; defaults match proposal section 5.
    initial_position
        Position the agent is assumed to enter the first day with. The
        first-day cost is ``|a_0 - initial_position|``.

    Returns
    -------
    pandas.Series
        Per-day reward, same index as ``prices``. The last day has
        ``NaN`` because ``r_{t+1}`` is undefined; the first
        ``vol_window`` days are NaN because ``c_t`` has insufficient
        history. Drop NaN before computing MC returns.
    """
    p = prices["Close"].astype(float)
    next_r = np.log(p).diff().shift(-1)

    a = pd.Series(np.asarray(actions, dtype=float), index=prices.index)
    if len(a) != len(prices):
        raise ValueError(
            f"actions length {len(a)} != prices length {len(prices)}"
        )
    a_prev = a.shift(1).fillna(initial_position)

    c_t = transaction_cost(prices, c0=c0, lam=lam, vol_window=vol_window)
    return a * next_r - c_t * np.abs(a - a_prev)
