"""Monte Carlo return targets (proposal eq 1).

Constructs

    y_t = sum_{k=0}^{H} gamma^k * R_{t+k}

with NO temporal-difference bootstrap and NO downstream Q plug-in. Targets
at the right edge of the sample are dropped: the constructor refuses to
emit ``y_t`` unless every reward in the window is observed.

The horizon ``H`` and discount ``gamma`` are fixed by the proposal at
``H in {20, 60}`` and ``gamma = 0.95``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GAMMA_DEFAULT: float = 0.95
H_DEFAULT: int = 60


def mc_returns(
    rewards: pd.Series,
    *,
    gamma: float = GAMMA_DEFAULT,
    H: int = H_DEFAULT,
) -> pd.Series:
    """Compute truncated MC returns from a per-step reward series.

    Parameters
    ----------
    rewards
        Per-step reward indexed by trading day. May contain NaN at the
        head (cost warmup) and at the tail (no ``r_{t+1}``); both are
        handled by skipping any window that contains NaN.
    gamma, H
        Discount factor and finite horizon (in days).

    Returns
    -------
    pandas.Series
        ``y_t`` indexed by the same dates as ``rewards``, with rows for
        which the window ``[t, t+H]`` is not fully observed dropped.
    """
    if H < 0:
        raise ValueError(f"H must be non-negative, got {H}")
    R = rewards.to_numpy(dtype=float)
    n = len(R)
    if n <= H:
        raise ValueError(f"rewards length {n} too short for H={H}")

    weights = gamma ** np.arange(H + 1)
    y = np.full(n, np.nan)
    for t in range(n - H):
        window = R[t : t + H + 1]
        if np.isnan(window).any():
            continue
        y[t] = float(np.dot(weights, window))

    return pd.Series(y, index=rewards.index).dropna()


def truncation_bias_bound(*, gamma: float = GAMMA_DEFAULT, H: int = H_DEFAULT) -> float:
    """Worst-case fractional bias of the H-truncated estimator vs the infinite-horizon
    geometric sum, assuming bounded per-step rewards. Equals ``gamma**(H+1)``."""
    return float(gamma ** (H + 1))
