"""Strictly causal rolling features (proposal section 5).

Every feature at time ``t`` may use information from time ``t`` and earlier
only. The leakage smoke test in ``tests/test_features_causality.py``
enforces this contract: scrambling prices at indices ``>= k`` must leave
features at indices ``< k`` unchanged.

Convention. Per proposal eq (20), 5-day momentum is
``sum_{i=1}^5 r_{t-i}`` -- strictly past returns, excluding ``r_t``. We
apply the same shift to the realized-vol calculation for symmetry. The
200-day MA at time ``t`` is allowed to include ``P_t`` itself, since
``P_t`` is already observed at decision time ``t``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_MOM_WINDOW: int = 5
DEFAULT_VOL_WINDOW: int = 20
DEFAULT_MA_WINDOW: int = 200
DEFAULT_Z_WINDOW: int = 252

FEATURE_COLS: tuple[str, ...] = (
    "x1_mom",
    "x2_logvol",
    "x3_p_vs_ma",
    "x4_intercept",
)


def build_features(
    prices: pd.DataFrame,
    *,
    mom_window: int = DEFAULT_MOM_WINDOW,
    vol_window: int = DEFAULT_VOL_WINDOW,
    ma_window: int = DEFAULT_MA_WINDOW,
    z_window: int = DEFAULT_Z_WINDOW,
) -> pd.DataFrame:
    """Build the four-column feature matrix from adjusted closes.

    Parameters
    ----------
    prices
        DataFrame with a ``"Close"`` column of adjusted closes, indexed by
        trading day.

    Returns
    -------
    pandas.DataFrame
        Same index as ``prices``, columns ``("x1_mom", "x2_logvol",
        "x3_p_vs_ma", "x4_intercept")``. Early rows are ``NaN`` until
        every rolling window is filled; callers are expected to drop NaN
        rows before fitting.
    """
    p = prices["Close"].astype(float)
    log_p = np.log(p)
    r = log_p.diff()

    # x1: 5-day momentum, z-scored over a trailing 252-day window.
    # Per eq (20): sum runs over r_{t-mom_window}..r_{t-1} (strictly past).
    mom = r.shift(1).rolling(mom_window).sum()
    mom_mu = mom.rolling(z_window).mean()
    mom_sd = mom.rolling(z_window).std()
    x1 = (mom - mom_mu) / mom_sd

    # x2: log of trailing 20-day realized vol, mean-centred over 252 days.
    realized_vol = r.shift(1).rolling(vol_window).std()
    log_vol = np.log(realized_vol)
    x2 = log_vol - log_vol.rolling(z_window).mean()

    # x3: deviation of P_t from its trailing 200-day mean (relative).
    ma = p.rolling(ma_window).mean()
    x3 = (p - ma) / ma

    # x4: intercept.
    x4 = pd.Series(1.0, index=p.index)

    return pd.DataFrame(
        {"x1_mom": x1, "x2_logvol": x2, "x3_p_vs_ma": x3, "x4_intercept": x4}
    )


def feature_warmup(
    *,
    mom_window: int = DEFAULT_MOM_WINDOW,
    vol_window: int = DEFAULT_VOL_WINDOW,
    ma_window: int = DEFAULT_MA_WINDOW,
    z_window: int = DEFAULT_Z_WINDOW,
) -> int:
    """Number of leading rows guaranteed to be NaN for at least one feature.

    The binding constraint is the slowest feature: x1 needs ``z_window``
    momentum samples, each of which needs ``mom_window`` past returns
    plus the shift(1), and similarly for x2. x3 needs ``ma_window``. The
    return is a safe upper bound on rows to drop after ``build_features``.
    """
    return max(z_window + mom_window, z_window + vol_window, ma_window) + 1
