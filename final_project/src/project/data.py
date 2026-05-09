"""Price-data download and caching.

Pulls daily prices for SPY (and TLT, the out-of-domain robustness check)
from Yahoo Finance via ``yfinance`` and caches them as parquet under
``data/raw/`` so notebooks do not re-download. All downstream code reads
the adjusted close from the ``"Close"`` column (yfinance's
``auto_adjust=True`` mode places adjusted prices there).

Splits per the proposal:
    pre-training: 1990-01-01 -> 1992-12-31  (calibrate sigma_y only)
    training:     1994-01-01 -> 2019-12-31  (Gibbs + policy iteration)
    test:         2020-01-01 -> 2026-04-30  (posterior predictive eval)

SPY launched 1993-01-22, so the pre-training window cannot use SPY itself.
We substitute ``^GSPC`` (the S&P 500 index that SPY tracks) for the prior
elicitation window. The substitution is sealed strictly before SPY's first
trading day, so no test-time information leaks into the prior.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from project.utils import RAW_DIR

SYMBOLS: tuple[str, ...] = ("SPY", "TLT")

SPY_START: str = "1993-01-22"
SPY_END: str = "2026-04-30"

PRETRAIN_SYMBOL: str = "^GSPC"
PRETRAIN_START: str = "1990-01-01"
PRETRAIN_END: str = "1992-12-31"

TRAIN_START: str = "1994-01-01"
TRAIN_END: str = "2019-12-31"
TEST_START: str = "2020-01-01"
TEST_END: str = "2026-04-30"


def _safe_filename(symbol: str) -> str:
    return symbol.lstrip("^")


def _cache_path(symbol: str, start: str, end: str) -> Path:
    return RAW_DIR / f"{_safe_filename(symbol)}_{start}_{end}.parquet"


def download_prices(
    symbol: str,
    *,
    start: str,
    end: str,
    force: bool = False,
) -> pd.DataFrame:
    """Download daily OHLCV for ``symbol`` and cache as parquet.

    Parameters
    ----------
    symbol
        Yahoo Finance ticker, e.g. ``"SPY"`` or ``"^GSPC"``.
    start, end
        ISO date strings. The end date is exclusive in yfinance's API, so
        pass the day *after* the last trading day you want.
    force
        If True, refresh the cache even if it exists.

    Returns
    -------
    pandas.DataFrame
        Daily OHLCV, indexed by ``DatetimeIndex`` named ``Date``. Columns
        come from yfinance with ``auto_adjust=True``, so ``"Close"`` is
        the split- and dividend-adjusted close.
    """
    out = _cache_path(symbol, start, end)
    if out.exists() and not force:
        return pd.read_parquet(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        raise RuntimeError(f"yfinance returned an empty frame for {symbol!r}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index.name = "Date"
    df.to_parquet(out)
    return df


def load_spy(*, force: bool = False) -> pd.DataFrame:
    """Load SPY 1993-01-22 -> 2026-04-30 (cached)."""
    return download_prices(
        "SPY", start=SPY_START, end=SPY_END, force=force,
    )


def load_tlt(*, force: bool = False) -> pd.DataFrame:
    """Load TLT 1993-01-22 -> 2026-04-30 (cached). Used for the robustness check."""
    return download_prices(
        "TLT", start=SPY_START, end=SPY_END, force=force,
    )


def load_pretrain(*, force: bool = False) -> pd.DataFrame:
    """Load ^GSPC 1990-01-01 -> 1992-12-31 for prior elicitation.

    SPY launched 1993-01-22, so the proposal's pre-1994 calibration window
    cannot be filled by SPY itself. ``^GSPC`` is the S&P 500 index that
    SPY tracks; ~750 trading days across the 1990 recession and the
    1991-92 recovery yield a more representative variance estimate than a
    single low-vol year would. The window closes strictly before SPY's
    first trading day, so no test-time information leaks into the prior.
    """
    return download_prices(
        PRETRAIN_SYMBOL,
        start=PRETRAIN_START,
        end=PRETRAIN_END,
        force=force,
    )


def split_windows(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Slice a SPY (or TLT) frame into the proposal's train and test windows.

    The pre-training window is filled by ``load_pretrain()`` from a
    different ticker (``^GSPC``), so it is intentionally not a slice of
    ``df``.

    Returns
    -------
    dict
        Keys ``"train"``, ``"test"`` mapping to disjoint slices of ``df``.
        Slices are inclusive of both endpoints.
    """
    return {
        "train": df.loc[TRAIN_START:TRAIN_END],
        "test": df.loc[TEST_START:TEST_END],
    }
