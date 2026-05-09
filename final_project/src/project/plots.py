"""Paper-quality plotting helpers (foundation set).

This module is the *only* place plot styling lives. Notebooks call these
helpers and never instantiate matplotlib axes inline. Every function

* returns ``(fig, ax)`` (or ``(fig, axes_array)``); never calls
  :func:`matplotlib.pyplot.show`,
* takes a ``style: dict | None = None`` kwarg merged on top of
  :data:`DEFAULT_STYLE`; never mutates ``plt.rcParams``,
* uses the project-wide colour discipline: Bayesian = blue (``"C0"``),
  classical = orange (``"C1"``), random benchmark = grey (``"0.5"``).

The set is deliberately minimal -- six functions covering the data layer
only. Posterior, credible-interval, action-frequency, and diagnostics
plots land in later increments paired with the notebooks that consume
them.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from project.env import (
    C0_DEFAULT,
    LAMBDA_DEFAULT,
    VOL_WINDOW_DEFAULT,
    transaction_cost,
)
from project.features import FEATURE_COLS

DEFAULT_STYLE: dict = {
    # figure sizes
    "figsize_single": (9.0, 4.0),
    "figsize_wide": (10.0, 3.5),
    "figsize_panel": (10.0, 7.5),
    "figsize_dual": (10.0, 4.5),
    "dpi": 100,
    # font sizes
    "title_fontsize": 11,
    "label_fontsize": 9,
    "tick_fontsize": 8,
    "legend_fontsize": 8,
    # colour discipline (locked across the project)
    "color_bayesian": "C0",
    "color_classical": "C1",
    "color_random": "0.5",
    "color_data": "black",
    "color_pretrain": "C3",
    "color_overlay_normal": "C2",
    "color_overlay_t": "C3",
    "color_action_short": "C3",
    "color_action_flat": "0.5",
    "color_action_long": "C0",
    # line / hist parameters
    "linewidth": 0.9,
    "linewidth_thin": 0.6,
    "linewidth_overlay": 1.4,
    "linestyle_overlay_normal": "--",
    "linestyle_overlay_t": "-",
    "alpha_hist": 0.55,
    "alpha_span": 0.10,
    "alpha_drawdown": 0.35,
    "hist_bins": 80,
    # Per-policy palette consumed by the path-visualisation set
    # (plot_cumulative_return, plot_action_timeline, ...). Tabular and
    # linear-FQI share the classical bucket and differ by linestyle.
    "policy_styles": {
        "random":     {"color": "0.5", "linestyle": "-",  "label": "Random"},
        "tabular":    {"color": "C1",  "linestyle": "--", "label": "Tabular Q (MC)"},
        "linear_fqi": {"color": "C1",  "linestyle": "-",  "label": "Linear FQI (MC)"},
        "bayesian":   {"color": "C0",  "linestyle": "-",  "label": "Bayesian Q (Thompson)"},
    },
    # Per-sampler palette consumed by the diagnostic set (plot_trace,
    # plot_posterior_density, plot_posterior_overlay, ...). C2 here means
    # "Student-t sampler's posterior" -- a different semantics from
    # color_overlay_normal above, which paints the Normal-fit overlay on
    # the empirical-returns histogram in nb 00. The two charts never share
    # a figure (nb 00 talks about distributional fits to data; nb 03 talks
    # about which sampler produced a posterior), so a single colour can
    # carry both meanings safely. If a future notebook ever needs to put
    # both meanings on one chart, that's the moment to revisit.
    "sampler_styles": {
        "gaussian":  {"color": "C0", "linestyle": "-", "label": "Gaussian Gibbs"},
        "student_t": {"color": "C2", "linestyle": "-", "label": "Student-t (collapsed RW-MH)"},
    },
    # Per-action palette consumed by the posterior-Q visualisation set
    # (plot_action_probability_heatmap, plot_q_posterior_at_state, ...).
    # Long (+1) deliberately shares "C0" with policy_styles['bayesian']
    # because in nb 04 the action heatmap will sit next to the Bayesian
    # credible band; "blue = Bayesian thinks long" reads coherently. The
    # two meanings never share a single axes (the heatmap is a stacked bar
    # while the band is a line + fill), so they cannot conflate visually.
    # If a future chart ever puts both meanings on one axes, revisit.
    "action_styles": {
        -1: {"color": "C3",  "label": "short"},
         0: {"color": "0.7", "label": "flat"},
        +1: {"color": "C0",  "label": "long"},
    },
}


def _resolve(style: dict | None) -> dict:
    """Merge user-supplied style on top of :data:`DEFAULT_STYLE`."""
    if style is None:
        return dict(DEFAULT_STYLE)
    return {**DEFAULT_STYLE, **style}


def _despine(ax: Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_price_series(
    prices: pd.DataFrame,
    *,
    splits: Mapping[str, tuple[pd.Timestamp, pd.Timestamp]] | None = None,
    log_scale: bool = True,
    style: dict | None = None,
) -> tuple[Figure, Axes]:
    """Plot adjusted close vs date with optional split shading.

    Parameters
    ----------
    prices
        DataFrame indexed by ``Date`` with a ``"Close"`` column.
    splits
        Optional mapping ``{name: (start, end)}`` of named windows to
        shade. ``start``/``end`` may be ``pd.Timestamp`` or anything
        accepted by ``pd.Timestamp(...)``. Splits are drawn as
        translucent ``axvspan`` regions; the alpha grows as the dict
        order advances so test/recent windows read more strongly.
    log_scale
        ``True`` plots ``Close`` on a log y-axis (default; appropriate
        for 30+ years of price data).
    style
        Optional override of :data:`DEFAULT_STYLE`.
    """
    import matplotlib.pyplot as plt

    s = _resolve(style)
    fig, ax = plt.subplots(figsize=s["figsize_single"], dpi=s["dpi"])
    ax.plot(
        prices.index,
        prices["Close"].astype(float),
        color=s["color_data"],
        linewidth=s["linewidth"],
    )
    if log_scale:
        ax.set_yscale("log")
    if splits is not None:
        for i, (name, (start, end)) in enumerate(splits.items()):
            ax.axvspan(
                pd.Timestamp(start),
                pd.Timestamp(end),
                color=s["color_data"],
                alpha=s["alpha_span"] * (1 + i),
                label=name,
            )
        ax.legend(
            loc="lower right",
            frameon=False,
            fontsize=s["legend_fontsize"],
            ncol=min(3, len(splits)),
        )
    ax.set_xlabel("date", fontsize=s["label_fontsize"])
    ax.set_ylabel(
        "adjusted close" + (" (log)" if log_scale else ""),
        fontsize=s["label_fontsize"],
    )
    ax.tick_params(labelsize=s["tick_fontsize"])
    _despine(ax)
    fig.tight_layout()
    return fig, ax


def plot_returns_distribution(
    returns: pd.Series,
    *,
    show_normal_overlay: bool = True,
    show_t_overlay: bool = True,
    bins: int | None = None,
    style: dict | None = None,
) -> tuple[Figure, Axes]:
    """Histogram of daily log returns with Normal / Student-t reference fits.

    The Student-t overlay is the visual argument for ``sampler_t.py``
    existing: if returns visibly leave the Normal envelope in the tails,
    the scale-mixture variant is justified.

    Parameters
    ----------
    returns
        Per-day log returns (NaNs are dropped before fitting / plotting).
    show_normal_overlay
        Overlay a fitted ``N(mu, sigma^2)`` density.
    show_t_overlay
        Overlay a fitted Student-t density (df estimated by MLE via
        :func:`scipy.stats.t.fit`).
    bins
        Histogram bin count (default from :data:`DEFAULT_STYLE`).
    style
        Optional override of :data:`DEFAULT_STYLE`.
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    s = _resolve(style)
    r = returns.dropna().to_numpy(dtype=float)
    if r.size == 0:
        raise ValueError("returns is empty after dropna()")

    fig, ax = plt.subplots(figsize=s["figsize_wide"], dpi=s["dpi"])
    ax.hist(
        r,
        bins=bins if bins is not None else s["hist_bins"],
        color=s["color_data"],
        alpha=s["alpha_hist"],
        density=True,
    )

    if show_normal_overlay or show_t_overlay:
        lo, hi = float(np.quantile(r, 0.0005)), float(np.quantile(r, 0.9995))
        grid = np.linspace(lo, hi, 400)
        if show_normal_overlay:
            mu, sigma = float(r.mean()), float(r.std(ddof=1))
            ax.plot(
                grid,
                stats.norm.pdf(grid, loc=mu, scale=sigma),
                color=s["color_overlay_normal"],
                linewidth=s["linewidth_overlay"],
                linestyle=s["linestyle_overlay_normal"],
                label=f"Normal(μ={mu:.4f}, σ={sigma:.4f})",
            )
        if show_t_overlay:
            df_t, loc_t, scale_t = stats.t.fit(r)
            ax.plot(
                grid,
                stats.t.pdf(grid, df=df_t, loc=loc_t, scale=scale_t),
                color=s["color_overlay_t"],
                linewidth=s["linewidth_overlay"],
                linestyle=s["linestyle_overlay_t"],
                label=f"Student-t (ν={df_t:.2f})",
            )
        ax.legend(loc="upper left", frameon=False, fontsize=s["legend_fontsize"])

    ax.set_xlabel("daily log return", fontsize=s["label_fontsize"])
    ax.set_ylabel("density", fontsize=s["label_fontsize"])
    ax.tick_params(labelsize=s["tick_fontsize"])
    _despine(ax)
    fig.tight_layout()
    return fig, ax


def plot_feature_panel(
    features: pd.DataFrame,
    *,
    feature_cols: Iterable[str] | None = None,
    style: dict | None = None,
) -> tuple[Figure, np.ndarray]:
    """Per-feature time series with a marginal histogram.

    Layout: one row per feature; left column is the time series, right
    column is the marginal histogram. Returns ``(fig, axes)`` with
    ``axes`` shape ``(n_features, 2)``.

    Parameters
    ----------
    features
        DataFrame indexed by date. NaN warmup rows are dropped before
        plotting.
    feature_cols
        Subset / order of columns to plot. Defaults to
        :data:`project.features.FEATURE_COLS`.
    style
        Optional override of :data:`DEFAULT_STYLE`.
    """
    import matplotlib.pyplot as plt

    s = _resolve(style)
    cols = tuple(feature_cols) if feature_cols is not None else FEATURE_COLS
    if not cols:
        raise ValueError("feature_cols is empty")
    df = features[list(cols)].dropna()

    n = len(cols)
    fig, axes = plt.subplots(
        n,
        2,
        figsize=(s["figsize_panel"][0], 1.6 * n + 0.6),
        dpi=s["dpi"],
        gridspec_kw={"width_ratios": [4, 1]},
        sharey="row",
    )
    if n == 1:
        axes = axes.reshape(1, 2)

    palette = (
        s["color_bayesian"],
        s["color_classical"],
        s["color_overlay_normal"],
        s["color_random"],
    )
    for i, col in enumerate(cols):
        color = palette[i % len(palette)]
        ax_ts, ax_hi = axes[i, 0], axes[i, 1]
        ax_ts.plot(df.index, df[col], color=color, linewidth=s["linewidth_thin"])
        ax_ts.axhline(0.0, color="black", linewidth=0.4, linestyle=":")
        ax_ts.set_ylabel(col, fontsize=s["label_fontsize"])
        ax_ts.tick_params(labelsize=s["tick_fontsize"])
        _despine(ax_ts)

        ax_hi.hist(
            df[col],
            bins=40,
            orientation="horizontal",
            color=color,
            alpha=s["alpha_hist"],
        )
        ax_hi.tick_params(labelsize=s["tick_fontsize"])
        ax_hi.set_xlabel("count", fontsize=s["label_fontsize"])
        _despine(ax_hi)

    axes[-1, 0].set_xlabel("date", fontsize=s["label_fontsize"])
    fig.tight_layout()
    return fig, axes


def plot_mc_target_distribution(
    targets: pd.Series,
    *,
    by_action: pd.Series | None = None,
    bins: int | None = None,
    style: dict | None = None,
) -> tuple[Figure, Axes]:
    """Histogram of MC return targets ``y_t``.

    Parameters
    ----------
    targets
        Series of ``y_t`` values (NaNs dropped).
    by_action
        Optional aligned series of integer actions in
        :data:`project.env.ACTIONS` (``-1, 0, +1``). When given, draws
        one overlay histogram per action using the project's locked
        action palette.
    bins
        Histogram bin count (default from :data:`DEFAULT_STYLE`).
    style
        Optional override of :data:`DEFAULT_STYLE`.
    """
    import matplotlib.pyplot as plt

    s = _resolve(style)
    y = targets.dropna()
    if y.size == 0:
        raise ValueError("targets is empty after dropna()")

    fig, ax = plt.subplots(figsize=s["figsize_wide"], dpi=s["dpi"])
    nb = bins if bins is not None else s["hist_bins"]

    if by_action is None:
        ax.hist(
            y.to_numpy(dtype=float),
            bins=nb,
            color=s["color_data"],
            alpha=s["alpha_hist"],
            density=True,
        )
    else:
        a = by_action.reindex(y.index)
        edges = np.linspace(float(y.min()), float(y.max()), nb + 1)
        action_palette = {
            -1: (s["color_action_short"], "a = -1"),
            0: (s["color_action_flat"], "a = 0"),
            1: (s["color_action_long"], "a = +1"),
        }
        for action_value, (color, label) in action_palette.items():
            mask = a == action_value
            if not mask.any():
                continue
            ax.hist(
                y[mask].to_numpy(dtype=float),
                bins=edges,
                color=color,
                alpha=s["alpha_hist"],
                density=True,
                label=label,
            )
        ax.legend(loc="upper right", frameon=False, fontsize=s["legend_fontsize"])

    ax.set_xlabel("y_t (truncated MC return)", fontsize=s["label_fontsize"])
    ax.set_ylabel("density", fontsize=s["label_fontsize"])
    ax.tick_params(labelsize=s["tick_fontsize"])
    _despine(ax)
    fig.tight_layout()
    return fig, ax


def plot_cost_model(
    prices: pd.DataFrame,
    *,
    c0: float = C0_DEFAULT,
    lam: float = LAMBDA_DEFAULT,
    vol_window: int = VOL_WINDOW_DEFAULT,
    style: dict | None = None,
) -> tuple[Figure, Axes]:
    """Visualise ``c_t = c_0 + λ · σ_t^(20)`` over the sample.

    Twin-axis layout: realised log return on the right axis, transaction
    cost ``c_t`` on the left. Makes the cost formula concrete in basis
    points relative to the day-to-day return scale.

    Parameters
    ----------
    prices
        DataFrame with ``"Close"``, indexed by date.
    c0, lam, vol_window
        Cost-formula parameters; defaults match :mod:`project.env`.
    style
        Optional override of :data:`DEFAULT_STYLE`.
    """
    import matplotlib.pyplot as plt

    s = _resolve(style)
    p = prices["Close"].astype(float)
    r = np.log(p).diff()
    c_t = transaction_cost(prices, c0=c0, lam=lam, vol_window=vol_window)

    fig, ax_left = plt.subplots(figsize=s["figsize_single"], dpi=s["dpi"])
    ax_right = ax_left.twinx()

    ax_right.plot(
        r.index,
        r,
        color=s["color_random"],
        linewidth=s["linewidth_thin"],
        alpha=0.5,
        label="r_t (log return)",
    )
    ax_left.plot(
        c_t.index,
        c_t,
        color=s["color_classical"],
        linewidth=s["linewidth"],
        label=f"c_t = {c0:g} + {lam:g}·σ_t^({vol_window})",
    )
    ax_left.axhline(c0, color="black", linewidth=0.4, linestyle=":", label=f"c_0 = {c0:g}")

    ax_left.set_xlabel("date", fontsize=s["label_fontsize"])
    ax_left.set_ylabel("transaction cost c_t", fontsize=s["label_fontsize"])
    ax_right.set_ylabel("daily log return r_t", fontsize=s["label_fontsize"])
    ax_left.tick_params(labelsize=s["tick_fontsize"])
    ax_right.tick_params(labelsize=s["tick_fontsize"])
    _despine(ax_left)
    ax_right.spines["top"].set_visible(False)

    h_l, l_l = ax_left.get_legend_handles_labels()
    h_r, l_r = ax_right.get_legend_handles_labels()
    ax_left.legend(
        h_l + h_r,
        l_l + l_r,
        loc="upper right",
        frameon=False,
        fontsize=s["legend_fontsize"],
    )
    fig.tight_layout()
    return fig, ax_left


def plot_regime_summary(
    prices_test: pd.DataFrame,
    *,
    rolling_window: int = 60,
    style: dict | None = None,
) -> tuple[Figure, np.ndarray]:
    """Two-panel test-window regime summary: rolling vol, running drawdown.

    Anchors the test-period narrative for the modelling notebooks:
    where vol cluster, where the drawdowns hit, what the agent will be
    evaluated against.

    Parameters
    ----------
    prices_test
        Test-window prices with a ``"Close"`` column.
    rolling_window
        Window for realised-vol estimation (days). Default 60.
    style
        Optional override of :data:`DEFAULT_STYLE`.
    """
    import matplotlib.pyplot as plt

    s = _resolve(style)
    p = prices_test["Close"].astype(float)
    r = np.log(p).diff()
    sigma = r.rolling(rolling_window).std() * np.sqrt(252)
    cummax = p.cummax()
    drawdown = p / cummax - 1.0

    fig, axes = plt.subplots(
        2,
        1,
        figsize=s["figsize_dual"],
        dpi=s["dpi"],
        sharex=True,
    )
    ax_vol, ax_dd = axes[0], axes[1]

    ax_vol.plot(
        sigma.index,
        sigma,
        color=s["color_classical"],
        linewidth=s["linewidth"],
    )
    ax_vol.set_ylabel(
        f"annualised vol\n({rolling_window}d rolling)",
        fontsize=s["label_fontsize"],
    )
    ax_vol.tick_params(labelsize=s["tick_fontsize"])
    _despine(ax_vol)

    ax_dd.fill_between(
        drawdown.index,
        drawdown,
        0.0,
        color=s["color_action_short"],
        alpha=s["alpha_drawdown"],
    )
    ax_dd.plot(
        drawdown.index,
        drawdown,
        color=s["color_action_short"],
        linewidth=s["linewidth_thin"],
    )
    ax_dd.axhline(0.0, color="black", linewidth=0.4, linestyle=":")
    ax_dd.set_ylabel("running drawdown", fontsize=s["label_fontsize"])
    ax_dd.set_xlabel("date", fontsize=s["label_fontsize"])
    ax_dd.tick_params(labelsize=s["tick_fontsize"])
    _despine(ax_dd)

    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Path-visualisation set (consumed by nb 02 onward)
#
# All five helpers below take per-policy series keyed by policy name (or, in
# the action-timeline case, a single series + an optional shared axes). Colour
# and linestyle for each policy are resolved from DEFAULT_STYLE['policy_styles']
# so that the project-wide palette stays in one place.
# ---------------------------------------------------------------------------


def _resolve_policy_style(s: dict, name: str) -> dict:
    """Look up ``(color, linestyle, label)`` for a policy name.

    Unknown policies fall back to a neutral grey solid line so callers can
    plot ad-hoc paths without first registering them in
    ``DEFAULT_STYLE['policy_styles']``.
    """
    palette = s.get("policy_styles", {})
    if name in palette:
        entry = palette[name]
        return {
            "color": entry.get("color", "0.3"),
            "linestyle": entry.get("linestyle", "-"),
            "label": entry.get("label", name),
        }
    return {"color": "0.3", "linestyle": "-", "label": name}


def plot_action_timeline(
    actions: pd.Series,
    *,
    label: str | None = None,
    policy: str | None = None,
    ax: Axes | None = None,
    style: dict | None = None,
) -> tuple[Figure, Axes]:
    """Step plot of actions over time.

    Used as the per-policy action panel in nb 02 and as a sub-overlay in
    later notebooks (e.g., overlaying a Bayesian action sequence on a
    posterior credible band). Accepts an optional ``ax`` so the same call
    can serve both stand-alone use and composition into a multi-panel
    figure.

    Parameters
    ----------
    actions
        Per-day integer action series with values in
        :data:`project.env.ACTIONS` (``-1, 0, +1``).
    label
        Legend entry. ``None`` (default) suppresses the legend; supply a
        non-empty string when stacking multiple timelines on a shared axes.
    policy
        Optional policy name for colour/linestyle resolution against
        ``DEFAULT_STYLE['policy_styles']`` (e.g., ``"random"``,
        ``"tabular"``, ``"linear_fqi"``, ``"bayesian"``). ``None``
        (default) uses the neutral ``color_data`` palette. The legend
        text is still controlled by ``label``; ``policy`` only drives
        the line aesthetic.
    ax
        Existing axes to draw into. ``None`` (default) creates a new
        single-axes figure sized via ``style['figsize_single']``. When an
        ``ax`` is supplied, the returned ``Figure`` is ``ax.figure`` and
        ``tight_layout`` is *not* called (to avoid disturbing the caller's
        layout).
    style
        Optional override of :data:`DEFAULT_STYLE`.
    """
    import matplotlib.pyplot as plt

    s = _resolve(style)
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=s["figsize_single"], dpi=s["dpi"])
    else:
        fig = ax.figure

    if policy is not None:
        ps_entry = _resolve_policy_style(s, policy)
        line_color = ps_entry["color"]
        line_ls = ps_entry["linestyle"]
    else:
        line_color = s["color_data"]
        line_ls = "-"

    ax.step(
        actions.index,
        actions.to_numpy(),
        where="post",
        color=line_color,
        linestyle=line_ls,
        linewidth=s["linewidth"],
        label=label,
    )
    ax.set_ylim(-1.5, 1.5)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["-1 short", "0 flat", "+1 long"])
    ax.axhline(0.0, color="black", linewidth=0.4, linestyle=":")
    ax.set_xlabel("date", fontsize=s["label_fontsize"])
    ax.set_ylabel("action", fontsize=s["label_fontsize"])
    ax.tick_params(labelsize=s["tick_fontsize"])
    _despine(ax)
    if label is not None:
        ax.legend(loc="upper right", frameon=False, fontsize=s["legend_fontsize"])
    if created:
        fig.tight_layout()
    return fig, ax


def plot_cumulative_return(
    paths: Mapping[str, pd.Series],
    *,
    style: dict | None = None,
) -> tuple[Figure, Axes]:
    """Overlay one cumulative-log-return line per policy.

    Parameters
    ----------
    paths
        Mapping ``{policy_name: cum_log_return_series}``. Policy names are
        resolved against :data:`DEFAULT_STYLE['policy_styles']` for colour
        and linestyle; unknown names fall back to a neutral grey solid
        line. Iteration order is preserved so the legend order matches
        the dict insertion order.
    style
        Optional override of :data:`DEFAULT_STYLE`.
    """
    import matplotlib.pyplot as plt

    if len(paths) == 0:
        raise ValueError("paths is empty")

    s = _resolve(style)
    fig, ax = plt.subplots(figsize=s["figsize_single"], dpi=s["dpi"])
    for name, series in paths.items():
        ps = _resolve_policy_style(s, name)
        ax.plot(
            series.index,
            series.to_numpy(),
            color=ps["color"],
            linestyle=ps["linestyle"],
            linewidth=s["linewidth"],
            label=ps["label"],
        )
    ax.axhline(0.0, color="black", linewidth=0.4, linestyle=":")
    ax.set_xlabel("date", fontsize=s["label_fontsize"])
    ax.set_ylabel("cumulative log return", fontsize=s["label_fontsize"])
    ax.tick_params(labelsize=s["tick_fontsize"])
    ax.legend(loc="upper left", frameon=False, fontsize=s["legend_fontsize"])
    _despine(ax)
    fig.tight_layout()
    return fig, ax


def plot_position_holdings(
    positions: Mapping[str, pd.Series],
    *,
    style: dict | None = None,
) -> tuple[Figure, np.ndarray]:
    """Per-policy panels of position over time.

    Stacked layout (one row per policy) is more legible than an overlay
    for discrete positions: each panel has a clean ``-1 / 0 / +1`` axis
    and the policy's palette colour, so regime-occupancy reads at a
    glance. The action-timeline overlay (nb 02 cell 11) covers the
    transition story; this panel covers the dwell story.

    Parameters
    ----------
    positions
        Mapping ``{policy_name: position_series}``. Series values are
        expected to be in :data:`project.env.ACTIONS`.
    style
        Optional override of :data:`DEFAULT_STYLE`.
    """
    import matplotlib.pyplot as plt

    if len(positions) == 0:
        raise ValueError("positions is empty")

    s = _resolve(style)
    n = len(positions)
    fig, axes = plt.subplots(
        n,
        1,
        figsize=(s["figsize_single"][0], 1.5 * n + 0.6),
        dpi=s["dpi"],
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes[:, 0]  # collapse the column dim, keep ndarray
    for ax_i, (name, series) in zip(axes, positions.items()):
        ps = _resolve_policy_style(s, name)
        ax_i.fill_between(
            series.index,
            series.to_numpy(),
            0.0,
            step="post",
            color=ps["color"],
            alpha=s["alpha_drawdown"],
        )
        ax_i.step(
            series.index,
            series.to_numpy(),
            where="post",
            color=ps["color"],
            linewidth=s["linewidth"],
        )
        ax_i.axhline(0.0, color="black", linewidth=0.4, linestyle=":")
        ax_i.set_ylim(-1.5, 1.5)
        ax_i.set_yticks([-1, 0, 1])
        ax_i.set_ylabel(ps["label"], fontsize=s["label_fontsize"])
        ax_i.tick_params(labelsize=s["tick_fontsize"])
        _despine(ax_i)
    axes[-1].set_xlabel("date", fontsize=s["label_fontsize"])
    fig.tight_layout()
    return fig, axes


def plot_action_frequency(
    actions: Mapping[str, pd.Series],
    *,
    style: dict | None = None,
) -> tuple[Figure, Axes]:
    """Grouped bar chart of action-occupancy fractions by policy.

    For each policy ``p`` and action ``a`` in :data:`project.env.ACTIONS`,
    plots the fraction of trading days the policy held action ``a``. The
    x-axis groups by action label; one bar per policy per group. With
    three policies this yields nine bars and a three-entry legend.

    Parameters
    ----------
    actions
        Mapping ``{policy_name: action_series}``.
    style
        Optional override of :data:`DEFAULT_STYLE`.
    """
    import matplotlib.pyplot as plt

    if len(actions) == 0:
        raise ValueError("actions is empty")

    s = _resolve(style)
    action_labels = (-1, 0, 1)
    n_actions = len(action_labels)
    n_policies = len(actions)

    fig, ax = plt.subplots(figsize=s["figsize_wide"], dpi=s["dpi"])
    group_centres = np.arange(n_actions, dtype=float)
    group_width = 0.8
    bar_width = group_width / max(n_policies, 1)

    for i, (name, series) in enumerate(actions.items()):
        ps = _resolve_policy_style(s, name)
        arr = series.dropna().to_numpy()
        total = max(len(arr), 1)
        freqs = [float(np.mean(arr == a)) for a in action_labels]
        offsets = group_centres - group_width / 2 + (i + 0.5) * bar_width
        ax.bar(
            offsets,
            freqs,
            width=bar_width,
            color=ps["color"],
            edgecolor="black",
            linewidth=0.4,
            label=ps["label"],
            alpha=0.85,
        )

    ax.set_xticks(group_centres)
    ax.set_xticklabels(["a = -1 (short)", "a = 0 (flat)", "a = +1 (long)"])
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("action", fontsize=s["label_fontsize"])
    ax.set_ylabel("fraction of test-window days", fontsize=s["label_fontsize"])
    ax.tick_params(labelsize=s["tick_fontsize"])
    ax.legend(loc="upper right", frameon=False, fontsize=s["legend_fontsize"])
    _despine(ax)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Diagnostic set (consumed by nb 03 and reused by nbs 04, 05, 09 appendices)
#
# Each helper consumes a sampler-output array directly (chains x draws, etc.)
# and resolves per-sampler colour from DEFAULT_STYLE['sampler_styles'].
# Per-chain rendering is a single matplotlib line per chain so that an
# inspector can see chain-wise mixing at a glance, not a smear.
# ---------------------------------------------------------------------------


def _resolve_sampler_style(s: dict, name: str | None) -> dict:
    """Look up ``(color, linestyle, label)`` for a sampler name.

    Unknown / ``None`` falls back to the data colour, solid line. The
    fallback exists so callers can plot ad-hoc traces (e.g., a synthetic
    sanity check) without first registering a sampler in
    ``DEFAULT_STYLE['sampler_styles']``.
    """
    palette = s.get("sampler_styles", {})
    if name is not None and name in palette:
        entry = palette[name]
        return {
            "color": entry.get("color", s["color_data"]),
            "linestyle": entry.get("linestyle", "-"),
            "label": entry.get("label", name),
        }
    return {"color": s["color_data"], "linestyle": "-", "label": name or ""}


def plot_trace(
    trace_array: np.ndarray,
    *,
    param_name: str,
    sampler: str | None = None,
    ax: Axes | None = None,
    style: dict | None = None,
) -> tuple[Figure, Axes]:
    """One line per chain through draw index for a scalar parameter.

    Parameters
    ----------
    trace_array
        Shape ``(n_chains, n_draws)`` -- caller has already sliced burn-in.
    param_name
        Used as the y-label and (with the sampler tag if any) the title.
    sampler
        Optional sampler name resolved against
        ``DEFAULT_STYLE['sampler_styles']`` for line colour. ``None``
        (default) draws in the neutral data colour.
    ax
        Existing axes to draw into; ``None`` (default) creates a fresh
        single-axes figure sized via ``style['figsize_single']``. When an
        ``ax`` is supplied, ``tight_layout`` is *not* called (the caller
        owns layout).
    style
        Optional override of :data:`DEFAULT_STYLE`.
    """
    import matplotlib.pyplot as plt

    s = _resolve(style)
    arr = np.asarray(trace_array)
    if arr.ndim != 2:
        raise ValueError(
            f"trace_array must be 2-D (n_chains, n_draws), got shape {arr.shape}"
        )

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=s["figsize_single"], dpi=s["dpi"])
    else:
        fig = ax.figure

    samp = _resolve_sampler_style(s, sampler)
    n_chains, n_draws = arr.shape
    x = np.arange(n_draws)
    for c in range(n_chains):
        ax.plot(
            x,
            arr[c],
            color=samp["color"],
            linestyle=samp["linestyle"],
            linewidth=s["linewidth_thin"],
            alpha=0.7,
        )

    ax.set_xlabel("draw (post-burn)", fontsize=s["label_fontsize"])
    ax.set_ylabel(param_name, fontsize=s["label_fontsize"])
    title = param_name if sampler is None else f"{param_name}  --  {samp['label']}"
    ax.set_title(title, fontsize=s["title_fontsize"])
    ax.tick_params(labelsize=s["tick_fontsize"])
    _despine(ax)
    if created:
        fig.tight_layout()
    return fig, ax


def plot_trace_grid(
    trace_dict: Mapping[str, np.ndarray],
    *,
    sampler: str | None = None,
    style: dict | None = None,
) -> tuple[Figure, np.ndarray]:
    """One trace panel per parameter laid out in a single column.

    Wraps :func:`plot_trace` per parameter so that callers pass an entire
    representative-subset dict in one call. Iteration order of
    ``trace_dict`` is preserved as panel order.

    Parameters
    ----------
    trace_dict
        Mapping ``{param_name: (n_chains, n_draws)}``.
    sampler
        Optional sampler name for line-colour resolution against
        ``DEFAULT_STYLE['sampler_styles']``; threaded through to each
        per-panel ``plot_trace`` call.
    style
        Optional override of :data:`DEFAULT_STYLE`.
    """
    import matplotlib.pyplot as plt

    if len(trace_dict) == 0:
        raise ValueError("trace_dict is empty")

    s = _resolve(style)
    n = len(trace_dict)
    fig, axes = plt.subplots(
        n,
        1,
        figsize=(s["figsize_panel"][0], 1.6 * n + 0.6),
        dpi=s["dpi"],
        squeeze=False,
    )
    axes = axes[:, 0]
    for ax_i, (name, arr) in zip(axes, trace_dict.items()):
        plot_trace(arr, param_name=name, sampler=sampler, ax=ax_i, style=style)
    fig.tight_layout()
    return fig, axes


def plot_posterior_density(
    samples: np.ndarray,
    *,
    param_name: str,
    sampler: str | None = None,
    ref_value: float | None = None,
    ax: Axes | None = None,
    style: dict | None = None,
) -> tuple[Figure, Axes]:
    """Histogram + KDE for a flat 1-D posterior sample.

    Parameters
    ----------
    samples
        Flat array ``(n_total_draws,)`` -- caller has already flattened
        chains and sliced burn-in.
    param_name
        Used as the x-label and (with sampler tag) the title.
    sampler
        Optional sampler name resolved against ``sampler_styles`` for the
        KDE line colour. The histogram uses the same colour at low alpha.
    ref_value
        Optional vertical-line reference (e.g., a synthetic-data truth).
    ax
        Existing axes to draw into. ``None`` (default) creates a fresh
        single-axes figure.
    style
        Optional override of :data:`DEFAULT_STYLE`.

    Notes
    -----
    With ``len(samples) == 1`` KDE bandwidth selection is undefined; the
    helper falls back to histogram-only rendering and emits no KDE line.
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    s = _resolve(style)
    arr = np.asarray(samples, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("samples is empty")

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=s["figsize_single"], dpi=s["dpi"])
    else:
        fig = ax.figure

    samp = _resolve_sampler_style(s, sampler)
    color = samp["color"]

    bins = s["hist_bins"]
    if arr.size < bins:
        bins = max(arr.size, 1)
    ax.hist(
        arr,
        bins=bins,
        density=True,
        color=color,
        alpha=s["alpha_hist"],
    )

    if arr.size >= 2 and float(arr.std(ddof=1)) > 0.0:
        lo, hi = float(arr.min()), float(arr.max())
        pad = 0.05 * (hi - lo) if hi > lo else 1e-6
        grid = np.linspace(lo - pad, hi + pad, 400)
        kde = stats.gaussian_kde(arr)
        ax.plot(
            grid,
            kde(grid),
            color=color,
            linewidth=s["linewidth"],
            label=samp["label"] or None,
        )

    if ref_value is not None:
        ax.axvline(
            ref_value,
            color="black",
            linewidth=0.6,
            linestyle="--",
            label=f"ref = {ref_value:g}",
        )

    ax.set_xlabel(param_name, fontsize=s["label_fontsize"])
    ax.set_ylabel("density", fontsize=s["label_fontsize"])
    title = param_name if sampler is None else f"{param_name}  --  {samp['label']}"
    ax.set_title(title, fontsize=s["title_fontsize"])
    ax.tick_params(labelsize=s["tick_fontsize"])
    _despine(ax)
    if created:
        fig.tight_layout()
    return fig, ax


def plot_posterior_density_grid(
    samples_dict: Mapping[str, np.ndarray],
    *,
    sampler: str | None = None,
    ref_values: Mapping[str, float] | None = None,
    style: dict | None = None,
) -> tuple[Figure, np.ndarray]:
    """One density panel per parameter laid out in a single column.

    Parameters
    ----------
    samples_dict
        Mapping ``{param_name: flat_samples_array}``. Iteration order is
        preserved as panel order.
    sampler
        Optional sampler name for KDE colour resolution; threaded through
        :func:`plot_posterior_density` per panel.
    ref_values
        Optional ``{param_name: reference_value}``. Only entries whose
        keys appear in ``samples_dict`` are honoured; the rest are
        silently ignored.
    style
        Optional override of :data:`DEFAULT_STYLE`.
    """
    import matplotlib.pyplot as plt

    if len(samples_dict) == 0:
        raise ValueError("samples_dict is empty")

    s = _resolve(style)
    refs = dict(ref_values) if ref_values is not None else {}
    n = len(samples_dict)
    fig, axes = plt.subplots(
        n,
        1,
        figsize=(s["figsize_panel"][0], 1.6 * n + 0.6),
        dpi=s["dpi"],
        squeeze=False,
    )
    axes = axes[:, 0]
    for ax_i, (name, samples) in zip(axes, samples_dict.items()):
        plot_posterior_density(
            samples,
            param_name=name,
            sampler=sampler,
            ref_value=refs.get(name),
            ax=ax_i,
            style=style,
        )
    fig.tight_layout()
    return fig, axes


def plot_posterior_overlay(
    samplers_dict: Mapping[str, np.ndarray],
    *,
    param_name: str,
    ref_value: float | None = None,
    ax: Axes | None = None,
    style: dict | None = None,
) -> tuple[Figure, Axes]:
    """Overlay one posterior KDE per sampler on a single axes.

    The Gaussian-vs-Student-t comparison plot used at the end of nb 03.
    Each sampler's flat samples become one KDE line in its locked colour.
    A small histogram per sampler sits underneath at low alpha so the
    reader can sanity-check the KDE.

    Parameters
    ----------
    samplers_dict
        Mapping ``{sampler_name: flat_samples_array}``. Sampler names are
        resolved against ``DEFAULT_STYLE['sampler_styles']``.
    param_name
        x-axis label and title (with sampler list appended).
    ref_value
        Optional vertical reference line.
    ax
        Existing axes to draw into; ``None`` (default) creates a fresh
        figure.
    style
        Optional override of :data:`DEFAULT_STYLE`.
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    if len(samplers_dict) == 0:
        raise ValueError("samplers_dict is empty")

    s = _resolve(style)
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=s["figsize_single"], dpi=s["dpi"])
    else:
        fig = ax.figure

    # Pick a shared grid spanning the union of all samplers' supports.
    all_arrays = [np.asarray(v, dtype=float).ravel() for v in samplers_dict.values()]
    if any(a.size == 0 for a in all_arrays):
        raise ValueError("at least one sampler has empty samples")
    lo = float(min(a.min() for a in all_arrays))
    hi = float(max(a.max() for a in all_arrays))
    pad = 0.05 * (hi - lo) if hi > lo else 1e-6
    grid = np.linspace(lo - pad, hi + pad, 400)

    for name, arr in zip(samplers_dict.keys(), all_arrays):
        samp = _resolve_sampler_style(s, name)
        ax.hist(
            arr,
            bins=s["hist_bins"],
            density=True,
            color=samp["color"],
            alpha=s["alpha_hist"] * 0.4,  # quieter than the singular helper
        )
        if arr.size >= 2 and float(arr.std(ddof=1)) > 0.0:
            kde = stats.gaussian_kde(arr)
            ax.plot(
                grid,
                kde(grid),
                color=samp["color"],
                linestyle=samp["linestyle"],
                linewidth=s["linewidth"],
                label=samp["label"],
            )

    if ref_value is not None:
        ax.axvline(
            ref_value,
            color="black",
            linewidth=0.6,
            linestyle="--",
            label=f"ref = {ref_value:g}",
        )

    ax.set_xlabel(param_name, fontsize=s["label_fontsize"])
    ax.set_ylabel("density", fontsize=s["label_fontsize"])
    ax.set_title(
        f"{param_name}  --  {' vs '.join(samplers_dict.keys())}",
        fontsize=s["title_fontsize"],
    )
    ax.tick_params(labelsize=s["tick_fontsize"])
    ax.legend(loc="upper right", frameon=False, fontsize=s["legend_fontsize"])
    _despine(ax)
    if created:
        fig.tight_layout()
    return fig, ax


def _convergence_summary(
    values_dict: Mapping[str, float],
    *,
    threshold: float,
    direction: str,
    threshold_label: str,
    xlabel: str,
    style: dict,
) -> tuple[Figure, Axes]:
    """Shared layout for ``plot_rhat_summary`` and ``plot_ess_summary``.

    Horizontal bars per parameter, threshold rule line, failing bars
    flagged with the project's "warn" colour. ``direction`` is
    ``"upper"`` (R-hat: fail when value > threshold) or ``"lower"``
    (ESS: fail when value < threshold).
    """
    import matplotlib.pyplot as plt

    if len(values_dict) == 0:
        raise ValueError("values_dict is empty")
    if direction not in ("upper", "lower"):
        raise ValueError("direction must be 'upper' or 'lower'")

    s = style
    names = list(values_dict.keys())
    values = np.asarray([float(v) for v in values_dict.values()])

    if direction == "upper":
        fail_mask = values > threshold
    else:
        fail_mask = values < threshold
    pass_color = s["color_data"]
    fail_color = s["color_action_short"]   # red, project-locked "warn" colour
    bar_colors = [fail_color if f else pass_color for f in fail_mask]

    height = max(0.4 * len(names) + 0.8, s["figsize_wide"][1])
    fig, ax = plt.subplots(
        figsize=(s["figsize_wide"][0], height), dpi=s["dpi"]
    )
    y_positions = np.arange(len(names))
    ax.barh(
        y_positions,
        values,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.4,
        alpha=0.85,
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(names, fontsize=s["tick_fontsize"])
    ax.invert_yaxis()  # first key on top -- reads top-to-bottom
    ax.axvline(
        threshold,
        color="black",
        linewidth=0.6,
        linestyle="--",
        label=threshold_label,
    )
    ax.set_xlabel(xlabel, fontsize=s["label_fontsize"])
    ax.tick_params(labelsize=s["tick_fontsize"])
    ax.legend(loc="lower right", frameon=False, fontsize=s["legend_fontsize"])
    _despine(ax)
    fig.tight_layout()
    return fig, ax


def plot_rhat_summary(
    rhat_dict: Mapping[str, float],
    *,
    threshold: float = 1.05,
    style: dict | None = None,
) -> tuple[Figure, Axes]:
    """Horizontal bar summary of R-hat across parameters.

    Bars exceeding ``threshold`` (default 1.05, the project gate from
    ``test_sampler_recovery``) are flagged red so they read at a glance.
    The threshold itself is drawn as a dashed vertical line.

    Parameters
    ----------
    rhat_dict
        Mapping ``{param_name: rhat_scalar}``. Iteration order becomes
        plot order, top-to-bottom.
    threshold
        Convergence gate (default 1.05).
    style
        Optional override of :data:`DEFAULT_STYLE`.
    """
    s = _resolve(style)
    return _convergence_summary(
        rhat_dict,
        threshold=threshold,
        direction="upper",
        threshold_label=f"R-hat = {threshold:g}",
        xlabel="R-hat",
        style=s,
    )


def plot_ess_summary(
    ess_dict: Mapping[str, float],
    *,
    threshold: float = 400.0,
    style: dict | None = None,
) -> tuple[Figure, Axes]:
    """Horizontal bar summary of effective sample size across parameters.

    Bars *below* ``threshold`` are flagged red. Default 400 matches the
    project's ESS gate on ``nu`` from ``test_t_recovery_single_replication``.

    Parameters
    ----------
    ess_dict
        Mapping ``{param_name: ess_scalar}``.
    threshold
        Floor for adequate effective sample size (default 400).
    style
        Optional override of :data:`DEFAULT_STYLE`.
    """
    s = _resolve(style)
    return _convergence_summary(
        ess_dict,
        threshold=threshold,
        direction="lower",
        threshold_label=f"ESS = {threshold:g}",
        xlabel="ESS",
        style=s,
    )


def plot_mh_acceptance(
    acceptance_trace: np.ndarray,
    *,
    target_low: float = 0.20,
    target_high: float = 0.45,
    style: dict | None = None,
) -> tuple[Figure, Axes]:
    """Rolling MH acceptance rate per chain with a shaded target band.

    Used for the ``nu`` block of the Student-t sampler. The 2-D input is
    a 0/1 acceptance trace (chains x draws) -- in nb 03 it is *derived*
    from the post-burn ``nu`` chain via
    ``np.diff(np.log(nu_trace), axis=1) != 0`` rather than read from the
    sampler's stored ``nu_acceptance`` aggregate (the stored field is
    chain-level only). The lossiness is bounded:
    ``propose_log_nu`` adds ``step * z`` with ``z ~ N(0, 1)``, so the
    probability that an accepted proposal lands on the previous
    ``log_nu`` is exactly zero in IEEE-754 floating point. Rejection is
    the only way two consecutive ``log_nu`` values match.

    Parameters
    ----------
    acceptance_trace
        Shape ``(n_chains, n_draws_post_burn)`` of 0/1 (or boolean).
    target_low, target_high
        Shaded reference band for adequate acceptance, defaults
        ``[0.20, 0.45]`` -- the project gate from
        ``test_t_recovery_single_replication``.
    style
        Optional override of :data:`DEFAULT_STYLE`.
    """
    import matplotlib.pyplot as plt

    arr = np.asarray(acceptance_trace, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            f"acceptance_trace must be 2-D (n_chains, n_draws), got shape {arr.shape}"
        )

    s = _resolve(style)
    n_chains, n_draws = arr.shape
    # Rolling window: a sane default is 5% of the chain, floor at 50.
    window = max(50, int(0.05 * n_draws))
    window = min(window, n_draws)
    kernel = np.ones(window) / window

    fig, ax = plt.subplots(figsize=s["figsize_single"], dpi=s["dpi"])
    ax.axhspan(
        target_low,
        target_high,
        color=s["color_random"],
        alpha=s["alpha_span"],
        label=f"target band [{target_low:g}, {target_high:g}]",
    )
    samp_color = s["sampler_styles"]["student_t"]["color"]
    x = np.arange(n_draws - window + 1) + window // 2
    for c in range(n_chains):
        rolled = np.convolve(arr[c], kernel, mode="valid")
        ax.plot(
            x,
            rolled,
            color=samp_color,
            linewidth=s["linewidth_thin"],
            alpha=0.7,
        )
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("draw (post-burn)", fontsize=s["label_fontsize"])
    ax.set_ylabel(f"rolling acceptance ({window}-draw window)", fontsize=s["label_fontsize"])
    ax.set_title("RW-MH acceptance for nu", fontsize=s["title_fontsize"])
    ax.tick_params(labelsize=s["tick_fontsize"])
    ax.legend(loc="upper right", frameon=False, fontsize=s["legend_fontsize"])
    _despine(ax)
    fig.tight_layout()
    return fig, ax


def plot_drawdown(
    cum_log_returns: Mapping[str, pd.Series],
    *,
    style: dict | None = None,
) -> tuple[Figure, Axes]:
    """Running peak-to-trough drawdown overlay.

    For each policy's cumulative-log-return series, draws
    ``cum - running_peak`` over time. By construction this is non-positive
    and reaches zero precisely at running maxima. Overlay (rather than
    stacked panels) reads well here because drawdown trajectories
    typically share inflection points across policies and overlap
    informatively.

    Parameters
    ----------
    cum_log_returns
        Mapping ``{policy_name: cum_log_return_series}``.
    style
        Optional override of :data:`DEFAULT_STYLE`.
    """
    import matplotlib.pyplot as plt

    if len(cum_log_returns) == 0:
        raise ValueError("cum_log_returns is empty")

    s = _resolve(style)
    fig, ax = plt.subplots(figsize=s["figsize_single"], dpi=s["dpi"])
    for name, cum in cum_log_returns.items():
        ps = _resolve_policy_style(s, name)
        c = cum.dropna()
        running_peak = c.cummax()
        dd = c - running_peak
        ax.plot(
            dd.index,
            dd.to_numpy(),
            color=ps["color"],
            linestyle=ps["linestyle"],
            linewidth=s["linewidth"],
            label=ps["label"],
        )
    ax.axhline(0.0, color="black", linewidth=0.4, linestyle=":")
    ax.set_xlabel("date", fontsize=s["label_fontsize"])
    ax.set_ylabel("drawdown (cum log return - running peak)", fontsize=s["label_fontsize"])
    ax.tick_params(labelsize=s["tick_fontsize"])
    ax.legend(loc="lower left", frameon=False, fontsize=s["legend_fontsize"])
    _despine(ax)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Posterior-Q visualisation set (consumed by nb 04 onward)
#
# These helpers consume posterior arrays directly -- usually obtained via
# `project.thompson.posterior_q(policy, x)` or a posterior rollout pickled to
# `data/processed/paths/`. They are the visual backbone of nb 04's
# "Bayesian on its own terms" story: regime-conditional Q-uncertainty,
# credible bands on cumulative-return paths, posterior action diversity, and
# scalar-metric posterior densities.
# ---------------------------------------------------------------------------


_PROJECT_ACTION_KEYS: tuple[int, int, int] = (-1, 0, +1)


def _resolve_action_palette(s: dict, n_actions: int) -> list[str]:
    """Per-column colour list for an n-action Q-posterior or action-path array.

    For the project default ``n_actions == 3`` we map column ``i`` to the
    locked palette via ``ACTIONS[i]`` (short = -1, flat = 0, long = +1) so
    that the colours are consistent with the env's action labels. For other
    arities we fall back to a deterministic ``"Ci"`` cycle, which is enough
    for ad-hoc plots (none ship in the current notebook set).
    """
    palette = s.get("action_styles", {})
    if n_actions == 3 and all(k in palette for k in _PROJECT_ACTION_KEYS):
        return [palette[k]["color"] for k in _PROJECT_ACTION_KEYS]
    return [f"C{i}" for i in range(n_actions)]


def plot_q_posterior_at_state(
    q_samples: np.ndarray,
    *,
    state_label: str,
    action_labels: Iterable[str] | None = None,
    ax: Axes | None = None,
    style: dict | None = None,
) -> tuple[Figure, Axes]:
    """Posterior of ``Q(s, a)`` at one state, with one density per action.

    Each action's posterior samples become a quiet density-style histogram
    underneath a KDE line, in its locked action colour, with a dotted
    vertical line at the posterior mean. The three densities share the
    x-axis so a reader can immediately see which action is favoured and how
    confidently.

    Parameters
    ----------
    q_samples
        Shape ``(n_draws, n_actions)`` -- typically the output of
        :func:`project.thompson.posterior_q` for a single state ``x``.
    state_label
        Used in the title and (when wrapped by :func:`plot_q_posterior_grid`)
        as the per-panel sub-title. Pass a meaningful regime label such as
        ``"low_vol_2024_05"`` so the figure self-documents.
    action_labels
        Optional list of per-action legend labels of length ``n_actions``.
        ``None`` (default) maps to ``["short", "flat", "long"]`` for the
        project default ``n_actions == 3`` and to ``["a0", "a1", ...]``
        otherwise.
    ax
        Existing axes to draw into; ``None`` (default) creates a fresh
        figure. Supplying ``ax`` lets :func:`plot_q_posterior_grid` use this
        helper as its per-panel renderer.
    style
        Optional override of :data:`DEFAULT_STYLE`.

    Notes
    -----
    With ``n_draws == 1`` KDE bandwidth selection is undefined; the helper
    falls back to histogram-only rendering and emits no KDE line. This keeps
    grid layouts robust to degenerate single-draw posteriors.
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    arr = np.asarray(q_samples, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            f"q_samples must be 2-D (n_draws, n_actions), got shape {arr.shape}"
        )
    n_draws, n_actions = arr.shape
    if n_actions == 0:
        raise ValueError("q_samples has zero actions")

    s = _resolve(style)
    if action_labels is None:
        if n_actions == 3:
            labels = ["short", "flat", "long"]
        else:
            labels = [f"a{i}" for i in range(n_actions)]
    else:
        labels = list(action_labels)
        if len(labels) != n_actions:
            raise ValueError(
                f"action_labels has {len(labels)} entries but q_samples "
                f"has {n_actions} actions"
            )
    colors = _resolve_action_palette(s, n_actions)

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=s["figsize_single"], dpi=s["dpi"])
    else:
        fig = ax.figure

    # Shared x-grid spanning the union of all actions' posterior supports.
    lo = float(arr.min())
    hi = float(arr.max())
    pad = 0.05 * (hi - lo) if hi > lo else 1e-6
    grid = np.linspace(lo - pad, hi + pad, 400)

    bins = s["hist_bins"]
    if n_draws < bins:
        bins = max(n_draws, 1)

    for a in range(n_actions):
        col = colors[a]
        col_arr = arr[:, a]
        ax.hist(
            col_arr, bins=bins, density=True,
            color=col, alpha=s["alpha_hist"] * 0.4,
        )
        if col_arr.size >= 2 and float(col_arr.std(ddof=1)) > 0.0:
            kde = stats.gaussian_kde(col_arr)
            ax.plot(
                grid, kde(grid),
                color=col,
                linewidth=s["linewidth"],
                label=labels[a],
            )
        ax.axvline(
            float(col_arr.mean()),
            color=col,
            linewidth=0.8,
            linestyle=":",
        )

    ax.set_xlabel("Q(s, a) posterior", fontsize=s["label_fontsize"])
    ax.set_ylabel("density", fontsize=s["label_fontsize"])
    ax.set_title(f"Q-posterior at state: {state_label}", fontsize=s["title_fontsize"])
    ax.tick_params(labelsize=s["tick_fontsize"])
    if any(ln.get_label() and not ln.get_label().startswith("_") for ln in ax.lines):
        ax.legend(loc="upper right", frameon=False, fontsize=s["legend_fontsize"])
    _despine(ax)
    if created:
        fig.tight_layout()
    return fig, ax


def plot_q_posterior_grid(
    q_samples_dict: Mapping[str, np.ndarray],
    *,
    action_labels: Iterable[str] | None = None,
    style: dict | None = None,
) -> tuple[Figure, np.ndarray]:
    """One :func:`plot_q_posterior_at_state` panel per representative state.

    Parameters
    ----------
    q_samples_dict
        Mapping ``{state_label: q_samples_array}`` where each array has
        shape ``(n_draws, n_actions)``. Iteration order is preserved as
        panel order.
    action_labels
        Optional list of per-action legend labels of length ``n_actions``;
        threaded through to every panel so they share the same legend
        vocabulary.
    style
        Optional override of :data:`DEFAULT_STYLE`.
    """
    import matplotlib.pyplot as plt

    if len(q_samples_dict) == 0:
        raise ValueError("q_samples_dict is empty")

    s = _resolve(style)
    n = len(q_samples_dict)
    fig, axes = plt.subplots(
        n,
        1,
        figsize=(s["figsize_panel"][0], 1.8 * n + 0.6),
        dpi=s["dpi"],
        squeeze=False,
    )
    axes = axes[:, 0]
    for ax_i, (state, q_arr) in zip(axes, q_samples_dict.items()):
        plot_q_posterior_at_state(
            q_arr,
            state_label=state,
            action_labels=action_labels,
            ax=ax_i,
            style=style,
        )
    fig.tight_layout()
    return fig, axes


def plot_credible_band(
    paths_array: np.ndarray,
    index: pd.DatetimeIndex,
    *,
    bands: tuple[float, ...] = (0.50, 0.95),
    show_median: bool = True,
    sample_paths: int = 0,
    policy: str | None = None,
    label: str | None = None,
    ax: Axes | None = None,
    style: dict | None = None,
) -> tuple[Figure, Axes]:
    """Posterior credible bands over a path-shaped posterior rollout.

    The visual idiom for any path-level posterior on a date axis: cumulative
    log return, drawdown, position holdings, exposure, etc. Each band level
    becomes one ``fill_between`` shaded region; wider bands are drawn first
    (lower alpha) and narrower bands on top (higher alpha) so the inner
    quantile range reads sharper. The median is overlaid as a single line in
    the policy's locked colour.

    Parameters
    ----------
    paths_array
        Shape ``(n_paths, n_timesteps)``. Posterior draws of a per-day
        scalar series. Quantiles are taken across the path axis at each
        timestep.
    index
        Length-``n_timesteps`` :class:`pandas.DatetimeIndex` for the
        x-axis. Length must match ``n_timesteps``.
    bands
        Iterable of central credible-mass levels in ``(0, 1)``. Default
        ``(0.50, 0.95)`` plots a 50% inner band and a 95% outer band.
    show_median
        Whether to overlay a per-timestep median line.
    sample_paths
        If positive, overlay this many evenly-spaced rows of
        ``paths_array`` at low alpha to give a sense of individual path
        behaviour. Capped at ``n_paths``.
    policy
        Optional policy name for colour resolution against
        ``DEFAULT_STYLE['policy_styles']`` (e.g., ``"bayesian"``,
        ``"linear_fqi"``). ``None`` (default) uses the neutral
        ``color_data`` palette.
    label
        Legend label for the median line. ``None`` defaults to the
        ``policy_styles`` label when ``policy`` is given, otherwise
        suppresses the median legend entry.
    ax
        Existing axes to draw into. ``None`` (default) creates a fresh
        single-axes figure. Supplying ``ax`` lets callers overlay multiple
        policies' bands on the same axis -- the closing Gaussian-vs-Student-t
        panel of nb 04, and the headline Bayesian-vs-classical panel of
        nb 05, both rely on this.
    style
        Optional override of :data:`DEFAULT_STYLE`.
    """
    import matplotlib.pyplot as plt

    arr = np.asarray(paths_array, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            f"paths_array must be 2-D (n_paths, n_timesteps), got shape {arr.shape}"
        )
    n_paths, n_t = arr.shape
    if n_t != len(index):
        raise ValueError(
            f"len(index)={len(index)} must equal n_timesteps={n_t}"
        )
    if not bands:
        raise ValueError("bands must be non-empty")
    for b in bands:
        if not 0.0 < b < 1.0:
            raise ValueError(f"each band must be strictly in (0, 1), got {b}")
    if sample_paths < 0:
        raise ValueError(f"sample_paths must be >= 0, got {sample_paths}")

    s = _resolve(style)
    if policy is not None:
        ps = _resolve_policy_style(s, policy)
        color = ps["color"]
        median_label = label if label is not None else ps["label"]
    else:
        color = s["color_data"]
        median_label = label

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=s["figsize_single"], dpi=s["dpi"])
    else:
        fig = ax.figure

    # Draw widest band first (faintest), narrowest last (most opaque) so the
    # inner quantile range stays visually distinct from the tail mass.
    sorted_bands = sorted(set(float(b) for b in bands), reverse=True)
    for i, b in enumerate(sorted_bands):
        alpha_low = (1.0 - b) / 2.0
        alpha_high = 1.0 - alpha_low
        lo = np.quantile(arr, alpha_low, axis=0)
        hi = np.quantile(arr, alpha_high, axis=0)
        ax.fill_between(
            index, lo, hi,
            color=color,
            alpha=0.18 + 0.10 * i,
            linewidth=0,
        )

    if sample_paths > 0:
        n_to_show = min(int(sample_paths), n_paths)
        sample_idxs = np.linspace(0, n_paths - 1, n_to_show, dtype=int)
        for k in sample_idxs:
            ax.plot(
                index, arr[int(k)],
                color=color,
                linewidth=s["linewidth_thin"],
                alpha=0.15,
            )

    if show_median:
        med = np.quantile(arr, 0.5, axis=0)
        ax.plot(
            index, med,
            color=color,
            linewidth=s["linewidth"],
            label=median_label,
        )

    ax.set_xlabel("date", fontsize=s["label_fontsize"])
    ax.tick_params(labelsize=s["tick_fontsize"])
    legend_lines = [
        ln for ln in ax.lines
        if ln.get_label() and not ln.get_label().startswith("_")
    ]
    if legend_lines:
        ax.legend(loc="upper left", frameon=False, fontsize=s["legend_fontsize"])
    _despine(ax)
    if created:
        fig.tight_layout()
    return fig, ax


def plot_action_probability_heatmap(
    actions_array: np.ndarray,
    index: pd.DatetimeIndex,
    *,
    n_actions: int = 3,
    bin_freq: str = "ME",
    action_labels: Iterable[str] | None = None,
    style: dict | None = None,
) -> tuple[Figure, Axes]:
    """Stacked bar across time bins of posterior action proportions.

    Aggregates a posterior rollout's action choices over time: at each
    timestep the per-path action distribution is computed, then averaged
    within each bin (default monthly). Each bar at bin ``b`` stacks
    ``P(a = -1 | b)``, ``P(a = 0 | b)``, ``P(a = +1 | b)`` summing to 1.

    The visual answer to "across the test window's feature distribution,
    what fraction of posterior draws picks each action?" -- regime-by-regime,
    you see when the posterior policy concentrates on long, when it goes
    flat, when it shorts.

    Parameters
    ----------
    actions_array
        Shape ``(n_paths, n_timesteps)``, integer action labels in
        :data:`project.env.ACTIONS` (``-1, 0, +1``).
    index
        Length-``n_timesteps`` :class:`pandas.DatetimeIndex`.
    n_actions
        Number of distinct actions; the project default is 3 (short / flat
        / long). Other arities use a deterministic ``"Ci"`` colour cycle.
    bin_freq
        Pandas resample frequency string (default ``"ME"`` = month end).
        Common alternatives: ``"QE"`` (quarterly), ``"YE"`` (yearly).
    action_labels
        Optional list of legend labels of length ``n_actions``. ``None``
        defaults to the labels stored in :data:`DEFAULT_STYLE['action_styles']`.
    style
        Optional override of :data:`DEFAULT_STYLE`.
    """
    import matplotlib.pyplot as plt

    arr = np.asarray(actions_array, dtype=int)
    if arr.ndim != 2:
        raise ValueError(
            f"actions_array must be 2-D (n_paths, n_timesteps), got shape {arr.shape}"
        )
    n_paths, n_t = arr.shape
    if n_t != len(index):
        raise ValueError(
            f"len(index)={len(index)} must equal n_timesteps={n_t}"
        )
    if n_actions < 1:
        raise ValueError(f"n_actions must be >= 1, got {n_actions}")

    s = _resolve(style)
    palette = s.get("action_styles", {})
    project_default = (n_actions == 3 and all(k in palette for k in _PROJECT_ACTION_KEYS))

    if project_default:
        action_keys = _PROJECT_ACTION_KEYS
        colors = [palette[k]["color"] for k in action_keys]
        default_labels = [palette[k]["label"] for k in action_keys]
    else:
        action_keys = tuple(range(n_actions))
        colors = [f"C{i}" for i in range(n_actions)]
        default_labels = [f"a{i}" for i in range(n_actions)]

    if action_labels is None:
        labels = list(default_labels)
    else:
        labels = list(action_labels)
        if len(labels) != n_actions:
            raise ValueError(
                f"action_labels has {len(labels)} entries but n_actions={n_actions}"
            )

    # Per-step action proportions across the path axis.
    probs = np.zeros((n_t, n_actions), dtype=float)
    for a_idx, a_key in enumerate(action_keys):
        probs[:, a_idx] = (arr == a_key).mean(axis=0)

    df = pd.DataFrame(probs, index=index, columns=list(action_keys))
    binned = df.resample(bin_freq).mean()
    # Drop any all-NaN rows (resample can introduce empty bins at boundaries).
    binned = binned.dropna(how="all")
    # Renormalise within each bin so each bar still sums to ~1 even after
    # ragged-edge effects from resample boundaries.
    row_sums = binned.sum(axis=1).replace(0.0, np.nan)
    binned = binned.div(row_sums, axis=0).fillna(0.0)

    fig, ax = plt.subplots(figsize=s["figsize_wide"], dpi=s["dpi"])
    bin_centres = binned.index
    # Width: 80% of the median spacing between bins, in days.
    if len(bin_centres) >= 2:
        spacings = np.diff(bin_centres.values).astype("timedelta64[D]").astype(int)
        bar_width = 0.8 * float(np.median(spacings))
    else:
        bar_width = 20.0  # ~monthly fallback for a degenerate 1-bin case

    bottom = np.zeros(len(bin_centres))
    for a_idx, a_key in enumerate(action_keys):
        heights = binned[a_key].to_numpy()
        ax.bar(
            bin_centres, heights,
            bottom=bottom,
            width=bar_width,
            color=colors[a_idx],
            edgecolor="white",
            linewidth=0.4,
            label=labels[a_idx],
            align="center",
        )
        bottom = bottom + heights

    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("date", fontsize=s["label_fontsize"])
    ax.set_ylabel("posterior action probability", fontsize=s["label_fontsize"])
    ax.tick_params(labelsize=s["tick_fontsize"])
    ax.legend(loc="upper right", frameon=False, fontsize=s["legend_fontsize"], ncol=n_actions)
    _despine(ax)
    fig.tight_layout()
    return fig, ax


def plot_posterior_metric_distribution(
    metric_samples_dict: Mapping[str, np.ndarray],
    *,
    sampler: str | None = None,
    style: dict | None = None,
) -> tuple[Figure, np.ndarray]:
    """One density panel per scalar metric across posterior paths.

    The visual statement that "Bayesian gives a *distribution* over Sharpe
    /MDD/turnover, not a point estimate". Each entry in
    ``metric_samples_dict`` is a 1-D array of per-path metric realisations
    (typically one value per posterior rollout); this helper draws a
    histogram + KDE for each.

    Renders by delegation to :func:`plot_posterior_density_grid` -- the
    structural payload is the same, only the semantics differ. This keeps
    the per-panel rendering (KDE colour resolution, single-sample
    fall-back, layout heuristics) in a single place.

    Parameters
    ----------
    metric_samples_dict
        Mapping ``{metric_name: posterior_path_metric_array}``. Each value
        is a 1-D array of shape ``(n_paths,)``. Iteration order is
        preserved as panel order.
    sampler
        Optional sampler name for KDE colour resolution against
        :data:`DEFAULT_STYLE['sampler_styles']`.
    style
        Optional override of :data:`DEFAULT_STYLE`.
    """
    if len(metric_samples_dict) == 0:
        raise ValueError("metric_samples_dict is empty")
    return plot_posterior_density_grid(
        metric_samples_dict,
        sampler=sampler,
        style=style,
    )


# ---------------------------------------------------------------------------
# Comparison set (consumed by nb 05 onward)
#
# Three helpers that consume the post-aggregation outputs of the project's
# two metric-distribution functions (``posterior_predictive_metrics`` and
# ``classical_baseline_distribution``) along with the raw posterior /
# seed-level paths. They are the visual backbone of nb 05's headline:
# Bayesian-vs-classical comparison on a forest-plot-style table, density
# overlay, and four-method cumulative-return panel.
#
# Cross-method consumers honour ``DEFAULT_STYLE['policy_styles']`` for
# colour and linestyle. The classical bucket (tabular + linear FQI) shares
# the C1 colour by linestyle differentiation; ``plot_path_comparison``
# additionally differentiates the two via fill_between alpha so they remain
# distinguishable when both render as seed-spread bands on one axes.
# ---------------------------------------------------------------------------


_DEFAULT_METRIC_ORDER: tuple[str, ...] = ("sharpe", "max_drawdown", "turnover")


def plot_metric_comparison_table(
    method_metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    metric_order: Iterable[str] | None = None,
    method_order: Iterable[str] | None = None,
    lo_key: str = "ci_lo",
    hi_key: str = "ci_hi",
    style: dict | None = None,
) -> tuple[Figure, Axes]:
    """Forest-plot-style mean + interval grid: rows = metrics, methods stacked
    vertically within each row.

    Reads ``{method: {metric: {mean, <lo_key>, <hi_key>}}}``. The lo / hi key
    names default to ``ci_lo`` / ``ci_hi`` (the classical-distribution
    convention from :func:`project.baselines.classical_baseline_distribution`)
    but can be overridden with ``lo_key="q025", hi_key="q975"`` to consume
    :func:`project.eval.posterior_predictive_metrics` output directly. The
    helper itself is interval-semantics agnostic -- the caller is responsible
    for ensuring the intervals being plotted answer the same question across
    methods (see nb 05's closing markdown for the project's stance).

    Parameters
    ----------
    method_metrics
        Nested mapping ``{method_name: {metric_name: {mean, lo_key, hi_key}}}``.
    metric_order
        Optional iterable of metric names controlling row order
        (top-to-bottom). Defaults to ``("sharpe", "max_drawdown",
        "turnover")``.
    method_order
        Optional iterable of method names controlling within-row stacking
        and legend order. Defaults to insertion order of ``method_metrics``.
    lo_key, hi_key
        Keys inside each metric dict that hold the interval endpoints.
        Defaults match the classical bucket; pass ``"q025"`` / ``"q975"``
        for the Bayesian bucket.
    style
        Optional override of :data:`DEFAULT_STYLE`.
    """
    import matplotlib.pyplot as plt

    if len(method_metrics) == 0:
        raise ValueError("method_metrics is empty")

    s = _resolve(style)
    methods = list(method_order) if method_order is not None else list(method_metrics.keys())
    if metric_order is not None:
        metrics = list(metric_order)
    else:
        # Use defaults that exist in the data; otherwise fall back to the union
        # of metric keys in insertion order of the first method's dict.
        first_method_metrics = list(next(iter(method_metrics.values())).keys())
        metrics = [m for m in _DEFAULT_METRIC_ORDER if m in first_method_metrics]
        if not metrics:
            metrics = first_method_metrics

    n_methods = len(methods)
    n_metrics = len(metrics)
    # Vertical offset per method within a metric row; total offset spread is
    # 0.6 (well inside the unit row spacing of 1.0 to keep rows visually
    # distinct).
    if n_methods > 1:
        offsets = np.linspace(-0.3, 0.3, n_methods)
    else:
        offsets = np.array([0.0])

    fig, ax = plt.subplots(
        figsize=(s["figsize_panel"][0], 1.0 * n_metrics + 1.5),
        dpi=s["dpi"],
    )

    # Track which methods have already been added to the legend (label only the
    # first metric row's marker per method).
    legend_added: set[str] = set()
    for j, method in enumerate(methods):
        ps = _resolve_policy_style(s, method)
        color = ps["color"]
        label = ps["label"]
        for i, metric in enumerate(metrics):
            entry = method_metrics[method].get(metric)
            if entry is None:
                continue
            mean = float(entry["mean"])
            lo = float(entry[lo_key])
            hi = float(entry[hi_key])
            y = i + offsets[j]
            ax.plot(
                [lo, hi], [y, y],
                color=color,
                linewidth=s["linewidth"],
                solid_capstyle="butt",
            )
            ln_label = label if method not in legend_added else None
            ax.plot(
                [mean], [y],
                marker="o",
                markersize=4.5,
                color=color,
                label=ln_label,
            )
            if ln_label is not None:
                legend_added.add(method)

    ax.set_yticks(np.arange(n_metrics))
    ax.set_yticklabels(metrics, fontsize=s["tick_fontsize"])
    ax.invert_yaxis()  # first metric on top -- reads top-to-bottom
    ax.set_xlabel("metric value (mean ± interval)", fontsize=s["label_fontsize"])
    ax.tick_params(labelsize=s["tick_fontsize"])
    ax.legend(loc="best", frameon=False, fontsize=s["legend_fontsize"])
    ax.axvline(0.0, color="black", linewidth=0.4, linestyle=":")
    _despine(ax)
    fig.tight_layout()
    return fig, ax


def plot_metric_distribution_overlay(
    method_samples: Mapping[str, np.ndarray],
    *,
    metric_name: str,
    style: dict | None = None,
) -> tuple[Figure, Axes]:
    """KDE per method on shared axes for a single scalar metric.

    The "see the distributions, not just the table" companion to
    :func:`plot_metric_comparison_table`. Each method contributes one density
    curve (histogram + KDE) in its locked policy colour.

    Parameters
    ----------
    method_samples
        Mapping ``{method_name: 1-D array of metric values}``. Methods are
        resolved against :data:`DEFAULT_STYLE['policy_styles']` for colour /
        linestyle.
    metric_name
        x-axis label and title -- the metric this density represents (e.g.,
        ``"sharpe"``, ``"max_drawdown"``, ``"turnover"``).
    style
        Optional override of :data:`DEFAULT_STYLE`.

    Notes
    -----
    A method whose ``samples`` array has fewer than two finite, non-constant
    values falls back to histogram-only rendering for that method (no KDE
    line). This keeps the overlay robust to degenerate single-seed
    classicals.
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    if len(method_samples) == 0:
        raise ValueError("method_samples is empty")

    s = _resolve(style)
    fig, ax = plt.subplots(figsize=s["figsize_single"], dpi=s["dpi"])

    # Shared x-grid spanning the union of all methods' supports.
    arrays = {m: np.asarray(v, dtype=float).ravel() for m, v in method_samples.items()}
    if any(a.size == 0 for a in arrays.values()):
        raise ValueError("at least one method has empty samples")
    lo = float(min(a.min() for a in arrays.values()))
    hi = float(max(a.max() for a in arrays.values()))
    pad = 0.05 * (hi - lo) if hi > lo else 1e-6
    grid = np.linspace(lo - pad, hi + pad, 400)

    bins = s["hist_bins"]
    for name, arr in arrays.items():
        ps = _resolve_policy_style(s, name)
        color = ps["color"]
        method_bins = bins if arr.size >= bins else max(arr.size, 1)
        ax.hist(
            arr,
            bins=method_bins,
            density=True,
            color=color,
            alpha=s["alpha_hist"] * 0.4,  # quieter than the singular density helper
        )
        if arr.size >= 2 and float(arr.std(ddof=1)) > 0.0:
            kde = stats.gaussian_kde(arr)
            ax.plot(
                grid, kde(grid),
                color=color,
                linestyle=ps["linestyle"],
                linewidth=s["linewidth"],
                label=ps["label"],
            )
            ax.axvline(
                float(arr.mean()),
                color=color,
                linewidth=0.6,
                linestyle=":",
            )

    ax.set_xlabel(metric_name, fontsize=s["label_fontsize"])
    ax.set_ylabel("density", fontsize=s["label_fontsize"])
    ax.set_title(
        f"{metric_name} -- per-method ensemble distribution",
        fontsize=s["title_fontsize"],
    )
    ax.tick_params(labelsize=s["tick_fontsize"])
    if any(ln.get_label() and not ln.get_label().startswith("_") for ln in ax.lines):
        ax.legend(loc="best", frameon=False, fontsize=s["legend_fontsize"])
    _despine(ax)
    fig.tight_layout()
    return fig, ax


# Per-method alpha multipliers used by plot_path_comparison's seed-spread
# bands. Tabular and linear_fqi share the locked C1 colour, so distinct
# alphas keep them visually separable when both render as fill_between
# bands on a single axes. Tuned by inspection in nb 05; revisit if any
# new policy joins ``policy_styles``.
_PATH_COMPARISON_BAND_ALPHA: dict[str, float] = {
    "bayesian":   0.22,
    "random":     0.18,
    "tabular":    0.18,
    "linear_fqi": 0.10,
}


def plot_path_comparison(
    method_paths: Mapping[str, Mapping[str, object]],
    *,
    bayesian_key: str = "bayesian",
    quantity: str = "cum_log_return",
    bands: tuple[float, ...] = (0.50, 0.95),
    style: dict | None = None,
) -> tuple[Figure, Axes]:
    """Headline cumulative-return panel for nb 05.

    Bayesian gets a full credible band (50% + 95% by default); each other
    method gets a thinner seed-spread band drawn from the same quantile
    levels across its seed axis, plus a median line in the locked policy
    colour. Tabular and linear FQI share the C1 colour and differ by both
    linestyle (locked in ``policy_styles``) and fill_between alpha
    (:data:`_PATH_COMPARISON_BAND_ALPHA`) so the four-method overlay
    remains visually distinguishable.

    Parameters
    ----------
    method_paths
        Mapping ``{method_name: {quantity: (n_runs, T) ndarray, "index":
        DatetimeIndex}}``. ``n_runs`` is ``n_paths`` for Bayesian, ``n_seeds``
        for classical baselines. The ``index`` length must equal ``T``.
    bayesian_key
        Which entry of ``method_paths`` holds the Bayesian posterior bundle
        (full credible band rather than seed-spread band). Defaults to
        ``"bayesian"``. Raises ``ValueError`` if not present.
    quantity
        Which array under each method's bundle to plot. Defaults to
        ``"cum_log_return"``.
    bands
        Iterable of central credible-mass levels in ``(0, 1)``. Default
        ``(0.50, 0.95)``. Applied identically to Bayesian and classical
        methods so the bands measure the same thing on both sides.
    style
        Optional override of :data:`DEFAULT_STYLE`.
    """
    import matplotlib.pyplot as plt

    if bayesian_key not in method_paths:
        raise ValueError(
            f"bayesian_key={bayesian_key!r} not in method_paths keys "
            f"{list(method_paths.keys())!r}"
        )
    if not bands:
        raise ValueError("bands must be non-empty")
    for b in bands:
        if not 0.0 < b < 1.0:
            raise ValueError(f"each band must be strictly in (0, 1), got {b}")

    s = _resolve(style)
    fig, ax = plt.subplots(figsize=s["figsize_single"], dpi=s["dpi"])

    sorted_bands = sorted(set(float(b) for b in bands), reverse=True)

    def _draw(method: str, bundle: Mapping[str, object], *, is_bayesian: bool) -> None:
        ps = _resolve_policy_style(s, method)
        color = ps["color"]
        linestyle = ps["linestyle"]
        label = ps["label"]
        arr = np.asarray(bundle[quantity], dtype=float)
        if arr.ndim != 2:
            raise ValueError(
                f"method {method!r} {quantity!r} must be 2-D (n_runs, T), "
                f"got shape {arr.shape}"
            )
        index = bundle["index"]
        if len(index) != arr.shape[1]:
            raise ValueError(
                f"method {method!r}: len(index)={len(index)} != T={arr.shape[1]}"
            )
        base_alpha = _PATH_COMPARISON_BAND_ALPHA.get(method, 0.15)
        # Inner bands are drawn last (most opaque) for both flavours; for
        # classical bands we scale the base alpha down to keep them quieter
        # than the Bayesian credible band.
        for i, b in enumerate(sorted_bands):
            alpha_low = (1.0 - b) / 2.0
            alpha_high = 1.0 - alpha_low
            lo = np.quantile(arr, alpha_low, axis=0)
            hi = np.quantile(arr, alpha_high, axis=0)
            band_alpha = base_alpha + 0.10 * i if is_bayesian else base_alpha
            ax.fill_between(
                index, lo, hi,
                color=color,
                alpha=band_alpha,
                linewidth=0,
            )
        med = np.quantile(arr, 0.5, axis=0)
        ax.plot(
            index, med,
            color=color,
            linestyle=linestyle,
            linewidth=s["linewidth"],
            label=label,
        )

    # Render order: Bayesian first (so its band sits underneath), then
    # classical bands, with random last so its grey line stays legible on top.
    method_order = [bayesian_key] + [m for m in method_paths.keys() if m != bayesian_key]
    for method in method_order:
        _draw(method, method_paths[method], is_bayesian=(method == bayesian_key))

    ax.axhline(0.0, color="black", linewidth=0.4, linestyle=":")
    ax.set_xlabel("date", fontsize=s["label_fontsize"])
    ax.set_ylabel(quantity, fontsize=s["label_fontsize"])
    ax.tick_params(labelsize=s["tick_fontsize"])
    ax.legend(loc="upper left", frameon=False, fontsize=s["legend_fontsize"])
    _despine(ax)
    fig.tight_layout()
    return fig, ax
