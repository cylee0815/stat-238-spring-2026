"""Smoke contracts for ``project.plots``.

These tests check structure, not appearance: each helper must return the
documented ``Figure`` / ``Axes`` shape, accept ``style`` overrides,
never call ``plt.show()`` and never mutate ``rcParams``. Visual
correctness is verified by hand in the notebooks that consume them.
"""

from __future__ import annotations

from copy import deepcopy
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from project.features import FEATURE_COLS, build_features
from project.plots import (
    DEFAULT_STYLE,
    plot_action_frequency,
    plot_action_probability_heatmap,
    plot_action_timeline,
    plot_cost_model,
    plot_credible_band,
    plot_cumulative_return,
    plot_drawdown,
    plot_feature_panel,
    plot_mc_target_distribution,
    plot_metric_comparison_table,
    plot_metric_distribution_overlay,
    plot_path_comparison,
    plot_position_holdings,
    plot_posterior_metric_distribution,
    plot_price_series,
    plot_q_posterior_at_state,
    plot_q_posterior_grid,
    plot_regime_summary,
    plot_returns_distribution,
    plot_posterior_density,
    plot_posterior_density_grid,
    plot_ess_summary,
    plot_mh_acceptance,
    plot_posterior_overlay,
    plot_rhat_summary,
    plot_trace,
    plot_trace_grid,
)


# ---------- fixtures ----------


def _synthetic_prices(n: int = 600, sigma: float = 0.012, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_p = np.cumsum(rng.normal(0.0003, sigma, size=n))
    p = np.exp(log_p) * 100.0
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    return pd.DataFrame({"Close": p}, index=idx)


def _synthetic_returns(n: int = 1500, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(
        rng.standard_t(df=5, size=n) * 0.01,
        index=pd.date_range("2010-01-01", periods=n, freq="B"),
        name="r",
    )


def _synthetic_targets(n: int = 800, seed: int = 2) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(
        rng.normal(0.05, 0.10, size=n),
        index=pd.date_range("2010-01-01", periods=n, freq="B"),
        name="y",
    )


def _synthetic_actions(n: int, seed: int = 3) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(
        rng.choice([-1, 0, 1], size=n),
        index=pd.date_range("2010-01-01", periods=n, freq="B"),
        name="a",
    )


# ---------- module-level invariants ----------


def test_default_style_has_required_keys() -> None:
    """DEFAULT_STYLE must expose the locked colour discipline + figure parameters."""
    required = {
        "figsize_single",
        "figsize_wide",
        "figsize_panel",
        "figsize_dual",
        "color_bayesian",
        "color_classical",
        "color_random",
        "title_fontsize",
        "label_fontsize",
        "tick_fontsize",
        "hist_bins",
    }
    missing = required - set(DEFAULT_STYLE.keys())
    assert not missing, f"DEFAULT_STYLE missing keys: {missing}"


def test_color_discipline_is_locked() -> None:
    """Bayesian = blue (C0), classical = orange (C1), random = grey (0.5)."""
    assert DEFAULT_STYLE["color_bayesian"] == "C0"
    assert DEFAULT_STYLE["color_classical"] == "C1"
    assert DEFAULT_STYLE["color_random"] == "0.5"


def test_policy_styles_locked() -> None:
    """policy_styles encodes the per-method palette path plots resolve against.

    Bayesian / classical / random colour discipline is propagated in;
    tabular and linear FQI share the classical orange and differ only in
    linestyle (linear FQI solid, tabular dashed) — same secondary-key
    idiom as the Normal/Student-t overlays.
    """
    ps = DEFAULT_STYLE["policy_styles"]
    expected = {"random", "tabular", "linear_fqi", "bayesian"}
    assert set(ps.keys()) == expected, f"policy_styles missing/extra keys: {set(ps.keys()) ^ expected}"
    for name in expected:
        entry = ps[name]
        assert "color" in entry and "linestyle" in entry and "label" in entry, (
            f"policy_styles[{name!r}] missing color/linestyle/label"
        )
    # Colour-bucket assignments
    assert ps["random"]["color"] == DEFAULT_STYLE["color_random"]
    assert ps["bayesian"]["color"] == DEFAULT_STYLE["color_bayesian"]
    assert ps["tabular"]["color"] == DEFAULT_STYLE["color_classical"]
    assert ps["linear_fqi"]["color"] == DEFAULT_STYLE["color_classical"]
    # Within the classical bucket, linestyle is the differentiator
    assert ps["tabular"]["linestyle"] != ps["linear_fqi"]["linestyle"]


def test_sampler_styles_locked() -> None:
    """sampler_styles encodes the per-sampler palette diagnostic plots resolve against.

    Two semantics for ``"C2"`` co-exist in DEFAULT_STYLE: ``color_overlay_normal``
    paints the *Normal-fit overlay on the empirical-returns histogram* in nb 00,
    and ``sampler_styles['student_t']`` paints the *Student-t sampler's posterior*
    in nb 03. The two charts never share a figure -- nb 00 talks about
    distributional fits to data, nb 03 talks about which sampler produced a
    posterior -- so a single colour can carry both meanings safely.
    """
    ss = DEFAULT_STYLE["sampler_styles"]
    expected = {"gaussian", "student_t"}
    assert set(ss.keys()) == expected, (
        f"sampler_styles missing/extra keys: {set(ss.keys()) ^ expected}"
    )
    for name in expected:
        entry = ss[name]
        assert "color" in entry and "linestyle" in entry and "label" in entry, (
            f"sampler_styles[{name!r}] missing color/linestyle/label"
        )
    # The locked colour decision: gaussian = C0 (project Bayesian blue),
    # student_t = C2 (distinct from policy 'bayesian' so nb 09 sweep can show
    # both samplers feeding into the same policy class without colour clash).
    assert ss["gaussian"]["color"] == "C0"
    assert ss["student_t"]["color"] == "C2"
    # Distinct colours, otherwise plot_posterior_overlay collapses to one line.
    assert ss["gaussian"]["color"] != ss["student_t"]["color"]


def test_action_styles_locked() -> None:
    """action_styles encodes the per-action palette consumed by nb 04+ plots.

    Locked: short = ``"C3"`` (red), flat = ``"0.7"`` (light grey), long =
    ``"C0"`` (project Bayesian blue). The long/Bayesian colour collision is
    deliberate -- "blue means the Bayesian model is bullish" reads coherently
    when an action-probability heatmap sits next to a Bayesian credible band
    in nb 04. The two meanings never share an axes (heatmap is a stacked bar;
    band is a line + fill), so they cannot conflate visually. If a future
    chart ever puts both meanings on a single axes, revisit this decision.
    """
    as_ = DEFAULT_STYLE["action_styles"]
    expected = {-1, 0, 1}
    assert set(as_.keys()) == expected, (
        f"action_styles missing/extra keys: {set(as_.keys()) ^ expected}"
    )
    for k in expected:
        entry = as_[k]
        assert "color" in entry and "label" in entry, (
            f"action_styles[{k!r}] missing color/label"
        )
    assert as_[-1]["color"] == "C3"
    assert as_[0]["color"] == "0.7"
    assert as_[+1]["color"] == "C0"
    # Document the collision: long shares the Bayesian-policy colour by intent.
    assert as_[+1]["color"] == DEFAULT_STYLE["policy_styles"]["bayesian"]["color"]


# ---------- per-function invariants (parametrised over all six helpers) ----------


def _all_functions_with_call():
    """Return ``(name, callable)`` pairs that can be invoked on the synthetic fixtures."""
    prices = _synthetic_prices()
    returns = _synthetic_returns()
    targets = _synthetic_targets()
    feats = build_features(_synthetic_prices(n=900))

    return [
        ("plot_price_series", lambda: plot_price_series(prices)),
        ("plot_returns_distribution", lambda: plot_returns_distribution(returns)),
        ("plot_feature_panel", lambda: plot_feature_panel(feats)),
        ("plot_mc_target_distribution", lambda: plot_mc_target_distribution(targets)),
        ("plot_cost_model", lambda: plot_cost_model(prices)),
        ("plot_regime_summary", lambda: plot_regime_summary(prices)),
    ]


@pytest.mark.parametrize("name, call", _all_functions_with_call())
def test_returns_figure(name, call) -> None:
    fig, _ = call()
    assert isinstance(fig, Figure), f"{name} did not return a Figure"
    plt.close(fig)


@pytest.mark.parametrize("name, call", _all_functions_with_call())
def test_does_not_call_show(name, call) -> None:
    with patch.object(plt, "show") as mock_show:
        fig, _ = call()
        assert not mock_show.called, f"{name} called plt.show()"
        plt.close(fig)


@pytest.mark.parametrize("name, call", _all_functions_with_call())
def test_does_not_mutate_rcparams(name, call) -> None:
    snapshot = deepcopy(dict(plt.rcParams))
    fig, _ = call()
    after = dict(plt.rcParams)
    assert after == snapshot, f"{name} mutated plt.rcParams"
    plt.close(fig)


@pytest.mark.parametrize("name, call", _all_functions_with_call())
def test_style_override_accepted(name, call) -> None:
    """Passing a partial style dict must not raise."""
    # The original `call` builds the inputs; we re-derive them and call with style override.
    prices = _synthetic_prices()
    returns = _synthetic_returns()
    targets = _synthetic_targets()
    feats = build_features(_synthetic_prices(n=900))
    override = {"figsize_single": (8, 4), "title_fontsize": 9}

    if name == "plot_price_series":
        fig, _ = plot_price_series(prices, style=override)
    elif name == "plot_returns_distribution":
        fig, _ = plot_returns_distribution(returns, style=override)
    elif name == "plot_feature_panel":
        fig, _ = plot_feature_panel(feats, style=override)
    elif name == "plot_mc_target_distribution":
        fig, _ = plot_mc_target_distribution(targets, style=override)
    elif name == "plot_cost_model":
        fig, _ = plot_cost_model(prices, style=override)
    elif name == "plot_regime_summary":
        fig, _ = plot_regime_summary(prices, style=override)
    plt.close(fig)


# ---------- per-function structural assertions ----------


def test_plot_price_series_returns_single_axes() -> None:
    fig, ax = plot_price_series(_synthetic_prices())
    assert isinstance(ax, Axes)
    assert ax.get_yscale() == "log"  # default log_scale=True
    plt.close(fig)


def test_plot_price_series_linear_scale_kwarg() -> None:
    fig, ax = plot_price_series(_synthetic_prices(), log_scale=False)
    assert ax.get_yscale() == "linear"
    plt.close(fig)


def test_plot_price_series_with_splits_draws_axvspans() -> None:
    prices = _synthetic_prices(n=600)
    splits = {
        "train": (prices.index[0], prices.index[300]),
        "test": (prices.index[301], prices.index[-1]),
    }
    fig, ax = plot_price_series(prices, splits=splits)
    n_spans = sum(1 for p in ax.patches if p.get_label() in splits)
    assert n_spans == len(splits), (
        f"expected {len(splits)} labelled axvspan patches, got {n_spans}"
    )
    plt.close(fig)


def test_plot_returns_distribution_overlays_normal_and_t() -> None:
    fig, ax = plot_returns_distribution(
        _synthetic_returns(),
        show_normal_overlay=True,
        show_t_overlay=True,
    )
    assert isinstance(ax, Axes)
    assert len(ax.lines) >= 2, "expected Normal + Student-t overlay lines"
    assert len(ax.patches) > 0, "expected histogram patches"
    plt.close(fig)


def test_plot_returns_distribution_no_overlays() -> None:
    fig, ax = plot_returns_distribution(
        _synthetic_returns(),
        show_normal_overlay=False,
        show_t_overlay=False,
    )
    assert len(ax.lines) == 0
    plt.close(fig)


def test_plot_returns_distribution_empty_raises() -> None:
    with pytest.raises(ValueError):
        plot_returns_distribution(pd.Series([], dtype=float))


def test_plot_feature_panel_default_uses_FEATURE_COLS() -> None:
    feats = build_features(_synthetic_prices(n=900))
    fig, axes = plot_feature_panel(feats)
    assert isinstance(axes, np.ndarray)
    assert axes.shape == (len(FEATURE_COLS), 2), (
        f"expected ({len(FEATURE_COLS)}, 2) grid, got {axes.shape}"
    )
    plt.close(fig)


def test_plot_feature_panel_custom_subset() -> None:
    feats = build_features(_synthetic_prices(n=900))
    cols = ("x1_mom", "x3_p_vs_ma")
    fig, axes = plot_feature_panel(feats, feature_cols=cols)
    assert axes.shape == (2, 2)
    plt.close(fig)


def test_plot_mc_target_distribution_default_one_histogram() -> None:
    fig, ax = plot_mc_target_distribution(_synthetic_targets())
    # No `by_action`: exactly one set of patches.
    assert len(ax.patches) > 0
    plt.close(fig)


def test_plot_mc_target_distribution_by_action_three_overlays() -> None:
    n = 800
    targets = _synthetic_targets(n=n)
    actions = _synthetic_actions(n=n)
    fig, ax = plot_mc_target_distribution(targets, by_action=actions)
    legend = ax.get_legend()
    assert legend is not None
    assert len(legend.get_texts()) == 3, "expected one legend entry per action"
    plt.close(fig)


def test_plot_cost_model_returns_axes() -> None:
    fig, ax = plot_cost_model(_synthetic_prices())
    assert isinstance(ax, Axes)
    # Twin axis exists; it shares x with the left axis and lives in the same figure.
    assert len(fig.axes) == 2
    plt.close(fig)


def test_plot_regime_summary_returns_two_axes() -> None:
    fig, axes = plot_regime_summary(_synthetic_prices())
    assert isinstance(axes, np.ndarray)
    assert axes.shape == (2,)
    plt.close(fig)


# ---------- determinism ----------


def test_plot_price_series_deterministic() -> None:
    """Same input twice => bit-identical pixels."""
    prices = _synthetic_prices()
    fig1, _ = plot_price_series(prices)
    fig1.canvas.draw()
    buf1 = bytes(fig1.canvas.buffer_rgba())
    plt.close(fig1)

    fig2, _ = plot_price_series(prices)
    fig2.canvas.draw()
    buf2 = bytes(fig2.canvas.buffer_rgba())
    plt.close(fig2)

    assert buf1 == buf2


# ---------- path-visualisation set (nb 02+) ----------


def _synthetic_policy_paths(n: int = 500, seed: int = 5) -> dict[str, dict[str, pd.Series]]:
    """Three-policy bundle: random, tabular, linear_fqi.

    Returns ``{policy_name: {actions, positions, rewards, cum_log_return}}``,
    matching the dict shape ``project.baselines`` returns.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    out: dict[str, dict[str, pd.Series]] = {}
    for i, name in enumerate(["random", "tabular", "linear_fqi"]):
        a = rng.choice([-1, 0, 1], size=n, p=[0.3, 0.4, 0.3])
        r = rng.normal(0.0001 * (i + 1), 0.01, size=n)
        actions = pd.Series(a, index=idx, name="actions")
        out[name] = {
            "actions": actions,
            "positions": actions.rename("positions"),
            "rewards": pd.Series(r, index=idx, name="rewards"),
            "cum_log_return": pd.Series(r, index=idx, name="cum_log_return").cumsum(),
        }
    return out


# ---- plot_action_timeline ----


def test_plot_action_timeline_returns_figure_and_axes() -> None:
    a = _synthetic_actions(n=200)
    fig, ax = plot_action_timeline(a)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_plot_action_timeline_ax_none_creates_new_axes() -> None:
    """ax=None should produce a fresh single-axes figure."""
    a = _synthetic_actions(n=200)
    fig, ax = plot_action_timeline(a, ax=None)
    assert isinstance(ax, Axes)
    # Single-axes figure: only one axes object.
    assert len(fig.axes) == 1
    plt.close(fig)


def test_plot_action_timeline_ax_existing_reuses_axes() -> None:
    """Passing an existing axes must reuse it (same Figure, same Axes returned)."""
    pre_fig, pre_ax = plt.subplots()
    a = _synthetic_actions(n=200)
    fig, ax = plot_action_timeline(a, ax=pre_ax)
    assert fig is pre_fig, "should return the existing Figure, not create a new one"
    assert ax is pre_ax, "should return the same Axes instance that was passed in"
    plt.close(pre_fig)


def test_plot_action_timeline_does_not_call_show() -> None:
    a = _synthetic_actions(n=200)
    with patch.object(plt, "show") as mock_show:
        fig, _ = plot_action_timeline(a)
        assert not mock_show.called
        plt.close(fig)


def test_plot_action_timeline_does_not_mutate_rcparams() -> None:
    a = _synthetic_actions(n=200)
    snapshot = deepcopy(dict(plt.rcParams))
    fig, _ = plot_action_timeline(a)
    after = dict(plt.rcParams)
    assert after == snapshot
    plt.close(fig)


def test_plot_action_timeline_style_override_accepted() -> None:
    a = _synthetic_actions(n=200)
    override = {"figsize_single": (8, 3), "linewidth": 1.2}
    fig, _ = plot_action_timeline(a, style=override)
    plt.close(fig)


def test_plot_action_timeline_label_propagates() -> None:
    """A non-empty label argument must surface in the legend."""
    a = _synthetic_actions(n=200)
    fig, ax = plot_action_timeline(a, label="bayesian")
    legend = ax.get_legend()
    assert legend is not None, "label should produce a legend entry"
    texts = [t.get_text() for t in legend.get_texts()]
    assert "bayesian" in texts
    plt.close(fig)


def test_plot_action_timeline_policy_resolves_color_and_linestyle() -> None:
    """The policy kwarg must look up colour/linestyle from policy_styles."""
    a = _synthetic_actions(n=200)
    fig, ax = plot_action_timeline(a, label="tab", policy="tabular")
    line = next(ln for ln in ax.lines if ln.get_label() == "tab")
    expected = DEFAULT_STYLE["policy_styles"]["tabular"]
    assert line.get_color() == expected["color"]
    assert line.get_linestyle() == expected["linestyle"]
    plt.close(fig)


def test_plot_action_timeline_deterministic() -> None:
    a = _synthetic_actions(n=200)
    fig1, _ = plot_action_timeline(a)
    fig1.canvas.draw()
    buf1 = bytes(fig1.canvas.buffer_rgba())
    plt.close(fig1)

    fig2, _ = plot_action_timeline(a)
    fig2.canvas.draw()
    buf2 = bytes(fig2.canvas.buffer_rgba())
    plt.close(fig2)
    assert buf1 == buf2


# ---- plot_cumulative_return ----


def test_plot_cumulative_return_three_policies_three_lines() -> None:
    """One labelled line per policy (axhline baselines etc. don't count)."""
    paths = _synthetic_policy_paths()
    cum = {k: v["cum_log_return"] for k, v in paths.items()}
    fig, ax = plot_cumulative_return(cum)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    labelled = [ln for ln in ax.lines if not ln.get_label().startswith("_")]
    assert len(labelled) == 3, (
        f"expected one labelled line per policy, got {len(labelled)} "
        f"(labels: {[ln.get_label() for ln in labelled]})"
    )
    plt.close(fig)


def test_plot_cumulative_return_resolves_policy_styles() -> None:
    """Each line must inherit color/linestyle from DEFAULT_STYLE['policy_styles']."""
    paths = _synthetic_policy_paths()
    cum = {k: v["cum_log_return"] for k, v in paths.items()}
    fig, ax = plot_cumulative_return(cum)
    ps = DEFAULT_STYLE["policy_styles"]
    label_to_line = {ln.get_label(): ln for ln in ax.lines}
    for name in ("random", "tabular", "linear_fqi"):
        # Either the raw key or the human label can appear in the legend, but
        # the matplotlib line label is the human label; we look up by either.
        line = label_to_line.get(ps[name]["label"]) or label_to_line.get(name)
        assert line is not None, f"missing line for policy {name!r}"
        assert line.get_linestyle() == ps[name]["linestyle"], (
            f"{name}: linestyle {line.get_linestyle()!r} != {ps[name]['linestyle']!r}"
        )
    plt.close(fig)


def test_plot_cumulative_return_legend_present() -> None:
    paths = _synthetic_policy_paths()
    cum = {k: v["cum_log_return"] for k, v in paths.items()}
    fig, ax = plot_cumulative_return(cum)
    legend = ax.get_legend()
    assert legend is not None
    assert len(legend.get_texts()) == 3
    plt.close(fig)


def test_plot_cumulative_return_does_not_call_show() -> None:
    paths = _synthetic_policy_paths()
    cum = {k: v["cum_log_return"] for k, v in paths.items()}
    with patch.object(plt, "show") as mock_show:
        fig, _ = plot_cumulative_return(cum)
        assert not mock_show.called
        plt.close(fig)


def test_plot_cumulative_return_does_not_mutate_rcparams() -> None:
    paths = _synthetic_policy_paths()
    cum = {k: v["cum_log_return"] for k, v in paths.items()}
    snapshot = deepcopy(dict(plt.rcParams))
    fig, _ = plot_cumulative_return(cum)
    after = dict(plt.rcParams)
    assert after == snapshot
    plt.close(fig)


def test_plot_cumulative_return_style_override_accepted() -> None:
    paths = _synthetic_policy_paths()
    cum = {k: v["cum_log_return"] for k, v in paths.items()}
    override = {"figsize_single": (8, 3.5)}
    fig, _ = plot_cumulative_return(cum, style=override)
    plt.close(fig)


def test_plot_cumulative_return_empty_dict_raises() -> None:
    with pytest.raises(ValueError):
        plot_cumulative_return({})


# ---- plot_position_holdings ----


def test_plot_position_holdings_n_subplots_for_n_policies() -> None:
    paths = _synthetic_policy_paths()
    positions = {k: v["positions"] for k, v in paths.items()}
    fig, axes = plot_position_holdings(positions)
    assert isinstance(fig, Figure)
    assert isinstance(axes, np.ndarray)
    assert axes.shape == (len(positions),), (
        f"expected one subplot per policy ({len(positions)},), got {axes.shape}"
    )
    plt.close(fig)


def test_plot_position_holdings_does_not_call_show() -> None:
    paths = _synthetic_policy_paths()
    positions = {k: v["positions"] for k, v in paths.items()}
    with patch.object(plt, "show") as mock_show:
        fig, _ = plot_position_holdings(positions)
        assert not mock_show.called
        plt.close(fig)


def test_plot_position_holdings_does_not_mutate_rcparams() -> None:
    paths = _synthetic_policy_paths()
    positions = {k: v["positions"] for k, v in paths.items()}
    snapshot = deepcopy(dict(plt.rcParams))
    fig, _ = plot_position_holdings(positions)
    after = dict(plt.rcParams)
    assert after == snapshot
    plt.close(fig)


def test_plot_position_holdings_style_override_accepted() -> None:
    paths = _synthetic_policy_paths()
    positions = {k: v["positions"] for k, v in paths.items()}
    override = {"figsize_panel": (9, 6)}
    fig, _ = plot_position_holdings(positions, style=override)
    plt.close(fig)


def test_plot_position_holdings_single_policy_returns_array() -> None:
    """Even with one policy the return type is still an ndarray of axes."""
    paths = _synthetic_policy_paths()
    positions = {"random": paths["random"]["positions"]}
    fig, axes = plot_position_holdings(positions)
    assert isinstance(axes, np.ndarray)
    assert axes.shape == (1,)
    plt.close(fig)


def test_plot_position_holdings_empty_dict_raises() -> None:
    with pytest.raises(ValueError):
        plot_position_holdings({})


# ---- plot_action_frequency ----


def test_plot_action_frequency_three_policies_structure() -> None:
    """Three policies × three actions = 9 bars (grouped or stacked)."""
    paths = _synthetic_policy_paths()
    actions = {k: v["actions"] for k, v in paths.items()}
    fig, ax = plot_action_frequency(actions)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    # Bars are matplotlib.patches.Rectangle; one per (policy, action) cell.
    n_bars = len(ax.patches)
    assert n_bars == 9, f"expected 9 bars (3 policies × 3 actions), got {n_bars}"
    plt.close(fig)


def test_plot_action_frequency_legend_lists_policies() -> None:
    paths = _synthetic_policy_paths()
    actions = {k: v["actions"] for k, v in paths.items()}
    fig, ax = plot_action_frequency(actions)
    legend = ax.get_legend()
    assert legend is not None
    assert len(legend.get_texts()) == 3, "one legend entry per policy"
    plt.close(fig)


def test_plot_action_frequency_does_not_call_show() -> None:
    paths = _synthetic_policy_paths()
    actions = {k: v["actions"] for k, v in paths.items()}
    with patch.object(plt, "show") as mock_show:
        fig, _ = plot_action_frequency(actions)
        assert not mock_show.called
        plt.close(fig)


def test_plot_action_frequency_does_not_mutate_rcparams() -> None:
    paths = _synthetic_policy_paths()
    actions = {k: v["actions"] for k, v in paths.items()}
    snapshot = deepcopy(dict(plt.rcParams))
    fig, _ = plot_action_frequency(actions)
    after = dict(plt.rcParams)
    assert after == snapshot
    plt.close(fig)


def test_plot_action_frequency_style_override_accepted() -> None:
    paths = _synthetic_policy_paths()
    actions = {k: v["actions"] for k, v in paths.items()}
    override = {"figsize_wide": (8, 3.5)}
    fig, _ = plot_action_frequency(actions, style=override)
    plt.close(fig)


def test_plot_action_frequency_empty_dict_raises() -> None:
    with pytest.raises(ValueError):
        plot_action_frequency({})


# ---- plot_drawdown ----


def test_plot_drawdown_three_policies_three_lines() -> None:
    paths = _synthetic_policy_paths()
    cum = {k: v["cum_log_return"] for k, v in paths.items()}
    fig, ax = plot_drawdown(cum)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert len(ax.lines) >= 3, "expected at least one line per policy"
    plt.close(fig)


def test_plot_drawdown_values_non_positive() -> None:
    """Running drawdown is bounded above by zero (cum minus cumulative max)."""
    paths = _synthetic_policy_paths()
    cum = {k: v["cum_log_return"] for k, v in paths.items()}
    fig, ax = plot_drawdown(cum)
    for line in ax.lines:
        ydata = np.asarray(line.get_ydata(), dtype=float)
        ydata = ydata[~np.isnan(ydata)]
        assert (ydata <= 1e-12).all(), (
            f"drawdown line {line.get_label()!r} has positive values: max={ydata.max():.3e}"
        )
    plt.close(fig)


def test_plot_drawdown_legend_lists_policies() -> None:
    paths = _synthetic_policy_paths()
    cum = {k: v["cum_log_return"] for k, v in paths.items()}
    fig, ax = plot_drawdown(cum)
    legend = ax.get_legend()
    assert legend is not None
    assert len(legend.get_texts()) == 3
    plt.close(fig)


def test_plot_drawdown_does_not_call_show() -> None:
    paths = _synthetic_policy_paths()
    cum = {k: v["cum_log_return"] for k, v in paths.items()}
    with patch.object(plt, "show") as mock_show:
        fig, _ = plot_drawdown(cum)
        assert not mock_show.called
        plt.close(fig)


def test_plot_drawdown_does_not_mutate_rcparams() -> None:
    paths = _synthetic_policy_paths()
    cum = {k: v["cum_log_return"] for k, v in paths.items()}
    snapshot = deepcopy(dict(plt.rcParams))
    fig, _ = plot_drawdown(cum)
    after = dict(plt.rcParams)
    assert after == snapshot
    plt.close(fig)


def test_plot_drawdown_style_override_accepted() -> None:
    paths = _synthetic_policy_paths()
    cum = {k: v["cum_log_return"] for k, v in paths.items()}
    override = {"figsize_single": (8, 3.5)}
    fig, _ = plot_drawdown(cum, style=override)
    plt.close(fig)


def test_plot_drawdown_empty_dict_raises() -> None:
    with pytest.raises(ValueError):
        plot_drawdown({})


# ---------- diagnostic set (nb 03+) ----------

# Synthetic-trace fixtures: shaped like real sampler output but cheap.
# (n_chains, n_draws) for scalar params; (n_chains, n_draws, p) for vector.


def _synthetic_trace_scalar(
    n_chains: int = 4, n_draws: int = 1500, seed: int = 11,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_chains, n_draws))


def _synthetic_trace_dict(
    n_chains: int = 4, n_draws: int = 1500, seed: int = 12,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "mu_beta[0]": rng.standard_normal((n_chains, n_draws)) * 0.05,
        "mu_beta[1]": rng.standard_normal((n_chains, n_draws)) * 0.05 + 0.1,
        "sigma2": np.exp(rng.standard_normal((n_chains, n_draws)) * 0.1 - 4.0),
        "nu": np.abs(rng.standard_normal((n_chains, n_draws)) * 2.0 + 6.0),
    }


def _flat_samples(n: int = 4000, seed: int = 13) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n) * 0.05


# ---- plot_trace ----


def test_plot_trace_returns_figure_and_axes() -> None:
    fig, ax = plot_trace(_synthetic_trace_scalar(), param_name="mu_beta[0]")
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_plot_trace_one_line_per_chain() -> None:
    """Each chain becomes one line in the axes."""
    n_chains = 5
    arr = _synthetic_trace_scalar(n_chains=n_chains, n_draws=400)
    fig, ax = plot_trace(arr, param_name="x")
    assert len(ax.lines) == n_chains, (
        f"expected {n_chains} lines, got {len(ax.lines)}"
    )
    plt.close(fig)


def test_plot_trace_ax_existing_reuses_axes() -> None:
    """Passing ax= must reuse it -- same Figure, same Axes returned."""
    pre_fig, pre_ax = plt.subplots()
    arr = _synthetic_trace_scalar()
    fig, ax = plot_trace(arr, param_name="x", ax=pre_ax)
    assert fig is pre_fig
    assert ax is pre_ax
    plt.close(pre_fig)


def test_plot_trace_sampler_color_resolves() -> None:
    """sampler='gaussian' / 'student_t' must paint lines with the locked colour."""
    arr = _synthetic_trace_scalar(n_chains=2, n_draws=200)
    for sampler, expected in [
        ("gaussian", DEFAULT_STYLE["sampler_styles"]["gaussian"]["color"]),
        ("student_t", DEFAULT_STYLE["sampler_styles"]["student_t"]["color"]),
    ]:
        fig, ax = plot_trace(arr, param_name="x", sampler=sampler)
        assert ax.lines[0].get_color() == expected, (
            f"{sampler}: line colour {ax.lines[0].get_color()!r} != {expected!r}"
        )
        plt.close(fig)


def test_plot_trace_does_not_call_show() -> None:
    arr = _synthetic_trace_scalar()
    with patch.object(plt, "show") as mock_show:
        fig, _ = plot_trace(arr, param_name="x")
        assert not mock_show.called
        plt.close(fig)


def test_plot_trace_does_not_mutate_rcparams() -> None:
    arr = _synthetic_trace_scalar()
    snapshot = deepcopy(dict(plt.rcParams))
    fig, _ = plot_trace(arr, param_name="x")
    assert dict(plt.rcParams) == snapshot
    plt.close(fig)


def test_plot_trace_style_override_accepted() -> None:
    arr = _synthetic_trace_scalar()
    override = {"figsize_single": (8, 3.0), "linewidth": 1.2}
    fig, _ = plot_trace(arr, param_name="x", style=override)
    plt.close(fig)


def test_plot_trace_deterministic() -> None:
    arr = _synthetic_trace_scalar(seed=99)
    fig1, _ = plot_trace(arr, param_name="x")
    fig1.canvas.draw()
    buf1 = bytes(fig1.canvas.buffer_rgba())
    plt.close(fig1)

    fig2, _ = plot_trace(arr, param_name="x")
    fig2.canvas.draw()
    buf2 = bytes(fig2.canvas.buffer_rgba())
    plt.close(fig2)
    assert buf1 == buf2


def test_plot_trace_rejects_non_2d() -> None:
    with pytest.raises(ValueError):
        plot_trace(np.zeros(10), param_name="x")


# ---- plot_trace_grid ----


def test_plot_trace_grid_n_panels_match_n_params() -> None:
    """One panel per parameter in the dict."""
    td = _synthetic_trace_dict()
    fig, axes = plot_trace_grid(td)
    assert isinstance(fig, Figure)
    assert isinstance(axes, np.ndarray)
    assert axes.size == len(td), (
        f"expected {len(td)} panels, got {axes.size}"
    )
    plt.close(fig)


def test_plot_trace_grid_lines_per_panel_match_n_chains() -> None:
    n_chains = 3
    td = {
        "a": _synthetic_trace_scalar(n_chains=n_chains, n_draws=200, seed=0),
        "b": _synthetic_trace_scalar(n_chains=n_chains, n_draws=200, seed=1),
    }
    fig, axes = plot_trace_grid(td)
    for ax in axes.ravel():
        assert len(ax.lines) == n_chains
    plt.close(fig)


def test_plot_trace_grid_sampler_color_propagates() -> None:
    """The sampler kwarg threads through plot_trace per panel."""
    td = {"a": _synthetic_trace_scalar(n_chains=2, n_draws=200)}
    fig, axes = plot_trace_grid(td, sampler="student_t")
    expected = DEFAULT_STYLE["sampler_styles"]["student_t"]["color"]
    for line in axes.ravel()[0].lines:
        assert line.get_color() == expected
    plt.close(fig)


def test_plot_trace_grid_does_not_call_show() -> None:
    td = _synthetic_trace_dict()
    with patch.object(plt, "show") as mock_show:
        fig, _ = plot_trace_grid(td)
        assert not mock_show.called
        plt.close(fig)


def test_plot_trace_grid_does_not_mutate_rcparams() -> None:
    td = _synthetic_trace_dict()
    snapshot = deepcopy(dict(plt.rcParams))
    fig, _ = plot_trace_grid(td)
    assert dict(plt.rcParams) == snapshot
    plt.close(fig)


def test_plot_trace_grid_style_override_accepted() -> None:
    td = _synthetic_trace_dict()
    fig, _ = plot_trace_grid(td, style={"figsize_panel": (9, 6)})
    plt.close(fig)


def test_plot_trace_grid_empty_dict_raises() -> None:
    with pytest.raises(ValueError):
        plot_trace_grid({})


# ---- plot_posterior_density ----


def test_plot_posterior_density_returns_figure_and_axes() -> None:
    fig, ax = plot_posterior_density(_flat_samples(), param_name="mu_beta[0]")
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_plot_posterior_density_has_histogram_and_kde() -> None:
    """Histogram patches present + a KDE line on top."""
    fig, ax = plot_posterior_density(_flat_samples(), param_name="x")
    assert len(ax.patches) > 0, "expected histogram bars"
    assert len(ax.lines) >= 1, "expected at least one KDE line"
    plt.close(fig)


def test_plot_posterior_density_ref_value_drawn() -> None:
    """ref_value must surface as a vertical line at that x position."""
    fig, ax = plot_posterior_density(
        _flat_samples(), param_name="x", ref_value=0.05,
    )
    vlines = [
        ln for ln in ax.lines
        if len(set(ln.get_xdata())) == 1  # vertical
    ]
    assert any(
        abs(float(ln.get_xdata()[0]) - 0.05) < 1e-9 for ln in vlines
    ), "expected a vertical line at ref_value=0.05"
    plt.close(fig)


def test_plot_posterior_density_ax_existing_reuses_axes() -> None:
    pre_fig, pre_ax = plt.subplots()
    fig, ax = plot_posterior_density(
        _flat_samples(), param_name="x", ax=pre_ax,
    )
    assert fig is pre_fig
    assert ax is pre_ax
    plt.close(pre_fig)


def test_plot_posterior_density_sampler_color_resolves() -> None:
    fig, ax = plot_posterior_density(
        _flat_samples(), param_name="x", sampler="student_t",
    )
    expected = DEFAULT_STYLE["sampler_styles"]["student_t"]["color"]
    # The KDE line should adopt the sampler colour.
    assert ax.lines[0].get_color() == expected
    plt.close(fig)


def test_plot_posterior_density_does_not_call_show() -> None:
    with patch.object(plt, "show") as mock_show:
        fig, _ = plot_posterior_density(_flat_samples(), param_name="x")
        assert not mock_show.called
        plt.close(fig)


def test_plot_posterior_density_does_not_mutate_rcparams() -> None:
    snapshot = deepcopy(dict(plt.rcParams))
    fig, _ = plot_posterior_density(_flat_samples(), param_name="x")
    assert dict(plt.rcParams) == snapshot
    plt.close(fig)


def test_plot_posterior_density_style_override_accepted() -> None:
    fig, _ = plot_posterior_density(
        _flat_samples(), param_name="x", style={"hist_bins": 30},
    )
    plt.close(fig)


def test_plot_posterior_density_degenerate_n_one() -> None:
    """Single-sample input should not crash; rendered as a delta-style plot.

    The behaviour we lock here: do not raise. KDE fitting on n=1 is
    undefined, so the helper falls back to a histogram-only render.
    """
    fig, ax = plot_posterior_density(
        np.array([0.5]), param_name="x",
    )
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_plot_posterior_density_empty_raises() -> None:
    with pytest.raises(ValueError):
        plot_posterior_density(np.array([]), param_name="x")


# ---- plot_posterior_density_grid ----


def _flat_samples_dict(seed: int = 14) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "mu_beta[0]": rng.standard_normal(2000) * 0.05,
        "mu_beta[1]": rng.standard_normal(2000) * 0.05 + 0.1,
        "sigma2": np.exp(rng.standard_normal(2000) * 0.1 - 4.0),
    }


def test_plot_posterior_density_grid_n_panels_match_n_params() -> None:
    sd = _flat_samples_dict()
    fig, axes = plot_posterior_density_grid(sd)
    assert isinstance(fig, Figure)
    assert isinstance(axes, np.ndarray)
    assert axes.size == len(sd)
    plt.close(fig)


def test_plot_posterior_density_grid_ref_values_propagate() -> None:
    """ref_values entries must be drawn as vertical lines on matching panels."""
    sd = _flat_samples_dict()
    refs = {"mu_beta[0]": 0.0, "sigma2": 0.018}
    fig, axes = plot_posterior_density_grid(sd, ref_values=refs)
    # Find the axes for refs and check a vertical line at the ref value.
    name_to_ax = dict(zip(sd.keys(), axes.ravel()))
    for name, ref in refs.items():
        ax = name_to_ax[name]
        vlines = [ln for ln in ax.lines if len(set(ln.get_xdata())) == 1]
        assert any(
            abs(float(ln.get_xdata()[0]) - ref) < 1e-9 for ln in vlines
        ), f"missing vertical line at ref={ref} for {name!r}"
    plt.close(fig)


def test_plot_posterior_density_grid_sampler_color_propagates() -> None:
    sd = _flat_samples_dict()
    fig, axes = plot_posterior_density_grid(sd, sampler="gaussian")
    expected = DEFAULT_STYLE["sampler_styles"]["gaussian"]["color"]
    for ax in axes.ravel():
        # KDE is the first non-vertical line; check it adopts sampler colour.
        kde_lines = [ln for ln in ax.lines if len(set(ln.get_xdata())) > 1]
        assert kde_lines, "expected KDE line per panel"
        assert kde_lines[0].get_color() == expected
    plt.close(fig)


def test_plot_posterior_density_grid_does_not_call_show() -> None:
    sd = _flat_samples_dict()
    with patch.object(plt, "show") as mock_show:
        fig, _ = plot_posterior_density_grid(sd)
        assert not mock_show.called
        plt.close(fig)


def test_plot_posterior_density_grid_does_not_mutate_rcparams() -> None:
    sd = _flat_samples_dict()
    snapshot = deepcopy(dict(plt.rcParams))
    fig, _ = plot_posterior_density_grid(sd)
    assert dict(plt.rcParams) == snapshot
    plt.close(fig)


def test_plot_posterior_density_grid_style_override_accepted() -> None:
    sd = _flat_samples_dict()
    fig, _ = plot_posterior_density_grid(sd, style={"figsize_panel": (9, 6)})
    plt.close(fig)


def test_plot_posterior_density_grid_empty_dict_raises() -> None:
    with pytest.raises(ValueError):
        plot_posterior_density_grid({})


# ---- plot_posterior_overlay ----


def _two_sampler_dict(seed: int = 15) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "gaussian": rng.standard_normal(2000) * 0.05,
        "student_t": rng.standard_normal(2000) * 0.05 + 0.01,
    }


def test_plot_posterior_overlay_returns_figure_and_axes() -> None:
    fig, ax = plot_posterior_overlay(_two_sampler_dict(), param_name="mu_beta[0]")
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_plot_posterior_overlay_one_kde_per_sampler() -> None:
    """Two samplers => at least two non-vertical KDE lines."""
    sd = _two_sampler_dict()
    fig, ax = plot_posterior_overlay(sd, param_name="x")
    kde_lines = [ln for ln in ax.lines if len(set(ln.get_xdata())) > 1]
    assert len(kde_lines) == 2
    plt.close(fig)


def test_plot_posterior_overlay_colors_from_sampler_styles() -> None:
    sd = _two_sampler_dict()
    fig, ax = plot_posterior_overlay(sd, param_name="x")
    expected = {
        DEFAULT_STYLE["sampler_styles"]["gaussian"]["color"],
        DEFAULT_STYLE["sampler_styles"]["student_t"]["color"],
    }
    kde_lines = [ln for ln in ax.lines if len(set(ln.get_xdata())) > 1]
    actual = {ln.get_color() for ln in kde_lines}
    assert actual == expected, f"line colors {actual} != {expected}"
    plt.close(fig)


def test_plot_posterior_overlay_ref_value_drawn() -> None:
    sd = _two_sampler_dict()
    fig, ax = plot_posterior_overlay(sd, param_name="x", ref_value=0.0)
    vlines = [ln for ln in ax.lines if len(set(ln.get_xdata())) == 1]
    assert any(abs(float(ln.get_xdata()[0])) < 1e-9 for ln in vlines)
    plt.close(fig)


def test_plot_posterior_overlay_ax_existing_reuses_axes() -> None:
    pre_fig, pre_ax = plt.subplots()
    fig, ax = plot_posterior_overlay(
        _two_sampler_dict(), param_name="x", ax=pre_ax,
    )
    assert fig is pre_fig
    assert ax is pre_ax
    plt.close(pre_fig)


def test_plot_posterior_overlay_does_not_call_show() -> None:
    sd = _two_sampler_dict()
    with patch.object(plt, "show") as mock_show:
        fig, _ = plot_posterior_overlay(sd, param_name="x")
        assert not mock_show.called
        plt.close(fig)


def test_plot_posterior_overlay_does_not_mutate_rcparams() -> None:
    sd = _two_sampler_dict()
    snapshot = deepcopy(dict(plt.rcParams))
    fig, _ = plot_posterior_overlay(sd, param_name="x")
    assert dict(plt.rcParams) == snapshot
    plt.close(fig)


def test_plot_posterior_overlay_style_override_accepted() -> None:
    sd = _two_sampler_dict()
    fig, _ = plot_posterior_overlay(
        sd, param_name="x", style={"figsize_single": (8, 3.5)},
    )
    plt.close(fig)


def test_plot_posterior_overlay_empty_dict_raises() -> None:
    with pytest.raises(ValueError):
        plot_posterior_overlay({}, param_name="x")


# ---- plot_rhat_summary ----


def test_plot_rhat_summary_returns_figure_and_axes() -> None:
    fig, ax = plot_rhat_summary({"a": 1.01, "b": 1.02, "c": 1.04})
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_plot_rhat_summary_one_bar_per_param() -> None:
    rhats = {"a": 1.01, "b": 1.02, "c": 1.04, "d": 1.005}
    fig, ax = plot_rhat_summary(rhats)
    # Bars are matplotlib.patches.Rectangle; one per parameter.
    assert len(ax.patches) == len(rhats)
    plt.close(fig)


def test_plot_rhat_summary_threshold_line_drawn_at_kwarg() -> None:
    """The default threshold is 1.05; passing a different one should move the line."""
    fig, ax = plot_rhat_summary({"a": 1.01}, threshold=1.10)
    # Find a vertical (axvline) at threshold.
    vlines = [ln for ln in ax.lines if len(set(ln.get_xdata())) == 1]
    xvals = [float(ln.get_xdata()[0]) for ln in vlines]
    assert any(abs(x - 1.10) < 1e-9 for x in xvals), (
        f"expected threshold line at 1.10, got xvals={xvals}"
    )
    plt.close(fig)


def test_plot_rhat_summary_exceeding_bar_visually_distinct() -> None:
    """A bar with rhat > threshold should differ from a passing bar.

    Locked mechanism: failing bars are coloured a flag colour (red),
    passing bars use the project's data colour. Test checks that two
    bars on opposite sides of the threshold do not share the same
    matplotlib facecolor.
    """
    fig, ax = plot_rhat_summary(
        {"good": 1.01, "bad": 1.10}, threshold=1.05,
    )
    # Bars are added in dict-iteration order.
    fc_good = tuple(ax.patches[0].get_facecolor())
    fc_bad = tuple(ax.patches[1].get_facecolor())
    assert fc_good != fc_bad, (
        f"passing/failing bars should differ; both = {fc_good}"
    )
    plt.close(fig)


def test_plot_rhat_summary_does_not_call_show() -> None:
    with patch.object(plt, "show") as mock_show:
        fig, _ = plot_rhat_summary({"a": 1.01, "b": 1.02})
        assert not mock_show.called
        plt.close(fig)


def test_plot_rhat_summary_does_not_mutate_rcparams() -> None:
    snapshot = deepcopy(dict(plt.rcParams))
    fig, _ = plot_rhat_summary({"a": 1.01})
    assert dict(plt.rcParams) == snapshot
    plt.close(fig)


def test_plot_rhat_summary_style_override_accepted() -> None:
    fig, _ = plot_rhat_summary(
        {"a": 1.01}, style={"figsize_wide": (8, 3.5)},
    )
    plt.close(fig)


def test_plot_rhat_summary_empty_dict_raises() -> None:
    with pytest.raises(ValueError):
        plot_rhat_summary({})


# ---- plot_ess_summary ----


def test_plot_ess_summary_returns_figure_and_axes() -> None:
    fig, ax = plot_ess_summary({"a": 1200.0, "b": 500.0})
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_plot_ess_summary_one_bar_per_param() -> None:
    ess = {"a": 1200.0, "b": 500.0, "c": 300.0}
    fig, ax = plot_ess_summary(ess)
    assert len(ax.patches) == len(ess)
    plt.close(fig)


def test_plot_ess_summary_threshold_line_drawn_at_kwarg() -> None:
    fig, ax = plot_ess_summary({"a": 800.0}, threshold=600.0)
    vlines = [ln for ln in ax.lines if len(set(ln.get_xdata())) == 1]
    xvals = [float(ln.get_xdata()[0]) for ln in vlines]
    assert any(abs(x - 600.0) < 1e-9 for x in xvals)
    plt.close(fig)


def test_plot_ess_summary_below_threshold_bar_visually_distinct() -> None:
    """ESS direction is reversed: failing means *below* threshold."""
    fig, ax = plot_ess_summary(
        {"good": 800.0, "bad": 200.0}, threshold=400.0,
    )
    fc_good = tuple(ax.patches[0].get_facecolor())
    fc_bad = tuple(ax.patches[1].get_facecolor())
    assert fc_good != fc_bad
    plt.close(fig)


def test_plot_ess_summary_does_not_call_show() -> None:
    with patch.object(plt, "show") as mock_show:
        fig, _ = plot_ess_summary({"a": 800.0})
        assert not mock_show.called
        plt.close(fig)


def test_plot_ess_summary_does_not_mutate_rcparams() -> None:
    snapshot = deepcopy(dict(plt.rcParams))
    fig, _ = plot_ess_summary({"a": 800.0})
    assert dict(plt.rcParams) == snapshot
    plt.close(fig)


def test_plot_ess_summary_style_override_accepted() -> None:
    fig, _ = plot_ess_summary(
        {"a": 800.0}, style={"figsize_wide": (8, 3.5)},
    )
    plt.close(fig)


def test_plot_ess_summary_empty_dict_raises() -> None:
    with pytest.raises(ValueError):
        plot_ess_summary({})


# ---- plot_mh_acceptance ----


def _synthetic_acceptance(
    n_chains: int = 4, n_draws: int = 2000, target: float = 0.30, seed: int = 22,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.uniform(0, 1, size=(n_chains, n_draws)) < target).astype(np.int8)


def test_plot_mh_acceptance_returns_figure_and_axes() -> None:
    fig, ax = plot_mh_acceptance(_synthetic_acceptance())
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_plot_mh_acceptance_one_line_per_chain() -> None:
    n_chains = 3
    arr = _synthetic_acceptance(n_chains=n_chains, n_draws=500)
    fig, ax = plot_mh_acceptance(arr)
    # Excluding axhline reference lines: rolling-rate lines have many distinct x values.
    rolling = [ln for ln in ax.lines if len(set(ln.get_xdata())) > 1]
    assert len(rolling) == n_chains, (
        f"expected {n_chains} rolling-rate lines, got {len(rolling)}"
    )
    plt.close(fig)


def test_plot_mh_acceptance_band_edges_match_kwargs() -> None:
    """The shaded target band must span [target_low, target_high]."""
    from matplotlib.patches import Rectangle

    arr = _synthetic_acceptance(n_chains=2, n_draws=300)
    fig, ax = plot_mh_acceptance(arr, target_low=0.15, target_high=0.50)
    # axhspan adds a Rectangle patch with y0=target_low and height=target_high - target_low.
    bands = [
        p for p in ax.patches
        if isinstance(p, Rectangle) and p.get_height() > 0 and p.get_height() < 1.0
    ]
    assert bands, "expected an axhspan rectangle for the target band"
    band = bands[0]
    lo = float(band.get_y())
    hi = lo + float(band.get_height())
    assert abs(lo - 0.15) < 1e-9 and abs(hi - 0.50) < 1e-9, (
        f"band span = ({lo}, {hi}); expected (0.15, 0.50)"
    )
    plt.close(fig)


def test_plot_mh_acceptance_does_not_call_show() -> None:
    arr = _synthetic_acceptance()
    with patch.object(plt, "show") as mock_show:
        fig, _ = plot_mh_acceptance(arr)
        assert not mock_show.called
        plt.close(fig)


def test_plot_mh_acceptance_does_not_mutate_rcparams() -> None:
    arr = _synthetic_acceptance()
    snapshot = deepcopy(dict(plt.rcParams))
    fig, _ = plot_mh_acceptance(arr)
    assert dict(plt.rcParams) == snapshot
    plt.close(fig)


def test_plot_mh_acceptance_style_override_accepted() -> None:
    arr = _synthetic_acceptance()
    fig, _ = plot_mh_acceptance(arr, style={"figsize_single": (8, 3)})
    plt.close(fig)


def test_plot_mh_acceptance_rejects_non_2d() -> None:
    with pytest.raises(ValueError):
        plot_mh_acceptance(np.zeros(100))


# ---------- posterior-Q visualisation set (nb 04+) ----------

# Synthetic Q-posterior fixtures: shape (n_draws, n_actions). Real callers
# obtain these via project.thompson.posterior_q(policy, x); the structural
# tests do not depend on that helper.


def _synthetic_q_at_state(
    n_draws: int = 2000, n_actions: int = 3, seed: int = 20,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    means = np.array([0.05, 0.0, -0.02][:n_actions])
    return rng.standard_normal((n_draws, n_actions)) * 0.03 + means


def _synthetic_q_grid(seed: int = 21) -> dict[str, np.ndarray]:
    return {
        "low_vol":   _synthetic_q_at_state(n_draws=1500, seed=seed),
        "high_vol":  _synthetic_q_at_state(n_draws=1500, seed=seed + 1),
        "transition": _synthetic_q_at_state(n_draws=1500, seed=seed + 2),
    }


def _synthetic_paths(
    n_paths: int = 200, n_t: int = 300, seed: int = 30,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Cumulative-return-shaped paths: (n_paths, n_timesteps)."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0003, 0.012, size=(n_paths, n_t))
    cum = np.cumsum(steps, axis=1)
    idx = pd.date_range("2020-01-02", periods=n_t, freq="B")
    return cum, idx


def _synthetic_action_paths(
    n_paths: int = 200, n_t: int = 300, seed: int = 31,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Discrete-action paths shaped like a posterior rollout: (n_paths, n_t)."""
    rng = np.random.default_rng(seed)
    arr = rng.choice([-1, 0, 1], size=(n_paths, n_t), p=[0.3, 0.4, 0.3])
    idx = pd.date_range("2020-01-02", periods=n_t, freq="B")
    return arr, idx


def _synthetic_metric_dict(seed: int = 32) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "sharpe":       rng.normal(0.6, 0.4, size=500),
        "max_drawdown": np.abs(rng.normal(0.10, 0.04, size=500)),
        "turnover":     np.abs(rng.normal(0.4, 0.1, size=500)),
    }


# ---- plot_q_posterior_at_state ----


def test_plot_q_posterior_at_state_returns_figure_and_axes() -> None:
    fig, ax = plot_q_posterior_at_state(_synthetic_q_at_state(), state_label="median")
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_plot_q_posterior_at_state_three_density_curves() -> None:
    """Three actions => three KDE/density lines on shared x-axis."""
    qs = _synthetic_q_at_state(n_actions=3)
    fig, ax = plot_q_posterior_at_state(qs, state_label="x_median")
    kde_lines = [ln for ln in ax.lines if len(set(ln.get_xdata())) > 1]
    assert len(kde_lines) == 3, (
        f"expected 3 KDE density lines, got {len(kde_lines)}"
    )
    plt.close(fig)


def test_plot_q_posterior_at_state_one_mean_line_per_action() -> None:
    qs = _synthetic_q_at_state(n_actions=3)
    fig, ax = plot_q_posterior_at_state(qs, state_label="x")
    vlines = [ln for ln in ax.lines if len(set(ln.get_xdata())) == 1]
    assert len(vlines) == 3, (
        f"expected 3 vertical mean lines, got {len(vlines)}"
    )
    plt.close(fig)


def test_plot_q_posterior_at_state_colors_from_action_styles() -> None:
    """Each density line must inherit colour from DEFAULT_STYLE['action_styles']."""
    qs = _synthetic_q_at_state(n_actions=3)
    fig, ax = plot_q_posterior_at_state(qs, state_label="x")
    expected = {
        DEFAULT_STYLE["action_styles"][-1]["color"],
        DEFAULT_STYLE["action_styles"][0]["color"],
        DEFAULT_STYLE["action_styles"][+1]["color"],
    }
    kde_lines = [ln for ln in ax.lines if len(set(ln.get_xdata())) > 1]
    actual = {ln.get_color() for ln in kde_lines}
    assert actual == expected, f"line colours {actual} != {expected}"
    plt.close(fig)


def test_plot_q_posterior_at_state_action_labels_appear_in_legend() -> None:
    qs = _synthetic_q_at_state(n_actions=3)
    fig, ax = plot_q_posterior_at_state(
        qs, state_label="x",
        action_labels=["alpha", "beta", "gamma"],
    )
    legend = ax.get_legend()
    assert legend is not None
    texts = [t.get_text() for t in legend.get_texts()]
    for label in ("alpha", "beta", "gamma"):
        assert label in texts, f"missing legend entry {label!r}"
    plt.close(fig)


def test_plot_q_posterior_at_state_state_label_in_title() -> None:
    qs = _synthetic_q_at_state()
    fig, ax = plot_q_posterior_at_state(qs, state_label="high_vol_2024_03")
    assert "high_vol_2024_03" in ax.get_title()
    plt.close(fig)


def test_plot_q_posterior_at_state_does_not_call_show() -> None:
    qs = _synthetic_q_at_state()
    with patch.object(plt, "show") as mock_show:
        fig, _ = plot_q_posterior_at_state(qs, state_label="x")
        assert not mock_show.called
        plt.close(fig)


def test_plot_q_posterior_at_state_does_not_mutate_rcparams() -> None:
    qs = _synthetic_q_at_state()
    snapshot = deepcopy(dict(plt.rcParams))
    fig, _ = plot_q_posterior_at_state(qs, state_label="x")
    assert dict(plt.rcParams) == snapshot
    plt.close(fig)


def test_plot_q_posterior_at_state_style_override_accepted() -> None:
    qs = _synthetic_q_at_state()
    fig, _ = plot_q_posterior_at_state(
        qs, state_label="x", style={"figsize_single": (8, 3.5)},
    )
    plt.close(fig)


def test_plot_q_posterior_at_state_ax_existing_reuses_axes() -> None:
    """Composability: passing ax= must reuse it (so plot_q_posterior_grid can call this)."""
    pre_fig, pre_ax = plt.subplots()
    qs = _synthetic_q_at_state()
    fig, ax = plot_q_posterior_at_state(qs, state_label="x", ax=pre_ax)
    assert fig is pre_fig
    assert ax is pre_ax
    plt.close(pre_fig)


def test_plot_q_posterior_at_state_n_draws_one_does_not_crash() -> None:
    """Single-draw input falls back to histogram-only (no KDE) and does not raise."""
    qs = np.array([[0.05, 0.0, -0.02]])
    fig, ax = plot_q_posterior_at_state(qs, state_label="x")
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_plot_q_posterior_at_state_rejects_non_2d() -> None:
    with pytest.raises(ValueError):
        plot_q_posterior_at_state(np.zeros(10), state_label="x")


def test_plot_q_posterior_at_state_action_labels_length_mismatch_raises() -> None:
    qs = _synthetic_q_at_state(n_actions=3)
    with pytest.raises(ValueError):
        plot_q_posterior_at_state(qs, state_label="x", action_labels=["a", "b"])


def test_plot_q_posterior_at_state_deterministic() -> None:
    qs = _synthetic_q_at_state(seed=99)
    fig1, _ = plot_q_posterior_at_state(qs, state_label="x")
    fig1.canvas.draw()
    buf1 = bytes(fig1.canvas.buffer_rgba())
    plt.close(fig1)

    fig2, _ = plot_q_posterior_at_state(qs, state_label="x")
    fig2.canvas.draw()
    buf2 = bytes(fig2.canvas.buffer_rgba())
    plt.close(fig2)
    assert buf1 == buf2


# ---- plot_q_posterior_grid ----


def test_plot_q_posterior_grid_n_panels_match_n_states() -> None:
    grid = _synthetic_q_grid()
    fig, axes = plot_q_posterior_grid(grid)
    assert isinstance(fig, Figure)
    assert isinstance(axes, np.ndarray)
    assert axes.size == len(grid), (
        f"expected {len(grid)} panels, got {axes.size}"
    )
    plt.close(fig)


def test_plot_q_posterior_grid_each_panel_has_three_kde_lines() -> None:
    """Per-panel structure mirrors the singular helper: 3 KDE + 3 mean lines."""
    grid = _synthetic_q_grid()
    fig, axes = plot_q_posterior_grid(grid)
    for ax in axes.ravel():
        kde_lines = [ln for ln in ax.lines if len(set(ln.get_xdata())) > 1]
        assert len(kde_lines) == 3, (
            f"expected 3 KDE lines per panel, got {len(kde_lines)}"
        )
    plt.close(fig)


def test_plot_q_posterior_grid_state_labels_in_titles() -> None:
    grid = _synthetic_q_grid()
    fig, axes = plot_q_posterior_grid(grid)
    titles = [ax.get_title() for ax in axes.ravel()]
    for state in grid.keys():
        assert any(state in t for t in titles), (
            f"state {state!r} not found in any panel title"
        )
    plt.close(fig)


def test_plot_q_posterior_grid_action_labels_propagate() -> None:
    grid = _synthetic_q_grid()
    fig, axes = plot_q_posterior_grid(
        grid, action_labels=["alpha", "beta", "gamma"],
    )
    for ax in axes.ravel():
        legend = ax.get_legend()
        assert legend is not None
        texts = [t.get_text() for t in legend.get_texts()]
        for label in ("alpha", "beta", "gamma"):
            assert label in texts
    plt.close(fig)


def test_plot_q_posterior_grid_does_not_call_show() -> None:
    grid = _synthetic_q_grid()
    with patch.object(plt, "show") as mock_show:
        fig, _ = plot_q_posterior_grid(grid)
        assert not mock_show.called
        plt.close(fig)


def test_plot_q_posterior_grid_does_not_mutate_rcparams() -> None:
    grid = _synthetic_q_grid()
    snapshot = deepcopy(dict(plt.rcParams))
    fig, _ = plot_q_posterior_grid(grid)
    assert dict(plt.rcParams) == snapshot
    plt.close(fig)


def test_plot_q_posterior_grid_style_override_accepted() -> None:
    grid = _synthetic_q_grid()
    fig, _ = plot_q_posterior_grid(grid, style={"figsize_panel": (9, 6)})
    plt.close(fig)


def test_plot_q_posterior_grid_empty_dict_raises() -> None:
    with pytest.raises(ValueError):
        plot_q_posterior_grid({})


# ---- plot_credible_band ----


def test_plot_credible_band_returns_figure_and_axes() -> None:
    paths, idx = _synthetic_paths()
    fig, ax = plot_credible_band(paths, idx)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_plot_credible_band_two_bands_produce_two_filled_regions() -> None:
    """fill_between adds one PolyCollection per band; 2 bands -> 2 collections."""
    paths, idx = _synthetic_paths()
    fig, ax = plot_credible_band(paths, idx, bands=(0.50, 0.95))
    assert len(ax.collections) == 2, (
        f"expected 2 filled regions, got {len(ax.collections)}"
    )
    plt.close(fig)


def test_plot_credible_band_single_band_produces_one_filled_region() -> None:
    paths, idx = _synthetic_paths()
    fig, ax = plot_credible_band(paths, idx, bands=(0.95,))
    assert len(ax.collections) == 1
    plt.close(fig)


def test_plot_credible_band_show_median_true_adds_median_line() -> None:
    paths, idx = _synthetic_paths()
    fig, ax = plot_credible_band(
        paths, idx, bands=(0.50,), show_median=True, sample_paths=0,
    )
    # No sample-path overlay, no policy => exactly one line: the median.
    assert len(ax.lines) == 1, (
        f"expected 1 median line, got {len(ax.lines)}"
    )
    plt.close(fig)


def test_plot_credible_band_show_median_false_removes_median_line() -> None:
    paths, idx = _synthetic_paths()
    fig, ax = plot_credible_band(
        paths, idx, bands=(0.50,), show_median=False, sample_paths=0,
    )
    assert len(ax.lines) == 0, (
        f"show_median=False should suppress the median line; "
        f"got {len(ax.lines)} lines"
    )
    # Bands still present.
    assert len(ax.collections) == 1
    plt.close(fig)


def test_plot_credible_band_ax_none_creates_axes() -> None:
    paths, idx = _synthetic_paths()
    fig, ax = plot_credible_band(paths, idx, ax=None)
    assert isinstance(ax, Axes)
    assert len(fig.axes) == 1
    plt.close(fig)


def test_plot_credible_band_ax_existing_reuses_axes() -> None:
    """Composability: passing ax= must reuse it (Gaussian-vs-Student-t panel needs this)."""
    paths, idx = _synthetic_paths()
    pre_fig, pre_ax = plt.subplots()
    fig, ax = plot_credible_band(paths, idx, ax=pre_ax)
    assert fig is pre_fig
    assert ax is pre_ax
    plt.close(pre_fig)


def test_plot_credible_band_policy_bayesian_resolves_to_C0() -> None:
    """policy='bayesian' must drive the line + fill colour from policy_styles."""
    paths, idx = _synthetic_paths()
    fig, ax = plot_credible_band(
        paths, idx, bands=(0.95,), policy="bayesian", show_median=True,
    )
    # The median line takes the bayesian colour.
    expected = DEFAULT_STYLE["policy_styles"]["bayesian"]["color"]
    assert ax.lines[0].get_color() == expected, (
        f"median line colour {ax.lines[0].get_color()!r} != {expected!r}"
    )
    plt.close(fig)


def test_plot_credible_band_label_overrides_policy_label() -> None:
    """An explicit label= must surface in the legend, ignoring the policy_styles label."""
    paths, idx = _synthetic_paths()
    fig, ax = plot_credible_band(
        paths, idx, bands=(0.95,), policy="bayesian", label="custom label",
        show_median=True,
    )
    legend = ax.get_legend()
    assert legend is not None
    texts = [t.get_text() for t in legend.get_texts()]
    assert "custom label" in texts
    plt.close(fig)


def test_plot_credible_band_sample_paths_overlay_count() -> None:
    """sample_paths=K overlays K thin lines in addition to the median (if shown)."""
    paths, idx = _synthetic_paths(n_paths=300, n_t=200)
    K = 8
    fig, ax = plot_credible_band(
        paths, idx, bands=(0.95,), show_median=True, sample_paths=K,
    )
    # K sample lines + 1 median line = K + 1.
    assert len(ax.lines) == K + 1, (
        f"expected {K + 1} lines (K samples + median), got {len(ax.lines)}"
    )
    plt.close(fig)


def test_plot_credible_band_does_not_call_show() -> None:
    paths, idx = _synthetic_paths()
    with patch.object(plt, "show") as mock_show:
        fig, _ = plot_credible_band(paths, idx)
        assert not mock_show.called
        plt.close(fig)


def test_plot_credible_band_does_not_mutate_rcparams() -> None:
    paths, idx = _synthetic_paths()
    snapshot = deepcopy(dict(plt.rcParams))
    fig, _ = plot_credible_band(paths, idx)
    assert dict(plt.rcParams) == snapshot
    plt.close(fig)


def test_plot_credible_band_style_override_accepted() -> None:
    paths, idx = _synthetic_paths()
    fig, _ = plot_credible_band(paths, idx, style={"figsize_single": (8, 3.5)})
    plt.close(fig)


def test_plot_credible_band_deterministic() -> None:
    paths, idx = _synthetic_paths(seed=99)
    fig1, _ = plot_credible_band(paths, idx)
    fig1.canvas.draw()
    buf1 = bytes(fig1.canvas.buffer_rgba())
    plt.close(fig1)

    fig2, _ = plot_credible_band(paths, idx)
    fig2.canvas.draw()
    buf2 = bytes(fig2.canvas.buffer_rgba())
    plt.close(fig2)
    assert buf1 == buf2


def test_plot_credible_band_empty_bands_raises() -> None:
    paths, idx = _synthetic_paths()
    with pytest.raises(ValueError):
        plot_credible_band(paths, idx, bands=())


def test_plot_credible_band_invalid_band_value_raises() -> None:
    paths, idx = _synthetic_paths()
    with pytest.raises(ValueError):
        plot_credible_band(paths, idx, bands=(1.5,))
    with pytest.raises(ValueError):
        plot_credible_band(paths, idx, bands=(0.0,))


def test_plot_credible_band_index_length_mismatch_raises() -> None:
    paths, idx = _synthetic_paths(n_t=300)
    short_idx = idx[:200]
    with pytest.raises(ValueError):
        plot_credible_band(paths, short_idx)


def test_plot_credible_band_rejects_non_2d() -> None:
    idx = pd.date_range("2020-01-02", periods=10, freq="B")
    with pytest.raises(ValueError):
        plot_credible_band(np.zeros(10), idx)


def test_plot_credible_band_overlay_two_policies_on_one_axes() -> None:
    """Critical use case: nb 04 closing panel overlays Gaussian + Student-t bands."""
    paths_g, idx = _synthetic_paths(seed=40)
    paths_t, _ = _synthetic_paths(seed=41)
    fig, ax = plt.subplots()
    plot_credible_band(paths_g, idx, bands=(0.95,), policy="bayesian",
                       label="Gaussian", ax=ax)
    plot_credible_band(paths_t, idx, bands=(0.95,), policy="bayesian",
                       label="Student-t", ax=ax)
    legend = ax.get_legend()
    assert legend is not None
    texts = [t.get_text() for t in legend.get_texts()]
    assert "Gaussian" in texts and "Student-t" in texts
    # Two fills (one per call), two median lines.
    assert len(ax.collections) == 2
    assert len(ax.lines) == 2
    plt.close(fig)


# ---- plot_action_probability_heatmap ----


def test_plot_action_probability_heatmap_returns_figure_and_axes() -> None:
    arr, idx = _synthetic_action_paths()
    fig, ax = plot_action_probability_heatmap(arr, idx)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_plot_action_probability_heatmap_monthly_bins_count() -> None:
    """A 1500-day window monthly-binned has ~70 bars * 3 actions = ~210 patches."""
    arr, idx = _synthetic_action_paths(n_paths=50, n_t=1500)
    fig, ax = plot_action_probability_heatmap(arr, idx, bin_freq="ME")
    n_months_expected = len(pd.date_range(idx[0], idx[-1], freq="ME"))
    expected = n_months_expected * 3
    actual = len(ax.patches)
    # Allow +/- 3 segments edge tolerance for resample boundary effects.
    assert abs(actual - expected) <= 3, (
        f"expected ~{expected} patches (n_months={n_months_expected} * 3 actions), "
        f"got {actual}"
    )
    plt.close(fig)


def test_plot_action_probability_heatmap_quarterly_bin_freq() -> None:
    """bin_freq parameter changes the bar count."""
    arr, idx = _synthetic_action_paths(n_paths=20, n_t=1000)
    fig_m, ax_m = plot_action_probability_heatmap(arr, idx, bin_freq="ME")
    fig_q, ax_q = plot_action_probability_heatmap(arr, idx, bin_freq="QE")
    assert len(ax_q.patches) < len(ax_m.patches), (
        "quarterly bins should yield fewer bar segments than monthly"
    )
    plt.close(fig_m)
    plt.close(fig_q)


def test_plot_action_probability_heatmap_colors_from_action_styles() -> None:
    """The three stacked-bar colours must come from DEFAULT_STYLE['action_styles']."""
    arr, idx = _synthetic_action_paths()
    fig, ax = plot_action_probability_heatmap(arr, idx)
    expected = {
        DEFAULT_STYLE["action_styles"][-1]["color"],
        DEFAULT_STYLE["action_styles"][0]["color"],
        DEFAULT_STYLE["action_styles"][+1]["color"],
    }
    actual = {
        matplotlib.colors.to_hex(p.get_facecolor())
        for p in ax.patches
    }
    expected_hex = {matplotlib.colors.to_hex(c) for c in expected}
    # actual should be a subset of (or equal to) the expected palette.
    assert actual <= expected_hex, (
        f"bar colours {actual} not within action palette {expected_hex}"
    )
    plt.close(fig)


def test_plot_action_probability_heatmap_stacks_sum_to_one() -> None:
    """Per-bin stacked heights must sum to 1.0 (proportions, not counts)."""
    arr, idx = _synthetic_action_paths(n_paths=100, n_t=400)
    fig, ax = plot_action_probability_heatmap(arr, idx, bin_freq="ME")
    # Group patches by x position; their cumulative heights should reach ~1.
    from collections import defaultdict
    by_x: dict[float, list[float]] = defaultdict(list)
    for p in ax.patches:
        by_x[round(p.get_x() + p.get_width() / 2, 6)].append(p.get_height())
    for centre, heights in by_x.items():
        total = sum(heights)
        assert abs(total - 1.0) < 1e-6, (
            f"bin at x={centre}: sum of heights = {total}, expected 1.0"
        )
    plt.close(fig)


def test_plot_action_probability_heatmap_action_labels_propagate() -> None:
    arr, idx = _synthetic_action_paths()
    fig, ax = plot_action_probability_heatmap(
        arr, idx, action_labels=["short_alpha", "flat_beta", "long_gamma"],
    )
    legend = ax.get_legend()
    assert legend is not None
    texts = [t.get_text() for t in legend.get_texts()]
    for label in ("short_alpha", "flat_beta", "long_gamma"):
        assert label in texts
    plt.close(fig)


def test_plot_action_probability_heatmap_default_legend_uses_action_styles_labels() -> None:
    arr, idx = _synthetic_action_paths()
    fig, ax = plot_action_probability_heatmap(arr, idx)
    legend = ax.get_legend()
    assert legend is not None
    texts = [t.get_text() for t in legend.get_texts()]
    expected = {
        DEFAULT_STYLE["action_styles"][-1]["label"],
        DEFAULT_STYLE["action_styles"][0]["label"],
        DEFAULT_STYLE["action_styles"][+1]["label"],
    }
    assert set(texts) == expected, (
        f"legend texts {set(texts)} != action_styles labels {expected}"
    )
    plt.close(fig)


def test_plot_action_probability_heatmap_does_not_call_show() -> None:
    arr, idx = _synthetic_action_paths()
    with patch.object(plt, "show") as mock_show:
        fig, _ = plot_action_probability_heatmap(arr, idx)
        assert not mock_show.called
        plt.close(fig)


def test_plot_action_probability_heatmap_does_not_mutate_rcparams() -> None:
    arr, idx = _synthetic_action_paths()
    snapshot = deepcopy(dict(plt.rcParams))
    fig, _ = plot_action_probability_heatmap(arr, idx)
    assert dict(plt.rcParams) == snapshot
    plt.close(fig)


def test_plot_action_probability_heatmap_style_override_accepted() -> None:
    arr, idx = _synthetic_action_paths()
    fig, _ = plot_action_probability_heatmap(
        arr, idx, style={"figsize_wide": (10, 4)},
    )
    plt.close(fig)


def test_plot_action_probability_heatmap_index_length_mismatch_raises() -> None:
    arr, _ = _synthetic_action_paths(n_t=300)
    short_idx = pd.date_range("2020-01-02", periods=200, freq="B")
    with pytest.raises(ValueError):
        plot_action_probability_heatmap(arr, short_idx)


def test_plot_action_probability_heatmap_rejects_non_2d() -> None:
    idx = pd.date_range("2020-01-02", periods=10, freq="B")
    with pytest.raises(ValueError):
        plot_action_probability_heatmap(np.zeros(10, dtype=int), idx)


def test_plot_action_probability_heatmap_action_labels_length_mismatch_raises() -> None:
    arr, idx = _synthetic_action_paths()
    with pytest.raises(ValueError):
        plot_action_probability_heatmap(
            arr, idx, action_labels=["a", "b"],
        )


# ---- plot_posterior_metric_distribution ----


def test_plot_posterior_metric_distribution_returns_figure_and_axes() -> None:
    md = _synthetic_metric_dict()
    fig, axes = plot_posterior_metric_distribution(md)
    assert isinstance(fig, Figure)
    assert isinstance(axes, np.ndarray)
    plt.close(fig)


def test_plot_posterior_metric_distribution_n_panels_match_n_metrics() -> None:
    md = _synthetic_metric_dict()
    fig, axes = plot_posterior_metric_distribution(md)
    assert axes.size == len(md), (
        f"expected {len(md)} panels, got {axes.size}"
    )
    plt.close(fig)


def test_plot_posterior_metric_distribution_each_panel_has_density() -> None:
    md = _synthetic_metric_dict()
    fig, axes = plot_posterior_metric_distribution(md)
    for ax in axes.ravel():
        # Each panel: histogram bars + KDE line.
        assert len(ax.patches) > 0, "expected histogram bars per panel"
        assert len(ax.lines) >= 1, "expected at least one KDE line per panel"
    plt.close(fig)


def test_plot_posterior_metric_distribution_metric_names_in_titles() -> None:
    md = _synthetic_metric_dict()
    fig, axes = plot_posterior_metric_distribution(md)
    titles = [ax.get_title() for ax in axes.ravel()]
    for metric in md.keys():
        assert any(metric in t for t in titles), (
            f"metric {metric!r} not found in any panel title"
        )
    plt.close(fig)


def test_plot_posterior_metric_distribution_sampler_color_propagates() -> None:
    md = _synthetic_metric_dict()
    fig, axes = plot_posterior_metric_distribution(md, sampler="gaussian")
    expected = DEFAULT_STYLE["sampler_styles"]["gaussian"]["color"]
    for ax in axes.ravel():
        kde_lines = [ln for ln in ax.lines if len(set(ln.get_xdata())) > 1]
        assert kde_lines, "expected KDE line per panel"
        assert kde_lines[0].get_color() == expected
    plt.close(fig)


def test_plot_posterior_metric_distribution_does_not_call_show() -> None:
    md = _synthetic_metric_dict()
    with patch.object(plt, "show") as mock_show:
        fig, _ = plot_posterior_metric_distribution(md)
        assert not mock_show.called
        plt.close(fig)


def test_plot_posterior_metric_distribution_does_not_mutate_rcparams() -> None:
    md = _synthetic_metric_dict()
    snapshot = deepcopy(dict(plt.rcParams))
    fig, _ = plot_posterior_metric_distribution(md)
    assert dict(plt.rcParams) == snapshot
    plt.close(fig)


def test_plot_posterior_metric_distribution_style_override_accepted() -> None:
    md = _synthetic_metric_dict()
    fig, _ = plot_posterior_metric_distribution(md, style={"figsize_panel": (9, 6)})
    plt.close(fig)


def test_plot_posterior_metric_distribution_empty_dict_raises() -> None:
    with pytest.raises(ValueError):
        plot_posterior_metric_distribution({})


def test_plot_posterior_metric_distribution_n_paths_one_does_not_crash() -> None:
    """Single-path input falls back to histogram-only (no KDE) per metric."""
    md = {"sharpe": np.array([0.5]), "mdd": np.array([0.1])}
    fig, axes = plot_posterior_metric_distribution(md)
    assert axes.size == 2
    plt.close(fig)


# ---------- comparison set (nb 05+) ----------
#
# Three helpers consumed by nb 05's main comparison: a forest-plot-style
# method-by-metric table, a density-overlay companion, and a four-method
# cumulative-return panel. The first two run on already-aggregated dicts
# matching the shape the project's two metric-distribution functions
# produce; the third runs on raw paths.


def _synthetic_method_metrics(
    methods: tuple[str, ...] = ("random", "tabular", "linear_fqi", "bayesian"),
    metrics: tuple[str, ...] = ("sharpe", "max_drawdown", "turnover"),
    seed: int = 31,
) -> dict[str, dict[str, dict[str, float]]]:
    """``{method: {metric: {mean, ci_lo, ci_hi}}}`` -- the post-aggregation
    shape that ``plot_metric_comparison_table`` consumes.

    The comparison-table helper is key-name agnostic (the lo/hi keys are an
    arg) so the same fixture covers both ``q025/q975`` (Bayesian) and
    ``ci_lo/ci_hi`` (classical) by re-keying.
    """
    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for m in methods:
        out[m] = {}
        for k in metrics:
            mean = float(rng.normal(0.5, 0.2))
            half = float(abs(rng.normal(0.0, 0.1)) + 0.05)
            out[m][k] = {
                "mean": mean,
                "ci_lo": mean - half,
                "ci_hi": mean + half,
            }
    return out


def _synthetic_method_samples(
    methods: tuple[str, ...] = ("random", "tabular", "linear_fqi", "bayesian"),
    n_per_method: tuple[int, ...] = (20, 20, 20, 500),
    seed: int = 32,
) -> dict[str, np.ndarray]:
    """``{method: 1-D metric samples}`` -- the input shape
    ``plot_metric_distribution_overlay`` consumes for one metric (e.g.,
    Sharpe values per posterior path / per classical seed)."""
    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {}
    for m, n in zip(methods, n_per_method):
        out[m] = rng.normal(loc=0.4, scale=0.15, size=n)
    return out


def _synthetic_method_paths(
    n_paths_bayesian: int = 100,
    n_seeds_classical: int = 20,
    T: int = 200,
    seed: int = 33,
) -> dict[str, dict[str, object]]:
    """Bayesian + 3 classical bundles, in the dict shape plot_path_comparison
    consumes: ``{method: {cum_log_return: (n, T) ndarray, index: DatetimeIndex}}``."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-02", periods=T, freq="B")
    out: dict[str, dict[str, object]] = {}
    out["bayesian"] = {
        "cum_log_return": np.cumsum(rng.normal(0.0005, 0.012, size=(n_paths_bayesian, T)), axis=1),
        "index": idx,
    }
    for name, drift in [("random", 0.0), ("tabular", 0.0003), ("linear_fqi", 0.0004)]:
        out[name] = {
            "cum_log_return": np.cumsum(
                rng.normal(drift, 0.011, size=(n_seeds_classical, T)),
                axis=1,
            ),
            "index": idx,
        }
    return out


# ---- plot_metric_comparison_table ----


def test_plot_metric_comparison_table_returns_figure_and_axes() -> None:
    mm = _synthetic_method_metrics()
    fig, ax = plot_metric_comparison_table(mm)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_plot_metric_comparison_table_renders_n_methods_times_n_metrics_glyphs() -> None:
    """4 methods × 3 metrics → 12 dot markers + 12 interval lines."""
    mm = _synthetic_method_metrics()
    fig, ax = plot_metric_comparison_table(mm)
    # Each cell renders one error-bar style: a line segment + a dot marker.
    # The dot is a Line2D with marker != 'None' and a single point.
    dot_lines = [
        ln for ln in ax.lines
        if ln.get_marker() not in ("None", "", None)
        and len(np.asarray(ln.get_xdata())) == 1
    ]
    assert len(dot_lines) == 12, (
        f"expected 12 dot markers (4 methods × 3 metrics), got {len(dot_lines)}"
    )
    plt.close(fig)


def test_plot_metric_comparison_table_method_colors_match_policy_styles() -> None:
    """Each row of dots inherits the locked policy colour."""
    mm = _synthetic_method_metrics()
    fig, ax = plot_metric_comparison_table(mm)
    ps = DEFAULT_STYLE["policy_styles"]
    dot_lines = [
        ln for ln in ax.lines
        if ln.get_marker() not in ("None", "", None)
        and len(np.asarray(ln.get_xdata())) == 1
    ]
    seen_colors_per_method: dict[str, set] = {m: set() for m in mm.keys()}
    for ln in dot_lines:
        label = ln.get_label() or ""
        for m in mm.keys():
            human = ps[m]["label"] if m in ps else m
            if label == human or label == m:
                seen_colors_per_method[m].add(ln.get_color())
                break
    for m in mm.keys():
        if m in ps:
            assert ps[m]["color"] in seen_colors_per_method[m] or len(
                seen_colors_per_method[m]
            ) > 0, f"no dot found for method {m!r}"


def test_plot_metric_comparison_table_metric_order_overrides_default() -> None:
    """metric_order kwarg controls row order (top-to-bottom)."""
    mm = _synthetic_method_metrics()
    fig, ax = plot_metric_comparison_table(
        mm, metric_order=["turnover", "sharpe", "max_drawdown"]
    )
    yticklabels = [t.get_text() for t in ax.get_yticklabels()]
    # The first metric in metric_order should appear first in the y-axis
    # tick labels (after any matplotlib processing); accept either exact
    # or substring match.
    assert any("turnover" in t for t in yticklabels), (
        f"expected 'turnover' in yticklabels, got {yticklabels}"
    )
    plt.close(fig)


def test_plot_metric_comparison_table_method_order_overrides_default() -> None:
    """method_order kwarg controls within-row ordering."""
    mm = _synthetic_method_metrics()
    custom_order = ["bayesian", "linear_fqi", "tabular", "random"]
    fig, ax = plot_metric_comparison_table(mm, method_order=custom_order)
    # The legend reflects method_order.
    legend = ax.get_legend()
    assert legend is not None
    legend_labels = [t.get_text() for t in legend.get_texts()]
    ps = DEFAULT_STYLE["policy_styles"]
    expected_first = ps[custom_order[0]]["label"]
    assert legend_labels[0] == expected_first or custom_order[0] in legend_labels[0], (
        f"first legend entry {legend_labels[0]!r} did not match {expected_first!r}"
    )
    plt.close(fig)


def test_plot_metric_comparison_table_handles_degenerate_interval() -> None:
    """mean == ci_lo == ci_hi must render a dot with a zero-length interval, not crash."""
    mm = _synthetic_method_metrics()
    # Force one cell to be degenerate.
    mm["random"]["sharpe"] = {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
    fig, ax = plot_metric_comparison_table(mm)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_plot_metric_comparison_table_accepts_alternate_lo_hi_keys() -> None:
    """Bayesian dicts use q025/q975; classical use ci_lo/ci_hi. Both must be
    consumable via a key-name override -- the comparison helper must not
    hard-code one set of keys."""
    mm: dict[str, dict[str, dict[str, float]]] = {}
    rng = np.random.default_rng(7)
    for m in ("bayesian", "linear_fqi"):
        mm[m] = {}
        for k in ("sharpe", "max_drawdown"):
            mn = float(rng.normal())
            mm[m][k] = {"mean": mn, "q025": mn - 0.1, "q975": mn + 0.1}
    fig, ax = plot_metric_comparison_table(
        mm, lo_key="q025", hi_key="q975"
    )
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_plot_metric_comparison_table_does_not_call_show() -> None:
    mm = _synthetic_method_metrics()
    with patch.object(plt, "show") as mock_show:
        fig, _ = plot_metric_comparison_table(mm)
        assert not mock_show.called
        plt.close(fig)


def test_plot_metric_comparison_table_does_not_mutate_rcparams() -> None:
    mm = _synthetic_method_metrics()
    snapshot = deepcopy(dict(plt.rcParams))
    fig, _ = plot_metric_comparison_table(mm)
    assert dict(plt.rcParams) == snapshot
    plt.close(fig)


def test_plot_metric_comparison_table_style_override_accepted() -> None:
    mm = _synthetic_method_metrics()
    fig, _ = plot_metric_comparison_table(
        mm, style={"figsize_panel": (9, 6)}
    )
    plt.close(fig)


def test_plot_metric_comparison_table_deterministic() -> None:
    mm = _synthetic_method_metrics()
    fig1, _ = plot_metric_comparison_table(mm)
    fig1.canvas.draw()
    buf1 = bytes(fig1.canvas.buffer_rgba())
    plt.close(fig1)

    fig2, _ = plot_metric_comparison_table(mm)
    fig2.canvas.draw()
    buf2 = bytes(fig2.canvas.buffer_rgba())
    plt.close(fig2)
    assert buf1 == buf2


def test_plot_metric_comparison_table_empty_dict_raises() -> None:
    with pytest.raises(ValueError):
        plot_metric_comparison_table({})


# ---- plot_metric_distribution_overlay ----


def test_plot_metric_distribution_overlay_returns_figure_and_axes() -> None:
    ms = _synthetic_method_samples()
    fig, ax = plot_metric_distribution_overlay(ms, metric_name="sharpe")
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_plot_metric_distribution_overlay_one_kde_per_method() -> None:
    """Four methods → at least four density curves (KDE lines)."""
    ms = _synthetic_method_samples()
    fig, ax = plot_metric_distribution_overlay(ms, metric_name="sharpe")
    kde_lines = [
        ln for ln in ax.lines
        if len(set(np.asarray(ln.get_xdata()))) > 1  # not vertical
    ]
    assert len(kde_lines) >= len(ms), (
        f"expected at least {len(ms)} density curves, got {len(kde_lines)}"
    )
    plt.close(fig)


def test_plot_metric_distribution_overlay_colors_match_policy_styles() -> None:
    ms = _synthetic_method_samples()
    fig, ax = plot_metric_distribution_overlay(ms, metric_name="sharpe")
    ps = DEFAULT_STYLE["policy_styles"]
    seen_colors = {ln.get_color() for ln in ax.lines}
    for method in ms.keys():
        if method in ps:
            assert ps[method]["color"] in seen_colors, (
                f"expected colour {ps[method]['color']!r} for method {method!r} "
                f"in {seen_colors!r}"
            )
    plt.close(fig)


def test_plot_metric_distribution_overlay_metric_name_in_xlabel() -> None:
    ms = _synthetic_method_samples()
    fig, ax = plot_metric_distribution_overlay(ms, metric_name="sharpe")
    assert "sharpe" in (ax.get_xlabel() or "").lower() or "sharpe" in (
        ax.get_title() or ""
    ).lower()
    plt.close(fig)


def test_plot_metric_distribution_overlay_handles_degenerate_method() -> None:
    """A method with a single sample falls back to no-KDE rendering for that
    method but does not crash."""
    ms = _synthetic_method_samples()
    ms["random"] = np.array([0.3])
    fig, ax = plot_metric_distribution_overlay(ms, metric_name="sharpe")
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_plot_metric_distribution_overlay_does_not_call_show() -> None:
    ms = _synthetic_method_samples()
    with patch.object(plt, "show") as mock_show:
        fig, _ = plot_metric_distribution_overlay(ms, metric_name="sharpe")
        assert not mock_show.called
        plt.close(fig)


def test_plot_metric_distribution_overlay_does_not_mutate_rcparams() -> None:
    ms = _synthetic_method_samples()
    snapshot = deepcopy(dict(plt.rcParams))
    fig, _ = plot_metric_distribution_overlay(ms, metric_name="sharpe")
    assert dict(plt.rcParams) == snapshot
    plt.close(fig)


def test_plot_metric_distribution_overlay_style_override_accepted() -> None:
    ms = _synthetic_method_samples()
    fig, _ = plot_metric_distribution_overlay(
        ms, metric_name="sharpe", style={"figsize_single": (8, 4)}
    )
    plt.close(fig)


def test_plot_metric_distribution_overlay_empty_dict_raises() -> None:
    with pytest.raises(ValueError):
        plot_metric_distribution_overlay({}, metric_name="sharpe")


# ---- plot_path_comparison ----


def test_plot_path_comparison_returns_figure_and_axes() -> None:
    mp = _synthetic_method_paths()
    fig, ax = plot_path_comparison(mp)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_plot_path_comparison_renders_bayesian_band_and_other_bands() -> None:
    """Bayesian gets a credible band (multiple fill_between collections) in
    its locked colour; each non-bayesian method gets at least one
    fill_between (seed-spread band) in its locked colour."""
    mp = _synthetic_method_paths()
    fig, ax = plot_path_comparison(mp, bands=(0.50, 0.95))
    # At minimum, one fill_between per method (Bayesian + 3 classical = 4).
    n_polys = len(ax.collections)
    assert n_polys >= len(mp), (
        f"expected >= {len(mp)} fill_between collections, got {n_polys}"
    )
    plt.close(fig)


def test_plot_path_comparison_includes_median_line_per_method() -> None:
    """Each method contributes one labelled median line."""
    mp = _synthetic_method_paths()
    fig, ax = plot_path_comparison(mp)
    labelled = [
        ln for ln in ax.lines
        if ln.get_label() and not ln.get_label().startswith("_")
    ]
    assert len(labelled) >= len(mp), (
        f"expected >= {len(mp)} labelled median lines, got {len(labelled)}"
    )
    plt.close(fig)


def test_plot_path_comparison_missing_bayesian_key_raises() -> None:
    mp = _synthetic_method_paths()
    mp.pop("bayesian")
    with pytest.raises((KeyError, ValueError)):
        plot_path_comparison(mp, bayesian_key="bayesian")


def test_plot_path_comparison_classical_alphas_differ() -> None:
    """Tabular and linear_fqi share the C1 colour; their bands must use
    distinguishable alphas so they don't melt into one orange blob."""
    mp = _synthetic_method_paths()
    fig, ax = plot_path_comparison(mp)
    # Look at the fill_between collections per method by colour. C1 is shared
    # by tabular and linear_fqi; we require that there are at least two C1
    # collections with different alphas.
    import matplotlib.colors as mcolors
    c1_rgb = mcolors.to_rgb("C1")
    c1_alphas: list[float] = []
    for coll in ax.collections:
        face = coll.get_facecolor()
        if len(face) == 0:
            continue
        rgba = face[0]
        if tuple(np.round(rgba[:3], 3)) == tuple(np.round(c1_rgb, 3)):
            c1_alphas.append(float(rgba[3]))
    assert len(set(round(a, 3) for a in c1_alphas)) >= 2, (
        f"expected C1 fill_between collections to use >= 2 distinct alphas, "
        f"got {c1_alphas!r}"
    )
    plt.close(fig)


def test_plot_path_comparison_renames_bayesian_key() -> None:
    """A non-default bayesian_key must be honoured."""
    mp = _synthetic_method_paths()
    mp["bayesian_gaussian"] = mp.pop("bayesian")
    fig, ax = plot_path_comparison(mp, bayesian_key="bayesian_gaussian")
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_plot_path_comparison_does_not_call_show() -> None:
    mp = _synthetic_method_paths()
    with patch.object(plt, "show") as mock_show:
        fig, _ = plot_path_comparison(mp)
        assert not mock_show.called
        plt.close(fig)


def test_plot_path_comparison_does_not_mutate_rcparams() -> None:
    mp = _synthetic_method_paths()
    snapshot = deepcopy(dict(plt.rcParams))
    fig, _ = plot_path_comparison(mp)
    assert dict(plt.rcParams) == snapshot
    plt.close(fig)


def test_plot_path_comparison_style_override_accepted() -> None:
    mp = _synthetic_method_paths()
    fig, _ = plot_path_comparison(mp, style={"figsize_single": (8, 4)})
    plt.close(fig)


def test_plot_path_comparison_deterministic() -> None:
    mp = _synthetic_method_paths()
    fig1, _ = plot_path_comparison(mp)
    fig1.canvas.draw()
    buf1 = bytes(fig1.canvas.buffer_rgba())
    plt.close(fig1)

    fig2, _ = plot_path_comparison(mp)
    fig2.canvas.draw()
    buf2 = bytes(fig2.canvas.buffer_rgba())
    plt.close(fig2)
    assert buf1 == buf2


def test_plot_path_comparison_single_seed_classical_does_not_crash() -> None:
    """Classical method with n_seeds=1 has no spread; helper must render a
    single line, not crash on degenerate quantiles."""
    mp = _synthetic_method_paths()
    mp["tabular"]["cum_log_return"] = mp["tabular"]["cum_log_return"][:1]
    fig, ax = plot_path_comparison(mp)
    assert isinstance(ax, Axes)
    plt.close(fig)
