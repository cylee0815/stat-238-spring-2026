"""Generate every figure used by ``slides.tex`` and save them to ``figures/``.

Run from the project root with the project's conda env::

    /Users/chihyulee815/miniconda3/envs/stat238-bayes-rl/bin/python \\
        presentation/generate_figures.py

The script reloads the cached Gibbs traces and rollout bundles produced by
notebooks 03-09 and calls the locked ``project.plots.plot_*`` helpers. It is
the *only* place in the project where ``fig.savefig`` is invoked; notebooks
keep their inline-display convention untouched. Each figure is saved as both
``.pdf`` (slides) and ``.png`` (preview).
"""
from __future__ import annotations

import hashlib
import pickle
import sys
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from project import data, env, features, plots, thompson  # noqa: E402
from project import eval as ev  # noqa: E402
from project.targets import mc_returns  # noqa: E402
from project.utils import set_seed  # noqa: E402

FIG_DIR = REPO / "figures"
TRACE_DIR = REPO / "data" / "processed" / "traces"
PATHS_DIR = REPO / "data" / "processed" / "paths"
CLASSICAL_DIR = REPO / "data" / "processed" / "classical"

SEED = 12345
N_PATHS = 500


def save(fig, name: str) -> None:
    """Write fig to figures/<name>.pdf and figures/<name>.png."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf + .png")


def _data_fingerprint(prices_train: pd.DataFrame, prices_test: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(f"train={prices_train.index[0]}|{prices_train.index[-1]}|n={len(prices_train)}".encode())
    h.update(f"test={prices_test.index[0]}|{prices_test.index[-1]}|n={len(prices_test)}".encode())
    return h.hexdigest()[:12]


def _classical_key(name: str, fp: str, n_seeds: int = 20, base_seed: int = 0) -> Path:
    h = hashlib.sha256()
    h.update(fp.encode())
    h.update(f"n_seeds={n_seeds}|base_seed={base_seed}".encode())
    return CLASSICAL_DIR / f"classical_{name}_{h.hexdigest()[:12]}.pkl"


def _load_pickle(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _paths_key(sampler_name: str, trace_hash: str, n_paths: int = N_PATHS,
               resample_every: object = None, seed: int = SEED) -> Path:
    """Mirror nb 04's path cache key so we pick the canonical bundle."""
    h = hashlib.sha256()
    h.update(trace_hash.encode())
    h.update(f"n_paths={n_paths}|resample_every={resample_every}|seed={seed}".encode())
    return PATHS_DIR / f"paths_{sampler_name}_{h.hexdigest()[:12]}.pkl"


def _bayes_metric_summary(paths_dict: dict, n_paths: int) -> dict:
    rewards = paths_dict["rewards"]
    cum = paths_dict["cum_log_return"]
    acts = paths_dict["actions"]
    idx = paths_dict["index"]
    s = np.array([ev.sharpe(pd.Series(rewards[i], index=idx)) for i in range(n_paths)])
    m = np.array([ev.max_drawdown(pd.Series(cum[i], index=idx)) for i in range(n_paths)])
    t = np.array([ev.turnover(pd.Series(acts[i], index=idx)) for i in range(n_paths)])
    return {
        "sharpe": {"mean": float(s.mean()), "q025": float(np.quantile(s, 0.025)), "q975": float(np.quantile(s, 0.975))},
        "max_drawdown": {"mean": float(m.mean()), "q025": float(np.quantile(m, 0.025)), "q975": float(np.quantile(m, 0.975))},
        "turnover": {"mean": float(t.mean()), "q025": float(np.quantile(t, 0.025)), "q975": float(np.quantile(t, 0.975))},
    }, s, m, t


def main() -> None:
    set_seed()
    print("Loading cached SPY data + splits ...")
    spy = data.load_spy()
    splits = data.split_windows(spy)
    prices_train = splits["train"]
    prices_test = splits["test"]
    feat = features.build_features(spy)
    feat_train = feat.reindex(prices_train.index)
    fp = _data_fingerprint(prices_train, prices_test)

    print("Loading cached Gaussian Gibbs trace + paths ...")
    trace_g_path = sorted(TRACE_DIR.glob("gaussian_[0-9a-f]*.pkl"))[0]
    trace_g = _load_pickle(trace_g_path)
    hash_g = trace_g_path.stem.rsplit("_", 1)[-1]
    paths_g_path = _paths_key("gaussian", hash_g)
    if not paths_g_path.exists():
        raise FileNotFoundError(f"canonical paths bundle missing: {paths_g_path.name}")
    paths_g = _load_pickle(paths_g_path)
    print(f"  trace={trace_g_path.name}  paths={paths_g_path.name}")

    print("Loading cached classical bundles ...")
    cls_tab = _load_pickle(_classical_key("tabular", fp))
    cls_fqi = _load_pickle(_classical_key("linear_fqi", fp))
    cls_rnd = _load_pickle(_classical_key("random", fp))

    # ---- Figure 1: SPY price + train/test split ----------------------------
    print("\n[1/10] SPY price series + splits")
    fig, _ = plots.plot_price_series(spy)
    fig.suptitle("SPY adjusted close, 1993-2026 (train 1994-2019, test 2020-04 -> 2026-04)",
                 fontsize=10, y=1.01)
    save(fig, "fig_data_overview")

    # ---- Figure 2: MC return distribution per action ----------------------
    print("[2/10] MC-return target distribution by action")
    rng = np.random.default_rng(SEED)
    behavior_idx = rng.integers(0, 3, size=len(prices_train))
    labels = np.asarray(env.ACTIONS, dtype=int)
    behavior_labels = labels[behavior_idx]
    train_rewards = env.step_reward(prices_train, behavior_labels)
    y_series = mc_returns(train_rewards)
    shared = feat_train.dropna().index.intersection(y_series.index)
    behavior_labels_series = pd.Series(
        labels[behavior_idx], index=prices_train.index, name="action"
    ).loc[shared]
    fig, _ = plots.plot_mc_target_distribution(
        y_series.loc[shared], by_action=behavior_labels_series,
    )
    fig.suptitle("MC return $y_t = \\sum_{k=0}^{H} \\gamma^k R_{t+k}$ by behavior action (training data)",
                 fontsize=10, y=1.01)
    save(fig, "fig_mc_targets")

    # ---- Figure 3: Gibbs R-hat summary ------------------------------------
    print("[3/10] Gaussian Gibbs sampler R-hat summary")
    posterior = {"mu_beta": trace_g["mu_beta"], "sigma2": trace_g["sigma2"], "beta": trace_g["beta"]}
    rhat = az.rhat(az.from_dict(posterior=posterior))
    p = trace_g["beta"].shape[-1]
    n_actions = trace_g["beta"].shape[-2]
    rhat_dict = {f"mu_beta[{j}]": float(rhat["mu_beta"].values[j]) for j in range(p)}
    rhat_dict["sigma2"] = float(rhat["sigma2"].item())
    for k in range(n_actions):
        for j in range(p):
            rhat_dict[f"beta[a={k}][{j}]"] = float(rhat["beta"].values[k, j])
    fig, _ = plots.plot_rhat_summary(rhat_dict)
    fig.suptitle("Gaussian Gibbs sampler -- $\\hat R$ across all scalars",
                 fontsize=10, y=1.01)
    save(fig, "fig_rhat")

    # ---- Figure 4: Q(s,a) posterior at a representative state -------------
    print("[4/10] Q(s,a) posterior at a representative test state")
    policy_g = thompson.policy_from_trace(trace_g)
    feat_test = feat.reindex(prices_test.index).dropna()
    x_rep = feat_test.iloc[len(feat_test) // 2].to_numpy()
    q_samp = thompson.posterior_q(policy_g, x_rep)
    fig, _ = plots.plot_q_posterior_at_state(q_samp,
        state_label=f"mid-test ({feat_test.index[len(feat_test)//2].date()})")
    save(fig, "fig_q_posterior")

    # ---- Figure 5: Test-window cumulative return -- 4-method credible band -
    print("[5/10] Test-window cumulative return -- 4-method comparison")
    method_paths = {
        "bayesian":   {"cum_log_return": paths_g["cum_log_return"], "index": paths_g["index"]},
        "linear_fqi": {"cum_log_return": cls_fqi["paths"]["cum_log_return"], "index": cls_fqi["index"]},
        "tabular":    {"cum_log_return": cls_tab["paths"]["cum_log_return"], "index": cls_tab["index"]},
        "random":     {"cum_log_return": cls_rnd["paths"]["cum_log_return"], "index": cls_rnd["index"]},
    }
    fig, ax = plots.plot_path_comparison(method_paths, bayesian_key="bayesian", bands=(0.50, 0.95))
    ax.set_ylabel("cumulative log return", fontsize=9)
    fig.suptitle("Test-window cumulative return -- Bayesian (Gaussian) vs classical",
                 fontsize=10, y=1.005)
    save(fig, "fig_cum_return")

    # ---- Figure 6: Forest plot of headline metrics ------------------------
    print("[6/10] Forest plot -- Sharpe / MDD / Turnover (4 methods)")
    bayes_metrics, bayes_s, bayes_m, bayes_t = _bayes_metric_summary(paths_g, N_PATHS)
    method_metrics = {
        "bayesian":   bayes_metrics,
        "linear_fqi": cls_fqi["metrics_seed_distribution"],
        "tabular":    cls_tab["metrics_seed_distribution"],
        "random":     cls_rnd["metrics_seed_distribution"],
    }
    fig, _ = plots.plot_metric_comparison_table(
        method_metrics,
        metric_order=["sharpe", "max_drawdown", "turnover"],
        method_order=["bayesian", "linear_fqi", "tabular", "random"],
        lo_key="q025", hi_key="q975",
    )
    fig.suptitle("Headline metrics -- mean and 2.5-97.5 percentile of realized outcomes",
                 fontsize=10, y=1.01)
    save(fig, "fig_metric_table")

    # ---- Figure 7: Per-metric distribution overlay (Sharpe only) ----------
    print("[7/10] Sharpe distribution overlay (KDE)")
    raw_sharpe = {
        "bayesian":   bayes_s,
        "linear_fqi": cls_fqi["raw_metric_samples"]["sharpe"],
        "tabular":    cls_tab["raw_metric_samples"]["sharpe"],
        "random":     cls_rnd["raw_metric_samples"]["sharpe"],
    }
    fig, _ = plots.plot_metric_distribution_overlay(raw_sharpe, metric_name="sharpe")
    save(fig, "fig_sharpe_overlay")

    # ---- Figure 8: Action frequency comparison ----------------------------
    print("[8/10] Action occupancy comparison")
    all_actions = {
        "bayesian":   pd.Series(paths_g["actions"].ravel(), name="actions"),
        "linear_fqi": pd.Series(cls_fqi["paths"]["actions"].ravel(), name="actions"),
        "tabular":    pd.Series(cls_tab["paths"]["actions"].ravel(), name="actions"),
        "random":     pd.Series(cls_rnd["paths"]["actions"].ravel(), name="actions"),
    }
    fig, _ = plots.plot_action_frequency(all_actions)
    fig.suptitle("Action occupancy -- fraction of (run, day) cells per action",
                 fontsize=10, y=1.005)
    save(fig, "fig_action_freq")

    # ---- Figure 9: Prior-strength invariance (nb 09 Panel A) --------------
    print("[9/10] Prior-strength invariance (sigma^2 prior sweep)")
    sweep_paths = sorted(PATHS_DIR.glob("paths_gaussian_prior_sweep_[0-9a-f]*.pkl"))
    sweep_traces = sorted(TRACE_DIR.glob("gaussian_prior_sweep_[0-9a-f]*.pkl"))
    if len(sweep_paths) >= 3 and len(sweep_traces) >= 3:
        # We can't directly match nu_sigma without recomputing keys; label
        # in load order, which matches the alphabetic sort of the 3 caches.
        prior_methods = {}
        for i, p_path in enumerate(sweep_paths[:3]):
            ns_label = ["nu_sigma=0.1", "nu_sigma=1.0", "nu_sigma=4.0"][i]
            paths_ns = _load_pickle(p_path)
            ms, _, _, _ = _bayes_metric_summary(paths_ns, N_PATHS)
            prior_methods[ns_label] = ms
        custom_style = {
            "policy_styles": {
                **plots.DEFAULT_STYLE["policy_styles"],
                "nu_sigma=0.1": {"color": "C0", "linestyle": "-", "label": "nu_sigma=0.1"},
                "nu_sigma=1.0": {"color": "C1", "linestyle": "-", "label": "nu_sigma=1.0"},
                "nu_sigma=4.0": {"color": "C2", "linestyle": "-", "label": "nu_sigma=4.0"},
            }
        }
        fig, _ = plots.plot_metric_comparison_table(
            prior_methods,
            metric_order=["sharpe", "max_drawdown", "turnover"],
            method_order=list(prior_methods.keys()),
            lo_key="q025", hi_key="q975",
            style=custom_style,
        )
        fig.suptitle("Prior-strength invariance -- $\\sigma^2$-prior sweep (nb 09 Panel A)",
                     fontsize=10, y=1.01)
        save(fig, "fig_prior_invariance")
    else:
        print("  skipped: prior-sweep caches not all present")

    # ---- Figure 10: Gaussian vs Student-t robustness ----------------------
    print("[10/10] Gaussian vs Student-t robustness")
    trace_t_path_list = sorted(TRACE_DIR.glob("student_t_[0-9a-f]*.pkl"))
    if trace_t_path_list:
        hash_t = trace_t_path_list[0].stem.rsplit("_", 1)[-1]
        paths_t_path = _paths_key("student_t", hash_t)
        if not paths_t_path.exists():
            print(f"  skipped: canonical student-t paths bundle missing ({paths_t_path.name})")
            paths_t_path_list = []
    if trace_t_path_list and paths_t_path.exists():
        paths_t = _load_pickle(paths_t_path)
        bayes_t_metrics, *_ = _bayes_metric_summary(paths_t, N_PATHS)
        method_metrics_robust = {
            "bayesian":   bayes_metrics,
            "bayesian_t": bayes_t_metrics,
            "linear_fqi": cls_fqi["metrics_seed_distribution"],
        }
        custom_style = {
            "policy_styles": {
                **plots.DEFAULT_STYLE["policy_styles"],
                "bayesian_t": {"color": "C2", "linestyle": "-", "label": "Bayesian Q (Student-t)"},
            }
        }
        fig, _ = plots.plot_metric_comparison_table(
            method_metrics_robust,
            metric_order=["sharpe", "max_drawdown", "turnover"],
            method_order=["bayesian", "bayesian_t", "linear_fqi"],
            lo_key="q025", hi_key="q975",
            style=custom_style,
        )
        fig.suptitle("Robustness -- Gaussian vs Student-t Bayesian, vs linear FQI",
                     fontsize=10, y=1.01)
        save(fig, "fig_student_t")
    else:
        print("  skipped: student-t paths cache not present")

    print("\nDone. figures/ now contains the slide deck's plots.")


if __name__ == "__main__":
    main()
