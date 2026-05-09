"""Synthetic-data parameter recovery for the Student-t Gibbs sampler.

Mirrors :mod:`tests.test_sampler_recovery` but extends coverage to ``nu``
and adds an ESS-on-``nu`` gate at the production chain budget (the
metric the user fixed at sign-off: ESS for ``nu`` >= 400 effective).
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pytest

from project.priors import make_priors_t
from project.sampler_t import gibbs_student_t


def _draw_synthetic_t(
    rng: np.random.Generator,
    *,
    N: int,
    p: int,
    A: int,
    nu_true: float = 5.0,
    sigma_y_scale: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Sample ``(X, y, a)`` from the t scale-mixture generative model."""
    X = rng.standard_normal((N, p))
    a = rng.integers(0, A, size=N)

    mu_true = sigma_y_scale * rng.standard_normal(p)
    Sigma_true = (0.5 * sigma_y_scale) ** 2 * np.eye(p)
    L = np.linalg.cholesky(Sigma_true)
    beta_true = mu_true[None, :] + (rng.standard_normal((A, p)) @ L.T)

    sigma_true = 0.5 * sigma_y_scale
    lam = rng.gamma(shape=0.5 * nu_true, scale=2.0 / nu_true, size=N)
    eps = rng.standard_normal(N) / np.sqrt(lam)
    y = (X * beta_true[a]).sum(axis=1) + sigma_true * eps

    return X, y, a, {
        "mu_true": mu_true,
        "Sigma_true": Sigma_true,
        "beta_true": beta_true,
        "sigma2_true": sigma_true ** 2,
        "nu_true": nu_true,
    }


def _credible_interval(samples: np.ndarray, mass: float = 0.95) -> tuple[float, float]:
    lo = float(np.quantile(samples, (1 - mass) / 2))
    hi = float(np.quantile(samples, 1 - (1 - mass) / 2))
    return lo, hi


def test_t_recovery_single_replication() -> None:
    """One replication, full production budget. Posterior centred near truth.

    Also gates on (i) ``nu`` ESS >= 400 (the user's locked threshold) and
    (ii) RW-MH acceptance falling inside [0.20, 0.45].
    """
    rng = np.random.default_rng(238)
    N, p, A = 600, 4, 5
    X, y, a, theta = _draw_synthetic_t(rng, N=N, p=p, A=A, nu_true=5.0)

    priors = make_priors_t(sigma_y_hat=float(np.std(y)), p=p, n_actions=A)
    trace = gibbs_student_t(
        X, y, a, priors,
        n_draws=6000, n_burn=2000, n_chains=4, seed=12345,
    )

    def _z(samples: np.ndarray, truth: float) -> float:
        return abs(samples.mean() - truth) / samples.std(ddof=1)

    mu_flat = trace["mu_beta"].reshape(-1, p)
    for j in range(p):
        z = _z(mu_flat[:, j], theta["mu_true"][j])
        assert z < 4.0, f"mu_beta[{j}] off by {z:.2f} SD"

    z = _z(trace["sigma2"].reshape(-1), theta["sigma2_true"])
    assert z < 4.0, f"sigma2 off by {z:.2f} SD"

    z = _z(trace["nu"].reshape(-1), theta["nu_true"])
    assert z < 4.0, f"nu off by {z:.2f} SD"

    # Acceptance within target band.
    acc = trace["nu_acceptance"]
    assert (0.20 <= acc).all() and (acc <= 0.45).all(), (
        f"nu MH acceptance per chain {acc} outside [0.20, 0.45]"
    )

    # ESS gate on nu (locked at 400 by sign-off).
    nu_id = az.from_dict(posterior={"nu": trace["nu"]})
    ess_nu = float(az.ess(nu_id)["nu"].item())
    assert ess_nu >= 400, f"ESS(nu) = {ess_nu:.0f} < 400"

    # R-hat gate: every scalar in (mu_beta, sigma2, nu) must be well-mixed.
    idata = az.from_dict(posterior={
        "mu_beta": trace["mu_beta"],
        "sigma2": trace["sigma2"],
        "nu": trace["nu"],
    })
    rhat = az.rhat(idata)
    rhat_mu = rhat["mu_beta"].to_numpy()
    rhat_s2 = float(rhat["sigma2"].item())
    rhat_nu = float(rhat["nu"].item())
    assert rhat_mu.max() < 1.05, f"R-hat(mu_beta) max = {rhat_mu.max():.3f} >= 1.05"
    assert rhat_s2 < 1.05, f"R-hat(sigma2) = {rhat_s2:.3f} >= 1.05"
    assert rhat_nu < 1.05, f"R-hat(nu) = {rhat_nu:.3f} >= 1.05"


def test_t_meta_block_present() -> None:
    rng = np.random.default_rng(0)
    N, p, A = 100, 3, 3
    X, y, a, _ = _draw_synthetic_t(rng, N=N, p=p, A=A, nu_true=10.0)
    priors = make_priors_t(sigma_y_hat=float(np.std(y)), p=p, n_actions=A)
    trace = gibbs_student_t(
        X, y, a, priors, n_draws=50, n_burn=20, n_chains=2, seed=42,
    )
    meta = trace["meta"]
    assert meta["sampler"] == "student_t"
    assert meta["seed"] == 42
    assert meta["n_chains"] == 2
    assert meta["n_draws"] == 50
    assert meta["n_burn"] == 20
    assert meta["nu_init"] == 10.0
    assert meta["keep_omega"] is False
    assert "omega" not in trace


def test_t_keep_omega_optional_save() -> None:
    """``keep_omega=True`` should add an (chain, draw, N) omega block."""
    rng = np.random.default_rng(0)
    N, p, A = 80, 3, 3
    X, y, a, _ = _draw_synthetic_t(rng, N=N, p=p, A=A, nu_true=10.0)
    priors = make_priors_t(sigma_y_hat=float(np.std(y)), p=p, n_actions=A)
    trace = gibbs_student_t(
        X, y, a, priors, n_draws=30, n_burn=20, n_chains=2, seed=42,
        keep_omega=True,
    )
    assert trace["meta"]["keep_omega"] is True
    assert trace["omega"].shape == (2, 30, N)
    assert np.all(trace["omega"] > 0)


def test_t_robbins_monro_freezes_after_half_burnin() -> None:
    """The log_step trace should be of length ``n_burn // 2`` and finite."""
    rng = np.random.default_rng(0)
    N, p, A = 100, 3, 3
    X, y, a, _ = _draw_synthetic_t(rng, N=N, p=p, A=A, nu_true=10.0)
    priors = make_priors_t(sigma_y_hat=float(np.std(y)), p=p, n_actions=A)
    n_burn = 200
    trace = gibbs_student_t(
        X, y, a, priors,
        n_draws=50, n_burn=n_burn, n_chains=2, seed=1,
    )
    assert trace["log_step_trace"].shape == (2, n_burn // 2)
    assert np.all(np.isfinite(trace["log_step_trace"]))


@pytest.mark.slow
def test_t_coverage_100_replications() -> None:
    """Frequentist coverage of 95% CrIs on (mu_beta, sigma2, nu) over 100 reps.

    Each parameter must achieve >= 90 / 100 coverage. ``beta_a`` is omitted
    from the strict gate to keep the parameter-space joint failure rate
    low; ``mu_beta`` and ``nu`` are the headline targets.
    """
    n_reps = 100
    N, p, A = 300, 3, 5
    nu_true = 5.0

    base = np.random.SeedSequence(7777)
    data_root, fit_root = base.spawn(2)
    data_seeds = data_root.spawn(n_reps)
    fit_seeds = fit_root.spawn(n_reps)

    cov_mu = np.zeros(p, dtype=int)
    cov_s2 = 0
    cov_nu = 0

    for r in range(n_reps):
        rng = np.random.default_rng(data_seeds[r])
        X, y, a, theta = _draw_synthetic_t(rng, N=N, p=p, A=A, nu_true=nu_true)
        priors = make_priors_t(sigma_y_hat=float(np.std(y)), p=p, n_actions=A)
        seed = int(fit_seeds[r].generate_state(1)[0])
        trace = gibbs_student_t(
            X, y, a, priors,
            n_draws=1500, n_burn=800, n_chains=1, seed=seed,
        )
        mu_flat = trace["mu_beta"].reshape(-1, p)
        for j in range(p):
            lo, hi = _credible_interval(mu_flat[:, j])
            cov_mu[j] += int(lo <= theta["mu_true"][j] <= hi)
        lo, hi = _credible_interval(trace["sigma2"].reshape(-1))
        cov_s2 += int(lo <= theta["sigma2_true"] <= hi)
        lo, hi = _credible_interval(trace["nu"].reshape(-1))
        cov_nu += int(lo <= nu_true <= hi)

    assert cov_mu.min() >= 90, (
        f"mu_beta per-component coverage out of 100: {cov_mu.tolist()}"
    )
    assert cov_s2 >= 90, f"sigma2 coverage out of 100: {cov_s2}"
    assert cov_nu >= 90, f"nu coverage out of 100: {cov_nu}"
