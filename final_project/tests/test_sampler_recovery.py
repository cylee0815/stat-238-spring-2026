"""Synthetic-data parameter recovery for the Gaussian Gibbs sampler.

Non-negotiable: this sampler must recover the data-generating ``mu_beta``
and ``sigma^2`` within 95% credible intervals before it is run on real
prices. The 100-replication coverage test is the linchpin -- if it fails,
every downstream notebook is built on sand.
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pytest

from project.priors import make_priors
from project.sampler import gibbs_gaussian


def _draw_synthetic(
    rng: np.random.Generator,
    *,
    N: int,
    p: int,
    A: int,
    sigma_y_scale: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Sample ``(X, y, a)`` from the hierarchical generative model.

    ``mu_true`` has scale ``sigma_y_scale``; ``Sigma_true`` is half that on
    each axis; the residual std is half the mu scale. These choices keep
    all three sources of variability comparable.
    """
    X = rng.standard_normal((N, p))
    a = rng.integers(0, A, size=N)

    mu_true = sigma_y_scale * rng.standard_normal(p)
    Sigma_true = (0.5 * sigma_y_scale) ** 2 * np.eye(p)
    L = np.linalg.cholesky(Sigma_true)
    beta_true = mu_true[None, :] + (rng.standard_normal((A, p)) @ L.T)

    sigma_true = 0.5 * sigma_y_scale
    y = (X * beta_true[a]).sum(axis=1) + sigma_true * rng.standard_normal(N)

    return X, y, a, {
        "mu_true": mu_true,
        "Sigma_true": Sigma_true,
        "beta_true": beta_true,
        "sigma2_true": sigma_true ** 2,
    }


def _credible_interval(samples: np.ndarray, mass: float = 0.95) -> tuple[float, float]:
    lo = float(np.quantile(samples, (1 - mass) / 2))
    hi = float(np.quantile(samples, 1 - (1 - mass) / 2))
    return lo, hi


def test_gibbs_recovers_single_replication() -> None:
    """Smoke test: posterior means are within 4 posterior SDs of truth.

    A 95% CrI containment check on a single replication has ~5% per-param
    flake rate; the rigorous coverage gate is
    :func:`test_gibbs_coverage_100_replications`. Here we only assert the
    posterior is *centred* near truth, which catches catastrophic bias bugs
    with negligible flake rate.
    """
    rng = np.random.default_rng(238)
    N, p, A = 600, 4, 5
    X, y, a, theta = _draw_synthetic(rng, N=N, p=p, A=A)

    priors = make_priors(sigma_y_hat=float(np.std(y)), p=p, n_actions=A)
    trace = gibbs_gaussian(
        X, y, a, priors,
        n_draws=2000, n_burn=1000, n_chains=4, seed=12345,
    )

    def _z_score(samples: np.ndarray, truth: float) -> float:
        return abs(samples.mean() - truth) / samples.std(ddof=1)

    mu_flat = trace["mu_beta"].reshape(-1, p)
    for j in range(p):
        z = _z_score(mu_flat[:, j], theta["mu_true"][j])
        assert z < 4.0, f"mu_beta[{j}] posterior off by {z:.2f} SD"

    s2_flat = trace["sigma2"].reshape(-1)
    z = _z_score(s2_flat, theta["sigma2_true"])
    assert z < 4.0, f"sigma2 posterior off by {z:.2f} SD"

    beta_flat = trace["beta"].reshape(-1, A, p)
    max_z = 0.0
    for k in range(A):
        for j in range(p):
            z = _z_score(beta_flat[:, k, j], theta["beta_true"][k, j])
            max_z = max(max_z, z)
    # Bonferroni-ish: with A*p = 20 entries, allow up to 4.5 SD anywhere.
    assert max_z < 4.5, f"max beta posterior off by {max_z:.2f} SD"

    # R-hat gate: every scalar in (mu_beta, sigma2) must be well-mixed.
    idata = az.from_dict(posterior={
        "mu_beta": trace["mu_beta"],
        "sigma2": trace["sigma2"],
    })
    rhat = az.rhat(idata)
    rhat_mu = rhat["mu_beta"].to_numpy()
    rhat_s2 = float(rhat["sigma2"].item())
    assert rhat_mu.max() < 1.05, f"R-hat(mu_beta) max = {rhat_mu.max():.3f} >= 1.05"
    assert rhat_s2 < 1.05, f"R-hat(sigma2) = {rhat_s2:.3f} >= 1.05"


@pytest.mark.slow
def test_gibbs_coverage_100_replications() -> None:
    """Frequentist coverage check on the 95% CrIs over 100 simulations.

    Ground truth is redrawn each replication; the sampler's 95% CrI on
    each scalar parameter is required to cover truth in at least 90 / 100
    replications. Falls within the binomial test for nominal coverage 95%.
    """
    n_reps = 100
    N, p, A = 300, 3, 5

    base = np.random.SeedSequence(7777)
    data_root, fit_root = base.spawn(2)
    data_seeds = data_root.spawn(n_reps)
    fit_seeds = fit_root.spawn(n_reps)

    cov_mu = np.zeros(p, dtype=int)
    cov_s2 = 0
    cov_beta = np.zeros((A, p), dtype=int)

    for r in range(n_reps):
        rng = np.random.default_rng(data_seeds[r])
        X, y, a, theta = _draw_synthetic(rng, N=N, p=p, A=A)
        priors = make_priors(sigma_y_hat=float(np.std(y)), p=p, n_actions=A)

        fit_seed = int(fit_seeds[r].generate_state(1)[0])
        trace = gibbs_gaussian(
            X, y, a, priors,
            n_draws=1200, n_burn=800, n_chains=1, seed=fit_seed,
        )

        mu_flat = trace["mu_beta"].reshape(-1, p)
        for j in range(p):
            lo, hi = _credible_interval(mu_flat[:, j])
            cov_mu[j] += int(lo <= theta["mu_true"][j] <= hi)

        s2_flat = trace["sigma2"].reshape(-1)
        lo, hi = _credible_interval(s2_flat)
        cov_s2 += int(lo <= theta["sigma2_true"] <= hi)

        beta_flat = trace["beta"].reshape(-1, A, p)
        for k in range(A):
            for j in range(p):
                lo, hi = _credible_interval(beta_flat[:, k, j])
                cov_beta[k, j] += int(lo <= theta["beta_true"][k, j] <= hi)

    # Per-parameter binomial coverage gates.
    assert cov_mu.min() >= 90, (
        f"mu_beta per-component coverage out of 100: {cov_mu.tolist()}; "
        f"min {cov_mu.min()} < 90"
    )
    assert cov_s2 >= 90, f"sigma2 coverage out of 100: {cov_s2} < 90"
    assert cov_beta.min() >= 90, (
        f"beta_a coverage out of 100 (per (a, j)):\n{cov_beta}\n"
        f"min {cov_beta.min()} < 90"
    )


def test_meta_block_present_and_correct() -> None:
    """Every saved trace must carry sampler/seed/count provenance."""
    rng = np.random.default_rng(0)
    N, p, A = 100, 3, 3
    X, y, a, _ = _draw_synthetic(rng, N=N, p=p, A=A)
    priors = make_priors(sigma_y_hat=float(np.std(y)), p=p, n_actions=A)
    trace = gibbs_gaussian(
        X, y, a, priors, n_draws=50, n_burn=20, n_chains=2, seed=42,
    )
    meta = trace["meta"]
    assert meta["seed"] == 42
    assert meta["n_chains"] == 2
    assert meta["n_draws"] == 50
    assert meta["n_burn"] == 20
    assert meta["n_actions"] == A
    assert meta["p"] == p
    assert meta["N"] == N
    assert meta["sampler"] == "gaussian"


def test_independent_chain_seeding_changes_with_seed() -> None:
    """Two different root seeds give two different posteriors on the same data."""
    rng = np.random.default_rng(0)
    N, p, A = 200, 3, 3
    X, y, a, _ = _draw_synthetic(rng, N=N, p=p, A=A)
    priors = make_priors(sigma_y_hat=float(np.std(y)), p=p, n_actions=A)
    t1 = gibbs_gaussian(X, y, a, priors, n_draws=200, n_burn=100, n_chains=2, seed=1)
    t2 = gibbs_gaussian(X, y, a, priors, n_draws=200, n_burn=100, n_chains=2, seed=999)
    # Same seed -> same draws (sanity).
    t1b = gibbs_gaussian(X, y, a, priors, n_draws=200, n_burn=100, n_chains=2, seed=1)
    np.testing.assert_array_equal(t1["mu_beta"], t1b["mu_beta"])
    assert not np.array_equal(t1["mu_beta"], t2["mu_beta"])
