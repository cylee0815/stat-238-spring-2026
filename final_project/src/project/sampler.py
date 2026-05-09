"""Hierarchical Gibbs sampler -- Gaussian likelihood (proposal section 2.5).

Implements the exact Gibbs sampler with the conjugate full conditionals.
Each block is a separate function that takes an explicit
``rng: np.random.Generator`` so chains seed independently and parallel
execution is a one-line change later.

Numerical-stability conventions (locked at sign-off):
- Normal full conditionals are sampled via Cholesky of the precision:
  given V^{-1} = L L^T, draw x = m + L^{-T} z with z ~ N(0, I) using two
  triangular solves. No inversion of V^{-1} is ever formed for the
  ``beta_a`` and ``mu_beta`` blocks.
- Sigma_beta is drawn via the Bartlett decomposition of the corresponding
  Wishart on Sigma_beta^{-1}.

Hierarchy:
    sigma^2     ~ IG(a0, b0)
    Sigma_beta  ~ IW(nu0, Psi0)
    mu_beta     ~ N(mu0, Lambda0)               # Lambda0 is COVARIANCE
    beta_a      ~ N(mu_beta, Sigma_beta)        # a = 0, ..., A - 1
    y_t         ~ N(x_t^T beta_{a_t}, sigma^2)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from project.priors import GaussianPriors
from project.utils import SEED


# ---------------------------------------------------------------------------
# Cholesky-based primitives
# ---------------------------------------------------------------------------

def _draw_normal_precision(
    precision: np.ndarray,
    nat_param: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``x ~ N(precision^{-1} @ nat_param, precision^{-1})`` via Cholesky.

    ``precision`` must be SPD. The mean is recovered by
    ``cho_solve((L, True), nat_param)``; the noise term is
    ``solve_triangular(L, z, lower=True, trans='T')``, equivalent to
    ``L^{-T} z``. No explicit inverse is formed.
    """
    L, low = cho_factor(precision, lower=True, check_finite=False)
    mean = cho_solve((L, low), nat_param, check_finite=False)
    z = rng.standard_normal(precision.shape[0])
    noise = solve_triangular(L, z, lower=True, trans="T", check_finite=False)
    return mean + noise


def _sample_invwishart(
    nu: float,
    Psi: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``Sigma ~ IW(nu, Psi)`` via Bartlett.

    Uses ``Sigma^{-1} = W ~ Wishart(nu, Psi^{-1})``. For the small ``p`` of
    this project (4) the explicit ``p x p`` inverses are negligible and
    keep the routine readable.

    Convention (matches scipy.stats.invwishart):
        E[Sigma] = Psi / (nu - p - 1) for nu > p + 1.
    """
    p = Psi.shape[0]
    if nu <= p - 1:
        raise ValueError(f"IW requires nu > p - 1, got nu={nu}, p={p}")
    Psi_inv = np.linalg.inv(Psi)
    M = np.linalg.cholesky(Psi_inv)            # Psi_inv = M @ M.T
    A = np.zeros((p, p))
    diag_df = nu - np.arange(p)                # nu, nu-1, ..., nu-p+1
    A[np.diag_indices(p)] = np.sqrt(rng.chisquare(df=diag_df))
    if p > 1:
        tril = np.tril_indices(p, k=-1)
        A[tril] = rng.standard_normal(size=tril[0].size)
    LA = M @ A
    W = LA @ LA.T
    Sigma = np.linalg.inv(W)
    return 0.5 * (Sigma + Sigma.T)             # suppress fp-asymmetry


# ---------------------------------------------------------------------------
# Per-block full conditionals
# ---------------------------------------------------------------------------

def update_beta(
    beta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    sigma2: float,
    mu_beta: np.ndarray,
    Sigma_beta_inv: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """In-place update of ``beta`` (shape ``(A, p)``).

    Per action k:
        V_k^{-1} = sigma^{-2} X_k^T X_k + Sigma_beta^{-1}
        m_k      = V_k (sigma^{-2} X_k^T y_k + Sigma_beta^{-1} mu_beta)
        beta_k   ~ N(m_k, V_k)

    If no observations have ``a == k`` the posterior collapses to the
    hierarchical prior ``N(mu_beta, Sigma_beta)``.
    """
    A, _ = beta.shape
    Sb_inv_mu = Sigma_beta_inv @ mu_beta
    inv_sigma2 = 1.0 / sigma2
    for k in range(A):
        mask = (a == k)
        if mask.any():
            X_k = X[mask]
            y_k = y[mask]
            precision = inv_sigma2 * (X_k.T @ X_k) + Sigma_beta_inv
            nat = inv_sigma2 * (X_k.T @ y_k) + Sb_inv_mu
        else:
            precision = Sigma_beta_inv
            nat = Sb_inv_mu
        beta[k] = _draw_normal_precision(precision, nat, rng)


def update_mu_beta(
    beta: np.ndarray,
    Sigma_beta_inv: np.ndarray,
    Lambda0_inv: np.ndarray,
    Lambda0_inv_mu0: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample ``mu_beta | beta_{0:A}, Sigma_beta``.

        precision = Lambda0^{-1} + A * Sigma_beta^{-1}
        nat       = Lambda0^{-1} mu0 + Sigma_beta^{-1} sum_a beta_a
    """
    A = beta.shape[0]
    precision = Lambda0_inv + A * Sigma_beta_inv
    nat = Lambda0_inv_mu0 + Sigma_beta_inv @ beta.sum(axis=0)
    return _draw_normal_precision(precision, nat, rng)


def update_Sigma_beta(
    beta: np.ndarray,
    mu_beta: np.ndarray,
    nu0: float,
    Psi0: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample ``Sigma_beta | beta_{0:A}, mu_beta`` from IW(nu_post, Psi_post)."""
    A = beta.shape[0]
    centered = beta - mu_beta[None, :]
    S = centered.T @ centered
    return _sample_invwishart(nu0 + A, Psi0 + S, rng)


def update_sigma2(
    beta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    a0: float,
    b0: float,
    rng: np.random.Generator,
) -> float:
    """Sample ``sigma^2 | rest`` from IG(a0 + N/2, b0 + 0.5 RSS)."""
    N = y.shape[0]
    pred = (X * beta[a]).sum(axis=1)
    resid = y - pred
    a_post = a0 + 0.5 * N
    b_post = b0 + 0.5 * float(resid @ resid)
    return float(b_post / rng.gamma(shape=a_post, scale=1.0))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def gibbs_gaussian(
    X: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    priors: GaussianPriors,
    *,
    n_draws: int = 2000,
    n_burn: int = 1000,
    n_chains: int = 4,
    seed: int = SEED,
    n_actions: int | None = None,
) -> dict[str, Any]:
    """Run the conjugate Gibbs sampler.

    Parameters
    ----------
    X, y, a
        Stacked design matrix ``(N, p)``, MC return targets ``(N,)``, and
        action ids in ``{0, ..., n_actions - 1}``. Time order is irrelevant
        because the model is iid given the parameters.
    priors
        Hyperparameters from :func:`project.priors.make_priors`.
    n_draws, n_burn, n_chains, seed
        Sweep counts and root seed. Per-chain ``Generator`` instances are
        spawned from ``np.random.SeedSequence(seed)`` so chains never share
        random state.
    n_actions
        Defaults to ``priors.n_actions``; override only for sensitivity
        runs that change the action set.

    Returns
    -------
    dict
        ``beta``        ``(n_chains, n_draws, n_actions, p)``
        ``mu_beta``     ``(n_chains, n_draws, p)``
        ``Sigma_beta``  ``(n_chains, n_draws, p, p)``
        ``sigma2``      ``(n_chains, n_draws)``
        ``meta``        provenance dict (seed, counts, sampler tag).
    """
    X = np.ascontiguousarray(X, dtype=float)
    y = np.ascontiguousarray(y, dtype=float)
    a = np.ascontiguousarray(a, dtype=np.int64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    N, p = X.shape
    if y.shape != (N,):
        raise ValueError(f"y has shape {y.shape}, expected ({N},)")
    if a.shape != (N,):
        raise ValueError(f"a has shape {a.shape}, expected ({N},)")
    if p != priors.p:
        raise ValueError(f"X has p={p} but priors.p={priors.p}")
    A = priors.n_actions if n_actions is None else int(n_actions)
    if a.min() < 0 or a.max() >= A:
        raise ValueError(
            f"action ids must lie in [0, {A}); got min={a.min()}, max={a.max()}"
        )
    if n_draws < 1 or n_burn < 0 or n_chains < 1:
        raise ValueError("n_draws >= 1, n_burn >= 0, n_chains >= 1 required")

    Lambda0_inv = np.linalg.inv(priors.Lambda0)
    Lambda0_inv_mu0 = Lambda0_inv @ priors.mu0

    out_beta = np.empty((n_chains, n_draws, A, p))
    out_mu = np.empty((n_chains, n_draws, p))
    out_Sigma = np.empty((n_chains, n_draws, p, p))
    out_sigma2 = np.empty((n_chains, n_draws))

    seed_seq = np.random.SeedSequence(seed)
    chain_seeds = seed_seq.spawn(n_chains)

    for c in range(n_chains):
        rng = np.random.default_rng(chain_seeds[c])
        # Dispersed init across chains
        sigma2 = float(rng.uniform(0.5, 2.0)) * priors.sigma_y_hat ** 2
        mu_beta = priors.sigma_y_hat * rng.standard_normal(p)
        Sigma_beta = priors.sigma_y_hat ** 2 * np.eye(p)
        beta = mu_beta[None, :] + priors.sigma_y_hat * rng.standard_normal((A, p))

        for it in range(n_burn + n_draws):
            Sigma_beta_inv = np.linalg.inv(Sigma_beta)
            update_beta(beta, X, y, a, sigma2, mu_beta, Sigma_beta_inv, rng)
            mu_beta = update_mu_beta(
                beta, Sigma_beta_inv, Lambda0_inv, Lambda0_inv_mu0, rng
            )
            Sigma_beta = update_Sigma_beta(beta, mu_beta, priors.nu0, priors.Psi0, rng)
            sigma2 = update_sigma2(beta, X, y, a, priors.a0, priors.b0, rng)

            if it >= n_burn:
                d = it - n_burn
                out_beta[c, d] = beta
                out_mu[c, d] = mu_beta
                out_Sigma[c, d] = Sigma_beta
                out_sigma2[c, d] = sigma2

    return {
        "beta": out_beta,
        "mu_beta": out_mu,
        "Sigma_beta": out_Sigma,
        "sigma2": out_sigma2,
        "meta": {
            "seed": int(seed),
            "n_chains": int(n_chains),
            "n_draws": int(n_draws),
            "n_burn": int(n_burn),
            "n_actions": int(A),
            "p": int(p),
            "N": int(N),
            "sampler": "gaussian",
        },
    }


def to_arviz(trace: dict[str, Any]):  # -> az.InferenceData
    """Wrap a trace dict as an ArviZ InferenceData for diagnostic plots.

    Imported lazily to keep ``project.sampler`` import cheap.
    """
    import arviz as az

    posterior = {
        "beta": trace["beta"],
        "mu_beta": trace["mu_beta"],
        "Sigma_beta": trace["Sigma_beta"],
        "sigma2": trace["sigma2"],
    }
    return az.from_dict(posterior=posterior)
