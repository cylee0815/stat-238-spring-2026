"""Hierarchical Gibbs sampler -- Student-t scale-mixture likelihood.

Same hierarchy as :mod:`project.sampler` plus per-observation latent
scales:

    lambda_t          ~ Gamma(nu / 2, rate = nu / 2)            # E[lambda] = 1
    y_t | beta, lambda, sigma^2  ~ N(x_t^T beta_{a_t}, sigma^2 / lambda_t)

so the marginal likelihood of ``y_t`` is Student-t with ``nu`` degrees of
freedom, location ``x_t^T beta_{a_t}``, and scale ``sigma``. The sampler
follows the proposal section 2.6 with one block per quantity:

  1. lambda_t | rest        -- Gamma full conditional, vectorised over t
  2. beta_a   | rest        -- weighted Normal, Cholesky draw
  3. mu_beta  | rest        -- Normal (reused from project.sampler)
  4. Sigma_beta | rest      -- Inverse-Wishart, Bartlett (reused)
  5. sigma^2  | rest        -- Inverse-Gamma with lambda-weighted RSS
  6. log nu   | rest        -- random-walk Metropolis on log nu, with
                                Robbins-Monro adaptation of log step-size
                                during the first half of burn-in only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import gammaln

from project.priors import StudentTPriors
from project.sampler import (
    _draw_normal_precision,
    _sample_invwishart,
    update_mu_beta,
    update_Sigma_beta,
)
from project.utils import SEED


# ---------------------------------------------------------------------------
# Student-t-specific full conditionals
# ---------------------------------------------------------------------------

def update_lambda(
    X: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    beta: np.ndarray,
    sigma2: float,
    nu: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample ``lambda_t ~ Gamma((nu + 1)/2, rate = (nu + r_t^2/sigma^2)/2)``."""
    pred = (X * beta[a]).sum(axis=1)
    resid = y - pred
    shape = 0.5 * (nu + 1.0)
    rate = 0.5 * (nu + resid * resid / sigma2)
    return rng.gamma(shape=shape, scale=1.0 / rate)


def update_beta_t(
    beta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    lam: np.ndarray,
    sigma2: float,
    mu_beta: np.ndarray,
    Sigma_beta_inv: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """In-place ``beta`` update under per-obs lambda weights.

    Per action k:
        V_k^{-1} = sigma^{-2} X_k^T diag(lambda_k) X_k + Sigma_beta^{-1}
        m_k      = V_k (sigma^{-2} X_k^T (lambda_k * y_k) + Sigma_beta^{-1} mu_beta)
    """
    A, _ = beta.shape
    Sb_inv_mu = Sigma_beta_inv @ mu_beta
    inv_sigma2 = 1.0 / sigma2
    for k in range(A):
        mask = (a == k)
        if mask.any():
            X_k = X[mask]
            y_k = y[mask]
            lam_k = lam[mask]
            XtWX = X_k.T @ (X_k * lam_k[:, None])
            XtWy = X_k.T @ (lam_k * y_k)
            precision = inv_sigma2 * XtWX + Sigma_beta_inv
            nat = inv_sigma2 * XtWy + Sb_inv_mu
        else:
            precision = Sigma_beta_inv
            nat = Sb_inv_mu
        beta[k] = _draw_normal_precision(precision, nat, rng)


def update_sigma2_t(
    beta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    lam: np.ndarray,
    a0: float,
    b0: float,
    rng: np.random.Generator,
) -> float:
    """Lambda-weighted ``sigma^2`` update: ``IG(a0 + N/2, b0 + 0.5 sum_t lambda_t r_t^2)``."""
    pred = (X * beta[a]).sum(axis=1)
    resid = y - pred
    weighted_rss = float(np.sum(lam * resid * resid))
    a_post = a0 + 0.5 * y.shape[0]
    b_post = b0 + 0.5 * weighted_rss
    return float(b_post / rng.gamma(shape=a_post, scale=1.0))


def _log_t_marginal(resid: np.ndarray, sigma2: float, nu: float) -> float:
    """``sum_t log f_t(y_t | beta, sigma^2, nu)`` -- the lambda-marginalized
    Student-t log-likelihood.

    Used in the *partially collapsed* MH on ``nu`` (Liu-Wong-Kong 1994).
    The MH target here is ``p(nu | y, beta, sigma^2)`` -- with the latent
    lambdas integrated out analytically (Gamma-Normal mixture -> Student-t)
    -- *not* the conditional ``p(nu | lambda)``. A reader checking the
    math against the standard conjugate-block decomposition will not find
    a ``p(nu | lambda)`` step in this sampler; the lambdas are still
    resampled every sweep for the beta and sigma^2 updates, but the nu
    block sees only their integrated effect.

    Numerical note. ``gammaln(a) - gammaln(b)`` stays finite as
    ``nu -> 0+``, so the expression does not blow up at the boundary; in
    any case the ``Exponential(nu_rate)`` prior pulls the chain away from
    that boundary.
    """
    n = resid.shape[0]
    a = 0.5 * (nu + 1.0)
    b = 0.5 * nu
    return float(
        n * (gammaln(a) - gammaln(b) - 0.5 * np.log(np.pi * nu * sigma2))
        - a * np.sum(np.log1p(resid * resid / (nu * sigma2)))
    )


def propose_log_nu(
    log_nu: float,
    step: float,
    resid: np.ndarray,
    sigma2: float,
    nu_rate: float,
    rng: np.random.Generator,
) -> tuple[float, bool]:
    """Partially-collapsed RW-MH on ``log nu``.

    Proposal ``log nu' = log nu + step * z`` with ``z ~ N(0, 1)``.
    Target on log nu is ``p(log nu | y, beta, sigma^2)`` -- i.e. the
    Student-t marginal log-likelihood plus the Exponential(``nu_rate``)
    prior, plus the Jacobian ``log(nu)`` for the log transform.
    """
    z = rng.standard_normal()
    log_nu_prop = log_nu + step * z
    nu = float(np.exp(log_nu))
    nu_prop = float(np.exp(log_nu_prop))

    log_lik_prop = _log_t_marginal(resid, sigma2, nu_prop)
    log_lik_curr = _log_t_marginal(resid, sigma2, nu)
    log_prior_prop = -nu_rate * nu_prop
    log_prior_curr = -nu_rate * nu
    log_jac_prop = log_nu_prop
    log_jac_curr = log_nu

    log_alpha = (
        (log_lik_prop - log_lik_curr)
        + (log_prior_prop - log_prior_curr)
        + (log_jac_prop - log_jac_curr)
    )
    if np.log(rng.uniform()) < log_alpha:
        return log_nu_prop, True
    return log_nu, False


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def gibbs_student_t(
    X: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    priors: StudentTPriors,
    *,
    n_draws: int = 6000,
    n_burn: int = 2000,
    n_chains: int = 4,
    seed: int = SEED,
    n_actions: int | None = None,
    nu_init: float = 10.0,
    log_step_init: float = float(np.log(0.5)),
    rm_target: float = 0.30,
    keep_omega: bool = False,
) -> dict[str, Any]:
    """Run the Student-t scale-mixture Gibbs sampler.

    Returns a dict with the same keys as :func:`project.sampler.gibbs_gaussian`
    plus:
        ``nu``               ``(n_chains, n_draws)``
        ``nu_acceptance``    ``(n_chains,)``    -- post-burn acceptance rate
        ``log_step_trace``   ``(n_chains, n_burn // 2)``
                                         -- Robbins-Monro path on log step,
                                            frozen after the first half of burn-in.
        ``omega``            ``(n_chains, n_draws, N)``   -- stored only if
                                         ``keep_omega=True``. ``omega = 1 / lambda``,
                                         the per-obs variance multiplier.

    The default chain budget (4 x 6000 + 2000) is larger than for the
    Gaussian sampler because the RW-MH on ``nu`` mixes much worse than the
    conjugate blocks. The ``ess_nu`` diagnostic in the recovery test is
    the gate for whether to bump ``n_draws`` further.
    """
    if not isinstance(priors, StudentTPriors):
        raise TypeError("priors must be a StudentTPriors instance")
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
        raise ValueError(f"action ids must lie in [0, {A})")
    if n_draws < 1 or n_burn < 1 or n_chains < 1:
        raise ValueError("n_draws, n_burn, n_chains must all be >= 1")
    if nu_init <= 0:
        raise ValueError("nu_init must be > 0")
    if not (0.0 < rm_target < 1.0):
        raise ValueError("rm_target must be in (0, 1)")

    Lambda0_inv = np.linalg.inv(priors.Lambda0)
    Lambda0_inv_mu0 = Lambda0_inv @ priors.mu0

    out_beta = np.empty((n_chains, n_draws, A, p))
    out_mu = np.empty((n_chains, n_draws, p))
    out_Sigma = np.empty((n_chains, n_draws, p, p))
    out_sigma2 = np.empty((n_chains, n_draws))
    out_nu = np.empty((n_chains, n_draws))
    out_omega = np.empty((n_chains, n_draws, N)) if keep_omega else None

    n_burn_half = max(n_burn // 2, 1)
    log_step_trace = np.empty((n_chains, n_burn_half))
    nu_acceptance = np.empty(n_chains)

    seed_seq = np.random.SeedSequence(seed)
    chain_seeds = seed_seq.spawn(n_chains)

    for c in range(n_chains):
        rng = np.random.default_rng(chain_seeds[c])
        sigma2 = float(rng.uniform(0.5, 2.0)) * priors.sigma_y_hat ** 2
        mu_beta = priors.sigma_y_hat * rng.standard_normal(p)
        Sigma_beta = priors.sigma_y_hat ** 2 * np.eye(p)
        beta = mu_beta[None, :] + priors.sigma_y_hat * rng.standard_normal((A, p))
        log_nu = float(np.log(nu_init))
        log_step = float(log_step_init)
        lam = np.ones(N)

        accept_post = 0
        for it in range(n_burn + n_draws):
            nu = float(np.exp(log_nu))
            lam = update_lambda(X, y, a, beta, sigma2, nu, rng)

            Sigma_beta_inv = np.linalg.inv(Sigma_beta)
            update_beta_t(beta, X, y, a, lam, sigma2, mu_beta, Sigma_beta_inv, rng)
            mu_beta = update_mu_beta(
                beta, Sigma_beta_inv, Lambda0_inv, Lambda0_inv_mu0, rng
            )
            Sigma_beta = update_Sigma_beta(beta, mu_beta, priors.nu0, priors.Psi0, rng)
            sigma2 = update_sigma2_t(beta, X, y, a, lam, priors.a0, priors.b0, rng)

            step = float(np.exp(log_step))
            resid_now = y - (X * beta[a]).sum(axis=1)
            log_nu, accepted = propose_log_nu(
                log_nu, step, resid_now, sigma2, priors.nu_rate, rng
            )

            if it < n_burn_half:
                gamma_i = (it + 1) ** (-0.6)
                log_step = log_step + gamma_i * (float(accepted) - rm_target)
                log_step_trace[c, it] = log_step

            if it >= n_burn:
                d = it - n_burn
                out_beta[c, d] = beta
                out_mu[c, d] = mu_beta
                out_Sigma[c, d] = Sigma_beta
                out_sigma2[c, d] = sigma2
                out_nu[c, d] = float(np.exp(log_nu))
                accept_post += int(accepted)
                if keep_omega:
                    out_omega[c, d] = 1.0 / lam

        nu_acceptance[c] = accept_post / n_draws

    result: dict[str, Any] = {
        "beta": out_beta,
        "mu_beta": out_mu,
        "Sigma_beta": out_Sigma,
        "sigma2": out_sigma2,
        "nu": out_nu,
        "nu_acceptance": nu_acceptance,
        "log_step_trace": log_step_trace,
        "meta": {
            "seed": int(seed),
            "n_chains": int(n_chains),
            "n_draws": int(n_draws),
            "n_burn": int(n_burn),
            "n_actions": int(A),
            "p": int(p),
            "N": int(N),
            "nu_init": float(nu_init),
            "log_step_init": float(log_step_init),
            "rm_target": float(rm_target),
            "keep_omega": bool(keep_omega),
            "sampler": "student_t",
        },
    }
    if keep_omega:
        result["omega"] = out_omega
    return result
