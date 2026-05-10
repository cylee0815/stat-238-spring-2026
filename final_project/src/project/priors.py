"""Prior elicitation for the hierarchical regression (proposal section 2.5).

Priors are elicited from the GSPC pre-training window through ``sigma_y_hat``
(the empirical standard deviation of MC return targets under the uniform
behaviour policy on 1990-1992 GSPC, frozen before SPY's first trading day).

The defaults here are the *weakly-informative on the data scale* choice from
Gelman et al.: the prior on each component of ``mu_beta`` has marginal
variance equal to the residual variance, and the prior on ``Sigma_beta``
has prior expectation equal to ``sigma_y_hat^2 * I``. The ``strength`` knob
exists for the sensitivity sweep in notebook 09.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GaussianPriors:
    """Conjugate priors for the hierarchical Gaussian regression.

    Hierarchy:
        sigma^2     ~ Inverse-Gamma(a0, b0)
        Sigma_beta  ~ Inverse-Wishart(nu0, Psi0)
        mu_beta     ~ Normal(mu0, Lambda0)        # Lambda0 is the COVARIANCE
        beta_a      ~ Normal(mu_beta, Sigma_beta)

    ``Lambda0`` follows the convention used by the user in the API
    sign-off: it is the prior covariance of ``mu_beta``, not the precision.
    The sampler inverts it once at fit time.
    """

    mu0: np.ndarray            # (p,)
    Lambda0: np.ndarray        # (p, p)  prior COVARIANCE of mu_beta
    nu0: float                 # IW dof for Sigma_beta
    Psi0: np.ndarray           # (p, p)  IW scale for Sigma_beta
    a0: float                  # IG shape for sigma^2
    b0: float                  # IG scale for sigma^2
    sigma_y_hat: float
    p: int
    n_actions: int


@dataclass(frozen=True)
class StudentTPriors(GaussianPriors):
    """``GaussianPriors`` plus an Exponential prior on the t degrees of freedom."""

    nu_rate: float = 0.1       # nu ~ Exponential(0.1) -> E[nu] = 10


def make_priors(
    sigma_y_hat: float,
    p: int = 4,
    n_actions: int = 3,
    *,
    strength: float = 1.0,
    nu_sigma: float = 0.1,
) -> GaussianPriors:
    """Weakly-informative priors scaled to the pretraining return variance.

    Default (``strength = 1.0``, ``nu_sigma = 0.1``):
        mu0     = 0
        Lambda0 = sigma_y_hat**2 * I              (prior var of mu_beta)
        nu0     = p + 2                           (proper, minimally informative IW)
        Psi0    = sigma_y_hat**2 * I              (E[Sigma_beta] = sigma_y_hat**2 * I)
        a0, b0  = nu_sigma/2, nu_sigma * sigma_y_hat**2 / 2

    Why the sigma^2 prior is much weaker than the others
    -----------------------------------------------------
    ``sigma_y_hat`` is calibrated from the empirical *marginal* variance of
    MC return targets on the pretraining window. The residual variance
    ``sigma^2`` is the *unexplained* part after fitting ``x' beta_a`` -- in
    general much smaller than the marginal variance, by a factor of
    ``(1 - R^2)``. With unknown ``R^2`` we cannot reliably elicit a prior
    centre for ``sigma^2``. Defaulting to ``nu_sigma = 0.1`` makes the
    prior contribute roughly 0.1 effective observations: the data
    dominates by orders of magnitude on real samples (N ~ thousands), and
    the synthetic-recovery test in ``test_sampler_recovery.py`` clears
    its 90/100 coverage gate at this setting.

    The ``strength`` knob multiplies ``Lambda0`` only and is intended for
    notebook 09's sensitivity sweep over ``{0.1, 1, 10}``. ``nu_sigma`` is
    similarly exposed so the sigma^2 prior can be tightened in sensitivity
    runs (e.g. ``nu_sigma in {1, 4}``) to confirm robustness.
    """
    if sigma_y_hat <= 0:
        raise ValueError(f"sigma_y_hat must be positive, got {sigma_y_hat}")
    if strength <= 0:
        raise ValueError(f"strength must be positive, got {strength}")
    if nu_sigma <= 0:
        raise ValueError(f"nu_sigma must be positive, got {nu_sigma}")
    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}")
    if n_actions < 1:
        raise ValueError(f"n_actions must be >= 1, got {n_actions}")

    s2 = float(sigma_y_hat) ** 2
    return GaussianPriors(
        mu0=np.zeros(p),
        Lambda0=s2 * float(strength) * np.eye(p),
        nu0=float(p + 2),
        Psi0=s2 * np.eye(p),
        a0=float(nu_sigma) / 2.0,
        b0=float(nu_sigma) * s2 / 2.0,
        sigma_y_hat=float(sigma_y_hat),
        p=int(p),
        n_actions=int(n_actions),
    )


def make_priors_t(
    sigma_y_hat: float,
    p: int = 4,
    n_actions: int = 3,
    *,
    strength: float = 1.0,
    nu_sigma: float = 0.1,
    nu_rate: float = 0.1,
) -> StudentTPriors:
    """``make_priors`` plus the Exponential rate for the t degrees of freedom."""
    base = make_priors(
        sigma_y_hat, p=p, n_actions=n_actions, strength=strength, nu_sigma=nu_sigma
    )
    return StudentTPriors(
        mu0=base.mu0,
        Lambda0=base.Lambda0,
        nu0=base.nu0,
        Psi0=base.Psi0,
        a0=base.a0,
        b0=base.b0,
        sigma_y_hat=base.sigma_y_hat,
        p=base.p,
        n_actions=base.n_actions,
        nu_rate=float(nu_rate),
    )
