---
name: stat238-bayes
description: >
  Coding conventions and reusable templates for STAT 238 Bayesian Statistics
  homework (UC Berkeley). Use this skill whenever working on Bayesian inference
  tasks involving: Metropolis-Hastings samplers, Gibbs samplers, PyMC models,
  grid/Laplace posterior approximations, posterior predictive distributions,
  hierarchical models, or trace-plot diagnostics. Triggers on any HW problem
  that mentions MCMC, posterior sampling, prior specification, or Bayesian
  inference in Python. Always read this skill before writing any sampler or
  PyMC model — it encodes established conventions from previous submissions
  that must be followed consistently.
---

# STAT 238 Bayesian Homework — Coding Skill

This skill encodes Thomas's established patterns from HW4. Follow these
conventions exactly for consistency across submissions.

---

## 1. Standard Imports

Always start every notebook/script with this block:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2, gamma as sp_gamma
from scipy.linalg import solve, solve_triangular
from scipy.linalg import svd as lsvd
from scipy.special import gammaln
import statsmodels.api as sm
import pymc as pm
import arviz as az
import pytensor.tensor as pt
import warnings
warnings.filterwarnings('ignore')
```

---

## 2. Grid Posterior (Numerical Integration)

Use for 1D or 2D posteriors when the unnormalized log-posterior is cheap to evaluate.

```python
# --- 1D grid ---
theta_grid = np.linspace(lo, hi, 500)
log_post = np.array([log_unnorm_post(t) for t in theta_grid])
log_post -= log_post.max()          # numerical stability
post = np.exp(log_post)
dtheta = theta_grid[1] - theta_grid[0]
post /= post.sum() * dtheta         # normalize

# posterior mean and 95% CI
post_mean = (theta_grid * post).sum() * dtheta
cdf = np.cumsum(post) * dtheta
ci_lo = theta_grid[np.searchsorted(cdf, 0.025)]
ci_hi = theta_grid[np.searchsorted(cdf, 0.975)]

# --- 2D grid (e.g. alpha, beta) ---
A, B = np.meshgrid(a_grid, b_grid)   # shape (n_b, n_a)
log_post2d = log_unnorm_post_2d(A, B) # vectorized
log_post2d -= log_post2d.max()
post2d = np.exp(log_post2d)
Z = post2d.sum() * da * db
post2d /= Z
```

---

## 3. Metropolis-Hastings Sampler

**Standard random-walk MH template** — used for every MH problem.

```python
def mh_sampler(log_post_fn, theta_init, proposal_sd, n_samples=20000, burn_in=5000):
    """
    Random-walk Metropolis-Hastings.
    
    Parameters
    ----------
    log_post_fn  : callable, takes theta -> scalar log posterior (unnormalized)
    theta_init   : array_like, starting point
    proposal_sd  : float or array, std of isotropic (or diagonal) Gaussian proposal
    n_samples    : int, total iterations including burn-in
    burn_in      : int, samples to discard
    
    Returns
    -------
    samples      : (n_samples - burn_in, dim) array of post-burn-in draws
    accept_rate  : float
    """
    theta = np.atleast_1d(np.array(theta_init, dtype=float))
    dim = len(theta)
    samples = np.zeros((n_samples, dim))
    log_p_curr = log_post_fn(theta)
    n_accept = 0

    for i in range(n_samples):
        proposal = theta + np.random.randn(dim) * proposal_sd
        log_p_prop = log_post_fn(proposal)
        log_alpha = log_p_prop - log_p_curr
        if np.log(np.random.rand()) < log_alpha:
            theta = proposal
            log_p_curr = log_p_prop
            n_accept += 1
        samples[i] = theta

    accept_rate = n_accept / n_samples
    return samples[burn_in:], accept_rate
```

**Tuning rule**: target acceptance rate 20–40% for multivariate, 40–60% for 1D.
Always print and comment on acceptance rate. Adjust `proposal_sd` until in range.

```python
# Typical usage
samples, ar = mh_sampler(log_post_fn, theta_init=theta_hat,
                          proposal_sd=0.5, n_samples=25000, burn_in=5000)
print(f"Acceptance rate: {ar:.3f}")  # aim for 0.20–0.40
```

**For 2D problems** (e.g. alpha/beta), use a 2-element `proposal_sd` array
or a full covariance matrix scaled from the Laplace inverse-Hessian:

```python
# Initialize proposal_sd from Laplace covariance (if available)
proposal_cov = post_cov          # from Laplace approx
proposal_sd = 2.38 / np.sqrt(dim) * np.linalg.cholesky(proposal_cov)

def mh_sampler_cov(log_post_fn, theta_init, L_prop, n_samples=25000, burn_in=5000):
    """MH with full covariance proposal: z = L_prop @ randn(dim)."""
    theta = np.array(theta_init, dtype=float)
    dim = len(theta)
    samples = np.zeros((n_samples, dim))
    log_p_curr = log_post_fn(theta)
    n_accept = 0
    for i in range(n_samples):
        proposal = theta + L_prop @ np.random.randn(dim)
        log_p_prop = log_post_fn(proposal)
        if np.log(np.random.rand()) < log_p_prop - log_p_curr:
            theta = proposal
            log_p_curr = log_p_prop
            n_accept += 1
        samples[i] = theta
    return samples[burn_in:], n_accept / n_samples
```

---

## 4. Gibbs Sampler

Template for a generic Gibbs sampler. Derive full conditionals analytically,
then sample from them exactly. See HW5 problems 5, 6, 7, 8.

```python
def gibbs_sampler(init_state, full_conditionals, n_samples=10000):
    """
    init_state       : dict or list of initial values for each block
    full_conditionals: list of callables, each takes current state -> new draw
    """
    state = list(init_state)
    dim = len(state)
    samples = np.zeros((n_samples, dim))
    for t in range(n_samples):
        for k, fc in enumerate(full_conditionals):
            state[k] = fc(state)
        samples[t] = state
    return samples
```

For conjugate full conditionals, use scipy:
```python
from scipy.stats import norm, gamma as sp_gamma, multivariate_normal

# Normal full conditional: x | rest ~ N(mu_fc, sigma_fc^2)
new_x = np.random.normal(mu_fc, sigma_fc)

# Gamma full conditional: tau^{-2} | rest ~ Gamma(a, b)  [shape, rate]
new_prec = np.random.gamma(shape=a_fc, scale=1.0/b_fc)

# Multivariate normal: beta | rest ~ N(mu_vec, Sigma_mat)
new_beta = np.random.multivariate_normal(mu_vec, Sigma_mat)
```

---

## 5. Trace Plots and Convergence Diagnostics

HW5 requires trace plots for every MCMC run. Use this standard layout:

```python
def plot_traces(samples, param_names, title="Trace plots", burn_in=0):
    """Plot trace plots and marginal histograms side by side."""
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, 2, figsize=(12, 3 * n_params))
    if n_params == 1:
        axes = axes[None, :]
    for i, name in enumerate(param_names):
        s = samples[:, i]
        # Trace
        axes[i, 0].plot(s, lw=0.5, alpha=0.7, color="steelblue")
        axes[i, 0].set_title(f"Trace: {name}"); axes[i, 0].set_xlabel("Iteration")
        # Histogram
        axes[i, 1].hist(s, bins=50, density=True, color="steelblue", alpha=0.7)
        axes[i, 1].set_title(f"Posterior: {name}")
    plt.suptitle(title, y=1.01); plt.tight_layout(); plt.show()

# Usage
plot_traces(samples, param_names=["alpha", "beta"])
```

**Convergence comment template** (write this after every trace plot):
> "The trace plots show [good/adequate/poor] mixing. The chains [appear stationary /
> show slow drift / exhibit clear periodicity]. Acceptance rate = {ar:.3f}, which is
> [within/outside] the target range. [The chain has / has not] mixed adequately."

---

## 6. PyMC Model Template

Established PyMC conventions — follow exactly.

```python
with pm.Model() as model_name:
    # --- Priors ---
    # Flat (improper): use pm.Flat, always provide initvals
    beta = pm.Flat("beta", shape=p)

    # Weakly informative: use explicit distributions
    mu   = pm.Normal("mu", mu=0, sigma=100)
    tau  = pm.HalfNormal("tau", sigma=10)
    
    # Precision -> std trick (for Gamma precision priors)
    prec = pm.Gamma("prec", alpha=0.001, beta=0.001)
    sigma = pm.Deterministic("sigma", 1.0 / pt.sqrt(prec))

    # --- Likelihood ---
    # Logistic:
    y_obs = pm.Bernoulli("y_obs", logit_p=X @ beta, observed=y)
    # Probit:
    p_prob = 0.5 * (1 + pt.erf((X @ beta) / np.sqrt(2.0)))
    p_clip = pt.clip(p_prob, 1e-7, 1 - 1e-7)
    y_obs = pm.Bernoulli("y_obs", p=p_clip, observed=y)
    # Normal:
    y_obs = pm.Normal("y_obs", mu=mu_expr, sigma=sigma, observed=y)
    # Poisson:
    y_obs = pm.Poisson("y_obs", mu=pt.exp(log_mu_expr), observed=y)
    # Binomial:
    y_obs = pm.Binomial("y_obs", n=n_i, p=p_expr, observed=y_i)

    # --- Sampling --- (always use these settings)
    trace = pm.sample(
        draws=2000,
        tune=1000,
        chains=2,
        target_accept=0.9,        # increase to 0.95 for difficult posteriors
        initvals={"beta": beta_hat},  # ALWAYS initialize at MLE/MAP for pm.Flat
        return_inferencedata=True,
        progressbar=False,
        random_seed=238,
    )
```

**Post-sampling checks** — always run:

```python
# Divergences and R-hat
n_div  = trace.sample_stats["diverging"].values.sum()
rhat   = az.summary(trace, var_names=[...])["r_hat"].max()
print(f"Divergences: {n_div}   Max R-hat: {rhat:.4f}")
# Good: n_div == 0, rhat < 1.01

# Summary table
print(az.summary(trace, var_names=[...], hdi_prob=0.95, round_to=4).to_string())
```

---

## 7. Laplace Approximation

Standard pattern for any log-posterior with a unique mode.

```python
from scipy.optimize import minimize

def laplace_approx(log_post_fn, theta_init, dim):
    """Returns (mode, cov_matrix, post_sd)."""
    result = minimize(lambda t: -log_post_fn(t), theta_init, method="BFGS")
    mode = result.x
    H = result.hess_inv   # BFGS inverse Hessian approximation
    # Or use numerical Hessian:
    from scipy.optimize import approx_fprime
    eps = 1e-5
    H_num = np.zeros((dim, dim))
    for j in range(dim):
        e = np.zeros(dim); e[j] = eps
        H_num[:, j] = (approx_fprime(mode, log_post_fn, eps) -
                       approx_fprime(mode - e, log_post_fn, eps)) / eps
    post_cov = np.linalg.inv(-H_num)
    post_sd  = np.sqrt(np.diag(post_cov))
    return mode, post_cov, post_sd
```

---

## 8. Comparison Table

Every problem that compares methods (Laplace vs. MH vs. PyMC) must print
a table in this format:

```python
def print_comparison(param_names, methods_dict):
    """
    methods_dict: {method_name: (means_array, sds_array)}
    """
    method_names = list(methods_dict.keys())
    header = f"{'':14}" + "".join(f"{'Mean':>10}{'SD':>8}" for _ in method_names)
    sub    = f"{'':14}" + "".join(f"  {m[:12]:>16}" for m in method_names)
    print(sub); print("-" * len(sub))
    for i, p in enumerate(param_names):
        row = f"  {p:<12}"
        for means, sds in methods_dict.values():
            row += f"  {means[i]:>8.4f}  {sds[i]:>6.4f}"
        print(row)

# Usage
print_comparison(
    param_names=["alpha", "beta"],
    methods_dict={
        "Laplace":  (laplace_means, laplace_sds),
        "MH":       (mh_samples.mean(0), mh_samples.std(0)),
        "PyMC":     (pymc_means, pymc_sds),
    }
)
```

---

## 9. Posterior Predictive

For computing posterior predictive distributions and credible intervals:

```python
def posterior_predictive_samples(param_samples, likelihood_sampler, n_pred=10000):
    """
    param_samples   : (N, dim) posterior draws
    likelihood_sampler: callable, takes one param draw -> one predictive draw
    """
    idx = np.random.choice(len(param_samples), size=n_pred, replace=True)
    return np.array([likelihood_sampler(param_samples[i]) for i in idx])

# Example: Poisson predictive
def poisson_pred(params):
    alpha, beta = params
    t_new = 11   # 1986 is year index 11 (1976=1, ..., 1985=10)
    lam = np.exp(alpha + beta * t_new)
    return np.random.poisson(lam)

y_pred = posterior_predictive_samples(mh_samples, poisson_pred)
ci_lo, ci_hi = np.percentile(y_pred, [2.5, 97.5])
print(f"95% predictive CI: [{ci_lo:.1f}, {ci_hi:.1f}]")

plt.figure(figsize=(8, 4))
plt.hist(y_pred, bins=30, density=True, color="steelblue", alpha=0.7)
plt.axvline(ci_lo, color="red", linestyle="--", label=f"95% CI: [{ci_lo:.0f}, {ci_hi:.0f}]")
plt.axvline(ci_hi, color="red", linestyle="--")
plt.xlabel("Predicted count"); plt.title("Posterior predictive distribution")
plt.legend(); plt.tight_layout(); plt.show()
```

---

## 10. Plotting Conventions

Always follow these style defaults (matches HW4):

```python
# Figure sizes
FIGSIZE_WIDE  = (11, 4)   # time series, traces, single-panel
FIGSIZE_STD   = (9, 5)    # scatter + fit
FIGSIZE_SMALL = (8, 4)    # histograms, 1D posteriors
FIGSIZE_MULTI = (14, 6)   # multi-panel grids

# Data scatter
plt.scatter(x, y, s=4, alpha=0.2, color="gray", label="Data")

# Fit curves
plt.plot(x, y_fit, lw=2.5, color="steelblue", label="Posterior mean")
plt.plot(x, y_alt, lw=2.0, color="darkorange", linestyle="--", label="Laplace/Ridge")

# Posterior samples (fan)
for j in range(0, N, 10):
    plt.plot(x, mu_samp[j], lw=0.6, alpha=0.4, color="steelblue")

# Always end with
plt.tight_layout(); plt.show()
```

---

## 11. Common Log-Posterior Patterns

### Cauchy likelihood (Problem 1, HW5)
```python
def log_post_cauchy(theta, x_obs, lo=0, hi=100):
    if not (lo < theta < hi):
        return -np.inf
    return -np.sum(np.log(1 + (x_obs - theta)**2))
```

### Binomial logistic (Problem 2, HW5)
```python
def log_post_bioassay(params, x_i, n_i, y_i):
    alpha, beta = params
    eta = alpha + beta * x_i
    log_p = eta - np.log1p(np.exp(eta))   # log sigmoid, stable
    log_1mp = -np.log1p(np.exp(eta))
    return np.sum(y_i * log_p + (n_i - y_i) * log_1mp)
```

### Poisson log-linear (Problem 4, HW5)
```python
def log_post_poisson(params, y_obs, t_vals, M=10):
    alpha, beta = params
    if abs(alpha) > M or abs(beta) > M:
        return -np.inf
    lam = np.exp(alpha + beta * t_vals)
    return np.sum(y_obs * np.log(lam) - lam)   # gammaln(y+1) is constant
```

### Multivariate Gaussian joint density (Problem 5, HW5)
```python
# pi(x) ∝ exp(-x' A x / 2 + ...)
# Full conditional x_k | x_{-k} ~ N(mu_k, sigma_k^2):
# Extract from precision matrix Q = A (symmetric)
# mu_k = -Q[k,k]^{-1} * sum_{j != k} Q[k,j] * x_j
# sigma_k^2 = 1/Q[k,k]
```

---

## 12. Quarto Notebook Format

Output is a Quarto `.qmd` file rendered to PDF.
- Header: `jupyter: pymc_env`
- Use `\newpage` between problems
- Every code cell should have visible output (print statements + plots)
- Math in `$$...$$` blocks, inline in `$...$`
- Always echo code: `execute: echo: true`

---

## 13. Random Seeds

- Use `np.random.seed(238)` at the top of any sampling section
- Use `random_seed=238` in all `pm.sample()` calls
- Use `random_seed=42` for data subsampling / train-test splits

---

## 14. HW5-Specific Problem Map

| Problem | Key methods needed |
|---------|-------------------|
| 1       | Grid posterior, MH (1D), PyMC Cauchy |
| 2       | Laplace approx, MH (2D), PyMC Binomial-logistic, conditional distributions |
| 3       | MH (p-dim), PyMC probit, trace plots |
| 4       | Laplace approx, MH (2D Poisson), PyMC Poisson, posterior predictive |
| 5       | Gibbs sampler (multivariate Gaussian), trace plots, empirical covariance |
| 6       | Gibbs sampler (hierarchical Normal), PyMC hierarchical model |
| 7       | Gibbs sampler (high-dim regression), comparison with lecture method |
| 8       | Gibbs sampler (two-school model), PyMC, P(delta>0), predictive probability |