import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# reproducibility
np.random.seed(238238)

# Compatibility wrapper for numpy versions
integrate = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

# 1. Data and Constants
x_obs = np.array([26.6, 38.5, 34.4, 34, 31, 23.6])
n_obs = len(x_obs)
n_fail = 2

theta_min, theta_max = 15, 50
ls_min, ls_max = -1, 3
grid_size = 500

# 2. Construct Grid
theta_vals = np.linspace(theta_min, theta_max, grid_size)
log_sigma_vals = np.linspace(ls_min, ls_max, grid_size)
T, LS = np.meshgrid(theta_vals, log_sigma_vals)
S = np.exp(LS)

# 3. Compute Unnormalized Posterior (Censored Likelihood)
rss = np.sum([(xi - T)**2 for xi in x_obs], axis=0)
# Probability of Falling in Range [23, 39]
p_in_range = norm.cdf((39 - T) / S) - norm.cdf((23 - T) / S)
# Probability of Failure (falling outside)
p_fail = 1.0 - p_in_range
p_fail = np.clip(p_fail, 1e-15, 1.0) # Numerical stability

# Log-posterior calculation
# log(sigma^-7) + log(exp(-rss/2s^2)) + 2*log(p_fail)
log_post = -7 * LS - rss / (2 * S**2) + n_fail * np.log(p_fail)

# Stabilize and exponentiate
post_unnorm = np.exp(log_post - np.max(log_post))

# 4. Normalize Numerically
integral_ls = integrate(post_unnorm, log_sigma_vals, axis=0)
total_volume = integrate(integral_ls, theta_vals)
post_norm = post_unnorm / total_volume

# 5. Marginalize for theta
marginal_theta = integrate(post_norm, log_sigma_vals, axis=0)

# Compute 95% Credible Interval
cdf_theta = np.array([integrate(marginal_theta[:i], theta_vals[:i]) 
                     for i in range(1, len(theta_vals) + 1)])
cdf_theta /= cdf_theta[-1]

ci_low = theta_vals[np.searchsorted(cdf_theta, 0.025)]
ci_high = theta_vals[np.searchsorted(cdf_theta, 0.975)]

# 6. Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot A: Joint Posterior
cp = ax1.contourf(T, LS, post_norm, levels=50, cmap='magma')
fig.colorbar(cp, ax=ax1, label='Normalized Density')
ax1.set_title(r'Joint Posterior $p(\theta, \log \sigma \mid \text{data})$ (Censored)')
ax1.set_xlabel(r'$\theta$')
ax1.set_ylabel(r'$\log \sigma$')

# Plot B: Marginal Posterior
ax2.plot(theta_vals, marginal_theta, color='green', lw=2)
ax2.fill_between(theta_vals, 0, marginal_theta, 
                 where=(theta_vals >= ci_low) & (theta_vals <= ci_high), 
                 color='green', alpha=0.2, label='95% Credible Interval')
ax2.axvline(ci_low, color='red', linestyle='--')
ax2.axvline(ci_high, color='red', linestyle='--')
ax2.set_title(r'Marginal Posterior $p(\theta \mid \text{data})$ (Censored)')
ax2.set_xlabel(r'$\theta$')
ax2.set_ylabel('Density')
ax2.legend()

plt.tight_layout()
plt.savefig('q6.png')
plt.show()

print(f"Censored 95% Credible Interval: [{ci_low:.2f}, {ci_high:.2f}]") # [23.35, 39.62]