import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# reproducibility
np.random.seed(238238)

# Compatibility for numpy versions (NumPy 2.0 renamed trapz to trapezoid)
integrate = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

# 1. Data and Constants
x = np.array([26.6, 38.5, 34.4, 34, 31, 23.6])
n = len(x)
theta_min, theta_max = 15, 50
ls_min, ls_max = -1, 3
grid_size = 500  # Creates a dense 500x500 grid

# 2. Construct Grid
theta_vals = np.linspace(theta_min, theta_max, grid_size)
log_sigma_vals = np.linspace(ls_min, ls_max, grid_size)
T, LS = np.meshgrid(theta_vals, log_sigma_vals)
S = np.exp(LS)

# 3. Compute Unnormalized Posterior
# Sum of squared residuals for the likelihood
rss = np.sum([(xi - T)**2 for xi in x], axis=0)

# Truncation correction: P(23 <= X <= 39 | theta, sigma)
phi_diff = norm.cdf((39 - T) / S) - norm.cdf((23 - T) / S)
phi_diff = np.clip(phi_diff, 1e-15, 1.0) # Stability

# Log-posterior: -n*log(sigma) - RSS/(2*sigma^2) - n*log(phi_diff)
log_post = -n * LS - rss / (2 * S**2) - n * np.log(phi_diff)

# Use Max-Log trick for stability before exponentiating
post_unnorm = np.exp(log_post - np.max(log_post))

# 4. Normalize Numerically using Trapezoidal Rule
# Integrate over log_sigma (axis 0), then over theta
integral_ls = integrate(post_unnorm, log_sigma_vals, axis=0)
total_volume = integrate(integral_ls, theta_vals)
post_norm = post_unnorm / total_volume

# 5. Marginalize for theta
marginal_theta = integrate(post_norm, log_sigma_vals, axis=0)

# Compute 95% Credible Interval via CDF
cdf_theta = np.array([integrate(marginal_theta[:i], theta_vals[:i]) 
                     for i in range(1, len(theta_vals) + 1)])
# Ensure CDF reaches exactly 1.0
cdf_theta /= cdf_theta[-1]

ci_low = theta_vals[np.searchsorted(cdf_theta, 0.025)]
ci_high = theta_vals[np.searchsorted(cdf_theta, 0.975)]

# 6. Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot A: 2D Joint Posterior
# Using log_sigma for the y-axis as requested by the grid range
cp = ax1.contourf(T, LS, post_norm, levels=50, cmap='magma')
fig.colorbar(cp, ax=ax1, label='Normalized Density')
ax1.set_title(r'Joint Posterior $p(\theta, \log \sigma \mid \text{data})$')
ax1.set_xlabel(r'$\theta$')
ax1.set_ylabel(r'$\log \sigma$')

# Plot B: Marginal Posterior for theta
ax2.plot(theta_vals, marginal_theta, color='blue', lw=2)
ax2.fill_between(theta_vals, 0, marginal_theta, 
                 where=(theta_vals >= ci_low) & (theta_vals <= ci_high), 
                 color='blue', alpha=0.2, label='95% Credible Interval')
ax2.axvline(ci_low, color='red', linestyle='--')
ax2.axvline(ci_high, color='red', linestyle='--')
ax2.set_title(r'Marginal Posterior $p(\theta \mid \text{data})$')
ax2.set_xlabel(r'$\theta$')
ax2.set_ylabel('Density')
ax2.legend()

plt.tight_layout()
plt.show()

print(f"95% Credible Interval: [{ci_low:.2f}, {ci_high:.2f}]") # [16.68, 48.60]

# Compared to non-truncated normal:
mean = np.mean(x)
# Sample standard deviation (ddof=1)
std_err = np.std(x, ddof=1) / np.sqrt(len(x))
# 95% CI using t-distribution with n-1 degrees of freedom
ci_usual = t.interval(0.95, df=len(x)-1, loc=mean, scale=std_err)

print(f"Usual Normal Interval: [{ci_usual[0]:.2f}, {ci_usual[1]:.2f}]") # [25.60, 37.10]