import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

# Reproducibility
np.random.seed(238238)

# Compatibility for numpy versions (NumPy 2.0 renamed trapz to trapezoid)
integrate = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

def get_laplace_posterior(theta_grid, data):
    """Computes marginalized Laplace posterior: f(theta) proportional to (1/M(theta))^n"""
    n = len(data)
    m_theta = np.sum(np.abs(data[:, np.newaxis] - theta_grid), axis=0)
    log_post = -n * np.log(m_theta)
    post = np.exp(log_post - np.max(log_post))
    return post / integrate(post, theta_grid)

def get_normal_posterior(theta_grid, data):
    """Computes the marginalized Normal posterior (t-distribution)"""
    n = len(data)
    x_bar = np.mean(data)
    s = np.std(data, ddof=1)
    se = s / np.sqrt(n)
    return t.pdf(theta_grid, df=n-1, loc=x_bar, scale=se)

def get_ci(theta_grid, density):
    """Computes 95% Bayesian credible interval from discretized density"""
    cdf = np.cumsum(density)
    cdf /= cdf[-1]
    low = theta_grid[np.searchsorted(cdf, 0.025)]
    high = theta_grid[np.searchsorted(cdf, 0.975)]
    return low, high

def plot_joint_posterior(x_data):
    """Generates the 2D Joint Posterior Plot for Part (b)"""
    theta_vals = np.linspace(15, 50, 500)
    log_sigma_vals = np.linspace(-1, 3, 500)
    T, LS = np.meshgrid(theta_vals, log_sigma_vals)
    S = np.exp(LS)
    
    m_theta = np.sum([np.abs(xi - T) for xi in x_data], axis=0)
    log_post = -(len(x_data) + 1) * LS - m_theta / S
    post_unnorm = np.exp(log_post - np.max(log_post))
    
    plt.figure(figsize=(10, 8))
    cp = plt.contourf(T, LS, post_unnorm, levels=50, cmap='magma')
    plt.colorbar(cp, label='Normalized Density')
    plt.scatter(x_data, np.zeros_like(x_data) + 2.8, color='cyan', marker='|', label='Data Locations')
    plt.title(r'Joint Posterior $p(\theta, \log \sigma \mid \text{data})$ (Laplace Model)')
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\log \sigma$')
    plt.legend()
    plt.savefig('q7_joint.png', dpi=300)
    plt.show()

def run_q7_analysis(x_data, x_outlier_data):
    # Grids
    grid_orig = np.linspace(15, 50, 2000)
    grid_outlier = np.linspace(0, 130, 5000)

    # --- Calculations ---
    post_lap_orig = get_laplace_posterior(grid_orig, x_data)
    post_norm_orig = get_normal_posterior(grid_orig, x_data)
    ci_lap_orig = get_ci(grid_orig, post_lap_orig)
    ci_norm_orig = get_ci(grid_orig, post_norm_orig)

    post_lap_out = get_laplace_posterior(grid_outlier, x_outlier_data)
    post_norm_out = get_normal_posterior(grid_outlier, x_outlier_data)
    ci_lap_out = get_ci(grid_outlier, post_lap_out)
    ci_norm_out = get_ci(grid_outlier, post_norm_out)

    # --- Part (b): Marginal Posterior (Jagged Nature) ---
    plt.figure(figsize=(10, 8))
    plt.plot(grid_orig, post_lap_orig, color='blue', lw=2)
    plt.scatter(x_data, np.zeros_like(x_data), color='red', marker='x', zorder=5, label='Data Points')
    plt.title(r'Marginal Posterior $p(\theta \mid \text{data})$ (Laplace Model)')
    plt.xlabel(r'$\theta$')
    plt.ylabel('Density')
    plt.grid(axis='y', alpha=0.2)
    plt.legend()
    plt.savefig('q7b_marginal.png', dpi=300)
    plt.show()

    # --- Part (d): Robustness Comparison ---
    plt.figure(figsize=(10, 8))
    plt.plot(grid_outlier, post_lap_out, color='blue', label='Laplace Model')
    plt.fill_between(grid_outlier, 0, post_lap_out, 
                     where=(grid_outlier >= ci_lap_out[0]) & (grid_outlier <= ci_lap_out[1]), 
                     color='blue', alpha=0.1, label='Laplace 95% CI')
    plt.axvline(ci_lap_out[0], color='blue', linestyle=':', alpha=0.6)
    plt.axvline(ci_lap_out[1], color='blue', linestyle=':', alpha=0.6)

    plt.plot(grid_outlier, post_norm_out, color='orange', linestyle='--', label='Normal Model')
    plt.axvline(ci_norm_out[0], color='red', linestyle='--', alpha=0.8, label='Normal 95% CI bounds')
    plt.axvline(ci_norm_out[1], color='red', linestyle='--', alpha=0.8)

    plt.title(r"Q7(d): Robustness to Outlier ($x_7=120$)")
    plt.xlabel(r"$\theta$")
    plt.ylabel("Density")
    plt.xlim(0, 130)
    plt.legend()
    plt.savefig('q7d_robustness.png', dpi=300)
    plt.show()

    # --- Summary Figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    ax1.plot(grid_orig, post_lap_orig, label='Laplace Model', color='blue')
    ax1.plot(grid_orig, post_norm_orig, label='Normal Model', color='red', linestyle='--')
    ax1.set_title("Original Data (n=6)")
    ax1.set_xlabel(r"$\theta$")
    ax1.legend()

    ax2.plot(grid_outlier, post_lap_out, label='Laplace Model', color='blue')
    ax2.plot(grid_outlier, post_norm_out, label='Normal Model', color='red', linestyle='--')
    ax2.set_title("With Outlier $x_7=120$")
    ax2.set_xlabel(r"$\theta$")
    ax2.legend()
    plt.tight_layout()
    plt.savefig('q7_comparison.png', dpi=300)
    plt.show()

    print(f"--- Credible Intervals ---")
    print(f"Original - Laplace CI: [{ci_lap_orig[0]:.2f}, {ci_lap_orig[1]:.2f}]")
    print(f"Original - Normal CI:  [{ci_norm_orig[0]:.2f}, {ci_norm_orig[1]:.2f}]")
    print(f"Outlier - Laplace CI:  [{ci_lap_out[0]:.2f}, {ci_lap_out[1]:.2f}]")
    print(f"Outlier - Normal CI:   [{ci_norm_out[0]:.2f}, {ci_norm_out[1]:.2f}]")

if __name__ == "__main__":
    x_orig = np.array([26.6, 38.5, 34.4, 34, 31, 23.6])
    x_outlier = np.append(x_orig, 120.0)
    
    plot_joint_posterior(x_orig)
    run_q7_analysis(x_orig, x_outlier)