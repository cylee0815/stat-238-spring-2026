import numpy as np
from scipy.stats import norm, t

# Reproducibility
np.random.seed(238238)

# 1. Constants
theta_true = 35.5
sigma = 5.5
alpha = 0.05
M = 100000
threshold = 25

# t-critical value depends on N, so we calculate it inside the loop
# However, if N=1, t_{0, alpha/2} is technically undefined. 
# For simulation, we handle N > 1.

C = []

for _ in range(M):
    samples = []
    while True:
        x = np.random.normal(theta_true, sigma)
        samples.append(x)
        if x <= threshold:
            break
    
    N = len(samples)
    x_bar = np.mean(samples)
    
    # Probability is not well-defined for N=1 in the t-dist context
    # but the logic of the stopping rule N=inf{n: Xn <= 25} 
    # allows N=1. We use the survival function for the t-stat.
    if N > 1:
        t_crit = t.ppf(1 - alpha/2, df=N-1)
        margin = sigma * (t_crit / np.sqrt(N))
        
        if (x_bar - margin) <= theta_true <= (x_bar + margin):
            C.append(1)
        else:
            C.append(0)
    else:
        # For N=1, the interval is technically undefined/infinite
        # In many stopping rule contexts, we treat N=1 separately 
        # or observe that the frequentist coverage fails.
        C.append(0) 

est_prob = np.mean(C)
print(f"Estimated Probability: {est_prob:.5f}")