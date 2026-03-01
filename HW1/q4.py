import numpy as np
from scipy.stats import t

# Reproducibility
np.random.seed(238238)

# Parameters
theta = 35.5
sigma = 5.5
alpha = 0.05
M = 100000

C = np.zeros(M)

for i in range(M):
    
    samples = []
    
    # Generate observations until X_n <= 25
    while True:
        x = np.random.normal(theta, sigma)
        samples.append(x)
        if x <= 25:
            break
    
    N = len(samples)
    xbar = np.mean(samples)
    
    # t critical value
    tcrit = t.ppf(1 - alpha/2, df=N-1)
    
    lower = xbar - sigma * tcrit / np.sqrt(N)
    upper = xbar + sigma * tcrit / np.sqrt(N)
    
    # Indicator
    C[i] = 1 if (lower <= theta <= upper) else 0

estimate = np.mean(C)

print("Estimated probability:", estimate)
print("1 - alpha:", 1 - alpha)