"""Hierarchical Bayesian regression for sequential ETF allocation.

Public modules:
    data        -- price data download and caching
    features    -- strictly causal rolling features
    env         -- TradingEnv with volatility-dependent transaction cost
    targets     -- Monte Carlo return construction
    sampler     -- hierarchical Gibbs sampler (Gaussian likelihood)
    sampler_t   -- hierarchical Gibbs sampler (Student-t scale-mixture)
    baselines   -- random benchmark, tabular MC-regression Q baseline,
                   linear FQI baseline, and the multi-seed orchestrator
                   `classical_baseline_distribution`
    thompson    -- Thompson-sampling decision rule and policy iteration
    rollout     -- per-episode / per-step posterior-sampled rollouts
    eval        -- posterior predictive metrics
    plots       -- paper-quality figures (foundation set; expanded by
                   later increments paired with their consuming notebooks)
    utils       -- seeds, IO helpers
"""

from project import plots  # noqa: F401  -- ensure import-side effects + namespace exposure

__version__ = "0.1.0"
