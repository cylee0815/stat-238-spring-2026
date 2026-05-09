# Project: Hierarchical Bayesian Regression for Sequential ETF Allocation
# Course: STAT 238 (Graduate Bayesian Statistics, UC Berkeley), Spring 2026

## Your role
You are implementing a graduate-level Bayesian statistics final project end-to-end.
The complete specification is in `docs/proposal.tex` (also available as `docs/proposal.pdf`).
**Read it in full before writing any code.** Every modeling choice, every equation, and
every evaluation metric is fixed by that document. Do not improvise on the model;
implementation choices (libraries, code organization, plotting style) are yours.

## Hard requirements (from the proposal — non-negotiable)
1. The likelihood is on **Monte Carlo returns** y_t = sum_{k=0..H} gamma^k R_{t+k}.
   Do NOT use TD bootstrapped targets. This is the central methodological point.
2. The model is **hierarchical Bayesian regression**: beta_a | mu_beta, Sigma_beta ~ N(...),
   with an Inverse-Wishart hyperprior on Sigma_beta and Inverse-Gamma on sigma^2.
   Implement the **exact Gibbs sampler** with the full conditionals from Section 2.5
   of the proposal. No PyMC/Stan shortcuts for the headline model — write the
   sampler yourself so the Bayesian machinery is visible. (You may use PyMC for
   a cross-check in a separate notebook cell.)
3. Implement **both** the Gaussian likelihood and the Student-t scale-mixture
   likelihood. Diagnostics decide which is the headline result.
4. Features must be **strictly causal**. No look-ahead. Use a pre-1994 window for
   prior hyperparameter calibration; never touch it again.
5. Transaction cost is volatility-dependent: c_t = c_0 + lambda * vol_t.
6. Evaluation is **posterior predictive only**. Sharpe / MDD / Turnover are
   distributions, not points. Report mean +/- 95% credible interval.
7. Pseudo-regret vs best fixed action in hindsight. NO oracle-Q regret.
8. Two baselines: tabular epsilon-greedy Q-learning (C1) and linear fitted-Q with
   epsilon-greedy (C2). Both must be tuned on the training set.
9. Reproducibility: seeded RNG everywhere; one command rebuilds every figure.

## Deliverables
- `notebooks/00_data.ipynb` ... `notebooks/10_discussion.ipynb` — one notebook per
  section of the proposal's notebook plan (Section 7).
- `src/` — clean Python modules; notebooks import from here, no logic inline.
- `figures/` — every plot saved as PDF + PNG, named to match the proposal's
  Visualisation Plan (Section 9).
- `results/` — pickled posterior samples, metric tables as CSV.
- `tests/` — pytest unit tests for the Gibbs sampler (synthetic recovery),
  the environment (cost calculation), and the MC-return constructor (no leakage).
- `report/` — a final 6-8 page write-up that mirrors the proposal but reports
  empirical findings.

## Workflow expectations
- **Always run the relevant SKILL.md first** before file creation tasks.
- After each notebook, run all cells top-to-bottom and check it executes clean.
- Use `uv` or `pip` with a pinned `requirements.txt`. Python 3.11+.
- Commit logical units. Suggested commits: data loader, environment, MC targets,
  Gibbs sampler, Student-t branch, baselines, evaluation, plots, report.
- When you hit a modeling ambiguity, **stop and ask** — do not paper over it.

## What I want you to do first (in this order)
1. Read `docs/proposal.tex` cover to cover. Summarize back to me your
   understanding of the model, the data flow, and the evaluation plan.
   Flag anything that looks ambiguous or under-specified.
2. Propose the file structure under `src/` (module names, function signatures
   for the public API). Wait for approval before implementing.
3. Implement the data loader and feature pipeline (notebook 00 + module).
   Show me the output before moving on.
4. Implement the Gibbs sampler with synthetic-data recovery test BEFORE
   running it on real data. The recovery test is non-negotiable.
5. Then proceed through the remaining notebooks in order.

## Style
- Type-hint everything. Docstrings in NumPy format.
- Plot style: matplotlib, no seaborn defaults. Black text on white. No emoji.
- Prefer `numpy` + `scipy.stats` for the sampler. `pandas` for data wrangling.
  `arviz` for MCMC diagnostics. `yfinance` for data download.
- Performance: vectorize the Gibbs sweep over actions. Target M=5000, B=1000
  in under 60s on a laptop.