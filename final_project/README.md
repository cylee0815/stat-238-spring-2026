# Hierarchical Bayesian Regression for Sequential ETF Allocation

Final project for STAT 238 (Bayesian Statistics, UC Berkeley, Spring 2026).

The full specification lives in [`docs/proposal.tex`](docs/proposal.tex). The
working brief for the implementation is in [`PROMPT.md`](PROMPT.md).

## Install

The authoritative environment spec is `environment.yml` (conda / mamba):

```bash
conda env create -f environment.yml
conda activate stat238-bayes-rl
```

Python 3.11 is pinned. Most packages come from `conda-forge`; `yfinance` is
installed via the nested `pip:` section because its conda-forge build lags.

A `requirements.txt` is provided as a fallback for reviewers who do not use
conda:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce all figures

```bash
# 1. Download and cache raw price data
python -m project.data --download

# 2. Run notebooks 00 -> 10 in order, top-to-bottom
for nb in notebooks/*.ipynb; do
    jupyter nbconvert --to notebook --execute --inplace "$nb"
done

# 3. Tests
pytest tests/
```

Every figure in `figures/` and every metric table in `results/` is regenerated
by the steps above. Random seeds are fixed in `src/project/utils.py`.

## Layout

- `src/project/` — Python modules (data loader, environment, Gibbs sampler, etc.)
- `notebooks/` — one notebook per section of the proposal's notebook plan
- `tests/` — pytest unit tests; the synthetic-recovery test for the Gibbs
  sampler is non-negotiable
- `data/raw/`, `data/processed/`, `results/` — gitignored artifacts
- `report/`, `presentation/` — final write-up and slide deck
