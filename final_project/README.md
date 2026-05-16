# Hierarchical Bayesian Regression for Sequential ETF Allocation

Final project for STAT 238 (Bayesian Statistics, UC Berkeley, Spring 2026).

[![Launch in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cylee0815/stat-238-spring-2026/main?urlpath=lab/tree/final_project/notebooks/00_data.ipynb)
[![Website](https://img.shields.io/badge/website-live-2ea44f)](https://cylee0815.github.io/stat-238-spring-2026/)

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
# 1. Download and cache raw price data (SPY, TLT, ^GSPC pre-train window)
python -c "from project import data; data.load_spy(); data.load_tlt(); data.load_pretrain()"

# 2. Run notebooks 00 -> 10 in order, top-to-bottom
for nb in notebooks/*.ipynb; do
    jupyter nbconvert --to notebook --execute --inplace "$nb"
done

# 3. Tests
pytest tests/
```

Every figure in `figures/` and every metric table in `results/` is regenerated
by the steps above. Random seeds are fixed in `src/project/utils.py`.

## Reproducibility

Two complementary paths, no local setup required:

- **Browse the results** — the [project website](https://cylee0815.github.io/stat-238-spring-2026/)
  is a MyST build of notebooks 00–10 rendering their *committed* outputs.
  It is rebuilt by `.github/workflows/deploy-site.yml` on every push to
  `main`; nothing is re-executed, so the published numbers always match the
  reviewed commit.
- **Re-run the analysis** — the Binder badge above launches a JupyterLab
  session in the exact pinned environment (`binder/environment.yml`, which
  mirrors `environment.yml` plus an editable install of the `project`
  package). Open `notebooks/00_data.ipynb` and *Restart Kernel & Run All*,
  then proceed 01 → 10 in order.

The Binder image and `environment.yml` are kept in lockstep: edit one, edit
the other. One-time repo setup for the website: **Settings → Pages → Source =
GitHub Actions**.

## Layout

- `src/project/` — Python modules (data loader, environment, Gibbs sampler, etc.)
- `notebooks/` — one notebook per section of the proposal's notebook plan
- `tests/` — pytest unit tests; the synthetic-recovery test for the Gibbs
  sampler is non-negotiable
- `data/raw/`, `data/processed/`, `results/` — gitignored artifacts
- `report/`, `presentation/` — final write-up and slide deck
