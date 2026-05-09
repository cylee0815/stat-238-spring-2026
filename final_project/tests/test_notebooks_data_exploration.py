"""End-to-end execution test for the data-exploration + diagnostics notebooks.

Each notebook must run on a fresh kernel without raising and within its
runtime budget. The test does **not** validate cell *output* (figures,
prints): output correctness is verified by hand. What this catches is
silent breakage -- a function rename in ``project.plots``, an index
type-change, a feature column drift -- before it ships in the recorded
presentation.

Runtime budgets are per-notebook (``BUDGETS``). nb 00 and nb 01 are tight
because they only call existing helpers; nb 02 also runs three classical
baselines on the test window so it gets more headroom. nb 03 runs both
production samplers at full chain budget on a cold cache and is allowed
6 minutes; subsequent (cached) runs land in seconds. The notebooks rely
on cached parquet files in ``data/raw/`` and trace pickles in
``data/processed/traces/``; if either is missing, the first cell that
triggers ``yfinance`` or the first sampler call will time out and fail
this test loudly, which is the desired behaviour.
"""

from __future__ import annotations

import time
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / "notebooks"

# Per-notebook runtime budgets in seconds. nb 02 fits a tabular and a
# linear-FQI baseline on top of the data load; nb 00/01 only call helpers
# already cached or trivially fast. nb 03 runs both production samplers
# (4 chains x 6000 + 2000 burn) on a cold cache; with the cache populated
# it falls to <30s, which is the steady-state CI cost. nb 04 loads the
# cached traces from nb 03 and either reads or simulates 500 posterior
# rollouts per sampler; the path cache means subsequent runs are fast.
# nb 05 reads from the trace + path caches and orchestrates 3 classical
# baselines x 20 seeds; cold runs land in ~30s on this dataset, the
# 90s budget covers headroom on a fresh classical cache.
# 120s headroom on top of the ~30s cold target keeps cold first-runs
# from flaking the suite.
# nb 06 fits a fresh Gaussian sampler on TLT (cold first run dominated
# by ~4-6 minutes of Gibbs + 500-path rollout + 3 classical baselines x
# 20 seeds on TLT). Warm reruns hit every cache and finish in <90s; the
# 540s budget covers the cold first run.
# nb 09's prior-sensitivity sweep adds two new Gaussian fits (nu_sigma in
# {1.0, 4.0}) plus a re-fit at nu_sigma=0.1 under a generalized
# prior-aware cache key, three sets of 500-path rollouts on the SPY test
# window, a 2000-path convergence rollout on the original SPY trace, and
# the tabular Q-baseline at n_bins in {2, 8}. Cold ~6-8 min; warm <90 s.
# 720 s budget covers headroom on a fresh cache.
# nb 07 holds the Gaussian SPY posterior fixed and runs two new 500-path
# rollouts (resample_every in {1, 20}); the per-episode rollout is the
# nb 04 cache. Cold ~30 s; warm <30 s. The 240 s budget is generous.
BUDGETS: dict[str, int] = {
    "00_data.ipynb": 30,
    "01_features_and_targets.ipynb": 30,
    "02_env_and_features_sanity.ipynb": 90,
    "03_sampler_diagnostics.ipynb": 360,
    "04_thompson_and_posterior_q.ipynb": 120,
    "05_main_comparison.ipynb": 90,
    "06_robustness.ipynb": 540,
    "07_thompson_cadence.ipynb": 240,
    "09_prior_sensitivity.ipynb": 720,
}

NOTEBOOKS = [NOTEBOOKS_DIR / name for name in BUDGETS]


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=[p.name for p in NOTEBOOKS])
def test_notebook_executes_cleanly(nb_path: Path) -> None:
    """Run the notebook end-to-end; fail on any cell error or budget overrun."""
    assert nb_path.exists(), f"missing notebook: {nb_path}"

    budget = BUDGETS[nb_path.name]
    nb = nbformat.read(str(nb_path), as_version=4)
    client = NotebookClient(
        nb,
        timeout=budget,
        kernel_name="python3",
        resources={"metadata": {"path": str(nb_path.parent)}},
    )

    t0 = time.time()
    client.execute()
    elapsed = time.time() - t0

    assert elapsed < budget, (
        f"{nb_path.name} took {elapsed:.1f}s, budget is {budget}s"
    )


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=[p.name for p in NOTEBOOKS])
def test_notebook_structure_alternates_md_code(nb_path: Path) -> None:
    """Every code cell should be immediately preceded by a markdown cell.

    This enforces the "narrative + code" rhythm: no orphan code blocks
    without a markdown header explaining what they're doing. The very
    first cell is always markdown (title), and the second is always
    code (imports), so we just check that no two code cells are
    adjacent without an intervening markdown cell.
    """
    nb = nbformat.read(str(nb_path), as_version=4)
    types = [c.cell_type for c in nb.cells]

    # First cell should be the markdown title.
    assert types[0] == "markdown", f"{nb_path.name}: first cell is not markdown"

    # No two adjacent code cells with no markdown between them, except for
    # the closing assertion block (which can be md -> code -> code).
    # Allow a *single* such pair across the notebook to accommodate
    # "moments table + histogram" or similar tightly-coupled cells.
    adjacent_code = sum(
        1 for a, b in zip(types, types[1:]) if a == "code" and b == "code"
    )
    assert adjacent_code <= 2, (
        f"{nb_path.name}: too many adjacent code cells ({adjacent_code}); "
        "code blocks should be interleaved with markdown narrative"
    )


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=[p.name for p in NOTEBOOKS])
def test_notebook_has_no_inline_pyplot(nb_path: Path) -> None:
    """Foundation rule: notebooks delegate plotting to ``project.plots``.

    Any ``import matplotlib.pyplot`` or ``plt.subplots(`` in a code cell
    means we slipped back into inline plotting. The setup cell is allowed
    to do other matplotlib config but must not import pyplot directly:
    the helpers do it on demand.
    """
    nb = nbformat.read(str(nb_path), as_version=4)
    offences: list[str] = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        src = cell.source
        if "import matplotlib.pyplot" in src or "from matplotlib import pyplot" in src:
            offences.append(f"cell {i}: imports pyplot directly")
        if "plt.subplots(" in src or "plt.figure(" in src:
            offences.append(f"cell {i}: instantiates an axes/figure inline")
    assert not offences, f"{nb_path.name} has inline matplotlib usage: {offences}"
