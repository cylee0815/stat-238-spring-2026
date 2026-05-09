"""Seeds, IO helpers, project paths."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
FIGURES_DIR: Path = PROJECT_ROOT / "figures"

SEED: int = 238238


def set_seed(seed: int = SEED) -> np.random.Generator:
    """Seed Python and NumPy global RNGs and return a fresh ``np.random.Generator``.

    Parameters
    ----------
    seed
        Integer seed. Defaults to the project-wide ``SEED``.

    Returns
    -------
    numpy.random.Generator
        A fresh PCG64 generator seeded from ``seed``. Use this rather than
        legacy global ``np.random.*`` calls in new code.
    """
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def ensure_dirs() -> None:
    """Create runtime output directories if they do not exist."""
    for d in (RAW_DIR, PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
