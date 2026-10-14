#!/usr/bin/env python3
"""Verify that r counts non-self neighbors in every packaged implementation.

Throughout this package, r is defined as the number of OTHER observations in
a point's neighborhood (self excluded). scikit-learn's kneighbors(X) always
puts the query point itself in column 0 of its result, so every correct
implementation here must request r+1 neighbors and then drop that first
column. This script does NOT re-run any experiment; it is a fast, static +
numerical sanity check that catches the single most consequential
implementation bug this package guards against (silently counting the
self-match as one of the r neighbors, which shifts every r value by one and
is easy to introduce accidentally when refactoring).

Run this before every full experiment, locally and again after transferring
to a compute cluster:

    python scripts/legacy/verify_r_convention.py

It must print PASS. Any AssertionError means some file's rNN implementation
has drifted from the shared convention and must be fixed before trusting any
numbers it produces.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[2]

# Every "standard" (non-Case-1) experiment script shares the identical
# pattern: query r+1 neighbors, then slice away column 0 (self). Listed here
# so assert_static_convention can grep each one's source text directly,
# without importing/running the (heavy, SLURM-oriented) scripts themselves.
STANDARD_FILES = [
    ROOT / "section_4_2_simulated_benchmarks/partA_type2_grid_comparison.py",
    ROOT / "section_4_3_nonconvex_benchmarks/partB_fixed_datasets_comparison.py",
    ROOT / "section_4_4_labeled_benchmarks/Benchmark_data_real.py",
    ROOT / "section_4_5_sensitivity/simulated_data/rstar_selection_landscape.py",
    ROOT / "section_4_5_sensitivity/nonconvex_data/rstar_selection_landscape.py",
    ROOT / "section_4_5_sensitivity/labeled_data/rstar_selection_landscape.py",
]

# Case 1's three notebooks reuse the same fixed-r (5/30/100) rNN affinity
# helper as the standard scripts, so they are checked with the identical
# literal-string test below, just parsed out of notebook JSON instead of a
# plain .py file.
NOTEBOOKS = [
    ROOT / "section_4_1_gaussian_simulation/gaussian_simulation.ipynb",
    ROOT / "section_4_1_gaussian_simulation/gaussian_simulation_boxplot.ipynb",
    ROOT / "section_4_1_gaussian_simulation/combined_boxplot_figure.ipynb",
]


def assert_static_convention() -> None:
    """Grep every implementation's source text for the two literal
    substrings that together prove it drops the self-match: the
    n_neighbors=r+1 query, and the [:, 1:] slice that discards column 0.
    A pure text search (rather than importing and introspecting the code)
    keeps this check fast and side-effect-free, and works uniformly across
    plain .py scripts and .ipynb notebooks.
    """
    for path in STANDARD_FILES:
        source = path.read_text(encoding="utf-8")
        assert "n_neighbors=r + 1" in source, path
        assert "dists[:, 1:], idx[:, 1:]" in source, path

    # Case 1's multimetric-landscape script queries neighbors ONCE up to the
    # largest r in R_VALUES (for efficiency, since it needs many r values
    # from the same fitted NearestNeighbors object), then slices out exactly
    # r non-self neighbors per r value -- a different code shape from the
    # STANDARD_FILES pattern above, so it gets its own pair of literal
    # substrings to check.
    multimetric = (
        ROOT / "section_4_1_gaussian_simulation/gaussian_multimetric_landscape_figure.py"
    ).read_text(encoding="utf-8")
    assert "n_neighbors=max(R_VALUES) + 1" in multimetric
    assert "idx_max[:, 1:r + 1]" in multimetric

    for path in NOTEBOOKS:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
        )
        assert "n_neighbors=r + 1" in source, path
        assert "dists[:, 1:], idx[:, 1:]" in source, path


def assert_numerical_convention() -> None:
    """Independent of any packaged source code: directly exercise
    scikit-learn's NearestNeighbors on a tiny, deliberately well-separated
    1-D point set (so every neighbor relationship is unambiguous), and
    confirm that requesting r+1 neighbors and dropping column 0 yields
    exactly r neighbors per point, none of which is the point itself. This
    guards against the convention being correct in every packaged script
    (checked above) but scikit-learn's own kneighbors() behavior having
    silently changed in a way that would break the shared assumption.
    """
    # Points are spaced so each nearest-neighbor identity is unambiguous.
    x = np.array([[0.0], [1.0], [3.0], [7.0], [12.0], [20.0]])
    r = 3
    nbrs = NearestNeighbors(n_neighbors=r + 1).fit(x)
    _, idx = nbrs.kneighbors(x)
    nonself = idx[:, 1:]
    assert nonself.shape == (len(x), r)
    assert all(i not in row for i, row in enumerate(nonself))
    assert all(len(set(row)) == r for row in nonself)


if __name__ == "__main__":
    assert_static_convention()
    assert_numerical_convention()
    print("PASS: every packaged rNN implementation uses exactly r non-self neighbors.")

