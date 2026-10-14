#!/usr/bin/env python3
"""Verify the larger-is-better convention for NID and WB correlations.

Every validity index and external criterion in this package is meant to be
oriented so that a LARGER Spearman rank correlation always means STRONGER
agreement with the reference partition -- this lets every table cell,
"highest value in row" bolding, and aggregate win-count be compared on a
single consistent scale. Two quantities are natively the opposite way round
and must be flipped before use:

  * NID (Normalized Information Distance) is an external DISTANCE: 0 means
    identical partitions, larger means less similar. Every Case 2--4 script
    reverses it (or, in Case 1's landscape script, uses 1 - NID) before
    correlating with anything.
  * WB is an internal index that is MINIMIZED by construction (unlike every
    other internal index in this package, which is maximized). Every
    Case 2--4 script negates it (raw_curves['WB'] = -np.array([...])) before
    correlating with anything.

This script does NOT re-run any experiment; it is a fast, static + numerical
sanity check that both reversals are actually present in the code. Run it
before every full experiment, locally and again after transferring to a
compute cluster:

    python scripts/legacy/verify_score_directions.py

It must print PASS. Any AssertionError means some file's score-direction
handling has drifted and must be fixed before trusting any correlation,
ranking, or "exceeds Silhouette" count it produces.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]

# The three Case 2--4 experiment scripts all share the same score-direction
# machinery (an explicit LOWER_IS_BETTER_EXTERNAL_METRICS set, an
# orient_external_values() helper, and an explicit WB negation) -- checked
# here by grepping each script's source text for those exact substrings.
EXPERIMENT_FILES = [
    ROOT / "section_4_2_simulated_benchmarks/partA_type2_grid_comparison.py",
    ROOT / "section_4_3_nonconvex_benchmarks/partB_fixed_datasets_comparison.py",
    ROOT / "section_4_4_labeled_benchmarks/Benchmark_data_real.py",
]


def assert_static_convention() -> None:
    for path in EXPERIMENT_FILES:
        source = path.read_text(encoding="utf-8")
        assert "LOWER_IS_BETTER_EXTERNAL_METRICS = {'NID'}" in source, path
        assert "def orient_external_values(metric, values):" in source, path
        assert "raw_curves['WB'] = -np.array([" in source, path
        assert "'score_direction_convention': SCORE_DIRECTION_CONVENTION" in source, path
        assert "nonself_oriented" in source, path

    case1 = (
        ROOT / "section_4_1_gaussian_simulation/gaussian_multimetric_landscape_figure.py"
    ).read_text(encoding="utf-8")
    assert 'return 1.0 - values if metric == "NID" else values' in case1
    assert "nonself_oriented" in case1


def assert_numerical_convention() -> None:
    external_quality = np.array([0.1, 0.3, 0.6, 0.9])
    nid_distance = 1.0 - external_quality
    wb_raw = np.array([9.0, 6.0, 3.0, 1.0])

    assert np.isclose(spearmanr(external_quality, nid_distance).statistic, -1.0)
    assert np.isclose(spearmanr(external_quality, -nid_distance).statistic, 1.0)
    assert np.isclose(spearmanr(wb_raw, external_quality).statistic, -1.0)
    assert np.isclose(spearmanr(-wb_raw, external_quality).statistic, 1.0)


if __name__ == "__main__":
    assert_static_convention()
    assert_numerical_convention()
    print(
        "PASS: NID and WB are oriented before correlation so larger rho "
        "consistently means stronger agreement."
    )
