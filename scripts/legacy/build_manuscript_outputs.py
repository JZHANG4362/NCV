#!/usr/bin/env python3
"""Build manuscript-ready table inputs and collect generated PDF figures.

This is the ONE script that produces the actual files main.tex and
supplement.tex compile against: the three "_clean_v6.tex" LaTeX table
fragments (written to manuscript_outputs/tables/, and mirrored one level up
in the project's own tables/ folder) and the ten manuscript PDF figures
(copied/renamed into manuscript_outputs/figures/). The per-case
`generate_*_tables.py` scripts in section_4_2_simulated_benchmarks/section_4_3_nonconvex_benchmarks/section_4_4_labeled_benchmarks produce a separate,
differently-formatted "_all_tables.tex" dump for ad hoc inspection/QA; they
are NOT what the manuscript actually includes.

Run from any directory after the Case 1--4 and sensitivity experiments have
produced their result pickles (paths below are all resolved relative to this
script's own location via ROOT, not the working directory). The script reads
the canonical result pickles produced by the merge scripts. It uses rNN_Hard
as the manuscript's rNN-based Silhouette score and omits the soft-neighborhood
diagnostic variant (rNN_Soft) from the manuscript tables -- rNN_Soft is only
used internally as a robustness check.

Usage:
    python scripts/legacy/build_manuscript_outputs.py            # best-effort
    python scripts/legacy/build_manuscript_outputs.py --strict    # fail loudly if any
                                                          # required table or
                                                          # figure is missing
"""

from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "manuscript_outputs"
TABLE_OUT = OUT / "tables"
FIGURE_OUT = OUT / "figures"

# The 7 external validity criteria and 3 clustering methods swept across
# every Case 2--4 table. "ACC" result pickles use the base filename (no
# suffix); the other 6 metrics were computed in separate re-runs and use a
# "_{METRIC}" filename suffix (see metric_path below).
METRICS = ["ACC", "Jaccard", "SS3", "ARI", "NMI", "Powers", "NID"]
CLUSTERING = ["KMeans", "Spectral", "Agglomerative"]
CLUSTER_LABEL = {
    "KMeans": "K-means clustering",
    "Spectral": "spectral clustering",
    "Agglomerative": "agglomerative clustering",
}

# The 9 internal validity indices shown as table columns, in the exact
# left-to-right order used throughout the manuscript. rNN_Hard is the
# proposed rNN-based Silhouette score; rNN_Soft is intentionally excluded
# here even though it exists in every results pickle (it is only reported in
# the boxplot robustness figures in Case 1, not in the Case 2--4 tables).
METHODS = [
    "rNN_Hard", "Silhouette", "VRC", "WG", "WB", "DBCV", "CVDD",
    "VIASCKDE", "DCSI",
]
METHOD_LABELS = {
    "rNN_Hard": r"$r$NN", "Silhouette": "Silhouette", "VRC": "VRC",
    "WG": "WG", "WB": "WB", "DBCV": "DBCV", "CVDD": "CVDD",
    "VIASCKDE": "VIASCKDE", "DCSI": "DCSI",
}

# Case 2's 12 data-generating settings: every combination of true cluster
# count K*, ambient dimension D, and cluster compactness (compact = tight
# clusters, sigma=0.1; sparse = diffuse clusters, sigma=0.8). Each setting
# was independently replicated 100 times.
CASE2_CASES = [
    (k, d, compactness)
    for k in [6, 30]
    for d in [2, 10, 200]
    for compactness in ["compact", "sparse"]
]
# Case 3's ten fixed non-globular (SIPU) benchmark datasets, keyed by their
# lowercase result-pickle key, mapped to each dataset's true cluster count.
CASE3_K = {
    "pathbased": 3, "spiral": 3, "aggregation": 7, "atom": 2,
    "lsun": 3, "zelnik1": 3, "zelnik5": 4, "zelnik6": 3,
    "triangle1": 4, "longsquare": 6,
}
# Capitalized display names for the Case 3 datasets (the result-pickle keys
# above are lowercase, but the manuscript tables use title case).
CASE3_DISPLAY = {
    "pathbased": "Pathbased", "spiral": "Spiral",
    "aggregation": "Aggregation", "atom": "Atom", "lsun": "Lsun",
    "zelnik1": "Zelnik1", "zelnik5": "Zelnik5", "zelnik6": "Zelnik6",
    "triangle1": "Triangle1", "longsquare": "LongSquare",
}
# Case 4's nine labeled real-data benchmarks, mapped to their true class
# count (used as K* in the table's metadata column).
CASE4_K = {
    "Digits": 10, "MNIST-test": 10, "Ecoli": 8, "Letter": 26,
    "Iris": 3, "Fashion-MNIST-test": 10, "Wine": 3, "Libras": 15,
    "Vowel": 11,
}


def load_results(path: Path) -> dict | None:
    """Load a result pickle and return its 'results' sub-dict if present.

    Returns None (rather than raising) when the file does not exist, so
    callers can skip a metric/case gracefully and report exactly what is
    missing instead of crashing partway through table generation.
    """
    if not path.exists():
        return None
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return payload.get("results", payload)


def metric_path(case: int, metric: str) -> Path:
    """Resolve the result-pickle path for a given case and external metric.

    ACC pickles use the base filename; every other metric was produced by a
    separate re-run of the same experiment script and is saved with a
    "_{metric}" suffix. Case 2's filenames additionally carry a historical
    "nonconvex_" prefix (Case 2 is NOT the nonconvex case -- that is Case 3
    -- but the filename was never renamed and is preserved here verbatim so
    existing result pickles keep resolving correctly).
    """
    suffix = "" if metric == "ACC" else f"_{metric}"
    if case == 2:
        name = "nonconvex_case_comparison_results_nonself_oriented.pkl" if metric == "ACC" else (
            f"nonconvex_case_comparison_results_nonself_oriented_{metric}.pkl"
        )
        return ROOT / "section_4_2_simulated_benchmarks/data" / name
    if case == 3:
        return ROOT / "section_4_3_nonconvex_benchmarks/data" / (
            f"nonconvex_fixed_results_paperkmax_buffer10_r130_nonself_oriented{suffix}.pkl"
        )
    return ROOT / "section_4_4_labeled_benchmarks/data" / (
        f"realdata_fixed_results_paperkmax_buffer10_r130_nonself_oriented{suffix}.pkl"
    )


def format_values(values: list[float]) -> list[str]:
    """Format one table row's 9 method values as LaTeX cell strings.

    Missing/non-finite values (e.g. DBCV on Fashion-MNIST-test) render as
    the literal string "NA" rather than being silently dropped. The single
    highest value in the row is wrapped in \\textbf{} (bold) -- this is the
    "bold indicates the highest value in each row" convention referenced in
    every table caption. atol=5e-13 is deliberately far smaller than the
    3-decimal display precision; it only guards against genuine
    floating-point ties, not near-ties that would round to the same digits.
    """
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    maximum = float(np.max(finite)) if len(finite) else None
    cells = []
    for value in arr:
        if not np.isfinite(value):
            cells.append("NA")
        elif maximum is not None and np.isclose(value, maximum, atol=5e-13):
            cells.append(rf"\textbf{{{value:.3f}}}")
        else:
            cells.append(f"{value:.3f}")
    return cells


def compact_table(
    caption: str,
    label: str,
    metadata_headers: list[str],
    rows: list[tuple[list[str], list[float]]],
    width: str,
) -> str:
    """Render one landscape-oriented LaTeX table (one metric x one method).

    metadata_headers are the leading identifying columns (e.g. K*, D,
    Compactness for Case 2; Dataset, K* for Cases 3/4); the remaining
    columns are always the 9 METHODS in the fixed order defined above.
    Wrapped in \\resizebox so wide 9-column tables still fit the page.
    """
    headers = metadata_headers + [METHOD_LABELS[m] for m in METHODS]
    spec = "".join("c" for _ in metadata_headers) + f"*{{{len(METHODS)}}}{{r}}"
    lines = [
        r"\begin{center}", r"\scriptsize", r"\setlength{\tabcolsep}{2.8pt}",
        r"\renewcommand{\arraystretch}{0.92}",
        rf"\captionof{{table}}{{{caption}}}", rf"\label{{{label}}}",
        rf"\resizebox{{{width}}}{{!}}{{%", rf"\begin{{tabular}}{{{spec}}}",
        r"\hline", " & ".join(headers) + " \\\\", r"\hline",
    ]
    for metadata, values in rows:
        lines.append(" & ".join(metadata + format_values(values)) + " \\\\")
    lines.extend([r"\hline", r"\end{tabular}", "}", r"\end{center}"])
    return "\n".join(lines)


def case2_tables() -> str:
    """Build the full Case 2 LaTeX table document (case2_tables_clean_v6.tex).

    Iterates all 7 external metrics x 3 clustering methods x {mean, median}
    (42 tables total, one per combination), each a 12-row table (one row per
    section_4_2_simulated_benchmarks setting) aggregating each setting's 100 replications down to a
    single summary statistic per index column. Missing metric/method
    combinations degrade gracefully to an explanatory LaTeX comment rather
    than raising, so a partially-finished experiment still produces a
    compilable (if incomplete) document.
    """
    pieces = [
        "% Auto-generated by scripts/legacy/build_manuscript_outputs.py.",
        "% r is the number of non-self neighbors; rNN_Soft is intentionally omitted.",
    ]
    for metric in METRICS:
        results = load_results(metric_path(2, metric))
        if results is None:
            pieces.append(f"% Missing Case 2 results for {metric}.")
            continue
        for cm_index, cm in enumerate(CLUSTERING):
            if cm not in results:
                pieces.append(f"% Missing Case 2 results for {metric}/{cm}.")
                continue
            pieces.extend([r"\clearpage", r"\begin{landscape}"])
            if cm_index == 0:
                pieces.append(rf"\subsection{{{metric}}}")
            # Two tables per (metric, method): one summarized by the mean
            # across the 100 replications, one by the median (the median
            # table backs the supplement's separate median-based claim).
            for stat_name, stat_fn in [("mean", np.mean), ("median", np.median)]:
                rows = []
                for k, d, compactness in CASE2_CASES:
                    key = f"k{k}_D{d}_{compactness}"
                    by_method = results.get(cm, {}).get(key, {})
                    values = []
                    for method in METHODS:
                        vector = np.asarray(by_method.get(method, []), dtype=float)
                        vector = vector[np.isfinite(vector)]
                        values.append(float(stat_fn(vector)) if len(vector) else np.nan)
                    rows.append(([str(k), str(d), compactness], values))
                caption = (
                    f"{stat_name.capitalize()} Spearman rank correlation ($\\rho$) "
                    f"with {metric} across $N_{{\\mathrm{{rep}}}}=100$ replications, "
                    f"{CLUSTER_LABEL[cm]}, for the 12 settings in Case 2. Bold "
                    "indicates the highest value in each row."
                )
                pieces.append(compact_table(
                    caption, f"tab:case2_{metric}_{cm}_{stat_name}",
                    [r"$K^*$", r"$D$", "Compactness"], rows, r"0.83\linewidth",
                ))
            pieces.append(r"\end{landscape}")
    return "\n\n".join(pieces) + "\n"


def fixed_case_tables(case: int) -> str:
    """Build the Case 3 or Case 4 LaTeX table document (case must be 3 or 4).

    Both cases share this single implementation because they have the same
    shape: a fixed list of datasets (no replications to aggregate, unlike
    Case 2), each contributing exactly one row per (metric, method) table.
    Produces nonconvex_tables_clean_v6.tex for case=3 and
    realdata_tables_clean_v6.tex for case=4.
    """
    is_case3 = case == 3
    dataset_k = CASE3_K if is_case3 else CASE4_K
    filename_prefix = "nonconvex" if is_case3 else "realdata"
    case_label = "ten non-globular benchmark datasets" if is_case3 else (
        "nine labeled benchmark datasets"
    )
    pieces = [
        "% Auto-generated by scripts/legacy/build_manuscript_outputs.py.",
        "% r is the number of non-self neighbors; rNN_Soft is intentionally omitted.",
    ]
    for metric in METRICS:
        results = load_results(metric_path(case, metric))
        if results is None:
            pieces.append(f"% Missing Case {case} results for {metric}.")
            continue
        for cm_index, cm in enumerate(CLUSTERING):
            pieces.extend([r"\clearpage", r"\begin{landscape}"])
            if cm_index == 0:
                pieces.append(rf"\subsection{{{metric}}}")
            rows = []
            for dataset, k in dataset_k.items():
                by_method = results.get(cm, {}).get(dataset, {})
                values = [float(by_method.get(method, np.nan)) for method in METHODS]
                display = CASE3_DISPLAY[dataset] if is_case3 else dataset
                rows.append(([display, str(k)], values))
            caption = (
                f"Spearman rank correlation ($\\rho$) with {metric} for the "
                f"{case_label}, {CLUSTER_LABEL[cm]}. Bold indicates the highest "
                "value in each row."
            )
            pieces.append(compact_table(
                caption, f"tab:{filename_prefix}_{metric}_{cm}",
                ["Dataset", r"$K^*$"], rows,
                r"0.86\linewidth" if is_case3 else r"0.87\linewidth",
            ))
            pieces.append(r"\end{landscape}")
    return "\n\n".join(pieces) + "\n"


def write_tables() -> None:
    """Generate all 3 manuscript table documents and write them to disk.

    Writes into manuscript_outputs/tables/ only. The project's top-level
    tables/ folder (what supplement.tex actually \\input{}s) is a separate
    copy and is NOT updated automatically by this function -- after
    reviewing the output here, copy the three *_clean_v6.tex files to
    ../../tables/ by hand.
    """
    TABLE_OUT.mkdir(parents=True, exist_ok=True)
    outputs = {
        "case2_tables_clean_v6.tex": case2_tables(),
        "nonconvex_tables_clean_v6.tex": fixed_case_tables(3),
        "realdata_tables_clean_v6.tex": fixed_case_tables(4),
    }
    for name, text in outputs.items():
        path = TABLE_OUT / name
        path.write_text(text, encoding="utf-8")
        print(f"Wrote {path.relative_to(ROOT)}")


def collect_figures(strict: bool) -> None:
    """Copy the 7 upstream-generated PDF figures into manuscript_outputs/figures/,
    renaming each to the exact filename main.tex/supplement.tex reference.

    Case 1's 4 figures come straight from its notebooks/script; the 3
    sensitivity figures are the "hard" (not "soft") neighborhood variant
    from section_4_5_sensitivity/summary_plots/, matching what the
    manuscript actually shows. With strict=True, any missing upstream file
    raises SystemExit after reporting everything that's missing (not just
    the first one), so a single run tells you the full remaining to-do list.
    """
    FIGURE_OUT.mkdir(parents=True, exist_ok=True)
    mapping = {
        ROOT / "section_4_1_gaussian_simulation/figures/combined/gaussian_combined_landscape_boxplot.pdf":
            "gaussian_combined_landscape_boxplot.pdf",
        ROOT / "section_4_1_gaussian_simulation/figures/combined/gaussian_multimetric_kmeans_landscape.pdf":
            "gaussian_multimetric_kmeans_landscape.pdf",
        ROOT / "section_4_1_gaussian_simulation/figures/combined/gaussian_multimetric_spectral_landscape.pdf":
            "gaussian_multimetric_spectral_landscape.pdf",
        ROOT / "section_4_1_gaussian_simulation/figures/combined/gaussian_multimetric_agglomerative_landscape.pdf":
            "gaussian_multimetric_agglomerative_landscape.pdf",
        ROOT / "section_4_5_sensitivity/summary_plots/figures/tau_sensitivity_hard.pdf":
            "rstar_tau_sensitivity.pdf",
        ROOT / "section_4_5_sensitivity/summary_plots/figures/khat_sensitivity_hard.pdf":
            "rstar_khat_sensitivity.pdf",
        ROOT / "section_4_5_sensitivity/summary_plots/figures/b_sensitivity_hard.pdf":
            "rstar_b_sensitivity.pdf",
    }
    missing = []
    for source, target_name in mapping.items():
        if source.exists():
            target = FIGURE_OUT / target_name
            shutil.copy2(source, target)
            print(f"Copied {target.relative_to(ROOT)}")
        else:
            missing.append(source)
            print(f"Missing upstream figure: {source.relative_to(ROOT)}")
    if strict and missing:
        raise SystemExit(f"{len(missing)} required figure(s) are missing")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict", action="store_true",
        help="fail if any required PDF figure has not yet been generated",
    )
    args = parser.parse_args()
    write_tables()
    collect_figures(args.strict)


if __name__ == "__main__":
    main()
