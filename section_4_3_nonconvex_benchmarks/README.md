# Section 4.3 — Case 3: Benchmarks with Non-Globular Cluster Structure

**Main manuscript:** Section 4.3, table `tab:main_nonconvex_acc_spectral`.
**Supplement:** Section S9, 21 result tables. Dataset characteristics are in S5.2.

`partB_fixed_datasets_comparison.py` evaluates the ten fixed nonconvex datasets
under three clustering algorithms and seven external criteria. `data/` holds
saved results; `data/raw_datasets/` contains the ten ARFF files, also shared by
Sections 4.5 and 4.6.

From the **repository root**, run `python reproduce.py tables` for the current
table fragments, or `python scripts/run_experiments.py case3 --execute` for a
fresh run in a separate directory.

The section's outputs are `outputs/tables/nonconvex_tables_clean_v6.tex` and
the Case 3 block in `outputs/tables/main_tables.tex`. The historical
`generate_nonconvex_tables.py` creates a separate inspection layout.
See the [output map](../docs/OUTPUT_MAP.md) and [dataset sources](../docs/DATA.md).

