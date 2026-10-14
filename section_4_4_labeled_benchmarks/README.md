# Section 4.4 — Case 4: Benchmarks with Labeled Data

**Main manuscript:** Section 4.4, table `tab:main_labeled_acc_kmeans`.
**Supplement:** Section S10, 21 result tables. Dataset characteristics are in S5.3.

`Benchmark_data_real.py` evaluates nine labeled datasets under three clustering
algorithms and seven external criteria. `data/` holds saved results.
`data/raw_datasets/` contains downloaded inputs shared with Sections 4.5 and
4.6; Digits, Iris, and Wine come from scikit-learn.

From the **repository root**, use `python reproduce.py tables` to rebuild the
tables, or `python scripts/run_experiments.py case4 --execute` for an isolated
fresh run. Large datasets can require substantial time and memory.

The section's outputs are `outputs/tables/realdata_tables_clean_v6.tex` and
the Case 4 block in `outputs/tables/main_tables.tex`. DBCV on original
Fashion-MNIST-test remains unavailable (`NA`). Historical merge and inspection
helpers are retained; use `reproduce.py` for current formatting.
See the [output map](../docs/OUTPUT_MAP.md) and [dataset sources](../docs/DATA.md).

