# Section 4.3 — Nonconvex benchmarks

**Main text:** Table 2. **Supplement:** Tables S46–S66.

After [environment setup](../README.md#environment-setup), run from the repository root:

```bash
python reproduce.py tables
```

The main text table is in `outputs/tables/main_tables.tex`.
The Supplement tables are in `outputs/tables/nonconvex_tables_clean_v6.tex`.

To recompute this section's results with `partB_fixed_datasets_comparison.py`:

```bash
python scripts/run_experiments.py case3 --execute
```

Fresh results are saved under `runs/case3/`.
See the [root README](../README.md#run-all-experiments-from-scratch) to run and export all experiments.
