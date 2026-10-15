# Section 4.2 — Simulated benchmarks

**Main text:** Table 1. **Supplement:** Tables S4–S45.

After [environment setup](../README.md#environment-setup), run from the repository root:

```bash
python reproduce.py tables
```

The main text table is in `outputs/tables/main_tables.tex`.
The Supplement tables are in `outputs/tables/case2_tables_clean_v6.tex`.

To recompute this section's results with `partA_type2_grid_comparison.py`:

```bash
python scripts/run_experiments.py case2 --execute
```

Fresh results are saved under `runs/case2/`.
See the [root README](../README.md#run-all-experiments-from-scratch) to run and export all experiments.
