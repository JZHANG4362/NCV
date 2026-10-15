# Section 4.4 — Labeled benchmarks

**Main text:** Table 3. **Supplement:** Tables S67–S87.

After [environment setup](../README.md#environment-setup), run from the repository root:

```bash
python reproduce.py tables
```

The main text table is in `outputs/tables/main_tables.tex`.
The Supplement tables are in `outputs/tables/realdata_tables_clean_v6.tex`.

To recompute this section's results with `Benchmark_data_real.py`:

```bash
python scripts/run_experiments.py case4 --execute
```

Fresh results are saved under `runs/case4/`.
See the [root README](../README.md#run-all-experiments-from-scratch) to run and export all experiments.
