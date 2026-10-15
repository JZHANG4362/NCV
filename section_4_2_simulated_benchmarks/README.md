# Section 4.2 — Case 2: Benchmark Study Using Simulated Datasets

**Main manuscript:** Section 4.2, table `tab:main_case2_acc`.
**Supplement:** Section S8, 42 mean/median tables. Dataset design is in S5.1.

`partA_type2_grid_comparison.py` runs the 12 simulated settings for the three
clustering algorithms and seven external validity criteria. Canonical results
are in `data/`; the historical `nonconvex_` filename prefix belongs to this
simulated study and does not indicate Case 3.

From the **repository root**:

```bash
python reproduce.py tables
python scripts/run_experiments.py case2 --execute
```

The first command rebuilds all manuscript table fragments from saved results.
The second runs this section's experiments in an isolated directory. Its
corresponding outputs are `outputs/tables/case2_tables_clean_v6.tex` and the
Case 2 block in `outputs/tables/main_tables.tex`.

**Known gap:** Spectral / `k30_D200_sparse` has 39 saved replications instead
of 100. See the [reproduction notes](../README.md#reproduction-notes).
