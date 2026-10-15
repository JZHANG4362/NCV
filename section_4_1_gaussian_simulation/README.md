# Section 4.1 — Gaussian simulation

**Main text:** Figure 2. **Supplement:** Figures S1–S3.

After [environment setup](../README.md#environment-setup), run from the repository root
to rerun only this section from scratch:

```bash
python scripts/run_experiments.py case1 --execute
```

This generates new simulation results, runs the 100-replication study, and
redraws all four figures. The supplied result caches are not copied into the run.

Find the PDFs in `runs/case1/section_4_1_gaussian_simulation/figures/combined/`
and the new numerical results in `runs/case1/section_4_1_gaussian_simulation/data/`:

| Figure | PDF |
|---|---|
| Main text Figure 2 | `gaussian_combined_landscape_boxplot.pdf` |
| Supplement Figure S1 — K-means | `gaussian_multimetric_kmeans_landscape.pdf` |
| Supplement Figure S2 — spectral clustering | `gaussian_multimetric_spectral_landscape.pdf` |
| Supplement Figure S3 — agglomerative clustering | `gaussian_multimetric_agglomerative_landscape.pdf` |

The command runs `gaussian_simulation.ipynb`, `gaussian_simulation_boxplot.ipynb`,
`combined_boxplot_figure.ipynb`, and `gaussian_multimetric_landscape_figure.py` in order.
For another fresh run, add `--workdir runs/case1_new` with a new directory name.

To reproduce outputs using the supplied results, follow the
[cached-output instructions](../README.md#reproduce-figures-and-tables-from-saved-results).
