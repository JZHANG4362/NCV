# Section 4.1 — Case 1: Mechanism-Focused Gaussian Simulation

**Main manuscript:** Section 4.1, figure `fig:scatter_acc`.
**Supplement:** Section S7, three additional external-criterion figures.

| Script/notebook | Purpose |
|---|---|
| `gaussian_simulation.ipynb` | Representative Gaussian dataset and fixed-neighborhood results |
| `gaussian_simulation_boxplot.ipynb` | 100-replication improvement study |
| `combined_boxplot_figure.ipynb` | Main-text combined scatter/boxplot figure |
| `gaussian_multimetric_landscape_figure.py` | Three supplementary figures for K-means, spectral, and agglomerative clustering |

The three saved numerical result files are in `data/`.
From the **repository root**, use `python reproduce.py cached` to rebuild the
manuscript outputs, or `python scripts/run_experiments.py case1 --execute` for
fresh computation in a separate run directory.

Outputs include `outputs/figures/gaussian_combined_landscape_boxplot.pdf` and
the three `outputs/figures/gaussian_multimetric_*_landscape.pdf` files.
See the [full output map](../docs/OUTPUT_MAP.md).

