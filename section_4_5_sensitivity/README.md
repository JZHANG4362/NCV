# Section 4.5 — Sensitivity Analysis

**Main manuscript:** Section 4.5.
**Supplement:** Section S11, sensitivity to the threshold, cluster-count bound,
and diagnostic-window buffer.

| Subfolder | Purpose |
|---|---|
| `simulated_data/` | 12 Case 2 settings, 30 replications each |
| `nonconvex_data/` | Ten Case 3 nonconvex datasets |
| `labeled_data/` | Six smaller Case 4 labeled datasets |
| `summary_plots/` | Combines the saved results into the three S11 figures |

Each dataset subfolder has `rstar_selection_landscape.py` and saved results in
`data/`. These experiments use K-means and ACC. The summary script is
`summary_plots/tau_khat_window_sensitivity.py`.

From the **repository root**, use `python reproduce.py cached` for the manuscript
figures, or `python scripts/run_experiments.py sensitivity --execute` for fresh
computation in a separate directory.

The three manuscript outputs are `outputs/figures/rstar_tau_sensitivity.pdf`,
`rstar_khat_sensitivity.pdf`, and `rstar_b_sensitivity.pdf`. This study has its
own replication design, distinct from the main benchmark in Section 4.2.

