# Section 4.5 — Sensitivity analysis

**Supplement:** Figures S4–S6, discussed in main text Section 4.5.

After [environment setup](../README.md#environment-setup), run from the repository root:

```bash
python reproduce.py cached
```

Find the PDFs in `outputs/figures/`:

| Supplement figure | PDF |
|---|---|
| Figure S4 — threshold | `rstar_tau_sensitivity.pdf` |
| Figure S5 — cluster-count bound | `rstar_khat_sensitivity.pdf` |
| Figure S6 — diagnostic-window buffer | `rstar_b_sensitivity.pdf` |

`summary_plots/tau_khat_window_sensitivity.py` draws these figures from the
saved results in `simulated_data/`, `nonconvex_data/`, and `labeled_data/`.
To rerun the experiments and plotting:

```bash
python scripts/run_experiments.py sensitivity --execute
```

Fresh results and plots are saved under `runs/sensitivity/`.
