# Manuscript-to-code map

Use labels rather than table/figure numbers, which can change as the manuscript
is edited. Paths below are relative to this repository.

| Manuscript location or label | Numerical source | Exporter / output |
|---|---|---|
| Case 1, `fig:scatter_acc` | `section_4_1_gaussian_simulation/gaussian_simulation.ipynb` and `gaussian_simulation_boxplot.ipynb` | `combined_boxplot_figure.ipynb` → `figures/gaussian_combined_landscape_boxplot.pdf` |
| S7, three additional Case 1 figures | `section_4_1_gaussian_simulation/gaussian_multimetric_landscape_figure.py` | Same script → `figures/gaussian_multimetric_{kmeans,spectral,agglomerative}_landscape.pdf` |
| `tab:main_case2_acc` | `section_4_2_simulated_benchmarks/partA_type2_grid_comparison.py`; ACC, KMeans, mean | `reproduce.py tables` → `tables/main_tables.tex` |
| S8, all Case 2 tables | Same Case 2 script; seven external criteria and three clustering methods | `tables/case2_tables_clean_v6.tex` |
| `tab:main_nonconvex_acc_spectral` | `section_4_3_nonconvex_benchmarks/partB_fixed_datasets_comparison.py`; ACC, Spectral | `tables/main_tables.tex` |
| S9, all Case 3 tables | Same Case 3 script | `tables/nonconvex_tables_clean_v6.tex` |
| `tab:main_labeled_acc_kmeans` | `section_4_4_labeled_benchmarks/Benchmark_data_real.py`; ACC, KMeans | `tables/main_tables.tex` |
| S10, all Case 4 tables | Same Case 4 script | `tables/realdata_tables_clean_v6.tex` |
| Section 4.5 and S11, tau / K-hat / buffer sensitivity | `section_4_5_sensitivity/{simulated_data,nonconvex_data,labeled_data}/rstar_selection_landscape.py` | `section_4_5_sensitivity/summary_plots/tau_khat_window_sensitivity.py` → three `figures/rstar_*_sensitivity.pdf` files |
| `tab:improvement_tuned_by_index` | Both tuned embedding packages, `manifold_tsne_umap_case{2,3,4}.py`, fixed-result caches | `reproduce.py tables` → `local_embedding_tables/tuned_by_index.tex` |
| `tab:improvement_oracle_by_index` | Same packages, parameter-sensitivity caches; maximum correlation over each candidate grid | `local_embedding_tables/oracle_by_index.tex` |
| S12, 31 per-dataset embedding plots | Same packages, fixed-result caches | Alternative-index package's `generate_manifold_bar_summary_tex.py` → `figures/local_embedding_bar_plots/case{2,3,4}/*.pdf` |

All output paths in the last column are under `outputs/` when using the reader
entry point. The embedding packages are:

- `section_4_6_local_embeddings/silhouette`: ordinary Silhouette, t-SNE
  Silhouette, UMAP Silhouette, and the rNN baseline.
- `section_4_6_local_embeddings/other_indices`: VRC, WG, WB, DBCV, CVDD,
  VIASCKDE, DCSI, and DB, each on original data, t-SNE, and UMAP.

The alternative-index summary imports the ordinary Silhouette results from the
first package. Both are required to recreate the nine-index embedding tables
and bar charts. CH is an alias of VRC in some saved dictionaries and is not an
additional reported method.

## Saved-result paths

| Study | Canonical cache pattern |
|---|---|
| Case 1 | `section_4_1_gaussian_simulation/data/gaussian_*.pkl` (three files) |
| Case 2 benchmark | `section_4_2_simulated_benchmarks/data/nonconvex_case_comparison_results_nonself_oriented{_METRIC}.pkl` |
| Case 3 benchmark | `section_4_3_nonconvex_benchmarks/data/nonconvex_fixed_results_paperkmax_buffer10_r130_nonself_oriented{_METRIC}.pkl` |
| Case 4 benchmark | `section_4_4_labeled_benchmarks/data/realdata_fixed_results_paperkmax_buffer10_r130_nonself_oriented{_METRIC}.pkl` |
| Sensitivity | `section_4_5_sensitivity/*/data/rstar_landscape_*.pkl` (28 files) |
| Tuned Silhouette embeddings | `section_4_6_local_embeddings/silhouette/data/caseN/manifold_{fixed_results,param_sensitivity_results,param_star_results}_tsne_umap_tuned_embedding_origkgrid_v1.pkl` |
| Other tuned indices | `section_4_6_local_embeddings/other_indices/data/caseN/manifold_{fixed_results,param_sensitivity_results,param_star_results}_tsne_umap_alt_indices_tuned_origkgrid_v1.pkl` |

ACC uses the base benchmark filename; the other external criteria append
`_Jaccard`, `_SS3`, `_ARI`, `_NMI`, `_Powers`, or `_NID`. The historical
`nonconvex_` prefix in the Case 2 filename does not mean it contains Case 3 data.
Additional r-sensitivity and selected-r caches for Cases 3 and 4 are included.

## Material authored in LaTeX

The theoretical interior/boundary illustration (`fig:interior_boundary`) is
drawn directly with TikZ in `main.tex`. The dataset-characteristic and software
tables in S5–S6 are descriptive tables authored in `supplement.tex`, not outputs
of the numerical experiments. Proofs in S1–S4 are also authored in LaTeX.
Those document sources are outside this code repository. Dataset loaders and
their expected dimension/class-count assertions are in the relevant experiment
scripts; the included raw data and [source notes](DATA.md) support those tables.

