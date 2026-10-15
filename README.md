# NCV: reproducing the numerical studies

Code and saved numerical results for **A Nearest-Neighbor Silhouette for Ranking
Clusterings Across Granularities**.

## Find the code for a manuscript section

| Main manuscript | Supplement | Folder |
|---|---|---|
| **4.1 — Case 1: Gaussian simulation** | **S7** | [section_4_1_gaussian_simulation/](section_4_1_gaussian_simulation/) |
| **4.2 — Case 2: Simulated benchmarks** | **S8** | [section_4_2_simulated_benchmarks/](section_4_2_simulated_benchmarks/) |
| **4.3 — Case 3: Nonconvex benchmarks** | **S9** | [section_4_3_nonconvex_benchmarks/](section_4_3_nonconvex_benchmarks/) |
| **4.4 — Case 4: Labeled benchmarks** | **S10** | [section_4_4_labeled_benchmarks/](section_4_4_labeled_benchmarks/) |
| **4.5 — Sensitivity analysis** | **S11** | [section_4_5_sensitivity/](section_4_5_sensitivity/) |
| **4.6 — Local embeddings and internal validity indices** | **S12** | [section_4_6_local_embeddings/](section_4_6_local_embeddings/) |

Each section folder has a README identifying its scripts and outputs.
Section 4.5 has separate data-study and summary-plot subfolders. Section 4.6
contains `silhouette/` and `other_indices/`; both are needed for its summaries.
See the [output map](#output-map) below for individual table and figure labels.

## Set up

Run all commands from the repository root. Create the tested Python 3.12 environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`. Alternatively, use
`conda env create -f environment.yml`, then `conda activate ncv-reproduction`.

## Reproduce tables and figures from saved results

```bash
python reproduce.py tables
python reproduce.py cached
```

`tables` calculates the 89 empirical tables from the saved numerical results.
`cached` also redraws all 38 empirical figures. It recalculates one representative
Case 1 selected-neighborhood panel; it does not repeat the full simulation study.

Outputs are written to `outputs/` as LaTeX tables and PDF figures. Use
`--output PATH` to choose a different destination. The Python exporter computes
the table values and applies the manuscript formatting directly.

## Run the numerical experiments afresh

Start with the small Iris example, then inspect or execute the full job plan:

```bash
python scripts/run_experiments.py smoke --execute
python scripts/run_experiments.py all
python scripts/run_experiments.py all --execute
```

The runner copies code and raw data into `runs/GROUP/` and computes fresh results
there, preserving the supplied results. Groups are `case1`, `case2`, `case3`,
`case4`, `sensitivity`, `embeddings`, `all`, and `smoke`. Full runs may require
substantial time and memory. The `all` command without `--execute` only prints
the job plan; `smoke` runs K-means/Iris through the benchmark and both embedding
pipelines. Its partial results cannot generate all manuscript tables.

To choose a run directory, resume it, and export a completed full run:

```bash
python scripts/run_experiments.py all --execute --workdir runs/full
python scripts/run_experiments.py all --execute --resume --workdir runs/full
python reproduce.py cached --results-root runs/full --output outputs/full
```

Use the same group and directory when resuming. Completed jobs are skipped;
an interrupted job restarts from its beginning. The run retains its original
code copy. Progress is recorded in `RUN_STATE.json`, with numbered logs in
`runs/full/logs/`. Jobs execute sequentially to avoid shared-file collisions.
Full fresh runs were not performed during packaging.

## Reproduction notes

- The saved Case 2 Spectral / `k30_D200_sparse` results have 39 replications,
  although the manuscript specifies 100. A fresh 100-replication run will
  generally change those entries.
- Original-data DBCV is unavailable for Fashion-MNIST-test. Its table entries
  are `NA`, and embedding improvement counts omit that unavailable comparison.
- The dependency pins describe the tested packaging environment. The original
  Python 3.11 experiment environment was not fully pinned. In a fresh Iris
  check, benchmark correlations matched the saved values, but fitted embedding
  correlations differed. Exact fresh reproduction of embedding results is not
  established; the supplied numerical results reproduce the reported values.

See [data and attribution](#data-and-attribution) below and [NOTICE.md](NOTICE.md)
for software and third-party license status.

## Output map

Paths are relative to the repository. Output paths in the last column are
under `outputs/`. Table and figure labels identify the corresponding manuscript
items even when their displayed numbers change.

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

Plotting logs are under `outputs/logs/`. Intermediate figures are written beside
the plotting scripts and ignored by Git. Fonts and library versions can affect
the appearance of redrawn figures.

Generated `.tex` files are fragments requiring the manuscript's LaTeX preamble.
The three main benchmark tables are embedded in `main.tex`; replace those table
environments with the matching generated blocks if updating the paper.
No manuscript source is changed automatically.

The theoretical TikZ illustration (`fig:interior_boundary`), proofs in S1–S4,
and descriptive dataset/software tables in S5–S6 are authored in the manuscript
LaTeX sources and are outside this numerical-code repository.

## Numerical and formatting conventions

- `r` counts non-self neighbors; the distance graph uses the symmetric union.
- `rNN_Hard` is rNN-Sil. `rNN_Soft` is a diagnostic variant used in the Case 1
  boxplot and omitted from benchmark tables.
- Score directions are handled before correlation so larger correlations
  consistently mean stronger agreement, including NID, WB, and DB.
- Case 2 means and medians use finite saved correlations. Missing replications
  are not padded or imputed.
- Benchmark tables use `Sil` and `rNN-Sil`, left alignment, and vertical rules
  around those columns. Only rNN-Sil is bold when its displayed value exceeds
  Sil. A superscript `\star` marks the largest available displayed value per row.
- Embedding improvement compares an index on an embedding with the same index
  on original data. Tuned values use the selection algorithm; oracle values
  use the maximum correlation over the candidate grid for each dataset.
- Overall percentages pool counts across Cases 2–4. Denominators are 31, or 30
  for DBCV because the original-data Fashion-MNIST-test value is unavailable.
  Integer percentages retain manuscript rounding.

Use `reproduce.py` for manuscript tables and figures. Some embedding plotting
helpers also expose older standalone exports; they remain because the current
bar-plot script imports their loading functions, dataset labels, and colors.

## Data and attribution

Synthetic data are generated by the scripts with their recorded seeds. Case 2
uses Gaussian blobs with the stated dimensions, cluster counts, and compactness
settings. Fixed raw files are bundled once in `section_4_3_nonconvex_benchmarks/data/raw_datasets/`
and `section_4_4_labeled_benchmarks/data/raw_datasets/`. The embedding scripts share these folders.

### Nonconvex benchmarks (Case 3)

The ten ARFF files preserve their original headers and citations:

| Files | Attribution recorded in the files |
|---|---|
| `pathbased.arff`, `spiral.arff` | Chang and Yeung (2008), *Robust path-based spectral clustering*; [SIPU dataset collection](http://cs.joensuu.fi/sipu/datasets/) |
| `aggregation.arff` | Gionis, Mannila, and Tsaparas (2007), *Clustering aggregation*; SIPU collection |
| `atom.arff`, `lsun.arff` | FCPS; Alfred Ultsch (2005), *Clustering with SOM: U*C*, Workshop on Self Organizing Feature Maps, pp. 31–37. The file headers request this citation when using the datasets. |
| `zelnik1.arff`, `zelnik5.arff`, `zelnik6.arff` | Zelnik-Manor and Perona (2004), *Self-tuning spectral clustering*; [original dataset page recorded in the headers](http://www.vision.caltech.edu/lihi/Demos/SelfTuningClustering.html) |
| `longsquare.arff` | Handl and Knowles (2004), *Multiobjective clustering with automatic determination of the number of clusters*, technical report TR-COMPSYSBIO-2004-02 |
| `triangle1.arff` | The supplied ARFF has no source/citation header. The original attribution was not recovered from this file. |

### Labeled benchmarks (Case 4)

The loaders in `section_4_4_labeled_benchmarks/Benchmark_data_real.py` specify the preprocessing,
test-set selection, and expected sample/dimension/class counts. They reuse a
local file if present and otherwise download from the following recorded
sources. These are loader URLs, not a claim that each endpoint was rechecked
when preparing the package.

| Dataset | Loader/source |
|---|---|
| Digits, Iris, Wine | `sklearn.datasets.load_digits`, `load_iris`, `load_wine`; bundled with scikit-learn |
| MNIST-test | Test split of [Keras MNIST NPZ](https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz) |
| Fashion-MNIST-test | Test images and labels from the [Fashion-MNIST repository](https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion) |
| Ecoli | [UCI Ecoli raw file](https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data) |
| Letter | [UCI Letter Recognition raw file](https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data) |
| Libras | [UCI Libras Movement raw file](https://archive.ics.uci.edu/ml/machine-learning-databases/libras/movement_libras.data) |
| Vowel | [UCI Vowel raw file](https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/vowel/vowel-context.data) |

These third-party datasets retain their own terms and attribution requirements.
Bundling a copy does not change their ownership or grant a new license.

### Saved numerical results

The 98 `.pkl` files contain Python/NumPy dictionaries, scalars, and arrays from
the numerical studies. They are the inputs for cached reproduction. Use the
bundled files from this repository; pickle is an executable serialization
format and should not be used with unrelated, untrusted downloads.
