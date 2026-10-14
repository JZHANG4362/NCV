# NCV: reproducing the numerical studies

Code, saved numerical results, and figure snapshots for **A Nearest-Neighbor
Silhouette for Ranking Clusterings Across Granularities**.

This repository covers the empirical tables and figures in Section 4 and
Supplementary Sections S7–S12. The [output map](docs/OUTPUT_MAP.md) connects each
manuscript output to its scripts, inputs, and generated files.

## Find the code for a manuscript section

| Main manuscript | Supplement | Folder |
|---|---|---|
| **4.1 — Case 1: Gaussian simulation** | **S7** | [section_4_1_gaussian_simulation/](section_4_1_gaussian_simulation/) |
| **4.2 — Case 2: Simulated benchmarks** | **S8** | [section_4_2_simulated_benchmarks/](section_4_2_simulated_benchmarks/) |
| **4.3 — Case 3: Nonconvex benchmarks** | **S9** | [section_4_3_nonconvex_benchmarks/](section_4_3_nonconvex_benchmarks/) |
| **4.4 — Case 4: Labeled benchmarks** | **S10** | [section_4_4_labeled_benchmarks/](section_4_4_labeled_benchmarks/) |
| **4.5 — Sensitivity analysis** | **S11** | [section_4_5_sensitivity/](section_4_5_sensitivity/) |
| **4.6 — Local embeddings and internal validity indices** | **S12** | [section_4_6_local_embeddings/](section_4_6_local_embeddings/) |

Each section folder has its own README listing the relevant scripts and
outputs. Section 4.6 contains `silhouette/` and `other_indices/`; both are
needed for its nine-index tables and figures. Section 4.5 contains
`simulated_data/`, `nonconvex_data/`, `labeled_data/`, and `summary_plots/`.

**Known replication gap:** the saved Case 2 results for spectral clustering,
`k30_D200_sparse`, contain **39 replications**, although the manuscript describes
100. The supplied results reproduce the current tables exactly. A complete new
100-replication run will generally change those entries. See the
[audit](docs/AUDIT.md) before interpreting this as a completed 100-replication
benchmark.

## Set up

Use Python 3.12 for the tested reproduction environment. From this directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead. Alternatively:

```bash
conda env create -f environment.yml
conda activate ncv-reproduction
```

The version pins record the environment tested while preparing this repository;
they are **not a recovered lockfile for the original experiments**. The
manuscript reports Python 3.11, and the original environment files did not pin
library versions. Full reruns can therefore differ from the saved results.
In the fresh Iris test, benchmark values matched but the fitted embedding
results differed; exact reproduction of those results currently relies on the
supplied caches. Details are in [VALIDATION.md](docs/VALIDATION.md).

## Rebuild the outputs

Start with the numerical tables; this takes seconds and does not fit models:

```bash
python reproduce.py tables
```

This recalculates all 89 empirical tables: 84 supplementary benchmark tables,
three main-text benchmark tables, and two embedding-improvement tables. It
checks 8,838 displayed cells against the manuscript snapshot and writes LaTeX
fragments, CSV values/counts, and a validation report under `outputs/`.

To redraw all 38 empirical figures as well:

```bash
python reproduce.py cached
```

This uses the saved numerical results. The main Case 1 figure additionally
recalculates its single representative dataset's selected-neighborhood panel;
it does **not** repeat the 100-replication study. Figure rendering may vary
slightly with fonts and library versions.

To rebuild the tables and collect the **exact archived manuscript PDFs** without
redrawing them:

```bash
python reproduce.py archive --output outputs/archive
```

`archive` copies figures; it is not a fresh numerical or plotting run.
For all modes, `--output PATH` chooses the destination. No manuscript source is
edited automatically.

## Run the experiments from scratch

First test the three numerical pipelines on Iris under K-means:

```bash
python scripts/run_experiments.py smoke --execute
```

Inspect the complete job plan, then run it when sufficient compute is available:

```bash
python scripts/run_experiments.py all
python scripts/run_experiments.py all --execute
```

The runner creates `runs/all/`, copies the code and raw datasets there, and starts
without saved numerical results. It preserves the results shipped with this
repository. Full runs include large dense distance matrices, 100-replication
benchmarks, and repeated t-SNE/UMAP fits; they are intended for a compute server.
See [reproduction details](docs/REPRODUCTION.md) for individual stages, logs,
resuming, resource considerations, and exporting newly computed results.

## Repository layout

| Location | Contents |
|---|---|
| `section_4_1_.../` through `section_4_6_.../` | Numerical studies organized in manuscript order; scripts and saved results are kept with their section |
| `reproduce.py`, `scripts/` | Reader entry points, current table formatting, checks, and isolated experiment runner |
| `templates/` | Current manuscript captions and table layout; result rows are placeholders |
| `reference/` | Expected displayed values, exact figure snapshots, source provenance, and file hashes |
| `docs/` | Output map, audit findings, reproduction instructions, and dataset sources |
| `outputs/`, `runs/` | Generated files; excluded from version control |

The historical folder called `codes_new_manifold` is **not** the complete source
for this manuscript. It runs a different six-method Silhouette experiment. The
original source packages were traced to the current outputs and reorganized
into the six section folders above. Original paths are retained in the source
provenance; [folder_renames.json](reference/folder_renames.json) maps the previous
repository layout to the new one.

## Checks and attribution

```bash
python scripts/check_package.py
```

This checks packaged file hashes, Python/notebook syntax, the neighborhood and
score-direction conventions, saved-result coverage, and agreement with the
manuscript tables. Add `--strict-replications` to require all Case 2 vectors to
contain 100 runs; that stricter check intentionally fails for the supplied snapshot.

Dataset origins and required citations are listed in [DATA.md](docs/DATA.md).
See [NOTICE.md](NOTICE.md) for the status of software and third-party licensing.

