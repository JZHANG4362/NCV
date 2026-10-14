# Source and reproducibility audit

Prepared 2026-10-14 against the manuscript source in
`20261007_EJS_Adjust_tables`. No manuscript numbers, captions, or source files
were changed during repository preparation.

## Is `codes_new_manifold` sufficient?

No. That folder runs six Silhouette-family methods, including affinity-based
t-SNE/UMAP variants, with ACC as the external criterion. It does not contain
the complete Case 1 study, seven-criterion benchmark, sensitivity study, or
current nine-index embedding analysis. Some embedding grids also differ.

The sources actually traced to the current empirical outputs were found in the
neighboring `20260715_overleaf` project:

1. `codes/`: Case 1, Cases 2–4 benchmarks, sensitivity experiments, and results.
2. `codes_new_manifold_tuned_embedding_orig_kgrid/`: tuned ordinary Silhouette
   embeddings, required by the nine-index summaries.
3. `codes_new_manifold_alt_indices_tuned_orig_kgrid/`: the eight other indices,
   embedding summaries, and bar figures.

These three original packages are included under six section-numbered folders,
with 98 canonical numerical cache files. The original directory names above
describe the upstream project, not the current repository layout. See the
[README section map](../README.md#find-the-code-for-a-manuscript-section) and
[`folder_renames.json`](../reference/folder_renames.json) for current locations.
Duplicate job shards, personal scheduler files, cluster logs, unrelated papers,
Word instruction documents, and obsolete experiment variants were omitted.

## What agrees with the manuscript?

- All **934 benchmark rows** agree at displayed precision: 903 supplementary
  rows and 31 rows in the three main benchmark tables, totaling 8,406 cells.
- All **54 embedding-summary rows** agree, totaling 432 percentage cells. Exact
  numerators and denominators are regenerated from per-dataset comparisons.
- Together these cover **89 empirical tables, 988 rows, and 8,838 cells**.
- The pooled benchmark counts also agree: **496/651** improve over ordinary
  Silhouette, **299/651** attain the row maximum, and **466/651** rank in the
  top three. These counts use unrounded correlations and Case 2 means.
- All **38 empirical figure assets** were traced to the source packages. Of
  those, 34 are byte-identical to the manuscript copies. The remaining four
  Case 1 files have identical extracted text and identical rendered images at
  a maximum dimension of 1,600 pixels, despite different PDF bytes.
- A cached redraw was also exercised, including the representative Case 1
  selected-r calculation. Rendering is not promised to be byte-identical.

`reference/table_values.json` records the expected displayed cells.
`reference/figure_provenance.json` records the manuscript and upstream figure
hashes and the comparison used. `reference/source_manifest.json` records the
source file hashes. The repository contains the exact manuscript PDFs in
`reference/figures/` for visual comparison and `archive` mode.

## Known gap: 39 rather than 100 Case 2 replications

For **Spectral / `k30_D200_sparse`**, the canonical results contain 39 values
per index/criterion rather than 100. The separate Spectral job files in the
source project contain the same 39 values. Other Case 2 vectors contain 100.
This affects all seven external criteria and all ten computed indices; nine
indices are shown in the manuscript tables. Thus 14 supplementary table rows
(seven criteria × mean/median) are based on 39 replications. The main Case 2
table uses K-means and is not affected by this particular gap.

The existing displayed values match the means and medians of those 39 runs.
They were preserved, without inventing or filling missing replications. The
exporter prints a warning and lists all 63 affected displayed-method vectors
in its validation report. The captions retain the author's existing text and
still state 100; that inconsistency has not been silently corrected here.

Before treating the study as fully replicated, either recover the completed
run from its original compute system or run the missing setting to completion
and update the affected results and manuscript claims. This repository does
not assert that a fresh 100-run result matches the existing 39-run table.

## Other boundaries

- DBCV on original Fashion-MNIST-test is unavailable. Benchmark tables retain
  `NA`; embedding comparisons exclude that dataset for DBCV, using Case 4
  denominator 8 and Overall denominator 30.
- Section 4.6 / S12 embedding results are ACC-only, unlike the seven-criterion
  benchmark. Case 2 embedding results use one dataset per setting, not the
  benchmark's 100-replication aggregation.
- The sensitivity study uses K-means/ACC, 30 simulated replications per setting,
  and six smaller real datasets; it is a distinct experiment.
- The original environments list Python 3.11 but do not pin scientific library
  versions. The tested packaging environment uses Python 3.12.2. Its version
  snapshot is included; it does not establish the original full-run environment.
- The full experiment battery was not rerun. A fresh Iris/K-means smoke test
  checks all three principal numerical pipelines. See `VALIDATION.md` for the
  observed verification results and any fresh-run differences.
- In that smoke test, all 70 benchmark correlations match the saved values
  within absolute tolerance `1e-12`. All 18 embedding-dependent correlations
  (t-SNE and UMAP for nine indices) differ; the 11 original-data/rNN reference
  values across the two embedding packages match. The original software
  versions were not recovered, and the cause of these differences was not
  isolated. Cached reproduction is established; exact fresh embedding
  reproduction is not.

## Changes made while packaging

Scientific experiment formulas, grids, seeds, and saved numerical result files
were preserved. In the four embedding Case 3/4 scripts, raw-data paths point
to the shared identical datasets in `section_4_3_nonconvex_benchmarks` and
`section_4_4_labeled_benchmarks`, avoiding
three copies of the same inputs. Notebook output and execution metadata were
cleared; code cells were retained.

New reader-facing scripts handle current table formatting, cached exports,
isolated fresh runs, writable JIT/plot caches, and validation. Layout templates
were extracted from the current manuscript, including left alignment,
separators, `Sil`, `rNN-Sil`, bold comparison markers, `\star`, and aggregated
percentages. The historical source generators remain as provenance/helpers.

