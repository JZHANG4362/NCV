# Reproduction workflow

## 1. Reconstruct the current manuscript outputs

Run `python reproduce.py tables` from the repository root. The exporter loads
the numerical pickles, computes mean/median correlations or improvement counts,
and inserts new values into the layout templates. Expected values in
`reference/table_values.json` are used to check the result, not to supply numeric
cells. Row metadata from the snapshot supplies dataset names and table labels.

Generated files:

| File under `outputs/` | Purpose |
|---|---|
| `tables/main_tables.tex` | Three main-text benchmark tables, identified by their existing labels |
| `tables/case2_tables_clean_v6.tex` | 42 Case 2 tables: 7 external criteria × 3 clustering methods × mean/median |
| `tables/nonconvex_tables_clean_v6.tex` | 21 Case 3 tables |
| `tables/realdata_tables_clean_v6.tex` | 21 Case 4 tables |
| `local_embedding_tables/tuned_by_index.tex` | Algorithm-selected embedding improvement percentages |
| `local_embedding_tables/oracle_by_index.tex` | Best-on-grid embedding improvement percentages |
| `tables/local_embedding_figures.tex` | Supplementary S12 figure inclusion statements and captions |
| `benchmark_values.csv` | Unrounded mean/median/scalar correlations in long form |
| `embedding_counts.csv` | Exact improved and valid dataset counts before percentage rounding |
| `summary_statistics.json` | Unrounded benchmark win/rank totals and pooled Silhouette embedding improvements used in the main-text discussion |
| `table_validation.json` | Row/cell totals, mismatches, and short replication vectors |
| `run_report.json` | Mode, Python version, and elapsed time |

`python reproduce.py cached` also creates `outputs/figures/` with the 38 PDFs.
The plotting scripts write intermediate figures beside their source files;
these generated directories are ignored by Git. Logs are in `outputs/logs/`.

The main Case 1 combined notebook reads two saved studies and recomputes the
single selected-r curve. The three supplementary Case 1 plots load the
multimetric cache. The sensitivity and 31 embedding plots are post-processing
of their corresponding cached results. Thus this mode is much cheaper than a
full rerun, but is not entirely free of numerical computation.

For exact copies of the historical PDFs, use `python reproduce.py archive`.
PDF metadata, font metrics, and overlapping label order can change when
redrawing a figure even when the underlying data agree.

The `.tex` files are fragments for inclusion in the manuscript, not standalone
documents. They retain the manuscript captions and references, including the
Algorithm reference in the embedding caption. Use them with the paper's usual
LaTeX preamble (`graphicx`, `booktabs`, `caption`, `pdflscape`, `float`, etc.).
The three main benchmark tables are currently embedded in `main.tex`; replace
those table environments with the matching generated blocks if updating the
paper. The other fragments and figures retain their manuscript-relative paths.

## 2. Test a small fresh run

```bash
python scripts/run_experiments.py smoke --execute
```

This runs K-means/Iris through all seven external criteria in the benchmark,
the tuned ordinary-Silhouette embedding study, and all eight alternative
embedding indices. It uses the original grids and scoring code, but only this
one small dataset/method combination. Its partial results are not sufficient
to build the full manuscript tables.

## 3. Run full numerical studies

The experiment runner prints commands unless `--execute` is supplied:

```bash
python scripts/run_experiments.py all
python scripts/run_experiments.py all --execute --workdir runs/full
```

Available groups are `case1`, `case2`, `case3`, `case4`, `sensitivity`,
`embeddings`, `all`, and `smoke`. Without `--workdir`, each uses `runs/GROUP/`.
The `all` group runs the first six groups sequentially in one run directory,
with no saved numerical caches copied in. Raw datasets are copied, so the
packaged data do not require network access during the run.

| Group | Work performed |
|---|---|
| `case1` | Gaussian simulation, 100-replication boxplot study, combined figure, supplementary multimetric figures |
| `case2` | All 12 settings, 100 replications, three clustering methods, seven external criteria |
| `case3` | Ten fixed nonconvex datasets, three clustering methods, seven criteria |
| `case4` | Nine labeled datasets, three clustering methods, seven criteria |
| `sensitivity` | 12 simulated settings with 30 replications; ten nonconvex and six smaller labeled datasets; K-means and ACC; aggregation of the parameter sweeps |
| `embeddings` | Both embedding packages on Cases 2–4, three clustering methods; ACC only; one simulated dataset per Case 2 setting |

Each successful job is recorded in `RUN_STATE.json`. To continue after a failure:

```bash
python scripts/run_experiments.py all --execute --resume --workdir runs/full
```

Completed jobs are skipped. An interrupted job is rerun from its beginning;
the historical scripts save checkpoints, but do not consistently resume at the
next unfinished replication. Keep the same group when resuming. Source files
are copied once when creating a run directory, so a resumed run retains that
code version.

Logs are numbered under `runs/full/logs/`. The runner uses the same Python
executable as the parent command, sets a writable Numba/Matplotlib cache, and
uses one thread per numerical worker. It clears an inherited `JOB_TAG` because
the sequential jobs write canonical result files. Independent parallel jobs
must not share those files. Historical merge scripts remain available, but
their expected job tags must match the jobs being merged; the provided
sequential workflow does not require merging.

Export a complete fresh run with:

```bash
python reproduce.py cached --results-root runs/full --output outputs/full --allow-changed-results
```

Differences from the historical manuscript are still listed in
`table_validation.json`. `--allow-changed-results` permits them instead of
stopping the exporter. Do not silently replace manuscript results without
reviewing that report. A new full Case 2 run supplies the missing replications
and is expected to change the affected spectral-clustering entries.

Full runs were **not** executed when preparing this repository. Dense pairwise
calculations scale quadratically in sample size, and the largest labeled
datasets and alternative-index embedding sweeps may require substantial time
and memory. No measured full-run resource estimate is available.

## 4. Numerical and formatting conventions

- `r` counts non-self neighbors. The distance graph uses the symmetric union.
- `rNN_Hard` is the manuscript's rNN-Sil. `rNN_Soft` is an additional diagnostic
  variant, used in the Case 1 boxplot but omitted from benchmark table columns.
- Scores are oriented before correlating so larger Spearman correlation means
  stronger agreement; NID, WB, and the embedding study's DB require their
  documented direction handling.
- Case 2 benchmark means/medians use finite saved correlations. The known
  39-replication cell is not padded or imputed.
- Benchmark tables use `Sil` and `rNN-Sil`, left alignment, and vertical rules
  around those two columns. Only rNN-Sil is bold, when its displayed value
  exceeds Sil. A superscript `\star` marks row maxima among displayed values.
- Embedding improvement compares each index on an embedding with that same
  index on original data. Oracle values are maxima over each dataset's candidate
  grid, whereas tuned values use the algorithm's selection.
- Overall percentages pool counts across Cases 2–4; they are not averages of
  case percentages. Denominators are 31, or 30 for DBCV. DBCV excludes
  Fashion-MNIST-test because its original-data value is unavailable.
- Integer percentages retain the manuscript rounding convention. The count CSV
  preserves exact numerators, denominators, and unrounded percentages.

Use the top-level entry points for current manuscript formatting. The retained
historical `generate_*_tex.py`, `generate_*_tables.py`, and
`scripts/legacy/build_manuscript_outputs.py` scripts include earlier standalone
layouts/captions; they are preserved as source provenance and helpers, not the
recommended current table exporter.

