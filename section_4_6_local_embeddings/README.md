# Section 4.6 — Effects of Local Embeddings on Internal Validity Indices

**Main manuscript:** Section 4.6, tables `tab:improvement_tuned_by_index` and
`tab:improvement_oracle_by_index`.
**Supplement:** Section S12, 31 per-dataset embedding bar plots.

| Subfolder | Indices covered |
|---|---|
| [silhouette/](silhouette/) | Ordinary Silhouette on original data, t-SNE, and UMAP, plus the rNN reference |
| [other_indices/](other_indices/) | VRC, WG, WB, DBCV, CVDD, VIASCKDE, DCSI, and DB, plus the rNN reference |

**Both subfolders are required.** The summary tables and figures combine the
Silhouette results with the eight other indices. Each subfolder contains
`manifold_tsne_umap_case2.py`, `manifold_tsne_umap_case3.py`, and
`manifold_tsne_umap_case4.py`, with saved results under `data/case2/`,
`data/case3/`, and `data/case4/`.

From the **repository root**:

```bash
python reproduce.py tables
python reproduce.py cached
python scripts/run_experiments.py embeddings --execute
```

The first two commands rebuild manuscript outputs from the supplied results.
The last runs both numerical pipelines afresh in a separate directory. Exact
fresh embedding reproduction is not established; see the
[reproduction notes](../README.md#reproduction-notes).

The main table fragments are in `outputs/local_embedding_tables/`.
The 31 figures are in `outputs/figures/local_embedding_bar_plots/` and are
drawn by `other_indices/generate_manifold_bar_summary_tex.py`.
Tuned results use algorithm-selected parameters; oracle results use the best
ACC rank correlation on each candidate grid. This study is ACC-only.

