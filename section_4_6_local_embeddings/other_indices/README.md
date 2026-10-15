# Section 4.6 / Supplement S12 — Other internal validity indices

The three `manifold_tsne_umap_case*.py` scripts compute VRC, WG, WB, DBCV,
CVDD, VIASCKDE, DCSI, and DB on original data and tuned t-SNE/UMAP embeddings.
Saved results and parameter curves are under `data/case2/`, `data/case3/`, and
`data/case4/`.

`generate_manifold_bar_summary_tex.py` combines these results with the sibling
`silhouette/` results to draw all 31 Supplement S12 figures. The other
`generate_*.py` files supply loading functions, labels, and colors used by that
plotter. Use the top-level `reproduce.py` command for current percentage tables
and formatting.

See the [Section 4.6 README](../README.md) for commands and output locations.
