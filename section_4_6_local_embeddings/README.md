# Section 4.6 — Local embeddings

**Main text:** Tables 4–5. **Supplement:** Figures S7–S37.

After [environment setup](../README.md#environment-setup), run from the repository root:

```bash
python reproduce.py cached
```

The tables are in `outputs/local_embedding_tables/tuned_by_index.tex`
(Table 4) and `oracle_by_index.tex` (Table 5). The figure paths below are
relative to `outputs/figures/local_embedding_bar_plots/`.

| Supplement figure | PDF |
|---|---|
| Figure S7 | `case2/k6_D2_compact.pdf` |
| Figure S8 | `case2/k6_D2_sparse.pdf` |
| Figure S9 | `case2/k6_D10_compact.pdf` |
| Figure S10 | `case2/k6_D10_sparse.pdf` |
| Figure S11 | `case2/k6_D200_compact.pdf` |
| Figure S12 | `case2/k6_D200_sparse.pdf` |
| Figure S13 | `case2/k30_D2_compact.pdf` |
| Figure S14 | `case2/k30_D2_sparse.pdf` |
| Figure S15 | `case2/k30_D10_compact.pdf` |
| Figure S16 | `case2/k30_D10_sparse.pdf` |
| Figure S17 | `case2/k30_D200_compact.pdf` |
| Figure S18 | `case2/k30_D200_sparse.pdf` |
| Figure S19 | `case3/pathbased.pdf` |
| Figure S20 | `case3/spiral.pdf` |
| Figure S21 | `case3/aggregation.pdf` |
| Figure S22 | `case3/atom.pdf` |
| Figure S23 | `case3/lsun.pdf` |
| Figure S24 | `case3/zelnik1.pdf` |
| Figure S25 | `case3/zelnik5.pdf` |
| Figure S26 | `case3/zelnik6.pdf` |
| Figure S27 | `case3/triangle1.pdf` |
| Figure S28 | `case3/longsquare.pdf` |
| Figure S29 | `case4/Digits.pdf` |
| Figure S30 | `case4/MNIST-test.pdf` |
| Figure S31 | `case4/Ecoli.pdf` |
| Figure S32 | `case4/Letter.pdf` |
| Figure S33 | `case4/Iris.pdf` |
| Figure S34 | `case4/Fashion-MNIST-test.pdf` |
| Figure S35 | `case4/Wine.pdf` |
| Figure S36 | `case4/Libras.pdf` |
| Figure S37 | `case4/Vowel.pdf` |

`other_indices/generate_manifold_bar_summary_tex.py` draws all 31 figures
using results from both `silhouette/` and `other_indices/`.
To rerun both pipelines for Cases 2–4:

```bash
python scripts/run_experiments.py embeddings --execute
```

Fresh results are saved under `runs/embeddings/`.
See the [root README](../README.md#run-all-experiments-from-scratch) to run and export all experiments.
