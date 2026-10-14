# Validation performed

Checked on 2026-10-14, using Python 3.12.2 on macOS arm64. Exact package versions,
smoke-job timings, and each fresh-versus-saved comparison are recorded in
[`reference/validation_environment.json`](../reference/validation_environment.json).

| Check | Result |
|---|---|
| Current table values | All 8,838 displayed cells match: 89 empirical tables / 988 rows |
| Main-text benchmark summary | 496 improvements, 299 row maxima, 466 top-three results out of 651 comparisons |
| Pooled Silhouette embedding counts | t-SNE: 67/93 tuned, 81/93 oracle; UMAP: 61/93 tuned, 74/93 oracle |
| Historical figure provenance | 34/38 byte-identical; other four identical in text and in 1,600-pixel raster comparisons |
| Archived-output command | Recomputed tables and copied all 38 exact manuscript PDFs |
| Cached redraw | Rebuilt all 38 PDFs, including the single representative Case 1 selected-r calculation |
| Source syntax | All included Python files and notebook code cells parse |
| Neighborhood and score-direction checks | Existing independent numerical/static checks pass |
| Fresh benchmark smoke | K-means / Iris, all ten computed indices and seven external criteria: 70 correlations match the saved values within absolute tolerance `1e-12` |
| Fresh ordinary-Silhouette embedding smoke | Runs successfully; original Silhouette and rNN match, while the two embedding correlations differ |
| Fresh eight-index embedding smoke | Runs successfully; eight original-data values plus rNN match, while all 16 embedding correlations differ |
| Complete fresh experiment battery | Not run |

The smoke jobs took approximately 21 seconds (benchmark), 20 seconds (ordinary
Silhouette embeddings), and 25 seconds (other indices) on the validation
machine. These timings describe Iris only and are not estimates for the full
studies. The embedding runner initially encountered a read-only installation's
Numba cache restriction; the public wrapper now supplies a writable cache
directory and the numerical jobs completed.

Fresh embedding differences are substantive, not merely display rounding. For
example, the cached Iris/K-means ordinary-Silhouette t-SNE correlation is about
0.8239, whereas the fresh run gives about 0.9129. The original library versions
were not pinned, and the cause was not isolated. Do not present the smoke test
as confirmation of exact numerical replication of the embedding study.

The table checks compare regenerated numbers against a separate snapshot of
the manuscript cells. They do not substitute those expected numbers for the
calculation. Cached redraws and archived figures preserve the published
numerical evidence, while the isolated runner provides a way to investigate
and complete fresh experiments without overwriting that evidence.

The documented Case 2 replication gap remains: Spectral / `k30_D200_sparse`
contains 39 rather than 100 runs. `check_package.py --strict-replications`
reports that as a failure; ordinary snapshot checks acknowledge it explicitly.

## Validation after organizing folders by manuscript section

The six `section_4_*` folders were checked after renaming. All 23 full-run job
paths exist, and all eight shared raw-data paths resolve in an isolated staged
copy. A cached reproduction using the new layout rebuilt all 38 figures and
matched all 8,838 displayed table cells. The 161 saved data, reference, and
template files checked against the previous layout are unchanged. The earlier
smoke-job paths in `validation_environment.json` use the original layout; see
[`folder_renames.json`](../reference/folder_renames.json) for their current locations.

