"""
Merges the per-job tagged results pickles (written when manifold_tsne_
umap_case4.py runs with the JOB_TAG environment variable set -- see the 18
large-dataset .dlx files, run_manifold_case4_{dataset}_{method}_{fast,
slow}.dlx) back into the canonical (untagged) results pickles in
data/case4/ that generate_manifold_tsne_umap_tex.py expects.

UNLIKE the untuned sibling's merge script: each (dataset, clustering
method) combination is now split into TWO separate jobs/files (fast/slow
index groups), each populating a DIFFERENT SUBSET of method keys within
the SAME (cm, dname) entry -- e.g. Letter_KMeans_fast.pkl has
{'rNN_Hard':.., 'VRC_Original':.., ...} and Letter_KMeans_slow.pkl has
{'rNN_Hard':.., 'DBCV_Original':.., ...} for the SAME (KMeans, Letter).
A naive dataset-level merge (replace dst[cm][dname] wholesale) would let
whichever job merges LAST silently discard the other job's columns -- so
this version merges recursively at every dict level, only overwriting
leaf values, so fast- and slow-group results for the same dataset combine
instead of clobbering each other. Safe to re-run at any point, including
when only ONE of the two groups has finished for a given dataset (that's
the whole point of the fast/slow split -- see the module docstring on
manifold_tsne_umap_case4.py).

Run this LOCALLY (no heavy conda env needed, just stdlib pickle) after
retrieving the whole data/ folder from the cluster:

    python merge_manifold_case4_results.py
"""

import os
import pickle

DATA_DIR = 'data/case4'
EXPERIMENT_TAG = 'tsne_umap_alt_indices_tuned_origkgrid_v1'
SCORE_DIRECTION_CONVENTION = 'larger_is_better_v1'

# Must match the JOB_TAG values set in all 18 per-job .dlx files
# (run_manifold_case4_{dataset}_{method}_{fast,slow}.dlx) exactly.
LARGE_DATASET_TAGS = ['MNISTtest', 'FashionMNISTtest', 'Letter']
CLUSTERING_METHODS = ['KMeans', 'Spectral', 'Agglomerative']
GROUPS = ['fast', 'slow']
JOB_TAGS = [
    f'{dtag}_{cm}_{group}'
    for dtag in LARGE_DATASET_TAGS
    for cm in CLUSTERING_METHODS
    for group in GROUPS
]

BASE_NAMES = [
    'manifold_fixed_results',
    'manifold_param_sensitivity_results',
    'manifold_param_star_results',
]

ALL_DATASETS = [
    'Digits', 'MNIST-test', 'Ecoli', 'Letter', 'Iris',
    'Fashion-MNIST-test', 'Wine', 'Libras', 'Vowel',
]
ALL_METHODS = ['KMeans', 'Spectral', 'Agglomerative']
INDEX_NAMES = ['VRC', 'WG', 'WB', 'DBCV', 'CVDD', 'VIASCKDE', 'DCSI', 'DB']
ALL_KEYS = ['rNN_Hard'] + [f'{idx}_{rep}' for idx in INDEX_NAMES for rep in ['Original', 'TSNE', 'UMAP']]


def path_for(base_name, job_tag=None):
    tag = f'{EXPERIMENT_TAG}_{job_tag}' if job_tag else EXPERIMENT_TAG
    return os.path.join(DATA_DIR, f'{base_name}_{tag}.pkl')


def deep_merge(dst, src):
    """Recursively merge src into dst IN PLACE, at every dict level --
    only overwrites leaf (non-dict) values. Critical here: fast- and
    slow-group job-tagged files populate DIFFERENT method-key subsets for
    the SAME (cm, dname), so a shallow dataset-level merge would let one
    group's file silently clobber the other's columns."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_merge(dst[k], v)
        else:
            dst[k] = v


def main():
    final_results = None  # merged manifold_fixed_results, for the completeness report

    for base_name in BASE_NAMES:
        canonical_path = path_for(base_name)
        if os.path.exists(canonical_path):
            with open(canonical_path, 'rb') as f:
                merged = pickle.load(f)['results']
        else:
            merged = {}

        n_merged = 0
        for job_tag in JOB_TAGS:
            tagged_path = path_for(base_name, job_tag)
            if not os.path.exists(tagged_path):
                continue
            with open(tagged_path, 'rb') as f:
                tagged_results = pickle.load(f)['results']
            deep_merge(merged, tagged_results)
            n_merged += 1

        if n_merged == 0 and not os.path.exists(canonical_path):
            print(f'{base_name}: nothing found yet -- skipping')
            continue

        out_payload = {
            'results': merged,
            'score_direction_convention': SCORE_DIRECTION_CONVENTION,
            'clustering_methods_computed': sorted(merged.keys()),
        }
        with open(canonical_path, 'wb') as f:
            pickle.dump(out_payload, f)
        print(f'{base_name}: merged {n_merged}/{len(JOB_TAGS)} job-tagged file(s) -> {canonical_path}')

        if base_name == 'manifold_fixed_results':
            final_results = merged

    if final_results:
        print('\n=== Completeness report (manifold_fixed_results, ACC) ===')
        print(f'{"dataset":20s} ' + ' '.join(f'{cm:>13s}' for cm in ALL_METHODS))
        for dname in ALL_DATASETS:
            counts = []
            for cm in ALL_METHODS:
                row = final_results.get(cm, {}).get(dname, {})
                n_present = sum(1 for key in ALL_KEYS if row.get(key) is not None)
                counts.append(f'{n_present:>10d}/{len(ALL_KEYS)}')
            print(f'{dname:20s} ' + ' '.join(counts))


if __name__ == '__main__':
    main()
