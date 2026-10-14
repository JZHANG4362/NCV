"""
Merges the per-job tagged results pickles (written when manifold_tsne_
umap_case4.py runs with the JOB_TAG environment variable set -- see the 9
large-dataset .dlx files, run_manifold_case4_{dataset}_{method}.dlx) back
into the canonical (untagged) results pickles in data/case4/ that
generate_manifold_tsne_umap_tex.py expects.

Mirrors section_4_4_labeled_benchmarks/merge_realdata_results.py exactly, adapted for this
folder's single-metric (ACC only) layout: data/case4/{base_name}_
{EXPERIMENT_TAG}[_{JOB_TAG}].pkl instead of a metric-suffixed filename.

Run this LOCALLY (no heavy conda env needed, just stdlib pickle) after
retrieving the whole data/ folder from the cluster:

    python merge_manifold_case4_results.py

Safe to re-run -- always starts from whatever canonical file already
exists (if any) and folds in every job-tagged file present, so partial
retrievals (e.g. only some of the 9 large-dataset jobs have finished so
far) work fine; just re-run after retrieving more.
"""

import os
import pickle

DATA_DIR = 'data/case4'
EXPERIMENT_TAG = 'tsne_umap_tuned_embedding_origkgrid_v1'
SCORE_DIRECTION_CONVENTION = 'larger_is_better_v1'

# Must match the JOB_TAG values set in all 9 per-job .dlx files
# (run_manifold_case4_MNIST-test_KMeans.dlx, etc.) exactly.
JOB_TAGS = [
    'MNISTtest_KMeans',
    'MNISTtest_Spectral',
    'MNISTtest_Agglomerative',
    'FashionMNISTtest_KMeans',
    'FashionMNISTtest_Spectral',
    'FashionMNISTtest_Agglomerative',
    'Letter_KMeans',
    'Letter_Spectral',
    'Letter_Agglomerative',
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


def path_for(base_name, job_tag=None):
    tag = f'{EXPERIMENT_TAG}_{job_tag}' if job_tag else EXPERIMENT_TAG
    return os.path.join(DATA_DIR, f'{base_name}_{tag}.pkl')


def deep_merge(dst, src):
    """dst, src: dict[cm][dataset][...]. Merges src into dst in place, at
    the dataset level (src's (cm, dataset) entries overwrite dst's if both
    are present -- shouldn't happen if job tags are disjoint by dataset)."""
    for cm, by_dataset in src.items():
        dst.setdefault(cm, {})
        for dname, payload in by_dataset.items():
            if dname in dst[cm]:
                print(f'    WARNING: {cm}/{dname} already present -- '
                      f'overwriting (check for overlapping job tags)')
            dst[cm][dname] = payload


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
        print(f'{base_name}: merged {n_merged} job-tagged file(s) -> {canonical_path}')

        if base_name == 'manifold_fixed_results':
            final_results = merged

    if final_results:
        print('\n=== Completeness report (manifold_fixed_results, ACC) ===')
        missing = [
            (cm, dname)
            for cm in ALL_METHODS
            for dname in ALL_DATASETS
            if dname not in final_results.get(cm, {})
        ]
        status = ('COMPLETE (all 27 dataset x method combos)' if not missing
                   else f'{len(missing)} missing: {missing}')
        print(f'  {status}')


if __name__ == '__main__':
    main()
