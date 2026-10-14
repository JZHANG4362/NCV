"""
Merges the per-job tagged results pickles (written when Benchmark_data_real.py
runs with the JOB_TAG environment variable set -- see the 8 remaining
large-dataset .dlx files, run_realdata_{dataset}_{method}.dlx) back into the
canonical (untagged) results pickles that everything else (table generators,
main.tex) expects.

Why this exists: running multiple Benchmark_data_real.py jobs concurrently
against the SAME (untagged) results files risks one job's periodic save
clobbering another's in-progress work (see cluster_setup_commands_realdata.docx
-- the original reason for the "run sequentially only" rule). Giving each job
its own private, disjoint output files via JOB_TAG removes that risk entirely,
enabling true parallel execution -- but the results then need to be combined
back into the canonical files afterward, which is what this script does.

Run this LOCALLY (no heavy Comp1 conda env needed, just stdlib pickle) after
scp'ing back the whole data/ folder from every job you ran:

    python merge_realdata_results.py

Safe to re-run -- always starts from whatever canonical file already exists
(if any) and folds in every job-tagged file present, so partial retrievals
(e.g. only some of the 8 jobs have finished so far) work fine; just re-run
after retrieving more.
"""

import os
import pickle

DATA_DIR = 'data'
EXPERIMENT_TAG = 'paperkmax_buffer10_r130_nonself_oriented'
SCORE_DIRECTION_CONVENTION = 'larger_is_better_v1'

# Must match the JOB_TAG values set in all 9 per-job .dlx files
# (run_realdata_MNIST-test_KMeans.dlx, etc.) exactly.
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
    'realdata_fixed_results',
    'realdata_rnn_r_sensitivity_results',
    'realdata_rnn_r_star_results',
]

METRICS = ['ACC', 'Jaccard', 'SS3', 'ARI', 'NMI', 'Powers', 'NID']

ALL_DATASETS = [
    'Digits', 'MNIST-test', 'Ecoli', 'Letter', 'Iris',
    'Fashion-MNIST-test', 'Wine', 'Libras', 'Vowel',
]
ALL_METHODS = ['KMeans', 'Spectral', 'Agglomerative']


def path_for(base_name, metric, job_tag=None):
    tag = f'{EXPERIMENT_TAG}_{job_tag}' if job_tag else EXPERIMENT_TAG
    suffix = '' if metric == 'ACC' else f'_{metric}'
    return os.path.join(DATA_DIR, f'{base_name}_{tag}{suffix}.pkl')


def deep_merge(dst, src):
    """dst, src: dict[cm][dataset][...]. Merges src into dst in place, at the
    dataset level (src's (cm, dataset) entries overwrite dst's if both are
    present -- shouldn't happen if job tags are disjoint by dataset)."""
    for cm, by_dataset in src.items():
        dst.setdefault(cm, {})
        for dname, payload in by_dataset.items():
            if dname in dst[cm]:
                print(f'    WARNING: {cm}/{dname} already present -- '
                      f'overwriting (check for overlapping job tags)')
            dst[cm][dname] = payload


def main():
    final_results = {}  # metric -> merged results dict, for the completeness report

    for base_name in BASE_NAMES:
        for metric in METRICS:
            canonical_path = path_for(base_name, metric)
            if os.path.exists(canonical_path):
                with open(canonical_path, 'rb') as f:
                    merged = pickle.load(f)['results']
            else:
                merged = {}

            n_merged = 0
            for job_tag in JOB_TAGS:
                tagged_path = path_for(base_name, metric, job_tag)
                if not os.path.exists(tagged_path):
                    continue
                with open(tagged_path, 'rb') as f:
                    tagged_results = pickle.load(f)['results']
                deep_merge(merged, tagged_results)
                n_merged += 1

            if n_merged == 0 and not os.path.exists(canonical_path):
                continue  # nothing at all yet for this (base_name, metric)

            out_payload = {
                'results': merged,
                'external_metric': metric,
                'score_direction_convention': SCORE_DIRECTION_CONVENTION,
                'clustering_methods_computed': sorted(merged.keys()),
            }
            with open(canonical_path, 'wb') as f:
                pickle.dump(out_payload, f)
            print(f'{base_name} / {metric}: merged {n_merged} job-tagged file(s) '
                  f'-> {canonical_path}')

            if base_name == 'realdata_fixed_results':
                final_results[metric] = merged

    if final_results:
        print('\n=== Completeness report (realdata_fixed_results) ===')
        for metric, results in final_results.items():
            missing = [
                (cm, dname)
                for cm in ALL_METHODS
                for dname in ALL_DATASETS
                if dname not in results.get(cm, {})
            ]
            status = 'COMPLETE (all 27 dataset x method combos)' if not missing else f'{len(missing)} missing: {missing}'
            print(f'  {metric}: {status}')


if __name__ == '__main__':
    main()
