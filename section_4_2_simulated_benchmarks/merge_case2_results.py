"""
Merges the per-job tagged results pickles (written when
partA_type2_grid_comparison.py runs with the JOB_TAG environment variable
set -- see run_partA_{KMeans,Spectral,Agglomerative}.dlx) back into the
canonical (untagged) results pickles that everything else (table
generators, main.tex) expects.

Why this exists: running the 3 clustering-method jobs concurrently against
the SAME (untagged) results files risks one job's periodic save clobbering
another's in-progress work -- the same issue fixed for Case4
(see Case4/merge_realdata_results.py). Giving each job its own private,
disjoint output files via JOB_TAG removes that risk entirely, enabling
true parallel execution with no --dependency chaining needed -- but the
results then need to be combined back into the canonical files afterward,
which is what this script does.

Run this LOCALLY (no heavy Comp1 conda env needed, just stdlib pickle)
after scp'ing back the whole data/ folder from every job you ran:

    python merge_case2_results.py

Safe to re-run -- always starts from whatever canonical file already
exists (if any) and folds in every job-tagged file present, so partial
retrievals work fine; just re-run after retrieving more.
"""

import os
import pickle

DATA_DIR = 'data'

# Must match the JOB_TAG values set in the 3 per-method .dlx files exactly.
JOB_TAGS = ['KMeans', 'Spectral', 'Agglomerative']

# One results pickle exists per external metric (see path_for below); ACC
# is the only one without a `_{metric}` filename suffix (historical --
# it was the original/only metric before the other 6 were added).
METRICS = ['ACC', 'Jaccard', 'SS3', 'ARI', 'NMI', 'Powers', 'NID']
SCORE_DIRECTION_CONVENTION = 'larger_is_better_v1'

# The 10 internal validity indices computed by partA_type2_grid_comparison.py
# (rNN_Soft is included here for completeness/QA even though it's excluded
# from the manuscript's main tables).
ALL_METHODS = [
    'VRC', 'Silhouette', 'WG', 'WB', 'DBCV', 'rNN_Hard', 'rNN_Soft',
    'CVDD', 'VIASCKDE', 'DCSI',
]

# The full 12-cell experiment grid: K* in {6,30} x D in {2,10,200} x
# compactness in {compact,sparse}. Must match partA_type2_grid_comparison.py's
# own CASES construction exactly, since this drives the completeness report
# below (which case/method combinations are missing or short of N_REP=100).
K_STAR_VALUES = [6, 30]
D_VALUES = [2, 10, 200]
COMPACTNESS_VALUES = ['compact', 'sparse']
CASES = [
    (k_star, D, comp)
    for k_star in K_STAR_VALUES
    for D in D_VALUES
    for comp in COMPACTNESS_VALUES
]


def case_key(k_star, D, comp):
    '''Same case identifier convention as partA_type2_grid_comparison.py's
    case_key, e.g. (6, 2, "compact") -> "k6_D2_compact".'''
    return f'k{k_star}_D{D}_{comp}'


def path_for(metric, job_tag=None):
    '''Mirror of partA_type2_grid_comparison.py's results_path_for: with
    job_tag=None this returns the CANONICAL (untagged) path that downstream
    consumers (generate_case2_tables.py, ../tools/build_manuscript_outputs.py)
    read; with job_tag set to one of JOB_TAGS it returns that job's private,
    disjoint checkpoint file instead. The "nonconvex_..." filename prefix is
    a known historical naming quirk -- Case 2 is not the nonconvex-shapes
    case -- retained so it keeps matching existing pickles on disk and the
    verbatim string checks in ../tools/verify_score_directions.py.'''
    suffix = f'_{job_tag}' if job_tag else ''
    if metric == 'ACC':
        return os.path.join(DATA_DIR, f'nonconvex_case_comparison_results_nonself_oriented{suffix}.pkl')
    return os.path.join(DATA_DIR, f'nonconvex_case_comparison_results_nonself_oriented{suffix}_{metric}.pkl')


def deep_merge(dst, src):
    """dst, src: dict[cm][case_key][method] -> list of N_REP floats. Merges
    src into dst in place, at the clustering-method level (src's cm entries
    overwrite dst's if both present -- shouldn't happen if job tags are
    disjoint by clustering method)."""
    for cm, by_case in src.items():
        if cm in dst:
            print(f'    WARNING: clustering method {cm!r} already present -- '
                  f'overwriting (check for overlapping job tags)')
        dst[cm] = by_case


def main():
    # ── Pass 1: merge every job-tagged pickle into the canonical file, per metric ──
    for metric in METRICS:
        canonical_path = path_for(metric)
        # Start from whatever's already in the canonical file (if any) so
        # this script is safe to re-run incrementally as more job-tagged
        # files get scp'd back from the cluster -- previously-merged
        # clustering methods are preserved even if their tagged file isn't
        # present in this run.
        if os.path.exists(canonical_path):
            with open(canonical_path, 'rb') as f:
                merged = pickle.load(f)['results']
        else:
            merged = {}

        n_merged = 0
        for job_tag in JOB_TAGS:
            tagged_path = path_for(metric, job_tag)
            if not os.path.exists(tagged_path):
                continue
            with open(tagged_path, 'rb') as f:
                tagged_results = pickle.load(f)['results']
            deep_merge(merged, tagged_results)
            n_merged += 1

        if n_merged == 0 and not os.path.exists(canonical_path):
            # Nothing to merge and no pre-existing canonical file for this
            # metric -- skip writing rather than creating an empty pickle.
            continue

        payload = {
            'results': merged,
            'external_metric': metric,
            'score_direction_convention': SCORE_DIRECTION_CONVENTION,
            'clustering_methods_computed': sorted(merged.keys()),
        }
        with open(canonical_path, 'wb') as f:
            pickle.dump(payload, f)
        print(f'{metric}: merged {n_merged} job-tagged file(s) -> {canonical_path}')

    # ── Pass 2: report, per metric x clustering method, whether every one of
    # the 12 cases has reached the full N_REP=100 replications for every one
    # of the 10 validity indices -- this is what generate_case2_tables.py
    # uses (via its own case_is_complete check) to decide which tables it
    # can safely produce. ──
    print('\n=== Completeness report ===')
    for metric in METRICS:
        canonical_path = path_for(metric)
        if not os.path.exists(canonical_path):
            print(f'  {metric}: no canonical file found')
            continue
        with open(canonical_path, 'rb') as f:
            results = pickle.load(f)['results']
        for cm in ['KMeans', 'Spectral', 'Agglomerative']:
            if cm not in results:
                print(f'  {metric} / {cm}: not started')
                continue
            missing = [
                case_key(*case) for case in CASES
                if case_key(*case) not in results[cm]
                or any(len(results[cm][case_key(*case)].get(m, [])) < 100
                       for m in ALL_METHODS)
            ]
            status = 'COMPLETE' if not missing else f'{len(missing)}/{len(CASES)} cases incomplete: {missing}'
            print(f'  {metric} / {cm}: {status}')


if __name__ == '__main__':
    main()
