r"""
For each case (2/3/4), plots the MEAN Spearman rho vs. ACC as a function
of the candidate parameter value (perplexity for t-SNE, n_neighbors for
UMAP), one line per index, averaged across every (clustering_method,
dataset) pair in that case -- straight from the ALREADY-SAVED per-
parameter sensitivity curves (manifold_param_sensitivity_results_*.pkl),
no re-running of any cluster job needed.

This answers the ORACLE question directly and visually: does there EXIST
a parameter range where an index correlates well with ACC (the curve's
peak), independent of whether Algorithm 1's K-curve-shape criterion can
actually FIND that peak from the data alone (Algorithm 1's pick would be
a single x-position on this same curve, not necessarily the peak).

Datasets within a case don't all share the same candidate grid (e.g. a
small dataset like Iris only survives the `3*p < n_samples-1` filter up
to perplexity=40) -- at each x-value, the mean is taken over whichever
(cm, dataset) pairs actually have that x in their grid, so the curve
narrows to fewer contributing points at the high end.

Run this LOCALLY after retrieving data/ from the cluster and (for Case 4)
running merge_manifold_case4_results.py:

    python generate_oracle_landscape_plots.py

Writes figures/oracle_landscape/{case_tag}_perplexity.{pdf,png} and
..._n_neighbors.{pdf,png} for each case.
"""

import os
import pickle
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

DATA_DIR = 'data'
OUT_DIR = 'figures/oracle_landscape'
os.makedirs(OUT_DIR, exist_ok=True)

EXPERIMENT_TAG = 'tsne_umap_alt_indices_tuned_origkgrid_v1'
SCORE_DIRECTION_CONVENTION = 'larger_is_better_v1'

INDEX_NAMES = ['VRC', 'WG', 'WB', 'DBCV', 'CVDD', 'VIASCKDE', 'DCSI', 'DB']
# tab10 has 10 colors; 8 indices fit with room to spare, kept consistent
# across all 3 cases/both panels for easy cross-plot comparison.
INDEX_COLORS = dict(zip(INDEX_NAMES, sns.color_palette('tab10', n_colors=len(INDEX_NAMES))))

CASES = [
    dict(tag='case2', title='Case 2 -- Simulated Data',
         path=os.path.join(DATA_DIR, 'case2', f'manifold_param_sensitivity_results_{EXPERIMENT_TAG}.pkl')),
    dict(tag='case3', title='Case 3 -- Nonconvex Data',
         path=os.path.join(DATA_DIR, 'case3', f'manifold_param_sensitivity_results_{EXPERIMENT_TAG}.pkl')),
    dict(tag='case4', title='Case 4 -- Real Labeled Data',
         path=os.path.join(DATA_DIR, 'case4', f'manifold_param_sensitivity_results_{EXPERIMENT_TAG}.pkl')),
]


def load_results(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        payload = pickle.load(f)
    if payload.get('score_direction_convention') != SCORE_DIRECTION_CONVENTION:
        raise ValueError(f'{path} does not use the expected score-direction convention.')
    return payload['results']


def aggregate_curves(results, param_key):
    """results: {cm: {dname: {'r':.., 'perplexity': {idx: {p: rho}}, 'n_neighbors': {...}}}}.
    param_key: 'perplexity' or 'n_neighbors'.
    Returns {idx: {param_value: (mean_rho, n_contributing)}}, only for
    indices that actually have data."""
    per_index = defaultdict(lambda: defaultdict(list))
    for cm, by_dataset in results.items():
        for dname, entry in by_dataset.items():
            for idx, curve in entry.get(param_key, {}).items():
                for p, rho in curve.items():
                    if rho is not None and np.isfinite(rho):
                        per_index[idx][p].append(rho)
    agg = {}
    for idx, by_param in per_index.items():
        agg[idx] = {p: (float(np.mean(vals)), len(vals)) for p, vals in by_param.items()}
    return agg


def plot_landscape(case_tag, case_title, agg, xlabel, param_name, out_path):
    present = [idx for idx in INDEX_NAMES if idx in agg]
    missing = [idx for idx in INDEX_NAMES if idx not in agg]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    if present:
        for idx in present:
            xs = sorted(agg[idx].keys())
            ys = [agg[idx][x][0] for x in xs]
            ns = [agg[idx][x][1] for x in xs]
            n_lo, n_hi = min(ns), max(ns)
            n_label = f'n={n_lo}' if n_lo == n_hi else f'n={n_lo}-{n_hi}'
            # thin support (few contributing dataset/method pairs so far)
            # is flagged directly in the legend rather than hidden -- a
            # curve averaged over n=1 is not yet a reliable case-level
            # summary, just a single dataset's result.
            ax.plot(xs, ys, marker='o', linewidth=2, color=INDEX_COLORS[idx],
                     label=f'{idx} ({n_label})', alpha=1.0 if n_lo >= 5 else 0.5,
                     linestyle='-' if n_lo >= 5 else '--')
        ax.axhline(0, color='grey', linewidth=1, linestyle='--')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(r'mean Spearman $\rho$ with ACC (averaged over datasets x clustering methods)', fontsize=11)
        title = f'{case_title} -- {param_name} landscape (mean over available datasets/methods)'
        if missing:
            title += f'\n(not yet available: {", ".join(missing)})'
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best', ncol=2)
    else:
        ax.text(0.5, 0.5, f'No {param_name} sensitivity data available yet for {case_tag}',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{case_title} -- {param_name} landscape', fontsize=13, fontweight='bold')

    plt.tight_layout()
    fig.savefig(f'{out_path}.pdf', bbox_inches='tight')
    fig.savefig(f'{out_path}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}.pdf  ({len(present)}/{len(INDEX_NAMES)} indices present'
          + (f', missing: {missing}' if missing else '') + ')')


def main():
    for case in CASES:
        results = load_results(case['path'])
        if results is None:
            print(f"{case['tag']}: NO SENSITIVITY DATA FILE FOUND at {case['path']} -- skipping")
            continue

        agg_p = aggregate_curves(results, 'perplexity')
        agg_k = aggregate_curves(results, 'n_neighbors')

        plot_landscape(case['tag'], case['title'], agg_p, 'perplexity', 't-SNE',
                        os.path.join(OUT_DIR, f"{case['tag']}_perplexity"))
        plot_landscape(case['tag'], case['title'], agg_k, '$n_{neighbors}$', 'UMAP',
                        os.path.join(OUT_DIR, f"{case['tag']}_n_neighbors"))


if __name__ == '__main__':
    main()
