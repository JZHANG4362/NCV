"""
Case 2 (simulated data): companion to section_4_2_simulated_benchmarks/partA_type2_grid_
comparison.py. Runs the FULL 12-setting grid (k* in {6,30} x D in
{2,10,200} x compactness in {compact,sparse} -- matches Table 1's own
grid exactly, same CASES construction as partA_type2_grid_comparison.py),
with N_REP=1 (no replication) per setting, and compares 4 silhouette-
family validity indices against ACC:

  1. Silhouette     -- ordinary silhouette, Euclidean distance on X.
  2. rNN_Hard       -- the existing rNN-Hard silhouette baseline (r*,
                        Algorithm 1).
  3. TSNE_Ordinary  -- ordinary (Euclidean) silhouette on the 2D t-SNE
                        embedding of X, perplexity* TUNED via Algorithm 1
                        (swept over PERPLEXITY_VALUES) -- NOT an affinity-
                        matrix substitution, the genuine t-SNE embedding.
  4. UMAP_Ordinary  -- ordinary (Euclidean) silhouette on the 2D UMAP
                        embedding of X, n_neighbors* TUNED via Algorithm 1
                        (swept over N_NEIGHBORS_VALUES) -- genuine UMAP
                        embedding, no affinity substitution.

This experiment deliberately does NOT include the affinity-substitution
methods (TSNE_JointProb / UMAP_FuzzyUnion) from codes_new_manifold /
codes_new_manifold_new_grid -- see manifold_tsne_umap_case3.py's module
docstring for the full rationale.

Only ACC is used as the external metric.

Parameter-selection algorithm (perplexity* for t-SNE, n_neighbors* for
UMAP): EXACTLY Algorithm 1 (select_r_topK_spearman), generalized in name
only (select_param_topK_spearman) -- identical convention (m=35,
threshold=0.95, same EVALUATION/SELECTION K-ranges) as the paper's own r*
selection, applied to the silhouette-vs-K curve computed on the genuine
embedding at each candidate perplexity/n_neighbors.

SCALE NOTE: each grid point requires a FULL t-SNE or UMAP embedding
optimization (not just an affinity-matrix build) -- see
manifold_tsne_umap_case3.py's module docstring.

Compute-batching: D_hard(r) / the t-SNE embeddings / the UMAP embeddings
depend only on X, never on the clustering method or K -- built ONCE per
setting here and reused across all 3 clustering methods.

Usage (clustering method via CLI arg, default = all 3 sequentially):
    python manifold_tsne_umap_case2.py             # all 3 methods
    python manifold_tsne_umap_case2.py KMeans
    python manifold_tsne_umap_case2.py Spectral
    python manifold_tsne_umap_case2.py Agglomerative

The main loop checkpoints after every setting, so an interrupted/timed-out
job can simply be resubmitted -- completed settings are preserved in the
results pickles.

Requires: numpy, scipy, scikit-learn, pandas, matplotlib, seaborn, umap-learn
"""

# ── cell: imports ─────────────────────────────────────────────────────
import os
import sys
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless-safe for remote/SLURM nodes (no display)
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.neighbors import NearestNeighbors

import umap

sns.set_style('whitegrid')
warnings.filterwarnings('ignore', message='Graph is not fully connected')
warnings.filterwarnings('ignore', category=UserWarning, module='umap')

# ── cell: helpers -- dataset generation (verbatim from partA) ──────────
def simulate_type2_data(n_samples, n_features, k_star, compactness, random_state=1):
    '''Approximate a Type-2-style dataset using Gaussian blobs; identical to
    partA_type2_grid_comparison.py's simulate_type2_data.'''
    X, y = make_blobs(n_samples=n_samples, n_features=n_features,
                      centers=k_star, cluster_std=compactness,
                      random_state=random_state)
    return X, y


def sample_size_for(k_star):
    '''Cluster sizes in {20, 100} per MDCGen's Type 2 description, capped at
    1000 total. Verbatim from partA_type2_grid_comparison.py.'''
    return min(1000, k_star * 50)


# ── cell: helpers -- rNN baseline (verbatim from partA/partB) ──────────
def rnn_affinity_distance(X, r, use_gaussian=False):
    '''
    Build rNN affinity distance matrix (precomputed, zeros on diagonal).
    Hard (binary): distance = 0 when either point lists the other among
    its r nearest neighbors (the symmetric union), and 1 otherwise.
    '''
    n = X.shape[0]
    if not 1 <= r < n:
        raise ValueError(f'r must satisfy 1 <= r < n; got r={r}, n={n}')
    nbrs = NearestNeighbors(n_neighbors=r + 1, algorithm='auto').fit(X)
    dists, idx = nbrs.kneighbors(X)
    dists, idx = dists[:, 1:], idx[:, 1:]

    A = np.zeros((n, n))
    if use_gaussian:
        for i in range(n):
            sigma = dists[i, -1] or 1e-10
            for j_pos, j in enumerate(idx[i]):
                A[i, j] = np.exp(-dists[i, j_pos]**2 / (2 * sigma**2))
    else:
        for i in range(n):
            A[i, idx[i]] = 1.0

    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0)
    a_min, a_max = A.min(), A.max()
    if a_max > a_min:
        A = (A - a_min) / (a_max - a_min)
    D = 1.0 - A
    np.fill_diagonal(D, 0)
    return D


# ── cell: helpers -- genuine t-SNE / UMAP embeddings (NO affinity substitution) ──
def tsne_embedding(X, perplexity, random_state=1):
    '''Genuine 2D t-SNE embedding -- ordinary Euclidean silhouette is
    computed on this directly (no affinity-matrix substitution).'''
    return TSNE(n_components=2, perplexity=perplexity, random_state=random_state,
                init='pca').fit_transform(X)


def umap_embedding(X, n_neighbors, random_state=1):
    '''Genuine 2D UMAP embedding -- ordinary Euclidean silhouette is
    computed on this directly (no affinity-matrix substitution).'''
    return umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                      random_state=random_state).fit_transform(X)


# ── cell: helpers -- Algorithm 1, generalized (verbatim logic) ─────────
def spearmanr_safe(a, b):
    '''Spearman correlation, returns NaN on degenerate input.'''
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(b)
    if mask.sum() < 2:
        return np.nan
    rho, _ = spearmanr(a[mask], b[mask])
    return float(rho)


def select_param_topK_spearman(sil_dict, param_values, m, n_k, threshold=0.95):
    '''Generalization of Algorithm 1 / select_r_topK_spearman to ANY tunable
    neighborhood-scale parameter -- see manifold_tsne_umap_case3.py for the
    full explanation. Logic copied verbatim.'''
    avail = sorted(p for p in param_values if p in sil_dict)
    if not avail:
        return param_values[0], {}
    m_eff   = max(2, min(m, n_k - 1))
    top_idx = list(range(n_k - m_eff, n_k))
    ideal   = np.arange(m_eff, 0, -1, dtype=float)
    c_vals  = {}
    for p in avail:
        obs       = np.asarray(sil_dict[p], dtype=float)[top_idx]
        rho       = spearmanr_safe(obs, ideal)
        c_vals[p] = rho if not np.isnan(rho) else -2.0
    p_star = next((p for p in avail if c_vals[p] >= threshold), None)
    if p_star is None:
        p_star = max(avail, key=lambda p: c_vals[p])
    return p_star, c_vals


def max_K_eval(k_star):
    '''max(25, 1.75*k*) -- the reference paper's own K-range cap.'''
    return int(max(25, round(1.75 * k_star)))


def max_K_select(k_star):
    '''max_K_eval(k*) + 10 -- buffer ADDED ONLY for parameter selection.'''
    return max_K_eval(k_star) + 10


# ── cell: external metric (ACC only -- see module docstring) ───────────
def clustering_accuracy(y_true, y_pred):
    '''Unsupervised clustering accuracy via Hungarian-matched contingency
    table.'''
    cm = contingency_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    return cm[row_ind, col_ind].sum() / cm.sum()


SCORE_DIRECTION_CONVENTION = 'larger_is_better_v1'

# ── cell: clustering-methods (verbatim) ─────────────────────────────────
def kmeans_labels(X, K, seed):
    return KMeans(n_clusters=K, random_state=seed, n_init='auto').fit_predict(X)


def spectral_labels(X, K, seed):
    return SpectralClustering(
        n_clusters=K, random_state=seed,
        affinity='nearest_neighbors', n_neighbors=10,
        assign_labels='kmeans',
    ).fit_predict(X)


def agglomerative_labels(X, K, seed):
    return AgglomerativeClustering(n_clusters=K, linkage='ward').fit_predict(X)


CLUSTERING_METHOD_FUNCS = {
    'KMeans':        kmeans_labels,
    'Spectral':      spectral_labels,
    'Agglomerative': agglomerative_labels,
}
CLUSTERING_METHODS = ['KMeans', 'Spectral', 'Agglomerative']

# ── cell: settings -- full 12-setting grid, N_REP=1 (verbatim grid from
#    partA_type2_grid_comparison.py) ─────────────────────────────────────
K_STAR_VALUES      = [6, 30]
D_VALUES           = [2, 10, 200]
COMPACTNESS_VALUES = {'compact': 0.1, 'sparse': 0.8}
RANDOM_STATE = 1

CASES = [
    (k_star, D, comp_label)
    for k_star in K_STAR_VALUES
    for D in D_VALUES
    for comp_label in COMPACTNESS_VALUES
]


def case_key(case):
    '''e.g. (6, 2, "compact") -> "k6_D2_compact".'''
    k_star, D, comp_label = case
    return f'k{k_star}_D{D}_{comp_label}'


CASE_META = {
    case_key(c): dict(K=c[0], N=sample_size_for(c[0]), D=c[1], compactness=c[2])
    for c in CASES
}
DATASETS_TO_RUN = [case_key(c) for c in CASES]

R_VALUES           = list(range(5, 135, 5))    # r grid: 5, 10, ..., 130
PERPLEXITY_VALUES  = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
N_NEIGHBORS_VALUES = [5, 10, 15, 20, 30, 50, 75, 100]
M                  = 35
THRESHOLD          = 0.95

ALL_METHODS = ['Silhouette', 'rNN_Hard', 'TSNE_Ordinary', 'UMAP_Ordinary']

if len(sys.argv) > 1:
    assert sys.argv[1] in CLUSTERING_METHODS, f'Unknown clustering method {sys.argv[1]!r}'
    CLUSTERING_METHODS_TO_RUN = [sys.argv[1]]
else:
    CLUSTERING_METHODS_TO_RUN = CLUSTERING_METHODS

CASE_TAG = 'case2'
EXPERIMENT_TAG = 'tsne_umap_tuned_embedding_origkgrid_v1'


def results_path_for(base_name):
    os.makedirs(f'data/{CASE_TAG}', exist_ok=True)
    return f'data/{CASE_TAG}/{base_name}_{EXPERIMENT_TAG}.pkl'


print(f'{len(CASES)} settings (N_REP=1, no replication):')
for c in CASES:
    ck = case_key(c)
    m = CASE_META[ck]
    print(f'  {ck:20s} k*={m["K"]}  D={m["D"]}  N={m["N"]}  compactness={m["compactness"]}  '
          f'max_K_select={max_K_select(m["K"])} max_K_eval={max_K_eval(m["K"])}')
print(f'R_VALUES: {R_VALUES[0]}..{R_VALUES[-1]} ({len(R_VALUES)})')
print(f'PERPLEXITY_VALUES: {PERPLEXITY_VALUES}')
print(f'N_NEIGHBORS_VALUES: {N_NEIGHBORS_VALUES}')
print(f'Clustering methods this session: {CLUSTERING_METHODS_TO_RUN}')

# ── cell: load-results ───────────────────────────────────────────────
def load_or_init(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            payload = pickle.load(f)
        results = payload['results']
        print(f'Loaded existing results from {path}')
        return results
    print(f'No existing results file found at {path} — starting fresh.')
    return {}


main_results        = load_or_init(results_path_for('manifold_fixed_results'))
sensitivity_results = load_or_init(results_path_for('manifold_param_sensitivity_results'))
param_star_results   = load_or_init(results_path_for('manifold_param_star_results'))


def save_results(results_dict, base_name):
    payload = {
        'results': results_dict,
        'score_direction_convention': SCORE_DIRECTION_CONVENTION,
        'clustering_methods_computed': sorted(results_dict.keys()),
    }
    with open(results_path_for(base_name), 'wb') as f:
        pickle.dump(payload, f)


def save_all():
    save_results(main_results, 'manifold_fixed_results')
    save_results(sensitivity_results, 'manifold_param_sensitivity_results')
    save_results(param_star_results, 'manifold_param_star_results')


# ── cell: main-loop -- one Gaussian-blob dataset per setting, N_REP=1 ────
# NOTE: D_hard(r) and the t-SNE/UMAP embedding sweeps depend only on X --
# built ONCE per setting, shared across all 3 clustering methods.
for dname in DATASETS_TO_RUN:
    meta = CASE_META[dname]
    k_star, D_features, n_samples_target = meta['K'], meta['D'], meta['N']
    compactness = COMPACTNESS_VALUES[meta['compactness']]

    X_raw, y = simulate_type2_data(
        n_samples=n_samples_target, n_features=D_features, k_star=k_star,
        compactness=compactness, random_state=RANDOM_STATE,
    )
    X = StandardScaler().fit_transform(X_raw)
    n_samples, Ddim = X.shape
    max_K_s = max_K_select(k_star)
    max_K_e = max_K_eval(k_star)
    K_TEST = list(range(2, max_K_s + 1))
    n_k = len(K_TEST)
    K_idx_lookup = {K: i for i, K in enumerate(K_TEST)}
    eval_idx = [K_idx_lookup[K] for K in K_TEST if K <= max_K_e]

    for cm_name in CLUSTERING_METHODS_TO_RUN:
        main_results.setdefault(cm_name, {})[dname] = main_results.get(cm_name, {}).get(dname, {})
        sensitivity_results.setdefault(cm_name, {})[dname] = {'r': {}, 'perplexity': {}, 'n_neighbors': {}}
        param_star_results.setdefault(cm_name, {})

    print(f'\n=== Setting {dname}  (k*={k_star}, D={Ddim}, N={n_samples}) ===')

    r_grid     = [r for r in R_VALUES if r < n_samples]
    perpl_grid = [p for p in PERPLEXITY_VALUES if 3 * p < n_samples - 1]
    nn_grid    = [k for k in N_NEIGHBORS_VALUES if k < n_samples - 1]

    print(f'  building rNN sweep ({len(r_grid)} r values)...')
    D_hard_by_r = {r: rnn_affinity_distance(X, r, use_gaussian=False) for r in r_grid}
    print(f'  building t-SNE embedding sweep ({len(perpl_grid)} perplexity values)...')
    Y_tsne_by_p = {p: tsne_embedding(X, p, random_state=1) for p in perpl_grid}
    print(f'  building UMAP embedding sweep ({len(nn_grid)} n_neighbors values)...')
    Y_umap_by_k = {k: umap_embedding(X, k, random_state=1) for k in nn_grid}

    for cm_name in CLUSTERING_METHODS_TO_RUN:
        cm_fn = CLUSTERING_METHOD_FUNCS[cm_name]
        seed = 1  # no replication -- fixed seed is enough for reproducibility

        labels_by_K = {K: cm_fn(X, K, seed) for K in K_TEST}
        multi_cluster = {K: len(np.unique(labels_by_K[K])) > 1 for K in K_TEST}

        ext_vals = np.array(
            [clustering_accuracy(y, labels_by_K[K]) for K in K_TEST]
        )[eval_idx]

        sil_curve = np.array([
            silhouette_score(X, labels_by_K[K], metric='euclidean') if multi_cluster[K] else np.nan
            for K in K_TEST
        ])
        main_results[cm_name][dname]['Silhouette'] = spearmanr_safe(sil_curve[eval_idx], ext_vals)

        sil_hard_by_r = {
            r: np.array([
                silhouette_score(D_hard_by_r[r], labels_by_K[K], metric='precomputed') if multi_cluster[K] else np.nan
                for K in K_TEST
            ])
            for r in r_grid
        }
        r_star, _ = select_param_topK_spearman(sil_hard_by_r, r_grid, m=M, n_k=n_k, threshold=THRESHOLD)
        rho_hard = spearmanr_safe(sil_hard_by_r[r_star][eval_idx], ext_vals)
        main_results[cm_name][dname]['rNN_Hard'] = rho_hard

        sil_tsne_by_p = {
            p: np.array([
                silhouette_score(Y_tsne_by_p[p], labels_by_K[K], metric='euclidean') if multi_cluster[K] else np.nan
                for K in K_TEST
            ])
            for p in perpl_grid
        }
        p_star, _ = select_param_topK_spearman(sil_tsne_by_p, perpl_grid, m=M, n_k=n_k, threshold=THRESHOLD)
        rho_tsne = spearmanr_safe(sil_tsne_by_p[p_star][eval_idx], ext_vals)
        main_results[cm_name][dname]['TSNE_Ordinary'] = rho_tsne

        sil_umap_by_k = {
            k: np.array([
                silhouette_score(Y_umap_by_k[k], labels_by_K[K], metric='euclidean') if multi_cluster[K] else np.nan
                for K in K_TEST
            ])
            for k in nn_grid
        }
        nn_star, _ = select_param_topK_spearman(sil_umap_by_k, nn_grid, m=M, n_k=n_k, threshold=THRESHOLD)
        rho_umap = spearmanr_safe(sil_umap_by_k[nn_star][eval_idx], ext_vals)
        main_results[cm_name][dname]['UMAP_Ordinary'] = rho_umap

        sensitivity_results[cm_name][dname]['r'] = {
            r: spearmanr_safe(sil_hard_by_r[r][eval_idx], ext_vals) for r in r_grid
        }
        sensitivity_results[cm_name][dname]['perplexity'] = {
            p: spearmanr_safe(sil_tsne_by_p[p][eval_idx], ext_vals) for p in perpl_grid
        }
        sensitivity_results[cm_name][dname]['n_neighbors'] = {
            k: spearmanr_safe(sil_umap_by_k[k][eval_idx], ext_vals) for k in nn_grid
        }
        param_star_results[cm_name][dname] = {
            'r_star':           r_star,   'rho_hard': rho_hard,
            'perplexity_star':  p_star,   'rho_tsne': rho_tsne,
            'n_neighbors_star': nn_star,  'rho_umap': rho_umap,
            'max_K_select':     max_K_s,  'max_K_eval': max_K_e,
        }

        print(f'{cm_name}: r*={r_star} (rho={rho_hard:.3f})  '
              f'perplexity*={p_star} (rho={rho_tsne:.3f})  '
              f'n_neighbors*={nn_star} (rho={rho_umap:.3f})')

    save_all()
    print(f'  setting {dname} done, progress saved.')

print('\nAll settings completed!')
save_all()
print('Results saved.')

# ── cell: summary-table ─────────────────────────────────────────────────
rows = []
for cm_name in main_results:
    for dname in main_results[cm_name]:
        meta = CASE_META.get(dname, {})
        for method in ALL_METHODS:
            rho = main_results[cm_name][dname].get(method)
            if rho is None or not np.isfinite(rho):
                continue
            rows.append({
                'clustering_method': cm_name, 'dataset': dname,
                'k*': meta.get('K'), 'N': meta.get('N'), 'D': meta.get('D'),
                'method': method, 'rho': round(float(rho), 4),
            })
summary_df = pd.DataFrame(rows)
print('\n=== Method-comparison summary (ACC) ===')
print(summary_df.to_string(index=False))

# ── cell: combined bar + landscape plots ────────────────────────────────
def plot_combined(cm_name, dname):
    '''3-panel figure: left = bar chart of the 4 methods' Spearman rho vs
    ACC; middle = TSNE_Ordinary rho vs. perplexity, with perplexity*
    marked; right = UMAP_Ordinary rho vs. n_neighbors, with n_neighbors*
    marked. Saved to figures/{CASE_TAG}/{cm_name}/{EXPERIMENT_TAG}/
    Combined_{dname}.{pdf,png}.'''
    bar_data    = main_results.get(cm_name, {}).get(dname, {})
    curve_data  = sensitivity_results.get(cm_name, {}).get(dname, {})
    star_data   = param_star_results.get(cm_name, {}).get(dname, {})
    if not bar_data:
        print(f'No results to plot for {cm_name} / {dname}')
        return

    meta = CASE_META.get(dname, {})
    k_star, n_samples, D = meta.get('K', '?'), meta.get('N', '?'), meta.get('D', '?')

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    ax = axes[0]
    available = [m for m in ALL_METHODS if bar_data.get(m) is not None and np.isfinite(bar_data[m])]
    if available:
        vals = [bar_data[m] for m in available]
        palette = sns.color_palette('tab10', n_colors=len(available))
        ax.bar(available, vals, color=palette, alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.axhline(0, color='grey', linewidth=1, linestyle='--')
        ax.set_ylabel('Spearman $\\rho$ with ACC', fontsize=13)
        ax.set_title('Method comparison', fontsize=13, fontweight='bold')
        ax.tick_params(axis='x', labelsize=9, rotation=30)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

    def landscape_panel(ax, curve, star_key, xlabel, title, color):
        if not curve:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=13, fontweight='bold')
            return
        xs = sorted(curve.keys())
        ax.plot(xs, [curve[x] for x in xs], marker='o', color=color, linewidth=2)
        star = star_data.get(star_key)
        if star is not None:
            ax.axvline(star, color=color, linestyle=':', linewidth=2.5,
                       label=f'${{{star_key}}}={star}$')
        oracle = max(curve, key=curve.get)
        ax.axvline(oracle, color=color, linestyle='--', linewidth=1.5, alpha=0.5,
                   label=f'oracle={oracle}')
        ax.axhline(0, color='grey', linewidth=1, linestyle='--')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Spearman $\\rho$ with ACC', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')

    landscape_panel(axes[1], curve_data.get('perplexity', {}), 'perplexity_star',
                    'perplexity', 'TSNE\\_Ordinary $\\rho$ vs. perplexity', 'seagreen')
    landscape_panel(axes[2], curve_data.get('n_neighbors', {}), 'n_neighbors_star',
                    '$n_{neighbors}$', 'UMAP\\_Ordinary $\\rho$ vs. $n_{neighbors}$', 'darkorange')

    fig.suptitle(f'{cm_name} — {dname}  ($k^*={k_star}$, $D={D}$, $N={n_samples}$)',
                fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out_dir = f'figures/{CASE_TAG}/{cm_name}/{EXPERIMENT_TAG}'
    os.makedirs(out_dir, exist_ok=True)
    fname = f'Combined_{dname}'
    plt.savefig(f'{out_dir}/{fname}.pdf', bbox_inches='tight')
    plt.savefig(f'{out_dir}/{fname}.png', dpi=150, bbox_inches='tight')
    print(f'Saved: {out_dir}/{fname}.pdf')
    plt.close(fig)


for cm_name in CLUSTERING_METHODS_TO_RUN:
    for dname in sorted(main_results.get(cm_name, {}).keys()):
        plot_combined(cm_name, dname)
