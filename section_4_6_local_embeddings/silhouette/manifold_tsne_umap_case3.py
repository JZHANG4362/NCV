"""
Case 3 (nonconvex data): companion to section_4_3_nonconvex_benchmarks/partB_fixed_datasets_
comparison.py. Runs all 10 fixed real (non-simulated) 2D/3D nonconvex-
cluster datasets (CVDD Table 3: pathbased, spiral, aggregation, atom, lsun,
zelnik1, zelnik5, zelnik6, triangle1, longsquare) and compares 4
silhouette-family validity indices against ACC:

  1. Silhouette     -- ordinary silhouette, Euclidean distance on X.
  2. rNN_Hard       -- the existing rNN-Hard silhouette baseline (r*,
                        Algorithm 1).
  3. TSNE_Ordinary  -- ordinary (Euclidean) silhouette on the 2D t-SNE
                        embedding of X. UNLIKE the earlier affinity-based
                        variant of this experiment (codes_new_manifold /
                        codes_new_manifold_new_grid's TSNE_JointProb),
                        this does NOT replace the pairwise distance with
                        any t-SNE-derived affinity matrix -- it is the
                        genuine t-SNE embedding, run through ordinary
                        Euclidean silhouette, exactly like Method 1
                        (TSNE_Ordinary) in the earlier experiment. The
                        difference from that earlier Method 1 is that
                        perplexity is now TUNED (perplexity* selected via
                        Algorithm 1, swept over PERPLEXITY_VALUES) instead
                        of fixed at 30.
  4. UMAP_Ordinary  -- ordinary (Euclidean) silhouette on the 2D UMAP
                        embedding of X, same relationship to the earlier
                        experiment's Method 3: genuine UMAP embedding, no
                        affinity substitution, but now with n_neighbors
                        TUNED (n_neighbors* via Algorithm 1, swept over
                        N_NEIGHBORS_VALUES) instead of fixed at 15.

This experiment deliberately does NOT include the affinity-substitution
methods (TSNE_JointProb / UMAP_FuzzyUnion) from codes_new_manifold /
codes_new_manifold_new_grid -- per explicit instruction, this is the
"original version": genuine t-SNE/UMAP embeddings, evaluated with the
ordinary Euclidean silhouette, with their own native hyperparameter
(perplexity / n_neighbors) tuned by the SAME Algorithm 1 criterion used
for rNN's r*, rather than substituting either method's internal affinity
matrix in for the pairwise distance.

Only ACC is used as the external metric (not the full 7-metric battery of
partB_fixed_datasets_comparison.py).

Parameter-selection algorithm (perplexity* for t-SNE, n_neighbors* for
UMAP): EXACTLY Algorithm 1 (select_r_topK_spearman), generalized in name
only (select_param_topK_spearman) -- identical convention (m=35,
threshold=0.95, same EVALUATION/SELECTION K-ranges) as the paper's own r*
selection. Applied here to the silhouette-vs-K curve computed on the
EMBEDDING at each candidate perplexity/n_neighbors, not on an affinity
matrix.

SCALE NOTE: unlike the affinity-substitution variant (where each grid
point only requires building one affinity/distance matrix, a single
closed-form step), each grid point here requires a FULL t-SNE or UMAP
embedding optimization (gradient descent iterations). Sweeping 10
perplexity values x 8 n_neighbors values means up to 18 separate
embedding fits per dataset -- slower than the affinity-substitution
version, though sklearn's default Barnes-Hut t-SNE and UMAP's approximate
nearest-neighbor search both scale far better than the affinity variant's
exact O(N^2) joint-probability computation.

Compute-batching: D_hard(r) / the t-SNE embeddings / the UMAP embeddings
all depend only on X, never on the clustering method or K -- built ONCE
per dataset here and reused across all 3 clustering methods.

Usage (clustering method via CLI arg, default = all 3 sequentially):
    python manifold_tsne_umap_case3.py             # all 3 methods
    python manifold_tsne_umap_case3.py KMeans
    python manifold_tsne_umap_case3.py Spectral
    python manifold_tsne_umap_case3.py Agglomerative

Requires: numpy, scipy, scikit-learn, pandas, matplotlib, seaborn, umap-learn
Raw data: bundled at data/case3_raw_datasets/*.arff.
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

from scipy.io import arff
from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment

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

# ── cell: helpers -- rNN baseline (verbatim from partB_fixed_datasets_comparison.py) ──
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
    neighborhood-scale parameter -- here, t-SNE perplexity and UMAP
    n_neighbors, not just rNN's r. Logic copied verbatim.'''
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

# ── cell: load-datasets (identical to partB_fixed_datasets_comparison.py) ──
RAW_DATA_DIR = '../../section_4_3_nonconvex_benchmarks/data/raw_datasets'

EXPECTED_TABLE3 = {
    'pathbased':   dict(K=3, N=300,  D=2),
    'spiral':      dict(K=3, N=312,  D=2),
    'aggregation': dict(K=7, N=788,  D=2),
    'atom':        dict(K=2, N=800,  D=3),
    'lsun':        dict(K=3, N=400,  D=2),
    'zelnik1':     dict(K=3, N=299,  D=2),
    'zelnik5':     dict(K=4, N=512,  D=2),
    'zelnik6':     dict(K=3, N=238,  D=2),
    'triangle1':   dict(K=4, N=1000, D=2),
    'longsquare':  dict(K=6, N=900,  D=2),
}
DATASET_NAMES = list(EXPECTED_TABLE3.keys())


def load_arff_dataset(path):
    data, meta = arff.loadarff(path)
    attr_names = meta.names()
    class_col = attr_names[-1]
    feature_cols = attr_names[:-1]
    X = np.column_stack([data[c].astype(float) for c in feature_cols])
    y_raw = data[class_col]
    y_str = np.array([v.decode('utf-8') if isinstance(v, bytes) else str(v) for v in y_raw])
    _, y = np.unique(y_str, return_inverse=True)
    return X, y


DATASETS = {}
print(f'{"dataset":12s} {"N":>6s} {"D":>3s} {"K*":>3s}  match?')
for name in DATASET_NAMES:
    X, y = load_arff_dataset(os.path.join(RAW_DATA_DIR, f'{name}.arff'))
    DATASETS[name] = (X, y)
    exp = EXPECTED_TABLE3[name]
    n, d, k = X.shape[0], X.shape[1], len(np.unique(y))
    ok = (n == exp['N']) and (d == exp['D']) and (k == exp['K'])
    print(f'{name:12s} {n:6d} {d:3d} {k:3d}  {"OK" if ok else "MISMATCH -- investigate!"}')
    assert ok, f'{name}: loaded (N={n},D={d},K={k}) does not match CVDD Table 3 (N={exp["N"]},D={exp["D"]},K={exp["K"]})'

# ── cell: settings ────────────────────────────────────────────────────
R_VALUES           = list(range(5, 135, 5))    # r grid: 5, 10, ..., 130
PERPLEXITY_VALUES  = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
N_NEIGHBORS_VALUES = [5, 10, 15, 20, 30, 50, 75, 100]
M                  = 35
THRESHOLD          = 0.95

ALL_METHODS = ['Silhouette', 'rNN_Hard', 'TSNE_Ordinary', 'UMAP_Ordinary']
DATASETS_TO_RUN = DATASET_NAMES

if len(sys.argv) > 1:
    assert sys.argv[1] in CLUSTERING_METHODS, f'Unknown clustering method {sys.argv[1]!r}'
    CLUSTERING_METHODS_TO_RUN = [sys.argv[1]]
else:
    CLUSTERING_METHODS_TO_RUN = CLUSTERING_METHODS

CASE_TAG = 'case3'
EXPERIMENT_TAG = 'tsne_umap_tuned_embedding_origkgrid_v1'


def results_path_for(base_name):
    os.makedirs(f'data/{CASE_TAG}', exist_ok=True)
    return f'data/{CASE_TAG}/{base_name}_{EXPERIMENT_TAG}.pkl'


print(f'{len(DATASETS_TO_RUN)} datasets: {DATASETS_TO_RUN}')
for name in DATASETS_TO_RUN:
    k_star = EXPECTED_TABLE3[name]['K']
    print(f'  {name:12s} k*={k_star}  max_K_select={max_K_select(k_star)} max_K_eval={max_K_eval(k_star)}  m={M}')
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


# ── cell: main-loop ─────────────────────────────────────────────────────
# NOTE: D_hard(r) and the t-SNE/UMAP embedding sweeps depend only on X --
# built ONCE per dataset here, shared across all 3 clustering methods.
for dname in DATASETS_TO_RUN:
    X_raw, y = DATASETS[dname]
    X = StandardScaler().fit_transform(X_raw)
    n_samples, Ddim = X.shape
    k_star = len(np.unique(y))
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

    print(f'\n=== Dataset {dname}  (k*={k_star}, D={Ddim}, N={n_samples}) ===')

    r_grid     = [r for r in R_VALUES if r < n_samples]
    perpl_grid = [p for p in PERPLEXITY_VALUES if 3 * p < n_samples - 1]
    nn_grid    = [k for k in N_NEIGHBORS_VALUES if k < n_samples - 1]

    # ---- rNN sweep (unchanged) ----
    print(f'  building rNN sweep ({len(r_grid)} r values)...')
    D_hard_by_r = {r: rnn_affinity_distance(X, r, use_gaussian=False) for r in r_grid}

    # ---- genuine t-SNE / UMAP embedding sweeps (one full embedding per
    #      candidate perplexity / n_neighbors -- see module SCALE NOTE) ----
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

        # ---- Silhouette (untuned baseline) ----
        sil_curve = np.array([
            silhouette_score(X, labels_by_K[K], metric='euclidean') if multi_cluster[K] else np.nan
            for K in K_TEST
        ])
        main_results[cm_name][dname]['Silhouette'] = spearmanr_safe(sil_curve[eval_idx], ext_vals)

        # ---- rNN_Hard sweep + Algorithm 1 selection ----
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

        # ---- TSNE_Ordinary sweep + Algorithm 1 selection of perplexity* ----
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

        # ---- UMAP_Ordinary sweep + Algorithm 1 selection of n_neighbors* ----
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

        print(f'  {cm_name}: r*={r_star} (rho={rho_hard:.3f})  '
              f'perplexity*={p_star} (rho={rho_tsne:.3f})  '
              f'n_neighbors*={nn_star} (rho={rho_umap:.3f})')

    save_all()
    print(f'  dataset {dname} done, progress saved.')

print('\nAll datasets completed!')
save_all()
print('Results saved.')

# ── cell: summary-table ─────────────────────────────────────────────────
rows = []
for cm_name in main_results:
    for dname in main_results[cm_name]:
        meta = EXPECTED_TABLE3.get(dname, {})
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
    '''3-panel figure for one (clustering_method, dataset): left = bar chart
    of the 4 methods' Spearman rho vs ACC; middle = TSNE_Ordinary rho vs.
    perplexity, with perplexity* marked; right = UMAP_Ordinary rho vs.
    n_neighbors, with n_neighbors* marked. Saved to
    figures/{CASE_TAG}/{cm_name}/{EXPERIMENT_TAG}/Combined_{dname}.{pdf,png}.'''
    bar_data    = main_results.get(cm_name, {}).get(dname, {})
    curve_data  = sensitivity_results.get(cm_name, {}).get(dname, {})
    star_data   = param_star_results.get(cm_name, {}).get(dname, {})
    if not bar_data:
        print(f'No results to plot for {cm_name} / {dname}')
        return

    meta = EXPECTED_TABLE3.get(dname, {})
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
