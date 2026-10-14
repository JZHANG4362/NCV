"""
Case 4 (real labeled data): companion to section_4_4_labeled_benchmarks/Benchmark_data_real.py.
Runs all 9 labeled real datasets and compares:

  1. rNN_Hard       -- the existing rNN-Hard silhouette baseline (r*,
                        Algorithm 1), on raw X.
  2-25. 8 alternative internal validity indices -- VRC, WG, WB, DBCV, CVDD,
        VIASCKDE, DCSI, DB -- each computed on THREE representations:
        <Index>_Original (raw X, NOT tuned -- no embedding involved),
        <Index>_TSNE, <Index>_UMAP.

UNLIKE codes_new_manifold_alt_indices_orig_kgrid (the untuned sibling):
perplexity/n_neighbors ARE tuned here, via Algorithm 1 -- and crucially,
EACH INDEX gets its OWN independently-selected perplexity*/n_neighbors*,
using THAT index's own rho-vs-K curve as the selection criterion. This is
the exact same relationship rNN_Hard's r* already has to rNN_Hard's own
curve -- not a single shared/proxy criterion. Concretely: for every
candidate perplexity value, a full t-SNE embedding is fit ONCE (shared
across indices), but then EVERY requested index's curve is computed on
that embedding (not just the eventual winner) -- so Algorithm 1 has, for
each index, a full param-vs-rho landscape to choose from, exactly as it
does for r*.

SCALE WARNING (per explicit instruction -- this IS much slower than the
untuned sibling): sweeping 10 perplexity + 8 n_neighbors values means up
to 18 embedding fits per dataset (same as the tuned-embedding siblings),
but ADDITIONALLY, at every one of those 18 embeddings, every requested
index must be computed -- for the O(N^2) indices (CVDD, VIASCKDE, DBCV,
and to a lesser extent DCSI) this multiplies their already-expensive
per-representation cost by up to 18x relative to the untuned sibling,
which only ever computes each index ONCE per representation.

FAST/SLOW INDEX-GROUP SPLIT (so early results are available sooner, and
so independent SLURM jobs can run in parallel instead of one job blocking
on the slowest index): pass 'fast', 'slow', or 'all' as the 2nd CLI
argument.
  FAST = VRC, WG, WB, DB   -- no O(N^2) precompute at all (VRC is a
                                  sklearn one-liner; WG/WB/DB are O(N*K)
                                  or O(N*D) loops over cluster centroids;
                                  no CVDD/VIASCKDE precompute is even
                                  built when running 'fast' alone).
  SLOW = DBCV, CVDD, VIASCKDE, DCSI  -- CVDD/VIASCKDE each precompute a
                                  dense N x N matrix (same order of cost
                                  as the rNN-Hard sweep) PER embedding,
                                  i.e. up to 18 times per dataset; DBCV
                                  (hdbscan) and DCSI (per-cluster pairwise
                                  cdist) are also non-trivial.
Both groups independently fit their OWN copy of the embedding sweep (kept
deliberately uncoordinated/stateless between jobs -- simpler and more
robust than trying to share cached embeddings across separate SLURM
processes). rNN_Hard is computed in BOTH groups' runs (cheap to just
redo; avoids a "which job owns rNN_Hard" dependency). Results merge
naturally: main_results[cm][dname] is loaded fresh each run and only the
requested group's index keys are written, so a later 'slow' run doesn't
clobber an earlier 'fast' run's columns (or vice versa) -- see save_all().

Usage (clustering method, then index group, then optional dataset subset
-- mirrors the untuned sibling's CLI with ONE extra positional arg):
    python manifold_tsne_umap_case4.py KMeans fast                # all 9 datasets, fast indices only
    python manifold_tsne_umap_case4.py KMeans slow                # all 9 datasets, slow indices only
    python manifold_tsne_umap_case4.py KMeans all Digits Iris     # both groups, 2 datasets
    python manifold_tsne_umap_case4.py all fast                   # all 3 clustering methods, fast only

Requires: numpy, scipy, scikit-learn, pandas, matplotlib, seaborn,
umap-learn, hdbscan (DBCV is NaN for every dataset if hdbscan is missing).
Raw data: downloaded/cached under data/case4_raw_datasets/ on first run
(Digits/Iris/Wine come bundled with scikit-learn and need no download).
"""

# ── cell: imports ─────────────────────────────────────────────────────
import os
import sys
import gzip
import pickle
import warnings
import urllib.request

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless-safe for remote/SLURM nodes (no display)
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import squareform, pdist, cdist

from sklearn.datasets import load_digits, load_iris, load_wine
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score, davies_bouldin_score, silhouette_score,
)
from sklearn.metrics.cluster import contingency_matrix
from sklearn.neighbors import NearestNeighbors, KernelDensity

import umap

sns.set_style('whitegrid')
warnings.filterwarnings('ignore', message='Graph is not fully connected')
warnings.filterwarnings('ignore', category=UserWarning, module='umap')

try:
    from hdbscan.validity import validity_index as _hdbscan_validity_index
    _HDBSCAN_AVAILABLE = True
except ImportError:
    _HDBSCAN_AVAILABLE = False
    print("hdbscan is not installed -- DBCV will be NaN for every dataset/"
          "representation. Run `pip install hdbscan` to enable it.")

# ── cell: helpers -- rNN baseline (verbatim from Benchmark_data_real.py) ──
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


# ── cell: helpers -- genuine t-SNE / UMAP embeddings (perplexity/
#          n_neighbors are candidate VALUES here -- tuning happens
#          per-index, below) ──────────────────────────────────────────
def tsne_embedding(X, perplexity, random_state=1):
    return TSNE(n_components=2, perplexity=perplexity, random_state=random_state,
                init='pca').fit_transform(X)


def umap_embedding(X, n_neighbors, random_state=1):
    return umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                      random_state=random_state).fit_transform(X)


# ── cell: 9 alternative internal validity indices (verbatim ports from
#          section_4_4_labeled_benchmarks/Benchmark_data_real.py) ─────────────────────────
def wemmert_gancarski_index(X, labels):
    '''Wemmert-Gancarski index (maximize).'''
    labels = np.asarray(labels)
    clusters = np.unique(labels)
    K = len(clusters)
    n = len(labels)
    if K < 2:
        return np.nan

    centroids = np.array([X[labels == k].mean(axis=0) for k in clusters])
    total = 0.0
    for idx, k in enumerate(clusters):
        mask = labels == k
        n_k = mask.sum()
        Xi = X[mask]
        d_own = np.linalg.norm(Xi - centroids[idx], axis=1)
        other_idx = [j for j in range(K) if j != idx]
        d_others = np.linalg.norm(
            Xi[:, None, :] - centroids[other_idx][None, :, :], axis=2)
        d_min_other = d_others.min(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            R = np.where(d_min_other > 0, d_own / d_min_other, np.inf)
        J_k = R.mean()
        total += max(0.0, n_k * (1.0 - J_k))
    return total / n


def wb_index(X, labels):
    '''WB-index (Zhao & Fränti, 2014): K * SSW / SSB (minimize).'''
    labels = np.asarray(labels)
    clusters = np.unique(labels)
    K = len(clusters)
    if K < 2:
        return np.nan

    grand_mean = X.mean(axis=0)
    ssw = 0.0
    ssb = 0.0
    for k in clusters:
        Xi = X[labels == k]
        c_k = Xi.mean(axis=0)
        ssw += np.sum((Xi - c_k) ** 2)
        ssb += Xi.shape[0] * np.sum((c_k - grand_mean) ** 2)
    if ssb == 0:
        return np.nan
    return K * ssw / ssb


def dbcv_index(X, labels):
    '''DBCV (Moulavi et al., 2014) via hdbscan's reference implementation
    (maximize).'''
    if not _HDBSCAN_AVAILABLE:
        return np.nan
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return np.nan
    try:
        return float(_hdbscan_validity_index(X.astype(np.float64), labels))
    except Exception:
        return np.nan


def _bottleneck_matrix_from_mst(mst_csr, n):
    '''Dense NxN matrix of minimax ("bottleneck") path values from an MST.'''
    mst_coo = mst_csr.tocoo()
    edges = sorted(zip(mst_coo.row, mst_coo.col, mst_coo.data), key=lambda e: e[2])

    parent = list(range(n))
    members = [[i] for i in range(n)]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    B = np.zeros((n, n))
    for u, v, w in edges:
        ru, rv = find(u), find(v)
        if ru == rv:
            continue
        if len(members[ru]) < len(members[rv]):
            ru, rv = rv, ru
        idx_a, idx_b = np.array(members[ru]), np.array(members[rv])
        B[np.ix_(idx_a, idx_b)] = w
        B[np.ix_(idx_b, idx_a)] = w
        parent[rv] = ru
        members[ru].extend(members[rv])
        members[rv] = None
    return B


def cvdd_precompute(X, k_nn=7):
    '''Precompute the density-involved distance (DD) matrix and the plain
    Euclidean distance matrix once per (dataset, embedding).'''
    n = X.shape[0]
    d = squareform(pdist(X, metric='euclidean'))

    k_use = min(k_nn, n - 1)
    nbrs = NearestNeighbors(n_neighbors=k_use + 1).fit(X)
    dists, _ = nbrs.kneighbors(X)
    Den = np.maximum(dists[:, 1:].mean(axis=1), 1e-12)
    fDen = Den / Den.max()

    Rel = Den[:, None] / Den[None, :]
    RelT = Den[None, :] / Den[:, None]
    fRel = 1.0 - np.exp(-(Rel + RelT - 2.0))
    nD = Den[:, None] + Den[None, :]
    drD = d + fRel * nD
    np.fill_diagonal(drD, 0.0)

    mst = minimum_spanning_tree(drD)
    conD = _bottleneck_matrix_from_mst(mst, n)
    DD = np.sqrt(np.outer(fDen, fDen)) * conD
    return {'d': d, 'DD': DD}


def cvdd_score(stats, labels):
    '''CVDD(pi) given precomputed stats and a partition. Maximize.'''
    labels = np.asarray(labels)
    clusters = np.unique(labels)
    K = len(clusters)
    if K < 2:
        return np.nan
    d, DD = stats['d'], stats['DD']

    idx_by_c = {c: np.where(labels == c)[0] for c in clusters}

    sep = np.zeros(K)
    for i, ci in enumerate(clusters):
        idx_i = idx_by_c[ci]
        best = np.inf
        for j, cj in enumerate(clusters):
            if i == j:
                continue
            sub = DD[np.ix_(idx_i, idx_by_c[cj])]
            best = min(best, sub.min())
        sep[i] = best

    com = np.zeros(K)
    for i, ci in enumerate(clusters):
        idx = idx_by_c[ci]
        ni = len(idx)
        if ni < 2:
            com[i] = 1e-12
            continue
        d_sub = d[np.ix_(idx, idx)]
        mst_sub = minimum_spanning_tree(d_sub)
        pD_sub = _bottleneck_matrix_from_mst(mst_sub, ni)
        iu = np.triu_indices(ni, k=1)
        vals = pD_sub[iu]
        mean_v = vals.mean()
        std_v = vals.std(ddof=1) if len(vals) > 1 else 0.0
        com[i] = max((1.0 / ni) * std_v * mean_v, 1e-12)

    return float(sep.sum() / com.sum())


def silverman_bandwidth(n, d):
    '''Dimension-adaptive KDE bandwidth (Silverman's rule of thumb).'''
    return n ** (-1.0 / (d + 4))


def viasckde_precompute(X, bandwidth=None, kernel='gaussian'):
    '''Precompute the min-max-scaled distance matrix and per-point KDE
    density once per (dataset, embedding).'''
    X01 = MinMaxScaler().fit_transform(X)
    d = squareform(pdist(X01, metric='euclidean'))
    np.fill_diagonal(d, np.inf)
    if bandwidth is None:
        bandwidth = silverman_bandwidth(X01.shape[0], X01.shape[1])
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel).fit(X01)
    log_dens = kde.score_samples(X01)
    dens = np.exp(log_dens - log_dens.max())
    return {'d': d, 'dens': dens}


def viasckde_score(stats, labels):
    '''VIASCKDE given precomputed stats and a partition. Maximize.'''
    labels = np.asarray(labels)
    n = len(labels)
    clusters = np.unique(labels)
    K = len(clusters)
    if K < 2:
        return np.nan
    d, dens = stats['d'], stats['dens']

    total = 0.0
    for ci in clusters:
        mask = labels == ci
        idx = np.where(mask)[0]
        ni = len(idx)

        dens_k = dens[idx]
        dmin, dmax = dens_k.min(), dens_k.max()
        kde_k = (dens_k - dmin) / (dmax - dmin) if dmax > dmin else np.zeros(ni)

        d_same = d[np.ix_(idx, idx)]
        a_i = d_same.min(axis=1)
        a_i = np.where(np.isinf(a_i), 0.0, a_i)

        other_idx = np.where(~mask)[0]
        d_other = d[np.ix_(idx, other_idx)]
        b_i = d_other.min(axis=1)

        denom = np.maximum(np.maximum(a_i, b_i), 1e-12)
        CoSeD = kde_k * (b_i - a_i) / denom
        total += ni * CoSeD.mean()

    return float(total / n)


def _dcsi_core_points(X_sub, min_pts):
    '''Core points of one class/cluster per DCSI's definition.'''
    n = X_sub.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=bool), None

    d_sub = squareform(pdist(X_sub, metric='euclidean'))
    k2 = min(2 * min_pts, n - 1)
    nbrs = NearestNeighbors(n_neighbors=k2 + 1).fit(X_sub)
    dists, _ = nbrs.kneighbors(X_sub)
    eps = np.median(dists[:, k2])

    k1 = min(min_pts, n - 1)
    nbrs1 = NearestNeighbors(n_neighbors=k1 + 1).fit(X_sub)
    dists1, _ = nbrs1.kneighbors(X_sub)
    core = dists1[:, k1] <= eps
    return core, d_sub


def _dcsi_conn(core_mask, d_sub):
    '''Max edge weight in the MST over a class's core points.'''
    idx = np.where(core_mask)[0]
    if len(idx) < 2:
        return 0.0
    d_core = d_sub[np.ix_(idx, idx)]
    mst = minimum_spanning_tree(d_core)
    return float(mst.data.max()) if mst.nnz > 0 else 0.0


def dcsi_index(X, labels, min_pts=5):
    '''DCSI (Gauss, Scheipl & Herrmann, 2025). Maximize (in [0,1)).'''
    labels = np.asarray(labels)
    clusters = np.unique(labels)
    K = len(clusters)
    if K < 2:
        return np.nan

    core_points_global = {}
    conn = {}
    for c in clusters:
        idx = np.where(labels == c)[0]
        X_sub = X[idx]
        core_mask, d_sub = _dcsi_core_points(X_sub, min_pts)
        core_points_global[c] = X_sub[core_mask] if d_sub is not None else np.empty((0, X.shape[1]))
        conn[c] = _dcsi_conn(core_mask, d_sub) if d_sub is not None else 0.0

    pairwise_vals = []
    for i in range(K):
        for j in range(i + 1, K):
            ci, cj = clusters[i], clusters[j]
            pts_i, pts_j = core_points_global[ci], core_points_global[cj]
            if len(pts_i) == 0 or len(pts_j) == 0:
                sep = 0.0
            else:
                sep = float(cdist(pts_i, pts_j, metric='euclidean').min())
            conn_max = max(conn[ci], conn[cj])
            if conn_max <= 0:
                q = np.inf if sep > 0 else 0.0
            else:
                q = sep / conn_max
            dcsi_pair = q / (1.0 + q) if np.isfinite(q) else 1.0
            pairwise_vals.append(dcsi_pair)

    return float(np.mean(pairwise_vals))


INDEX_NAMES = ['VRC', 'WG', 'WB', 'DBCV', 'CVDD', 'VIASCKDE', 'DCSI', 'DB']
FAST_INDICES = ['VRC', 'WG', 'WB', 'DB']
SLOW_INDICES = ['DBCV', 'CVDD', 'VIASCKDE', 'DCSI']
INDEX_GROUPS = {'fast': FAST_INDICES, 'slow': SLOW_INDICES, 'all': INDEX_NAMES}


def compute_index_curves(Z, labels_by_K, K_TEST, multi_cluster, index_names):
    '''Compute the requested subset of indices' raw per-K curves on ONE
    representation Z. CVDD/VIASCKDE precompute is skipped entirely unless
    actually requested (the whole point of the fast/slow split).'''
    cvdd_stats = cvdd_precompute(Z) if 'CVDD' in index_names else None
    viasckde_stats = viasckde_precompute(Z) if 'VIASCKDE' in index_names else None

    curves = {name: np.full(len(K_TEST), np.nan) for name in index_names}
    for i, K in enumerate(K_TEST):
        if not multi_cluster[K]:
            continue
        labels = labels_by_K[K]
        if 'VRC' in index_names:
            curves['VRC'][i] = calinski_harabasz_score(Z, labels)
        if 'WG' in index_names:
            curves['WG'][i] = wemmert_gancarski_index(Z, labels)
        if 'WB' in index_names:
            curves['WB'][i] = -wb_index(Z, labels)
        if 'DBCV' in index_names:
            curves['DBCV'][i] = dbcv_index(Z, labels)
        if 'CVDD' in index_names:
            curves['CVDD'][i] = cvdd_score(cvdd_stats, labels)
        if 'VIASCKDE' in index_names:
            curves['VIASCKDE'][i] = viasckde_score(viasckde_stats, labels)
        if 'DCSI' in index_names:
            curves['DCSI'][i] = dcsi_index(Z, labels)
        if 'DB' in index_names:
            curves['DB'][i] = -davies_bouldin_score(Z, labels)
    return curves


# ── cell: helpers -- Algorithm 1 (verbatim logic) ───────────────────────
def spearmanr_safe(a, b):
    '''Spearman correlation, returns NaN on degenerate input.'''
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(b)
    if mask.sum() < 2:
        return np.nan
    rho, _ = spearmanr(a[mask], b[mask])
    return float(rho)


def select_param_topK_spearman(sil_dict, param_values, m, n_k, threshold=0.95):
    '''Generalization of Algorithm 1 / select_r_topK_spearman. Logic
    copied verbatim.'''
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


def tune_index_over_grid(Y_by_param, param_grid, labels_by_K, K_TEST, multi_cluster,
                          index_names, eval_idx, ext_vals, m, threshold):
    '''For every candidate param value, fit-independent embedding Y_by_param[param]
    is already given; compute ALL requested indices' curves on it. Then,
    per index INDEPENDENTLY, select that index's own best param via
    Algorithm 1 (mirroring r*'s relationship to rNN_Hard's own curve) and
    record its rho at that param. Returns (param_star, rho_at_star,
    curves_by_param) -- all dicts keyed first by param then by index (for
    curves_by_param) or directly by index (for param_star/rho_at_star).'''
    n_k = len(K_TEST)
    curves_by_param = {
        param: compute_index_curves(Y_by_param[param], labels_by_K, K_TEST, multi_cluster, index_names)
        for param in param_grid
    }
    param_star, rho_at_star = {}, {}
    for idx in index_names:
        sil_dict = {param: curves_by_param[param][idx] for param in param_grid}
        p_star, _ = select_param_topK_spearman(sil_dict, param_grid, m=m, n_k=n_k, threshold=threshold)
        param_star[idx] = p_star
        rho_at_star[idx] = spearmanr_safe(sil_dict[p_star][eval_idx], ext_vals)
    return param_star, rho_at_star, curves_by_param


def max_K_eval(k_star):
    '''max(25, 1.75*k*) -- the reference paper's own K-range cap.'''
    return int(max(25, round(1.75 * k_star)))


def max_K_select(k_star):
    '''max_K_eval(k*) + 10 -- buffer ADDED ONLY for parameter selection.'''
    return max_K_eval(k_star) + 10


# ── cell: external metric (ACC only) ────────────────────────────────────
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

# ── cell: load-datasets (verbatim loaders from Benchmark_data_real.py) ──
RAW_DATA_DIR = '../../section_4_4_labeled_benchmarks/data/raw_datasets'
os.makedirs(RAW_DATA_DIR, exist_ok=True)

EXPECTED_TABLE_DATA = {
    'Digits':             dict(K=10, N=1797,  D=64),
    'MNIST-test':         dict(K=10, N=10000, D=784),
    'Ecoli':              dict(K=8,  N=336,   D=7),
    'Letter':             dict(K=26, N=20000, D=16),
    'Iris':               dict(K=3,  N=150,   D=4),
    'Fashion-MNIST-test': dict(K=10, N=10000, D=784),
    'Wine':               dict(K=3,  N=178,   D=13),
    'Libras':             dict(K=15, N=360,   D=90),
    'Vowel':              dict(K=11, N=990,   D=10),
}
DATASET_NAMES = list(EXPECTED_TABLE_DATA.keys())


def _fetch(url, fname):
    path = os.path.join(RAW_DATA_DIR, fname)
    if not os.path.exists(path):
        print(f'  downloading {url} ...')
        urllib.request.urlretrieve(url, path)
    return path


def _load_idx_images(gz_path):
    with gzip.open(gz_path) as f:
        _, n, rows, cols = np.frombuffer(f.read(16), dtype='>i4')
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)
    return data.astype(float)


def _load_idx_labels(gz_path):
    with gzip.open(gz_path) as f:
        _, n = np.frombuffer(f.read(8), dtype='>i4')
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def load_digits_dataset():
    d = load_digits()
    return d.data.astype(float), d.target


def load_iris_dataset():
    d = load_iris()
    return d.data.astype(float), d.target


def load_wine_dataset():
    d = load_wine()
    return d.data.astype(float), d.target


def load_mnist_test():
    path = _fetch('https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz', 'mnist.npz')
    with np.load(path) as f:
        X, y = f['x_test'], f['y_test']
    return X.reshape(X.shape[0], -1).astype(float), y


def load_fashion_mnist_test():
    img_path = _fetch(
        'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-images-idx3-ubyte.gz',
        'fmnist_test_images.gz')
    lbl_path = _fetch(
        'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-labels-idx1-ubyte.gz',
        'fmnist_test_labels.gz')
    return _load_idx_images(img_path), _load_idx_labels(lbl_path)


def load_ecoli():
    path = _fetch('https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data', 'ecoli.data')
    rows = [l.split() for l in open(path) if l.strip()]
    X = np.array([[float(v) for v in r[1:-1]] for r in rows])
    y_str = np.array([r[-1] for r in rows])
    _, y = np.unique(y_str, return_inverse=True)
    return X, y


def load_letter():
    path = _fetch(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data',
        'letter-recognition.data')
    rows = [l.strip().split(',') for l in open(path) if l.strip()]
    X = np.array([[float(v) for v in r[1:]] for r in rows])
    y_str = np.array([r[0] for r in rows])
    _, y = np.unique(y_str, return_inverse=True)
    return X, y


def load_libras():
    path = _fetch(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/libras/movement_libras.data',
        'movement_libras.data')
    rows = [l.strip().split(',') for l in open(path) if l.strip()]
    X = np.array([[float(v) for v in r[:-1]] for r in rows])
    y_str = np.array([r[-1] for r in rows])
    _, y = np.unique(y_str, return_inverse=True)
    return X, y


def load_vowel():
    path = _fetch(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/vowel/vowel-context.data',
        'vowel-context.data')
    rows = [l.strip().split() for l in open(path) if l.strip()]
    X = np.array([[float(v) for v in r[3:13]] for r in rows])
    y_str = np.array([r[13] for r in rows])
    _, y = np.unique(y_str, return_inverse=True)
    return X, y


DATASET_LOADERS = {
    'Digits':             load_digits_dataset,
    'MNIST-test':         load_mnist_test,
    'Ecoli':              load_ecoli,
    'Letter':             load_letter,
    'Iris':               load_iris_dataset,
    'Fashion-MNIST-test': load_fashion_mnist_test,
    'Wine':               load_wine_dataset,
    'Libras':             load_libras,
    'Vowel':              load_vowel,
}

# ── cell: CLI -- clustering method, index group, optional dataset subset ──
_cli_cm = sys.argv[1] if len(sys.argv) > 1 else 'all'
if _cli_cm == 'all':
    CLUSTERING_METHODS_TO_RUN = CLUSTERING_METHODS
else:
    assert _cli_cm in CLUSTERING_METHODS, f'Unknown clustering method {_cli_cm!r}'
    CLUSTERING_METHODS_TO_RUN = [_cli_cm]

_cli_group = sys.argv[2] if len(sys.argv) > 2 else 'all'
assert _cli_group in INDEX_GROUPS, f'Unknown index group {_cli_group!r}; choose fast/slow/all'
GROUP_INDICES = INDEX_GROUPS[_cli_group]

DATASET_NAMES_TO_LOAD = (
    [a for a in sys.argv[3:]] if len(sys.argv) > 3 else DATASET_NAMES
)

DATASETS = {}
print(f'{"dataset":20s} {"N":>7s} {"D":>4s} {"K*":>3s}  match?')
for name in DATASET_NAMES_TO_LOAD:
    X, y = DATASET_LOADERS[name]()
    DATASETS[name] = (X, y)
    exp = EXPECTED_TABLE_DATA[name]
    n, d, k = X.shape[0], X.shape[1], len(np.unique(y))
    ok = (n == exp['N']) and (d == exp['D']) and (k == exp['K'])
    print(f'{name:20s} {n:7d} {d:4d} {k:3d}  {"OK" if ok else "MISMATCH -- investigate!"}')
    assert ok, f'{name}: loaded (N={n},D={d},K={k}) does not match expected (N={exp["N"]},D={exp["D"]},K={exp["K"]})'

# ── cell: settings ────────────────────────────────────────────────────
R_VALUES           = list(range(5, 135, 5))    # r grid: 5, 10, ..., 130
PERPLEXITY_VALUES  = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
N_NEIGHBORS_VALUES = [5, 10, 15, 20, 30, 50, 75, 100]
M                  = 35
THRESHOLD          = 0.95

ALL_METHODS = ['rNN_Hard'] + [
    f'{idx}_{rep}' for idx in INDEX_NAMES for rep in ['Original', 'TSNE', 'UMAP']
]
DATASETS_TO_RUN = DATASET_NAMES_TO_LOAD
LARGE_DATASETS = ['MNIST-test', 'Fashion-MNIST-test', 'Letter']

CASE_TAG = 'case4'
EXPERIMENT_TAG = 'tsne_umap_alt_indices_tuned_origkgrid_v1'

JOB_TAG = os.environ.get('JOB_TAG', '').strip()
_run_tag = f'{EXPERIMENT_TAG}_{JOB_TAG}' if JOB_TAG else EXPERIMENT_TAG


def results_path_for(base_name):
    os.makedirs(f'data/{CASE_TAG}', exist_ok=True)
    return f'data/{CASE_TAG}/{base_name}_{_run_tag}.pkl'


print(f'{len(DATASETS_TO_RUN)} datasets to run: {DATASETS_TO_RUN}')
print(f'Of these, large/slow datasets present: {[d for d in DATASETS_TO_RUN if d in LARGE_DATASETS]}')
for name in DATASETS_TO_RUN:
    k_star = EXPECTED_TABLE_DATA[name]['K']
    print(f'  {name:20s} k*={k_star}  max_K_select={max_K_select(k_star)} max_K_eval={max_K_eval(k_star)}  m={M}')
print(f'R_VALUES: {R_VALUES[0]}..{R_VALUES[-1]} ({len(R_VALUES)})')
print(f'PERPLEXITY_VALUES: {PERPLEXITY_VALUES}')
print(f'N_NEIGHBORS_VALUES: {N_NEIGHBORS_VALUES}')
print(f'Clustering method(s) this session: {CLUSTERING_METHODS_TO_RUN}')
print(f'Index group this session: {_cli_group!r} -> {GROUP_INDICES}')

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
        main_results.setdefault(cm_name, {}).setdefault(dname, {})
        sensitivity_results.setdefault(cm_name, {}).setdefault(dname, {'r': {}, 'perplexity': {}, 'n_neighbors': {}})
        param_star_results.setdefault(cm_name, {}).setdefault(dname, {})

    print(f'\n=== Dataset {dname}  (k*={k_star}, D={Ddim}, N={n_samples}) ===')
    if dname in LARGE_DATASETS:
        print(f'  NOTE: {dname} is a large dataset -- the embedding sweep '
              f'(up to 18 fits) x the {_cli_group!r} index group is the '
              f'dominant cost here (see module SCALE WARNING).')

    r_grid     = [r for r in R_VALUES if r < n_samples]
    perpl_grid = [p for p in PERPLEXITY_VALUES if 3 * p < n_samples - 1]
    nn_grid    = [k for k in N_NEIGHBORS_VALUES if k < n_samples - 1]

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

        # ---- rNN_Hard: always computed (cheap to redo across group runs);
        #      one r's dense NxN matrix at a time (memory fix, see sibling
        #      packages) ----
        sil_hard_by_r = {}
        for r in r_grid:
            D_r = rnn_affinity_distance(X, r, use_gaussian=False)
            sil_hard_by_r[r] = np.array([
                silhouette_score(D_r, labels_by_K[K], metric='precomputed') if multi_cluster[K] else np.nan
                for K in K_TEST
            ])
            del D_r
        r_star, _ = select_param_topK_spearman(sil_hard_by_r, r_grid, m=M, n_k=n_k, threshold=THRESHOLD)
        rho_hard = spearmanr_safe(sil_hard_by_r[r_star][eval_idx], ext_vals)
        main_results[cm_name][dname]['rNN_Hard'] = rho_hard

        # ---- Original (raw X): NOT tuned, no embedding -- computed once,
        #      only for the requested index group ----
        curves_orig = compute_index_curves(X, labels_by_K, K_TEST, multi_cluster, GROUP_INDICES)
        for idx in GROUP_INDICES:
            main_results[cm_name][dname][f'{idx}_Original'] = spearmanr_safe(
                curves_orig[idx][eval_idx], ext_vals)
        del curves_orig

        # ---- t-SNE: per-index tuning over perpl_grid ----
        p_star, rho_tsne, curves_tsne_by_p = tune_index_over_grid(
            Y_tsne_by_p, perpl_grid, labels_by_K, K_TEST, multi_cluster,
            GROUP_INDICES, eval_idx, ext_vals, M, THRESHOLD)
        for idx in GROUP_INDICES:
            main_results[cm_name][dname][f'{idx}_TSNE'] = rho_tsne[idx]

        # ---- UMAP: per-index tuning over nn_grid ----
        nn_star, rho_umap, curves_umap_by_k = tune_index_over_grid(
            Y_umap_by_k, nn_grid, labels_by_K, K_TEST, multi_cluster,
            GROUP_INDICES, eval_idx, ext_vals, M, THRESHOLD)
        for idx in GROUP_INDICES:
            main_results[cm_name][dname][f'{idx}_UMAP'] = rho_umap[idx]

        sensitivity_results[cm_name][dname]['r'] = {
            r: spearmanr_safe(sil_hard_by_r[r][eval_idx], ext_vals) for r in r_grid
        }
        sensitivity_results[cm_name][dname].setdefault('perplexity', {})
        sensitivity_results[cm_name][dname].setdefault('n_neighbors', {})
        for idx in GROUP_INDICES:
            sensitivity_results[cm_name][dname]['perplexity'][idx] = {
                p: spearmanr_safe(curves_tsne_by_p[p][idx][eval_idx], ext_vals) for p in perpl_grid
            }
            sensitivity_results[cm_name][dname]['n_neighbors'][idx] = {
                k: spearmanr_safe(curves_umap_by_k[k][idx][eval_idx], ext_vals) for k in nn_grid
            }

        param_star_results[cm_name][dname].update({
            'r_star': r_star, 'rho_hard': rho_hard,
            'max_K_select': max_K_s, 'max_K_eval': max_K_e,
        })
        param_star_results[cm_name][dname].setdefault('perplexity_star', {}).update(p_star)
        param_star_results[cm_name][dname].setdefault('n_neighbors_star', {}).update(nn_star)

        print(f'  {cm_name} [{_cli_group}]: r*={r_star} (rho={rho_hard:.3f})')
        for idx in GROUP_INDICES:
            print(f'    {idx:10s} Orig={main_results[cm_name][dname][f"{idx}_Original"]:+.3f}  '
                  f'p*={p_star[idx]:>4} TSNE={rho_tsne[idx]:+.3f}  '
                  f'nn*={nn_star[idx]:>4} UMAP={rho_umap[idx]:+.3f}')

    save_all()
    print(f'  dataset {dname} done, progress saved.')

print('\nAll datasets completed!')
save_all()
print('Results saved.')

# ── cell: summary-table ─────────────────────────────────────────────────
rows = []
for cm_name in main_results:
    for dname in main_results[cm_name]:
        meta = EXPECTED_TABLE_DATA.get(dname, {})
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

# ── cell: combined bar + r-landscape plot ───────────────────────────────
def plot_combined(cm_name, dname):
    '''2-panel figure: left = bar chart of whatever methods are present
    for this (cm, dname) so far (may be a subset if only one index group
    has been retrieved); right = rNN_Hard rho vs. r. Saved to
    figures/{CASE_TAG}/{cm_name}/{EXPERIMENT_TAG}/Combined_{dname}.{pdf,png}.'''
    bar_data    = main_results.get(cm_name, {}).get(dname, {})
    curve_data  = sensitivity_results.get(cm_name, {}).get(dname, {})
    star_data   = param_star_results.get(cm_name, {}).get(dname, {})
    if not bar_data:
        print(f'No results to plot for {cm_name} / {dname}')
        return

    meta = EXPECTED_TABLE_DATA.get(dname, {})
    k_star, n_samples, D = meta.get('K', '?'), meta.get('N', '?'), meta.get('D', '?')

    fig, axes = plt.subplots(1, 2, figsize=(26, 7), gridspec_kw={'width_ratios': [3, 1]})

    ax = axes[0]
    available = [m for m in ALL_METHODS if bar_data.get(m) is not None and np.isfinite(bar_data[m])]
    if available:
        vals = [bar_data[m] for m in available]
        palette = sns.color_palette('tab20', n_colors=len(available))
        ax.bar(available, vals, color=palette, alpha=0.9, edgecolor='black', linewidth=0.4)
        ax.axhline(0, color='grey', linewidth=1, linestyle='--')
        ax.set_ylabel('Spearman $\\rho$ with ACC', fontsize=13)
        ax.set_title(f'Method comparison ({len(available)}/{len(ALL_METHODS)} methods retrieved so far)',
                     fontsize=13, fontweight='bold')
        ax.tick_params(axis='x', labelsize=7, rotation=90)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

    ax2 = axes[1]
    curve = curve_data.get('r', {})
    if curve:
        xs = sorted(curve.keys())
        ax2.plot(xs, [curve[x] for x in xs], marker='o', color='steelblue', linewidth=2)
        star = star_data.get('r_star')
        if star is not None:
            ax2.axvline(star, color='steelblue', linestyle=':', linewidth=2.5, label=f'$r^*={star}$')
        oracle = max(curve, key=curve.get)
        ax2.axvline(oracle, color='steelblue', linestyle='--', linewidth=1.5, alpha=0.5, label=f'oracle={oracle}')
        ax2.axhline(0, color='grey', linewidth=1, linestyle='--')
        ax2.set_xlabel('$r$', fontsize=12)
        ax2.set_ylabel('Spearman $\\rho$ with ACC', fontsize=12)
        ax2.set_title('rNN\\_Hard $\\rho$ vs. $r$', fontsize=13, fontweight='bold')
        ax2.set_ylim(-1.05, 1.05)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9, loc='best')
    else:
        ax2.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax2.transAxes)

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
