"""
Standalone script for partB_fixed_datasets_comparison.ipynb, extracted for
remote (SLURM) execution. Runs all 10 fixed real (non-simulated) 2D/3D
nonconvex-cluster datasets (CVDD Table 3: pathbased, spiral, aggregation,
atom, lsun, zelnik1, zelnik5, zelnik6, triangle1, longsquare), all 10
validity indices (both rNN_Hard and rNN_Soft), against ALL 7 external
metrics (Jaccard, SS3, ARI, NMI, Powers, NID, ACC) in a single pass -- see
"Multi-metric batching" below. Clustering method is selected via a
command-line argument, one of KMeans / Spectral / Agglomerative (defaults
to running all three sequentially if omitted -- this whole script is fast,
on the order of minutes per clustering method, since there is no N_REP
loop: each dataset is fixed, so every (dataset, clustering_method) pair is
computed exactly once with a fixed seed):

    python partB_fixed_datasets_comparison.py             # all 3 methods
    python partB_fixed_datasets_comparison.py KMeans       # one method
    python partB_fixed_datasets_comparison.py Spectral
    python partB_fixed_datasets_comparison.py Agglomerative

Multi-metric batching: r* selection (select_r_topK_spearman) and every
validity index's raw score curve -- including the full rNN Hard/Soft
r-sensitivity sweep over R_VALUES, the single most expensive part of this
script -- depend only on (X, labels_by_K) for a given dataset, NOT on
which external metric is being validated against. So instead of rerunning
clustering + the rNN affinity sweep once per external metric (7x the
cost), this script computes each index's raw K-curve, the full r-
sensitivity sweep, and rNN's r* ONCE per (dataset, clustering_method),
then loops over all 7 EXTERNAL_METRIC_FUNCS to compute the Spearman rho
against each metric's ext_vals. Results for each metric are still saved
to their own separate pickle files (main / sensitivity / r* results),
preserving the existing per-metric file naming convention exactly (see
results_path_for below), so no downstream code (main.tex table
generation) needs to change.

r* selection method (identical convention to Part A / Part C):
- EVALUATION (the reported rho for every index, not just rNN): computed
  only over K = 2..max(25, 1.75*k*) -- the reference paper's ("Benchmarking
  of Clustering Validity Measures Revisited") own K-range cap, Section 3.2.
- SELECTION of r* (rNN_Hard and rNN_Soft independently): select_r_topK_spearman
  (top-m monotonic-decrease-vs-ideal-sequence Spearman criterion), m=35,
  over a wider range K = 2..(max(25,1.75*k*)+10).

The main loop checkpoints after every dataset, so an interrupted/timed-out
job can simply be resubmitted -- completed datasets are preserved in the
results pickles.

Requires: numpy, scipy, scikit-learn, pandas, matplotlib, seaborn, hdbscan
"""


# ── cell: imports ─────────────────────────────────────────────────────
import os
import sys
import pickle
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless-safe for remote/SLURM nodes (no display)
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.io import arff
from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import squareform, pdist, cdist

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score, calinski_harabasz_score,
    normalized_mutual_info_score, mutual_info_score,
)
from sklearn.metrics.cluster import pair_confusion_matrix, entropy, contingency_matrix
from sklearn.neighbors import NearestNeighbors, KernelDensity

sns.set_style('whitegrid')

warnings.filterwarnings('ignore', message='Graph is not fully connected')

try:
    from hdbscan.validity import validity_index as _hdbscan_validity_index
    _HDBSCAN_AVAILABLE = True
except ImportError:
    _HDBSCAN_AVAILABLE = False
    print("hdbscan is not installed — DBCV will be reported as NaN for every "
          "dataset. Run `pip install hdbscan` to enable it.")

# ── cell: helpers ─────────────────────────────────────────────────────
def simulate_type2_data(n_samples, n_features, k_star, compactness, random_state=1):
    '''Approximate a Type-2-style dataset using Gaussian blobs; identical to
    type2_case_comparison.ipynb's simulate_type2_data.

    NOTE: vestigial / not called anywhere in this script. This file only
    ever clusters the 10 fixed real (SIPU) datasets loaded below, never
    simulated blobs -- this function is carried over unmodified from the
    simulated-data sibling scripts (Case 2's Part A / Type-2 grid) purely
    for template consistency, and would raise NameError if invoked here
    since make_blobs is not imported in this file. Left in place rather
    than deleted so this script's helper section stays a recognizable
    match to the other Parts' scripts.'''
    X, y = make_blobs(n_samples=n_samples, n_features=n_features,
                      centers=k_star, cluster_std=compactness,
                      random_state=random_state)
    return X, y


def rnn_affinity_distance(X, r, use_gaussian=False):
    '''
    Build rNN affinity distance matrix (precomputed, zeros on diagonal).
    Hard (binary): distance = 0 when either point lists the other among
    its r nearest neighbors (the symmetric union), and 1 otherwise.
    Soft (Gaussian): Gaussian-weighted affinity converted to distance.
    '''
    n = X.shape[0]
    # Exclude the query point itself: r denotes exactly r other observations.
    # "Corrected r convention" enforced package-wide (see
    # ../tools/verify_r_convention.py): sklearn's kneighbors always returns
    # the query point itself as its own closest neighbor at column 0, so to
    # get exactly r NON-self neighbors we must ask sklearn for r+1 neighbors
    # and then drop that self column before using the results.
    if not 1 <= r < n:
        raise ValueError(f'r must satisfy 1 <= r < n; got r={r}, n={n}')
    nbrs = NearestNeighbors(n_neighbors=r + 1, algorithm='auto').fit(X)
    dists, idx = nbrs.kneighbors(X)
    dists, idx = dists[:, 1:], idx[:, 1:]

    # Build the (asymmetric, pre-symmetrization) affinity matrix: for each
    # point i, mark/weight its r nearest neighbors j.
    A = np.zeros((n, n))
    if use_gaussian:
        # Soft variant: Gaussian kernel weight using i's own r-th neighbor
        # distance as the local bandwidth sigma (adaptive per point).
        for i in range(n):
            sigma = dists[i, -1] or 1e-10
            for j_pos, j in enumerate(idx[i]):
                A[i, j] = np.exp(-dists[i, j_pos]**2 / (2 * sigma**2))
    else:
        # Hard variant: binary indicator, 1.0 if j is among i's r nearest.
        for i in range(n):
            A[i, idx[i]] = 1.0

    # Symmetrize via elementwise max (the "symmetric union": i-j is an edge
    # if EITHER i lists j or j lists i among its r nearest neighbors), then
    # min-max normalize to [0,1] and convert affinity -> distance (1 - A) so
    # the result can be fed to sklearn's silhouette_score as a precomputed
    # distance matrix.
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0)
    a_min, a_max = A.min(), A.max()
    if a_max > a_min:
        A = (A - a_min) / (a_max - a_min)
    D = 1.0 - A
    np.fill_diagonal(D, 0)
    return D


def spearmanr_safe(a, b):
    '''Spearman correlation, returns NaN on degenerate input.'''
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(b)
    if mask.sum() < 2:
        return np.nan
    rho, _ = spearmanr(a[mask], b[mask])
    return float(rho)


def select_r_topK_spearman(sil_dict, r_values, m, n_k, threshold=0.95):
    '''
    Select r* using the top-m overestimate Spearman criterion: checks
    whether the silhouette curve decreases monotonically across the top-m
    largest K's via Spearman correlation against the ideal strictly-
    decreasing sequence [m,...,1]. Picks the smallest r whose score
    clears threshold; falls back to argmax over r otherwise.

    m is clipped to n_k-1 (m_eff) so this is always well-defined even
    when the requested window is wider than the available K range.

    This is the manuscript's Algorithm 1 (r* selection), run independently
    per (dataset, clustering_method) and per rNN variant (Hard/Soft). It is
    intentionally blind to every external metric: it only ever looks at the
    rNN-affinity silhouette curve's own shape (sil_dict), never at ACC/ARI/
    etc. -- the idea being that an over-large K should make ANY well-behaved
    internal validity curve trend downward across the largest-K tail, so
    picking the r whose curve best matches that "ideal" decreasing shape is
    a purely internal, label-free proxy for picking a good r.
    '''
    # sil_dict may not have every candidate r_values entry (e.g. a caller
    # passed a partial sweep); restrict to r's actually present.
    avail = sorted(r for r in r_values if r in sil_dict)
    if not avail:
        return r_values[0], {}
    # Effective window size: min(m, n_k-1), floored at 2 so the Spearman
    # correlation below is always computed over at least 2 points.
    m_eff   = max(2, min(m, n_k - 1))
    # Indices of the top-m_eff largest K's within the K_TEST grid (the
    # "large-K tail" where we expect the curve to be monotonically falling).
    top_idx = list(range(n_k - m_eff, n_k))
    # The "ideal" strictly-decreasing reference sequence [m_eff, ..., 1]
    # that a perfectly-behaved silhouette-vs-K curve would rank-correlate
    # with perfectly (rho=1) on that tail.
    ideal   = np.arange(m_eff, 0, -1, dtype=float)
    c_vals  = {}
    for r in avail:
        obs        = np.asarray(sil_dict[r], dtype=float)[top_idx]
        rho        = spearmanr_safe(obs, ideal)
        # Rank a degenerate/NaN correlation below every real score (-2 is
        # outside Spearman's [-1,1] range) so it never wins the argmax
        # fallback below, without needing separate NaN-handling logic.
        c_vals[r]  = rho if not np.isnan(rho) else -2.0
    # Prefer the SMALLEST r that clears the monotonicity threshold (a
    # smaller neighborhood is preferred when several r's are "good enough");
    # if none clears it, fall back to whichever r best matches the ideal
    # decreasing shape.
    r_star = next((r for r in avail if c_vals[r] >= threshold), None)
    if r_star is None:
        r_star = max(avail, key=lambda r: c_vals[r])
    return r_star, c_vals


def max_K_eval(k_star):
    '''max(25, 1.75*k*) -- the reference paper's own K-range cap (Section
    3.2), used as the EVALUATION scope: every reported rho (all 10
    indices, not just rNN) is computed only over K=2..max_K_eval.'''
    return int(max(25, round(1.75 * k_star)))


def max_K_select(k_star):
    '''max_K_eval(k*) + 10 -- a small buffer ADDED ONLY for r* selection
    (rNN_Hard / rNN_Soft), giving the m=35 window headroom past the
    evaluation boundary while still overlapping substantially back into
    the evaluation range.'''
    return max_K_eval(k_star) + 10

# ── cell: custom-indices ──────────────────────────────────────────────
def wemmert_gancarski_index(X, labels):
    '''
    Wemmert-Gancarski index (maximize). For each point, the ratio of its
    distance to its own cluster centroid over the minimum distance to any
    other cluster centroid; averaged (with a max(0, .) floor) per cluster
    and combined with cluster-size weights.
    '''
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
    '''
    DBCV (Moulavi et al., 2014) via hdbscan's reference implementation
    (maximize). Returns NaN if hdbscan is unavailable or the labeling is
    degenerate.
    '''
    if not _HDBSCAN_AVAILABLE:
        return np.nan
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return np.nan
    try:
        return float(_hdbscan_validity_index(X.astype(np.float64), labels))
    except Exception:
        return np.nan

# ── cell: cvdd-index ──────────────────────────────────────────────────
def _bottleneck_matrix_from_mst(mst_csr, n):
    '''
    Given a sparse MST (scipy.sparse.csgraph.minimum_spanning_tree), return
    the dense NxN matrix of minimax ("bottleneck") path values, i.e. the
    maximum edge weight on the unique path between every pair of nodes in
    the MST.

    Implementation: process MST edges in ascending weight order (Kruskal
    reconstruction). Whenever two components merge at weight w, every pair
    (x in component A, y in component B) gets bottleneck value w -- and
    every such pair is assigned exactly once across the whole run, giving
    an exact O(n^2) algorithm.
    '''
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
    '''
    Precompute the density-involved distance (DD) matrix and the plain
    Euclidean distance matrix once per dataset -- both are independent of
    any particular partition/K, so this is shared across every clustering
    method and every K in the sweep for a given replicate.
    '''
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
    '''CVDD(pi) given precomputed stats (from cvdd_precompute) and a
    partition. Maximize.'''
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

# ── cell: viasckde-index ──────────────────────────────────────────────
def silverman_bandwidth(n, d):
    '''
    Dimension-adaptive KDE bandwidth (Silverman's rule of thumb,
    isotropic/simplified form): h = n^(-1/(d+4)). Data is assumed
    min-max-scaled to [0,1]^d, so no additional per-feature std/scale
    factor is applied.

    A FIXED bandwidth (e.g. 0.05) is only sensible at low d: at high d
    (e.g. 784 for MNIST/Fashion-MNIST), the Gaussian kernel exp(-||x-x_i||^2
    / (2h^2)) underflows to ~0 for every cross term once pairwise distances
    exceed h by even a little, making every point look equally "dense" --
    the KDE carries no signal (empirically verified: Spearman correlation
    between such a density estimate and nearest-neighbor distance is
    ~0.0001, i.e. noise). This scales h with n and d so the kernel stays
    informative regardless of dimensionality (verified: same check gives
    ~-0.9 at the d=784, n=10000 scale).
    '''
    return n ** (-1.0 / (d + 4))


def viasckde_precompute(X, bandwidth=None, kernel='gaussian'):
    '''Precompute the min-max-scaled distance matrix and per-point KDE
    density once per dataset (both independent of the partition).
    bandwidth=None (default) uses silverman_bandwidth(n, d) -- pass an
    explicit value to override.'''
    X01 = MinMaxScaler().fit_transform(X)
    d = squareform(pdist(X01, metric='euclidean'))
    np.fill_diagonal(d, np.inf)  # exclude self when finding nearest
    if bandwidth is None:
        bandwidth = silverman_bandwidth(X01.shape[0], X01.shape[1])
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel).fit(X01)
    log_dens = kde.score_samples(X01)
    # Subtract the max log-density before exponentiating (softmax-style
    # stabilization) -- avoids exp() overflow at high D. VIASCKDE only ever
    # uses `dens` through a per-cluster min-max normalization, which is
    # invariant to a common multiplicative rescaling of dens, so this is an
    # EXACT fix, not an approximation.
    dens = np.exp(log_dens - log_dens.max())
    return {'d': d, 'dens': dens}


def viasckde_score(stats, labels):
    '''VIASCKDE given precomputed stats (from viasckde_precompute) and a
    partition. Maximize.'''
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
        a_i = np.where(np.isinf(a_i), 0.0, a_i)  # singleton clusters

        other_idx = np.where(~mask)[0]
        d_other = d[np.ix_(idx, other_idx)]
        b_i = d_other.min(axis=1)

        denom = np.maximum(np.maximum(a_i, b_i), 1e-12)
        CoSeD = kde_k * (b_i - a_i) / denom
        total += ni * CoSeD.mean()

    return float(total / n)

# ── cell: dcsi-index ──────────────────────────────────────────────────
def _dcsi_core_points(X_sub, min_pts):
    '''
    Core points of one class/cluster per DCSI's definition: x is core if at
    least min_pts other points of the SAME class lie within eps(x), where
    eps is the per-class heuristic (median distance to the
    (2*min_pts)-th nearest neighbor within the class). Returns boolean core
    mask and the pairwise distance submatrix.
    '''
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
    '''Max edge weight in the MST over a class's core points (0 if <2 core points).'''
    idx = np.where(core_mask)[0]
    if len(idx) < 2:
        return 0.0
    d_core = d_sub[np.ix_(idx, idx)]
    mst = minimum_spanning_tree(d_core)
    return float(mst.data.max()) if mst.nnz > 0 else 0.0


def dcsi_index(X, labels, min_pts=5):
    '''
    DCSI (Gauss, Scheipl & Herrmann, 2025). Maximize (in [0,1)).
    See markdown above for the exact formulas implemented.
    '''
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
                # cdist avoids materializing the (n_i, n_j, D) broadcast
                # tensor that a manual pts_i[:,None,:]-pts_j[None,:,:]
                # difference would -- at large D with unbalanced clusters
                # that tensor can reach many GB in one allocation.
                sep = float(cdist(pts_i, pts_j, metric='euclidean').min())
            conn_max = max(conn[ci], conn[cj])
            if conn_max <= 0:
                q = np.inf if sep > 0 else 0.0
            else:
                q = sep / conn_max
            dcsi_pair = q / (1.0 + q) if np.isfinite(q) else 1.0
            pairwise_vals.append(dcsi_pair)

    return float(np.mean(pairwise_vals))

# ── cell: external-metrics ────────────────────────────────────────────
def _pair_counts(y_true, y_pred):
    '''a,b,c,d pair-counting quantities from sklearn's pair_confusion_matrix.
    Rows=truth, cols=predicted convention:
      a = true-same & pred-same   (TP)
      b = true-same & pred-diff   (FN)
      c = true-diff & pred-same   (FP)
      d = true-diff & pred-diff   (TN)
    '''
    C = pair_confusion_matrix(y_true, y_pred)
    d, c = C[0, 0], C[0, 1]
    b, a = C[1, 0], C[1, 1]
    return float(a), float(b), float(c), float(d)


def jaccard_pairs(y_true, y_pred):
    '''Jaccard coefficient (pair-counting external index).'''
    a, b, c, d = _pair_counts(y_true, y_pred)
    denom = a + b + c
    return a / denom if denom > 0 else 0.0


def sokal_sneath3(y_true, y_pred):
    '''Sokal & Sneath 3 pair-counting index: (a+d)/(b+c).'''
    a, b, c, d = _pair_counts(y_true, y_pred)
    denom = b + c
    if denom <= 0:
        return np.inf if (a + d) > 0 else 0.0
    return (a + d) / denom


def powers_informedness(y_true, y_pred):
    ''''Powers' (Bookmaker Informedness / Youden's J) on the pair-counting
    confusion matrix: TPR + TNR - 1.'''
    a, b, c, d = _pair_counts(y_true, y_pred)
    tpr = a / (a + b) if (a + b) > 0 else 0.0
    tnr = d / (c + d) if (c + d) > 0 else 0.0
    return tpr + tnr - 1.0


def nid_distance(y_true, y_pred):
    '''Normalized Information Distance (Vinh, Epps & Bailey, 2010):
    NID = 1 - I(U,V) / max(H(U), H(V)). A DISTANCE (0 = identical
    partitions), not a similarity -- see markdown above.'''
    mi = mutual_info_score(y_true, y_pred)
    h_true = entropy(y_true)
    h_pred = entropy(y_pred)
    denom = max(h_true, h_pred)
    if denom <= 0:
        return 0.0
    return 1.0 - mi / denom


def clustering_accuracy(y_true, y_pred):
    '''Unsupervised clustering accuracy via Hungarian-matched contingency
    table (optimal one-to-one matching between predicted clusters and true
    classes, then fraction of points in a matched pair). Handles
    n_pred != n_true by leaving extra clusters/classes unmatched (0 credit).'''
    cm = contingency_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    return cm[row_ind, col_ind].sum() / cm.sum()


# Registry mapping each external-metric name to the function that computes
# it from (y_true, y_pred) label arrays. This is the set of 7 metrics this
# script validates every internal index against; ACC gets the "no suffix"
# results filename (see results_path_for below) while the other 6 get a
# `_{METRIC}` suffix -- purely a file-naming convention, not a difference
# in how the metrics themselves are treated.
EXTERNAL_METRIC_FUNCS = {
    'Jaccard': jaccard_pairs,
    'SS3':     sokal_sneath3,
    'ARI':     adjusted_rand_score,
    'NMI':     normalized_mutual_info_score,
    'Powers':  powers_informedness,
    'NID':     nid_distance,
    'ACC':     clustering_accuracy,
}

# Orient every score so that larger means better before Spearman correlation.
# NID and WB are the two lower-is-better quantities in this study.
# NID (defined above) is a DISTANCE (0=identical partitions, larger=worse),
# so it is flipped here via orient_external_values before every Spearman
# correlation is computed, so that "larger rho" always means "the internal
# index agrees more with the external ground truth" consistently across all
# 7 external metrics. WB (an internal validity index, not external -- see
# raw_curves['WB'] below) is the other lower-is-better quantity in this
# study and is negated separately, at the point its raw K-curve is built.
LOWER_IS_BETTER_EXTERNAL_METRICS = {'NID'}
SCORE_DIRECTION_CONVENTION = 'larger_is_better_v1'


def orient_external_values(metric, values):
    '''Flip sign of values for any metric in LOWER_IS_BETTER_EXTERNAL_METRICS
    (currently just NID) so every returned array follows the
    larger-is-better convention; all other metrics pass through unchanged.'''
    values = np.asarray(values, dtype=float)
    return -values if metric in LOWER_IS_BETTER_EXTERNAL_METRICS else values

# ── cell: clustering-methods ──────────────────────────────────────────
# Thin wrappers giving each of the 3 clustering algorithms this study
# compares a uniform (X, K, seed) -> labels signature, so the main loop
# below can call CLUSTERING_METHOD_FUNCS[cm_name](X, K, seed) generically.
def kmeans_labels(X, K, seed):
    '''K-means with automatic n_init selection (sklearn's 'auto' heuristic).'''
    return KMeans(n_clusters=K, random_state=seed, n_init='auto').fit_predict(X)


def spectral_labels(X, K, seed):
    '''Spectral clustering with a k-NN (10 neighbors) affinity graph and
    k-means used for the final label assignment step in eigenspace.'''
    return SpectralClustering(
        n_clusters=K, random_state=seed,
        affinity='nearest_neighbors', n_neighbors=10,
        assign_labels='kmeans',
    ).fit_predict(X)


def agglomerative_labels(X, K, seed):
    '''Agglomerative (hierarchical) clustering with Ward linkage. Ward's
    variance-minimizing merge criterion is deterministic given X and K, so
    unlike the other two methods this one ignores `seed`.'''
    return AgglomerativeClustering(n_clusters=K, linkage='ward').fit_predict(X)


CLUSTERING_METHOD_FUNCS = {
    'KMeans':        kmeans_labels,
    'Spectral':       spectral_labels,
    'Agglomerative':  agglomerative_labels,
}

CLUSTERING_METHODS = ['KMeans', 'Spectral', 'Agglomerative']

# ── cell: load-datasets ───────────────────────────────────────────────
RAW_DATA_DIR = 'data/raw_datasets'

# The known-ground-truth (N, D, K*) for each of the 10 bundled SIPU-repository
# .arff files, taken from the CVDD paper's own Table 3 (the source that
# assembled this particular 10-dataset non-globular/nonconvex benchmark
# collection). Used purely as a REPRODUCIBILITY SAFEGUARD below: after each
# .arff file is parsed, its actual shape/class-count is compared against
# these expected values and the run is halted (via assert) on any mismatch.
# This guards against silent corruption of results from an accidentally
# swapped/edited/truncated data file, a parsing bug, or an .arff whose class
# column encoding changed -- any of which would otherwise let the script run
# to completion on wrong data without any visible error.
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
    '''Parse one SIPU-format .arff file into (X, y): X is the (N, D) float
    feature matrix (every attribute column except the last, which is taken
    to be the class/label column); y is an integer-coded label array
    (0..K*-1) obtained by relabeling whatever raw class values the .arff
    file uses (scipy.io.arff.loadarff returns nominal/string attributes as
    Python bytes objects, hence the decode step below) via np.unique's
    return_inverse -- i.e. the actual class names/order in the file don't
    matter, only how many distinct classes there are and which rows share
    one, since only unsupervised clustering + external validity comparison
    against y is done downstream, never a name-dependent lookup.'''
    data, meta = arff.loadarff(path)
    attr_names = meta.names()
    class_col = attr_names[-1]
    feature_cols = attr_names[:-1]
    X = np.column_stack([data[c].astype(float) for c in feature_cols])
    y_raw = data[class_col]
    y_str = np.array([v.decode('utf-8') if isinstance(v, bytes) else str(v) for v in y_raw])
    _, y = np.unique(y_str, return_inverse=True)
    return X, y


# Load and validate all 10 datasets up front (before any clustering/scoring
# work begins) so a data problem fails fast with a clear message rather than
# surfacing as a confusing downstream numeric anomaly. For each dataset:
# load the .arff file, compare its (N, D, K*) against EXPECTED_TABLE3, print
# a per-dataset OK/MISMATCH status line, and assert on any mismatch --
# intentionally hard-failing the whole script rather than silently
# continuing with unexpected data, since every result in the paper depends
# on these exact datasets.
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
R_VALUES = list(range(5, 135, 5))      # r grid: 5, 10, ..., 130
M         = 35
THRESHOLD = 0.95

ALL_METHODS = [
    'VRC', 'Silhouette', 'WG', 'WB', 'DBCV', 'rNN_Hard', 'rNN_Soft',
    'CVDD', 'VIASCKDE', 'DCSI',
]
DATASETS_TO_RUN = DATASET_NAMES

# ── Choose the clustering algorithm(s) for this run ─────────────────────
# One job = one (or all) clustering methods -- all 7 external metrics are
# always batched together within a job (see docstring). Pass the method
# name as a command-line argument, e.g. `python
# partB_fixed_datasets_comparison.py Spectral`; with no argument, all
# three run sequentially (this whole script is fast).
if len(sys.argv) > 1:
    assert sys.argv[1] in CLUSTERING_METHODS, f'Unknown clustering method {sys.argv[1]!r}'
    CLUSTERING_METHODS_TO_RUN = [sys.argv[1]]
else:
    CLUSTERING_METHODS_TO_RUN = CLUSTERING_METHODS

# ── External validity metrics to batch in this run ──────────────────────
EXTERNAL_METRICS_TO_RUN = list(EXTERNAL_METRIC_FUNCS.keys())

EXPERIMENT_TAG = 'paperkmax_buffer10_r130_nonself_oriented'

def results_path_for(base_name, metric):
    '''Build the on-disk pickle path for one (base_name, external_metric)
    combination. ACC (the "primary" external metric this whole pipeline was
    originally built around) keeps the plain `{base_name}_{EXPERIMENT_TAG}.pkl`
    name with no suffix; every other metric gets a `_{METRIC}` suffix. This
    naming convention is also independently relied on by
    generate_nonconvex_tables.py and ../tools/build_manuscript_outputs.py,
    so it must stay in sync with results_path_for there.'''
    suffix = '' if metric == 'ACC' else f'_{metric}'
    return f'data/{base_name}_{EXPERIMENT_TAG}{suffix}.pkl'

print(f'{len(DATASETS_TO_RUN)} datasets: {DATASETS_TO_RUN}')
for name in DATASETS_TO_RUN:
    k_star = EXPECTED_TABLE3[name]['K']
    print(f'  {name:12s} k*={k_star}  max_K_select={max_K_select(k_star)} max_K_eval={max_K_eval(k_star)}  m={M}')
print(f'R_VALUES: {R_VALUES[0]}..{R_VALUES[-1]} ({len(R_VALUES)} values)')
print(f'Clustering methods this session: {CLUSTERING_METHODS_TO_RUN}')
print(f'External metrics batched this session: {EXTERNAL_METRICS_TO_RUN}')

# ── cell: load-results ────────────────────────────────────────────────
# main_results[metric][cm][dataset][method]              -> scalar rho
# sensitivity_results[metric][cm][dataset]['hard'/'soft'][r] -> scalar rho
# rstar_results[metric][cm][dataset]                      -> dict (r*, rho, ...)
def load_or_init(path):
    '''Resume-friendly results loader: if a results pickle from a previous
    (possibly interrupted/timed-out) run already exists at `path`, load and
    return its `results` dict so completed (dataset, clustering_method)
    entries are preserved and skipped/overwritten in place rather than
    recomputed; otherwise start from an empty dict. This is what makes the
    checkpoint-after-every-dataset design (see save_all below) actually
    resumable across SLURM job resubmissions.'''
    if os.path.exists(path):
        with open(path, 'rb') as f:
            payload = pickle.load(f)
        results = payload['results']
        print(f'Loaded existing results from {path}')
        for cm_name in results:
            print(' ', cm_name, ':', sorted(results[cm_name].keys()))
        return results
    print(f'No existing results file found at {path} — starting fresh.')
    return {}


os.makedirs('data', exist_ok=True)
# Three parallel result stores, one dict per external metric (see the
# "main_results[metric][cm][dataset][method]" layout comment above):
#   main_results        -- the final Spearman rho for every (metric, cm,
#                           dataset, validity-index) combination; this is
#                           what the downstream table-generation scripts read.
#   sensitivity_results  -- rNN Hard/Soft rho AT EVERY r in R_VALUES (the
#                           full sweep), used only for the diagnostic
#                           plots/oracle-gap analysis below, not the main
#                           tables.
#   rstar_results        -- the selected r* (and its rho) for rNN Hard/Soft
#                           per (metric, cm, dataset), i.e. the actual output
#                           of Algorithm 1's selection step.
main_results        = {}
sensitivity_results = {}
rstar_results        = {}
for metric in EXTERNAL_METRICS_TO_RUN:
    main_results[metric]        = load_or_init(results_path_for('nonconvex_fixed_results', metric))
    sensitivity_results[metric] = load_or_init(results_path_for('nonconvex_rnn_r_sensitivity_results', metric))
    rstar_results[metric]       = load_or_init(results_path_for('nonconvex_rnn_r_star_results', metric))


def save_results(results_dict, base_name, metric):
    '''Pickle one results dict to disk along with a small metadata payload:
    which external metric it belongs to, the score-direction convention it
    was computed under (so a future reader can detect/reject a stale file
    computed under a different convention -- see generate_nonconvex_tables.py's
    load_results, which checks this exact field), and which clustering
    methods have been computed so far.'''
    payload = {
        'results': results_dict,
        'external_metric': metric,
        'score_direction_convention': SCORE_DIRECTION_CONVENTION,
        'clustering_methods_computed': sorted(results_dict.keys()),
    }
    with open(results_path_for(base_name, metric), 'wb') as f:
        pickle.dump(payload, f)


def save_all():
    '''Checkpoint every result store (main/sensitivity/r*) for every
    external metric in one call. Invoked after each dataset finishes inside
    the main loop below, so an interrupted job can simply be resubmitted --
    already-completed datasets are preserved in the pickles on disk and
    load_or_init will pick them back up.'''
    for metric in EXTERNAL_METRICS_TO_RUN:
        save_results(main_results[metric], 'nonconvex_fixed_results', metric)
        save_results(sensitivity_results[metric], 'nonconvex_rnn_r_sensitivity_results', metric)
        save_results(rstar_results[metric], 'nonconvex_rnn_r_star_results', metric)

# ── cell: main-loop (Hard + Soft, multi-metric batched) ─────────────────
# NOTE (multi-metric batching): r* selection, every index's raw per-K
# score curve, AND the full rNN Hard/Soft r-sensitivity sweep over
# R_VALUES (the single most expensive part of this script) depend only on
# (X, labels_by_K) -- never on the external metric. So each is computed
# exactly ONCE per (dataset, clustering_method), and only the final
# Spearman-vs-ext_vals step is repeated per metric below.

for dname in DATASETS_TO_RUN:
    X_raw, y = DATASETS[dname]
    # Standardize features (zero mean, unit variance per column) before
    # clustering/scoring -- required since these datasets mix raw pixel/
    # coordinate scales (e.g. atom is 3D, others are 2D with differing
    # native ranges) and every distance-based method here (K-means,
    # Euclidean silhouette, rNN neighbor search, ...) is scale-sensitive.
    X = StandardScaler().fit_transform(X_raw)
    n_samples, D = X.shape
    k_star = len(np.unique(y))
    # The two K-ranges from Algorithm 1 (see max_K_select/max_K_eval
    # docstrings above): max_K_s is the WIDER range clustering is actually
    # run over (also used for r* selection), max_K_e is the NARROWER range
    # every reported rho is restricted to.
    max_K_s = max_K_select(k_star)
    max_K_e = max_K_eval(k_star)
    K_TEST = list(range(2, max_K_s + 1))
    n_k = len(K_TEST)
    # eval_idx: positions within K_TEST (and therefore within every raw
    # per-K score curve computed below) that fall within the narrower
    # evaluation range K<=max_K_e. Every Spearman correlation reported in
    # main_results is computed only over these positions -- clustering
    # itself still runs on the full K_TEST range up to max_K_s so r*
    # selection has its extra +10 headroom.
    K_idx_lookup = {K: i for i, K in enumerate(K_TEST)}
    eval_idx = [K_idx_lookup[K] for K in K_TEST if K <= max_K_e]

    for metric in EXTERNAL_METRICS_TO_RUN:
        for cm_name in CLUSTERING_METHODS_TO_RUN:
            main_results[metric].setdefault(cm_name, {})
            main_results[metric][cm_name][dname] = {}
            sensitivity_results[metric].setdefault(cm_name, {})
            sensitivity_results[metric][cm_name][dname] = {'hard': {}, 'soft': {}}
            rstar_results[metric].setdefault(cm_name, {})

    print(f'\n=== Dataset {dname}  (k*={k_star}, D={D}, N={n_samples}) ===')

    cvdd_stats     = cvdd_precompute(X)
    viasckde_stats = viasckde_precompute(X)

    for cm_name in CLUSTERING_METHODS_TO_RUN:
        cm_fn = CLUSTERING_METHOD_FUNCS[cm_name]
        seed = 1  # no replication -- fixed seed is enough for reproducibility

        labels_by_K = {K: cm_fn(X, K, seed) for K in K_TEST}
        multi_cluster = {K: len(np.unique(labels_by_K[K])) > 1 for K in K_TEST}

        # ext_vals computed once per metric, shared across every index below
        ext_vals_by_metric = {
            metric: orient_external_values(
                metric,
                np.array([EXTERNAL_METRIC_FUNCS[metric](y, labels_by_K[K]) for K in K_TEST])[eval_idx],
            )
            for metric in EXTERNAL_METRICS_TO_RUN
        }

        # ---- 8 non-rNN indices: raw per-K curves, computed once ----
        # Each entry is the index's own raw score at every K in K_TEST
        # (NaN wherever a K produced a degenerate single-cluster labeling).
        # All 8 of these plus rNN_Hard/rNN_Soft are meant to be
        # larger-is-better BEFORE the final Spearman step -- WB (below) is
        # the one exception among these internal indices and is negated
        # here so raw_curves consistently holds larger-is-better values
        # throughout, matching the SCORE_DIRECTION_CONVENTION also applied
        # to the external metrics via orient_external_values above.
        raw_curves = {}
        raw_curves['VRC'] = np.array([
            calinski_harabasz_score(X, labels_by_K[K]) if multi_cluster[K] else np.nan
            for K in K_TEST
        ])
        raw_curves['Silhouette'] = np.array([
            silhouette_score(X, labels_by_K[K], metric='euclidean') if multi_cluster[K] else np.nan
            for K in K_TEST
        ])
        raw_curves['WG'] = np.array([
            wemmert_gancarski_index(X, labels_by_K[K]) if multi_cluster[K] else np.nan
            for K in K_TEST
        ])
        # WB is minimized; negate it so larger consistently means better.
        raw_curves['WB'] = -np.array([
            wb_index(X, labels_by_K[K]) if multi_cluster[K] else np.nan
            for K in K_TEST
        ])
        raw_curves['DBCV'] = np.array([
            dbcv_index(X, labels_by_K[K]) if multi_cluster[K] else np.nan
            for K in K_TEST
        ])
        raw_curves['CVDD'] = np.array([
            cvdd_score(cvdd_stats, labels_by_K[K]) if multi_cluster[K] else np.nan
            for K in K_TEST
        ])
        raw_curves['VIASCKDE'] = np.array([
            viasckde_score(viasckde_stats, labels_by_K[K]) if multi_cluster[K] else np.nan
            for K in K_TEST
        ])
        raw_curves['DCSI'] = np.array([
            dcsi_index(X, labels_by_K[K]) if multi_cluster[K] else np.nan
            for K in K_TEST
        ])

        for metric in EXTERNAL_METRICS_TO_RUN:
            ext_vals = ext_vals_by_metric[metric]
            for method, curve in raw_curves.items():
                main_results[metric][cm_name][dname][method] = spearmanr_safe(curve[eval_idx], ext_vals)

        # ---- rNN Hard + Soft sweep: computed ONCE, shared by every metric ----
        sil_hard_by_r, sil_soft_by_r = {}, {}
        for r in R_VALUES:
            D_hard = rnn_affinity_distance(X, r=r, use_gaussian=False)
            D_soft = rnn_affinity_distance(X, r=r, use_gaussian=True)
            sil_hard_by_r[r] = np.array([
                silhouette_score(D_hard, labels_by_K[K], metric='precomputed') if multi_cluster[K] else np.nan
                for K in K_TEST
            ])
            sil_soft_by_r[r] = np.array([
                silhouette_score(D_soft, labels_by_K[K], metric='precomputed') if multi_cluster[K] else np.nan
                for K in K_TEST
            ])
            # sensitivity curves: rNN Hard/Soft correlation at EVERY r, per metric
            for metric in EXTERNAL_METRICS_TO_RUN:
                ext_vals = ext_vals_by_metric[metric]
                sensitivity_results[metric][cm_name][dname]['hard'][r] = spearmanr_safe(sil_hard_by_r[r][eval_idx], ext_vals)
                sensitivity_results[metric][cm_name][dname]['soft'][r] = spearmanr_safe(sil_soft_by_r[r][eval_idx], ext_vals)

        # r* selection never looks at the external metric -- done once per dataset
        r_star_h, _ = select_r_topK_spearman(sil_hard_by_r, R_VALUES, m=M, n_k=n_k, threshold=THRESHOLD)
        r_star_s, _ = select_r_topK_spearman(sil_soft_by_r, R_VALUES, m=M, n_k=n_k, threshold=THRESHOLD)

        for metric in EXTERNAL_METRICS_TO_RUN:
            ext_vals = ext_vals_by_metric[metric]
            rho_hard = spearmanr_safe(sil_hard_by_r[r_star_h][eval_idx], ext_vals)
            rho_soft = spearmanr_safe(sil_soft_by_r[r_star_s][eval_idx], ext_vals)
            main_results[metric][cm_name][dname]['rNN_Hard'] = rho_hard
            main_results[metric][cm_name][dname]['rNN_Soft'] = rho_soft
            rstar_results[metric][cm_name][dname] = {
                'r_star_hard':  r_star_h,
                'rho_hard':     rho_hard,
                'r_star_soft':  r_star_s,
                'rho_soft':     rho_soft,
                'max_K_select': max_K_s,
                'max_K_eval':   max_K_e,
            }

        rho_hard_acc = main_results['ACC'][cm_name][dname]['rNN_Hard']
        rho_soft_acc = main_results['ACC'][cm_name][dname]['rNN_Soft']
        print(f'  {cm_name}: r*_hard={r_star_h} (ACC rho={rho_hard_acc:.3f})  '
              f'r*_soft={r_star_s} (ACC rho={rho_soft_acc:.3f})  '
              f'max_K_select={max_K_s}, max_K_eval={max_K_e}')

    save_all()
    print(f'  dataset {dname} done, progress saved for metrics: {EXTERNAL_METRICS_TO_RUN}')

print('\nAll datasets completed!')

# ── cell: save-results ────────────────────────────────────────────────
save_all()
print('Results saved for all batched metrics.')

# ── cell: summary-table ───────────────────────────────────────────────
for metric in EXTERNAL_METRICS_TO_RUN:
    rows = []
    for cm_name in main_results[metric]:
        for dname in main_results[metric][cm_name]:
            meta = EXPECTED_TABLE3.get(dname, {})
            for method in ALL_METHODS:
                rho = main_results[metric][cm_name][dname].get(method)
                if rho is None or not np.isfinite(rho):
                    continue
                rows.append({
                    'clustering_method': cm_name, 'external_metric': metric,
                    'dataset': dname, 'k*': meta.get('K'), 'N': meta.get('N'), 'D': meta.get('D'),
                    'method': method, 'rho': round(float(rho), 4),
                })
    main_summary_df = pd.DataFrame(rows)
    print(f'\n=== Method-comparison summary ({metric}) ===')
    print(main_summary_df.to_string(index=False))

for metric in EXTERNAL_METRICS_TO_RUN:
    rows2 = []
    for cm_name in rstar_results[metric]:
        for dname in rstar_results[metric][cm_name]:
            entry = rstar_results[metric][cm_name][dname]
            curve = sensitivity_results[metric].get(cm_name, {}).get(dname, {})
            for variant, r_key, rho_key in [('hard', 'r_star_hard', 'rho_hard'), ('soft', 'r_star_soft', 'rho_soft')]:
                variant_curve = curve.get(variant, {})
                r_oracle = max(variant_curve, key=variant_curve.get) if variant_curve else None
                rows2.append({
                    'clustering_method': cm_name, 'dataset': dname, 'variant': variant,
                    'r_star': entry.get(r_key), 'rho': round(entry[rho_key], 4) if rho_key in entry else None,
                    'r_oracle (diagnostic)': r_oracle,
                    'rho_oracle': round(variant_curve[r_oracle], 4) if r_oracle is not None else None,
                    'gap_to_oracle': round((variant_curve[r_oracle] - entry[rho_key]), 4) if r_oracle is not None else None,
                })
    rstar_summary_df = pd.DataFrame(rows2)
    print(f'\n=== r* summary ({metric}, with diagnostic oracle r) ===')
    print(rstar_summary_df.to_string(index=False))
    for variant in ['hard', 'soft']:
        sub = rstar_summary_df[rstar_summary_df['variant'] == variant]
        print(f'Mean gap-to-oracle ({metric}, {variant}): {sub["gap_to_oracle"].mean():.4f}')

# ── cell: combined-plots ──────────────────────────────────────────────
def plot_combined(metric, cm_name, dname):
    '''Diagnostic 2-panel figure for one (external_metric, clustering_method,
    dataset) combination, saved to figures/{cm_name}/{metric}/{EXPERIMENT_TAG}/
    Combined_{dname}.{pdf,png}: left panel is a bar chart of every index's
    Spearman rho against `metric` (from main_results); right panel is the
    rNN Hard/Soft rho-vs-r sensitivity curve (from sensitivity_results) with
    the selected r* marked (dotted vertical line) alongside the diagnostic
    "oracle" r -- i.e. whichever r actually maximizes rho in hindsight, which
    r* selection (blind to the external metric) cannot see. These plots are
    exploratory/QA outputs, not figures used directly in the manuscript.'''
    bar_data   = main_results[metric].get(cm_name, {}).get(dname, {})
    curve_data = sensitivity_results[metric].get(cm_name, {}).get(dname, {})
    rstar_data = rstar_results[metric].get(cm_name, {}).get(dname, {})

    if not bar_data and not curve_data:
        print(f'No results to plot for {metric} / {cm_name} / {dname}')
        return

    meta = EXPECTED_TABLE3.get(dname, {})
    k_star, n_samples, D = meta.get('K', '?'), meta.get('N', '?'), meta.get('D', '?')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    available = [m for m in ALL_METHODS if bar_data.get(m) is not None and np.isfinite(bar_data[m])]
    if available:
        vals = [bar_data[m] for m in available]
        palette = sns.color_palette('tab10', n_colors=len(available))
        ax.bar(available, vals, color=palette, alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.axhline(0, color='grey', linewidth=1, linestyle='--')
        ax.set_ylabel(f'Spearman $\\rho$ with {metric}', fontsize=13)
        ax.set_title('Method comparison', fontsize=13, fontweight='bold')
        ax.tick_params(axis='x', labelsize=10, rotation=30)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Method comparison', fontsize=13, fontweight='bold')

    ax = axes[1]
    hard_curve = curve_data.get('hard', {})
    soft_curve = curve_data.get('soft', {})
    if hard_curve or soft_curve:
        if hard_curve:
            rs_h = sorted(hard_curve.keys())
            ax.plot(rs_h, [hard_curve[r] for r in rs_h], marker='o', color='steelblue',
                    linewidth=2, label='rNN Hard $\\rho(r)$')
            r_star_hard = rstar_data.get('r_star_hard')
            if r_star_hard is not None:
                ax.axvline(r_star_hard, color='steelblue', linestyle=':', linewidth=2.5,
                          label=f'$r^*_{{hard}} = {r_star_hard}$')
            r_oracle_h = max(hard_curve, key=hard_curve.get)
            ax.axvline(r_oracle_h, color='steelblue', linestyle='--', linewidth=1.5, alpha=0.5,
                      label=f'oracle (hard) $r={r_oracle_h}$')

        if soft_curve:
            rs_s = sorted(soft_curve.keys())
            ax.plot(rs_s, [soft_curve[r] for r in rs_s], marker='s', color='darkorange',
                    linewidth=2, label='rNN Soft $\\rho(r)$')
            r_star_soft = rstar_data.get('r_star_soft')
            if r_star_soft is not None:
                ax.axvline(r_star_soft, color='darkorange', linestyle=':', linewidth=2.5,
                          label=f'$r^*_{{soft}} = {r_star_soft}$')
            r_oracle_s = max(soft_curve, key=soft_curve.get)
            ax.axvline(r_oracle_s, color='darkorange', linestyle='--', linewidth=1.5, alpha=0.5,
                      label=f'oracle (soft) $r={r_oracle_s}$')

        ax.axhline(0, color='grey', linewidth=1, linestyle='--')
        ax.set_xlabel('$r$ (rNN neighborhood size)', fontsize=12)
        ax.set_ylabel(f'Spearman $\\rho$ with {metric}', fontsize=12)
        ax.set_title('rNN Hard/Soft $\\rho$ vs. $r$  (eval $k_{max}$=max(25,1.75$k^*$), select +10, m=35)',
                    fontsize=13, fontweight='bold')
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('rNN Hard/Soft $\\rho$ vs. $r$', fontsize=13, fontweight='bold')

    fig.suptitle(f'{cm_name} — {dname}  ($k^*={k_star}$, $D={D}$, $N={n_samples}$)  '
                f'[eval kmax=max(25,1.75k*), select +10, m=35, hard+soft, r≤130]',
                fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out_dir = f'figures/{cm_name}/{metric}/{EXPERIMENT_TAG}'
    os.makedirs(out_dir, exist_ok=True)
    fname = f'Combined_{dname}'
    plt.savefig(f'{out_dir}/{fname}.pdf', bbox_inches='tight')
    plt.savefig(f'{out_dir}/{fname}.png', dpi=150, bbox_inches='tight')
    print(f'Saved: {out_dir}/{fname}.pdf')
    plt.close(fig)


for metric in EXTERNAL_METRICS_TO_RUN:
    for cm_name in CLUSTERING_METHODS_TO_RUN:
        for dname in sorted(main_results[metric].get(cm_name, {}).keys()):
            plot_combined(metric, cm_name, dname)

# ── cell: highlight-table ─────────────────────────────────────────────
# (CSV export, since a remote/headless job has no notebook to render
# pandas .style HTML in -- see the notebook for the interactive version.)
for metric in EXTERNAL_METRICS_TO_RUN:
    pivot_rows = []
    for cm_name in main_results[metric]:
        for dname in main_results[metric][cm_name]:
            row = {'clustering_method': cm_name, 'dataset': dname}
            for method in ALL_METHODS:
                val = main_results[metric][cm_name][dname].get(method)
                row[method] = round(float(val), 4) if val is not None and np.isfinite(val) else np.nan
            pivot_rows.append(row)
    pivot_df = pd.DataFrame(pivot_rows).set_index(['clustering_method', 'dataset'])
    out_csv = f'data/highlight_table_{EXPERIMENT_TAG}_{metric}.csv'
    pivot_df.to_csv(out_csv)
    print(f'Saved: {out_csv}')
