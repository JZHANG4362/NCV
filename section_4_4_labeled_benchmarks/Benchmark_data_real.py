"""
Standalone script for Benchmark_data_real.ipynb, extracted for remote
(SLURM) execution. Runs all 9 real benchmark datasets from main.tex's
Table~\ref{tab:data} (Digits, MNIST-test, Ecoli, Letter, Iris,
Fashion-MNIST-test, Wine, Libras, Vowel), all 10 validity indices (both
rNN_Hard and rNN_Soft), against ALL 7 external metrics (Jaccard, SS3,
ARI, NMI, Powers, NID, ACC) in a single pass -- see "Multi-metric
batching" below. Clustering method matrix matches Part B
(Benchmark_data_simulated_nonconvex): all 3 of KMeans, Spectral,
Agglomerative -- NOT just the KMeans-only protocol described in the
paper's Real Data Analysis section (sec:real). Defaults to all 3
(sequentially, in this one process), overridable via a command-line
argument to restrict to a single method:

    python Benchmark_data_real.py                  # all 3 (default)
    python Benchmark_data_real.py all               # all 3 (explicit)
    python Benchmark_data_real.py KMeans            # one method only
    python Benchmark_data_real.py Spectral
    python Benchmark_data_real.py Agglomerative

SCALE WARNING: MNIST-test, Fashion-MNIST-test (N=10000, D=784) and Letter
(N=20000, D=16) are far larger than anything else in this project. The
rNN affinity sweep (rnn_affinity_distance, O(N^2) per r) and several
custom indices (CVDD/VIASCKDE/DCSI precompute steps) do not scale well to
N=10000-20000 in pure NumPy/Python loops -- expect these three datasets to
dominate the wall-clock time by a wide margin over the other 6 combined.
Agglomerative makes this WORSE than the KMeans-only design this was
originally built for: agglomerative_labels() below re-fits the FULL ward
linkage from scratch for every K in K_TEST (there is no cheap way to cut
one fitted tree at multiple K's through sklearn's high-level API) -- for
Letter that's ~55 full O(N^2)-ish linkage fits at N=20000, which may be
impractically slow; watch the first run closely and consider restricting
K_TEST or dropping Agglomerative for the largest datasets if it doesn't
converge in reasonable time. DATASETS_TO_RUN can be restricted (see
settings below) to stage the 6 small/medium datasets separately from the
3 large ones; the recommended remote-compute split (small as one
all-methods job, large as 9 per-dataset-per-method jobs -- see the
checkpointing NOTE below for why) is documented in
cluster_setup_commands_realdata.docx.

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

r* selection method (identical convention to Part A / Part B / Part C):
- EVALUATION (the reported rho for every index, not just rNN): computed
  only over K = 2..max(25, 1.75*k*) -- the reference paper's ("Benchmarking
  of Clustering Validity Measures Revisited") own K-range cap, Section 3.2.
- SELECTION of r* (rNN_Hard and rNN_Soft independently): select_r_topK_spearman
  (top-m monotonic-decrease-vs-ideal-sequence Spearman criterion), m=35,
  over a wider range K = 2..(max(25,1.75*k*)+10).

The main loop checkpoints after every dataset, so an interrupted/timed-out
job can simply be resubmitted -- completed datasets are preserved in the
results pickles. Raw dataset downloads are cached under
data/raw_datasets/ so a resubmitted job does not re-download anything.

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

from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import squareform, pdist, cdist

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.datasets import load_digits, load_iris, load_wine
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score, calinski_harabasz_score,
    normalized_mutual_info_score, mutual_info_score,
)
from sklearn.metrics.cluster import pair_confusion_matrix, entropy, contingency_matrix
from sklearn.neighbors import NearestNeighbors, KernelDensity

import urllib.request
import gzip
import io

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
    type2_case_comparison.ipynb's simulate_type2_data.'''
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
    '''
    avail = sorted(r for r in r_values if r in sil_dict)
    if not avail:
        return r_values[0], {}
    m_eff   = max(2, min(m, n_k - 1))
    top_idx = list(range(n_k - m_eff, n_k))
    ideal   = np.arange(m_eff, 0, -1, dtype=float)
    c_vals  = {}
    for r in avail:
        obs        = np.asarray(sil_dict[r], dtype=float)[top_idx]
        rho        = spearmanr_safe(obs, ideal)
        c_vals[r]  = rho if not np.isnan(rho) else -2.0
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
    # Subtract the max log-density before exponentiating (the standard
    # softmax-style stabilization trick) -- at high D (e.g. 784 for
    # MNIST/Fashion-MNIST) the fixed bandwidth makes raw log-density values
    # large enough that exp() overflows to inf for every point. VIASCKDE
    # only ever uses `dens` through a per-cluster min-max normalization,
    # which is invariant to a common multiplicative rescaling of dens (i.e.
    # a common additive shift of log_dens), so this is an EXACT fix, not
    # an approximation.
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
                # difference would -- at D=784 (MNIST/Fashion-MNIST) with
                # unbalanced clusters that tensor can reach tens to
                # hundreds of GB in one allocation and OOM the job.
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
LOWER_IS_BETTER_EXTERNAL_METRICS = {'NID'}
SCORE_DIRECTION_CONVENTION = 'larger_is_better_v1'


def orient_external_values(metric, values):
    values = np.asarray(values, dtype=float)
    return -values if metric in LOWER_IS_BETTER_EXTERNAL_METRICS else values

# ── cell: clustering-methods ──────────────────────────────────────────
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
    'Spectral':       spectral_labels,
    'Agglomerative':  agglomerative_labels,
}

CLUSTERING_METHODS = ['KMeans', 'Spectral', 'Agglomerative']

# ── cell: load-datasets ───────────────────────────────────────────────
RAW_DATA_DIR = 'data/raw_datasets'
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
    '''Download url to RAW_DATA_DIR/fname if not already cached; return the local path.'''
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
    X = np.array([[float(v) for v in r[1:-1]] for r in rows])  # drop sequence-name col + class col
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
    '''Deterding Vowel Recognition (Connectionist Bench). Columns: train/test
    flag, speaker id, sex, 10 features, class label.'''
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

# ── datasets to load/run this session (see settings below for the
#    command-line override) -- restrict here to stage large datasets
#    separately, e.g. DATASET_NAMES_TO_LOAD = ['Digits', 'Iris', 'Wine']
DATASET_NAMES_TO_LOAD = (
    [a for a in sys.argv[2:]] if len(sys.argv) > 2 else DATASET_NAMES
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
    assert ok, f'{name}: loaded (N={n},D={d},K={k}) does not match main.tex Table~\\ref{{tab:data}} (N={exp["N"]},D={exp["D"]},K={exp["K"]})'

# ── cell: settings ────────────────────────────────────────────────────
R_VALUES = list(range(5, 135, 5))      # r grid: 5, 10, ..., 130
M         = 35
THRESHOLD = 0.95

ALL_METHODS = [
    'VRC', 'Silhouette', 'WG', 'WB', 'DBCV', 'rNN_Hard', 'rNN_Soft',
    'CVDD', 'VIASCKDE', 'DCSI',
]
# All nine datasets by default, matching main.tex Table~\ref{tab:data}
# exactly -- unless a subset was named on the command line (see
# DATASET_NAMES_TO_LOAD above), which restricts both loading AND running
# to that subset (so the large-dataset job need not even download the
# other 6).
DATASETS_TO_RUN = DATASET_NAMES_TO_LOAD
LARGE_DATASETS = ['MNIST-test', 'Fashion-MNIST-test', 'Letter']

# ── Choose the clustering algorithm(s) for this run ──────────────────────
# Same matrix as Part B (Benchmark_data_simulated_nonconvex): all 3
# clustering methods -- KMeans, Spectral, Agglomerative -- not just the
# KMeans-only protocol described in the paper's Real Data Analysis
# section (sec:real). Pass a specific method name as the FIRST
# command-line argument to restrict to just that one (e.g. `python
# Benchmark_data_real.py Spectral Digits Iris Wine`); omit it, or pass
# 'all' explicitly, to run all 3 sequentially in this process (matching
# Part B's default) -- dataset names (if any) are all subsequent
# arguments either way (see DATASET_NAMES_TO_LOAD above).
#
# NOTE (checkpointing): save_all() below is called once per DATASET,
# after every requested clustering method has finished for it -- so for
# a job that requests 'all' 3 methods on a SINGLE large dataset (see the
# per-method large-dataset .dlx files), nothing is saved until KMeans,
# Spectral, AND Agglomerative have all completed for that one dataset.
# That's why the 3 large datasets (MNIST-test, Fashion-MNIST-test,
# Letter) are each still split into 3 separate per-method jobs rather
# than one 'all'-methods job like the small-dataset batch -- otherwise a
# timeout partway through, say, Agglomerative would silently discard
# already-finished KMeans + Spectral results for that dataset.
_cli_cm = sys.argv[1] if len(sys.argv) > 1 else 'all'
if _cli_cm == 'all':
    CLUSTERING_METHODS_TO_RUN = CLUSTERING_METHODS
else:
    assert _cli_cm in CLUSTERING_METHODS, f'Unknown clustering method {_cli_cm!r}'
    CLUSTERING_METHODS_TO_RUN = [_cli_cm]

# ── External validity metrics to batch in this run ──────────────────────
EXTERNAL_METRICS_TO_RUN = list(EXTERNAL_METRIC_FUNCS.keys())

EXPERIMENT_TAG = 'paperkmax_buffer10_r130_nonself_oriented'

# ── Optional per-job output tag, for safe PARALLEL execution ───────────
# Set via the JOB_TAG environment variable (see the per-job .dlx files).
# When set, this job's results are written to their OWN private files
# (EXPERIMENT_TAG + '_' + JOB_TAG), completely disjoint from every other
# job's files -- so any number of jobs can run at the same time with zero
# risk of one job's periodic save clobbering another's, no matter what
# (dataset, method) combination each one covers. Merge everything back
# into the canonical (untagged) files afterward with merge_realdata_results.py.
JOB_TAG = os.environ.get('JOB_TAG', '').strip()
_run_tag = f'{EXPERIMENT_TAG}_{JOB_TAG}' if JOB_TAG else EXPERIMENT_TAG

def results_path_for(base_name, metric):
    suffix = '' if metric == 'ACC' else f'_{metric}'
    return f'data/{base_name}_{_run_tag}{suffix}.pkl'

print(f'{len(DATASETS_TO_RUN)} datasets to run: {DATASETS_TO_RUN}')
print(f'Of these, large/slow datasets present: {[d for d in DATASETS_TO_RUN if d in LARGE_DATASETS]}')
for name in DATASETS_TO_RUN:
    k_star = EXPECTED_TABLE_DATA[name]['K']
    print(f'  {name:20s} k*={k_star}  max_K_select={max_K_select(k_star)} max_K_eval={max_K_eval(k_star)}  m={M}')
print(f'R_VALUES: {R_VALUES[0]}..{R_VALUES[-1]} ({len(R_VALUES)} values)')
print(f'Clustering method(s) this session: {CLUSTERING_METHODS_TO_RUN}')
print(f'External metrics batched this session: {EXTERNAL_METRICS_TO_RUN}')

# ── cell: load-results ────────────────────────────────────────────────
# main_results[metric][cm][dataset][method]              -> scalar rho
# sensitivity_results[metric][cm][dataset]['hard'/'soft'][r] -> scalar rho
# rstar_results[metric][cm][dataset]                      -> dict (r*, rho, ...)
def load_or_init(path):
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
main_results        = {}
sensitivity_results = {}
rstar_results        = {}
for metric in EXTERNAL_METRICS_TO_RUN:
    main_results[metric]        = load_or_init(results_path_for('realdata_fixed_results', metric))
    sensitivity_results[metric] = load_or_init(results_path_for('realdata_rnn_r_sensitivity_results', metric))
    rstar_results[metric]       = load_or_init(results_path_for('realdata_rnn_r_star_results', metric))


def save_results(results_dict, base_name, metric):
    payload = {
        'results': results_dict,
        'external_metric': metric,
        'score_direction_convention': SCORE_DIRECTION_CONVENTION,
        'clustering_methods_computed': sorted(results_dict.keys()),
    }
    with open(results_path_for(base_name, metric), 'wb') as f:
        pickle.dump(payload, f)


def save_all():
    for metric in EXTERNAL_METRICS_TO_RUN:
        save_results(main_results[metric], 'realdata_fixed_results', metric)
        save_results(sensitivity_results[metric], 'realdata_rnn_r_sensitivity_results', metric)
        save_results(rstar_results[metric], 'realdata_rnn_r_star_results', metric)

# ── cell: main-loop (Hard + Soft, multi-metric batched) ─────────────────
# NOTE (multi-metric batching): r* selection, every index's raw per-K
# score curve, AND the full rNN Hard/Soft r-sensitivity sweep over
# R_VALUES (the single most expensive part of this script, and the one
# that scales worst with N -- see SCALE WARNING in the module docstring)
# depend only on (X, labels_by_K) -- never on the external metric. So
# each is computed exactly ONCE per (dataset, clustering_method), and
# only the final Spearman-vs-ext_vals step is repeated per metric below.

for dname in DATASETS_TO_RUN:
    X_raw, y = DATASETS[dname]
    X = StandardScaler().fit_transform(X_raw)
    n_samples, D = X.shape
    k_star = len(np.unique(y))
    max_K_s = max_K_select(k_star)
    max_K_e = max_K_eval(k_star)
    K_TEST = list(range(2, max_K_s + 1))
    n_k = len(K_TEST)
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
            meta = EXPECTED_TABLE_DATA.get(dname, {})
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
    bar_data   = main_results[metric].get(cm_name, {}).get(dname, {})
    curve_data = sensitivity_results[metric].get(cm_name, {}).get(dname, {})
    rstar_data = rstar_results[metric].get(cm_name, {}).get(dname, {})

    if not bar_data and not curve_data:
        print(f'No results to plot for {metric} / {cm_name} / {dname}')
        return

    meta = EXPECTED_TABLE_DATA.get(dname, {})
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

    out_dir = f'figures/{cm_name}/{metric}/{_run_tag}'
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
    out_csv = f'data/highlight_table_{_run_tag}_{metric}.csv'
    pivot_df.to_csv(out_csv)
    print(f'Saved: {out_csv}')
