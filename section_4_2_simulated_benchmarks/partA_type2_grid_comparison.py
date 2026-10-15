"""Main text Section 4.2: Table 1; Supplement Section S8: Tables S4--S45.

Generate 100 replications of each of 12 Gaussian-mixture settings:
K*=6/30, D=2/10/200, and standard deviation 0.1/0.8. Standardize each
dataset, then fit one clustering algorithm over the candidate K grid.
Select r with Algorithm 1 on internal score curves and assess rankings
against ACC, Jaccard, SS3, ARI, NMI, Powers, and NID. The same partitions
and internal curves are reused for all seven external criteria.

The nine published indices include rNN-Silhouette with binary neighbor weights. Table 1
uses K-means/ACC means; the Supplement gives means and medians for all
three algorithms. scripts/build_tables.py performs this aggregation.
The supplied Spectral k30_D200_sparse cache has 39 replications; this
script requests 100 for a fresh run and does not fill missing results.

Run here: python partA_type2_grid_comparison.py KMeans
Use Spectral or Agglomerative for the other algorithms."""


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

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score, calinski_harabasz_score,
    normalized_mutual_info_score, mutual_info_score,
)
from sklearn.metrics.cluster import pair_confusion_matrix, entropy, contingency_matrix
from sklearn.neighbors import NearestNeighbors, KernelDensity

sns.set_style('whitegrid')

# Suppress repeated disconnected-graph warnings during the K sweep.
# Spectral affinity uses 10 neighbors independently of the rNN score grid.
warnings.filterwarnings('ignore', message='Graph is not fully connected')

try:
    from hdbscan.validity import validity_index as _hdbscan_validity_index
    _HDBSCAN_AVAILABLE = True
except ImportError:
    _HDBSCAN_AVAILABLE = False
    print("hdbscan is not installed — DBCV will be reported as NaN for every "
          "replicate. Run `pip install hdbscan` to enable it.")

# ── cell: helpers ─────────────────────────────────────────────────────
def simulate_type2_data(n_samples, n_features, k_star, compactness, random_state=1):
    """Generate the Gaussian mixtures used in Case 2 and its follow-up studies.
    
    make_blobs draws centers uniformly in [-10, 10] in each coordinate,
    uses isotropic standard deviation compactness, and distributes samples
    as evenly as possible among k_star components. The caller standardizes
    features before clustering and scoring (Supplement Section S5)."""
    # Supplement Section S5: mu_c ~ Uniform([-10,10]^D), then
    # X_i | y_i=c ~ N(mu_c, compactness^2 I_D). compactness is sigma, not variance.
    # make_blobs balances component sizes up to one observation and shuffles rows.
    # y records component membership for external evaluation; it is not a feature.
    X, y = make_blobs(n_samples=n_samples, n_features=n_features,
                      centers=k_star, cluster_std=compactness,
                      random_state=random_state)
    return X, y


def rnn_affinity_distance(X, r):
    """Build the binary-neighbor distance used by rNN-Silhouette.
    
    Query r+1 neighbors and discard the first returned match. For distinct
    observations this leaves r other points. Take the symmetric union of
    directed neighbor links and return 1-A with a zero diagonal. Reuse this
    distance matrix across candidate cluster counts for the same dataset."""
    n = X.shape[0]
    # Query r+1 neighbors and drop the first returned match. For distinct
    # observations this removes self and retains r other observations.
    if not 1 <= r < n:
        raise ValueError(f'r must satisfy 1 <= r < n; got r={r}, n={n}')
    nbrs = NearestNeighbors(n_neighbors=r + 1, algorithm='auto').fit(X)
    dists, idx = nbrs.kneighbors(X)
    # With distinct points, column 0 is the query itself, leaving r other neighbors.
    # For coincident points, the neighbor library determines the order of zero-distance ties.
    dists, idx = dists[:, 1:], idx[:, 1:]

    # Section 3.1: first form the directed binary affinity B_ij = 1{j in N_r(i)}.
    # Neighbor search uses features only; neither the reference labels nor K enter B.
    A = np.zeros((n, n))
    for i in range(n):
        A[i, idx[i]] = 1.0

    # Symmetric union: A_ij = max(B_ij, B_ji) = 1{j in N_r(i) OR i in N_r(j)}.
    # An edge in either direction is enough; this is not a mutual-neighbor graph.
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0)
    # The affinity is already binary. This retained min-max step leaves its 0/1 values unchanged.
    a_min, a_max = A.min(), A.max()
    if a_max > a_min:
        A = (A - a_min) / (a_max - a_min)
    # The paper uses dissimilarities: d_r(i,j)=1-A_ij off the diagonal, d_r(i,i)=0.
    # Thus linked points have distance 0 and unlinked points have distance 1.
    # Pass this N-by-N matrix as metric="precomputed", not as N feature vectors.
    D = 1.0 - A
    np.fill_diagonal(D, 0)
    return D


def spearmanr_safe(a, b):
    '''Spearman correlation, returns NaN on degenerate input.'''
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    # Compare paired entries only. In the experiments an entry is a candidate
    # partition (one K), not an observation; SciPy averages tied ranks.
    mask = ~np.isnan(a) & ~np.isnan(b)
    if mask.sum() < 2:
        return np.nan
    rho, _ = spearmanr(a[mask], b[mask])
    return float(rho)


def select_r_topK_spearman(sil_dict, r_values, m, n_k, threshold=0.95):
    """Apply Algorithm 1 to score curves ordered by increasing K.
    
    Correlate the last m_eff scores with a strictly decreasing sequence,
    where m_eff = max(2, min(m, n_k - 1)). Choose the smallest candidate
    whose correlation reaches the threshold; if none does, choose a maximizer
    (the smallest candidate breaks ties). Reference labels and external
    correlations are not used here. Return the choice and diagnostic scores.
    The high-K window can overlap the evaluation range."""
    # Algorithm 1 scans candidates in increasing order. Here m is the paper's
    # requested window size s, and threshold is tau; n_k counts candidate K values.
    avail = sorted(r for r in r_values if r in sil_dict)
    if not avail:
        return r_values[0], {}
    # Implemented window convention: keep at most n_k-1 entries (at least two).
    # For K=2,...,35 and m=35 this gives 33 entries, K=3,...,35.
    # This clipping differs by one from the paper's K_lo formula when s spans
    # the entire grid; retain it here to preserve the existing numerical results.
    m_eff   = max(2, min(m, n_k - 1))
    top_idx = list(range(n_k - m_eff, n_k))
    # Algorithm 1 constructs v=(m_eff,...,1). Positive rank agreement with v
    # means the internal score decreases as K increases in the upper K window.
    ideal   = np.arange(m_eff, 0, -1, dtype=float)
    c_vals  = {}
    for r in avail:
        # C(candidate) = Spearman(score(K) over K_top, v). This is an internal
        # shape diagnostic; it does not compare the score with ACC or reference labels.
        obs        = np.asarray(sil_dict[r], dtype=float)[top_idx]
        rho        = spearmanr_safe(obs, ideal)
        # An undefined correlation receives -2, below the valid range [-1,1].
        # If every candidate is undefined, the fallback below returns the smallest one.
        c_vals[r]  = rho if not np.isnan(rho) else -2.0
    # Threshold branch of Algorithm 1: choose the FIRST (smallest) candidate
    # with C >= tau, rather than maximizing an external agreement score.
    r_star = next((r for r in avail if c_vals[r] >= threshold), None)
    if r_star is None:
        # Fallback branch: maximize C when the threshold is unmet. Sorted candidate
        # order makes the smallest candidate win an exact tie.
        r_star = max(avail, key=lambda r: c_vals[r])
    return r_star, c_vals


def max_K_eval(k_star):
    """Return K-hat = max(25, round(1.75 K*)) for the benchmark evaluation grid.
    
    Reported rank correlations use candidate counts K=2,...,K-hat. K*
    sets this experimental range; Algorithm 1 then selects from internal
    score curves without comparing them with the reference labels."""
    # K* specifies the benchmark range only. Each K is a requested partition size;
    # the experiment assesses its ranking and does not return an estimated best K.
    return int(max(25, round(1.75 * k_star)))


def max_K_select(k_star):
    """Add ten candidate cluster counts for Algorithm 1's diagnostic window.
    
    Scores at these extra K values help select the neighborhood or embedding
    parameter; they do not enter the reported external rank correlations."""
    # Algorithm 1 sets K_max = K_hat + b, using b=10 in the main comparisons.
    # The extra partitions help select r; reported agreement uses K <= K_hat.
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

    # For each proposed cluster, compare its points with its own centroid and
    # the closest competing centroid; no reference classes enter this calculation.
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
        # The cluster contribution is n_c * max(0, 1 - mean_i[d_own(i)/d_other(i)]).
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
        # SSW sums squared distances to each point's own centroid; SSB weights each
        # centroid's squared distance to the grand mean by its cluster size.
        ssw += np.sum((Xi - c_k) ** 2)
        ssb += Xi.shape[0] * np.sum((c_k - grand_mean) ** 2)
    if ssb == 0:
        return np.nan
    # WB = K*SSW/SSB is smaller-is-better. Callers negate it before comparing ranks.
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
    # Delegate density-based validity to hdbscan using float64 coordinates.
    # Unavailable or failed scores remain NaN so summaries can track missingness.
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
    # Process MST edges from shortest to longest, as in Kruskal's construction.
    # At a merge of weight w, w is the largest edge joining the two components.
    edges = sorted(zip(mst_coo.row, mst_coo.col, mst_coo.data), key=lambda e: e[2])

    # Initially each point is its own component. Track membership explicitly
    # so a merge can fill all newly connected point pairs at once.
    parent = list(range(n))
    members = [[i] for i in range(n)]

    def find(x):
        """Find the union-find component containing a point, compressing the search path."""
        # Follow parent links to the component root and shorten the path for later merges.
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
        # Every cross-component pair now has bottleneck distance w. Fill both
        # directions because the underlying distance is symmetric.
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
    # Mean distance to the k_use non-self neighbors measures local sparsity.
    # These quantities depend on the representation, so reuse them across K.
    Den = np.maximum(dists[:, 1:].mean(axis=1), 1e-12)
    fDen = Den / Den.max()

    Rel = Den[:, None] / Den[None, :]
    RelT = Den[None, :] / Den[:, None]
    # Density mismatch penalty: 1-exp(-(Den_i/Den_j + Den_j/Den_i - 2)).
    # Add this penalty times (Den_i+Den_j) to the Euclidean distance.
    fRel = 1.0 - np.exp(-(Rel + RelT - 2.0))
    nD = Den[:, None] + Den[None, :]
    drD = d + fRel * nD
    np.fill_diagonal(drD, 0.0)

    mst = minimum_spanning_tree(drD)
    # The MST yields the minimax connection distance: the largest edge on the
    # path between i and j. Density weights then turn conD into DD.
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

    # For each proposed cluster, separation is its smallest DD to any other cluster.
    sep = np.zeros(K)
    for i, ci in enumerate(clusters):
        idx_i = idx_by_c[ci]
        best = np.inf
        for j, cj in enumerate(clusters):
            if i == j:
                continue
            # Find the closest density-involved connection to this other cluster,
            # then retain the smallest separation across all competing clusters.
            sub = DD[np.ix_(idx_i, idx_by_c[cj])]
            best = min(best, sub.min())
        sep[i] = best

    # Compactness uses within-cluster Euclidean MST path distances. The final
    # index divides total separation by total compactness across clusters.
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
        # Use each unordered within-cluster pair once, excluding the diagonal,
        # when measuring the mean and spread of MST bottleneck distances.
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
    # Use h=n^(-1/(d+4)) on the min-max-scaled features used by the KDE index.
    return n ** (-1.0 / (d + 4))


def viasckde_precompute(X, bandwidth=None, kernel='gaussian'):
    '''Precompute the min-max-scaled distance matrix and per-point KDE
    density once per dataset (both independent of the partition).
    bandwidth=None (default) uses silverman_bandwidth(n, d) -- pass an
    explicit value to override.'''
    # This index additionally scales features to [0,1] before its KDE step.
    # It leaves the caller's clustering data and fixed partitions unchanged.
    X01 = MinMaxScaler().fit_transform(X)
    d = squareform(pdist(X01, metric='euclidean'))
    np.fill_diagonal(d, np.inf)  # exclude self when finding nearest
    if bandwidth is None:
        bandwidth = silverman_bandwidth(X01.shape[0], X01.shape[1])
    # Estimate density with bandwidth h=n^(-1/(D+4)) unless supplied explicitly.
    # The Gaussian kernel here belongs to VIASCKDE, not to an rNN weighting variant.
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
        # Normalize density inside each proposed cluster to weight its interior points.
        kde_k = (dens_k - dmin) / (dmax - dmin) if dmax > dmin else np.zeros(ni)

        d_same = d[np.ix_(idx, idx)]
        # The distance diagonal was set to infinity during precomputation, so
        # this minimum excludes the point itself; singletons are handled next.
        a_i = d_same.min(axis=1)
        a_i = np.where(np.isinf(a_i), 0.0, a_i)  # singleton clusters

        other_idx = np.where(~mask)[0]
        d_other = d[np.ix_(idx, other_idx)]
        b_i = d_other.min(axis=1)

        denom = np.maximum(np.maximum(a_i, b_i), 1e-12)
        # a_i/b_i are the nearest same/other-cluster distances, respectively.
        # Weight (b_i-a_i)/max(a_i,b_i) by normalized density and average over points.
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
    # Choose the within-cluster radius as the median distance to neighbor 2*min_pts.
    # A point is core if its min_pts-th non-self neighbor lies within that radius.
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
    # The largest core-MST edge is the radius needed to connect all core points.
    # This connectedness value is compared with cross-cluster separation in DCSI.
    mst = minimum_spanning_tree(d_core)
    return float(mst.data.max()) if mst.nnz > 0 else 0.0


def dcsi_index(X, labels, min_pts=5):
    '''
    DCSI (Gauss, Scheipl & Herrmann, 2025). Maximize (in [0,1)).
    The inline steps below define the implemented pairwise scores.
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
            # For this pair of clusters, separation is the smallest core-to-core distance;
            # connectedness is the larger within-cluster core MST edge. Form their ratio q.
            conn_max = max(conn[ci], conn[cj])
            if conn_max <= 0:
                q = np.inf if sep > 0 else 0.0
            else:
                q = sep / conn_max
            # Map q to q/(1+q), then average equally over cluster pairs for the final score.
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
    # Counts refer to ordered pairs of distinct observations. The common factor
    # of two cancels in the pair-count ratios used by Jaccard, SS3, and Powers.
    C = pair_confusion_matrix(y_true, y_pred)
    d, c = C[0, 0], C[0, 1]
    b, a = C[1, 0], C[1, 1]
    return float(a), float(b), float(c), float(d)


def jaccard_pairs(y_true, y_pred):
    '''Jaccard coefficient (pair-counting external index).'''
    # Count pairs kept together in both partitions relative to all pairs
    # kept together in at least one partition; pairs separated by both do not contribute.
    a, b, c, d = _pair_counts(y_true, y_pred)
    denom = a + b + c
    return a / denom if denom > 0 else 0.0


def sokal_sneath3(y_true, y_pred):
    '''Sokal & Sneath 3 pair-counting index: (a+d)/(b+c).'''
    # Compare agreeing pairs (a+d) with disagreeing pairs (b+c).
    # Perfect pair agreement can give infinity, which is retained in the score array.
    a, b, c, d = _pair_counts(y_true, y_pred)
    denom = b + c
    if denom <= 0:
        return np.inf if (a + d) > 0 else 0.0
    return (a + d) / denom


def powers_informedness(y_true, y_pred):
    ''''Powers' (Bookmaker Informedness / Youden's J) on the pair-counting
    confusion matrix: TPR + TNR - 1.'''
    # Combine the ability to keep same-class pairs together and to separate
    # different-class pairs: TPR + TNR - 1, with zero-denominator guards below.
    a, b, c, d = _pair_counts(y_true, y_pred)
    tpr = a / (a + b) if (a + b) > 0 else 0.0
    tnr = d / (c + d) if (c + d) > 0 else 0.0
    return tpr + tnr - 1.0


def nid_distance(y_true, y_pred):
    '''Normalized Information Distance (Vinh, Epps & Bailey, 2010):
    NID = 1 - I(U,V) / max(H(U), H(V)). A DISTANCE (0 = identical
    partitions), not a similarity -- see markdown above.'''
    # Compute mutual information first, then normalize by the larger of the
    # two entropy quantities below and subtract from one to form the distance.
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
    # ACC = max over one-to-one label matchings of (matched observations)/N.
    # The assignment solver minimizes cost, so negation or max-count subtraction
    # converts maximum overlap to its cost convention. Unmatched groups earn no credit.
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

# Every reported correlation is oriented so that a larger value means
# stronger agreement with the reference partition. NID is a distance, hence
# its raw values must be negated before correlation.
LOWER_IS_BETTER_EXTERNAL_METRICS = {'NID'}
SCORE_DIRECTION_CONVENTION = 'larger_is_better_v1'


def orient_external_values(metric, values):
    """Orient external scores so that larger values mean better agreement.
    
    For rank correlation, -NID and the Supplement's 1-NID induce the same
    ranking: they differ only by an additive constant."""
    # Orient external criteria before rank comparison so larger means better.
    # For NID, reversing its order is necessary because the raw quantity is a distance.
    values = np.asarray(values, dtype=float)
    return -values if metric in LOWER_IS_BETTER_EXTERNAL_METRICS else values

# ── cell: clustering-methods ──────────────────────────────────────────
# Each wrapper takes the SAME (X, K, seed) -> labels signature so the main
# loop below can call whichever clustering method is selected for this job
# without any per-method branching. K is the candidate cluster count being
# swept over K_TEST; seed is the per-replicate deterministic seed (see
# `seed = ...` in the main loop) so K-means/Spectral's internal random
# initialization is reproducible across reruns of the same replicate.
def kmeans_labels(X, K, seed):
    # Fit a separate K-means partition for this K by minimizing within-cluster
    # squared Euclidean distances. Reuse these labels for all validity scores.
    """Fit the requested K-means partition on X using the experiment seed."""
    return KMeans(n_clusters=K, random_state=seed, n_init='auto').fit_predict(X)


def spectral_labels(X, K, seed):
    # Affinity is built from a k-NN graph (fixed at n_neighbors=10,
    # independent of the rNN r grid used by the validity indices below) --
    # this is standard sklearn spectral clustering, not the paper's rNN
    # affinity. assign_labels='kmeans' runs k-means on the spectral
    # embedding as the final labeling step.
    # Construct the fixed 10-neighbor affinity graph, then partition its spectral
    # representation into K groups. Its 10 neighbors are independent of validation r.
    """Fit K spectral clusters using the fixed nearest-neighbor affinity graph."""
    return SpectralClustering(
        n_clusters=K, random_state=seed,
        affinity='nearest_neighbors', n_neighbors=10,
        assign_labels='kmeans',
    ).fit_predict(X)


def agglomerative_labels(X, K, seed):
    # Ward linkage is a deterministic bottom-up merge given (X, K), so
    # `seed` is accepted only to keep the shared 3-argument signature and
    # is otherwise unused here.
    # Ward linkage merges groups to minimize the increase in within-cluster
    # sum of squares, stopping at the requested K groups.
    """Fit K agglomerative clusters using Ward linkage on Euclidean features."""
    return AgglomerativeClustering(n_clusters=K, linkage='ward').fit_predict(X)


CLUSTERING_METHOD_FUNCS = {
    'KMeans':        kmeans_labels,
    'Spectral':       spectral_labels,
    'Agglomerative':  agglomerative_labels,
}

# ── cell: settings ────────────────────────────────────────────────────
K_STAR_VALUES      = [6, 30]
D_VALUES           = [2, 10, 200]
COMPACTNESS_VALUES = {'compact': 0.1, 'sparse': 0.8}
COMPACTNESS_INDEX  = {'compact': 0, 'sparse': 1}   # for deterministic seeding only

CASES = [
    (k_star, D, comp_label)
    for k_star in K_STAR_VALUES
    for D in D_VALUES
    for comp_label in COMPACTNESS_VALUES
]

N_REP     = 100
R_VALUES  = list(range(5, 135, 5))      # r grid for the rNN affinities, up to 130
# Selection notation: code M corresponds to window size s in Algorithm 1;
# THRESHOLD corresponds to tau. Neither is the number of clusters K.
M         = 35                          # fixed m for the r* Spearman criterion
THRESHOLD = 0.95
# K_TEST is now per-case (depends on k_star via max_K_select/max_K_eval above)
# -- computed inside the main loop, not as a single flat range.

ALL_METHODS = ['VRC', 'Silhouette', 'WG', 'WB', 'DBCV', 'rNN_Hard', 'CVDD', 'VIASCKDE', 'DCSI']
CLUSTERING_METHODS = ['KMeans', 'Spectral', 'Agglomerative']

# ── Choose which method(s) to (re-)compute ──────────────────────────────
METHODS_TO_RUN = ALL_METHODS

# ── Choose the clustering algorithm for this run ────────────────────────
# One job = one clustering method (all 7 external metrics are always
# batched together within a job -- see docstring). Pass the method name as
# a command-line argument, e.g. `python partA_type2_grid_comparison.py
# Spectral`; defaults to KMeans if no argument is given (e.g. when run
# from a notebook).
_cli_cm = sys.argv[1] if len(sys.argv) > 1 else 'KMeans'
assert _cli_cm in CLUSTERING_METHODS, f'Unknown clustering method {_cli_cm!r}'
CLUSTERING_METHODS_TO_RUN = [_cli_cm]

# ── External validity metrics to batch in this run ──────────────────────
# All 7 by default (Jaccard, SS3, ARI, NMI, Powers, NID, ACC) -- computing
# extra metrics here is nearly free since clustering + rNN r* selection
# are shared across metrics (see docstring).
EXTERNAL_METRICS_TO_RUN = list(EXTERNAL_METRIC_FUNCS.keys())

def case_key(case):
    '''Human-readable identifier for one (k_star, D, comp_label) grid cell,
    e.g. (6, 2, "compact") -> "k6_D2_compact". Used as the dict key that
    indexes into `results[metric][clustering_method][...]` and as the
    lookup key downstream in merge_case2_results.py / generate_case2_tables.py.'''
    # Encode K*, dimension, and compactness in a stable key shared by
    # the experiment cache and downstream tables; no statistical quantity is computed here.
    k_star, D, comp_label = case
    return f'k{k_star}_D{D}_{comp_label}'

def sample_size_for(k_star):
    """Use N = min(1000, 50 K*): 300 points for K*=6 and 1000 for K*=30."""
    # Cap N at 1000 while targeting 50 observations per true component:
    # K*=6 gives N=300, and K*=30 gives N=1000 with nearly balanced component sizes.
    return min(1000, k_star * 50)

# ── Optional per-job output tag, for safe PARALLEL execution ───────────
# Set via the JOB_TAG environment variable (see the per-method .dlx
# files). When set, this job's results are written to their OWN private
# files, completely disjoint from every other job's files -- so all 3
# clustering-method jobs can run at the same time with zero risk of one
# job's periodic save clobbering another's, no matter what stage each is
# at. Merge everything back into the canonical (untagged) files
# afterward with merge_case2_results.py.
JOB_TAG = os.environ.get('JOB_TAG', '').strip()

def results_path_for(metric):
    """Return the per-metric cache path, with an optional job-specific suffix.
    The historical nonconvex_ prefix refers to simulated Case 2 here; the
    fixed nonconvex benchmarks are Case 3. The repository runner uses no suffix."""
    # Keep the established cache naming scheme so the table/plot readers
    # can find results for the same experiment without changing the saved data format.
    suffix = f'_{JOB_TAG}' if JOB_TAG else ''
    return (
        f'data/nonconvex_case_comparison_results_nonself_oriented{suffix}.pkl' if metric == 'ACC'
        else f'data/nonconvex_case_comparison_results_nonself_oriented{suffix}_{metric}.pkl'
    )

print(f'{len(CASES)} cases:')
for c in CASES:
    print(' ', case_key(c), '->', c)
print(f'\nN_REP = {N_REP}')
for k_star in K_STAR_VALUES:
    print(f'k*={k_star}: max_K_select={max_K_select(k_star)} max_K_eval={max_K_eval(k_star)}  m={M}')
print(f'Methods to run this session: {METHODS_TO_RUN}')
print(f'Clustering method this session: {CLUSTERING_METHODS_TO_RUN[0]}')
print(f'External metrics batched this session: {EXTERNAL_METRICS_TO_RUN}')

# ── cell: load-results ────────────────────────────────────────────────
# results[metric][clustering_method][case_key][method] -> list of rho, one per rep
os.makedirs('data', exist_ok=True)
results = {}
for metric in EXTERNAL_METRICS_TO_RUN:
    path = results_path_for(metric)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            payload = pickle.load(f)
        results[metric] = payload['results']
        print(f'Loaded existing results from {path}:')
        for cm_name in results[metric]:
            print('   ', cm_name, ':')
            for ckey in results[metric][cm_name]:
                print('      ', ckey, '->', sorted(results[metric][cm_name][ckey].keys()))
    else:
        results[metric] = {}
        print(f'No existing results file found at {path} — starting fresh.')

def save_results(results):
    """Write one result snapshot per external criterion, including score orientation.
    Periodic writes retain partial results, but rerunning this script recomputes
    requested cases. The experiment runner resumes only at completed-job boundaries."""
    for metric in EXTERNAL_METRICS_TO_RUN:
        # Store correlations together with the metadata below, including the
        # score-direction convention, so readers can interpret the cached values.
        payload = {
            'results': results[metric],
            'external_metric': metric,
            'score_direction_convention': SCORE_DIRECTION_CONVENTION,
            'clustering_methods_computed': sorted(results[metric].keys()),
        }
        # Write the currently accumulated result store as a checkpoint for later
        # table generation and plotting; this step does not aggregate the scores.
        with open(results_path_for(metric), 'wb') as f:
            pickle.dump(payload, f)

# ── cell: main-loop ───────────────────────────────────────────────────
# NOTE (multi-metric batching): r* selection and every index's raw
# per-K score curve depend only on (X, labels_by_K) -- never on the
# external metric. So each is computed exactly ONCE per (case, rep), and
# only the final Spearman-vs-ext_vals step is repeated per metric below.

need_rnn      = 'rNN_Hard' in METHODS_TO_RUN
need_cvdd     = 'CVDD' in METHODS_TO_RUN
need_viasckde = 'VIASCKDE' in METHODS_TO_RUN

for case in CASES:
    k_star, D, comp_label = case
    ckey = case_key(case)
    compactness = COMPACTNESS_VALUES[comp_label]
    n_samples = sample_size_for(k_star)
    max_K_s = max_K_select(k_star)
    max_K_e = max_K_eval(k_star)
    # K_TEST is the FULL candidate-K sweep for this case, out to the wider
    # selection cap (max_K_select) -- every raw per-K score curve below is
    # computed over this whole range. eval_idx then picks out just the
    # positions with K <= max_K_eval, i.e. the narrower range that every
    # reported Spearman rho is actually computed over (see max_K_eval's
    # docstring). r* selection (rNN_Hard) uses the FULL K_TEST
    # range, not eval_idx -- that's the whole point of the +10 buffer.
    # K indexes alternative partitions of the same data. Sweep through K_hat+b
    # for selection, then restrict to the evaluation subset when reporting rho.
    K_TEST = list(range(2, max_K_s + 1))
    n_k = len(K_TEST)
    K_idx_lookup = {K: i for i, K in enumerate(K_TEST)}
    eval_idx = [K_idx_lookup[K] for K in K_TEST if K <= max_K_e]

    for metric in EXTERNAL_METRICS_TO_RUN:
        for cm_name in CLUSTERING_METHODS_TO_RUN:
            results[metric].setdefault(cm_name, {})
            results[metric][cm_name].setdefault(ckey, {})
            for method in METHODS_TO_RUN:
                results[metric][cm_name][ckey][method] = []

    print(f'\n=== Case {ckey}  (k*={k_star}, D={D}, compactness={compactness}, N={n_samples}) ===')

    for rep in range(N_REP):
        if (rep + 1) % 20 == 0:
            print(f'  rep {rep+1}/{N_REP}')
            save_results(results)

        # Deterministic seed (no hash() — avoids PYTHONHASHSEED nondeterminism
        # across separate staged runs). Independent of clustering method, so
        # every method sees the exact same generated dataset per replicate.
        seed = k_star * 100_000 + D * 100 + COMPACTNESS_INDEX[comp_label] * 10 + rep

        X, y = simulate_type2_data(n_samples, D, k_star, compactness, random_state=seed)
        # Preprocess each feature as (x - sample mean)/sample standard deviation
        # (StandardScaler uses ddof=0 and leaves constant features at zero).
        # Use the same transformed X for clustering and every original-data score.
        X = StandardScaler().fit_transform(X)

        cvdd_stats     = cvdd_precompute(X) if need_cvdd else None
        viasckde_stats = viasckde_precompute(X) if need_viasckde else None

        for cm_name in CLUSTERING_METHODS_TO_RUN:
            cm_fn = CLUSTERING_METHOD_FUNCS[cm_name]

            # Cluster once per candidate K (shared by every validity index
            # and every external metric below -- clustering is by far the
            # most expensive repeated step, so this dict is the crux of the
            # "multi-metric batching" savings described in the docstring).
            # Fit each candidate partition once. All indices and all r values below
            # rank these same partitions within this one clustering algorithm.
            labels_by_K = {K: cm_fn(X, K, seed) for K in K_TEST}
            # Some indices (e.g. silhouette, VRC) are undefined for a
            # degenerate single-cluster labeling; multi_cluster flags which
            # K's actually produced >1 cluster so those spots can be NaN'd
            # out below instead of raising.
            multi_cluster = {K: len(np.unique(labels_by_K[K])) > 1 for K in K_TEST}

            # ext_vals computed once per metric, shared across every index below.
            # Step 1: look up each requested metric's scoring function.
            ext_vals_by_metric = {
                metric: EXTERNAL_METRIC_FUNCS[metric]
                for metric in EXTERNAL_METRICS_TO_RUN
            }
            # Step 2: evaluate that function against the true labels `y` for
            # every K in K_TEST, restrict to the eval_idx subset (K <=
            # max_K_eval), and orient the result (flip NID's sign) so a
            # larger value always means "more agreement with truth" --
            # required before every downstream Spearman correlation.
            ext_vals_by_metric = {
                metric: orient_external_values(
                    metric,
                    np.array([fn(y, labels_by_K[K]) for K in K_TEST])[eval_idx],
                )
                for metric, fn in ext_vals_by_metric.items()
            }

            # ── raw per-K score curves, computed once regardless of metric count ──
            raw_curves = {}

            if 'VRC' in METHODS_TO_RUN:
                raw_curves['VRC'] = np.array([
                    calinski_harabasz_score(X, labels_by_K[K]) if multi_cluster[K] else np.nan
                    for K in K_TEST
                ])

            if 'Silhouette' in METHODS_TO_RUN:
                raw_curves['Silhouette'] = np.array([
                    silhouette_score(X, labels_by_K[K], metric='euclidean') if multi_cluster[K] else np.nan
                    for K in K_TEST
                ])

            if 'WG' in METHODS_TO_RUN:
                raw_curves['WG'] = np.array([
                    wemmert_gancarski_index(X, labels_by_K[K]) if multi_cluster[K] else np.nan
                    for K in K_TEST
                ])

            if 'WB' in METHODS_TO_RUN:
                # WB is minimized; negate it so larger consistently means better.
                raw_curves['WB'] = -np.array([
                    wb_index(X, labels_by_K[K]) if multi_cluster[K] else np.nan
                    for K in K_TEST
                ])

            if 'DBCV' in METHODS_TO_RUN:
                raw_curves['DBCV'] = np.array([
                    dbcv_index(X, labels_by_K[K]) if multi_cluster[K] else np.nan
                    for K in K_TEST
                ])

            if need_cvdd:
                raw_curves['CVDD'] = np.array([
                    cvdd_score(cvdd_stats, labels_by_K[K]) if multi_cluster[K] else np.nan
                    for K in K_TEST
                ])

            if need_viasckde:
                raw_curves['VIASCKDE'] = np.array([
                    viasckde_score(viasckde_stats, labels_by_K[K]) if multi_cluster[K] else np.nan
                    for K in K_TEST
                ])

            if 'DCSI' in METHODS_TO_RUN:
                raw_curves['DCSI'] = np.array([
                    dcsi_index(X, labels_by_K[K]) if multi_cluster[K] else np.nan
                    for K in K_TEST
                ])

            if need_rnn:
                # Cache the binary-neighbor Silhouette curve for each r over the full K grid.
                # These same curves support parameter selection and subsequent evaluation.
                # For partition C_K, a_i is mean d_r within its cluster (excluding self),
                # b_i is the smallest mean d_r to another cluster, and
                # S_r(K) = mean_i[(b_i-a_i)/max(a_i,b_i)], with 0 for a_i=b_i=0 or a singleton.
                # Vary r to change d_r while retaining C_K; each array scores one r across K.
                sil_hard_by_r = {}
                for r in R_VALUES:
                    D_hard = rnn_affinity_distance(X, r=r)
                    sil_hard_by_r[r] = np.array([
                        silhouette_score(D_hard, labels_by_K[K], metric='precomputed') if multi_cluster[K] else np.nan
                        for K in K_TEST
                    ])

                # Select r from the internal curve shape without consulting external scores.
                # Retain the selected curve for the external rank-correlation calculation.
                if 'rNN_Hard' in METHODS_TO_RUN:
                    r_star_h, _ = select_r_topK_spearman(sil_hard_by_r, R_VALUES, m=M, n_k=n_k, threshold=THRESHOLD)
                    raw_curves['rNN_Hard'] = sil_hard_by_r[r_star_h]


            # ── only this final step is repeated per external metric ──
            # raw_curves now holds one full K_TEST-length score curve per
            # validity index (index -> array), computed exactly once for
            # this (case, clustering_method, rep). For each external
            # metric, restrict each curve to eval_idx (the narrower
            # max_K_eval range) and record its Spearman rho against that
            # metric's oriented values -- one rho appended per (metric,
            # index) per replicate, accumulated into `results` for later
            # averaging in scripts/build_tables.py.
            for metric in EXTERNAL_METRICS_TO_RUN:
                ext_vals = ext_vals_by_metric[metric]
                for method, curve in raw_curves.items():
                    results[metric][cm_name][ckey][method].append(
                        spearmanr_safe(curve[eval_idx], ext_vals)
                    )

    save_results(results)
    print(f'  case {ckey} done, progress saved for metrics: {EXTERNAL_METRICS_TO_RUN}')

print('\nAll cases completed!')

# ── cell: save-results ────────────────────────────────────────────────
save_results(results)
print(f'Results saved for metrics: {[results_path_for(m) for m in EXTERNAL_METRICS_TO_RUN]}')

# ── cell: summary-table ───────────────────────────────────────────────
# Optional in-session sanity check (only meaningful when this script is run
# interactively/as a notebook cell with `results` already populated in
# memory for this session's clustering method -- NOT a substitute for
# generate_case2_tables.py, which re-derives its own tables straight from
# the saved pickles, across all 3 clustering methods, after merging).
# Flattens `results` into one row per (metric, clustering_method, case,
# index), with the across-replications mean/std Spearman rho for a quick
# eyeball check before the results are shipped off-cluster.
rows = []
for metric in EXTERNAL_METRICS_TO_RUN:
    for cm_name in CLUSTERING_METHODS:
        for case in CASES:
            ckey = case_key(case)
            k_star, D, comp_label = case
            for method in ALL_METHODS:
                vals = np.array(results.get(metric, {}).get(cm_name, {}).get(ckey, {}).get(method, []), dtype=float)
                vals = vals[np.isfinite(vals)]  # drop NaN and +/-inf (e.g. degenerate SS3)
                if len(vals) == 0:
                    continue
                rows.append({
                    'clustering_method': cm_name,
                    'external_metric': metric,
                    'k*': k_star, 'D': D, 'compactness': comp_label,
                    'method': method, 'n': len(vals),
                    'mean_rho': round(float(vals.mean()), 4),
                    'std_rho': round(float(vals.std()), 4),
                })

summary_df = pd.DataFrame(rows)
summary_df

# ── cell: boxplots ────────────────────────────────────────────────────
def plot_case_boxplot(results, metric, cm_name, case, methods_order):
    '''Draw and save one boxplot (one box per validity index in
    methods_order) of the per-replicate Spearman rho values for a single
    (metric, clustering_method, case) combination -- a quick visual
    diagnostic of each index's rho distribution/spread across the N_REP
    replications, saved as both PDF and PNG under
    ../../../../figures/nonconvex_clusters/<cm_name>/<metric>/. Like the
    summary-table cell above, this is a session-local convenience (only
    plots whatever `results` already holds in memory) and is independent
    of the LaTeX tables produced by generate_case2_tables.py.'''
    ckey = case_key(case)
    cm_results = results.get(metric, {}).get(cm_name, {}).get(ckey, {})
    available = [m for m in methods_order if len(cm_results.get(m, [])) > 0]
    if not available:
        print(f'No results to plot for {metric} / {cm_name} / {ckey}')
        return

    # Each method supplies its replication-level correlations for this one setting.
    # Filter nonfinite entries before summarizing their distribution in a box.
    data = []
    for m in available:
        vals = np.array(cm_results[m], dtype=float)
        data.append(vals[np.isfinite(vals)])

    fig, ax = plt.subplots(figsize=(1.6 * len(available) + 2, 6))
    # Boxes show quartiles and medians across replications, using one box per index.
    # This diagnostic complements the means/medians exported to the paper tables.
    bp = ax.boxplot(data, tick_labels=available, patch_artist=True)
    palette = sns.color_palette('tab10', n_colors=len(available))
    for patch, color in zip(bp['boxes'], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.axhline(0, color='grey', linewidth=1, linestyle='--')
    ax.set_ylabel(f'Spearman $\\rho$ with {metric}', fontsize=13)
    k_star, D, comp_label = case
    ax.set_title(f'{cm_name} — $k^*={k_star}$, $D={D}$, compactness={comp_label}',
                fontsize=13, fontweight='bold')
    ax.tick_params(axis='x', labelsize=10, rotation=30)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    # Keep diagnostic exports separated by experiment/algorithm so successive
    # plots do not overwrite results from a different comparison.
    out_dir = f'../../../../figures/nonconvex_clusters/{cm_name}/{metric}'
    os.makedirs(out_dir, exist_ok=True)
    fname = f'NonconvexCase_{ckey}'
    plt.savefig(f'{out_dir}/{fname}.pdf', bbox_inches='tight')
    plt.savefig(f'{out_dir}/{fname}.png', dpi=150, bbox_inches='tight')
    plt.show()


for metric in EXTERNAL_METRICS_TO_RUN:
    for cm_name in CLUSTERING_METHODS:
        for case in CASES:
            plot_case_boxplot(results, metric, cm_name, case, ALL_METHODS)
