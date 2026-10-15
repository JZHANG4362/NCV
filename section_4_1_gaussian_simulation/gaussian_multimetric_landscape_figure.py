"""Supplement Section S7: Figures S1 (K-means), S2 (spectral), S3 (Ward).

Use the representative Case 1 dataset: N=2000, D=20, K*=10, component
standard deviation 1, data seed 1, and no feature standardization.
For each clustering algorithm, select hard rNN's r from 5,10,...,130
using Algorithm 1 on K=2,...,35. Compare ordinary and selected rNN
Silhouette rankings with external criteria on K=2,...,25.
Figure S1 omits ACC because it appears in main text Figure 2; Figures S2
and S3 include all seven criteria. Each algorithm has its own selection.

Run here: python gaussian_multimetric_landscape_figure.py
The script reuses its saved numerical cache when available."""

from __future__ import annotations

import os
import pickle
from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy, spearmanr
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import (
    adjusted_rand_score,
    mutual_info_score,
    normalized_mutual_info_score,
    pair_confusion_matrix,
    silhouette_score,
)
from sklearn.metrics.cluster import contingency_matrix
from sklearn.neighbors import NearestNeighbors


# Paths are relative to this script; run from section_4_1_gaussian_simulation/.
OUT_DIR = "figures/combined"
CACHE_PATH = "data/gaussian_multimetric_landscape_results_unstandardized_nonself_oriented.pkl"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

N = 2000            # number of samples
D = 20              # ambient dimensionality
K_STAR = 10         # true number of Gaussian clusters (ground-truth K*)
SEED = 1            # seed for the representative dataset (matches gaussian_simulation.ipynb)
K_MAX_EVAL = 25      # max(25, 1.75*K_STAR) -- evaluation cap in main text Section 4.1
K_MAX_SELECT = 35    # K_MAX_EVAL + 10 buffer -- wider K-range used only for r* SELECTION (Algorithm 1)
# K indexes alternative partitions of the same data. Sweep through K_hat+b
# for selection, then restrict to the evaluation subset when reporting rho.
K_TEST = list(range(2, K_MAX_SELECT + 1))   # K sweep for clustering & r* selection window (2..35)
K_EVAL = list(range(2, K_MAX_EVAL + 1))     # K subrange used for every reported/plotted rho (2..25)
EVAL_IDX = np.array([K_TEST.index(k) for k in K_EVAL])   # indices into K_TEST-ordered arrays that restrict to K_EVAL
R_VALUES = list(range(5, 135, 5))    # candidate neighborhood sizes r for r* selection: 5,10,...,130
# Selection notation: code M corresponds to window size s in Algorithm 1;
# THRESHOLD corresponds to tau. Neither is the number of clusters K.
M = 35            # window size for the top-m monotonic-decrease selection criterion
THRESHOLD = 0.95   # minimum Spearman rho (vs. ideal decreasing sequence) required to accept a candidate r

# Clustering methods and external validity metrics swept over by this
# script (vs. gaussian_simulation.ipynb's single K-means / ACC-ARI-NMI).
METHODS = ("KMeans", "Spectral", "Agglomerative")
METRIC_ORDER = ("Jaccard", "SS3", "ARI", "Powers", "NMI", "NID", "ACC")
METRIC_LABELS = {
    "Jaccard": "Jaccard",
    "SS3": "SS3",
    "ARI": "ARI",
    "Powers": "Powers' informedness",
    "NMI": "NMI",
    "NID": "NID",
    "ACC": "ACC",
}
METRIC_AXIS_LABELS = {**METRIC_LABELS, "NID": "$1-\\mathrm{NID}$"}
SCORE_DIRECTION_CONVENTION = "larger_is_better_v1"
PANEL_LETTERS = dict(zip(METRIC_ORDER, "abcdefg"))


def _pair_counts(y_true, y_pred):
    """Return the four pair-counting-based confusion counts (a, b, c, d)
    used by several external validity metrics below:
      a = pairs in the SAME true class and SAME predicted cluster
      b = pairs in the SAME true class but DIFFERENT predicted clusters
      c = pairs in DIFFERENT true classes but the SAME predicted cluster
      d = pairs in DIFFERENT true classes and DIFFERENT predicted clusters
    sklearn's pair_confusion_matrix returns a 2x2 matrix laid out as
    [[d, c], [b, a]], so the indices below just unpack that ordering.
    """
    # Counts refer to ordered pairs of distinct observations. The common factor
    # of two cancels in the pair-count ratios used by Jaccard, SS3, and Powers.
    matrix = pair_confusion_matrix(y_true, y_pred)
    d, c = matrix[0, 0], matrix[0, 1]
    b, a = matrix[1, 0], matrix[1, 1]
    return float(a), float(b), float(c), float(d)


def jaccard_pairs(y_true, y_pred):
    """Jaccard index over point pairs: a / (a + b + c) -- the fraction of
    "agreeing" pairs (same-cluster in both partitions) among all pairs
    that agree in at least one of the two partitions."""
    # Count pairs kept together in both partitions relative to all pairs
    # kept together in at least one partition; pairs separated by both do not contribute.
    a, b, c, _ = _pair_counts(y_true, y_pred)
    denominator = a + b + c
    return a / denominator if denominator > 0 else 0.0


def sokal_sneath3(y_true, y_pred):
    """Sokal-Sneath 3 pair-counting index: (a + d) / (b + c) -- ratio of
    agreeing pairs to disagreeing pairs. Unbounded above (can be +inf when
    b + c = 0, i.e. perfect agreement); display_values() below maps +inf
    to a finite value just above the panel's other points for plotting."""
    # Compare agreeing pairs (a+d) with disagreeing pairs (b+c).
    # Perfect pair agreement can give infinity, which is retained in the score array.
    a, b, c, d = _pair_counts(y_true, y_pred)
    denominator = b + c
    if denominator <= 0:
        return np.inf if (a + d) > 0 else 0.0
    return (a + d) / denominator


def powers_informedness(y_true, y_pred):
    """Powers' informedness (a.k.a. Youden's J / bookmaker informedness)
    for pair-counting: TPR + TNR - 1, where TPR = a/(a+b) is the fraction
    of same-class pairs correctly kept together, and TNR = d/(c+d) is the
    fraction of different-class pairs correctly kept apart."""
    # Combine the ability to keep same-class pairs together and to separate
    # different-class pairs: TPR + TNR - 1, with zero-denominator guards below.
    a, b, c, d = _pair_counts(y_true, y_pred)
    tpr = a / (a + b) if (a + b) > 0 else 0.0
    tnr = d / (c + d) if (c + d) > 0 else 0.0
    return tpr + tnr - 1.0


def nid_distance(y_true, y_pred):
    """Retain the historical NID calculation used for Figures S1--S3.
    This script applies scipy.stats.entropy directly to label arrays, unlike
    the benchmark helpers, which use sklearn partition entropy. The convention
    is preserved here to keep saved figure results unchanged. The plotting
    step uses 1-NID to orient the resulting values."""
    # Compute mutual information first, then normalize by the larger of the
    # two entropy quantities below and subtract from one to form the distance.
    mi = mutual_info_score(y_true, y_pred)
    # This historical figure helper applies scipy entropy to numeric labels;
    # retain that behavior here rather than silently changing the saved convention.
    denominator = max(entropy(y_true), entropy(y_pred))
    return 0.0 if denominator <= 0 else 1.0 - mi / denominator


def clustering_accuracy(y_true, y_pred):
    """Clustering accuracy via the Hungarian algorithm: builds the
    predicted-cluster x true-class contingency matrix, finds the best
    1-to-1 label matching maximizing the matched count (linear_sum_assignment
    minimizes, hence the negated matrix), and returns that count as a
    fraction of all points."""
    matrix = contingency_matrix(y_true, y_pred)
    # ACC = max over one-to-one label matchings of (matched observations)/N.
    # The assignment solver minimizes cost, so negation or max-count subtraction
    # converts maximum overlap to its cost convention. Unmatched groups earn no credit.
    row_ind, col_ind = linear_sum_assignment(-matrix)
    return matrix[row_ind, col_ind].sum() / matrix.sum()


METRIC_FUNCS = OrderedDict(
    Jaccard=jaccard_pairs,
    SS3=sokal_sneath3,
    ARI=adjusted_rand_score,
    Powers=powers_informedness,
    NMI=normalized_mutual_info_score,
    NID=nid_distance,
    ACC=clustering_accuracy,
)


def orient_external_values(metric, values):
    """Orient external scores so that larger values mean better agreement.
    
    For rank correlation, -NID and the Supplement's 1-NID induce the same
    ranking: they differ only by an additive constant."""
    # Orient external criteria before rank comparison so larger means better.
    # For NID, reversing its order is necessary because the raw quantity is a distance.
    values = np.asarray(values, dtype=float)
    # NID is a distance. Using 1-NID preserves a familiar [0, 1] scale while
    # reversing its ranking direction for the correlation and plot.
    return 1.0 - values if metric == "NID" else values


def cluster_labels(method, X, k, seed):
    """Dispatch to one of the three clustering algorithms this script
    compares, all called with a fixed number of clusters k. Spectral
    clustering here uses a k-nearest-neighbor affinity graph (n_neighbors=10,
    a fixed construction choice unrelated to the r* grid used for the rNN
    silhouette itself) and Agglomerative uses Ward linkage on Euclidean
    distance -- both standard defaults for comparing against K-means."""
    # Fit a new partition at the supplied cluster count k. Reuse its labels
    # for the ordinary and rNN scores so the comparison changes only the validity index.
    if method == "KMeans":
        return KMeans(n_clusters=k, random_state=seed, n_init="auto").fit_predict(X)
    if method == "Spectral":
        return SpectralClustering(
            n_clusters=k,
            random_state=seed,
            affinity="nearest_neighbors",
            n_neighbors=10,
            assign_labels="kmeans",
        ).fit_predict(X)
    if method == "Agglomerative":
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    raise ValueError(f"Unknown clustering method: {method}")


def spearmanr_safe(a, b):
    """Spearman correlation, returning NaN instead of raising when fewer
    than 2 valid (non-NaN, paired) observations remain."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    # Compare paired entries only. In the experiments an entry is a candidate
    # partition (one K), not an observation; SciPy averages tied ranks.
    mask = ~np.isnan(a) & ~np.isnan(b)
    if mask.sum() < 2:
        return np.nan
    return float(spearmanr(a[mask], b[mask]).statistic)


def select_r_topK_spearman(curves, r_values, m, n_k, threshold=0.95):
    """Apply Algorithm 1 to score curves ordered by increasing K.
    
    Correlate the last m_eff scores with a strictly decreasing sequence,
    where m_eff = max(2, min(m, n_k - 1)). Choose the smallest candidate
    whose correlation reaches the threshold; if none does, choose a maximizer
    (the smallest candidate breaks ties). Reference labels and external
    correlations are not used here. Return the choice and diagnostic scores.
    The high-K window can overlap the evaluation range."""
    # Algorithm 1 scans candidates in increasing order. Here m is the paper's
    # requested window size s, and threshold is tau; n_k counts candidate K values.
    available = sorted(r for r in r_values if r in curves)
    # Implemented window convention: keep at most n_k-1 entries (at least two).
    # For K=2,...,35 and m=35 this gives 33 entries, K=3,...,35.
    # This clipping differs by one from the paper's K_lo formula when s spans
    # the entire grid; retain it here to preserve the existing numerical results.
    m_eff = max(2, min(m, n_k - 1))                  # can't exceed the number of available K's
    top_idx = list(range(n_k - m_eff, n_k))            # indices of the top-m_eff largest K's
    # Algorithm 1 constructs v=(m_eff,...,1). Positive rank agreement with v
    # means the internal score decreases as K increases in the upper K window.
    ideal = np.arange(m_eff, 0, -1, dtype=float)       # ideal strictly-decreasing reference [m_eff, ..., 1]
    # C(candidate) = Spearman(score(K) over K_top, v). This is an internal
    # shape diagnostic; it does not compare the score with ACC or reference labels.
    scores = {
        r: spearmanr_safe(np.asarray(curves[r])[top_idx], ideal)
        for r in available
    }
    # Threshold branch of Algorithm 1: choose the FIRST (smallest) candidate
    # with C >= tau, rather than maximizing an external agreement score.
    selected = next(
        (r for r in available if np.isfinite(scores[r]) and scores[r] >= threshold),
        None,
    )
    if selected is None:
        # Fallback branch: maximize C when the threshold is unmet. Sorted candidate
        # order makes the smallest candidate win an exact tie.
        selected = max(
            available,
            key=lambda r: scores[r] if np.isfinite(scores[r]) else -2.0,
        )
    return selected, scores


def hard_rnn_distance(dists_max, idx_max, r):
    """Build the Hard rNN affinity distance matrix for one candidate r,
    reusing neighbor indices already queried out to max(R_VALUES) so this
    doesn't need to re-run a k-NN search for every r (see compute_results
    below, which queries `max(R_VALUES) + 1` neighbors once up front)."""
    n = idx_max.shape[0]
    # idx_max includes each query point in column zero.  Discard it so r
    # counts exactly r other observations (Case 1 non-self-neighbor
    # convention).
    idx = idx_max[:, 1:r + 1]
    # Section 3.1: first form the directed binary affinity B_ij = 1{j in N_r(i)}.
    # Neighbor search uses features only; neither the reference labels nor K enter B.
    affinity = np.zeros((n, n), dtype=float)
    affinity[np.arange(n)[:, None], idx] = 1.0
    # Symmetric union: A_ij = max(B_ij, B_ji) = 1{j in N_r(i) OR i in N_r(j)}.
    # An edge in either direction is enough; this is not a mutual-neighbor graph.
    affinity = np.maximum(affinity, affinity.T)   # symmetrise (OR rule)
    np.fill_diagonal(affinity, 0.0)
    # The paper uses dissimilarities: d_r(i,j)=1-A_ij off the diagonal, d_r(i,i)=0.
    # Thus linked points have distance 0 and unlinked points have distance 1.
    # Pass this N-by-N matrix as metric="precomputed", not as N feature vectors.
    distance = 1.0 - affinity
    np.fill_diagonal(distance, 0.0)
    return distance


def compute_results():
    """Run the full pipeline from scratch: generate the representative
    dataset, cluster it with each of the three methods across K=2..35,
    score against every external metric (restricted to K=2..25), build the
    rNN-Hard silhouette curves for every candidate r=5..130 (reusing one
    shared k-NN query), select r* per method via Algorithm 1, and package
    everything into a dict that gets pickled to CACHE_PATH so subsequent
    runs of this script can skip straight to plotting (see main()).
    """
    print("Generating the unstandardized representative Gaussian dataset...")
    # Figures S1--S3 use the Case 1 model: uniform centers in [-10,10]^20,
    # then 200 observations per component with covariance I_20. Keep raw features
    # for these representative-data plots; seed 1 fixes this particular draw.
    X, y_true = make_blobs(
        n_samples=N,
        n_features=D,
        centers=K_STAR,
        cluster_std=1,
        random_state=SEED,
    )
    labels = {}
    ordinary = {}
    external = {}
    for method in METHODS:
        print(f"Clustering with {method}: K=2..{K_MAX_SELECT}")
        # Cluster once per (method, K) over the WIDE K_TEST range (2..35) --
        # these labels are reused both for the ordinary-silhouette curve
        # and, further below, for every candidate r's rNN silhouette curve.
        labels[method] = {
            k: cluster_labels(method, X, k, SEED) for k in K_TEST
        }
        ordinary[method] = np.array(
            [silhouette_score(X, labels[method][k]) for k in K_TEST]
        )
        # External metrics are computed only over the narrower K_EVAL range
        # (2..25) and immediately oriented so "larger is always better"
        # (see orient_external_values / NID's special-casing above).
        external[method] = {
            metric: orient_external_values(
                metric,
                np.array(
                    [fn(y_true, labels[method][k]) for k in K_EVAL], dtype=float
                ),
            )
            for metric, fn in METRIC_FUNCS.items()
        }

    print("Computing the shared nearest-neighbor ordering through r=130...")
    # Query neighbors ONCE out to the largest candidate r (130 + 1 for
    # self), then hard_rnn_distance() slices this shared result down to
    # each smaller r -- avoids repeating an expensive k-NN search per r.
    neighbors = NearestNeighbors(n_neighbors=max(R_VALUES) + 1).fit(X)
    dists_max, idx_max = neighbors.kneighbors(X)

    rnn_by_method = {method: {} for method in METHODS}
    for r in R_VALUES:
        print(f"  r={r}")
        distance = hard_rnn_distance(dists_max, idx_max, r)
        for method in METHODS:
            # Reuse the SAME cluster labels computed above for this method
            # (labels don't depend on r) -- only the distance matrix passed
            # to silhouette_score changes with r.
            rnn_by_method[method][r] = np.array(
                [
                    silhouette_score(
                        distance, labels[method][k], metric="precomputed"
                    )
                    for k in K_TEST
                ]
            )

    # Select r* independently per clustering method, from the rNN-Hard
    # silhouette curves' own shape alone (external metrics not involved).
    selected_r = {}
    rnn_selected = {}
    selection_scores = {}
    for method in METHODS:
        selected_r[method], selection_scores[method] = select_r_topK_spearman(
            rnn_by_method[method],
            R_VALUES,
            m=M,
            n_k=len(K_TEST),
            threshold=THRESHOLD,
        )
        rnn_selected[method] = rnn_by_method[method][selected_r[method]]
        print(f"Selected r* for {method}: {selected_r[method]}")

    # Bundle everything needed to redraw the figures without recomputing,
    # plus a `config` snapshot that main() compares against on future runs
    # to detect whether the cache is stale (see main() below).
    payload = {
        "config": {
            "N": N,
            "D": D,
            "K_STAR": K_STAR,
            "SEED": SEED,
            "K_TEST": K_TEST,
            "K_EVAL": K_EVAL,
            "R_VALUES": R_VALUES,
            "M": M,
            "THRESHOLD": THRESHOLD,
            "standardized": False,
            "score_direction_convention": SCORE_DIRECTION_CONVENTION,
        },
        "ordinary": ordinary,
        "rnn_selected": rnn_selected,
        "selected_r": selected_r,
        "selection_scores": selection_scores,
        "external": external,
    }
    with open(CACHE_PATH, "wb") as handle:
        pickle.dump(payload, handle)
    print(f"Saved cache: {CACHE_PATH}")
    return payload


def get_point_colors(k_values):
    """Blue shades for K < K_STAR (under-partition), orange shades for
    K > K_STAR (over-partition); color intensity increases the closer K
    is to K_STAR, so points near the truth stand out more."""
    k_array = np.asarray(k_values)
    # Use distance from K* only for color intensity, not for computing validity scores.
    # Darker points correspond to partition counts closer to the reference count.
    distance = np.abs(k_array - K_STAR).astype(float)
    max_distance = distance.max() if distance.max() > 0 else 1.0
    intensity = 1.0 - distance / max_distance
    # Blue denotes K<K* and orange K>K*. The plotting helper handles the best-score star.
    return [
        plt.cm.Blues(0.3 + 0.7 * intensity[i])
        if k < K_STAR
        else plt.cm.Oranges(0.3 + 0.7 * intensity[i])
        for i, k in enumerate(k_array)
    ]


def display_values(values):
    """Clip +/-inf metric values (e.g. Sokal-Sneath 3 can be +inf when
    perfectly separating classes) to a finite value just beyond the
    finite range's edge, offset by 8% of the observed spread (or 8% of
    the max magnitude, whichever is larger, as a floor for near-constant
    data). This lets +/-inf points still be PLOTTED (above/below all
    finite points) without distorting the axis scale the way an actual
    +/-inf value would; plot_landscape_panel() below annotates the panel
    when this remapping was used for SS3."""
    # Work on a copy: moving infinite markers for display must not alter saved scores.
    values = np.asarray(values, dtype=float).copy()
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values)
    spread = np.ptp(finite)
    # Choose a visible margin even for nearly constant finite values, then place
    # positive/negative infinities beyond the corresponding end of the finite range.
    step = max(0.08 * spread, 0.08 * max(abs(finite).max(), 1.0))
    values[np.isposinf(values)] = finite.max() + step
    values[np.isneginf(values)] = finite.min() - step
    return values


def add_k_labels(ax, x_values, y_values, k_values):
    """Place K number labels on a scatter panel with overlap avoidance:
    points whose labels would collide (within a fixed 0.035 normalized-axes
    threshold) get their label offset outward along a "repulsion" direction
    (the sum of unit vectors away from all sufficiently-close neighboring
    points), connected back to the marker with a thin arrow; the K*
    marker's label is additionally always forced outward (fixed diagonal
    offset) since it sits under the red best-K star rather than a
    circle/square marker. Points without a collision get their label drawn
    directly on the marker in white bold text. Same collision/repulsion
    logic as gaussian_simulation.ipynb's `_add_k_labels`, but without that
    version's manual per-K override mechanism (this script produces one
    figure per clustering method automatically rather than hand-tuning
    each panel)."""
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    k_values = np.asarray(k_values, dtype=int)
    ax.autoscale_view()
    # Normalize each axis before measuring proximity, so label placement does
    # not depend on whether the index and external scores have different units.
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xspan = max(xlim[1] - xlim[0], 1e-9)
    yspan = max(ylim[1] - ylim[0], 1e-9)
    xn = (x_values - xlim[0]) / xspan
    yn = (y_values - ylim[0]) / yspan

    # Flag close marker pairs for displaced labels; isolated labels remain centered.
    offset_indices = set()
    for i in range(len(x_values)):
        for j in range(i + 1, len(x_values)):
            if np.hypot(xn[i] - xn[j], yn[i] - yn[j]) < 0.035:
                offset_indices.add(i)
                offset_indices.add(j)

    # For crowded points, sum directions away from nearby markers and normalize.
    # If the directions cancel, alternate up/down to separate coincident labels.
    directions = {}
    for i in sorted(offset_indices):
        fx = fy = 0.0
        for j in range(len(x_values)):
            if i == j:
                continue
            dx, dy = xn[i] - xn[j], yn[i] - yn[j]
            distance = np.hypot(dx, dy)
            if 0 < distance < 0.18:
                fx += dx / distance
                fy += dy / distance
        norm = np.hypot(fx, fy)
        if norm <= 1e-6:
            fx, fy = ((0.0, 1.0) if i % 2 == 0 else (0.0, -1.0))
        else:
            fx, fy = fx / norm, fy / norm
        directions[i] = (fx, fy)

    # Always offset the K* annotation as an additional layout rule, independently
    # of which partition attains the maximum external score.
    star_matches = np.where(k_values == K_STAR)[0]
    if star_matches.size:
        star_index = int(star_matches[0])
        offset_indices.add(star_index)
        directions[star_index] = (0.75, 0.75)

    for i, (x, y, k) in enumerate(zip(x_values, y_values, k_values)):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        if i not in offset_indices:
            ax.text(
                x,
                y,
                str(k),
                ha="center",
                va="center",
                fontsize=5.3,
                fontweight="bold",
                color="white",
                zorder=6,
            )
        else:
            fx, fy = directions[i]
            ax.annotate(
                str(k),
                xy=(x, y),
                xytext=(x + fx * 0.075 * xspan, y + fy * 0.075 * yspan),
                ha="center",
                va="center",
                fontsize=5.8,
                fontweight="bold",
                color="black",
                arrowprops=dict(
                    arrowstyle="-",
                    color="gray",
                    linewidth=0.45,
                    shrinkA=0,
                    shrinkB=3,
                ),
                zorder=7,
            )


def plot_landscape_panel(ax, index_values, metric_values, metric, index_title):
    """Draw one scatter panel: an internal score (index_values -- either
    ordinary silhouette or rNN-based silhouette at r*) on the x-axis vs.
    one external validity metric on the y-axis, one point per K in
    K_EVAL. The panel title reports the Spearman rho between the two,
    which is the key diagnostic this whole figure family visualizes."""
    k_values = np.asarray(K_EVAL)
    raw_metric = np.asarray(metric_values, dtype=float)
    # For display only, map infinite external values just beyond the finite range.
    # Retain raw_metric below for the actual rank-correlation calculation.
    shown_metric = display_values(raw_metric)
    index_values = np.asarray(index_values, dtype=float)
    # Keep paired, non-NaN internal/external scores so each visible point
    # refers to the same partition K on both axes.
    valid = ~np.isnan(index_values) & ~np.isnan(raw_metric)
    colors = get_point_colors(k_values)

    # Draw under- and over-partitioned regions separately. Lines follow increasing
    # K within each region; the markers show the individual candidate partitions.
    for mask, cmap, marker, size in (
        (k_values < K_STAR, plt.cm.Blues, "o", 95),
        (k_values > K_STAR, plt.cm.Oranges, "s", 115),
    ):
        current = mask & valid
        if not np.any(current):
            continue
        indices = np.where(current)[0]
        ax.plot(
            index_values[current],
            shown_metric[current],
            color=cmap(0.5),
            linewidth=1.1,
            alpha=0.3,
            zorder=1,
        )
        ax.scatter(
            index_values[current],
            shown_metric[current],
            c=[colors[i] for i in indices],
            s=size,
            marker=marker,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )

    valid_indices = np.where(valid)[0]
    # Find the best external score among valid paired entries, then map its
    # position back to the full K array for the red reference star.
    best_local = int(np.nanargmax(raw_metric[valid]))
    best_index = valid_indices[best_local]
    ax.scatter(
        index_values[best_index],
        shown_metric[best_index],
        marker="*",
        s=190,
        color="red",
        edgecolors="darkred",
        linewidths=0.45,
        zorder=5,
    )

    add_k_labels(
        ax,
        index_values[valid],
        shown_metric[valid],
        k_values[valid],
    )
    # Report rank correlation across K, using the score arrays themselves.
    # Marker positions, color choices, and label offsets do not enter this calculation.
    rho = spearmanr_safe(index_values[valid], raw_metric[valid])
    ax.set_title(
        f"{index_title}\nSpearman $\\rho$: {rho:.3f}",
        fontsize=10,
        fontweight="bold",
        pad=5,
    )
    ax.set_xlabel("Index Values", fontsize=10)
    ax.set_ylabel(METRIC_AXIS_LABELS[metric], fontsize=10)
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.grid(True, linestyle=":", alpha=0.4)

    if metric == "SS3" and np.isposinf(raw_metric).any():
        ax.text(
            0.02,
            0.98,
            "$\\infty$ mapped above finite range",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
        )


def legend_handles():
    """Proxy legend entries (best-K star / under-partition circle /
    over-partition square) shared across every panel in a figure."""
    # Create legend-only artists with the same marker styles as the plotted points.
    # They explain best external score, K<K*, and K>K* without adding observations.
    return [
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor="red",
            markeredgecolor="darkred",
            markersize=12,
            label="Best $K$",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="steelblue",
            markeredgecolor="white",
            markersize=9,
            label="$K <$ True $K^*$ (under-partition)",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="darkorange",
            markeredgecolor="white",
            markersize=9,
            label="$K >$ True $K^*$ (over-partition)",
        ),
    ]


def plot_method_figure(payload, method):
    """Draw one Supplement figure, pairing ordinary and selected rNN Silhouette
    for each external criterion. Figure S1 omits K-means/ACC because main text
    Figure 2 already displays it. Figures S2--S3 include all seven criteria."""
    metrics = list(METRIC_ORDER)
    if method == "KMeans":
        metrics.remove("ACC")

    # 2 metrics per row (4 columns: ordinary+rNN panel-pair, twice).
    n_rows = (len(metrics) + 1) // 2
    fig, axes = plt.subplots(n_rows, 4, figsize=(19, 4 * n_rows))
    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        top=0.87,
        bottom=0.075,
        wspace=0.30,
        hspace=0.72,
    )
    # Both curves are restricted to EVAL_IDX (K=2..25) before plotting,
    # even though rnn_selected was computed at r* over the wider K=2..35
    # selection range.
    ordinary = np.asarray(payload["ordinary"][method])[EVAL_IDX]
    rnn_selected = np.asarray(payload["rnn_selected"][method])[EVAL_IDX]

    # Place the ordinary and selected-rNN panels side by side for each criterion
    # so their rankings can be compared on identical candidate partitions.
    for metric_index, metric in enumerate(metrics):
        group_row = metric_index // 2
        group_col = metric_index % 2
        left_col = group_col * 2
        ordinary_ax = axes[group_row, left_col]
        rnn_ax = axes[group_row, left_col + 1]
        metric_values = payload["external"][method][metric]
        plot_landscape_panel(
            ordinary_ax,
            ordinary,
            metric_values,
            metric,
            "Ordinary Silhouette Score",
        )
        plot_landscape_panel(
            rnn_ax,
            rnn_selected,
            metric_values,
            metric,
            "rNN-based Silhouette Score",
        )

    # If the number of metrics is odd, the last row has two unused axes
    # (e.g. Spectral/Agglomerative have 7 metrics -> 4 rows, last row only
    # uses the first pair) -- turn those off rather than leaving them blank.
    used_axes = 2 * len(metrics)
    for flat_ax in axes.flat[used_axes:]:
        flat_ax.axis("off")

    # Panel-group labels like "(a) Jaccard" are positioned using each
    # pair's ACTUAL rendered bounding box (hence the draw() call before
    # querying get_position()), so they stay centered above their panel
    # pair regardless of exact layout.
    fig.canvas.draw()
    for metric_index, metric in enumerate(metrics):
        group_row = metric_index // 2
        group_col = metric_index % 2
        left_ax = axes[group_row, group_col * 2]
        right_ax = axes[group_row, group_col * 2 + 1]
        left_pos = left_ax.get_position()
        right_pos = right_ax.get_position()
        fig.text(
            (left_pos.x0 + right_pos.x1) / 2,
            max(left_pos.y1, right_pos.y1) + 0.025,
            f"({PANEL_LETTERS[metric]}) {METRIC_LABELS[metric]}",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    fig.suptitle(
        f"Gaussian simulation — {method} clustering",
        fontsize=18,
        fontweight="bold",
        y=0.985,
    )
    fig.legend(
        handles=legend_handles(),
        loc="lower center",
        ncol=3,
        fontsize=11,
        framealpha=0.9,
        bbox_to_anchor=(0.5, 0.012),
    )

    stem = f"gaussian_multimetric_{method.lower()}_landscape"
    pdf_path = os.path.join(OUT_DIR, f"{stem}.pdf")
    png_path = os.path.join(OUT_DIR, f"{stem}.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


def main():
    """Entry point: reuse the cached results pickle if present AND its
    embedded `config` snapshot matches the current constants exactly
    (guards against silently plotting stale results after a constant like
    N, K_STAR, R_VALUES, etc. changes); otherwise (re)run compute_results()
    from scratch. Then render and save one supplementary PDF/PNG figure
    per clustering method.

    Remember: this script relies on relative paths for its cache and
    figure outputs (OUT_DIR/CACHE_PATH above), so it must be invoked with
    section_4_1_gaussian_simulation/ as the working directory.
    """
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as handle:
            payload = pickle.load(handle)
        # Check the saved study configuration before reusing numerical results.
        # If sample size, K/r grids, or other recorded settings differ, recompute.
        expected = {
            "N": N,
            "D": D,
            "K_STAR": K_STAR,
            "SEED": SEED,
            "K_TEST": K_TEST,
            "K_EVAL": K_EVAL,
            "R_VALUES": R_VALUES,
            "M": M,
            "THRESHOLD": THRESHOLD,
            "standardized": False,
            "score_direction_convention": SCORE_DIRECTION_CONVENTION,
        }
        if payload.get("config") != expected:
            payload = compute_results()
        else:
            print(f"Loaded cache: {CACHE_PATH}")
    else:
        payload = compute_results()

    # Render one figure for each clustering algorithm: K-means, spectral,
    # and Ward correspond to Supplement Figures S1, S2, and S3.
    for method in METHODS:
        plot_method_figure(payload, method)


if __name__ == "__main__":
    main()
