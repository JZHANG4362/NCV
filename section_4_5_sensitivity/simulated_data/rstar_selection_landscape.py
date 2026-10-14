r"""
r* selection-landscape diagnostics (rstar_sensitivity_experiment_plan.pdf,
Sections 4.1-4.6, "minimum defensible package" items 1-6) for the Type-2
Gaussian grid, K-means only, ACC only. See
nonconvex_data/rstar_selection_landscape.py for the
full design rationale (window-definition note, what's post-processing vs.
new compute, etc.) -- identical logic, adapted for REPLICATION (N_REP
reps per case: report median curves + IQR per the plan's Section 4.1) and
for the plan's Section 4.6 (dimension-by-r sensitivity, D in {2,10,200})
by defaulting to the FULL 12-case production grid (all D, both
compactness levels, k* in {6,30}) rather than a hand-picked subset --
"you may incorporate multiple settings" is used here to mean exactly the
case grid this project already uses everywhere else, so results are
directly comparable.

Unlike partA_type2_grid_comparison.py, this script does NOT compute the
8 non-rNN indices -- only KMeans clustering + the rNN Hard/Soft sweep +
ACC are needed for r* diagnostics -- but it DOES need a WIDER K range per
case than the production pipeline (to post-hoc perturb K-hat and b
without re-clustering), which partially offsets that saving. N_REP
defaults to 30 (not the production 100) to keep this tractable as a first
pass; bump N_REP_DIAGNOSTIC below and rerun (ideally on the cluster, see
the accompanying .dlx) for full consistency with the rest of the paper.

Usage:
    python rstar_selection_landscape.py                  # all 12 cases, N_REP=30
    python rstar_selection_landscape.py --cases 2 --nrep 3   # smoke test
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.neighbors import NearestNeighbors

# This script is run from its own subfolder (cwd = simulated_data/),
# so these outputs land next to the script: data/ holds one pickle per case
# with the full cached r-landscape, figures/ holds the per-case diagnostic
# plots, tables/ holds the combined summary CSV. tau_khat_window_sensitivity.py
# (Layer 2, one level up) globs data/*.pkl from here as one of its three inputs.
OUT_DATA_DIR = 'data'
OUT_FIG_DIR = 'figures'
OUT_TABLE_DIR = 'tables'
os.makedirs(OUT_DATA_DIR, exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)
os.makedirs(OUT_TABLE_DIR, exist_ok=True)

CLUSTERING_METHOD = 'KMeans'
EXTERNAL_METRIC = 'ACC'
# The full r grid over which the rNN-affinity Silhouette landscape is swept
# and cached: r = 5, 10, 15, ..., 130 (non-self neighbor count). Algorithm 1's
# r* is chosen from this same grid, so caching every r on it is what lets all
# of the tau/K-hat/b sensitivity analysis below run as pure post-processing.
R_VALUES = list(range(5, 135, 5))
M = 35                              # top-m window length used by Algorithm 1's C(r) statistic
TAU0 = 0.95                         # primary/production rank-correlation threshold tau
B0 = 10                             # primary/production buffer b added on top of K-hat
# Grids swept for the sensitivity analysis proper -- tau, K-hat, and b are each
# varied one at a time (holding the other two at their TAU0/khat0/B0 values)
# entirely by re-reading the cached silhouette-vs-K arrays, never by re-clustering.
TAU_GRID = [0.85, 0.90, 0.95, 1.00]
KHAT_GRID = list(range(20, 61, 10))
B_GRID = [5, 10, 20]
LOCAL_DELTAS = [5, 10]              # +/- r window sizes for the local-sensitivity diagnostic L_delta
PLATEAU_EPS = [0.02, 0.05]          # tolerance bands for the rho_E(r) plateau-width diagnostic W_eps
FIXED_R_COMPARISON = [5, 10, 30, 50, 100]   # fixed-r baselines reported alongside r* for context

K_STAR_VALUES = [6, 30]           # matches the Case 2 production grid
D_VALUES = [2, 10, 200]
COMPACTNESS_VALUES = {'compact': 0.1, 'sparse': 0.8}
COMPACTNESS_INDEX = {'compact': 0, 'sparse': 1}

N_REP_DIAGNOSTIC = 30              # 100 in production; see module docstring


def all_cases():
    """Enumerate the full 12-case Case 2 production grid: 2 values of k*
    (true cluster count) x 3 values of D (ambient dimension) x 2 compactness
    levels (compact/sparse cluster_std). Each combination is one "setting"
    that gets its own cached landscape pickle and its own row in the
    Layer 2 (tau_khat_window_sensitivity) aggregation."""
    return [
        (k_star, D, comp)
        for k_star in K_STAR_VALUES
        for D in D_VALUES
        for comp in COMPACTNESS_VALUES
    ]


def case_key(case):
    """Short filesystem-safe identifier for a case, e.g. 'k30_D10_compact'.
    Used to name the per-case .pkl/.pdf outputs and, later, as the row label
    Layer 2 uses when it reloads these pickles."""
    k_star, D, comp = case
    return f'k{k_star}_D{D}_{comp}'


def sample_size_for(k_star):
    """Sample size scales with the true number of clusters (50 points per
    cluster), capped at 1000 to keep the O(N^2) rNN sweep over R_VALUES
    tractable for this diagnostic (which needs many more (r, K) pairs than
    the production pipeline)."""
    return min(1000, k_star * 50)


def simulate_type2_data(n_samples, n_features, k_star, compactness, random_state=1):
    """Draw one replication of Case 2's "Type-2 Gaussian grid" data: k_star
    isotropic Gaussian blobs (sklearn's make_blobs) in n_features dimensions,
    with cluster_std=compactness controlling how separated the blobs are."""
    X, y = make_blobs(n_samples=n_samples, n_features=n_features,
                       centers=k_star, cluster_std=compactness,
                       random_state=random_state)
    return X, y


# ── helpers (copied verbatim from partA_type2_grid_comparison.py) ──────
def rnn_affinity_distance(X, r, use_gaussian=False):
    """Build the rNN-affinity distance matrix used by the rNN-Silhouette
    score, for a given non-self-neighbor count r. For each point, find its r
    nearest OTHER points (sklearn's kneighbors is asked for r+1 neighbors and
    column 0, the point itself at distance 0, is dropped -- see the r
    convention note in the module docstring / verify_r_convention.py). Two
    affinity variants are supported:
      - "hard" (use_gaussian=False): binary affinity, 1.0 if j is among i's r
        nearest neighbors, else 0.
      - "soft" (use_gaussian=True): Gaussian-kernel affinity weighted by
        distance, with per-point bandwidth sigma set to the distance to the
        r-th (farthest-kept) neighbor.
    The raw affinity is symmetrized (A = max(A, A^T)), min-max normalized to
    [0, 1], and converted to a distance D = 1 - A (with a zero diagonal) so it
    can be fed to sklearn's silhouette_score(metric='precomputed'). This
    script calls this function once per r in R_VALUES per variant, on the
    SAME dataset X, to build the full landscape cached below."""
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
    """Spearman rank correlation that tolerates NaNs (dropping any index
    where either input is NaN) and returns NaN instead of raising when fewer
    than 2 valid pairs remain -- used both for the internal C(r) statistic
    and for the external rho_E(r) agreement curve."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(b)
    if mask.sum() < 2:
        return np.nan
    rho, _ = spearmanr(a[mask], b[mask])
    return float(rho)


def select_r_topK_spearman(sil_dict, r_values, m, n_k, threshold=0.95):
    """Core of Algorithm 1's internal (label-free) r* selection rule. For a
    candidate r, take the top-m largest-K silhouette values within the
    current diagnostic window (n_k = number of K values in that window) and
    compute their Spearman correlation against an "ideal" strictly
    decreasing ranking (m_eff, m_eff-1, ..., 1). A high correlation C(r)
    means the rNN-Silhouette score is behaving monotonically (decreasing) as
    K grows across the top of the window -- the internal proxy for "this r
    gives a well-behaved score." r* is the SMALLEST r in r_values whose C(r)
    reaches `threshold` (tau); if no r reaches it, fall back to the r with
    the highest C(r) (the fallback / cap-hit case tracked elsewhere). Note
    this never looks at any external label or accuracy value -- that's the
    whole point of caching C(r) separately from rho_E(r) below."""
    avail = sorted(r for r in r_values if r in sil_dict)
    if not avail:
        return r_values[0], {}
    m_eff = max(2, min(m, n_k - 1))
    top_idx = list(range(n_k - m_eff, n_k))
    ideal = np.arange(m_eff, 0, -1, dtype=float)
    c_vals = {}
    for r in avail:
        obs = np.asarray(sil_dict[r], dtype=float)[top_idx]
        rho = spearmanr_safe(obs, ideal)
        c_vals[r] = rho if not np.isnan(rho) else -2.0
    r_star = next((r for r in avail if c_vals[r] >= threshold), None)
    if r_star is None:
        r_star = max(avail, key=lambda r: c_vals[r])
    return r_star, c_vals


def max_K_eval(k_star):
    """Default K-hat0: the production diagnostic-window upper bound used
    elsewhere in this project, max(25, round(1.75 * k_star)). This is the
    "primary configuration" value that TAU_GRID/KHAT_GRID/B_GRID are swept
    around."""
    return int(max(25, round(1.75 * k_star)))


def clustering_accuracy(y_true, y_pred):
    """Clustering accuracy (ACC): best one-to-one matching between predicted
    cluster labels and true class labels via the Hungarian algorithm
    (linear_sum_assignment on the negated contingency matrix), then the
    fraction of points correctly matched. This is the single external metric
    used throughout this diagnostic (EXTERNAL_METRIC = 'ACC')."""
    cm = contingency_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    return cm[row_ind, col_ind].sum() / cm.sum()


def kmeans_labels(X, K, seed):
    """Thin wrapper around sklearn KMeans, matching the production
    pipeline's clustering method for Case 2 (CLUSTERING_METHOD = 'KMeans')."""
    return KMeans(n_clusters=K, random_state=seed, n_init='auto').fit_predict(X)


# ── diagnostics (identical logic to the Part B version) ─────────────────
def rho_E_curve(sil_by_r, ext_vals, eval_idx):
    """The RETROSPECTIVE external agreement curve rho_E(r): for each r,
    Spearman-correlate the rNN-Silhouette score S_r(K) against the external
    ACC(K) across the K values in eval_idx. This is the "how good would r
    actually have been" curve -- unlike C(r), it DOES use the labels, which
    is why it is reported only for diagnostic/regret purposes and never used
    to pick r* itself."""
    return {r: spearmanr_safe(sil_by_r[r][eval_idx], ext_vals[eval_idx]) for r in R_VALUES}


def select_rstar_for_window(sil_by_r, K_TEST, khat, b, m, tau):
    """Given the WIDE cached silhouette-vs-K arrays (sil_by_r, indexed over
    all of K_TEST), slice out just the K's within the diagnostic window
    implied by (khat, b) -- i.e. K <= khat + b -- and re-run
    select_r_topK_spearman on that slice. This is the function that makes
    the K-hat/b sensitivity sweep "free": since sil_by_r already covers every
    K up to the widest window needed by any grid point, changing khat or b
    only changes which columns of the cached array are read, never requires
    reclustering or rerunning the rNN sweep. eval_idx (K <= khat, i.e.
    excluding the +b buffer) is returned separately for use with
    rho_E_curve."""
    max_select = khat + b
    idx_in_range = [i for i, K in enumerate(K_TEST) if K <= max_select]
    n_k = len(idx_in_range)
    sliced = {r: sil_by_r[r][idx_in_range] for r in R_VALUES}
    r_star, c_vals = select_r_topK_spearman(sliced, R_VALUES, m=m, n_k=n_k, threshold=tau)
    eval_idx = [i for i, K in enumerate(K_TEST) if K <= khat]
    return r_star, c_vals, eval_idx


def local_sensitivity(rhoE, r_star):
    """Two extra diagnostics computed from the already-cached rho_E(r)
    curve: L_delta = how much rho_E drops from its value at r* to its worst
    value within +/-delta of r* (small L_delta = locally stable choice), and
    W_eps = how many r's on the whole grid achieve rho_E within eps of the
    curve's best value (a wide "plateau" means many r choices would have
    performed comparably well)."""
    out = {}
    for delta in LOCAL_DELTAS:
        neighbors = [r for r in R_VALUES if abs(r - r_star) <= delta]
        vals = [rhoE[r] for r in neighbors if not np.isnan(rhoE[r])]
        out[f'L_{delta}'] = (rhoE[r_star] - min(vals)) if vals else np.nan
    finite = [v for v in rhoE.values() if not np.isnan(v)]
    best = max(finite) if finite else np.nan
    for eps in PLATEAU_EPS:
        out[f'W_{eps}'] = sum(1 for v in rhoE.values() if not np.isnan(v) and v >= best - eps)
    return out


def run_one_rep(X, y, K_TEST, khat0, seed):
    """One replication: cluster + rNN sweep + ACC over the wide K_TEST.
    Returns per-variant r*, c_vals, rhoE and all derived diagnostics for
    the PRIMARY window, plus tau/khat/b sensitivity."""
    # Cluster ONCE per K over the wide K_TEST range (shared by every r and
    # every variant below) -- this is the expensive step that must not be
    # repeated, which is exactly why K_TEST is made wide enough up front to
    # cover every K-hat/b combination the sensitivity sweep will need.
    labels_by_K = {K: kmeans_labels(X, K, seed) for K in K_TEST}
    ext_vals = np.array([clustering_accuracy(y, labels_by_K[K]) for K in K_TEST])

    # Full landscape sweep: for every r on the grid, build both the hard and
    # soft rNN-affinity distance matrices ONCE and score them against every K
    # already clustered above. This produces sil_hard_by_r[r] / sil_soft_by_r[r]
    # as full silhouette-vs-K arrays (one entry per K in K_TEST) -- the
    # complete cache that lets every subsequent r* computation (primary
    # config, tau sweep, K-hat sweep, b sweep) run as post-processing instead
    # of recomputing any clustering or silhouette scores.
    sil_hard_by_r, sil_soft_by_r = {}, {}
    for r in R_VALUES:
        D_hard = rnn_affinity_distance(X, r=r, use_gaussian=False)
        D_soft = rnn_affinity_distance(X, r=r, use_gaussian=True)
        sil_hard_by_r[r] = np.array([
            silhouette_score(D_hard, labels_by_K[K], metric='precomputed')
            if len(np.unique(labels_by_K[K])) > 1 else np.nan
            for K in K_TEST
        ])
        sil_soft_by_r[r] = np.array([
            silhouette_score(D_soft, labels_by_K[K], metric='precomputed')
            if len(np.unique(labels_by_K[K])) > 1 else np.nan
            for K in K_TEST
        ])

    khat_values = KHAT_GRID
    out = {}
    for variant, sil_by_r in [('hard', sil_hard_by_r), ('soft', sil_soft_by_r)]:
        # Primary/production configuration (tau0, K-hat0, b0): this is the
        # r* and rho_E(r*) that Algorithm 1 as actually shipped would select
        # and achieve.
        r_star0, c_vals0, eval_idx0 = select_rstar_for_window(sil_by_r, K_TEST, khat0, B0, M, TAU0)
        rhoE0 = rho_E_curve(sil_by_r, ext_vals, eval_idx0)

        finite_rhoE = {r: v for r, v in rhoE0.items() if not np.isnan(v)}
        r_oracle = max(finite_rhoE, key=finite_rhoE.get) if finite_rhoE else R_VALUES[0]
        regret = rhoE0[r_oracle] - rhoE0[r_star0]
        loc_sens = local_sensitivity(rhoE0, r_star0)

        max_c = max(c_vals0.values())
        # tau sweep: re-threshold the SAME primary-window c_vals0 at each
        # candidate tau -- no re-slicing or reclustering needed since tau
        # only changes which r first crosses the threshold.
        tau_sens = {}
        for tau in TAU_GRID:
            r_s = next((r for r in sorted(c_vals0) if c_vals0[r] >= tau), None)
            if r_s is None:
                r_s = max(c_vals0, key=lambda r: c_vals0[r])
            tau_sens[tau] = dict(r_star=r_s, rhoE=rhoE0.get(r_s, np.nan))

        # K-hat sweep: re-slice sil_by_r/K_TEST at each candidate K-hat
        # (buffer b held at B0) via select_rstar_for_window -- purely reading
        # different columns of the wide cached arrays.
        khat_sens = {}
        for khat in khat_values:
            r_s, c_v, eidx = select_rstar_for_window(sil_by_r, K_TEST, khat, B0, M, TAU0)
            rE = rho_E_curve(sil_by_r, ext_vals, eidx)
            khat_sens[khat] = dict(r_star=r_s, rhoE=rE.get(r_s, np.nan))

        # b sweep: same idea, varying the buffer b (K-hat held at khat0).
        b_sens = {}
        for b in B_GRID:
            r_s, c_v, eidx = select_rstar_for_window(sil_by_r, K_TEST, khat0, b, M, TAU0)
            rE = rho_E_curve(sil_by_r, ext_vals, eidx)
            b_sens[b] = dict(r_star=r_s, rhoE=rE.get(r_s, np.nan))

        out[variant] = dict(
            r_star=r_star0, c_vals=c_vals0, rhoE=rhoE0,
            r_oracle=r_oracle, regret=regret, local_sensitivity=loc_sens,
            cap_hit=(r_star0 == R_VALUES[-1]), floor_hit=(r_star0 == R_VALUES[0]),
            fallback=(max_c < TAU0), threshold_margin=c_vals0[r_star0] - TAU0,
            tau_sensitivity=tau_sens, khat_sensitivity=khat_sens, b_sensitivity=b_sens,
        )
    return out


def run_one_case(case, n_rep):
    """Run N_REP_DIAGNOSTIC (default 30) independent replications of one
    (k_star, D, compactness) case, each with a freshly simulated dataset and
    its own seed. K_TEST is built wide enough (2..max over all khat+b grid
    combinations) that every downstream tau/K-hat/b sensitivity query for
    every replication can be answered without further clustering."""
    k_star, D, comp_label = case
    ckey = case_key(case)
    compactness = COMPACTNESS_VALUES[comp_label]
    n_samples = sample_size_for(k_star)
    khat0 = max_K_eval(k_star)
    khat_values = KHAT_GRID
    max_select_needed = max(khat + b for khat in khat_values for b in B_GRID)
    K_TEST = list(range(2, max_select_needed + 1))
    print(f'\n=== {ckey} (N={n_samples}, K-hat0={khat0}, wide K_TEST=2..{max_select_needed}, n_rep={n_rep}) ===')

    reps = {'hard': [], 'soft': []}
    for rep in range(n_rep):
        seed = k_star * 100_000 + D * 100 + COMPACTNESS_INDEX[comp_label] * 10 + rep
        X, y = simulate_type2_data(n_samples, D, k_star, compactness, random_state=seed)
        X = StandardScaler().fit_transform(X)
        rep_result = run_one_rep(X, y, K_TEST, khat0, seed)
        for variant in ['hard', 'soft']:
            reps[variant].append(rep_result[variant])
        if (rep + 1) % 10 == 0:
            print(f'  rep {rep+1}/{n_rep}')

    return dict(case=case, ckey=ckey, N=n_samples, khat0=khat0, K_TEST=K_TEST, n_rep=n_rep, reps=reps)


def summarize_case(result):
    """Median + IQR across reps, per variant, matching plan Section 4.1's
    'report the median curve and an interquartile band' for replicated
    simulations."""
    rows = []
    for variant in ['hard', 'soft']:
        reps = result['reps'][variant]
        r_stars = [rp['r_star'] for rp in reps]
        rhoE_at_rstar = [rp['rhoE'][rp['r_star']] for rp in reps]
        regrets = [rp['regret'] for rp in reps]
        L5 = [rp['local_sensitivity']['L_5'] for rp in reps]
        fallback_rate = np.mean([rp['fallback'] for rp in reps])
        cap_rate = np.mean([rp['cap_hit'] for rp in reps])
        rows.append({
            'case': result['ckey'], 'variant': variant, 'N': result['N'], 'n_rep': result['n_rep'],
            'median_r_star': float(np.median(r_stars)),
            'IQR_r_star': float(np.percentile(r_stars, 75) - np.percentile(r_stars, 25)),
            'median_r_star/N': float(np.median(r_stars)) / result['N'],
            'median_rhoE(r*)': float(np.median(rhoE_at_rstar)),
            'median_regret': float(np.median(regrets)),
            'median_L_5': float(np.nanmedian(L5)),
            'fallback_rate': fallback_rate,
            'cap_hit_rate': cap_rate,
        })
    return rows


def plot_landscape_median(result, variant, out_path):
    """C(r) and rho_E(r): median curve + IQR band across reps."""
    reps = result['reps'][variant]
    c_matrix = np.array([[rp['c_vals'][r] for r in R_VALUES] for rp in reps])
    rE_matrix = np.array([[rp['rhoE'].get(r, np.nan) for r in R_VALUES] for rp in reps])
    r_stars = [rp['r_star'] for rp in reps]
    median_r_star = int(np.median(r_stars))

    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    ax = axes[0]
    med = np.median(c_matrix, axis=0)
    q1, q3 = np.percentile(c_matrix, [25, 75], axis=0)
    ax.plot(R_VALUES, med, marker='o', color='steelblue', label='median $C(r)$')
    ax.fill_between(R_VALUES, q1, q3, color='steelblue', alpha=0.2, label='IQR')
    ax.axhline(TAU0, color='gray', linestyle='--', label=f'$\\tau_0$={TAU0}')
    ax.axvline(median_r_star, color='red', linestyle=':', label=f'median $r^*$={median_r_star}')
    ax.set_ylabel('$C(r)$ (internal)')
    variant_suffix = '' if variant == 'hard' else f' ({variant})'
    ax.set_title(f"{result['ckey']}{variant_suffix} -- K-means, ACC, {result['n_rep']} reps")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    med = np.nanmedian(rE_matrix, axis=0)
    q1, q3 = np.nanpercentile(rE_matrix, [25, 75], axis=0)
    ax.plot(R_VALUES, med, marker='s', color='darkorange', label='median $\\rho_E(r)$')
    ax.fill_between(R_VALUES, q1, q3, color='darkorange', alpha=0.2, label='IQR')
    ax.axvline(median_r_star, color='red', linestyle=':', label=f'median $r^*$={median_r_star}')
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.set_ylabel('$\\rho_E(r)$ (retrospective, ACC)')
    ax.set_xlabel('$r$')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', type=int, default=None,
                         help='Number of cases to run (default: all 12). Use a small number for smoke tests.')
    parser.add_argument('--case-keys', nargs='+', default=None,
                        help='Run only the named case keys, e.g. k30_D2_compact k30_D2_sparse.')
    parser.add_argument('--nrep', type=int, default=N_REP_DIAGNOSTIC,
                         help=f'Replications per case (default: {N_REP_DIAGNOSTIC}).')
    parser.add_argument('--summarize-saved', action='store_true',
                        help='Rebuild the combined summary table from saved case files without rerunning.')
    args = parser.parse_args()

    cases = all_cases()
    # --summarize-saved: skip all computation entirely and just reload the
    # already-cached per-case pickles from a previous full run to rebuild
    # tables/summary_table.csv -- handy after manually editing/removing a
    # case file, or after changing summarize_case()'s output columns.
    if args.summarize_saved:
        all_rows = []
        for case in cases:
            path = os.path.join(OUT_DATA_DIR, f'rstar_landscape_{case_key(case)}.pkl')
            with open(path, 'rb') as f:
                all_rows.extend(summarize_case(pickle.load(f)))
        summary_df = pd.DataFrame(all_rows)
        summary_df.to_csv(os.path.join(OUT_TABLE_DIR, 'summary_table.csv'), index=False)
        print(summary_df.to_string(index=False))
        print('\nRebuilt summary_table.csv from 12 saved case files.')
        return
    if args.case_keys is not None:
        case_lookup = {case_key(case): case for case in cases}
        unknown = [key for key in args.case_keys if key not in case_lookup]
        if unknown:
            parser.error(f'Unknown case key(s): {", ".join(unknown)}')
        cases = [case_lookup[key] for key in args.case_keys]
    if args.cases is not None:
        cases = cases[:args.cases]

    # Main loop: for each case, run all replications, cache the FULL raw
    # result (every rep's c_vals/rhoE/sensitivity dicts, not just the
    # summary) to data/rstar_landscape_<ckey>.pkl -- this is the file Layer 2
    # (../summary_plots/) later globs and reloads -- then emit
    # the median +/- IQR landscape figures and accumulate summary rows.
    all_rows = []
    for case in cases:
        result = run_one_case(case, args.nrep)
        with open(os.path.join(OUT_DATA_DIR, f'rstar_landscape_{result["ckey"]}.pkl'), 'wb') as f:
            pickle.dump(result, f)
        for variant in ['hard', 'soft']:
            out_path = os.path.join(OUT_FIG_DIR, f'landscape_{result["ckey"]}_{variant}.pdf')
            plot_landscape_median(result, variant, out_path)
        all_rows.extend(summarize_case(result))
        for row in summarize_case(result):
            print(f"  [{row['variant']}] median r*={row['median_r_star']:.0f}  "
                  f"median rhoE(r*)={row['median_rhoE(r*)']:.3f}  "
                  f"fallback_rate={row['fallback_rate']:.2f}  cap_hit_rate={row['cap_hit_rate']:.2f}")

    summary_df = pd.DataFrame(all_rows)
    if args.case_keys is None and args.cases is None:
        summary_df.to_csv(os.path.join(OUT_TABLE_DIR, 'summary_table.csv'), index=False)
    print('\n' + summary_df.to_string(index=False))
    if args.case_keys is None and args.cases is None:
        print(f'\nWrote summary_table.csv, {len(cases)} pkl files, {2*len(cases)} landscape figures.')
    else:
        print(f'\nWrote {len(cases)} pkl files and {2*len(cases)} landscape figures; '
              'combined summary_table.csv unchanged.')


if __name__ == '__main__':
    main()
