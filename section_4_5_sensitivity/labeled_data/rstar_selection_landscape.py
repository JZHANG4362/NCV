"""Main text Section 4.5; Supplement Section S11, Figures S4--S6.

Study six Case 4 datasets under K-means: Digits, Ecoli, Iris, Libras, Vowel, Wine.
Standardize features and evaluate agreement with ACC only.

Cache cluster assignments and hard rNN score curves over a K range
covering every tested K-hat and buffer b. Then vary one selection setting
at a time: tau=0.85/0.90/0.95/1.00, K-hat=20/30/40/50/60, or b=5/10/20.
The high-K diagnostic window has length m=35, clipped to the available
range; it can overlap the evaluation range. Changing K-hat changes both
the selection window and the K values used for the reported correlation.
Threshold and buffer sweeps keep the evaluation range fixed.

summary_plots/tau_khat_window_sensitivity.py builds the three published
figures from hard-rNN results, averaging the simulated replications.
Local sensitivity and oracle regret are additional diagnostics.
Run here: python rstar_selection_landscape.py

This working-folder version queries r+1 neighbors and drops the first
match. Its fresh results can differ from the supplied sensitivity figures,
which used a self-inclusive r-neighbor query. The refactored ZIP retains
that historical convention for reproducing Figures S4--S6."""

import gzip
import os
import pickle
import sys
import urllib.request

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment

from sklearn.datasets import load_digits, load_iris, load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.neighbors import NearestNeighbors

# Outputs are written relative to this script's own subfolder (cwd =
# Benchmark_data_real/ under Sensitivity/): data/ holds one pickle per
# dataset with the full cached r-landscape, figures/ holds the per-dataset
# diagnostic plots, tables/ holds the combined summary CSV.
# ../summary_plots/tau_khat_window_sensitivity.py (Layer 2)
# globs data/*.pkl from here as one of its three inputs.
OUT_DATA_DIR = 'data'
OUT_FIG_DIR = 'figures'
OUT_TABLE_DIR = 'tables'
# Share the fixed raw datasets with section_4_4_labeled_benchmarks.
# Paths are relative to this script's folder; use the repository runner.
RAW_DATA_DIR = '../../section_4_4_labeled_benchmarks/data/raw_datasets'
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
# Selection notation: code M corresponds to window size s in Algorithm 1;
# TAU0 is the baseline tau varied in the sensitivity sweep.
M = 35                              # top-m window length used by Algorithm 1's C(r) statistic
TAU0 = 0.95                         # primary/production rank-correlation threshold tau
B0 = 10                             # primary/production buffer b added on top of K-hat
# Grids swept for the sensitivity analysis proper -- tau, K-hat, and b are each
# varied one at a time (holding the other two at their TAU0/khat0/B0 values)
# entirely by re-reading the cached silhouette-vs-K arrays, never by
# re-clustering.
TAU_GRID = [0.85, 0.90, 0.95, 1.00]
KHAT_GRID = list(range(20, 61, 10))
B_GRID = [5, 10, 20]
LOCAL_DELTAS = [5, 10]              # +/- r window sizes for the local-sensitivity diagnostic L_delta
PLATEAU_EPS = [0.02, 0.05]          # tolerance bands for the rho_E(r) plateau-width diagnostic W_eps
FIXED_R_COMPARISON = [5, 10, 30, 50, 100]   # fixed-r baselines reported alongside r* for context

SMALL_MEDIUM_DATASETS = ['Digits', 'Ecoli', 'Iris', 'Wine', 'Libras', 'Vowel']
LARGE_DATASETS = ['MNIST-test', 'Fashion-MNIST-test', 'Letter']

# Expected (K = true class count, N = sample size, D = ambient dimension) for
# every Case 4 dataset this project knows how to load -- includes the 3
# large datasets (excluded by default, see module docstring) so they remain
# available via an explicit opt-in on the command line.
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


# ── dataset loaders ──────
# All loaders below are byte-for-byte copies of the loaders in
# ../../section_4_4_labeled_benchmarks/Benchmark_data_real.py, kept here so this diagnostic script can
# run standalone. _fetch() downloads-and-caches raw files under RAW_DATA_DIR
# (shared with Case4's own cache, so if Case4 has already been run these
# downloads are skipped entirely).
def _fetch(url, fname):
    """Return the locally cached raw dataset file, downloading it only when absent."""
    # Reuse the local raw-data file when present; otherwise download it to
    # the dataset cache so subsequent experiments read the same input file.
    path = os.path.join(RAW_DATA_DIR, fname)
    if not os.path.exists(path):
        print(f'  downloading {url} ...')
        urllib.request.urlretrieve(url, path)
    return path


def _load_idx_images(gz_path):
    """Decode an IDX image file and flatten each image into one feature vector."""
    with gzip.open(gz_path) as f:
        _, n, rows, cols = np.frombuffer(f.read(16), dtype='>i4')
        # Flatten each image into one row of pixel features; the matching label file
        # supplies the reference classes independently of the pixels.
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)
    return data.astype(float)


def _load_idx_labels(gz_path):
    """Decode the IDX class labels in the same observation order as the image file."""
    # Skip the 8-byte IDX header, then read one unsigned-byte class per image.
    # Preserve file order to keep labels aligned with the flattened image rows.
    with gzip.open(gz_path) as f:
        _, n = np.frombuffer(f.read(8), dtype='>i4')
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


# Small/medium loaders (sklearn built-ins, no download needed) --------------
def load_digits_dataset():
    """Return the Digits features and reference classes listed in Supplement Table S2."""
    # Return numeric features and reference labels separately; standardization
    # and clustering take place in the experiment caller, not in this loader.
    d = load_digits()
    return d.data.astype(float), d.target


def load_iris_dataset():
    """Return the Iris features and reference classes listed in Supplement Table S2."""
    # Return numeric features and reference labels separately; standardization
    # and clustering take place in the experiment caller, not in this loader.
    d = load_iris()
    return d.data.astype(float), d.target


def load_wine_dataset():
    """Return the Wine features and reference classes listed in Supplement Table S2."""
    # Return numeric features and reference labels separately; standardization
    # and clustering take place in the experiment caller, not in this loader.
    d = load_wine()
    return d.data.astype(float), d.target


# Large-dataset loaders (downloaded on first use; slow, excluded by default) -
def load_mnist_test():
    """Load the MNIST test split and flatten its pixels, preserving reference labels."""
    path = _fetch('https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz', 'mnist.npz')
    with np.load(path) as f:
        # Use the test split listed in Supplement Table S2; training images are not pooled in.
        X, y = f['x_test'], f['y_test']
    return X.reshape(X.shape[0], -1).astype(float), y


def load_fashion_mnist_test():
    """Load the Fashion-MNIST test images and matching labels from cached raw files."""
    # Read the matching test-image and test-label files; decode images to
    # flat pixel rows and labels to their original observation order.
    img_path = _fetch(
        'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-images-idx3-ubyte.gz',
        'fmnist_test_images.gz')
    lbl_path = _fetch(
        'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-labels-idx1-ubyte.gz',
        'fmnist_test_labels.gz')
    return _load_idx_images(img_path), _load_idx_labels(lbl_path)


# UCI-hosted loaders (downloaded on first use; small/medium size) -----------
def load_ecoli():
    """Separate Ecoli numeric features from sequence identifiers and reference classes."""
    path = _fetch('https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data', 'ecoli.data')
    rows = [l.split() for l in open(path) if l.strip()]
    # Select only the measured feature columns. The label column is decoded
    # separately below for external evaluation and is never included in X.
    X = np.array([[float(v) for v in r[1:-1]] for r in rows])
    y_str = np.array([r[-1] for r in rows])
    # Encode class names as integers without changing which observations share a class.
    # These labels are used for external criteria and the benchmark K* count.
    _, y = np.unique(y_str, return_inverse=True)
    return X, y


def load_letter():
    """Separate the Letter descriptor features from the first-column class labels."""
    path = _fetch(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data',
        'letter-recognition.data')
    rows = [l.strip().split(',') for l in open(path) if l.strip()]
    # Select only the measured feature columns. The label column is decoded
    # separately below for external evaluation and is never included in X.
    X = np.array([[float(v) for v in r[1:]] for r in rows])
    y_str = np.array([r[0] for r in rows])
    # Encode class names as integers without changing which observations share a class.
    # These labels are used for external criteria and the benchmark K* count.
    _, y = np.unique(y_str, return_inverse=True)
    return X, y


def load_libras():
    """Separate Libras movement features from the final-column reference labels."""
    path = _fetch(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/libras/movement_libras.data',
        'movement_libras.data')
    rows = [l.strip().split(',') for l in open(path) if l.strip()]
    # Select only the measured feature columns. The label column is decoded
    # separately below for external evaluation and is never included in X.
    X = np.array([[float(v) for v in r[:-1]] for r in rows])
    y_str = np.array([r[-1] for r in rows])
    # Encode class names as integers without changing which observations share a class.
    # These labels are used for external criteria and the benchmark K* count.
    _, y = np.unique(y_str, return_inverse=True)
    return X, y


def load_vowel():
    """Extract the ten Vowel acoustic features and the reference class, excluding identifiers."""
    path = _fetch(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/vowel/vowel-context.data',
        'vowel-context.data')
    rows = [l.strip().split() for l in open(path) if l.strip()]
    # Keep the ten acoustic features, excluding the split flag, speaker ID, sex, and class.
    X = np.array([[float(v) for v in r[3:13]] for r in rows])
    y_str = np.array([r[13] for r in rows])
    # Encode class names as integers without changing which observations share a class.
    # These labels are used for external criteria and the benchmark K* count.
    _, y = np.unique(y_str, return_inverse=True)
    return X, y


# Registry mapping dataset name -> loader function, used by run_one_dataset()
# and main() below to dispatch on the --dataset-name-list command-line args.
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


# ── helpers ──────────────
def rnn_affinity_distance(X, r):
    """Build the binary-neighbor distance used by rNN-Silhouette.
    
    Query r+1 neighbors and discard the first returned match. For distinct
    observations this leaves r other points. Take the symmetric union of
    directed neighbor links and return 1-A with a zero diagonal. Reuse this
    distance matrix across candidate cluster counts for the same dataset."""
    n = X.shape[0]
    # Exclude the query point itself: r denotes exactly r other observations.
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
    """Spearman rank correlation that tolerates NaNs (dropping any index
    where either input is NaN) and returns NaN instead of raising when fewer
    than 2 valid pairs remain -- used both for the internal C(r) statistic
    and for the external rho_E(r) agreement curve."""
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
    m_eff = max(2, min(m, n_k - 1))
    top_idx = list(range(n_k - m_eff, n_k))
    # Algorithm 1 constructs v=(m_eff,...,1). Positive rank agreement with v
    # means the internal score decreases as K increases in the upper K window.
    ideal = np.arange(m_eff, 0, -1, dtype=float)
    c_vals = {}
    for r in avail:
        # C(candidate) = Spearman(score(K) over K_top, v). This is an internal
        # shape diagnostic; it does not compare the score with ACC or reference labels.
        obs = np.asarray(sil_dict[r], dtype=float)[top_idx]
        rho = spearmanr_safe(obs, ideal)
        # An undefined correlation receives -2, below the valid range [-1,1].
        # If every candidate is undefined, the fallback below returns the smallest one.
        c_vals[r] = rho if not np.isnan(rho) else -2.0
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


def clustering_accuracy(y_true, y_pred):
    """Clustering accuracy (ACC): best one-to-one matching between predicted
    cluster labels and true class labels via the Hungarian algorithm
    (linear_sum_assignment on the negated contingency matrix), then the
    fraction of points correctly matched."""
    cm = contingency_matrix(y_true, y_pred)
    # ACC = max over one-to-one label matchings of (matched observations)/N.
    # The assignment solver minimizes cost, so negation or max-count subtraction
    # converts maximum overlap to its cost convention. Unmatched groups earn no credit.
    row_ind, col_ind = linear_sum_assignment(-cm)
    return cm[row_ind, col_ind].sum() / cm.sum()


def kmeans_labels(X, K, seed):
    """Thin wrapper around sklearn KMeans, matching the production
    pipeline's clustering method for Case 4 (CLUSTERING_METHOD = 'KMeans')."""
    # Fit a separate K-means partition for this K by minimizing within-cluster
    # squared Euclidean distances. Reuse these labels for all validity scores.
    return KMeans(n_clusters=K, random_state=seed, n_init='auto').fit_predict(X)


# ── diagnostics (identical logic to the other two subfolders) ──────────
def rho_E_curve(sil_by_r, ext_vals, eval_idx):
    """The RETROSPECTIVE external agreement curve rho_E(r): for each r,
    Spearman-correlate the rNN-Silhouette score S_r(K) against the external
    ACC(K) across the K values in eval_idx. Unlike C(r), this DOES use the
    labels, so it is reported only for diagnostic/regret purposes and never
    used to pick r* itself."""
    # For each r, correlate its internal score curve with the reference curve
    # on eval_idx only; the wider K range remains reserved for selection diagnostics.
    return {r: spearmanr_safe(sil_by_r[r][eval_idx], ext_vals[eval_idx]) for r in R_VALUES}


def select_rstar_for_window(sil_by_r, K_TEST, khat, b, m, tau):
    """Given the WIDE cached silhouette-vs-K arrays (sil_by_r, indexed over
    all of K_TEST), slice out just the K's within the diagnostic window
    implied by (khat, b) -- i.e. K <= khat + b -- and re-run
    select_r_topK_spearman on that slice. This is what makes the K-hat/b
    sensitivity sweep "free": since sil_by_r already covers every K up to the
    widest window needed by any grid point, changing khat or b only changes
    which columns of the cached array are read, never requires reclustering
    or rerunning the rNN sweep. eval_idx (K <= khat, excluding the +b buffer)
    is returned separately for use with rho_E_curve."""
    # Changing K_hat or b changes K_max=K_hat+b, so first restrict the
    # precomputed curves before applying the upper-window diagnostic.
    max_select = khat + b
    idx_in_range = [i for i, K in enumerate(K_TEST) if K <= max_select]
    n_k = len(idx_in_range)
    sliced = {r: sil_by_r[r][idx_in_range] for r in R_VALUES}
    r_star, c_vals = select_r_topK_spearman(sliced, R_VALUES, m=m, n_k=n_k, threshold=tau)
    # Reported rho_E uses K<=K_hat. Changing b alone leaves this set unchanged;
    # changing K_hat changes both the selection and evaluation ranges.
    eval_idx = [i for i, K in enumerate(K_TEST) if K <= khat]
    return r_star, c_vals, eval_idx


def local_sensitivity(rhoE, r_star):
    """L_delta = drop in rho_E from its value at r* to its worst value
    within +/-delta of r* (small L_delta = locally stable choice); W_eps =
    how many r's on the whole grid achieve rho_E within eps of the curve's
    best value (a wide "plateau" means many r choices perform comparably)."""
    out = {}
    for delta in LOCAL_DELTAS:
        # Restrict the existing r grid to |r-r*|<=delta, then find the worst available
        # rho_E there; subtracting it from rho_E(r*) gives the local drop L_delta.
        neighbors = [r for r in R_VALUES if abs(r - r_star) <= delta]
        vals = [rhoE[r] for r in neighbors if not np.isnan(rhoE[r])]
        out[f'L_{delta}'] = (rhoE[r_star] - min(vals)) if vals else np.nan
    finite = [v for v in rhoE.values() if not np.isnan(v)]
    best = max(finite) if finite else np.nan
    # W_eps counts grid candidates within eps of the global best rho_E.
    # It is a count of candidates, not an interval width in units of r.
    for eps in PLATEAU_EPS:
        out[f'W_{eps}'] = sum(1 for v in rhoE.values() if not np.isnan(v) and v >= best - eps)
    return out


def run_one_dataset(dname):
    """Run the full landscape computation for one real benchmark dataset:
    load + standardize the data, cluster once per K over a WIDE K_TEST range
    (wide enough to cover every K-hat/b combination in the sensitivity
    grids), then sweep every r in R_VALUES to build the full hard
    silhouette-vs-K cache. Everything after that -- the primary r*, oracle
    regret, local sensitivity, and the tau/K-hat/b sensitivity sweeps -- is
    pure post-processing over these cached arrays."""
    print(f'\n=== {dname} ===')
    X_raw, y = DATASET_LOADERS[dname]()
    # Preprocess each feature as (x - sample mean)/sample standard deviation
    # (StandardScaler uses ddof=0 and leaves constant features at zero).
    # Use the same transformed X for clustering and every original-data score.
    X = StandardScaler().fit_transform(X_raw)
    k_star = EXPECTED_TABLE_DATA[dname]['K']
    N = X.shape[0]

    khat0 = max_K_eval(k_star)
    khat_values = KHAT_GRID
    max_select_needed = max(khat + b for khat in khat_values for b in B_GRID)
    K_TEST = list(range(2, max_select_needed + 1))
    print(f'  k*={k_star}, N={N}, K-hat0={khat0}, wide K_TEST=2..{max_select_needed} ({len(K_TEST)} values)')

    # Cluster ONCE per K over the wide K_TEST range (shared by every r and
    # variant below) -- the expensive step that must not be repeated.
    seed = 1
    # Fit each candidate partition once. All indices and all r values below
    # rank these same partitions within this one clustering algorithm.
    labels_by_K = {K: kmeans_labels(X, K, seed) for K in K_TEST}
    ext_vals = np.array([clustering_accuracy(y, labels_by_K[K]) for K in K_TEST])

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
            silhouette_score(D_hard, labels_by_K[K], metric='precomputed')
            if len(np.unique(labels_by_K[K])) > 1 else np.nan
            for K in K_TEST
        ])

    result = {'dataset': dname, 'k_star': k_star, 'N': N, 'khat0': khat0, 'K_TEST': K_TEST,
               'ext_vals': ext_vals, 'variants': {}}

    for variant, sil_by_r in [('hard', sil_hard_by_r)]:
        # Primary/production configuration (tau0, K-hat0, b0): the r* and
        # rho_E(r*) that Algorithm 1 as actually shipped would select and
        # achieve on this dataset.
        r_star0, c_vals0, eval_idx0 = select_rstar_for_window(sil_by_r, K_TEST, khat0, B0, M, TAU0)
        # C(r) measures the internal decreasing trend; rho_E(r) measures agreement
        # with ACC. Keep them separate: only C(r) is used to select r*.
        rhoE0 = rho_E_curve(sil_by_r, ext_vals, eval_idx0)

        finite_rhoE = {r: v for r, v in rhoE0.items() if not np.isnan(v)}
        # Optional diagnostic: the oracle maximizes ACC agreement across r.
        # Its advantage over selected r* is regret; it does not feed Algorithm 1.
        r_oracle = max(finite_rhoE, key=finite_rhoE.get) if finite_rhoE else R_VALUES[0]
        regret = rhoE0[r_oracle] - rhoE0[r_star0]
        fixed_r_rhoE = {r: rhoE0.get(r, np.nan) for r in FIXED_R_COMPARISON if r in rhoE0}
        loc_sens = local_sensitivity(rhoE0, r_star0)

        cap_hit = (r_star0 == R_VALUES[-1])
        floor_hit = (r_star0 == R_VALUES[0])
        max_c = max(c_vals0.values())
        fallback = max_c < TAU0
        threshold_margin = c_vals0[r_star0] - TAU0
        n_crossings = sum(1 for r in R_VALUES if c_vals0[r] >= TAU0)

        # tau sweep: re-threshold the SAME primary-window c_vals0 at each
        # candidate tau -- no re-slicing or reclustering needed.
        tau_sens = {}
        # Figure S4: vary only tau, reusing the same C(r) values and evaluation set.
        for tau in TAU_GRID:
            r_s = next((r for r in sorted(c_vals0) if c_vals0[r] >= tau), None)
            if r_s is None:
                r_s = max(c_vals0, key=lambda r: c_vals0[r])
            tau_sens[tau] = dict(r_star=r_s, rhoE=rhoE0.get(r_s, np.nan),
                                   fallback=max_c < tau, cap_hit=(r_s == R_VALUES[-1]))

        # K-hat sweep: re-slice sil_by_r/K_TEST at each candidate K-hat
        # (buffer b held at B0) -- purely reading different columns of the
        # wide cached arrays.
        khat_sens = {}
        # Figure S5: vary K_hat; recompute the window selection and rho_E on its
        # corresponding evaluation range, reusing the already fitted partitions.
        for khat in khat_values:
            r_s, c_v, eidx = select_rstar_for_window(sil_by_r, K_TEST, khat, B0, M, TAU0)
            rE = rho_E_curve(sil_by_r, ext_vals, eidx)
            khat_sens[khat] = dict(r_star=r_s, rhoE=rE.get(r_s, np.nan))

        # b sweep: same idea, varying the buffer b (K-hat held at khat0).
        b_sens = {}
        # Figure S6: vary buffer b only, holding K_hat, tau, and requested s fixed.
        for b in B_GRID:
            r_s, c_v, eidx = select_rstar_for_window(sil_by_r, K_TEST, khat0, b, M, TAU0)
            rE = rho_E_curve(sil_by_r, ext_vals, eidx)
            b_sens[b] = dict(r_star=r_s, rhoE=rE.get(r_s, np.nan))

        result['variants'][variant] = dict(
            r_star=r_star0, c_vals=c_vals0, rhoE=rhoE0, eval_idx=eval_idx0,
            r_oracle=r_oracle, regret=regret, fixed_r_rhoE=fixed_r_rhoE,
            local_sensitivity=loc_sens, cap_hit=cap_hit, floor_hit=floor_hit,
            fallback=fallback, threshold_margin=threshold_margin,
            n_threshold_crossings=n_crossings, tau_sensitivity=tau_sens,
            khat_sensitivity=khat_sens, b_sensitivity=b_sens,
        )
        print(f'  [{variant}] r*={r_star0}  rho_E(r*)={rhoE0[r_star0]:.3f}  '
              f'oracle r={r_oracle} regret={regret:.3f}  fallback={fallback} cap_hit={cap_hit}')

    return result


def plot_landscape(result, variant, out_path):
    """Draw the internal C(r) and external rho_E(r) diagnostics for one dataset.
    
    The upper panel shows the threshold used by Algorithm 1. The lower
    panel assesses agreement with ACC, carrying the same selected r*.
    These per-dataset diagnostics supplement the aggregate Figures S4--S6."""
    v = result['variants'][variant]
    r_vals = R_VALUES
    # Extract C(r) and rho_E(r) in the same candidate order. Only C(r)
    # determines r*; rho_E(r) evaluates the resulting external agreement.
    c = [v['c_vals'][r] for r in r_vals]
    rE = [v['rhoE'].get(r, np.nan) for r in r_vals]

    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    ax = axes[0]
    ax.plot(r_vals, c, marker='o', color='steelblue')
    ax.axhline(TAU0, color='gray', linestyle='--', label=f'$\\tau_0$={TAU0}')
    ax.axvline(v['r_star'], color='red', linestyle=':', label=f"$r^*$={v['r_star']}")
    ax.set_ylabel('$C(r)$ (internal)')
    variant_suffix = '' if variant == 'hard' else f' ({variant})'
    ax.set_title(f"{result['dataset']}{variant_suffix} -- K-means, ACC\n"
                 f"$K^*$={result['k_star']}, $N$={result['N']}, "
                 f"$\\widehat K$={result['khat0']}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Carry the selected r* into the ACC panel to compare the internal
    # selection rule with the retrospective performance landscape.
    ax = axes[1]
    ax.plot(r_vals, rE, marker='s', color='darkorange')
    ax.axvline(v['r_star'], color='red', linestyle=':', label=f"$r^*$={v['r_star']}")
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.set_ylabel('$\\rho_E(r)$ (retrospective, ACC)')
    ax.set_xlabel('$r$')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    # Default to the 6 small/medium datasets (see module docstring SCOPE
    # note); any command-line arguments override this list entirely
    # (including opt-in to the 3 large datasets).
    """Run the requested sensitivity datasets and save both full curves and summaries."""
    dataset_names = sys.argv[1:] if len(sys.argv) > 1 else SMALL_MEDIUM_DATASETS
    unknown = [d for d in dataset_names if d not in EXPECTED_TABLE_DATA]
    assert not unknown, f'Unknown dataset(s): {unknown}'
    large_requested = [d for d in dataset_names if d in LARGE_DATASETS]
    if large_requested:
        print(f'NOTE: {large_requested} are large datasets -- this will be slow '
              f'(see module docstring / SCALE WARNING in Benchmark_data_real.py).')

    # For each requested dataset: run the full landscape computation, cache
    # the FULL raw result (c_vals/rhoE/sensitivity dicts for every variant,
    # not just the summary) to data/rstar_landscape_<dname>.pkl -- this is
    # the file Layer 2 (../summary_plots/) later globs and
    # reloads -- and emit the per-dataset landscape figures.
    all_results = {}
    # Run each fixed dataset independently, retaining the full C(r), rho_E(r),
    # and selected-r results for later plotting and sensitivity summaries.
    for dname in dataset_names:
        all_results[dname] = run_one_dataset(dname)
        with open(os.path.join(OUT_DATA_DIR, f'rstar_landscape_{dname}.pkl'), 'wb') as f:
            pickle.dump(all_results[dname], f)
        for variant in ['hard']:
            out_path = os.path.join(OUT_FIG_DIR, f'landscape_{dname}_{variant}.pdf')
            plot_landscape(all_results[dname], variant, out_path)

    rows = []
    for dname, result in all_results.items():
        for variant in ['hard']:
            v = result['variants'][variant]
            rows.append({
                'dataset': dname, 'variant': variant, 'N': result['N'],
                'k_star': result['k_star'], 'r_star': v['r_star'],
                'rho_E(r*)': round(v['rhoE'][v['r_star']], 4),
                'r_star/N': round(v['r_star'] / result['N'], 5),
                'fallback': v['fallback'], 'cap_hit': v['cap_hit'], 'floor_hit': v['floor_hit'],
                'oracle_r': v['r_oracle'], 'regret': round(v['regret'], 4),
                'L_5': round(v['local_sensitivity']['L_5'], 4),
                'L_10': round(v['local_sensitivity']['L_10'], 4),
                'threshold_margin': round(v['threshold_margin'], 4),
            })
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(os.path.join(OUT_TABLE_DIR, 'summary_table.csv'), index=False)
    print('\n' + summary_df.to_string(index=False))
    print(f"\nWrote summary_table.csv, {len(all_results)} pkl files, "
          f"{len(all_results)} landscape figures.")


if __name__ == '__main__':
    main()
