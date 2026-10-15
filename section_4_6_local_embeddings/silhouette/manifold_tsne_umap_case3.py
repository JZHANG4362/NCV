"""Main text Section 4.6, Tables 4--5; Supplement Section S12, Figures S19--S28.

Case 3: the ten fixed nonconvex datasets. This pipeline evaluates
ordinary Silhouette. All comparisons use ACC.

1. Standardize X and fit t-SNE/UMAP candidates in two dimensions, seed 1.
   Test perplexities 10,20,...,100 subject to 3*p < N-1, and UMAP neighbor
   counts 5,10,15,20,30,50,75,100 subject to k < N-1.
2. Fit each clustering algorithm on original standardized X over the K
   grid. Keep these assignments fixed while scoring X and each embedding.
   Each embedding is reused across algorithms within this pipeline.
3. Apply Algorithm 1 separately to each index's candidate score curves
   on the wider selection grid. Evaluate the selected curve against ACC
   only on K=2,...,max(25, round(1.75 K*)).
4. Save selected correlations for Table 4 and the bar plots, plus every
   candidate's ACC correlation for the retrospective oracle in Table 5.
   The hard rNN-Sil reference uses original X and its own selected r.

These experiments use one dataset per setting, with no replication.
The exporter combines this pipeline with its sibling to assemble all nine
indices. Bar annotations compare each embedding with that index on X.
Run here: python manifold_tsne_umap_case3.py KMeans
Use Spectral or Agglomerative to select another clustering algorithm."""

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

# cell: helpers -- rNN baseline
def rnn_affinity_distance(X, r):
    """Build the binary-neighbor distance used by rNN-Silhouette.
    
    Query r+1 neighbors and discard the first returned match. For distinct
    observations this leaves r other points. Take the symmetric union of
    directed neighbor links and return 1-A with a zero diagonal. Reuse this
    distance matrix across candidate cluster counts for the same dataset."""
    n = X.shape[0]
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


# ── cell: helpers -- genuine t-SNE / UMAP embeddings (NO affinity substitution) ──
def tsne_embedding(X, perplexity, random_state=1):
    """Fit one two-dimensional t-SNE candidate with PCA initialization and the given seed."""
    # Fit one 2D candidate from X; perplexity controls local neighborhoods.
    # Keep original-data cluster labels fixed when scoring this embedding later.
    return TSNE(n_components=2, perplexity=perplexity, random_state=random_state,
                init='pca').fit_transform(X)


def umap_embedding(X, n_neighbors, random_state=1):
    """Fit one two-dimensional UMAP candidate with the given neighbor count and seed."""
    # Fit one 2D candidate using n_neighbors to build the UMAP graph.
    # This parameter affects the representation, not the number of clustering groups K.
    return umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                      random_state=random_state).fit_transform(X)


# cell: helpers -- Algorithm 1, generalized
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


def select_param_topK_spearman(sil_dict, param_values, m, n_k, threshold=0.95):
    """Apply Algorithm 1 to score curves ordered by increasing K.
    
    Correlate the last m_eff scores with a strictly decreasing sequence,
    where m_eff = max(2, min(m, n_k - 1)). Choose the smallest candidate
    whose correlation reaches the threshold; if none does, choose a maximizer
    (the smallest candidate breaks ties). Reference labels and external
    correlations are not used here. Return the choice and diagnostic scores.
    The high-K window can overlap the evaluation range.
    For Section 4.6, candidates are perplexities or UMAP neighbor counts,
    with a separate choice for each internal index and clustering algorithm."""
    # Algorithm 1 scans candidates in increasing order. Here m is the paper's
    # requested window size s, and threshold is tau; n_k counts candidate K values.
    avail = sorted(p for p in param_values if p in sil_dict)
    if not avail:
        return param_values[0], {}
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
    for p in avail:
        # C(candidate) = Spearman(score(K) over K_top, v). This is an internal
        # shape diagnostic; it does not compare the score with ACC or reference labels.
        obs       = np.asarray(sil_dict[p], dtype=float)[top_idx]
        rho       = spearmanr_safe(obs, ideal)
        # An undefined correlation receives -2, below the valid range [-1,1].
        # If every candidate is undefined, the fallback below returns the smallest one.
        c_vals[p] = rho if not np.isnan(rho) else -2.0
    # Threshold branch of Algorithm 1: choose the FIRST (smallest) candidate
    # with C >= tau, rather than maximizing an external agreement score.
    p_star = next((p for p in avail if c_vals[p] >= threshold), None)
    if p_star is None:
        # Fallback branch: maximize C when the threshold is unmet. Sorted candidate
        # order makes the smallest candidate win an exact tie.
        p_star = max(avail, key=lambda p: c_vals[p])
    return p_star, c_vals


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


# ── cell: external metric (ACC only -- see module docstring) ───────────
def clustering_accuracy(y_true, y_pred):
    '''Unsupervised clustering accuracy via Hungarian-matched contingency
    table.'''
    cm = contingency_matrix(y_true, y_pred)
    # ACC = max over one-to-one label matchings of (matched observations)/N.
    # The assignment solver minimizes cost, so negation or max-count subtraction
    # converts maximum overlap to its cost convention. Unmatched groups earn no credit.
    row_ind, col_ind = linear_sum_assignment(-cm)
    return cm[row_ind, col_ind].sum() / cm.sum()


SCORE_DIRECTION_CONVENTION = 'larger_is_better_v1'

# cell: clustering-methods
def kmeans_labels(X, K, seed):
    # Fit a separate K-means partition for this K by minimizing within-cluster
    # squared Euclidean distances. Reuse these labels for all validity scores.
    """Fit the requested K-means partition on X using the experiment seed."""
    return KMeans(n_clusters=K, random_state=seed, n_init='auto').fit_predict(X)


def spectral_labels(X, K, seed):
    # Construct the fixed 10-neighbor affinity graph, then partition its spectral
    # representation into K groups. Its 10 neighbors are independent of validation r.
    """Fit K spectral clusters using the fixed nearest-neighbor affinity graph."""
    return SpectralClustering(
        n_clusters=K, random_state=seed,
        affinity='nearest_neighbors', n_neighbors=10,
        assign_labels='kmeans',
    ).fit_predict(X)


def agglomerative_labels(X, K, seed):
    # Ward linkage merges groups to minimize the increase in within-cluster
    # sum of squares, stopping at the requested K groups.
    """Fit K agglomerative clusters using Ward linkage on Euclidean features."""
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
    """Load numeric coordinates and separately encode the final ARFF attribute as reference labels."""
    data, meta = arff.loadarff(path)
    attr_names = meta.names()
    class_col = attr_names[-1]
    # Supplement Table S1: the last ARFF attribute is the reference class.
    # Only the preceding coordinate columns become clustering features X.
    feature_cols = attr_names[:-1]
    X = np.column_stack([data[c].astype(float) for c in feature_cols])
    y_raw = data[class_col]
    y_str = np.array([v.decode('utf-8') if isinstance(v, bytes) else str(v) for v in y_raw])
    # Encode class names as integers without changing which observations share a class.
    # These labels are used for external criteria and the benchmark K* count.
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
# Selection notation: code M corresponds to window size s in Algorithm 1;
# THRESHOLD corresponds to tau. Neither is the number of clusters K.
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
    """Resolve the experiment cache filename used by the result readers."""
    # Keep the established cache naming scheme so the table/plot readers
    # can find results for the same experiment without changing the saved data format.
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
    """Load a previous result dictionary if present, otherwise begin with an empty store."""
    # Load existing entries when available so unrelated stored results survive.
    # The caller decides which requested entries to recompute; loading alone does not skip work.
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
    """Serialize the current result dictionary with its experiment metadata."""
    # Store correlations together with the metadata below, including the
    # score-direction convention, so readers can interpret the cached values.
    payload = {
        'results': results_dict,
        'score_direction_convention': SCORE_DIRECTION_CONVENTION,
        'clustering_methods_computed': sorted(results_dict.keys()),
    }
    # Write the currently accumulated result store as a checkpoint for later
    # table generation and plotting; this step does not aggregate the scores.
    with open(results_path_for(base_name), 'wb') as f:
        pickle.dump(payload, f)


def save_all():
    """Checkpoint the selected correlations, candidate-parameter sweeps, and selected parameters."""
    # Save each result store under its own filename: selected correlations,
    # full parameter sweeps, and selected parameter values serve different downstream tasks.
    save_results(main_results, 'manifold_fixed_results')
    save_results(sensitivity_results, 'manifold_param_sensitivity_results')
    save_results(param_star_results, 'manifold_param_star_results')


# ── cell: main-loop ─────────────────────────────────────────────────────
# NOTE: D_hard(r) and the t-SNE/UMAP embedding sweeps depend only on X --
# built ONCE per dataset here, shared across all 3 clustering methods.
for dname in DATASETS_TO_RUN:
    X_raw, y = DATASETS[dname]
    # Preprocess each feature as (x - sample mean)/sample standard deviation
    # (StandardScaler uses ddof=0 and leaves constant features at zero).
    # Use the same transformed X for clustering and every original-data score.
    X = StandardScaler().fit_transform(X_raw)
    n_samples, Ddim = X.shape
    k_star = len(np.unique(y))
    max_K_s = max_K_select(k_star)
    max_K_e = max_K_eval(k_star)
    # K indexes alternative partitions of the same data. Sweep through K_hat+b
    # for selection, then restrict to the evaluation subset when reporting rho.
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
    # t-SNE perplexity controls its effective local neighborhood. The 3*p
    # sample-size filter is the candidate-grid convention of these experiments.
    perpl_grid = [p for p in PERPLEXITY_VALUES if 3 * p < n_samples - 1]
    # Here lowercase k is UMAP's neighbor count, not the partition count K
    # and not r in the rNN-Silhouette reference.
    nn_grid    = [k for k in N_NEIGHBORS_VALUES if k < n_samples - 1]

    # ---- rNN sweep (unchanged) ----
    print(f'  building rNN sweep ({len(r_grid)} r values)...')
    # For partition C_K, a_i is mean d_r within its cluster (excluding self),
    # b_i is the smallest mean d_r to another cluster, and
    # S_r(K) = mean_i[(b_i-a_i)/max(a_i,b_i)], with 0 for a_i=b_i=0 or a singleton.
    # Vary r to change d_r while retaining C_K; each array scores one r across K.
    D_hard_by_r = {r: rnn_affinity_distance(X, r) for r in r_grid}

    # ---- genuine t-SNE / UMAP embedding sweeps (one full embedding per
    #      candidate perplexity / n_neighbors -- see module SCALE NOTE) ----
    print(f'  building t-SNE embedding sweep ({len(perpl_grid)} perplexity values)...')
    # Fit each embedding once per dataset and candidate, then reuse it for
    # all requested clustering algorithms within this pipeline.
    Y_tsne_by_p = {p: tsne_embedding(X, p, random_state=1) for p in perpl_grid}
    print(f'  building UMAP embedding sweep ({len(nn_grid)} n_neighbors values)...')
    Y_umap_by_k = {k: umap_embedding(X, k, random_state=1) for k in nn_grid}

    for cm_name in CLUSTERING_METHODS_TO_RUN:
        cm_fn = CLUSTERING_METHOD_FUNCS[cm_name]
        seed = 1  # no replication -- fixed seed is enough for reproducibility

        # Fit partitions on original standardized X. Keep these labels for all
        # representations, so changes in agreement measure the validity scores.
        # Fit each candidate partition once. All indices and all r values below
        # rank these same partitions within this one clustering algorithm.
        labels_by_K = {K: cm_fn(X, K, seed) for K in K_TEST}
        multi_cluster = {K: len(np.unique(labels_by_K[K])) > 1 for K in K_TEST}

        # ACC(K) is the external reference for Tables 4--5 and Figures S7--S37.
        # These values assess the selected rankings; they do not select parameters.
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
        # Evaluate ordinary Euclidean Silhouette in each 2D t-SNE representation,
        # keeping the original-data partitions fixed. Apply Algorithm 1 to choose p.
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
        # Repeat the same fixed-partition comparison for UMAP. Each method gets
        # its own selected embedding parameter from the internal score curves.
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

        # Retain candidate-wise ACC correlations for retrospective comparisons.
        # These values do not enter Algorithm 1's parameter selection.
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
    # Read the selected-score summary, complete parameter sweep, and chosen parameter
    # from separate stores; plotting does not recompute their statistical results.
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
    # Preserve the configured method order while omitting unavailable correlations.
    # Each bar is agreement across K for this dataset and clustering algorithm.
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
        """Show ACC agreement across one embedding-parameter grid.
        
        Read the selected parameter from the saved Algorithm 1 result, then
        mark the retrospective ACC-maximizing candidate separately. This
        diagnostic plot does not retune the parameter or rerun clustering."""
        if not curve:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=13, fontweight='bold')
            return
        # Sort the parameter grid before drawing its saved external-correlation curve.
        # The curve is a retrospective evaluation, not the decreasing-trend diagnostic.
        xs = sorted(curve.keys())
        ax.plot(xs, [curve[x] for x in xs], marker='o', color=color, linewidth=2)
        # The dotted marker locates the parameter already selected from internal scores.
        star = star_data.get(star_key)
        if star is not None:
            ax.axvline(star, color=color, linestyle=':', linewidth=2.5,
                       label=f'${{{star_key}}}={star}$')
        # The dashed oracle marker maximizes the displayed external agreement.
        # Its use of reference labels distinguishes it from the selected parameter.
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

    # Keep diagnostic exports separated by experiment/algorithm so successive
    # plots do not overwrite results from a different comparison.
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
