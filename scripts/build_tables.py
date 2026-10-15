#!/usr/bin/env python3
"""Build main text Tables 1--5 and Supplement Tables S4--S87 from saved results.

Case 2 tables summarize replication-level Spearman correlations by mean
or median. Cases 3 and 4 each have one result per dataset and algorithm.
Table 4 counts strict improvements at algorithm-selected embedding
parameters; Table 5 uses the largest ACC correlation on each candidate
grid. Overall percentages pool dataset counts across Cases 2--4, separately
for each index, clustering algorithm, and embedding method.
Run python reproduce.py tables from the repository root."""
from __future__ import annotations
import argparse
from pathlib import Path
import pickle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT
METRICS = ["ACC", "Jaccard", "SS3", "ARI", "NMI", "Powers", "NID"]

CMS = ["KMeans", "Spectral", "Agglomerative"]

METHODS = [
    "rNN_Hard", "Silhouette", "VRC", "WG", "WB", "DBCV", "CVDD",
    "VIASCKDE", "DCSI",
]

CASE2_CASES = [
    (k, d, compactness)
    for k in [6, 30]
    for d in [2, 10, 200]
    for compactness in ["compact", "sparse"]
]

CASE3_K = {
    "pathbased": 3, "spiral": 3, "aggregation": 7, "atom": 2,
    "lsun": 3, "zelnik1": 3, "zelnik5": 4, "zelnik6": 3,
    "triangle1": 4, "longsquare": 6,
}

CASE4_K = {
    "Digits": 10, "MNIST-test": 10, "Ecoli": 8, "Letter": 26,
    "Iris": 3, "Fashion-MNIST-test": 10, "Wine": 3, "Libras": 15,
    "Vowel": 11,
}

def metric_path(case: int, metric: str) -> Path:
    """Locate one external criterion's cache from the batched experiment.
    ACC has no metric suffix; other criteria use their names. The historical
    nonconvex_ prefix in Case 2 filenames still refers to the simulated grid."""
    # Map the case to its experiment folder, preserving the filename suffix
    # that identifies the external criterion in the original result cache.
    suffix = "" if metric == "ACC" else f"_{metric}"
    if case == 2:
        name = "nonconvex_case_comparison_results_nonself_oriented.pkl" if metric == "ACC" else (
            f"nonconvex_case_comparison_results_nonself_oriented_{metric}.pkl"
        )
        return DATA_ROOT / "section_4_2_simulated_benchmarks/data" / name
    if case == 3:
        return DATA_ROOT / "section_4_3_nonconvex_benchmarks/data" / (
            f"nonconvex_fixed_results_paperkmax_buffer10_r130_nonself_oriented{suffix}.pkl"
        )
    return DATA_ROOT / "section_4_4_labeled_benchmarks/data" / (
        f"realdata_fixed_results_paperkmax_buffer10_r130_nonself_oriented{suffix}.pkl"
    )

INDICES = ['Silhouette','VRC','WG','WB','DBCV','CVDD','VIASCKDE','DCSI','DB']
PACKAGES = {
    'alt': ('section_4_6_local_embeddings/other_indices','tsne_umap_alt_indices_tuned_origkgrid_v1'),
    'sil': ('section_4_6_local_embeddings/silhouette','tsne_umap_tuned_embedding_origkgrid_v1'),
}

def load(path):
    """Read a result cache and verify that larger scores mean better agreement."""
    with path.open('rb') as f:
        payload = pickle.load(f)
    # Reject caches with a different score orientation; otherwise signs could
    # reverse the meaning of an apparent ranking improvement.
    if payload.get('score_direction_convention') != 'larger_is_better_v1':
        raise ValueError(f'Unexpected score convention: {path}')
    return payload['results']

def benchmark_rows():
    """Aggregate finite replication correlations for Case 2; retain single values
    for Cases 3--4. Record incomplete runs rather than imputing replications.
    Main text Tables 1--3 reuse the corresponding ACC rows from the Supplement."""
    rows = {}; raw = []; short_runs=[]
    for case in [2,3,4]:
        for metric in METRICS:
            results = load(metric_path(case,metric))
            settings = ([f'k{k}_D{d}_{c}' for k,d,c in CASE2_CASES] if case==2 else list(CASE3_K if case==3 else CASE4_K))
            for cm in CMS:
                for stat in (['mean','median'] if case==2 else ['value']):
                    label = (f'tab:case2_{metric}_{cm}_{stat}' if case==2 else f'tab:{"nonconvex" if case==3 else "realdata"}_{metric}_{cm}')
                    for i,dataset in enumerate(settings):
                        source = results[cm][dataset]; vals=[]
                        for method in METHODS:
                            v = source[method]
                            if case==2:
                                if len(v)!=100 and stat=='mean':
                                    short_runs.append(dict(metric=metric,clustering=cm,dataset=dataset,method=method,expected=100,actual=len(v)))
                                finite=np.asarray(v,dtype=float);finite=finite[np.isfinite(finite)]
                                # Each value is already a rank correlation across K from one replication.
                                # Summarize these correlations, rather than correlating averaged score curves.
                                v=float((np.mean if stat=='mean' else np.median)(finite)) if len(finite) else np.nan
                            vals.append(float(v))
                            raw.append(dict(case=case,metric=metric,clustering=cm,summary=stat,dataset=dataset,method=method,value=v))
                        rows[f'{label}/{i}']=[f'{x:.3f}' if np.isfinite(x) else 'NA' for x in vals]
    for target,source,n in [('tab:main_case2_acc','tab:case2_ACC_KMeans_mean',12),('tab:main_nonconvex_acc_spectral','tab:nonconvex_ACC_Spectral',10),('tab:main_labeled_acc_kmeans','tab:realdata_ACC_KMeans',9)]:
        for i in range(n):rows[f'{target}/{i}']=rows[f'{source}/{i}']
    return rows,raw,short_runs

def embedding_rows():
    """Count strict embedding improvements separately for each index and algorithm.
    Table 4 uses the selected parameter; Table 5 takes the maximum finite ACC
    correlation from the candidate grid. A dataset enters the denominator only
    when both original and embedded correlations are finite. Fashion-MNIST
    DBCV is the expected missing pair: overall N=30 for DBCV and 31 otherwise.
    Overall pools counts across cases, rather than averaging case percentages."""
    counts=[]; rows={}; allstats={}
    for scenario in ['tuned','oracle']:
        for case in [2,3,4]:
            fixed={};sens={}
            for kind,(package,tag) in PACKAGES.items():
                folder=DATA_ROOT/package/f'data/case{case}'
                fixed[kind]=load(folder/f'manifold_fixed_results_{tag}.pkl')
                sens[kind]=load(folder/f'manifold_param_sensitivity_results_{tag}.pkl')
            datasets=([f'k{k}_D{d}_{c}' for k,d,c in CASE2_CASES] if case==2 else list(CASE3_K if case==3 else CASE4_K))
            for cm in CMS:
                for index in INDICES:
                    kind='sil' if index=='Silhouette' else 'alt'
                    for rep,curve in [('TSNE','perplexity'),('UMAP','n_neighbors')]:
                        improved=valid=0
                        for dataset in datasets:
                            f=fixed[kind][cm][dataset]
                            orig=f['Silhouette' if kind=='sil' else f'{index}_Original']
                            if scenario=='tuned':
                                value=f[f'{rep}_Ordinary' if kind=='sil' else f'{index}_{rep}']
                            else:
                                values=sens[kind][cm][dataset][curve]
                                if kind=='alt':values=values[index]
                                finite=[v for v in values.values() if v is not None and np.isfinite(v)]
                                # Table 5 oracle uses ACC to choose the best candidate retrospectively.
                                # Table 4 above uses the parameter chosen by the internal diagnostic.
                                value=max(finite) if finite else np.nan
                            available=orig is not None and value is not None and np.isfinite(orig) and np.isfinite(value)
                            if available:
                                # Count an improvement only when the embedded correlation is strictly higher.
                                valid+=1;improved+=int(value>orig)
                            elif not (case==4 and index=='DBCV' and dataset=='Fashion-MNIST-test'):
                                raise ValueError(f'Unexpected missing embedding pair: {scenario}/{case}/{cm}/{dataset}/{index}/{rep}')
                        allstats[scenario,case,cm,index,rep]=(improved,valid)
                        counts.append(dict(scenario=scenario,case=case,clustering=cm,index=index,representation=rep,improved=improved,valid=valid,percent=100*improved/valid))
        for ci,cm in enumerate(CMS):
            for ii,index in enumerate(INDICES):
                cells=[]
                for rep in ['TSNE','UMAP']:
                    stat=[allstats[scenario,c,cm,index,rep] for c in [2,3,4]]
                    # Matches the current manuscript's integer percentages;
                    # Overall is pooled counts, never an average of percentages.
                    # Overall percentage = 100 * sum_case(improved) / sum_case(valid).
                    # Pool dataset counts, so the 12/10/9-dataset cases have their proper weights.
                    stat.append((sum(s[0] for s in stat),sum(s[1] for s in stat)))
                    cells.extend(f'{round(n/d,2)*100:.0f}\\%' for n,d in stat)
                rows[f'embedding_{scenario}/{ci*len(INDICES)+ii}']=cells
    return rows,counts

def format_cells(key,values):
    """Apply the displayed-number notation used in the paper tables.
    Benchmark bold marks only rNN-Sil values exceeding Sil on the same row;
    a star marks each largest displayed value. Embedding bold marks Overall
    percentages above 50. Comparisons here use the already formatted values."""
    # Embedding rows have two four-column groups. Their fourth columns are
    # the pooled Overall percentages, where bold marks values above 50%.
    if key.startswith('embedding_'):
        return [rf'\textbf{{{v}}}' if i in [3,7] and int(v[:-2])>50 else v for i,v in enumerate(values)]
    finite=[float(x) for x in values if x!='NA']
    maximum=max(finite) if finite else None
    out=[]
    for i,v in enumerate(values):
        c=v
        # Benchmark columns begin with rNN-Sil then Sil: bold only the first
        # when its displayed correlation exceeds the second on this same row.
        if i==0 and v!='NA' and values[1]!='NA' and float(v)>float(values[1]):c=rf'\textbf{{{v}}}'
        # Mark the largest displayed finite value with a star, independently of bold.
        # The comparisons use rounded display values, matching the visible table.
        if v!='NA' and float(v)==maximum:c+=r'$^{\star}$'
        out.append(c)
    return out

MAIN_CAPTIONS = {
    2: 'Mean Spearman rank correlation for Case 2 with ACC and K-means clustering across 100 replications. Sil and $r$NN-Sil denote the ordinary and $r$NN-based Silhouette scores. Bold marks $r$NN-Sil $>$ Sil; $^{\\star}$ marks the largest available displayed value per row.',
    3: 'Spearman rank correlation with ACC for Case 3 with non-globular cluster structures under spectral clustering. Sil and $r$NN-Sil denote the ordinary and $r$NN-based Silhouette scores. Bold marks $r$NN-Sil $>$ Sil; $^{\\star}$ marks the largest available displayed value per row.',
    4: 'Spearman rank correlation with ACC for Case 4 under K-means clustering. Sil and $r$NN-Sil denote the ordinary and $r$NN-based Silhouette scores. Bold marks $r$NN-Sil $>$ Sil; $^{\\star}$ marks the largest available displayed value per row. \nFor Fashion-MNIST-test, DBCV is unavailable because floating-point underflow\nin the 784-dimensional all-points-core-distance calculation produces infinite\nmutual-reachability distances and an assertion failure during internal\nminimum-spanning-tree reconstruction.',
}

EMBEDDING_CAPTIONS = {
    'tuned': 'Percentage of datasets where an internal validity index, applied with t-SNE- or UMAP-transformed data, results in an improvement over its result on the untransformed data. Numbers in parentheses give the dataset counts. Overall percentages are the total numbers of datasets showing improvement across Cases~2--4 divided by $12+10+9=31$ and multiplied by 100. DBCV excludes Fashion-MNIST-test because of errors in the original computation, so its Case~4 and overall denominators are 8 and 30, respectively. Perplexity in t-SNE and the number of neighborhoods in UMAP are obtained via Algorithm~\\ref{alg:r_selection},  with $r$ replaced by perplexity in t-SNE or the number of neighbors in UMAP. Bold numbers in the Overall columns indicate improvement for more than half of the datasets (percentage $>50\\%$).',
    'oracle': 'Percentages obtained using the best perplexity in t-SNE and number of neighbors in UMAP for each dataset, chosen to maximize the Spearman rank correlation with ACC over the respective candidate grids rather than selected by Algorithm~\\ref{alg:r_selection}. The notation is the same as in Table~\\ref{tab:improvement_tuned_by_index}.',
}

CLUSTER_LABELS = {
    'KMeans': 'K-means clustering',
    'Spectral': 'spectral clustering',
    'Agglomerative': 'agglomerative clustering',
}
BENCHMARK_NOTE = (
    r'Sil and $r$NN-Sil denote the ordinary and $r$NN-based Silhouette scores. '
    r'Bold marks $r$NN-Sil $>$ Sil; $^{\star}$ marks the largest available displayed value per row.'
)


def benchmark_table(rows, case, metric, clustering, stat='value', main=False):
    """Render one table using the paper's captions, column order and notation."""
    # Select the caption and row key for either main text Tables 1--3 or the
    # matching Supplement table; both use the same stored numeric rows.
    if main:
        label = {2: 'tab:main_case2_acc', 3: 'tab:main_nonconvex_acc_spectral',
                 4: 'tab:main_labeled_acc_kmeans'}[case]
        caption = MAIN_CAPTIONS[case]
    elif case == 2:
        label = f'tab:case2_{metric}_{clustering}_{stat}'
        caption = (
            f'{stat.capitalize()} Spearman rank correlation ($\\rho$) with {metric} '
            r'across $N_{\mathrm{rep}}=100$ replications, '
            f'{CLUSTER_LABELS[clustering]}, for the 12 settings in Case 2. {BENCHMARK_NOTE}'
        )
    else:
        label = f'tab:{"nonconvex" if case == 3 else "realdata"}_{metric}_{clustering}'
        datasets = 'ten non-globular' if case == 3 else 'nine labeled'
        caption = (f'Spearman rank correlation ($\\rho$) with {metric} for the '
                   f'{datasets} benchmark datasets, {CLUSTER_LABELS[clustering]}. {BENCHMARK_NOTE}')
    if case == 2:
        metadata = [[str(k), str(d), compactness] for k, d, compactness in CASE2_CASES]
        headers = [r'$K^*$', r'$D$', 'Compactness']
        columns = 'lll|ll|*{7}{l}'
    else:
        dataset_counts = CASE3_K if case == 3 else CASE4_K
        metadata = [[('LongSquare' if name == 'longsquare' else name.capitalize()) if case == 3 else name,
                     str(k)] for name, k in dataset_counts.items()]
        headers = ['Dataset', r'$K^*$']
        columns = 'll|ll|*{7}{l}'
    # Put rNN-Sil and Sil first and separate this pair with vertical rules.
    # The column specifications above left-align all benchmark entries.
    headers += [r'$r$NN-Sil', 'Sil', *METHODS[2:]]
    lines = [r'\begin{table}[H]', r'\centering'] if main else [r'\begin{center}']
    if case == 2 and not main:
        lines += [r'\begin{minipage}{\linewidth}', r'\centering']
    lines += [r'\scriptsize',
              r'\setlength{\tabcolsep}{' + ('2.4pt' if main else '2.8pt') + '}',
              r'\renewcommand{\arraystretch}{' + ('0.94' if main else '0.92') + '}',
              (r'\caption{' if main else r'\captionof{table}{') + caption + '}',
              r'\label{' + label + '}']
    width = r'\textwidth' if main else {2: r'0.80\linewidth', 3: r'0.86\linewidth', 4: r'0.87\linewidth'}[case]
    lines += [r'\resizebox{' + width + r'}{!}{%', r'\begin{tabular}{' + columns + '}',
              r'\hline', ' & '.join(headers) + r' \\', r'\hline']
    # Attach dataset metadata to the already formatted correlations, then apply
    # bold/star notation without recomputing or rerounding the underlying results.
    for i, metadata_cells in enumerate(metadata):
        key = f'{label}/{i}'
        lines.append(' & '.join(metadata_cells + format_cells(key, rows[key])) + r' \\')
    lines += [r'\hline', r'\end{tabular}', '}']
    if case == 2 and not main:
        lines.append(r'\end{minipage}')
    lines.append(r'\end{table}' if main else r'\end{center}')
    return '\n'.join(lines)


def supplementary_tables(rows, case):
    """Assemble the Supplement benchmark tables by external criterion and algorithm."""
    # Loop over the seven external criteria and three algorithms in a fixed
    # order; Case 2 adds both mean and median tables, while Cases 3--4 use single values.
    pieces = ['% Auto-generated by scripts/build_tables.py.', '% r is the number of non-self neighbors; rNN-Sil uses binary neighbor weights.']
    for metric in METRICS:
        for clustering in CMS:
            pieces += [r'\clearpage', r'\begin{landscape}']
            if clustering == CMS[0]:
                pieces.append(r'\subsection{' + metric + '}')
            for stat in (['mean', 'median'] if case == 2 else ['value']):
                pieces.append(benchmark_table(rows, case, metric, clustering, stat))
            pieces.append(r'\end{landscape}')
    return '\n\n'.join(pieces) + '\n'


def embedding_table(rows, scenario):
    """Render main text Table 4 or 5 from already computed improvement percentages."""
    lines = [r'\begin{table}[!ht]', r'\centering', r'\scriptsize',
             r'\setlength{\tabcolsep}{2.4pt}', r'\renewcommand{\arraystretch}{0.62}',
             r'\caption{' + EMBEDDING_CAPTIONS[scenario] + '}',
             r'\label{tab:improvement_' + scenario + '_by_index}',
             r'\resizebox{\textwidth}{!}{%', r'\begin{tabular}{lccc|c|ccc|c|}',
             r'\hline', r'\noalign{\vskip 3pt}',
             r' & \multicolumn{4}{c}{t-SNE} & \multicolumn{4}{c}{UMAP} \\',
             r'\cmidrule(lr){2-5} \cmidrule(lr){6-9}',
             r'Index & Case 2 (12) & Case 3 (10) & Case 4 (9) & Overall (31) & Case 2 (12) & Case 3 (10) & Case 4 (9) & Overall (31) \\']
    # Stack the three algorithms as row sections. Each section uses the same
    # t-SNE/UMAP case columns and pooled Overall columns from embedding_rows().
    for ci, heading in enumerate(['K-means clustering', 'Spectral Clustering', 'Agglomerative Clustering']):
        lines += [r'\hline', r'\multicolumn{9}{l}{\textbf{' + heading + r'}} \\', r'\hline']
        for ii, index in enumerate(INDICES):
            key = f'embedding_{scenario}/{ci * len(INDICES) + ii}'
            lines.append(' & '.join([index] + format_cells(key, rows[key])) + r' \\')
    lines += [r'\hline', r'\end{tabular}', '}', r'\end{table}']
    return '\n'.join(lines) + '\n'


def write_tables(output, rows):
    """Write the six LaTeX fragments for main text Tables 1--5 and Supplement benchmark tables."""
    tables = output/'tables'
    embeddings = output/'local_embedding_tables'
    tables.mkdir(parents=True, exist_ok=True)
    embeddings.mkdir(parents=True, exist_ok=True)
    # Main text Tables 1--3 use Case 2 K-means means, Case 3 spectral results,
    # and Case 4 K-means results, respectively, all evaluated against ACC.
    main = [benchmark_table(rows, 2, 'ACC', 'KMeans', 'mean', main=True),
            benchmark_table(rows, 3, 'ACC', 'Spectral', main=True),
            benchmark_table(rows, 4, 'ACC', 'KMeans', main=True)]
    (tables/'main_tables.tex').write_text('\n\n'.join(main) + '\n')
    for case, filename in [(2, 'case2_tables_clean_v6.tex'),
                           (3, 'nonconvex_tables_clean_v6.tex'),
                           (4, 'realdata_tables_clean_v6.tex')]:
        (tables/filename).write_text(supplementary_tables(rows, case))
    # Write tuned and oracle embedding summaries separately for Tables 4 and 5.
    for scenario in ['tuned', 'oracle']:
        (embeddings/f'{scenario}_by_index.tex').write_text(embedding_table(rows, scenario))


def build(output,results_root=None):
    """Aggregate the requested result directory and export the paper table fragments."""
    # Resolve which result tree supplies the values, then combine benchmark
    # correlations and embedding-improvement counts before rendering the LaTeX fragments.
    global DATA_ROOT
    DATA_ROOT=Path(results_root).resolve() if results_root else ROOT
    rows,raw,short_runs=benchmark_rows();erows,counts=embedding_rows();rows.update(erows)
    output.mkdir(parents=True,exist_ok=True)
    write_tables(output,rows)
    report={'table_rows':len(rows),'benchmark_rows':len(rows)-len(erows),'embedding_rows':len(erows),'benchmark_cells':len(raw)+31*9,'embedding_cells':len(erows)*8,'short_case2_runs':short_runs}
    print(f'Built {len(rows)} table rows from numerical results.')
    if short_runs: print(f'WARNING: {len(short_runs)} metric/method vectors have fewer than 100 Case 2 replications. These saved results are used as supplied; see the README reproduction notes.')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs')
    parser.add_argument('--results-root',type=Path,default=ROOT)
    args=parser.parse_args();build(args.output.resolve(),args.results_root)
