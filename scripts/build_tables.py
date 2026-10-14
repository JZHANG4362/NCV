#!/usr/bin/env python3
"""Recalculate manuscript cells from numerical caches, then apply layout templates.

The templates contain captions, labels and layout, but no result cells.
reference/table_values.json is used ONLY for independent comparison, never
as an input to the numerical calculations.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import importlib.util
import json
from pathlib import Path
import pickle
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT
spec = importlib.util.spec_from_file_location('benchmark', ROOT/'scripts/legacy/build_manuscript_outputs.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
CMS = benchmark.CLUSTERING
INDICES = ['Silhouette','VRC','WG','WB','DBCV','CVDD','VIASCKDE','DCSI','DB']
PACKAGES = {
    'alt': ('section_4_6_local_embeddings/other_indices','tsne_umap_alt_indices_tuned_origkgrid_v1'),
    'sil': ('section_4_6_local_embeddings/silhouette','tsne_umap_tuned_embedding_origkgrid_v1'),
}

def load(path):
    with path.open('rb') as f:
        payload = pickle.load(f)
    if payload.get('score_direction_convention') != 'larger_is_better_v1':
        raise ValueError(f'Unexpected score convention: {path}')
    return payload['results']

def benchmark_rows():
    rows = {}; raw = []; short_runs=[]
    for case in [2,3,4]:
        for metric in benchmark.METRICS:
            results = load(benchmark.metric_path(case,metric))
            settings = ([f'k{k}_D{d}_{c}' for k,d,c in benchmark.CASE2_CASES] if case==2 else list(benchmark.CASE3_K if case==3 else benchmark.CASE4_K))
            for cm in CMS:
                for stat in (['mean','median'] if case==2 else ['value']):
                    label = (f'tab:case2_{metric}_{cm}_{stat}' if case==2 else f'tab:{"nonconvex" if case==3 else "realdata"}_{metric}_{cm}')
                    for i,dataset in enumerate(settings):
                        source = results[cm][dataset]; vals=[]
                        for method in benchmark.METHODS:
                            v = source[method]
                            if case==2:
                                if len(v)!=100 and stat=='mean':
                                    short_runs.append(dict(metric=metric,clustering=cm,dataset=dataset,method=method,expected=100,actual=len(v)))
                                finite=np.asarray(v,dtype=float);finite=finite[np.isfinite(finite)]
                                v=float((np.mean if stat=='mean' else np.median)(finite)) if len(finite) else np.nan
                            vals.append(float(v))
                            raw.append(dict(case=case,metric=metric,clustering=cm,summary=stat,dataset=dataset,method=method,value=v))
                        rows[f'{label}/{i}']=[f'{x:.3f}' if np.isfinite(x) else 'NA' for x in vals]
    for target,source,n in [('tab:main_case2_acc','tab:case2_ACC_KMeans_mean',12),('tab:main_nonconvex_acc_spectral','tab:nonconvex_ACC_Spectral',10),('tab:main_labeled_acc_kmeans','tab:realdata_ACC_KMeans',9)]:
        for i in range(n):rows[f'{target}/{i}']=rows[f'{source}/{i}']
    return rows,raw,short_runs

def embedding_rows():
    counts=[]; rows={}; allstats={}
    for scenario in ['tuned','oracle']:
        for case in [2,3,4]:
            fixed={};sens={}
            for kind,(package,tag) in PACKAGES.items():
                folder=DATA_ROOT/package/f'data/case{case}'
                fixed[kind]=load(folder/f'manifold_fixed_results_{tag}.pkl')
                sens[kind]=load(folder/f'manifold_param_sensitivity_results_{tag}.pkl')
            datasets=([f'k{k}_D{d}_{c}' for k,d,c in benchmark.CASE2_CASES] if case==2 else list(benchmark.CASE3_K if case==3 else benchmark.CASE4_K))
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
                                value=max(finite) if finite else np.nan
                            available=orig is not None and value is not None and np.isfinite(orig) and np.isfinite(value)
                            if available:
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
                    stat.append((sum(s[0] for s in stat),sum(s[1] for s in stat)))
                    cells.extend(f'{round(n/d,2)*100:.0f}\\%' for n,d in stat)
                rows[f'embedding_{scenario}/{ci*len(INDICES)+ii}']=cells
    return rows,counts

def format_cells(key,values):
    if key.startswith('embedding_'):
        return [rf'\textbf{{{v}}}' if i in [3,7] and int(v[:-2])>50 else v for i,v in enumerate(values)]
    finite=[float(x) for x in values if x!='NA']
    maximum=max(finite) if finite else None
    out=[]
    for i,v in enumerate(values):
        c=v
        if i==0 and v!='NA' and values[1]!='NA' and float(v)>float(values[1]):c=rf'\textbf{{{v}}}'
        if v!='NA' and float(v)==maximum:c+=r'$^{\star}$'
        out.append(c)
    return out

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def summary_statistics(raw,counts):
    groups=defaultdict(dict)
    for row in raw:
        if row['summary']!='median':
            groups[row['case'],row['metric'],row['clustering'],row['dataset']][row['method']]=row['value']
    cases={}
    for case in [2,3,4]:
        total=improved=best=top3=0
        for key,values in groups.items():
            if key[0]!=case:continue
            value=values['rNN_Hard'];finite=[x for x in values.values() if np.isfinite(x)]
            total+=1;improved+=int(value>values['Silhouette'])
            best+=int(value==max(finite));top3+=int(sum(x>value for x in finite)<3)
        cases[str(case)]=dict(comparisons=total,improves_on_sil=improved,row_maximum=best,top_three=top3)
    pooled={k:sum(c[k] for c in cases.values()) for k in cases['2']}
    silhouette={}
    for scenario in ['tuned','oracle']:
        silhouette[scenario]={}
        for rep in ['TSNE','UMAP']:
            selected=[r for r in counts if r['scenario']==scenario and r['representation']==rep and r['index']=='Silhouette']
            n=sum(r['improved'] for r in selected);d=sum(r['valid'] for r in selected)
            silhouette[scenario][rep]=dict(improved=n,valid=d,percent=100*n/d)
    return dict(benchmark_by_case=cases,benchmark_pooled=pooled,silhouette_embeddings_pooled=silhouette)

def build(output,verify=True,results_root=None):
    global DATA_ROOT
    DATA_ROOT=Path(results_root).resolve() if results_root else ROOT
    benchmark.ROOT=DATA_ROOT
    rows,raw,short_runs=benchmark_rows();erows,counts=embedding_rows();rows.update(erows)
    expected=json.loads((ROOT/'reference/table_values.json').read_text())
    if set(rows)!=set(expected):raise ValueError('Result/reference row keys differ')
    mismatches=[{'row':k,'computed':rows[k],'manuscript':v['values']} for k,v in expected.items() if rows[k]!=v['values']]
    output.mkdir(parents=True,exist_ok=True)
    report={'table_rows':len(rows),'benchmark_rows':len(rows)-len(erows),'embedding_rows':len(erows),'benchmark_cells':len(raw)+31*9,'embedding_cells':len(erows)*8,'short_case2_runs':short_runs,'mismatches':mismatches}
    (output/'table_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    write_csv(output/'benchmark_values.csv',raw)
    write_csv(output/'embedding_counts.csv',counts)
    (output/'summary_statistics.json').write_text(json.dumps(summary_statistics(raw,counts),indent=2)+'\n')
    for path in (ROOT/'templates').glob('*.tex'):
        text=path.read_text()
        def replace(match):
            key=match[1]
            return ' & '.join(expected[key]['metadata']+format_cells(key,rows[key]))+r' \\'
        text=re.sub(r'@@ROW:(.*?)@@',replace,text)
        dest=output/('local_embedding_tables' if path.stem in ['tuned_by_index','oracle_by_index'] else 'tables')/path.name
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(text)
    print(f'Built {len(rows)} rows; {len(mismatches)} rows differ from the manuscript snapshot.')
    if short_runs: print(f'WARNING: {len(short_runs)} metric/method vectors have fewer than 100 Case 2 replications. See table_validation.json.')
    if mismatches and verify:raise SystemExit('Mismatch: see '+str(output/'table_validation.json'))
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs')
    parser.add_argument('--allow-changed-results',action='store_true',help='write changed results after a fresh run, while still reporting differences')
    parser.add_argument('--results-root',type=Path,default=ROOT)
    args=parser.parse_args();build(args.output.resolve(),not args.allow_changed_results,args.results_root)
