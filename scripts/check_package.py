#!/usr/bin/env python3
"""Validate the supplied manuscript snapshot without rerunning full experiments."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_tables import build

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strict-replications',action='store_true')
    parser.add_argument('--skip-hashes',action='store_true',help='for intentionally edited source copies')
    args=parser.parse_args()
    if not args.skip_hashes:
        lines=(ROOT/'reference/SHA256SUMS').read_text().splitlines()
        for line in lines:
            digest,name=line.split('  ',1)
            path=ROOT/name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                raise SystemExit('File differs from packaged snapshot: '+name)
        print(f'PASS: {len(lines)} packaged file hashes.')
    code_files=0
    for folder in sorted(ROOT.glob('section_4_*'))+[ROOT/'scripts']:
        for path in folder.rglob('*.py'):
            ast.parse(path.read_text(),filename=str(path));code_files+=1
        for path in folder.rglob('*.ipynb'):
            notebook=json.loads(path.read_text())
            for i,cell in enumerate(notebook['cells']):
                if cell['cell_type']=='code':ast.parse(''.join(cell['source']),filename=f'{path}:cell{i}')
            code_files+=1
    ast.parse((ROOT/'reproduce.py').read_text());code_files+=1
    print(f'PASS: syntax of {code_files} Python files/notebooks.')
    report=build(ROOT/'outputs/validation')
    short=report['short_case2_runs']
    if len(short)!=63 or any((s['clustering'],s['dataset'],s['actual'])!=('Spectral','k30_D200_sparse',39) for s in short):
        raise SystemExit('Case 2 replication coverage differs from the documented snapshot.')
    for name in ['verify_r_convention.py','verify_score_directions.py']:
        subprocess.run([sys.executable,str(ROOT/'scripts/legacy'/name)],check=True)
    figures=json.loads((ROOT/'reference/figure_provenance.json').read_text())
    if len(figures)!=38:raise SystemExit('Expected 38 figure snapshots')
    for item in figures:
        path=ROOT/'reference'/item['output']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['manuscript_sha256']:
            raise SystemExit('Archived figure changed: '+item['output'])
    print('PASS: 38 archived figure snapshots and 8,838 displayed table cells.')
    if args.strict_replications:
        raise SystemExit('STRICT CHECK FAILED: Spectral / k30_D200_sparse has 39 rather than 100 replications. See docs/AUDIT.md.')
    print('PASS with the documented Case 2 replication gap; this is a snapshot check, not full numerical replication.')

if __name__=='__main__':main()
