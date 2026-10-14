#!/usr/bin/env python3
"""Run a trusted repository notebook's code cells in order, without saving outputs."""
import argparse
import json
import os
from pathlib import Path

def run(path):
    path=Path(path).resolve()
    notebook=json.loads(path.read_text())
    os.chdir(path.parent)
    namespace={'__name__':'__main__','__file__':str(path)}
    for i,cell in enumerate(notebook['cells']):
        if cell['cell_type']=='code':
            print(f'{path.name}: cell {i}',flush=True)
            exec(compile(''.join(cell['source']),f'{path.name}:cell{i}','exec'),namespace)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebook',type=Path)
    args=parser.parse_args();run(args.notebook)
