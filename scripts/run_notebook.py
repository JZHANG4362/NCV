#!/usr/bin/env python3
"""Execute a repository notebook's code cells in order, from its own directory.

Used for the Case 1 calculations and main text Figure 2. This runner
writes the files requested by the cells but leaves notebook outputs and
execution counts unchanged."""
import argparse
import json
import os
from pathlib import Path

def run(path):
    """Execute code cells in notebook order in one shared Python namespace."""
    path=Path(path).resolve()
    notebook=json.loads(path.read_text())
    # Run from the notebook's folder so data/ and figures/ paths resolve as
    # they do during an interactive execution of the same notebook.
    os.chdir(path.parent)
    # Reuse one namespace across cells: later plotting cells consume arrays
    # and functions defined by earlier data-generation and scoring cells.
    namespace={'__name__':'__main__','__file__':str(path)}
    for i,cell in enumerate(notebook['cells']):
        if cell['cell_type']=='code':
            print(f'{path.name}: cell {i}',flush=True)
            # Compile with a cell-specific filename for useful tracebacks. Files written
            # by the cell persist, but the notebook's stored outputs are not rewritten.
            exec(compile(''.join(cell['source']),f'{path.name}:cell{i}','exec'),namespace)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebook',type=Path)
    args=parser.parse_args();run(args.notebook)
