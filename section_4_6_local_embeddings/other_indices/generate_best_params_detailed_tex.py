r"""
DETAILED companion to generate_best_params_tex.py (the per-CASE oracle
summary): for each case, clustering method, and INDIVIDUAL DATASET,
reports the ORACLE-best perplexity (t-SNE) and n_neighbors (UMAP) for
each of the 8 alternative validity indices -- the parameter value that
maximizes THAT dataset's own Spearman rho with ACC, using its own
per-parameter sensitivity curve (no case-level averaging here, unlike
generate_best_params_tex.py). Same relationship as manifold_tsne_umap_
summary.tex (per-dataset detail) has to manifold_tsne_umap_averages.tex
(per-case average) -- this file is the per-dataset detail, best_params_
by_case_and_index.tex is the per-case average, for the SAME oracle-best-
parameter question.

Computed directly from the already-saved per-parameter sensitivity curves
-- no re-running of any cluster job needed.

IMPORTANT: like the case-level version, this is the ORACLE choice (best
parameter GIVEN knowledge of ACC) for THIS dataset specifically -- not
what Algorithm 1 would actually select from that dataset's K-curve shape
alone (Algorithm 1's actual pick is already reported as the {idx}_TSNE /
{idx}_UMAP columns in manifold_tsne_umap_summary.tex).

Run this LOCALLY after retrieving data/ from the cluster and (for Case 4)
running merge_manifold_case4_results.py:

    python generate_best_params_detailed_tex.py

Writes tables/best_params_detailed.tex. Compile with (run TWICE, in ONE
command, for the table of contents):
    cd tables && pdflatex best_params_detailed.tex && pdflatex best_params_detailed.tex
"""

import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from generate_manifold_tsne_umap_tex import (
    DATA_DIR, EXPERIMENT_TAG, SCORE_DIRECTION_CONVENTION, CLUSTERING_METHODS,
    CM_LABELS, CASE2_META, CASE2_DISPLAY, CASE2_ORDER,
    CASE3_META, CASE3_DISPLAY, CASE3_ORDER,
    CASE4_META, CASE4_DISPLAY, CASE4_ORDER,
)
from generate_oracle_landscape_plots import INDEX_NAMES, INDEX_COLORS

sns.set_style('whitegrid')

OUT_DIR = 'tables'
FIG_DIR = 'figures/best_params_detailed'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

CASES = [
    dict(tag='case2', title='Case 2 -- Simulated Data', meta=CASE2_META,
         display=CASE2_DISPLAY, order=CASE2_ORDER,
         path=os.path.join(DATA_DIR, 'case2', f'manifold_param_sensitivity_results_{EXPERIMENT_TAG}.pkl'),
         star_path=os.path.join(DATA_DIR, 'case2', f'manifold_param_star_results_{EXPERIMENT_TAG}.pkl')),
    dict(tag='case3', title='Case 3 -- Nonconvex Data', meta=CASE3_META,
         display=CASE3_DISPLAY, order=CASE3_ORDER,
         path=os.path.join(DATA_DIR, 'case3', f'manifold_param_sensitivity_results_{EXPERIMENT_TAG}.pkl'),
         star_path=os.path.join(DATA_DIR, 'case3', f'manifold_param_star_results_{EXPERIMENT_TAG}.pkl')),
    dict(tag='case4', title='Case 4 -- Real Labeled Data', meta=CASE4_META,
         display=CASE4_DISPLAY, order=CASE4_ORDER,
         path=os.path.join(DATA_DIR, 'case4', f'manifold_param_sensitivity_results_{EXPERIMENT_TAG}.pkl'),
         star_path=os.path.join(DATA_DIR, 'case4', f'manifold_param_star_results_{EXPERIMENT_TAG}.pkl')),
]


def load_results(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        payload = pickle.load(f)
    if payload.get('score_direction_convention') != SCORE_DIRECTION_CONVENTION:
        raise ValueError(f'{path} does not use the expected score-direction convention.')
    return payload['results']


def best_param(curve):
    """curve: {param_value: rho}. Returns (best_param, best_rho) or (None, None)."""
    finite = {p: r for p, r in curve.items() if r is not None and np.isfinite(r)}
    if not finite:
        return None, None
    best_p = max(finite, key=finite.get)
    return best_p, finite[best_p]


def to_latex_table(case_tag, cm, meta, display, order, results):
    '''One landscape table per (case, clustering_method): rows = datasets,
    columns = 8 indices x (best perplexity + rho, best n_neighbors + rho).
    `--' marks a dataset/index combination not yet retrieved.'''
    lines = []
    lines.append(r'\begin{landscape}')
    lines.append(r'\begin{table}[h]')
    lines.append(r'\scriptsize')
    lines.append(r'\setlength{\tabcolsep}{2.0pt}')
    lines.append(
        rf"\caption{{Oracle-best perplexity/$n_{{neighbors}}$ PER DATASET, {CM_LABELS.get(cm, cm)} "
        rf"({case_tag}) -- the parameter value maximizing THIS dataset's own Spearman "
        rf"$\rho$ with ACC (its own curve, not case-averaged). Format per index: "
        rf"best-perplexity(rho) / best-$n_{{neighbors}}$(rho). `--' marks a dataset/"
        rf"index combination not yet retrieved.}}"
    )
    lines.append(r'\centering')
    lines.append(r'\resizebox{\textwidth}{!}{%')
    lines.append(r'\begin{tabular}{l' + 'c' * (3 + len(INDEX_NAMES)) + '}')
    lines.append(r'\hline')
    header = ['Dataset', '$K^*$', '$N$', '$D$'] + INDEX_NAMES
    lines.append(' & '.join(header) + r' \\')
    lines.append(r'\hline')
    for dname in order:
        m = meta[dname]
        entry = results.get(cm, {}).get(dname, {})
        perpl_curves = entry.get('perplexity', {})
        nn_curves = entry.get('n_neighbors', {})
        cells = []
        for idx in INDEX_NAMES:
            bp, bpr = best_param(perpl_curves.get(idx, {}))
            bk, bkr = best_param(nn_curves.get(idx, {}))
            p_s = f'{bp}({bpr:+.2f})' if bp is not None else '--'
            k_s = f'{bk}({bkr:+.2f})' if bk is not None else '--'
            cells.append(f'{p_s} / {k_s}')
        lines.append(
            f"{display[dname]} & {m['K']} & {m['N']} & {m['D']} & "
            + ' & '.join(cells) + r' \\'
        )
    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    lines.append('}')
    lines.append(rf'\label{{tab:best_params_detailed_{case_tag}_{cm}}}')
    lines.append(r'\end{table}')
    lines.append(r'\end{landscape}')
    return '\n'.join(lines) + '\n'


def _panel(ax, curves, star_by_idx, xlabel, title):
    '''One landscape panel: one line per index, a filled circle at
    Algorithm 1's selected parameter, a star at the oracle-best parameter
    (same marker position if they coincide).'''
    any_data = False
    for idx in INDEX_NAMES:
        curve = curves.get(idx, {})
        finite = {p: r for p, r in curve.items() if r is not None and np.isfinite(r)}
        if not finite:
            continue
        any_data = True
        xs = sorted(finite.keys())
        ys = [finite[x] for x in xs]
        color = INDEX_COLORS[idx]
        ax.plot(xs, ys, linewidth=1.6, color=color, alpha=0.9)

        star = star_by_idx.get(idx)
        if star is not None and star in finite:
            ax.plot(star, finite[star], marker='o', color=color, markersize=7,
                     markeredgecolor='black', markeredgewidth=0.6, zorder=5)

        oracle_p = max(finite, key=finite.get)
        ax.plot(oracle_p, finite[oracle_p], marker='*', color=color, markersize=13,
                 markeredgecolor='black', markeredgewidth=0.6, zorder=4)

    ax.axhline(0, color='grey', linewidth=1, linestyle='--')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(r'Spearman $\rho$ with ACC', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)
    if not any_data:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)


def plot_dataset_landscape(case_tag, cm, dname, meta_entry, sens_entry, star_entry):
    '''2-panel figure for one (case, cm, dataset): left = t-SNE perplexity
    landscape (9 index lines), right = UMAP n_neighbors landscape.
    Filled circle = Algorithm 1's actual selected parameter for that
    index; star = the oracle-best parameter (this dataset's own curve).
    Saved to figures/best_params_detailed/{case_tag}/{cm}/{dname}.{pdf,png}.'''
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    p_star = star_entry.get('perplexity_star', {}) if star_entry else {}
    nn_star = star_entry.get('n_neighbors_star', {}) if star_entry else {}

    _panel(axes[0], sens_entry.get('perplexity', {}), p_star, 'perplexity', 't-SNE landscape')
    _panel(axes[1], sens_entry.get('n_neighbors', {}), nn_star, '$n_{neighbors}$', 'UMAP landscape')

    color_handles = [Line2D([0], [0], color=INDEX_COLORS[idx], lw=2, label=idx) for idx in INDEX_NAMES]
    marker_handles = [
        Line2D([0], [0], marker='o', color='grey', lw=0, markersize=7,
               markeredgecolor='black', label='Algorithm 1 selected'),
        Line2D([0], [0], marker='*', color='grey', lw=0, markersize=12,
               markeredgecolor='black', label='oracle best'),
    ]
    fig.legend(handles=color_handles + marker_handles, loc='lower center',
               ncol=6, fontsize=9, bbox_to_anchor=(0.5, -0.06))

    k_star, n_samples, D = meta_entry.get('K', '?'), meta_entry.get('N', '?'), meta_entry.get('D', '?')
    fig.suptitle(f'{cm} — {dname}  ($k^*={k_star}$, $D={D}$, $N={n_samples}$)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.06, 1, 0.93])

    out_dir = os.path.join(FIG_DIR, case_tag, cm)
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, f'{dname}.pdf')
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, f'{dname}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    return pdf_path


def figure_block(case_tag, cm, dname, display, pdf_path):
    lines = []
    lines.append(r'\begin{figure}[h]')
    lines.append(r'\centering')
    lines.append(rf'\includegraphics[width=\textwidth]{{{os.path.join("..", pdf_path)}}}')
    lines.append(
        rf'\caption{{{display[dname]} -- {CM_LABELS.get(cm, cm)} ({case_tag}). '
        r'One line per index; filled circle = Algorithm~1\'s actual selected '
        r'parameter; star = this dataset\'s oracle-best parameter. When circle '
        r'and star coincide, Algorithm~1 found the peak; when they differ, the '
        r'gap between them is what Algorithm~1 leaves on the table.}'
    )
    lines.append(r'\end{figure}')
    return '\n'.join(lines) + '\n'


DOC_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{pdflscape}

\title{Oracle-Best Perplexity / $n_{neighbors}$ PER DATASET\\
\large (auto-generated by generate\_best\_params\_detailed\_tex.py -- do not edit by hand)}
\author{}
\date{}

\begin{document}
\maketitle
\tableofcontents
\clearpage

\section*{What this reports}
\addcontentsline{toc}{section}{What this reports}

This is the per-dataset companion to \texttt{best\_params\_by\_case\_and\_
index.tex} (which averages across an entire case). Here, for every
individual dataset and clustering method, each of the 8 alternative
validity indices gets its own ORACLE-best perplexity (t-SNE) and
$n_{neighbors}$ (UMAP) -- the value that maximizes THAT dataset's own
Spearman $\rho$ with ACC, read directly off its own per-parameter
sensitivity curve. Format in each cell: \texttt{best-perplexity(} $\rho$
\texttt{) / best-}$n_{neighbors}$\texttt{(}$\rho$\texttt{)}.

\textbf{This is still an ORACLE value, not Algorithm~1's actual pick.}
Algorithm~1 (used for the \texttt{\_TSNE} / \texttt{\_UMAP} columns in
\texttt{manifold\_tsne\_umap\_summary.tex}) selects a parameter using only
the $K$-curve's SHAPE, never looking at ACC -- it will often NOT match the
oracle value shown here. Comparing the two directly answers: how much is
Algorithm~1 leaving on the table for this specific dataset/index?

Each table is followed by one landscape-curve figure per dataset (one
line per index, colored consistently with \texttt{figures/oracle\_
landscape/}): a filled circle marks Algorithm~1's actual selected
parameter, a star marks this dataset's oracle-best parameter. When the
two coincide, Algorithm~1 found the peak for that index on that dataset;
when they diverge, the visual gap between circle and star is exactly what
Algorithm~1 is leaving on the table.

"""

DOC_CLOSING = r"""\end{document}
"""


def main():
    parts = [DOC_PREAMBLE]

    for case in CASES:
        results = load_results(case['path'])
        star_results = load_results(case['star_path']) or {}
        parts.append(f"\\section{{{case['title']}}}\n")
        if results is None:
            parts.append(f"No sensitivity data file found at \\texttt{{{case['path'].replace('_', r'\_')}}} -- skipping.\n\n")
            print(f"{case['tag']}: NO DATA -- skipped")
            continue

        for cm in CLUSTERING_METHODS:
            if cm not in results:
                print(f"{case['tag']} / {cm}: no results -- skipping table")
                continue
            parts.append(f"\\subsection{{{CM_LABELS.get(cm, cm)}}}\n")
            parts.append(to_latex_table(case['tag'], cm, case['meta'], case['display'], case['order'], results))
            parts.append('')

            n_figs = 0
            for dname in case['order']:
                sens_entry = results.get(cm, {}).get(dname)
                if not sens_entry:
                    continue
                star_entry = star_results.get(cm, {}).get(dname, {})
                pdf_path = plot_dataset_landscape(
                    case['tag'], cm, dname, case['meta'][dname], sens_entry, star_entry)
                parts.append(figure_block(case['tag'], cm, dname, case['display'], pdf_path))
                parts.append('')
                n_figs += 1
            print(f"{case['tag']} / {cm}: table with {len(case['order'])} dataset rows, {n_figs} figures generated+embedded")
        parts.append(r'\clearpage')
        parts.append('')

    parts.append(DOC_CLOSING)
    combined = '\n'.join(parts)
    combined_path = os.path.join(OUT_DIR, 'best_params_detailed.tex')
    with open(combined_path, 'w') as f:
        f.write(combined)
    print(f'\nWrote: {combined_path}')
    print('Compile with (run pdflatex TWICE, in ONE command):')
    print('  cd tables && pdflatex best_params_detailed.tex && pdflatex best_params_detailed.tex')


if __name__ == '__main__':
    main()
