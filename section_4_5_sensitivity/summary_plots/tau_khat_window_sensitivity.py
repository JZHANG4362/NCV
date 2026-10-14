r"""
Dedicated analysis of r* sensitivity to (tau, K-hat, diagnostic-window
length b) -- plan Section 4.4 -- across all three data families. Pure
post-processing: every quantity here was already computed and cached by
the three sibling rstar_selection_landscape.py scripts (see their
tau_sensitivity/khat_sensitivity/b_sensitivity fields), so this script
does NOT re-cluster or re-sweep anything -- it only loads, aggregates
(mean across replications for simulated_data), tabulates,
and plots.

Usage:
    python tau_khat_window_sensitivity.py
"""

import glob
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Must match the grids used by all three Layer 1 scripts exactly (they are
# what determines which keys exist in each pickle's tau_sensitivity /
# khat_sensitivity / b_sensitivity dicts) -- these are NOT independently
# swept here, only used to know which keys to read back out.
TAU_GRID = [0.85, 0.90, 0.95, 1.00]
KHAT_GRID = list(range(20, 61, 10))
B_GRID = [5, 10, 20]

OUT_TABLE_DIR = 'tables'
OUT_FIG_DIR = 'figures'
os.makedirs(OUT_TABLE_DIR, exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)

# The three landscape caches live in sibling folders under Section 4.5.
# Run from summary_plots/ (the top-level reproduction command does this).
# `replicated=True` for Case 2 means
# each pickle holds a LIST of per-replication results (result['reps'][variant])
# that must be averaged; Case 3/4 pickles hold a single result per dataset
# (result['variants'][variant]).
FAMILIES = {
    'simulated_data': dict(
        glob_pattern='../simulated_data/data/rstar_landscape_*.pkl',
        replicated=True, label='Case 2'),
    'nonconvex_data': dict(
        glob_pattern='../nonconvex_data/data/rstar_landscape_*.pkl',
        replicated=False, label='Case 3'),
    'labeled_data': dict(
        glob_pattern='../labeled_data/data/rstar_landscape_*.pkl',
        replicated=False, label='Case 4'),
}


def load_family(family_key):
    """Load every cached landscape pickle for one data family (Case 2, 3, or
    4) and extract, for each (dataset/case, hard-or-soft variant), the
    already-computed tau/K-hat/b sensitivity results -- no recomputation,
    just reading fields that Layer 1's run_one_case/run_one_dataset already
    populated. For Case 2 (replicated=True), each pickle holds one result
    per replication; this averages r* and rho_E(r*) across replications for
    each grid point so Case 2 contributes one row per case like the other
    two families contribute one row per dataset."""
    info = FAMILIES[family_key]
    rows = []
    for path in sorted(glob.glob(info['glob_pattern'])):
        with open(path, 'rb') as f:
            result = pickle.load(f)
        name = result.get('ckey', result.get('dataset'))
        for variant in ['hard', 'soft']:
            if info['replicated']:
                # Case 2: average r* and rho_E(r*) across the N_REP
                # replications, separately for each tau/K-hat/b grid value.
                reps = result['reps'][variant]
                # mean r* and rho per grid point, across replications
                tau_row = {}
                for tau in TAU_GRID:
                    r_stars = [rp['tau_sensitivity'][tau]['r_star'] for rp in reps]
                    rhoEs = [rp['tau_sensitivity'][tau]['rhoE'] for rp in reps]
                    tau_row[tau] = dict(r_star=float(np.mean(r_stars)),
                                         rhoE=float(np.nanmean(rhoEs)))
                khat_row = {}
                for khat in KHAT_GRID:
                    r_stars = [rp['khat_sensitivity'][khat]['r_star'] for rp in reps]
                    rhoEs = [rp['khat_sensitivity'][khat]['rhoE'] for rp in reps]
                    khat_row[khat] = dict(r_star=float(np.mean(r_stars)),
                                           rhoE=float(np.nanmean(rhoEs)))
                b_row = {}
                for b in B_GRID:
                    r_stars = [rp['b_sensitivity'][b]['r_star'] for rp in reps]
                    rhoEs = [rp['b_sensitivity'][b]['rhoE'] for rp in reps]
                    b_row[b] = dict(r_star=float(np.mean(r_stars)),
                                     rhoE=float(np.nanmean(rhoEs)))
                n_units = len(reps)
            else:
                # Case 3/4: a single (unreplicated) result per dataset --
                # read the sensitivity dicts straight through.
                v = result['variants'][variant]
                tau_row = {tau: dict(r_star=v['tau_sensitivity'][tau]['r_star'],
                                      rhoE=v['tau_sensitivity'][tau]['rhoE'])
                           for tau in TAU_GRID}
                khat_row = {khat: dict(r_star=d['r_star'], rhoE=d['rhoE'])
                            for khat, d in v['khat_sensitivity'].items()}
                b_row = {b: dict(r_star=d['r_star'], rhoE=d['rhoE'])
                         for b, d in v['b_sensitivity'].items()}
                n_units = 1

            rows.append(dict(family=family_key, name=name, variant=variant,
                              n_units=n_units, tau=tau_row, khat=khat_row, b=b_row))
    return rows


def build_wide_table(rows, sweep_key, variant_filter='hard'):
    """One row per (family, name), one column per grid value, cell = r*.
    sweep_key selects which of 'tau'/'khat'/'b' to pivot on; variant_filter
    restricts to the hard or soft rNN variant. Produces the
    {tau,khat,b}_sensitivity_{hard,soft}.csv tables: for every one of the 28
    settings (12 Case2 + 10 Case3 + 6 Case4), what r* would Algorithm 1 have
    picked and what rho_E(r*) would it have achieved, at every grid value."""
    out_rows = []
    for r in rows:
        if r['variant'] != variant_filter:
            continue
        row = {'family': FAMILIES[r['family']]['label'], 'name': r['name']}
        for key, d in r[sweep_key].items():
            row[f'r*(={key})'] = d['r_star']
        for key, d in r[sweep_key].items():
            row[f'rhoE(={key})'] = round(d['rhoE'], 3) if not np.isnan(d['rhoE']) else np.nan
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def swing_summary(rows, sweep_key, variant_filter='hard'):
    """How much does r* and rhoE(r*) swing across the grid, per row? For
    each (family, name), computes the range (max-min) of r* and of
    rho_E(r*) across the grid values of sweep_key -- a small range means the
    method's output is insensitive to that tuning parameter for that
    setting. Feeds both the *_swing_{hard,soft}.csv tables and the LaTeX
    swing table below."""
    out_rows = []
    for r in rows:
        if r['variant'] != variant_filter:
            continue
        r_stars = [d['r_star'] for d in r[sweep_key].values()]
        rhoEs = [d['rhoE'] for d in r[sweep_key].values() if not np.isnan(d['rhoE'])]
        out_rows.append(dict(
            family=FAMILIES[r['family']]['label'], name=r['name'],
            r_star_range=max(r_stars) - min(r_stars),
            rhoE_range=(max(rhoEs) - min(rhoEs)) if rhoEs else np.nan,
            rhoE_min=min(rhoEs) if rhoEs else np.nan,
            rhoE_max=max(rhoEs) if rhoEs else np.nan,
        ))
    return pd.DataFrame(out_rows)


def write_swing_latex(swing, sweep_key, out_path):
    """Write the hard-variant swing table included by the report."""
    captions = {
        'tau': (r'Range of $r^*$ and $\rho_E(r^*)$ swept across '
                r'$\tau\in\{0.85,0.90,0.95,1.00\}$, hard $r$NN variant, '
                r'sorted by $\rho_E$ range within each family.'),
        'khat': (r'Range of $r^*$ and $\rho_E(r^*)$ swept across '
                 r'$\widehat K\in\{20,30,40,50,60\}$, hard $r$NN variant.'),
        'b': (r'Range of $r^*$ and $\rho_E(r^*)$ swept across buffer '
              r'$b\in\{5,10,20\}$, hard $r$NN variant.'),
    }
    family_order = [info['label'] for info in FAMILIES.values()]
    lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\small',
        rf'\caption{{{captions[sweep_key]}}}',
        r'\begin{tabular}{llrrrr}',
        r'\toprule',
        r'Family & Setting & $r^*$ range & $\rho_E(r^*)$ range & $\rho_E$ min & $\rho_E$ max \\',
        r'\midrule',
    ]
    for family_index, family in enumerate(family_order):
        if family_index:
            lines.append(r'\addlinespace')
        part = swing[swing['family'] == family].sort_values('rhoE_range', ascending=False)
        for _, row in part.iterrows():
            setting = str(row['name']).replace('_', r'\_')
            lines.append(
                f"{family} & {setting} & {row['r_star_range']:.1f} & "
                f"{row['rhoE_range']:.3f} & {row['rhoE_min']:.3f} & "
                f"{row['rhoE_max']:.3f} \\\\"
            )
    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        rf'\label{{tab:{sweep_key}-swing}}',
        r'\end{table}',
    ])
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def plot_sweep(rows, sweep_key, xlabel, out_path, variant_filter='hard'):
    """One line per (family, name) setting, showing rho_E(r*) as sweep_key
    varies across its grid, colored by data family (Case 2/3/4). This is the
    figure that becomes rstar_{tau,khat,b}_sensitivity.pdf in the
    manuscript (hard variant only; see ../../tools/build_manuscript_outputs.py)."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = {'Case 2': 'steelblue',
              'Case 3': 'darkorange',
              'Case 4': 'seagreen'}
    seen_labels = set()
    for r in rows:
        if r['variant'] != variant_filter:
            continue
        label = FAMILIES[r['family']]['label']
        xs = sorted(r[sweep_key].keys())
        ys = [r[sweep_key][x]['rhoE'] for x in xs]
        plot_label = label if label not in seen_labels else None
        seen_labels.add(label)
        ax.plot(xs, ys, marker='o', alpha=0.5, color=colors[label], label=plot_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'$\rho$')
    variant_suffix = '' if variant_filter == 'hard' else f' ({variant_filter})'
    ax.set_title(f'Sensitivity of $\\rho$ to {xlabel}{variant_suffix}')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    # Load and combine all three Layer 1 caches into one flat list of rows
    # (one row per (family, dataset/case, hard-or-soft variant)) -- 28
    # settings x 2 variants = 56 rows total.
    all_rows = []
    for family_key in FAMILIES:
        all_rows.extend(load_family(family_key))
    print(f'Loaded {len(all_rows)} (family, name, variant) rows.')

    # For each of the three tuning parameters (tau, K-hat, b) and each
    # rNN variant (hard, soft): write the wide r*/rhoE table, the swing
    # summary, and (hard only) the LaTeX swing table, plus the sensitivity
    # plot. This is where the 6 {tau,khat,b}_sensitivity_{hard,soft}.pdf
    # figures and matching CSVs are produced.
    for sweep_key, xlabel, fname in [
        ('tau', r'$\tau$', 'tau'),
        ('khat', r'$\widehat K$', 'khat'),
        ('b', r'buffer $b$', 'b'),
    ]:
        for variant in ['hard', 'soft']:
            wide = build_wide_table(all_rows, sweep_key, variant)
            wide.to_csv(os.path.join(OUT_TABLE_DIR, f'{fname}_sensitivity_{variant}.csv'), index=False)

            swing = swing_summary(all_rows, sweep_key, variant)
            swing.to_csv(os.path.join(OUT_TABLE_DIR, f'{fname}_swing_{variant}.csv'), index=False)
            if variant == 'hard':
                write_swing_latex(
                    swing, sweep_key,
                    os.path.join(OUT_TABLE_DIR, f'{fname}_swing_table.tex'),
                )

            plot_sweep(all_rows, sweep_key, xlabel,
                       os.path.join(OUT_FIG_DIR, f'{fname}_sensitivity_{variant}.pdf'), variant)

        print(f'\n=== {sweep_key} swing summary (hard) ===')
        print(swing_summary(all_rows, sweep_key, 'hard').to_string(index=False))

    print(f'\nWrote tables to {OUT_TABLE_DIR}/, figures to {OUT_FIG_DIR}/.')


if __name__ == '__main__':
    main()
