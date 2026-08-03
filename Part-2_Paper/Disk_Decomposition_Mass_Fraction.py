import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Patch

import setup
from importlib import reload
reload(setup)

from setup import load_halotree_and_pfs, load_timings, sec_branch_compute
from setup import codetp_list, label_list

"""
Stacked-bar census of WHERE the DISK's stellar mass comes from, split by MERGER
GROUP, for one or more AGORA codes.

A single row of stacked bars is drawn (one bar per code):
    disk = Thin(3) + Thick(4) [+ disc(-1)]
Each bar is subdivided into the five merger groups
    Old | Infall | First passage | Cls + post-cls | Deposit
and the height of a segment is that group's fraction of the disk's mass
(so each bar sums to ~1 by default; see NORMALIZE). The disk-to-total mass
fraction D/T = M_disk / (M_disk + M_spheroid) is printed atop each bar.

Two D/T values are annotated per bar:
    dt_eval -- D/T at the evaluation epoch idx_eval (first passage + 0.6 Gyr,
               snapped to cadence). This is the epoch the stacked bars are built
               from.
    dt_pre  -- D/T one cadence step BEFORE the merger begins, at
               idx_pre = idx_begin - step, where idx_begin is the merger-begin
               snapshot from load_timings and step is the tree cadence.
The annotation reads:   D/T
                        dt_eval -> dt_pre
The spheroid mass is still used (to form each ratio) even though it is not
plotted.

Inputs (per code). The stacked bars + dt_eval come from the idx_eval pair,
addressed with the SAME idx_eval / prog_branch recipe as
circularity_compute_decompose():

  circularity_ProgBranch-%s_Idx_%s_%s_ver2013_Abadi2003method.npy
      % (codetp, prog_branch, idx_eval)              -> ID, mass, label
  circularity_decomposed_ProgBranch-%s_Idx_%s_%s_ver2013_Abadi2003method.npy
      % (merger_number, idx_eval, codetp)            -> ID_old/infall/pass/cls/deposit

dt_pre needs only the label+mass file at the pre-merger snapshot idx_pre:

  circularity_ProgBranch-%s_Idx_%s_%s_ver2013_Abadi2003method.npy
      % (codetp, prog_branch, idx_pre)               -> ID, mass, label

This is a SEPARATE, read-only plotting script: it does NOT modify or re-run the
circularity / decomposition pipeline, it only consumes their saved .npy outputs.

Run:
    python plot_circularity_mergergroups.py
"""

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
ANALYSIS_DIR  = '/work/hdd/bezm/tnguyen2/AGORA/analysis'
HALOTREE_VER  = 2013
MERGER_NUMBER = '0'

# Codes to draw (left -> right). Codes whose inputs are missing are skipped.
CODES = codetp_list

# Optional pretty display names for the x-axis (codetp -> shown label). Codes
# not listed here fall back to the codetp string itself.
CODE_LABELS = {}        # e.g. {'GADGET3': 'GADGET-3', 'ART': 'ART-I'}

# Component definition via assign_label() codes:
#   1 = bulge, 2 = halo, 3 = thin disc, 4 = thick disc, -1 = disc (unsplit)
SPHEROID_LABELS = (1, 2)            # not plotted, but needed for D/T
DISK_LABELS     = (3, 4, -1)        # -1 kept so disks survive a failed etacut

# Merger groups: key -> (display name, colour) and the circularity_decomposed
# key holding that group's star IDs.
GROUP_STYLE = {
    'old':     ('Old',            'black'),
    'infall':  ('Infall',         'royalblue'),
    'pass':    ('First passage',  'orange'),
    'cls':     ('Cls + post-cls', 'red'),
    'deposit': ('Deposit',        'limegreen'),
}
GROUP_IDKEY = {'old': 'ID_old', 'infall': 'ID_infall', 'pass': 'ID_pass',
               'cls': 'ID_cls', 'deposit': 'ID_deposit'}
# Bottom -> top stacking order.
GROUP_ORDER = ['cls', 'pass', 'infall', 'old', 'deposit']

# Single row now: spheroid panel removed, D/T reported instead.
ROWS = [('disk', 'Disk')]

# 'component' : each bar normalized by the DISK mass (bars sum to ~1)
# 'total'     : bars normalized by total galaxy stellar mass, so bar HEIGHTS
#               then encode the absolute disk share
NORMALIZE   = 'component'
ANNOTATE_DT = True                  # write D/T (dt_eval -> dt_pre) atop each bar

SAVE_PATH = '/work/hdd/bezm/tnguyen2/figures/AGORA/November2025/Disk_Decomposition_Fraction_ver2013_ver3.png'


# ----------------------------------------------------------------------
# x-axis label styling (adopted from the multi-panel set_title convention)
# ----------------------------------------------------------------------
def _code_title(codetp, label):
    """(text, color) for a code's x-axis label.
    Blue for ART/GADGET3/GADGET4/GIZMO; '*' for ART/RAMSES/GADGET3/GEAR;
    dagger for CHANGA. NOTE: the CHANGA r'$\\text{...}^{\\dagger}$' form needs
    text.usetex=True (amsmath); switch \\text -> \\mathrm for plain mathtext."""
    if codetp in ('ART', 'GADGET3', 'GADGET4', 'GIZMO'):
        color = 'blue'
    else:
        color = 'black'
    label = label_list[codetp_list.index(codetp)]
    if codetp in ('ART', 'RAMSES', 'GADGET3', 'GEAR'):
        text = label + '*'
    elif codetp == 'CHANGA':
        text = r'$\text{%s}^{\dagger}$' % label
    else:
        text = label
    return text, color


# ----------------------------------------------------------------------
# Snapshot resolution (verbatim recipe from circularity_compute_decompose)
# ----------------------------------------------------------------------
def _resolve_eval_indices(codetp):
    """(prog_branch, idx_eval, idx_pre) for `codetp`.

    idx_eval : evaluation epoch = first passage + 0.6 Gyr, snapped to the tree
               cadence, with GEAR's -1 offset (the bars + dt_eval epoch).
    idx_pre  : one cadence step before the merger begins, idx_begin - step.
               idx_begin is taken directly from load_timings (already a snapshot
               index); step is the tree cadence from load_halotree_and_pfs.
    """
    _, time_list, _, step = load_halotree_and_pfs(codetp, HALOTREE_VER, rawtree_skip=True)
    prog_branch, _, _ = sec_branch_compute(codetp, MERGER_NUMBER)
    (idx_begin, idx_endinfall, idx_1stpass, idx_maxdist, idx_cls,
     time_begin, time_endinfall, time_1stpass, time_maxdist, time_cls) = \
        load_timings(codetp, HALOTREE_VER, MERGER_NUMBER)

    time_eval = time_1stpass + 0.6
    k = np.argmin(abs(time_eval - time_list / 1e3))
    idx_eval = k - (k % step) - (1 if codetp == 'GEAR' else 0)

    idx_pre = idx_begin - step
    return prog_branch, idx_eval, idx_pre


# ----------------------------------------------------------------------
# File-path helpers
# ----------------------------------------------------------------------
def _circ_path(codetp, prog_branch, idx):
    """Path to the per-star circularity file (ID, mass, label) at `idx`."""
    return os.path.join(
        ANALYSIS_DIR,
        'circularity_ProgBranch-%s_Idx_%s_%s_ver2013_Abadi2003method.npy'
        % (codetp, prog_branch, idx))


def _dec_path(codetp, idx):
    """Path to the merger-group decomposition file at `idx`."""
    return os.path.join(
        ANALYSIS_DIR,
        'circularity_decomposed_ProgBranch-%s_Idx_%s_%s_ver2013_Abadi2003method.npy'
        % (MERGER_NUMBER, idx, codetp))


def _dt_from_circularity(path):
    """D/T = M_disk / (M_disk + M_spheroid) from a circularity file, or None if
    the file is missing or the galaxy carries no labelled stellar mass."""
    if not os.path.exists(path):
        return None
    c = np.load(path, allow_pickle=True).tolist()
    mass  = np.asarray(c['mass'], dtype=float)
    label = np.asarray(c['label'])
    m_disk = mass[np.isin(label, DISK_LABELS)].sum()
    m_sph  = mass[np.isin(label, SPHEROID_LABELS)].sum()
    tot = m_disk + m_sph
    return float(m_disk / tot) if tot > 0 else None


# ----------------------------------------------------------------------
# Data assembly
# ----------------------------------------------------------------------
def load_component_group_mass(codetp, prog_branch, idx_eval):
    """Return (M, totals) for `codetp` at idx_eval, or None if inputs missing.

    M[comp][group] : stellar mass [Msun] of stars in BOTH component `comp`
                     (spheroid/disk) and merger `group`.
    totals[comp]   : total stellar mass of `comp` [Msun].
    """
    circ_path = _circ_path(codetp, prog_branch, idx_eval)
    dec_path  = _dec_path(codetp, idx_eval)

    missing = [p for p in (circ_path, dec_path) if not os.path.exists(p)]
    if missing:
        print('[skip %s] missing: %s' % (codetp, missing[0]))
        return None

    c = np.load(circ_path, allow_pickle=True).tolist()
    d = np.load(dec_path,  allow_pickle=True).tolist()

    ID    = np.asarray(c['ID'])
    mass  = np.asarray(c['mass'], dtype=float)
    if codetp == 'GEAR':
        label = np.asarray(c['label_fix'])
    else:
        label = np.asarray(c['label'])

    comp_mask = {
        'spheroid': np.isin(label, SPHEROID_LABELS),
        'disk':     np.isin(label, DISK_LABELS),
    }
    group_ids = {g: np.asarray(d[GROUP_IDKEY[g]]) for g in GROUP_STYLE}

    M, totals = {comp: {} for comp in comp_mask}, {}
    for comp, cmask in comp_mask.items():
        ID_c, mass_c = ID[cmask], mass[cmask]
        totals[comp] = float(mass_c.sum())
        for g in GROUP_STYLE:
            M[comp][g] = float(mass_c[np.isin(ID_c, group_ids[g])].sum())
    return M, totals


# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
def make_plot(codes=CODES):
    data, used, galaxy_total, dt_pre = {}, [], {}, {}
    for code in codes:
        prog_branch, idx_eval, idx_pre = _resolve_eval_indices(code)

        res = load_component_group_mass(code, prog_branch, idx_eval)
        if res is None:
            continue
        M, totals = res
        data[code] = (M, totals)
        galaxy_total[code] = totals['spheroid'] + totals['disk']
        used.append(code)

        # pre-merger D/T (idx_begin - step). Missing file -> None (annotated --).
        pre_path = _circ_path(code, prog_branch, idx_pre)
        dt_pre[code] = _dt_from_circularity(pre_path)
        if dt_pre[code] is None:
            print('  [%s] no dt_pre at idx_pre=%s (file missing/empty): %s'
                  % (code, idx_pre, pre_path))

        # coverage diagnostic: do the five groups account for the whole component?
        for comp in ('spheroid', 'disk'):
            grp_sum = sum(M[comp].values())
            cov = grp_sum / totals[comp] if totals[comp] > 0 else 0.0
            print('  %-8s %-9s  M=%.3e Msun  group-coverage=%.3f'
                  % (code, comp, totals[comp], cov))

    if not used:
        raise RuntimeError('No code produced loadable inputs; nothing to plot.')

    x = np.arange(len(used))
    comp, comp_label = ROWS[0]                       # 'disk'
    fig, ax = plt.subplots(1, 1, figsize=(1.45 * len(used) + 2.5, 4.8))

    bottoms = np.zeros(len(used))
    for g in GROUP_ORDER:                            # bottom -> top
        name, color = GROUP_STYLE[g]
        heights = np.array([
            (data[code][0][comp][g] /
             (data[code][1][comp] if NORMALIZE == 'component' else galaxy_total[code]))
            if (data[code][1][comp] if NORMALIZE == 'component' else galaxy_total[code]) > 0
            else 0.0
            for code in used])
        ax.bar(x, heights, bottom=bottoms, width=0.62, color=color,
               edgecolor='black', linewidth=0.6, label=name)
        print(codes, g, heights)
        bottoms += heights

    if ANNOTATE_DT:
        for i, code in enumerate(used):
            tot = galaxy_total[code]
            if tot <= 0:
                continue
            dt_eval = data[code][1]['disk'] / tot                 # D/T at idx_eval
            pre = dt_pre.get(code)
            pre_txt = '%.2f' % pre if pre is not None else '--'
            ax.text(x[i], bottoms[i] + 0.02,
                    'D/T\n%s → %.2f' % (pre_txt, dt_eval),
                    ha='center', va='bottom', fontsize=14, color='0.25',
                    linespacing=1.1)

    ax.set_ylabel('%s Mass Fraction' % comp_label, fontsize=14)
    ax.set_ylim(0, max(1.0, bottoms.max()) * 1.16)
    ax.margins(x=0.04)
    ax.tick_params(axis='both', labelsize=14)
    ax.axhline(1.0, color='0.7', lw=0.8, ls=':')

    # x-axis code labels: per-code text + colour (ART/GADGET/RAMSES/... scheme)
    labels_txt, label_colors = [], []
    for code in used:
        disp = CODE_LABELS.get(code, code)
        t, col = _code_title(code, disp)
        labels_txt.append(t)
        label_colors.append(col)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_txt, fontsize=14)
    for tick, col in zip(ax.get_xticklabels(), label_colors):
        tick.set_color(col)

    handles = [Patch(facecolor=GROUP_STYLE[g][1], edgecolor='black',
                     label=GROUP_STYLE[g][0]) for g in reversed(GROUP_ORDER)]
    fig.legend(handles=handles, ncol=5, fontsize=14, frameon=False,
               loc='lower center', bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.08, 1, 0.98])
    #fig.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
    #plt.close(fig)
    print('saved %s   (codes: %s)' % (SAVE_PATH, ', '.join(used)))


if __name__ == '__main__':
    make_plot(CODES)