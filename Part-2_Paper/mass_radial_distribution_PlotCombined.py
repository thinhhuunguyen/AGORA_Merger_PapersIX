import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import yt
yt.set_log_level(0)

import setup
from importlib import reload
reload(setup)

from setup import load_timings
from setup import codetp_list, label_list

halotree_ver = 2013
merger_number = '0'

code_to_label = dict(zip(codetp_list, label_list))
# Row order requested: ART-I, GADGET-3, GADGET-4, GIZMO, ENZO, AREPO-T, RAMSES, CHANGA, GEAR
row_codetp_list = ['ART', 'GADGET3', 'GADGET4', 'GIZMO', 'ENZO', 'AREPO', 'RAMSES', 'CHANGA', 'GEAR']

n_rows = len(row_codetp_list)
n_cols = 7

# Figure is wider than a plain single grid would need: the extra width is spent
# on the two widened gaps (column 0 - column 1, and column 1 - column 2) plus a
# right margin reserved for the "Group N" labels/separators, while every panel
# keeps the exact same width/height as a uniform 7-column grid.
fig = plt.figure(figsize=(27.74, 34))

# Column 0, column 1, and columns 2-6 each get their own gridspec, with
# left/right chosen so all three grids produce identical panel sizes (same
# top/bottom/hspace too) -- only the gaps/margins around the grids differ. The
# right margin was widened (figure widened to match) to fit the row-group labels
# and separators; column 0's panel, column 1's panel, the column 0-1 gap, the
# column 1-2 gap, and the columns 2-6 block are all unchanged in absolute size
# from before.
gs_col0 = fig.add_gridspec(n_rows, 1, left=0.065554, right=0.175805, top=0.93, bottom=0.04,
                            hspace=0.18)
gs_col1 = fig.add_gridspec(n_rows, 1, left=0.225780, right=0.335884, top=0.93, bottom=0.04,
                            hspace=0.18)
gs_rest2 = fig.add_gridspec(n_rows, n_cols - 2, left=0.359330, right=0.963966, top=0.93, bottom=0.04,
                             wspace=0.12, hspace=0.18)

xticks = [0, 2, 4, 6, 8, 10]
hist_bin_log = np.linspace(0, 10, 45)

font_size = 26

col_titles = [
    'Before merger\n' + r'($t_{\rm pre-infall}$)',
    'After merger\n' + r'($t_{\rm post/eq}$)',
    'old + accretion',
    'infall',
    '1st passage',
    'coalescence',
    'deposit',
]

component_colors = {
    'old': 'black',
    'infall': 'royalblue',
    'pass': 'orange',
    'cls': 'red',
    'deposit': 'limegreen',
}

# (first_row, last_row, label) -- row indices are inclusive, into row_codetp_list
row_groups = [
    (0, 3, 'Group 1'),  # ART-I, GADGET-3, GADGET-4, GIZMO
    (4, 5, 'Group 2'),  # ENZO, AREPO-T
    (6, 8, 'Group 3'),  # RAMSES, CHANGA, GEAR
]

axes_grid = np.empty((n_rows, n_cols), dtype=object)

for i, codetp in enumerate(row_codetp_list):
    label = code_to_label[codetp]
    if codetp == 'CHANGA':
        step = 2
    elif codetp == 'GEAR':
        step = 3
    else:
        step = 1
    #
    idx_begin, idx_endinfall, idx_1stpass, idx_maxdist, idx_cls, \
    time_begin, time_endinfall, time_1stpass, time_maxdist, time_cls = load_timings(codetp, halotree_ver, merger_number)
    time_list = np.loadtxt('/work/hdd/bezm/gtg115x/Halo_Finding/%s/pfs_allsnaps_%s.txt' % (codetp, halotree_ver), dtype=str)[:, 2].astype(float)
    #
    time_eval = time_1stpass + 0.6
    if codetp == 'GEAR':
        idx_eval = np.argmin(abs(time_eval - time_list/1e3)) - (np.argmin(abs(time_eval - time_list/1e3)) % step) - 1
    else:
        idx_eval = np.argmin(abs(time_eval - time_list/1e3)) - (np.argmin(abs(time_eval - time_list/1e3)) % step)
    #
    dist_gal_begin, mass_gal_begin = np.load(
        '/work/hdd/bezm/tnguyen2/AGORA/analysis/radial_mass_distribution_ProgBranch-%s_Idx_%s_%s_ver2013.npy' % (merger_number, idx_begin - step, codetp),
        allow_pickle=True).tolist().values()
    #
    ID_old, dist_old, mass_old, ID_infall, dist_infall, mass_infall, ID_pass, dist_pass, mass_pass, \
    ID_cls, dist_cls, mass_cls, ID_deposit, dist_deposit, mass_deposit, \
    ID_gal, dist_gal, mass_gal = np.load(
        '/work/hdd/bezm/tnguyen2/AGORA/analysis/radial_mass_distribution_decomposed_ProgBranch-%s_Idx_%s_%s_ver2013.npy' % (merger_number, idx_eval, codetp),
        allow_pickle=True).tolist().values()
    #
    mass_total = mass_gal.sum()
    #
    # (dist, mass, color, percent-of-total or None)
    col_data = [
        (dist_gal_begin, mass_gal_begin, 'black', None),
        (dist_gal, mass_gal, 'black', None),
        (dist_old, mass_old, component_colors['old'], mass_old.sum() / mass_total * 100),
        (dist_infall, mass_infall, component_colors['infall'], mass_infall.sum() / mass_total * 100),
        (dist_pass, mass_pass, component_colors['pass'], mass_pass.sum() / mass_total * 100),
        (dist_cls, mass_cls, component_colors['cls'], mass_cls.sum() / mass_total * 100),
        (dist_deposit, mass_deposit, component_colors['deposit'], mass_deposit.sum() / mass_total * 100),
    ]
    #
    for j, (dist, mass, color, pct) in enumerate(col_data):
        if j == 0:
            ax = fig.add_subplot(gs_col0[i, 0])
        elif j == 1:
            ax = fig.add_subplot(gs_col1[i, 0])
        else:
            ax = fig.add_subplot(gs_rest2[i, j - 2])
        axes_grid[i, j] = ax
        if j == 1:
            # "After merger" total, stacked from the five components (bottom to top,
            # matching the component columns' color coding).
            stack_order = ['old', 'deposit', 'infall', 'pass', 'cls']
            stack_dist = {'old': dist_old, 'infall': dist_infall, 'pass': dist_pass,
                          'cls': dist_cls, 'deposit': dist_deposit}
            stack_mass = {'old': mass_old, 'infall': mass_infall, 'pass': mass_pass,
                          'cls': mass_cls, 'deposit': mass_deposit}
            ax.hist([stack_dist[c] for c in stack_order],
                    weights=[stack_mass[c] for c in stack_order],
                    bins=hist_bin_log, histtype='step', stacked=True, linewidth=3,
                    color=[component_colors[c] for c in stack_order])
        else:
            ax.hist(dist, weights=mass, bins=hist_bin_log, histtype='step', color=color, linewidth=3)
        #
        if pct is not None:
            if codetp == 'GIZMO' and j in (2, 3):  # old, infall columns get extra precision for GIZMO
                pct_str = '%.1f%%' % pct
            else:
                pct_str = '%.0f%%' % pct
            ax.text(0.95, 0.88, pct_str, transform=ax.transAxes, ha='right', va='top',
                     fontsize=font_size - 3, color=color, weight="bold")
        #
        if j == 0:
            ax.set_yscale('log')
            ax.set_ylim(1e6, 6e8)
        elif j == 1:
            ax.set_yscale('log')
            ax.set_ylim(1e6, 7e9)
        # else: columns 2-6 (old, infall, pass, cls, deposit) use a linear
        # y-scale with a matplotlib-determined range instead of a shared log one.
        ax.set_xticks(xticks)
        if i == n_rows - 1:
            ax.set_xlabel('d (kpc)', fontsize=font_size)
            ax.tick_params('x', labelsize=font_size)
        else:
            ax.set_xticklabels([])
        if j == 0 or j == 1:
            ax.set_ylabel(r'$M_{\bigstar} (M_\odot)$', fontsize=font_size)
        if j <= 1:
            ax.tick_params('y', labelsize=font_size)
            ax.yaxis.get_offset_text().set_fontsize(font_size - 4)
        else:
            ax.tick_params(labelleft=False)
        #
        if i == 0:
            ax.set_title(col_titles[j], fontsize=font_size - 2)
    #
    # Code name in the top-right corner of the column-0 panel (color/asterisk/dagger
    # encoding preserved from the original per-panel titles)
    if codetp in ('ART', 'GADGET3', 'GADGET4', 'GIZMO'):
        title_color = 'blue'
    else:
        title_color = 'black'
    if codetp in ('ART', 'RAMSES', 'GADGET3', 'GEAR'):
        row_text = label + '*'
    elif codetp == 'CHANGA':
        row_text = r'$\text{%s}^{\dagger}$' % label
    else:
        row_text = label
    axes_grid[i, 0].text(0.95, 0.93, row_text, transform=axes_grid[i, 0].transAxes,
                          ha='right', va='top', fontsize=font_size, color=title_color)
    #
    # Funnel connector between column 1 ("after merger" total) and column 2
    # ("old + accretion"): two lines from the midpoint of column 1's right edge
    # diverging out to column 2's top-left and bottom-left corners.
    pos_col1 = axes_grid[i, 1].get_position()
    pos_col2 = axes_grid[i, 2].get_position()
    apex_x, apex_y = pos_col1.x1, (pos_col1.y0 + pos_col1.y1) / 2
    for corner_y in (pos_col2.y0, pos_col2.y1):
        fig.add_artist(Line2D([apex_x, pos_col2.x0], [apex_y, corner_y], transform=fig.transFigure,
                               color='black', linewidth=1.0))

# Row-group labels ("Group 1/2/3") and separators, in the right margin freed up
# by gs_rest2's right edge (0.963966) and the figure's right edge (~1.0).
margin_x0 = 0.9660   # just right of the columns 2-6 block
margin_x1 = 0.9950   # short of the figure edge
label_x = (margin_x0 + margin_x1) / 2

for start, end, group_label in row_groups:
    y_top = axes_grid[start, 0].get_position().y1
    y_bottom = axes_grid[end, 0].get_position().y0
    fig.text(label_x, (y_top + y_bottom) / 2, group_label, rotation=-90,
              ha='center', va='center', fontsize=font_size, weight='bold')

# Separator between consecutive groups, at the midpoint of the gap between them
for (_, prev_end, _), (next_start, _, _) in zip(row_groups[:-1], row_groups[1:]):
    y_prev_bottom = axes_grid[prev_end, 0].get_position().y0
    y_next_top = axes_grid[next_start, 0].get_position().y1
    sep_y = (y_prev_bottom + y_next_top) / 2
    fig.add_artist(Line2D([margin_x0, margin_x1], [sep_y, sep_y], transform=fig.transFigure,
                           color='black', linewidth=1.0))


plt.savefig('/work/hdd/bezm/tnguyen2/figures/AGORA/November2025/mass_radialdistribution_CombinedFirstMerger_ProgBranch-%s_ver2013_logscale_ver3.png' % merger_number, dpi=300, bbox_inches='tight')
