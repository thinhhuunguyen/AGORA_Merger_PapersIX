import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde

import setup
from importlib import reload
reload(setup)

from setup import load_halotree_and_pfs, load_timings, sec_branch_compute
from setup import codetp_list, label_list, color_list

# Create figure
fig = plt.figure(figsize=(20, 20))

# Outer GridSpec: 1 top row (2 panels) + a bottom block (3 rows of 5 panels each)
gs = GridSpec(
    nrows=2,
    ncols=10,
    figure=fig,
    height_ratios=[1, 3],
    hspace=0.16,
    wspace=0.5
)

# -------------------------
# Top row (2 equal panels)
# -------------------------
ax_left_top = fig.add_subplot(gs[0, 0:5])
ax_right_top = fig.add_subplot(gs[0, 5:10])

# -------------------------
# Bottom block: 3 stacked rows of 5 panels each, tightly spaced
# Row 1: ART-I, GADGET-3, GADGET-4, GIZMO
# Row 2: ENZO, AREPO-T
# Row 3: RAMSES, CHANGA, GEAR
# -------------------------
gs_bot = gs[1, :].subgridspec(nrows=3, ncols=10, hspace=0.15, wspace=0.5)
ax_bot_row1, ax_bot_row2, ax_bot_row3 = [], [], []
for i in range(5):
    ax_bot_row1.append(fig.add_subplot(gs_bot[0, i*2:(i+1)*2]))
    ax_bot_row2.append(fig.add_subplot(gs_bot[1, i*2:(i+1)*2]))
    ax_bot_row3.append(fig.add_subplot(gs_bot[2, i*2:(i+1)*2]))

# Which of the 3 bottom rows each code's lines belong to
code_row_map = {
    'ART': ax_bot_row1, 'GADGET3': ax_bot_row1, 'GADGET4': ax_bot_row1, 'GIZMO': ax_bot_row1,
    'ENZO': ax_bot_row2, 'AREPO': ax_bot_row2,
    'RAMSES': ax_bot_row3, 'CHANGA': ax_bot_row3, 'GEAR': ax_bot_row3,
}

halotree_ver = 2013
merger_number = '0'
font_size = 18
epsilon_lim = 1.5


xlim = (-epsilon_lim, epsilon_lim)
ylim = (-0.02, 1.62)
hist_bin_log = np.linspace(-epsilon_lim,epsilon_lim,100)

# Row group labels, on the left-most panel of each of the 3 bottom rows
for ax, group_label in zip([ax_bot_row1[0], ax_bot_row2[0], ax_bot_row3[0]], ['Group 1', 'Group 2', 'Group 3']):
    ax.text(
        0.08, 0.92, group_label,
        transform=ax.transAxes, fontsize=font_size*0.8, fontweight='bold',
        va='top', ha='left', zorder=20,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2)
    )


for k in range(len(codetp_list)):
#for k in [0]:
    codetp = codetp_list[k]
    #
    dist_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/dist_relvel_j_ProgBranch-%s_FirstMerger_%s_ver2013.npy' % (merger_number, codetp), allow_pickle=True).tolist()
    prog_branch, sec_branch, sec_branch_2 = sec_branch_compute(codetp, merger_number)
    redshift_list, time_list, pfs, step = load_halotree_and_pfs(codetp, halotree_ver, rawtree_skip=True)
    idx_begin, idx_endinfall, idx_1stpass, idx_maxdist, idx_cls, time_begin, time_endinfall, time_1stpass, time_maxdist, time_cls = load_timings(codetp, halotree_ver, merger_number)
    time_eval = time_1stpass + 0.6
    if codetp == 'GEAR':
        idx_eval = np.argmin(abs(time_eval - time_list/1e3)) - (np.argmin(abs(time_eval - time_list/1e3)) % step) - 1
    else:
        idx_eval = np.argmin(abs(time_eval - time_list/1e3)) - (np.argmin(abs(time_eval - time_list/1e3)) % step)
    #
    #Pre-infall
    epsilon_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/circularity_ProgBranch-%s_Idx_%s_%s_ver2013_Abadi2003method.npy' % (codetp, prog_branch, idx_begin - step), allow_pickle=True).tolist()
    epsilon = epsilon_data['circ']
    mass_gal = epsilon_data['mass']
    dist_gal = epsilon_data['dist']
    kde = gaussian_kde(epsilon, weights=mass_gal, bw_method='scott')
    x = np.linspace(-epsilon_lim, epsilon_lim, 1000)
    kde_values = kde(x)
    if codetp == 'ART' or codetp == 'RAMSES' or codetp == 'GADGET3' or codetp == 'GEAR':
        ax_left_top.plot(x, kde_values, linewidth=2, label=label_list[k] + '*', color=color_list[k])
    elif codetp == 'CHANGA':
        ax_left_top.plot(x, kde_values, linewidth=2, label=r'$\text{%s}^{\dagger}$' % label_list[k], color=color_list[k])
    else:
        ax_left_top.plot(x, kde_values, linewidth=2, label=label_list[k], color=color_list[k])
    ax_left_top.set_xlabel(r'$\epsilon = J_{z}/J_\text{circ}(E)$', fontsize = font_size)
    ax_left_top.set_ylabel(r'$F(\epsilon)$', fontsize = font_size)
    ax_left_top.axvline(0, linestyle='--', color='k')
    ax_left_top.axvline(1, linestyle='--', color='k')
    ax_left_top.tick_params('both', labelsize = font_size)
    ax_left_top.set_title(r'Before the merger ($t_\text{pre-infall}$)', fontsize = font_size)
    ax_left_top.set_xlim(xlim)
    ax_left_top.set_ylim(ylim)
    #
    #AFTER MERGER (equivalent timestep)
    #
    ID_old, epsilon_old, mass_old, dist_old, ID_infall, epsilon_infall, mass_infall, dist_infall, \
                ID_pass, epsilon_pass, mass_pass, dist_pass, \
                ID_cls, epsilon_cls, mass_cls, dist_cls,\
                ID_deposit, epsilon_deposit, mass_deposit, dist_deposit,\
                ID_gal, epsilon_gal, mass_gal, dist_gal = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/circularity_decomposed_ProgBranch-%s_Idx_%s_%s_ver2013_Abadi2003method.npy' % (merger_number, idx_eval, codetp), allow_pickle=True).tolist().values()
    #
    # Equivalent time-step
    #epsilon_hist = np.histogram(epsilon_gal[(np.abs(epsilon_gal)<1e99)*(dist_gal<10)], weights=mass_gal[(np.abs(epsilon_gal)<1e99)*(dist_gal<10)], bins=hist_bin_log)
    kde = gaussian_kde(epsilon_gal, weights=mass_gal, bw_method='scott')
    x = np.linspace(-epsilon_lim, epsilon_lim, 1000)
    kde_values = kde(x)
    ax_right_top.plot(x, kde_values, linewidth=2, label=label_list[k], color=color_list[k], zorder=10)
    ax_right_top.set_xlabel(r'$\epsilon = J_{z}/J_\text{circ}(E)$', fontsize = font_size)
    ax_right_top.axvline(0, linestyle='--', color='k', zorder=1, alpha=0.3)
    ax_right_top.axvline(1, linestyle='--', color='k', zorder=1, alpha=0.3)
    ax_right_top.tick_params('both', labelsize = font_size)
    ax_right_top.set_title(r'After the merger ($t_\text{post/eq}$)', fontsize = font_size)
    ax_right_top.set_xlim(xlim)
    ax_right_top.set_ylim(ylim)
    ax_right_top.set_yticks([])
    #
    # Route this code's lines to its assigned bottom row
    row_axes = code_row_map[codetp]
    is_top_group = row_axes is ax_bot_row1
    is_bottom_group = row_axes is ax_bot_row3

    categories = [
        ('Old', epsilon_old, mass_old),
        ('Infall', epsilon_infall, mass_infall),
        ('First passage', epsilon_pass, mass_pass),
        ('Cls + Post-cls', epsilon_cls, mass_cls),
        ('Deposit', epsilon_deposit, mass_deposit),
    ]

    for i, (cat_title, eps_arr, mass_arr) in enumerate(categories):
        ax = row_axes[i]
        kde = gaussian_kde(eps_arr, weights=mass_arr, bw_method='scott')
        x = np.linspace(-epsilon_lim, epsilon_lim, 1000)
        kde_values = kde(x)
        ax.plot(x, kde_values, linewidth=2, label=label_list[k], color=color_list[k], zorder=10)
        ax.axvline(0, linestyle='--', color='k', zorder=1, alpha=0.3)
        ax.axvline(1, linestyle='--', color='k', zorder=1, alpha=0.3)
        ax.tick_params('both', labelsize = font_size)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if i == 0:
            ax.set_ylabel(r'$F(\epsilon)$', fontsize = font_size)
        else:
            ax.set_yticklabels([])
        if is_top_group:
            ax.set_title(cat_title, fontsize = font_size)
        if is_bottom_group:
            if i == 0:
                ax.set_xlabel(r'$\epsilon = J_{z}/J_\text{circ}(E)$', fontsize = font_size)
        else:
            ax.set_xticklabels([])


handles, labels = ax_left_top.get_legend_handles_labels()

leg = fig.legend(
    handles,
    labels,
    loc='lower center',
    ncol=len(labels),          # one column per code
    fontsize=15,
    frameon=True,
    bbox_to_anchor=(0.5, 0.06)
)

for i in [0, 4, 5, 8]:
    leg.get_texts()[i].set_color("blue")

# Make room for the legend
fig.subplots_adjust(bottom=0.127)
plt.savefig('/work/hdd/bezm/tnguyen2/figures/AGORA/November2025/circularity_distribution_ProgBranch-%s_ver2013_Abadi2003method_ver3.png' % merger_number, dpi=300, bbox_inches='tight')