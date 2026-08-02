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
fig = plt.figure(figsize=(20, 10))

# Create a 10x2 GridSpec
gs = GridSpec(
    nrows=2,
    ncols=10,
    figure=fig,
    height_ratios=[1, 1],
    hspace=0.33,
    wspace=0.5
)

# -------------------------
# Left column (2 equal panels)
# -------------------------
ax_left_top = fig.add_subplot(gs[0, 0:5])
ax_right_top = fig.add_subplot(gs[0, 5:10])

# -------------------------
# Right column (5 stacked panels)
# Each spans 2 rows
# -------------------------
ax_bot = []
for i in range(5):
    ax = fig.add_subplot(gs[1, i*2:(i+1)*2])
    ax_bot.append(ax)

ax_bot0, ax_bot1, ax_bot2, ax_bot3, ax_bot4 = ax_bot[0], ax_bot[1], ax_bot[2], ax_bot[3], ax_bot[4]

halotree_ver = 2013
merger_number = '0'
font_size = 18
epsilon_lim = 1.5


xlim = (-epsilon_lim, epsilon_lim)
ylim = (-0.02, 1.62)
hist_bin_log = np.linspace(-epsilon_lim,epsilon_lim,100)


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
    # Old stars
    kde = gaussian_kde(epsilon_old, weights=mass_old, bw_method='scott')
    x = np.linspace(-epsilon_lim, epsilon_lim, 1000)
    kde_values = kde(x)
    ax_bot0.plot(x, kde_values, linewidth=2, label=label_list[k], color=color_list[k], zorder=10)
    ax_bot0.set_xlabel(r'$\epsilon = J_{z}/J_\text{circ}(E)$', fontsize = font_size)
    ax_bot0.set_ylabel(r'$F(\epsilon)$', fontsize = font_size)
    ax_bot0.axvline(0, linestyle='--', color='k', zorder=1, alpha=0.3)
    ax_bot0.axvline(1, linestyle='--', color='k', zorder=1, alpha=0.3)
    ax_bot0.tick_params('both', labelsize = font_size)
    ax_bot0.set_title('Old', fontsize = font_size)
    ax_bot0.set_xlim(xlim)
    ax_bot0.set_ylim(ylim)
    #
    # Infall stage stars
    kde = gaussian_kde(epsilon_infall, weights=mass_infall, bw_method='scott')
    x = np.linspace(-epsilon_lim, epsilon_lim, 1000)
    kde_values = kde(x)
    ax_bot1.plot(x, kde_values, linewidth=2, label=label_list[k], color=color_list[k], zorder=10)
    ax_bot1.axvline(0, linestyle='--', color='k', zorder=1, alpha=0.3)
    ax_bot1.axvline(1, linestyle='--', color='k', zorder=1, alpha=0.3)
    ax_bot1.tick_params('both', labelsize = font_size)
    ax_bot1.set_title('Infall', fontsize = font_size)
    ax_bot1.set_xlim(xlim)
    ax_bot1.set_ylim(ylim)
    ax_bot1.set_yticklabels([])
    #
    # First-passage stars
    kde = gaussian_kde(epsilon_pass, weights=mass_pass, bw_method='scott')
    x = np.linspace(-epsilon_lim, epsilon_lim, 1000)
    kde_values = kde(x)
    ax_bot2.plot(x, kde_values, linewidth=2, label=label_list[k], color=color_list[k], zorder=10)
    ax_bot2.axvline(0, linestyle='--', color='k', zorder=1, alpha=0.3)
    ax_bot2.axvline(1, linestyle='--', color='k', zorder=1, alpha=0.3)
    ax_bot2.tick_params('both', labelsize = font_size)
    ax_bot2.set_title('First passage', fontsize = font_size)
    ax_bot2.set_xlim(xlim)
    ax_bot2.set_ylim(ylim)
    ax_bot2.set_yticklabels([])
    #
    # Cls stars
    kde = gaussian_kde(epsilon_cls, weights=mass_cls, bw_method='scott')
    x = np.linspace(-epsilon_lim, epsilon_lim, 1000)
    kde_values = kde(x)
    ax_bot3.plot(x, kde_values, linewidth=2, label=label_list[k], color=color_list[k], zorder=10)
    ax_bot3.axvline(0, linestyle='--', color='k', zorder=1, alpha=0.3)
    ax_bot3.axvline(1, linestyle='--', color='k', zorder=1, alpha=0.3)
    ax_bot3.tick_params('both', labelsize = font_size)
    ax_bot3.set_title('Cls + Post-cls', fontsize = font_size)
    ax_bot3.set_xlim(xlim)
    ax_bot3.set_ylim(ylim)
    ax_bot3.set_yticklabels([])
    #
    # Deposit stars
    kde = gaussian_kde(epsilon_deposit, weights=mass_deposit, bw_method='scott')
    x = np.linspace(-epsilon_lim, epsilon_lim, 1000)
    kde_values = kde(x)
    ax_bot4.plot(x, kde_values, linewidth=2, label=label_list[k], color=color_list[k], zorder=10)
    ax_bot4.axvline(0, linestyle='--', color='k', zorder=1, alpha=0.3)
    ax_bot4.axvline(1, linestyle='--', color='k', zorder=1, alpha=0.3)
    ax_bot4.tick_params('both', labelsize = font_size)
    ax_bot4.set_title('Deposit', fontsize = font_size)
    ax_bot4.set_xlim(xlim)
    ax_bot4.set_ylim(ylim)
    ax_bot4.set_yticklabels([])

    
handles, labels = ax_left_top.get_legend_handles_labels()

leg = fig.legend(
    handles,
    labels,
    loc='lower center',
    ncol=len(labels),          # one column per code
    fontsize=15,
    frameon=True,
    bbox_to_anchor=(0.5, -0.018)
)

for i in [0, 4, 5, 8]:
    leg.get_texts()[i].set_color("blue")

# Make room for the legend
fig.subplots_adjust(bottom=0.2)
plt.savefig('/work/hdd/bezm/tnguyen2/figures/AGORA/November2025/circularity_distribution_ProgBranch-%s_ver2013_Abadi2003method_ver2.png' % merger_number, dpi=300, bbox_inches='tight')