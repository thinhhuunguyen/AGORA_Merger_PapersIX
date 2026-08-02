import numpy as np
import matplotlib.pyplot as plt
import yt
yt.set_log_level(0)
    
import setup
from importlib import reload
reload(setup)

from setup import load_timings
from setup import codetp_list, label_list

halotree_ver = 2013
merger_number = '0'

fig = plt.figure(figsize=(23, 19))

# Two 3x3 blocks side by side. Small internal wspace/hspace shrinks the gaps
# within each block; the gap between left (right=0.48) and right (left=0.55)
# blocks is preserved independently.
gs_left  = fig.add_gridspec(3, 3, left=0.05, right=0.48, top=0.95, bottom=0.12,
                            wspace=0.16, hspace=0.15)
gs_right = fig.add_gridspec(3, 3, left=0.55, right=0.98, top=0.95, bottom=0.12,
                            wspace=0.16, hspace=0.15)

xticks = [0, 2, 4, 6, 8, 10]

# ====================================================================
# LEFT BLOCK (Code 1): total radial mass distribution
# ====================================================================
font_size = 23
for j in range(len(codetp_list)):
    codetp = codetp_list[j]
    if codetp == 'CHANGA':
        step = 2
    elif codetp == 'GEAR':
        step = 3
    else:
        step = 1
    idx_begin, idx_endinfall, idx_1stpass, idx_maxdist, idx_cls, \
    time_begin, time_endinfall, time_1stpass, time_maxdist, time_cls = load_timings(codetp, halotree_ver, merger_number)
    #
    ax = fig.add_subplot(gs_left[j // 3, j % 3])
    dist_gal, mass_gal = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/radial_mass_distribution_ProgBranch-%s_Idx_%s_%s_ver2013.npy' % (merger_number, idx_begin - step, codetp), allow_pickle=True).tolist().values()
    hist_bin_log = np.linspace(0, 10, 100)
    ax.hist(dist_gal, weights=mass_gal, bins=hist_bin_log, color='black')
    if j >= 6:
        ax.set_xlabel('d (kpc)', fontsize=font_size)
    if j == 0 or j == 3 or j == 6:
        ax.set_ylabel(r'$M_{\bigstar} (M_\odot)$', fontsize=font_size)
    ax.tick_params('both', labelsize=font_size)
    ax.yaxis.get_offset_text().set_fontsize(font_size-2)
    if j >= 6:
        ax.set_xticks(xticks)
    else:
        ax.set_xticks(xticks,['','','','','',''])
    ax.set_yscale('log')
    ax.set_ylim(1e5, 4e8)
    if codetp not in ('ART', 'CHANGA', 'GEAR'):
        ax.tick_params(labelleft=False)
    if codetp in ('ART', 'GADGET3', 'GADGET4', 'GIZMO'):
        title_color = 'blue'
    else:
        title_color = 'black'
    if codetp in ('ART', 'RAMSES', 'GADGET3', 'GEAR'):
        ax.set_title(label_list[j] + '*', fontsize=font_size, color=title_color)
    elif codetp == 'CHANGA':
        ax.set_title(r'$\text{%s}^{\dagger}$' % label_list[j], fontsize=font_size, color=title_color)
    else:
        ax.set_title(label_list[j], fontsize=font_size, color=title_color)

# ====================================================================
# RIGHT BLOCK (Code 2): decomposed radial mass distribution
# ====================================================================
font_size = 22
leg_handles, leg_labels = [], []
for j in range(len(codetp_list)):
    codetp = codetp_list[j]
    idx_begin, idx_endinfall, idx_1stpass, idx_maxdist, idx_cls, \
    time_begin, time_endinfall, time_1stpass, time_maxdist, time_cls = load_timings(codetp, halotree_ver, merger_number)
    time_list = np.loadtxt('/work/hdd/bezm/gtg115x/Halo_Finding/%s/pfs_allsnaps_%s.txt' % (codetp, halotree_ver), dtype=str)[:, 2].astype(float)
    if codetp == 'GEAR':
        step = 3
    elif codetp == 'CHANGA':
        step = 2
    else:
        step = 1
    #
    ax = fig.add_subplot(gs_right[j // 3, j % 3])
    #
    time_eval = time_1stpass + 0.6
    if codetp == 'GEAR':
        idx_eval = np.argmin(abs(time_eval - time_list/1e3)) - (np.argmin(abs(time_eval - time_list/1e3)) % step) - 1
    else:
        idx_eval = np.argmin(abs(time_eval - time_list/1e3)) - (np.argmin(abs(time_eval - time_list/1e3)) % step)
    #
    ID_old, dist_old, mass_old, ID_infall, dist_infall, mass_infall, ID_pass, dist_pass, mass_pass, \
    ID_cls, dist_cls, mass_cls, ID_deposit, dist_deposit, mass_deposit, \
    ID_gal, dist_gal, mass_gal = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/radial_mass_distribution_decomposed_ProgBranch-%s_Idx_%s_%s_ver2013.npy' % (merger_number, idx_eval, codetp), allow_pickle=True).tolist().values()

    hist_bin_log = np.linspace(0, 10, 100)
    #
    infall_hist  = np.histogram(dist_infall,  weights=mass_infall,  bins=hist_bin_log)
    pass_hist    = np.histogram(dist_pass,    weights=mass_pass,     bins=hist_bin_log)
    cls_hist     = np.histogram(dist_cls,     weights=mass_cls,      bins=hist_bin_log)
    deposit_hist = np.histogram(dist_deposit, weights=mass_deposit,  bins=hist_bin_log)
    old_hist     = np.histogram(dist_old,     weights=mass_old,      bins=hist_bin_log)

    ax.bar(hist_bin_log[:-1], old_hist[0], width=np.diff(hist_bin_log), color='black', align='edge', label='old + accretion')
    ax.bar(hist_bin_log[:-1], deposit_hist[0], bottom=old_hist[0], width=np.diff(hist_bin_log), color='limegreen', align='edge', label='deposit')
    ax.bar(hist_bin_log[:-1], infall_hist[0], bottom=old_hist[0] + deposit_hist[0], width=np.diff(hist_bin_log), color='royalblue', align='edge', label='infall')
    ax.bar(hist_bin_log[:-1], pass_hist[0], bottom=old_hist[0] + deposit_hist[0] + infall_hist[0], width=np.diff(hist_bin_log), color='orange', align='edge', label='1st passage')
    ax.bar(hist_bin_log[:-1], cls_hist[0], bottom=old_hist[0] + deposit_hist[0] + infall_hist[0] + pass_hist[0], width=np.diff(hist_bin_log), color='red', align='edge', label='coalescence')

    # ----- inset percentage bars -----
    left, bottom, width, height = [0.45, 0.85, 0.5, 0.1]
    ax_cls = ax.inset_axes([left, bottom, width, height])
    ax_cls.get_xaxis().set_visible(False)
    ax_cls.get_yaxis().set_visible(False)
    ax_cls.bar(hist_bin_log[:-1], cls_hist[0], width=np.diff(hist_bin_log), color='red', align='edge', label='cls + post-cls')
    ax_cls.text(0.65, 0.6, s='%.0f%%' % ((mass_cls.sum()/mass_gal.sum())*100), transform=ax_cls.transAxes, fontsize=font_size - 8)

    left, bottom, width, height = [0.45, 0.75, 0.5, 0.1]
    ax_pass = ax.inset_axes([left, bottom, width, height])
    ax_pass.get_xaxis().set_visible(False)
    ax_pass.get_yaxis().set_visible(False)
    ax_pass.bar(hist_bin_log[:-1], pass_hist[0], width=np.diff(hist_bin_log), color='orange', align='edge', label='first passage')
    ax_pass.text(0.65, 0.6, s='%.0f%%' % ((mass_pass.sum()/mass_gal.sum())*100), transform=ax_pass.transAxes, fontsize=font_size - 8)

    left, bottom, width, height = [0.45, 0.65, 0.5, 0.1]
    ax_infall = ax.inset_axes([left, bottom, width, height])
    ax_infall.get_xaxis().set_visible(False)
    ax_infall.get_yaxis().set_visible(False)
    ax_infall.bar(hist_bin_log[:-1], infall_hist[0], width=np.diff(hist_bin_log), color='royalblue', align='edge', label='infall')
    if codetp == 'GIZMO':
        ax_infall.text(0.65, 0.6, s='%.1f%%' % ((mass_infall.sum()/mass_gal.sum())*100), transform=ax_infall.transAxes, fontsize=font_size - 8)
    else:
        ax_infall.text(0.65, 0.6, s='%.0f%%' % ((mass_infall.sum()/mass_gal.sum())*100), transform=ax_infall.transAxes, fontsize=font_size - 8)

    left, bottom, width, height = [0.45, 0.55, 0.5, 0.1]
    ax_deposit = ax.inset_axes([left, bottom, width, height])
    ax_deposit.get_xaxis().set_visible(False)
    ax_deposit.get_yaxis().set_visible(False)
    ax_deposit.bar(hist_bin_log[:-1], deposit_hist[0], width=np.diff(hist_bin_log), color='limegreen', align='edge', label='deposit')
    ax_deposit.text(0.65, 0.6, s='%.0f%%' % ((mass_deposit.sum()/mass_gal.sum())*100), transform=ax_deposit.transAxes, fontsize=font_size - 8)

    left, bottom, width, height = [0.45, 0.45, 0.5, 0.1]
    ax_old = ax.inset_axes([left, bottom, width, height])
    ax_old.get_yaxis().set_visible(False)
    ax_old.bar(hist_bin_log[:-1], old_hist[0], width=np.diff(hist_bin_log), color='black', align='edge', label='old')
    if codetp == 'GIZMO':
        ax_old.text(0.65, 0.6, s='%.1f%%' % ((mass_old.sum()/mass_gal.sum())*100), transform=ax_old.transAxes, fontsize=font_size - 8)
    else:
        ax_old.text(0.65, 0.6, s='%.0f%%' % ((mass_old.sum()/mass_gal.sum())*100), transform=ax_old.transAxes, fontsize=font_size - 8)
    ax_old.set_xticks(xticks, ['0', '', '', '', '', '10'])
    ax_old.tick_params('x', labelsize=font_size - 8)
    #
    if j >= 6:
        ax.set_xlabel('d (kpc)', fontsize=font_size)
    if j == 0 or j == 3 or j == 6:
        ax.set_ylabel(r'$M_{\bigstar} (M_\odot)$', fontsize=font_size)
    ax.tick_params('both', labelsize=font_size)
    ax.yaxis.get_offset_text().set_fontsize(font_size-2)
    if j >= 6:
        ax.set_xticks(xticks)
    else:
        ax.set_xticks(xticks, ['', '', '', '', '', ''])
    ax.set_yscale('log')
    ax.set_ylim(1e6, 3.5e9)
    if codetp not in ('ART', 'CHANGA', 'GEAR'):
        ax.tick_params(labelleft=False)
    if codetp in ('ART', 'GADGET3', 'GADGET4', 'GIZMO'):
        title_color = 'blue'
    else:
        title_color = 'black'
    if codetp in ('ART', 'RAMSES', 'GADGET3', 'GEAR'):
        ax.set_title(label_list[j] + '*', fontsize=font_size, color=title_color)
    elif codetp == 'CHANGA':
        ax.set_title(r'$\text{%s}^{\dagger}$' % label_list[j], fontsize=font_size, color=title_color)
    else:
        ax.set_title(label_list[j], fontsize=font_size, color=title_color)
    #
    if j == len(codetp_list) - 1:
        h1, l1 = ax_old.get_legend_handles_labels()
        h2, l2 = ax_infall.get_legend_handles_labels()
        h3, l3 = ax_pass.get_legend_handles_labels()
        h4, l4 = ax_cls.get_legend_handles_labels()
        h5, l5 = ax_deposit.get_legend_handles_labels()
        leg_handles = h1 + h2 + h3 + h4 + h5
        leg_labels  = l1 + l2 + l3 + l4 + l5

# Legend centered under the RIGHT block (block spans x = 0.55-0.98, center ~0.765)
fig.legend(leg_handles, leg_labels, loc='upper center',
           bbox_to_anchor=(0.765, 0.07), ncol=5, fontsize=18)

fig.text(0.265, 0.98, r'Before the merger ($t_\text{pre-infall}$)', ha='center', va='bottom', fontsize=24)
fig.text(0.765, 0.98, r'After the merger ($t_\text{post/eq}$)', ha='center', va='bottom', fontsize=24)

fig.text(0.265, 0.015, '(a)', ha='center', va='bottom', fontsize=24)
fig.text(0.765, 0.015, '(b)', ha='center', va='bottom', fontsize=24)

plt.savefig('/work/hdd/bezm/tnguyen2/figures/AGORA/November2025/mass_radialdistribution_CombinedFirstMerger_ProgBranch-%s_ver2013_logscale_ver2.png' % merger_number, dpi=300, bbox_inches='tight')