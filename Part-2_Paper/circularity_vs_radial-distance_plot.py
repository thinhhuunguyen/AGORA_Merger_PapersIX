import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import ImageGrid as AxesGrid

import setup
from importlib import reload
reload(setup)

from setup import load_timings
from setup import codetp_list, label_list


halotree_ver = 2013
merger_number = '0'
fig = plt.figure()

axs = AxesGrid(
    fig,
    (0.1, 0.1, 2, 1),
    nrows_ncols=(2, 5),
    axes_pad=(0.12, 0.45),
    label_mode="1",
    share_all=True,
    aspect=False,  
) 

font_size = 18
epsilon_lim = 1.5
levels_list = [0.1, 0.5, 0.95]

for j in range(len(codetp_list)):
    codetp = codetp_list[j]
    idx_begin, idx_endinfall, idx_1stpass, idx_maxdist, idx_cls, time_begin, time_endinfall, time_1stpass, time_maxdist, time_cls = load_timings(codetp, halotree_ver, merger_number)
    dist_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/dist_relvel_j_ProgBranch-%s_FirstMerger_%s_ver2013.npy' % (merger_number, codetp), allow_pickle=True).tolist()
    time_list = np.loadtxt('/work/hdd/bezm/gtg115x/Halo_Finding/%s/pfs_allsnaps_%s.txt' % (codetp, halotree_ver), dtype=str)[:,2].astype(float)
    if codetp == 'GEAR':
        step = 3
    elif codetp == 'CHANGA':
        step = 2
    else:
        step = 1
    #
    time_eval = time_1stpass + 0.6
    if codetp == 'GEAR':
        idx_eval = np.argmin(abs(time_eval - time_list/1e3)) - (np.argmin(abs(time_eval - time_list/1e3)) % step) - 1
    else:
        idx_eval = np.argmin(abs(time_eval - time_list/1e3)) - (np.argmin(abs(time_eval - time_list/1e3)) % step)
    # Circularity
    ID_old0, epsilon_old, mass_old, dist_old, ID_infall0, epsilon_infall, mass_infall, dist_infall, \
                ID_pass0, epsilon_pass, mass_pass, dist_pass, \
                ID_cls0, epsilon_cls, mass_cls, dist_cls,\
                ID_deposit0, epsilon_deposit, mass_deposit, dist_deposit,\
                ID_gal0, epsilon_gal, mass_gal, dist_gal = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/circularity_decomposed_ProgBranch-%s_Idx_%s_%s_ver2013_Abadi2003method.npy' % (merger_number, idx_eval, codetp), allow_pickle=True).tolist().values()

    #Load the previous dist data because the center is better (center of mass of the stellar core; the new data uses center of baryonic, which is more suitable for doing angular momentum)
    ID_old, dist_old, mass_old, ID_infall, dist_infall, mass_infall, ID_pass, dist_pass, mass_pass, \
    ID_cls, dist_cls, mass_cls, ID_deposit, dist_deposit, mass_deposit, \
    ID_gal, dist_gal, mass_gal = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/radial_mass_distribution_decomposed_ProgBranch-%s_Idx_%s_%s_ver2013.npy' % (merger_number, idx_eval, codetp), allow_pickle=True).tolist().values()

    epsilon_old = epsilon_old[np.intersect1d(ID_old, ID_old0, return_indices=True)[2]]
    epsilon_infall = epsilon_infall[np.intersect1d(ID_infall, ID_infall0, return_indices=True)[2]]
    epsilon_pass = epsilon_pass[np.intersect1d(ID_pass, ID_pass0, return_indices=True)[2]]
    epsilon_cls = epsilon_cls[np.intersect1d(ID_cls, ID_cls0, return_indices=True)[2]]

    dist_old = dist_old[np.intersect1d(ID_old, ID_old0, return_indices=True)[2]]
    dist_infall = dist_infall[np.intersect1d(ID_infall, ID_infall0, return_indices=True)[2]]
    dist_pass = dist_pass[np.intersect1d(ID_pass, ID_pass0, return_indices=True)[2]]
    dist_cls = dist_cls[np.intersect1d(ID_cls, ID_cls0, return_indices=True)[2]]

    mass_old = mass_old[np.intersect1d(ID_old, ID_old0, return_indices=True)[2]]
    mass_infall = mass_infall[np.intersect1d(ID_infall, ID_infall0, return_indices=True)[2]]
    mass_pass = mass_pass[np.intersect1d(ID_pass, ID_pass0, return_indices=True)[2]]
    mass_cls = mass_cls[np.intersect1d(ID_cls, ID_cls0, return_indices=True)[2]]
    #
    dist_lim = 0.2*dist_data['prog_radius'][dist_data['idx'] == idx_eval][0]
    #
    bool_old = (dist_old <= dist_lim)*(abs(epsilon_old) <= epsilon_lim)
    sns.kdeplot(x=dist_old[bool_old], y=epsilon_old[bool_old], weights=mass_old[bool_old], color='black', levels=levels_list, ax=axs[j], zorder=10)
    
    bool_infall = (dist_infall <= dist_lim)*(abs(epsilon_infall) <= epsilon_lim)
    sns.kdeplot(x=dist_infall[bool_infall], y=epsilon_infall[bool_infall], weights=mass_infall[bool_infall], color='royalblue', levels=levels_list, ax=axs[j], zorder=10)
    
    bool_pass = (dist_pass <= dist_lim)*(abs(epsilon_pass) <= epsilon_lim)
    sns.kdeplot(x=dist_pass[bool_pass], y=epsilon_pass[bool_pass], weights=mass_pass[bool_pass], color='orange', levels=levels_list, ax=axs[j], zorder=10)
    
    bool_cls = (dist_cls <= dist_lim)*(abs(epsilon_cls) <= epsilon_lim)
    sns.kdeplot(x=dist_cls[bool_cls], y=epsilon_cls[bool_cls], weights=mass_cls[bool_cls], color='red', levels=levels_list, ax=axs[j], zorder=10)
    #
    axs[j].set_ylim(-epsilon_lim, epsilon_lim)
    axs[j].set_xlim(-0.3, 10)
    #
    axs[j].set_xlabel('d (kpc)', fontsize=font_size)
    axs[j].set_ylabel(r'$\epsilon = j_{z}/j_{c}$', fontsize=font_size)
    axs[j].tick_params('both', labelsize=font_size)
    #axs[j].set_xticks([0,2.5,5,7.5,10], ['0','2.5','5','7.5','10'])
    axs[j].set_xticks([0,3,6,9], ['0','3','6','9'])
    #
    axs[j].axhline(0, linestyle='--', color='grey', zorder=0, alpha=0.5)
    axs[j].axhline(1, linestyle='--', color='grey', zorder=0, alpha=0.5)
    #
    if codetp == 'ART' or codetp == 'GADGET3' or codetp == 'GADGET4' or codetp == 'GIZMO':
        title_color = 'blue'
    else:
        title_color = 'black'
    if codetp == 'ART' or codetp == 'RAMSES' or codetp == 'GADGET3' or codetp == 'GEAR':
        axs[j].set_title(label_list[j] + '*', fontsize=font_size, color=title_color)
    elif codetp == 'CHANGA':
        axs[j].set_title(r'$\text{%s}^{\dagger}$' % label_list[j], fontsize=font_size, color=title_color)
    else:
        axs[j].set_title(label_list[j], fontsize=font_size, color=title_color)
    
axs[9].set_axis_off()
axs[0].plot([],[],color='black', label='old')
axs[0].plot([],[],color='royalblue', label='infall')
axs[0].plot([],[],color='orange', label='first passage')
axs[0].plot([],[],color='red', label='cls + post-cls')
handles1, labels1 = axs[0].get_legend_handles_labels()
fig.legend(handles1, labels1 , bbox_to_anchor=(1.91, 0.14), loc='lower center',ncol=1,fontsize=font_size-1)
fig.tight_layout()

plt.savefig('/work/hdd/bezm/tnguyen2/figures/AGORA/November2025/circularity_vs_distance_ProgBranch-%s_ver2013_Abadi2003method_ver2.png' % merger_number, dpi=300, bbox_inches='tight')