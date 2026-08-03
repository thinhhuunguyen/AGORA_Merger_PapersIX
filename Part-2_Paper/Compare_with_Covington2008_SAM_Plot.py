import numpy as np
import matplotlib.pyplot as plt
import yt
yt.set_log_level(0)
from astropy.constants import G
G_u = G.to('kpc**3/(Msun*Gyr**2)').value

import setup
from importlib import reload
reload(setup)

from setup import load_halotree_and_pfs, load_timings
from setup import codetp_list, label_list, color_list, marker_list


merger_number = '0'
halotree_ver = 2013
font_size = 18

fig, axs = plt.subplots(figsize=(18,4.5),nrows=1, ncols=4)

for j in range(len(codetp_list)):
    codetp = codetp_list[j]
    idx_begin, idx_endinfall, idx_1stpass, idx_maxdist, idx_cls, time_begin, time_endinfall, time_1stpass, time_maxdist, time_cls = load_timings(codetp, halotree_ver, merger_number)
    redshift_list, time_list, pfs, step = load_halotree_and_pfs(codetp, halotree_ver, rawtree_skip=True)
    time_endloop = time_cls + 0.3
    idx_endloop = np.argmin(abs(time_endloop - time_list/1e3))
    data = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/Comparison_CovingtonEtal2008_ProgBranch-%s_%s_ver2013_ver7.npy' % (merger_number, codetp), allow_pickle=True).tolist()
    # Fine tuning Crad value to match the predicted Rf with the true Rf
    rf_true = data['rf_sim'][data['idx']==idx_cls][0]
    rf_predicted = data['rf']
    error_list = (rf_predicted - rf_true)**2
    chosen_predicted_index = int(np.nanargmin(error_list))
    #
    axs[1].scatter(data['rf_sim'][data['idx']==idx_cls], data['rf'][int(chosen_predicted_index)], color=color_list[j], marker=marker_list[j], label=r'%s: $C_{rad}$ = %.2f' % (label_list[j], data['Crad_list'][int(chosen_predicted_index)]), s=100)
    #
    axs[0].scatter(data['mf_sim'][data['idx']==idx_cls], data['mf'], color=color_list[j], marker=marker_list[j], s=100, label=codetp)
    #
    axs[2].scatter(data['fdm_f_sim'][data['idx']==idx_cls], data['fdm_f'], color=color_list[j], marker=marker_list[j], s=100, label=codetp)
    #
    axs[3].scatter(data['sigma_f_sim'][data['idx']==idx_cls], data['sigma_f'][int(chosen_predicted_index)], color=color_list[j], marker=marker_list[j], label=codetp, s=100)

axs[1].set_xlim(-0.1, 3.4)
axs[1].set_ylim(-0.1, 3.4)
axs[1].axline([1,1], [2,2], color='black', linestyle='--')
axs[1].set_xlabel(r'True $R_{f}$ (kpc)', fontsize=font_size)
axs[1].set_ylabel(r'Predicted $R_{f}$ (kpc)', fontsize=font_size)
axs[1].tick_params('both', labelsize=font_size)

axs[0].set_xlim(1.2e9, 1e10)
axs[0].set_ylim(1.2e9, 1e10)
axs[0].axline([1,1], [2,2], color='black', linestyle='--')
axs[0].set_xlabel(r'True $M_{f}$ $(M_\odot)$', fontsize=font_size)
axs[0].set_ylabel(r'Predicted $M_{f}$ $(M_\odot)$', fontsize=font_size)
axs[0].tick_params('both', labelsize=font_size, which='both')
axs[0].set_xticks([2e9, 4e9, 6e9, 8e9, 1e10])
axs[0].set_yticks([2e9, 4e9, 6e9, 8e9, 1e10])

axs[3].set_xlim(50, 400)
axs[3].set_ylim(50, 400)
axs[3].axline([1,1], [2,2], color='black', linestyle='--')
axs[3].tick_params('both', labelsize=font_size)
axs[3].set_xlabel(r'True $\sigma_{f}$ (kpc/Gyr)', fontsize=font_size)
axs[3].set_ylabel(r'Predicted $\sigma_{f}$ (kpc/Gyr)', fontsize=font_size)
axs[3].tick_params('both', labelsize=font_size)
axs[3].set_yticks([100,200,300, 400])
axs[3].set_xticks([100,200,300, 400])

axs[2].set_xlim(0, 1.05)
axs[2].set_ylim(0, 1.05)
axs[2].axline([1,1], [2,2], color='black', linestyle='--')
axs[2].tick_params('both', labelsize=font_size)
axs[2].set_xlabel(r'True $f_{dm,f}$ (kpc)', fontsize=font_size)
axs[2].set_ylabel(r'Predicted $f_{dm,f}$ (kpc)', fontsize=font_size)
axs[2].tick_params('both', labelsize=font_size)
axs[2].set_xticks([0,0.2,0.4,0.6,0.8,1.0])
axs[2].set_yticks([0,0.2,0.4,0.6,0.8,1.0])

handles1, labels1 = axs[1].get_legend_handles_labels()
fig.legend(handles1, labels1 , bbox_to_anchor=(0.5, -0.19), loc='lower center',ncol=5,fontsize=15)


fig.tight_layout()
plt.savefig('/work/hdd/bezm/tnguyen2/figures/AGORA/November2025/Compare_with_Covington2008_SAM_ver3.png', dpi=300, bbox_inches='tight')