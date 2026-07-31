import numpy as np
import matplotlib.pyplot as plt
import yt
yt.set_log_level(0)

import setup
from importlib import reload
reload(setup)

from setup import load_halotree_and_pfs, load_timings, sec_branch_compute
from setup import codetp_list, label_list, color_list

font_size = 12

fig, grid = plt.subplots(ncols=1, nrows=4, figsize=(6,13))
fig.subplots_adjust(wspace=0.2, hspace=0.1)
grid = grid.flatten()

halotree_ver = 2013
merger_number = '0'

time_begin_min = np.inf
time_begin_max = -np.inf

for k in range(len(codetp_list)):
    codetp = codetp_list[k]
    #
    redshift_list, time_list, pfs, step = load_halotree_and_pfs(codetp, halotree_ver, rawtree_skip=True)
    metadata_dir = '/work/hdd/bezm/tnguyen2/AGORA/%s/metadata/' % codetp
    #
    dist_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/dist_relvel_j_ProgBranch-%s_FirstMerger_%s_ver2013.npy' % (merger_number, codetp), allow_pickle=True).tolist()
    prog_branch, sec_branch, sec_branch_2 = sec_branch_compute(codetp, merger_number)
    idx_begin, idx_endinfall, idx_1stpass, idx_maxdist, idx_cls, time_begin, time_endinfall, time_1stpass, time_maxdist, time_cls = load_timings(codetp, halotree_ver, merger_number)
    time_eval = time_1stpass + 0.6
    if codetp == 'GEAR':
        idx_eval = np.argmin(abs(time_eval - time_list/1e3)) - (np.argmin(abs(time_eval - time_list/1e3)) % step) - 1
    else:
        idx_eval = np.argmin(abs(time_eval - time_list/1e3)) - (np.argmin(abs(time_eval - time_list/1e3)) % step)
    #
    time_lim = 3
    idx_lim = np.argmin(abs(time_lim - time_list/1e3))    
    time_begin_min = min(time_begin_min, time_begin)
    time_begin_max = max(time_begin_max, time_begin)
    #
    gas_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/GasZ_ProgBranch-0_%s_ver2013_ConvexHull_preInfall_ver2.npy' % codetp, allow_pickle=True).tolist()
    gas_Z_hull = gas_data['gas_Z_hull']
    gas_Z_galaxy = gas_data['gas_Z_galaxy']
    gas_massZ_hull = gas_data['gas_massZ_hull']
    gas_massZ_galaxy = gas_data['gas_massZ_galaxy']
    time_plot = np.array(gas_data['time'])
    
    gas_data_tcool = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/GasCoolingTime_ProgBranch-0_%s_ver2013_ConvexHull_preInfall_ver2.npy' % codetp, allow_pickle=True).tolist()
    gas_mass_hull = gas_data_tcool['gas_mass_hull']
    gas_tcool_hull = gas_data_tcool['gas_tcool_hull']    
    gas_tff_hull = gas_data_tcool['gas_tff_hull']
    gas_dist_hull = gas_data_tcool['gas_dist_hull']
    gas_radialvel_hull = gas_data_tcool['gas_radialvel_hull']
    time_plot_tcool = np.array(gas_data_tcool['time'])
    gas_mass_total = np.array([])
    for m in range(len(time_plot_tcool)):
        gas_mass_total = np.append(gas_mass_total, gas_mass_hull[m].sum())
    
    #Plotting
    grid[0].plot(time_plot_tcool/1e3, gas_mass_total, color=color_list[k])
    grid[0].set_ylabel(r'$M_\text{gas}$ $(M_\odot)$', fontsize=font_size)
    grid[0].set_xlim(0.6, 1.17)
    grid[0].set_ylim(1.5e9, 1.15e10)
    grid[0].tick_params('both', labelsize=font_size)
    grid[0].set_xticklabels([])
    
    if codetp == 'GIZMO':
        grid[1].plot(time_plot[:-1]/1e3, gas_Z_hull[:-1], color=color_list[k], label=label_list[k])
    else:
        grid[1].plot(time_plot/1e3, gas_Z_hull, color=color_list[k], label=label_list[k])
    grid[1].set_ylabel(r'$Z$ $(Z_\odot)$', fontsize=font_size)
    grid[1].set_yscale('log')
    grid[1].set_xlim(0.6, 1.17)
    grid[1].set_ylim(5e-3, 2e-1)
    grid[1].tick_params('both', labelsize=font_size)
    grid[1].set_xticklabels([])
    #
    gas_mass_plot = []
    gas_tcool_plot = []
    for j in range(len(gas_tcool_hull)):
        reduced = (gas_radialvel_hull[j] < 0)*(np.maximum(gas_tcool_hull[j], gas_tff_hull[j]) < 0.1)
        gas_mass_plot.append(np.array(gas_mass_hull[j])[reduced].sum())
        gas_tcool_plot.append(np.average(np.maximum(gas_tcool_hull[j], gas_tff_hull[j])[reduced], weights=np.array(gas_mass_hull[j])[reduced]))

    if codetp == 'GIZMO': #there is an issue with the hull of GIZMO is shifted too much at the beginning of the major merger
        grid[3].plot(time_plot_tcool[:-1]/1e3, gas_tcool_plot[:-1], linestyle='-', color=color_list[k], label=label_list[k])
    else:
        grid[3].plot(time_plot_tcool/1e3, gas_tcool_plot, linestyle='-', color=color_list[k], label=label_list[k])
    grid[3].set_ylabel(r'$\bar{t}_\text{cl} [v_r < 0, t_\text{cl} < 0.1\,\text{Gyr}]$ (Gyr)', fontsize=font_size)
    grid[3].set_xlabel(r'Time since Big Bang (Gyr)', fontsize=font_size)
    grid[3].set_ylim(2e-2, 0.11)
    grid[3].set_xlim(0.6, 1.17)
    grid[3].tick_params('both', labelsize=font_size)

    if codetp == 'GIZMO':
        grid[2].plot(time_plot_tcool[:-1]/1e3, gas_mass_plot[:-1], linestyle='-', color=color_list[k], label=label_list[k])
    else:
        grid[2].plot(time_plot_tcool/1e3, gas_mass_plot, linestyle='-', color=color_list[k], label=label_list[k])
    grid[2].set_ylabel(r'$M_\text{gas} [v_r < 0, t_\text{cl} < 0.1\,\text{Gyr}]$ $(M_\odot)$', fontsize=font_size)
    grid[2].set_yscale('log')
    grid[2].set_ylim(1e8, 5e9)
    grid[2].set_xlim(0.6, 1.17)
    grid[2].set_xticklabels([])
    grid[2].tick_params('both', labelsize=font_size)

grid[0].fill_between (x = np.linspace(time_begin_min , time_begin_max), y1 = -1e99, y2 = 1e99, alpha=0.09, color='grey')
grid[1].fill_between (x = np.linspace(time_begin_min , time_begin_max), y1 = -1e99, y2 = 1e99, alpha=0.09, color='grey')
grid[2].fill_between (x = np.linspace(time_begin_min , time_begin_max), y1 = -1e99, y2 = 1e99, alpha=0.09, color='grey')
grid[3].fill_between (x = np.linspace(time_begin_min , time_begin_max), y1 = -1e99, y2 = 1e99, alpha=0.09, color='grey')

grid[2].text(x=0.18, y=0.1, s=r'$t_\text{cl} = \max(t_\text{cool}, t_\text{ff})$', transform=grid[2].transAxes, fontsize=font_size)
grid[3].legend(ncols=3)

#Adding redshift axis
time_values = [0.6628458623299167, 0.7861791877466925, 0.9601452183348662, 1.076]
specific_redshifts = [8, 7, 6, 5.5]
ax_z = grid[0].twiny()
ax_z.set_xlim([0.6, 1.17])
ax_z.set_xlabel('z',fontsize=font_size)
ax_z.xaxis.label.set_color('black') 
# Set the redshift labels based on the calculated values
ax_z.set_xticks(time_values)
ax_z.set_xticklabels(['{:.1f}'.format(z) for z in specific_redshifts])
ax_z.tick_params(axis='x',labelcolor='black',labelsize=font_size)

plt.savefig('/work/hdd/bezm/tnguyen2/figures/AGORA/November2025/GasMetallicity_and_CoolingTime_PreMerger_ProgBranch-%s_ver2013_PartApaper_ver2.png' % merger_number, dpi=300, bbox_inches='tight')