import numpy as np
import matplotlib.pyplot as plt
import yt
yt.set_log_level(0)

import setup
from importlib import reload
reload(setup)

from setup import load_halotree_and_pfs, load_timings, load_ds, load_tracking_dist_data, sec_branch_compute
from setup import codetp_list, label_list, color_list


merger_number = '0'

font_size = 18
halotree_ver = 2013 

property_list = [r'$M_\bigstar$ $(M_\odot)$',r'SFR $(M_\odot/\text{yr})$', r'$M_\text{gas}$ $(M_\odot)$', r'$M_\text{gas-total}/M_\text{all}$']

max_stellarmass_begin = -np.inf
min_stellarmass_begin = np.inf
max_stellarmass_end = -np.inf
min_stellarmass_end = np.inf

fig, axs = plt.subplots(ncols=len(codetp_list), nrows=len(property_list), figsize=(20,3*len(property_list)))
plt.subplots_adjust(wspace=0.1)

for i in range(len(codetp_list)):
    #
    codetp = codetp_list[i]
    #
    redshift_list = np.loadtxt('/work/hdd/bezm/gtg115x/Halo_Finding/%s/pfs_allsnaps_%s.txt' % (codetp, halotree_ver), dtype=str)[:,1].astype(float)
    time_list = np.loadtxt('/work/hdd/bezm/gtg115x/Halo_Finding/%s/pfs_allsnaps_%s.txt' % (codetp, halotree_ver), dtype=str)[:,2].astype(float)
    pfs = np.loadtxt('/work/hdd/bezm/gtg115x/Halo_Finding/%s/pfs_allsnaps_%s.txt' % (codetp, halotree_ver), dtype=str)[:,0]
    dist_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/dist_relvel_j_ProgBranch-%s_FirstMerger_%s_ver2013.npy' % (merger_number, codetp), allow_pickle=True).tolist()
    prog_branch, sec_branch, sec_branch_2 = sec_branch_compute(codetp, merger_number)
    idx_begin, idx_endinfall, idx_1stpass, idx_maxdist, idx_cls, time_begin, time_endinfall, time_1stpass, time_maxdist, time_cls = load_timings(codetp, halotree_ver, merger_number)
    #
    time_eval = time_1stpass + 0.6
    #
    output = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/SF_data_for_Merger_%s_ver2013_ConvexHull.npy' % codetp, allow_pickle=True).tolist()
    #
    stellar_mass = np.array(output['stellar_mass'])
    sfr = np.array(output['sfr'])
    ssfr = np.array(output['ssfr'])
    time_plot = np.array(output['time'])
    
    gas_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/GasMassFraction_ProgBranch-0_%s_ver2013_ConvexHull_ver3.npy' % codetp, allow_pickle=True).tolist()
    gas_mass_total = gas_data['gas_mass_total']
    gas_mass_sf = gas_data['gas_mass_sf']
    gas_mass_fraction = gas_data['gas_mass_fraction']
    gas_mass_sf_fraction = gas_data['gas_mass_sf_fraction']
    time_plot_gas = np.array(gas_data['time'])/1e3
    
    time_anchor = time_plot_gas[0]

    max_stellarmass_begin = max(max_stellarmass_begin, stellar_mass[np.argmin(abs(time_plot - time_anchor - 0))])
    min_stellarmass_begin = min(min_stellarmass_begin, stellar_mass[np.argmin(abs(time_plot - time_anchor - 0))])
    max_stellarmass_end = max(max_stellarmass_end, stellar_mass[np.argmin(abs(time_plot - time_anchor - 1))])
    min_stellarmass_end = min(min_stellarmass_end, stellar_mass[np.argmin(abs(time_plot - time_anchor - 1))])
    
    for k in range(len(property_list)):
        ax = axs[k,i]
        
        #
        properties = [stellar_mass, sfr, [gas_mass_total, gas_mass_sf], gas_mass_fraction]
        #
        if k == 2:
            ax.plot(time_plot_gas - time_anchor, properties[k][0],'-',color=color_list[i], lw=3)
            ax.plot(time_plot_gas - time_anchor, properties[k][1],':',color=color_list[i], lw=3)
            if i == 1:
                ax.plot([], [],'-',color='black', lw=2, label=r'total')
                ax.plot([], [],':',color='black', lw=2, label=r'star-forming')
                ax.legend(ncols=1, loc='lower right')
        elif k == 3:
            ax.plot(time_plot_gas - time_anchor, properties[k],'-',color=color_list[i], lw=3)
        else:
            ax.plot(time_plot - time_anchor, properties[k],'-',color=color_list[i], lw=3)
        
        ax.fill_between (x = np.linspace((time_begin + time_maxdist)/2 , time_begin) - time_anchor, y1 = -1e99, y2 = 1e99, alpha=0.09, color='cyan')
        ax.fill_between (x = np.linspace(time_maxdist, (time_begin + time_maxdist)/2 ) - time_anchor, y1 = -1e99, y2 = 1e99, alpha=0.09, color='orange')
        ax.fill_between (x = np.linspace(time_cls, time_maxdist) - time_anchor, y1 = -1e99, y2 = 1e99, alpha=0.09, color='red')
        ax.fill_between (x = np.linspace(5, time_cls) - time_anchor, y1 = -1e99, y2 = 1e99, alpha=0.09, color='darkred')
        
        if k == 3:
            ax_twin = ax.twinx()
            ax_twin.plot(time_plot_gas - time_anchor, np.array(gas_mass_sf)/np.array(gas_mass_total),'--',color=color_list[i], lw=3)
            #
            if i != 8:
                ax_twin.set_yticklabels([])
            ax_twin.set_ylim(0.1,0.8)
            ax_twin.tick_params('both', labelsize=font_size)
            if i == 8:
                ax_twin.set_ylabel(r'$M_\text{gas-sf}/M_\text{gas-total}$', fontsize=font_size)
        
        if k == 0:
            ax.set_yscale('log')
        if i == 0:
            ax.set_ylabel(property_list[k], fontsize=font_size)

        if merger_number == '0':
            if k == 0:
                ax.set_ylim(9e7, 1.1e10)
            elif k == 1:
                ax.set_ylim(-3, 40)
            elif k == 2:
                ax.set_ylim(4e8, 6e10)
                ax.set_yscale('log')
            elif k == 3:
                ax.set_ylim(0.05, 0.22)
        elif merger_number == '1':
            if k == 0:
                ax.set_ylim(5e6, 6e8)
        #
        ax.set_xlim(0, 1)
        ax.tick_params('both', labelsize=font_size)
        if i != 0:
            ax.set_yticklabels([])
        #
        if k == len(property_list) - 1:
            if i == 0:
                ax.set_xticks([0, 0.5, 1])
                ax.set_xlabel(r'Time elapsed since $t_\text{start}$ (Gyr)', fontsize=font_size)
                ax.xaxis.set_label_coords(0.8, -0.2)
            else:
                ax.set_xticks([0, 0.5, 1], ['','',''])
        else:
            ax.set_xticks([0, 0.5, 1], ['','',''])
        #
        if k == 0:
            if codetp == 'ART' or codetp == 'GADGET3' or codetp == 'GADGET4' or codetp == 'GIZMO':
                title_color = 'blue'
            else:
                title_color = 'black'
            if codetp == 'ART' or codetp == 'RAMSES' or codetp == 'GADGET3' or codetp == 'GEAR':
                ax.set_title(label_list[i] + '*', fontsize=font_size, color=title_color)
            elif codetp == 'CHANGA':
                ax.set_title(r'$\text{%s}^{\dagger}$' % label_list[i], fontsize=font_size, color=title_color)
            else:
                ax.set_title(label_list[i], fontsize=font_size, color=title_color)


plt.savefig('/work/hdd/bezm/tnguyen2/figures/AGORA/November2025/All_Properties_Evolution_ProgBranch-%s_ver2013_ver11.png' % merger_number, dpi=300, bbox_inches='tight')
#print('Max and min of stellar mass in the beginning (log10):', np.log10(max_stellarmass_begin), np.log10(min_stellarmass_begin))
#print('Max and min of stellar mass 1 Gyr after the beginning (log10):',np.log10(max_stellarmass_end), np.log10(min_stellarmass_end))