import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import yt
yt.set_log_level(0)
import os
import scipy
    
import setup
from importlib import reload
reload(setup)

from setup import load_halotree_and_pfs, load_timings, sec_branch_compute
from setup import codetp_list, label_list, color_list, marker_list

def get_all_sec_starIDs(assignment, prog_branch, sec_branch, sec_branch_2, idx_lim):
    #this function obtains the ID of all stars that even belong to the secondary galaxy 
    sec_allIDs = np.array([])
    for idx in range(0, min(max(list(assignment['ids'][sec_branch].keys())), idx_lim) + 1, 1):
        if idx in assignment['ids'][sec_branch].keys():
            overlap_ids = np.intersect1d(assignment['ids'][prog_branch][idx], assignment['ids'][sec_branch][idx])
            overlap_energies_prog = (assignment['energies'][prog_branch][idx])[np.intersect1d(overlap_ids, assignment['ids'][prog_branch][idx], return_indices=True)[2]]
            overlap_energies_sec = (assignment['energies'][sec_branch][idx])[np.intersect1d(overlap_ids, assignment['ids'][sec_branch][idx], return_indices=True)[2]]
            prog_ids_add = overlap_ids[overlap_energies_prog < overlap_energies_sec]
            sec_ids_add = overlap_ids[overlap_energies_sec < overlap_energies_prog]
            prog_ids_unique = np.setdiff1d(assignment['ids'][prog_branch][idx], assignment['ids'][sec_branch][idx])
            sec_ids_unique = np.setdiff1d(assignment['ids'][sec_branch][idx], assignment['ids'][prog_branch][idx])
            prog_ids_all = np.append(prog_ids_unique, prog_ids_add)
            sec_ids_all = np.append(sec_ids_unique, sec_ids_add)
            #
            sec_allIDs = np.unique(np.append(sec_allIDs, sec_ids_all))
    
    if sec_branch_2 != None: #sec_branch_2 are branches that are part of the secondary halo but still have the merger mass ratio > 0.25. Another case happens in AREPO, sec_branch is a sub-halo of sec_branch_2.
        for sec_branch_2_i in sec_branch_2:
            for idx in range(0, min(max(list(assignment['ids'][sec_branch_2_i].keys())), idx_lim) + 1, 1):
                if idx in assignment['ids'][sec_branch_2_i].keys():
                    overlap_ids = np.intersect1d(assignment['ids'][prog_branch][idx], assignment['ids'][sec_branch_2_i][idx])
                    overlap_energies_prog = (assignment['energies'][prog_branch][idx])[np.intersect1d(overlap_ids, assignment['ids'][prog_branch][idx], return_indices=True)[2]]
                    overlap_energies_sec = (assignment['energies'][sec_branch_2_i][idx])[np.intersect1d(overlap_ids, assignment['ids'][sec_branch_2_i][idx], return_indices=True)[2]]
                    prog_ids_add = overlap_ids[overlap_energies_prog < overlap_energies_sec]
                    sec_ids_add = overlap_ids[overlap_energies_sec < overlap_energies_prog]
                    prog_ids_unique = np.setdiff1d(assignment['ids'][prog_branch][idx], assignment['ids'][sec_branch_2_i][idx])
                    sec_ids_unique = np.setdiff1d(assignment['ids'][sec_branch_2_i][idx], assignment['ids'][prog_branch][idx])
                    prog_ids_all = np.append(prog_ids_unique, prog_ids_add)
                    sec_ids_all = np.append(sec_ids_unique, sec_ids_add)
                    #
                    sec_allIDs = np.unique(np.append(sec_allIDs, sec_ids_all))
    
    return sec_allIDs


def calc_r2_linregress(x,y):
    _,_,r,_,_, = scipy.stats.linregress(x, y)
    r2 = r*r
    return r2

#%%%%%%%%%%%%%%%%%%%%%%%%%% Figure 10 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
font_size = 14

fig, grid = plt.subplots(nrows=2, ncols=7, figsize=(16,6))
fig.subplots_adjust(wspace=0.1, hspace=0.1)
grid = grid.flatten()

halotree_ver = 2013
merger_number = '0'

burst_fraction_list = []
codetp_bf_list = []
label_bf_list = []
color_bf_list = []
marker_bf_list = []

for j in range(len(codetp_list)):
    codetp = codetp_list[j]
    if codetp == 'GADGET3' or codetp == 'GADGET4':
        continue
    # Adjust for plotting 7 codes
    if j < 4:
        k = j
    else:
        k = j - 2
    #
    redshift_list, time_list, pfs, step = load_halotree_and_pfs(codetp, halotree_ver, rawtree_skip=True)
    metadata_dir = '/work/hdd/bezm/tnguyen2/AGORA/%s/metadata/' % codetp
    if os.path.exists('/work/hdd/bezm/tnguyen2/AGORA/analysis/SF_data_for_burstFractionCalc_%s_ver2013.npy' % codetp) == False:
        assignment = np.load('/work/hdd/bezm/gtg115x/Halo_Finding/%s/star_id_%s_final_firstmerger.npy' % (codetp, halotree_ver), allow_pickle=True).tolist()
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
    if os.path.exists('/work/hdd/bezm/tnguyen2/AGORA/analysis/SF_data_for_burstFractionCalc_%s_ver2013.npy' % codetp) == True:
        output = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/SF_data_for_burstFractionCalc_%s_ver2013.npy' % codetp, allow_pickle=True).tolist()
    else:
        #All stars that ever belong to the secondary galaxy (to subtract to get the burst fraction)
        sec_allIDs = get_all_sec_starIDs(assignment, prog_branch, sec_branch, sec_branch_2, idx_lim)
        #
        output = {}
        output['stellar_mass'] = []
        output['sfr'] = []
        output['ssfr'] = []
        output['time'] = []
        output['idx'] = []
        
        if codetp == 'GEAR':
            idx_loop_start = 2
        else:
            idx_loop_start = 0

        for idx in range(idx_loop_start, idx_lim, step):
            if os.path.exists(metadata_dir + 'star_metadata_allbox_%s.npy' % idx) == False:
                continue
            if idx not in assignment['ids'][prog_branch].keys():
                continue
            metadata = np.load(metadata_dir + 'star_metadata_allbox_%s.npy' % idx, allow_pickle=True).tolist()
            mass_all = metadata['mass']
            age_all = metadata['age']
            if codetp == 'RAMSES':
                age_all = (time_list[idx].astype(float)/1000) - age_all
            pos_all = metadata['pos']
            ID_all = metadata['ID']
            #
            sID = assignment['ids'][prog_branch][idx]
            sID = np.setdiff1d(sID, sec_allIDs) #remove all the deposit stars + stars that are uniquely bound to the secondary galaxy at any points
            spos = pos_all[np.intersect1d(sID, ID_all, return_indices=True)[2]]
            smass = mass_all[np.intersect1d(sID, ID_all, return_indices=True)[2]]
            sage = age_all[np.intersect1d(sID, ID_all, return_indices=True)[2]]
            sfr = smass[sage < 10/1e3].sum()/(1e7) #averaged over 10 Myr
            ssfr = (sfr/smass.sum())/1e-9
            #
            output['stellar_mass'].append(smass.sum())
            output['sfr'].append(sfr)
            output['ssfr'].append(ssfr)
            output['time'].append(time_list[idx].astype(float)/1000)
            output['idx'].append(idx)

        output['stellar_mass'] = np.array(output['stellar_mass'])
        output['sfr'] = np.array(output['sfr'])
        output['ssfr'] = np.array(output['ssfr'])
        output['time'] = np.array(output['time'])
        output['idx'] = np.array(output['idx'])
        np.save('/work/hdd/bezm/tnguyen2/AGORA/analysis/SF_data_for_burstFractionCalc_%s_ver2013.npy' % codetp, output)
        del assignment
    #
    boolean = (output['time'] >= (time_begin - 0.1))*(output['time'] < time_begin)
    baseline_ssfr = np.min(output['ssfr'][boolean])
    #
    baseline_sm = output['stellar_mass'][output['idx'] < idx_begin]
    baseline_sfr = output['sfr'][output['idx'] < idx_begin]
    sm_step = output['stellar_mass'][output['idx'] == (idx_begin-step)][0]
    for idx in range(idx_begin, idx_lim, step):
        sfr_step = (sm_step*baseline_ssfr) #unit in Msun/Gyr
        baseline_sfr = np.append(baseline_sfr, sfr_step*1e-9)
        sm_step += sfr_step*(time_list[idx]/1e3 - time_list[idx - step]/1e3)
        baseline_sm = np.append(baseline_sm, sm_step)

    #Calculate the burst fraction
    burst_fraction = (output['stellar_mass'][output['idx'] == idx_cls][0] - baseline_sm[output['idx'] == idx_cls][0])/output['stellar_mass'][output['idx'] == idx_cls][0] 
    if codetp == 'GEAR':
        burst_fraction_vs_time = []
        time_bf_list = []
        for idx in range(idx_begin, idx_cls+step, step):
            time_bf_list.append(time_list[idx]/1e3)
            burst_fraction_vs_time.append((output['stellar_mass'][output['idx'] == idx][0] - baseline_sm[output['idx'] == idx][0])/output['stellar_mass'][output['idx'] == idx][0])  
        burst_fraction = max(burst_fraction_vs_time)
        time_bf = time_bf_list[np.argmax(burst_fraction_vs_time)]

    #Storing the results
    burst_fraction_list.append(burst_fraction)
    codetp_bf_list.append(codetp)
    label_bf_list.append(label_list[j])
    color_bf_list.append(color_list[j])
    marker_bf_list.append(marker_list[j])
    #
    #Plotting
    grid[k].plot(output['time'] - time_begin, output['ssfr'], color=color_list[j])
    if codetp == 'GEAR':
        grid[k].axvline(time_bf - time_begin, linestyle=':', color='black')
    else:
        grid[k].axvline(time_cls - time_begin, linestyle=':', color='black')
    grid[k].axhline(baseline_ssfr, color=color_list[j], linestyle='--')
    grid[k].fill_between(x=np.linspace(time_begin - 0.1 - time_begin, time_begin - time_begin, 100), y1=-100, y2=100, color='gray', alpha=0.15)
    grid[0].set_ylabel('sSFR (1/Gyr)', fontsize=font_size)
    grid[k].set_ylim(-1, 36)
    grid[k].set_xlim(time_begin - 0.2 - time_begin, time_cls + 0.2 - time_begin)
    grid[k].tick_params('both', labelsize=font_size)
    grid[k].set_xticks([])
    grid[k].text(x=time_begin - 0.1 - time_begin, y = 30, s=r'%.2f $\mathbf{\mathrm{\mathbf{Gyr}}^{-1}}$' % baseline_ssfr, ha='left', weight='bold')
    grid[k].set_title(label_list[j], fontsize = font_size)
    #
    if codetp == 'ART' or codetp == 'GADGET3' or codetp == 'GADGET4' or codetp == 'GIZMO':
        title_color = 'blue'
    else:
        title_color = 'black'
    if codetp == 'ART' or codetp == 'RAMSES' or codetp == 'GADGET3' or codetp == 'GEAR':
        grid[k].set_title(label_list[j] + '*', fontsize=font_size, color=title_color)
    elif codetp == 'CHANGA':
        grid[k].set_title(r'$\text{%s}^{\dagger}$' % label_list[j], fontsize=font_size, color=title_color)
    else:
        grid[k].set_title(label_list[j], fontsize=font_size, color=title_color)
    #
    grid[k].xaxis.set_major_locator(ticker.MaxNLocator(3))
    grid[k].set_xticklabels([])
    if k != 0:
        grid[k].set_yticks([])
    #
    grid[k+7].plot(output['time'] - time_begin, output['stellar_mass'], color=color_list[j], linestyle='-')
    grid[k+7].plot(output['time'] - time_begin, baseline_sm, color=color_list[j], linestyle='--')
    grid[7].set_ylabel(r'$M_\bigstar$ $(M_\odot)$', fontsize=font_size)
    grid[k+7].set_ylim(2e7, 5e10)
    grid[k+7].set_yscale('log')
    if codetp == 'GEAR':
        grid[k+7].axvline(time_bf - time_begin, linestyle=':', color='black')
    else:
        grid[k+7].axvline(time_cls - time_begin, linestyle=':', color='black')
    grid[k+7].fill_between(x=np.linspace(time_begin - 0.1 - time_begin, time_begin - time_begin, 100), y1=-100, y2=1e12, color='gray', alpha=0.15)
    grid[k+7].set_xlim(time_begin - 0.2 - time_begin, time_cls + 0.2 - time_begin)
    grid[k+7].tick_params('both', labelsize=font_size)
    grid[k+7].xaxis.set_major_locator(ticker.MaxNLocator(3))
    if k != 0:
        grid[k+7].set_yticklabels([])
    grid[7].set_xlabel('Time since beginning of infall (Gyr)', fontsize=font_size, loc='left')
    axtwin = grid[k+7].twinx()
    axtwin.plot(output['time'] - time_begin, output['sfr'], color='black', linestyle='-',alpha=0.4, lw=0.8)
    axtwin.plot(output['time'] - time_begin, baseline_sfr, color='black', linestyle='--',alpha=0.4, lw=0.8)
    axtwin.set_ylim(-2, 36)
    axtwin.tick_params('both', labelsize=font_size)
    if codetp == 'GEAR':
        axtwin.text(x=time_begin - 0.1 - time_begin, y = 30, s=r'$\mathbf{f_{sb}}*$ = %.3f' % burst_fraction, ha='left', weight='semibold')
    else:
        axtwin.text(x=time_begin - 0.1 - time_begin, y = 30, s=r'$\mathbf{f_{sb}}$ = %.3f' % burst_fraction, ha='left', weight='semibold')
    if k == 6:
        axtwin.set_ylabel(r'SFR ($M_\odot$/yr)', fontsize=font_size)
    if k != 6:
        axtwin.set_yticklabels([])
        
plt.savefig('/work/hdd/bezm/tnguyen2/figures/AGORA/November2025/BurstFraction_and_BaselinesSFR_ProgBranch-%s_ver2013_ver7.png' % merger_number, dpi=300, bbox_inches='tight')



#%%%%%%%%%%%%%%%%%%%%%%%%%% Figure 11 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

bf_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/Gas_Properties_preInfall_allCodes.npy', allow_pickle=True).tolist()

total_gas_to_all_list = []
totalmass_ratio_bf_list = []


for codetp in codetp_bf_list:
    redshift_list, time_list, pfs, step = load_halotree_and_pfs(codetp, halotree_ver, rawtree_skip=True)
    idx_begin, idx_endinfall, idx_1stpass, idx_maxdist, idx_cls, time_begin, time_endinfall, time_1stpass, time_maxdist, time_cls = load_timings(codetp, halotree_ver, merger_number)
    output_star_prog = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/SF_data_for_Merger_%s_ver2013_ConvexHull.npy' % codetp, allow_pickle=True).tolist()
    output_star_sec = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/SF_secondary_data_for_Merger_%s_ver2013_ConvexHull.npy' % codetp, allow_pickle=True).tolist()
    #
    gasmass = bf_data[codetp]['prog_gas_mass_total'] + bf_data[codetp]['sec_gas_mass_total']
    allmass = bf_data[codetp]['prog_gas_mass_total']/bf_data[codetp]['prog_gas_mass_fraction'] + bf_data[codetp]['sec_gas_mass_total']/bf_data[codetp]['sec_gas_mass_fraction']
    starmass = (output_star_prog['stellar_mass'])[output_star_prog['idx'] == (idx_begin - step)][0] + (output_star_sec['stellar_mass'])[output_star_sec['idx'] == (idx_begin - step)][0]
    total_gas_to_all_list.append(gasmass/allmass)
    #
    #the file "merger_mass_ratio_%s_ver2013.npy" is generated from the "merger_timing_plot.py" script in the "Part-1_Paper" folder.
    ratio_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/merger_mass_ratio_%s_ver2013.npy' % codetp, allow_pickle=True).tolist()
    totalmass_ratio_bf_list.append(ratio_data['total_ratio'])

font_size = 14
fig, grid = plt.subplots(ncols=2, nrows=1, figsize=(15/2,3.5), sharey=True)
fig.subplots_adjust(wspace=0.18, hspace=0.3)
grid = grid.flatten()

halotree_ver = 2013
merger_number = '0'

for n in range(len(codetp_bf_list)):
    grid[0].scatter(total_gas_to_all_list[n], burst_fraction_list[n], color=color_bf_list[n], marker=marker_bf_list[n], label=label_bf_list[n], s=60)
    grid[1].scatter(totalmass_ratio_bf_list[n], burst_fraction_list[n], color=color_bf_list[n], marker=marker_bf_list[n], label=label_bf_list[n], s=60)

grid[0].text(0.07, 0.07, s='$R^{2}$ = %.2f' % calc_r2_linregress(total_gas_to_all_list, burst_fraction_list), transform=grid[0].transAxes, fontsize=font_size-2)
grid[0].set_xlabel(r'$f_\text{gas}$', fontsize=font_size)
grid[0].set_ylabel(r'$f_\text{sb}$', fontsize=font_size)
grid[0].tick_params('both', labelsize=font_size)
grid[0].set_xticks([0.11, 0.13, 0.15, 0.17, 0.19])
#------------------------------------------------------------------------------------
grid[1].text(0.07, 0.07, s='$R^{2}$ = %.2f' % calc_r2_linregress(totalmass_ratio_bf_list, burst_fraction_list), transform=grid[1].transAxes, fontsize=font_size-2)
grid[1].set_xlabel(r'$\mu_\text{total}$', fontsize=font_size)
grid[1].tick_params('both', labelsize=font_size)
grid[1].set_xticks([0.5, 0.6, 0.7, 0.8])

handles1, labels1 = grid[0].get_legend_handles_labels()
fig.legend(handles1, labels1 , ncols=4, bbox_to_anchor=(0.5, -0.3), loc='lower center',ncol=1,fontsize=font_size-1)
plt.savefig('/work/hdd/bezm/tnguyen2/figures/AGORA/November2025/BurstFraction_Correlation_withPreMergerProperties_ProgBranch-%s_ver2013_PartApaper_ver2.png' % merger_number, dpi=300, bbox_inches='tight')
