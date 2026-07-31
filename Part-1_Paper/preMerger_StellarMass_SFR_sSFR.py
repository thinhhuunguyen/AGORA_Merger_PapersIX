import numpy as np
import matplotlib.pyplot as plt
import yt
yt.set_log_level(0)
import os

import setup
from importlib import reload
reload(setup)

from setup import load_halotree_and_pfs, load_timings, load_ds, sec_branch_compute
from setup import in_hull
from setup import codetp_list, label_list, color_list


font_size = 14

fig, grid = plt.subplots(ncols=1, nrows=3, figsize=(6,9))
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
    if os.path.exists('/work/hdd/bezm/tnguyen2/AGORA/analysis/SF_data_for_Merger_%s_ver2013_ConvexHull.npy' % codetp) == False:
        assignment = np.load('/work/hdd/bezm/gtg115x/Halo_Finding/%s/star_id_%s_final_firstmerger.npy' % (codetp, halotree_ver), allow_pickle=True).tolist()
        hullv = np.load('/work/hdd/bezm/gtg115x/Halo_Finding/%s/hullv_%s_withPos_final.npy' % (codetp, halotree_ver), allow_pickle=True).tolist()
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
    time_lim = 2.5
    idx_lim = np.argmin(abs(time_lim - time_list/1e3))    
    if os.path.exists('/work/hdd/bezm/tnguyen2/AGORA/analysis/SF_data_for_Merger_%s_ver2013_ConvexHull.npy' % codetp) == True:
        output = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/SF_data_for_Merger_%s_ver2013_ConvexHull.npy' % codetp, allow_pickle=True).tolist()
    else:
        #
        output = {}
        output['stellar_mass'] = []
        output['sfr'] = []
        output['ssfr'] = []
        output['time'] = []
        output['idx'] = []

        time_loop_start = 0.5
        idx_loop_start = np.argmin(abs(time_loop_start - time_list/1e3))  

        for idx in range(idx_loop_start, idx_lim, 1):
            if os.path.exists(metadata_dir + 'star_metadata_allbox_%s.npy' % idx) == False:
                continue
            if idx not in assignment['ids'][prog_branch].keys():
                continue
            metadata = np.load(metadata_dir + 'star_metadata_allbox_%s.npy' % idx, allow_pickle=True).tolist()
            mass_all = metadata['mass']
            age_all = metadata['age']
            if codetp == 'RAMSES':
                age_all = (time_list[idx].astype(float)/1000) - age_all
                
            if codetp == 'ART' or codetp == 'GADGET3' or codetp == 'GADGET4' or codetp == 'GEAR' or codetp == 'GIZMO' or codetp == 'AREPO' or codetp == 'AREPO-TNG':
                mass_init_all = metadata['mass_init']
            elif codetp == 'ENZO':
                mask = age_all*1e3 < 5 #age less than 5 Myr
                mass_init_all = mass_all
                mass_init_all[mask] = mass_init_all[mask] / (1 - 0.163) #If star particle's age is more than 5Myr, it loses 16.3% of its mass. 
            elif codetp == 'RAMSES':
                mass_init_all = mass_all #RAMSES does not implement stellar mass loss
            elif codetp == 'CHANGA':
                mass_init_all = 56525.92803654*np.ones(len(mass_all)) #CHANGA implements fixed initial stellar mass

            pos_all = metadata['pos']
            ID_all = metadata['ID']
            #
            if idx > max(list(assignment['ids'][sec_branch].keys())):
                prog_ids_all = assignment['ids'][prog_branch][idx]
            else:
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
            sID = prog_ids_all
            spos = pos_all[np.intersect1d(sID, ID_all, return_indices=True)[2]]
            smass = mass_all[np.intersect1d(sID, ID_all, return_indices=True)[2]]
            smass_init = mass_init_all[np.intersect1d(sID, ID_all, return_indices=True)[2]]
            sage = age_all[np.intersect1d(sID, ID_all, return_indices=True)[2]]
            sID = ID_all[np.intersect1d(sID, ID_all, return_indices=True)[2]]
            #
            ds = load_ds(codetp, idx, pfs)
            hull_bool = in_hull((spos*ds.units.code_length).to('m').v, hullv[idx][prog_branch])
            spos = spos[hull_bool]
            smass = smass[hull_bool]
            smass_init = smass_init[hull_bool]
            sage = sage[hull_bool]
            sID = sID[hull_bool]
            sfr = smass_init[sage < 10/1e3].sum()/(1e7) #averaged over 10 Myr
            ssfr = (sfr/smass.sum())/1e-9
            #
            output['stellar_mass'].append(smass.sum())
            output['sfr'].append(sfr)
            output['ssfr'].append(ssfr)
            output['time'].append(time_list[idx].astype(float)/1000)
            output['idx'].append(idx)
            del ds

        output['stellar_mass'] = np.array(output['stellar_mass'])
        output['sfr'] = np.array(output['sfr'])
        output['ssfr'] = np.array(output['ssfr'])
        output['time'] = np.array(output['time'])
        output['idx'] = np.array(output['idx'])
        np.save('/work/hdd/bezm/tnguyen2/AGORA/analysis/SF_data_for_Merger_%s_ver2013_ConvexHull.npy' % codetp, output)
        print('Done with', codetp)
        del assignment, hullv
    #
    time_begin_min = min(time_begin_min, time_begin)
    time_begin_max = max(time_begin_max, time_begin)
    #
    #Plotting
    #
    grid[0].plot(output['time'], output['stellar_mass'], color=color_list[k], label=label_list[k])
    grid[0].set_ylabel(r'$M_\bigstar$ $(M_\odot)$', fontsize=font_size)
    grid[0].set_yscale('log')
    grid[0].set_xlim(0.6, 1.17)
    grid[0].set_ylim(1e7, 2.5e9)
    grid[0].tick_params('both', labelsize=font_size)
    grid[0].set_xticklabels([])
    #
    grid[1].plot(output['time'], output['sfr'], color=color_list[k], label=label_list[k])
    grid[1].set_ylabel(r'SFR ($M_\odot$/yr)', fontsize=font_size)
    grid[1].set_ylim(-0.5, 11)
    grid[1].set_xlim(0.6, 1.17)
    grid[1].tick_params('both', labelsize=font_size)
    grid[1].set_xticklabels([])
    #
    grid[2].plot(output['time'], output['ssfr'], color=color_list[k], label=label_list[k])
    grid[2].set_ylabel('sSFR (1/Gyr)', fontsize=font_size)
    grid[2].set_xlabel(r'Time since Big Bang (Gyr)', fontsize=font_size)
    grid[2].set_ylim(-0.5, 23)
    grid[2].set_xlim(0.6, 1.17)
    grid[2].tick_params('both', labelsize=font_size)

grid[0].fill_between (x = np.linspace(time_begin_min , time_begin_max), y1 = -1e99, y2 = 1e99, alpha=0.09, color='grey')
grid[1].fill_between (x = np.linspace(time_begin_min , time_begin_max), y1 = -1e99, y2 = 1e99, alpha=0.09, color='grey')
grid[2].fill_between (x = np.linspace(time_begin_min , time_begin_max), y1 = -1e99, y2 = 1e99, alpha=0.09, color='grey')

grid[1].legend(ncols=2)

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

plt.savefig('/work/hdd/bezm/tnguyen2/figures/AGORA/November2025/GasMass_StellarMass_SFR_PreMerger_ProgBranch-%s_ver2013_PartApaper_ver2.png' % merger_number, dpi=300, bbox_inches='tight')
