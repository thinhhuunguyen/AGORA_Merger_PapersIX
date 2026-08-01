import yt
yt.set_log_level(0)
import numpy as np
import matplotlib.pyplot as plt
import glob as glob
import os
    
import setup
from importlib import reload
reload(setup)

from setup import load_halotree_and_pfs, load_timings, sec_branch_compute


codetp = 'CHANGA'
merger_number = '0'
halotree_ver = 2013

redshift_list, time_list, pfs, step = load_halotree_and_pfs(codetp, halotree_ver, rawtree_skip=True)
metadata_dir = '/work/hdd/bezm/tnguyen2/AGORA/%s/metadata/' % codetp
    
dist_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/dist_relvel_j_ProgBranch-%s_FirstMerger_%s_ver2013.npy' % (merger_number, codetp), allow_pickle=True).tolist()
prog_branch, sec_branch, sec_branch_2 = sec_branch_compute(codetp, merger_number)
idx_begin, idx_endinfall, idx_1stpass, idx_maxdist, idx_cls, time_begin, time_endinfall, time_1stpass, time_maxdist, time_cls = load_timings(codetp, halotree_ver, merger_number)
time_eval = time_1stpass + 0.6

if codetp == 'GEAR':
    idx_eval = np.argmin(abs(time_eval - time_list/1e3)) - (np.argmin(abs(time_eval - time_list/1e3)) % step) - 1
else:
    idx_eval = np.argmin(abs(time_eval - time_list/1e3)) - (np.argmin(abs(time_eval - time_list/1e3)) % step)

font_size = 18

if os.path.exists('/work/hdd/bezm/tnguyen2/AGORA/analysis/smass_sfr_evolution_ProgBranch-0_%s_ver%s.npy' % (codetp, halotree_ver)):
    output = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/smass_sfr_evolution_ProgBranch-0_%s_ver%s.npy' % (codetp, halotree_ver), allow_pickle=True).tolist()
else:
    metadata_dir = '/work/hdd/bezm/tnguyen2/AGORA/%s/metadata/' % codetp
    assignment = np.load('/work/hdd/bezm/gtg115x/Halo_Finding/%s/star_id_%s_final_firstmerger.npy' % (codetp, halotree_ver), allow_pickle=True).tolist()
    if codetp == 'CHANGA':
        step = 2
    elif codetp == 'GEAR':
        step = 3
    else:
        step = 1
    #
    output = {}
    output['stellar_mass'] = []
    output['sfr'] = []
    output['ssfr'] = []
    output['time'] = []
    output['idx'] = []

    for idx in range(idx_begin, dist_data['idx'][-1] + step, step):
        metadata = np.load(metadata_dir + 'star_metadata_allbox_%s.npy' % idx, allow_pickle=True).tolist()
        mass_all = metadata['mass']
        age_all = metadata['age']
        if codetp == 'CHANGA':
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
        smass = mass_all[np.intersect1d(sID, ID_all, return_indices=True)[2]]
        smass_init = mass_init_all[np.intersect1d(sID, ID_all, return_indices=True)[2]]
        sage = age_all[np.intersect1d(sID, ID_all, return_indices=True)[2]]
        sfr = smass_init[sage < 10/1e3].sum()/(1e7) #averaged over 10 Myr
        ssfr = (sfr/smass.sum())/1e-9
        #
        output['stellar_mass'].append(smass.sum())
        output['sfr'].append(sfr)
        output['ssfr'].append(ssfr)
        output['time'].append(time_list[idx].astype(float)/1000)
        output['idx'].append(idx)
        np.save('/work/hdd/bezm/tnguyen2/AGORA/analysis/smass_sfr_evolution_ProgBranch-0_%s_ver%s.npy' % (codetp, halotree_ver), output)

time_anchor = time_begin 
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(output['time'] - time_anchor, output['sfr'], '.-', color='tab:pink', linewidth=3, markersize=10)
ax.fill_between (x = np.linspace((time_begin + time_maxdist)/2 , time_begin) - time_anchor, y1 = -1e99, y2 = 1e99, alpha=0.09, color='cyan')
ax.fill_between (x = np.linspace(time_maxdist, (time_begin + time_maxdist)/2 ) - time_anchor, y1 = -1e99, y2 = 1e99, alpha=0.09, color='orange')
ax.fill_between (x = np.linspace(time_cls, time_maxdist) - time_anchor, y1 = -1e99, y2 = 1e99, alpha=0.09, color='red')
ax.fill_between (x = np.linspace(5, time_cls) - time_anchor, y1 = -1e99, y2 = 1e99, alpha=0.09, color='darkred')
ax.set_ylim(-1, 105)
ax.set_xlim(0, 2.1)

ax.axvline(time_list[266]/1e3 - time_begin, linestyle='--', color='k')
ax.axvline(time_list[250]/1e3 - time_begin, linestyle='--', color='k')

ax.set_ylabel(r'SFR $(M_\odot/\text{yr})$', fontsize=font_size)
ax.set_xlabel('Time since beginning of infall (Gyr)', fontsize=font_size)
ax.tick_params('both', labelsize=font_size)

fig.savefig('/work/hdd/bezm/tnguyen2/figures/AGORA/November2025/SFR_CHANGA_delayedStarburst_ver2.png', dpi=300, bbox_inches='tight')