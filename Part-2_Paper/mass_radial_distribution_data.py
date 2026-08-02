import numpy as np
import yt
yt.set_log_level(0)
import sys

import setup
from importlib import reload
reload(setup)

from setup import load_timings, load_ds, sec_branch_compute

codetp = sys.argv[1]
merger_number = '0'
halotree_ver = 2013

if codetp == 'GEAR':
    step = 3
elif codetp == 'CHANGA':
    step = 2
else:
    step = 1

redshift_list = np.loadtxt('/work/hdd/bezm/gtg115x/Halo_Finding/%s/pfs_allsnaps_%s.txt' % (codetp, halotree_ver), dtype=str)[:,1].astype(float)
time_list = np.loadtxt('/work/hdd/bezm/gtg115x/Halo_Finding/%s/pfs_allsnaps_%s.txt' % (codetp, halotree_ver), dtype=str)[:,2].astype(float)
pfs = np.loadtxt('/work/hdd/bezm/gtg115x/Halo_Finding/%s/pfs_allsnaps_%s.txt' % (codetp, halotree_ver), dtype=str)[:,0]
dist_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/dist_relvel_j_ProgBranch-%s_FirstMerger_%s_ver2013.npy' % (merger_number, codetp), allow_pickle=True).tolist()
prog_branch, sec_branch, sec_branch_2 = sec_branch_compute(codetp, merger_number)
idx_begin, idx_endinfall, idx_1stpass, idx_maxdist, idx_cls, time_begin, time_endinfall, time_1stpass, time_maxdist, time_cls = load_timings(codetp, halotree_ver, merger_number)
time_eval = time_1stpass + 0.6
if codetp == 'GEAR':
    idx_eval = np.argmin(abs(time_eval - time_list/1e3)) - (np.argmin(abs(time_eval - time_list/1e3)) % step) - 1
else:
    idx_eval = np.argmin(abs(time_eval - time_list/1e3)) - (np.argmin(abs(time_eval - time_list/1e3)) % step)

metadata_dir = '/work/hdd/bezm/tnguyen2/AGORA/%s/metadata/' % codetp
assignment = np.load('/work/hdd/bezm/gtg115x/Halo_Finding/%s/star_id_%s_final_firstmerger.npy' % (codetp, halotree_ver), allow_pickle=True).tolist()


def radial_mass_compute(codetp, idx, use_baryonic_center = False):
    #
    ds = load_ds(codetp, idx, pfs)
    #
    if use_baryonic_center == True:
        gal_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/%s/radius_2000_%s/Galaxy_Halo_%s_Snapshot_%s_comR2000.npy' % (codetp, codetp, prog_branch, idx), allow_pickle=True).tolist()
        try:
            gal_center = gal_data['com']
        except:
            gal_center = gal_data['gal_com']
    else:
        if idx == idx_begin - step:
            gal_center, gal_center2, v_bulk, v_bulk2, _ = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/premerger_centers_%s_ver2013_ver2.npy' % (codetp), allow_pickle=True).tolist().values()
        else:
            gal_center = np.array(dist_data['prog_com_plot'])[dist_data['idx'] == idx][0]
    #
    metadata = np.load(metadata_dir + 'star_metadata_allbox_%s.npy' % idx, allow_pickle=True).tolist()
    mass_all = metadata['mass']
    pos_all = metadata['pos']
    ID_all = metadata['ID']
    dist_all = np.linalg.norm(pos_all - gal_center, axis=1)
    gal_bool = np.intersect1d(assignment['ids'][prog_branch][idx], ID_all, return_indices=True)[2]
    #
    mass_gal = mass_all[gal_bool]
    dist_gal = dist_all[gal_bool]
    dist_gal = (dist_gal*ds.units.code_length).to('kpc').v
    #
    output = {}
    output['dist_gal'] = dist_gal
    output['mass_gal'] = mass_gal
    np.save('/work/hdd/bezm/tnguyen2/AGORA/analysis/radial_mass_distribution_ProgBranch-%s_Idx_%s_%s_ver2013.npy' % (merger_number, idx, codetp), output)
    return dist_gal, mass_gal


def get_all_sec_starIDs(assignment, sec_branch, step, idx_lim):
    #this function obtains the ID of all stars that even belong to the secondary galaxy 
    sec_allIDs = np.array([])
    for idx in range(0, min(max(list(assignment['ids'][sec_branch].keys())), idx_lim) + 1, 1):
        if idx in assignment['ids'][sec_branch].keys():
            metadata = np.load(metadata_dir + 'star_metadata_allbox_%s.npy' % idx, allow_pickle=True).tolist()
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
                    metadata = np.load(metadata_dir + 'star_metadata_allbox_%s.npy' % idx, allow_pickle=True).tolist()
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


def radial_mass_compute_decompose(codetp, idx, use_baryonic_center = True):
    #
    ds = load_ds(codetp, idx, pfs)
    idx_begin, idx_endinfall, idx_1stpass, idx_maxdist, idx_cls, time_begin, time_endinfall, time_1stpass, time_maxdist, time_cls = load_timings(codetp, halotree_ver, merger_number)
    #
    if use_baryonic_center == True:
        gal_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/%s/radius_2000_%s/Galaxy_Halo_%s_Snapshot_%s_comR2000.npy' % (codetp, codetp, prog_branch, idx), allow_pickle=True).tolist()
        try:
            gal_center = gal_data['com']
        except:
            gal_center = gal_data['gal_com']
    else:
        if idx == idx_begin - step:
            gal_center, gal_center2, v_bulk, v_bulk2, _ = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/premerger_centers_%s_ver2013_ver2.npy' % (codetp), allow_pickle=True).tolist().values()
        else:
            gal_center = np.array(dist_data['prog_com_plot'])[dist_data['idx'] == idx][0]
    #
    metadata = np.load(metadata_dir + 'star_metadata_allbox_%s.npy' % idx, allow_pickle=True).tolist()
    mass_all = metadata['mass']
    pos_all = metadata['pos']
    ID_all = metadata['ID']
    age_all = metadata['age']
    if codetp == 'RAMSES':
        age_all = (time_list[idx].astype(float)/1000) - age_all
    dist_all = np.linalg.norm(pos_all - gal_center, axis=1)
    #
    gal_bool = np.intersect1d(assignment['ids'][prog_branch][idx], ID_all, return_indices=True)[2]
    #
    mass_gal = mass_all[gal_bool]
    dist_gal = dist_all[gal_bool]
    ID_gal = ID_all[gal_bool]
    age_gal = age_all[gal_bool]
    ftime_gal = (time_list[idx].astype(float)/1000) - age_gal
    dist_gal = (dist_gal*ds.units.code_length).to('kpc').v
    #
    # Deposit stars
    ID_deposit = get_all_sec_starIDs(assignment, sec_branch, step, idx)
    dist_deposit = dist_gal[np.intersect1d(ID_deposit, ID_gal, return_indices=True)[2]]
    mass_deposit = mass_gal[np.intersect1d(ID_deposit, ID_gal, return_indices=True)[2]]    
    # 
    # Old stars + Accretion
    ID_old = ID_gal[ftime_gal < time_begin]
    ID_old = np.setdiff1d(ID_old, ID_deposit)
    dist_old = dist_gal[np.intersect1d(ID_old, ID_gal, return_indices=True)[2]]
    mass_old = mass_gal[np.intersect1d(ID_old, ID_gal, return_indices=True)[2]]
    # Infall 
    ID_infall = ID_gal[(ftime_gal >= time_begin)*(ftime_gal  < (time_begin + time_maxdist)/2)]
    ID_infall = np.setdiff1d(ID_infall, ID_deposit)
    dist_infall = dist_gal[np.intersect1d(ID_infall, ID_gal, return_indices=True)[2]]
    mass_infall = mass_gal[np.intersect1d(ID_infall, ID_gal, return_indices=True)[2]]
    # First passage
    ID_pass = ID_gal[(ftime_gal >= (time_begin + time_maxdist)/2)*(ftime_gal  < time_maxdist)]
    ID_pass = np.setdiff1d(ID_pass, ID_deposit)
    dist_pass = dist_gal[np.intersect1d(ID_pass, ID_gal, return_indices=True)[2]]
    mass_pass = mass_gal[np.intersect1d(ID_pass, ID_gal, return_indices=True)[2]]
    # Coalesence + Postcoalescence
    #ID_cls = ID_gal[(ftime_gal >= time_maxdist)*(ftime_gal  <= time_cls)]
    ID_cls = ID_gal[ftime_gal >= time_maxdist]
    ID_cls = np.setdiff1d(ID_cls, ID_deposit)
    dist_cls = dist_gal[np.intersect1d(ID_cls, ID_gal, return_indices=True)[2]]
    mass_cls = mass_gal[np.intersect1d(ID_cls, ID_gal, return_indices=True)[2]]
    #
    output = {}
    output['ID_old'] = ID_old
    output['dist_old'] = dist_old
    output['mass_old'] = mass_old
    output['ID_infall'] = ID_infall
    output['dist_infall'] = dist_infall
    output['mass_infall'] = mass_infall
    output['ID_pass'] = ID_pass
    output['dist_pass'] = dist_pass
    output['mass_pass'] = mass_pass
    output['ID_cls'] = ID_cls
    output['dist_cls'] = dist_cls
    output['mass_cls'] = mass_cls
    output['ID_deposit'] = ID_deposit
    output['dist_deposit'] = dist_deposit
    output['mass_deposit'] = mass_deposit
    output['ID_gal'] = ID_gal
    output['dist_gal'] = dist_gal
    output['mass_gal'] = mass_gal
    np.save('/work/hdd/bezm/tnguyen2/AGORA/analysis/radial_mass_distribution_decomposed_ProgBranch-%s_Idx_%s_%s_ver2013.npy' % (merger_number, idx, codetp), output)
    return ID_old, dist_old, mass_old, ID_infall, dist_infall, mass_infall, ID_pass, dist_pass, mass_pass,\
            ID_cls, dist_cls, mass_cls, ID_deposit, dist_deposit, mass_deposit, ID_gal, dist_gal, mass_gal

##################################
#Generate data for the left panel of Figure 2
radial_mass_compute(codetp, idx_begin - step, use_baryonic_center = False) #pre-infall timestep (no need for decomposition)
#Generate data for the right panel of Figure 2
radial_mass_compute_decompose(codetp, idx_eval, use_baryonic_center = True) #equivalent timestep (with decomposition to multiple merger stages)