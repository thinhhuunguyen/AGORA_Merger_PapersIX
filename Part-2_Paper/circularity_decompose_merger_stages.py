import numpy as np
import sys

import setup
from importlib import reload
reload(setup)

from setup import load_halotree_and_pfs, load_timings, sec_branch_compute


def circularity_compute_decompose(codetp):
    # EVALUATED AT IDX_EVAL
    halotree_ver = 2013
    merger_number = '0'
    redshift_list, time_list, pfs, step = load_halotree_and_pfs(codetp, halotree_ver, rawtree_skip=True)
    prog_branch, sec_branch, sec_branch_2 = sec_branch_compute(codetp, merger_number)
    idx_begin, idx_endinfall, idx_1stpass, idx_maxdist, idx_cls, time_begin, time_endinfall, time_1stpass, time_maxdist, time_cls = load_timings(codetp, halotree_ver, merger_number)
    time_eval = time_1stpass + 0.6
    if codetp == 'GEAR':
        idx_eval = np.argmin(abs(time_eval - time_list/1e3)) - (np.argmin(abs(time_eval - time_list/1e3)) % step) - 1
    else:
        idx_eval = np.argmin(abs(time_eval - time_list/1e3)) - (np.argmin(abs(time_eval - time_list/1e3)) % step)
    #
    epsilon_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/circularity_ProgBranch-%s_Idx_%s_%s_ver2013_Abadi2003method.npy' % (codetp, prog_branch, idx_eval), allow_pickle=True).tolist()
    epsilon_gal = epsilon_data['circ']
    ID_gal = epsilon_data['ID']
    dist_gal = epsilon_data['dist']
    mass_gal = epsilon_data['mass']
    #
    metadata_dir = '/work/hdd/bezm/tnguyen2/AGORA/%s/metadata/' % codetp
    metadata = np.load(metadata_dir + 'star_metadata_allbox_%s.npy' % idx_eval, allow_pickle=True).tolist()
    ID_all = metadata['ID']
    age_all = metadata['age']
    if codetp == 'RAMSES':
        age_all = (time_list[idx_eval].astype(float)/1000) - age_all

    gal_bool = np.intersect1d(ID_gal, ID_all, return_indices=True)[2] #this works becasue ID_gal is already sorted 
    age_gal = age_all[gal_bool]
    ftime_gal = (time_list[idx_eval].astype(float)/1000) - age_gal
    #
    _, _, _, _, _, _, _, _, _, \
    _, _, _, ID_deposit, _, _, \
    _, _, _ = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/radial_mass_distribution_decomposed_ProgBranch-%s_Idx_%s_%s_ver2013.npy' % (merger_number, idx_eval, codetp), allow_pickle=True).tolist().values()

    # Deposit stars
    epsilon_deposit = epsilon_gal[np.intersect1d(ID_deposit, ID_gal, return_indices=True)[2]]
    mass_deposit = mass_gal[np.intersect1d(ID_deposit, ID_gal, return_indices=True)[2]] 
    dist_deposit = dist_gal[np.intersect1d(ID_deposit, ID_gal, return_indices=True)[2]]      
    # 
    # Old stars + Accretion
    ID_old = ID_gal[ftime_gal < time_begin]
    ID_old = np.setdiff1d(ID_old, ID_deposit)
    epsilon_old = epsilon_gal[np.intersect1d(ID_old, ID_gal, return_indices=True)[2]]
    mass_old = mass_gal[np.intersect1d(ID_old, ID_gal, return_indices=True)[2]]
    dist_old = dist_gal[np.intersect1d(ID_old, ID_gal, return_indices=True)[2]]
    # Infall 
    ID_infall = ID_gal[(ftime_gal >= time_begin)*(ftime_gal  < (time_begin + time_maxdist)/2)]
    ID_infall = np.setdiff1d(ID_infall, ID_deposit)
    epsilon_infall = epsilon_gal[np.intersect1d(ID_infall, ID_gal, return_indices=True)[2]]
    mass_infall = mass_gal[np.intersect1d(ID_infall, ID_gal, return_indices=True)[2]]
    dist_infall = dist_gal[np.intersect1d(ID_infall, ID_gal, return_indices=True)[2]]
    # First passage
    ID_pass = ID_gal[(ftime_gal >= (time_begin + time_maxdist)/2)*(ftime_gal  < time_maxdist)]
    ID_pass = np.setdiff1d(ID_pass, ID_deposit)
    epsilon_pass = epsilon_gal[np.intersect1d(ID_pass, ID_gal, return_indices=True)[2]]
    mass_pass = mass_gal[np.intersect1d(ID_pass, ID_gal, return_indices=True)[2]]
    dist_pass = dist_gal[np.intersect1d(ID_pass, ID_gal, return_indices=True)[2]]
    # Coalesence + Postcoalescence
    #ID_cls = ID_gal[(ftime_gal >= time_maxdist)*(ftime_gal  <= time_cls)]
    ID_cls = ID_gal[ftime_gal >= time_maxdist]
    ID_cls = np.setdiff1d(ID_cls, ID_deposit)
    epsilon_cls = epsilon_gal[np.intersect1d(ID_cls, ID_gal, return_indices=True)[2]]
    mass_cls = mass_gal[np.intersect1d(ID_cls, ID_gal, return_indices=True)[2]]
    dist_cls = dist_gal[np.intersect1d(ID_cls, ID_gal, return_indices=True)[2]]
    #
    output = {}
    output['ID_old'] = ID_old
    output['epsilon_old'] = epsilon_old
    output['mass_old'] = mass_old
    output['dist_old'] = dist_old
    output['ID_infall'] = ID_infall
    output['epsilon_infall'] = epsilon_infall
    output['mass_infall'] = mass_infall
    output['dist_infall'] = dist_infall
    output['ID_pass'] = ID_pass
    output['epsilon_pass'] = epsilon_pass
    output['mass_pass'] = mass_pass
    output['dist_pass'] = dist_pass
    output['ID_cls'] = ID_cls
    output['epsilon_cls'] = epsilon_cls
    output['mass_cls'] = mass_cls
    output['dist_cls'] = dist_cls
    output['ID_deposit'] = ID_deposit
    output['epsilon_deposit'] = epsilon_deposit
    output['mass_deposit'] = mass_deposit
    output['dist_deposit'] = dist_deposit
    output['ID_gal'] = ID_gal
    output['epsilon_gal'] = epsilon_gal
    output['mass_gal'] = mass_gal
    output['dist_gal'] = dist_gal
    np.save('/work/hdd/bezm/tnguyen2/AGORA/analysis/circularity_decomposed_ProgBranch-%s_Idx_%s_%s_ver2013_Abadi2003method.npy' % (merger_number, idx_eval, codetp), output)
    return ID_old, epsilon_old, mass_old, dist_old, ID_infall, epsilon_infall, mass_infall, dist_infall, \
            ID_pass, epsilon_pass, mass_pass, dist_pass, \
            ID_cls, epsilon_cls, mass_cls, dist_cls,\
            ID_deposit, epsilon_deposit, mass_deposit, dist_deposit,\
            ID_gal, epsilon_gal, mass_gal, dist_gal

#%%%%%%%%%%%%%%%%%% Run the function %%%%%%%%%%%%%%%%%%
codetp = sys.argv[1]
circularity_compute_decompose(codetp)