import numpy as np
import yt
yt.set_log_level(0)
import sys

import setup
from importlib import reload
reload(setup)

from setup import load_halotree_and_pfs, load_timings, sec_branch_compute

def remove_outliers(arr, threshold=1.5):
    q3 = np.percentile(arr, 75)
    q1 = np.percentile(arr, 25)
    iqr = np.percentile(arr, 75) - np.percentile(arr, 25)
    return (arr > q1 - threshold*iqr)*(arr < q3 + threshold*iqr)

def remove_outliers_3d(pos, vel, mass, threshold=1.5):
    pos_bool = remove_outliers(pos[:,0])*remove_outliers(pos[:,1])*remove_outliers(pos[:,2])
    vel_bool = remove_outliers(vel[:,0])*remove_outliers(vel[:,1])*remove_outliers(vel[:,2])
    return pos[pos_bool*vel_bool], vel[pos_bool*vel_bool], mass[pos_bool*vel_bool]

def determine_center(pos, vel, mass, threshold=1.5):
    pos_f, vel_f, mass_f = remove_outliers_3d(pos, vel, mass, threshold)
    center = np.average(pos_f, weights=mass_f, axis=0)
    return center

def determine_center_and_velocity(pos, vel, mass, threshold=1.5):
    pos_f, vel_f, mass_f = remove_outliers_3d(pos, vel, mass, threshold)
    center = np.average(pos_f, weights=mass_f, axis=0)
    velocity = np.average(vel_f, weights=mass_f, axis=0)
    return center, velocity

codetp = sys.argv[1]
merger_number = '0'
halotree_ver = 2013

if codetp == 'GEAR':
    step = 3
elif codetp == 'CHANGA':
    step = 2
else:
    step = 1

redshift_list, time_list, pfs, step = load_halotree_and_pfs(codetp, halotree_ver, rawtree_skip=True)
dist_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/dist_relvel_j_ProgBranch-%s_FirstMerger_%s_ver2013.npy' % (merger_number, codetp), allow_pickle=True).tolist()
prog_branch, sec_branch, sec_branch_2 = sec_branch_compute(codetp, merger_number)
idx_begin, idx_endinfall, idx_1stpass, idx_maxdist, idx_cls, time_begin, time_endinfall, time_1stpass, time_maxdist, time_cls = load_timings(codetp, halotree_ver, merger_number)
idx_preinfall = idx_begin - step

time_eval = time_1stpass + 0.6
if codetp == 'GEAR':
    idx_eval = np.argmin(abs(time_eval - time_list/1e3)) - (np.argmin(abs(time_eval - time_list/1e3)) % step) - 1
else:
    idx_eval = np.argmin(abs(time_eval - time_list/1e3)) - (np.argmin(abs(time_eval - time_list/1e3)) % step)

metadata_dir = '/work/hdd/bezm/tnguyen2/AGORA/%s/metadata/' % codetp
assignment = np.load('/work/hdd/bezm/gtg115x/Halo_Finding/%s/star_id_%s_final_firstmerger.npy' % (codetp, halotree_ver), allow_pickle=True).tolist()

#Calculate the center and velocity of the two progenitors at idx_preinfall
allstars = np.load(metadata_dir + '/' + 'star_metadata_allbox_%s.npy' % idx_preinfall, allow_pickle=True).tolist()
allpos = allstars['pos']
allvel = allstars['vel']
allmass = allstars['mass']
allID = allstars['ID'].astype(int)

overlap_ids = np.intersect1d(assignment['ids'][prog_branch][idx_preinfall], assignment['ids'][sec_branch][idx_preinfall])
overlap_energies_prog = (assignment['energies'][prog_branch][idx_preinfall])[np.intersect1d(overlap_ids, assignment['ids'][prog_branch][idx_preinfall], return_indices=True)[2]]
overlap_energies_sec = (assignment['energies'][sec_branch][idx_preinfall])[np.intersect1d(overlap_ids, assignment['ids'][sec_branch][idx_preinfall], return_indices=True)[2]]
prog_ids_add = overlap_ids[overlap_energies_prog < overlap_energies_sec]
sec_ids_add = overlap_ids[overlap_energies_sec < overlap_energies_prog]
prog_ids_unique = np.setdiff1d(assignment['ids'][prog_branch][idx_preinfall], assignment['ids'][sec_branch][idx_preinfall])
sec_ids_unique = np.setdiff1d(assignment['ids'][sec_branch][idx_preinfall], assignment['ids'][prog_branch][idx_preinfall])
prog_ids_all = np.append(prog_ids_unique, prog_ids_add)
sec_ids_all = np.append(sec_ids_unique, sec_ids_add)

bound1_ID = prog_ids_all
bound1_pos = allpos[np.intersect1d(bound1_ID, allID, return_indices=True)[2]]
bound1_vel = allvel[np.intersect1d(bound1_ID, allID, return_indices=True)[2]]
bound1_mass = allmass[np.intersect1d(bound1_ID, allID, return_indices=True)[2]]

bound2_ID = sec_ids_all
bound2_pos = allpos[np.intersect1d(bound2_ID, allID, return_indices=True)[2]]
bound2_vel = allvel[np.intersect1d(bound2_ID, allID, return_indices=True)[2]]
bound2_mass = allmass[np.intersect1d(bound2_ID, allID, return_indices=True)[2]]

center1, vel1 = determine_center_and_velocity(bound1_pos, bound1_vel, bound1_mass)
center2, vel2 = determine_center_and_velocity(bound2_pos, bound2_vel, bound2_mass)

# Saving the output
center_output = {}
center_output['center1'] = center1
center_output['center2'] = center2
center_output['vel1'] = vel1
center_output['vel2'] = vel2
center_output['idx'] = idx_preinfall

np.save('/work/hdd/bezm/tnguyen2/AGORA/analysis/premerger_centers_%s_ver2013_ver2.npy' % (codetp), center_output)
