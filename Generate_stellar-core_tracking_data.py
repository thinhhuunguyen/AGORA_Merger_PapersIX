import numpy as np
import unyt
import yt
from scipy.spatial.distance import cdist
from scipy.interpolate import CubicSpline
from tqdm import tqdm
import os, sys

import setup
from importlib import reload
reload(setup)

from setup import load_halotree_and_pfs, load_ds, sec_branch_compute
from setup import get_haloradius


def std_weighted(values, weights):
    average = np.average(values, weights=weights, axis=0)
    variance = np.average((values - average)**2, weights=weights, axis=0)*len(weights)/(len(weights) - 1)
    return np.sqrt(variance)


def remove_outliers(arr, threshold=1.5):
    median = np.median(arr)
    iqr = np.percentile(arr, 75) - np.percentile(arr, 25)
    return (arr > median - threshold*iqr)*(arr < median + threshold*iqr)


def monte_carlo_distance_error(x, y, sx, sy, ntrials=200000, seed=None):
    rng = np.random.default_rng(seed)
    n = len(x)
    # draw samples: shape (ntrials, n)
    xs = rng.normal(loc=x, scale=sx, size=(ntrials, n))
    ys = rng.normal(loc=y, scale=sy, size=(ntrials, n))
    diffs = xs - ys
    ds = np.linalg.norm(diffs, axis=1)  # array of distances
    return np.array([np.percentile(ds, 16), np.percentile(ds, 50), np.average(ds), np.percentile(ds, 84)])


codetp = sys.argv[1] 
idx_begin = int(sys.argv[2]) #manually input from the "Finding_the_infall_timestep_of_the_merger_non-spherical_halos.ipynb" notebook

merger_number = '0'
halotree_ver = 2013
rawtree, redshift_list, time_list, pfs, step = load_halotree_and_pfs(codetp, halotree_ver)
prog_branch, sec_branch, sec_branch_2 = sec_branch_compute(codetp, merger_number)
assignment = np.load('/work/hdd/bezm/gtg115x/Halo_Finding/%s/star_id_%s_final_firstmerger.npy' % (codetp, halotree_ver), allow_pickle=True).tolist()


ds = load_ds(codetp, idx_begin, pfs)

# Load the top 30% most bound particles in the progenitor and secondary halos at the infall timestep (initial filter)
init_percent  = 0.3 #default 15%-30%
ntrack_init_prog = int(init_percent*len(assignment['ids'][prog_branch][idx_begin]))
ntrack_init_sec = int(init_percent*len(assignment['ids'][sec_branch][idx_begin]))

track_ids_prog_init = assignment['ids'][prog_branch][idx_begin][np.argsort(assignment['energies'][prog_branch][idx_begin])][:ntrack_init_prog]
track_ids_sec_init = assignment['ids'][sec_branch][idx_begin][np.argsort(assignment['energies'][sec_branch][idx_begin])][:ntrack_init_sec]
track_energies_prog_init = np.sort(assignment['energies'][prog_branch][idx_begin])[:ntrack_init_prog]
track_energies_sec_init = np.sort(assignment['energies'][sec_branch][idx_begin])[:ntrack_init_sec]

# Load star metadata in the whole simulation box
metadata_dir = '/work/hdd/bezm/tnguyen2/AGORA/%s/metadata/' % codetp
metadata = np.load(metadata_dir + 'star_metadata_allbox_%s.npy' % idx_begin, allow_pickle=True).tolist()
mass_all = metadata['mass']
pos_all = metadata['pos']
vel_all = metadata['vel']
ID_all = metadata['ID']


track_pos_prog = pos_all[np.intersect1d(track_ids_prog_init, ID_all, return_indices=True)[2]]
track_pos_sec = pos_all[np.intersect1d(track_ids_sec_init, ID_all, return_indices=True)[2]]
track_vel_prog = vel_all[np.intersect1d(track_ids_prog_init, ID_all, return_indices=True)[2]]
track_vel_sec = vel_all[np.intersect1d(track_ids_sec_init, ID_all, return_indices=True)[2]]
track_mass_prog = mass_all[np.intersect1d(track_ids_prog_init, ID_all, return_indices=True)[2]]
track_mass_sec = mass_all[np.intersect1d(track_ids_sec_init, ID_all, return_indices=True)[2]]
track_ids_prog = ID_all[np.intersect1d(track_ids_prog_init, ID_all, return_indices=True)[2]]
track_ids_sec = ID_all[np.intersect1d(track_ids_sec_init, ID_all, return_indices=True)[2]]
#because np.intersect1d sorts the ids array, we rearrange the energy array too
track_energies_prog = track_energies_prog_init[np.argsort(track_ids_prog_init)]
track_energies_sec = track_energies_sec_init[np.argsort(track_ids_sec_init)]

# Remove outliers beyond 1.5*IQR (interquartile range) from the median in position and velocity for both progenitor and secondary halos
outlier_threshold = 1.5 #set 1 for ART, 1.5 for the rest
outliers_prog = remove_outliers(track_pos_prog[:,0], outlier_threshold)*remove_outliers(track_pos_prog[:,1], outlier_threshold)*remove_outliers(track_pos_prog[:,2], outlier_threshold)*\
    remove_outliers(track_vel_prog[:,0], outlier_threshold)*remove_outliers(track_vel_prog[:,1], outlier_threshold)*remove_outliers(track_vel_prog[:,2], outlier_threshold)
outliers_sec = remove_outliers(track_pos_sec[:,0], outlier_threshold)*remove_outliers(track_pos_sec[:,1], outlier_threshold)*remove_outliers(track_pos_sec[:,2], outlier_threshold)*\
    remove_outliers(track_vel_sec[:,0], outlier_threshold)*remove_outliers(track_vel_sec[:,1], outlier_threshold)*remove_outliers(track_vel_sec[:,2], outlier_threshold)

track_pos_prog = track_pos_prog[outliers_prog]
track_pos_sec = track_pos_sec[outliers_sec]
track_ids_prog = track_ids_prog[outliers_prog]
track_ids_sec = track_ids_sec[outliers_sec]
track_vel_prog = track_vel_prog[outliers_prog]
track_vel_sec = track_vel_sec[outliers_sec]
track_mass_prog = track_mass_prog[outliers_prog]
track_mass_sec = track_mass_sec[outliers_sec]
track_energies_prog = track_energies_prog[outliers_prog]
track_energies_sec = track_energies_sec[outliers_sec]

# Select the top 10% most bound particles in the progenitor and secondary halos at the infall timestep (final filter)
ntrack_f_prog = int(0.1*len(assignment['ids'][prog_branch][idx_begin]))
ntrack_f_sec = int(0.1*len(assignment['ids'][sec_branch][idx_begin]))

track_pos_prog = track_pos_prog[np.argsort(track_energies_prog)][:ntrack_f_prog]
track_pos_sec = track_pos_sec[np.argsort(track_energies_sec)][:ntrack_f_sec]
track_ids_prog = track_ids_prog[np.argsort(track_energies_prog)][:ntrack_f_prog]
track_ids_sec = track_ids_sec[np.argsort(track_energies_sec)][:ntrack_f_sec]
track_vel_prog = track_vel_prog[np.argsort(track_energies_prog)][:ntrack_f_prog]
track_vel_sec = track_vel_sec[np.argsort(track_energies_sec)][:ntrack_f_sec]
track_mass_prog = track_mass_prog[np.argsort(track_energies_prog)][:ntrack_f_prog]
track_mass_sec = track_mass_sec[np.argsort(track_energies_sec)][:ntrack_f_sec]
track_energies_prog = track_energies_prog[np.argsort(track_energies_prog)][:ntrack_f_prog]
track_energies_sec = track_energies_sec[np.argsort(track_energies_sec)][:ntrack_f_sec]

# Compute the center of mass position and velocity of the progenitor stellar cores
track_com_prog = np.average(track_pos_prog, weights=track_mass_prog, axis=0)
track_com_sec = np.average(track_pos_sec, weights=track_mass_sec, axis=0)
track_comvel_prog = np.average(track_vel_prog, weights=track_mass_prog, axis=0)
track_comvel_sec = np.average(track_vel_sec, weights=track_mass_sec, axis=0)
dist = np.linalg.norm(track_com_prog - track_com_sec)
relvel = track_comvel_sec - track_comvel_prog
relvelmag = np.linalg.norm(relvel) #in km/s
#Compute the error on the dist and relvel mag calculation
prog_comstd = std_weighted(track_pos_prog, track_mass_prog)
sec_comstd = std_weighted(track_pos_sec, track_mass_sec)
comrelstd = np.sqrt(prog_comstd**2 + sec_comstd**2)
comrel = track_com_prog - track_com_sec
dist_std = (1/dist)*np.sqrt(np.sum((comrel**2)*(comrelstd**2)))
#
ds = load_ds(codetp, idx_begin, pfs)
dist = (dist*ds.units.code_length).to('kpc').v.tolist()
dist_std = (dist_std*ds.units.code_length).to('kpc').v.tolist()
#
prog_comvel_std = std_weighted(track_vel_prog, track_mass_prog)
sec_comvel_std = std_weighted(track_vel_sec, track_mass_sec)
relvel_std = np.sqrt(prog_comvel_std**2 + sec_comvel_std**2)
relvelmag_std = (1/relvelmag)*np.sqrt(np.sum((relvel**2)*(relvel_std**2)))


# Make the output file for the stellar core tracking data
dist_data = {}
dist_data['prog_ids_plot'] = track_ids_prog
dist_data['sec_ids_plot'] = track_ids_sec
dist_data['dist'] = np.array([dist])
dist_data['dist_std'] = np.array([dist_std])
dist_data['relvel'] = [relvel]
dist_data['relvelmag'] = np.array([relvelmag])
dist_data['relvelmag_std'] = np.array([relvelmag_std])
dist_data['idx'] = np.array([idx_begin])
dist_data['time'] = np.array([time_list[idx_begin]/1e3])
dist_data['z'] = np.array([redshift_list[idx_begin]])
dist_data['prog_pos_plot'] = [track_pos_prog]
dist_data['sec_pos_plot'] = [track_pos_sec]
dist_data['prog_vel_plot'] = [track_vel_prog]
dist_data['sec_vel_plot'] = [track_vel_sec]
dist_data['prog_mass_plot'] = [track_mass_prog]
dist_data['sec_mass_plot'] = [track_mass_sec]
dist_data['codelength_to_meters'] = np.array([ds.length_unit.to('m')])

# Tracking the stellar cores
idx_stopeval = np.argmin(abs(time_list - 3200))

for idx_i in tqdm(range(idx_begin + step, idx_stopeval + step, step)):
    metadata = np.load(metadata_dir + 'star_metadata_allbox_%s.npy' % idx_i, allow_pickle=True).tolist()
    mass_all = metadata['mass']
    pos_all = metadata['pos']
    vel_all = metadata['vel']
    ID_all = metadata['ID']
    #
    track_pos_prog = pos_all[np.intersect1d(track_ids_prog, ID_all, return_indices=True)[2]]
    track_pos_sec = pos_all[np.intersect1d(track_ids_sec, ID_all, return_indices=True)[2]]
    track_vel_prog = vel_all[np.intersect1d(track_ids_prog, ID_all, return_indices=True)[2]]
    track_vel_sec = vel_all[np.intersect1d(track_ids_sec, ID_all, return_indices=True)[2]]
    track_mass_prog = mass_all[np.intersect1d(track_ids_prog, ID_all, return_indices=True)[2]]
    track_mass_sec = mass_all[np.intersect1d(track_ids_sec, ID_all, return_indices=True)[2]]
    #
    track_com_prog = np.average(track_pos_prog, weights=track_mass_prog, axis=0)
    track_com_sec = np.average(track_pos_sec, weights=track_mass_sec, axis=0)
    track_comvel_prog = np.average(track_vel_prog, weights=track_mass_prog, axis=0)
    track_comvel_sec = np.average(track_vel_sec, weights=track_mass_sec, axis=0)
    dist = np.linalg.norm(track_com_prog - track_com_sec)
    relvel = track_comvel_sec - track_comvel_prog
    relvelmag = np.linalg.norm(relvel) #in km/s
    #
    ds = load_ds(codetp, idx_i, pfs)
    #Compute the error on the dist and relvel mag calculation
    prog_comstd = std_weighted(track_pos_prog, track_mass_prog)
    sec_comstd = std_weighted(track_pos_sec, track_mass_sec)
    comrelstd = np.sqrt(prog_comstd**2 + sec_comstd**2)
    comrel = track_com_prog - track_com_sec
    dist_std = (1/dist)*np.sqrt(np.sum((comrel**2)*(comrelstd**2)))
    #
    dist = (dist*ds.units.code_length).to('kpc').v.tolist()
    dist_std = (dist_std*ds.units.code_length).to('kpc').v.tolist()
    #
    prog_comvel_std = std_weighted(track_vel_prog, track_mass_prog)
    sec_comvel_std = std_weighted(track_vel_sec, track_mass_sec)
    relvel_std = np.sqrt(prog_comvel_std**2 + sec_comvel_std**2)
    relvelmag_std = (1/relvelmag)*np.sqrt(np.sum((relvel**2)*(relvel_std**2)))
    #
    dist_data['dist'] = np.append(dist_data['dist'], dist)
    dist_data['dist_std'] = np.append(dist_data['dist_std'], dist_std)
    dist_data['relvel'].append(relvel)
    dist_data['relvelmag'] = np.append(dist_data['relvelmag'], relvelmag)
    dist_data['relvelmag_std'] = np.append(dist_data['relvelmag_std'], relvelmag_std)
    dist_data['idx'] = np.append(dist_data['idx'], idx_i)
    dist_data['time'] = np.append(dist_data['time'], time_list[idx_i]/1e3)
    dist_data['z'] = np.append(dist_data['z'], redshift_list[idx_i])
    dist_data['prog_pos_plot'].append(track_pos_prog)
    dist_data['sec_pos_plot'].append(track_pos_sec)
    dist_data['prog_vel_plot'].append(track_vel_prog)
    dist_data['sec_vel_plot'].append(track_vel_sec)
    dist_data['prog_mass_plot'].append(track_mass_prog)
    dist_data['sec_mass_plot'].append(track_mass_sec)
    dist_data['codelength_to_meters'] = np.append(dist_data['codelength_to_meters'], ds.length_unit.to('m'))

# Add additional variables for plotting and error analysis
dist_data['prog_com_plot'] = []
dist_data['sec_com_plot'] = []
dist_data['dist_mc'] = []
for k in range(len(dist_data['prog_pos_plot'])):
    dist_data['prog_com_plot'].append(np.average(dist_data['prog_pos_plot'][k],  weights=dist_data['prog_mass_plot'][k], axis=0))
    dist_data['sec_com_plot'].append(np.average(dist_data['sec_pos_plot'][k],  weights=dist_data['sec_mass_plot'][k], axis=0))
    #
    x = np.average(dist_data['prog_pos_plot'][k],  weights=dist_data['prog_mass_plot'][k], axis=0)
    sx = std_weighted(dist_data['prog_pos_plot'][k], dist_data['prog_mass_plot'][k])
    y = np.average(dist_data['sec_pos_plot'][k],  weights=dist_data['sec_mass_plot'][k], axis=0)
    sy = std_weighted(dist_data['sec_pos_plot'][k], dist_data['sec_mass_plot'][k])
    err = monte_carlo_distance_error(x, y, sx, sy)
    err *= dist_data['codelength_to_meters'][k]*3.24078e-20 #1m = 3.24078e-20kpc
    dist_data['dist_mc'].append(err)
dist_data['dist_mc'] = np.array(dist_data['dist_mc'])

# Add radius of the progenitor halo
dist_data['prog_radius']  = (np.array([get_haloradius(rawtree, prog_branch, idx, printerror=True) for idx in dist_data['idx']])*np.array(dist_data['codelength_to_meters'])*unyt.m).to('kpc').v

np.save('/work/hdd/bezm/tnguyen2/AGORA/analysis/dist_relvel_j_ProgBranch-%s_FirstMerger_%s_ver%s.npy' % (merger_number, codetp, halotree_ver), dist_data)