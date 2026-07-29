import numpy as np
import yt
yt.set_log_level(0)
    
import setup
from importlib import reload
reload(setup)

from setup import load_halotree_and_pfs, load_ds, sec_branch_compute
from setup import codetp_list


def std_weighted(values, weights):
    average = np.average(values, weights=weights, axis=0)
    variance = np.average((values - average)**2, weights=weights, axis=0)*len(weights)/(len(weights) - 1)
    return np.sqrt(variance)

def vel_dispersion_compute(velocities, mass):
    #Calculate the 3D velocity dispersion
    return np.sqrt(std_weighted(velocities[:,0], mass)**2 + std_weighted(velocities[:,1], mass)**2 + std_weighted(velocities[:,2], mass)**2) 

def cls_timestep_compute(dist_data, cls_dist_lim = 0.025, cls_vel_lim = 0.1, softening_length=0.08):
    dist_list = np.array(dist_data['dist'])
    prog_r_list = np.array(dist_data['prog_radius'])
    relvelmag_list = np.array(dist_data['relvelmag'])
    prog_velsigma = np.array([vel_dispersion_compute(dist_data['prog_vel_plot'][k], dist_data['prog_mass_plot'][k]) for k in range(len(dist_data['idx']))])
    sec_velsigma = np.array([vel_dispersion_compute(dist_data['sec_vel_plot'][k], dist_data['sec_mass_plot'][k]) for k in range(len(dist_data['idx']))])
    # 
    cls_cond = np.where((np.array(dist_list) < np.maximum(cls_dist_lim*np.array(prog_r_list), 10*softening_length)) &
                   (np.array(relvelmag_list)/np.array(prog_velsigma) < cls_vel_lim) & 
                   (np.array(relvelmag_list)/np.array(sec_velsigma) < cls_vel_lim))[0][0]
    #
    cls_idx = dist_data['idx'][cls_cond]
    return cls_idx

def convert_pccm_to_pc(value, idx, codetp):
    pfs = np.loadtxt('/work/hdd/bezm/gtg115x/Halo_Finding/%s/pfs_allsnaps_%s.txt' % (codetp, halotree_ver), dtype=str)[:,0]
    ds = load_ds(codetp, idx, pfs)
    return (value*ds.units.pccm).to('pc').v.tolist()

merger_number = '0'
halotree_ver = 2013
cls_dist_lim = 0.025
cls_vel_lim = 0.1

for j in np.flip(range(len(codetp_list))):
    codetp = codetp_list[j]
    rawtree, redshift_list, time_list, pfs, step = load_halotree_and_pfs(codetp, halotree_ver)
    prog_branch, sec_branch, sec_branch_2 = sec_branch_compute(codetp, merger_number)
    #
    if codetp == 'CHANGA':
        step = 2
    elif codetp == 'GEAR':
        step = 3
    else:
        step = 1
    dist_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/dist_relvel_j_ProgBranch-%s_FirstMerger_%s_ver2013.npy' % (merger_number, codetp), allow_pickle=True).tolist()
    time_list = np.loadtxt('/work/hdd/bezm/gtg115x/Halo_Finding/%s/pfs_allsnaps_%s.txt' % (codetp, halotree_ver), dtype=str)[:,2].astype(float)
    #
    start_time = time_list[dist_data['idx'][0]]/1e3
    if codetp == 'CHANGA' or codetp == 'GADGET3' or codetp == 'GADGET4' or codetp == 'GEAR' or codetp == 'GIZMO' or codetp == 'AREPO':
        epsilon = 80/1000    
    elif codetp == 'ENZO' or codetp == 'ART' or codetp == 'RAMSES':
        epsilon = np.array([])
        for idx in dist_data['idx']:
            epsilon = np.append(epsilon, convert_pccm_to_pc(163, idx, codetp))/1000
    end_time = time_list[cls_timestep_compute(dist_data, cls_dist_lim , cls_vel_lim, epsilon)]/1e3
    # Printing the code, the coalescence timestep, and the coalescence time in Gyr
    print(codetp, cls_timestep_compute(dist_data, cls_dist_lim , cls_vel_lim, epsilon), end_time)