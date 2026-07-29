import numpy as np
import matplotlib.pyplot as plt
import yt
yt.set_log_level(0)
from scipy.interpolate import CubicSpline
import unyt
    
import setup
from importlib import reload
reload(setup)

from setup import load_timings
from setup import codetp_list, label_list, color_list


def std_weighted(values, weights):
    average = np.average(values, weights=weights, axis=0)
    variance = np.average((values - average)**2, weights=weights, axis=0)*len(weights)/(len(weights) - 1)
    return np.sqrt(variance)

def vel_dispersion_compute(velocities, mass):
    #Calculate the 3D velocity dispersion
    return np.sqrt(std_weighted(velocities[:,0], mass)**2 + std_weighted(velocities[:,1], mass)**2 + std_weighted(velocities[:,2], mass)**2) 


fig, axs = plt.subplots(ncols=9, nrows=5, figsize=(20,15), sharex=True)

font_size = 18
merger_number = '0'
halotree_ver = 2013

for i in range(len(codetp_list)):
    codetp = codetp_list[i]
    dist_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/dist_relvel_j_ProgBranch-%s_FirstMerger_%s_ver2013.npy' % (merger_number, codetp), allow_pickle=True).tolist()
    idx_begin, idx_endinfall, idx_1stpass, idx_maxdist, idx_cls, time_begin, time_endinfall, time_1stpass, time_maxdist, time_cls = load_timings(codetp, halotree_ver, merger_number)
    #
    # Distance/Radius row
    if codetp == 'RAMSES':
        #manually adjusted using CubicSpline on the whole data set (All_Properties_vs_Time_Plot.ipynb) --> Clip Index 118:125
        dist_data['prog_radius'][:7] *= np.array([10.76999422,  9.86647663,  3.42627787,  2.47154105,  2.37751782, 2.05280175,  1.04861337]) 
    elif codetp == 'CHANGA':
        #manually adjusted using CubicSpline on the whole data set (All_Properties_vs_Time_Plot.ipynb) --> Clip Index 71:75
        dist_data['prog_radius'][:3] *= np.array([3.10666631, 3.02595101, 2.203989  ]) 
    
    if codetp != 'RAMSES' and codetp != 'CHANGA' and codetp != 'ENZO':
        if codetp == 'GADGET3':
            clip_i, clip_f = 3, 9
        elif codetp == 'GADGET4':
            clip_i, clip_f = 1, 5
        elif codetp == 'GEAR':
            clip_i, clip_f = 2, 3
        elif codetp == 'GIZMO':
            clip_i, clip_f = 1, 8
        elif codetp == 'ART':
            clip_i, clip_f = 3, 8
        elif codetp == 'AREPO':
            clip_i, clip_f = 1, 10
        cs = CubicSpline(np.append(dist_data['time'][:clip_i], dist_data['time'][clip_f:]), np.append(dist_data['prog_radius'][:clip_i], dist_data['prog_radius'][clip_f:])) #correcting the sudden drop in radius
        dist_data['prog_radius'] = cs(dist_data['time'])
    
    axs[0,i].plot(dist_data['time'] - dist_data['time'][0], dist_data['dist']/dist_data['prog_radius'], color=color_list[i])
    axs[0,i].set_title(label_list[i], fontsize=font_size)
    axs[0,i].set_xlim(0, 1.4)
    axs[0,i].axvline(time_cls - dist_data['time'][0], color='k', linestyle='--')
    if i != 0:
        axs[0,i].set_yticklabels([])
    axs[0,i].set_ylim(-0.1,1)
    axs[0,i].tick_params('both', labelsize=font_size)
    if i == 0:
        axs[0,i].set_ylabel(r'$d/R_\text{200c, prim}$', fontsize=font_size)
    #
    # Velocity
    rawtree_prog = np.load('/work/hdd/bezm/gtg115x/Halo_Finding/%s/halotree_%s_final_onlyBranch0.npy' % (codetp, halotree_ver), allow_pickle=True).tolist()
    
    axs[1,i].plot(dist_data['time'] - dist_data['time'][0], dist_data['relvelmag'], color=color_list[i])
    axs[1,i].set_ylim(-20, 380)
    axs[1,i].set_yticks([0,100,200,300])
    axs[1,i].axvline(time_cls - dist_data['time'][0], color='k', linestyle='--')
    axs[1,i].tick_params('both', labelsize=font_size)
    if i == 0:
        axs[1,i].set_ylabel(r'$v_\text{rel}$ (km/s)', fontsize=font_size)
    if i != 0:
        axs[1,i].set_yticklabels([])
    #
    #Velocity dispersion row
    prog_velsigma = np.array([vel_dispersion_compute(dist_data['prog_vel_plot'][k], dist_data['prog_mass_plot'][k]) for k in range(len(dist_data['idx']))])
    sec_velsigma = np.array([vel_dispersion_compute(dist_data['sec_vel_plot'][k], dist_data['sec_mass_plot'][k]) for k in range(len(dist_data['idx']))])

    axs[2,i].plot(dist_data['time'] - dist_data['time'][0], prog_velsigma, color=color_list[i])
    axs[2,i].plot(dist_data['time'] - dist_data['time'][0], sec_velsigma, color=color_list[i], linestyle=':')
    axs[2,i].axvline(time_cls - dist_data['time'][0], color='k', linestyle='--')
    axs[2,i].set_ylim(40, 410)
    axs[2,i].set_yticks([100,200,300,400])
    if i != 0:
        axs[2,i].set_yticklabels([])
    axs[2,i].tick_params('both', labelsize=font_size)
    if i == 0:
        axs[2,i].set_ylabel(r'$\sigma$ (km/s)', fontsize=font_size)
    if i == 1:
        axs[2,i].plot([], [],'-',color='black', lw=2, label=r'prim')
        axs[2,i].plot([], [],':',color='black', lw=2, label=r'sec')
        axs[2,i].legend(ncols=1, loc='upper right')
    
    # Velocity/Velocity_Dispersion row
    axs[3,i].plot(dist_data['time'] - dist_data['time'][0], dist_data['relvelmag']/np.array(prog_velsigma), color=color_list[i])
    axs[3,i].plot(dist_data['time'] - dist_data['time'][0], dist_data['relvelmag']/np.array(sec_velsigma), color=color_list[i], linestyle=':')
    axs[3,i].set_ylim(-0.2, 6.5)
    axs[3,i].axvline(time_cls - dist_data['time'][0], color='k', linestyle='--')
    if i != 0:
        axs[3,i].set_yticklabels([])
    axs[3,i].tick_params('both', labelsize=font_size)
    if i == 0:
        axs[3,i].set_ylabel(r'$v_\text{rel}/\sigma$', fontsize=font_size)
    if i == 1:
        axs[3,i].plot([], [],'-',color='black', lw=2, label=r'prim')
        axs[3,i].plot([], [],':',color='black', lw=2, label=r'sec')
        axs[3,i].legend(ncols=1, loc='upper right')
    #
    # Angular momentum row
    relcom = (np.array(dist_data['sec_com_plot']) - np.array(dist_data['prog_com_plot']))*dist_data['codelength_to_meters'].v[:,np.newaxis]
    relcom = (relcom*unyt.m).to('kpc').value
    j = np.cross(relcom, dist_data['relvel']) #unit is kpc*km/s
    jmag = np.linalg.norm(j, axis=1)
    axs[4,i].plot(dist_data['time'] - dist_data['time'][0], jmag, color=color_list[i])
    axs[4,i].set_yscale('log')
    axs[4,i].set_ylim(5e-2, 5e3)
    axs[4,i].set_yticks([1e-1,1e0,1e1,1e2,1e3])
    if i != 0:
        axs[4,i].set_yticklabels([])
    axs[4,i].set_xticks([0, 0.5, 1], ['0','0.5','1'])
    axs[4,i].axvline(time_cls - dist_data['time'][0], color='k', linestyle='--')
    axs[4,i].tick_params('both', labelsize=font_size)
    if i == 1:
        axs[4,i].set_xlabel('Time since beginning of infall (Gyr)', fontsize=font_size, ha='center')
    if i == 0:
        axs[4,i].set_ylabel(r'$j_\bigstar$ (kpc$\cdot$km/s)', fontsize=font_size)
    
    #Testing the angular momentum condition in literature (j/j_i < 0.05)
    #idx_cross = dist_data['idx'][np.where(dist_data['dist'] < dist_data['prog_radius'])[0][0]]
    #print(codetp, idx_begin, idx_cross, 100*jmag[dist_data['idx'] == idx_cls][0]/jmag[dist_data['idx'] == idx_cross][0])
    #print(codetp, 100*jmag[dist_data['idx'] == idx_cls][0]/jmag[0])
                
plt.savefig('/work/hdd/bezm/tnguyen2/figures/AGORA/November2025/d_v_j_ProgBranch-%s_ver2013_ver3.png' % merger_number, dpi=300, bbox_inches='tight')
