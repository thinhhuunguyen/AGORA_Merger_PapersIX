import numpy as np
import matplotlib.pyplot as plt
import yt
yt.set_log_level(0)
    
import setup
from importlib import reload
reload(setup)

from setup import load_halotree_and_pfs, load_timings, sec_branch_compute
from setup import codetp_list, label_list, color_list

def angle_between(v1, v2):
    """ Returns the angle in degrees between unit vectors 'v1' and 'v2' """
    # Clip the dot product to [-1.0, 1.0] to avoid errors from floating point jitter
    dot_product = np.dot(v1, v2)
    # Calculate angle in radians and convert to degrees
    radians = np.arccos(dot_product)
    return np.degrees(radians)


fig = plt.figure(figsize=(7.3*2, 3*2))
plt.subplots_adjust(wspace=-0.1)
plt.subplots_adjust(hspace=0)
font_size = 17

axs = []
for i in range(9):
    ax = fig.add_subplot(2, 5, i+1, projection='3d')
    axs.append(ax)

ticks = [-1,-0.5, 0, 0.5, 1]
ticks_label = ['-1', '', 0, '', 1]

for i in range(len(codetp_list)):
    codetp = codetp_list[i]
    metadata_dir = '/work/hdd/bezm/tnguyen2/AGORA/%s/metadata/' % codetp
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
    time_eval = time_1stpass + 0.6
    
    if codetp == 'GEAR':
        idx_eval = np.argmin(abs(time_eval - time_list/1e3)) - (np.argmin(abs(time_eval - time_list/1e3)) % step) - 1
    else:
        idx_eval = np.argmin(abs(time_eval - time_list/1e3)) - (np.argmin(abs(time_eval - time_list/1e3)) % step)
    #
    ### Rotational angular momentum at t_eval
    gal_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/%s/radius_2000_%s/Galaxy_Halo_%s_Snapshot_%s_comR2000.npy' % (codetp, codetp, prog_branch, idx_eval), allow_pickle=True).tolist()
    try:
        gal_center = gal_data['com']
    except:
        gal_center = gal_data['gal_com']
    
    metadata = np.load(metadata_dir + 'star_metadata_allbox_%s.npy' % idx_eval, allow_pickle=True).tolist()
    mass_all = metadata['mass']
    pos_all = metadata['pos']
    vel_all = metadata['vel'] #unit of km/s
    ID_all = metadata['ID']
    dist_all = np.linalg.norm(pos_all - gal_center, axis=1)

    epsilon_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/circularity_ProgBranch-%s_Idx_%s_%s_ver2013_Abadi2003method.npy' % (codetp, prog_branch, idx_eval), allow_pickle=True).tolist()
    ID_disk = epsilon_data['ID'][(epsilon_data['label'] == 3) | (epsilon_data['label'] == 4) | (epsilon_data['label'] == -1)]

    disk_bool = np.intersect1d(ID_disk, ID_all, return_indices=True)[2]
    pos_disk = pos_all[disk_bool]
    vel_disk = vel_all[disk_bool]
    mass_disk = mass_all[disk_bool]
    relpos_disk = pos_disk - gal_center
    com_vel = np.average(np.array(dist_data['prog_vel_plot'])[dist_data['idx'] == idx_eval][0], weights=np.array(dist_data['prog_mass_plot'])[dist_data['idx'] == idx_eval][0], axis=0)
    relvel_disk = vel_disk - com_vel
    #Calculate the angular momentum axis of the galaxy using only young stars (because they are the main component of the disk (if exists))
    J_each = mass_disk[:,np.newaxis]*np.cross(relpos_disk, relvel_disk)
    com_J = np.sum(J_each, axis=0)
    com_j = com_J/np.sum(mass_disk)
    com_j_unitvec = com_j/np.linalg.norm(com_j)
    com_j_unitvec_eval = com_j_unitvec
    #
    ### Rotational angular momentum at t_start
    center_begin = np.array(dist_data['prog_com_plot'])[dist_data['idx'] == idx_begin][0]
    
    metadata = np.load(metadata_dir + 'star_metadata_allbox_%s.npy' % idx_begin, allow_pickle=True).tolist()
    mass_all = metadata['mass']
    pos_all = metadata['pos']
    vel_all = metadata['vel'] #unit of km/s
    ID_all = metadata['ID']
    dist_all = np.linalg.norm(pos_all - center_begin, axis=1)

    assignment = np.load('/work/hdd/bezm/gtg115x/Halo_Finding/%s/star_id_%s_final_firstmerger.npy' % (codetp, halotree_ver), allow_pickle=True).tolist()
    prog_ID = assignment['ids'][prog_branch][idx_begin]
    gal_bool = np.intersect1d(prog_ID, ID_all, return_indices=True)[2]
    mass_gal = mass_all[gal_bool]
    pos_gal = pos_all[gal_bool]
    vel_gal = vel_all[gal_bool]
    dist_gal = dist_all[gal_bool]
    
    j_bool = dist_gal < 0.2*(np.array(dist_data['prog_radius'])[dist_data['idx'] == idx_begin][0]/3.24077929e-20)/np.array(dist_data['codelength_to_meters'])[dist_data['idx'] == idx_begin][0]
    
    relpos_gal = pos_gal[j_bool] - center_begin
    com_vel = np.sum(vel_gal[j_bool]*mass_gal[j_bool][:,np.newaxis],axis=0)/np.sum(mass_gal[j_bool])
    relvel_gal = vel_gal[j_bool] - com_vel
    J_each = mass_gal[j_bool][:,np.newaxis]*np.cross(relpos_gal, relvel_gal)
    com_J = np.sum(J_each, axis=0)
    com_j = com_J/np.sum(mass_gal[j_bool])
    com_j_unitvec = com_j/np.linalg.norm(com_j)
    com_j_unitvec_begin = com_j_unitvec
    #
    ### Orbital angular momentum at t_fp (the first periapsis)
    relpos_vec = np.array(dist_data['sec_com_plot'])[dist_data['idx'] == idx_1stpass][0] - np.array(dist_data['prog_com_plot'])[dist_data['idx'] == idx_1stpass][0]
    relvel_vec = np.array(dist_data['relvel'])[dist_data['idx'] == idx_1stpass][0]
    orbital_j = np.cross(relpos_vec, relvel_vec)
    orbital_j_unitvec = orbital_j/np.linalg.norm(orbital_j)
    #Calculate angle between j vectors
    theta_eval_begin = angle_between(com_j_unitvec_eval, com_j_unitvec_begin)
    theta_eval_orbit = angle_between(com_j_unitvec_eval, orbital_j_unitvec)
    #
    axs[i].quiver(0,0,0, com_j_unitvec_begin[0], com_j_unitvec_begin[1], com_j_unitvec_begin[2],
                  linestyle='-', linewidth=2, color='black')
    axs[i].quiver(0,0,0, com_j_unitvec_eval[0], com_j_unitvec_eval[1], com_j_unitvec_eval[2],
                  linestyle='-', linewidth=2, color=color_list[i])
    axs[i].quiver(0,0,0, orbital_j_unitvec[0], orbital_j_unitvec[1], orbital_j_unitvec[2],
                  linestyle='--', linewidth=2, color=color_list[i])


    axs[i].text2D(0.1, 0.92, s=r'$\Delta\theta_\text{e-s} = %.0f^{\circ}$' % theta_eval_begin, fontsize=font_size-3, transform=axs[i].transAxes, va='top')
    axs[i].text2D(0.1, 0.82, s=r'$\Delta\theta_\text{e-o} = %.0f^{\circ}$' % theta_eval_orbit, fontsize=font_size-3, transform=axs[i].transAxes, va='top')

    if codetp == 'ART' or codetp == 'GADGET3' or codetp == 'GADGET4' or codetp == 'GIZMO':
        title_color = 'blue'
    else:
        title_color = 'black'
    if codetp == 'ART' or codetp == 'RAMSES' or codetp == 'GADGET3' or codetp == 'GEAR':
        axs[i].set_title(label_list[i] + '*', fontsize=font_size, color=title_color)
    elif codetp == 'CHANGA':
        axs[i].set_title(r'$\text{%s}^{\dagger}$' % label_list[i], fontsize=font_size, color=title_color)
    else:
        axs[i].set_title(label_list[i], fontsize=font_size, color=title_color)
    axs[i].set_xlim([-1, 1])
    axs[i].set_ylim([-1, 1])
    axs[i].set_zlim([-1, 1])
    axs[i].set_xticks(ticks, ticks_label)
    axs[i].set_yticks(ticks, ticks_label)
    axs[i].set_zticks(ticks, ticks_label)
    axs[i].tick_params('both', labelsize=font_size-4)
    axs[i].view_init(10, 5+65)
    
    if i != 5:
        axs[i].set_xticklabels([])
        axs[i].set_yticklabels([])
        axs[i].set_zticklabels([])
    else:
        axs[i].set_xlabel('x', fontsize=font_size)
        axs[i].set_ylabel('y', fontsize=font_size)
        axs[i].zaxis.set_label_text('z', fontsize=font_size)
        axs[i].zaxis.labelpad = -5

    if i == 0:
        axs[i].quiver(0,0,0,0,0,0, linestyle='-', linewidth=2, color='k', label=r'$j_\text{start}$')
        axs[i].quiver(0,0,0,0,0,0, linestyle='-',  linewidth=2, color=color_list[1], label=r'$j_\text{post/eq}$')
        axs[i].quiver(0,0,0,0,0,0, linestyle='--', linewidth=2, color=color_list[1], label=r'$j_\text{orb}$')
        handles1, labels1 = axs[i].get_legend_handles_labels()
        fig.legend(handles1, labels1, bbox_to_anchor=(0.81, 0.35), loc='center', ncol=1, fontsize=font_size)
    
plt.savefig('/work/hdd/bezm/tnguyen2/figures/AGORA/November2025/angular_momentum_direction_change_ver2013_ver7.png', dpi=300, bbox_inches='tight')