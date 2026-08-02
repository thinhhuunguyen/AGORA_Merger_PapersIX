import numpy as np
import matplotlib.pyplot as plt
import yt
from yt import YTArray
yt.set_log_level(0)
from scipy.spatial import ConvexHull, Delaunay
import os
    
import setup
from importlib import reload
reload(setup)

from setup import load_halotree_and_pfs, load_timings, load_ds, sec_branch_compute
from setup import codetp_list, label_list, color_list, marker_list


merger_number = '0'
halotree_ver = 2013

def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    #
    return hull.find_simplex(p)>=0

def compute_halfmass_radius(codetp, idx, sec_branch_on=False):
    #Loading data
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
    metadata_dir = '/work/hdd/bezm/tnguyen2/AGORA/%s/metadata/' % codetp
    assignment = np.load('/work/hdd/bezm/gtg115x/Halo_Finding/%s/star_id_%s_final_firstmerger.npy' % (codetp, halotree_ver), allow_pickle=True).tolist()
    hullv = np.load('/work/hdd/bezm/gtg115x/Halo_Finding/%s/hullv_%s_withPos_final.npy' % (codetp, halotree_ver), allow_pickle=True).tolist()
    #
    metadata = np.load(metadata_dir + 'star_metadata_allbox_%s.npy' % idx, allow_pickle=True).tolist()
    mass_all = metadata['mass']
    pos_all = metadata['pos']
    vel_all = metadata['vel']
    ID_all = metadata['ID']
    #
    if idx <= idx_begin:
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
        if sec_branch_on == True:
            sID = sec_ids_all
        else:
            sID = prog_ids_all
    else:
        sID = assignment['ids'][prog_branch][idx]

    spos = pos_all[np.intersect1d(sID, ID_all, return_indices=True)[2]]
    svel = vel_all[np.intersect1d(sID, ID_all, return_indices=True)[2]]
    smass = mass_all[np.intersect1d(sID, ID_all, return_indices=True)[2]]
    #
    if idx < idx_begin:
        ds = load_ds(codetp, idx, pfs)
        codelength_to_meters = (1*ds.units.code_length).to('m').v
        if sec_branch_on == True:
            hull_bool = in_hull(spos*codelength_to_meters, hullv[idx][sec_branch])
        else:
            hull_bool = in_hull(spos*codelength_to_meters, hullv[idx][prog_branch])
    else:
        if sec_branch_on == True:
            hull_bool = in_hull(spos*np.array(dist_data['codelength_to_meters'])[dist_data['idx'] == idx][0], hullv[idx][sec_branch])  
        else:
            hull_bool = in_hull(spos*np.array(dist_data['codelength_to_meters'])[dist_data['idx'] == idx][0], hullv[idx][prog_branch])
    spos = spos[hull_bool]
    svel = svel[hull_bool]
    smass = smass[hull_bool]
    #
    if idx in dist_data['idx']:
        center = np.array(dist_data['prog_com_plot'])[dist_data['idx'] == idx][0]
    elif idx == idx_begin - step:
        #This is obtained using the "premerger_center_and_velocity.py" file
        center, center2, v_bulk, v_bulk2, _ = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/premerger_centers_%s_ver2013_ver2.npy' % (codetp), allow_pickle=True).tolist().values()
    elif os.path.exists('/work/hdd/bezm/tnguyen2/AGORA/%s/radius_2000_%s/Galaxy_Halo_0_Snapshot_%s_comR2000.npy' % (codetp, codetp, idx)) == True:
        center = np.load('/work/hdd/bezm/tnguyen2/AGORA/%s/radius_2000_%s/Galaxy_Halo_0_Snapshot_%s_comR2000.npy' % (codetp, codetp, idx), allow_pickle=True).tolist()
        center = center['com']
        print('Change center')
    else:
        rawtree = np.load('/work/hdd/bezm/gtg115x/Halo_Finding/%s/halotree_%s_final.npy' % (codetp, halotree_ver), allow_pickle=True).tolist()
        center = rawtree[prog_branch][idx]['Halo_Center']
        print('Change center')
    
    #
    if sec_branch_on == True:
        sdist = np.linalg.norm(spos - center2, axis=1)
    else:
        sdist = np.linalg.norm(spos - center, axis=1)
    #
    #Calculating half-mass radius
    sdist_sort = np.sort(sdist)
    smass_sort = smass[np.argsort(sdist)]
    smass_cumsum = np.cumsum(smass_sort)
    smass_cumsum_percent = smass_cumsum/smass_cumsum[-1]
    rhalf = sdist_sort[np.argmin(abs(smass_cumsum_percent - 0.5))]
    if idx < idx_begin:
        rhalf = YTArray(rhalf*codelength_to_meters, 'm')
    else:
        rhalf = YTArray(rhalf*np.array(dist_data['codelength_to_meters'])[dist_data['idx'] == idx][0], 'm')
    #
    return rhalf.to('kpc'), spos, center


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Calculate the Half-mass radius data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if os.path.exists('/work/hdd/bezm/tnguyen2/AGORA/analysis/HalfMassRadius_data_ver2013_ver2.npy') == False:
    for codetp in codetp_list:
        redshift_list, time_list, pfs, step = load_halotree_and_pfs(codetp, halotree_ver, rawtree_skip=True)
        idx_begin, idx_endinfall, idx_1stpass, idx_maxdist, idx_cls, time_begin, time_endinfall, time_1stpass, time_maxdist, time_cls = load_timings(codetp, halotree_ver, merger_number)
        time_eval = time_1stpass + 0.6
        if codetp == 'GEAR':
            idx_eval = np.argmin(abs(time_eval - time_list/1e3)) - (np.argmin(abs(time_eval - time_list/1e3)) % step) - 1
        else:
            idx_eval = np.argmin(abs(time_eval - time_list/1e3)) - (np.argmin(abs(time_eval - time_list/1e3)) % step)
        #Calculate half-mass radius at pre-infall and equivalent timesteps
        rhalf_preinfall, spos_preinfall, center_preinfall = compute_halfmass_radius(codetp, idx_begin - step, sec_branch_on=False)
        rhalf_eval, spos_eval, center_eval = compute_halfmass_radius(codetp, idx_eval, sec_branch_on=False)
    output = {}
    output[codetp] = {}
    output[codetp]['rhalf_preinfall'] = rhalf_preinfall
    output[codetp]['rhalf_eval'] = rhalf_eval
    np.save('/work/hdd/bezm/tnguyen2/AGORA/analysis/HalfMassRadius_data_ver2013_ver2.npy', output)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Plotting Half-mass radius ratio vs burst fraction %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
font_size = 18
bf_list = [0.819, 0.648, 0.736, 0.464, np.nan, np.nan, 0.263, 0.799, 0.995] #This is obtained from the "Part-1_Paper/Calculate_BurstFraction_and_its_correlations.py" file

data_rhalf = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/HalfMassRadius_data_ver2013_ver2.npy', allow_pickle=True).tolist()

rhalf_preinfall_list = np.array([])
rhalf_eval_list = np.array([])

for codetp in codetp_list:
    rhalf_preinfall_list = np.append(rhalf_preinfall_list, data_rhalf[codetp]['rhalf_preinfall'].value)
    rhalf_eval_list = np.append(rhalf_eval_list, data_rhalf[codetp]['rhalf_eval'].value)

fig, ax = plt.subplots(figsize=(9/1.2,7.5/1.2))

for k in range(len(codetp_list)):
    if codetp_list[k] == 'GADGET3' or codetp_list[k] == 'GADGET4':
        continue
    ax.scatter(bf_list[k], rhalf_eval_list[k]/rhalf_preinfall_list[k], label=label_list[k], marker=marker_list[k], color=color_list[k],s=150)

ax.legend(loc='lower left', ncols=2, fontsize=font_size-3)
ax.set_xlabel('$f_{sb}$', fontsize=font_size)
ax.set_ylabel(r'$R_{1/2,\text{post/eq}}/R_{1/2,\text{pre-infall}}$', fontsize=font_size)
ax.axhline(1, color='k', linestyle='--')
ax.tick_params('both', labelsize=font_size)
ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xlim(0.2, 1.03)

plt.savefig('/work/hdd/bezm/tnguyen2/figures/AGORA/November2025/HalfMassRadiusRatio_vs_burstFraction_ver5.png', dpi=300, bbox_inches='tight')
