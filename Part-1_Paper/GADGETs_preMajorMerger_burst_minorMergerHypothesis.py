import numpy as np
import matplotlib.pyplot as plt
import yt
yt.set_log_level(0)
import sys, os
from scipy.spatial import ConvexHull
from mpl_toolkits.axes_grid1 import AxesGrid

import setup
from importlib import reload
reload(setup)

from setup import load_halotree_and_pfs, load_timings, load_ds, sec_branch_compute
from setup import add_metallicity_fields, extract_and_order_snapshotIdx

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


#RUN THIS CODE FOR GADGET3 AND GADGET4, THEN MERGE THE TWO FIGURES VERTICALLY TOGETHER IN A THIRD-PARTY SOFTWARE OR LINUX COMMAND LINE.
codetp = sys.argv[1] #GADGET3 or GADGET4
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

metadata_dir = '/work/hdd/bezm/tnguyen2/AGORA/%s/metadata/' % codetp
assignment = np.load('/work/hdd/bezm/gtg115x/Halo_Finding/%s/star_id_%s_final_firstmerger.npy' % (codetp, halotree_ver), allow_pickle=True).tolist()
hullv = np.load('/work/hdd/bezm/gtg115x/Halo_Finding/%s/hullv_%s_withPos_final.npy' % (codetp, halotree_ver), allow_pickle=True).tolist()

plot_bound = 20 #kpc
outliers_threshold = 1.5
fontsize = 14
prj_list = ['z','x','y']


fig2 = plt.figure(figsize=(9*3,6*3))
grid = AxesGrid(
    fig2,
    (0.075, 0.075, 1, 1),
    nrows_ncols=(1, 4),
    axes_pad=0.35,
    label_mode="L",
    cbar_mode='each',
    cbar_location='bottom',
    cbar_pad="23%",
    aspect=True
)

j = 1 #plotting y vs z

if codetp == 'GADGET4':
    idx_list = [86, 98]
elif codetp == 'GADGET3':
    idx_list = [77, 98]

for k in range(len(idx_list)):
    idx = idx_list[k]
    ds = load_ds(codetp, idx, pfs)
    add_metallicity_fields(ds, codetp)
    codelength_to_kpc = (1*ds.units.code_length).to('kpc').v
    allstars = np.load(metadata_dir + '/' + 'star_metadata_allbox_%s.npy' % idx, allow_pickle=True).tolist()
    allpos = allstars['pos']
    allvel = allstars['vel']
    allmass = allstars['mass']
    allID = allstars['ID'].astype(int)
    bound_ID = assignment['ids']['0'][idx]
    bound_pos = allpos[np.intersect1d(bound_ID, allID, return_indices=True)[2]]
    bound_vel = allvel[np.intersect1d(bound_ID, allID, return_indices=True)[2]]
    bound_mass = allmass[np.intersect1d(bound_ID, allID, return_indices=True)[2]]
    center = determine_center(bound_pos, bound_vel, bound_mass)
    #%%%%%%%%%%%%%%% Plotting %%%%%%%%%%%%%%%%%%%%%%%
    prj = yt.ProjectionPlot(
        ds,
        prj_list[j],
        ("gas", "density"),
        center = center,
        width = 2*plot_bound/codelength_to_kpc,
        weight_field=("gas", "density")
    )
    prj.set_cmap(("gas", "density"), 'viridis')
    prj.set_font_size(fontsize)
    prj.set_unit(("gas", "density"), "g/cm**3")
    prj.set_zlim(("gas", "density"), 7e-27, 1e-21)
    prj.set_colorbar_label(("gas", "density"), r'$\rho_\text{gas} (\frac{\text{g}}{\text{cm}^{3}})$')
    prj.annotate_title('t = %.2f Gyr' % (float(time_list[idx])/1e3))
    if k == 0:
        if codetp == 'GADGET4':
            prj.annotate_text([0.07,0.87], 'GADGET-4', coord_system="axis", text_args={"color": "white", "fontsize":fontsize}) 
        elif codetp == 'GADGET3':
            prj.annotate_text([0.07,0.87], 'GADGET-3', coord_system="axis", text_args={"color": "white", "fontsize":fontsize}) 
    
    plot = prj.plots[("gas", "density")]
    plot.figure = fig2
    plot.axes = grid[2*k].axes
    plot.cax = grid.cbar_axes[2*k]
    prj.render()
    #
    #Plotting layers of the DM halo on top
    if j == 0:
        axis1, axis2 = 0, 1
    elif j == 1:
        axis1, axis2 = 1, 2
    elif j == 2:
        axis1, axis2 = 2, 0
    
    hullv_pos_prog = ((hullv[idx][prog_branch]*ds.units.m).to('code_length').v - center)*codelength_to_kpc
    hullv_pos_vert  = np.append(ConvexHull(hullv_pos_prog[:,[axis1,axis2]]).vertices, ConvexHull(hullv_pos_prog[:,[axis1,axis2]]).vertices[0])
    plot.axes.plot(hullv_pos_prog[hullv_pos_vert][:,axis1], hullv_pos_prog[hullv_pos_vert][:,axis2], color='white')
    #
    if codetp == 'GADGET4':
        target_branch = '0_160'
    elif codetp == 'GADGET3':
        target_branch = '0_190'
    
    hullv_pos_target = ((hullv[idx][target_branch]*ds.units.m).to('code_length').v - center)*codelength_to_kpc
    hullv_pos_target_vert  = np.append(ConvexHull(hullv_pos_target[:,[axis1,axis2]]).vertices, ConvexHull(hullv_pos_target[:,[axis1,axis2]]).vertices[0])
    plot.axes.plot(hullv_pos_target[hullv_pos_target_vert][:,axis1], hullv_pos_target[hullv_pos_target_vert][:,axis2], color='red')
    
    plot.axes.set_xlim(-20, 20)
    plot.axes.set_ylim(-20, 20)
    
    prj = yt.ProjectionPlot(
        ds,
        prj_list[j],
        ("gas", "agora_metallicity"),
        center = center,
        width = 2*plot_bound/codelength_to_kpc,
        weight_field=("gas", "density")
    )
    prj.set_cmap(("gas", "agora_metallicity"), 'cividis')
    prj.set_font_size(fontsize)
    prj.set_zlim(("gas", "agora_metallicity"), 2e-4, 1e-1)
    prj.set_colorbar_label(("gas", "agora_metallicity"), r'$Z_\text{gas} (Z_\odot)$')
    
    plot = prj.plots[("gas", "agora_metallicity")]
    plot.figure = fig2
    plot.axes = grid[2*k + 1].axes
    plot.cax = grid.cbar_axes[2*k + 1]
    plot.cax.set_label("Temperature [K]")
    prj.render()
    #
    #Plotting layers of the DM halo on top
    if j == 0:
        axis1, axis2 = 0, 1
    elif j == 1:
        axis1, axis2 = 1, 2
    elif j == 2:
        axis1, axis2 = 2, 0
    
    hullv_pos_prog = ((hullv[idx][prog_branch]*ds.units.m).to('code_length').v - center)*codelength_to_kpc
    hullv_pos_vert  = np.append(ConvexHull(hullv_pos_prog[:,[axis1,axis2]]).vertices, ConvexHull(hullv_pos_prog[:,[axis1,axis2]]).vertices[0])
    plot.axes.plot(hullv_pos_prog[hullv_pos_vert][:,axis1], hullv_pos_prog[hullv_pos_vert][:,axis2], color='white')
    #
    if codetp == 'GADGET4':
        target_branch = '0_160'
    elif codetp == 'GADGET3':
        target_branch = '0_190'
    
    hullv_pos_target = ((hullv[idx][target_branch]*ds.units.m).to('code_length').v - center)*codelength_to_kpc
    hullv_pos_target_vert  = np.append(ConvexHull(hullv_pos_target[:,[axis1,axis2]]).vertices, ConvexHull(hullv_pos_target[:,[axis1,axis2]]).vertices[0])
    plot.axes.plot(hullv_pos_target[hullv_pos_target_vert][:,axis1], hullv_pos_target[hullv_pos_target_vert][:,axis2], color='red')
    
    plot.axes.set_xlim(-20, 20)
    plot.axes.set_ylim(-20, 20)

output = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/SF_data_for_burstFractionCalc_%s_ver2013_ConvexHull.npy' % codetp, allow_pickle=True).tolist()

if os.path.exists('/work/hdd/bezm/tnguyen2/AGORA/analysis/premergerburst_halo_distanceTrack_%s_ver2013.npy' % codetp) == False:
    rawtree, _, _, _, _ = load_halotree_and_pfs(codetp, halotree_ver, rawtree_skip=False)
    if codetp == 'GADGET4':
        target_branch = '0_160'
    elif codetp == 'GADGET3':
        target_branch = '0_190'

    distance_target_list = []
    distance_norm_target_list = []
    time_target_list = []
    idx_target_list = []
    for idx_i in range(45, max(extract_and_order_snapshotIdx(rawtree, target_branch)) + 1):
        ds_i = load_ds(codetp, idx_i, pfs)
        distance_target_list.append((np.linalg.norm(rawtree[target_branch][idx_i]['Halo_Center'] - rawtree['0'][idx_i]['Halo_Center'])*ds_i.units.code_length).to('kpc').v)
        distance_norm_target_list.append(np.linalg.norm(rawtree[target_branch][idx_i]['Halo_Center'] - rawtree['0'][idx_i]['Halo_Center'])/rawtree['0'][idx_i]['Halo_Radius'])
        time_target_list.append(time_list[idx_i]/1e3)
        idx_target_list.append(idx_i)

    distance_target_list = np.array(distance_target_list)
    distance_norm_target_list = np.array(distance_norm_target_list)
    time_target_list = np.array(time_target_list)
    idx_target_list = np.array(idx_target_list)

    premergerburst_halo_output = {}
    premergerburst_halo_output['distance'] = distance_target_list
    premergerburst_halo_output['distance_norm'] = distance_norm_target_list
    premergerburst_halo_output['time'] = time_target_list
    premergerburst_halo_output['idx'] = idx_target_list
    np.save('/work/hdd/bezm/tnguyen2/AGORA/analysis/premergerburst_halo_distanceTrack_%s_ver2013.npy' % codetp, premergerburst_halo_output)
else:
    premergerburst_halo_output = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/premergerburst_halo_distanceTrack_%s_ver2013.npy' % codetp, allow_pickle=True).tolist()

ax_sfr = fig2.add_axes([1.15, 0.41, 0.25, 0.327])  
ax_sfr.plot(output['time'], output['sfr'], zorder=1)
ax_sfr.fill_betweenx(y=np.arange(-20,20,0.1), x1=time_begin,x2=2, color='grey', linestyle='-', alpha=0.3)
for k in range(len(idx_list)):
    idx = idx_list[k]
    ax_sfr.scatter(output['time'][output['idx'] == idx][0], output['sfr'][output['idx'] == idx][0], color='blue', s=50, zorder=10)
ax_sfr.set_xlim(0.5, 1.15)
ax_sfr.set_ylim(-0.3, 10)
ax_sfr.set_xlabel('Time since Big Bang (Gyr)', fontsize=12)
ax_sfr.set_ylabel(r'SFR $(M_\odot/\text{yr})$', fontsize=12)
ax_sfr.tick_params('both', labelsize=12)

ax_twin = ax_sfr.twinx()
ax_twin.plot(premergerburst_halo_output['time'], premergerburst_halo_output['distance_norm'], color='tab:orange', zorder=1)
for k in range(len(idx_list)):
    idx = idx_list[k]
    ax_twin.scatter(premergerburst_halo_output['time'][ premergerburst_halo_output['idx'] == idx ][0], premergerburst_halo_output['distance_norm'][ premergerburst_halo_output['idx'] == idx ][0], color='red', s=50, zorder=10)
ax_twin.set_ylim(0.1, 3)
ax_twin.set_ylabel(r'$d/R_\text{main halo}$', fontsize=12)
ax_twin.tick_params('both', labelsize=12)


fig2.savefig('/work/hdd/bezm/tnguyen2/figures/AGORA/November2025/%s_preMajorMerger_burst_minorMergerHypothesis_ver2.png' % (codetp), dpi=300, bbox_inches='tight')
    