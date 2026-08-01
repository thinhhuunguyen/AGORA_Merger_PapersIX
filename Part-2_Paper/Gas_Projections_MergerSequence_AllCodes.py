import os

import numpy as np
import matplotlib.pyplot as plt
import yt
yt.set_log_level(0)
import h5py
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import AxesGrid
    
import setup
from importlib import reload
reload(setup)

from setup import load_halotree_and_pfs, load_timings, load_ds, sec_branch_compute
from setup import codetp_list, label_list

def get_trajectory(codetp, plot_bound = 15):
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Loading data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    redshift_list, time_list, pfs, step = load_halotree_and_pfs(codetp, halotree_ver, rawtree_skip=True)
    dist_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/dist_relvel_j_ProgBranch-%s_FirstMerger_%s_ver2013.npy' % (merger_number, codetp), allow_pickle=True).tolist()
    prog_branch, sec_branch, sec_branch_2 = sec_branch_compute(codetp, merger_number)
    idx_begin, idx_endinfall, idx_1stpass, idx_maxdist, idx_cls, time_begin, time_endinfall, time_1stpass, time_maxdist, time_cls = load_timings(codetp, halotree_ver, merger_number)
    time_eval = time_1stpass + 0.6
    if codetp == 'GEAR':
        idx_eval = np.argmin(abs(time_eval - time_list/1e3)) - (np.argmin(abs(time_eval - time_list/1e3)) % step) - 1
    else:
        idx_eval = np.argmin(abs(time_eval - time_list/1e3)) - (np.argmin(abs(time_eval - time_list/1e3)) % step)
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Calculate the projection axis %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    gal_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/%s/radius_2000_%s/Galaxy_Halo_%s_Snapshot_%s_comR2000.npy' % (codetp, codetp, prog_branch, idx_eval), allow_pickle=True).tolist()
    try:
        gal_center = gal_data['com']
    except:
        gal_center = gal_data['gal_com']
    
    metadata_dir = '/work/hdd/bezm/tnguyen2/AGORA/%s/metadata/' % codetp
    metadata = np.load(metadata_dir + 'star_metadata_allbox_%s.npy' % idx_eval, allow_pickle=True).tolist()
    mass_all = metadata['mass']
    pos_all = metadata['pos']
    vel_all = metadata['vel'] #unit of km/s
    ID_all = metadata['ID']
    dist_all = np.linalg.norm(pos_all - gal_center, axis=1)
    # The file in the following line is generate by the script "circularity_distribution_analysis.py" in the same directory
    ID_cls = list(np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/circularity_decomposed_ProgBranch-%s_Idx_%s_%s_ver2013.npy' % (merger_number, idx_eval, codetp), allow_pickle=True).tolist().values())[12]
    cls_bool = np.intersect1d(ID_cls, ID_all, return_indices=True)[2]
    pos_cls = pos_all[cls_bool]
    vel_cls = vel_all[cls_bool]
    mass_cls = mass_all[cls_bool]
    relpos_cls = pos_cls - gal_center
    com_vel = np.sum(vel_cls*mass_cls[:,np.newaxis],axis=0)/np.sum(mass_cls)
    relvel_cls = vel_cls - com_vel
    #Calculate the angular momentum axis of the galaxy using only young stars (because they are the main component of the disk (if exists))
    J_each = mass_cls[:,np.newaxis]*np.cross(relpos_cls, relvel_cls)
    com_J = np.sum(J_each, axis=0)
    com_j = com_J/np.sum(mass_cls)
    com_j_unitvec = com_j/np.linalg.norm(com_j)
    east_vec = np.array(dist_data['sec_com_plot'])[dist_data['idx'] == idx_1stpass][0] - np.array(dist_data['prog_com_plot'])[dist_data['idx'] == idx_1stpass][0]
    east_unitvec = east_vec/np.linalg.norm(east_vec)
    north_unitvec = np.cross(com_j_unitvec, east_unitvec)
    rel_vec_cartesian = (np.array(dist_data['sec_com_plot']) - np.array(dist_data['prog_com_plot']))*np.array(dist_data['codelength_to_meters'])[:,np.newaxis]*3.24077929e-20 # in kpc
    rel_vec_east = np.dot(rel_vec_cartesian, east_unitvec)
    rel_vec_north = np.dot(rel_vec_cartesian, north_unitvec)
    if codetp == 'ART' or codetp == 'GADGET3' or codetp == 'ENZO' or codetp == 'RAMSES' or codetp == 'AREPO':
        idx_secinframe = idx_1stpass - 2*step #the timestep where the secondary galaxy shows up in the plotting frame
    elif codetp == 'GADGET4' or codetp == 'CHANGA' or codetp == 'GEAR' or codetp == 'GIZMO':
        idx_secinframe = idx_1stpass - 1*step
    if codetp == 'ENZO' or codetp == 'GIZMO':
        idx_southwest = dist_data['idx'][(rel_vec_north > -plot_bound)*(rel_vec_east > -plot_bound)][0] - 2*step #15 is the plot bound
    elif codetp == 'GADGET3':
        idx_southwest = dist_data['idx'][(rel_vec_north > -plot_bound)*(rel_vec_east > -plot_bound)][0] - 3*step #15 is the plot bound
    elif codetp == 'CHANGA' or codetp == 'GADGET4' or codetp == 'GEAR' or codetp == 'AREPO' or codetp == 'RAMSES' or codetp == 'ART':
        idx_southwest = dist_data['idx'][(rel_vec_north > -plot_bound)*(rel_vec_east > -plot_bound)][0] - step #15 is the plot bound
    southwest_vec = np.array(dist_data['sec_com_plot'])[dist_data['idx'] == idx_southwest][0] - np.array(dist_data['prog_com_plot'])[dist_data['idx'] == idx_southwest][0]
    southwest_unitvec = southwest_vec/np.linalg.norm(southwest_vec)
    northeast_unitvec = southwest_unitvec*(-1)
    southeast_univec = np.cross(com_j_unitvec, southwest_unitvec)
    east_vec = northeast_unitvec + southeast_univec
    east_unitvec = east_vec/np.linalg.norm(east_vec)
    north_unitvec = np.cross(com_j_unitvec, east_unitvec)
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Calculate the projected trajectory %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    rel_vec_cartesian = (np.array(dist_data['sec_com_plot']) - np.array(dist_data['prog_com_plot']))*np.array(dist_data['codelength_to_meters'])[:,np.newaxis]*3.24077929e-20 # in kpc
    rel_vec_cartesian = rel_vec_cartesian[dist_data['idx'] <= idx_cls]
    rel_vec_east = np.dot(rel_vec_cartesian, east_unitvec)
    rel_vec_north = np.dot(rel_vec_cartesian, north_unitvec)
    #
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Plotting the gas projection and resave the data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def plot_gasprojection(idx, codetp, idx_plotloc):
        ds = load_ds(codetp, idx, pfs)
        codelength_to_kpc = (1*ds.units.code_length).to('kpc').v
        center = np.array(dist_data['prog_com_plot'])[dist_data['idx'] == idx][0]
    
        if codetp == 'ART' or codetp == 'ENZO' or codetp == 'RAMSES':
            reg = ds.sphere(center, (3*plot_bound/codelength_to_kpc, 'code_length'))
            prj = yt.OffAxisProjectionPlot(
                ds,
                com_j_unitvec,
                ("gas", "density"),
                center = center,
                width = 2*plot_bound/codelength_to_kpc,
                weight_field=("gas", "density"), 
                fontsize = font_size,
                north_vector=north_unitvec,
                data_source = reg
            )
        else: 
            prj = yt.OffAxisProjectionPlot(
                ds,
                com_j_unitvec,
                ("gas", "density"),
                center = center,
                width = 2*plot_bound/codelength_to_kpc,
                fontsize = font_size,
                north_vector=north_unitvec,
                weight_field=("gas", "density")
            )
        prj.set_cmap(("gas", "density"), 'viridis')
        prj.set_font_size(font_size)
        prj.set_unit(("gas", "density"), "g/cm**3")
        prj.set_zlim(("gas", "density"), 7e-27, 1e-18)
        #Saving the plot data
        image_data = np.array(prj.frb[("gas", "density")])
        output_path = "/work/hdd/bezm/tnguyen2/AGORA/analysis/Gas_Project_MergerSequence_AllCodes_datasave/%s_idxplotloc_%s_ver2013.h5" % (codetp, idx_plotloc)
        with h5py.File(output_path, "w") as f:
            f.create_dataset("density", data=image_data)
    for idx_plotloc in range(0,6):
        if os.path.exists("/work/hdd/bezm/tnguyen2/AGORA/analysis/Gas_Project_MergerSequence_AllCodes_datasave/%s_idxplotloc_%s_ver2013.h5" % (codetp, idx_plotloc)) == False:
            if idx_plotloc == 0:
                plot_gasprojection(idx_begin, codetp, idx_plotloc)
            elif idx_plotloc == 1:
                plot_gasprojection(idx_secinframe, codetp, idx_plotloc)
            elif idx_plotloc == 2:
                plot_gasprojection(idx_1stpass, codetp, idx_plotloc)
            elif idx_plotloc == 3:
                plot_gasprojection(idx_maxdist, codetp, idx_plotloc)
            elif idx_plotloc == 4:
                plot_gasprojection(idx_cls, codetp, idx_plotloc)
            elif idx_plotloc == 5:
                plot_gasprojection(idx_eval, codetp, idx_plotloc)
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Plotting the gas projection and resave the data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    return rel_vec_east, rel_vec_north, (dist_data['idx'])[dist_data['idx'] <= idx_cls], [idx_begin, idx_secinframe, idx_1stpass, idx_maxdist, idx_cls, idx_eval]


merger_number = '0'
halotree_ver = 2013

fig = plt.figure()

plot_bound = 15 #kpc
outliers_threshold = 1.5
font_size = 18

zlim_min, zlim_max = 7e-27, 1e-21

grid = AxesGrid(
    fig,
    (0.1, 0.1, 0.87*2, 1.75*2),
    nrows_ncols=(len(codetp_list), 6),
    axes_pad=0,
    label_mode="1",
    share_all=True,
    aspect=False,
) 

for codetp in codetp_list:
    codetp_plotloc = codetp_list.index(codetp) 
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Loading projected trajectory data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    rel_vec_east, rel_vec_north, idx_list, idx_plot = get_trajectory(codetp)
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Loading projected trajectory data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #
    ax_title = grid[5 + 6*codetp_plotloc].axes
    if codetp == 'ART' or codetp == 'GADGET3' or codetp == 'GADGET4' or codetp == 'GIZMO':
        title_color = 'blue'
    else:
        title_color = 'black'
    if codetp == 'ART' or codetp == 'RAMSES' or codetp == 'GADGET3' or codetp == 'GEAR':
        ax_title.text(1.1, 0.5, label_list[codetp_plotloc] + '*', transform=ax_title.transAxes, color=title_color, fontsize=font_size, va='center', ha='left')
    elif codetp == 'CHANGA':
        ax_title.text(1.1, 0.5, r'$\text{%s}^{\dagger}$' % label_list[codetp_plotloc], transform=ax_title.transAxes, color=title_color, fontsize=font_size, va='center', ha='left')
    else:
        ax_title.text(1.1, 0.5, '%s' % (label_list[codetp_plotloc]), transform=ax_title.transAxes, color=title_color, fontsize=font_size, va='center', ha='left')
    #
    for idx_plotloc in range(0, 6):
        input_path = "/work/hdd/bezm/tnguyen2/AGORA/analysis/Gas_Project_MergerSequence_AllCodes_datasave/%s_idxplotloc_%s_ver2013.h5" % (codetp, idx_plotloc)
        with h5py.File(input_path, "r") as f:
            density_image = f["density"][:]    
        
        ax = grid[idx_plotloc + 6*codetp_plotloc].axes
        
        im = ax.imshow(
            density_image,
            origin="lower",
            cmap="viridis",
            norm=mcolors.LogNorm(vmin=zlim_min, vmax=zlim_max),
            extent=[-plot_bound, plot_bound, -plot_bound, plot_bound]
        )
        
        ax.set_xticks([-15, -10, -5, 0, 5, 10, 15], ['','-10','','0','','10',''])
        ax.set_yticks([-15, -10, -5, 0, 5, 10, 15], ['','-10','','0','','10',''])
        ax.tick_params(
            which='both',
            left=True, right=True,
            top=True, bottom=True,
            direction='in',                     # ticks pointing inward (common in ApJ style)
            colors='white',
            labelcolor='black',
            labelsize=font_size
        )
        
        if codetp_plotloc == 8 and idx_plotloc == 0:
            ax.set_xlabel("kpc", fontsize=font_size)
            ax.set_ylabel("kpc", fontsize=font_size)
        #Plotting projected trajectory
        if idx_plotloc < 5 and idx_plotloc > 0:
            ax.plot(rel_vec_east, rel_vec_north, color='white', linestyle='--', alpha=0.55)
            ax.scatter(rel_vec_east[idx_list == idx_plot[idx_plotloc]], rel_vec_north[idx_list == idx_plot[idx_plotloc]], color='red', s=20, alpha=0.7)
            ax.set_xlim(-plot_bound, plot_bound)
            ax.set_ylim(-plot_bound, plot_bound)

        if codetp_plotloc == 0:
            if idx_plotloc == 0:
                ax.text(0.13, 0.8, r"$t_\text{start}$", transform=ax.transAxes, va='center', ha='left', fontsize=font_size - 2, color='white')
            elif idx_plotloc == 1:
                ax.text(0.13, 0.8, r"$t_\text{start}$ < t < $t_\text{fp}$", transform=ax.transAxes, va='center', ha='left', fontsize=font_size - 2, color='white')
            elif idx_plotloc == 2:
                ax.text(0.13, 0.8, r"$t_\text{fp}$", transform=ax.transAxes, va='center', ha='left', fontsize=font_size - 2, color='white')
            elif idx_plotloc == 3:
                ax.text(0.13, 0.8, r"$t_\text{max}$", transform=ax.transAxes, va='center', ha='left', fontsize=font_size - 2, color='white')
            elif idx_plotloc == 4:
                ax.text(0.13, 0.8, r"$t_\text{cls}$", transform=ax.transAxes, va='center', ha='left', fontsize=font_size - 2, color='white')
            elif idx_plotloc == 5:
                ax.text(0.13, 0.8, r"$t_\text{post/eq}$", transform=ax.transAxes, va='center', ha='left', fontsize=font_size - 2, color='white')
            
top_ax    = grid[0].axes                          # row 0, col 0
bottom_ax = grid[6 * (len(codetp_list) - 1)].axes # last row, col 0

top_bbox    = top_ax.get_position()
bottom_bbox = bottom_ax.get_position()

cbar_width  = 0.08                        # tweak to taste
cbar_gap    = 0.04                         # gap between colorbar and grid left edge
cbar_left   = top_bbox.x0 - cbar_gap - cbar_width
cbar_bottom = bottom_bbox.y0 + 0.43
cbar_height = top_bbox.y1 - bottom_bbox.y0 - 0.43

cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])

cbar = fig.colorbar(im, cax=cax, orientation="vertical")
cbar.ax.yaxis.set_ticks_position('left')
cbar.ax.yaxis.set_label_position('left')
cbar.set_label(
    r'$\rho_\mathrm{gas}\,(\mathrm{g}\,\mathrm{cm}^{-3})$',
    fontsize=font_size
)
cbar.ax.tick_params(labelsize=font_size)

fig.savefig('/work/hdd/bezm/tnguyen2/figures/AGORA/November2025/Gas_Project_MergerSequence_AllCodes_ver2013_ver5.png', dpi=300, bbox_inches='tight')