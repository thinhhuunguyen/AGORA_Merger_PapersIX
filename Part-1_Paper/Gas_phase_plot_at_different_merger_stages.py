import numpy as np
import matplotlib.pyplot as plt
import yt
yt.set_log_level(0)
from scipy.spatial.distance import cdist
from astropy.constants import G
from tqdm import tqdm
import os
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu, bws_test
from scipy.stats import norm
from scipy.spatial import ConvexHull, Delaunay
from scipy.interpolate import CubicSpline
from yt.data_objects.particle_filters import add_particle_filter
if int((yt.__version__).split('.')[0]) >= 4 and int((yt.__version__).split('.')[1]) >= 2: #ParticleUnion is only available in yt 4.2 and later
    from yt.data_objects.unions import ParticleUnion
else:
    from yt.data_objects.unions import Union as ParticleUnion
import unyt
from mpl_toolkits.axes_grid1 import AxesGrid, ImageGrid
from matplotlib.colors import LogNorm
    
import setup
from importlib import reload
reload(setup)

from setup import load_halotree_and_pfs, load_timings, load_ds, load_tracking_dist_data, sec_branch_compute
from setup import gas_name_dict, gas_temp_dict, gas_mass_dict
from setup import add_metallicity_fields, add_cooling_fields, add_radialdist_to_halocenter_field, extract_and_order_snapshotIdx
from setup import get_haloradius, infall_timestep_compute_hullv
from setup import codetp_list, label_list, color_list, marker_list
from setup import codetp_list10, label_list10, color_list10, marker_list10

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

def obtain_gas_properties(idx, codetp, pfs, hullv, rawtree, retrieve_sec = False):
    #
    ds = load_ds(codetp, idx, pfs)
    # Gas mass
    if codetp == 'ART' or codetp == 'ENZO' or codetp == 'RAMSES':
        prog_reg = ds.sphere(rawtree[prog_branch][idx]['Halo_Center'], 2*rawtree[prog_branch][idx]['Halo_Radius'])
        gas_x = prog_reg[(gas_name_dict[codetp],"x")].to('m').v
        gas_y = prog_reg[(gas_name_dict[codetp],"y")].to('m').v
        gas_z = prog_reg[(gas_name_dict[codetp],"z")].to('m').v
        prog_gascoor = np.vstack((gas_x, gas_y, gas_z)).T
        prog_gasmass = prog_reg[(gas_name_dict[codetp],gas_mass_dict[codetp])].to('Msun').v
        prog_gastemp = prog_reg[(gas_name_dict[codetp],gas_temp_dict[codetp])].v
        prog_gasden = prog_reg['gas','density'].to('g/cm**3').v

        prog_bool = in_hull(prog_gascoor, hullv[idx][prog_branch])
        if np.sum(prog_bool)/len(prog_bool) > 0.75:
            print('Need to select a bigger region to conver the convex hull', np.sum(prog_bool)/len(prog_bool))

        prog_center = np.array(dist_data['prog_com_plot'])[np.array(dist_data['idx']) == idx][0]*np.array(dist_data['codelength_to_meters'])[np.array(dist_data['idx']) == idx][0]
        prog_gascoor = prog_gascoor[prog_bool]
        prog_gasdist = np.linalg.norm(prog_gascoor - prog_center, axis=1)
        prog_gasdist = (prog_gasdist*unyt.m).to('kpc').v
        prog_gasmass = prog_gasmass[prog_bool]
        prog_gastemp = prog_gastemp[prog_bool]
        prog_gasden = prog_gasden[prog_bool]
        #
        if retrieve_sec == True:
            sec_reg = ds.sphere(rawtree[sec_branch][idx]['Halo_Center'], 1.5*rawtree[sec_branch][idx]['Halo_Radius'])
            gas_x = sec_reg[("gas","x")].to('m').v
            gas_y = sec_reg[("gas","y")].to('m').v
            gas_z = sec_reg[("gas","z")].to('m').v
            sec_gascoor = np.vstack((gas_x, gas_y, gas_z)).T
            sec_gasmass = sec_reg[(gas_name_dict[codetp],gas_mass_dict[codetp])].to('Msun').v
            sec_gastemp = sec_reg[(gas_name_dict[codetp],gas_temp_dict[codetp])].v
            sec_gasden = sec_reg['gas','density'].to('g/cm**3').v*0.752

            sec_bool = in_hull(sec_gascoor, hullv[idx][sec_branch])
            if np.sum(sec_bool)/len(sec_bool) > 0.75:
                print('Need to select a bigger region to conver the convex hull')

            sec_center = np.array(dist_data['sec_com_plot'])[np.array(dist_data['idx']) == idx][0]*np.array(dist_data['codelength_to_meters'])[np.array(dist_data['idx']) == idx][0]
            sec_gascoor = sec_gascoor[sec_bool]
            sec_gasdist = np.linalg.norm(sec_gascoor - sec_center, axis=1)
            sec_gasdist = (sec_gasdist*unyt.m).to('kpc').v
            sec_gasmass = sec_gasmass[sec_bool]
            sec_gastemp = sec_gastemp[sec_bool]
            sec_gasden = sec_gasden[sec_bool]
            
    else: #for SPH and hybrid codes
        reg = ds.all_data()
        gascoor = reg[gas_name_dict[codetp], 'particle_position'].to('m').v
        gasmass = reg[gas_name_dict[codetp], 'particle_mass'].to('Msun').v
        gastemp = reg[gas_name_dict[codetp], gas_temp_dict[codetp]].v
        gasden = reg['gas','density'].to('g/cm**3').v
        
        prog_bool = in_hull(gascoor, hullv[idx][prog_branch])
        prog_center = np.array(dist_data['prog_com_plot'])[np.array(dist_data['idx']) == idx][0]*np.array(dist_data['codelength_to_meters'])[np.array(dist_data['idx']) == idx][0]
        prog_gascoor = gascoor[prog_bool]
        prog_gasdist = np.linalg.norm(prog_gascoor - prog_center, axis=1)
        prog_gasdist = (prog_gasdist*unyt.m).to('kpc').v
        prog_gasmass = gasmass[prog_bool]
        prog_gastemp = gastemp[prog_bool]
        prog_gasden = gasden[prog_bool]
        
        if retrieve_sec == True:
            sec_bool = in_hull(gascoor, hullv[idx][sec_branch])
            sec_center = np.array(dist_data['sec_com_plot'])[np.array(dist_data['idx']) == idx][0]*np.array(dist_data['codelength_to_meters'])[np.array(dist_data['idx']) == idx][0]
            sec_gascoor = gascoor[sec_bool]
            sec_gasdist = np.linalg.norm(sec_gascoor - sec_center, axis=1)
            sec_gasdist = (sec_gasdist*unyt.m).to('kpc').v
            sec_gasmass = gasmass[sec_bool]
            sec_gastemp = gastemp[sec_bool]
            sec_gasden = gasden[sec_bool]
    
    if retrieve_sec == True:
        output = {index: value for index, value in enumerate([prog_gascoor, prog_gasdist, prog_gasmass, prog_gastemp, prog_gasden, sec_gascoor, sec_gasdist, sec_gasmass, sec_gastemp, sec_gasden])}
        np.save('/work/hdd/bezm/tnguyen2/AGORA/analysis/gas_data_for_phase_plot_%s_Idx_%s_ver2013.npy' % (codetp, idx), output)
        return prog_gascoor, prog_gasdist, prog_gasmass, prog_gastemp, prog_gasden, sec_gascoor, sec_gasdist, sec_gasmass, sec_gastemp, sec_gasden
    else:
        output = {index: value for index, value in enumerate([prog_gascoor, prog_gasdist, prog_gasmass, prog_gastemp, prog_gasden])}
        np.save('/work/hdd/bezm/tnguyen2/AGORA/analysis/gas_data_for_phase_plot_%s_Idx_%s_ver2013.npy' % (codetp, idx), output)
        return prog_gascoor, prog_gasdist, prog_gasmass, prog_gastemp, prog_gasden


halotree_ver = 2013
merger_number = '0'

plotbins=150
plotbins2=320
font_size = 15

dens_min = np.log10(1e-30)
dens_max = np.log10(1e-20)
t_min = np.log10(5.95)
t_max = np.log10(5e8)
mass_min = 1e0
mass_max = 1e8

dist_min = -1
dist_max = 10
mass_min2 = 1e0
mass_max2 = 5e7


fig = plt.figure() #this fig is for the phase plot
grid = AxesGrid(
    fig,
    (0.1, 0.1, 2.1, 1.9*4/5),
    nrows_ncols=(4, len(codetp_list)),
    axes_pad=0.1,
    label_mode="1",
    share_all=True,
    cbar_location="right",
    cbar_mode="single",
    cbar_size="3%",
    cbar_pad="2%",
    aspect=False,
) 

fig2 = plt.figure()  #this fig is for the density-distance plot
grid2 = AxesGrid(
    fig2,
    (0.1, 0.1, 2.1, 1.9*4/5),
    nrows_ncols=(4, len(codetp_list)),
    axes_pad=0.1,
    label_mode="1",
    share_all=True,
    cbar_location="right",
    cbar_mode="single",
    cbar_size="3%",
    cbar_pad="2%",
    aspect=False,
) 



for i in range(len(codetp_list)):
    codetp = codetp_list[i]
    #
    if codetp == 'GEAR':
        step = 3
    elif codetp == 'CHANGA':
        step = 2
    else:
        step = 1
        
    time_list = np.loadtxt('/work/hdd/bezm/gtg115x/Halo_Finding/%s/pfs_allsnaps_%s.txt' % (codetp, halotree_ver), dtype=str)[:,2].astype(float)
    idx_begin, idx_endinfall, idx_1stpass, idx_maxdist, idx_cls, time_begin, time_endinfall, time_1stpass, time_maxdist, time_cls = load_timings(codetp, halotree_ver, merger_number) 
    time_eval = time_1stpass + 0.6
    if codetp == 'GEAR':
        idx_eval = np.argmin(abs(time_eval - time_list/1e3)) - (np.argmin(abs(time_eval - time_list/1e3)) % step) - 1
    else:
        idx_eval = np.argmin(abs(time_eval - time_list/1e3)) - (np.argmin(abs(time_eval - time_list/1e3)) % step)
    #
    if os.path.exists('/work/hdd/bezm/tnguyen2/AGORA/analysis/gas_data_for_phase_plot_%s_Idx_%s_ver2013.npy' % (codetp, idx_eval)) == False:
        rawtree, redshift_list, time_list, pfs, step = load_halotree_and_pfs(codetp, halotree_ver)
        hullv = np.load('/work/hdd/bezm/gtg115x/Halo_Finding/%s/hullv_%s_withPos_final.npy' % (codetp, halotree_ver), allow_pickle=True).tolist()
    else:
        redshift_list, time_list, pfs, step = load_halotree_and_pfs(codetp, halotree_ver, rawtree_skip=True)
        rawtree = 0
        hullv = 0
    #
    dist_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/dist_relvel_j_ProgBranch-%s_FirstMerger_%s_ver2013.npy' % (merger_number, codetp), allow_pickle=True).tolist()
    prog_branch, sec_branch, sec_branch_2 = sec_branch_compute(codetp, merger_number)  
    #------------------------------------------------------------------------------------
    #Phase plot of the primary and secondary galaxy at the start of infall
    if os.path.exists('/work/hdd/bezm/tnguyen2/AGORA/analysis/gas_data_for_phase_plot_%s_Idx_%s_ver2013.npy' % (codetp, idx_begin)) == False:
        prog_gascoor, prog_gasdist, prog_gasmass, prog_gastemp, prog_gasden, sec_gascoor, sec_gasdist, sec_gasmass, sec_gastemp, sec_gasden = obtain_gas_properties(idx_begin, codetp, pfs, hullv, rawtree, retrieve_sec=True)
    else:
        prog_gascoor, prog_gasdist, prog_gasmass, prog_gastemp, prog_gasden, sec_gascoor, sec_gasdist, sec_gasmass, sec_gastemp, sec_gasden = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/gas_data_for_phase_plot_%s_Idx_%s_ver2013.npy' % (codetp, idx_begin), allow_pickle=True).tolist().values()
    
    p = grid[i].hist2d(np.log10(prog_gasden), np.log10(prog_gastemp), weights=prog_gasmass, norm=LogNorm(vmin=mass_min, vmax=mass_max),  bins=plotbins, cmap='inferno', zorder=100)

    grid[i].axvline(np.log10(1.67e-24/0.752), linestyle='--', color='k', linewidth=0.5, zorder=200)

    if codetp == 'ART' or codetp == 'GADGET3' or codetp == 'GADGET4' or codetp == 'GIZMO':
        title_color = 'blue'
    else:
        title_color = 'black'
    if codetp == 'ART' or codetp == 'RAMSES' or codetp == 'GADGET3' or codetp == 'GEAR':
        grid[i].set_title(label_list[i] + '*', fontsize=font_size, color=title_color)
    elif codetp == 'CHANGA':
        grid[i].set_title(r'$\text{%s}^{\dagger}$' % label_list[i], fontsize=font_size, color=title_color)
    else:
        grid[i].set_title(label_list[i], fontsize=font_size, color=title_color)
    
    
    grid[i].set_xlim(dens_min, dens_max)
    grid[i].set_ylim(t_min, t_max)
    grid[i].set_xticks([-30, -27.5, -25, -22.5, -20], [r'$10^{-30}$', '', r'$10^{-25}$', '', r'$10^{-20}$'])
    grid[i].set_yticks([2, 4, 6, 8], [r'$10^{2}$', r'$10^{4}$', r'$10^{6}$', r'$10^{8}$'])

    cbar = fig.colorbar(p[3], cax=grid.cbar_axes[0])
    cbar.ax.tick_params('both', labelsize=font_size)
    cbar.set_label(r'$M_\text{gas} (M_\odot)$', fontsize=font_size)
    #Distance-Density plot of the system at the start of infall
    p2 = grid2[i].hist2d(np.log10(prog_gasden), prog_gasdist, weights=prog_gasmass, norm=LogNorm(vmin=mass_min2, vmax=mass_max2),  bins=plotbins2, cmap='viridis', zorder=100)

    grid2[i].axvline(np.log10(1.67e-24/0.752), linestyle='--', color='k', linewidth=0.5, zorder=200)

    if codetp == 'ART' or codetp == 'GADGET3' or codetp == 'GADGET4' or codetp == 'GIZMO':
        title_color = 'blue'
    else:
        title_color = 'black'
    if codetp == 'ART' or codetp == 'RAMSES' or codetp == 'GADGET3' or codetp == 'GEAR':
        grid2[i].set_title(label_list[i] + '*', fontsize=font_size, color=title_color)
    elif codetp == 'CHANGA':
        grid2[i].set_title(r'$\text{%s}^{\dagger}$' % label_list[i], fontsize=font_size, color=title_color)
    else:
        grid2[i].set_title(label_list[i], fontsize=font_size, color=title_color)
    
    grid2[i].set_xlim(dens_min, dens_max)
    grid2[i].set_ylim(dist_min, dist_max)
    grid2[i].set_xticks([-30, -27.5, -25, -22.5, -20], [r'$10^{-30}$', '', r'$10^{-25}$', '', r'$10^{-20}$'])
    grid2[i].set_yticks([0, 5, 10, 15, 20], [r'0', r'5', r'10', '', r'20'])

    cbar2 = fig2.colorbar(p2[3], cax=grid2.cbar_axes[0])
    cbar2.ax.tick_params('both', labelsize=font_size)
    cbar2.set_label(r'$M_\text{gas} (M_\odot)$', fontsize=font_size)
    #
    
    #------------------------------------------------------------------------------------
    #Phase plot of the system (end of infall)
    if os.path.exists('/work/hdd/bezm/tnguyen2/AGORA/analysis/gas_data_for_phase_plot_%s_Idx_%s_ver2013.npy' % (codetp, idx_endinfall)) == False:
        prog_gascoor, prog_gasdist, prog_gasmass, prog_gastemp, prog_gasden = obtain_gas_properties(idx_endinfall, codetp, pfs, hullv, rawtree)
    else:
        prog_gascoor, prog_gasdist, prog_gasmass, prog_gastemp, prog_gasden = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/gas_data_for_phase_plot_%s_Idx_%s_ver2013.npy' % (codetp, idx_endinfall), allow_pickle=True).tolist().values()
    #
    grid[i+1*len(codetp_list)].hist2d(np.log10(prog_gasden), np.log10(prog_gastemp), weights=prog_gasmass, norm=LogNorm(vmin=mass_min, vmax=mass_max),  bins=plotbins, cmap='inferno', zorder=100)
    grid[i+1*len(codetp_list)].axvline(np.log10(1.67e-24/0.752), linestyle='--', color='k', linewidth=0.5, zorder=200)
    grid[i + 1*len(codetp_list)].set_xlim(dens_min, dens_max)
    grid[i + 1*len(codetp_list)].set_ylim(t_min, t_max)
    #Distance-Density plot of the system (end of infall)
    grid2[i+1*len(codetp_list)].hist2d(np.log10(prog_gasden), prog_gasdist, weights=prog_gasmass, norm=LogNorm(vmin=mass_min2, vmax=mass_max2),  bins=plotbins2, cmap='viridis', zorder=100)
    grid2[i+1*len(codetp_list)].axvline(np.log10(1.67e-24/0.752), linestyle='--', color='k', linewidth=0.5, zorder=200)
    grid2[i + 1*len(codetp_list)].set_xlim(dens_min, dens_max)
    grid2[i + 1*len(codetp_list)].set_ylim(dist_min, dist_max)

    #------------------------------------------------------------------------------------
    #Phase plot of the system (First apocenter)
    if os.path.exists('/work/hdd/bezm/tnguyen2/AGORA/analysis/gas_data_for_phase_plot_%s_Idx_%s_ver2013.npy' % (codetp, idx_maxdist)) == False:
        prog_gascoor, prog_gasdist, prog_gasmass, prog_gastemp, prog_gasden = obtain_gas_properties(idx_maxdist, codetp, pfs, hullv, rawtree)
    else:
        prog_gascoor, prog_gasdist, prog_gasmass, prog_gastemp, prog_gasden = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/gas_data_for_phase_plot_%s_Idx_%s_ver2013.npy' % (codetp, idx_maxdist), allow_pickle=True).tolist().values()
    #
    grid[i+2*len(codetp_list)].hist2d(np.log10(prog_gasden), np.log10(prog_gastemp), weights=prog_gasmass, norm=LogNorm(vmin=mass_min, vmax=mass_max),  bins=plotbins, cmap='inferno', zorder=100)
    grid[i+2*len(codetp_list)].axvline(np.log10(1.67e-24/0.752), linestyle='--', color='k', linewidth=0.5, zorder=200)
    grid[i + 2*len(codetp_list)].set_xlim(dens_min, dens_max)
    grid[i + 2*len(codetp_list)].set_ylim(t_min, t_max)
    #Distance-Density plot of the system (First apocenter)
    grid2[i+2*len(codetp_list)].hist2d(np.log10(prog_gasden), prog_gasdist, weights=prog_gasmass, norm=LogNorm(vmin=mass_min2, vmax=mass_max2),  bins=plotbins2, cmap='viridis', zorder=100)
    grid2[i+2*len(codetp_list)].axvline(np.log10(1.67e-24/0.752), linestyle='--', color='k', linewidth=0.5, zorder=200)
    grid2[i + 2*len(codetp_list)].set_xlim(dens_min, dens_max)
    grid2[i + 2*len(codetp_list)].set_ylim(dist_min, dist_max)


    
    #------------------------------------------------------------------------------------
    #Phase plot of the system (Evaluate timestep)
    if os.path.exists('/work/hdd/bezm/tnguyen2/AGORA/analysis/gas_data_for_phase_plot_%s_Idx_%s_ver2013.npy' % (codetp, idx_eval)) == False:
        prog_gascoor, prog_gasdist, prog_gasmass, prog_gastemp, prog_gasden = obtain_gas_properties(idx_eval, codetp, pfs, hullv, rawtree)
    else:
        prog_gascoor, prog_gasdist, prog_gasmass, prog_gastemp, prog_gasden = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/gas_data_for_phase_plot_%s_Idx_%s_ver2013.npy' % (codetp, idx_eval), allow_pickle=True).tolist().values()
    #
    grid[i+3*len(codetp_list)].hist2d(np.log10(prog_gasden), np.log10(prog_gastemp), weights=prog_gasmass, norm=LogNorm(vmin=mass_min, vmax=mass_max),  bins=plotbins, cmap='inferno', zorder=100)
    grid[i+3*len(codetp_list)].axvline(np.log10(1.67e-24/0.752), linestyle='--', color='k', linewidth=0.5, zorder=200)
    grid[i + 3*len(codetp_list)].set_xlim(dens_min, dens_max)
    grid[i + 3*len(codetp_list)].set_ylim(t_min, t_max)
    #Distance-Density plot of the system (Evaluate timestep)
    grid2[i+3*len(codetp_list)].hist2d(np.log10(prog_gasden), prog_gasdist, weights=prog_gasmass, norm=LogNorm(vmin=mass_min2, vmax=mass_max2),  bins=plotbins2, cmap='viridis', zorder=100)
    grid2[i+3*len(codetp_list)].axvline(np.log10(1.67e-24/0.752), linestyle='--', color='k', linewidth=0.5, zorder=200)
    grid2[i + 3*len(codetp_list)].set_xlim(dens_min, dens_max)
    grid2[i + 3*len(codetp_list)].set_ylim(dist_min, dist_max)
    #
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------
    del rawtree, hullv
    print('Done for %s' % codetp)
    

grid[0 + 4].text(np.log10(5e-30), 7.4, "Start Pri.", ha='left', va='bottom', fontsize=font_size-2, zorder=500)
grid[1*len(codetp_list) + 4].text(np.log10(5e-30), 7.4, "End Infall", ha='left', va='bottom', fontsize=font_size-2, zorder=500)
grid[2*len(codetp_list) + 4].text(np.log10(5e-30), 7.4, "1st Apo.", ha='left', va='bottom', fontsize=font_size-2, zorder=500)
grid[3*len(codetp_list) + 4].text(np.log10(5e-30), 7.4, "Cls/Post-cls", ha='left', va='bottom', fontsize=font_size-2, zorder=500)

grid[3*len(codetp_list)].tick_params('both', labelsize=font_size)
grid[3*len(codetp_list)].set_xlabel(r'$\rho_{gas} \left( \frac{g}{cm^{3}} \right)$', fontsize=font_size)
grid[3*len(codetp_list)].set_ylabel(r'T (K)', fontsize=font_size)


grid2[0 + 4].text(np.log10(5e-30), 8, "Start Pri.", ha='left', fontsize=font_size-2, zorder=500)
grid2[1*len(codetp_list) + 4].text(np.log10(5e-30), 8, "End Infall", ha='left', fontsize=font_size-2, zorder=500)
grid2[2*len(codetp_list) + 4].text(np.log10(5e-30), 8, "1st Apo.", ha='left', fontsize=font_size-2, zorder=500)
grid2[3*len(codetp_list) + 4].text(np.log10(5e-30), 8, "Cls/Post-cls", ha='left', fontsize=font_size-2, zorder=500)

grid2[3*len(codetp_list)].tick_params('both', labelsize=font_size)
grid2[3*len(codetp_list)].set_xlabel(r'$\rho_{gas} \left( \frac{g}{cm^{3}} \right)$', fontsize=font_size)
grid2[3*len(codetp_list)].set_ylabel(r'd (kpc)', fontsize=font_size)
    
fig.savefig('/work/hdd/bezm/tnguyen2/figures/AGORA/November2025/GasPhasePlots_ProgBranch-%s_ver2013_PartApaper_ver4.png' % merger_number, dpi=300, bbox_inches='tight')
fig2.savefig('/work/hdd/bezm/tnguyen2/figures/AGORA/November2025/GasDensityDistancePlots_ProgBranch-%s_ver2013_PartApaper_ver3.png' % merger_number, dpi=300, bbox_inches='tight')

