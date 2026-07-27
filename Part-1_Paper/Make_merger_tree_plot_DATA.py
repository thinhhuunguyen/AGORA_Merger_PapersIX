import yt
yt.set_log_level(0)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib.cm as cm
import matplotlib.colors as mcolors


import setup
from importlib import reload
reload(setup)

from setup import load_halotree_and_pfs, infall_timestep_compute_spherical, infall_timestep_compute_hullv
from setup import extract_and_order_snapshotIdx
from setup import codetp_list10



fig = plt.figure()

grid = AxesGrid(
    fig,
    (0.1, 0.1, 2, 1.2),
    nrows_ncols=(2, 5),
    #nrows_ncols=(2, 8),
    axes_pad=0.35,
    label_mode="1",
    share_all=True,
    cbar_location="bottom",
    cbar_mode="single",
    cbar_size="2%",
    cbar_pad="5.5%",
    aspect=False
) 

font_size = 14

halotree_ver = 2013
merger_number = '0'

massratio_ratio_lim = 0

output = {}

for k in range(len(codetp_list10)):
    codetp = codetp_list10[k]
    rawtree, redshift_list, time_list, pfs, step = load_halotree_and_pfs(codetp, halotree_ver)
    hullv = np.load('/work/hdd/bezm/gtg115x/Halo_Finding/%s/hullv_%s_withPos_final.npy' % (codetp, halotree_ver), allow_pickle=True).tolist()
    hullv_branchFirst = {}
    for idx in hullv.keys():
        for branch in hullv[idx].keys():
            if branch not in hullv_branchFirst.keys():
                hullv_branchFirst[branch] = {}
            hullv_branchFirst[branch][idx] = hullv[idx][branch]
    #
    idx_limplot = np.argmin(abs(time_list - 2000)) - ((np.argmin(abs(time_list - 2000)) - 2) % step)
    if codetp == 'GEAR' and halotree_ver == 2013:
        anchor = rawtree['0'][idx_limplot]['Halo_Center']
    else:
        anchor = rawtree['0'][idx_limplot]['Halo_Center']
    
    
    
    loc_plot = []
    time_plot = []

    for idx in extract_and_order_snapshotIdx(rawtree, '0'):
        if codetp == 'GADGET3' or codetp == 'AREPO' or codetp == 'GADGET4' or codetp == 'AREPO-TNG':
            factor = 60
        elif codetp == 'GEAR' or codetp == 'GIZMO':
            factor = 60000
        else:
            factor = 1
        loc_plot.append(np.linalg.norm(rawtree['0'][idx]['Halo_Center'] - anchor)/factor)
        time_plot.append(time_list[idx]/1e3)

    grid[k].plot(loc_plot, time_plot, color='k', lw=2, linestyle='--', zorder=100)


    # Define a colormap and normalization range
    # (Adjust vmin/vmax depending on your mass ratio range)
    vmin, vmax = massratio_ratio_lim, 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)   # Log scale is often good for ratios
    cmap = cm.viridis_r
    
    output[codetp] = {}
    output[codetp]['anchor'] = anchor
    output[codetp]['loc_0'] = loc_plot
    output[codetp]['time_0'] = time_plot
    output[codetp]['branch'] = []
    output[codetp]['loc'] = []
    output[codetp]['time'] = []
    output[codetp]['massratio'] = []
    output[codetp]['infall_index'] = []
    
    for branch in hullv_branchFirst.keys():
        infall_index_init = infall_timestep_compute_spherical(rawtree, '0', branch, step, halo_radius=True)
        if infall_index_init == None:
            continue
        if min(extract_and_order_snapshotIdx(rawtree, branch)) >= idx_limplot: #for better visualization on the plot, remove the branch that starts after idx_limplot
            continue
        infall_index = infall_timestep_compute_hullv(hullv, '0', branch, step, infall_index_init)
        if infall_index != None:
            #pre-infall mass ratio
            massratio = rawtree[branch][infall_index-step]['Halo_Mass']/rawtree['0'][infall_index-step]['Halo_Mass']
            if massratio > massratio_ratio_lim:                
                loc_plot_branch = []
                time_plot_branch = []
                for idx in extract_and_order_snapshotIdx(rawtree, branch):
                    if codetp == 'GADGET3' or codetp == 'AREPO' or codetp == 'GADGET4' or codetp == 'AREPO-TNG':
                        factor = 60
                    elif codetp == 'GEAR' or codetp == 'GIZMO':
                        factor = 60000
                    else:
                        factor = 1
                    loc_plot_branch.append(np.linalg.norm(rawtree[branch][idx]['Halo_Center'] - anchor)/factor)
                    time_plot_branch.append(time_list[idx]/1e3)
                if  idx <= idx_limplot or (np.linalg.norm(rawtree[branch][idx_limplot]['Halo_Center'] - rawtree['0'][idx_limplot]['Halo_Center']) < rawtree['0'][idx_limplot]['Halo_Radius']):
                    #For better visual on the plot, we elimite the branches that are still to far away from '0' at z = 2
                    if idx < max(extract_and_order_snapshotIdx(rawtree, '0')):
                        #this takes care of the situation that a branch ends at the same time as the '0' branch
                        loc_plot_branch.append(np.linalg.norm(rawtree['0'][idx+step]['Halo_Center'] - anchor)/factor)
                        time_plot_branch.append(time_list[idx+step]/1e3)
                    if massratio > 0.25:
                        lw = 2.5
                        zorder = 10
                        alpha = 0.8
                    else:
                        lw = 1.5
                        zorder = 1
                        alpha = 0.5
                    grid[k].plot(loc_plot_branch, time_plot_branch, alpha=alpha, color=cmap(norm(massratio)), lw=lw, zorder=zorder)
                    #
                    output[codetp]['branch'].append(branch)
                    output[codetp]['loc'].append(loc_plot_branch)
                    output[codetp]['time'].append(time_plot_branch)
                    output[codetp]['massratio'].append(massratio)
                    output[codetp]['infall_index'].append(infall_index)
    

np.save('/work/hdd/bezm/tnguyen2/AGORA/analysis/MergerTreeMainHalo_ProgBranch-%s_ver2013_plotdata.npy' % merger_number, output)

