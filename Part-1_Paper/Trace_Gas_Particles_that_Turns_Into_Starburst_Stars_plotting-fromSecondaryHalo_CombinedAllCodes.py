import yt
yt.set_log_level(0)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle, ConnectionPatch
import os, sys  
import setup
from importlib import reload
reload(setup)

from setup import load_halotree_and_pfs, load_timings, load_ds, load_tracking_dist_data, sec_branch_compute
from setup import gas_name_dict, gas_temp_dict
from setup import add_metallicity_fields, add_cooling_fields, add_radialdist_to_halocenter_field, extract_and_order_snapshotIdx
from setup import get_haloradius, infall_timestep_compute_hullv
from setup import codetp_list, label_list, color_list, marker_list
from setup import codetp_list10, label_list10, color_list10, marker_list10


class LoadPlottingData:
    def __init__(self, codetp, merger_number = '0', halotree_ver = 2013):
        self.codetp = codetp
        self.redshift_list, self.time_list, self.pfs, self.step = load_halotree_and_pfs(codetp, halotree_ver, rawtree_skip=True)
        self.dist_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/dist_relvel_j_ProgBranch-%s_FirstMerger_%s_ver2013.npy' % (merger_number, codetp), allow_pickle=True).tolist()
        self.prog_branch, self.sec_branch, self.sec_branch_2 = sec_branch_compute(codetp, merger_number)
        self.idx_begin, self.idx_endinfall, self.idx_1stpass, self.idx_maxdist, self.idx_cls, self.time_begin, self.time_endinfall, self.time_1stpass, self.time_maxdist, self.time_cls = load_timings(codetp, halotree_ver, merger_number)
        time_eval = self.time_1stpass + 0.6
        if codetp == 'GEAR':
            self.idx_eval = np.argmin(abs(time_eval - self.time_list/1e3)) - (np.argmin(abs(time_eval - self.time_list/1e3)) % self.step) - 1
        else:
            self.idx_eval = np.argmin(abs(time_eval - self.time_list/1e3)) - (np.argmin(abs(time_eval - self.time_list/1e3)) % self.step)
        #####################################################################3
        self.output = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/Trace_Gas_Particles_that_Turns_Into_Starburst_Stars_ProgBranch-0_%s_ver2013_ver2.npy' % codetp, allow_pickle=True).tolist()
        gas_index_bound = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/bound_gas_index_sec_galaxy_ProgBranch-0_%s_ver2013.npy' % codetp, allow_pickle=True).tolist()
        self.gas_idx_plot = np.intersect1d(gas_index_bound, np.array(list(self.output.keys())))
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def get_starform_time(self, gas_idx):
        time_lastgas = self.output[gas_idx]['time'][-1]
        idx_lastgas = np.argmin(abs(self.time_list/1e3 - time_lastgas))
        idx_firststar = idx_lastgas + self.step
        return self.time_list[idx_firststar]/1e3
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def create_radialvel_plotdata(self, idx_test):
        plotting = {}
        plotting[idx_test] = {}
        plotting[idx_test]['pos'] = []
        plotting[idx_test]['vel'] = []
        plotting[idx_test]['radialvel'] = []
        #    
        vel_com = np.average(np.array(self.dist_data['prog_vel_plot'])[self.dist_data['idx'] == idx_test][0], weights=np.array(self.dist_data['prog_mass_plot'])[self.dist_data['idx'] == idx_test][0], axis=0) # in km/s
        codelength_to_kpc = np.array(self.dist_data['codelength_to_meters'])[np.array(self.dist_data['idx']) == idx_test][0]*3.24077929e-20 #convert to kpc
        prog_center = np.array(self.dist_data['prog_com_plot'])[np.array(self.dist_data['idx']) == idx_test][0]*codelength_to_kpc
        #
        for gas_idx in self.gas_idx_plot:  
            if self.codetp == 'CHANGA':
                if (self.output[gas_idx]['time'][0] < self.time_list[idx_test]/1e3) or (self.output[gas_idx]['time'][0] > self.time_list[self.idx_eval]/1e3): 
                    continue
            elif self.codetp == 'GADGET4':
                if (self.output[gas_idx]['time'][-1] < self.time_list[idx_test]/1e3)  or (self.output[gas_idx]['firststar_time'][0] > self.time_list[self.idx_eval+self.step]/1e3): 
                    continue
            else:
                if (self.output[gas_idx]['time'][-1] < self.time_list[idx_test]/1e3) or (self.output[gas_idx]['time'][-1] > self.time_list[self.idx_eval]/1e3): 
                    continue
            prog_to_gas = (self.output[gas_idx]['pos'])[np.isclose(self.output[gas_idx]['time'], self.time_list[idx_test]/1e3, atol=1e-10)][0]
            vel_gas = (self.output[gas_idx]['vel'])[np.isclose(self.output[gas_idx]['time'], self.time_list[idx_test]/1e3, atol=1e-10)][0] - vel_com
            radialvel_gas = np.dot(vel_gas, prog_to_gas/np.linalg.norm(prog_to_gas))
            #
            plotting[idx_test]['pos'].append(prog_to_gas)
            plotting[idx_test]['vel'].append(vel_gas)
            plotting[idx_test]['radialvel'].append(radialvel_gas)
            #
        plotting[idx_test]['pos'] = np.array(plotting[idx_test]['pos'])
        plotting[idx_test]['vel'] = np.array(plotting[idx_test]['vel'])
        plotting[idx_test]['radialvel'] = np.array(plotting[idx_test]['radialvel'])
        plotting[idx_test]['time'] = self.time_list[idx_test]/1e3
        plotting[idx_test]['sec_pos'] = np.array(self.dist_data['sec_com_plot'])[np.array(self.dist_data['idx']) == idx_test][0]*codelength_to_kpc - prog_center
        plotting[idx_test]['sec_vel'] = np.array(self.dist_data['relvel'])[self.dist_data['idx'] == idx_test][0]
        #
        plotting[idx_test]['rel_pos'] = []
        for idx_k in range(self.idx_begin, idx_test+self.step, self.step):
            codelength_to_kpc_k = np.array(self.dist_data['codelength_to_meters'])[np.array(self.dist_data['idx']) == idx_k][0]*3.24077929e-20 #convert to kpc
            plotting[idx_test]['rel_pos'].append(codelength_to_kpc_k*(np.array(self.dist_data['sec_com_plot'])[np.array(self.dist_data['idx']) == idx_k][0] - np.array(self.dist_data['prog_com_plot'])[np.array(self.dist_data['idx']) == idx_k][0]))
        plotting[idx_test]['rel_pos'] = np.array(plotting[idx_test]['rel_pos'])
        return plotting
        

merger_number = '0'
halotree_ver = 2013

axlim_xmin, axlim_xmax, axlim_ymin, axlim_ymax = -30, 30, -50, 25

line_width = 1
alpha = 0.05 #0.3 before
alpha_scat = 0.8
size_scat = 4
font_size = 22
norm_time = plt.Normalize(0, 1)
norm_temp = mcolors.LogNorm(10, 5e7)

radialvel_axlim_xmin, radialvel_axlim_xmax, radialvel_axlim_ymin, radialvel_axlim_ymax = -5.6, 5.6, -7, 7

index_spacing = 1 #set this to higher number if want to plot fewer particles for quicker saving

#%%%%%%%%%%%%%%%% Set up figure %%%%%%%%%%%%%%%
fig, ax = plt.subplots(3, 5, sharey = False, sharex = False, figsize=(12*2,6*3))
fig.subplots_adjust(wspace=0.05, hspace=0.12)

codetp_plotlist = ['CHANGA', 'GADGET3', 'GADGET4', 'GEAR', 'GIZMO']

for j in range(0, 5):
    
    ax_time = ax[0,j]
    ax_temp = ax[1,j]
    ax_radialvec = ax[2, j]
    
    #Plot parameters
    ax_time.set_aspect('equal')
    ax_time.set_xlim(axlim_xmin,axlim_xmax)
    ax_time.set_ylim(axlim_ymin,axlim_ymax)
    ax_time.tick_params('both', labelsize=font_size)

    ax_temp.set_aspect('equal')
    ax_temp.set_xlim(axlim_xmin,axlim_xmax)
    ax_temp.set_ylim(axlim_ymin,axlim_ymax)
    ax_temp.tick_params('both', labelsize=font_size)

    ax_radialvec.set_aspect('equal')
    ax_radialvec.set_xlim(radialvel_axlim_xmin,radialvel_axlim_xmax)
    ax_radialvec.set_ylim(radialvel_axlim_ymin,radialvel_axlim_ymax)
    
    if j == 0:
        ax_radialvec.set_xlabel('x (kpc)',fontsize=font_size)
        ax_radialvec.set_ylabel('y (kpc)',fontsize=font_size)

    #
    codetp = codetp_plotlist[j]
    if codetp == 'ART' or codetp == 'GADGET3' or codetp == 'GADGET4' or codetp == 'GIZMO':
        title_color = 'blue'
    else:
        title_color = 'black'
    if codetp == 'ART' or codetp == 'RAMSES' or codetp == 'GADGET3' or codetp == 'GEAR':
        ax_time.set_title(label_list[codetp_list.index(codetp)] + '*', fontsize=font_size, color=title_color)
    elif codetp == 'CHANGA':
        ax_time.set_title(r'$\text{%s}^{\dagger}$' % label_list[codetp_list.index(codetp)], fontsize=font_size, color=title_color)
    else:
        ax_time.set_title(label_list[codetp_list.index(codetp)], fontsize=font_size, color=title_color)
    #
    if j == 0:
        ax_time.tick_params(axis='both', labelbottom=True, labelleft=True)
        ax_temp.tick_params(axis='both', labelbottom=True, labelleft=True)
        ax_radialvec.tick_params(axis='both', labelbottom=True, labelleft=True, labelsize=font_size)
    else:
        ax_time.tick_params(axis='both', labelbottom=True, labelleft=False)
        ax_temp.tick_params(axis='both', labelbottom=True, labelleft=False)
        ax_radialvec.tick_params(axis='both', labelbottom=True, labelleft=False, labelsize=font_size)

# [left, bottom, width, height] in *figure coordinates* (0 → 1)
cbar_left   = 0.295   # move left/right
cbar_bottom = 0.06   # move up/down
cbar_width  = 0.28   # length of the colorbar
cbar_height = 0.012  # thickness

cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])            
ax_cbar_time = ax_time.scatter([],[],c=[], norm=norm_time, cmap='rainbow', alpha=1)    
cbar_time = fig.colorbar(ax_cbar_time, cax=cax, fraction=0.046, pad=0.04, orientation='horizontal')
cbar_time.set_label(r'Time elapsed since $t_\text{start}$ (Gyr)',fontsize=font_size)
cbar_time.ax.tick_params(labelsize=font_size)


# [left, bottom, width, height] in *figure coordinates* (0 → 1)
cbar_left   = 0.61   # move left/right
cbar_bottom = 0.06   # move up/down
cbar_width  = 0.28   # length of the colorbar
cbar_height = 0.012  # thickness

cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])  
ax_cbar_temp = ax_temp.scatter([],[],c=[], cmap='plasma', norm=norm_temp, alpha=1)
cbar_temp = fig.colorbar(ax_cbar_temp, cax=cax, fraction=0.046, pad=0.04, orientation='horizontal')
cbar_temp.set_label('T (K)',fontsize=font_size)
cbar_temp.ax.tick_params(labelsize=font_size)

#%%%%%%%%%%%%%%%% Set up figure %%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%% Looping through each code to plot %%%%%%%%%%%%%%%
for k in range(len(codetp_plotlist)):
#for k in [0]:
    codetp = codetp_plotlist[k]
    #%%%%%%%%%%%%%%% Load data %%%%%%%%%%%%%%%
    PlotData = LoadPlottingData(codetp)
    #%%%%%%%%%%%%%%% Plotting %%%%%%%%%%%%%%%
    ax_time = ax[0,k]
    ax_temp = ax[1,k]
    ax_radialvec = ax[2,k]

    for gas_idx_i in range(0,len(PlotData.gas_idx_plot), index_spacing):
        gas_idx = int(PlotData.gas_idx_plot[gas_idx_i])
        #
        if codetp == 'CHANGA' and (PlotData.output[gas_idx]['time'][0] - PlotData.time_begin > 1):
            continue
        #
        x = np.array(PlotData.output[gas_idx]['pos'])[:,0]
        y = np.array(PlotData.output[gas_idx]['pos'])[:,1]

        z_time = np.array(PlotData.output[gas_idx]['time']) - PlotData.time_begin
        z_temp = np.array(PlotData.output[gas_idx]['temp'])
        if codetp == 'GADGET4':
            time_limit = max(PlotData.output[gas_idx]['firststar_time']) - PlotData.time_begin
            time_limit_bool = z_time < time_limit
            points = np.array([x[time_limit_bool], y[time_limit_bool]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
        else:
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
        #
        if codetp == 'CHANGA':
            zorder = np.argmin(abs(PlotData.output[gas_idx]['time'][0] - PlotData.dist_data['time']))*10
        else:
            zorder = np.argmin(abs(PlotData.output[gas_idx]['time'][-1] - PlotData.dist_data['time']))*10
        # Create a continuous norm to map from data points to colors
        lc_time = LineCollection(segments, cmap='rainbow', norm=norm_time, alpha=alpha, zorder=zorder)
        lc_time.set_array(z_time)
        lc_time.set_linewidth(line_width)
        line_time = ax_time.add_collection(lc_time)
        #
        lc_temp = LineCollection(segments, cmap='plasma', norm=norm_temp, alpha=alpha, zorder=zorder)
        lc_temp.set_array(z_temp)
        lc_temp.set_linewidth(line_width)
        line_temp = ax_temp.add_collection(lc_temp)
        #
        if codetp == 'GADGET4':
            ax_time.scatter(PlotData.output[gas_idx]['firststar_pos'][:,0], PlotData.output[gas_idx]['firststar_pos'][:,1], marker='o', c=PlotData.output[gas_idx]['firststar_time'] - PlotData.time_begin , norm=norm_time, s=size_scat, alpha=alpha_scat, cmap='rainbow', zorder=999999999, edgecolors='black', linewidths=0.15)
        elif codetp == 'CHANGA':
            ax_time.scatter(x[0], y[0], marker='o', c=z_time[0] , norm=norm_time, s=size_scat, alpha=alpha_scat, cmap='rainbow', zorder=999999999, edgecolors='black', linewidths=0.15)
        else:
            ax_time.scatter(PlotData.output[gas_idx]['firststar_pos'][0], PlotData.output[gas_idx]['firststar_pos'][1], marker='o', c=PlotData.get_starform_time(gas_idx) - PlotData.time_begin , norm=norm_time, s=size_scat, alpha=alpha_scat, cmap='rainbow', zorder=999999999, edgecolors='black', linewidths=0.15)

    #%%%%%%%%%%%%%%% Plotting radialvel %%%%%%%%%%%%%%%
    if codetp == 'GADGET3' or codetp == 'GIZMO' or codetp == 'CHANGA' or codetp == 'GEAR':
        idx_radialvel = PlotData.idx_maxdist - 2*PlotData.step
    elif codetp == 'GADGET4':
        idx_radialvel = PlotData.idx_maxdist - 1*PlotData.step
    
    plotting = PlotData.create_radialvel_plotdata(idx_radialvel)
    # Plotting the gas particles with radialvel < 0
    cond = plotting[idx_radialvel]['radialvel'] < 0
    ax_radialvec.quiver(plotting[idx_radialvel]['pos'][cond][:, 0], plotting[idx_radialvel]['pos'][cond][:, 1],
               plotting[idx_radialvel]['vel'][cond][:, 0], plotting[idx_radialvel]['vel'][cond][:, 1],
               alpha=0.2,
               scale=4500,          # adjust to control arrow length
               width=0.003,        # arrow shaft width
               headwidth=4, zorder=1e9)
    #Plotting the center of the main galaxy 
    ax_radialvec.scatter(0,0,color='red',marker='x',s=40,zorder=1e10)
    #Plotting the velocity direction of the secondary galaxy 
    ax_radialvec.quiver(plotting[idx_radialvel]['sec_pos'][0], plotting[idx_radialvel]['sec_pos'][1],
               plotting[idx_radialvel]['sec_vel'][0], plotting[idx_radialvel]['sec_vel'][1],
               alpha=1,
               scale=1000,          # adjust to control arrow length
               width=0.006,        # arrow shaft width
               headwidth=4, color='red')
    #Plotting the merger's trajectory
    ax_radialvec.plot(plotting[idx_radialvel]['rel_pos'][:,0], plotting[idx_radialvel]['rel_pos'][:,1], linestyle='--', color='red', alpha=0.54, zorder=1)
    #Annotating
    if k == 0:
        ax_radialvec.text(0.08, 0.09, s=r'$t_\text{fp} < t < t_\text{max}$',  transform=ax_radialvec.transAxes, va='center', ha='left', fontsize=font_size-2)
        ax_radialvec.text(0.08, 0.91, s=r'$v_{r} < 0$',  transform=ax_radialvec.transAxes, va='center', ha='left', fontsize=font_size-2)
    #Create a subplot, showing the zoom in bounds
    zoom_x1, zoom_x2 = radialvel_axlim_xmin,radialvel_axlim_xmax
    zoom_y1, zoom_y2 = radialvel_axlim_ymin,radialvel_axlim_ymax
    # Draw the rectangle on ax_temp (second row)
    rect = Rectangle(
        (zoom_x1, zoom_y1),
        zoom_x2 - zoom_x1,
        zoom_y2 - zoom_y1,
        linewidth=0.8,
        edgecolor='black',
        facecolor='none',
        linestyle='-',
        zorder=1e10,
        alpha=0.6
    )
    ax_temp.add_patch(rect)
    # Draw connecting lines from two corners of the rectangle to ax_radialvec
    con1 = ConnectionPatch(
        xyA=(zoom_x1, zoom_y2), coordsA=ax_temp.transData,
        xyB=(zoom_x1, zoom_y2), coordsB=ax_radialvec.transData,
        color='black', linewidth=0.8, linestyle='-', alpha=0.6
    )
    # Right-bottom corner → right-bottom of ax_radialvec
    con2 = ConnectionPatch(
        xyA=(zoom_x2, zoom_y1), coordsA=ax_temp.transData,
        xyB=(zoom_x2, zoom_y1), coordsB=ax_radialvec.transData,
        color='black', linewidth=0.8, linestyle='-', alpha=0.6
    )
    fig.add_artist(con1)
    fig.add_artist(con2)
    print(codetp)
#%%%%%%%%%%%%%%%% Looping through each code to plot %%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%% Saving %%%%%%%%%%%%%%%
#fig.tight_layout()
fig.savefig('/work/hdd/bezm/tnguyen2/figures/AGORA/November2025/Trace_Gas_Particles_that_Turns_Into_Starburst_Stars_CombinedAllCodes_ProgBranch-%s_ver2013_fromSecondaryHalo_ver4.png' % (merger_number), dpi=300, bbox_inches='tight')

