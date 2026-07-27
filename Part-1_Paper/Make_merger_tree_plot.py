import numpy as np
import matplotlib.pyplot as plt
import yt
yt.set_log_level(0)
from scipy.stats import norm
if int((yt.__version__).split('.')[0]) >= 4 and int((yt.__version__).split('.')[1]) >= 2: #ParticleUnion is only available in yt 4.2 and later
    from yt.data_objects.unions import ParticleUnion
else:
    from yt.data_objects.unions import Union as ParticleUnion
from mpl_toolkits.axes_grid1 import AxesGrid, ImageGrid
import matplotlib.colors as mcolors
import matplotlib.cm as cm
    
import setup
from importlib import reload
reload(setup)

from setup import load_halotree_and_pfs, load_timings, load_ds, load_tracking_dist_data, sec_branch_compute
from setup import gas_name_dict, gas_temp_dict
from setup import add_metallicity_fields, add_cooling_fields, add_radialdist_to_halocenter_field, extract_and_order_snapshotIdx, in_hull
from setup import get_haloradius, infall_timestep_compute_hullv
from setup import codetp_list, label_list, color_list, marker_list
from setup import codetp_list10, label_list10, color_list10, marker_list10


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    return mcolors.LinearSegmentedColormap.from_list(
        f'trunc_{cmap.name}_{minval:.2f}_{maxval:.2f}',
        cmap(np.linspace(minval, maxval, n))
    )

fig = plt.figure()

grid = AxesGrid(
    fig,
    (0.1, 0.1, 1.3, 2.7),
    nrows_ncols=(3, 3),
    #nrows_ncols=(2, 8),
    axes_pad=0.42,
    label_mode="1",
    share_all=True,
    cbar_location="bottom",
    cbar_mode="single",
    cbar_size="2%",
    cbar_pad="9%",
    aspect=False
) 

font_size = 21

halotree_ver = 2013
merger_number = '0'

massratio_ratio_lim = 0.005

output = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/MergerTreeMainHalo_ProgBranch-%s_ver2013_plotdata.npy' % merger_number, allow_pickle=True).tolist()


for k in range(len(codetp_list)):
    codetp = codetp_list[k]
    anchor = output[codetp]['anchor']
    #
    grid[k].plot(output[codetp]['loc_0'], output[codetp]['time_0'], color='k', lw=2, linestyle='--', zorder=100)
    # Define a colormap and normalization range
    # (Adjust vmin/vmax depending on your mass ratio range)
    vmin, vmax = 0, 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)   # Log scale is often good for ratios
    cmap = truncate_colormap(cm.viridis_r, 0.03, 1.0)
    
    for n in range(len(output[codetp]['branch'])):
        if output[codetp]['massratio'][n] < massratio_ratio_lim:
            continue
        if output[codetp]['massratio'][n] > 0.25:
            lw = 3
            zorder = 10
            alpha = 0.8
        else:
            lw = 2
            zorder = 1
            alpha = 0.7
        grid[k].plot(output[codetp]['loc'][n], output[codetp]['time'][n], color=cmap(norm(output[codetp]['massratio'][n])), alpha=alpha, lw=lw, zorder=zorder)
    
    idx_begin, idx_endinfall, idx_1stpass, idx_maxdist, idx_cls, time_begin, time_endinfall, time_1stpass, time_maxdist, time_cls = load_timings(codetp, halotree_ver, merger_number)
    grid[k].set_ylim(0.6, 1.8)
    grid[k].set_xlim(0, 0.02)
    grid[k].invert_xaxis()
    grid[k].invert_yaxis()
    grid[k].set_title(label_list[k], fontsize=font_size)
    grid[k].set_ylabel('Time since Big Bang (Gyr)', fontsize=font_size)
    grid[k].tick_params('both', labelsize=font_size)
    grid[k].set_xticks([0, 0.01, 0.02])
    if k == 6:
        grid[k].set_xlabel(r'$d_\text{normalized}$ (code_length)', fontsize=font_size)
        
    time_values = np.array([659, 786, 960, 1209, 1587])/1e3 
    specific_redshifts = [8, 7, 6, 5, 4] 
    ax_z = grid[k].twinx()
    ax_z.set_ylim(grid[k].get_ylim())
    ax_z.set_yticks(time_values)
    if k == 8:
        ax_z.set_ylabel("z", fontsize=font_size, rotation=270, labelpad=18)
        ax_z.set_yticklabels(['{:.1f}'.format(z) for z in specific_redshifts])
        ax_z.tick_params(axis='y',labelcolor='k',labelsize=font_size)
    else:
        ax_z.set_yticklabels(['' for z in range(len(specific_redshifts))])

sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = grid.cbar_axes[0].colorbar(sm, ax=grid[k])
cbar.set_label(r'$\mu_\text{DM}$', fontsize = font_size)
cbar.ax.tick_params(labelsize = font_size)

fig.tight_layout()

plt.savefig('/work/hdd/bezm/tnguyen2/figures/AGORA/November2025/MergerTreeMainHalo_ProgBranch-%s_ver2013_ver4.png' % merger_number, dpi=300, bbox_inches='tight')
