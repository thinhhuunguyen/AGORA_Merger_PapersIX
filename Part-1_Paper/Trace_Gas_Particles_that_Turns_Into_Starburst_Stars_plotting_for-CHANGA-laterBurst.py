import yt
yt.set_log_level(0)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
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

def get_starform_time(gas_idx):
    time_lastgas = output[gas_idx]['time'][-1]
    idx_lastgas = np.argmin(abs(time_list/1e3 - time_lastgas))
    idx_firststar = idx_lastgas + step
    return time_list[idx_firststar]/1e3



codetp = 'CHANGA'
merger_number = '0'
halotree_ver = 2013

redshift_list, time_list, pfs, step = load_halotree_and_pfs(codetp, halotree_ver, rawtree_skip=True)
metadata_dir = '/work/hdd/bezm/tnguyen2/AGORA/%s/metadata/' % codetp
    
dist_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/dist_relvel_j_ProgBranch-%s_FirstMerger_%s_ver2013.npy' % (merger_number, codetp), allow_pickle=True).tolist()
prog_branch, sec_branch, sec_branch_2 = sec_branch_compute(codetp, merger_number)
idx_begin, idx_endinfall, idx_1stpass, idx_maxdist, idx_cls, time_begin, time_endinfall, time_1stpass, time_maxdist, time_cls = load_timings(codetp, halotree_ver, merger_number)
time_eval = time_1stpass + 0.6

if codetp == 'GEAR':
    idx_eval = np.argmin(abs(time_eval - time_list/1e3)) - (np.argmin(abs(time_eval - time_list/1e3)) % step) - 1
else:
    idx_eval = np.argmin(abs(time_eval - time_list/1e3)) - (np.argmin(abs(time_eval - time_list/1e3)) % step)
    
#----------------------------------------------------------------------------------------------------------------------------------------
output = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/Trace_Gas_Particles_that_Turns_Into_Starburst_Stars_ProgBranch-0_%s_ver2013.npy' % codetp, allow_pickle=True).tolist()

axlim = 70 #kpc
line_width = 1
alpha = 0.05 #0.3 before
alpha_scat = 0.5
if axlim < 30:
    size_scat = 5
else:
    size_scat = 2
font_size = 20

# ---------------------------------------------------------------------------------------------------------------------------------------
# Figure: 2 rows x 3 columns.
#   columns -> 0: trajectory colored by time, 1: trajectory colored by temperature,
#              2: distance-vs-time colored by temperature (Code 2 top / Code 3 bottom)
#   rows    -> 0 (top): gas starting before coalescence (mt_*),
#              1 (bottom): gas starting within a snapshot window (db_*)
# NOTE: sharex/sharey removed so the right column can carry its own (time, distance)
#       axes and so every panel shows its own ticks + labels.
# ---------------------------------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(2, 3, figsize=(19, 13), constrained_layout=True)

norm_time = plt.Normalize(0, 1.358)
norm_temp = mcolors.LogNorm(10, 5e7)

# ---- spatial panels (columns 0 & 1): equal aspect, +/- axlim ----
spatial_axes = [ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]]
for a in spatial_axes:
    a.set_aspect('equal')
    a.set_xlim(-axlim, axlim)
    a.set_ylim(-axlim, axlim)
    a.set_xticks([-60, -40, -20, 0, 20, 40, 60])
    a.set_ylabel('y (kpc)', fontsize=font_size)
    a.tick_params('both', labelsize=font_size)

# x label on the bottom spatial row only
for a in [ax[1, 0], ax[1, 1]]:
    a.set_xlabel('x (kpc)', fontsize=font_size)

# ---- phase-space panels (column 2): auto aspect, own units ----
phase_axes = [ax[0, 2], ax[1, 2]]
for a in phase_axes:
    a.set_aspect('auto')
    a.set_ylabel('Distance (kpc)', fontsize=font_size)
    a.tick_params('both', labelsize=font_size)

# x label on the bottom phase panel only
ax[1, 2].set_xlabel(r'Time elapsed since $t_\text{start}$ (Gyr)', fontsize=font_size)

# title on the top-middle panel
ax[0, 1].set_title(codetp, fontsize=font_size)


def plot_trajectory(gas_idx, ax_time, ax_temp):
    x = np.array(output[gas_idx]['pos'])[:, 0]
    y = np.array(output[gas_idx]['pos'])[:, 1]
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    z_time = np.array(output[gas_idx]['time']) - time_begin
    z_temp = np.array(output[gas_idx]['temp'])
    #
    #zorder = np.argmin(abs(output[gas_idx]['time'][0] - dist_data['time'])) * 10
    #
    lc_time = LineCollection(segments, cmap='rainbow', norm=norm_time, alpha=alpha, zorder=1)
    lc_time.set_array(z_time)
    lc_time.set_linewidth(line_width)
    ax_time.add_collection(lc_time)
    #
    lc_temp = LineCollection(segments, cmap='plasma', norm=norm_temp, alpha=alpha, zorder=1)
    lc_temp.set_array(z_temp)
    lc_temp.set_linewidth(line_width)
    ax_temp.add_collection(lc_temp)
    #
    ax_time.scatter(x[0], y[0], marker='o', c=z_time[0], norm=norm_time, s=size_scat,
                    alpha=alpha_scat, cmap='rainbow', zorder=999999999, edgecolors='black', linewidths=0.15)


# ------------------------------------------------------------------
# distance-vs-time phase-space panels (column 2)  <- Code 2 / Code 3
# randomly select n_sample particles for plotting
# ------------------------------------------------------------------
def plot_phase(ax_phase, times, dists, temps, n_sample, seg_alpha, dist_cut=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    # build the pool of eligible indices AFTER applying the distance cut
    if dist_cut is None:
        eligible = np.arange(len(times))
    else:
        eligible = np.array([j for j in range(len(times))
                             if np.asarray(dists[j])[-1] <= dist_cut])
    # randomly draw n_sample from the eligible pool
    idxs = rng.choice(eligible, size=min(n_sample, len(eligible)), replace=False)
    for i in idxs:
        t = np.asarray(times[i])
        d = np.asarray(dists[i])
        T = np.asarray(temps[i])
        points = np.array([t, d]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap='plasma', norm=norm_temp, alpha=seg_alpha, zorder=100)
        lc.set_array(0.5 * (T[:-1] + T[1:]))            # temp at segment midpoint
        lc.set_linewidth(0.1)
        ax_phase.add_collection(lc)


db_time, db_dist, db_temp = [], [], []   # db = 'delayed burst'
mt_time, mt_dist, mt_temp = [], [], []   # mt = 'merger timescale'

for gas_idx in list(output.keys())[240000::]:
    t0 = output[gas_idx]['time'][0]
    # Top row
    if not (t0 > time_list[idx_cls]/1e3):
        plot_trajectory(gas_idx, ax[0, 0], ax[0, 1])
        mt_time.append(output[gas_idx]['time'] - time_begin)
        mt_dist.append(np.linalg.norm(output[gas_idx]['pos'], axis=1))
        mt_temp.append(output[gas_idx]['temp'])
    # Bottom row
    elif not (t0 > time_list[266]/1e3 or t0 < time_list[250]/1e3): #ver 3 (250-266); ver4 (248-256)
        plot_trajectory(gas_idx, ax[1, 0], ax[1, 1])
        db_time.append(output[gas_idx]['time'] - time_begin)
        db_dist.append(np.linalg.norm(output[gas_idx]['pos'], axis=1))
        db_temp.append(output[gas_idx]['temp'])

# ---- Code 2 -> top-right ;  Code 3 -> bottom-right ----
rng = np.random.default_rng(12345)  # shared generator; drop the seed for a fresh draw each run
plot_phase(ax[0, 2], mt_time, mt_dist, mt_temp, n_sample=2000, seg_alpha=0.6, dist_cut=None, rng=rng)
plot_phase(ax[1, 2], db_time, db_dist, db_temp, n_sample=2000, seg_alpha=0.6, dist_cut=90, rng=rng)

# LineCollection does not autoscale the axes, so set limits manually
all_t = np.concatenate([np.asarray(x) for x in mt_time])
all_d = np.concatenate([np.asarray(x) for x in mt_dist])
ax[0, 2].set_xlim(all_t.min(), 1.41) #ver3: max=1.41
ax[0, 2].set_ylim(all_d.min(), 100)

all_t = np.concatenate([np.asarray(x) for x in db_time])
all_d = np.concatenate([np.asarray(x) for x in db_dist])
ax[1, 2].set_xlim(all_t.min(), 1.41) #ver3: max=1.41
ax[1, 2].set_ylim(all_d.min(), 100)

ax[0, 2].plot(dist_data['time'] - time_begin, dist_data['prog_radius'], linestyle='--', dashes=(5, 10), color='k', linewidth=0.7, zorder=1)
ax[1, 2].plot(dist_data['time'] - time_begin, dist_data['prog_radius'], linestyle='--', dashes=(5, 10), color='k', linewidth=0.7, zorder=1, label='$R_{200c}$')
ax[0, 2].plot(dist_data['time'] - time_begin, 0.2*dist_data['prog_radius'], linestyle=':', color='k', linewidth=0.7, zorder=100)
ax[1, 2].plot(dist_data['time'] - time_begin, 0.2*dist_data['prog_radius'], linestyle=':', color='k', linewidth=0.7, zorder=100, label='$0.2R_{200c}$')
ax[1, 2].legend(fontsize=font_size, loc='upper right', framealpha=0.9)


# ---------------------------------------------------------------------------------------------------------------------------------------
# Horizontal colorbars, one per column, placed beneath each column
#   col 0 -> time (rainbow) ; col 1 -> temperature (plasma, trajectories)
#   col 2 -> temperature (plasma), SHARED by Code 2 (top) and Code 3 (bottom)
# ---------------------------------------------------------------------------------------------------------------------------------------
sm_time = plt.cm.ScalarMappable(norm=norm_time, cmap='rainbow'); sm_time.set_array([])
cbar_time = fig.colorbar(sm_time, ax=ax[:, 0], location='bottom', fraction=0.046, pad=0.04)
cbar_time.set_label(r'Time elapsed since $t_\mathrm{start}$ (Gyr)', fontsize=font_size)
cbar_time.ax.tick_params(labelsize=font_size)

sm_temp1 = plt.cm.ScalarMappable(norm=norm_temp, cmap='plasma'); sm_temp1.set_array([])
cbar_temp1 = fig.colorbar(sm_temp1, ax=ax[:, 1], location='bottom', fraction=0.046, pad=0.04)
cbar_temp1.set_label('T (K)', fontsize=font_size)
cbar_temp1.ax.tick_params(labelsize=font_size)

sm_temp2 = plt.cm.ScalarMappable(norm=norm_temp, cmap='plasma'); sm_temp2.set_array([])
cbar_temp2 = fig.colorbar(sm_temp2, ax=ax[:, 2], location='bottom', fraction=0.046, pad=0.04)
cbar_temp2.set_label('T (K)', fontsize=font_size)
cbar_temp2.ax.tick_params(labelsize=font_size)


fig.savefig('/work/hdd/bezm/tnguyen2/figures/AGORA/November2025/Trace_Gas_Particles_that_Turns_Into_Starburst_Stars_%s_axlim_%s_ProgBranch-%s_ver2013_delayedStarburst_combined_ver5.png' % (codetp, merger_number, axlim), dpi=300, bbox_inches='tight')