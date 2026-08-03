from astropy.constants import G
import astropy.units as u
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker
import matplotlib.colors
import matplotlib.colorbar
from matplotlib.ticker import AutoMinorLocator, LinearLocator

import setup
from importlib import reload
reload(setup)

from setup import load_halotree_and_pfs, load_timings, load_ds, sec_branch_compute


# ----------------------------------------------------------------------
# Arguments and branches (unchanged)
# ----------------------------------------------------------------------
codetp = 'GEAR'
halotree_ver = 2013
merger_number = '0'
redshift_list, time_list, pfs, step = load_halotree_and_pfs(codetp, halotree_ver, rawtree_skip=True)
prog_branch, sec_branch, sec_branch_2 = sec_branch_compute(codetp, merger_number, halotree_ver)
dist_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/dist_relvel_j_ProgBranch-%s_FirstMerger_%s_ver2013.npy' % (merger_number, codetp), allow_pickle=True).tolist()
idx_begin, idx_endinfall, idx_1stpass, idx_maxdist, idx_cls, time_begin, time_endinfall, time_1stpass, time_maxdist, time_cls = load_timings(codetp, halotree_ver, merger_number)
time_eval = time_1stpass + 0.6
if codetp == 'GEAR':
    idx_eval = np.argmin(abs(time_eval - time_list/1e3)) - (np.argmin(abs(time_eval - time_list/1e3)) % step) - 1
else:
    idx_eval = np.argmin(abs(time_eval - time_list/1e3)) - (np.argmin(abs(time_eval - time_list/1e3)) % step)

Gv = G.to('kpc**3/ (Msun*s**2)').value
KPC_TO_KM = u.kpc.to('km') 

# ----------------------------------------------------------------------
def align_coordinates_with_angular_momentum(coords, j_direc):
    """Rotate coords so that j_direc -> z-axis (identical to decomposition.py)."""
    z_axis = j_direc
    x_axis = np.cross([0, 0, 1], z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    M = np.vstack((x_axis, y_axis, z_axis)).T
    return np.dot(coords, M)

z_axis_save = np.array([ 0.72083623, -0.54198586,  0.43202599])
epsilon_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/circularity_ProgBranch-%s_Idx_%s_%s_ver2013_Abadi2003method.npy' % ('GEAR', '0', idx_eval), allow_pickle=True).tolist()

#----------------------------------------------------------------------
# Matplotlib styling (from decomposition.py)
plt.style.use(["default"])
matplotlib.rc('lines', linewidth=3)
matplotlib.rc('font', family='monospace', weight='normal', size=16)
_c_frame = (0, 0, 0, .8)
for _tick in ('xtick', 'ytick'):
    matplotlib.rc(_tick + '.major', width=2, size=8)
    matplotlib.rc(_tick + '.minor', width=1.5, size=4, visible=True)
    matplotlib.rc(_tick, color=_c_frame, labelsize=15, direction='in')
matplotlib.rc('xtick', top=True)
matplotlib.rc('ytick', right=True)
matplotlib.rc('axes', linewidth=2.5, edgecolor=_c_frame, labelweight='normal')

# ----------------------------------------------------------------------
# Diagnostic imaging (Liang et al. 2024)
# Ported verbatim from decomposition.py (Liang & Jiang). plot_part_all renders
# one face-on or edge-on stellar surface-density panel (aligning each component
# by its OWN angular momentum); image_plot tiles bulge/halo/thin/thick + total
# face-on and edge-on, annotates mass fractions, and writes a PDF.
# Only the savefig path is changed: it now lands in the script's output dir and
# is stamped with codetp so parallel runs over different codes don't collide.
# ----------------------------------------------------------------------
def plot_part_all(ax, pos, Masses, vels, binmax, binwidth, vmin, vmax,
                  edge=True, fontsize=20, xlabel=False, ylabel=False, z_axis=np.array([None, None, None])):

    if (z_axis == None).all():
        total_angular_momentum = np.sum(np.cross(pos, vels * Masses[:, np.newaxis]), axis=0) / np.sum(Masses)
        z_axis = total_angular_momentum / np.linalg.norm(total_angular_momentum)
 
    newcoord = align_coordinates_with_angular_momentum(pos, z_axis)
    coordx = newcoord.T[0]
    coordy = newcoord.T[1]
    coordz = newcoord.T[2]
 
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_major_locator(LinearLocator(numticks=5))
    if edge == True:
        ax.yaxis.set_major_locator(LinearLocator(numticks=3))
    else:
        ax.yaxis.set_major_locator(LinearLocator(numticks=5))
    ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:.0f}'))
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:.0f}'))
 
    xbinned = [-binmax, binmax]  # kpc
    ybinned = [-binmax, binmax]
    zbinned = [-binmax / 2, binmax / 2]
 
    range_bin = [xbinned, ybinned]
    extent = [xbinned[0], xbinned[1], ybinned[0], ybinned[1]]
    range_bin2 = [xbinned, zbinned]
    extent2 = [xbinned[0], xbinned[1], zbinned[0], zbinned[1]]
 
    xnbin = (xbinned[1] - xbinned[0]) / binwidth
    ynbin = (ybinned[1] - ybinned[0]) / binwidth
    znbin = (zbinned[1] - zbinned[0]) / binwidth
 
    nbins = np.array([np.ceil(xnbin).astype(int), np.ceil(ynbin).astype(int)])
    nbins2 = np.array([np.ceil(xnbin).astype(int), np.ceil(znbin).astype(int)])
 
    if edge == True:
        R, tedges, zedges = np.histogram2d(coordx, coordz, weights=Masses / binwidth / binwidth, bins=nbins2, range=range_bin2, density=False)
        skill = np.where(R == 0.)
        R[skill] = 1
        R = np.log10(R)
        skill = np.where(R.T == 0.)
        alpha_mask = np.ones((R.T.shape[0], R.T.shape[1]))
        alpha_mask[skill] = 0
        ax.imshow(R.T, cmap='coolwarm', interpolation='spline16', extent=extent2, origin='lower',
                  aspect='auto', vmin=vmin, vmax=vmax, alpha=alpha_mask)
        if xlabel == True:
            ax.set_xlabel('X [kpc]', fontsize=fontsize)
            ax.tick_params(axis='x', labelsize=fontsize)
        else:
            ax.tick_params(axis='x', labelsize=0)
        if ylabel == True:
            ax.set_ylabel('Z [kpc]', fontsize=fontsize)
            ax.tick_params(axis='y', labelsize=fontsize)
        else:
            ax.tick_params(axis='y', labelsize=0)
    else:
        R, tedges, zedges = np.histogram2d(coordx, coordy, weights=Masses / binwidth / binwidth, bins=nbins, range=range_bin, density=False)
        skill = np.where(R == 0.)
        R[skill] = 1
        R = np.log10(R)
        skill = np.where(R.T == 0.)
        alpha_mask = np.ones((R.T.shape[0], R.T.shape[1]))
        alpha_mask[skill] = 0
        ax.imshow(R.T, cmap='coolwarm', interpolation='spline16', extent=extent, origin='lower',
                  aspect='auto', vmin=vmin, vmax=vmax, alpha=alpha_mask)
        if xlabel == True:
            ax.set_xlabel('X [kpc]', fontsize=fontsize)
            ax.tick_params(axis='x', labelsize=fontsize)
        else:
            ax.tick_params(axis='x', labelsize=0)
        if ylabel == True:
            ax.set_ylabel('Y [kpc]', fontsize=fontsize)
            ax.tick_params(axis='y', labelsize=fontsize)
        else:
            ax.tick_params(axis='y', labelsize=0)


def image_plot(pos_s, mass_s, vel_s, decomposition, prog_branch, snap):
    PLOT_LIM = 12.0  # kpc; x/y axis limit (edge-on z stays PLOT_LIM/3.3)

    fig = plt.figure(figsize=(24, 9))                 # 4 columns now, not 5
    gs = gridspec.GridSpec(2, 4, height_ratios=[0.4, 0.2])

    # Define axes for each subplot
    ax = []
    for i in range(2):
        ax_row = []
        for j in range(4):
            ax_row.append(fig.add_subplot(gs[i, j]))
        ax.append(ax_row)
    # Set equal aspect for the face-on (top) row
    for j in range(4):
        ax[0][j].set(adjustable="box", aspect="equal")

    #vmin = np.log10(mass_s.min() / 0.05 / 0.05)
    #vmax = np.log10(np.sum(mass_s) / 1500 / 0.05 / 0.05)
    vmin = 7.3
    vmax = 9.0
    
    # Spheroid = Bulge(1) + Halo(2); ThinDisk = 3; ThickDisk = 4
    spheroid_mask = (decomposition == 1) | (decomposition == 2)
    masks = [spheroid_mask, decomposition == 3, decomposition == 4]

    for i, mask in enumerate(masks):
        if i == 0:
            plot_part_all(ax[0][i], pos_s[mask], mass_s[mask], vel_s[mask], binmax=PLOT_LIM, binwidth=0.05, edge=False, vmin=vmin, vmax=vmax, ylabel=True, z_axis=z_axis_save)
            plot_part_all(ax[1][i], pos_s[mask], mass_s[mask], vel_s[mask], binmax=PLOT_LIM, binwidth=0.05, edge=True, vmin=vmin, vmax=vmax, ylabel=True, xlabel=True, z_axis=z_axis_save)
        else:
            plot_part_all(ax[0][i], pos_s[mask], mass_s[mask], vel_s[mask], binmax=PLOT_LIM, binwidth=0.05, edge=False, vmin=vmin, vmax=vmax, z_axis=z_axis_save)
            plot_part_all(ax[1][i], pos_s[mask], mass_s[mask], vel_s[mask], binmax=PLOT_LIM, binwidth=0.05, edge=True, vmin=vmin, vmax=vmax, xlabel=True, z_axis=z_axis_save)

    # Total in the last column
    plot_part_all(ax[0][3], pos_s, mass_s, vel_s, binmax=PLOT_LIM, binwidth=0.05, edge=False, vmin=vmin, vmax=vmax, z_axis=z_axis_save)
    plot_part_all(ax[1][3], pos_s, mass_s, vel_s, binmax=PLOT_LIM, binwidth=0.05, edge=True, vmin=vmin, vmax=vmax, xlabel=True, z_axis=z_axis_save)

    ax[0][0].text(s=r"Spheroid", x=0.03, y=0.9, transform=ax[0][0].transAxes, color="darkred", fontsize=20)
    ax[0][1].text(s=r"ThinDisk", x=0.03, y=0.9, transform=ax[0][1].transAxes, color="blue", fontsize=20)
    ax[0][2].text(s=r"ThickDisk", x=0.03, y=0.9, transform=ax[0][2].transAxes, color="darkgreen", fontsize=20)
    ax[0][3].text(s=r"Total", x=0.03, y=0.9, transform=ax[0][3].transAxes, color="black", fontsize=20)
    # (Total-panel text removed per request)

    axadd = fig.add_axes([0.91, 0.12, 0.01, 0.755])
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cb = matplotlib.colorbar.ColorbarBase(axadd, cmap=plt.cm.coolwarm, norm=norm, orientation='vertical')
    cb.set_label('$\\log_{10} \\Sigma_*/(M_\\odot kpc^{-2})$', fontsize=20)
    axadd.tick_params(axis='x', labelsize=20)
    plt.subplots_adjust(wspace=0.08, hspace=0.1)
    fig.suptitle(r'$t = t_\text{post/eq}$', x=0.5, y=0.95, fontsize=20, fontfamily='monospace')
    #fig.suptitle(r'$t = t_\text{post/eq} - %.2f$ Gyr' % (float(time_list[idx]/1e3) - float(time_list[idx - 2*step]/1e3)), x=0.5, y=0.95, fontsize=20, fontfamily='monospace')
    #fig.suptitle(r'$t = t_\text{post/eq} + %.2f$ Gyr' % (float(time_list[idx + 2*step]/1e3) - float(time_list[idx]/1e3)), x=0.5, y=0.95, fontsize=20, fontfamily='monospace')
    plt.savefig('/work/hdd/bezm/tnguyen2/figures/AGORA/decomposition_%s_snap%d_%d.png' % (codetp, snap, prog_branch), dpi=300, bbox_inches="tight")
    plt.show()

# RUN THESE LINES FOR SNAPSHOTS 446, 452 (IDX_EVAL), AND 458, THEN COMBINE THE FIGURES VERTICALLY USING THIRD PARTY SOFTWARE OR COMMAND LINES
idx = idx_eval
ds = load_ds(codetp, idx, pfs)
try:
    gal_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/%s/radius_2000_%s/Galaxy_Halo_%s_Snapshot_%s_comR2000.npy' % (codetp, codetp, prog_branch, idx), allow_pickle=True).tolist()
    try:
        center = gal_data['com']
    except:
        center = gal_data['gal_com']
except:
    dist_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/dist_relvel_j_ProgBranch-%s_FirstMerger_%s_ver2013.npy' % (merger_number, codetp), allow_pickle=True).tolist()
    center = np.array(dist_data['prog_com_plot'])[dist_data['idx'] == idx][0]
center_kpc = (center * ds.units.code_length).to('kpc').v

star_plot = epsilon_data['pos'] - center_kpc  # Centering the stellar position
mass_plot = epsilon_data['mass']
vel_plot = np.zeros(shape=star_plot.shape) #we don't need the velocity information when we already specify z_axis_save
labels_plot = epsilon_data['label']

image_plot(star_plot, mass_plot, vel_plot, labels_plot, prog_branch, idx)
print('saved decomposition image: decomp_snap%d_%d.png' % (idx, prog_branch))
