from astropy.constants import G
import astropy.units as u
import numpy as np
from scipy.ndimage import gaussian_filter1d
from matplotlib.lines import Line2D

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

idx = idx_eval
ds = load_ds(codetp, idx, pfs)
gal_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/%s/radius_2000_%s/Galaxy_Halo_%s_Snapshot_%s_comR2000.npy' % (codetp, codetp, prog_branch, idx), allow_pickle=True).tolist()
try:
    center = gal_data['com']
except:
    center = gal_data['gal_com']
center_kpc = (center * ds.units.code_length).to('kpc').v
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

# Matplotlib styling (from decomposition.py)
plt.style.use(["default"])
matplotlib.rc('lines', linewidth=3)
matplotlib.rc('font', family='sans-serif', weight='normal', size=16)
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
def plot_part_all(ax, pos, Masses, binmax, binwidth, vmin, vmax,
                  edge=True, fontsize=20, xlabel=False, ylabel=False, z_axis=np.array([None, None, None])):

    #if (z_axis == None).all():
    #    total_angular_momentum = np.sum(np.cross(pos, vels * Masses[:, np.newaxis]), axis=0) / np.sum(Masses)
    #    z_axis = total_angular_momentum / np.linalg.norm(total_angular_momentum)
 
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

def make_combined_figure(pos_s, epsilon_data, prog_branch, snap):
    font_size = 14
    PLOT_LIM  = 12.0
    vmin, vmax = 7.3, 9.0

    mass_s        = np.asarray(epsilon_data['mass'])
    decomposition = np.asarray(epsilon_data['label_fix'])

    fig = plt.figure(figsize=(16, 5.3))
    # Single row: two histograms, then a stacked pair of edge-on panels.
    outer = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1], wspace=0.30)

    ax1 = fig.add_subplot(outer[0, 0])   # Liang et al. 2025
    ax2 = fig.add_subplot(outer[0, 1])   # Kannan et al. 2015

    # Right column split vertically: spheroid edge-on on top, disk edge-on below.
    gs_right = outer[0, 2].subgridspec(2, 1, hspace=0.15)
    ax_sph  = fig.add_subplot(gs_right[0, 0])
    ax_disk = fig.add_subplot(gs_right[1, 0])

    bins    = np.linspace(-1.5, 1.5, 100)
    centers = 0.5 * (bins[:-1] + bins[1:])
    total_mass = np.sum(epsilon_data['mass'])

    # ===== Left: Liang et al. 2025 method =====
    label_names   = {1: "Bulge", 2: "Halo", 3: "Thin disc", 4: "Thick disc"}
    color_names_1 = {1: "red", 3: "blue", 4: "darkgreen"}

    hist, _ = np.histogram(epsilon_data['circ'], bins=bins, weights=epsilon_data['mass'])
    ax1.plot(centers, gaussian_filter1d(hist, sigma=2), lw=2, color='black', label='All')

    for label_i in [1]:
        mask = (epsilon_data['label'] == label_i)
        frac = np.sum(epsilon_data['mass'][mask]) / total_mass
        hist, _ = np.histogram(epsilon_data['circ'][mask], bins=bins, weights=epsilon_data['mass'][mask])
        ax1.plot(centers, gaussian_filter1d(hist, sigma=2), lw=2,
                 color=color_names_1[label_i], linestyle='-', alpha=1,
                 label=f"{label_names[label_i]} ({frac:.0%})")

    mask = (epsilon_data['label'] == 3) | (epsilon_data['label'] == 4)
    frac = np.sum(epsilon_data['mass'][mask]) / total_mass
    hist, _ = np.histogram(epsilon_data['circ'][mask], bins=bins, weights=epsilon_data['mass'][mask])
    ax1.plot(centers, gaussian_filter1d(hist, sigma=2), lw=2, color='orange',
             linestyle='-', label=f"Disk ({frac:.0%})")

    ax1.text(0.09, 0.84,
             'D/T = %.2f' % (1 - epsilon_data['mass'][epsilon_data['label'] == 1].sum() / epsilon_data['mass'].sum()),
             transform=ax1.transAxes, fontsize=font_size)
    ax1.axvline(0, color='grey', linestyle='--', alpha=0.7)
    ax1.axvline(epsilon_data['etacut'], color='blue', linestyle=':', alpha=0.7, lw=2)
    ax1.set_ylim(bottom=-0.01e8)
    ax1.set_title(r'Liang et al. 2025 method', fontsize=font_size, y=1.05)

    # ===== Middle: Kannan et al. 2015 method =====
    hist, _ = np.histogram(epsilon_data['circ'], bins=bins, weights=epsilon_data['mass'])
    ax2.plot(centers, gaussian_filter1d(hist, sigma=2), lw=2, color='black', label='All')

    color_names_2 = {1: "red", -1: "orange"}
    for label_i in [1, -1]:
        mask = (epsilon_data['label_fix'] == label_i)
        hist, _ = np.histogram(epsilon_data['circ'][mask], bins=bins, weights=epsilon_data['mass'][mask])
        ax2.plot(centers, gaussian_filter1d(hist, sigma=2), lw=2, color=color_names_2[label_i])

    eps_star = 0.243
    ax2.axvline(eps_star, color='grey', linestyle='--', alpha=0.7)
    ax2.text(0.09, 0.84,
             'D/T = %.2f' % (1 - epsilon_data['mass'][epsilon_data['label_fix'] == 1].sum() / epsilon_data['mass'].sum()),
             transform=ax2.transAxes, fontsize=font_size)
    ax2.set_ylim(bottom=-0.01e8)
    ax2.set_title(r'Kannan et al. 2015 (K15) method', fontsize=font_size, y=1.05)

    for ax in (ax1, ax2):
        ax.set_xlabel(r'$\epsilon = J_{z}/J_\text{circ}(E)$', fontsize=font_size)
        ax.tick_params('both', labelsize=font_size)
        ax.set_xticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
        ax.set_ylabel(r'$M_\bigstar (M_\odot)$', fontsize=font_size)

    # ===== Right column: stacked edge-on spheroid (top) + disk (bottom) =====
    plot_part_all(ax_sph,  pos_s[decomposition == 1],  mass_s[decomposition == 1],
                  binmax=PLOT_LIM, binwidth=0.05, edge=True, vmin=vmin, vmax=vmax,
                  fontsize=font_size, ylabel=True, xlabel=False, z_axis=z_axis_save)
    plot_part_all(ax_disk, pos_s[decomposition == -1], mass_s[decomposition == -1],
                  binmax=PLOT_LIM, binwidth=0.05, edge=True, vmin=vmin, vmax=vmax,
                  fontsize=font_size, ylabel=True, xlabel=True, z_axis=z_axis_save)

    ax_sph.text(0.03, 0.82, r"Spheroid (K15)", transform=ax_sph.transAxes, color="darkred", fontsize=font_size)
    ax_disk.text(0.03, 0.82, r"Disk (K15)", transform=ax_disk.transAxes, color="blue", fontsize=font_size)

    # Colorbar to the right of the stacked pair, spanning both edge-on panels.
    pos_top = ax_sph.get_position()
    pos_bot = ax_disk.get_position()
    cax = fig.add_axes([pos_bot.x1 + 0.012, pos_bot.y0, 0.012, pos_top.y1 - pos_bot.y0])
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cb = matplotlib.colorbar.ColorbarBase(cax, cmap=plt.cm.coolwarm, norm=norm, orientation='vertical')
    cb.set_label(r'$\log_{10} \Sigma_*/(M_\odot kpc^{-2})$', fontsize=font_size)
    cax.tick_params(labelsize=font_size)

    # ===== Shared legend (below the row) =====
    legend_elements = [
        Line2D([0], [0], color='black',  linestyle='-',  lw=2, label='Total'),
        Line2D([0], [0], color='red',    linestyle='-',  lw=2, label='Spheroid'),
        Line2D([0], [0], color='orange', linestyle='-',  lw=2, label='Disk'),
        Line2D([0], [0], color='grey',   linestyle='--', lw=2, label=r'$\epsilon_\mathrm{spheroid\text{-}center}$'),
        Line2D([0], [0], color='blue',   linestyle=':',  lw=2, label='Thin disk / thick disk threshold'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0),
               ncol=5, fontsize=font_size, frameon=True)

    plt.savefig('/work/hdd/bezm/tnguyen2/figures/AGORA/November2025/decomposition_combined_%s_snap%d_%d_Kannan2015method.png'
                % (codetp, snap, prog_branch), dpi=300, bbox_inches="tight")


# Run the function for plotting
star_pos = epsilon_data['pos'] - center_kpc  # Centering the stellar positions
make_combined_figure(star_pos, epsilon_data, prog_branch, idx)