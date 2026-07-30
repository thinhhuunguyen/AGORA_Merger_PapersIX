import yt
yt.set_log_level(0)
import numpy as np
import matplotlib.pyplot as plt

import setup
from importlib import reload
reload(setup)

from setup import load_halotree_and_pfs, load_timings
from setup import codetp_list, label_list, color_list


fig, ax = plt.subplots(figsize=(9,8))
ax2 = ax.twinx()
font_size = 20
merger_number   = '0'
halotree_ver    = 2013
sph_codetp_list = ['CHANGA','GADGET3', 'GADGET4', 'GEAR', 'GIZMO']

firstpass_handle = None  # for a single "First Pass" legend entry

for codetp in sph_codetp_list:
    (idx_begin, idx_endinfall, idx_1stpass, idx_maxdist, idx_cls,
     time_begin, time_endinfall, time_1stpass, time_maxdist, time_cls) = \
        load_timings(codetp, halotree_ver, merger_number)

    redshift_list, time_list, pfs, step = load_halotree_and_pfs(
        codetp, halotree_ver, rawtree_skip=True)

    data = np.load(
        '/work/hdd/bezm/tnguyen2/AGORA/analysis/'
        'Select_gas_with_negative_radialvel_infalling_gasparticletracing_%s_ver2013.npy' % codetp,
        allow_pickle=True).tolist()

    # Absolute landmark times (consistent units: Gyr)
    t_ei = time_endinfall
    t_fp = time_list[idx_1stpass] / 1e3
    t_md = time_list[idx_maxdist] / 1e3

    time_raw  = []
    mass_plot = []
    momentum_plot = []
    idx_plot  = []

    for idx in data.keys():
        if idx > idx_maxdist:
            continue

        temp_cond      = data[idx]['temp']      < np.inf
        dist_cond      = data[idx]['dist']      < 20   # kpc
        radialvel_cond = data[idx]['radialvel'] < 0
        combined = dist_cond & radialvel_cond & temp_cond

        mass_plot.append(np.sum(data[idx]['mass'][combined]))
        momentum_plot.append(np.sum(data[idx]['mass'][combined]*(-1)*data[idx]['radialvel'][combined]))
        time_raw.append(data[idx]['time'])
        idx_plot.append(idx)

    time_raw  = np.array(time_raw)
    mass_plot = np.array(mass_plot)
    momentum_plot = np.array(momentum_plot)
    idx_plot  = np.array(idx_plot)

    # ── simple linear normalization: endinfall=0, maxdist=1 ──────────────
    time_norm  = (time_raw - t_ei) / (t_md - t_ei)
    fp_norm    = (t_fp     - t_ei) / (t_md - t_ei)  # firstpass: wherever it falls

    color = color_list[codetp_list.index(codetp)]
    ax.plot(time_norm, mass_plot, label=label_list[codetp_list.index(codetp)], color=color, linewidth=2.5)

    if codetp == 'ART' or codetp == 'RAMSES' or codetp == 'GADGET3' or codetp == 'GEAR':
        ax2.plot(time_norm, momentum_plot, label=label_list[codetp_list.index(codetp)] + '*', color=color, linewidth=2.5, linestyle='--')
    elif codetp == 'CHANGA':
        ax2.plot(time_norm, momentum_plot, label=r'$\text{%s}^{\dagger}$' % label_list[codetp_list.index(codetp)], color=color, linewidth=2.5, linestyle='--')
    else:
        ax2.plot(time_norm, momentum_plot, label=label_list[codetp_list.index(codetp)], color=color, linewidth=2.5, linestyle='--')
    
    # Firstpass scatter at its true normalized position
    ax.scatter(fp_norm, mass_plot[idx_plot == idx_1stpass],
                    color=color, zorder=5, marker='|', s=200, linewidths=2)
    ax2.scatter(fp_norm, momentum_plot[idx_plot == idx_1stpass],
                    color=color, zorder=5, marker='|', s=200, linewidths=2)


# ── x-axis: only fixed landmarks as ticks ─────────────────────────────────
ax.set_ylim(bottom=2e6, top=1.8e9)
ax.set_xlim(0, 1)
ax.set_xticks([0.0, 1.0])
ax.set_yscale('log')
ax.set_ylabel(r'$M_\text{gas} (M_\odot)$', fontsize=font_size)
ax.tick_params('both', labelsize=font_size)

ax.text(0.04, 0.9, 
        r'$v_\text{r} < 0$' + '\n' + r'$r < 20\,\text{kpc}$',
        fontsize=font_size, transform=ax.transAxes,
        va='center', ha='left', linespacing=1.8,
        bbox=dict(boxstyle='square,pad=0.4', facecolor='none', edgecolor='black', linewidth=1.5))

ax.fill_between(x = np.linspace(0,0.64), y1 = -1e99, y2 = 1e99, alpha=0.15, color='grey')

ax2.set_ylim(bottom=2e8, top=4e11)
ax2.set_xlim(0, 1)
ax2.set_xticks([0.0, 1.0])
ax2.set_xticklabels([r'$[t_\text{start} + t_\text{max}]/2$' + '\n' '(End Infall)', r'$t_\text{max}$' + '\n' + '(First Apoapsis)'])
ax2.set_yscale('log')
ax2.set_ylabel(r'$p_{v_{r} < 0} = m|v_\text{r}| (M_\odot\text{km}\text{s}^{-1})$', fontsize=font_size)
ax2.tick_params('both', labelsize=font_size)

# Append "First Pass" marker to legend
handles, labels = ax2.get_legend_handles_labels()
leg = fig.legend(handles, labels, fontsize=font_size-4, ncol=3,  bbox_to_anchor=(0.86, 0.25))

for i in [1, 2, 4]:
    leg.get_texts()[i].set_color("blue")

from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color='black', linewidth=2.5, linestyle='-',  label=r'$M_\text{gas}$'),
    Line2D([0], [0], color='black', linewidth=2.5, linestyle='--', label=r'$p_{v_r < 0}$'),
]

fig.legend(handles=legend_elements, fontsize=font_size, bbox_to_anchor=(0.86, 0.35), ncol=2)  # nudge below the existing code legend

marker_handle = ax.scatter([], [], color='black', marker='|', s=200,  linewidths=2, label=r'$t_\text{fp}$ (First Periapsis)')

fig.legend(handles=[marker_handle], fontsize=font_size, bbox_to_anchor=(0.86, 0.45), ncol=1)

plt.tight_layout()
plt.savefig('/work/hdd/bezm/tnguyen2/figures/AGORA/November2025/infalling_gasmass_and_momentum_maxdist_stage_ver7.png', dpi=300, bbox_inches='tight')