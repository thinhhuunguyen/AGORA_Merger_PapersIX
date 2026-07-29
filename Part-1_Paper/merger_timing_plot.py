import numpy as np
import matplotlib.pyplot as plt
import yt
import os
yt.set_log_level(0)
    
import setup
from importlib import reload
reload(setup)

from setup import load_halotree_and_pfs, load_timings, sec_branch_compute
from setup import codetp_list, label_list, color_list


fig, axs = plt.subplots(figsize=(10,8), ncols=2,  width_ratios=[2.21, 1])
ax = axs[0]
ax_cut = axs[1]
font_size = 16
merger_number = '0'
halotree_ver = 2013

cls_dist_lim = 0.025
cls_vel_lim = 0.1

for j in np.flip(range(len(codetp_list))):
    codetp = codetp_list[j]
    if os.path.exists('/work/hdd/bezm/tnguyen2/AGORA/analysis/merger_mass_ratio_%s_ver2013.npy' % codetp) == False:
        rawtree, redshift_list, time_list, pfs, step = load_halotree_and_pfs(codetp, halotree_ver)
        prog_branch, sec_branch, sec_branch_2 = sec_branch_compute(codetp, merger_number)
        idx_begin, idx_endinfall, idx_1stpass, idx_maxdist, idx_cls, time_begin, time_endinfall, time_1stpass, time_maxdist, time_cls = load_timings(codetp, halotree_ver, merger_number)
        idx_preinfall = idx_begin - step
        # Star information
        output_star1 = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/SF_data_for_Merger_%s_ver2013_ConvexHull.npy' % codetp, allow_pickle=True).tolist()
        mass_star1 = np.array(output_star1['stellar_mass'])
        idx_star1 = np.array(output_star1['idx'])
        output_star2 = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/SF_secondary_data_for_Merger_%s_ver2013_ConvexHull.npy' % codetp, allow_pickle=True).tolist()
        mass_star2 = np.array(output_star2['stellar_mass'])
        idx_star2 = np.array(output_star2['idx'])
        # Gas information
        output_gas1 = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/GasCoolingTime_ProgBranch-0_%s_ver2013_ConvexHull_preInfall_ver2.npy' % codetp, allow_pickle=True).tolist()
        mass_gas1 = output_gas1['gas_mass_hull']
        idx_gas1 = output_gas1['idx']
        output_gas2 = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/GasMassFraction_SecBranch-%s_%s_ver%s_ConvexHull_preInfall_ver2.npy' % (merger_number, codetp, halotree_ver), allow_pickle=True).tolist()
        mass_gas2 = np.array(output_gas2['gas_mass_total'])
        time_gas2 = np.array(output_gas2['time'])
        #Add mass ratio
        idx_preinfall = idx_begin - step
        dm_ratio = rawtree[sec_branch][idx_preinfall]['Halo_Mass']/rawtree[prog_branch][idx_preinfall]['Halo_Mass']
        star_ratio = mass_star2[idx_star2 == idx_preinfall][0]/mass_star1[idx_star1 == idx_preinfall][0]
        gasandstar_ratio = (mass_star2[idx_star2 == idx_preinfall][0]  +  mass_gas2[np.argmin(abs(time_gas2 - time_list[idx_preinfall]))]  )/(mass_star1[idx_star1 == idx_preinfall][0] + mass_gas1[np.argmin(abs(np.array(idx_gas1) - idx_preinfall))].sum() )
        total_ratio = (rawtree[sec_branch][idx_preinfall]['Halo_Mass'] + mass_star2[idx_star2 == idx_preinfall][0]  +  mass_gas2[np.argmin(abs(time_gas2 - time_list[idx_preinfall]))])/ ( rawtree[prog_branch][idx_preinfall]['Halo_Mass'] + mass_star1[idx_star1 == idx_preinfall][0] + mass_gas1[np.argmin(abs(np.array(idx_gas1) - idx_preinfall))].sum() )
        output = {}
        output['dm_ratio'] = dm_ratio
        output['star_ratio'] = star_ratio
        output['gasandstar_ratio'] = gasandstar_ratio
        output['total_ratio'] = total_ratio
        np.save('/work/hdd/bezm/tnguyen2/AGORA/analysis/merger_mass_ratio_%s_ver2013.npy' % codetp, output)
        print(codetp, dm_ratio, star_ratio, gasandstar_ratio, total_ratio)
    else:
        redshift_list, time_list, pfs, step = load_halotree_and_pfs(codetp, halotree_ver, rawtree_skip = True)
        prog_branch, sec_branch, sec_branch_2 = sec_branch_compute(codetp, merger_number)
        idx_begin, idx_endinfall, idx_1stpass, idx_maxdist, idx_cls, time_begin, time_endinfall, time_1stpass, time_maxdist, time_cls = load_timings(codetp, halotree_ver, merger_number)
        idx_preinfall = idx_begin - step
        #
        ratio_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/merger_mass_ratio_%s_ver2013.npy' % codetp, allow_pickle=True).tolist()
        dm_ratio, star_ratio, gasandstar_ratio, gassfandstar_ratio, gascoolandstar_ratio, total_ratio = ratio_data.values()

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # Adding merger timing
    ax.barh(label_list[j],time_cls - time_begin, left=time_begin, color=color_list[j],alpha=0.6)
    ax_cut.barh(label_list[j],time_cls - time_begin, left=time_begin, color=color_list[j],alpha=0.6)

    # Adding time of first pericentric
    ax.barh(label_list[j], 0.005, left=(time_begin+time_maxdist)/2-0.0025, color=color_list[j])
    ax.barh(label_list[j], 0.005, left=time_maxdist-0.0025, color=color_list[j])
    #ax.barh(label_list[j], 0.005, left=time_1stpass-0.0025, color=color_list[j])

    #
    ax_cut.text(2.47, label_list[j], '%.2f' % dm_ratio,color='grey', fontsize=font_size-2, va='center')
    ax_cut.text(2.47 + 0.14, label_list[j], '%.2f' % star_ratio,color='brown', fontsize=font_size-2, va='center')
    ax_cut.text(2.47 + 0.14*2, label_list[j], '%.2f' % gascoolandstar_ratio,color='blue', fontsize=font_size-2, va='center')


    if codetp == 'GEAR':
        y_pos = 2.38     # the y-value of the bar
        bar_height = 0.1             # adjust if your bars have different height
        bar_top = y_pos + bar_height/2
        bar_bottom = y_pos - bar_height/2
        # Choose where along x you want the slash marks
        x1 = time_begin + 0.71
        x2 = x1 + 0.03     # short diagonal segment length
        # First diagonal slash
        ax.plot([x1, x2], [bar_bottom, bar_top],
                transform=ax.transData, color=color_list[j], lw=1.5, clip_on=False)
        ax_cut.plot([1.981, 1.981+0.03], [bar_bottom, bar_top],
                transform=ax.transData, color=color_list[j], lw=1.5, clip_on=False)
        #
        y_pos = 1.62     # the y-value of the bar
        bar_height = 0.1             # adjust if your bars have different height
        bar_top = y_pos + bar_height/2
        bar_bottom = y_pos - bar_height/2
        ax.plot([x1, x2], [bar_bottom, bar_top],
                transform=ax.transData, color=color_list[j], lw=1.5, clip_on=False)
        ax_cut.plot([1.981, 1.981+0.03], [bar_bottom, bar_top],
                transform=ax.transData, color=color_list[j], lw=1.5, clip_on=False)

    print('Done for %s' % codetp)

    
        
ax.set_xlim(left=0.95, right=1.9)
ax.set_xlabel('Time since Big Bang (Gyr)', fontsize=font_size, loc='right')
ax.tick_params('both', labelsize=font_size)

ax_cut.set_xlim(2.3, 2.9)

ax.spines['right'].set_visible(False)
ax_cut.spines['left'].set_visible(False)
ax_cut.set_yticks([])
ax_cut.yaxis.set_visible(False)
ax_cut.tick_params('both', labelsize=font_size)
ax_cut.set_xticks([2.4, 2.6, 2.8])

d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax.plot([1, 1], [0, 1], transform=ax.transAxes, **kwargs)
ax_cut.plot([0, 0], [0, 1], transform=ax_cut.transAxes, **kwargs)

#Adding redshift axis

time_values = [1.2090205225967145, 1.3787629847766211, 1.5869003581794816, 1.853794946402306]
specific_redshifts = [5, 4.5, 4, 3.5]
ax_z = ax.twiny()
ax_z.set_xlim([0.95,1.9])
ax_z.set_xlabel('z',fontsize=font_size)
ax_z.xaxis.set_label_coords(0.8, 1.07) 
ax_z.xaxis.label.set_color('black') 
# Set the redshift labels based on the calculated values
ax_z.set_xticks(time_values)
ax_z.set_xticklabels(['{:.1f}'.format(z) for z in specific_redshifts])
ax_z.tick_params(axis='x',labelcolor='black',labelsize=font_size)

time_values = [2.478555046117643, 2.685865869218459]
specific_redshifts = [2.7, 2.5]
ax_z_cut = ax_cut.twiny()
ax_z_cut.set_xlim([2.3, 2.73])
ax_z_cut.xaxis.label.set_color('black') 
# Set the redshift labels based on the calculated values
ax_z_cut.set_xticks(time_values)
ax_z_cut.set_xticklabels(['{:.1f}'.format(z) for z in specific_redshifts])
ax_z_cut.tick_params(axis='x',labelcolor='black',labelsize=font_size)

ax_z.spines['right'].set_visible(False)
ax_z_cut.spines['left'].set_visible(False)
ax_z_cut.set_yticks([])
ax_z_cut.yaxis.set_visible(False)

# --- make room above the top bar and add column headers ---
ymin, ymax = ax_cut.get_ylim()
ax_cut.set_ylim(ymin, ymax + 0.35)
ax.set_ylim(ymin, ymax + 0.35)   # keep both panels vertically aligned

header_y = ymax - 0.1
ax_cut.text(2.48,          header_y, r'$\mu_\mathrm{DM}$',       color='grey',  fontsize=font_size-2, va='center')
ax_cut.text(2.47 + 0.165,   header_y, r'$\mu_\bigstar$',          color='brown', fontsize=font_size-2, va='center')
ax_cut.text(2.47 + 0.265, header_y, r'$\mu_\mathrm{baryon}$', color='blue',  fontsize=font_size-2, va='center')

# --- thin grey separators between text columns ---
sep_x = [2.59, 2.735]                 # midpoints between the three columns
sep_bottom = ymin + 0.5
sep_top = ymax - 0.6              # extend up to just above the headers
ax_cut.vlines(sep_x, sep_bottom, sep_top, color='grey', lw=0.5, alpha=0.7)

fig.tight_layout()
plt.savefig('/work/hdd/bezm/tnguyen2/figures/AGORA/November2025/Timing_ProgBranch-%s_ver2013_ver10.png' % merger_number, dpi=300, bbox_inches='tight')
