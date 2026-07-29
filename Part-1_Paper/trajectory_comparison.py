import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

import setup
from importlib import reload
reload(setup)

from setup import codetp_list, label_list, color_list

fig, ax = plt.subplots(figsize=(8,7))
fontsize = 20

merger_number = '0'
for j in range(len(codetp_list)):
    codetp = codetp_list[j]
    data = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/dist_relvel_j_ProgBranch-%s_FirstMerger_%s_ver2013.npy' % (merger_number, codetp), allow_pickle=True).tolist()
    #
    time_list = np.array(data['time'])
    dist_list = np.array(data['dist'])
    #
    #Determine the 1st pass time and maxdist time using Cubic Spline to interpolate the distance curve
    cs = CubicSpline(time_list, dist_list)
    time_spline = np.linspace(min(time_list), max(time_list), 1000)
    time_1stpass = time_spline[np.where(np.diff(cs(time_spline))>0)[0][0]]
    time_plot = time_list - time_1stpass
    #
    ax.plot(time_plot, dist_list, color=color_list[j], linewidth=2.7, alpha=1, label=label_list[j])

ax.legend(fontsize=fontsize-2, ncol=2)
ax.set_ylim(bottom=-2)
ax.set_xlim(left = -0.4, right=0.5)
ax.set_xlabel('Normalized time (Gyr)', fontsize=fontsize)
ax.set_ylabel('Distance (kpc)', fontsize=fontsize)
ax.tick_params('both', labelsize=fontsize)
fig.tight_layout()
plt.savefig('/work/hdd/bezm/tnguyen2/figures/AGORA/November2025/distance_vs_time_ProgBranch-%s_ver2013_ver4.png' % merger_number, dpi=300, bbox_inches='tight')