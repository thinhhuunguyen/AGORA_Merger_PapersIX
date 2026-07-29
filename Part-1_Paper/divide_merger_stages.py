import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

import setup
from importlib import reload
reload(setup)

from setup import load_timings
from setup import codetp_list

j = 2 #use RAMSES as an example
codetp = codetp_list[j]
merger_number = '0'
halotree_ver = 2013
idx_begin, idx_endinfall, idx_1stpass, idx_maxdist, idx_cls, time_begin, time_endinfall, time_1stpass, time_maxdist, time_cls = load_timings(codetp, halotree_ver, merger_number)

font_size = 18

data = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/dist_relvel_j_ProgBranch-%s_FirstMerger_%s_ver2013.npy' % (merger_number, codetp), allow_pickle=True).tolist()
#
time_list = np.array(data['time'])
dist_list = np.array(data['dist'])


cs = CubicSpline(time_list, dist_list)
time_spline = np.linspace(min(time_list), max(time_list), 1000)
time_1stpass = time_spline[np.where(np.diff(cs(time_spline))>0)[0][0]]
time_maxdist = time_spline[time_spline > time_1stpass][np.argmax(cs(time_spline)[time_spline > time_1stpass])]
time_begin = time_list[0]
time_cls = time_list[data['idx'] == idx_cls][0]


plt.figure(figsize=(9,8))
plt.plot(time_list, dist_list, '.-', color='black', linewidth=2)
plt.axvline(time_maxdist, color='black', linestyle=':')
plt.axvline(time_cls, color='black', linestyle=':')

plt.fill_between (x = np.linspace((time_begin + time_maxdist)/2, time_list[0]), y1 = -2, y2 = 41, alpha=0.2, color='cyan')
plt.text(x=1.16, y=20, s='infall\n stage', fontsize=font_size - 2, weight='bold', ha='center') 
plt.fill_between (x = np.linspace(time_maxdist, (time_begin + time_maxdist)/2), y1 = -2, y2 = 41, alpha=0.2, color='orange')
plt.text(x=1.39, y=20, s='first \npassage\n stage', fontsize=font_size - 2, weight='bold', ha='center') #for GADGET3
plt.fill_between (x = np.linspace(time_cls, time_maxdist), y1 = -2, y2 = 41, alpha=0.15, color='red')
plt.text(x=1.615, y=20, s='coalescence\n stage', fontsize=font_size - 2, weight='bold', ha='center') #for GADGET3
plt.fill_between (x = np.linspace(time_cls, time_cls + 0.2), y1 = -2, y2 = 41, alpha=0.15, color='darkred')
plt.text(x=1.845, y=20, s='post-\n coalescence\n stage', fontsize=font_size - 2, weight='bold', ha='center') #for GADGET3

plt.xlabel('Time Since Big Bang (Gyr)', fontsize=font_size)
plt.ylabel(r'Distance (kpc)', fontsize=font_size)
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)

plt.ylim(-2, 41)
plt.xlim(left=time_begin, right=time_cls + 0.2)
plt.tight_layout()

plt.savefig('/work/hdd/bezm/tnguyen2/figures/AGORA/November2025/define_merger_stages_ver3.png', dpi=300, bbox_inches='tight')