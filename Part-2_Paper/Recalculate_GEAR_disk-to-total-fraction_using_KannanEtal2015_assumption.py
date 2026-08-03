from scipy.ndimage import gaussian_filter1d
from astropy.constants import G
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt

import setup
from importlib import reload
reload(setup)

from setup import load_halotree_and_pfs, load_timings, sec_branch_compute

# Loading data
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

#------------------------------------------
Gv = G.to('kpc**3/ (Msun*s**2)').value
KPC_TO_KM = u.kpc.to('km') 
idx = idx_eval
# Load the circularity assuming the bulge's circularity symmetry around zero
output = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/circularity_ProgBranch-%s_Idx_%s_%s_ver2013_Abadi2003method.npy' % (codetp, prog_branch, idx), allow_pickle=True).tolist()

#------------------------------------------
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Recalculate the D/T of GEAR (using Kannan et al 2015 method)
rng = np.random.default_rng(42)
# ---- Load ----
data = output
circ = np.asarray(data['circ'], float)
mass = np.asarray(data['mass'], float)
g = np.isfinite(circ) & np.isfinite(mass)
circ, mass = circ[g], mass[g]
Mtot = mass.sum()

# ---- Smoothed circularity distribution function ----
lo, hi, nb = -1.5, 1.5, 300
edges = np.linspace(lo, hi, nb+1)
ctr = 0.5*(edges[:-1]+edges[1:]); bw = ctr[1]-ctr[0]
h, _ = np.histogram(circ, bins=edges, weights=mass)

SIGMA_EPS = 0.05                      # fiducial smoothing (in units of epsilon)
f  = gaussian_filter1d(h, SIGMA_EPS/bw, mode='nearest')
f1 = np.gradient(f, ctr)
f2 = np.gradient(f1, ctr)

# ---- Kannan separation point, searched only in -0.3 <= eps <= 0.3 ----
WIN = (-0.3, 0.3)
inwin = (ctr >= WIN[0]) & (ctr <= WIN[1])

def interp_zero(x, y, i):             # linear interp of y=0 between i-1 and i
    return x[i-1] - y[i-1]*(x[i]-x[i-1])/(y[i]-y[i-1])

eps_star, method = None, None
# 1) local maximum: f1 changes + -> -
for i in range(1, len(ctr)-1):
    if inwin[i] and f1[i] > 0 >= f1[i+1]:
        eps_star, method = interp_zero(ctr, f1, i+1), 'local maximum (f\'=0, +->-)'
        break
# 2) fallback: inflection where f2 changes - -> +
if eps_star is None:
    for i in range(1, len(ctr)-1):
        if inwin[i] and f2[i] < 0 <= f2[i+1]:
            eps_star, method = interp_zero(ctr, f2, i+1), 'inflection (f\'\'=0, -->+)'
            break

if eps_star is None:
    raise SystemExit("No max or inflection in [-0.3,0.3] -> Kannan: no bulge component.")

print(f"Separation point  eps* = {eps_star:+.3f}   via {method}")

# ---- Reflection decomposition (bulge symmetric about eps*) ----
fsm = np.interp(np.clip(2*eps_star - circ, lo, hi), ctr, f)   # mirrored DF at each particle
ftot = np.interp(circ, ctr, f)
p_bulge = np.where(circ <= eps_star, 1.0,
                   np.clip(np.divide(fsm, ftot, out=np.zeros_like(fsm), where=ftot>0), 0, 1))
is_bulge = rng.random(circ.size) < p_bulge
is_disc  = ~is_bulge


# analytic B/T = 2 * M(eps<eps*) / Mtot
M_left = mass[circ < eps_star].sum()
BT_analytic = min(1.0, 2*M_left/Mtot)
BT_discrete = mass[is_bulge].sum()/Mtot
print(f"B/T (analytic 2*M_left/M)   = {BT_analytic:.3f}")
print(f"B/T (discrete assignment)   = {BT_discrete:.3f}")
print(f"Bulge particles = {is_bulge.sum()},  Disc particles = {is_disc.sum()}")


# ---- Sensitivity of eps* to smoothing ----
print("\nSensitivity of eps* to smoothing bandwidth:")
for se in [0.03, 0.04, 0.045, 0.05, 0.055, 0.06, 0.08]:
    ff = gaussian_filter1d(h, se/bw, mode='nearest')
    ff2 = np.gradient(np.gradient(ff, ctr), ctr)
    ff1 = np.gradient(ff, ctr)
    es, mt = None, '--'
    for i in range(1, len(ctr)-1):
        if inwin[i] and ff1[i-1] > 0 >= ff1[i+1]:
            es, mt = interp_zero(ctr, ff1, i+1), 'max'; break
    if es is None:
        for i in range(1, len(ctr)-1):
            if inwin[i] and ff2[i-1] < 0 <= ff2[i+1]:
                es, mt = interp_zero(ctr, ff2, i+1), 'infl'; break
    bt = min(1.0, 2*mass[circ < es].sum()/Mtot) if es is not None else float('nan')
    print(f"  sigma_eps={se:.2f}:  eps*={'  none' if es is None else f'{es:+.3f}'}"
          f"  ({mt})   B/T={bt:.3f}")

# ---- Plot (for checking and visualization) ----
fig, ax = plt.subplots(2, 1, figsize=(7.2, 7.4), sharex=True,
                       gridspec_kw=dict(height_ratios=[2.2, 1]))

h_b, _ = np.histogram(circ[is_bulge], bins=edges, weights=mass[is_bulge])
h_d, _ = np.histogram(circ[is_disc],  bins=edges, weights=mass[is_disc])
ax[0].bar(ctr, h, width=bw, color='0.85', edgecolor='0.7', lw=0.3, label='Total')
ax[0].bar(ctr, h_b, width=bw, color='C3', alpha=0.55, label=f'Bulge (B/T={BT_discrete:.2f})')
ax[0].bar(ctr, h_d, width=bw, color='C0', alpha=0.55, label='Disc')
ax[0].plot(ctr, f, 'k-', lw=1.4, label=f'Smoothed DF ($\\sigma_\\epsilon$={SIGMA_EPS})')
ax[0].axvline(eps_star, color='k', ls=':', lw=1.5)
ax[0].axvspan(*WIN, color='gold', alpha=0.12)
ax[0].annotate(f'$\\epsilon^*={eps_star:+.3f}$', (eps_star, ax[0].get_ylim()[1]*0.92),
               xytext=(8,0), textcoords='offset points', fontsize=9)
ax[0].set_ylabel('Mass per bin'); ax[0].legend(frameon=False, fontsize=8.5)
ax[0].set_ylim(bottom=0)

ax[1].axhline(0, color='0.6', lw=0.7)
ax[1].plot(ctr, f1/np.abs(f1).max(), color='C2', lw=1.3, label="f' (norm.)")
ax[1].plot(ctr, f2/np.abs(f2).max(), color='C4', lw=1.3, label="f'' (norm.)")
ax[1].axvline(eps_star, color='k', ls=':', lw=1.5)
ax[1].axvspan(*WIN, color='gold', alpha=0.12, label='search window')
ax[1].set_xlabel(r'Circularity  $\epsilon = j_z/j_c$')
ax[1].set_ylabel('normalised derivative')
ax[1].set_xlim(lo, hi); ax[1].legend(frameon=False, fontsize=8.5, ncol=2)

fig.tight_layout()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Update the output dictionary with the new labels (from the new D/T decomposition)
labels_sel_fix = np.ones(len(output['label']))
labels_sel_fix[is_disc] = -1
labels_sel_fix[is_bulge] = 1
output['label_fix'] = labels_sel_fix
np.save('/work/hdd/bezm/tnguyen2/AGORA/analysis/circularity_ProgBranch-%s_Idx_%s_%s_ver2013_Abadi2003method.npy' % (codetp, prog_branch, idx), output)