from astropy.constants import G
import astropy.units as u
import numpy as np
import sys
from scipy.interpolate import interp1d
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, UnivariateSpline
from scipy import ndimage                              # smooth_curve (get_etacut helper)
from sklearn.mixture import GaussianMixture as GMM     # pip install scikit-learn
from intersect import intersection  

import setup
from importlib import reload
reload(setup)

from setup import load_halotree_and_pfs, load_timings, load_ds, sec_branch_compute

 
from pytreegrav import ConstructTree, PotentialTarget, AccelTarget 

def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K` dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
THETA       = 0.5    # pytreegrav opening angle
N_RAD       = 100    # radial bin EDGES for the reference grid (-> N_RAD-1 centers)
N_AZ        = 4      # in-plane azimuthal samples per radius (decomposition.py uses 4)
RMAXORHALF  = 5.0    # alignment uses stars within RMAXORHALF * half-mass radius
 
PARTICLE_CODES = ('CHANGA', 'GADGET3', 'GADGET4', 'GEAR', 'AREPO', 'GIZMO')
MESH_CODES     = ('ART', 'ENZO', 'RAMSES')
EPS_PLUMMER_PARTICLE_KPC = 0.080        # 80 pc, fixed for the particle codes


# ----------------------------------------------------------------------
# Arguments and branches (unchanged)
# ----------------------------------------------------------------------
codetp = sys.argv[1]
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
# Set the snapshot index where the stellar circularity is calculated 
# ----------------------------------------------------------------------
idx = idx_begin - step
#idx = idx_eval
ds = load_ds(codetp, idx, pfs)

# ----------------------------------------------------------------------
# Softening (M4 spline support radius) by code family
# ----------------------------------------------------------------------
def _code_family(name):
    n = name
    if any(n == c or n.startswith(c) for c in PARTICLE_CODES):
        return 'particle'
    if any(n == c or n.startswith(c) for c in MESH_CODES):
        return 'mesh'
    return None
 
_family = _code_family(codetp)
if _family == 'particle':
    H_SUPPORT = 1 * EPS_PLUMMER_PARTICLE_KPC
elif _family == 'mesh':
    H_SUPPORT = float(ds.index.get_smallest_dx().to('kpc').v)
else:
    raise ValueError("Unknown code '%s': add it to PARTICLE_CODES or MESH_CODES." % codetp)



# ----------------------------------------------------------------------
# Center, star metadata, membership (unchanged)
# ----------------------------------------------------------------------
if idx == idx_begin - step:
    center, center2, v_bulk, v_bulk2, _ = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/premerger_centers_%s_ver2013_ver2.npy' % (codetp), allow_pickle=True).tolist().values()
else:
    gal_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/%s/radius_2000_%s/Galaxy_Halo_%s_Snapshot_%s_comR2000.npy' % (codetp, codetp, prog_branch, idx), allow_pickle=True).tolist()
    try:
        center = gal_data['com']
    except:
        center = gal_data['gal_com']
center_kpc = (center * ds.units.code_length).to('kpc').v
 
metadata_dir = '/work/hdd/bezm/tnguyen2/AGORA/%s/metadata/' % codetp
metadata = np.load(metadata_dir + 'star_metadata_allbox_%s.npy' % idx, allow_pickle=True).tolist()
mass_all = metadata['mass']
pos_all = metadata['pos']
vel_all = metadata['vel']          # km/s
ID_all = metadata['ID']
 
assignment = np.load('/work/hdd/bezm/gtg115x/Halo_Finding/%s/star_id_%s_final_firstmerger.npy' % (codetp, halotree_ver), allow_pickle=True).tolist()
hullv = np.load('/work/hdd/bezm/gtg115x/Halo_Finding/%s/hullv_%s_withPos_final.npy' % (codetp, halotree_ver), allow_pickle=True).tolist()

ID_include = assignment['ids'][prog_branch][idx]
gal_bool = np.intersect1d(ID_include, ID_all, return_indices=True)[2]
del assignment
 
mass_gal = mass_all[gal_bool]
ID_gal = ID_all[gal_bool]
vel_gal = vel_all[gal_bool]                                       # km/s
pos_gal = (pos_all[gal_bool] * ds.units.code_length).to('m').v  # absolute kpc

#Limit to within the Convex Hull
hull_bool = in_hull(pos_gal, np.asarray(hullv[idx][prog_branch]) )

mass_gal = mass_gal[hull_bool]
ID_gal = ID_gal[hull_bool]
vel_gal = vel_gal[hull_bool]
pos_gal = pos_gal[hull_bool]
pos_gal = (pos_gal * ds.units.m).to('kpc').v  # absolute kpc

star_pos = pos_gal - center_kpc                                   # galactocentric kpc
radius_gal = np.linalg.norm(star_pos, axis=1)
 
# bulk velocity [km/s]
if idx != idx_begin - step:
    v_bulk = np.average(np.array(dist_data['prog_vel_plot'])[dist_data['idx'] == idx][0], weights=np.array(dist_data['prog_mass_plot'])[dist_data['idx'] == idx][0], axis=0)   # (3,) km/s
vpec_kms = vel_gal - v_bulk         

# ----------------------------------------------------------------------
# Total matter sources (baryon + DM) within the convex hull, galactocentric kpc
# ----------------------------------------------------------------------
#The dm_bary_data file is calculated in the file "extract_darkmatter_bary_MassPos_ConvexHull.py"
dm_bary_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/%s/metadata/dm_bary_Branch_%s_Idx_%s_ver2013_ConvexHull.npy'
                       % (codetp, prog_branch, idx), allow_pickle=True).tolist()
mass_total = np.append(dm_bary_data['bary_mass'], dm_bary_data['dm_mass'])     # Msun  #!! incl. stars
coor_total = np.vstack((dm_bary_data['bary_coor'], dm_bary_data['dm_coor']))   # kpc
src_pos = coor_total - center_kpc                                              # galactocentric kpc, 'src' = source
src_soft = np.full(len(mass_total), H_SUPPORT, dtype=float)                    # softening length


def align_coordinates_with_angular_momentum(coords, j_direc):
    """Rotate coords so that j_direc -> z-axis (identical to decomposition.py)."""
    z_axis = j_direc
    x_axis = np.cross([0, 0, 1], z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    M = np.vstack((x_axis, y_axis, z_axis)).T
    return np.dot(coords, M)


src_pos = np.ascontiguousarray(src_pos, dtype=np.float64)
mass_total = np.ascontiguousarray(mass_total, dtype=np.float64)
src_soft = np.ascontiguousarray(src_soft, dtype=np.float64)

# --- (1) disk alignment: J of stars within the Convex Hull ---
Jvec = np.sum(np.cross(star_pos, vpec_kms * mass_gal[:,np.newaxis]), axis=0) / np.sum(mass_gal)
z_axis = Jvec / np.linalg.norm(Jvec)

star_pos_a = align_coordinates_with_angular_momentum(star_pos, z_axis)
src_pos_a  = align_coordinates_with_angular_momentum(src_pos,  z_axis)


# --- build the tree once on aligned sources -------------------------------
tree = ConstructTree(src_pos_a, mass_total, softening=src_soft)
# --- star binding energies: full per-particle tree potential --------------
h_star = np.full(len(star_pos_a), H_SUPPORT)
phi_star = PotentialTarget(np.ascontiguousarray(star_pos_a), None, None,
                           softening_target=h_star, tree=tree,
                           G=Gv, theta=THETA, parallel=True)          # kpc^2/s^2
vpec = vpec_kms / KPC_TO_KM                                            # kpc/s
speed2 = np.sum(vpec**2, axis=1)
E_star = 0.5 * speed2 + phi_star                                      # kpc^2/s^2


# --- (2,3) midplane reference: log radial grid over stellar extent --------
r = radius_gal
M_r = 1.01 * np.max(r)
m_r = 0.99 * np.min(r[r > 0])
pbins = np.logspace(np.log10(m_r), np.log10(M_r), N_RAD, endpoint=True)
rxy = 0.5 * (pbins[:-1] + pbins[1:])                                  # bin centers [kpc]

phis = np.linspace(0.0, 2.0 * np.pi, N_AZ, endpoint=False)
nhat = np.column_stack([np.cos(phis), np.sin(phis), np.zeros(N_AZ)])  # (N_AZ,3) in-plane
pos_mid = rxy[:, None, None] * nhat[None, :, :]                       # (N_RAD-1, N_AZ, 3)
mid_pts = np.ascontiguousarray(pos_mid.reshape(-1, 3))

h_mid = np.full(len(mid_pts), H_SUPPORT)
phi_mid = PotentialTarget(mid_pts, None, None, softening_target=h_mid, tree=tree,
                          G=Gv, theta=THETA, parallel=True)
acc_mid = AccelTarget(mid_pts, None, None, softening_target=h_mid, tree=tree,
                      G=Gv, theta=THETA, parallel=True)

phi_mid = phi_mid.reshape(len(rxy), N_AZ)
acc_mid = acc_mid.reshape(len(rxy), N_AZ, 3)

mid_pots = phi_mid.mean(axis=1)                                       # < Phi >_phi   [kpc^2/s^2]
vc2 = np.mean(np.sum(-acc_mid * pos_mid, axis=2), axis=1)             # < -a.r >_phi  = v_c^2
vc2 = np.clip(vc2, 0.0, None)
vc = np.sqrt(vc2)                                                     # kpc/s

E_circ = 0.5 * vc2 + mid_pots                                        # kpc^2/s^2
j_circ = KPC_TO_KM * rxy * vc                                        # km*kpc/s



# --- (4) log-log interpolation  j_c(E) = 10**f(log10(-E)) -----------------
xp = np.log10(-E_circ)            # E_circ<0 for bound reference orbits
fp = np.log10(j_circ)
ok = np.isfinite(xp) & np.isfinite(fp)
xp, fp = xp[ok], fp[ok]
so = np.argsort(xp)
xp, fp = xp[so], fp[so]
keep = np.concatenate(([True], np.diff(xp) > 0))                      # strictly increasing
xp, fp = xp[keep], fp[keep]
j_from_E = interp1d(xp, fp, fill_value='extrapolate', bounds_error=False)

with np.errstate(invalid='ignore'):
    jcE = 10.0 ** j_from_E(np.log10(np.where(E_star < 0, -E_star, np.nan)))

# edge handling exactly as decomposition.py
jcE[E_star > E_circ.max()] = np.inf                                  # nearly unbound -> spheroid
jcE[E_star < E_circ.min()] = j_circ[0]                               # over-bound -> innermost floor


# --- (5) orbital circularity  eps = j_z / j_c(E) --------------------------
# j_z projects each star's specific angular momentum onto the disk axis
# (z_axis), computed in the ORIGINAL (unrotated) frame and dotted with
# z_axis -- the dot product is rotation-invariant, so this matches
# decomposition.py without re-rotating the velocities. Using pos[kpc] x
# v[km/s] yields j in km*kpc/s, the SAME units as j_circ above, so the
# ratio is dimensionless.
j_star  = np.cross(star_pos, vpec_kms)        # km*kpc/s, galactocentric rest frame
jz_star = j_star @ z_axis                      # km*kpc/s, signed (disk-axis projection)
jp_star = np.sqrt(np.clip(np.sum(j_star**2, axis=1) - jz_star**2, 0.0, None))  # in-plane |j|

circ  = jz_star / jcE                          # j_z / j_c(E);  -> 0 where jcE = inf
polar = jp_star / jcE                          # j_p / j_c(E);  -> 0 where jcE = inf


# --- (6) normalized binding energy  e_b = E / |E|_max ---------------------
eb_star = E_star / np.abs(E_star).max()        # dimensionless; <0 => bound, -1 => most bound


# ----------------------------------------------------------------------
# Energy-threshold finder (Zana et al. 2022 / Liang et al. 2024)
# Ported verbatim from decomposition.py (Liang & Jiang); only numpy is used.
# get_Ecut detects a local minimum in the e_b distribution; FindMin and
# RefineMin are its helpers. e_b MUST be the normalized binding energy
# E/|E|_max, and the Emin=-0.9 default is tied to that normalization.
# ----------------------------------------------------------------------
def get_Ecut(eb, masses, nbins=25, M_bin=400, m_bin=80, toll=1.5, shrink=2,
             Mmin=0.05, Emin=-0.9):
    """
    Calculate energy threshold for subhalo using method from Zana et al. 2022.
 
    Parameters
    ----------
    eb : array_like
        scaled binding energy of the selected particles
    masses : array_like
        mass of the selected particles [Msun]
 
    Returns
    ----------
    Ecut : float
        energy threshold found by algorithm
    """
    if len(eb) < 100:
        print('Ecut = ', 0)
        return 0
 
    # Fix the number of bins as a function of Npart
    NbinMax = max(min(int(0.5 * np.sqrt(len(eb))), M_bin), m_bin)
 
    # This is to exclude the outer tail of bound particles
    M_E = np.quantile(eb, 0.9)
    m_E = np.quantile(eb, 0.01)
    Ecut, E_val = FindMin(eb, m_E, M_E, nbins)
 
    # If no minimum is found or the only minimum is too close to -1 (Maybe a GC?)
    if len(Ecut) == 0 or (len(Ecut) == 1 and Ecut < Emin):
        M_E = np.max(eb)
        Ecut, E_val = FindMin(eb, m_E, M_E, nbins)
        Ecut = Ecut
    # If one or none minima are found
    if len(Ecut) <= 1:
        D = (M_E - m_E) / float(nbins)
        # Avoid the following loop
        nbins = NbinMax + 1
    else:
        D = (M_E - m_E) / float(nbins)
        lb = Ecut - (toll * D)
        rb = Ecut + (toll * D)
 
    # -------
    while nbins < NbinMax:
        nbins = shrink * nbins
        D = D / shrink
        pos_E_refined, val_refined = FindMin(eb, m_E, M_E, nbins)
        EcutTEMP = []
        E_valTEMP = []
        for i, v in enumerate(E_val):
            pTEMP = pos_E_refined[(pos_E_refined <= rb[i]) * (pos_E_refined >= lb[i])]
            vTEMP = val_refined[(pos_E_refined <= rb[i]) * (pos_E_refined >= lb[i])]
            if len(pTEMP) > 0:
                # A refined position and value for each original minimum is stored.
                # The value of the minima is summed to the original ones to avoid
                # strange local minima
                EcutTEMP.append(pTEMP[np.argmin(vTEMP)])
                E_valTEMP.append(v + np.min(vTEMP))
 
        Ecut = np.array(EcutTEMP)
        E_val = np.array(E_valTEMP)
 
        if len(Ecut) <= 1:
            break
 
        lb = Ecut - (toll * D)
        rb = Ecut + (toll * D)
 
    # -------
 
    # If no energy cut is found
    if len(Ecut) == 0:
        Ecut = 0
    else:
        # Try to avoid strange nuclear minima with low mass if there are better alternatives
        rel_filt = [bool((np.sum(masses[eb < E]) / np.sum(masses) >= Mmin) + (E >= Emin)) for E in Ecut]
        if len(Ecut[rel_filt]) == 0:
            Ecut = Ecut[np.argmin(E_val)]
        else:
            Ecut = Ecut[rel_filt][np.argmin(E_val[rel_filt])]
        Ecut = RefineMin(eb, Ecut, D, (M_E - m_E) / NbinMax, shrink)
 
    print('Ecut = ', Ecut)
    print('nbins = ', nbins)
 
    return Ecut
 
 
def FindMin(q, m_E, M_E, nbins):
    """
    Look for the minima in the distribution of energies q in [m_E, M_E] with
    nbins bins. Returns the central positions of the minima and their values
    (the values depend on nbins).
    """
    # Minimum number of particles to perform a reliable Jcirc decomposition
    if len(q) >= 1e4:
        Npart_min = 1000
    elif 1e3 <= len(q) < 1e4:
        Npart_min = 100
    else:
        Npart_min = 10
 
    MinPart = max(Npart_min, 0.01 * len(q))
    arr = q[(q >= m_E) * (q <= M_E)]
    # Build the histogram
    hist = np.histogram(arr, bins=np.linspace(m_E, M_E, nbins))
 
    # Evaluate the increment on both sides A
    diff = hist[0][1:] - hist[0][:-1]
    left = diff[:-1]
    right = diff[1:]
    # Find the minima
    id_E = np.where(((left < 0) * (right >= 0)) + ((left <= 0) * (right > 0)))
 
    # C
    R_part = np.array([np.sum(hist[0][i + 1:]) for i in id_E[0]])
    id_E = id_E[0][R_part > MinPart]
    id_E_flag = [True] * len(id_E)
 
    # B
    for i, ids in enumerate(id_E):
        if len(hist[0]) > ids + 3:
            id_E_flag[i] *= hist[0][ids + 3] > hist[0][ids + 1]
        if ids > 0:
            id_E_flag[i] *= hist[0][ids - 1] > hist[0][ids + 1]
 
    id_E = id_E[id_E_flag]
 
    # Return the central position of the bins
    return 0.5 * (hist[1][id_E + 2] + hist[1][id_E + 1]), hist[0][id_E + 1]
 
 
def RefineMin(q, Vmin, D, Dmin, shrink):
    """
    Recursively refine a minimum Vmin of the energy distribution q, within an
    interval of size D centred on Vmin, shrinking by `shrink` each cycle while
    D > Dmin.
    """
    arr = []
    if D <= Dmin:
        if len(q) >= 1e4:
            coe = 0.5
        elif 1e3 <= len(q) < 1e4:
            coe = 2
        else:
            coe = 2.5
        while len(arr) == 0:
            m_E = Vmin - coe * D
            M_E = Vmin + coe * D
            arr = q[(q >= m_E) * (q <= M_E)]
            coe = coe + 0.2
        Vmin = np.median(arr)
 
    while D > Dmin:
        if len(q) >= 1e3:
            coe = 1.5
        else:
            coe = 4
        m_E = max(Vmin - coe * D, q.min() + D)
        M_E = Vmin + coe * D
        D = D / shrink
        arr = q[(q >= m_E) * (q <= M_E)]
        hist = np.histogram(arr, bins=np.arange(m_E, M_E, D))
        hist_min = (hist[0][np.where(hist[0] != 0)]).min()
        pid = np.where(hist[0] == hist_min)[0][0]
        arr = arr[(arr >= hist[1][pid]) * (arr <= (hist[1][pid + 1]))]
        # Get the energy as the median within the selected bin
        Vmin = np.median(arr)
 
    return Vmin


# ----------------------------------------------------------------------
# Component labelling (Liang et al. 2024)
# Ported verbatim from decomposition.py (Liang & Jiang). Depends only on numpy
# and scipy.interpolate.UnivariateSpline.
#   Bulge = 1, Halo = 2, Thin disc = 3, Thick disc = 4.
# With etacut=None the disc is left un-split (disky stars keep label -1); pass a
# circularity threshold (e.g. the standard 0.7, or the output of get_etacut) to
# obtain the full four-component split.
# ----------------------------------------------------------------------
def assign_label(eb, jzojc, masses, Ecut, etacut=None):
    """
    Decompose galaxies into bulge, halo, thin disc and thick disc
    (method from Liang et al. 2024).
 
    Returns
    ----------
    labels_4comp : label index for the selected particles
        Bulge=1, Halo=2, Thin Disc=3, Thick Disc=4.
        If etacut is None (or both discs under-resolved), disky stars carry
        label -1.
    """
    # create empty label
    labels_4comp = np.zeros(len(eb)) - 1
    # separate energy into low-energy and high-energy components
    E_low = eb <= Ecut
    E_low_where = np.where(eb <= Ecut)[0]
 
    # Bulge:
    dist_low = np.histogram(jzojc[E_low],
                            bins=np.arange(np.nanmin(jzojc[E_low]), np.nanmax(jzojc[E_low]), 0.01),
                            weights=masses[E_low])
 
    if len(dist_low[0]) >= 4:
        PositiveCirc = jzojc[E_low] > 0
        c = 0.5 * (dist_low[1][1:] + dist_low[1][:-1])
        # distribution function of circularity
        Bspl = UnivariateSpline(c, dist_low[0], s=0)
        # mirrored circularity distribution function
        yBspl = Bspl(-jzojc[E_low][PositiveCirc])
        # The seed is fixed for reproducibility
        #np.random.seed(309)
        np.random.seed(10)
        # Probability for particles with circularity > 0 to be assigned bulge,
        # via the distribution function of negative circularity
        p = yBspl / Bspl(jzojc[E_low][PositiveCirc])
        # [0;1)
        ra = np.random.random(len(yBspl))
        id_pos = np.where(E_low * (jzojc > 0))[0]
        # MCMC sampling
        id_b = id_pos[ra <= p]
        # All particles with circularity <= 0 are bulge stars
        bulge = np.where((eb <= Ecut) * (jzojc <= 0))
 
        bulge = np.concatenate((bulge[0], id_b))
        labels_4comp[bulge] = 1
 
    # Similar to bulge but only when there is a halo in the galaxy, i.e. Ecut<0
    if Ecut < 0:
        # Halo
        dist_high = np.histogram(jzojc[~E_low],
                                 bins=np.arange(np.nanmin(jzojc[~E_low]), np.nanmax(jzojc[~E_low]), 0.01),
                                 weights=masses[~E_low])
 
        if len(dist_high[0]) >= 4:
            PositiveCirc = jzojc[~E_low] > 0
            c = 0.5 * (dist_high[1][1:] + dist_high[1][:-1])
            Hspl = UnivariateSpline(c, dist_high[0], s=0)
            yHspl = Hspl(-jzojc[~E_low][PositiveCirc])
 
            # Ratio between negative tail and positive part
            p = yHspl / Hspl(jzojc[~E_low][PositiveCirc])
            ra = np.random.random(len(yHspl))
            id_pos = np.where((~E_low) * (jzojc > 0))[0]
            id_h = id_pos[ra <= p]
 
            halo = np.where((eb > Ecut) * (jzojc <= 0))
            halo = np.concatenate((halo[0], id_h))
            labels_4comp[halo] = 2
 
    # Need a threshold to split disky stars into thin and thick disc; otherwise
    # disky stars stay labelled -1. Standard choice: 0.7
    if etacut == None:
        return labels_4comp
 
    # Disky stars with circularity >= threshold -> thin disc; < threshold ->
    # thick disc. Require at least 100 particles for each disc; if one fails it
    # is folded into the other disky component, if both fail they go to bulge or
    # halo by their energy. One can change or delete this requirement.
    else:
        ThinDisk = np.where((labels_4comp != 1) & (labels_4comp != 2) & (jzojc >= etacut))
        ThickDisk = np.where((labels_4comp != 1) & (labels_4comp != 2) & (jzojc < etacut))
 
        if len(ThinDisk[0]) < 100 and len(ThickDisk[0]) < 100:
            labels_4comp[np.where((labels_4comp != 1) & (labels_4comp != 2) & (jzojc >= etacut) & (eb > Ecut))] = 2
            labels_4comp[np.where((labels_4comp != 1) & (labels_4comp != 2) & (jzojc >= etacut) & (eb <= Ecut))] = 1
 
        elif len(ThinDisk[0]) < 100 and len(ThickDisk[0]) >= 100:
            labels_4comp[np.where((labels_4comp != 1) & (labels_4comp != 2) & (jzojc >= etacut))] = 4
            labels_4comp[np.where((labels_4comp != 1) & (labels_4comp != 2) & (jzojc >= etacut))] = 4
        else:
            labels_4comp[ThinDisk] = 3
 
        if len(ThickDisk[0]) < 100 and len(ThinDisk[0]) < 100:
            labels_4comp[np.where((labels_4comp != 1) & (labels_4comp != 2) & (jzojc < etacut) & (eb > Ecut))] = 2
            labels_4comp[np.where((labels_4comp != 1) & (labels_4comp != 2) & (jzojc < etacut) & (eb <= Ecut))] = 1
        elif len(ThickDisk[0]) < 100 and len(ThinDisk[0]) >= 100:
            labels_4comp[np.where((labels_4comp != 1) & (labels_4comp != 2) & (jzojc < etacut))] = 3
            labels_4comp[np.where((labels_4comp != 1) & (labels_4comp != 2) & (jzojc < etacut))] = 3
        else:
            labels_4comp[ThickDisk] = 4
 
        # Require at least 30 particles for bulges and halos; if one fails it is
        # folded into the other spheroidal component. One can change or delete.
        Bulge = np.where(labels_4comp == 1)
        Halo = np.where(labels_4comp == 2)
        if len(Bulge[0]) < 30 and len(Halo[0]) >= 30:
            labels_4comp[Bulge] = 2
        elif len(Bulge[0]) >= 30 and len(Halo[0]) < 30:
            labels_4comp[Halo] = 1
 
        return labels_4comp

# ----------------------------------------------------------------------
# Circularity-threshold finder (Liang et al. 2024)
# Ported verbatim from decomposition.py (Liang & Jiang). get_etacut takes the
# disky stars (assign_label etacut=None -> label -1), fits a 2-component GMM in
# (circ, polar, e_b) space, and locates the circularity where the two
# components' circularity histograms cross -- that crossing is the thin/thick
# boundary eta_cut. smooth_curve is its Gaussian-smoothing helper.
# Extra dependencies: scipy.ndimage, sklearn.mixture.GaussianMixture, intersect.
# ----------------------------------------------------------------------
def smooth_curve(x, y, sigma=1):
    """Gaussian-smooth a curve (x, y); sigma sets the smoothing level."""
    x_sm = np.array(x)
    y_sm = np.array(y)
 
    x_g1d = ndimage.gaussian_filter1d(x_sm, sigma)
 
    if len(y_sm.shape) > 1:
        y_g1d = []
        for i in y_sm:
            y_g1d.append(ndimage.gaussian_filter1d(i, sigma))
        y_g1d = np.array(y_g1d)
    else:
        y_g1d = ndimage.gaussian_filter1d(y_sm, sigma)
 
    return x_g1d, y_g1d
 
 
def get_etacut(jzojc, jpojc, eb, mass, Ecut, smoothing=True, sigma=1):
    """
    Obtain the circularity threshold eta_cut (Liang et al. 2024).
 
    Feed circularity, polarity and binding energy of the disky stars to a
    2-component GMM. The component with the smaller circularity peak is the
    thick disc, the other the thin disc; eta_cut is where their circularity
    histograms intersect.
 
    Returns
    ----------
    eta_cut : float
        circularity threshold (0.0 on failure / under-resolved components)
    """
    # Separate disky stars based on Zana's method
    temp_label = assign_label(eb, jzojc, mass, Ecut, etacut=None)
    # Disky stars' label is assigned as -1
    non_sph = np.where(temp_label == -1)
 
    # GMM setting
    # n_init=10 adapted from Du et al. 2019; n_components=2 to separate discs
    aclus = GMM(n_components=2, covariance_type='full', n_init=10)
    data = (np.array([jzojc, jpojc, eb]).T)[non_sph]
    aclus.fit(data)
    GMMlabel = aclus.predict(data)
 
    # GMM label is random; bin circularity for each gaussian component
    min_val = 0
    max_val = 1.
    n_bins = int(np.sqrt(len(data.T[0])))  # or use another method to set bin count
    bins = np.linspace(min_val, max_val, n_bins)
 
    hist1 = np.histogram(data.T[0][np.where(GMMlabel == 0)], bins=bins)
    hist2 = np.histogram(data.T[0][np.where(GMMlabel == 1)], bins=bins)
 
    if len(np.where(GMMlabel == 0)[0]) <= 30 or len(np.where(GMMlabel == 1)[0]) <= 30:
        eta_cut = 0.
        print('Etacut = ', eta_cut)
        return eta_cut
 
    # Bin centre and density at this bin
    x1 = (hist1[1][1:] + hist1[1][:-1]) / 2
    y1 = hist1[0]
    x2 = (hist2[1][1:] + hist2[1][:-1]) / 2
    y2 = hist2[0]
 
    x1 = x1[y1 > 0]
    y1 = y1[y1 > 0]
    x2 = x2[y2 > 0]
    y2 = y2[y2 > 0]
 
    # Maximum density and corresponding bin for the two gaussian components
    tempy1 = y1[np.where((y1 != y1[0]) & (y1 != y1[-1]))]
    tempy2 = y2[np.where((y2 != y2[0]) & (y2 != y2[-1]))]
    hmax1 = x1[np.where(y1 == tempy1.max())]
    hmax2 = x2[np.where(y2 == tempy2.max())]
 
    # Safety check: avoid multiple maximum values, which is rare
    if len(hmax1) > 1:
        hmax1 = np.mean(hmax1)
    if len(hmax2) > 1:
        hmax2 = np.mean(hmax2)
 
    # Check which corresponding circularity is larger; if component 0 has the
    # larger peak, swap labels and re-bin
    circ_m1 = min(hmax1, hmax2)
    circ_m2 = max(hmax1, hmax2)
 
    if not isinstance(circ_m1, float):
        circ_m1 = circ_m1[0]
    if not isinstance(circ_m2, float):
        circ_m2 = circ_m2[0]
 
    if hmax2 == circ_m2:
        pass
    else:
        hist1 = np.histogram(data.T[0][np.where(GMMlabel == 1)], bins=bins)
        hist2 = np.histogram(data.T[0][np.where(GMMlabel == 0)], bins=bins)
 
        x1 = (hist1[1][1:] + hist1[1][:-1]) / 2
        y1 = hist1[0]
        x2 = (hist2[1][1:] + hist2[1][:-1]) / 2
        y2 = hist2[0]
 
        x1 = x1[y1 > 0]
        y1 = y1[y1 > 0]
        x2 = x2[y2 > 0]
        y2 = y2[y2 > 0]
 
    # Smooth curves and find the threshold
    if smoothing == True:
        x1, y1 = smooth_curve(x1, y1, sigma=sigma)
        x2, y2 = smooth_curve(x2, y2, sigma=sigma)
 
    x, y = intersection(x1, y1, x2, y2)[:2]  # old intersect returns (x,y); new returns (x,y,t1,t2)
 
    # If there are multiple thresholds or none, use criteria to reduce them;
    # some cases yield no threshold and return 0
    if len(x) == 0:
        min1 = np.min(x1)
        min2 = np.min(x2)
        max1 = np.max(x1)
        max2 = np.max(x2)
 
        if (max1 <= circ_m2 and min2 >= circ_m1):
            eta_cut = ((min2 + max1) / 2)
        else:
            eta_cut = 0.
            print("strange intersect 0")
    elif len(x) > 1:
        x_mask = np.where((x <= circ_m2) & (x >= circ_m1))
        if len(x[x_mask]) == 1:
            eta_cut = (x[x_mask][0])
        elif len(x[x_mask]) > 1:
            eta_cut = (np.mean(x[x_mask]))
            print("more than 1 intersect")
        elif len(x[x_mask]) == 0:
            eta_cut = 0.
            print("strange intersect 1")
    else:
        x_mask = np.where((x <= circ_m2) & (x >= circ_m1))
        if len(x[x_mask]) == 0:
            eta_cut = 0.
            print("strange intersect 1")
        else:
            eta_cut = x[0]
 
    print('Etacut = ', eta_cut)
 
    return eta_cut


eps_plummer = EPS_PLUMMER_PARTICLE_KPC if _family == 'particle' else H_SUPPORT
ecut_sel =  ((np.abs(circ) <= 1.5) & (polar <= 1.5) & (eb_star <= 0))
Ecut = get_Ecut(eb_star[ecut_sel], mass_gal[ecut_sel])


# --- (9) circularity threshold  eta_cut  via get_etacut (Liang+2024) ------
# get_etacut re-derives the disky stars internally (assign_label etacut=None),
# fits a 2-component GMM in (circ, polar, e_b), and returns the thin/thick
# circularity boundary. It can raise for under-resolved / degenerate GMM
# fits, and the whole point of this script is unattended batch runs, so the
# call is guarded: on failure (or eta_cut==0) the decomposition falls back to
# the spheroid/disc split (disc stays -1). Remove the try/except to mirror
# decomposition.py exactly. Set ETACUT_FIXED to a float (e.g. 0.7) to skip
# get_etacut and use a fixed threshold instead.
ETACUT_FIXED = None
if ETACUT_FIXED is not None:
    etacut = float(ETACUT_FIXED)
else:
    try:
        etacut = get_etacut(circ[ecut_sel], polar[ecut_sel],
                            eb_star[ecut_sel], mass_gal[ecut_sel], Ecut)
    except Exception as _e:
        print('get_etacut failed (%s); falling back to etacut=None' % _e)
        etacut = None


labels_sel = assign_label(eb_star[ecut_sel], circ[ecut_sel],
                          mass_gal[ecut_sel], Ecut, etacut=etacut)
label = np.full(len(ID_gal), -99.0)              # -99 = not in decomposition sample
label[ecut_sel] = labels_sel


# --- save -----------------------------------------------------------------
output = {}
output['jc']    = jcE[ecut_sel]       # j_c(E) [km*kpc/s]; inf => epsilon->0
output['jz']    = jz_star[ecut_sel]  # j_z   [km*kpc/s], signed
output['jp']    = jp_star[ecut_sel]   # j_p   [km*kpc/s], in-plane magnitude
output['circ']  = circ[ecut_sel]      # jz/jc(E)  (orbital circularity)
output['polar'] = polar[ecut_sel]     # jp/jc(E)  (polarity)
output['ID']    = ID_gal[ecut_sel]
output['mass']    = mass_gal[ecut_sel]
output['pos']    = pos_gal[ecut_sel]
output['eb']    = eb_star[ecut_sel]
output['E']     = E_star    # [kpc^2/s^2]
output['dist']     = radius_gal # [kpc]
output['label'] = labels_sel
output['etacut'] = etacut
output['Ecut'] = Ecut

print(codetp)
print('stars: %d   forced-spheroid(inf): %d' % (len(jcE), int(np.sum(~np.isfinite(jcE)))))
_disk = np.isfinite(circ)
print('circularity: median=%.3f   thin-disk frac(circ>0.7)=%.3f'
      % (np.median(circ[_disk]),
         np.sum(_disk & (circ > 0.7)) / float(len(circ))))
np.save('/work/hdd/bezm/tnguyen2/AGORA/analysis/circularity_ProgBranch-%s_Idx_%s_%s_ver2013_Abadi2003method.npy' % (codetp, prog_branch, idx), output)