import yt
import numpy as np
from yt.data_objects.unions import ParticleUnion
from yt.data_objects.particle_filters import add_particle_filter
from scipy.spatial import Delaunay
import sys

import setup
from importlib import reload
reload(setup)

from setup import load_halotree_and_pfs, load_timings, load_ds, load_tracking_dist_data, sec_branch_compute
from setup import gas_name_dict, gas_temp_dict
from setup import add_metallicity_fields, add_cooling_fields, add_radialdist_to_halocenter_field, extract_and_order_snapshotIdx
from setup import get_haloradius, infall_timestep_compute_hullv
from setup import codetp_list, label_list, color_list, marker_list
from setup import codetp_list10, label_list10, color_list10, marker_list10
from setup import gas_name_dict, gas_mass_dict, star_name_dict, star_mass_dict, dm_name_dict


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


def extract_dm_and_bary(index, branch, codetp):
    if codetp == 'AREPO' or codetp == 'GADGET3' or codetp == 'GADGET4':
        ds = yt.load(pfs[index], unit_base = {"length": (1.0, "Mpccm/h")})
    else:
        ds = yt.load(pfs[index])
    #-------------------------
    if codetp == 'ENZO':
        def darkmatter_init(pfilter, data):
            filter_darkmatter0 = np.logical_or(data["all", "particle_type"] == 1, data["all", "particle_type"] == 4)
            filter_darkmatter = np.logical_and(filter_darkmatter0,data['all', 'particle_mass'].to('Msun') > 1)
            return filter_darkmatter
        add_particle_filter("DarkMatter",function=darkmatter_init,filtered_type='all',requires=["particle_type","particle_mass"])
        ds.add_particle_filter("DarkMatter")
        #
        def stars_init(pfilter, data):
            filter_stars = np.logical_and(data["all", "particle_type"] == 2, data["all", "particle_mass"].to('Msun') > 1)
            return filter_stars
        add_particle_filter("stars", function=stars_init, filtered_type="all", requires=["particle_type","particle_mass"])
        ds.add_particle_filter("stars")
    elif codetp == 'GEAR':
        dm = ParticleUnion("DarkMatter",["PartType5","PartType2"])
        ds.add_particle_union(dm)
    elif codetp == 'GADGET3' or codetp == 'GADGET4':
        dm = ParticleUnion("DarkMatter",["PartType5","PartType1"])
        ds.add_particle_union(dm)
    elif codetp == 'AREPO' or codetp == 'GIZMO':
        dm = ParticleUnion("DarkMatter",["PartType2","PartType1"])
        ds.add_particle_union(dm)

    # =================================================================
    # Convex hull for this snapshot/branch.
    # hullv[index][branch] is the (M, 3) array of hull-vertex positions
    # in METERS (same convention as your gas-mass-fraction script). The
    # in_hull test below is therefore done in meters as well.
    # =================================================================
    hull_pts_m = np.asarray(hullv[index][branch])   # (M, 3), meters

    # Region pre-selection that bounds the hull.
    if codetp == 'ART' or codetp == 'ENZO' or codetp == 'RAMSES':
        hull_pts_cl = (hull_pts_m * ds.units.m).to('code_length').v
        hull_center = np.mean(hull_pts_cl, axis=0)
        hull_radius = 1.2 * np.max(np.linalg.norm(hull_pts_cl - hull_center, axis=1))
        reg = ds.sphere(hull_center, (hull_radius, 'code_length'))
    else:
        reg = ds.all_data()

    #-----------------------------
    #Extract dark matter metadata (hull cut in meters)
    dm_pos_m = reg[(dm_name_dict[codetp], 'particle_position')].to('m').v
    dm_m_all = reg[(dm_name_dict[codetp], 'particle_mass')].to('Msun').v
    in_dm   = in_hull(dm_pos_m, hull_pts_m)
    dm_m    = dm_m_all[in_dm]
    dm_coor = (dm_pos_m[in_dm] * ds.units.m).to('kpc').v

    #-----------------------------
    #Extract gas+star metadata (combine, then one hull cut in meters)
    s_pos_m = reg[(star_name_dict[codetp], 'particle_position')].to('m').v
    s_m_all = reg[(star_name_dict[codetp], star_mass_dict[codetp])].to('Msun').v

    if codetp == 'ART' or codetp == 'ENZO' or codetp == 'RAMSES':
        gas_x = reg[("gas", "x")].to('m').v
        gas_y = reg[("gas", "y")].to('m').v
        gas_z = reg[("gas", "z")].to('m').v
        gas_pos_m = np.vstack((gas_x, gas_y, gas_z)).T
    else:
        gas_pos_m = reg[(gas_name_dict[codetp], 'particle_position')].to('m').v
    gas_m_all = reg[(gas_name_dict[codetp], gas_mass_dict[codetp])].to('Msun').v

    if len(s_m_all) == 0:  # if there are no star particles
        bary_pos_m = gas_pos_m
        bary_m_all = gas_m_all
    else:
        bary_pos_m = np.concatenate((s_pos_m, gas_pos_m), axis=0)
        bary_m_all = np.concatenate((s_m_all, gas_m_all), axis=0)

    in_bary   = in_hull(bary_pos_m, hull_pts_m)
    bary_m    = bary_m_all[in_bary]
    bary_coor = (bary_pos_m[in_bary] * ds.units.m).to('kpc').v

    # Coverage guard for the grid-code bounding sphere: if almost everything
    # in the pre-selected region is inside the hull, the sphere may be
    # clipping the hull and should be enlarged.
    if codetp == 'ART' or codetp == 'ENZO' or codetp == 'RAMSES':
        frac = np.sum(in_bary) / len(in_bary)
        if frac > 0.75:
            print('Bounding sphere may clip the convex hull (in-hull fraction %.2f) at idx %s' % (frac, index))

    #-----------------------------
    output = {'dm_mass':dm_m, 'dm_coor':dm_coor, 'bary_mass':bary_m, 'bary_coor':bary_coor}
    np.save('/work/hdd/bezm/tnguyen2/AGORA/%s/metadata/dm_bary_Branch_%s_Idx_%s_ver%s_ConvexHull.npy' % (codetp, branch, index, halotree_ver),output)

codetp = sys.argv[1]

merger_number = '0'
halotree_ver = 2013
redshift_list, time_list, pfs, step = load_halotree_and_pfs(codetp, halotree_ver, rawtree_skip=True)
hullv = np.load('/work/hdd/bezm/gtg115x/Halo_Finding/%s/hullv_%s_withPos_final.npy' % (codetp, halotree_ver), allow_pickle=True).tolist()
prog_branch, sec_branch, sec_branch_2 = sec_branch_compute(codetp, merger_number)
idx_begin, idx_endinfall, idx_1stpass, idx_maxdist, idx_cls, time_begin, time_endinfall, time_1stpass, time_maxdist, time_cls = load_timings(codetp, halotree_ver, merger_number)
time_eval = time_1stpass + 0.6
if codetp == 'GEAR':
    idx_eval = np.argmin(abs(time_eval - time_list/1e3)) - (np.argmin(abs(time_eval - time_list/1e3)) % step) - 1
else:
    idx_eval = np.argmin(abs(time_eval - time_list/1e3)) - (np.argmin(abs(time_eval - time_list/1e3)) % step)

#index = idx_begin - step
index = idx_eval + 2*step

# Only run if a non-empty hull exists for this branch/snapshot.
if (prog_branch in hullv[index]) and (len(hullv[index][prog_branch]) != 0):
    extract_dm_and_bary(index, prog_branch, codetp)
else:
    print('No hull vertices for branch %s at idx %s; skipping.' % (prog_branch, index))