import yt
yt.set_log_level(0)
import numpy as np
import sys, os
from scipy.spatial import ConvexHull, Delaunay

import setup
from importlib import reload
reload(setup)

from setup import load_halotree_and_pfs, load_timings, load_ds, sec_branch_compute
from setup import gas_name_dict, gas_mass_dict
from setup import add_metallicity_fields, add_cooling_fields, add_radialdist_to_halocenter_field, add_radialvel_to_halocenter_field


yt.enable_parallelism()
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
nprocs = comm.size

def remove_outliers(arr, threshold=1.5):
    q3 = np.percentile(arr, 75)
    q1 = np.percentile(arr, 25)
    iqr = np.percentile(arr, 75) - np.percentile(arr, 25)
    return (arr > q1 - threshold*iqr)*(arr < q3 + threshold*iqr)

def remove_outliers_3d(pos, vel, mass, threshold=1.5):
    pos_bool = remove_outliers(pos[:,0])*remove_outliers(pos[:,1])*remove_outliers(pos[:,2])
    vel_bool = remove_outliers(vel[:,0])*remove_outliers(vel[:,1])*remove_outliers(vel[:,2])
    return pos[pos_bool*vel_bool], vel[pos_bool*vel_bool], mass[pos_bool*vel_bool]

def determine_center(pos, vel, mass, threshold=1.5):
    pos_f, vel_f, mass_f = remove_outliers_3d(pos, vel, mass, threshold)
    center = np.average(pos_f, weights=mass_f, axis=0)
    return center

codetp = sys.argv[1]

def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    #
    return hull.find_simplex(p)>=0

def calc_gas_coolingTime(idx, codetp, pfs, hullv):
    #
    ds = load_ds(codetp, idx, pfs)
    add_metallicity_fields(ds, codetp)
    #
    metadata_dir = '/work/hdd/bezm/tnguyen2/AGORA/%s/metadata/' % codetp
    #
    allstars = np.load(metadata_dir + '/' + 'star_metadata_allbox_%s.npy' % idx, allow_pickle=True).tolist()
    allpos = allstars['pos']
    allvel = allstars['vel']
    allmass = allstars['mass']
    allID = allstars['ID'].astype(int)
    #
    if codetp == 'CHANGA' and (idx == 178 or idx == 180):
        bound_ID = assignment_unique['0'][idx]
    elif codetp == 'RAMSES' and (idx in range(105, 120)):
        bound_ID = assignment_unique['0'][idx]
    elif codetp == 'ART' and (idx in range(110, 124)):
        bound_ID = assignment_unique['0'][idx]
    else:
        bound_ID = assignment['ids']['0'][idx]
    bound_pos = allpos[np.intersect1d(bound_ID, allID, return_indices=True)[2]]
    bound_vel = allvel[np.intersect1d(bound_ID, allID, return_indices=True)[2]]
    bound_mass = allmass[np.intersect1d(bound_ID, allID, return_indices=True)[2]]
    #
    galactic_center = determine_center(bound_pos, bound_vel, bound_mass)
    galactic_velcom = (rawtree_velcom[prog_branch][idx]['Vel_Com']*ds.units.code_length/ds.units.s).to('km/s').v
    #
    add_radialdist_to_halocenter_field(ds, galactic_center)
    add_radialvel_to_halocenter_field(ds, galactic_center, galactic_velcom)
    add_cooling_fields(ds, codetp, idx, redshift_list, galactic_center)
    #
    # Gas mass
    if codetp == 'ART' or codetp == 'ENZO' or codetp == 'RAMSES':
        hullv_pos_prog = (hullv[idx][prog_branch]*ds.units.m).to('code_length').v
        prog_center = np.mean(ConvexHull(hullv_pos_prog).points, axis=0)
        prog_reg = ds.sphere(prog_center, (1.1*max(np.linalg.norm(ConvexHull(hullv_pos_prog).points - prog_center,axis=1)), 'code_length')) #choosing a sphere enclosing the convex hull
        gas_x = prog_reg[(gas_name_dict[codetp],"x")].to('m')
        gas_y = prog_reg[(gas_name_dict[codetp],"y")].to('m')
        gas_z = prog_reg[(gas_name_dict[codetp],"z")].to('m')
        prog_gascoor = np.vstack((gas_x.v, gas_y.v, gas_z.v)).T
        
        prog_bool = in_hull(prog_gascoor, hullv[idx][prog_branch])
            
        prog_gasmass = prog_reg[(gas_name_dict[codetp],gas_mass_dict[codetp])].to('Msun').v
        #Calculating the cooling time 
        prog_tcool = prog_reg['gas', "tCool"].to('Gyr').v
        prog_tff = prog_reg['gas', "tFF"].to('Gyr').v
        #Calculating distance to the center
        prog_dist = np.linalg.norm(np.vstack((gas_x.to('code_length').v, gas_y.to('code_length').v, gas_z.to('code_length').v)).T - galactic_center, axis=1)
        prog_dist = (prog_dist*ds.units.code_length).to('kpc').v
        #
        prog_radialvel = prog_reg['gas', "radial_vel"].to('km/s').v
        return prog_tcool[prog_bool], prog_tff[prog_bool], prog_gasmass[prog_bool], prog_dist[prog_bool], prog_radialvel[prog_bool]
    #-------------------------------------------------------------------------------------
    else: #for SPH and hybrid codes
        reg = ds.all_data()
        gascoor = reg[gas_name_dict[codetp], 'particle_position'].to('m')
        gasmass = reg[gas_name_dict[codetp], 'particle_mass'].to('Msun').v
        
        prog_bool = in_hull(gascoor.v, hullv[idx][prog_branch])
        gasdist = np.linalg.norm(gascoor.to('code_length').v - galactic_center, axis=1)
        gasdist = (gasdist*ds.units.code_length).to('kpc').v
        gasradialvel = reg['gas','radial_vel'].to('km/s').v
        #Calculating the cooling time 
        tcool = reg['gas', "tCool"].to('Gyr').v
        if codetp != 'CHANGA':
            tff = reg['gas', "tFF"].to('Gyr').v
        else:
            radData = reg['gas', "radii"].in_units("kpc").value #*
            all_massData = reg['all', 'particle_mass'].in_units("Msun").value
            all_posData = reg['all', 'particle_position'].in_units("code_length").value
            all_radData = np.linalg.norm(all_posData - galactic_center, axis=1)
            all_radData = (all_radData*ds.units.code_length).to('kpc').v
            gravConst = 4.51710305e-30 #in pc^3/(Msun*s^2)
            # Sort all particles by radius once
            sort_idx     = np.argsort(all_radData)
            sorted_rad   = all_radData[sort_idx]
            sorted_mass  = all_massData[sort_idx]
            cumul_mass   = np.cumsum(sorted_mass)
            # For each gas cell, find the enclosed mass using searchsorted (exact, no bins)
            idx        = np.searchsorted(sorted_rad, radData)
            gravM      = cumul_mass[idx]
            # Mean density inside sphere of radius r (convert kpc -> pc: *1000)
            r_pc       = radData * 1000.0          # kpc → pc
            densM      = gravM / ((4/3) * np.pi * r_pc**3)
            # Free-fall time
            tff       = np.sqrt((3.0 * np.pi) / (32.0 * gravConst * densM))
            tff = tff/3.15576e16 #converting from seconds to Gyr
        
        return tcool[prog_bool], tff[prog_bool], gasmass[prog_bool], gasdist[prog_bool], gasradialvel[prog_bool]
    
#---------------------------------------------------------------------------------------------------------------------------------
halotree_ver = 2013
merger_number = '0'
    
redshift_list, time_list, pfs, step = load_halotree_and_pfs(codetp, halotree_ver, rawtree_skip = True)
dist_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/dist_relvel_j_ProgBranch-%s_FirstMerger_%s_ver2013.npy' % (merger_number, codetp), allow_pickle=True).tolist()
prog_branch, sec_branch, sec_branch_2 = sec_branch_compute(codetp, merger_number)
idx_begin, idx_endinfall, idx_1stpass, idx_maxdist, idx_cls, time_begin, time_endinfall, time_1stpass, time_maxdist, time_cls = load_timings(codetp, halotree_ver, merger_number)
rawtree_cdenrad = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/halotree_%s_%s_Branch0_cdenrad.npy' % (codetp, halotree_ver), allow_pickle=True).tolist()
rawtree_velcom = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/halotree_%s_%s_Branch0_velcom.npy' % (codetp, halotree_ver), allow_pickle=True).tolist()

metadata_dir = '/work/hdd/bezm/tnguyen2/AGORA/%s/metadata/' % codetp
hullv = np.load('/work/hdd/bezm/gtg115x/Halo_Finding/%s/hullv_%s_withPos_final.npy' % (codetp, halotree_ver), allow_pickle=True).tolist()
assignment = np.load('/work/hdd/bezm/gtg115x/Halo_Finding/%s/star_id_%s_final_firstmerger.npy' % (codetp, halotree_ver), allow_pickle=True).tolist()

if codetp == 'CHANGA' or codetp == 'ART' or codetp == 'RAMSES': #these codes have some of the snapshots whose the galaxy center is not computed correctly using assignment 
    assignment_unique = np.load('/work/hdd/bezm/gtg115x/Halo_Finding/%s/star_id_%s_final_unique.npy' % (codetp, halotree_ver), allow_pickle=True).tolist()


if yt.is_root():
    print('Start for %s' % codetp)

output = {}
output['gas_tcool_hull'] = []
output['gas_tff_hull'] = []
output['gas_mass_hull'] = []
output['gas_dist_hull'] = []
output['gas_radialvel_hull'] = []
output['time'] = []
output['idx'] = []


if codetp == 'CHANGA':
    idx_loopstart = np.argmin(abs((time_list/1e3) - 0.5)) - 1
else:
    idx_loopstart = np.argmin(abs((time_list/1e3) - 0.5))


my_storage = {}
for sto, i in yt.parallel_objects(range(idx_loopstart, dist_data['idx'][0] + step, step), nprocs, storage = my_storage):
    if prog_branch in hullv[i].keys():
        if os.path.exists('/work/hdd/bezm/tnguyen2/AGORA/analysis/GasCoolingTime_checkpoint/GasCoolingTime_%s_checkpoint/GasCoolingTime_ProgBranch-%s_%s_ver%s_ConvexHull_preInfall_%s_ver2.npy' % (codetp, merger_number, codetp, halotree_ver, i)):
            continue
        #
        gas_tcool_hull, gas_tff_hull, gas_mass_hull, gas_dist_hull, gas_radialvel_hull = calc_gas_coolingTime(i, codetp, pfs, hullv)
        if codetp == 'RAMSES' or codetp == 'ENZO' or codetp == 'ART' or codetp == 'AREPO' or codetp == 'CHANGA':
            output_cp = {}
            output_cp['gas_tcool_hull'] = gas_tcool_hull
            output_cp['gas_tff_hull'] = gas_tff_hull
            output_cp['gas_mass_hull'] = gas_mass_hull
            output_cp['gas_dist_hull'] = gas_dist_hull
            output_cp['gas_radialvel_hull'] = gas_radialvel_hull
            output_cp['time'] = time_list[i]
            output_cp['idx'] = i
            np.save('/work/hdd/bezm/tnguyen2/AGORA/analysis/GasCoolingTime_checkpoint/GasCoolingTime_%s_checkpoint/GasCoolingTime_ProgBranch-%s_%s_ver%s_ConvexHull_preInfall_%s_ver2.npy' % (codetp, merger_number, codetp, halotree_ver, i), output_cp)
        #
        sto.result = {}
        sto.result[0] = gas_tcool_hull
        sto.result[1] = gas_tff_hull
        sto.result[2] = gas_mass_hull
        sto.result[3] = gas_dist_hull
        sto.result[6] = gas_radialvel_hull
        sto.result[4] = time_list[i]
        sto.result[5] = i
        print('Done for time step %s' % i)


for c, vals in sorted(my_storage.items()):
    if vals != None:
        output['gas_tcool_hull'].append(vals[0])
        output['gas_tff_hull'].append(vals[1])
        output['gas_mass_hull'].append(vals[2])
        output['gas_dist_hull'].append(vals[3])
        output['gas_radialvel_hull'].append(vals[6])
        output['time'].append(vals[4])
        output['idx'].append(vals[5])

    
if yt.is_root():
    print('Start saving %s' % codetp)
    np.save('/work/hdd/bezm/tnguyen2/AGORA/analysis/GasCoolingTime_ProgBranch-%s_%s_ver%s_ConvexHull_preInfall_ver2.npy' % (merger_number, codetp, halotree_ver), output)
