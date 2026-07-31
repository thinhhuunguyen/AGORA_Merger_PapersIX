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
from setup import add_metallicity_fields, add_radialdist_to_halocenter_field


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

def calc_gas_Z(idx, codetp, pfs, hullv):
    #
    ds = load_ds(codetp, idx, pfs)
    add_metallicity_fields(ds, codetp)
    #
    metadata_dir = '/work/hdd/bezm/tnguyen2/AGORA/%s/metadata/' % codetp
    
    allstars = np.load(metadata_dir + '/' + 'star_metadata_allbox_%s.npy' % idx, allow_pickle=True).tolist()
    allpos = allstars['pos']
    allvel = allstars['vel']
    allmass = allstars['mass']
    allID = allstars['ID'].astype(int)
    
    bound_ID = assignment['ids']['0'][idx]
    bound_pos = allpos[np.intersect1d(bound_ID, allID, return_indices=True)[2]]
    bound_vel = allvel[np.intersect1d(bound_ID, allID, return_indices=True)[2]]
    bound_mass = allmass[np.intersect1d(bound_ID, allID, return_indices=True)[2]]
    
    galactic_center = determine_center(bound_pos, bound_vel, bound_mass)
    #
    add_radialdist_to_halocenter_field(ds, galactic_center)
    #
    # Gas mass
    if codetp == 'ART' or codetp == 'ENZO' or codetp == 'RAMSES':
        if idx in dist_data['idx']:
            prog_center = np.array(dist_data['prog_com_plot'])[np.array(dist_data['idx']) == idx][0]
        else:
            hullv_pos_prog = (hullv[idx][prog_branch]*ds.units.m).to('code_length').v
            prog_center = np.mean(ConvexHull(hullv_pos_prog).points, axis=0)
        prog_reg = ds.sphere(prog_center, (4*dist_data['prog_radius'][-1], 'kpc'))
        gas_x = prog_reg[(gas_name_dict[codetp],"x")].to('m')
        gas_y = prog_reg[(gas_name_dict[codetp],"y")].to('m')
        gas_z = prog_reg[(gas_name_dict[codetp],"z")].to('m')
        prog_gascoor = np.vstack((gas_x.v, gas_y.v, gas_z.v)).T
        prog_gasmass = prog_reg[(gas_name_dict[codetp],gas_mass_dict[codetp])].to('Msun').v
        prog_gasmassZ = prog_reg[(gas_name_dict[codetp],'metal_mass')].to('Msun').v
        prog_gasZ = prog_reg['gas','agora_metallicity']
        
        prog_bool = in_hull(prog_gascoor, hullv[idx][prog_branch])
        if np.sum(prog_bool)/len(prog_bool) > 0.75:
            print('Need to select a bigger region to conver the convex hull', np.sum(prog_bool)/len(prog_bool), idx)
        
        #Gas Metallicity within the convex hull
        gas_Z_hull = np.average(prog_gasZ[prog_bool], weights=prog_gasmass[prog_bool])
        gas_massZ_hull = np.sum(prog_gasmassZ[prog_bool])

        #Gas Metallicity within 0.2 Rvir of the halo (recentered)
        prog_gascoor_codelength = np.vstack((gas_x.to('code_length').v, gas_y.to('code_length').v, gas_z.to('code_length').v)).T
        gal_bool = np.linalg.norm(prog_gascoor_codelength - galactic_center, axis=1) < 0.2*rawtree_cdenrad['0'][idx]['cden_rad'] 
        gas_Z_galaxy = np.average(prog_gasZ[gal_bool], weights=prog_gasmass[gal_bool])
        gas_massZ_galaxy = np.sum(prog_gasmassZ[gal_bool])

        del prog_gascoor, prog_gasmass, prog_bool
        
    #-------------------------------------------------------------------------------------
            
    else: #for SPH and hybrid codes
        reg = ds.all_data()
        gascoor = reg[gas_name_dict[codetp], 'particle_position'].to('m')
        gasmass = reg[gas_name_dict[codetp], 'particle_mass'].to('Msun').v
        gasZ = reg['gas', 'agora_metallicity']
        if codetp != 'GEAR':
            gasmassZ = reg['gas', 'metal_mass'].to('Msun').v
        else:
            gasmassZ = gasZ*0.02041*gasmass #0.02041 is the agora_Zsun
        
        prog_bool = in_hull(gascoor.v, hullv[idx][prog_branch])
        
        #Gas Metallicity within the convex hull
        gas_Z_hull = np.average(gasZ[prog_bool], weights=gasmass[prog_bool])
        gas_massZ_hull = np.sum(gasmassZ[prog_bool])
        #Gas Metallicity within 0.2 Rvir of the halo (recentered)
        gal_bool = np.linalg.norm(gascoor.to('code_length').v - galactic_center, axis=1) < 0.2*rawtree_cdenrad['0'][idx]['cden_rad'] 
        gas_Z_galaxy = np.average(gasZ[gal_bool], weights=gasmass[gal_bool])
        gas_massZ_galaxy = np.sum(gasmassZ[gal_bool])

        
    return gas_Z_hull, gas_Z_galaxy, gas_massZ_hull, gas_massZ_galaxy
    
#---------------------------------------------------------------------------------------------------------------------------------
halotree_ver = 2013
merger_number = '0'
    
redshift_list, time_list, pfs, step = load_halotree_and_pfs(codetp, halotree_ver, rawtree_skip = True)
dist_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/dist_relvel_j_ProgBranch-%s_FirstMerger_%s_ver2013.npy' % (merger_number, codetp), allow_pickle=True).tolist()
prog_branch, sec_branch, sec_branch_2 = sec_branch_compute(codetp, merger_number)
idx_begin, idx_endinfall, idx_1stpass, idx_maxdist, idx_cls, time_begin, time_endinfall, time_1stpass, time_maxdist, time_cls = load_timings(codetp, halotree_ver, merger_number)
rawtree_cdenrad = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/halotree_%s_%s_Branch0_cdenrad.npy' % (codetp, halotree_ver), allow_pickle=True).tolist()

metadata_dir = '/work/hdd/bezm/tnguyen2/AGORA/%s/metadata/' % codetp
hullv = np.load('/work/hdd/bezm/gtg115x/Halo_Finding/%s/hullv_%s_withPos_final.npy' % (codetp, halotree_ver), allow_pickle=True).tolist()
assignment = np.load('/work/hdd/bezm/gtg115x/Halo_Finding/%s/star_id_%s_final_firstmerger.npy' % (codetp, halotree_ver), allow_pickle=True).tolist()

if yt.is_root():
    print('Start for %s' % codetp)
output = {}
output['gas_Z_hull'] = []
output['gas_Z_galaxy'] = []
output['gas_massZ_hull'] = []
output['gas_massZ_galaxy'] = []
output['time'] = []
output['idx'] = []

if codetp == 'CHANGA':
    idx_loopstart = np.argmin(abs((time_list/1e3) - 0.5)) - 1
else:
    idx_loopstart = np.argmin(abs((time_list/1e3) - 0.5))


my_storage = {}
for sto, i in yt.parallel_objects(range(idx_loopstart, dist_data['idx'][0] + step, 1), nprocs, storage = my_storage):
    if prog_branch in hullv[i].keys():
        if os.path.exists('/work/hdd/bezm/tnguyen2/AGORA/analysis/GasZ_%s_checkpoint/GasZ_ProgBranch-%s_%s_ver%s_ConvexHull_preInfall_%s_ver3.npy' % (codetp, merger_number, codetp, halotree_ver, i)):
            continue
        #
        gas_Z_hull, gas_Z_galaxy, gas_massZ_hull, gas_massZ_galaxy = calc_gas_Z(i, codetp, pfs, hullv)
        if codetp == 'RAMSES' or codetp == 'ENZO' or codetp == 'ART':
            output_cp = {}
            output_cp['gas_Z_hull'] = gas_Z_hull
            output_cp['gas_Z_galaxy'] = gas_Z_galaxy
            output_cp['gas_massZ_hull'] = gas_massZ_hull
            output_cp['gas_massZ_galaxy'] = gas_massZ_galaxy
            output_cp['time'] = time_list[i]
            output_cp['idx'] = i
            np.save('/work/hdd/bezm/tnguyen2/AGORA/analysis/GasZ_%s_checkpoint/GasZ_ProgBranch-%s_%s_ver%s_ConvexHull_preInfall_%s_ver3.npy' % (codetp, merger_number, codetp, halotree_ver, i), output_cp)
        #
        sto.result = {}
        sto.result[0] = gas_Z_hull
        sto.result[1] = gas_Z_galaxy
        sto.result[2] = gas_massZ_hull
        sto.result[3] = gas_massZ_galaxy
        sto.result[4] = time_list[i]
        sto.result[5] = i
        print('Done for time step %s' % i)


for c, vals in sorted(my_storage.items()):
    if vals != None:
        output['gas_Z_hull'].append(vals[0])
        output['gas_Z_galaxy'].append(vals[1])
        output['gas_massZ_hull'].append(vals[2])
        output['gas_massZ_galaxy'].append(vals[3])
        output['time'].append(vals[4])
        output['idx'].append(vals[5])
    
if yt.is_root():
    print('Start saving %s' % codetp)
    np.save('/work/hdd/bezm/tnguyen2/AGORA/analysis/GasZ_ProgBranch-%s_%s_ver%s_ConvexHull_preInfall_ver2.npy' % (merger_number, codetp, halotree_ver), output)

