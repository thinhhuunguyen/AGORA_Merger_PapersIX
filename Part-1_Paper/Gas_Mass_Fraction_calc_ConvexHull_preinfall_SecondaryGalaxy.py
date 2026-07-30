import yt
yt.set_log_level(0)
import numpy as np
import sys, os
from scipy.spatial import ConvexHull, Delaunay

import setup
from importlib import reload
reload(setup)

from setup import load_halotree_and_pfs, load_timings, load_ds, sec_branch_compute
from setup import gas_name_dict, gas_temp_dict, gas_mass_dict


yt.enable_parallelism()
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
nprocs = comm.size

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

def calc_gas_mass_frac(idx, codetp, pfs, hullv):
    #
    ds = load_ds(codetp, idx, pfs)
    # Gas mass
    if codetp == 'ART' or codetp == 'ENZO' or codetp == 'RAMSES':
        hullv_pos_sec = (hullv[idx][sec_branch]*ds.units.m).to('code_length').v
        sec_center = np.mean(ConvexHull(hullv_pos_sec).points, axis=0)
        sec_reg = ds.sphere(sec_center, (1.2*max(np.linalg.norm(ConvexHull(hullv_pos_sec).points - sec_center,axis=1)), 'code_length'))
        gas_x = sec_reg[(gas_name_dict[codetp],"x")].to('m').v
        gas_y = sec_reg[(gas_name_dict[codetp],"y")].to('m').v
        gas_z = sec_reg[(gas_name_dict[codetp],"z")].to('m').v
        sec_gascoor = np.vstack((gas_x, gas_y, gas_z)).T
        sec_gasmass = sec_reg[(gas_name_dict[codetp],gas_mass_dict[codetp])].to('Msun').v
        sec_gastemp = sec_reg[(gas_name_dict[codetp],gas_temp_dict[codetp])].v
        sec_gasnH = sec_reg['gas','number_density'].to('cm**-3').v*12/13
        
        sec_bool = in_hull(sec_gascoor, hullv[idx][sec_branch])
        if np.sum(sec_bool)/len(sec_bool) > 0.75:
            print('Need to select a bigger region to conver the convex hull', np.sum(sec_bool)/len(sec_bool), idx)

        sf_bool = sec_gasnH > 1
        cool_bool = sec_gastemp < 10**(4.5)
        
        gas_mass_total = sec_gasmass[sec_bool].sum()
        gas_mass_sf = sec_gasmass[sec_bool*sf_bool].sum()
        gas_mass_cool = sec_gasmass[sec_bool*cool_bool].sum()
        del sec_gascoor, sec_gasmass, sec_gastemp, sec_gasnH, sec_bool, sf_bool
        
        all_mass = sec_reg['all', 'particle_mass'].to('Msun').v
        all_pos = sec_reg['all', 'particle_position'].to('m').v
        all_bool = in_hull(all_pos, hullv[idx][sec_branch])
        del all_pos
        
        gas_mass_fraction = gas_mass_total/(all_mass[all_bool].sum() + gas_mass_total)
        gas_mass_sf_fraction = gas_mass_sf/(all_mass[all_bool].sum() + gas_mass_total)
        
    #-------------------------------------------------------------------------------------
            
    else: #for SPH and hybrid codes
        reg = ds.all_data()
        gascoor = reg[gas_name_dict[codetp], 'particle_position'].to('m').v
        gasmass = reg[gas_name_dict[codetp], 'particle_mass'].to('Msun').v
        gastemp = reg[gas_name_dict[codetp], gas_temp_dict[codetp]].v
        gasnH = reg['gas','number_density'].to('cm**-3').v*12/13
        
        sec_bool = in_hull(gascoor, hullv[idx][sec_branch])

        sf_bool = gasnH > 1
        cool_bool = gastemp < 10**(4.5)
        
        gas_mass_total = gasmass[sec_bool].sum()
        gas_mass_sf = gasmass[sec_bool*sf_bool].sum()
        gas_mass_cool = gasmass[sec_bool*cool_bool].sum()
        
        all_mass = reg['all', 'particle_mass'].to('Msun').v
        all_pos = reg['all', 'particle_position'].to('m').v
        all_bool = in_hull(all_pos, hullv[idx][sec_branch])
        
        gas_mass_fraction = gas_mass_total/all_mass[all_bool].sum() 
        gas_mass_sf_fraction = gas_mass_sf/all_mass[all_bool].sum()
        
    #return sec_gascoor, sec_gasdist, sec_gasmass, sec_gastemp, sec_gasden
    return gas_mass_total, gas_mass_sf, gas_mass_cool, gas_mass_fraction, gas_mass_sf_fraction
    
#---------------------------------------------------------------------------------------------------------------------------------
halotree_ver = 2013
merger_number = '0'
    
redshift_list, time_list, pfs, step = load_halotree_and_pfs(codetp, halotree_ver, rawtree_skip = True)
dist_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/dist_relvel_j_ProgBranch-%s_FirstMerger_%s_ver2013.npy' % (merger_number, codetp), allow_pickle=True).tolist()
prog_branch, sec_branch, sec_branch_2 = sec_branch_compute(codetp, merger_number)
idx_begin, idx_endinfall, idx_1stpass, idx_maxdist, idx_cls, time_begin, time_endinfall, time_1stpass, time_maxdist, time_cls = load_timings(codetp, halotree_ver, merger_number)

metadata_dir = '/work/hdd/bezm/tnguyen2/AGORA/%s/metadata/' % codetp
hullv = np.load('/work/hdd/bezm/gtg115x/Halo_Finding/%s/hullv_%s_withPos_final.npy' % (codetp, halotree_ver), allow_pickle=True).tolist()

if yt.is_root():
    print('Start for %s' % codetp)
output = {}
output['gas_mass_total'] = []
output['gas_mass_fraction'] = []
output['gas_mass_sf'] = []
output['gas_mass_sf_fraction'] = []
output['gas_mass_cool'] = []
output['time'] = []

if codetp == 'CHANGA':
    idx_loopstart = np.argmin(abs((time_list/1e3) - 0.5)) - 1
else:
    idx_loopstart = np.argmin(abs((time_list/1e3) - 0.5))


my_storage = {}
for sto, i in yt.parallel_objects(range(dist_data['idx'][0] - 4*step, dist_data['idx'][0] + step, step), nprocs, storage = my_storage):
    if sec_branch in hullv[i].keys():
        if len(hullv[i][sec_branch]) != 0:
            #
            gas_mass_total, gas_mass_sf, gas_mass_cool, gas_mass_fraction, gas_mass_sf_fraction = calc_gas_mass_frac(i, codetp, pfs, hullv)
            #
            sto.result = {}
            sto.result[0] = gas_mass_total
            sto.result[1] = gas_mass_fraction
            sto.result[2] = gas_mass_sf
            sto.result[3] = gas_mass_sf_fraction
            sto.result[4] = time_list[i]
            sto.result[5] = gas_mass_cool
            print('Done for time step %s' % i)


for c, vals in sorted(my_storage.items()):
    if vals != None:
        output['gas_mass_total'].append(vals[0])
        output['gas_mass_fraction'].append(vals[1])
        output['gas_mass_sf'].append(vals[2])
        output['gas_mass_sf_fraction'].append(vals[3])
        output['gas_mass_cool'].append(vals[5])
        output['time'].append(vals[4])
    
if yt.is_root():
    print('Start saving %s' % codetp)
    np.save('/work/hdd/bezm/tnguyen2/AGORA/analysis/GasMassFraction_SecBranch-%s_%s_ver%s_ConvexHull_preInfall_ver3.npy' % (merger_number, codetp, halotree_ver), output)

