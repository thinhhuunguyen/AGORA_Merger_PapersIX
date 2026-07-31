import yt
yt.set_log_level(0)
import numpy as np
from scipy.spatial import ConvexHull, Delaunay

import setup
from importlib import reload
reload(setup)

from setup import load_halotree_and_pfs, load_timings, load_ds, sec_branch_compute
from setup import gas_name_dict, gas_mass_dict
from setup import add_metallicity_fields
from setup import codetp_list

yt.enable_parallelism()
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
nprocs = comm.size

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
    if codetp == 'GEAR':
        add_metallicity_fields(ds, codetp)
    if codetp == 'ART' or codetp == 'ENZO' or codetp == 'RAMSES':
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Primary Galaxy %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        hullv_pos_prog = (hullv[idx][prog_branch]*ds.units.m).to('code_length').v
        prog_center = np.mean(ConvexHull(hullv_pos_prog).points, axis=0)
        prog_reg = ds.sphere(prog_center, (1.2*max(np.linalg.norm(ConvexHull(hullv_pos_prog).points - prog_center,axis=1)), 'code_length'))
        gas_x = prog_reg[(gas_name_dict[codetp],"x")].to('m').v
        gas_y = prog_reg[(gas_name_dict[codetp],"y")].to('m').v
        gas_z = prog_reg[(gas_name_dict[codetp],"z")].to('m').v
        prog_gascoor = np.vstack((gas_x, gas_y, gas_z)).T
        prog_gasmass = prog_reg[(gas_name_dict[codetp],gas_mass_dict[codetp])].to('Msun').v
        prog_gasmassZ = prog_reg[(gas_name_dict[codetp],'metal_mass')].to('Msun').v
        prog_gasnH = prog_reg['gas','number_density'].to('cm**-3').v*12/13
        #prog_gastemp = prog_reg[(gas_name_dict[codetp],gas_temp_dict[codetp])].v
        
        prog_bool = in_hull(prog_gascoor, hullv[idx][prog_branch])
        #sf_bool = (prog_gastemp < 10**(4))*(prog_gasnH > 1) 
        prog_sf_bool = prog_gasnH > 1 
        
        prog_gas_mass_total = prog_gasmass[prog_bool].sum()
        prog_gas_massZ = prog_gasmassZ[prog_bool].sum()
        prog_gas_mass_sf = prog_gasmass[prog_bool*prog_sf_bool].sum()
        del prog_gascoor, prog_gasmass, prog_gasnH, prog_bool, prog_sf_bool
        
        prog_all_mass = prog_reg['all', 'particle_mass'].to('Msun').v
        prog_all_pos = prog_reg['all', 'particle_position'].to('m').v
        prog_all_bool = in_hull(prog_all_pos, hullv[idx][prog_branch])
        del prog_all_pos
        
        prog_gas_mass_fraction = prog_gas_mass_total/(prog_all_mass[prog_all_bool].sum() + prog_gas_mass_total)
        #prog_gas_mass_sf_fraction = prog_gas_mass_sf/(prog_all_mass[prog_all_bool].sum() + prog_gas_mass_total)
        
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Secondary Galaxy %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        hullv_pos_sec = (hullv[idx][sec_branch]*ds.units.m).to('code_length').v
        sec_center = np.mean(ConvexHull(hullv_pos_sec).points, axis=0)
        sec_reg = ds.sphere(sec_center, (1.2*max(np.linalg.norm(ConvexHull(hullv_pos_sec).points - sec_center,axis=1)), 'code_length'))
        gas_x = sec_reg[(gas_name_dict[codetp],"x")].to('m').v
        gas_y = sec_reg[(gas_name_dict[codetp],"y")].to('m').v
        gas_z = sec_reg[(gas_name_dict[codetp],"z")].to('m').v
        sec_gascoor = np.vstack((gas_x, gas_y, gas_z)).T
        sec_gasmass = sec_reg[(gas_name_dict[codetp],gas_mass_dict[codetp])].to('Msun').v
        sec_gasmassZ = sec_reg[(gas_name_dict[codetp],'metal_mass')].to('Msun').v
        sec_gasnH = sec_reg['gas','number_density'].to('cm**-3').v*12/13
        #sec_gastemp = sec_reg[(gas_name_dict[codetp],gas_temp_dict[codetp])].v
        
        sec_bool = in_hull(sec_gascoor, hullv[idx][sec_branch])
        #sf_bool = (sec_gastemp < 10**(4))*(sec_gasnH > 1) 
        sec_sf_bool = sec_gasnH > 1 
        
        sec_gas_mass_total = sec_gasmass[sec_bool].sum()
        sec_gas_massZ = sec_gasmassZ[sec_bool].sum()
        sec_gas_mass_sf = sec_gasmass[sec_bool*sec_sf_bool].sum()
        del sec_gascoor, sec_gasmass, sec_gasnH, sec_bool, sec_sf_bool
        
        sec_all_mass = sec_reg['all', 'particle_mass'].to('Msun').v
        sec_all_pos = sec_reg['all', 'particle_position'].to('m').v
        sec_all_bool = in_hull(sec_all_pos, hullv[idx][sec_branch])
        del sec_all_pos
        
        sec_gas_mass_fraction = sec_gas_mass_total/(sec_all_mass[sec_all_bool].sum() + sec_gas_mass_total)
        #sec_gas_mass_sf_fraction = sec_gas_mass_sf/(sec_all_mass[sec_all_bool].sum() + sec_gas_mass_total)
    #------------------------------------------------------------------------------------- 
    else: #for SPH and hybrid codes
        reg = ds.all_data()
        gascoor = reg[gas_name_dict[codetp], 'particle_position'].to('m').v
        gasmass = reg[gas_name_dict[codetp], 'particle_mass'].to('Msun').v
        #gastemp = reg[gas_name_dict[codetp], gas_temp_dict[codetp]].v
        gasnH = reg['gas','number_density'].to('cm**-3').v*12/13

        if codetp != 'GEAR':
            gasmassZ = reg['gas', 'metal_mass'].to('Msun').v
        else:
            gasZ = reg['gas', 'agora_metallicity']
            gasmassZ = gasZ*0.02041*gasmass #0.02041 is the agora_Zsun
        
        prog_bool = in_hull(gascoor, hullv[idx][prog_branch])
        sec_bool = in_hull(gascoor, hullv[idx][sec_branch])
        #sf_bool = (gastemp < 10**(4))*(gasnH > 1) 
        sf_bool = gasnH > 1 
        
        prog_gas_mass_total = gasmass[prog_bool].sum()
        prog_gas_massZ = gasmassZ[prog_bool].sum()
        prog_gas_mass_sf = gasmass[prog_bool*sf_bool].sum()
        sec_gas_mass_total = gasmass[sec_bool].sum()
        sec_gas_massZ = gasmassZ[sec_bool].sum()
        sec_gas_mass_sf = gasmass[sec_bool*sf_bool].sum()
        
        all_mass = reg['all', 'particle_mass'].to('Msun').v
        all_pos = reg['all', 'particle_position'].to('m').v
        prog_all_bool = in_hull(all_pos, hullv[idx][prog_branch])
        sec_all_bool = in_hull(all_pos, hullv[idx][sec_branch])
        
        prog_gas_mass_fraction = prog_gas_mass_total/all_mass[prog_all_bool].sum() 
        sec_gas_mass_fraction = sec_gas_mass_total/all_mass[sec_all_bool].sum() 
        #gas_mass_sf_fraction = gas_mass_sf/all_mass[all_bool].sum()
        
    return prog_gas_mass_total, prog_gas_mass_sf, prog_gas_massZ, prog_gas_mass_fraction, sec_gas_mass_total, sec_gas_mass_sf, sec_gas_massZ, sec_gas_mass_fraction
    
#---------------------------------------------------------------------------------------------------------------------------------
halotree_ver = 2013
merger_number = '0'


my_storage = {}
for sto, codetp in yt.parallel_objects(codetp_list, nprocs, storage = my_storage):
    redshift_list, time_list, pfs, step = load_halotree_and_pfs(codetp, halotree_ver, rawtree_skip = True)
    dist_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/dist_relvel_j_ProgBranch-%s_FirstMerger_%s_ver2013.npy' % (merger_number, codetp), allow_pickle=True).tolist()
    prog_branch, sec_branch, sec_branch_2 = sec_branch_compute(codetp, merger_number)
    idx_begin, idx_endinfall, idx_1stpass, idx_maxdist, idx_cls, time_begin, time_endinfall, time_1stpass, time_maxdist, time_cls = load_timings(codetp, halotree_ver, merger_number)
    
    metadata_dir = '/work/hdd/bezm/tnguyen2/AGORA/%s/metadata/' % codetp
    hullv = np.load('/work/hdd/bezm/gtg115x/Halo_Finding/%s/hullv_%s_withPos_final.npy' % (codetp, halotree_ver), allow_pickle=True).tolist()
    #
    prog_gas_mass_total, prog_gas_mass_sf, prog_gas_massZ, prog_gas_mass_fraction, sec_gas_mass_total, sec_gas_mass_sf, sec_gas_massZ, sec_gas_mass_fraction = calc_gas_mass_frac(idx_begin - step, codetp, pfs, hullv)
    #
    sto.result = {}
    sto.result[0] = prog_gas_mass_total
    sto.result[1] = prog_gas_mass_sf
    sto.result[2] = prog_gas_massZ
    sto.result[3] = prog_gas_mass_fraction
    sto.result[4] = sec_gas_mass_total
    sto.result[5] = sec_gas_mass_sf
    sto.result[6] = sec_gas_massZ
    sto.result[7] = sec_gas_mass_fraction
    sto.result[8] = idx_begin - step
    sto.result[9] = codetp


output = {}
for c, vals in sorted(my_storage.items()):
    if vals != None:
        output[vals[9]] = {}
        output[vals[9]]['prog_gas_mass_total'] = vals[0]
        output[vals[9]]['prog_gas_mass_sf'] = vals[1]
        output[vals[9]]['prog_gas_massZ'] = vals[2]
        output[vals[9]]['prog_gas_mass_fraction'] = vals[3]
        output[vals[9]]['sec_gas_mass_total'] = vals[4]
        output[vals[9]]['sec_gas_mass_sf'] = vals[5]
        output[vals[9]]['sec_gas_massZ'] = vals[6]
        output[vals[9]]['sec_gas_mass_fraction'] = vals[7]
        output[vals[9]]['idx'] = vals[8]
    
if yt.is_root():
    np.save('/work/hdd/bezm/tnguyen2/AGORA/analysis/Gas_Properties_preInfall_allCodes.npy', output)

