import yt
yt.set_log_level(0)
import numpy as np
import os, sys

import setup
from importlib import reload
reload(setup)

from setup import load_halotree_and_pfs, load_timings, sec_branch_compute

yt.enable_parallelism()
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
nprocs = comm.size

codetp = sys.argv[1]
merger_number = '0'
halotree_ver = 2013

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% General data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

redshift_list, time_list, pfs, step = load_halotree_and_pfs(codetp, halotree_ver, rawtree_skip=True)
metadata_dir = '/work/hdd/bezm/tnguyen2/AGORA/%s/metadata/' % codetp

dist_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/dist_relvel_j_ProgBranch-%s_FirstMerger_%s_ver2013.npy' % (merger_number, codetp), allow_pickle=True).tolist()
prog_branch, sec_branch, sec_branch_2 = sec_branch_compute(codetp, merger_number)
idx_begin, idx_endinfall, idx_1stpass, idx_maxdist, idx_cls, time_begin, time_endinfall, time_1stpass, time_maxdist, time_cls = load_timings(codetp, halotree_ver, merger_number)
time_eval = time_1stpass + 0.6

if codetp == 'GEAR':
    idx_eval = np.argmin(abs(time_eval - time_list/1e3)) - (np.argmin(abs(time_eval - time_list/1e3)) % step) - 1
else:
    idx_eval = np.argmin(abs(time_eval - time_list/1e3)) - (np.argmin(abs(time_eval - time_list/1e3)) % step)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Gas tracing data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

output = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/Trace_Gas_Particles_that_Turns_Into_Starburst_Stars_ProgBranch-0_%s_ver2013_ver2.npy' % codetp, allow_pickle=True).tolist()
star_id = np.array(list(output.keys())).astype(int)
gas_index_bound = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/bound_gas_index_sec_galaxy_ProgBranch-0_%s_ver2013.npy' % codetp, allow_pickle=True).tolist()
gas_from_sec = np.intersect1d(star_id, gas_index_bound)
#gas_not_sec = np.setdiff1d(star_id, gas_index_bound)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Analysis %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


my_storage = {}
for sto, idx_test in yt.parallel_objects(range(idx_begin, idx_eval + step, step), nprocs, storage = my_storage):
    dist_list = []
    radialvel_list = []
    mass_list = []
    temp_list = []
    ID_list = []
    #    
    vel_com = np.average(np.array(dist_data['prog_vel_plot'])[dist_data['idx'] == idx_test][0], weights=np.array(dist_data['prog_mass_plot'])[dist_data['idx'] == idx_test][0], axis=0) # in km/s
    for gas_idx in gas_from_sec:  
        if codetp == 'CHANGA':
            if (output[gas_idx]['time'][0] < time_list[idx_test]/1e3) or (output[gas_idx]['time'][0] > time_list[idx_eval]/1e3): 
                continue
        elif codetp == 'GADGET4':
            if (output[gas_idx]['time'][-1] < time_list[idx_test]/1e3)  or (output[gas_idx]['firststar_time'][0] > time_list[idx_eval+step]/1e3): 
                continue
        else:
            if (output[gas_idx]['time'][-1] < time_list[idx_test]/1e3) or (output[gas_idx]['time'][-1] > time_list[idx_eval]/1e3): 
                continue
        codelength_to_kpc = np.array(dist_data['codelength_to_meters'])[np.array(dist_data['idx']) == idx_test][0]*3.24077929e-20 #convert to kpc
        prog_center = np.array(dist_data['prog_com_plot'])[np.array(dist_data['idx']) == idx_test][0]*codelength_to_kpc
        sec_center = np.array(dist_data['sec_com_plot'])[np.array(dist_data['idx']) == idx_test][0]*codelength_to_kpc
        prog_to_sec = sec_center - prog_center
        prog_to_gas = (output[gas_idx]['pos'])[np.isclose(output[gas_idx]['time'], time_list[idx_test]/1e3, atol=1e-10)][0]
        sec_to_gas = prog_to_gas - prog_to_sec
        #
        vel_gas = (output[gas_idx]['vel'])[np.isclose(output[gas_idx]['time'], time_list[idx_test]/1e3, atol=1e-10)][0] - vel_com
        radialvel_gas = np.dot(vel_gas, prog_to_gas/np.linalg.norm(prog_to_gas))
        #
        dist_list.append(np.linalg.norm(prog_to_gas))
        radialvel_list.append(radialvel_gas)
        mass_list.append((output[gas_idx]['mass'])[np.isclose(output[gas_idx]['time'], time_list[idx_test]/1e3, atol=1e-10)][0])
        temp_list.append((output[gas_idx]['temp'])[np.isclose(output[gas_idx]['time'], time_list[idx_test]/1e3, atol=1e-10)][0])
        ID_list.append(gas_idx)
        
    sto.result = {}
    sto.result[0] = np.array(dist_list)
    sto.result[1] = np.array(radialvel_list)
    sto.result[2] = np.array(mass_list)
    sto.result[3] = np.array(temp_list)
    sto.result[4] = np.array(ID_list)
    sto.result[5] = time_list[idx_test]/1e3
    sto.result[6] = idx_test

analysis = {}
for c, vals in sorted(my_storage.items()):
    if vals != None:
        analysis[vals[6]] = {}
        analysis[vals[6]]['dist'] = vals[0]
        analysis[vals[6]]['radialvel'] = vals[1]
        analysis[vals[6]]['mass'] = vals[2]
        analysis[vals[6]]['temp'] = vals[3]
        analysis[vals[6]]['ID'] = vals[4]
        analysis[vals[6]]['time'] = vals[5]

if yt.is_root():
    print(codetp)
    np.save('/work/hdd/bezm/tnguyen2/AGORA/analysis/Select_gas_with_negative_radialvel_infalling_gasparticletracing_%s_ver2013.npy' % codetp, analysis)
    


