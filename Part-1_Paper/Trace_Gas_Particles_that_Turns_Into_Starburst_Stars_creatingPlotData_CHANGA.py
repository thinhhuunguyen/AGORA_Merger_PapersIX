import yt
yt.set_log_level(0)
import numpy as np
import sys
import unyt
 
import setup
from importlib import reload
reload(setup)

from setup import load_halotree_and_pfs, load_timings, load_ds, load_tracking_dist_data, sec_branch_compute
from setup import gas_name_dict, gas_temp_dict, gas_mass_dict
from setup import add_metallicity_fields, add_cooling_fields, add_radialdist_to_halocenter_field, extract_and_order_snapshotIdx
from setup import get_haloradius, infall_timestep_compute_hullv
from setup import codetp_list, label_list, color_list, marker_list
from setup import codetp_list10, label_list10, color_list10, marker_list10

def add_new_entry(output, ID):
    output[ID] = {}
    output[ID]['pos'] = np.empty(shape=(0,3))
    output[ID]['vel'] = np.empty(shape=(0,3))
    output[ID]['time'] = np.array([])
    output[ID]['mass'] = np.array([])
    output[ID]['temp'] = np.array([])
    return output

codetp = sys.argv[1] #only for CHANGA and GADGET4
print(codetp)
merger_number = '0'
halotree_ver = 2013

#rawtree, redshift_list, time_list, pfs, step = load_halotree_and_pfs(codetp, halotree_ver)
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
    

output = {}
track_sID = np.array([])

idx_0 = dist_data['idx'][-1]
ds0 = load_ds(codetp, idx_0, pfs)
reg0 = ds0.all_data()
if codetp == 'CHANGA':
    gas_ID0 = reg0[gas_name_dict[codetp], 'iord'].astype(int).v
else:
    gas_ID0 = reg0[gas_name_dict[codetp], 'particle_index'].astype(int).v
del ds0, reg0

for idx in np.flip(dist_data['idx'])[1:]:
    ds = load_ds(codetp, idx, pfs)    
    #
    #Normalize the center
    center = np.array(dist_data['prog_com_plot'])[dist_data['idx'] == idx][0]
    center = (center*ds.units.code_length).to('kpc').v
    #
    reg = ds.all_data()
    if codetp == 'CHANGA':
        gas_ID = reg[gas_name_dict[codetp], 'iord'].astype(int).v
    else:
        gas_ID = reg[gas_name_dict[codetp], 'particle_index'].astype(int).v
    gas_pos = reg[gas_name_dict[codetp], 'particle_position'].to('kpc').v
    gas_vel = reg[gas_name_dict[codetp], 'particle_velocity'].to('km/s').v
    gas_mass = reg[gas_name_dict[codetp], gas_mass_dict[codetp]].to('Msun').v
    gas_temp = reg[gas_name_dict[codetp], gas_temp_dict[codetp]].v
    gas_dist = np.linalg.norm(gas_pos - center, axis=1)
    #Add the ones formed earlier in the loop
    if len(track_sID) != 0:
        track_bool = np.intersect1d(track_sID, gas_ID, return_indices=True)[2]
        track_spos = gas_pos[track_bool]
        track_svel = gas_vel[track_bool]
        track_smass = gas_mass[track_bool]
        track_stemp = gas_temp[track_bool]
        track_sID = gas_ID[track_bool]
        track_spos = track_spos - center
        for j in range(len(track_sID)):
            output[track_sID[j]]['pos'] = np.vstack((output[track_sID[j]]['pos'],track_spos[j]))
            output[track_sID[j]]['vel'] = np.vstack((output[track_sID[j]]['vel'],track_svel[j]))
            output[track_sID[j]]['mass'] = np.append(output[track_sID[j]]['mass'], track_smass[j])
            output[track_sID[j]]['temp'] = np.append(output[track_sID[j]]['temp'], track_stemp[j])
            output[track_sID[j]]['time'] = np.append(output[track_sID[j]]['time'], time_list[idx]/1e3)
    #
    #Add the new stars formed during this iteration
    ts_ID = np.setdiff1d(gas_ID, gas_ID0)
    ts_bool = np.intersect1d(ts_ID, gas_ID, return_indices=True)[2]
    ts_pos = gas_pos[ts_bool]
    ts_vel = gas_vel[ts_bool]
    ts_mass = gas_mass[ts_bool]
    ts_temp = gas_temp[ts_bool]
    ts_ID = gas_ID[ts_bool]
    ts_dist = gas_dist[ts_bool]
    #
    #Limit to the main galaxy
    if idx <= idx_1stpass:
        dist_bool = ts_dist < 50 #kpc
    elif idx > idx_1stpass and idx <= 220:
        dist_bool = ts_dist < 20 #kpc
    else:
        dist_bool = ts_dist < 0.2*np.array(dist_data['prog_radius'])[np.array(dist_data['idx']) == idx][0] #kpc
    ts_mass = ts_mass[dist_bool]
    ts_vel = ts_vel[dist_bool]
    ts_temp = ts_temp[dist_bool]
    ts_ID = ts_ID[dist_bool]
    ts_pos = ts_pos[dist_bool]
    ts_pos = ts_pos - center
    #
    for j in range(len(ts_ID)):
        if ts_ID[j] not in output.keys():
            output = add_new_entry(output, ts_ID[j])
        output[ts_ID[j]]['pos'] = np.vstack((output[ts_ID[j]]['pos'],ts_pos[j]))
        output[ts_ID[j]]['vel'] = np.vstack((output[ts_ID[j]]['vel'],ts_vel[j]))
        output[ts_ID[j]]['mass'] = np.append(output[ts_ID[j]]['mass'], ts_mass[j])
        output[ts_ID[j]]['temp'] = np.append(output[ts_ID[j]]['temp'], ts_temp[j])
        output[ts_ID[j]]['time'] = np.append(output[ts_ID[j]]['time'], time_list[idx]/1e3)
    
    #Adding the new gas into the track lis
    track_sID = np.unique(np.append(track_sID, ts_ID))
    
    #Before moving to the next timestep, reassign the variables 
    gas_ID0 = gas_ID
    print('Done with Snapshot', idx)
    del gas_pos, gas_mass, gas_temp, gas_dist, gas_vel
    
np.save('/work/hdd/bezm/tnguyen2/AGORA/analysis/Trace_Gas_Particles_that_Turns_Into_Starburst_Stars_ProgBranch-%s_%s_ver2013_ver2.npy' % (merger_number, codetp), output)
    

    