import yt
yt.set_log_level(0)
import numpy as np
import sys
import unyt
 
import setup
from importlib import reload
reload(setup)

from setup import load_halotree_and_pfs, load_timings, load_ds, load_tracking_dist_data, sec_branch_compute
from setup import gas_name_dict, gas_temp_dict
from setup import add_metallicity_fields, add_cooling_fields, add_radialdist_to_halocenter_field, extract_and_order_snapshotIdx
from setup import get_haloradius, infall_timestep_compute_hullv
from setup import codetp_list, label_list, color_list, marker_list
from setup import codetp_list10, label_list10, color_list10, marker_list10


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
print(codetp)
merger_number = '0'
halotree_ver = 2013

#rawtree, redshift_list, time_list, pfs, step = load_halotree_and_pfs(codetp, halotree_ver)
redshift_list, time_list, pfs, step = load_halotree_and_pfs(codetp, halotree_ver, rawtree_skip=True)
metadata_dir = '/work/hdd/bezm/tnguyen2/AGORA/%s/metadata/' % codetp
assignment = np.load('/work/hdd/bezm/gtg115x/Halo_Finding/%s/star_id_%s_final_firstmerger.npy' % (codetp, halotree_ver), allow_pickle=True).tolist()
    
dist_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/dist_relvel_j_ProgBranch-%s_FirstMerger_%s_ver2013.npy' % (merger_number, codetp), allow_pickle=True).tolist()
prog_branch, sec_branch, sec_branch_2 = sec_branch_compute(codetp, merger_number)
idx_begin, idx_endinfall, idx_1stpass, idx_maxdist, idx_cls, time_begin, time_endinfall, time_1stpass, time_maxdist, time_cls = load_timings(codetp, halotree_ver, merger_number)
time_eval = time_1stpass + 0.6

if codetp == 'GEAR':
    idx_eval = np.argmin(abs(time_eval - time_list/1e3)) - (np.argmin(abs(time_eval - time_list/1e3)) % step) - 1
else:
    idx_eval = np.argmin(abs(time_eval - time_list/1e3)) - (np.argmin(abs(time_eval - time_list/1e3)) % step)
    
    
merger_sID = assignment['ids'][prog_branch][idx_eval]

metadata = np.load(metadata_dir + 'star_metadata_allbox_%s.npy' % idx_eval, allow_pickle=True).tolist()
mass_all = metadata['mass']
age_all = metadata['age']
if codetp == 'RAMSES':
    age_all = (time_list[idx_eval].astype(float)/1000) - age_all
pos_all = metadata['pos']
ID_all = metadata['ID']

merger_sage = age_all[np.intersect1d(merger_sID, ID_all, return_indices=True)[2]]
merger_sID = ID_all[np.intersect1d(merger_sID, ID_all, return_indices=True)[2]]
merger_spos = pos_all[np.intersect1d(merger_sID, ID_all, return_indices=True)[2]]
merger_sftime = time_list[idx_eval]/1e3 - merger_sage
center = np.array(dist_data['prog_com_plot'])[np.array(dist_data['idx']) == idx_eval][0]

pos_bool = (np.linalg.norm(merger_spos - center, axis=1)*np.array(dist_data['codelength_to_meters'])[np.array(dist_data['idx']) == idx_eval][0]*unyt.m).to('kpc').v < 0.2*np.array(dist_data['prog_radius'])[np.array(dist_data['idx']) == idx_eval][0] #kpc
time_bool = (merger_sftime - time_begin) > 0
track_sID = merger_sID[pos_bool*time_bool]
#track_spos = merger_spos[pos_bool*time_bool]

for n in range(1,10): #every time a star is formed from a gas particle, its ID gets an increment of 34428280. This code traces the star IDs back to the parent gas IDs
    track_sID[track_sID > 34428280] -= 34428280

track_sID = np.unique(track_sID)

output = {}
for ID in track_sID:
    output[ID] = {}
    output[ID]['pos'] = np.empty(shape=(0,3))
    output[ID]['vel'] = np.empty(shape=(0,3))
    output[ID]['time'] = np.array([])
    output[ID]['temp'] = np.array([])
    output[ID]['mass'] = np.array([])
    output[ID]['firststar_pos'] = np.empty(shape=(0,3))
    output[ID]['firststar_time'] = np.array([])
    
for idx in range(idx_begin - step, idx_eval + step, step):
    ds = load_ds(codetp, idx, pfs)
    #
    reg = ds.all_data()
    gas_ID = reg[gas_name_dict[codetp], 'particle_index'].astype(int).v
    gas_pos = reg[gas_name_dict[codetp], 'particle_position'].to('kpc').v
    gas_vel = reg[gas_name_dict[codetp], 'particle_velocity'].to('km/s').v
    gas_temp = reg[gas_name_dict[codetp], gas_temp_dict[codetp]].v
    gas_mass = reg[gas_name_dict[codetp], 'particle_mass'].to('Msun').v
    #
    metadata = np.load(metadata_dir + 'star_metadata_allbox_%s.npy' % idx, allow_pickle=True).tolist()
    age_all = metadata['age']
    if codetp == 'RAMSES':
        age_all = (time_list[idx].astype(float)/1000) - age_all
    pos_all = metadata['pos']
    vel_all = metadata['vel']
    mass_all = metadata['mass']
    ID_all = metadata['ID']
    #
    #Normalize the center
    if idx == idx_begin - step:
        bound_ID = assignment['ids']['0'][idx]
        bound_pos = pos_all[np.intersect1d(bound_ID, ID_all, return_indices=True)[2]]
        bound_vel = vel_all[np.intersect1d(bound_ID, ID_all, return_indices=True)[2]]
        bound_mass = mass_all[np.intersect1d(bound_ID, ID_all, return_indices=True)[2]]
        center = determine_center(bound_pos, bound_vel, bound_mass)
    else:
        center = np.array(dist_data['prog_com_plot'])[dist_data['idx'] == idx][0]
    center = (center*ds.units.code_length).to('kpc').v
    #
    newstars_bool = age_all <= (time_list[idx] - time_list[idx - step])/1e3
    newstars_ID = ID_all[newstars_bool]
    newstars_pos = pos_all[newstars_bool]
    for n in range(1,10): #every time a star is formed from a gas particle, its ID gets an increment of 34428280. This code traces the star IDs back to the parent gas IDs
        newstars_ID[newstars_ID > 34428280] -= 34428280

    for k in range(len(newstars_ID)):
        if newstars_ID[k] in track_sID:
            matching_pos = newstars_pos[k]
            matching_pos = (matching_pos*ds.units.code_length).to('kpc').v
            matching_pos = matching_pos - center
            output[newstars_ID[k]]['firststar_pos'] = np.vstack((output[newstars_ID[k]]['firststar_pos'], matching_pos))
            output[newstars_ID[k]]['firststar_time'] = np.append(output[newstars_ID[k]]['firststar_time'], time_list[idx]/1e3)
    #
    ts_ID = np.intersect1d(track_sID, gas_ID).astype(int) #ts = turned-stars
    ts_bool = np.intersect1d(ts_ID, gas_ID, return_indices=True)[2]
    ts_pos = gas_pos[ts_bool]
    ts_vel = gas_vel[ts_bool]
    ts_temp = gas_temp[ts_bool]
    ts_mass = gas_mass[ts_bool]
    ts_ID = gas_ID[ts_bool]
    #
    ts_pos = ts_pos - center
    #
    for j in range(len(ts_ID)):
        output[ts_ID[j]]['pos'] = np.vstack((output[ts_ID[j]]['pos'],ts_pos[j]))
        output[ts_ID[j]]['vel'] = np.vstack((output[ts_ID[j]]['vel'],ts_vel[j]))
        output[ts_ID[j]]['temp'] = np.append(output[ts_ID[j]]['temp'], ts_temp[j])
        output[ts_ID[j]]['mass'] = np.append(output[ts_ID[j]]['mass'], ts_mass[j])
        output[ts_ID[j]]['time'] = np.append(output[ts_ID[j]]['time'], time_list[idx]/1e3)
    del ds, metadata
    
np.save('/work/hdd/bezm/tnguyen2/AGORA/analysis/Trace_Gas_Particles_that_Turns_Into_Starburst_Stars_ProgBranch-%s_%s_ver2013_ver2.npy' % (merger_number, codetp), output)
    

    