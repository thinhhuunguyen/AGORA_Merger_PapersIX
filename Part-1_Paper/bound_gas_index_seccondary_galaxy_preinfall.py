import yt
yt.set_log_level(0)
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull, Delaunay 
import glob as glob
import sys
from astropy.constants import G
    
import setup
from importlib import reload
reload(setup)

from setup import load_halotree_and_pfs, load_timings, load_ds, sec_branch_compute
from setup import gas_name_dict


def find_star_orbital_energy_square(dmpos,dmmass,velcom,starpos, starvel, starid, Numlength = 1000):
    N = len(dmpos)
    if len(dmpos) >0:
        diam = dmpos[:,0].max()- dmpos[:,0].min()
        diam = diam/N
    else:
        diam = 0
    numloops_dm = int(np.ceil(len(dmpos)/Numlength))
    numloops_star = int(np.ceil(len(starpos)/Numlength))
    pot = np.zeros(len(starpos))
    #print(N,Numlength,numloops)
    inds_dm = np.arange(len(dmpos))
    inds_star = np.arange(len(starpos))
    for i in range(numloops_star):
        ind_star  = inds_star[i*Numlength:min((i+1)*Numlength,len(starpos))]
        #print('ind dm shape', ind_dm.shape)
        for j in range(numloops_dm):
            ind_dm  = inds_dm[j*Numlength:min((j+1)*Numlength,len(dmpos))]
            #print('ind star shape', ind_star.shape)
            r = cdist(starpos[ind_star],dmpos[ind_dm])
            #print(r)
            bool = r ==0
            with np.errstate(divide='ignore'):
                r = 1/np.maximum(r,diam)
            mkg = dmmass[ind_dm]
            r[bool+~np.isfinite(r)] = 0
            s = np.sum((mkg*r), axis = 1)
            #print(s)
            if j == 0:
                U = s
            else:
                U += s
            #print(U.shape)
        # if i == 0:
        #     pot = U
        # else:
        #     pot = np.append(pot,U)
        pot[ind_star] = U
    #
    PE = -G.value*pot
    #velcom = np.average(dmvel, axis=0, weights=dmmass)
    KE = 0.5*np.linalg.norm(starvel - velcom, axis=1)**2
    E = KE + PE
    #print(PE)
    #print(KE)
    return E<0

def check_boundness(branch, idx, velcom, starpos, starvel, starid, dm_pos, dm_mass, Numlength = 1000):
    #This function loads the cut DM particle data of a halo (branch) at a timestep (idx) and check if the provided star/gas particles are bound to this halo
    return find_star_orbital_energy_square(dm_pos,dm_mass,velcom,starpos, starvel, starid)


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

    return hull.find_simplex(p)>=0


codetp = sys.argv[1]
merger_number = '0'
halotree_ver = 2013

rawtree, redshift_list, time_list, pfs, step = load_halotree_and_pfs(codetp, halotree_ver)
redshift_list, time_list, pfs, step = load_halotree_and_pfs(codetp, halotree_ver, rawtree_skip=True)
metadata_dir = '/work/hdd/bezm/tnguyen2/AGORA/%s/metadata/' % codetp
hullv = np.load('/work/hdd/bezm/gtg115x/Halo_Finding/%s/hullv_%s_withPos_final.npy' % (codetp, halotree_ver), allow_pickle=True).tolist()

dist_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/dist_relvel_j_ProgBranch-%s_FirstMerger_%s_ver2013.npy' % (merger_number, codetp), allow_pickle=True).tolist()
prog_branch, sec_branch, sec_branch_2 = sec_branch_compute(codetp, merger_number)
idx_begin, idx_endinfall, idx_1stpass, idx_maxdist, idx_cls, time_begin, time_endinfall, time_1stpass, time_maxdist, time_cls = load_timings(codetp, halotree_ver, merger_number)
time_eval = time_1stpass + 0.6

if codetp == 'GEAR':
    idx_eval = np.argmin(abs(time_eval - time_list/1e3)) - (np.argmin(abs(time_eval - time_list/1e3)) % step) - 1
else:
    idx_eval = np.argmin(abs(time_eval - time_list/1e3)) - (np.argmin(abs(time_eval - time_list/1e3)) % step)


save_part = '/work/hdd/bezm/gtg115x/Halo_Finding/%s/particle_save/' % (codetp)
part_dict = np.load(save_part + 'part_dict.npy', allow_pickle=True).tolist()

def load_dm_particles(left, right, meter): #make sure that the mass is in kg and the position is in m
    ind_array = np.arange(len(part_dict[idx]))
    ll = part_dict[idx]['ll']
    ur = part_dict[idx]['ur']
    ind_array = np.arange(len(ll))
    bool_overlap = (np.sum( ur <= left,axis=1)==0)*(np.sum(ll >= right,axis=1)==0)
    ind_array = ind_array[bool_overlap]
    mass,pos,vel,ids = np.array([]),np.array([[]]),np.array([[]]),np.array([])
    #
    for i in ind_array:
        part = np.load(save_part+'/part_%s_%s.npy' % (idx,i),allow_pickle= True).tolist()
        if len(mass) ==0:
            mass = part['mass']
            pos = part['pos']
            vel = part['vel']
            ids = part['ids']
        else:
            mass = np.append(mass,part['mass'])
            pos = np.vstack((pos,part['pos']))
            vel = np.vstack((vel,part['vel']))
            ids = np.append(ids,part['ids'])
    if len(mass)>0 and len(pos)>0:
        bool_in = (np.sum(pos >= left*meter,axis=1) ==3)*(np.sum(pos < right*meter,axis=1) ==3)
        mass,pos,vel,ids = mass[bool_in],pos[bool_in],vel[bool_in],ids[bool_in]
    return mass, pos, vel, ids


#THIS CODE CELLS SELECT GAS PARTICLES TO TRACE 
#Set the branch
branch = sec_branch
idx = idx_begin - step
ds = load_ds(codetp, idx, pfs)
meters = ds.length_unit.in_units('m')
vel_sec = rawtree[branch][idx]['Vel_Com']*meters.v
del rawtree
#
reg = ds.all_data()
gas_sphere_pos = reg[gas_name_dict[codetp],'particle_position'].to('m')
gas_sphere_mass = reg[gas_name_dict[codetp],'particle_mass'].to('Msun')
gas_sphere_vel = reg[gas_name_dict[codetp],'particle_velocity'].to('m/s')
if codetp != 'CHANGA':
    gas_sphere_index = reg[gas_name_dict[codetp],'particle_index'].astype(int).v
else:
    gas_sphere_index = reg[gas_name_dict[codetp],'iord'].astype(int).v

dm_mass, dm_pos, _, _ = load_dm_particles(-1e99, 1e99, meters)
    
hullv_pos = hullv[idx][sec_branch]

gas_inhull_bool = in_hull(gas_sphere_pos, hullv_pos) 
gas_pos_hull = gas_sphere_pos[gas_inhull_bool]
gas_mass_hull = gas_sphere_mass[gas_inhull_bool]
gas_vel_hull = gas_sphere_vel[gas_inhull_bool]
gas_index_hull = gas_sphere_index[gas_inhull_bool]

dm_inhull_bool = in_hull(dm_pos, hullv_pos) 
dm_pos_hull = dm_pos[dm_inhull_bool]
dm_mass_hull = dm_mass[dm_inhull_bool]

bound_bool = check_boundness(branch, idx, vel_sec, gas_pos_hull.v, gas_vel_hull.v, gas_index_hull, dm_pos_hull, dm_mass_hull)
gas_index_bound = gas_index_hull[bound_bool]
np.save('/work/hdd/bezm/tnguyen2/AGORA/analysis/bound_gas_index_sec_galaxy_ProgBranch-0_%s_ver2013.npy' % codetp, gas_index_bound)
