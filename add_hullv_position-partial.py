import numpy as np
import yt
import sys, os
from yt.data_objects.particle_filters import add_particle_filter
if int((yt.__version__).split('.')[0]) >= 4 and int((yt.__version__).split('.')[1]) >= 2: #ParticleUnion is only available in yt 4.2 and later
    from yt.data_objects.unions import ParticleUnion
else:
    from yt.data_objects.unions import Union as ParticleUnion
from setup import load_ds
    
yt.enable_parallelism()
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
nprocs = comm.size
    
codetp = sys.argv[1]
halotree_ver = sys.argv[2]
if yt.is_root():
    print(codetp)

hullv = np.load('/work/hdd/bezm/gtg115x/Halo_Finding/%s/hullv_%s_final.npy' % (codetp, halotree_ver), allow_pickle=True).tolist()
hullv_pos = {}
pfs = np.loadtxt('/work/hdd/bezm/gtg115x/Halo_Finding/%s/pfs_allsnaps_%s.txt' % (codetp, halotree_ver), dtype=str)[:,0]

hullv_snapFirst = {}
for idx in range(0, len(pfs)):
    hullv_snapFirst[idx] = {}
    for branch in hullv.keys():
        if codetp == 'GADGET4' and ((branch.startswith('0') and branch.count('_') <= 1) or branch == '4'):
            if idx in hullv[branch].keys():
                hullv_snapFirst[idx][branch] = hullv[branch][idx]
        elif codetp == 'ART' and ((branch.startswith('0') and branch.count('_') <= 1) or branch == '5'):
            if idx in hullv[branch].keys():
                hullv_snapFirst[idx][branch] = hullv[branch][idx]
        elif codetp == 'RAMSES' and ((branch.startswith('0') and branch.count('_') <= 1) or branch == '577_0'):
            if idx in hullv[branch].keys():
                hullv_snapFirst[idx][branch] = hullv[branch][idx]
        elif codetp == 'GADGET3' and ((branch.startswith('0') and branch.count('_') <= 1) or branch == '3'):
            if idx in hullv[branch].keys():
                hullv_snapFirst[idx][branch] = hullv[branch][idx]
        elif codetp == 'AREPO' and ((branch.startswith('0') and branch.count('_') <= 1) or branch == '0_170_0'):
            if idx in hullv[branch].keys():
                hullv_snapFirst[idx][branch] = hullv[branch][idx]
        else:
            if branch.startswith('0') and branch.count('_') <= 1:
                if idx in hullv[branch].keys():
                    hullv_snapFirst[idx][branch] = hullv[branch][idx]
del hullv
            
            
def load_dm_particles(idx):
    save_part = '/work/hdd/bezm/gtg115x/Halo_Finding/%s/particle_save/' % (codetp)
    if os.path.exists(save_part + 'part_dict.npy'):
        ds = load_ds(codetp, idx, pfs)
        meter = ds.length_unit.in_units('m')
        #
        left, right = -1e99, 1e99 #load whole box
        part_dict = np.load(save_part + 'part_dict.npy', allow_pickle=True).tolist()
        ind_array = np.arange(len(part_dict[idx]))
        ll = part_dict[idx]['ll']
        ur = part_dict[idx]['ur']
        ind_array = np.arange(len(ll))
        bool_overlap = (np.sum( ur <= left,axis=1)==0)*(np.sum(ll >= right,axis=1)==0)
        ind_array = ind_array[bool_overlap]
        pos,ids = np.array([[]]),np.array([])
        #
        for i in ind_array:
            part = np.load(save_part+'/part_%s_%s.npy' % (idx,i),allow_pickle= True).tolist()
            if len(ids) ==0:
                pos = part['pos']
                ids = part['ids']
            else:
                pos = np.vstack((pos,part['pos']))
                ids = np.append(ids,part['ids'])
        if len(ids)>0:
            bool_in = (np.sum(pos >= left*meter,axis=1) ==3)*(np.sum(pos < right*meter,axis=1) ==3)
            pos,ids = pos[bool_in],ids[bool_in]
        return pos.v, ids
    else:
        ds = load_ds(codetp, idx, pfs)
        if codetp == 'AREPO' or codetp == 'AREPO-TNG':
            reg = ds.all_data()
        else:
            left, right = np.array([1e89,1e89,1e89]),-1*np.array([1e89,1e89,1e89])
            for times in range(len(pfs)):
                if os.path.exists('/work/hdd/bezm/gtg115x/Halo_Finding/%s/Refined/refined_region_%s.npy' % (codetp, int(times))):
                    ll_o,ur_o = np.load('/work/hdd/bezm/gtg115x/Halo_Finding/%s/Refined/refined_region_%s.npy' % (codetp, int(times)),allow_pickle=True).tolist()
                    left = np.minimum(left,ll_o)
                    right = np.maximum(right,ur_o)
            buffer = (np.array(left) - np.array(right))*0.1
            left,right = np.array(left)-buffer,np.array(right)+buffer
            #
            reg = ds.box(left, right)
        dm_name_dict = {'ENZO':'DarkMatter','GEAR': 'DarkMatter', 'GADGET3': 'DarkMatter', 'AREPO': 'DarkMatter', 'AREPO-TNG': 'DarkMatter', 'GIZMO': 'DarkMatter', 'RAMSES': 'DM', 'ART': 'darkmatter', 'CHANGA': 'DarkMatter'}
        if codetp == 'ENZO':
            def darkmatter_init(pfilter, data):
                filter_darkmatter0 = np.logical_or(data["all", "particle_type"] == 1, data["all", "particle_type"] == 4)
                filter_darkmatter = np.logical_and(filter_darkmatter0,data['all', 'particle_mass'].to('Msun') > 1)
                return filter_darkmatter
            add_particle_filter("DarkMatter",function=darkmatter_init,filtered_type='all',requires=["particle_type","particle_mass"])
            ds.add_particle_filter("DarkMatter")
        if codetp == 'AREPO' or codetp == 'AREPO-TNG':
            dm = ParticleUnion("DarkMatter",["PartType2","PartType1"])
            ds.add_particle_union(dm)
        pos = reg[dm_name_dict[codetp], 'particle_position'].to('m').v
        ids = reg[dm_name_dict[codetp], 'particle_index'].v.astype(int) 
        return pos, ids

def add_hullv_pos(idx):
    hullv_pos_idx = {}
    pos, ids = load_dm_particles(idx)
    for branch_i in hullv_snapFirst[idx].keys():
        hullv_pos_i = pos[np.intersect1d(hullv_snapFirst[idx][branch_i], ids, return_indices=True)[2]]
        if len(hullv_pos_i) != len(hullv_snapFirst[idx][branch_i]):
            print('Error - Missing dm particles -- at Idx %s and Branch %s' % (idx, branch_i))
        hullv_pos_idx[branch_i] = hullv_pos_i
    return hullv_pos_idx

my_storage = {}
if codetp == 'AREPO':
    starting_idx = 6
else:
    starting_idx = 0
for sto, j in yt.parallel_objects(range(starting_idx, len(pfs)), nprocs, storage = my_storage):
    hullv_pos_idx = add_hullv_pos(j)
    sto.result = {}
    sto.result[0] = j
    sto.result[1] = hullv_pos_idx
    
for c, vals in sorted(my_storage.items()):
    if vals != None:
        hullv_pos[vals[0]] = vals[1]

if yt.is_root():
    np.save('/work/hdd/bezm/gtg115x/Halo_Finding/%s/hullv_%s_withPos_final.npy' % (codetp, halotree_ver), hullv_pos)

