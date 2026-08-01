import yt
import numpy as np
from Find_Galactic_Center import Find_Galactic_Center
import sys

from setup import load_halotree_and_pfs, sec_branch_compute

codetp = sys.argv[1]
branch_idx = sys.argv[2]
start_idx = int(sys.argv[3])
end_idx = int(sys.argv[4])
use_previous_gal_com = True #if this is True, then the process cannot be parallelized
expand_factor = 2
halotree_ver = 2013
merger_number = '0'

rawtree, redshift_list, time_list, pfs, step = load_halotree_and_pfs(codetp, halotree_ver)
prog_branch, sec_branch, sec_branch_2 = sec_branch_compute(codetp, merger_number)

for idx in range(start_idx, end_idx-step, -step):
    if idx not in rawtree[branch_idx].keys():
        continue
    if codetp == 'GADGET3' or codetp == 'AREPO':
        ds = yt.load(pfs[int(idx)], unit_base = {"length": (1.0, "Mpccm/h")})
    else:
        ds = yt.load(pfs[int(idx)])
    metadata = np.load('/work/hdd/bezm/tnguyen2/AGORA/%s/metadata/star_metadata_allbox_%s.npy' % (codetp, idx), allow_pickle=True).tolist()
    #
    center = Find_Galactic_Center(ds = ds, oden = 2000, halo_center = rawtree[branch_idx][idx]['Halo_Center'],
                                  halo_rvir = expand_factor*rawtree[branch_idx][idx]['Halo_Radius'],
                                  star_pos = metadata['pos'], star_mass = metadata['mass'])
    if use_previous_gal_com == False:
        new_com, new_virRad = center.Find_Com_and_virRad()
    else:
        previous_gal_com = np.load('/work/hdd/bezm/tnguyen2/AGORA/%s/radius_2000_%s/Galaxy_Halo_%s_Snapshot_%s_comR2000.npy' % (codetp, codetp, branch_idx, int(idx)+step), allow_pickle=True).tolist()['com']
        new_com, new_virRad = center.Find_Com_and_virRad(initial_gal_com_manual = True, initial_gal_com = (previous_gal_com*ds.units.code_length).to('m').v.tolist())
    #
    output_each = {}
    output_each['com'] = (new_com*center.ds.units.m).to('code_length').v
    output_each['r2000'] = (new_virRad*center.ds.units.m).to('code_length').v.tolist()
    #
    np.save('/work/hdd/bezm/tnguyen2/AGORA/%s/radius_2000_%s/Galaxy_Halo_%s_Snapshot_%s_comR2000.npy' % (codetp, codetp, branch_idx, idx), output_each)
    print(idx, output_each['r2000']/rawtree[branch_idx][idx]['Halo_Radius'])
    del center, ds, metadata

    
    
