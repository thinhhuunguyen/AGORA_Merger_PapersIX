import numpy as np
import yt
from yt import YTArray
from scipy.spatial import ConvexHull, Delaunay
from scipy.interpolate import CubicSpline
yt.set_log_level(0)

agora_Zsun = 0.02041

codetp_list = ['ART', 'ENZO', 'RAMSES', 'CHANGA', 'GADGET3', 'GADGET4', 'GEAR', 'AREPO', 'GIZMO']
label_list = ['ART-I', 'ENZO', 'RAMSES', 'CHANGA', 'GADGET-3', 'GADGET-4', 'GEAR', 'AREPO-T', 'GIZMO']
color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:pink', 'tab:brown', 'tab:olive', 'tab:purple', 'tab:red', 'tab:grey']
marker_list = ['s', 'o', 'p', 'v', '^', '<', '*', '>', 'h']

codetp_list10 = ['ART', 'ENZO', 'RAMSES', 'CHANGA', 'GADGET3', 'GADGET4', 'GEAR', 'AREPO', 'AREPO-TNG', 'GIZMO']
label_list10 = ['ART-I', 'ENZO', 'RAMSES', 'CHANGA', 'GADGET-3', 'GADGET-4', 'GEAR', 'AREPO-T', 'AREPO-TNG', 'GIZMO']
color_list10 = ['tab:blue', 'tab:orange', 'tab:green', 'tab:pink', 'tab:brown', 'tab:olive', 'tab:purple', 'tab:red', 'tab:cyan', 'tab:grey']
marker_list10 = ['s', 'o', 'p', 'v', '^', '<', '*', '>', 'D', 'h']

gas_name_dict = {'ENZO':'gas','GADGET3':'PartType0','GEAR':'PartType0','AREPO':'PartType0',\
                'GIZMO':'PartType0','RAMSES':'gas','ART':'gas','CHANGA':'Gas','GADGET4':'PartType0', 'AREPO-TNG':'PartType0'}

star_name_dict = {'ENZO':'stars','GADGET3':'PartType4','GEAR':'PartType1','AREPO':'PartType4',\
                'GIZMO':'PartType4','RAMSES':'star','ART':'stars','CHANGA':'Stars', 'GADGET4':'PartType4', 'AREPO-TNG':'PartType4'}

dm_name_dict = {'ENZO':'DarkMatter','GEAR': 'DarkMatter', 'GADGET3': 'DarkMatter','GADGET4': 'DarkMatter', 'AREPO': 'DarkMatter',\
                'GIZMO': 'DarkMatter', 'RAMSES': 'DM', 'ART': 'darkmatter', 'CHANGA': 'DarkMatter'}

gas_temp_dict = {'ART':'temperature', 'ENZO':'temperature', 'RAMSES':'temperature', 'CHANGA':'tempEff', 'GADGET3':'temperature','GADGET4':'temperature', 'GEAR':'temperature', 'AREPO':'temperature', 'GIZMO':'temperature'}

gas_mass_dict = {'ART':'cell_mass','RAMSES':'cell_mass', 'ENZO':'cell_mass', 'AREPO':'mass','GADGET3':'particle_mass','GADGET4':'particle_mass', 'GEAR':'particle_mass', 'CHANGA':'particle_mass', 'GIZMO':'particle_mass'}

star_mass_dict = {'ENZO':'particle_mass','GADGET3':'particle_mass','GADGET4':'particle_mass','GEAR':'particle_mass','AREPO':'particle_mass',\
                'GIZMO':'particle_mass','RAMSES':'particle_mass','ART':'particle_mass','CHANGA':'Mass'}


def extract_and_order_snapshotIdx(rawtree, branch):
    #this function extract only the snapshot key (i.e. the integer value) from the rawtree halotree output
    keys = list(rawtree[branch].keys())
    snapshotIdx = [x for x in keys if not isinstance(x, str)]
    snapshotIdx.sort()
    return snapshotIdx

def sec_branch_compute(codetp, merger_number, halotree_ver = 2013):
    if halotree_ver == 2013:
        if merger_number == '0':
            if codetp == 'ENZO':
                prog_branch = '0'
                sec_branch = '0_12'
                sec_branch_2 = None
            elif codetp == 'AREPO':
                prog_branch = '0'
                sec_branch = '0_170_0'
                sec_branch_2 = ['0_170']
            elif codetp == 'AREPO-TNG':
                prog_branch = '0'
                sec_branch = '0_192'
                sec_branch_2 = None
            elif codetp == 'GADGET3':
                prog_branch = '0'
                sec_branch = '0_150'
                sec_branch_2 = ['0_186']
            elif codetp == 'GADGET4':
                prog_branch = '0'
                sec_branch = '4'
                sec_branch_2 = ['0_102', '0_92']
            elif codetp == 'CHANGA':
                prog_branch = '0'
                sec_branch = '0_185'
                sec_branch_2 = None
            elif codetp == 'GIZMO':
                prog_branch = '0'
                sec_branch = '0_159'
                sec_branch_2 = None
            elif codetp == 'GEAR':
                prog_branch = '0'
                sec_branch = '0_120'
                sec_branch_2 = None
            elif codetp == 'ART':
                prog_branch = '0'
                sec_branch = '0_144'
                sec_branch_2 = None
            elif codetp == 'RAMSES':
                prog_branch = '0'
                sec_branch = '0_54'
                sec_branch_2 = None  
        if merger_number == '1': #the second merger of the main galaxy
            if codetp == 'ENZO':
                prog_branch = '0'
                sec_branch = '0_49'
                sec_branch_2 = None
            elif codetp == 'AREPO':
                prog_branch = '0'
                sec_branch = '0_74'
                sec_branch_2 = None
            elif codetp == 'GADGET3':
                prog_branch = '0'
                sec_branch = '0_53'
                sec_branch_2 = None
            elif codetp == 'GADGET4':
                prog_branch = '0'
                sec_branch = '0_49'
                sec_branch_2 = None
            elif codetp == 'CHANGA':
                prog_branch = '0'
                sec_branch = '0_37'
                sec_branch_2 = None
            elif codetp == 'GIZMO':
                prog_branch = '0'
                sec_branch = '0_1'
                sec_branch_2 = None
            elif codetp == 'GEAR':
                prog_branch = '0'
                sec_branch = '0_61'
                sec_branch_2 = None
            elif codetp == 'ART':
                prog_branch = '0'
                sec_branch = '5'
                sec_branch_2 = None
            elif codetp == 'RAMSES':
                prog_branch = '0'
                sec_branch = '0_17'
                sec_branch_2 = None  
        if merger_number == '2': #the third merger of the main galaxy
            if codetp == 'ENZO':
                prog_branch = '0'
                sec_branch = '0_0'
                sec_branch_2 = None
            elif codetp == 'AREPO':
                prog_branch = '0'
                sec_branch = '0_51'
                sec_branch_2 = None
            elif codetp == 'GADGET3':
                prog_branch = '0'
                sec_branch = '3'
                sec_branch_2 = None
            elif codetp == 'GADGET4':
                prog_branch = '0'
                sec_branch = '0_29'
                sec_branch_2 = None
            elif codetp == 'CHANGA':
                prog_branch = '0'
                sec_branch = '0_72'
                sec_branch_2 = None
            elif codetp == 'GIZMO':
                prog_branch = '0'
                sec_branch = '0_0'
                sec_branch_2 = None
            elif codetp == 'GEAR':
                prog_branch = '0'
                sec_branch = '0_40'
                sec_branch_2 = None
            elif codetp == 'ART':
                prog_branch = '0'
                sec_branch = '0_13'
                sec_branch_2 = None
            elif codetp == 'RAMSES':
                prog_branch = '0'
                sec_branch = '577_0'
                sec_branch_2 = None  
    return prog_branch, sec_branch, sec_branch_2


def load_halotree_and_pfs(codetp, halotree_ver, rawtree_skip = False):
    data_dir = '/work/hdd/bezm/tnguyen2/AGORA/%s/metadata' % codetp
    if rawtree_skip == False:
        rawtree = np.load('/work/hdd/bezm/gtg115x/Halo_Finding/%s/halotree_%s_final.npy' % (codetp, halotree_ver), allow_pickle=True).tolist()
    redshift_list = np.loadtxt('/work/hdd/bezm/gtg115x/Halo_Finding/%s/pfs_allsnaps_%s.txt' % (codetp, halotree_ver), dtype=str)[:,1].astype(float)
    time_list = np.loadtxt('/work/hdd/bezm/gtg115x/Halo_Finding/%s/pfs_allsnaps_%s.txt' % (codetp, halotree_ver), dtype=str)[:,2].astype(float)
    pfs = np.loadtxt('/work/hdd/bezm/gtg115x/Halo_Finding/%s/pfs_allsnaps_%s.txt' % (codetp, halotree_ver), dtype=str)[:,0]
    if codetp == 'CHANGA':
        step = 2
    elif codetp == 'GEAR':
        step = 3
    else:
        step = 1
    if rawtree_skip == False:
        return rawtree, redshift_list, time_list, pfs, step
    else:
        return redshift_list, time_list, pfs, step

def load_timings(codetp, halotree_ver, merger_number):
    if codetp == 'CHANGA':
        step = 2
    elif codetp == 'GEAR':
        step = 3
    else:
        step = 1
    #
    time_list = np.loadtxt('/work/hdd/bezm/gtg115x/Halo_Finding/%s/pfs_allsnaps_%s.txt' % (codetp, halotree_ver), dtype=str)[:,2].astype(float)
    dist_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/dist_relvel_j_ProgBranch-%s_FirstMerger_%s_ver2013.npy' % (merger_number, codetp), allow_pickle=True).tolist()
    idx_begin = dist_data['idx'][0]
    time_begin = dist_data['time'][0]
    #
    cs = CubicSpline(dist_data['time'], dist_data['dist'])
    time_spline = np.linspace(min(dist_data['time']), max(dist_data['time']), 1000)
    #
    time_1stpass = time_spline[np.where(np.diff(cs(time_spline))>0)[0][0]]
    idx_1stpass = np.argmin(abs(time_1stpass - time_list/1e3))
    if codetp == 'GEAR':
        idx_1stpass -= 1
    #
    time_maxdist = time_spline[time_spline > time_1stpass][np.argmax(cs(time_spline)[time_spline > time_1stpass])]
    if codetp == 'GEAR':
        idx_maxdist = np.argmin(abs(time_maxdist - time_list/1e3)) - (np.argmin(abs(time_maxdist - time_list/1e3)) % step) - 1
    else:
        idx_maxdist = np.argmin(abs(time_maxdist - time_list/1e3))
    #
    if halotree_ver == 2013: #this is calculated in 
        if codetp == 'ART':
            idx_cls = 178
        elif codetp == 'ENZO':
            idx_cls = 158
        elif codetp == 'RAMSES':
            idx_cls = 161
        elif codetp == 'CHANGA':
            idx_cls = 230 
        elif codetp == 'GADGET3':
            idx_cls = 145
        elif codetp == 'GADGET4':
            idx_cls = 126
        elif codetp == 'GEAR':
            idx_cls = 491
        elif codetp == 'AREPO':
            idx_cls = 147
        elif codetp == 'GIZMO':
            idx_cls = 164
    time_cls = time_list[idx_cls]/1e3
    #
    time_endinfall = (time_begin + time_maxdist)/2
    if codetp == 'GEAR':
        idx_endinfall = np.argmin(abs(time_endinfall - time_list/1e3)) - (np.argmin(abs(time_endinfall - time_list/1e3)) % step) - 1
    else:
        idx_endinfall = np.argmin(abs(time_endinfall - time_list/1e3)) - (np.argmin(abs(time_endinfall - time_list/1e3)) % step)
    #
    return idx_begin, idx_endinfall, idx_1stpass, idx_maxdist, idx_cls, time_begin, time_endinfall, time_1stpass, time_maxdist, time_cls


def load_tracking_dist_data(codetp, halotree_ver, merger_number):
    dist_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/dist_relvel_j_ProgBranch-%s_FirstMerger_%s_ver%s.npy' % (merger_number, codetp, halotree_ver), allow_pickle=True).tolist()
    return dist_data

def load_ds(codetp, idx, pfs):
    if codetp == 'AREPO' or codetp == 'GADGET3' or codetp == 'GADGET4' or codetp == 'AREPO-TNG':
        ds = yt.load(pfs[idx], unit_base = {"length": (1.0, "Mpccm/h")})
    else:
        ds = yt.load(pfs[idx])
    return ds

def get_r_oden(rawtree, branch, idx, oden):
    halo = rawtree[branch][idx]
    r_odenkey = 'r%s' % oden
    #this function return r200 value or find the closest value to it (in case the halo does not have 'r200' radius)
    if r_odenkey in halo.keys():
        r_oden = halo[r_odenkey]
    else:
        key_list = list(halo.keys())
        r_keys = np.array([x[1:] for x in key_list if x[0] =='r'])
        r_key = r_keys[abs(r_keys.astype(float)-oden)==abs(r_keys.astype(float)-oden).min()][0]
        r_oden = halo['r'+r_key]
    return r_oden

def get_haloradius(rawtree, branch, idx, halo_radius = False, printerror = True):
    if halo_radius == False:
        if (get_r_oden(rawtree, branch, idx, 200) < get_r_oden(rawtree, branch, idx, 250)):
            if printerror:
                print('r200 error at Idx %s and Branch %s' % (idx, branch))
            if ((rawtree[branch][idx]['Halo_Radius'] <= get_r_oden(rawtree, branch, idx, 150)) and (rawtree[branch][idx]['Halo_Radius'] >= get_r_oden(rawtree, branch, idx, 250))) and rawtree[branch][idx]['cden'] < 250 and rawtree[branch][idx]['cden'] > 150:
                if printerror:
                    print('Replaced by Halo_Radius')
                return rawtree[branch][idx]['Halo_Radius']
            elif get_r_oden(rawtree, branch, idx, 150) >= get_r_oden(rawtree, branch, idx, 250):
                if printerror:
                    print('Replaced by closest value to r150')
                return get_r_oden(rawtree, branch, idx, 150)
            elif get_r_oden(rawtree, branch, idx, 150) < get_r_oden(rawtree, branch, idx, 250):
                if printerror:
                    print('Replaced by closest value to r250')
                return get_r_oden(rawtree, branch, idx, 250)
        else:
            return get_r_oden(rawtree, branch, idx, 200)
    else:
        return rawtree[branch][idx]['Halo_Radius']

def infall_timestep_compute_spherical(rawtree, progenitor_branch, sec_branch, step, halo_radius = False, printerror = False):
    # This is the beginning of the merger, when the two halo radii start to overlap
    #NOTE: WE CALCULATED THE MERGER MASS RATIO ONE TIMESTEP BEFORE THE BEGINNING OF THE MERGER
    infall_index = max(min(extract_and_order_snapshotIdx(rawtree, progenitor_branch)),  min(extract_and_order_snapshotIdx(rawtree, sec_branch)))
    #Obtain the coordinate of the halos (from Kirk's merger tree)
    coor_merging = rawtree[sec_branch][infall_index]['Halo_Center']
    coor_prog = rawtree[progenitor_branch][infall_index]['Halo_Center']
    #
    #Calculate the distance between two halos
    dist = np.linalg.norm(coor_prog - coor_merging)
    #
    #Obtain the radius of the halos (from Kirk's merger tree)
    radius_merging = get_haloradius(rawtree, sec_branch, infall_index, halo_radius = halo_radius, printerror = printerror) 
    radius_prog = get_haloradius(rawtree, progenitor_branch, infall_index, halo_radius = halo_radius, printerror = printerror) 
    flag = 0
    while dist > (radius_merging + radius_prog):
    #while dist > (radius_prog):
        infall_index = infall_index + step
        #   
        #Sometimes, a halo is removed from the branch because it violates the constraints (coor, radius, or mass).
        #In this case, there is a gap in the branch (for example, Halo 1 then Halo 3 without Halo 2).
        #Thus, the code cannot read the merging_mass_index. In this happens, we skip the rest of the
        #while loop and continue to subtract 1 to the merging_mass_index
        if infall_index > max(extract_and_order_snapshotIdx(rawtree, sec_branch)):
            flag = 1
            break
        if infall_index not in extract_and_order_snapshotIdx(rawtree, sec_branch) or infall_index not in extract_and_order_snapshotIdx(rawtree, progenitor_branch):
            continue
        #
        coor_merging = rawtree[sec_branch][infall_index]['Halo_Center']
        coor_prog = rawtree[progenitor_branch][infall_index]['Halo_Center']
        #
        radius_merging = get_haloradius(rawtree, sec_branch, infall_index, halo_radius = halo_radius, printerror = printerror)
        radius_prog = get_haloradius(rawtree, progenitor_branch, infall_index, halo_radius = halo_radius, printerror = printerror)
        #
        dist = np.linalg.norm(coor_prog - coor_merging)
    if flag == 0:
        return infall_index
    else:
        return None
    
def infall_timestep_compute_hullv(hullv, prog_branch, sec_branch, step, infall_index_init):
    #Calculate the first timestep when two non-spherical halos overlap each other.
    #The infall_index_init should be set as the infall_timestep_compute_spherical - 5*Step
    def hullv_convert(hullv):
        #arrange the output to be branchFirst instead of snapFirst
        hullv_branchFirst = {}
        for idx in hullv.keys():
            for branch in hullv[idx].keys():
                if branch not in hullv_branchFirst.keys():
                    hullv_branchFirst[branch] = {}
                hullv_branchFirst[branch][idx] = hullv[idx][branch]
        return hullv_branchFirst
    #
    def normalize(v):
        n = np.linalg.norm(v)
        if n < 1e-12:
            return None
        return v / n
    #
    def project_points(points, axis):
        """Project 3D points onto a 3D axis and return min/max projections."""
        projections = points @ axis
        return projections.min(), projections.max()
    #
    def intervals_overlap(a_min, a_max, b_min, b_max):
        return not (a_max < b_min or b_max < a_min)
    #
    def get_face_normals(hull):
        normals = []
        for simplex in hull.simplices:
            pts = hull.points[simplex]
            # Compute face normal from triangle
            v1 = pts[1] - pts[0]
            v2 = pts[2] - pts[0]
            n = np.cross(v1, v2)
            n = normalize(n)
            if n is not None:
                normals.append(n)
        return normals
    #
    def get_edges(hull):
        edges = set()
        for simplex in hull.simplices:
            i, j, k = simplex
            edges.add(tuple(sorted((i, j))))
            edges.add(tuple(sorted((j, k))))
            edges.add(tuple(sorted((k, i))))
        # Convert to vectors
        edge_vectors = []
        for i, j in edges:
            v = hull.points[j] - hull.points[i]
            if np.linalg.norm(v) > 1e-12:
                edge_vectors.append(v)
        return edge_vectors
    #
    def convex_hulls_overlap_3d(verticesA, verticesB):
        hullA = ConvexHull(verticesA)
        hullB = ConvexHull(verticesB)
        #
        normalsA = get_face_normals(hullA)
        normalsB = get_face_normals(hullB)
        #
        edgesA = get_edges(hullA)
        edgesB = get_edges(hullB)
        #
        axes = []
        #
        # 1. Face normals
        axes.extend(normalsA)
        axes.extend(normalsB)
        #
        # 2. Cross products of edges
        for e1 in edgesA:
            for e2 in edgesB:
                axis = np.cross(e1, e2)
                axis = normalize(axis)
                if axis is not None:
                    axes.append(axis)
        #
        # SAT (Separating Axis Theorem) test: check all axes
        # SAT states that two convex shapes do NOT overlap if and only if there exists at least one axis on which their projections do not overlap.
        for axis in axes:
            amin, amax = project_points(verticesA, axis)
            bmin, bmax = project_points(verticesB, axis)
            #
            if not intervals_overlap(amin, amax, bmin, bmax):
                # Found separating axis → no overlap
                return False
        # No separating axis → they overlap
        return True
    #-------------------------------------------------------------
    hullv_branchFirst = hullv_convert(hullv)
    #
    #infall_index = max(min(extract_and_order_snapshotIdx(hullv_branchFirst, prog_branch)),  min(extract_and_order_snapshotIdx(hullv_branchFirst, sec_branch)))
    infall_index = infall_index_init
    hullv_pos_prog = hullv_branchFirst[prog_branch][infall_index]
    hullv_pos_sec = hullv_branchFirst[sec_branch][infall_index]
    while len(hullv_pos_prog) == 0 or len(hullv_pos_sec) == 0:
        if (infall_index - step) in hullv_branchFirst[prog_branch].keys() and (infall_index - step) in hullv_branchFirst[sec_branch].keys():
            infall_index -= step 
            hullv_pos_prog = hullv_branchFirst[prog_branch][infall_index]
            hullv_pos_sec = hullv_branchFirst[sec_branch][infall_index]
        else:
            break
    
    if len(hullv_pos_prog) == 0 or len(hullv_pos_sec) == 0:
        return None
    else:
        overlap_bool = convex_hulls_overlap_3d(hullv_pos_prog, hullv_pos_sec)
    #
    flag = 0
    if overlap_bool == True:
        while overlap_bool == True:
            #
            infall_index = infall_index - step
            #   
            #Sometimes, a halo is removed from the branch because it violates the constraints (coor, radius, or mass).
            #In this case, there is a gap in the branch (for example, Halo 1 then Halo 3 without Halo 2).
            #Thus, the code cannot read the merging_mass_index. In this happens, we skip the rest of the
            #while loop and continue to subtract 1 to the merging_mass_index
            if infall_index < min(extract_and_order_snapshotIdx(hullv_branchFirst, sec_branch)):
                flag = 1
                break
            if infall_index not in extract_and_order_snapshotIdx(hullv_branchFirst, sec_branch) or infall_index not in extract_and_order_snapshotIdx(hullv_branchFirst, prog_branch):
                continue
            #
            hullv_pos_prog = hullv_branchFirst[prog_branch][infall_index]
            hullv_pos_sec = hullv_branchFirst[sec_branch][infall_index]
            if len(hullv_pos_prog) == 0 or len(hullv_pos_sec) == 0:
                continue
            overlap_bool = convex_hulls_overlap_3d(hullv_pos_prog, hullv_pos_sec)
        #
        if flag == 0:
            return infall_index + step
        else:
            return None
    #
    elif overlap_bool == False:
        while overlap_bool == False:
            #
            infall_index = infall_index + step
            #   
            #Sometimes, a halo is removed from the branch because it violates the constraints (coor, radius, or mass).
            #In this case, there is a gap in the branch (for example, Halo 1 then Halo 3 without Halo 2).
            #Thus, the code cannot read the merging_mass_index. In this happens, we skip the rest of the
            #while loop and continue to subtract 1 to the merging_mass_index
            if infall_index > max(extract_and_order_snapshotIdx(hullv_branchFirst, sec_branch)):
                flag = 1
                break
            if infall_index not in extract_and_order_snapshotIdx(hullv_branchFirst, sec_branch) or infall_index not in extract_and_order_snapshotIdx(hullv_branchFirst, prog_branch):
                continue
            #
            hullv_pos_prog = hullv_branchFirst[prog_branch][infall_index]
            hullv_pos_sec = hullv_branchFirst[sec_branch][infall_index]
            if len(hullv_pos_prog) == 0 or len(hullv_pos_sec) == 0:
                continue
            overlap_bool = convex_hulls_overlap_3d(hullv_pos_prog, hullv_pos_sec)
        #
        if flag == 0:
            return infall_index
        else:
            return None
    
    
def get_all_sec_starIDs(assignment, sec_branch, step):
    #this function obtains the ID of all stars that even belong to the secondary galaxy 
    sec_allIDs = np.array([])
    for idx in range(0, max(list(assignment['ids'][sec_branch].keys())) + step, step):
        if idx in assignment['ids'][sec_branch].keys():
            metadata = np.load(metadata_dir + 'star_metadata_allbox_%s.npy' % idx, allow_pickle=True).tolist()
            ID_all = metadata['ID']
            overlap_ids = np.intersect1d(assignment['ids'][prog_branch][idx], assignment['ids'][sec_branch][idx])
            overlap_energies_prog = (assignment['energies'][prog_branch][idx])[np.intersect1d(overlap_ids, assignment['ids'][prog_branch][idx], return_indices=True)[2]]
            overlap_energies_sec = (assignment['energies'][sec_branch][idx])[np.intersect1d(overlap_ids, assignment['ids'][sec_branch][idx], return_indices=True)[2]]
            prog_ids_add = overlap_ids[overlap_energies_prog < overlap_energies_sec]
            sec_ids_add = overlap_ids[overlap_energies_sec < overlap_energies_prog]
            prog_ids_unique = np.setdiff1d(assignment['ids'][prog_branch][idx], assignment['ids'][sec_branch][idx])
            sec_ids_unique = np.setdiff1d(assignment['ids'][sec_branch][idx], assignment['ids'][prog_branch][idx])
            prog_ids_all = np.append(prog_ids_unique, prog_ids_add)
            sec_ids_all = np.append(sec_ids_unique, sec_ids_add)
            #
            sec_allIDs = np.unique(np.append(sec_allIDs, sec_ids_all))
    return sec_allIDs

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

    
#------------------------------------------------------------------------------------------------------------------------------------
#Adding yt fields

def add_metallicity_fields(pf, codetp):
    if codetp == 'GADGET3' or codetp == 'GADGET4' or codetp == 'CHANGA' or codetp == 'GIZMO' or codetp == 'GEAR':
        code_based = 'particle'
    elif codetp == 'ENZO' or codetp == 'AREPO' or codetp == 'ART' or codetp == 'RAMSES':
        code_based = 'cell'
    #
    if codetp == 'ART': 
        def art_gas_metallicity(field, data):
            return (data["gas", "metal_ii_density"] + data["gas", "metal_ia_density"]) / \
                    data["gas", "density"] / agora_Zsun
        pf.add_field(("gas", "agora_metallicity"), 
                     function=art_gas_metallicity, 
                     force_override=True, 
                     display_name="Metallicity", 
                     sampling_type = code_based,
                     take_log=True, 
                    #it is in solar metallicity units!
                    #however unyt does not automatically
                    #convert them so we say units = ''
                     units="")
    elif codetp == 'GADGET3' or codetp == 'GADGET4':
        def gadget_metallicity(field, data):
                return (data[(gas_name_dict[codetp], "Metallicity")]+YTArray(1e-4*agora_Zsun,''))/agora_Zsun
        pf.add_field(('gas', 'agora_metallicity'), 
                     function=gadget_metallicity, 
                     force_override=True,
                     take_log=True, 
                     sampling_type = code_based,
                     display_name="Metallicity", 
                    #it is in solar metallicity units!
                    #however unyt does not automatically
                    #convert them so we say units = ''                     
                     units="")
    elif codetp == 'GEAR':
        def gear_metallicity(field, data):
            if len(data[gas_name_dict[codetp], "metallicity"].shape) == 1:
                return data[gas_name_dict[codetp],"metallicity"].in_units("")/agora_Zsun
            else:
                return data[gas_name_dict[codetp], "metallicity"][:,9].in_units("")/agora_Zsun
            # in_units("") turned out to be crucial!; otherwise code_metallicity 
            #will be used and it will mess things up
        pf.add_field(('gas', 'agora_metallicity'), 
                     function=gear_metallicity,
                     force_override=True,
                     sampling_type = code_based,
                     display_name="Metallicity",
                     take_log=True, 
                    #it is in solar metallicity units!
                    #however unyt does not automatically
                    #convert them so we say units = ''                      
                     units="")
    elif codetp == 'ENZO':
        def enzo_metallicity(field, data):
            return data["gas", "metal_density"] / data["gas", "density"]/agora_Zsun
        pf.add_field(("gas", "agora_metallicity"), 
                     function=enzo_metallicity, 
                     force_override=True, 
                     display_name="Metallicity", 
                     sampling_type = 'cell',
                     take_log=True, 
                    #it is in solar metallicity units!
                    #however unyt does not automatically
                    #convert them so we say units = ''                      
                     units="")
    else:
        def gas_metallicity(field, data):
            return data['gas',"metallicity"].in_units('')/agora_Zsun
        pf.add_field(('gas','agora_metallicity'),
                    sampling_type = code_based,
                    function = gas_metallicity,
                    force_override = True,
                    display_name="Metallicity", 
                    #it is in solar metallicity units!
                    #however unyt does not automatically
                    #convert them so we say units = ''  
                     units = '')
    def primordial_gas_metallicity(field, data):
        return data['gas',"agora_metallicity"]*0.0+1e-4
    pf.add_field(('gas','agora_primordial_metallicity'),
                sampling_type = code_based,
                function = primordial_gas_metallicity,
                force_override = True,
                display_name="Primordial Metallicity", 
                #it is in solar metallicity units!
                #however unyt does not automatically
                #convert them so we say units = ''  
                 units = '')

def add_radialdist_to_halocenter_field(ds, gal_com):
    def _radialdist_to_halocenter(field, data):
        x = data[('gas','x')] - gal_com[0]*ds.units.code_length
        y = data[('gas','y')] - gal_com[1]*ds.units.code_length
        z = data[('gas','z')] - gal_com[2]*ds.units.code_length
        r = np.sqrt(x**2 + y**2 + z**2)
        return r
    #
    ds.add_field(
        ("gas", "radii"),
        function=_radialdist_to_halocenter,
        sampling_type="local",
        units="kpc",
        force_override = True
    )

def add_radialvel_to_halocenter_field(ds, gal_com, com_gasvel):
    def _radialvel_to_halocenter(field, data):
        x = data[('gas','x')] - gal_com[0]*ds.units.code_length
        y = data[('gas','y')] - gal_com[1]*ds.units.code_length
        z = data[('gas','z')] - gal_com[2]*ds.units.code_length
        r = np.sqrt(x**2 + y**2 + z**2)
        vx = data[('gas','velocity_x')] - com_gasvel[0]*ds.units.km/ds.units.s
        vy = data[('gas','velocity_y')] - com_gasvel[1]*ds.units.km/ds.units.s
        vz = data[('gas','velocity_z')] - com_gasvel[2]*ds.units.km/ds.units.s
        return (vx*x + vy*y + vz*z)/r
    #
    ds.add_field(
        ('gas', "radial_vel"),
        function=_radialvel_to_halocenter,
        sampling_type="local",
        units="km/s",
        force_override = True
    )

    
def add_cooling_fields(ds, codetp, idx, redshift_list, gal_com):
    from astropy.constants import k_B
    k_B = k_B.to("g*cm**2*s**-2*K**-1").value
    from scipy.interpolate import RegularGridInterpolator as regInt
    import h5py
    #
    def __H_numden(field, data):
        return data['gas','number_density']*0.92 #primordial hydrogen mass fraction is 0.752, which translates to the number density fraction of 0.92
    #
    ds.add_field(
        name = ('gas', "H_numden"),
        function = lambda field, data : __H_numden(field, data),
        sampling_type = "local",
        force_override=True,
        units = "1/cm**3")
    #-----------------------------------------------------------------------------------
    def coolingInterp(rateType):
        redsh=np.array([0.0,0.12202,0.25893,0.41254,0.58489,0.77828,0.99526,1.2387,1.5119,1.8184,
            2.1623,2.5481,2.9811,3.4668,4.0119,4.6234,5.3096,6.0795,6.9433,7.9125,
            9.0000,10.22,11.589,13.125,14.849,100.00])
        #
        temp=np.arange(1,9.05,0.05)
        dens=np.arange(-10,4.5,0.5)
        #
        if codetp == "ART":
            if "Primordial" in rateType:
                f = np.loadtxt("/work/hdd/bezm/tnguyen2/AGORA/clcool_primordial.dat", skiprows=3)
                f = np.reshape(f[:,4],(26,29,161))
                interp = regInt((dens, redsh, temp), np.transpose(f,axes=[1,0,2]), method="nearest")
            elif "Metals" in rateType:
                f = np.loadtxt("/work/hdd/bezm/tnguyen2/AGORA/clcool_Zsun.dat", skiprows=3)
                f = np.reshape(f[:,4],(26,29,161))
                interp = regInt((dens, redsh, temp), np.transpose(f,axes=[1,0,2]), method="nearest")
        else:
            f=h5py.File('/work/hdd/bezm/tnguyen2/AGORA/CloudyData_UVB=HM2012_shielded.h5','r') #Density (i), Redshift (j), Temperature (k)
            interp = regInt((dens, redsh, temp), f[f'CoolingRates/{rateType}'][:][:][:], method="nearest")
            f.close()
        return interp
    #-----------------------------------------------------------------------------------
    def denszT(data, codetp, idx):
        #data is the simulated region 
        if codetp == 'CHANGA':
            cTemp = data['Gas', gas_temp_dict[codetp]].value #*
        else:
            cTemp = data['gas', gas_temp_dict[codetp]].value #*
        cDens = data['gas', 'H_numden'].to('cm**-3').value #*
        #cDens = data['gas', 'number_density'].value #*
        #print(f"Dens: {np.max(np.log10(cDens))}, {np.min(cDens)} | Temp: {np.max(np.log10(cTemp))}, {np.min(cTemp)}")
        czNum = redshift_list[idx] #originally it's self.cz, I guessed that it is redshift, but not entirely sure
        if czNum < 0:
            czNum = 0
        cz = np.full_like(cDens, czNum)
        if codetp == 'ART' or codetp == 'ENZO' or codetp == 'RAMSES':
            cTemp = cTemp.ravel()
            cDens = cDens.ravel()
            cz = cz.ravel()
        #BIG SOURCE OF ERROR PERHAPS? MUST EXIST FOR GEAR AND ENZO, BUT WHYYYY???
        cTemp[cTemp < 11] = 11 
        cTemp[cTemp > 1e9] = 1e9
        cDens[cDens < 1e-10] = 1e-10
        cDens[cDens > 1e4] = 1e4
        return ((np.vstack((np.log10(cDens), cz, np.log10(cTemp)))).T).tolist()
    #-----------------------------------------------------------------------------------
    def __MMW(field, data):
        #
        if codetp == 'ART':
            return data['gas','mean_molecular_weight']
        else:
            denszTvals = denszT(data, codetp, idx)
            return coolingInterp("Primordial/MMW")(denszTvals)
    #
    ds.add_field(
        name = ('gas', "MMW"),
        function = lambda field, data : __MMW(field, data),
        sampling_type = "local",
        units = "",
        force_override=True
    )
    #-----------------------------------------------------------------------------------
    def __gasPressure(field, data):
        if codetp == 'CHANGA':
            return (data['gas', 'density'].in_units("g/cm**3")* #*
                    yt.YTArray(k_B,"g*cm**2/s**2/K")*yt.YTArray(data['Gas', gas_temp_dict[codetp]].value,"K"))/(data['gas', "MMW"]*yt.YTArray(1.67356e-24, "g")) #*
        elif codetp == 'ENZO' or codetp == 'RAMSES':
            return (np.ravel(data['gas', 'density'].in_units("g/cm**3"))* yt.YTArray(k_B,"g*cm**2/s**2/K")*np.ravel(data['gas', gas_temp_dict[codetp]]))/(data['gas', "MMW"]*yt.YTArray(1.67356e-24, "g")) #*
        else:
            return (data['gas', 'density'].in_units("g/cm**3")* yt.YTArray(k_B,"g*cm**2/s**2/K")*yt.YTArray(data['gas', gas_temp_dict[codetp]].value,"K"))/(data['gas', "MMW"]*yt.YTArray(1.67356e-24, "g")) #*
    #
    ds.add_field(
        name = ('gas', "gPressure"),
        function = lambda field, data : __gasPressure(field, data),
        sampling_type = "local",
        units = "g/cm/s**2",
        force_override=True)
    #-----------------------------------------------------------------------------------
    def __coolingRate(field, data):
        if codetp == 'ART':
            return yt.YTArray(coolingInterp("Primordial/Cooling")(denszT(data, codetp, idx)) + 
                    coolingInterp("Metals/Cooling")(denszT(data, codetp, idx))*np.ravel(data['gas', 'metal_mass_fraction']), "g*cm**5/s**3") #* #ART has an issue when loading agora_metallicity, so switch to metal_mass_fraction here
        elif codetp == 'ENZO' or codetp == 'RAMSES':
            return yt.YTArray(coolingInterp("Primordial/Cooling")(denszT(data, codetp, idx)) + 
                    coolingInterp("Metals/Cooling")(denszT(data, codetp, idx))*np.ravel(data['gas', 'agora_metallicity'])*agora_Zsun, "g*cm**5/s**3") #* #ART has an issue when loading agora_metallicity, so switch to metal_mass_fraction here
        else:
            return yt.YTArray(coolingInterp("Primordial/Cooling")(denszT(data, codetp, idx)) + 
                    coolingInterp("Metals/Cooling")(denszT(data, codetp, idx))*data['gas', 'agora_metallicity']*agora_Zsun, "g*cm**5/s**3") #*
    #
    ds.add_field(
        name = ('gas', "CoolingRate"),
        function = lambda field, data : __coolingRate(field, data),
        sampling_type = "local",
        units = "g*cm**5/s**3",
        force_override=True)
    #-----------------------------------------------------------------------------------
    def __coolingTime(field, data):
        return (3/2)*np.ravel(data['gas', "gPressure"])/((np.ravel(data['gas', 'H_numden'])**2) * data['gas', "CoolingRate"]) #*
    #
    ds.add_field(
        name = ('gas', "tCool"),
        function = lambda field, data : __coolingTime(field, data),
        sampling_type = "local",
        force_override=True,
        units = "s")
    #-----------------------------------------------------------------------------------
    def __ffTime(field, data):
        radData = data['gas', "radii"].in_units("kpc").value #*
        if codetp == 'GADGET3' or codetp == 'GADGET4' or codetp == 'AREPO' or codetp == 'GEAR' or codetp == 'GIZMO' or codetp == 'CHANGA':
            all_massData = data['all', 'particle_mass'].in_units("Msun").value
            all_posData = data['all', 'particle_position'].in_units("code_length").value
            all_radData = np.linalg.norm(all_posData - gal_com, axis=1)
            all_radData = (all_radData*ds.units.code_length).to('kpc').v
        else:
            particle_massData = data['all', 'particle_mass'].in_units("Msun").value
            particle_posData = data['all', 'particle_position'].in_units("code_length").value
            particle_radData = np.linalg.norm(particle_posData - gal_com, axis=1)
            particle_radData = (particle_radData*ds.units.code_length).to('kpc').v
            massData = data['gas', 'mass'].in_units("Msun").value #*
            all_massData = np.append(particle_massData, massData)
            all_radData = np.append(particle_radData, radData)
            
        gravConst = 4.51710305e-30 #in pc^3/(Msun*s^2)

        # Sort all particles by radius once
        sort_idx     = np.argsort(all_radData)
        sorted_rad   = all_radData[sort_idx]
        sorted_mass  = all_massData[sort_idx]
        cumul_mass   = np.cumsum(sorted_mass)
    
        # For each gas cell, find the enclosed mass using searchsorted (exact, no bins)
        idx        = np.searchsorted(sorted_rad, radData)
        idx[idx == len(cumul_mass)] = len(cumul_mass) - 1 #numerical float round-up can make element of radData > sorted_rad 
        gravM      = cumul_mass[idx]
    
        # Mean density inside sphere of radius r (convert kpc -> pc: *1000)
        r_pc       = radData * 1000.0          # kpc → pc
        densM      = gravM / ((4/3) * np.pi * r_pc**3)
    
        # Free-fall time
        tffA       = np.sqrt((3.0 * np.pi) / (32.0 * gravConst * densM))

        return yt.YTArray(tffA, "s")
    #
    ds.add_field(
        name = ('gas', "tFF"),
        function = lambda field, data : __ffTime(field, data),
        sampling_type = "local",
        force_override=True,
        units = "s")

    #-----------------------------------------------------------------------------------
    def __coolOFF(field, data):
        #result = data['gas', "tCool"]/np.ravel(data['gas', "tFF"])
        #result[result == 0] = 1e-99
        return data['gas', "tCool"]/np.ravel(data['gas', "tFF"])
    #
    ds.add_field(
        name = ('gas', "coolOFF"),
        function = lambda field, data : __coolOFF(field, data),
        sampling_type = "local",
        force_override=True,
        units = "") 
    

    