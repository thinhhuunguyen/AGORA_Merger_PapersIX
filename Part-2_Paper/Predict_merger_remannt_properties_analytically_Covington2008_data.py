import numpy as np
import yt
yt.set_log_level(0)
import os, sys
from yt.data_objects.particle_filters import add_particle_filter
if int((yt.__version__).split('.')[0]) >= 4 and int((yt.__version__).split('.')[1]) >= 2: #ParticleUnion is only available in yt 4.2 and later
    from yt.data_objects.unions import ParticleUnion
else:
    from yt.data_objects.unions import Union as ParticleUnion
from scipy.integrate import odeint
from astropy.constants import G

import setup
from importlib import reload
reload(setup)

from setup import load_halotree_and_pfs, load_timings, load_ds, sec_branch_compute
from setup import gas_name_dict, star_name_dict, dm_name_dict


#---------------------------------------------------------------------------------------------------------------------------------------------

def compute_half_mass_radius(mass, pos, center):
    radius = np.linalg.norm(pos - center, axis=1)
    radius_sort = np.sort(radius)
    mass_sort = mass[np.argsort(radius)]
    mass_cumsum_percentile = np.cumsum(mass_sort)/np.sum(mass)
    return radius_sort[np.argmin(abs(mass_cumsum_percentile - 0.5))]

def weighted_std(values, weights):
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return np.sqrt(variance)

def compute_1d_veldispersion(velocities, masses, num_projections=50):
    """
    Calculates the average mass-weighted velocity dispersion from 
    50 random line-of-sight projections.
    
    Parameters:
    velocities (np.array): (N, 3) array of stellar velocities
    masses (np.array): (N,) array of stellar masses
    num_projections (int): Number of random LOS angles to average over
    
    Returns:
    float: The mean mass-weighted velocity dispersion
    """
    dispersions = []
    
    for _ in range(num_projections):
        # 1. Generate a random unit vector (LOS direction)
        # Using a normal distribution ensures isotropic distribution on the sphere
        vec = np.random.rand(3) - np.random.rand(3)
        unit_vector = vec / np.linalg.norm(vec)
        
        # 2. Project 3D velocities onto this line-of-sight
        # Result is a 1D array of radial velocities
        v_los = np.dot(velocities, unit_vector)
        
        # 3. Calculate mass-weighted dispersion using your function
        sigma_weighted = weighted_std(v_los, masses)
        dispersions.append(sigma_weighted)
    
    # Return the average of all sampled projections
    return np.mean(dispersions), np.std(dispersions)
    
#---------------------------------------------------------------------------------------------------------------------------------------------
codetp = sys.argv[1]
merger_number = '0'
halotree_ver = 2013

if codetp == 'GEAR':
    step = 3
elif codetp == 'CHANGA':
    step = 2
else:
    step = 1

rawtree, redshift_list, time_list, pfs, step = load_halotree_and_pfs(codetp, halotree_ver, rawtree_skip=False)
dist_data = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/dist_relvel_j_ProgBranch-%s_FirstMerger_%s_ver2013.npy' % (merger_number, codetp), allow_pickle=True).tolist()
prog_branch, sec_branch, sec_branch_2 = sec_branch_compute(codetp, merger_number)
idx_begin, idx_endinfall, idx_1stpass, idx_maxdist, idx_cls, time_begin, time_endinfall, time_1stpass, time_maxdist, time_cls = load_timings(codetp, halotree_ver, merger_number)

time_eval = time_1stpass + 0.6
if codetp == 'GEAR':
    idx_eval = np.argmin(abs(time_eval - time_list/1e3)) - (np.argmin(abs(time_eval - time_list/1e3)) % step) - 1
else:
    idx_eval = np.argmin(abs(time_eval - time_list/1e3)) - (np.argmin(abs(time_eval - time_list/1e3)) % step)

metadata_dir = '/work/hdd/bezm/tnguyen2/AGORA/%s/metadata/' % codetp
assignment = np.load('/work/hdd/bezm/gtg115x/Halo_Finding/%s/star_id_%s_final_firstmerger.npy' % (codetp, halotree_ver), allow_pickle=True).tolist()

#---------------------------------------------------------------------------------------------------------------------------------------------
#Load the galaxies pre-merger
idx = idx_begin - step
ds = load_ds(codetp, idx, pfs)

center1, center2, _ = np.load('/work/hdd/bezm/tnguyen2/AGORA/analysis/premerger_centers_%s_ver2013.npy' % (codetp), allow_pickle=True).tolist().values()

if codetp == 'GADGET3' or codetp == 'GADGET4':
    dm = ParticleUnion("DarkMatter",["PartType5","PartType1"])
    ds.add_particle_union(dm)
elif codetp == 'AREPO' or codetp == 'GIZMO':
    dm = ParticleUnion("DarkMatter",["PartType2","PartType1"])
    ds.add_particle_union(dm)
elif codetp == 'GEAR':
    dm = ParticleUnion("DarkMatter",["PartType5","PartType2"])
    ds.add_particle_union(dm)
elif codetp == 'ENZO':
    def darkmatter_init(pfilter, data):
        filter_darkmatter0 = np.logical_or(data["all", "particle_type"] == 1, data["all", "particle_type"] == 4)
        filter_darkmatter = np.logical_and(filter_darkmatter0,data['all', 'particle_mass'].to('Msun') > 1)
        return filter_darkmatter
    add_particle_filter("DarkMatter",function=darkmatter_init,filtered_type='all',requires=["particle_type","particle_mass"])
    ds.add_particle_filter("DarkMatter")
    #
    def stars(pfilter, data):
        filter_stars = np.logical_and(data[pfilter.filtered_type, "particle_type"] == 2, data[pfilter.filtered_type, "particle_mass"].to('Msun') > 1)
        return filter_stars
    add_particle_filter("stars", function=stars, filtered_type="all", requires=["particle_type","particle_mass"])
    ds.add_particle_filter("stars")

#Load the simulation region
if codetp == 'ART' or codetp == 'ENZO' or codetp == 'RAMSES':
    prog_reg = ds.sphere(center1, (rawtree[prog_branch][idx]['cden_rad']))
    sec_reg = ds.sphere(center2, (rawtree[sec_branch][idx]['cden_rad']))
else:
    prog_reg = ds.all_data()
    sec_reg = ds.all_data()

#%%%%%%%%%%%%%%%%%%%%%%%%%
#Total mass of the galaxies
if codetp == 'ENZO' or codetp == 'ART' or codetp == 'RAMSES':
    m1tot = prog_reg['all','particle_mass'].to('Msun').sum().v.tolist()
    m2tot = sec_reg['all','particle_mass'].to('Msun').sum().v.tolist()
    m1tot = m1tot + prog_reg[gas_name_dict[codetp],'mass'].to('Msun').sum().v.tolist()
    m2tot = m2tot + sec_reg[gas_name_dict[codetp],'mass'].to('Msun').sum().v.tolist()
else:
    all1_mass_each = prog_reg['all','particle_mass'].to('Msun').v
    all1_pos_each = prog_reg['all', 'particle_position'].to('code_length').v
    all1_vel_each = prog_reg['all', 'particle_velocity'].to('kpc/Gyr').v
    all1_d_each = np.linalg.norm(all1_pos_each - center1, axis=1)
    halo1_mass_each = all1_mass_each[all1_d_each <= rawtree[prog_branch][idx]['cden_rad']]
    halo1_pos_each = all1_pos_each[all1_d_each <= rawtree[prog_branch][idx]['cden_rad']]
    halo1_vel_each = all1_vel_each[all1_d_each <= rawtree[prog_branch][idx]['cden_rad']]
    m1tot = halo1_mass_each.sum()
    #
    all2_mass_each = sec_reg['all','particle_mass'].to('Msun').v
    all2_pos_each = sec_reg['all', 'particle_position'].to('code_length').v
    all2_vel_each = sec_reg['all', 'particle_velocity'].to('kpc/Gyr').v
    all2_d_each = np.linalg.norm(all2_pos_each - center2, axis=1)
    halo2_mass_each = all2_mass_each[all2_d_each <= rawtree[sec_branch][idx]['cden_rad']]
    halo2_pos_each = all2_pos_each[all2_d_each <= rawtree[sec_branch][idx]['cden_rad']]
    halo2_vel_each = all2_vel_each[all2_d_each <= rawtree[sec_branch][idx]['cden_rad']]
    m2tot = halo2_mass_each.sum()

#%%%%%%%%%%%%%%%%%%%%%%%%%
#Total half-mass radius of the galaxies
if codetp == 'ENZO' or codetp == 'ART' or codetp == 'RAMSES':
    prog_tot_mass = np.append(prog_reg['all','particle_mass'].to('Msun').v, prog_reg[gas_name_dict[codetp],'cell_mass'].to('Msun').v)
    prog_gas_pos = np.vstack((prog_reg[gas_name_dict[codetp],'x'].to('code_length').v, prog_reg[gas_name_dict[codetp],'y'].to('code_length').v, prog_reg[gas_name_dict[codetp],'z'].to('code_length').v)).T
    prog_tot_pos = np.vstack((prog_reg['all','particle_position'].to('code_length').v, prog_gas_pos))
    r1tot = compute_half_mass_radius(prog_tot_mass, prog_tot_pos, center1)
    sec_tot_mass = np.append(sec_reg['all','particle_mass'].to('Msun').v, sec_reg[gas_name_dict[codetp],'cell_mass'].to('Msun').v)
    sec_gas_pos = np.vstack((sec_reg[gas_name_dict[codetp],'x'].to('code_length').v, sec_reg[gas_name_dict[codetp],'y'].to('code_length').v, sec_reg[gas_name_dict[codetp],'z'].to('code_length').v)).T
    sec_tot_pos = np.vstack((sec_reg['all','particle_position'].to('code_length').v, sec_gas_pos))
    r2tot = compute_half_mass_radius(sec_tot_mass, sec_tot_pos, center2)
else:
    r1tot = compute_half_mass_radius(halo1_mass_each, halo1_pos_each, center1)
    r2tot = compute_half_mass_radius(halo2_mass_each, halo2_pos_each, center2)
r1tot = (r1tot*ds.units.code_length).to('kpc').v.tolist()
r2tot = (r2tot*ds.units.code_length).to('kpc').v.tolist()

#%%%%%%%%%%%%%%%%%%%%%%%%%
#Stellar mass of the galaxies
allstars = np.load(metadata_dir + '/' + 'star_metadata_allbox_%s.npy' % idx, allow_pickle=True).tolist()
allmass = allstars['mass']
allpos = allstars['pos']
allID = allstars['ID'].astype(int)

m1_each = allmass[np.intersect1d(assignment['ids'][prog_branch][idx], allID, return_indices=True)[2]]
pos1_each = allpos[np.intersect1d(assignment['ids'][prog_branch][idx], allID, return_indices=True)[2]]
bool1_each = np.linalg.norm(pos1_each - center1, axis=1) <= rawtree[prog_branch][idx]['cden_rad'] #restrict to the halo
m1 = m1_each[ bool1_each ].sum()
m2_each = allmass[np.intersect1d(assignment['ids'][sec_branch][idx], allID, return_indices=True)[2]]
pos2_each = allpos[np.intersect1d(assignment['ids'][sec_branch][idx], allID, return_indices=True)[2]]
bool2_each = np.linalg.norm(pos2_each - center2, axis=1) <= rawtree[sec_branch][idx]['cden_rad'] #restrict to the halo
m2 = m2_each[ bool2_each ].sum()

#%%%%%%%%%%%%%%%%%%%%%%%%%
#Stellar half-mass radius of the galaxies
r1 = compute_half_mass_radius(m1_each[bool1_each], pos1_each[bool1_each], center1) #restrict to the halo
r2 = compute_half_mass_radius(m2_each[bool2_each], pos2_each[bool2_each], center2) #restrict to the halo

#%%%%%%%%%%%%%%%%%%%%%%%%%
#Gas mass fraction of the galaxies
if codetp == 'ENZO' or codetp == 'ART' or codetp == 'RAMSES':
    fg1 = prog_reg[gas_name_dict[codetp],'cell_mass'].to('Msun').sum().v/m1tot
    fg2 = sec_reg[gas_name_dict[codetp],'cell_mass'].to('Msun').sum().v/m2tot
else:
    gas1_mass_each = prog_reg[gas_name_dict[codetp],'particle_mass'].to('Msun').v
    gas1_pos_each = prog_reg[gas_name_dict[codetp], 'particle_position'].to('code_length').v
    gas1_d_each = np.linalg.norm(gas1_pos_each - center1, axis=1)
    #
    gas2_mass_each = sec_reg[gas_name_dict[codetp],'particle_mass'].to('Msun').v
    gas2_pos_each = sec_reg[gas_name_dict[codetp], 'particle_position'].to('code_length').v
    gas2_d_each = np.linalg.norm(gas2_pos_each - center2, axis=1)
    #    
    fg1 = gas1_mass_each[gas1_d_each <= rawtree[prog_branch][idx]['cden_rad']].sum()/m1tot
    fg2 = gas2_mass_each[gas2_d_each <= rawtree[sec_branch][idx]['cden_rad']].sum()/m2tot

#%%%%%%%%%%%%%%%%%%%%%%%%%
#Dark matter mass inside 1/2 stellar half mass radius
if codetp == 'ART' or codetp == 'ENZO' or codetp == 'RAMSES':
    reg_dm1 = ds.sphere(center1, (0.5*r1, 'code_length'))
    reg_dm2 = ds.sphere(center2, (0.5*r2, 'code_length'))
    mdm1 = reg_dm1[dm_name_dict[codetp], 'particle_mass'].to('Msun').sum().v.tolist()
    mdm2 = reg_dm2[dm_name_dict[codetp], 'particle_mass'].to('Msun').sum().v.tolist()
else:
    dm1_mass_each = prog_reg[dm_name_dict[codetp],'particle_mass'].to('Msun').v
    dm1_pos_each = prog_reg[dm_name_dict[codetp], 'particle_position'].to('code_length').v
    dm1_d_each = np.linalg.norm(dm1_pos_each - center1, axis=1)
    #
    dm2_mass_each = sec_reg[dm_name_dict[codetp],'particle_mass'].to('Msun').v
    dm2_pos_each = sec_reg[dm_name_dict[codetp], 'particle_position'].to('code_length').v
    dm2_d_each = np.linalg.norm(dm2_pos_each - center2, axis=1)
    #
    mdm1 = dm1_mass_each[dm1_d_each <= 0.5*r1].sum()
    mdm2 = dm2_mass_each[dm2_d_each <= 0.5*r2].sum()

#%%%%%%%%%%%%%%%%%%%%%%%%%
#Total internal kinetic energy 
velcom1 = rawtree[prog_branch][idx]['Vel_Com']*ds.units.code_length/ds.units.s
velcom1 = velcom1.to('kpc/Gyr').v
velcom2 = rawtree[sec_branch][idx]['Vel_Com']*ds.units.code_length/ds.units.s
velcom2 = velcom2.to('kpc/Gyr').v

if codetp == 'ART' or codetp == 'ENZO' or codetp == 'RAMSES':
    Ktot1 = 0.5*prog_reg['all','particle_mass'].to('Msun').v*np.linalg.norm(prog_reg['all','particle_velocity'].to('kpc/Gyr').v - velcom1, axis=1)**2
    Ktot1 = np.sum(Ktot1)
    gas_vx_1 = prog_reg[gas_name_dict[codetp],'velocity_x'].to('kpc/Gyr').v - velcom1[0]
    gas_vy_1 = prog_reg[gas_name_dict[codetp],'velocity_y'].to('kpc/Gyr').v - velcom1[1]
    gas_vz_1 = prog_reg[gas_name_dict[codetp],'velocity_z'].to('kpc/Gyr').v - velcom1[2]
    gas_mass_1 = prog_reg[gas_name_dict[codetp],'cell_mass'].to('Msun').v
    Kgas1 = 0.5*gas_mass_1*(gas_vx_1**2 + gas_vy_1**2 + gas_vz_1**2)
    Ktot1 += np.sum(Kgas1)
    #
    Ktot2 = 0.5*sec_reg['all','particle_mass'].to('Msun').v*np.linalg.norm(sec_reg['all','particle_velocity'].to('kpc/Gyr').v - velcom2, axis=1)**2
    Ktot2 = np.sum(Ktot2)
    gas_vx_2 = sec_reg[gas_name_dict[codetp],'velocity_x'].to('kpc/Gyr').v - velcom2[0]
    gas_vy_2 = sec_reg[gas_name_dict[codetp],'velocity_y'].to('kpc/Gyr').v - velcom2[1]
    gas_vz_2 = sec_reg[gas_name_dict[codetp],'velocity_z'].to('kpc/Gyr').v - velcom2[2]
    gas_mass_2 = sec_reg[gas_name_dict[codetp],'cell_mass'].to('Msun').v
    Kgas2 = 0.5*gas_mass_2*(gas_vx_2**2 + gas_vy_2**2 + gas_vz_2**2)
    Ktot2 += np.sum(Kgas2)
else:
    Ktot1 = 0.5*halo1_mass_each*np.linalg.norm(halo1_vel_each - velcom1, axis=1)**2
    Ktot1 = np.sum(Ktot1)
    Ktot2 = 0.5*halo2_mass_each*np.linalg.norm(halo2_vel_each - velcom2, axis=1)**2
    Ktot2 = np.sum(Ktot2)


#%%%%%%%%%%%%%%%%%%%%%%%%%
#Theoretical pericentric center and pericentric velocity (assuming point mass, 2 body problem)
idx_inside = idx
velrel_init = (rawtree[sec_branch][idx_inside]['Vel_Com'] - rawtree[prog_branch][idx_inside]['Vel_Com'])*ds.units.code_length/ds.units.s
velrel_init = velrel_init.to('kpc/Gyr').v
distrel_init = rawtree[sec_branch][idx_inside]['Halo_Center'] - rawtree[prog_branch][idx_inside]['Halo_Center']
distrel_init = (distrel_init*ds.units.code_length).to('kpc').v

def model_2BP(state, t):
    mu = G.to('kpc**3/Msun*Gyr**2').value*((m1tot*ds.units.Msun).to('Msun').v + (m2tot*ds.units.Msun).to('Msun').v)  # Earth's gravitational parameter  
                          # [km^3/s^2]
    x = state[0]
    y = state[1]
    z = state[2]
    x_dot = state[3]
    y_dot = state[4]
    z_dot = state[5]
    x_ddot = -mu * x / (x ** 2 + y ** 2 + z ** 2) ** (3 / 2)
    y_ddot = -mu * y / (x ** 2 + y ** 2 + z ** 2) ** (3 / 2)
    z_ddot = -mu * z / (x ** 2 + y ** 2 + z ** 2) ** (3 / 2)
    dstate_dt = [x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot]
    return dstate_dt

# Initial Conditions
X_0 = distrel_init[0]  # [kpc]
Y_0 = distrel_init[1]  # [kpc]
Z_0 = distrel_init[2]  # [kpc]
VX_0 = velrel_init[0]  # [kpc/Gyr]
VY_0 = velrel_init[1]  # [kpc/Gyr]
VZ_0 = velrel_init[2]  # [kpc/Gyr]
state_0 = [X_0, Y_0, Z_0, VX_0, VY_0, VZ_0]
# Time Array
t = np.linspace(0, 0.3, 5000) 
# Solving ODE
sol = odeint(model_2BP, state_0, t, rtol=1e-12, atol=1e-200)
X_sol = sol[:, 0]  # kpc
Y_sol = sol[:, 1]  # kpc
Z_sol = sol[:, 2]  # kpc
VX_sol = sol[:, 3]  # kpc/Gyr
VY_sol = sol[:, 4]  # kpc/Gyr
VZ_sol = sol[:, 5]  # kpc/Gyr
R_sol = np.linalg.norm(np.array([X_sol, Y_sol, Z_sol]).T, axis=1)
V_sol = np.linalg.norm(np.array([VX_sol, VY_sol, VZ_sol]).T, axis=1)
r_peri = np.min(R_sol) #in kpc
v_peri = V_sol[np.argmin(R_sol)] #in kpc/Gyr

i_peri = np.argmin(R_sol)
if i_peri == 0:
    raise ValueError("No interior pericenter in the integration window")


#%%%%%%%%%%%%%%%%%%%%%%%%%
#Set fitting parameters
A = 1.6
B = 1.0
C = 0.006
Cnew = 0.3
Cinit = 0.5 #default is 0.5
#Crad = 0.7 #default is 1
Crad_list = np.linspace(0.01, 2.5, 300)
Csig = 0.3
Cstars = 0.35
Cvir = 0.3

G_u = G.to('kpc**3/(Msun*Gyr**2)').value

#Calculate Impulse
delta_E1 = A*(G_u**2)*(m1tot**2)*m2tot/((v_peri**2)*(r_peri**2 + B*r1tot*r_peri + C*r1tot**2))
delta_E2 = A*(G_u**2)*(m2tot**2)*m1tot/((v_peri**2)*(r_peri**2 + B*r2tot*r_peri + C*r2tot**2))
#Dissipative strength
fk1 = delta_E1/Ktot1
fk2 = delta_E2/Ktot2
#Mass of new stars
mnew1 = Cnew*m1tot*fg1*fk1
mnew2 = Cnew*m2tot*fg2*fk2
#The final dark matter fraction
fdm_f = (mdm1 + mdm2)/(mdm1 + mdm2 + Cstars*(m1 + m2 + mnew1 + mnew2))
#The initial internal energies
r1 = (r1*ds.units.code_length).to('kpc').v.tolist()
r2 = (r2*ds.units.code_length).to('kpc').v.tolist()
Eint_i = -Cinit*G_u*( ((m1 + mnew1)**2/r1) + ((m2 + mnew2)**2/r2) )
#The final stellar mass of the merger remnant
mf = m1 + mnew1 + m2 + mnew2

#Calculate the relative velocity in the pair's center-of-mass frame
mw1, mw2 = m1+mnew1, m2+mnew2
vcom_pair = (mw1*velcom1 + mw2*velcom2)/(mw1+mw2)
v1, v2 = velcom1 - vcom_pair, velcom2 - vcom_pair

Rsep = np.linalg.norm(distrel_init)
Eorb = (-G_u*mw1*mw2/Rsep) + 0.5*mw1*np.linalg.norm(v1)**2 + 0.5*mw2*np.linalg.norm(v2)**2

def calculate_SAM_predictions(Crad):
    #The radiated energy term
    Erad = -Crad*( Ktot1*fg1*fk1*(1+fk1) +  Ktot2*fg2*fk2*(1+fk2) )
    #The orbital energy
    Rsep = np.linalg.norm(distrel_init)
    Eorb = (-G_u*mw1*mw2/Rsep) + 0.5*mw1*np.linalg.norm(v1)**2 + 0.5*mw2*np.linalg.norm(v2)**2
    #The final internal energies
    Eint_f = Eint_i + Erad + Eorb
    #The final stellar half-mass radius of the merger remnant
    rf = (-Cinit*G_u*(m1 + mnew1 + m2 + mnew2)**2)/Eint_f
    #Calculate the velocity dispersion
    sigma_f2 = Cvir*G_u*mf/( rf*(1-fdm_f) )
    sigma_f = np.sqrt(sigma_f2)
    return rf, sigma_f

rf_list = np.array([])
sigma_f_list = np.array([])
for Crad in Crad_list:
    rf, sigma_f = calculate_SAM_predictions(Crad)
    rf_list = np.append(rf_list, rf)
    sigma_f_list = np.append(sigma_f_list, sigma_f)
    

#---------------------------------------------------------------------------------------------------------------------------------------------

def obtain_remnant_properties(idx_evaluate):
    ds_cls = load_ds(codetp, idx_evaluate, pfs)
    #%%%%%%%%%%%%%%%%%%%%%%%%%
    if codetp == 'GADGET3' or codetp == 'GADGET4':
        dm = ParticleUnion("DarkMatter",["PartType5","PartType1"])
        ds_cls.add_particle_union(dm)
    elif codetp == 'AREPO' or codetp == 'GIZMO':
        dm = ParticleUnion("DarkMatter",["PartType2","PartType1"])
        ds_cls.add_particle_union(dm)
    elif codetp == 'GEAR':
        dm = ParticleUnion("DarkMatter",["PartType5","PartType2"])
        ds_cls.add_particle_union(dm)
    elif codetp == 'ENZO':
        def darkmatter_init(pfilter, data):
            filter_darkmatter0 = np.logical_or(data["all", "particle_type"] == 1, data["all", "particle_type"] == 4)
            filter_darkmatter = np.logical_and(filter_darkmatter0,data['all', 'particle_mass'].to('Msun') > 1)
            return filter_darkmatter
        add_particle_filter("DarkMatter",function=darkmatter_init,filtered_type='all',requires=["particle_type","particle_mass"])
        ds_cls.add_particle_filter("DarkMatter")
        #
        def stars(pfilter, data):
            filter_stars = np.logical_and(data[pfilter.filtered_type, "particle_type"] == 2, data[pfilter.filtered_type, "particle_mass"].to('Msun') > 1)
            return filter_stars
        add_particle_filter("stars", function=stars, filtered_type="all", requires=["particle_type","particle_mass"])
        ds_cls.add_particle_filter("stars")
    #%%%%%%%%%%%%%%%%%%%%%%%%%
    center = np.array(dist_data['prog_com_plot'])[np.array(dist_data['idx']) == idx_evaluate][0]
    #reg_cls = ds_cls.sphere(rawtree[prog_branch][idx_evaluate]['Halo_Center'], (rawtree[prog_branch][idx_evaluate]['cden_rad'], 'code_length'))
    #%%%%%%%%%%%%%%%%%%%%%%%%%
    #Stellar mass of the galaxies
    allstars_cls = np.load(metadata_dir + '/' + 'star_metadata_allbox_%s.npy' % idx_evaluate, allow_pickle=True).tolist()
    allmass_cls = allstars_cls['mass']
    allvel_cls = allstars_cls['vel']
    allpos_cls = allstars_cls['pos']
    allID_cls = allstars_cls['ID'].astype(int)
    m1_cls_each = allmass_cls[np.intersect1d(assignment['ids'][prog_branch][idx_evaluate], allID_cls, return_indices=True)[2]]
    pos1_cls_each = allpos_cls[np.intersect1d(assignment['ids'][prog_branch][idx_evaluate], allID_cls, return_indices=True)[2]]
    bool1_cls_each = np.linalg.norm(pos1_cls_each - center, axis=1) <= rawtree[prog_branch][idx_evaluate]['cden_rad']
    m1_cls = m1_cls_each[ bool1_cls_each ].sum()
    #%%%%%%%%%%%%%%%%%%%%%%%%%
    #Stellar half-mass radius of the galaxies
    rf_sim = compute_half_mass_radius(m1_cls_each[bool1_cls_each], pos1_cls_each[bool1_cls_each], center)
    mf_sim = m1_cls
    if codetp == 'ART' or codetp == 'ENZO' or codetp == 'RAMSES':
        reg_dm = ds_cls.sphere(center, (0.5*rf_sim, 'code_length'))
        reg_halfmass = ds_cls.sphere(center, (rf_sim, 'code_length'))
        rf_sim = (rf_sim*ds_cls.units.code_length).to('kpc').v.tolist()
        #Dark matter fraction
        fdm_f_sim = reg_dm[dm_name_dict[codetp], 'particle_mass'].to('Msun').sum().v.tolist()/(reg_dm[dm_name_dict[codetp], 'particle_mass'].to('Msun').sum().v.tolist() + reg_dm[star_name_dict[codetp], 'particle_mass'].to('Msun').sum().v.tolist())
    else:
        reg_cls = ds_cls.all_data()
        dm_allmass_cls = reg_cls[dm_name_dict[codetp], 'particle_mass'].to('Msun').v
        dm_allpos_cls = reg_cls[dm_name_dict[codetp], 'particle_position'].to('code_length').v
        dm_alld_cls = np.linalg.norm(dm_allpos_cls - center, axis=1)
        alld_cls = np.linalg.norm(allpos_cls - center, axis=1) #for stars
        fdm_f_sim = dm_allmass_cls[dm_alld_cls < 0.5*rf_sim].sum()/( dm_allmass_cls[dm_alld_cls < 0.5*rf_sim].sum() +        allmass_cls[alld_cls < 0.5*rf_sim].sum() )
        
    #%%%%%%%%%%%%%%%%%%%%%%%%%
    if codetp == 'ART' or codetp == 'ENZO' or codetp == 'RAMSES':
        star_vel_sim = reg_halfmass[star_name_dict[codetp],'particle_velocity'].to('code_length/s') - rawtree[prog_branch][idx_evaluate]['Vel_Com']*ds_cls.units.code_length/ds_cls.units.s
        star_vel_sim = star_vel_sim.to('kpc/Gyr').v
        star_mass_sim = reg_halfmass[star_name_dict[codetp],'particle_mass'].to('Msun').v
        #%%%%%%%%%%%%%%%%%%%%%%%%%
        #Velocity dispersion
        #sigma_f_sim = np.sqrt(weighted_std(star_vel_sim[:,0], star_mass_sim)**2 + weighted_std(star_vel_sim[:,1], star_mass_sim)**2 + weighted_std(star_vel_sim[:,2], star_mass_sim)**2)
        sigma_f_sim, sigma_f_sim_std = compute_1d_veldispersion(star_vel_sim, star_mass_sim)
    else:
        star_vel_sim = (allvel_cls[alld_cls < rf_sim]*ds_cls.units.km/ds_cls.units.s).to('code_length/s') - rawtree[prog_branch][idx_evaluate]['Vel_Com']*ds_cls.units.code_length/ds_cls.units.s
        star_vel_sim = star_vel_sim.to('kpc/Gyr').v
        star_mass_sim = allmass_cls[alld_cls < rf_sim]
        #Velocity dispersion
        #sigma_f_sim = np.sqrt(weighted_std(star_vel_sim[:,0], star_mass_sim)**2 + weighted_std(star_vel_sim[:,1], star_mass_sim)**2 + weighted_std(star_vel_sim[:,2], star_mass_sim)**2)
        sigma_f_sim, sigma_f_sim_std = compute_1d_veldispersion(star_vel_sim, star_mass_sim)
        rf_sim = (rf_sim*ds_cls.units.code_length).to('kpc').v.tolist()
        
    return mf_sim, rf_sim, fdm_f_sim, sigma_f_sim, sigma_f_sim_std 

mf_sim_list, rf_sim_list, fdm_f_sim_list, sigma_f_sim_list, sigma_f_sim_std_list = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]) 

time_startloop = time_cls - 0.1
time_endloop = time_cls + 0.3
if step == 1:
    idx_startloop = np.argmin(abs(time_startloop - time_list/1e3)) 
    idx_endloop = np.argmin(abs(time_endloop - time_list/1e3))
elif codetp == 'GEAR' or codetp == 'CHANGA':
    idx_startloop = np.argmin(abs(time_startloop - time_list/1e3)) 
    idx_endloop = np.argmin(abs(time_endloop - time_list/1e3)) + 1

idx_loop_list = np.array([])
#for idx_i in range(idx_startloop, idx_endloop + step, step):
for idx_i in [idx_cls]:
    mf_sim, rf_sim, fdm_f_sim, sigma_f_sim, sigma_f_sim_std = obtain_remnant_properties(idx_i)
    mf_sim_list = np.append(mf_sim_list, mf_sim)
    rf_sim_list = np.append(rf_sim_list, rf_sim)
    fdm_f_sim_list = np.append(fdm_f_sim_list, fdm_f_sim)
    sigma_f_sim_list = np.append(sigma_f_sim_list, sigma_f_sim)
    sigma_f_sim_std_list = np.append(sigma_f_sim_std_list, sigma_f_sim_std)
    idx_loop_list = np.append(idx_loop_list, idx_i)

output = {}
output['Crad_list'] = Crad_list
output['rf'] = rf_list
output['rf_sim'] = rf_sim_list
output['mf'] = mf
output['mf_sim'] = mf_sim_list
output['fdm_f'] = fdm_f
output['fdm_f_sim'] = fdm_f_sim_list
output['sigma_f'] = sigma_f_list
output['sigma_f_sim'] = sigma_f_sim_list
output['sigma_f_sim_std'] = sigma_f_sim_std_list
output['idx'] = idx_loop_list
output['mdm_halfmass'] = mdm1 + mdm2
output['Eorb'] = Eorb

np.save('/work/hdd/bezm/tnguyen2/AGORA/analysis/Comparison_CovingtonEtal2008_ProgBranch-%s_%s_ver2013_ver7.npy' % (merger_number, codetp), output)




