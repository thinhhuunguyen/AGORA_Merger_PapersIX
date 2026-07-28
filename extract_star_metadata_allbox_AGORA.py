import numpy as np
import yt
import sys
from yt.utilities.cosmology import Cosmology

yt.enable_parallelism()
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
nprocs = comm.size


def extract_star_metadata_persnap(pfs, metadata_dir, codetp):
    #Runnning parallel to load the star information
    my_storage = {}
    for sto, idx in yt.parallel_objects(range(len(pfs)), nprocs, storage = my_storage, dynamic = False):
        #
        if codetp == 'GADGET3' or codetp == 'AREPO' or codetp == 'GADGET4' or codetp == 'AREPO-TNG':
            ds = yt.load(pfs[idx],unit_base = {"length": (1.0, "Mpccm/h")})
        else:
            ds = yt.load(pfs[idx])
        try:
            reg = ds.all_data()
        except:
            print('Error at Snapshot', idx)
            continue
        #
        if codetp == 'GEAR':
            co = Cosmology(hubble_constant = ds.hubble_constant, omega_matter = ds.omega_matter, 
                    omega_lambda = ds.omega_lambda, omega_radiation = ds.omega_radiation)
            pmass = reg[("PartType1", "particle_mass")].in_units("Msun").v
            pmass_init = reg['PartType1', 'InitialMass'].to('Msun').v
            ppos = reg[("PartType1", "particle_position")].in_units("code_length").v
            pvel = reg[("PartType1", "particle_velocity")].in_units("km/s").v
            pftime = reg[('PartType1', 'StarFormationTime')]
            if len(pftime) > 0:  
                pftime = co.t_from_a(pftime).in_units('Gyr').v
            else: 
                pftime = np.array([])
            page = ds.current_time.to('Gyr').v - pftime
            pmet = reg['PartType1','StarMetals'].v #this field is dimensionless in yt
            pID = reg[("PartType1", "ParticleIDs")].v
        #
        elif codetp == 'GIZMO':
            pmass = reg[("PartType4", "particle_mass")].in_units("Msun").v
            ppos = reg[("PartType4", "particle_position")].in_units("code_length").v
            pvel = reg[("PartType4", "particle_velocity")].in_units("km/s").v
            page = reg['PartType4','age'].to('Gyr').v 
            pmet = reg['PartType4','metallicity'].to('Zsun').v 
            pID = reg[("PartType4", "ParticleIDs")].v
            #GIZMO does not store the initial stellar mass, so reconstruct it from the
            #current mass and the age-dependent cumulative mass-loss fraction.
            def mass_loss_fraction_calc(star_age):
                M_total = 6.5774186e-4*np.minimum(star_age,0.005089) ##Gyr
                mask = star_age > 0.005089
                if np.any(mask):  # Perform the calculation only where the condition is True
                    M_total[mask] += (
                        0.00017845 * 0.0381 / (1 - 0.648) *
                        (
                            np.power(np.minimum(star_age[mask], 0.0381) / 0.0381, 1 - 0.648) -
                            np.power(0.005089 / 0.0381, 1 - 0.648)
                        )
                    )
                M_total *= 1e3*14.8/1.19693;
                return M_total #M_total is the mass-loss fraction
            mass_loss_fraction = mass_loss_fraction_calc(np.array(page)) #page already in Gyr
            pmass_init = pmass / (1 - mass_loss_fraction)
        #
        elif codetp == 'CHANGA':
            try:
                pmass = reg[("Stars", "Mass")].in_units("Msun").v
                pmass_init = 56525.92803654*np.ones(len(pmass)) #CHANGA does not implement stellar mass loss
            except:
                continue
            ppos = reg[("Stars", "particle_position")].in_units("code_length").v
            pvel = reg[("Stars", "particle_velocity")].in_units("km/s").v
            pftime = reg['Stars','FormationTime'].to('Gyr').v
            page = ds.current_time.to('Gyr').v - pftime
            pmet = reg['Stars','Metals'].to('Zsun').v 
            pID = np.arange(len(pftime)) + 15368024
        #
        elif codetp == 'GADGET3' or codetp == 'GADGET4':
            co = Cosmology(hubble_constant = ds.hubble_constant, omega_matter = ds.omega_matter, 
                    omega_lambda = ds.omega_lambda, omega_radiation = ds.omega_radiation)
            try:
                pmass = reg['PartType4', 'Masses'].to('Msun').v
            except:
                continue
            pmass_init = (reg['PartType4', 'InitialStellarMass']*ds.units.code_mass).to('Msun').v
            ppos = reg['PartType4', 'particle_position'].to('code_length').v
            pvel = reg['PartType4', 'particle_velocity'].to('km/s').v
            pftime = reg[('PartType4', 'StellarFormationTime')]
            if len(pftime) > 0:  
                pftime = co.t_from_a(pftime).in_units('Gyr').v
            else: 
                pftime = np.array([])
            page = ds.current_time.to('Gyr').v - pftime
            pmet = reg['PartType4','Metallicity'].to('Zsun').v
            pID = reg['PartType4','ParticleIDs'].v
        #
        elif codetp == 'AREPO':
            co = Cosmology(hubble_constant = ds.hubble_constant, omega_matter = ds.omega_matter, 
                    omega_lambda = ds.omega_lambda, omega_radiation = ds.omega_radiation)
            pmass = reg['PartType4', 'particle_mass'].to('Msun').v
            pmass_init = (reg['PartType4', 'OriginalMass']*ds.units.code_mass).to('Msun').v
            ppos = reg['PartType4', 'particle_position'].to('code_length').v
            pvel = reg['PartType4', 'particle_velocity'].to('km/s').v
            pftime = reg[('PartType4', 'StellarFormationTime')]
            if len(pftime) > 0:  
                pftime = co.t_from_a(pftime).in_units('Gyr').v
            else: 
                pftime = np.array([])
            page = ds.current_time.to('Gyr').v - pftime
            pmet = reg['PartType4','Metallicity'].to('Zsun').v
            pID = reg['PartType4','ParticleIDs'].v
        #
        elif codetp == 'AREPO-TNG':
            co = Cosmology(hubble_constant = ds.hubble_constant, omega_matter = ds.omega_matter, 
                    omega_lambda = ds.omega_lambda, omega_radiation = ds.omega_radiation)
            try:
                pmass = reg['PartType4', 'Masses'].to('Msun').v
            except:
                continue
            pmass_init = (reg['PartType4', 'GFM_InitialMass']*ds.units.code_mass).to('Msun').v
            ppos = reg['PartType4', 'particle_position'].to('code_length').v
            pvel = reg['PartType4', 'particle_velocity'].to('km/s').v
            pftime = reg[('PartType4', 'GFM_StellarFormationTime')].v
            pmet = reg['PartType4','metallicity'].to('Zsun').v
            pID = reg['PartType4','ParticleIDs'].v
            #
            #pftime < 0 corresponds to gas cells in wind phase. Only pftime > 0 corresponds to real stars
            pmass = pmass[pftime > 0]
            pmass_init = pmass_init[pftime > 0]
            ppos = ppos[pftime > 0]
            pvel = pvel[pftime > 0]
            pmet = pmet[pftime > 0]
            pID = pID[pftime > 0]
            pftime = pftime[pftime > 0]
            #
            if len(pftime) > 0:  
                pftime = co.t_from_a(pftime).in_units('Gyr').v
            else: 
                pftime = np.array([])
            page = ds.current_time.to('Gyr').v - pftime

        elif codetp == 'ART':
            try:
                pmass = reg['stars', 'particle_mass'].to('Msun').v
                pmass_init = reg['stars', 'particle_mass_initial'].to('Msun').v
            except:
                continue
            ppos = reg['stars', 'particle_position'].to('code_length').v
            pvel = reg['stars', 'particle_velocity'].to('km/s').v
            pftime = reg[('stars', 'particle_creation_time')].to('Gyr').v
            page = ds.current_time.to('Gyr').v - pftime
            agora_Zsun = 0.02041
            pmet = (reg['stars','particle_metallicity1'].v + reg['stars','particle_metallicity2'].v)/agora_Zsun
            pID = reg['stars','particle_index'].v
        
        elif codetp == 'ENZO':
            ptype = reg['all', 'particle_type'].v
            pmass = reg['all', 'particle_mass'].to('Msun').v
            ppos = reg['all', 'particle_position'].to('code_length').v
            pvel = reg['all', 'particle_velocity'].to('km/s').v
            page = reg['all','age'].to('Gyr').v
            pmet = reg['all','metallicity_fraction'].v
            pID = reg['all','particle_index'].v
            star_bool = np.logical_and(ptype == 2, pmass > 1)
            pmass = pmass[star_bool]
            ppos = ppos[star_bool]
            pvel = pvel[star_bool]
            page = page[star_bool]
            pmet = pmet[star_bool]
            pID = pID[star_bool]
            #ENZO does not store the initial stellar mass. Reconstruct it from the current mass.
            mask = page*1e3 < 5 #age less than 5 Myr
            pmass_init = pmass.copy() #.copy() so the stored 'mass' array is not modified in place
            pmass_init[mask] = pmass_init[mask] / (1 - 0.163) #If star particle's age is more than 5Myr, it loses 16.3% of its mass. 
        
        elif codetp == 'RAMSES':
            pmass = reg['star', 'particle_mass'].to('Msun').v
            pmass_init = pmass #RAMSES does not implement stellar mass loss
            ppos = reg['star', 'particle_position'].to('code_length').v
            pvel = reg['star', 'particle_velocity'].to('km/s').v
            page = reg['star','age'].to('Gyr').v
            pmet = reg['star','particle_metallicity'].v
            pID = reg['star','particle_index'].v
        #
        output = {}
        output['mass'] = pmass
        output['pos'] = ppos
        output['vel'] = pvel
        output['age'] = page
        output['met'] = pmet
        output['ID'] = pID
        output['mass_init'] = pmass_init
        print('Star metadata in Snapshot Idx %s is extracted' % idx)
        np.save(metadata_dir + '/star_metadata_allbox_'+str(idx)+'.npy', output)
    return None


#---------------------------------------------------------------------------------
halo_dir = sys.argv[1]
metadata_dir = sys.argv[2]
halotree_ver = sys.argv[3]
codetp = sys.argv[4]
pfs = np.loadtxt(halo_dir + '/pfs_allsnaps_%s.txt' % halotree_ver, dtype=str)[:,0]
if yt.is_root():
    print('Done loading data')
    print(metadata_dir)

extract_star_metadata_persnap(pfs, metadata_dir, codetp)

