# AGORA_Merger_PapersIX
Analysis codes for the two AGORA Papers IX (Part 1 and Part 2). The studies analyze the effects of a major merger on star formation and stellar morphology of a Milky Way-mass Galaxy Progenitor and compare the results across nine state-of-the-art galaxy simulation codes.

Links to papers:
- Part 1 (Effects on star formation): https://arxiv.org/abs/2607.21709
- Part 2 (Effects on stellar morphology): https://arxiv.org/abs/2607.21710

## Details of each file

[setup.py](setup.py): This module provides the shared setup and helper routines for analyzing the AGORA multi-code simulation suite, including the code list, yt-field name for each code, plotting parameters for each code, halo merger-tree and snapshot loading, and merger infall-timing calculations. It also defines derived yt fields—such as AGORA-normalized metallicity, cooling time, and free-fall time, so that datasets from the different simulation codes can be loaded and compared on a common footing.

[extract_star_metadata_allbox_AGORA.py](extract_star_metadata_allbox_AGORA.py): extract the metadata (pos, vel, mass, initial mass, ID, age, metallicity) of all star particles in a simulation snapshot.

[visualizing_stars_assignment_and_DM_halos.ipynb](visualizing_stars_assignment_and_DM_halos.ipynb): this notebook visualizes the DM halos with their assigned stars and hence helps identifies the secondary halos of the major merger 

[Finding_the_infall_timestep_of_the_merger_non-spherical_halos.ipynb](Finding_the_infall_timestep_of_the_merger_non-spherical_halos.ipynb): this notebook identifies the starting timestep of the merger, aka. when the two convex hulls overlap for the first time 

[Generate_stellar-core_tracking_data.py](Generate_stellar-core_tracking_data.py): generate the tracking data of the stellar cores of the two progenitor galaxies. The output of this code is the "dist_data" variable loaded in the following codes. 

[determine_coalescence_timestep.py](determine_coalescence_timestep.py): determines the coalescence timestep for each code

### Part-1_Paper:
- [Make_merger_tree_plot.py](Part-1_Paper/Make_merger_tree_plot.py): Figure 1 
    - *Make_merger_tree_plot_DATA.py*: Generate data to plot Figure 1
- *trajectory_comparison.py*: Figure 2
- *divide_merger_stages.py*: Figure 3 (left subplot)
- *merger_timing_plot.py*: Figure 3 (right subplot)
    - *Gas_Mass_Fraction_calc_ConvexHull_preInfall_PrimaryGalaxy.py*: Generate data to plot Figure 3 (right subplot)
    - *Gas_Mass_Fraction_calc_ConvexHull_preinfall_SecondaryGalaxy.py*: Generate data to plot Figure 3 (right subplot)
- *All_Properties_vs_Time_Plot.py*: Figure 4
- *Gas_phase_plot_at_different_merger_stages.py*: Figure 5 + Figure 6 
- *Trace_Gas_Particles_that_Turns_Into_Starburst_Stars_plotting-fromSecondaryHalo_CombinedAllCodes.py*: Figure 7
    - *bound_gas_index_seccondary_galaxy_preinfall.py*: Generate data to plot Figure 7
    - *Trace_Gas_Particles_that_Turns_Into_Starburst_Stars_creatingPlotData_GADGET3_GEAR_GIZMO.py*: Generate data to plot Figure 7
    - *Trace_Gas_Particles_that_Turns_Into_Starburst_Stars_creatingPlotData_GADGET4.py*: Generate data to plot Figure 7
    - *Trace_Gas_Particles_that_Turns_Into_Starburst_Stars_creatingPlotData_CHANGA.py*: Generate data to plot Figure 7
- *infalling_gasmass_and_momentum_maxdist_stage.py*: Figure 8
    - *Select_gas_with_negative_radialvel_infalling_gasparticletracing.py*: Generate data to plot Figure 8
- *preMerger_StellarMass_SFR_sSFR.py*: Figure 9
- *Calculate_BurstFraction_and_its_correlations.py*: Figures 10 + Figure 11
    - *Gas_Properties_preInfall_allCodes.py*: Generate data to plot Figure 11
- *evolution_of_angular-momentum_and_variables_defining_coalescence.py*: Figure 12
- *Trace_Gas_Particles_that_Turns_Into_Starburst_Stars_plotting_for-CHANGA-laterBurst.py*: Figure 16