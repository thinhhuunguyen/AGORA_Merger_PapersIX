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

[add_hullv_position-partial.py](add_hullv_position-partial.py): add the vertices' coordinate of the convex hulls (of only relevant halos to the analysis to save space)

### Part-1_Paper:
- [Make_merger_tree_plot.py](Part-1_Paper/Make_merger_tree_plot.py): Figure 1 
    - *Make_merger_tree_plot_DATA.py*: Generate data to plot Figure 1
- [trajectory_comparison.py](Part-1_Paper/trajectory_comparison.py): Figure 2
- [divide_merger_stages.py](Part-1_Paper/divide_merger_stages.py): Figure 3 (left subplot)
- [merger_timing_plot.py](Part-1_Paper/merger_timing_plot.py): Figure 3 (right subplot)
    - *Gas_Mass_Fraction_calc_ConvexHull_preInfall_PrimaryGalaxy.py*: Generate data to plot Figure 3 (right subplot)
    - *Gas_Mass_Fraction_calc_ConvexHull_preinfall_SecondaryGalaxy.py*: Generate data to plot Figure 3 (right subplot)
- [All_Properties_vs_Time_Plot.py](Part-1_Paper/All_Properties_vs_Time_Plot.py): Figure 4
- [Gas_phase_plot_at_different_merger_stages.py](Part-1_Paper/Gas_phase_plot_at_different_merger_stages.py): Figure 5 + Figure 6 
- [Trace_Gas_Particles_that_Turns_Into_Starburst_Stars_plotting-fromSecondaryHalo_CombinedAllCodes.py](Part-1_Paper/Trace_Gas_Particles_that_Turns_Into_Starburst_Stars_plotting-fromSecondaryHalo_CombinedAllCodes.py): Figure 7
    - *bound_gas_index_seccondary_galaxy_preinfall.py*: Generate data to plot Figure 7
    - *Trace_Gas_Particles_that_Turns_Into_Starburst_Stars_creatingPlotData_GADGET3_GEAR_GIZMO.py*: Generate data to plot Figure 7
    - *Trace_Gas_Particles_that_Turns_Into_Starburst_Stars_creatingPlotData_GADGET4.py*: Generate data to plot Figure 7
    - *Trace_Gas_Particles_that_Turns_Into_Starburst_Stars_creatingPlotData_CHANGA.py*: Generate data to plot Figure 7
- [infalling_gasmass_and_momentum_maxdist_stage.py](Part-1_Paper/infalling_gasmass_and_momentum_maxdist_stage.py): Figure 8
    - *Select_gas_with_negative_radialvel_infalling_gasparticletracing.py*: Generate data to plot Figure 8
- [preMerger_StellarMass_SFR_sSFR.py](Part-1_Paper/preMerger_StellarMass_SFR_sSFR.py): Figure 9
- [Calculate_BurstFraction_and_its_correlations.py](Part-1_Paper/Calculate_BurstFraction_and_its_correlations.py): Figures 10 + Figure 11
    - *Gas_Properties_preInfall_allCodes.py*: Generate data to plot Figure 11
- [evolution_of_angular-momentum_and_variables_defining_coalescence.py](Part-1_Paper/evolution_of_angular-momentum_and_variables_defining_coalescence.py): Figure 12
- [GasMetallicity_and_CoolingTime_PreMerger.py](Part-1_Paper/GasMetallicity_and_CoolingTime_PreMerger.py): Figure 13
    - *Gas_Metallicity_calc_ConvexHull.py*: Generate data to plot Figure 13
    - *Gas_CoolingTime_calc_ConvexHull.py*: Generate data to plot Figure 13
- [GADGETs_preMajorMerger_burst_minorMergerHypothesis.py](Part-1_Paper/GADGETs_preMajorMerger_burst_minorMergerHypothesis.py): Figure 14
- [CHANGA_delayed_starburst_SFR.py](Part-1_Paper/CHANGA_delayed_starburst_SFR.py): Figure 15
- [Trace_Gas_Particles_that_Turns_Into_Starburst_Stars_plotting_for-CHANGA-laterBurst.py](Part-1_Paper/Trace_Gas_Particles_that_Turns_Into_Starburst_Stars_plotting_for-CHANGA-laterBurst.py): Figure 16

### Part-2_Paper:
- [Find_Galactic_Center.py](Part-2_Paper/Find_Galactic_Center.py): General Python Class to find a galactic center following Section 2.4 of Part-2 Paper
    - [Find_Galactic_Center_in_AGORA.py](Part-2_Paper/Find_Galactic_Center_in_AGORA.py): Applying the Find_Galactic_Center() class to the AGORA simulations (automatically find the centers through the snapshot list).
- [Gas_Projections_MergerSequence_AllCodes.py](Part-2_Paper/Gas_Projections_MergerSequence_AllCodes.py): Figure 1
- [mass_radial_distribution_PlotCombined.py](Part-2_Paper/mass_radial_distribution_PlotCombined.py): Figure 2
    - *mass_radial_distribution_data.py*: Generate data to plot Figure 2
    - *premerger_center_and_velocity.py*: Compute the center and velocity of the progenitors pre-infall, used for Figures 2, 3, and 4
- [Calculate_StellarHalfMassRadius_and_its_correlation_with_BurstFraction.py](Part-2_Paper/Calculate_StellarHalfMassRadius_and_its_correlation_with_BurstFraction.py): Figure 3
- [circularity_distribution_plot.py](Part-2_Paper/circularity_distribution_plot.py): Figure 4
    - *stellar_orbital_circularity_calculation_Abadi2003method.py*: Calculate the stellar orbital circularity to plot Figure 4
        - *extract_darkmatter_bary_MassPos_ConvexHull.py*: Calculate the dark matter and baryonic mass for the circularity calculation
    - *circularity_decompose_merger_stages.py*: Decompose the calculated stellar circularity into stellar groups from different merger stages
- [Disk_Decomposition_Mass_Fraction.py](Part-2_Paper/Disk_Decomposition_Mass_Fraction.py): Figure 5
- [circularity_vs_radial-distance_plot.py](Part-2_Paper/circularity_vs_radial-distance_plot.py): Figure 6
- [angular_momentum_direction_change.py](Part-2_Paper/angular_momentum_direction_change.py): Figure 7
