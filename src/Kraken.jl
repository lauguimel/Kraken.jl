"""
    Kraken

GPU-native Lattice Boltzmann Method (LBM) framework in Julia.

Supports 2D (D2Q9) and 3D (D3Q19) simulations with automatic GPU
acceleration via KernelAbstractions.jl.
"""
module Kraken

# --- Lattice definitions ---
include("lattice/lattice.jl")
include("lattice/d2q9.jl")
include("lattice/d3q19.jl")

# --- Rheology models ---
include("rheology/models.jl")
include("rheology/viscosity.jl")
include("rheology/strain_rate.jl")
include("rheology/linalg.jl")

# --- GPU kernels ---
include("kernels/equilibrium_helpers.jl")
include("kernels/collide_stream_2d.jl")
include("kernels/collide_stream_3d.jl")
include("kernels/stream_periodic_2d.jl")
include("kernels/collide_guo_2d.jl")
include("kernels/collide_guo_3d.jl")
include("kernels/macroscopic.jl")
include("kernels/boundary_2d.jl")
include("kernels/boundary_3d.jl")
include("kernels/thermal_2d.jl")
include("kernels/fused_thermal_2d.jl")
include("kernels/collide_mrt_2d.jl")
include("kernels/species_2d.jl")
include("kernels/multiphase_2d.jl")
include("kernels/vof_2d.jl")
include("kernels/dualgrid_2d.jl")
include("kernels/phasefield_2d.jl")
include("kernels/advect_prescribed_2d.jl")
include("kernels/collide_rheology_2d.jl")
include("kernels/collide_twophase_rheology_2d.jl")
include("kernels/viscoelastic_2d.jl")

# --- Simulation ---
include("simulation.jl")

# --- Grid refinement ---
include("refinement/refinement.jl")
include("kernels/refinement_exchange_2d.jl")
include("refinement/time_stepping.jl")
include("refinement/thermal_refinement.jl")

# --- I/O ---
include("io/vtk_writer.jl")
include("io/diagnostics.jl")
include("io/expression.jl")
include("io/stl_reader.jl")
include("io/voxelizer.jl")
include("io/kraken_parser.jl")

# --- Spatial boundary kernels ---
include("kernels/boundary_spatial_2d.jl")

# --- Generic simulation runner ---
include("simulation_runner.jl")

# --- Post-processing helpers ---
include("postprocess.jl")

# Lattice types and functions
export AbstractLattice, D2Q9, D3Q19
export lattice_dim, lattice_q, weights, velocities_x, velocities_y, velocities_z
export opposite, cs2, equilibrium

# Kernels
export stream_2d!, collide_2d!, stream_3d!, collide_3d!
export stream_periodic_x_wall_y_2d!, stream_fully_periodic_2d!, stream_periodic_x_axisym_2d!
export stream_axisym_inlet_2d!
export collide_guo_2d!, collide_guo_field_2d!
export collide_guo_3d!, collide_guo_field_3d!
export compute_macroscopic_2d!, compute_macroscopic_3d!, compute_macroscopic_forced_2d!
export compute_macroscopic_forced_3d!, compute_macroscopic_pressure_2d!
export apply_zou_he_north_2d!, apply_zou_he_south_2d!
export apply_zou_he_west_2d!, apply_zou_he_pressure_east_2d!, apply_extrapolate_east_2d!
export apply_zou_he_top_3d!
export apply_zou_he_bottom_3d!, apply_zou_he_west_3d!, apply_zou_he_east_3d!
export apply_zou_he_south_3d!, apply_zou_he_north_3d!
export apply_zou_he_pressure_east_3d!, apply_zou_he_pressure_top_3d!
export apply_bounce_back_walls_3d!

# Simulation
export LBMConfig, omega, reynolds
export initialize_2d, initialize_3d
export run_cavity_2d, run_cavity_3d
export run_poiseuille_2d, run_couette_2d
export initialize_taylor_green_2d, run_taylor_green_2d
export initialize_cylinder_2d, run_cylinder_2d, compute_drag_mea_2d
export collide_thermal_2d!, compute_temperature_2d!
export apply_fixed_temp_south_2d!, apply_fixed_temp_north_2d!
export apply_fixed_temp_west_2d!, apply_fixed_temp_east_2d!
export run_rayleigh_benard_2d, run_natural_convection_2d, run_natural_convection_refined_2d
export ThermalPatchArrays, create_thermal_patch_arrays, advance_thermal_refined_step!
export collide_boussinesq_2d!, collide_boussinesq_vt_2d!, collide_boussinesq_vt_modified_2d!
export fused_natconv_step!, fused_natconv_vt_step!
export collide_axisymmetric_2d!, collide_li_axisym_2d!, run_hagen_poiseuille_2d

# MRT
export collide_mrt_2d!, collide_twophase_mrt_2d!

# Species transport
export collide_species_2d!, compute_concentration_2d!
export apply_fixed_conc_south_2d!, apply_fixed_conc_north_2d!

# Multiphase (Shan-Chen)
export compute_psi_2d!, compute_sc_force_2d!, collide_sc_2d!
export run_spinodal_2d, benchmark_mlups

# VOF PLIC
export compute_vof_normal_2d!, advect_vof_2d!
export compute_hf_curvature_2d!, compute_surface_tension_2d!
export collide_twophase_2d!, run_static_droplet_2d, run_plateau_pinch_2d
export add_azimuthal_curvature_2d!, add_axisym_viscous_correction_2d!, set_vof_west_2d!
export apply_density_correction_2d!
export run_rp_axisym_2d, run_cij_jet_axisym_2d, run_cij_jet_phasefield_2d

# Phase-field (Allen-Cahn + pressure-based)
export phasefield_params, compute_phi_2d!, compute_chemical_potential_2d!
export add_azimuthal_chemical_potential_2d!, compute_phasefield_force_2d!
export compute_vof_from_phi_2d!, compute_antidiffusion_flux_2d!
export collide_allen_cahn_2d!, add_azimuthal_allen_cahn_source_2d!
export collide_pressure_phasefield_mrt_2d!, compute_macroscopic_phasefield_2d!
export set_phasefield_west_2d!, extrapolate_phasefield_east_2d!
export init_phasefield_equilibrium, init_pressure_equilibrium
export run_static_droplet_phasefield_2d

# Dual-grid VOF
export prolongate_bilinear_2d!, restrict_average_2d!
export compute_hf_curvature_dx_2d!, compute_surface_tension_dx_2d!
export run_static_droplet_dualgrid_2d


# Prescribed-velocity advection
export clamp_field_2d!, advect_vof_step!, advect_vof_plic_step!
export advect_vof_plic_2d!, fill_velocity_field!, init_vof_field!
export run_advection_2d

# I/O
export write_vtk, create_pvd, write_vtk_to_pvd
export setup_output_dir, write_snapshot_2d!, write_snapshot_3d!
export write_vtk_multiblock, write_snapshot_refined_2d!
export DiagnosticsLogger, open_diagnostics, log_diagnostics!, close_diagnostics!

# .krk config system
export KrakenExpr, parse_kraken_expr, evaluate, has_variable, is_time_dependent, is_spatial
export SimulationSetup, DomainSetup, PhysicsSetup, GeometryRegion, BoundarySetup, RheologySetup
export InitialSetup, OutputSetup, DiagnosticsSetup, STLSource, RefineSetup
export load_kraken, parse_kraken, build_rheology_model
export run_simulation

# Grid refinement
export RefinementPatch, RefinedDomain
export create_patch, create_refined_domain, rescaled_omega
export rescaling_factor_c2f, rescaling_factor_f2c
export prolongate_f_rescaled_2d!, restrict_f_rescaled_2d!
export temporal_interpolate_2d!, copy_macroscopic_overlap_2d!
export advance_refined_step!
export TwophaseRefinedArrays, create_twophase_patch_arrays, advance_twophase_refined_step!

# STL geometry
export STLTriangle, STLMesh, read_stl, transform_mesh
export voxelize_2d, voxelize_3d

# Post-processing
export extract_line, field_error, probe, domain_stats
export load_basilisk_interfaces, load_basilisk_interface_contour
export find_basilisk_snapshot, compare_interfaces

# Rheology
export AbstractRheology, GeneralizedNewtonian, Viscoelastic
export AbstractThermalCoupling, IsothermalCoupling, ArrheniusCoupling, WLFCoupling
export Newtonian, PowerLaw, CarreauYasuda, Cross, Bingham, HerschelBulkley
export OldroydB, FENEP, Saramito
export StressFormulation, LogConfFormulation
export effective_viscosity, effective_viscosity_thermal, thermal_shift_factor
export strain_rate_magnitude_2d, strain_rate_magnitude_3d
export collide_rheology_2d!, collide_rheology_guo_2d!, collide_rheology_thermal_2d!
export collide_twophase_rheology_2d!

# Viscoelastic
export eigen_sym2x2, mat_exp_sym2x2, mat_log_sym2x2, decompose_velocity_gradient
export compute_polymeric_force_2d!
export evolve_stress_2d!, evolve_logconf_2d!
export compute_stress_from_conf_2d!, compute_stress_from_logconf_2d!

# Spatial boundary kernels
export apply_zou_he_north_spatial_2d!, apply_zou_he_south_spatial_2d!
export apply_zou_he_west_spatial_2d!, apply_zou_he_pressure_east_spatial_2d!
export apply_zou_he_pressure_inlet_west_2d!

end # module Kraken
