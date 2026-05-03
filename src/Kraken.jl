"""
    Kraken

GPU-native Lattice Boltzmann Method (LBM) framework in Julia.

Supports 2D (D2Q9) and 3D (D3Q19) simulations with automatic GPU
acceleration via KernelAbstractions.jl.
"""
module Kraken

# PNG/GIF output hooks — populated by KrakenMakieExt when CairoMakie is loaded
const _png_saver = Ref{Any}(nothing)
const _gif_saver = Ref{Any}(nothing)

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
include("kernels/equilibrium_helpers_3d.jl")
include("kernels/collide_stream_2d.jl")
include("kernels/collide_stream_3d.jl")
include("kernels/stream_periodic_2d.jl")
include("kernels/collide_guo_2d.jl")
include("kernels/collide_guo_3d.jl")
include("kernels/macroscopic.jl")
include("kernels/boundary_2d.jl")
include("kernels/boundary_3d.jl")
include("kernels/thermal_2d.jl")
include("kernels/thermal_3d.jl")
include("kernels/fused_thermal_2d.jl")
include("kernels/collide_mrt_2d.jl")
include("kernels/species_2d.jl")
include("kernels/multiphase_2d.jl")
include("kernels/vof_2d.jl")
include("kernels/dualgrid_2d.jl")
include("kernels/phasefield_2d.jl")
include("kernels/pressure_vof_2d.jl")
include("kernels/smooth_vof_2d.jl")
include("kernels/ghost_fluid_2d.jl")
include("kernels/fused_bgk_2d.jl")
include("kernels/fused_trt_2d.jl")
include("kernels/li_bb_2d.jl")
include("kernels/aa_bgk_2d.jl")
include("kernels/persistent_bgk_2d.jl")
include("kernels/advect_prescribed_2d.jl")
include("kernels/collide_rheology_2d.jl")
include("kernels/collide_twophase_rheology_2d.jl")
include("kernels/viscoelastic_2d.jl")

# --- Kernel DSL (runtime fusion) ---
include("kernels/dsl/lbm_spec.jl")
include("kernels/dsl/bricks.jl")
include("kernels/dsl/bricks_3d.jl")
include("kernels/dsl/lbm_builder.jl")

# --- Kernels built from the DSL (must come after li_bb_2d.jl + DSL) ---
include("kernels/li_bb_2d_v2.jl")
include("kernels/li_bb_3d_v2.jl")

# --- Modular BC system (uses TRT rates + feq helpers; compiles face
#     kernels per BC type via Julia dispatch).
include("kernels/boundary_rebuild.jl")

# --- GPU-native drag reductions (replace host-side per-step transfers)
include("kernels/drag_gpu.jl")

# --- Analytic geometry derivatives for AD shape optimization ---
include("kernels/enzyme_rules.jl")

# --- Simulation drivers ---
include("drivers/basic.jl")
include("drivers/cylinder_libb.jl")
include("drivers/thermal.jl")
include("drivers/axisymmetric.jl")
include("drivers/multiphase.jl")
include("drivers/rheology.jl")
include("drivers/viscoelastic.jl")

# --- Curvilinear (body-fitted) mesh — v0.2 SLBM path ---
include("curvilinear/mesh.jl")
include("curvilinear/generators.jl")
include("curvilinear/slbm.jl")
include("curvilinear/mesh_3d.jl")
include("curvilinear/slbm_3d.jl")
include("curvilinear/mesh_from_arrays.jl")
include("curvilinear/mesh_gmsh.jl")

# --- Multi-block structured (v0.3) ---
include("multiblock/multiblock.jl")

# --- Grid refinement ---
include("refinement/refinement.jl")
include("refinement/conservative_tree_2d.jl")
include("refinement/conservative_tree_topology_2d.jl")
include("refinement/conservative_tree_streaming_2d.jl")
include("kernels/refinement_exchange_2d.jl")
include("refinement/time_stepping.jl")
include("refinement/thermal_refinement.jl")
include("kernels/refinement_exchange_3d.jl")
include("refinement/refinement_3d.jl")

# --- I/O ---
include("io/vtk_writer.jl")
include("io/diagnostics.jl")
include("io/expression.jl")
include("io/stl_reader.jl")
include("io/voxelizer.jl")
include("io/stl_libb.jl")
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
export apply_bounce_back_walls_3d!, apply_bounce_back_wall_3d!
export apply_bounce_back_walls_2d!, apply_bounce_back_wall_2d!

# Simulation
export LBMConfig, omega, reynolds
export initialize_2d, initialize_3d
export run_cavity_2d, run_cavity_3d
export run_poiseuille_2d, run_couette_2d
export initialize_taylor_green_2d, run_taylor_green_2d
export initialize_cylinder_2d, run_cylinder_2d, compute_drag_mea_2d
export run_cylinder_libb_2d, compute_drag_libb_2d, compute_drag_libb_mei_2d
export rebuild_inlet_outlet_libb_2d!, rebuild_inlet_outlet_libb_3d!
export AbstractBC, HalfwayBB, InterfaceBC, ZouHeVelocity, ZouHePressure
export BCSpec2D, BCSpec3D, apply_bc_rebuild_2d!, apply_bc_rebuild_3d!
export compute_drag_libb_mei_2d_gpu!, compute_drag_libb_3d_gpu!
export CutLinkList, CutLinkList3D, build_cut_link_list_2d, build_cut_link_list_3d
export collide_thermal_2d!, compute_temperature_2d!
export apply_fixed_temp_south_2d!, apply_fixed_temp_north_2d!
export apply_fixed_temp_west_2d!, apply_fixed_temp_east_2d!
export run_rayleigh_benard_2d, run_natural_convection_2d, run_natural_convection_refined_2d
export run_natural_convection_3d
export ThermalPatchArrays, create_thermal_patch_arrays, advance_thermal_refined_step!
export collide_boussinesq_2d!, collide_boussinesq_vt_2d!, collide_boussinesq_vt_modified_2d!
export fused_natconv_step!, fused_natconv_vt_step!
export collide_thermal_3d!, compute_temperature_3d!, collide_boussinesq_3d!
export apply_fixed_temp_west_3d!, apply_fixed_temp_east_3d!
export apply_fixed_temp_south_3d!, apply_fixed_temp_north_3d!
export apply_fixed_temp_bottom_3d!, apply_fixed_temp_top_3d!
export fused_bgk_step!, aa_even_step!, aa_odd_step!
export fused_trt_step!, trt_rates
export fused_trt_libb_step!, fused_trt_libb_v2_step!, fused_trt_libb_v2_step_3d!, precompute_q_wall_cylinder
export dq_wall_dR_cylinder
export precompute_q_wall_sphere_3d, compute_drag_libb_3d, run_sphere_libb_3d
export precompute_q_wall_annulus
export wall_velocity_rotating_cylinder, wall_velocity_rotating_inner
export persistent_fused_bgk!, persistent_aa_bgk!
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
export run_rp_axisym_2d, run_rp_pressure_vof_2d, run_rp_hybrid_2d
export run_cij_jet_axisym_2d, run_cij_jet_phasefield_2d, run_cij_jet_hybrid_2d
export collide_pressure_vof_mrt_2d!, compute_surface_tension_weighted_2d!
export init_pressure_vof_equilibrium, smooth_vof_2d!, correct_mass_2d!
export add_axisym_viscous_weighted_2d!
export extrapolate_velocity_ghost_2d!, reset_feq_ghost_2d!

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
export open_paraview
export DiagnosticsLogger, open_diagnostics, log_diagnostics!, close_diagnostics!

# .krk config system
export KrakenExpr, parse_kraken_expr, evaluate, has_variable, is_time_dependent, is_spatial
export SimulationSetup, DomainSetup, PhysicsSetup, GeometryRegion, BoundarySetup, RheologySetup
export InitialSetup, OutputSetup, DiagnosticsSetup, STLSource, RefineSetup, SanityIssue
export load_kraken, parse_kraken, build_rheology_model,
       parse_kraken_sweep, load_kraken_sweep, sanity_check, sanity_check_sweep
export LBMParams, lbm_params, lbm_params_table
export run_simulation

# Curvilinear mesh (v0.2 SLBM path)
export CurvilinearMesh, build_mesh, validate_mesh, compute_metric
export polar_mesh, stretched_box_mesh, cartesian_mesh, cylinder_focused_mesh
export cell_area, domain_extent
export SLBMGeometry, build_slbm_geometry, transfer_slbm_geometry
export slbm_bgk_step!, slbm_bgk_moving_step!, slbm_mrt_step!
export slbm_trt_libb_step!, slbm_trt_libb_step_local_2d!, slbm_trt_libb_step_local_biquad_2d!
export slbm_reg_libb_step_local_2d!
export precompute_q_wall_slbm_cylinder_2d
export compute_local_omega_2d
export PullSLBM, CollideTRTLocalDirect
# 3D SLBM
export CurvilinearMesh3D, build_mesh_3d, validate_mesh_3d
export stretched_box_mesh_3d, cartesian_mesh_3d
export SLBMGeometry3D, build_slbm_geometry_3d, transfer_slbm_geometry_3d
export slbm_bgk_step_3d!
export slbm_trt_libb_step_3d!, slbm_trt_libb_step_local_3d!
export precompute_q_wall_slbm_sphere_3d, compute_local_omega_3d
export PullSLBM_3D, CollideTRTLocalDirect_3D
# gmsh / external mesh import
export load_gmsh_mesh_2d, load_gmsh_mesh_3d, GmshPhysicalGroups

# Multi-block structured (v0.3)
export Block, Interface, MultiBlockMesh2D
export EDGE_SYMBOLS_2D, INTERFACE_TAG
export getblock, edge_length, edge_coords
export MultiBlockSanityIssue, sanity_check_multiblock
export BlockState2D, allocate_block_state_2d, interior_f, interior_macro, ext_dims
export exchange_ghost_2d!, exchange_ghost_shared_node_2d!, fill_ghost_corners_2d!, fill_physical_wall_ghost_2d!, fill_slbm_wall_ghost_2d!
export load_gmsh_multiblock_2d
export reorient_block, autoreorient_blocks, transpose_multiblock
export extend_mesh_2d, build_block_slbm_geometry_extended, extend_interior_field_2d

# Grid refinement
export RefinementPatch, RefinedDomain
export create_patch, create_refined_domain, rescaled_omega
export rescaling_factor_c2f, rescaling_factor_f2c
export prolongate_f_rescaled_2d!, restrict_f_rescaled_2d!
export temporal_interpolate_2d!, copy_macroscopic_overlap_2d!
export advance_refined_step!
export d2q9_cx, d2q9_cy
export d2q9_opposite
export coalesce_F_2d!, explode_uniform_F_2d!
export mass_F, momentum_F, moments_F
export fill_equilibrium_integrated_D2Q9!
export conservative_tree_parent_index
export split_coarse_to_fine_vertical_F_2d!, coalesce_fine_to_coarse_vertical_F
export split_coarse_to_fine_face_F_2d!, coalesce_fine_to_coarse_face_F
export split_coarse_to_fine_corner_F_2d!, coalesce_fine_to_coarse_corner_F
export coarse_to_fine_patch_boundary_F_2d!, fine_to_coarse_patch_boundary_F_2d!
export collide_BGK_integrated_D2Q9!
export collide_Guo_integrated_D2Q9!
export stream_fully_periodic_F_2d!, stream_periodic_x_wall_y_F_2d!
export stream_periodic_x_moving_wall_y_F_2d!
export cylinder_solid_mask_leaf_2d, square_solid_mask_leaf_2d
export backward_facing_step_solid_mask_leaf_2d
export vertical_facing_step_solid_mask_leaf_2d
export stream_periodic_x_wall_y_solid_F_2d!
export stream_bounceback_xy_solid_F_2d!
export apply_zou_he_west_F_2d!, apply_zou_he_pressure_east_F_2d!
export compute_drag_mea_solid_F_2d
export ConservativeTreePatch2D, create_conservative_tree_patch_2d
export coalesce_patch_to_shadow_F_2d!, explode_shadow_to_patch_uniform_F_2d!
export active_population_sums_F, active_mass_F, active_momentum_F, active_moments_F
export composite_to_leaf_F_2d!, leaf_to_composite_F_2d!
export stream_composite_fully_periodic_leaf_F_2d!
export stream_composite_periodic_x_wall_y_leaf_F_2d!
export stream_composite_periodic_x_moving_wall_y_leaf_F_2d!
export collide_BGK_composite_F_2d!, collide_Guo_composite_F_2d!
export LinkKind, SAME_LEVEL, COARSE_TO_FINE, FINE_TO_COARSE, BOUNDARY
export RouteKind, DIRECT, SPLIT_FACE, SPLIT_CORNER, COALESCE_FACE
export COALESCE_CORNER, ROUTE_BOUNDARY
export AbstractCellMetrics, CartesianMetrics2D
export ConservativeTreeCell2D, ConservativeTreeLink2D, ConservativeTreeRoute2D
export ConservativeTreeTopology2D, create_conservative_tree_topology_2d
export ConservativeTreeBlock2D, ConservativeTreePackedRoute2D
export ConservativeTreePackedTopology2D, pack_conservative_tree_topology_2d
export active_volume, morton_key_2d
export stream_composite_routes_interior_F_2d!
export stream_composite_routes_periodic_x_F_2d!
export stream_composite_routes_periodic_x_wall_y_F_2d!
export stream_composite_routes_periodic_x_moving_wall_y_F_2d!
export stream_composite_routes_periodic_x_wall_y_solid_F_2d!
export collide_Guo_composite_solid_F_2d!
export regrid_conservative_tree_patch_F_2d!
export regrid_conservative_tree_patch_direct_F_2d!
export conservative_tree_solid_mask_patch_range_2d
export conservative_tree_indicator_patch_range_2d
export conservative_tree_gradient_indicator_2d
export conservative_tree_hysteresis_patch_range_2d
export conservative_tree_velocity_gradient_patch_range_2d
export adapt_conservative_tree_patch_to_solid_mask_2d
export adapt_conservative_tree_patch_to_velocity_gradient_2d
export ConservativeTreeAdaptiveRun2D
export ConservativeTreeSolidAdaptiveRun2D
export ConservativeTreeMacroFlow2D, ConservativeTreeCylinderResult2D
export ConservativeTreeCylinderChannelResult2D
export ConservativeTreeSolidFlowResult2D
export composite_leaf_mean_ux_profile
export composite_leaf_velocity_field_2d
export couette_analytic_profile_2d, poiseuille_analytic_profile_2d
export run_conservative_tree_couette_macroflow_2d
export run_conservative_tree_poiseuille_macroflow_2d
export run_conservative_tree_couette_route_native_2d
export run_conservative_tree_poiseuille_route_native_2d
export run_conservative_tree_poiseuille_adaptive_route_native_2d
export run_conservative_tree_poiseuille_gradient_adaptive_route_native_2d
export validate_conservative_tree_route_native_phase_p_2d
export run_conservative_tree_square_obstacle_macroflow_2d
export run_conservative_tree_square_obstacle_route_native_2d
export run_conservative_tree_vfs_route_native_2d
export run_conservative_tree_vfs_mask_adaptive_route_native_2d
export run_conservative_tree_bfs_macroflow_2d
export run_conservative_tree_cylinder_macroflow_2d
export run_conservative_tree_cylinder_channel_macroflow_2d
export RefinementPatch3D, RefinedDomain3D
export create_patch_3d, create_refined_domain_3d, advance_refined_step_3d!
export prolongate_f_rescaled_3d!, prolongate_f_rescaled_full_3d!
export prolongate_f_rescaled_temporal_3d!, restrict_f_rescaled_3d!
export ThermalPatchArrays3D, create_thermal_patch_arrays_3d
export advance_thermal_refined_step_3d!, build_patch_thermal_bcs_3d
export fill_thermal_full_3d!, fill_thermal_ghost_3d!, restrict_thermal_to_coarse_3d!
export build_patch_flow_bcs_3d
export TwophaseRefinedArrays, create_twophase_patch_arrays, advance_twophase_refined_step!

# STL geometry
export STLTriangle, STLMesh, read_stl, transform_mesh
export voxelize_2d, voxelize_3d
export precompute_q_wall_from_stl_2d, precompute_q_wall_from_stl_3d

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
export run_viscoelastic_cylinder_2d

# Spatial boundary kernels
export apply_zou_he_north_spatial_2d!, apply_zou_he_south_spatial_2d!
export apply_zou_he_west_spatial_2d!, apply_zou_he_pressure_east_spatial_2d!
export apply_zou_he_pressure_inlet_west_2d!

end # module Kraken
