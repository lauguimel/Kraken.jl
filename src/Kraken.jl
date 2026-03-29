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

# --- GPU kernels ---
include("kernels/collide_stream_2d.jl")
include("kernels/collide_stream_3d.jl")
include("kernels/stream_periodic_2d.jl")
include("kernels/collide_guo_2d.jl")
include("kernels/macroscopic.jl")
include("kernels/boundary_2d.jl")
include("kernels/boundary_3d.jl")
include("kernels/thermal_2d.jl")
include("kernels/collide_mrt_2d.jl")
include("kernels/species_2d.jl")
include("kernels/multiphase_2d.jl")
include("kernels/vof_2d.jl")
include("kernels/dualgrid_2d.jl")
include("kernels/levelset_2d.jl")
include("kernels/clsvof_2d.jl")

# --- Simulation ---
include("simulation.jl")

# --- I/O ---
include("io/vtk_writer.jl")
include("io/diagnostics.jl")
include("io/expression.jl")
include("io/kraken_parser.jl")

# --- Spatial boundary kernels ---
include("kernels/boundary_spatial_2d.jl")

# --- Generic simulation runner ---
include("simulation_runner.jl")

# Lattice types and functions
export AbstractLattice, D2Q9, D3Q19
export lattice_dim, lattice_q, weights, velocities_x, velocities_y, velocities_z
export opposite, cs2, equilibrium

# Kernels
export stream_2d!, collide_2d!, stream_3d!, collide_3d!
export stream_periodic_x_wall_y_2d!, stream_fully_periodic_2d!, stream_periodic_x_axisym_2d!
export collide_guo_2d!, collide_guo_field_2d!
export compute_macroscopic_2d!, compute_macroscopic_3d!, compute_macroscopic_forced_2d!
export apply_zou_he_north_2d!, apply_zou_he_south_2d!
export apply_zou_he_west_2d!, apply_zou_he_pressure_east_2d!
export apply_zou_he_top_3d!

# Simulation
export LBMConfig, omega, reynolds
export initialize_2d, initialize_3d
export run_cavity_2d, run_cavity_3d
export run_poiseuille_2d, run_couette_2d
export initialize_taylor_green_2d, run_taylor_green_2d
export initialize_cylinder_2d, run_cylinder_2d, compute_drag_mea_2d
export collide_thermal_2d!, compute_temperature_2d!
export apply_fixed_temp_south_2d!, apply_fixed_temp_north_2d!
export run_rayleigh_benard_2d
export collide_boussinesq_2d!, collide_boussinesq_vt_2d!
export collide_axisymmetric_2d!, collide_li_axisym_2d!, run_hagen_poiseuille_2d

# MRT
export collide_mrt_2d!

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
export add_azimuthal_curvature_2d!, run_rp_axisym_2d

# Dual-grid VOF
export prolongate_bilinear_2d!, restrict_average_2d!
export compute_hf_curvature_dx_2d!, compute_surface_tension_dx_2d!
export run_static_droplet_dualgrid_2d

# Level-set
export advect_ls_2d!, reinit_ls_2d!, curvature_ls_2d!, add_azimuthal_curvature_ls_2d!

# CLSVOF
export ls_from_vof_2d!, surface_tension_clsvof_2d!
export run_static_droplet_clsvof_2d, run_rp_clsvof_2d

# I/O
export write_vtk, create_pvd, write_vtk_to_pvd
export setup_output_dir, write_snapshot_2d!
export DiagnosticsLogger, open_diagnostics, log_diagnostics!, close_diagnostics!

# .krk config system
export KrakenExpr, parse_kraken_expr, evaluate, has_variable, is_time_dependent, is_spatial
export SimulationSetup, DomainSetup, PhysicsSetup, GeometryRegion, BoundarySetup
export InitialSetup, OutputSetup, DiagnosticsSetup
export load_kraken, parse_kraken
export run_simulation

# Spatial boundary kernels
export apply_zou_he_north_spatial_2d!, apply_zou_he_south_spatial_2d!
export apply_zou_he_west_spatial_2d!, apply_zou_he_pressure_east_spatial_2d!

end # module Kraken
