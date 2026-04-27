"""
    Kraken

GPU-native Lattice Boltzmann Method (LBM) framework in Julia.

Supports 2D (D2Q9) and 3D (D3Q19) simulations with automatic GPU
acceleration via KernelAbstractions.jl.

v0.1.0 scope: single-phase Newtonian + thermal (DDF) flows.
"""
module Kraken

# PNG/GIF output hooks — populated by KrakenMakieExt when CairoMakie is loaded
const _png_saver = Ref{Any}(nothing)
const _gif_saver = Ref{Any}(nothing)

# --- Lattice definitions ---
include("lattice/lattice.jl")
include("lattice/d2q9.jl")
include("lattice/d3q19.jl")

# --- GPU kernels (Newtonian + thermal) ---
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

# --- I/O ---
include("io/vtk_writer.jl")
include("io/diagnostics.jl")
include("io/expression.jl")
include("io/kraken_parser.jl")

# --- Spatial boundary kernels ---
include("kernels/boundary_spatial_2d.jl")

# --- Simulation drivers ---
include("drivers/basic.jl")
include("drivers/thermal.jl")

# --- Generic simulation runner ---
include("simulation_runner.jl")

# --- Post-processing helpers ---
include("postprocess.jl")

# =====================================================================
# Public API
# =====================================================================

# Lattice types and functions
export AbstractLattice, D2Q9, D3Q19
export lattice_dim, lattice_q, weights, velocities_x, velocities_y, velocities_z
export opposite, cs2, equilibrium

# Collision & streaming kernels
export stream_2d!, collide_2d!, stream_3d!, collide_3d!
export stream_periodic_x_wall_y_2d!, stream_fully_periodic_2d!
export collide_guo_2d!, collide_guo_field_2d!
export collide_guo_3d!, collide_guo_field_3d!

# Macroscopic quantities
export compute_macroscopic_2d!, compute_macroscopic_3d!
export compute_macroscopic_forced_2d!, compute_macroscopic_forced_3d!

# Boundary conditions (Zou-He + bounce-back)
export apply_zou_he_north_2d!, apply_zou_he_south_2d!
export apply_zou_he_west_2d!, apply_zou_he_pressure_east_2d!, apply_extrapolate_east_2d!
export apply_zou_he_top_3d!, apply_zou_he_bottom_3d!
export apply_zou_he_west_3d!, apply_zou_he_east_3d!
export apply_zou_he_south_3d!, apply_zou_he_north_3d!
export apply_zou_he_pressure_east_3d!, apply_zou_he_pressure_top_3d!
export apply_bounce_back_walls_2d!, apply_bounce_back_wall_2d!
export apply_bounce_back_walls_3d!, apply_bounce_back_wall_3d!

# Spatial boundary kernels
export apply_zou_he_north_spatial_2d!, apply_zou_he_south_spatial_2d!
export apply_zou_he_west_spatial_2d!, apply_zou_he_pressure_east_spatial_2d!
export apply_zou_he_pressure_inlet_west_2d!

# Thermal kernels (DDF)
export collide_thermal_2d!, compute_temperature_2d!
export collide_thermal_3d!, compute_temperature_3d!
export apply_fixed_temp_south_2d!, apply_fixed_temp_north_2d!
export apply_fixed_temp_west_2d!, apply_fixed_temp_east_2d!
export apply_fixed_temp_west_3d!, apply_fixed_temp_east_3d!
export apply_fixed_temp_south_3d!, apply_fixed_temp_north_3d!
export apply_fixed_temp_bottom_3d!, apply_fixed_temp_top_3d!
export collide_boussinesq_2d!, collide_boussinesq_vt_2d!, collide_boussinesq_vt_modified_2d!
export collide_boussinesq_3d!
export fused_natconv_step!, fused_natconv_vt_step!

# Simulation drivers
export LBMConfig, omega, reynolds
export initialize_2d, initialize_3d
export run_cavity_2d, run_cavity_3d
export run_poiseuille_2d, run_couette_2d
export initialize_taylor_green_2d, run_taylor_green_2d
export initialize_cylinder_2d, run_cylinder_2d, compute_drag_mea_2d
export run_rayleigh_benard_2d, run_natural_convection_2d
export run_natural_convection_3d

# I/O
export write_vtk, create_pvd, write_vtk_to_pvd
export setup_output_dir, write_snapshot_2d!, write_snapshot_3d!
export open_paraview
export DiagnosticsLogger, open_diagnostics, log_diagnostics!, close_diagnostics!

# .krk config system
export KrakenExpr, parse_kraken_expr, evaluate, has_variable, is_time_dependent, is_spatial
export SimulationSetup, DomainSetup, PhysicsSetup, GeometryRegion, BoundarySetup
export InitialSetup, OutputSetup, DiagnosticsSetup, SanityIssue
export load_kraken, parse_kraken,
       parse_kraken_sweep, load_kraken_sweep, sanity_check, sanity_check_sweep
export LBMParams, lbm_params, lbm_params_table
export run_simulation

# Post-processing
export extract_line, field_error, probe, domain_stats

end # module Kraken
