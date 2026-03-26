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

# --- Simulation ---
include("simulation.jl")

# --- I/O ---
include("io/vtk_writer.jl")

# Lattice types and functions
export AbstractLattice, D2Q9, D3Q19
export lattice_dim, lattice_q, weights, velocities_x, velocities_y, velocities_z
export opposite, cs2, equilibrium

# Kernels
export stream_2d!, collide_2d!, stream_3d!, collide_3d!
export stream_periodic_x_wall_y_2d!, stream_fully_periodic_2d!
export collide_guo_2d!
export compute_macroscopic_2d!, compute_macroscopic_3d!, compute_macroscopic_forced_2d!
export apply_zou_he_north_2d!, apply_zou_he_south_2d!
export apply_zou_he_west_2d!, apply_extrapolation_east_2d!
export apply_zou_he_top_3d!

# Simulation
export LBMConfig, omega, reynolds
export initialize_2d, initialize_3d
export run_cavity_2d, run_cavity_3d
export run_poiseuille_2d, run_couette_2d
export initialize_taylor_green_2d, run_taylor_green_2d
export initialize_cylinder_2d, run_cylinder_2d, compute_drag_2d

# I/O
export write_vtk, create_pvd, write_vtk_to_pvd

end # module Kraken
