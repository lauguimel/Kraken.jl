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
export compute_macroscopic_2d!, compute_macroscopic_3d!
export apply_zou_he_north_2d!
export apply_zou_he_top_3d!

# Simulation
export LBMConfig, omega, reynolds
export initialize_2d, initialize_3d
export run_cavity_2d, run_cavity_3d

# I/O
export write_vtk, create_pvd, write_vtk_to_pvd

end # module Kraken
