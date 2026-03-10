"""
    Kraken

GPU-native multi-physics CFD framework in Julia.

Kraken.jl provides composable operators for computational fluid dynamics
simulations with automatic GPU acceleration via KernelAbstractions.jl.
"""
module Kraken

"""
    greet() -> String

Return the package name string.

# Returns
- `String`: the string `"Kraken.jl"`.
"""
greet() = "Kraken.jl"

include("operators/laplacian.jl")
include("operators/gradient.jl")
include("operators/divergence.jl")
include("operators/advection.jl")
include("solvers/poisson_fft.jl")
include("solvers/poisson_cg.jl")
include("solvers/helmholtz.jl")
include("solvers/multigrid.jl")
include("solvers/projection.jl")
include("io/vtk_writer.jl")
include("io/config_parser.jl")
include("physics/boussinesq.jl")
include("amr/quadtree.jl")
include("amr/operators.jl")
include("amr/poisson_amr.jl")
include("amr/projection_amr.jl")

export greet, laplacian!, gradient!, divergence!, advect!
export solve_poisson_fft!, solve_poisson_cg!, solve_poisson_neumann!, solve_poisson_neumann_dct!, solve_poisson_mg!
export solve_helmholtz!, solve_helmholtz_dct!
export projection_step!, projection_step_implicit!, run_cavity, apply_velocity_bc!, apply_pressure_neumann_bc!, available_backends
export advance_temperature!, buoyancy_force!, run_rayleigh_benard
export write_vtk, create_pvd, write_vtk_to_pvd
export load_config, SimulationConfig, GeometryConfig, PhysicsConfig
export BCConfig, OutputConfig, StudyConfig
export QuadTree, add_field!, get_field, set_field!, cell_size, cell_center
export refine!, coarsen!, adapt!, enforce_balance!
export find_neighbor, foreach_leaf, nleaves, initialize_field!, rebuild_leaf_list!
export laplacian_amr!, divergence_amr!, gradient_amr!, advect_amr!
export neighbor_value, neighbor_distance
export refine_uniformly!, solve_poisson_amr!, vcycle_amr!
export compute_residual_all!, compute_residual_level!, residual_norm
export smooth_level!, restrict_level!, prolongate_level!
export run_cavity_amr, projection_step_amr!, apply_velocity_bc_amr!

end # module Kraken
