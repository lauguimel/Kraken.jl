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
include("solvers/projection.jl")

export greet, laplacian!, gradient!, divergence!, advect!
export solve_poisson_fft!, solve_poisson_cg!, solve_poisson_neumann!
export projection_step!, run_cavity, apply_velocity_bc!

end # module Kraken
