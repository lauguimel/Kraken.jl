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
include("solvers/poisson_fft.jl")
include("solvers/poisson_cg.jl")

export greet, laplacian!, solve_poisson_fft!, solve_poisson_cg!

end # module Kraken
