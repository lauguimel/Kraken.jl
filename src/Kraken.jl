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

export greet

end # module Kraken
