using Test
using Kraken

@testset "Kraken.jl LBM" begin
    include("test_lbm_basic.jl")
    include("test_poiseuille.jl")
    include("test_couette.jl")
    include("test_taylor_green.jl")
    include("test_thermal.jl")
    include("test_axisymmetric.jl")
    include("test_cavity.jl")
    include("test_cylinder.jl")
end
