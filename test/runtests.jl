using Test
using Kraken

@testset "Kraken.jl LBM" begin
    include("test_lbm_basic.jl")
    include("test_poiseuille.jl")
    include("test_couette.jl")
    include("test_taylor_green.jl")
    include("test_thermal.jl")
    include("test_axisymmetric.jl")
    include("test_mrt.jl")
    include("test_species.jl")
    include("test_multiphase.jl")
    include("test_vof.jl")
    include("test_clsvof.jl")
    include("test_benchmark.jl")
    include("test_cavity.jl")
    include("test_cylinder.jl")
    include("test_expression.jl")
    include("test_kraken_parser.jl")
    include("test_simulation_runner.jl")
    include("test_stl.jl")
    include("test_krk_examples.jl")
end
