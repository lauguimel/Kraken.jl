using Test
using Kraken

@testset "Kraken.jl v0.1.0" begin
    include("test_lbm_basic.jl")
    include("test_poiseuille.jl")
    include("test_poiseuille_3d.jl")
    include("test_couette.jl")
    include("test_taylor_green.jl")
    include("test_thermal.jl")
    include("test_cavity.jl")
    include("test_cavity_3d.jl")
    include("test_cylinder.jl")
    include("test_expression.jl")
    include("test_kraken_parser.jl")
    include("test_simulation_runner.jl")
    include("test_krk_examples.jl")
    include("test_vtk_3d.jl")
    include("test_postprocess.jl")
end
