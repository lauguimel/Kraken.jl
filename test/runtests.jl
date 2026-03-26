using Test
using Kraken

@testset "Kraken.jl LBM" begin
    include("test_lbm_basic.jl")
    include("test_cavity.jl")
end
