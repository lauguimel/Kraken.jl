using Test
using Kraken

@testset "Kraken.jl" begin
    @test Kraken.greet() == "Kraken.jl"
end

include("test_laplacian.jl")
