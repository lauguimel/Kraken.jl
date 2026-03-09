using Test
using Kraken

@testset "Kraken.jl" begin
    @test Kraken.greet() == "Kraken.jl"
end

include("test_laplacian.jl")
include("test_poisson.jl")
include("test_cavity.jl")
include("test_io.jl")
