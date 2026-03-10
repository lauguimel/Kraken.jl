using Test
using Kraken

@testset "Kraken.jl" begin
    @test Kraken.greet() == "Kraken.jl"
end

include("test_laplacian.jl")
include("test_poisson.jl")
include("test_cavity.jl")
include("test_io.jl")
include("test_multigrid.jl")
include("test_advection.jl")
include("test_helmholtz.jl")
include("test_boussinesq.jl")
include("test_quadtree.jl")
include("test_amr_operators.jl")
include("test_poisson_amr.jl")
include("test_cavity_amr.jl")
