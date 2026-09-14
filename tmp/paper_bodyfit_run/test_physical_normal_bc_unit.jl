using Test

include(joinpath(@__DIR__, "paper_physical_normal_bc_unit.jl"))

@testset "paper physical-normal BC unit" begin
    results = pb_run_all()

    @test length(results.velocity) == 16
    @test length(results.pressure) == 16
    @test length(results.wall) == 16
    @test length(results.gmsh) == 6

    for group in (results.velocity, results.pressure, results.wall, results.gmsh)
        pop_err, moment_err = pb_group_error(group)
        @test pop_err < 1e-14
        @test moment_err < 1e-14
    end
end
