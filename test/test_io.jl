using Test
using Kraken
using WriteVTK: vtk_save

@testset "IO" begin

    @testset "VTK writer" begin
        nx, ny = 8, 8
        dx = 1.0 / nx
        u = rand(nx, ny)
        v = rand(nx, ny)
        p = rand(nx, ny)
        fields = Dict("U" => u, "V" => v, "P" => p)

        tmpdir = mktempdir()
        filepath = joinpath(tmpdir, "test_output")

        # Write VTK file
        outfiles = Kraken.write_vtk(filepath, nx, ny, dx, fields)
        vtr_path = outfiles[1]

        @test isfile(vtr_path)
        @test endswith(vtr_path, ".vtr")

        # Verify file content contains expected field names
        content = read(vtr_path, String)
        @test occursin("U", content)
        @test occursin("V", content)
        @test occursin("P", content)

        # Test PVD time series
        pvd_path = joinpath(tmpdir, "timeseries")
        pvd = Kraken.create_pvd(pvd_path)

        for (i, t) in enumerate([0.0, 0.1, 0.2])
            step_file = joinpath(tmpdir, "step_$i")
            Kraken.write_vtk_to_pvd(pvd, step_file, nx, ny, dx, Dict("P" => p), t)
        end
        vtk_save(pvd)

        pvd_file = pvd_path * ".pvd"
        @test isfile(pvd_file)
        pvd_content = read(pvd_file, String)
        @test occursin("timestep", pvd_content)
    end

    @testset "YAML config parser" begin
        config_path = joinpath(@__DIR__, "..", "examples", "cavity.yaml")
        cfg = Kraken.load_config(config_path)

        # Geometry
        @test cfg.geometry.domain == (1.0, 1.0)
        @test cfg.geometry.resolution == (64, 64)

        # Physics
        @test cfg.physics.Re == 100.0
        @test cfg.physics.equation == "navier-stokes"

        # Boundary conditions
        @test haskey(cfg.boundary_conditions, "top")
        @test cfg.boundary_conditions["top"].type == "wall"
        @test cfg.boundary_conditions["top"].values["velocity"] == [1.0, 0.0]
        @test haskey(cfg.boundary_conditions, "bottom")
        @test haskey(cfg.boundary_conditions, "left")
        @test haskey(cfg.boundary_conditions, "right")

        # Study
        @test cfg.study.type == "transient"
        @test cfg.study.dt == "auto"
        @test cfg.study.max_steps == 10000
        @test cfg.study.convergence_tol == 1.0e-6

        # Output
        @test cfg.output.format == "vtk"
        @test cfg.output.frequency == 100
    end

    @testset "YAML config defaults" begin
        tmpdir = mktempdir()
        minimal_yaml = joinpath(tmpdir, "minimal.yaml")
        open(minimal_yaml, "w") do f
            write(f, "physics:\n  Re: 50.0\n")
        end

        cfg = Kraken.load_config(minimal_yaml)
        @test cfg.geometry.domain == (1.0, 1.0)
        @test cfg.geometry.resolution == (32, 32)
        @test cfg.physics.Re == 50.0
        @test cfg.study.dt == "auto"
        @test cfg.output.format == "vtk"
    end

end
