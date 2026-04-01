using WriteVTK: vtk_save

@testset "VTK 3D snapshot" begin
    Nx, Ny, Nz = 4, 4, 4
    dx = 0.25

    mktempdir() do tmpdir
        # --- Scalar fields ---
        rho = ones(Float64, Nx, Ny, Nz)
        ux  = rand(Float64, Nx, Ny, Nz)

        fields = Dict{String,Any}("rho" => rho, "ux" => ux)
        write_snapshot_3d!(tmpdir, 100, Nx, Ny, Nz, dx, fields)

        vtr_file = joinpath(tmpdir, "snapshot_0000100.vtr")
        @test isfile(vtr_file)

        # --- PVD time-series ---
        pvd_path = joinpath(tmpdir, "timeseries")
        pvd = create_pvd(pvd_path)
        write_snapshot_3d!(tmpdir, 0, Nx, Ny, Nz, dx, fields; pvd=pvd, time=0.0)
        write_snapshot_3d!(tmpdir, 1, Nx, Ny, Nz, dx, fields; pvd=pvd, time=0.1)
        vtk_save(pvd)

        @test isfile(joinpath(tmpdir, "snapshot_0000000.vtr"))
        @test isfile(joinpath(tmpdir, "snapshot_0000001.vtr"))
        @test isfile(pvd_path * ".pvd")

        # --- Vector field (velocity tuple) ---
        uy = rand(Float64, Nx, Ny, Nz)
        uz = rand(Float64, Nx, Ny, Nz)
        vec_fields = Dict{String,Any}("velocity" => (ux, uy, uz), "rho" => rho)
        write_snapshot_3d!(tmpdir, 200, Nx, Ny, Nz, dx, vec_fields)

        vtr_vec = joinpath(tmpdir, "snapshot_0000200.vtr")
        @test isfile(vtr_vec)
    end
end
