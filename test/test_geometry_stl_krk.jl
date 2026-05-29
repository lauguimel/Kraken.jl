# KRK-GEO — end-to-end test for the Cartesian immersed-boundary STL path.
#
# An `Obstacle … stl(file=…)` declared in a .krk file must voxelize into the
# same is_solid mask as the analytical `Obstacle … { (x-cx)^2+(y-cy)^2 <= R^2 }`,
# and `run_simulation` must complete and reproduce the analytical-cylinder flow.
#
# This test also guards the M-GEO-2 regression fix: `run_simulation` references
# `setup.mesh`, so it fails with `FieldError: no field mesh` if the `mesh` field
# is ever dropped from `SimulationSetup`.

using Kraken
using Test

@testset "Geometry: STL obstacle .krk (Cartesian immersed boundary)" begin
    root        = normpath(joinpath(@__DIR__, ".."))
    example_dir = joinpath(root, "examples", "geometry_stl")
    stl_case    = joinpath(example_dir, "cylinder_stl_flow.krk")
    anal_case   = joinpath(example_dir, "cylinder_analytic_flow.krk")
    stl_file    = joinpath(example_dir, "cylinder.stl")
    output_dir  = joinpath(root, "output")

    @test isfile(stl_file)

    mask_of(setup) = begin
        dom = setup.domain
        dx = dom.Lx / dom.Nx
        dy = dom.Ly / dom.Ny
        m  = falses(dom.Nx, dom.Ny)
        Kraken._apply_geometry!(m, setup, dx, dy)
        (m, dx, dy)
    end

    # .krk stl(file=…) paths are repo-root-relative; resolve from there.
    cd(root) do
        stl_setup  = load_kraken(stl_case)
        anal_setup = load_kraken(anal_case)

        stl_mask, dx, dy   = mask_of(stl_setup)
        anal_mask, dx2, dy2 = mask_of(anal_setup)
        @test (dx, dy) == (dx2, dy2)

        # --- GATE-A: voxelized STL mask matches the analytical disc (rim band only)
        dom   = anal_setup.domain
        total = dom.Nx * dom.Ny
        disagree = stl_mask .!= anal_mask
        ndis  = sum(disagree)
        cx, cy = anal_setup.user_vars[:cx], anal_setup.user_vars[:cy]
        R      = anal_setup.user_vars[:R]
        maxrim = 0.0
        for j in 1:dom.Ny, i in 1:dom.Nx
            disagree[i, j] || continue
            x = (i - 0.5) * dx
            y = (j - 0.5) * dy
            maxrim = max(maxrim, abs(hypot(x - cx, y - cy) - R))
        end
        @test 100.0 * ndis / total <= 2.0            # ≤2% of cells disagree
        @test maxrim <= 1.5 * dx + 100 * eps(Float64) # …and only in the rim band

        # --- GATE-B: STL run completes, writes VTK, and reproduces the flow
        pvd_stl = joinpath(output_dir, stl_setup.name * ".pvd")
        isfile(pvd_stl) && rm(pvd_stl; force=true)   # ensure freshness

        stl_res  = run_simulation(stl_setup)
        anal_res = run_simulation(anal_setup)

        @test isfile(pvd_stl)                        # end-to-end: VTK written

        common = .!(stl_mask .| anal_mask)
        @test any(common)
        sp_stl = maximum(sqrt.(stl_res.ux[common] .^ 2 .+ stl_res.uy[common] .^ 2))
        sp_an  = maximum(sqrt.(anal_res.ux[common] .^ 2 .+ anal_res.uy[common] .^ 2))
        denom  = max(abs(sp_stl), abs(sp_an), eps(Float64))
        @test 100.0 * abs(sp_stl - sp_an) / denom <= 2.0
    end
end
