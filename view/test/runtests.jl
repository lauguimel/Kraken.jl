using Test
using CairoMakie  # headless backend for CI
CairoMakie.activate!()

using KrakenView
using Kraken

const CAVITY = joinpath(@__DIR__, "..", "..", "examples", "cavity.krk")

@testset "KrakenView smoke" begin
    @test isfile(CAVITY)

    scene = view_krk(CAVITY)
    @test scene isa KrakenView.KrakenScene
    @test scene.figure isa Makie.Figure

    # Figure should contain at least one Axis
    axes = [c for c in scene.figure.content if c isa Makie.Axis]
    @test length(axes) >= 1

    # Field Observable starts as zero matrix of right size
    Nx, Ny = scene.setup.domain.Nx, scene.setup.domain.Ny
    @test size(scene.field[]) == (Nx, Ny)

    # Exercise callback with synthetic state
    cb = KrakenView.make_callback(scene)
    fake_ux = fill(0.1, Nx, Ny)
    fake_uy = fill(0.2, Nx, Ny)
    fake_rho = ones(Nx, Ny)
    cb(100, (; rho=fake_rho, ux=fake_ux, uy=fake_uy))

    @test maximum(scene.field[]) > 0            # Observable updated
    @test length(scene.snapshots) == 1           # snapshot stored
    @test scene.snapshots[1].step == 100

    # 3D rejection: synthesize a D3Q19 setup by swapping lattice
    @test_throws ErrorException KrakenView.build_scene(
        (; lattice=:D3Q19, setup=nothing,
           domain=scene.setup.domain, name="x",
           boundaries=scene.setup.boundaries,
           regions=scene.setup.regions))
end
