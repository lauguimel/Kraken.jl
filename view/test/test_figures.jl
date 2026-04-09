# Headless tests for KrakenView figure primitives.

using Test
using CairoMakie
using KrakenView

@testset "KrakenView figures" begin
    tmpdir = mktempdir()

    @testset "heatmap_field" begin
        fig = heatmap_field(rand(32, 32); field_name="test", title="hm")
        @test fig isa Makie.Figure
        axes = [c for c in fig.content if c isa Makie.Axis]
        @test length(axes) == 1
        hms = [p for p in axes[1].scene.plots if p isa Makie.Heatmap]
        @test length(hms) >= 1
    end

    @testset "profile_plot (no reference)" begin
        field = [sin(2π * j / 64) for i in 1:32, j in 1:64]
        fig = profile_plot(field, :vertical; title="p")
        @test fig isa Makie.Figure
        axes = [c for c in fig.content if c isa Makie.Axis]
        lines = [p for p in axes[1].scene.plots if p isa Makie.Lines]
        @test length(lines) == 1
    end

    @testset "profile_plot (with reference)" begin
        field = rand(32, 32)
        fig = profile_plot(field, :horizontal;
                           reference=x -> 0.5,
                           title="p+ref")
        axes = [c for c in fig.content if c isa Makie.Axis]
        lines = [p for p in axes[1].scene.plots if p isa Makie.Lines]
        @test length(lines) >= 2
    end

    @testset "convergence_plot (order-2 fit)" begin
        N = [32, 64, 128, 256]
        err = [1.0, 0.25, 0.0625, 0.015625]
        slope, _ = KrakenView.fit_loglog_slope(N, err)
        @test abs(slope + 2.0) < 0.05
        fig = convergence_plot(N, err; theoretical_order=2.0)
        @test fig isa Makie.Figure
    end

    @testset "streamline_plot" begin
        Nx, Ny = 32, 32
        # Solid-body rotation-ish
        ux = [Float64(j - Ny/2) for i in 1:Nx, j in 1:Ny]
        uy = [Float64(-(i - Nx/2)) for i in 1:Nx, j in 1:Ny]
        fig = streamline_plot(ux, uy; n_lines=6, color_by_speed=true)
        @test fig isa Makie.Figure
        axes = [c for c in fig.content if c isa Makie.Axis]
        lines = [p for p in axes[1].scene.plots if p isa Makie.Lines]
        @test length(lines) >= 4
    end

    @testset "save_figure PNG" begin
        fig = heatmap_field(rand(16, 16))
        path = joinpath(tmpdir, "hm.png")
        save_figure(fig, path)
        @test isfile(path)
        bytes = open(read, path)
        @test length(bytes) >= 8
        # PNG magic: 0x89 0x50 0x4E 0x47
        @test bytes[1] == 0x89 && bytes[2] == 0x50 &&
              bytes[3] == 0x4E && bytes[4] == 0x47
    end

    @testset "generate_figures" begin
        spec = [
            (case="demo", figure_type=:heatmap, output="demo_hm.png",
             options=(colormap=:viridis,),
             data=(; field=rand(32, 32))),
            (case="order2", figure_type=:convergence, output="order2.png",
             options=(theoretical_order=2.0,),
             data=(; N_values=[32, 64, 128, 256],
                     errors=[1.0, 0.25, 0.0625, 0.015625])),
        ]
        paths = generate_figures(spec; output_dir=tmpdir)
        @test length(paths) == 2
        @test all(isfile, paths)
    end
end
