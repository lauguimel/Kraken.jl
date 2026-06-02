using Test

include(joinpath(dirname(@__DIR__), "benchmarks",
                 "amr_d_quicklook_from_krk_2d.jl"))

@testset "AMR D .krk quicklook artifacts" begin
    mktempdir() do dir
        artifacts = run_amr_d_quicklook_from_krk_2d(
            joinpath(dirname(@__DIR__), "benchmarks", "krk",
                     "amr_d_convergence_2d",
                     "poiseuille_nested4_channel.krk");
            outdir=dir, steps_override=1, include_reference=false,
            make_plots=false)

        @test length(artifacts) == 1
        a = only(artifacts)
        @test a.method == :amr_d
        @test a.status == :ok
        @test isfile(a.status_csv)
        @test isfile(a.mesh_csv)
        @test isfile(a.fields_csv)
        @test isfile(a.profiles_csv)
        @test isfile(joinpath(dir, "summary.csv"))
        @test length(readlines(a.fields_csv)) > 10
        @test occursin("mean_y", read(a.profiles_csv, String))
    end

    mktempdir() do dir
        artifacts = run_amr_d_quicklook_from_krk_2d(
            joinpath(dirname(@__DIR__), "benchmarks", "krk",
                     "amr_d_convergence_2d",
                     "poiseuille_nested4_channel.krk");
            outdir=dir, steps_override=1, include_reference=true,
            make_plots=false)

        @test length(artifacts) == 2
        @test Set(a.method for a in artifacts) == Set([:amr_d, :cartesian_classic])
        case_dir = first(artifacts).outdir
        values_csv = joinpath(case_dir, "values.csv")
        @test isfile(values_csv)
        values_text = read(values_csv, String)
        @test occursin("linf_profile_vs_reference", values_text)
        @test occursin("cartesian_classic", values_text)
        @test occursin("speed_max", values_text)
    end

    mktempdir() do dir
        nx, ny = 5, 4
        y = collect(range(0.0, 1.0; length=ny))
        analytic = [0.01 * yy * (1 - yy) for yy in y]
        ux_cart = [analytic[j] for i in 1:nx, j in 1:ny]
        ux_amr = [analytic[j] * (i <= 3 ? 1.0 : 1.05)
                  for i in 1:nx, j in 1:ny]
        cart_state = (;
            fields=(; rho=ones(nx, ny), ux=ux_cart, uy=zeros(nx, ny),
                    speed=abs.(ux_cart)),
            is_solid=falses(nx, ny),
            level=fill(2, nx, ny),
            patch=nothing,
            leaf_nx=nx,
            leaf_ny=ny)
        amr_state = (;
            fields=(; rho=ones(nx, ny) .+ 1e-6 .* ux_amr, ux=ux_amr,
                    uy=zeros(nx, ny), speed=abs.(ux_amr)),
            is_solid=falses(nx, ny),
            level=[i <= 3 ? 1 : 2 for i in 1:nx, j in 1:ny],
            patch=nothing,
            leaf_nx=nx,
            leaf_ny=ny)
        cart_result = (ux_profile=analytic, analytic_ux_profile=analytic)
        amr_result = (ux_profile=vec(sum(ux_amr; dims=1) ./ nx),
                      analytic_ux_profile=analytic)
        path = joinpath(dir, "debug_dashboard.png")
        _ql_plot_debug_dashboard(path, amr_result, amr_state,
                                 cart_result, cart_state; title="unit")
        @test isfile(path)
    end

    mktempdir() do dir
        artifacts = run_amr_d_quicklook_from_krk_2d(
            joinpath(dirname(@__DIR__), "benchmarks", "krk",
                     "amr_d_convergence_2d",
                     "poiseuille_xband_nested4_debug.krk");
            outdir=dir, steps_override=1, include_reference=true,
            make_plots=false)

        @test length(artifacts) == 2
        @test Set(a.method for a in artifacts) == Set([:amr_d, :cartesian_classic])
        mesh_text = read(joinpath(first(artifacts).outdir, "mesh_amr_d.csv"), String)
        @test occursin("\n4,", mesh_text)
    end

    mktempdir() do dir
        artifacts = run_amr_d_quicklook_from_krk_2d(
            joinpath(dirname(@__DIR__), "benchmarks", "krk",
                     "amr_d_convergence_2d",
                     "cylinder_lift_nested4_probe.krk");
            outdir=dir, steps_override=1, include_reference=false,
            make_plots=false)

        @test length(artifacts) == 1
        a = only(artifacts)
        @test a.method == :none
        @test a.status == :nested_obstacle_runtime_pending
        @test isfile(a.status_csv)
        @test isfile(a.mesh_csv)
        setup = load_kraken(joinpath(dirname(@__DIR__), "benchmarks", "krk",
                                     "amr_d_convergence_2d",
                                     "cylinder_lift_nested4_probe.krk"))
        case = conservative_tree_amr_d_case_from_krk_2d(setup)
        mask = _ql_static_solid_mask(setup, case, 512, 256)
        @test any(mask)
        @test a.fields_csv == ""
        @test a.profiles_csv == ""
    end
end
