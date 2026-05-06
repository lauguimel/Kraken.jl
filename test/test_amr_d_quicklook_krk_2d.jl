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
        @test Set(a.method for a in artifacts) == Set([:amr_d, :leaf_cartesian])
        case_dir = first(artifacts).outdir
        values_csv = joinpath(case_dir, "values.csv")
        @test isfile(values_csv)
        values_text = read(values_csv, String)
        @test occursin("linf_profile_vs_reference", values_text)
        @test occursin("leaf_cartesian", values_text)
    end

    mktempdir() do dir
        artifacts = run_amr_d_quicklook_from_krk_2d(
            joinpath(dirname(@__DIR__), "benchmarks", "krk",
                     "amr_d_showoff_2d",
                     "cylinder_lift_re100_long_channel.krk");
            outdir=dir, steps_override=1, include_reference=false,
            make_plots=false)

        @test length(artifacts) == 1
        a = only(artifacts)
        @test a.method == :none
        @test a.status == :nested_obstacle_runtime_pending
        @test isfile(a.status_csv)
        @test isfile(a.mesh_csv)
        @test a.fields_csv == ""
        @test a.profiles_csv == ""
    end
end
