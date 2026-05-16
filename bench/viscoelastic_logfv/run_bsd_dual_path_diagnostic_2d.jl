#!/usr/bin/env julia

using Kraken
using Printf
using Test

const KernelAbstractions = Kraken.KernelAbstractions

const CAVITY_N = 32
const CAVITY_END_TIME = 2.0
const CAVITY_NU_S = 0.1
const CAVITY_NU_P = 0.1
const CAVITY_U_MAX = 0.005
const CAVITY_BSD_FRACTION = 0.75
const CAVITY_MODEL = :oldroydb
const CAVITY_BSD_KIND = :fd

function bsd_dual_output_dir()
    root = normpath(joinpath(@__DIR__, "..", ".."))
    dir = joinpath(root, "results", "viscoelastic_logfv", "bsd_dual_path_diagnostic")
    mkpath(dir)
    return dir
end

function bsd_dual_csv_path(lambda_phys::Float64)
    filename = "bsd_dual_path_lambda$(lambda_phys)_N$(CAVITY_N)_t2.csv"
    return joinpath(bsd_dual_output_dir(), filename)
end

function write_bsd_dual_csv(path::AbstractString, series)
    open(path, "w") do io
        println(io, "step,time,rel_L2")
        for (step, t_phys, rel_l2) in series
            @printf(io, "%d,%.16e,%.16e\n", step, t_phys, rel_l2)
        end
    end
    return nothing
end

function run_bsd_dual_case(lambda_phys::Float64)
    result = run_viscoelastic_logfv_cavity_coupled_2d(;
        N=CAVITY_N,
        end_time=CAVITY_END_TIME,
        nu_s=CAVITY_NU_S,
        nu_p=CAVITY_NU_P,
        u_max=CAVITY_U_MAX,
        lambda_phys=lambda_phys,
        bsd_fraction=CAVITY_BSD_FRACTION,
        polymer_model=CAVITY_MODEL,
        bsd_kind=CAVITY_BSD_KIND,
        diagnose_bsd_dual=true,
        sample_times=Float64[],
        diagnostic_stride=0,
        backend=KernelAbstractions.CPU(),
    )

    series = result.bsd_dual_path_diagnostic
    rels = [rel_l2 for (_, _, rel_l2) in series]
    @assert !isempty(rels) "empty BSD dual-path diagnostic series"
    @assert all(isfinite, rels) "non-finite BSD dual-path relative L2"

    avg_rel_l2 = sum(rels) / length(rels)
    max_rel_l2 = maximum(rels)
    path = bsd_dual_csv_path(lambda_phys)
    write_bsd_dual_csv(path, series)

    @printf(
        "RESULT lambda_phys=%.1e N=%d t=%.1f n_steps=%d avg_rel_L2=%.16e max_rel_L2=%.16e\n",
        lambda_phys, CAVITY_N, CAVITY_END_TIME, length(series), avg_rel_l2, max_rel_l2,
    )

    return (; lambda_phys, n_steps=length(series), avg_rel_l2, max_rel_l2, path, rels)
end

function run_self_test()
    summaries = NamedTuple[]
    @testset "M14 BSD dual-path diagnostic" begin
        for lambda_phys in (0.001, 1.0)
            summary = run_bsd_dual_case(lambda_phys)
            push!(summaries, summary)
            @test summary.n_steps > 0
            @test all(isfinite, summary.rels)
            @test isfinite(summary.avg_rel_l2)
            @test isfinite(summary.max_rel_l2)
        end
    end
    return summaries
end

function main(args::Vector{String}=ARGS)
    try
        if isempty(args) || (length(args) == 1 && first(args) == "--self-test")
            run_self_test()
        else
            error("usage: julia --project=. $(PROGRAM_FILE) [--self-test]")
        end
    catch err
        println(stderr, "BSD dual-path diagnostic failed: ", sprint(showerror, err))
        exit(1)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
