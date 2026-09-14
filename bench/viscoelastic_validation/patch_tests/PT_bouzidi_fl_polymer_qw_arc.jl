# M47 H1 discriminator: straightened q_w arc -> FVFD ∇u -> log-FV C.
using DelimitedFiles
using KernelAbstractions
using Kraken

const OUTDIR = joinpath("scratch", "M47_runverdict")
const CSV = joinpath(OUTDIR, "PT_bouzidi_fl_polymer_qw_arc.csv")
const Nx = 64
const Ny = 4
const U_INF = 0.005
const R_CYL = 60.0
const LAMBDA = 12_000.0
const NSTEPS = 1_000
const LOG_EVERY = 100

function trace_from_log_fields(psixx, psixy, psiyy)
    trace_c = similar(psixx)
    for idx in eachindex(psixx)
        cxx, _, cyy = Kraken.logfv_exp_sym2_2d(psixx[idx], psixy[idx], psiyy[idx])
        trace_c[idx] = cxx + cyy
    end
    return trace_c
end

function build_arc_velocity(rule::Symbol, q_w::Vector{Float64})
    ux = zeros(Float64, Nx, Ny)
    uy = zeros(Float64, Nx, Ny)
    for i in 1:Nx
        theta = 2pi * (i - 1) / Nx
        smooth_wall = 0.5 * U_INF * cos(theta)
        for j in 1:Ny
            r = R_CYL + j - 1
            ux[i, j] = U_INF * cos(theta) * (1 - R_CYL / r)
            uy[i, j] = 0.25 * U_INF * sin(theta) * (1 - R_CYL / r)
        end
        ux[i, 1] = rule === :halfwayBB ? smooth_wall : smooth_wall + U_INF * cos(theta) * (q_w[i] - 0.5)
    end
    return ux, uy
end

function run_case(rule::Symbol, q_w::Vector{Float64})
    ux, uy = build_arc_velocity(rule, q_w)
    is_solid = falses(Nx, Ny)
    is_solid[:, 0 + 1] .= false
    bc = Kraken.fvfd_periodicx_wally_bcspec_2d()
    dudx, dudy, dvdx, dvdy = (zeros(Float64, Nx, Ny) for _ in 1:4)
    Kraken.fvfd_velocity_gradient_2d!(dudx, dudy, dvdx, dvdy, ux, uy, is_solid, 1.0, 1.0, bc)

    psixx = zeros(Float64, Nx, Ny); psixy = zeros(Float64, Nx, Ny); psiyy = zeros(Float64, Nx, Ny)
    nxtx = similar(psixx); nxty = similar(psixy); ntyy = similar(psiyy)
    model_code = Kraken.logfv_constitutive_model_code(:oldroydb)
    rows = Vector{NTuple{5, Float64}}()
    baseline_cells = findall(abs.(q_w .- 0.5) .< 1e-12)
    isempty(baseline_cells) && (baseline_cells = [argmin(abs.(q_w .- 0.5))])

    for step in 0:NSTEPS
        if step % LOG_EVERY == 0
            tc = trace_from_log_fields(psixx, psixy, psiyy)
            wall = tc[:, 1]
            base = maximum(abs.(wall[baseline_cells] .- 2.0))
            raw = maximum(wall)
            resid = maximum(abs.(wall .- 2.0)) - base
            push!(rows, (Float64(step), raw, base, resid, maximum(abs.(dudy[:, 1]))))
        end
        step == NSTEPS && break
        Kraken.logfv_step_constitutive_log_2d!(
            nxtx, nxty, ntyy, psixx, psixy, psiyy,
            dudx, dudy, dvdx, dvdy, LAMBDA, 1.0, model_code, Inf,
        )
        psixx, nxtx = nxtx, psixx
        psixy, nxty = nxty, psixy
        psiyy, ntyy = ntyy, psiyy
    end
    return rows, trace_from_log_fields(psixx, psixy, psiyy)
end

function run_PT_bouzidi_fl_polymer_qw_arc()::Bool
    mkpath(OUTDIR)
    q_w = [0.5 + 0.45 * sin(2pi * (i - 1) / Nx) for i in 1:Nx]
    rows_a, tc_a = run_case(:halfwayBB, q_w)
    rows_b, tc_b = run_case(:bouzidi_fl, q_w)
    open(CSV, "w") do io
        println(io, "case,step,max_trace_C,baseline_qw05_excess,residual_excess,max_abs_dudy_wall")
        for row in rows_a
            println(io, join(("halfwayBB", row...), ","))
        end
        for row in rows_b
            println(io, join(("bouzidi_fl_qw_arc", row...), ","))
        end
    end
    max_trace_a = maximum(tc_a[:, 1])
    max_trace_b = maximum(tc_b[:, 1])
    delta_ba = maximum(tc_b[:, 1] .- tc_a[:, 1])
    criterion = delta_ba > 10 * max(1e-12, maximum(abs.(tc_a[:, 1] .- 2.0)))
    println("PT_bouzidi_fl_polymer_qw_arc: max_trace_A=$(max_trace_a), max_trace_B=$(max_trace_b), delta_BA=$(delta_ba), pass=$(criterion)")
    return criterion
end

const PT_RESULT = run_PT_bouzidi_fl_polymer_qw_arc()
exit(PT_RESULT ? 0 : 1)
