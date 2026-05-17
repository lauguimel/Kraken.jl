#!/usr/bin/env julia

# M23: confined-cylinder Cd convergence with BSD disabled.

using KernelAbstractions
using Kraken
using Kraken: logfv_exp_sym2_2d
using Printf

const RHEOTOOL_CD = Dict(
    0.1 => 130.427804221,
    0.2 => 126.839657948,
)

const CSV_COLUMNS = [
    :mission, :backend, :FT, :R, :Wi, :bsd_fraction, :beta, :u_mean, :Re_R,
    :nu_total, :nu_s, :nu_p, :lambda, :max_steps, :polymer_substeps_used,
    :Cd_kraken, :Cd_s, :Cd_p, :Cd_bsd, :Cd_rheotool, :Cd_err_pct,
    :min_det_C, :u_max_abs, :tau_xx_max_abs, :tau_xy_max_abs, :tau_yy_max_abs,
    :n_drag_samples, :completed_steps, :first_nonfinite_step,
    :first_nonfinite_field, :nan_flag, :walltime_s, :MLUPS,
]

const METAL_MOD = if Sys.isapple()
    try
        @eval using Metal
        getfield(Main, :Metal)
    catch
        nothing
    end
else
    nothing
end

const CUDA_MOD = try
    @eval using CUDA
    getfield(Main, :CUDA)
catch
    nothing
end

function backend_available(mod, functional_name::Symbol)
    mod === nothing && return false
    return try
        Base.invokelatest(getfield(mod, functional_name))
    catch
        false
    end
end

function select_backend()
    requested = lowercase(strip(get(ENV, "KRAKEN_BACKEND", "auto")))
    requested in ("auto", "metal", "cuda", "cpu") ||
        error("KRAKEN_BACKEND must be auto, metal, cuda, or cpu; got $(requested)")

    if requested in ("auto", "metal") && backend_available(METAL_MOD, :functional)
        backend = Base.invokelatest(getfield(METAL_MOD, :MetalBackend))
        return backend, "metal", Float32
    elseif requested == "metal"
        error("KRAKEN_BACKEND=metal requested but Metal is unavailable")
    end

    if requested in ("auto", "cuda") && backend_available(CUDA_MOD, :functional)
        backend = Base.invokelatest(getfield(CUDA_MOD, :CUDABackend))
        return backend, "cuda", Float64
    elseif requested == "cuda"
        error("KRAKEN_BACKEND=cuda requested but CUDA is unavailable")
    end

    return KernelAbstractions.CPU(), "cpu", Float64
end

function wi_tag(Wi::Real)
    return replace(@sprintf("%.6g", Float64(Wi)), "." => "p", "-" => "m")
end

function csv_path(R::Int, Wi::Real)
    return joinpath("bench", "scratch", "cyl_cd_M23_bsd_off_R$(R)_Wi$(wi_tag(Wi)).csv")
end

function csv_cell(x)
    if x isa AbstractFloat
        return isfinite(x) ? @sprintf("%.16g", Float64(x)) : string(Float64(x))
    elseif x isa Bool
        return string(x)
    end
    s = string(x)
    if occursin(',', s) || occursin('"', s) || occursin('\n', s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

function write_csv(path::AbstractString, row)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(string.(CSV_COLUMNS), ","))
        println(io, join((csv_cell(get(row, col, "")) for col in CSV_COLUMNS), ","))
    end
end

function pct_error(value::Real, ref::Real)
    return 100.0 * (Float64(value) - Float64(ref)) / Float64(ref)
end

function max_steps_for_R(R::Int)
    return round(Int, 100_000 * (R / 30)^2)
end

function fluid_diagnostics(res)
    min_det = Inf
    u_max_abs = 0.0
    tauxx_max = 0.0
    tauxy_max = 0.0
    tauyy_max = 0.0
    fluid_cells = 0

    for j in 1:res.Ny, i in 1:res.Nx
        res.is_solid[i, j] && continue
        fluid_cells += 1
        cxx, cxy, cyy = logfv_exp_sym2_2d(
            res.psixx[i, j], res.psixy[i, j], res.psiyy[i, j],
        )
        det = Float64(cxx) * Float64(cyy) - Float64(cxy)^2
        min_det = min(min_det, det)
        u_max_abs = max(u_max_abs, hypot(Float64(res.ux[i, j]), Float64(res.uy[i, j])))
        tauxx_max = max(tauxx_max, abs(Float64(res.tauxx[i, j])))
        tauxy_max = max(tauxy_max, abs(Float64(res.tauxy[i, j])))
        tauyy_max = max(tauyy_max, abs(Float64(res.tauyy[i, j])))
    end

    fluid_cells == 0 && return (NaN, NaN, NaN, NaN, NaN)
    return (min_det, u_max_abs, tauxx_max, tauxy_max, tauyy_max)
end

function base_row(; backend_kind, FT, R, Wi, max_steps)
    beta = 0.5
    u_mean = 0.005
    Re_R = 1.0
    nu_total = u_mean * R / Re_R
    nu_s = beta * nu_total
    nu_p = (1.0 - beta) * nu_total
    lambda = Wi * R / u_mean
    ref = RHEOTOOL_CD[Wi]
    return Dict{Symbol,Any}(
        :mission => "M23",
        :backend => backend_kind,
        :FT => string(FT),
        :R => R,
        :Wi => Wi,
        :bsd_fraction => 0.0,
        :beta => beta,
        :u_mean => u_mean,
        :Re_R => Re_R,
        :nu_total => nu_total,
        :nu_s => nu_s,
        :nu_p => nu_p,
        :lambda => lambda,
        :max_steps => max_steps,
        :polymer_substeps_used => NaN,
        :Cd_kraken => NaN,
        :Cd_s => NaN,
        :Cd_p => NaN,
        :Cd_bsd => NaN,
        :Cd_rheotool => ref,
        :Cd_err_pct => NaN,
        :min_det_C => NaN,
        :u_max_abs => NaN,
        :tau_xx_max_abs => NaN,
        :tau_xy_max_abs => NaN,
        :tau_yy_max_abs => NaN,
        :n_drag_samples => 0,
        :completed_steps => 0,
        :first_nonfinite_step => 0,
        :first_nonfinite_field => :none,
        :nan_flag => true,
        :walltime_s => 0.0,
        :MLUPS => NaN,
    )
end

function run_case!(; backend, backend_kind, FT, R::Int, Wi::Float64, max_steps::Int)
    row = base_row(; backend_kind, FT, R, Wi, max_steps)
    H = round(Int, 4 * R)
    avg_window = max(1, max_steps ÷ 5)
    result, walltime, nan_flag = try
        t0 = time()
        res = Kraken.run_viscoelastic_logfv_cylinder_coupled_2d(;
            radius=R,
            H,
            L_up=4,
            L_down=8,
            nu_s=FT(row[:nu_s]),
            nu_p=FT(row[:nu_p]),
            lambda=FT(row[:lambda]),
            polymer_model=:oldroydb,
            u_mean=FT(row[:u_mean]),
            Fx_body=FT(0.0),
            bsd_fraction=FT(0.0),
            polymer_substeps=:auto,
            subcycle_relative_tolerance=FT(0.01),
            max_deformation_increment=FT(0.05),
            max_memory_deformation_increment=FT(0.07),
            max_polymer_substeps=64,
            max_steps,
            avg_window,
            drag_stride=200,
            force_boundary_fill=:bc_aware,
            embedded_gradient=false,
            embedded_advection=false,
            embedded_force=false,
            embedded_drag=false,
            backend,
            T=FT,
        )
        dt = time() - t0
        (res, dt, res.first_nonfinite_step > 0)
    catch err
        @warn "M23 case crashed" R Wi err=sprint(showerror, err)
        (nothing, 0.0, true)
    end

    if result !== nothing
        min_det, u_max, tauxx_max, tauxy_max, tauyy_max = fluid_diagnostics(result)
        row[:polymer_substeps_used] = result.polymer_substeps
        row[:Cd_kraken] = result.Cd
        row[:Cd_s] = result.Cd_s
        row[:Cd_p] = result.Cd_p
        row[:Cd_bsd] = result.Cd_bsd
        row[:Cd_err_pct] = pct_error(result.Cd, row[:Cd_rheotool])
        row[:min_det_C] = min_det
        row[:u_max_abs] = u_max
        row[:tau_xx_max_abs] = tauxx_max
        row[:tau_xy_max_abs] = tauxy_max
        row[:tau_yy_max_abs] = tauyy_max
        row[:n_drag_samples] = result.n_drag_samples
        row[:completed_steps] = result.completed_steps
        row[:first_nonfinite_step] = result.first_nonfinite_step
        row[:first_nonfinite_field] = result.first_nonfinite_field
        row[:nan_flag] = nan_flag
        row[:walltime_s] = walltime
        row[:MLUPS] = walltime > 0 ?
            Float64(result.Nx) * Float64(result.Ny) *
            Float64(result.completed_steps) / walltime / 1e6 : NaN
    else
        row[:first_nonfinite_field] = :crash
    end

    write_csv(csv_path(R, Wi), row)
    return row
end

function print_case_summary(row)
    err_text = Int(row[:R]) == 30 ?
        @sprintf("%.3f%%", Float64(row[:Cd_err_pct])) : "NA"
    @printf("M23 R=%d Wi=%.6g bsd=0: Cd_kraken=%.6f, Cd_rheotool=%.6f, err=%s, min_detC=%.4f, nan_flag=%s\n",
            Int(row[:R]), Float64(row[:Wi]), Float64(row[:Cd_kraken]),
            Float64(row[:Cd_rheotool]), err_text, Float64(row[:min_det_C]),
            string(row[:nan_flag]))
end

function assert_self_test_csv!(path::AbstractString)
    lines = readlines(path)
    @assert length(lines) == 2 "self-test CSV must contain header plus one row"
    @assert lines[1] == join(string.(CSV_COLUMNS), ",") "self-test CSV header mismatch"
end

function main(args)
    modes = Set(args)
    valid_mode = xor("--self-test" in modes, "--full" in modes) && length(modes) == 1
    valid_mode ||
        error("usage: julia --project=. bench/viscoelastic_logfv/run_cyl_cd_convergence_bsd_off_2d.jl --self-test|--full")
    backend, backend_kind, FT = select_backend()
    println("M23 backend=$(backend_kind) FT=$(FT)")

    if "--self-test" in modes
        row = run_case!(; backend, backend_kind, FT, R=20, Wi=0.1, max_steps=5_000)
        assert_self_test_csv!(csv_path(20, 0.1))
        print_case_summary(row)
        @printf("M23 SELF-TEST OK Cd_kraken=%.6f\n", Float64(row[:Cd_kraken]))
        return nothing
    end

    for R in (20, 30, 40, 50), Wi in (0.1, 0.2)
        row = run_case!(; backend, backend_kind, FT, R, Wi, max_steps=max_steps_for_R(R))
        print_case_summary(row)
    end
    return nothing
end

main(ARGS)
