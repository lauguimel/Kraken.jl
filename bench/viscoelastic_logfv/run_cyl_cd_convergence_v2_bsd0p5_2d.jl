#!/usr/bin/env julia

using Kraken, Printf
using Metal

const _METAL_OK = Metal.functional()
if _METAL_OK
    using KernelAbstractions
    backend, backend_label, FT = Metal.MetalBackend(), "metal", Float32
else
    @warn "Metal not available — falling back to CPU F64 (NOT recommended for M24C_bsd0p5)"
    using KernelAbstractions
    backend, backend_label, FT = KernelAbstractions.CPU(), "cpu", Float64
end

const CSV_COLUMNS = [
    :R, :Wi, :bsd_fraction, :beta, :u_mean, :lambda, :nu_s, :nu_p,
    :H, :L_up, :L_down, :Nx, :Ny,
    :max_steps, :completed_steps, :avg_window, :polymer_substeps,
    :backend, :float_type,
    :Cd_kraken, :Cd_rheotool, :Cd_err_pct,
    :min_det_C, :min_c_eig,
    :u_max_abs, :tau_xx_max_abs, :tau_xy_max_abs, :tau_yy_max_abs,
    :trace_C_max, :trace_C_mean, :N1_max_abs, :N1_mean_abs,
    :nan_flag, :dt_s, :mlups,
]

const SCRATCH_DIR = normpath(joinpath(@__DIR__, "..", "scratch"))
const RHEOTOOL_DIR = normpath(joinpath(@__DIR__, "..", "rheotool"))
const BSD_FRACTION = 0.5; const BETA = 0.5
const U_MEAN = 0.005; const RE_R = 1.0
const L_UP_FACTOR = 15.0; const L_DOWN_FACTOR = 15.0

function mode_from_args(args)
    length(args) <= 1 || error("expected at most one flag: --self-test or --full")
    flag = isempty(args) ? "--self-test" : args[1]
    flag == "--self-test" && return :selftest
    flag == "--full" && return :full
    error("unknown flag $(flag); expected --self-test or --full")
end

wi_output_tag(wi) = wi == 0.1 ? "0p1" : wi == 0.2 ? "0p2" :
    replace(@sprintf("%.6g", Float64(wi)), "." => "p", "-" => "m")

wi_input_tag(wi) = wi == 0.1 ? "0.1" : wi == 0.2 ? "0.2" :
    @sprintf("%.6g", Float64(wi))
case_csv_path(R, wi) =
    joinpath(SCRATCH_DIR, "cyl_v2_bsd0p5_R$(R)_wi$(wi_output_tag(wi)).csv")

function csv_cell(x)
    x === nothing && return ""
    if x isa AbstractFloat
        isnan(x) && return "NaN"
        return @sprintf("%.16g", Float64(x))
    end
    s = string(x)
    if occursin(',', s) || occursin('"', s) || occursin('\n', s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

function write_csv_row(path, row)
    open(path, "w") do io
        println(io, join(string.(CSV_COLUMNS), ","))
        println(io, join((csv_cell(get(row, col, "")) for col in CSV_COLUMNS), ","))
    end
end

function rheotool_last_cd(R, wi)
    R == 30 || return NaN
    path = joinpath(RHEOTOOL_DIR, "cylinder_wi$(wi_input_tag(wi))", "Cd.txt")
    cd = NaN
    open(path, "r") do io
        for line in eachline(io)
            s = strip(line)
            isempty(s) && continue
            startswith(s, "#") && continue
            parts = split(s)
            length(parts) >= 2 && (cd = parse(Float64, parts[end]))
        end
    end
    return cd
end

function pct_error(value, ref)
    isfinite(value) && isfinite(ref) || return NaN
    return 100.0 * (value - ref) / ref
end

function base_row(R, wi, max_steps)
    H = 4 * R
    Nx = ceil(Int, (L_UP_FACTOR + L_DOWN_FACTOR) * R)
    Ny = H
    nu_total = U_MEAN * R / RE_R
    nu_s = BETA * nu_total
    nu_p = (1.0 - BETA) * nu_total
    lambda = wi * R / U_MEAN
    avg_window = max(1, max_steps ÷ 5)
    return Dict{Symbol,Any}(
        :R => R,
        :Wi => wi,
        :bsd_fraction => BSD_FRACTION,
        :beta => BETA,
        :u_mean => U_MEAN,
        :lambda => lambda,
        :nu_s => nu_s,
        :nu_p => nu_p,
        :H => H,
        :L_up => L_UP_FACTOR,
        :L_down => L_DOWN_FACTOR,
        :Nx => Nx,
        :Ny => Ny,
        :max_steps => max_steps,
        :completed_steps => NaN,
        :avg_window => avg_window,
        :polymer_substeps => NaN,
        :backend => backend_label,
        :float_type => string(FT),
        :Cd_kraken => NaN,
        :Cd_rheotool => rheotool_last_cd(R, wi),
        :Cd_err_pct => NaN,
        :min_det_C => NaN,
        :min_c_eig => NaN,
        :u_max_abs => NaN,
        :tau_xx_max_abs => NaN,
        :tau_xy_max_abs => NaN,
        :tau_yy_max_abs => NaN,
        :trace_C_max => NaN,
        :trace_C_mean => NaN,
        :N1_max_abs => NaN,
        :N1_mean_abs => NaN,
        :nan_flag => true,
        :dt_s => NaN,
        :mlups => NaN,
    )
end

function max_abs_fluid(field, is_solid)
    found = false
    best = 0.0
    for idx in CartesianIndices(is_solid)
        is_solid[idx] && continue
        v = abs(Float64(field[idx]))
        isfinite(v) || return NaN
        best = found ? max(best, v) : v
        found = true
    end
    return found ? best : NaN
end

function max_speed_fluid(ux, uy, is_solid)
    found = false
    best = 0.0
    for idx in CartesianIndices(is_solid)
        is_solid[idx] && continue
        v = hypot(Float64(ux[idx]), Float64(uy[idx]))
        isfinite(v) || return NaN
        best = found ? max(best, v) : v
        found = true
    end
    return found ? best : NaN
end

function min_det_c_fluid(psixx, psixy, psiyy, is_solid)
    found = false
    best = Inf
    try
        for idx in CartesianIndices(is_solid)
            is_solid[idx] && continue
            cxx, cxy, cyy = Kraken.logfv_exp_sym2_2d(
                psixx[idx], psixy[idx], psiyy[idx],
            )
            det_c = Float64(cxx) * Float64(cyy) - Float64(cxy)^2
            isfinite(det_c) || return NaN, true
            best = found ? min(best, det_c) : det_c
            found = true
        end
    catch err
        err isa DomainError || rethrow()
        return NaN, true
    end
    return found ? best : NaN, false
end

function trace_C_and_N1_stats(psixx, psixy, psiyy, tauxx, tauyy, is_solid)
    found = false
    trace_max = -Inf; trace_sum = 0.0
    n1_max = 0.0; n1_abs_sum = 0.0
    n_fluid = 0
    bad = false
    try
        for idx in CartesianIndices(is_solid)
            is_solid[idx] && continue
            cxx, cxy, cyy = Kraken.logfv_exp_sym2_2d(
                psixx[idx], psixy[idx], psiyy[idx],
            )
            tr_c = Float64(cxx) + Float64(cyy)
            n1 = Float64(tauxx[idx]) - Float64(tauyy[idx])
            (isfinite(tr_c) && isfinite(n1)) || (bad = true; break)
            trace_max = found ? max(trace_max, tr_c) : tr_c
            trace_sum += tr_c
            n1_max = found ? max(n1_max, abs(n1)) : abs(n1)
            n1_abs_sum += abs(n1)
            n_fluid += 1
            found = true
        end
    catch err
        err isa DomainError || rethrow()
        bad = true
    end
    if bad || !found
        return NaN, NaN, NaN, NaN
    end
    return trace_max, trace_sum / n_fluid, n1_max, n1_abs_sum / n_fluid
end

function fill_result_row!(row, result, dt_s)
    min_det_c, det_bad = min_det_c_fluid(
        result.psixx, result.psixy, result.psiyy, result.is_solid,
    )
    cd = Float64(result.Cd)
    row[:completed_steps] = result.completed_steps
    row[:polymer_substeps] = result.polymer_substeps
    row[:Cd_kraken] = cd
    row[:Cd_err_pct] = pct_error(cd, row[:Cd_rheotool])
    row[:min_det_C] = min_det_c
    row[:min_c_eig] = Float64(result.min_c_eig)
    row[:u_max_abs] = max_speed_fluid(result.ux, result.uy, result.is_solid)
    row[:tau_xx_max_abs] = max_abs_fluid(result.tauxx, result.is_solid)
    row[:tau_xy_max_abs] = max_abs_fluid(result.tauxy, result.is_solid)
    row[:tau_yy_max_abs] = max_abs_fluid(result.tauyy, result.is_solid)
    tc_max, tc_mean, n1_max, n1_mean = trace_C_and_N1_stats(
        result.psixx, result.psixy, result.psiyy,
        result.tauxx, result.tauyy, result.is_solid,
    )
    row[:trace_C_max] = tc_max
    row[:trace_C_mean] = tc_mean
    row[:N1_max_abs] = n1_max
    row[:N1_mean_abs] = n1_mean
    row[:dt_s] = dt_s
    row[:mlups] = dt_s > 0 ?
        Float64(result.Nx) * Float64(result.Ny) *
        Float64(result.completed_steps) / dt_s / 1e6 : NaN
    checked = (
        row[:Cd_kraken], row[:min_det_C], row[:u_max_abs],
        row[:tau_xx_max_abs], row[:tau_xy_max_abs], row[:tau_yy_max_abs],
    )
    row[:nan_flag] = det_bad || any(v -> !isfinite(Float64(v)), checked) ||
        result.first_nonfinite_step > 0
    return row
end

fmt_nan(v, fmt) = isfinite(Float64(v)) ?
    Printf.format(Printf.Format(fmt), Float64(v)) : "NaN"

function print_summary(row)
    @printf(
        "M24C_bsd0p5 R=%d Wi=%s bsd=0.5: Cd_kraken=%s, Cd_rheotool=%s, err=%s%%, min_detC=%s, trace_C_max=%s, trace_C_mean=%s, N1_max=%s, N1_mean=%s, nan_flag=%s, dt=%.1fs\n",
        row[:R],
        wi_input_tag(row[:Wi]),
        fmt_nan(row[:Cd_kraken], "%.6f"),
        fmt_nan(row[:Cd_rheotool], "%.6f"),
        fmt_nan(row[:Cd_err_pct], "%.3f"),
        fmt_nan(row[:min_det_C], "%.4g"),
        fmt_nan(row[:trace_C_max], "%.4g"),
        fmt_nan(row[:trace_C_mean], "%.4g"),
        fmt_nan(row[:N1_max_abs], "%.4g"),
        fmt_nan(row[:N1_mean_abs], "%.4g"),
        string(row[:nan_flag]),
        Float64(row[:dt_s]),
    )
end

function run_case(R, wi, max_steps)
    row = base_row(R, wi, max_steps)
    result = nothing
    status = :ok
    nan_step_local = -1
    t0 = time()
    try
        result = Kraken.run_viscoelastic_logfv_cylinder_coupled_2d(;
            radius=R, H=row[:H], L_up=L_UP_FACTOR, L_down=L_DOWN_FACTOR,
            nu_s=FT(row[:nu_s]), nu_p=FT(row[:nu_p]), lambda=FT(row[:lambda]),
            polymer_model=:oldroydb, L_max=FT(10.0), u_mean=FT(U_MEAN),
            Fx_body=FT(0.0), bsd_fraction=FT(BSD_FRACTION),
            polymer_substeps=:auto,
            subcycle_relative_tolerance=FT(0.01),
            max_deformation_increment=FT(0.05),
            max_memory_deformation_increment=FT(0.07),
            max_polymer_substeps=64, max_steps=max_steps,
            avg_window=row[:avg_window], drag_stride=200, diagnostic_stride=0,
            embedded_geometry=:qwall, embedded_gradient=false,
            embedded_advection=false, embedded_force=false, embedded_drag=false,
            embedded_circle_samples=32, force_boundary_fill=:bc_aware,
            backend, T=FT,
        )
    catch err
        if err isa DomainError || err isa BoundsError
            status = :crashed_pre_nan
            nan_step_local = 0
        else
            rethrow(err)
        end
    end
    dt_s = time() - t0
    row[:dt_s] = dt_s
    if status == :ok
        fill_result_row!(row, result, dt_s)
    elseif status == :crashed_pre_nan
        nan_step_local == 0 || error("unexpected crash sentinel")
        row[:nan_flag] = true
    end
    write_csv_row(case_csv_path(R, wi), row)
    print_summary(row)
    return row
end

function assert_csv_schema(path)
    open(path, "r") do io
        header = split(strip(readline(io)), ",")
        @assert header == string.(CSV_COLUMNS) "CSV header mismatch in $(path)"
        row = split(strip(readline(io)), ",")
        @assert length(row) == length(CSV_COLUMNS) "CSV column count mismatch in $(path)"
    end
end

function main()
    mode = mode_from_args(ARGS)
    println("M24C_bsd0p5 mode: $(mode == :full ? "full" : "selftest") backend=$(backend_label) FT=$(FT) bsd=$(BSD_FRACTION)")
    mkpath(SCRATCH_DIR)
    cases = mode == :full ?
        [(R, wi, 50_000) for R in (20, 30, 40, 50) for wi in (0.1, 0.2)] :
        [(20, 0.1, 5_000)]
    rows = [run_case(R, wi, max_steps) for (R, wi, max_steps) in cases]
    if mode == :selftest
        path = case_csv_path(20, 0.1)
        assert_csv_schema(path)
        @assert isfinite(Float64(rows[1][:Cd_kraken])) "self-test Cd is not finite"
    end
end

main()
