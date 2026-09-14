#!/usr/bin/env julia

# Confined-cylinder Cd convergence harness for the production cell-centered
# log-FV polymer backend. This intentionally does not call the legacy
# Liu-style population conformation driver.

using Dates
using KernelAbstractions
using Kraken
using Printf
using Serialization

const CUDA_MOD = try
    @eval using CUDA
    getfield(Main, :CUDA)
catch
    nothing
end

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

const LIU_REF = Dict(
    (20, "0.1") => 129.42,
    (20, "0.5") => 125.17,
    (20, "1") => 164.26,
    (25, "0.1") => 129.61,
    (30, "0.1") => 130.36,
    (30, "0.5") => 126.31,
    (30, "1") => 151.31,
    (35, "0.1") => 130.77,
    (35, "0.5") => 127.72,
    (35, "1") => 149.04,
    (40, "0.1") => 130.79,
    (48, "0.1") => 130.83,
)

const RHEOTOOL_REF_MEAN = Dict(
    "0.05" => 131.813036363,
    "0.1" => 130.428774404,
    "0.2" => 126.831069913,
    "0.5" => 119.288284200,
    "1" => 116.995396566,
)

const RHEOTOOL_REF_LAST = Dict(
    "0.05" => 131.813486383,
    "0.1" => 130.429053837,
    "0.2" => 126.839657948,
    "0.5" => 119.713781971,
    "1" => 120.400592705,
)

const RHEOTOOL_NEWTONIAN = 132.362236515

const CSV_COLUMNS = [
    :timestamp, :suite, :status, :error,
    :backend, :float_type, :case_name,
    :R, :Nx, :Ny, :H, :L_up, :L_down,
    :Re_R, :Re_D, :beta, :Wi, :u_mean,
    :nu_total, :nu_s, :nu_p, :lambda, :stress_prefactor,
    :polymer_model, :L_max,
    :bsd_fraction, :Fx_body,
    :steps, :completed_steps, :avg_window, :drag_stride, :diagnostic_stride,
    :embedded_geometry, :embedded_gradient, :embedded_advection,
    :embedded_force, :embedded_drag, :embedded_circle_samples,
    :embedded_cut_count, :embedded_wall_length,
    :embedded_min_fluid_cell_fraction,
    :embedded_zero_fluid_cell_fraction_count,
    :embedded_normal_radial_min, :embedded_normal_radial_mean,
    :embedded_normal_radial_samples,
    :force_boundary_fill,
    :polymer_substeps_requested, :polymer_substeps,
    :raw_substeps, :relax_substeps, :deformation_substeps,
    :memory_deformation_substeps,
    :substeps_clamped, :max_grad_norm_estimate,
    :subcycle_relative_tolerance, :max_deformation_increment,
    :memory_deformation_increment, :max_memory_deformation_increment,
    :dt_s, :lups,
    :Cd, :Cd_s, :Cd_p, :Cd_bsd, :Cl, :Fx_s, :Fx_p, :Fx_bsd, :Fy_s, :Fy_p, :Fy_bsd,
    :n_drag_samples,
    :rho_min, :rho_max, :max_speed, :min_c_eig,
    :max_c_trace, :min_fene_denom, :max_fene_factor,
    :first_nonfinite_step, :first_nonfinite_field,
    :first_nonfinite_i, :first_nonfinite_j,
    :max_abs_psi, :max_abs_tau, :max_abs_poly_force, :max_abs_total_force,
    :newtonian_ref, :err_newtonian_ref_pct,
    :newtonian_cd_same_run, :err_newtonian_same_run_pct,
    :liu_ref, :err_liu_pct,
    :rheotool_ref_mean, :err_rheotool_mean_pct,
    :rheotool_ref_last, :err_rheotool_last_pct,
    :field_dump, :vtk_snapshot, :field_error,
]

function env_items(raw::AbstractString)
    return (strip(x) for x in split(replace(raw, ';' => ','), ',')
            if !isempty(strip(x)))
end

parse_list(::Type{T}, raw::AbstractString) where {T} =
    [parse(T, x) for x in env_items(raw)]

function parse_bool_env(name::AbstractString, default::Bool)
    raw = lowercase(strip(get(ENV, name, default ? "1" : "0")))
    raw in ("1", "true", "yes", "on") && return true
    raw in ("0", "false", "no", "off") && return false
    error("$(name) must be boolean-like, got $(raw)")
end

function parse_polymer_substeps(raw::AbstractString)
    stripped = lowercase(strip(raw))
    stripped == "auto" && return :auto
    return parse(Int, stripped)
end

function parse_logfv_polymer_model(raw::AbstractString)
    normalized = Symbol(replace(lowercase(strip(raw)), '-' => '_'))
    normalized in (:oldroydb, :oldroyd_b, :ob) && return :oldroydb
    normalized in (:fenep, :fene_p, :fene_peterlin) && return :fenep
    error("KRAKEN_LOGFV_POLYMER_MODEL must be oldroydb or fenep, got $(raw)")
end

function wi_key(wi::Real)
    return @sprintf("%.6g", Float64(wi))
end

function select_backend()
    requested = lowercase(get(ENV, "KRAKEN_BACKEND", "auto"))
    if requested in ("auto", "cuda") && CUDA_MOD !== nothing
        try
            if Base.invokelatest(getfield(CUDA_MOD, :functional))
                backend = Base.invokelatest(getfield(CUDA_MOD, :CUDABackend))
                device = Base.invokelatest(getfield(CUDA_MOD, :device))
                name = Base.invokelatest(getfield(CUDA_MOD, :name), device)
                return backend, "cuda", "CUDA $(name)"
            end
        catch err
            requested == "cuda" && rethrow(err)
        end
    end
    if requested in ("auto", "metal") && METAL_MOD !== nothing
        try
            if Base.invokelatest(getfield(METAL_MOD, :functional))
                backend = Base.invokelatest(getfield(METAL_MOD, :MetalBackend))
                return backend, "metal", "Metal"
            end
        catch err
            requested == "metal" && rethrow(err)
        end
    end
    requested in ("auto", "cpu") ||
        error("unknown or unavailable KRAKEN_BACKEND=$(requested)")
    return KernelAbstractions.CPU(), "cpu", "CPU"
end

function select_float_type(backend_kind::AbstractString)
    raw = lowercase(get(ENV, "KRAKEN_FT", "auto"))
    FT = if raw == "auto"
        backend_kind == "metal" ? Float32 : Float64
    elseif raw in ("float32", "single", "f32")
        Float32
    elseif raw in ("float64", "double", "f64")
        Float64
    else
        error("unknown KRAKEN_FT=$(raw); expected auto, float32, or float64")
    end
    backend_kind == "metal" && FT === Float64 &&
        error("Metal backend is local-debug only here and must use Float32")
    return FT
end

function suite_defaults(suite::AbstractString, smoke::Bool)
    if smoke || suite == "smoke"
        return (; R="4", Wi="0.001,0.1", run_newtonian=true,
                run_viscoelastic=true, steps_low_wi=80, steps=80,
                step_cap=200, scale_steps=false, avg_divisor=2,
                drag_stride=10)
    elseif suite == "newtonian"
        return (; R="20,30,35", Wi="0.001", run_newtonian=true,
                run_viscoelastic=false, steps_low_wi=100_000, steps=100_000,
                step_cap=0, scale_steps=true, avg_divisor=5,
                drag_stride=200)
    elseif suite == "nearnewtonian"
        return (; R="20,30", Wi="0.001", run_newtonian=true,
                run_viscoelastic=true, steps_low_wi=100_000, steps=100_000,
                step_cap=0, scale_steps=true, avg_divisor=5,
                drag_stride=200)
    elseif suite == "rheotool"
        return (; R="30", Wi="0.05,0.1,0.2,0.5,1.0",
                run_newtonian=true, run_viscoelastic=true,
                steps_low_wi=100_000, steps=200_000, step_cap=0,
                scale_steps=false, avg_divisor=5, drag_stride=200)
    elseif suite == "postfix_embedded_circle"
        return (; R="6,10,14", Wi="0.001,0.05,0.1",
                run_newtonian=true, run_viscoelastic=true,
                steps_low_wi=2_000, steps=5_000, step_cap=5_000,
                scale_steps=false, avg_divisor=4, drag_stride=50)
    elseif suite == "liu"
        return (; R="20,30,35", Wi="0.1,0.5,1.0",
                run_newtonian=true, run_viscoelastic=true,
                steps_low_wi=100_000, steps=200_000, step_cap=0,
                scale_steps=true, avg_divisor=5, drag_stride=200)
    elseif suite == "both"
        return (; R="20,30,35", Wi="0.001,0.05,0.1,0.2,0.5,1.0",
                run_newtonian=true, run_viscoelastic=true,
                steps_low_wi=100_000, steps=200_000, step_cap=0,
                scale_steps=true, avg_divisor=5, drag_stride=200)
    end
    error("unknown KRAKEN_LOGFV_CYLINDER_SUITE=$(suite); expected smoke, newtonian, nearnewtonian, rheotool, postfix_embedded_circle, liu, or both")
end

function csv_cell(x)
    x === nothing && return ""
    if x isa AbstractFloat
        return isfinite(x) ? @sprintf("%.16g", x) : string(x)
    end
    s = string(x)
    if occursin(',', s) || occursin('"', s) || occursin('\n', s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

function append_csv_row(path::AbstractString, row::Dict{Symbol,Any})
    first = !isfile(path)
    open(path, first ? "w" : "a") do io
        if first
            println(io, join(string.(CSV_COLUMNS), ","))
        end
        println(io, join((csv_cell(get(row, col, "")) for col in CSV_COLUMNS), ","))
        flush(io)
    end
end

function pct_error(value, ref)
    (value isa Real && ref isa Real && isfinite(Float64(value)) &&
     isfinite(Float64(ref))) || return NaN
    return 100.0 * (Float64(value) - Float64(ref)) / Float64(ref)
end

function steps_for_R(R::Int, Wi::Float64, steps_low_wi::Int, steps::Int,
                     scale_steps::Bool, cap::Int)
    base = Wi < 0.01 ? steps_low_wi : steps
    scaled = scale_steps ? round(Int, base * (R / 30)^2) : base
    bounded = cap > 0 ? min(scaled, cap) : scaled
    return max(1, bounded)
end

function guard_local_run!(backend_kind, Nx, Ny, steps, allow_long_local,
                          max_local_updates)
    backend_kind == "cuda" && return nothing
    allow_long_local && return nothing
    updates = Float64(Nx) * Float64(Ny) * Float64(steps)
    updates <= max_local_updates && return nothing
    error("refusing long local $(backend_kind) run ($(updates) lattice updates); set KRAKEN_ALLOW_LONG_LOCAL=1 for an intentional local benchmark")
end

function result_case_tag(row::Dict{Symbol,Any})
    wi = get(row, :Wi, 0)
    wi_tag = replace(@sprintf("%.6g", Float64(wi)), "." => "p", "-" => "m")
    name = replace(string(get(row, :case_name, "case")), r"[^A-Za-z0-9_]+" => "_")
    return @sprintf("R%d_Wi%s_%s_steps%d",
                    Int(get(row, :R, 0)), wi_tag, name,
                    Int(get(row, :steps, 0)))
end

function logfv_reconstruct_c_fields(psixx, psixy, psiyy)
    Nx, Ny = size(psixx)
    cxx = Matrix{Float64}(undef, Nx, Ny)
    cxy = Matrix{Float64}(undef, Nx, Ny)
    cyy = Matrix{Float64}(undef, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        cxx_ij, cxy_ij, cyy_ij = Kraken.logfv_exp_sym2_2d(
            psixx[i, j], psixy[i, j], psiyy[i, j],
        )
        cxx[i, j] = Float64(cxx_ij)
        cxy[i, j] = Float64(cxy_ij)
        cyy[i, j] = Float64(cyy_ij)
    end
    return cxx, cxy, cyy
end

function logfv_field_dict(result)
    cxx, cxy, cyy = logfv_reconstruct_c_fields(result.psixx, result.psixy, result.psiyy)
    ux = Float64.(result.ux)
    uy = Float64.(result.uy)
    ctrace = cxx .+ cyy
    fields = Dict{String,Matrix{Float64}}(
        "rho" => Float64.(result.rho),
        "ux" => ux,
        "uy" => uy,
        "u_mag" => hypot.(ux, uy),
        "solid" => Float64.(result.is_solid),
        "psi_xx" => Float64.(result.psixx),
        "psi_xy" => Float64.(result.psixy),
        "psi_yy" => Float64.(result.psiyy),
        "C_xx" => cxx,
        "C_xy" => cxy,
        "C_yy" => cyy,
        "C_trace" => ctrace,
        "tau_xx" => Float64.(result.tauxx),
        "tau_xy" => Float64.(result.tauxy),
        "tau_yy" => Float64.(result.tauyy),
        "fx_poly" => Float64.(result.fx_poly),
        "fy_poly" => Float64.(result.fy_poly),
        "fx_total" => Float64.(result.fx_total),
        "fy_total" => Float64.(result.fy_total),
    )
    if getproperty(result, :polymer_model) === :fenep
        L2 = Float64(result.L_max)^2
        fields["fene_factor"] = (L2 - 2.0) ./ (L2 .- ctrace)
    end
    return fields
end

function save_logfv_fields!(row::Dict{Symbol,Any}, result, field_root::AbstractString;
                            write_vtk::Bool=true)
    tag = result_case_tag(row)
    case_dir = joinpath(field_root, tag)
    mkpath(case_dir)
    fields = logfv_field_dict(result)
    metadata = Dict{Symbol,Any}(row)
    metadata[:completed_steps] = result.completed_steps
    metadata[:first_nonfinite_step] = result.first_nonfinite_step
    metadata[:first_nonfinite_field] = result.first_nonfinite_field
    metadata[:first_nonfinite_i] = result.first_nonfinite_i
    metadata[:first_nonfinite_j] = result.first_nonfinite_j
    metadata[:geometry_name] = result.geometry.name
    dump_path = joinpath(case_dir, "fields.jls")
    open(dump_path, "w") do io
        serialize(io, (; metadata, fields))
    end
    row[:field_dump] = dump_path
    if write_vtk
        Kraken.write_snapshot_2d!(
            case_dir, Int(result.completed_steps), Int(result.Nx), Int(result.Ny),
            1.0, fields; time=Float64(result.completed_steps),
        )
        row[:vtk_snapshot] = joinpath(case_dir, @sprintf("snapshot_%07d.vtr", Int(result.completed_steps)))
    end
    return dump_path
end

function maybe_save_logfv_fields!(row::Dict{Symbol,Any}, result, save_fields::Bool,
                                  field_root::AbstractString, write_vtk::Bool)
    save_fields || return nothing
    try
        save_logfv_fields!(row, result, field_root; write_vtk)
    catch err
        row[:field_error] = sprint(showerror, err)
        @warn "Log-FV field save failed" case=result_case_tag(row) error=row[:field_error]
    end
    return nothing
end

function row_base(; timestamp, suite, backend_label, FT, case_name, R, Nx, Ny,
                  H, L_up, L_down, Re_R, beta, Wi, u_mean, nu_total,
                  nu_s, nu_p, lambda, polymer_model, L_max,
                  bsd_fraction, Fx_body, steps,
                  avg_window, drag_stride, diagnostic_stride,
                  embedded_geometry, embedded_gradient, embedded_advection,
                  embedded_force, embedded_drag, embedded_circle_samples,
                  force_boundary_fill, polymer_substeps,
                  subcycle_relative_tolerance, max_deformation_increment,
                  max_memory_deformation_increment)
    return Dict{Symbol,Any}(
        :timestamp => timestamp,
        :suite => suite,
        :status => "started",
        :error => "",
        :backend => backend_label,
        :float_type => string(FT),
        :case_name => case_name,
        :R => R,
        :Nx => Nx,
        :Ny => Ny,
        :H => H,
        :L_up => L_up,
        :L_down => L_down,
        :Re_R => Re_R,
        :Re_D => 2 * Re_R,
        :beta => beta,
        :Wi => Wi,
        :u_mean => u_mean,
        :nu_total => nu_total,
        :nu_s => nu_s,
        :nu_p => nu_p,
        :lambda => lambda,
        :stress_prefactor => nu_p / lambda,
        :polymer_model => polymer_model,
        :L_max => L_max,
        :bsd_fraction => bsd_fraction,
        :Fx_body => Fx_body,
        :steps => steps,
        :completed_steps => 0,
        :avg_window => avg_window,
        :drag_stride => drag_stride,
        :diagnostic_stride => diagnostic_stride,
        :embedded_geometry => embedded_geometry,
        :embedded_gradient => embedded_gradient,
        :embedded_advection => embedded_advection,
        :embedded_force => embedded_force,
        :embedded_drag => embedded_drag,
        :embedded_circle_samples => embedded_circle_samples,
        :force_boundary_fill => force_boundary_fill,
        :polymer_substeps_requested => polymer_substeps,
        :subcycle_relative_tolerance => subcycle_relative_tolerance,
        :max_deformation_increment => max_deformation_increment,
        :max_memory_deformation_increment => max_memory_deformation_increment,
    )
end

function fill_result_row!(row, result, dt, R, u_mean, newtonian_cd_same_run,
                          liu_ref, rheo_mean, rheo_last)
    row[:status] = "ok"
    row[:dt_s] = dt
    row[:completed_steps] = result.completed_steps
    row[:lups] = Float64(result.Nx) * Float64(result.Ny) *
                 Float64(result.completed_steps) / dt
    row[:polymer_model] = result.polymer_model
    row[:L_max] = result.L_max
    row[:polymer_substeps] = result.polymer_substeps
    row[:raw_substeps] = result.subcycle_estimate.raw_substeps
    row[:relax_substeps] = result.subcycle_estimate.relax_substeps
    row[:deformation_substeps] = result.subcycle_estimate.deformation_substeps
    row[:memory_deformation_substeps] = result.subcycle_estimate.memory_deformation_substeps
    row[:substeps_clamped] = result.subcycle_estimate.clamped
    row[:max_grad_norm_estimate] = result.max_grad_norm_estimate
    row[:memory_deformation_increment] = result.subcycle_estimate.memory_deformation_increment
    row[:max_memory_deformation_increment] =
        result.subcycle_estimate.max_memory_deformation_increment
    row[:Cd] = result.Cd
    row[:Cd_s] = result.Cd_s
    row[:Cd_p] = result.Cd_p
    row[:Cd_bsd] = result.Cd_bsd
    row[:Cl] = 2.0 * result.Fy_drag / (u_mean^2 * 2.0 * R)
    row[:Fx_s] = result.Fx_s
    row[:Fx_p] = result.Fx_p
    row[:Fx_bsd] = result.Fx_bsd
    row[:Fy_s] = result.Fy_s
    row[:Fy_p] = result.Fy_p
    row[:Fy_bsd] = result.Fy_bsd
    row[:n_drag_samples] = result.n_drag_samples
    row[:rho_min] = result.rho_min
    row[:rho_max] = result.rho_max
    row[:max_speed] = result.max_speed
    row[:min_c_eig] = result.min_c_eig
    row[:max_c_trace] = result.max_c_trace
    row[:min_fene_denom] = result.min_fene_denom
    row[:max_fene_factor] = result.max_fene_factor
    row[:first_nonfinite_step] = result.first_nonfinite_step
    row[:first_nonfinite_field] = result.first_nonfinite_field
    row[:first_nonfinite_i] = result.first_nonfinite_i
    row[:first_nonfinite_j] = result.first_nonfinite_j
    row[:max_abs_psi] = result.max_abs_psi
    row[:max_abs_tau] = result.max_abs_tau
    row[:max_abs_poly_force] = result.max_abs_poly_force
    row[:max_abs_total_force] = result.max_abs_total_force
    row[:embedded_geometry] = result.embedded_geometry
    row[:embedded_gradient] = result.embedded_gradient
    row[:embedded_advection] = result.embedded_advection
    row[:embedded_force] = result.embedded_force
    row[:embedded_drag] = result.embedded_drag
    row[:embedded_circle_samples] = result.embedded_circle_samples
    row[:embedded_cut_count] = result.embedded_cut_count
    row[:embedded_wall_length] = result.embedded_wall_length
    row[:embedded_min_fluid_cell_fraction] = result.embedded_min_fluid_cell_fraction
    row[:embedded_zero_fluid_cell_fraction_count] = result.embedded_zero_fluid_cell_fraction_count
    row[:embedded_normal_radial_min] = result.embedded_normal_radial_min
    row[:embedded_normal_radial_mean] = result.embedded_normal_radial_mean
    row[:embedded_normal_radial_samples] = result.embedded_normal_radial_samples
    row[:newtonian_ref] = RHEOTOOL_NEWTONIAN
    row[:err_newtonian_ref_pct] = pct_error(result.Cd, RHEOTOOL_NEWTONIAN)
    row[:newtonian_cd_same_run] = newtonian_cd_same_run
    row[:err_newtonian_same_run_pct] = pct_error(result.Cd, newtonian_cd_same_run)
    row[:liu_ref] = liu_ref
    row[:err_liu_pct] = pct_error(result.Cd, liu_ref)
    row[:rheotool_ref_mean] = rheo_mean
    row[:err_rheotool_mean_pct] = pct_error(result.Cd, rheo_mean)
    row[:rheotool_ref_last] = rheo_last
    row[:err_rheotool_last_pct] = pct_error(result.Cd, rheo_last)
    return row
end

function mark_nonfinite_result!(row::Dict{Symbol,Any}, errors::Vector{String},
                                label::AbstractString; continue_on_error::Bool)
    scalar_fields = (
        :Cd, :Cd_s, :Cd_p, :Cd_bsd, :Cl,
        :rho_min, :rho_max, :max_speed, :min_c_eig,
        :max_c_trace,
        :max_abs_psi, :max_abs_tau, :max_abs_poly_force, :max_abs_total_force,
    )
    bad = Symbol[]
    for field in scalar_fields
        value = get(row, field, NaN)
        if !(value isa Real) || !isfinite(Float64(value))
            push!(bad, field)
        end
    end
    first_bad_step = get(row, :first_nonfinite_step, 0)
    has_first_bad = first_bad_step isa Integer && first_bad_step > 0
    isempty(bad) && !has_first_bad && return false

    row[:status] = "nonfinite"
    detail = has_first_bad ?
        @sprintf("first nonfinite at step %d field=%s i=%d j=%d",
                 first_bad_step, string(get(row, :first_nonfinite_field, :unknown)),
                 Int(get(row, :first_nonfinite_i, 0)),
                 Int(get(row, :first_nonfinite_j, 0))) :
        ""
    row[:error] = isempty(bad) ? detail :
        "nonfinite fields: " * join(string.(bad), ";") *
        (isempty(detail) ? "" : "; " * detail)
    push!(errors, "$(label): $(row[:error])")
    continue_on_error || error(row[:error])
    return true
end

backend, backend_kind, backend_label = select_backend()
FT = select_float_type(backend_kind)

smoke = parse_bool_env("KRAKEN_SMOKE", false)
suite = lowercase(get(ENV, "KRAKEN_LOGFV_CYLINDER_SUITE",
                      get(ENV, "KRAKEN_CYLINDER_SUITE", smoke ? "smoke" : "smoke")))
defaults = suite_defaults(suite, smoke)

R_values = parse_list(Int, get(ENV, "KRAKEN_R_LIST", defaults.R))
Wi_values = parse_list(Float64, get(ENV, "KRAKEN_WI_LIST", defaults.Wi))
run_newtonian = parse_bool_env("KRAKEN_RUN_NEWTONIAN", defaults.run_newtonian)
run_viscoelastic = parse_bool_env("KRAKEN_RUN_VISCOELASTIC",
                                  defaults.run_viscoelastic)

beta = parse(Float64, get(ENV, "KRAKEN_BETA", "0.59"))
u_mean = parse(Float64, get(ENV, "KRAKEN_U_MEAN", "0.005"))
Re_R = parse(Float64, get(ENV, "KRAKEN_RE_R", "1.0"))
L_up = parse(Float64, get(ENV, "KRAKEN_L_UP_R", "15"))
L_down = parse(Float64, get(ENV, "KRAKEN_L_DOWN_R", "15"))
H_factor = parse(Float64, get(ENV, "KRAKEN_H_OVER_R", "4"))
Fx_body = parse(Float64, get(ENV, "KRAKEN_FX_BODY", "0.0"))
bsd_fraction = parse(Float64, get(ENV, "KRAKEN_LOGFV_BSD_FRACTION", "1.0"))
polymer_substeps = parse_polymer_substeps(get(ENV, "KRAKEN_POLYMER_SUBSTEPS", "auto"))
subcycle_relative_tolerance =
    parse(Float64, get(ENV, "KRAKEN_SUBCYCLE_RELATIVE_TOLERANCE", "0.01"))
max_deformation_increment =
    parse(Float64, get(ENV, "KRAKEN_MAX_DEFORMATION_INCREMENT", "0.05"))
max_memory_deformation_increment =
    parse(Float64, get(ENV, "KRAKEN_MAX_MEMORY_DEFORMATION_INCREMENT", "0.07"))
max_polymer_substeps = parse(Int, get(ENV, "KRAKEN_MAX_POLYMER_SUBSTEPS", "64"))
polymer_model = parse_logfv_polymer_model(
    get(ENV, "KRAKEN_LOGFV_POLYMER_MODEL", get(ENV, "KRAKEN_POLYMER_MODEL", "oldroydb")),
)
L_max = parse(Float64, get(ENV, "KRAKEN_L_MAX", "10.0"))

steps_low_wi = parse(Int, get(ENV, "KRAKEN_STEPS_LOW_WI",
                              string(defaults.steps_low_wi)))
steps_default = parse(Int, get(ENV, "KRAKEN_STEPS", string(defaults.steps)))
step_cap = parse(Int, get(ENV, "KRAKEN_MAX_STEPS_CAP", string(defaults.step_cap)))
scale_steps = parse_bool_env("KRAKEN_SCALE_STEPS_WITH_R", defaults.scale_steps)
avg_divisor = parse(Int, get(ENV, "KRAKEN_AVG_DIVISOR",
                             string(defaults.avg_divisor)))
drag_stride = parse(Int, get(ENV, "KRAKEN_DRAG_STRIDE",
                             string(defaults.drag_stride)))
diagnostic_stride = parse(Int, get(ENV, "KRAKEN_DIAGNOSTIC_STRIDE", "0"))
embedded_geometry = Symbol(lowercase(get(ENV, "KRAKEN_LOGFV_EMBEDDED_GEOMETRY", "qwall")))
embedded_geometry in (:qwall, :circle) ||
    error("KRAKEN_LOGFV_EMBEDDED_GEOMETRY must be qwall or circle")
embedded_gradient = parse_bool_env("KRAKEN_LOGFV_EMBEDDED_GRADIENT", false)
embedded_advection = parse_bool_env("KRAKEN_LOGFV_EMBEDDED_ADVECTION", false)
embedded_force = parse_bool_env("KRAKEN_LOGFV_EMBEDDED_FORCE", false)
embedded_drag = parse_bool_env("KRAKEN_LOGFV_EMBEDDED_DRAG", false)
embedded_circle_samples = parse(Int, get(ENV, "KRAKEN_LOGFV_EMBEDDED_CIRCLE_SAMPLES", "32"))
force_boundary_fill = Symbol(lowercase(get(ENV, "KRAKEN_FORCE_BOUNDARY_FILL", "bc_aware")))
force_boundary_fill in (:bc_aware, :nearest, :none) ||
    error("KRAKEN_FORCE_BOUNDARY_FILL must be bc_aware, nearest, or none")
allow_long_local = parse_bool_env("KRAKEN_ALLOW_LONG_LOCAL", false)
max_local_updates = parse(Float64, get(ENV, "KRAKEN_MAX_LOCAL_UPDATES", "5e7"))
continue_on_error = parse_bool_env("KRAKEN_CONTINUE_ON_ERROR", !smoke)
save_fields = parse_bool_env("KRAKEN_SAVE_FIELDS", false)
write_field_vtk = parse_bool_env("KRAKEN_WRITE_FIELD_VTK", save_fields)

out_dir = get(ENV, "KRAKEN_OUTPUT_DIR",
              joinpath("tmp", "viscoelastic_logfv", "logfv_cylinder_cd_convergence"))
mkpath(out_dir)
timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
csv_path = get(ENV, "KRAKEN_OUTPUT_CSV",
               joinpath(out_dir, "logfv_cylinder_cd_convergence_$(timestamp).csv"))
field_root = get(ENV, "KRAKEN_FIELD_OUTPUT_DIR", joinpath(out_dir, "fields"))
save_fields && mkpath(field_root)

println("="^78)
println("Cell-centered log-FV confined-cylinder Cd convergence")
println("Backend: $(backend_label), FT=$(FT), suite=$(suite), smoke=$(smoke)")
println("R=$(join(R_values, ",")) Wi=$(join(Wi_values, ","))")
println("geometry: H=$(H_factor)R L_up=$(L_up)R L_down=$(L_down)R Fx_body=$(Fx_body)")
println("flow: Re_R=$(Re_R) beta=$(beta) u_mean=$(u_mean)")
println("polymer_model=$(polymer_model) L_max=$(polymer_model === :fenep ? L_max : 0.0)")
println("embedded_geometry=$(embedded_geometry) embedded_gradient=$(embedded_gradient) embedded_advection=$(embedded_advection) embedded_force=$(embedded_force) embedded_drag=$(embedded_drag) circle_samples=$(embedded_circle_samples) force_boundary_fill=$(force_boundary_fill)")
println("output=$(csv_path)")
save_fields && println("field_output=$(field_root) vtk=$(write_field_vtk)")
println("="^78)

errors = String[]
newtonian_cd_by_R = Dict{Int,Float64}()

for R in R_values
    H = round(Int, H_factor * R)
    Nx = ceil(Int, (L_up + L_down) * R)
    Ny = H
    nu_total = u_mean * R / Re_R
    steps_newtonian = steps_for_R(R, 0.0, steps_low_wi, steps_default,
                                  scale_steps, step_cap)
    avg_window_newtonian = max(1, steps_newtonian ÷ avg_divisor)

    if run_newtonian
        nu_s = nu_total
        nu_p = 0.0
        lambda = 1.0
        newtonian_polymer_substeps = 1
        row = row_base(; timestamp, suite, backend_label, FT,
            case_name="logfv_newtonian", R, Nx, Ny, H, L_up, L_down,
            Re_R, beta, Wi=0.0, u_mean, nu_total, nu_s, nu_p, lambda,
            polymer_model=:none, L_max=0.0,
            bsd_fraction=0.0, Fx_body, steps=steps_newtonian,
            avg_window=avg_window_newtonian, drag_stride,
            diagnostic_stride,
            embedded_geometry,
            embedded_gradient,
            embedded_advection,
            embedded_force,
            embedded_drag,
            embedded_circle_samples,
            force_boundary_fill,
            polymer_substeps=newtonian_polymer_substeps,
            subcycle_relative_tolerance, max_deformation_increment,
            max_memory_deformation_increment)
        try
            guard_local_run!(backend_kind, Nx, Ny, steps_newtonian,
                             allow_long_local, max_local_updates)
            @printf("R=%d Newtonian log-FV steps=%d ...\n", R, steps_newtonian)
            t0 = time()
            result = Kraken.run_viscoelastic_logfv_cylinder_coupled_2d(;
                radius=R,
                H,
                L_up,
                L_down,
                nu_s=FT(nu_s),
                nu_p=FT(nu_p),
                lambda=FT(lambda),
                polymer_model=:oldroydb,
                L_max=FT(0),
                u_mean=FT(u_mean),
                Fx_body=FT(Fx_body),
                bsd_fraction=FT(0),
                polymer_substeps=newtonian_polymer_substeps,
                subcycle_relative_tolerance=FT(subcycle_relative_tolerance),
                max_deformation_increment=FT(max_deformation_increment),
                max_memory_deformation_increment=FT(max_memory_deformation_increment),
                max_polymer_substeps,
                max_steps=steps_newtonian,
                avg_window=avg_window_newtonian,
                drag_stride,
                diagnostic_stride,
                embedded_geometry,
                embedded_gradient,
                embedded_advection,
                embedded_force,
                embedded_drag,
                embedded_circle_samples,
                force_boundary_fill,
                backend,
                T=FT,
            )
            dt = time() - t0
            fill_result_row!(row, result, dt, R, u_mean, NaN, NaN, NaN, NaN)
            newtonian_cd_by_R[R] = result.Cd
            bad = mark_nonfinite_result!(
                row, errors, "R=$(R) newtonian"; continue_on_error,
            )
            @printf("  status=%s Cd=%.9g Cd_s=%.9g Cd_bsd=%.9g n=%d err_newt=%.4g%% substeps=%d clamped=%s first_bad=%d MLUPS=%.2f dt=%.1fs\n",
                    bad ? "nonfinite" : "ok",
                    result.Cd, result.Cd_s, result.Cd_bsd, result.n_drag_samples,
                    row[:err_newtonian_ref_pct], result.polymer_substeps,
                    string(result.subcycle_estimate.clamped),
                    result.first_nonfinite_step, row[:lups] / 1e6, dt)
            maybe_save_logfv_fields!(row, result, save_fields, field_root, write_field_vtk)
        catch err
            row[:status] = "error"
            row[:error] = sprint(showerror, err)
            push!(errors, "R=$(R) newtonian: $(row[:error])")
            @warn "Log-FV Newtonian case failed" R error=row[:error]
            append_csv_row(csv_path, row)
            continue_on_error || rethrow()
        end
        append_csv_row(csv_path, row)
    end

    run_viscoelastic || continue

    for Wi in Wi_values
        lambda = Wi * R / u_mean
        nu_s = beta * nu_total
        nu_p = (1.0 - beta) * nu_total
        max_steps = steps_for_R(R, Wi, steps_low_wi, steps_default,
                                scale_steps, step_cap)
        avg_window = max(1, max_steps ÷ avg_divisor)
        key = wi_key(Wi)
        liu_ref = get(LIU_REF, (R, key), NaN)
        rheo_mean = get(RHEOTOOL_REF_MEAN, key, NaN)
        rheo_last = get(RHEOTOOL_REF_LAST, key, NaN)
        newt_same = get(newtonian_cd_by_R, R, NaN)
        case_name = Wi < 0.01 ? "logfv_near_newtonian" : "logfv_$(polymer_model)"
        row = row_base(; timestamp, suite, backend_label, FT,
            case_name, R, Nx, Ny, H, L_up, L_down, Re_R, beta, Wi,
            u_mean, nu_total, nu_s, nu_p, lambda, polymer_model, L_max,
            bsd_fraction, Fx_body,
            steps=max_steps, avg_window, drag_stride, polymer_substeps,
            diagnostic_stride,
            embedded_geometry,
            embedded_gradient,
            embedded_advection,
            embedded_force,
            embedded_drag,
            embedded_circle_samples,
            force_boundary_fill,
            subcycle_relative_tolerance, max_deformation_increment,
            max_memory_deformation_increment)
        try
            guard_local_run!(backend_kind, Nx, Ny, max_steps,
                             allow_long_local, max_local_updates)
            @printf("R=%d Wi=%.6g log-FV model=%s steps=%d ...\n",
                    R, Wi, string(polymer_model), max_steps)
            t0 = time()
            result = Kraken.run_viscoelastic_logfv_cylinder_coupled_2d(;
                radius=R,
                H,
                L_up,
                L_down,
                nu_s=FT(nu_s),
                nu_p=FT(nu_p),
                lambda=FT(lambda),
                polymer_model,
                L_max=FT(L_max),
                u_mean=FT(u_mean),
                Fx_body=FT(Fx_body),
                bsd_fraction=FT(bsd_fraction),
                polymer_substeps,
                subcycle_relative_tolerance=FT(subcycle_relative_tolerance),
                max_deformation_increment=FT(max_deformation_increment),
                max_memory_deformation_increment=FT(max_memory_deformation_increment),
                max_polymer_substeps,
                max_steps,
                avg_window,
                drag_stride,
                diagnostic_stride,
                embedded_geometry,
                embedded_gradient,
                embedded_advection,
                embedded_force,
                embedded_drag,
                embedded_circle_samples,
                force_boundary_fill,
                backend,
                T=FT,
            )
            dt = time() - t0
            fill_result_row!(row, result, dt, R, u_mean, newt_same,
                             liu_ref, rheo_mean, rheo_last)
            bad = mark_nonfinite_result!(
                row, errors, "R=$(R) Wi=$(Wi)"; continue_on_error,
            )
            @printf("  status=%s Cd=%.9g Cd_s=%.9g Cd_p=%.9g Cd_bsd=%.9g n=%d Liu_err=%.4g%% Rheo_err=%.4g%% Newt_same=%.4g%% minCeig=%.4g trCmax=%.4g fmax=%.4g substeps=%d clamped=%s first_bad=%d MLUPS=%.2f dt=%.1fs\n",
                    bad ? "nonfinite" : "ok",
                    result.Cd, result.Cd_s, result.Cd_p, result.Cd_bsd, result.n_drag_samples,
                    row[:err_liu_pct], row[:err_rheotool_mean_pct],
                    row[:err_newtonian_same_run_pct], result.min_c_eig,
                    result.max_c_trace, result.max_fene_factor,
                    result.polymer_substeps,
                    string(result.subcycle_estimate.clamped),
                    result.first_nonfinite_step, row[:lups] / 1e6, dt)
            maybe_save_logfv_fields!(row, result, save_fields, field_root, write_field_vtk)
        catch err
            row[:status] = "error"
            row[:error] = sprint(showerror, err)
            push!(errors, "R=$(R) Wi=$(Wi): $(row[:error])")
            @warn "Log-FV viscoelastic case failed" R Wi error=row[:error]
            append_csv_row(csv_path, row)
            continue_on_error || rethrow()
        end
        append_csv_row(csv_path, row)
    end
end

println("="^78)
println("CSV: $(csv_path)")
if isempty(errors)
    println("Done without case errors.")
else
    println("Done with $(length(errors)) case error(s):")
    foreach(err -> println("  - ", err), errors)
    continue_on_error || error("log-FV cylinder convergence failed")
end
