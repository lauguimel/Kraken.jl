#!/usr/bin/env julia

using Dates
using KernelAbstractions
using Printf
using WriteVTK: vtk_save

using Kraken

const _METAL_BACKEND_STATUS = if Sys.isapple()
    try
        @eval using Metal
        Metal.functional() ? (:Metal, Metal.MetalBackend(), Float32, nothing) :
        (:none, nothing, Float32, "Metal is not functional")
    catch err
        (:none, nothing, Float32, err)
    end
else
    (:none, nothing, Float32, "Metal is only supported on macOS")
end

const _CUDA_BACKEND_STATUS = try
    @eval using CUDA
    if Base.invokelatest(CUDA.functional)
        backend = Base.invokelatest(CUDA.CUDABackend)
        (:CUDA, backend, Float64, nothing)
    else
        (:none, nothing, Float64, "CUDA is not functional")
    end
catch err
    (:none, nothing, Float64, err)
end

const SUMMARY_COLUMNS = [
    :case,
    :status,
    :backend,
    :T,
    :Nx,
    :Ny,
    :steps,
    :polymer_substeps,
    :nu_s,
    :nu_p,
    :Fx_body,
    :lambda,
    :bsd_fraction,
    :min_c_eig,
    :max_speed,
    :max_abs_psi,
    :max_abs_tau,
    :rho_min,
    :rho_max,
    :max_rel_error,
    :first_nonfinite_step,
    :pvd,
]

function _csv_cell(x)
    x === nothing && return ""
    s = string(x)
    if occursin(',', s) || occursin('"', s) || occursin('\n', s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

function _append_summary!(path::String, row)
    new_file = !isfile(path)
    open(path, "a") do io
        if new_file
            println(io, join(string.(SUMMARY_COLUMNS), ","))
        end
        println(io, join((_csv_cell(get(row, col, "")) for col in SUMMARY_COLUMNS), ","))
    end
    return nothing
end

function _backend_from_env()
    requested = lowercase(get(ENV, "KRAKEN_BACKEND", "cpu"))
    if requested == "cuda"
        cuda_name, cuda_backend, cuda_ft, cuda_error = _CUDA_BACKEND_STATUS
        if cuda_backend !== nothing
            return (cuda_name, cuda_backend, cuda_ft)
        end
        @warn "CUDA requested but unavailable; falling back to CPU" reason=cuda_error
    elseif requested == "metal"
        metal_name, metal_backend, metal_ft, metal_error = _METAL_BACKEND_STATUS
        if metal_backend !== nothing
            return (metal_name, metal_backend, metal_ft)
        end
        @warn "Metal requested but unavailable; falling back to CPU" reason=metal_error
    elseif requested != "cpu"
        @warn "Unsupported KRAKEN_BACKEND=$(requested); expected cpu, metal, or cuda; falling back to CPU"
    end
    return (:CPU, KernelAbstractions.CPU(), Float64)
end

function _matrix_or_zeros(result, name::Symbol)
    hasproperty(result, name) && return Matrix{Float64}(getproperty(result, name))
    return zeros(Float64, result.Nx, result.Ny)
end

function _solid_mask(result)
    hasproperty(result, :is_solid) && return Matrix{Bool}(result.is_solid)
    return falses(result.Nx, result.Ny)
end

function _conformation_from_psi(result)
    Nx, Ny = result.Nx, result.Ny
    cxx = Matrix{Float64}(undef, Nx, Ny)
    cxy = Matrix{Float64}(undef, Nx, Ny)
    cyy = Matrix{Float64}(undef, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        cxx_ij, cxy_ij, cyy_ij = Kraken.logfv_exp_sym2_2d(
            result.psixx[i, j], result.psixy[i, j], result.psiyy[i, j],
        )
        cxx[i, j] = Float64(cxx_ij)
        cxy[i, j] = Float64(cxy_ij)
        cyy[i, j] = Float64(cyy_ij)
    end
    return cxx, cxy, cyy
end

function _stress_fields(result, cxx, cxy, cyy)
    if hasproperty(result, :tauxx) && hasproperty(result, :tauxy) && hasproperty(result, :tauyy)
        return (
            Matrix{Float64}(result.tauxx),
            Matrix{Float64}(result.tauxy),
            Matrix{Float64}(result.tauyy),
        )
    end
    prefactor = Float64(result.nu_p / result.lambda)
    return (
        prefactor .* (cxx .- 1.0),
        prefactor .* cxy,
        prefactor .* (cyy .- 1.0),
    )
end

function _vtk_fields(result)
    cxx, cxy, cyy = _conformation_from_psi(result)
    tauxx, tauxy, tauyy = _stress_fields(result, cxx, cxy, cyy)
    is_solid = _solid_mask(result)
    ux = Matrix{Float64}(result.ux)
    uy = Matrix{Float64}(result.uy)
    fields = Dict{String,AbstractMatrix}(
        "rho" => _matrix_or_zeros(result, :rho),
        "ux" => ux,
        "uy" => uy,
        "speed" => hypot.(ux, uy),
        "psixx" => Matrix{Float64}(result.psixx),
        "psixy" => Matrix{Float64}(result.psixy),
        "psiyy" => Matrix{Float64}(result.psiyy),
        "cxx" => cxx,
        "cxy" => cxy,
        "cyy" => cyy,
        "tauxx" => tauxx,
        "tauxy" => tauxy,
        "tauyy" => tauyy,
        "fx_poly" => _matrix_or_zeros(result, :fx_poly),
        "fy_poly" => _matrix_or_zeros(result, :fy_poly),
        "fx_total" => _matrix_or_zeros(result, :fx_total),
        "fy_total" => _matrix_or_zeros(result, :fy_total),
        "is_solid" => Float64.(is_solid),
    )
    return fields
end

function _finite_fields(result)
    for name in (:rho, :ux, :uy, :psixx, :psixy, :psiyy, :tauxx, :tauxy, :tauyy, :fx_total, :fy_total)
        hasproperty(result, name) || continue
        all(isfinite, getproperty(result, name)) || return false
    end
    return true
end

function _max_speed(result)
    hasproperty(result, :max_speed) && return Float64(result.max_speed)
    return maximum(hypot.(Matrix{Float64}(result.ux), Matrix{Float64}(result.uy)))
end

function _max_abs_psi(result)
    hasproperty(result, :max_abs_psi) && return Float64(result.max_abs_psi)
    return max(maximum(abs, result.psixx), maximum(abs, result.psixy), maximum(abs, result.psiyy))
end

function _max_abs_tau(result, fields)
    hasproperty(result, :max_abs_tau) && return Float64(result.max_abs_tau)
    return max(maximum(abs, fields["tauxx"]), maximum(abs, fields["tauxy"]), maximum(abs, fields["tauyy"]))
end

function _rho_bounds(result)
    hasproperty(result, :rho_min) && hasproperty(result, :rho_max) &&
        return (Float64(result.rho_min), Float64(result.rho_max))
    rho = Matrix{Float64}(result.rho)
    return (minimum(rho), maximum(rho))
end

function _case_status(result; min_c_eig_floor, max_speed_ceiling, rho_min_floor, rho_max_ceiling)
    first_bad = hasproperty(result, :first_nonfinite_step) ? result.first_nonfinite_step : 0
    rho_min, rho_max = _rho_bounds(result)
    return _finite_fields(result) &&
           first_bad == 0 &&
           Float64(result.min_c_eig) > min_c_eig_floor &&
           _max_speed(result) < max_speed_ceiling &&
           rho_min > rho_min_floor &&
           rho_max < rho_max_ceiling
end

function _run_case!(
    rows::Vector{Dict{Symbol,Any}},
    summary_path::String,
    output_root::String,
    case_name::String,
    runner,
    backend_name::Symbol,
    backend,
    FT;
    min_c_eig_floor::Real,
    max_speed_ceiling::Real,
    rho_min_floor::Real,
    rho_max_ceiling::Real,
)
    @info "Running simple log-FV validation case" case=case_name backend=backend_name FT
    result = runner(backend, FT)

    case_dir = Kraken.setup_output_dir(joinpath(output_root, case_name))
    pvd_path = joinpath(case_dir, case_name)
    pvd = Kraken.create_pvd(pvd_path)
    fields = _vtk_fields(result)
    step = hasproperty(result, :completed_steps) ? result.completed_steps :
           hasproperty(result, :max_steps) ? result.max_steps : 0
    Kraken.write_snapshot_2d!(
        case_dir, Int(step), result.Nx, result.Ny, 1.0, fields;
        pvd, time=Float64(step),
    )
    vtk_save(pvd)

    ok = _case_status(
        result;
        min_c_eig_floor,
        max_speed_ceiling,
        rho_min_floor,
        rho_max_ceiling,
    )
    rho_min, rho_max = _rho_bounds(result)
    row = Dict{Symbol,Any}(
        :case => case_name,
        :status => ok ? "pass" : "fail",
        :backend => backend_name,
        :T => FT,
        :Nx => result.Nx,
        :Ny => result.Ny,
        :steps => step,
        :polymer_substeps => hasproperty(result, :polymer_substeps) ? result.polymer_substeps : "",
        :nu_s => result.nu_s,
        :nu_p => result.nu_p,
        :Fx_body => hasproperty(result, :Fx_body) ? result.Fx_body : "",
        :lambda => result.lambda,
        :bsd_fraction => hasproperty(result, :bsd_fraction) ? result.bsd_fraction : "",
        :min_c_eig => result.min_c_eig,
        :max_speed => _max_speed(result),
        :max_abs_psi => _max_abs_psi(result),
        :max_abs_tau => _max_abs_tau(result, fields),
        :rho_min => rho_min,
        :rho_max => rho_max,
        :max_rel_error => hasproperty(result, :max_rel_error) ? result.max_rel_error : "",
        :first_nonfinite_step => hasproperty(result, :first_nonfinite_step) ? result.first_nonfinite_step : 0,
        :pvd => pvd_path * ".pvd",
    )
    _append_summary!(summary_path, row)
    push!(rows, row)

    @info "Finished simple log-FV validation case" case=case_name status=row[:status] pvd=row[:pvd]
    return ok
end

function main()
    backend_name, backend, FT = _backend_from_env()
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    output_root = Kraken.setup_output_dir(
        get(ENV, "KRAKEN_OUTPUT_DIR", joinpath("tmp", "logfv_simple_validation_outputs", timestamp)),
    )
    summary_path = joinpath(output_root, "summary.csv")
    rows = Dict{Symbol,Any}[]

    cases = (
        (
            name="poiseuille_coupled",
            min_c=0.95,
            max_speed=0.02,
            rho_min=0.98,
            rho_max=1.02,
            run=(backend, FT) -> Kraken.run_viscoelastic_logfv_poiseuille_coupled_2d(;
                Nx=6, Ny=16,
                nu_s=0.04, nu_p=0.06, Fx_body=1e-5,
                lambda=5.0, bsd_fraction=1.0,
                polymer_substeps=:auto,
                max_steps=4000,
                backend, T=FT,
            ),
        ),
        (
            name="square_periodic",
            min_c=0.90,
            max_speed=0.003,
            rho_min=0.99,
            rho_max=1.01,
            run=(backend, FT) -> Kraken.run_viscoelastic_logfv_square_periodic_2d(;
                Nx=28, Ny=14, side=4,
                nu_s=0.08, nu_p=0.02, Fx_body=1e-6,
                lambda=5.0, bsd_fraction=1.0,
                polymer_substeps=:auto,
                max_steps=150,
                backend, T=FT,
            ),
        ),
        (
            name="square_channel",
            min_c=0.90,
            max_speed=0.05,
            rho_min=0.98,
            rho_max=1.03,
            run=(backend, FT) -> Kraken.run_viscoelastic_logfv_square_channel_coupled_2d(;
                H=12, side=4, L_up=2, L_down=3,
                nu_s=0.08, nu_p=0.02, lambda=5.0,
                u_mean=0.01, Fx_body=2e-7,
                bsd_fraction=1.0,
                max_steps=20,
                diagnostic_stride=1,
                backend, T=FT,
            ),
        ),
        (
            name="bfs_coupled",
            min_c=0.90,
            max_speed=0.05,
            rho_min=0.98,
            rho_max=1.02,
            run=(backend, FT) -> Kraken.run_viscoelastic_logfv_bfs_coupled_2d(;
                H_in=4, expansion_ratio=2, L_up=2, L_down=4,
                nu_s=0.08, nu_p=0.02, lambda=5.0,
                u_mean=0.01, Fx_body=2e-7,
                bsd_fraction=1.0,
                max_steps=40,
                diagnostic_stride=1,
                backend, T=FT,
            ),
        ),
    )

    ok = true
    for case in cases
        ok &= _run_case!(
            rows, summary_path, output_root, case.name, case.run,
            backend_name, backend, FT;
            min_c_eig_floor=case.min_c,
            max_speed_ceiling=case.max_speed,
            rho_min_floor=case.rho_min,
            rho_max_ceiling=case.rho_max,
        )
    end

    println("Output directory: ", output_root)
    println("Summary CSV: ", summary_path)
    for row in rows
        @printf(
            "%-18s %-4s minC=% .6e maxU=% .6e rho=[% .6e,% .6e] pvd=%s\n",
            row[:case],
            row[:status],
            Float64(row[:min_c_eig]),
            Float64(row[:max_speed]),
            Float64(row[:rho_min]),
            Float64(row[:rho_max]),
            row[:pvd],
        )
    end

    ok || error("at least one simple log-FV validation case failed; see $(summary_path)")
    return nothing
end

main()
