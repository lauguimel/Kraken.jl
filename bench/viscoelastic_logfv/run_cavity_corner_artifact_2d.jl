#!/usr/bin/env julia

using KernelAbstractions
using Printf

using Kraken

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

function pick_backend()
    requested = lowercase(get(ENV, "KRAKEN_BACKEND", "auto"))
    if requested in ("metal", "mtl") && METAL_MOD !== nothing
        return :metal, METAL_MOD.MetalBackend(), Float32
    elseif requested in ("cuda", "gpu") && CUDA_MOD !== nothing
        return :cuda, CUDA_MOD.CUDABackend(), Float64
    elseif requested == "cpu"
        return :cpu, KernelAbstractions.CPU(), Float64
    end
    if METAL_MOD !== nothing && Sys.isapple()
        return :metal, METAL_MOD.MetalBackend(), Float32
    elseif CUDA_MOD !== nothing && CUDA_MOD.functional()
        return :cuda, CUDA_MOD.CUDABackend(), Float64
    end
    return :cpu, KernelAbstractions.CPU(), Float64
end

function sample_horizontal_kraken(
    field::AbstractMatrix, Nx::Integer, Ny::Integer, y_frac::Real,
    x_target::AbstractVector,
)
    j_real = y_frac * Ny + 0.5
    j_lo = clamp(floor(Int, j_real), 1, Ny - 1)
    j_hi = j_lo + 1
    wy = clamp(j_real - j_lo, 0.0, 1.0)
    row = [(1 - wy) * field[i, j_lo] + wy * field[i, j_hi] for i in 1:Nx]
    x_phys = [(i - 0.5) / Nx for i in 1:Nx]
    out = similar(x_target, Float64)
    for (k, x) in enumerate(x_target)
        if x <= x_phys[1]
            out[k] = row[1]
        elseif x >= x_phys[end]
            out[k] = row[end]
        else
            i_lo = searchsortedlast(x_phys, x)
            i_hi = i_lo + 1
            wx = (x - x_phys[i_lo]) / (x_phys[i_hi] - x_phys[i_lo])
            out[k] = (1 - wx) * row[i_lo] + wx * row[i_hi]
        end
    end
    return (x=collect(x_target), values=out)
end

function parse_mode(args::Vector{String})
    if isempty(args) || args == ["--self-test"]
        return (mode=:self_test, N=32, end_time=2.0)
    elseif args == ["--full"]
        return (mode=:full, N=64, end_time=8.0)
    end
    error("usage: julia --project=. bench/viscoelastic_logfv/run_cavity_corner_artifact_2d.jl [--self-test|--full]")
end

function run_profile(; N::Int, end_time::Float64, backend, T, skip_top_corners::Bool)
    result = run_viscoelastic_logfv_cavity_coupled_2d(;
        N=N,
        nu_s=0.1,
        nu_p=0.1,
        lambda_phys=1.0,
        bsd_fraction=0.75,
        u_max=0.005,
        polymer_model=:oldroydb,
        end_time=end_time,
        sample_times=Float64[end_time],
        skip_top_corners=skip_top_corners,
        backend=backend,
        T=T,
    )
    @assert result.first_nonfinite_step == 0
    snapshot_key = sort(collect(keys(result.snapshots)))[end]
    psixy_host = Array(result.snapshots[snapshot_key].psixy)
    x_target = [(i - 0.5) / N for i in 1:N]
    sampled = sample_horizontal_kraken(psixy_host, N, N, 0.75, x_target)
    return (result=result, x=sampled.x, psi_xy=sampled.values)
end

function write_profile_csv(path::AbstractString, x::AbstractVector, psi_xy::AbstractVector)
    @assert length(x) == length(psi_xy)
    open(path, "w") do io
        write(io, "x,psi_xy\n")
        for k in eachindex(x)
            @printf(io, "%.10g,%.10g\n", x[k], psi_xy[k])
        end
    end
    return path
end

function main()
    cfg = parse_mode(ARGS)
    _, backend, T = pick_backend()

    default = run_profile(;
        N=cfg.N, end_time=cfg.end_time, backend=backend, T=T,
        skip_top_corners=false,
    )
    skipped = run_profile(;
        N=cfg.N, end_time=cfg.end_time, backend=backend, T=T,
        skip_top_corners=true,
    )

    mktempdir() do dir
        default_path = write_profile_csv(
            joinpath(dir, "cavity_corner_default_psixy_y075.csv"),
            default.x, default.psi_xy,
        )
        skipped_path = write_profile_csv(
            joinpath(dir, "cavity_corner_skip_psixy_y075.csv"),
            skipped.x, skipped.psi_xy,
        )
        @assert isfile(default_path)
        @assert isfile(skipped_path)
        @assert countlines(default_path) == cfg.N + 1
        @assert countlines(skipped_path) == cfg.N + 1

        delta = abs.(skipped.psi_xy .- default.psi_xy)
        corner_dmax = maximum(delta[default.x .<= 0.3])
        bulk_dmax = maximum(delta[(default.x .>= 0.3) .& (default.x .<= 0.7)])
        @assert isfinite(corner_dmax)
        @assert isfinite(bulk_dmax)

        println("region   | max|Δpsi_xy|")
        println("-------- | --------------")
        println("corner   | $(corner_dmax)")
        println("bulk     | $(bulk_dmax)")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
