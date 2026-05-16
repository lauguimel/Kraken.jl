#!/usr/bin/env julia

using KernelAbstractions
using Kraken
using Printf
using Test

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

function parse_mode(args::Vector{String})
    isempty(args) || args == ["--self-test"] || args == ["--full"] ||
        error("usage: julia --project=. bench/viscoelastic_logfv/run_wall_stencil_audit_2d.jl [--self-test|--full]")
    return (N=32, end_time=2.0)
end

function run_cavity_snapshot(; N::Int, end_time::Float64, backend, T, polymer_wall_extrap::Symbol)
    result = run_viscoelastic_logfv_cavity_coupled_2d(;
        N=N,
        end_time=end_time,
        backend=backend,
        T=T,
        nu_s=0.1,
        nu_p=0.1,
        lambda_phys=1.0,
        bsd_fraction=0.75,
        u_max=0.005,
        polymer_model=:oldroydb,
        sample_times=Float64[end_time],
        skip_top_corners=false,
        bsd_kind=:fd,
        polymer_wall_extrap=polymer_wall_extrap,
    )
    @assert result.first_nonfinite_step == 0
    snapshot_key = sort(collect(keys(result.snapshots)))[end]
    return result.snapshots[snapshot_key]
end

function polymer_force(snapshot, polymer_wall_extrap::Symbol)
    tauxx = snapshot.tauxx
    tauxy = snapshot.tauxy
    tauyy = snapshot.tauyy
    T = eltype(tauxx)
    N = size(tauxx, 1)
    fx = zeros(T, N, N)
    fy = zeros(T, N, N)
    Kraken.logfv_polymer_force_bc_aware_2d!(
        fx, fy, tauxx, tauxy, tauyy, falses(N, N), one(T), one(T),
        Kraken.logfv_wallxwally_bcspec_2d();
        polymer_wall_extrap=polymer_wall_extrap,
    )
    return fx, fy
end

function force_metrics(fq, fl)
    qx, qy = fq
    lx, ly = fl
    Nx, Ny = size(qx)
    wall_delta2 = 0.0
    wall_ref2 = 0.0
    bulk_far_max_abs = 0.0
    for j in 1:Ny, i in 1:Nx
        dx = Float64(lx[i, j] - qx[i, j])
        dy = Float64(ly[i, j] - qy[i, j])
        if i <= 2 || i >= Nx - 1 || j <= 2 || j >= Ny - 1
            wall_delta2 += dx * dx + dy * dy
            wall_ref2 += Float64(qx[i, j])^2 + Float64(qy[i, j])^2
        end
        if 4 <= i <= Nx - 3 && 4 <= j <= Ny - 3
            bulk_far_max_abs = max(bulk_far_max_abs, abs(dx), abs(dy))
        end
    end
    return sqrt(wall_delta2) / sqrt(wall_ref2), bulk_far_max_abs
end

function write_centerline_csv(path::AbstractString, quad, lin)
    N = size(quad.ux, 1)
    i0 = N ÷ 2
    i1 = i0 + 1
    open(path, "w") do io
        write(io, "y,u_quadratic,u_linear\n")
        for j in 1:N
            y = (j - 0.5) / N
            uq = 0.5 * (Float64(quad.ux[i0, j]) + Float64(quad.ux[i1, j]))
            ul = 0.5 * (Float64(lin.ux[i0, j]) + Float64(lin.ux[i1, j]))
            @printf(io, "%.10g,%.10g,%.10g\n", y, uq, ul)
        end
    end
    @assert countlines(path) == N + 1
    return path
end

function run_audit()
    cfg = parse_mode(ARGS)
    _, backend, T = pick_backend()
    quad = run_cavity_snapshot(;
        N=cfg.N, end_time=cfg.end_time, backend=backend, T=T, polymer_wall_extrap=:quadratic,
    )
    lin = run_cavity_snapshot(;
        N=cfg.N, end_time=cfg.end_time, backend=backend, T=T, polymer_wall_extrap=:linear,
    )
    fq = polymer_force(quad, :quadratic)
    fl = polymer_force(lin, :linear)
    wall_row_rel_l2, bulk_far_max_abs = force_metrics(fq, fl)
    # bulk_far_max_abs is non-zero because the wall-stencil change propagates
    # inward through advection over the N_LBM steps of the simulation; the
    # bar < 1e-3 is well below the wall-row signal (< 0.5) and above the
    # numerical noise floor observed on the smoke (~5e-8).
    status = isfinite(wall_row_rel_l2) && wall_row_rel_l2 > 0.0 &&
             wall_row_rel_l2 < 0.5 && bulk_far_max_abs < 1.0e-3
    mktempdir() do dir
        write_centerline_csv(joinpath(dir, "wall_stencil_centerline_u.csv"), quad, lin)
        @printf("RESULT wall_row_rel_l2=%.16e  bulk_far_max_abs=%.16e  status=%s\n",
                wall_row_rel_l2, bulk_far_max_abs, status ? "PASS" : "FAIL")
        @assert status "wall stencil audit failed"
        @testset "M6-B wall stencil audit" begin
            @test isfinite(wall_row_rel_l2)
            @test wall_row_rel_l2 > 0.0
            @test wall_row_rel_l2 < 0.5
            @test bulk_far_max_abs < 1.0e-3
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_audit()
end
