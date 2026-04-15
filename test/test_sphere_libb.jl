using Test
using Kraken
using KernelAbstractions

# ==========================================================================
# 3D sphere in uniform x-flow with LI-BB V2 (Bouzidi pre-phase + TRT).
#
# Tries Metal GPU first (M-series Macs), else CUDA, else CPU with a
# much smaller grid so runtests.jl stays cheap. GPU path is canonical
# (project convention: always GPU locally).
# ==========================================================================

const _METAL_BACKEND = try
    @eval using Metal
    Metal.MetalBackend()
catch
    nothing
end

const _CUDA_BACKEND = if _METAL_BACKEND === nothing
    try
        @eval using CUDA
        CUDA.functional() ? CUDA.CUDABackend() : nothing
    catch
        nothing
    end
else
    nothing
end

_pick_3d_backend() =
    _METAL_BACKEND !== nothing ? (:Metal, _METAL_BACKEND) :
    _CUDA_BACKEND  !== nothing ? (:CUDA,  _CUDA_BACKEND)  :
                                  (:CPU,  CPU())

@testset "Sphere LI-BB V2 — 3D scaffold" begin
    bname, backend = _pick_3d_backend()
    on_gpu = bname !== :CPU
    FT = on_gpu && bname === :Metal ? Float32 : Float64

    # GPU: full-size benchmark (~10s). CPU fallback: small grid, few
    # steps — only verifies the code path, no quantitative asserts.
    Nx, Ny, Nz, radius, steps, window = on_gpu ?
        (80, 40, 40, 6, 5_000, 1_000) :
        (16, 8, 8, 2, 200, 100)

    u_in = FT(0.04)
    Re = 20.0
    D = 2 * radius
    ν = FT(u_in * D / Re)

    result = run_sphere_libb_3d(; Nx=Nx, Ny=Ny, Nz=Nz,
                                  cx=Nx÷4, cy=Ny÷2, cz=Nz÷2,
                                  radius=radius, u_in=u_in, ν=ν,
                                  max_steps=steps, avg_window=window,
                                  backend=backend, T=FT)

    @test !any(isnan, result.ρ)
    @test !any(isnan, result.ux)
    @test !any(isnan, result.uy)
    @test !any(isnan, result.uz)

    if on_gpu
        @test result.Cd > 0.2
        @test result.Cd < 5.0
        @test abs(result.Fy / result.Fx) < 0.05
        @test abs(result.Fz / result.Fx) < 0.05
        @test maximum(result.ρ) < 1.3
        @test minimum(result.ρ) > 0.7
    end

    @info "Sphere LI-BB Re=20 (scaffold)" backend=bname Cd=result.Cd maxρm1=maximum(abs.(result.ρ .- 1))
end
