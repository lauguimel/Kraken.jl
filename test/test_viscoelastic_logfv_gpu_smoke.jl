using Test
using KernelAbstractions
using Kraken

const _LOGFV_METAL_BACKEND = try
    @eval using Metal
    Metal.functional() ? Metal.MetalBackend() : nothing
catch
    nothing
end

const _LOGFV_CUDA_BACKEND = if _LOGFV_METAL_BACKEND === nothing
    try
        @eval using CUDA
        CUDA.functional() ? CUDA.CUDABackend() : nothing
    catch
        nothing
    end
else
    nothing
end

_logfv_gpu_backend() =
    _LOGFV_METAL_BACKEND !== nothing ? (:Metal, _LOGFV_METAL_BACKEND, Float32) :
    _LOGFV_CUDA_BACKEND !== nothing ? (:CUDA, _LOGFV_CUDA_BACKEND, Float64) :
                                      (:none, nothing, Float32)

function _copy_to_backend(backend, A::AbstractArray{T}) where {T}
    B = KernelAbstractions.allocate(backend, T, size(A)...)
    copyto!(B, A)
    return B
end

@testset "Log-FV GPU smoke" begin
    backend_name, backend, FT = _logfv_gpu_backend()
    if backend === nothing
        @info "No Metal/CUDA backend available; skipping log-FV GPU smoke"
        @test true
    else
        Nx, Ny = 10, 9
        dx, dy = FT(0.3), FT(0.2)
        ax, ay = FT(0.04), FT(-0.03)
        bx, by = FT(-0.02), FT(0.05)
        ux_h = [FT(0.12) + ax * (FT(i) - FT(0.5)) * dx + ay * (FT(j) - FT(0.5)) * dy
                for i in 1:Nx, j in 1:Ny]
        uy_h = [FT(-0.08) + bx * (FT(i) - FT(0.5)) * dx + by * (FT(j) - FT(0.5)) * dy
                for i in 1:Nx, j in 1:Ny]
        is_solid_h = fill(false, Nx, Ny)
        is_solid_h[4:6, 4:6] .= true

        ux = _copy_to_backend(backend, ux_h)
        uy = _copy_to_backend(backend, uy_h)
        is_solid = _copy_to_backend(backend, is_solid_h)
        dudx = KernelAbstractions.zeros(backend, FT, Nx, Ny)
        dudy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
        dvdx = KernelAbstractions.zeros(backend, FT, Nx, Ny)
        dvdy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
        Kraken.logfv_velocity_gradient_solid_aware_2d!(dudx, dudy, dvdx, dvdy, ux, uy, is_solid, dx, dy)

        dudx_h = Array(dudx)
        dudy_h = Array(dudy)
        dvdx_h = Array(dvdx)
        dvdy_h = Array(dvdy)
        atol = FT === Float32 ? 2e-5 : 2e-12
        for j in 1:Ny, i in 1:Nx
            if is_solid_h[i, j]
                @test dudx_h[i, j] == 0
                @test dudy_h[i, j] == 0
                @test dvdx_h[i, j] == 0
                @test dvdy_h[i, j] == 0
            else
                x_has_neighbor = (i > 1 && !is_solid_h[i - 1, j]) || (i < Nx && !is_solid_h[i + 1, j])
                y_has_neighbor = (j > 1 && !is_solid_h[i, j - 1]) || (j < Ny && !is_solid_h[i, j + 1])
                @test Float64(dudx_h[i, j]) ≈ Float64(x_has_neighbor ? ax : 0) atol=atol rtol=atol
                @test Float64(dvdx_h[i, j]) ≈ Float64(x_has_neighbor ? bx : 0) atol=atol rtol=atol
                @test Float64(dudy_h[i, j]) ≈ Float64(y_has_neighbor ? ay : 0) atol=atol rtol=atol
                @test Float64(dvdy_h[i, j]) ≈ Float64(y_has_neighbor ? by : 0) atol=atol rtol=atol
            end
        end

        psixx = KernelAbstractions.zeros(backend, FT, Nx, Ny)
        psixy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
        psiyy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
        outxx = KernelAbstractions.zeros(backend, FT, Nx, Ny)
        outxy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
        outyy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
        Kraken.logfv_step_oldroydb_log_2d!(
            outxx, outxy, outyy,
            psixx, psixy, psiyy,
            dudx, dudy, dvdx, dvdy,
            FT(5), FT(0.1),
        )
        KernelAbstractions.synchronize(backend)
        @test all(isfinite, Array(outxx))
        @test all(isfinite, Array(outxy))
        @test all(isfinite, Array(outyy))

        coupled = Kraken.run_viscoelastic_logfv_poiseuille_coupled_2d(;
            Nx=6, Ny=12, nu_s=0.04, nu_p=0.06, Fx_body=1e-5,
            lambda=5.0, bsd_fraction=1.0, polymer_substeps=:auto,
            max_steps=200, backend=backend, T=FT,
        )
        @test coupled.min_c_eig > 0
        @test all(isfinite, coupled.ux)
        @test all(isfinite, coupled.psixx)
        @info "Log-FV GPU smoke passed" backend=backend_name FT=FT substeps=coupled.polymer_substeps
    end
end
