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

        tauxx_h = [FT(0.1) + FT(0.07) * (FT(i) - FT(0.5)) * dx - FT(0.01) * (FT(j) - FT(0.5)) * dy
                   for i in 1:Nx, j in 1:Ny]
        tauxy_h = [FT(-0.2) - FT(0.03) * (FT(i) - FT(0.5)) * dx + FT(0.02) * (FT(j) - FT(0.5)) * dy
                   for i in 1:Nx, j in 1:Ny]
        tauyy_h = [FT(0.3) + FT(0.04) * (FT(i) - FT(0.5)) * dx + FT(0.05) * (FT(j) - FT(0.5)) * dy
                   for i in 1:Nx, j in 1:Ny]
        tauxx = _copy_to_backend(backend, tauxx_h)
        tauxy = _copy_to_backend(backend, tauxy_h)
        tauyy = _copy_to_backend(backend, tauyy_h)
        fx = KernelAbstractions.zeros(backend, FT, Nx, Ny)
        fy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
        Kraken.logfv_polymer_force_solid_aware_2d!(fx, fy, tauxx, tauxy, tauyy, is_solid, dx, dy)
        fx_h = Array(fx)
        fy_h = Array(fy)
        for j in 1:Ny, i in 1:Nx
            if is_solid_h[i, j]
                @test fx_h[i, j] == 0
                @test fy_h[i, j] == 0
            else
                x_has_neighbor = (i > 1 && !is_solid_h[i - 1, j]) || (i < Nx && !is_solid_h[i + 1, j])
                y_has_neighbor = (j > 1 && !is_solid_h[i, j - 1]) || (j < Ny && !is_solid_h[i, j + 1])
                @test Float64(fx_h[i, j]) ≈ Float64((x_has_neighbor ? FT(0.07) : FT(0)) + (y_has_neighbor ? FT(0.02) : FT(0))) atol=atol rtol=atol
                @test Float64(fy_h[i, j]) ≈ Float64((x_has_neighbor ? FT(-0.03) : FT(0)) + (y_has_neighbor ? FT(0.05) : FT(0))) atol=atol rtol=atol
            end
        end

        ux_quad_h = [FT(0.1) + FT(0.06) * ((FT(i) - FT(0.5)) * dx)^2 -
                     FT(0.03) * ((FT(j) - FT(0.5)) * dy)^2
                     for i in 1:Nx, j in 1:Ny]
        uy_quad_h = [FT(-0.2) - FT(0.07) * ((FT(i) - FT(0.5)) * dx)^2 +
                     FT(0.01) * ((FT(j) - FT(0.5)) * dy)^2
                     for i in 1:Nx, j in 1:Ny]
        fx_poly_h = [FT(0.03) + FT(0.01) * (FT(i) - FT(0.5)) * dx
                     for i in 1:Nx, j in 1:Ny]
        fy_poly_h = [FT(-0.02) + FT(0.02) * (FT(j) - FT(0.5)) * dy
                     for i in 1:Nx, j in 1:Ny]
        ux_quad = _copy_to_backend(backend, ux_quad_h)
        uy_quad = _copy_to_backend(backend, uy_quad_h)
        fx_poly_bsd = _copy_to_backend(backend, fx_poly_h)
        fy_poly_bsd = _copy_to_backend(backend, fy_poly_h)
        fx_bsd = KernelAbstractions.zeros(backend, FT, Nx, Ny)
        fy_bsd = KernelAbstractions.zeros(backend, FT, Nx, Ny)
        Kraken.logfv_bsd_correct_force_solid_aware_2d!(
            fx_bsd, fy_bsd, fx_poly_bsd, fy_poly_bsd,
            ux_quad, uy_quad, is_solid, FT(0.6), FT(0.17), dx, dy,
        )
        fx_bsd_h = Array(fx_bsd)
        fy_bsd_h = Array(fy_bsd)
        for j in 1:Ny, i in 1:Nx
            if is_solid_h[i, j]
                @test fx_bsd_h[i, j] == 0
                @test fy_bsd_h[i, j] == 0
            else
                x_second =
                    (i > 1 && !is_solid_h[i - 1, j] && i < Nx && !is_solid_h[i + 1, j]) ||
                    (i + 2 <= Nx && !is_solid_h[i + 1, j] && !is_solid_h[i + 2, j]) ||
                    (i - 2 >= 1 && !is_solid_h[i - 1, j] && !is_solid_h[i - 2, j])
                y_second =
                    (j > 1 && !is_solid_h[i, j - 1] && j < Ny && !is_solid_h[i, j + 1]) ||
                    (j + 2 <= Ny && !is_solid_h[i, j + 1] && !is_solid_h[i, j + 2]) ||
                    (j - 2 >= 1 && !is_solid_h[i, j - 1] && !is_solid_h[i, j - 2])
                lap_ux = (x_second ? FT(0.12) : FT(0)) + (y_second ? FT(-0.06) : FT(0))
                lap_uy = (x_second ? FT(-0.14) : FT(0)) + (y_second ? FT(0.02) : FT(0))
                @test Float64(fx_bsd_h[i, j]) ≈ Float64(fx_poly_h[i, j] - FT(0.6) * FT(0.17) * lap_ux) atol=atol rtol=atol
                @test Float64(fy_bsd_h[i, j]) ≈ Float64(fy_poly_h[i, j] - FT(0.6) * FT(0.17) * lap_uy) atol=atol rtol=atol
            end
        end

        ux_face = KernelAbstractions.zeros(backend, FT, Nx + 1, Ny)
        uy_face = KernelAbstractions.zeros(backend, FT, Nx, Ny + 1)
        Kraken.logfv_cell_velocity_to_faces_solid_aware_2d!(ux_face, uy_face, ux, uy, is_solid)
        psixx_const_h = fill(FT(0.3), Nx, Ny)
        psixy_const_h = fill(FT(-0.04), Nx, Ny)
        psiyy_const_h = fill(FT(0.2), Nx, Ny)
        psixx_const = _copy_to_backend(backend, psixx_const_h)
        psixy_const = _copy_to_backend(backend, psixy_const_h)
        psiyy_const = _copy_to_backend(backend, psiyy_const_h)
        advxx = KernelAbstractions.zeros(backend, FT, Nx, Ny)
        advxy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
        advyy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
        Kraken.logfv_advect_upwind_solid_aware_2d!(
            advxx, advxy, advyy,
            psixx_const, psixy_const, psiyy_const,
            ux_face, uy_face, is_solid, FT(0.2),
        )
        advxx_h = Array(advxx)
        advxy_h = Array(advxy)
        advyy_h = Array(advyy)
        for j in 1:Ny, i in 1:Nx
            if is_solid_h[i, j]
                @test advxx_h[i, j] == 0
                @test advxy_h[i, j] == 0
                @test advyy_h[i, j] == 0
            else
                @test Float64(advxx_h[i, j]) ≈ 0.3 atol=atol rtol=atol
                @test Float64(advxy_h[i, j]) ≈ -0.04 atol=atol rtol=atol
                @test Float64(advyy_h[i, j]) ≈ 0.2 atol=atol rtol=atol
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

        square = Kraken.run_viscoelastic_logfv_square_periodic_2d(;
            Nx=20, Ny=12, side=4, nu_s=0.08, nu_p=0.02, Fx_body=1e-6,
            lambda=5.0, bsd_fraction=1.0, polymer_substeps=:auto, max_steps=30,
            backend=backend, T=FT,
        )
        @test square.bsd_fraction == 1.0
        @test square.min_c_eig > 0
        @test square.max_speed > 0
        @test all(isfinite, square.ux)
        @test all(isfinite, square.psixx)

        low_beta_square = Kraken.run_viscoelastic_logfv_square_periodic_2d(;
            Nx=20, Ny=12, side=4, nu_s=0.002, nu_p=0.098, Fx_body=5e-6,
            lambda=50.0, bsd_fraction=1.0, polymer_substeps=:auto, max_steps=5,
            backend=backend, T=FT,
        )
        @test low_beta_square.nu_lbm ≈ low_beta_square.nu_total
        @test low_beta_square.min_c_eig > 0.9
        @test low_beta_square.max_speed > 0
        @test low_beta_square.rho_min > 0.99
        @test low_beta_square.rho_max < 1.01
        @test all(isfinite, low_beta_square.ux)
        @test all(isfinite, low_beta_square.psixx)

        @info "Log-FV GPU smoke passed" backend=backend_name FT=FT substeps=coupled.polymer_substeps square_substeps=square.polymer_substeps low_beta_square_substeps=low_beta_square.polymer_substeps
    end
end
