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

function _gpu_fluid_x_neighbor(is_solid, i, j)
    Nx, _ = size(is_solid)
    return (i > 1 && !is_solid[i - 1, j]) || (i < Nx && !is_solid[i + 1, j])
end

function _gpu_fluid_y_neighbor(is_solid, i, j)
    _, Ny = size(is_solid)
    return (j > 1 && !is_solid[i, j - 1]) || (j < Ny && !is_solid[i, j + 1])
end

function _gpu_fluid_x_second(is_solid, i, j)
    Nx, _ = size(is_solid)
    return (i > 1 && !is_solid[i - 1, j] && i < Nx && !is_solid[i + 1, j]) ||
           (i + 2 <= Nx && !is_solid[i + 1, j] && !is_solid[i + 2, j]) ||
           (i - 2 >= 1 && !is_solid[i - 1, j] && !is_solid[i - 2, j])
end

function _gpu_fluid_y_second(is_solid, i, j)
    _, Ny = size(is_solid)
    return (j > 1 && !is_solid[i, j - 1] && j < Ny && !is_solid[i, j + 1]) ||
           (j + 2 <= Ny && !is_solid[i, j + 1] && !is_solid[i, j + 2]) ||
           (j - 2 >= 1 && !is_solid[i, j - 1] && !is_solid[i, j - 2])
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

        bfs = Kraken.backward_facing_step_geometry_2d(;
            H_in=3, expansion_ratio=2, L_up=2, L_down=2, FT=FT,
        )
        bNx, bNy = bfs.Nx, bfs.Ny
        bdx, bdy = FT(0.4), FT(0.25)
        bax, bay = FT(0.03), FT(-0.02)
        bbx, bby = FT(-0.04), FT(0.05)
        bfs_solid_h = bfs.is_solid
        bux_h = [FT(0.12) + bax * (FT(i) - FT(0.5)) * bdx + bay * (FT(j) - FT(0.5)) * bdy
                 for i in 1:bNx, j in 1:bNy]
        buy_h = [FT(-0.08) + bbx * (FT(i) - FT(0.5)) * bdx + bby * (FT(j) - FT(0.5)) * bdy
                 for i in 1:bNx, j in 1:bNy]
        bfs_solid = _copy_to_backend(backend, bfs_solid_h)
        bux = _copy_to_backend(backend, bux_h)
        buy = _copy_to_backend(backend, buy_h)
        bdudx = KernelAbstractions.zeros(backend, FT, bNx, bNy)
        bdudy = KernelAbstractions.zeros(backend, FT, bNx, bNy)
        bdvdx = KernelAbstractions.zeros(backend, FT, bNx, bNy)
        bdvdy = KernelAbstractions.zeros(backend, FT, bNx, bNy)
        Kraken.logfv_velocity_gradient_solid_aware_2d!(
            bdudx, bdudy, bdvdx, bdvdy, bux, buy, bfs_solid, bdx, bdy,
        )
        bdudx_h = Array(bdudx)
        bdudy_h = Array(bdudy)
        bdvdx_h = Array(bdvdx)
        bdvdy_h = Array(bdvdy)
        for j in 1:bNy, i in 1:bNx
            if bfs_solid_h[i, j]
                @test bdudx_h[i, j] == 0
                @test bdudy_h[i, j] == 0
                @test bdvdx_h[i, j] == 0
                @test bdvdy_h[i, j] == 0
            else
                @test Float64(bdudx_h[i, j]) ≈ Float64(_gpu_fluid_x_neighbor(bfs_solid_h, i, j) ? bax : FT(0)) atol=atol rtol=atol
                @test Float64(bdvdx_h[i, j]) ≈ Float64(_gpu_fluid_x_neighbor(bfs_solid_h, i, j) ? bbx : FT(0)) atol=atol rtol=atol
                @test Float64(bdudy_h[i, j]) ≈ Float64(_gpu_fluid_y_neighbor(bfs_solid_h, i, j) ? bay : FT(0)) atol=atol rtol=atol
                @test Float64(bdvdy_h[i, j]) ≈ Float64(_gpu_fluid_y_neighbor(bfs_solid_h, i, j) ? bby : FT(0)) atol=atol rtol=atol
            end
        end

        bux_face = KernelAbstractions.zeros(backend, FT, bNx + 1, bNy)
        buy_face = KernelAbstractions.zeros(backend, FT, bNx, bNy + 1)
        Kraken.logfv_cell_velocity_to_faces_solid_aware_2d!(bux_face, buy_face, bux, buy, bfs_solid)
        bpsixx_const = _copy_to_backend(backend, fill(FT(0.25), bNx, bNy))
        bpsixy_const = _copy_to_backend(backend, fill(FT(-0.03), bNx, bNy))
        bpsiyy_const = _copy_to_backend(backend, fill(FT(0.11), bNx, bNy))
        badvxx = KernelAbstractions.zeros(backend, FT, bNx, bNy)
        badvxy = KernelAbstractions.zeros(backend, FT, bNx, bNy)
        badvyy = KernelAbstractions.zeros(backend, FT, bNx, bNy)
        Kraken.logfv_advect_upwind_solid_aware_2d!(
            badvxx, badvxy, badvyy,
            bpsixx_const, bpsixy_const, bpsiyy_const,
            bux_face, buy_face, bfs_solid, FT(0.2),
        )
        badvxx_h = Array(badvxx)
        badvxy_h = Array(badvxy)
        badvyy_h = Array(badvyy)
        for j in 1:bNy, i in 1:bNx
            if bfs_solid_h[i, j]
                @test badvxx_h[i, j] == 0
                @test badvxy_h[i, j] == 0
                @test badvyy_h[i, j] == 0
            else
                @test Float64(badvxx_h[i, j]) ≈ 0.25 atol=atol rtol=atol
                @test Float64(badvxy_h[i, j]) ≈ -0.03 atol=atol rtol=atol
                @test Float64(badvyy_h[i, j]) ≈ 0.11 atol=atol rtol=atol
            end
        end

        btx_xx, bty_xx = FT(0.07), FT(-0.01)
        btx_xy, bty_xy = FT(-0.03), FT(0.02)
        btx_yy, bty_yy = FT(0.04), FT(0.05)
        btauxx_h = [FT(0.1) + btx_xx * (FT(i) - FT(0.5)) * bdx + bty_xx * (FT(j) - FT(0.5)) * bdy
                    for i in 1:bNx, j in 1:bNy]
        btauxy_h = [FT(-0.2) + btx_xy * (FT(i) - FT(0.5)) * bdx + bty_xy * (FT(j) - FT(0.5)) * bdy
                    for i in 1:bNx, j in 1:bNy]
        btauyy_h = [FT(0.3) + btx_yy * (FT(i) - FT(0.5)) * bdx + bty_yy * (FT(j) - FT(0.5)) * bdy
                    for i in 1:bNx, j in 1:bNy]
        btauxx = _copy_to_backend(backend, btauxx_h)
        btauxy = _copy_to_backend(backend, btauxy_h)
        btauyy = _copy_to_backend(backend, btauyy_h)
        bfx_stress = KernelAbstractions.zeros(backend, FT, bNx, bNy)
        bfy_stress = KernelAbstractions.zeros(backend, FT, bNx, bNy)
        Kraken.logfv_polymer_force_solid_aware_2d!(
            bfx_stress, bfy_stress, btauxx, btauxy, btauyy, bfs_solid, bdx, bdy,
        )
        bfx_stress_h = Array(bfx_stress)
        bfy_stress_h = Array(bfy_stress)
        for j in 1:bNy, i in 1:bNx
            if bfs_solid_h[i, j]
                @test bfx_stress_h[i, j] == 0
                @test bfy_stress_h[i, j] == 0
            else
                expected_fx = (_gpu_fluid_x_neighbor(bfs_solid_h, i, j) ? btx_xx : FT(0)) +
                              (_gpu_fluid_y_neighbor(bfs_solid_h, i, j) ? bty_xy : FT(0))
                expected_fy = (_gpu_fluid_x_neighbor(bfs_solid_h, i, j) ? btx_xy : FT(0)) +
                              (_gpu_fluid_y_neighbor(bfs_solid_h, i, j) ? bty_yy : FT(0))
                @test Float64(bfx_stress_h[i, j]) ≈ Float64(expected_fx) atol=atol rtol=atol
                @test Float64(bfy_stress_h[i, j]) ≈ Float64(expected_fy) atol=atol rtol=atol
            end
        end

        bqxx, bqxy = FT(0.04), FT(-0.01)
        bqyx, bqyy = FT(-0.06), FT(0.03)
        bux_quad_h = [FT(0.1) + bqxx * ((FT(i) - FT(0.5)) * bdx)^2 +
                      bqxy * ((FT(j) - FT(0.5)) * bdy)^2
                      for i in 1:bNx, j in 1:bNy]
        buy_quad_h = [FT(-0.2) + bqyx * ((FT(i) - FT(0.5)) * bdx)^2 +
                      bqyy * ((FT(j) - FT(0.5)) * bdy)^2
                      for i in 1:bNx, j in 1:bNy]
        bfx_poly_h = [FT(0.02) + FT(0.01) * (FT(i) - FT(0.5)) * bdx
                      for i in 1:bNx, j in 1:bNy]
        bfy_poly_h = [FT(-0.03) + FT(0.02) * (FT(j) - FT(0.5)) * bdy
                      for i in 1:bNx, j in 1:bNy]
        bux_quad = _copy_to_backend(backend, bux_quad_h)
        buy_quad = _copy_to_backend(backend, buy_quad_h)
        bfx_poly = _copy_to_backend(backend, bfx_poly_h)
        bfy_poly = _copy_to_backend(backend, bfy_poly_h)
        bfx_total = KernelAbstractions.zeros(backend, FT, bNx, bNy)
        bfy_total = KernelAbstractions.zeros(backend, FT, bNx, bNy)
        Kraken.logfv_bsd_correct_force_solid_aware_2d!(
            bfx_total, bfy_total, bfx_poly, bfy_poly,
            bux_quad, buy_quad, bfs_solid, FT(0.75), FT(0.09), bdx, bdy,
        )
        bfx_total_h = Array(bfx_total)
        bfy_total_h = Array(bfy_total)
        for j in 1:bNy, i in 1:bNx
            if bfs_solid_h[i, j]
                @test bfx_total_h[i, j] == 0
                @test bfy_total_h[i, j] == 0
            else
                blap_ux = (_gpu_fluid_x_second(bfs_solid_h, i, j) ? FT(2) * bqxx : FT(0)) +
                          (_gpu_fluid_y_second(bfs_solid_h, i, j) ? FT(2) * bqxy : FT(0))
                blap_uy = (_gpu_fluid_x_second(bfs_solid_h, i, j) ? FT(2) * bqyx : FT(0)) +
                          (_gpu_fluid_y_second(bfs_solid_h, i, j) ? FT(2) * bqyy : FT(0))
                @test Float64(bfx_total_h[i, j]) ≈ Float64(bfx_poly_h[i, j] - FT(0.75) * FT(0.09) * blap_ux) atol=atol rtol=atol
                @test Float64(bfy_total_h[i, j]) ≈ Float64(bfy_poly_h[i, j] - FT(0.75) * FT(0.09) * blap_uy) atol=atol rtol=atol
            end
        end

        bf_in_h = zeros(FT, bNx, bNy, 9)
        for j in 1:bNy, i in 1:bNx, q in 1:9
            ux0 = bfs_solid_h[i, j] ? FT(0) : FT(0.015)
            bf_in_h[i, j, q] = Kraken.equilibrium(D2Q9(), FT(1), ux0, FT(0), q)
        end
        bf_in = _copy_to_backend(backend, bf_in_h)
        bq_wall = _copy_to_backend(backend, bfs.q_wall)
        buw_x = KernelAbstractions.zeros(backend, FT, bNx, bNy, 9)
        buw_y = KernelAbstractions.zeros(backend, FT, bNx, bNy, 9)
        bf_ref = KernelAbstractions.zeros(backend, FT, bNx, bNy, 9)
        bf_force = KernelAbstractions.zeros(backend, FT, bNx, bNy, 9)
        brho_ref = KernelAbstractions.zeros(backend, FT, bNx, bNy)
        bux_ref = KernelAbstractions.zeros(backend, FT, bNx, bNy)
        buy_ref = KernelAbstractions.zeros(backend, FT, bNx, bNy)
        brho_force = KernelAbstractions.zeros(backend, FT, bNx, bNy)
        bux_force = KernelAbstractions.zeros(backend, FT, bNx, bNy)
        buy_force = KernelAbstractions.zeros(backend, FT, bNx, bNy)
        bfx_zero = KernelAbstractions.zeros(backend, FT, bNx, bNy)
        bfy_zero = KernelAbstractions.zeros(backend, FT, bNx, bNy)
        Kraken.fused_trt_libb_v2_step!(
            bf_ref, bf_in, brho_ref, bux_ref, buy_ref, bfs_solid,
            bq_wall, buw_x, buw_y, bNx, bNy, FT(0.08),
        )
        Kraken.fused_trt_libb_v2_guo_field_step!(
            bf_force, bf_in, brho_force, bux_force, buy_force, bfs_solid,
            bq_wall, buw_x, buw_y, bfx_zero, bfy_zero, bNx, bNy, FT(0.08),
        )
        @test Array(bf_force) == Array(bf_ref)
        @test Array(brho_force) == Array(brho_ref)
        @test Array(bux_force) == Array(bux_ref)
        @test Array(buy_force) == Array(buy_ref)

        bfx_drive_h = [bfs_solid_h[i, j] ? FT(0) : FT(1e-5) for i in 1:bNx, j in 1:bNy]
        bfy_drive_h = [bfs_solid_h[i, j] ? FT(0) : FT(-3e-6) for i in 1:bNx, j in 1:bNy]
        bfx_drive = _copy_to_backend(backend, bfx_drive_h)
        bfy_drive = _copy_to_backend(backend, bfy_drive_h)
        bf_driven = KernelAbstractions.zeros(backend, FT, bNx, bNy, 9)
        brho_driven = KernelAbstractions.zeros(backend, FT, bNx, bNy)
        bux_driven = KernelAbstractions.zeros(backend, FT, bNx, bNy)
        buy_driven = KernelAbstractions.zeros(backend, FT, bNx, bNy)
        Kraken.fused_trt_libb_v2_guo_field_step!(
            bf_driven, bf_in, brho_driven, bux_driven, buy_driven, bfs_solid,
            bq_wall, buw_x, buw_y, bfx_drive, bfy_drive, bNx, bNy, FT(0.08),
        )
        @test all(isfinite, Array(bf_driven))
        @test all(isfinite, Array(brho_driven))
        @test all(isfinite, Array(bux_driven))
        @test maximum(abs, Array(bf_driven) .- Array(bf_ref)) > 0

        bgeom = Kraken.transfer_step_geometry_2d(bfs, backend)
        bprofile_h = Kraken.parabolic_face_profile_2d(bfs; face=:west, mean_velocity=FT(0.01), FT=FT)
        bprofile = _copy_to_backend(backend, bprofile_h)
        bbcspec = Kraken.default_step_bcspec_2d(bgeom, bprofile, FT(1))
        bhydro_in_h = zeros(FT, bNx, bNy, 9)
        for j in 1:bNy, i in 1:bNx, q in 1:9
            ux0 = bfs_solid_h[i, j] ? FT(0) : bprofile_h[j]
            bhydro_in_h[i, j, q] = Kraken.equilibrium(D2Q9(), FT(1), ux0, FT(0), q)
        end
        bhydro_in = _copy_to_backend(backend, bhydro_in_h)
        bhydro_out = KernelAbstractions.zeros(backend, FT, bNx, bNy, 9)
        brho_hydro = KernelAbstractions.zeros(backend, FT, bNx, bNy)
        bux_hydro = KernelAbstractions.zeros(backend, FT, bNx, bNy)
        buy_hydro = KernelAbstractions.zeros(backend, FT, bNx, bNy)
        bfx_hydro_h = [bfs_solid_h[i, j] ? FT(0) : FT(2e-7) for i in 1:bNx, j in 1:bNy]
        bfy_hydro_h = fill(FT(0), bNx, bNy)
        bfx_hydro = _copy_to_backend(backend, bfx_hydro_h)
        bfy_hydro = _copy_to_backend(backend, bfy_hydro_h)
        for _ in 1:5
            Kraken.fused_trt_libb_v2_guo_field_step!(
                bhydro_out, bhydro_in, brho_hydro, bux_hydro, buy_hydro,
                bgeom.is_solid, bgeom.q_wall, buw_x, buw_y,
                bfx_hydro, bfy_hydro, bNx, bNy, FT(0.08),
            )
            Kraken.apply_bc_rebuild_2d!(bhydro_out, bhydro_in, bbcspec, FT(0.08), bNx, bNy)
            bhydro_in, bhydro_out = bhydro_out, bhydro_in
        end
        Kraken.logfv_compute_macroscopic_forced_field_2d!(
            brho_hydro, bux_hydro, buy_hydro, bhydro_in, bfx_hydro, bfy_hydro,
        )
        brho_h = Array(brho_hydro)
        bux_h = Array(bux_hydro)
        buy_h = Array(buy_hydro)
        bfs_fluid = .!bfs_solid_h
        @test all(isfinite, brho_h[bfs_fluid])
        @test all(isfinite, bux_h[bfs_fluid])
        @test all(isfinite, buy_h[bfs_fluid])
        @test minimum(brho_h[bfs_fluid]) > 0.95
        @test maximum(brho_h[bfs_fluid]) < 1.05
        @test maximum(abs, bux_h[bfs_fluid]) > 1e-5

        ox_Nx, ox_Ny = 6, 5
        ox_dt = FT(0.25)
        ox_u = FT(0.2)
        ox_solid_h = fill(false, ox_Nx, ox_Ny)
        ox_solid = _copy_to_backend(backend, ox_solid_h)
        ox_ux = _copy_to_backend(backend, fill(ox_u, ox_Nx, ox_Ny))
        ox_uy = _copy_to_backend(backend, fill(FT(0), ox_Nx, ox_Ny))
        ox_west_u = _copy_to_backend(backend, fill(ox_u, ox_Ny))
        ox_east_u = _copy_to_backend(backend, fill(ox_u, ox_Ny))
        ox_ux_face = KernelAbstractions.zeros(backend, FT, ox_Nx + 1, ox_Ny)
        ox_uy_face = KernelAbstractions.zeros(backend, FT, ox_Nx, ox_Ny + 1)
        Kraken.logfv_cell_velocity_to_faces_openx_solid_aware_2d!(
            ox_ux_face, ox_uy_face, ox_ux, ox_uy, ox_solid, ox_west_u, ox_east_u,
        )
        ox_axx, ox_axy, ox_ayy = FT(0.03), FT(-0.02), FT(0.01)
        ox_psixx_h = [FT(0.2) + ox_axx * FT(i) for i in 1:ox_Nx, j in 1:ox_Ny]
        ox_psixy_h = [FT(-0.1) + ox_axy * FT(i) for i in 1:ox_Nx, j in 1:ox_Ny]
        ox_psiyy_h = [FT(0.05) + ox_ayy * FT(i) for i in 1:ox_Nx, j in 1:ox_Ny]
        ox_psixx = _copy_to_backend(backend, ox_psixx_h)
        ox_psixy = _copy_to_backend(backend, ox_psixy_h)
        ox_psiyy = _copy_to_backend(backend, ox_psiyy_h)
        ox_outxx = KernelAbstractions.zeros(backend, FT, ox_Nx, ox_Ny)
        ox_outxy = KernelAbstractions.zeros(backend, FT, ox_Nx, ox_Ny)
        ox_outyy = KernelAbstractions.zeros(backend, FT, ox_Nx, ox_Ny)
        Kraken.logfv_advect_upwind_openx_solid_aware_2d!(
            ox_outxx, ox_outxy, ox_outyy,
            ox_psixx, ox_psixy, ox_psiyy,
            _copy_to_backend(backend, fill(FT(0.2), ox_Ny)),
            _copy_to_backend(backend, fill(FT(-0.1), ox_Ny)),
            _copy_to_backend(backend, fill(FT(0.05), ox_Ny)),
            _copy_to_backend(backend, fill(FT(0.2) + ox_axx * FT(ox_Nx + 1), ox_Ny)),
            _copy_to_backend(backend, fill(FT(-0.1) + ox_axy * FT(ox_Nx + 1), ox_Ny)),
            _copy_to_backend(backend, fill(FT(0.05) + ox_ayy * FT(ox_Nx + 1), ox_Ny)),
            ox_ux_face, ox_uy_face, ox_solid, ox_dt,
        )
        ox_outxx_h = Array(ox_outxx)
        ox_outxy_h = Array(ox_outxy)
        ox_outyy_h = Array(ox_outyy)
        for j in 2:(ox_Ny - 1), i in 1:ox_Nx
            @test Float64(ox_outxx_h[i, j]) ≈ Float64(ox_psixx_h[i, j] - ox_dt * ox_u * ox_axx) atol=atol rtol=atol
            @test Float64(ox_outxy_h[i, j]) ≈ Float64(ox_psixy_h[i, j] - ox_dt * ox_u * ox_axy) atol=atol rtol=atol
            @test Float64(ox_outyy_h[i, j]) ≈ Float64(ox_psiyy_h[i, j] - ox_dt * ox_u * ox_ayy) atol=atol rtol=atol
        end

        bfs_ux_open_h = [bfs_solid_h[i, j] ? FT(0) : FT(0.015) for i in 1:bNx, j in 1:bNy]
        bfs_ux_open = _copy_to_backend(backend, bfs_ux_open_h)
        bfs_uy_open = KernelAbstractions.zeros(backend, FT, bNx, bNy)
        bfs_ux_face = KernelAbstractions.zeros(backend, FT, bNx + 1, bNy)
        bfs_uy_face = KernelAbstractions.zeros(backend, FT, bNx, bNy + 1)
        Kraken.logfv_cell_velocity_to_faces_openx_solid_aware_2d!(
            bfs_ux_face, bfs_uy_face, bfs_ux_open, bfs_uy_open, bfs_solid,
            _copy_to_backend(backend, fill(FT(0.015), bNy)),
            _copy_to_backend(backend, fill(FT(0.015), bNy)),
        )
        bfs_cxx, bfs_cxy, bfs_cyy = FT(0.25), FT(-0.03), FT(0.11)
        bfs_psixx = _copy_to_backend(backend, fill(bfs_cxx, bNx, bNy))
        bfs_psixy = _copy_to_backend(backend, fill(bfs_cxy, bNx, bNy))
        bfs_psiyy = _copy_to_backend(backend, fill(bfs_cyy, bNx, bNy))
        bfs_outxx = KernelAbstractions.zeros(backend, FT, bNx, bNy)
        bfs_outxy = KernelAbstractions.zeros(backend, FT, bNx, bNy)
        bfs_outyy = KernelAbstractions.zeros(backend, FT, bNx, bNy)
        Kraken.logfv_advect_upwind_openx_solid_aware_2d!(
            bfs_outxx, bfs_outxy, bfs_outyy,
            bfs_psixx, bfs_psixy, bfs_psiyy,
            _copy_to_backend(backend, fill(bfs_cxx, bNy)),
            _copy_to_backend(backend, fill(bfs_cxy, bNy)),
            _copy_to_backend(backend, fill(bfs_cyy, bNy)),
            _copy_to_backend(backend, fill(bfs_cxx, bNy)),
            _copy_to_backend(backend, fill(bfs_cxy, bNy)),
            _copy_to_backend(backend, fill(bfs_cyy, bNy)),
            bfs_ux_face, bfs_uy_face, bfs_solid, FT(0.2),
        )
        bfs_outxx_h = Array(bfs_outxx)
        bfs_outxy_h = Array(bfs_outxy)
        bfs_outyy_h = Array(bfs_outyy)
        for j in 1:bNy, i in 1:bNx
            if bfs_solid_h[i, j]
                @test bfs_outxx_h[i, j] == 0
                @test bfs_outxy_h[i, j] == 0
                @test bfs_outyy_h[i, j] == 0
            else
                @test Float64(bfs_outxx_h[i, j]) ≈ Float64(bfs_cxx) atol=atol rtol=atol
                @test Float64(bfs_outxy_h[i, j]) ≈ Float64(bfs_cxy) atol=atol rtol=atol
                @test Float64(bfs_outyy_h[i, j]) ≈ Float64(bfs_cyy) atol=atol rtol=atol
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

        couette = Kraken.run_viscoelastic_logfv_channel_2d(;
            Nx=9, Ny=11, flow=:couette, height=FT(1), width=FT(1),
            uwall=FT(0.07), lambda=FT(5), prefactor=FT(0.11),
            bsd_fraction=FT(1), backend=backend, T=FT,
        )
        @test couette.flow === :couette
        @test couette.min_c_eig > 0
        @test couette.max_tau_error < 1e-5
        @test couette.max_poly_force_error < 1e-5
        @test couette.max_total_force_error < 1e-5
        @test all(isfinite, couette.tauxx)
        @test all(isfinite, couette.fx_total)

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

        passive_bfs = Kraken.run_viscoelastic_logfv_bfs_passive_2d(;
            H_in=3, expansion_ratio=2, L_up=2, L_down=2,
            nu_s=0.08, nu_p=0.02, lambda=5.0,
            u_mean=0.01, Fx_body=2e-7,
            hydro_steps=5, polymer_steps=2,
            backend=backend, T=FT,
        )
        @test passive_bfs.min_c_eig > 0.9
        @test passive_bfs.max_abs_psi >= 0
        @test passive_bfs.rho_min > 0.95
        @test passive_bfs.rho_max < 1.05
        @test all(isfinite, passive_bfs.psixx)
        @test all(isfinite, passive_bfs.tauxx)

        coupled_bfs = Kraken.run_viscoelastic_logfv_bfs_coupled_2d(;
            H_in=3, expansion_ratio=2, L_up=2, L_down=2,
            nu_s=0.08, nu_p=0.02, lambda=5.0,
            u_mean=0.01, Fx_body=2e-7,
            bsd_fraction=1.0, max_steps=2,
            backend=backend, T=FT,
        )
        @test coupled_bfs.nu_lbm ≈ coupled_bfs.nu_total
        @test coupled_bfs.min_c_eig > 0.9
        @test coupled_bfs.max_abs_tau >= 0
        @test coupled_bfs.rho_min > 0.95
        @test coupled_bfs.rho_max < 1.05
        @test all(isfinite, coupled_bfs.psixx)
        @test all(isfinite, coupled_bfs.fx_total)

        low_beta_bfs = Kraken.run_viscoelastic_logfv_bfs_coupled_2d(;
            H_in=3, expansion_ratio=2, L_up=2, L_down=2,
            nu_s=0.002, nu_p=0.098, lambda=50.0,
            u_mean=0.003, Fx_body=5e-8,
            bsd_fraction=1.0, max_steps=1,
            backend=backend, T=FT,
        )
        @test low_beta_bfs.nu_lbm ≈ low_beta_bfs.nu_total
        @test low_beta_bfs.min_c_eig > 0.8
        @test low_beta_bfs.rho_min > 0.95
        @test low_beta_bfs.rho_max < 1.05
        @test all(isfinite, low_beta_bfs.psixx)
        @test all(isfinite, low_beta_bfs.fx_total)

        @info "Log-FV GPU smoke passed" backend=backend_name FT=FT substeps=coupled.polymer_substeps square_substeps=square.polymer_substeps low_beta_square_substeps=low_beta_square.polymer_substeps
    end
end
