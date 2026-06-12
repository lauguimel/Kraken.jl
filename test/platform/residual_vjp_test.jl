using Test
using Kraken
using Random

@testset "platform residual Phase 2a (Enzyme-free)" begin

    @testset "LBMGeomParams construction" begin
        fwd = Kraken.ad_forward_solve(; Nx=16, Ny=8, cx=4.0, cy=4.0,
                                        radius=2.0, u_in=0.05, nu=0.05,
                                        inlet=:parabolic, tol=1e-10,
                                        max_steps=50_000)
        p = LBMGeomParams(fwd.q_wall, fwd.is_solid, fwd.u_profile,
                          fwd.rho_out, fwd.s_plus, fwd.s_minus, fwd.Nx, fwd.Ny)
        @test p.Nx == 16
        @test p.Ny == 8
        @test size(p.q_wall) == (16, 8, 9)
        @test size(p.is_solid) == (16, 8)
    end

    @testset "residual returns same size as input" begin
        fwd = Kraken.ad_forward_solve(; Nx=16, Ny=8, cx=4.0, cy=4.0,
                                        radius=2.0, u_in=0.05, nu=0.05,
                                        inlet=:parabolic, tol=1e-10,
                                        max_steps=50_000)
        p = LBMGeomParams(fwd.q_wall, fwd.is_solid, fwd.u_profile,
                          fwd.rho_out, fwd.s_plus, fwd.s_minus, fwd.Nx, fwd.Ny)
        R = residual(nothing, LBM(), fwd.f_star, p)
        @test size(R) == size(fwd.f_star)
        @test eltype(R) == Float64
    end

    @testset "residual at fixed point < 1e-10" begin
        fwd = Kraken.ad_forward_solve(; Nx=16, Ny=8, cx=4.0, cy=4.0,
                                        radius=2.0, u_in=0.05, nu=0.05,
                                        inlet=:parabolic, tol=1e-10,
                                        max_steps=50_000)
        @test fwd.converged
        p = LBMGeomParams(fwd.q_wall, fwd.is_solid, fwd.u_profile,
                          fwd.rho_out, fwd.s_plus, fwd.s_minus, fwd.Nx, fwd.Ny)
        R = residual(nothing, LBM(), fwd.f_star, p)
        rel = sqrt(sum(abs2, R)) / sqrt(sum(abs2, fwd.f_star))
        @test rel < 1e-10
    end

    @testset "LBMThermalParams construction" begin
        fwd = Kraken.ad_thermal_forward_solve(; N=8, Ra=1e3, Pr=0.71,
                                                tol=1e-10, max_steps=100_000)
        @test fwd.converged
        p = LBMThermalParams(fwd.q_wall, fwd.params, fwd.Nx, fwd.Ny)
        @test p.Nx == 8
        @test p.Ny == 8
        @test size(p.q_wall) == size(fwd.q_wall)
    end

    @testset "thermal residual at fixed point" begin
        fwd = Kraken.ad_thermal_forward_solve(; N=8, Ra=1e3, Pr=0.71,
                                                tol=1e-10, max_steps=100_000)
        @test fwd.converged
        p = LBMThermalParams(fwd.q_wall, fwd.params, fwd.Nx, fwd.Ny)
        R = residual(nothing, LBM(), fwd.w_star, p)
        @test length(R) == length(fwd.w_star)
        rel = sqrt(sum(abs2, R)) / sqrt(sum(abs2, fwd.w_star))
        @test rel < 1e-9
    end

    @testset "LBMVEParams construction" begin
        Nx, Ny = 12, 12
        s_plus, s_minus = Kraken.ad_ve_trt_rates(0.08)
        p_ve = Kraken.ADVECoupledParams(Nx, Ny, 0.5, 0.05, 4, 0.04, 0.08,
                                        2e-4, s_plus, s_minus)
        geom = Kraken.ad_ve_build_geom(Nx, Ny, 6.35, 5.65, 2.5;
                                       samples=8, u_mean=2e-4)
        p = LBMVEParams(geom.g, geom.q_wall, geom.u_profile, p_ve)
        @test p.p.Nx == Nx
        @test p.p.Ny == Ny
    end

    @testset "SteadyResidual in LBM capabilities" begin
        @test SteadyResidual in capabilities(LBM())
    end

    @testset "capabilities introspection (updated)" begin
        caps = capabilities(LBM())
        @test ForwardSolve in caps
        @test GPUExecution in caps
        @test SteadyAdjoint in caps
        @test SteadyResidual in caps
    end
end

@testset "platform residual Phase 2b-1 (Enzyme-free)" begin

    @testset "LBMScalarParams construction from LBMGeomParams + ν" begin
        fwd = Kraken.ad_forward_solve(; Nx=16, Ny=8, cx=4.0, cy=4.0,
                                        radius=2.0, u_in=0.05, nu=0.05,
                                        inlet=:parabolic, tol=1e-10,
                                        max_steps=50_000)
        geom = LBMGeomParams(fwd.q_wall, fwd.is_solid, fwd.u_profile,
                              fwd.rho_out, fwd.s_plus, fwd.s_minus, fwd.Nx, fwd.Ny)
        ν = 0.05
        p = LBMScalarParams(geom, ν)
        @test p.ν == ν
        @test p.Nx == 16
        @test p.Ny == 8
        s_p, s_m = Kraken.ad_trt_rates_inline(ν)
        @test p.s_plus ≈ s_p
        @test p.s_minus ≈ s_m
    end

    @testset "LBMScalarParams residual parity vs LBMGeomParams (bit-exact)" begin
        fwd = Kraken.ad_forward_solve(; Nx=16, Ny=8, cx=4.0, cy=4.0,
                                        radius=2.0, u_in=0.05, nu=0.05,
                                        inlet=:parabolic, tol=1e-10,
                                        max_steps=50_000)
        geom = LBMGeomParams(fwd.q_wall, fwd.is_solid, fwd.u_profile,
                              fwd.rho_out, fwd.s_plus, fwd.s_minus, fwd.Nx, fwd.Ny)
        p_scalar = LBMScalarParams(geom, 0.05)
        R_geom = residual(nothing, LBM(), fwd.f_star, geom)
        R_scalar = residual(nothing, LBM(), fwd.f_star, p_scalar)
        @test maximum(abs.(R_scalar .- R_geom)) == 0.0
    end

    @testset "LBMScalarParams residual at fixed point < 1e-10" begin
        fwd = Kraken.ad_forward_solve(; Nx=16, Ny=8, cx=4.0, cy=4.0,
                                        radius=2.0, u_in=0.05, nu=0.05,
                                        inlet=:parabolic, tol=1e-10,
                                        max_steps=50_000)
        @test fwd.converged
        geom = LBMGeomParams(fwd.q_wall, fwd.is_solid, fwd.u_profile,
                              fwd.rho_out, fwd.s_plus, fwd.s_minus, fwd.Nx, fwd.Ny)
        p = LBMScalarParams(geom, 0.05)
        R = residual(nothing, LBM(), fwd.f_star, p)
        rel = sqrt(sum(abs2, R)) / sqrt(sum(abs2, fwd.f_star))
        @test rel < 1e-10
    end

    @testset "ENZYME_FREE_OK: LBMScalarParams exported, extension check" begin
        @test isdefined(Kraken, :LBMScalarParams)
        if Base.get_extension(Kraken, :KrakenADExt) === nothing
            println("ENZYME_FREE_OK: extension=nothing, LBMScalarParams defined")
        end
    end
end

if Base.get_extension(Kraken, :KrakenADExt) !== nothing
    using LinearAlgebra

    @testset "adjoint_vjp bit-exact vs internal (Phase 2a, Enzyme)" begin

        @testset "Newtonian VJP parity" begin
            fwd = Kraken.ad_forward_solve(; Nx=48, Ny=16, cx=12.0, cy=8.0,
                                            radius=3.75, u_in=0.05, nu=0.05,
                                            inlet=:parabolic, tol=1e-12,
                                            max_steps=120_000)
            @test fwd.converged
            p = LBMGeomParams(fwd.q_wall, fwd.is_solid, fwd.u_profile,
                              fwd.rho_out, fwd.s_plus, fwd.s_minus, fwd.Nx, fwd.Ny)
            rng = Random.MersenneTwister(0xdeadbeef)
            v = rand(rng, size(fwd.f_star)...)

            internal = v .- Kraken._ad_vjp_GtT(fwd.f_star, v, fwd.q_wall,
                              fwd.is_solid, fwd.u_profile, fwd.rho_out,
                              fwd.s_plus, fwd.s_minus, fwd.Nx, fwd.Ny)
            exposed = adjoint_vjp(nothing, LBM(), fwd.f_star, p, v)

            delta = norm(exposed .- internal) / norm(internal)
            @test delta == 0.0
        end

        @testset "Thermal VJP parity" begin
            fwd = Kraken.ad_thermal_forward_solve(; N=16, Ra=1e3, Pr=0.71,
                                                    tol=1e-11, max_steps=450_000)
            @test fwd.converged
            p = LBMThermalParams(fwd.q_wall, fwd.params, fwd.Nx, fwd.Ny)
            rng = Random.MersenneTwister(0xcafe1234)
            v = rand(rng, length(fwd.w_star))

            internal = v .- Kraken._ad_thermal_vjp_GtT(fwd.w_star, v,
                              fwd.q_wall, fwd.q_wall, fwd.params)
            exposed = adjoint_vjp(nothing, LBM(), fwd.w_star, p, v)

            delta = norm(exposed .- internal) / norm(internal)
            @test delta == 0.0
        end

        @testset "VE VJP parity" begin
            Nx, Ny = 24, 24
            s_plus, s_minus = Kraken.ad_ve_trt_rates(0.08)
            p_ve = Kraken.ADVECoupledParams(Nx, Ny, 0.5, 0.05, 4,
                                            0.02/0.5, 0.08, 2e-4,
                                            s_plus, s_minus)
            geom = Kraken.ad_ve_build_geom(Nx, Ny, 12.35, 11.65, 5.13;
                                           samples=16, u_mean=2e-4)
            w0 = Kraken.ad_ve_initial_state(geom.g, Nx, Ny, 0.05)
            fwd = Kraken.ad_ve_forward_solve(w0, geom, p_ve; fwd_tol=1e-13)
            @test fwd.converged

            p = LBMVEParams(geom.g, geom.q_wall, geom.u_profile, p_ve)
            rng = Random.MersenneTwister(0xabcd5678)
            v = rand(rng, length(fwd.w_star))

            internal = v .- Kraken._ad_ve_vjp_GtT(fwd.w_star, v, geom.g,
                              geom.q_wall, geom.u_profile, p_ve)
            exposed = adjoint_vjp(nothing, LBM(), fwd.w_star, p, v)

            delta = norm(exposed .- internal) / norm(internal)
            @test delta == 0.0
        end
    end
end

if Base.get_extension(Kraken, :KrakenADExt) !== nothing
    using LinearAlgebra

    @testset "Phase 2b-1 Enzyme-gated" begin

        @testset "LBMScalarParams adjoint_vjp parity vs LBMGeomParams" begin
            fwd = Kraken.ad_forward_solve(; Nx=32, Ny=16, cx=100.0, cy=8.0,
                                            radius=0.001, u_in=0.05, nu=0.05,
                                            inlet=:parabolic, tol=1e-12,
                                            max_steps=200_000)
            @test fwd.converged
            geom = LBMGeomParams(fwd.q_wall, fwd.is_solid, fwd.u_profile,
                                  fwd.rho_out, fwd.s_plus, fwd.s_minus, fwd.Nx, fwd.Ny)
            p_scalar = LBMScalarParams(geom, 0.05)
            rng = Random.MersenneTwister(0xdeadbeef)
            v = rand(rng, size(fwd.f_star)...)
            vjp_geom = adjoint_vjp(nothing, LBM(), fwd.f_star, geom, v)
            vjp_scalar = adjoint_vjp(nothing, LBM(), fwd.f_star, p_scalar, v)
            @test norm(vjp_scalar .- vjp_geom) == 0.0
        end

        @testset "ν gradient check: adjoint dL/dν vs central-FD (rel < 1e-2)" begin
            function ux_profile_from_f(f, x_obs::Int, Ny::Int)
                ux = zeros(Float64, Ny)
                @inbounds for j in 1:Ny
                    rho = sum(@view f[x_obs, j, :])
                    ux[j] = (f[x_obs, j, 2] - f[x_obs, j, 4] +
                             f[x_obs, j, 6] - f[x_obs, j, 7] -
                             f[x_obs, j, 8] + f[x_obs, j, 9]) / rho
                end
                return ux
            end

            ν_true = 0.05
            ν0 = 0.06
            Nx, Ny = 32, 16

            fwd_true = Kraken.ad_forward_solve(; Nx=Nx, Ny=Ny, cx=100.0, cy=8.0,
                                                 radius=0.001, u_in=0.05, nu=ν_true,
                                                 inlet=:parabolic, tol=1e-12,
                                                 max_steps=200_000)
            @test fwd_true.converged

            x_obs = Nx ÷ 2
            data_ux = ux_profile_from_f(fwd_true.f_star, x_obs, Ny)

            fwd0 = Kraken.ad_forward_solve(; Nx=Nx, Ny=Ny, cx=100.0, cy=8.0,
                                             radius=0.001, u_in=0.05, nu=ν0,
                                             inlet=:parabolic, tol=1e-12,
                                             max_steps=200_000)
            @test fwd0.converged

            pred_ux = ux_profile_from_f(fwd0.f_star, x_obs, Ny)
            L0 = 0.5 * sum(abs2, pred_ux .- data_ux)
            @test isfinite(L0)

            geom0 = LBMGeomParams(fwd0.q_wall, fwd0.is_solid, fwd0.u_profile,
                                   fwd0.rho_out, fwd0.s_plus, fwd0.s_minus, Nx, Ny)
            p0 = LBMScalarParams(geom0, ν0)

            cx_q = [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0]
            dLdf = zeros(size(fwd0.f_star))
            for j in 1:Ny
                rho_xj = sum(@view fwd0.f_star[x_obs, j, :])
                ux_xj = pred_ux[j]
                res_j = pred_ux[j] - data_ux[j]
                for q in 1:9
                    dLdf[x_obs, j, q] = res_j * (cx_q[q] - ux_xj) / rho_xj
                end
            end

            apply_GtT = v -> Kraken._ad_vjp_GtT(fwd0.f_star, v, fwd0.q_wall,
                                                 fwd0.is_solid, fwd0.u_profile,
                                                 fwd0.rho_out, fwd0.s_plus,
                                                 fwd0.s_minus, Nx, Ny)
            adj = Kraken.gmres_adjoint(apply_GtT, dLdf;
                                       tol=1e-11, restart=240, max_restarts=20)
            @test adj.converged

            dLdν_adj = Kraken._ad_pvjp_nu(fwd0.f_star, adj.lambda, p0)

            h = 1e-4 * ν0
            fwd_p = Kraken.ad_forward_solve(; Nx=Nx, Ny=Ny, cx=100.0, cy=8.0,
                                              radius=0.001, u_in=0.05, nu=ν0+h,
                                              inlet=:parabolic, tol=1e-12,
                                              max_steps=200_000)
            fwd_m = Kraken.ad_forward_solve(; Nx=Nx, Ny=Ny, cx=100.0, cy=8.0,
                                              radius=0.001, u_in=0.05, nu=ν0-h,
                                              inlet=:parabolic, tol=1e-12,
                                              max_steps=200_000)
            @test fwd_p.converged
            @test fwd_m.converged

            pred_p = ux_profile_from_f(fwd_p.f_star, x_obs, Ny)
            pred_m = ux_profile_from_f(fwd_m.f_star, x_obs, Ny)
            Lp = 0.5 * sum(abs2, pred_p .- data_ux)
            Lm = 0.5 * sum(abs2, pred_m .- data_ux)
            dLdν_fd = (Lp - Lm) / (2h)

            rel = abs(dLdν_adj - dLdν_fd) / max(abs(dLdν_fd), 1e-15)
            @info "ν gradient check" dLdν_adj dLdν_fd rel h
            @test rel < 1e-2
        end
    end
end
