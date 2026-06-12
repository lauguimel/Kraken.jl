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
