using Test
using LinearAlgebra
using Kraken, KernelAbstractions

@testset "Log-conformation 3D primitives" begin
    @testset "symmetric 3x3 exp/log round trip" begin
        psi = (0.18, 0.04, -0.03, -0.07, 0.02, 0.11)
        C = mat_exp_sym3x3(psi...)
        psi_back = mat_log_spd_sym3x3(C...)
        @test psi_back[1] ≈ psi[1] atol=2e-12 rtol=2e-12
        @test psi_back[2] ≈ psi[2] atol=2e-12 rtol=2e-12
        @test psi_back[3] ≈ psi[3] atol=2e-12 rtol=2e-12
        @test psi_back[4] ≈ psi[4] atol=2e-12 rtol=2e-12
        @test psi_back[5] ≈ psi[5] atol=2e-12 rtol=2e-12
        @test psi_back[6] ≈ psi[6] atol=2e-12 rtol=2e-12

        @test_throws DomainError mat_log_spd_sym3x3(-1.0, 0.0, 0.0, 1.0, 0.0, 1.0)
    end

    @testset "Psi to C kernels preserve identity and round trip" begin
        backend = KernelAbstractions.CPU()
        FT = Float64
        Nx, Ny, Nz = 4, 3, 2
        psi_xx_h = [FT(0.02i - 0.01j + 0.003k) for i in 1:Nx, j in 1:Ny, k in 1:Nz]
        psi_xy_h = [FT(0.004 * (i + j)) for i in 1:Nx, j in 1:Ny, k in 1:Nz]
        psi_xz_h = [FT(-0.003 * (i + k)) for i in 1:Nx, j in 1:Ny, k in 1:Nz]
        psi_yy_h = [FT(-0.015i + 0.006j) for i in 1:Nx, j in 1:Ny, k in 1:Nz]
        psi_yz_h = [FT(0.002 * (j - k)) for i in 1:Nx, j in 1:Ny, k in 1:Nz]
        psi_zz_h = [FT(0.01k - 0.004j) for i in 1:Nx, j in 1:Ny, k in 1:Nz]

        psi_xx = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz); copyto!(psi_xx, psi_xx_h)
        psi_xy = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz); copyto!(psi_xy, psi_xy_h)
        psi_xz = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz); copyto!(psi_xz, psi_xz_h)
        psi_yy = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz); copyto!(psi_yy, psi_yy_h)
        psi_yz = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz); copyto!(psi_yz, psi_yz_h)
        psi_zz = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz); copyto!(psi_zz, psi_zz_h)

        C_xx = similar(psi_xx); C_xy = similar(psi_xx); C_xz = similar(psi_xx)
        C_yy = similar(psi_xx); C_yz = similar(psi_xx); C_zz = similar(psi_xx)
        out_xx = similar(psi_xx); out_xy = similar(psi_xx); out_xz = similar(psi_xx)
        out_yy = similar(psi_xx); out_yz = similar(psi_xx); out_zz = similar(psi_xx)

        psi_to_C_3d!(C_xx, C_xy, C_xz, C_yy, C_yz, C_zz,
                     psi_xx, psi_xy, psi_xz, psi_yy, psi_yz, psi_zz)
        C_to_psi_3d!(out_xx, out_xy, out_xz, out_yy, out_yz, out_zz,
                     C_xx, C_xy, C_xz, C_yy, C_yz, C_zz)

        @test maximum(abs.(Array(out_xx) .- psi_xx_h)) < 3e-12
        @test maximum(abs.(Array(out_xy) .- psi_xy_h)) < 3e-12
        @test maximum(abs.(Array(out_xz) .- psi_xz_h)) < 3e-12
        @test maximum(abs.(Array(out_yy) .- psi_yy_h)) < 3e-12
        @test maximum(abs.(Array(out_yz) .- psi_yz_h)) < 3e-12
        @test maximum(abs.(Array(out_zz) .- psi_zz_h)) < 3e-12
    end

    @testset "stress from 3D log-conformation matches G(C-I)" begin
        backend = KernelAbstractions.CPU()
        FT = Float64
        Nx, Ny, Nz = 3, 2, 2
        psi_xx = fill(FT(0.12), Nx, Ny, Nz)
        psi_xy = fill(FT(0.03), Nx, Ny, Nz)
        psi_xz = fill(FT(-0.02), Nx, Ny, Nz)
        psi_yy = fill(FT(-0.05), Nx, Ny, Nz)
        psi_yz = fill(FT(0.01), Nx, Ny, Nz)
        psi_zz = fill(FT(0.08), Nx, Ny, Nz)
        tau_xx = similar(psi_xx); tau_xy = similar(psi_xx); tau_xz = similar(psi_xx)
        tau_yy = similar(psi_xx); tau_yz = similar(psi_xx); tau_zz = similar(psi_xx)
        G = 0.17

        compute_stress_from_logconf_3d!(tau_xx, tau_xy, tau_xz, tau_yy, tau_yz, tau_zz,
                                        psi_xx, psi_xy, psi_xz, psi_yy, psi_yz, psi_zz;
                                        G)

        cxx, cxy, cxz, cyy, cyz, czz = mat_exp_sym3x3(0.12, 0.03, -0.02, -0.05, 0.01, 0.08)
        @test tau_xx[1, 1, 1] ≈ G * (cxx - 1)
        @test tau_xy[1, 1, 1] ≈ G * cxy
        @test tau_xz[1, 1, 1] ≈ G * cxz
        @test tau_yy[1, 1, 1] ≈ G * (cyy - 1)
        @test tau_yz[1, 1, 1] ≈ G * cyz
        @test tau_zz[1, 1, 1] ≈ G * (czz - 1)
    end

    @testset "3D log source has identity-gradient limit" begin
        lambda = 5.0
        grad = (0.03, 0.04, -0.02,
                0.01, -0.02, 0.005,
                -0.003, 0.007, 0.01)
        src = ntuple(c -> logconf_source_3d(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            grad..., lambda, c,
        ), 6)

        @test src[1] ≈ 2 * grad[1] atol=1e-14
        @test src[2] ≈ grad[2] + grad[4] atol=1e-14
        @test src[3] ≈ grad[3] + grad[7] atol=1e-14
        @test src[4] ≈ 2 * grad[5] atol=1e-14
        @test src[5] ≈ grad[6] + grad[8] atol=1e-14
        @test src[6] ≈ 2 * grad[9] atol=1e-14
    end

    @testset "3D log source relaxes diagonal Psi analytically" begin
        lambda = 4.0
        psi = (log(1.4), 0.0, 0.0, log(0.9), 0.0, log(1.1))
        src = ntuple(c -> logconf_source_3d(
            psi..., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            lambda, c,
        ), 6)
        @test src[1] ≈ -(1 - 1 / 1.4) / lambda atol=1e-14
        @test src[2] ≈ 0.0 atol=1e-14
        @test src[3] ≈ 0.0 atol=1e-14
        @test src[4] ≈ -(1 - 1 / 0.9) / lambda atol=1e-14
        @test src[5] ≈ 0.0 atol=1e-14
        @test src[6] ≈ -(1 - 1 / 1.1) / lambda atol=1e-14
    end

    @testset "3D log source matches finite-difference Dlog(C)[Cdot]" begin
        psi = (0.18, 0.04, -0.03, -0.07, 0.02, 0.11)
        grad = (0.012, -0.018, 0.007,
                0.021, -0.006, -0.004,
                -0.005, 0.009, -0.006)
        @test abs(grad[1] + grad[5] + grad[9]) < 1e-15
        lambda = 3.7
        eps_fd = 1e-6

        c = mat_exp_sym3x3(psi...)
        C = [c[1] c[2] c[3];
             c[2] c[4] c[5];
             c[3] c[5] c[6]]
        L = [grad[1] grad[2] grad[3];
             grad[4] grad[5] grad[6];
             grad[7] grad[8] grad[9]]
        Cdot = L * C + C * transpose(L) - (C - I) / lambda
        Ceps = C + eps_fd * Cdot
        psi_eps = mat_log_spd_sym3x3(
            Ceps[1, 1], Ceps[1, 2], Ceps[1, 3],
            Ceps[2, 2], Ceps[2, 3], Ceps[3, 3],
        )
        fd = ntuple(i -> (psi_eps[i] - psi[i]) / eps_fd, 6)
        src = ntuple(i -> logconf_source_3d(psi..., grad..., lambda, i), 6)

        for i in 1:6
            @test src[i] ≈ fd[i] atol=2e-7 rtol=2e-5
        end
    end

    @testset "3D logconf collision injects source moments at identity" begin
        backend = KernelAbstractions.CPU()
        FT = Float64
        Nx, Ny, Nz = 7, 7, 7
        ic, jc, kc = 4, 4, 4
        lambda = 10.0
        tau_plus = 1.0
        grad = (0.02, 0.03, -0.01,
                0.04, -0.015, 0.006,
                -0.002, 0.009, -0.005)
        @test abs(grad[1] + grad[5] + grad[9]) < 1e-15

        ux_h = zeros(FT, Nx, Ny, Nz)
        uy_h = zeros(FT, Nx, Ny, Nz)
        uz_h = zeros(FT, Nx, Ny, Nz)
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            x = FT(i - ic)
            y = FT(j - jc)
            z = FT(k - kc)
            ux_h[i, j, k] = grad[1] * x + grad[2] * y + grad[3] * z
            uy_h[i, j, k] = grad[4] * x + grad[5] * y + grad[6] * z
            uz_h[i, j, k] = grad[7] * x + grad[8] * y + grad[9] * z
        end
        ux = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz); copyto!(ux, ux_h)
        uy = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz); copyto!(uy, uy_h)
        uz = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz); copyto!(uz, uz_h)
        is_solid = KernelAbstractions.zeros(backend, Bool, Nx, Ny, Nz)

        fields = ntuple(_ -> KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz), 6)
        g = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19)
        init_conformation_field_3d!(g, fields[2], ux, uy, uz)
        collide_logconf_3d!(
            g, fields[2], ux, uy, uz,
            fields[1], fields[2], fields[3], fields[4], fields[5], fields[6],
            is_solid, tau_plus, lambda; component=2,
        )
        out = similar(fields[2])
        compute_conformation_macro_3d!(out, g)

        expected_xy = grad[2] + grad[4]
        @test Array(out)[ic, jc, kc] ≈ expected_xy atol=2e-14
    end

    @testset "3D logconf local shear evolution reaches Oldroyd-B steady state" begin
        backend = KernelAbstractions.CPU()
        FT = Float64
        Nx, Ny, Nz = 7, 15, 7
        ic, jc, kc = 4, 8, 4
        γdot = FT(0.005)
        λ = FT(40.0)
        G = FT(0.003)
        Wi = λ * γdot

        C_xy_an = Wi
        C_xx_an = one(FT) + FT(2) * Wi^2
        τ_xy_an = G * C_xy_an
        N1_an = G * (C_xx_an - one(FT))

        ux_h = zeros(FT, Nx, Ny, Nz)
        uy_h = zeros(FT, Nx, Ny, Nz)
        uz_h = zeros(FT, Nx, Ny, Nz)
        yc = FT(jc)
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            ux_h[i, j, k] = γdot * (FT(j) - yc)
        end
        ux = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz); copyto!(ux, ux_h)
        uy = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz); copyto!(uy, uy_h)
        uz = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz); copyto!(uz, uz_h)
        is_solid = KernelAbstractions.zeros(backend, Bool, Nx, Ny, Nz)

        Ψ_xx = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
        Ψ_xy = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
        Ψ_xz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
        Ψ_yy = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
        Ψ_yz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
        Ψ_zz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
        g_xx = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19)
        g_xy = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19)
        g_xz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19)
        g_yy = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19)
        g_yz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19)
        g_zz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19)
        init_conformation_field_3d!(g_xx, Ψ_xx, ux, uy, uz)
        init_conformation_field_3d!(g_xy, Ψ_xy, ux, uy, uz)
        init_conformation_field_3d!(g_xz, Ψ_xz, ux, uy, uz)
        init_conformation_field_3d!(g_yy, Ψ_yy, ux, uy, uz)
        init_conformation_field_3d!(g_yz, Ψ_yz, ux, uy, uz)
        init_conformation_field_3d!(g_zz, Ψ_zz, ux, uy, uz)

        for _ in 1:(20 * Int(round(λ)))
            compute_conformation_macro_3d!(Ψ_xx, g_xx)
            compute_conformation_macro_3d!(Ψ_xy, g_xy)
            compute_conformation_macro_3d!(Ψ_xz, g_xz)
            compute_conformation_macro_3d!(Ψ_yy, g_yy)
            compute_conformation_macro_3d!(Ψ_yz, g_yz)
            compute_conformation_macro_3d!(Ψ_zz, g_zz)

            collide_logconf_3d!(g_xx, Ψ_xx, ux, uy, uz,
                                Ψ_xx, Ψ_xy, Ψ_xz, Ψ_yy, Ψ_yz, Ψ_zz,
                                is_solid, 1.0, λ; component=1)
            collide_logconf_3d!(g_xy, Ψ_xy, ux, uy, uz,
                                Ψ_xx, Ψ_xy, Ψ_xz, Ψ_yy, Ψ_yz, Ψ_zz,
                                is_solid, 1.0, λ; component=2)
            collide_logconf_3d!(g_xz, Ψ_xz, ux, uy, uz,
                                Ψ_xx, Ψ_xy, Ψ_xz, Ψ_yy, Ψ_yz, Ψ_zz,
                                is_solid, 1.0, λ; component=3)
            collide_logconf_3d!(g_yy, Ψ_yy, ux, uy, uz,
                                Ψ_xx, Ψ_xy, Ψ_xz, Ψ_yy, Ψ_yz, Ψ_zz,
                                is_solid, 1.0, λ; component=4)
            collide_logconf_3d!(g_yz, Ψ_yz, ux, uy, uz,
                                Ψ_xx, Ψ_xy, Ψ_xz, Ψ_yy, Ψ_yz, Ψ_zz,
                                is_solid, 1.0, λ; component=5)
            collide_logconf_3d!(g_zz, Ψ_zz, ux, uy, uz,
                                Ψ_xx, Ψ_xy, Ψ_xz, Ψ_yy, Ψ_yz, Ψ_zz,
                                is_solid, 1.0, λ; component=6)
        end

        compute_conformation_macro_3d!(Ψ_xx, g_xx)
        compute_conformation_macro_3d!(Ψ_xy, g_xy)
        compute_conformation_macro_3d!(Ψ_xz, g_xz)
        compute_conformation_macro_3d!(Ψ_yy, g_yy)
        compute_conformation_macro_3d!(Ψ_yz, g_yz)
        compute_conformation_macro_3d!(Ψ_zz, g_zz)

        C_xx = similar(Ψ_xx); C_xy = similar(Ψ_xx); C_xz = similar(Ψ_xx)
        C_yy = similar(Ψ_xx); C_yz = similar(Ψ_xx); C_zz = similar(Ψ_xx)
        psi_to_C_3d!(C_xx, C_xy, C_xz, C_yy, C_yz, C_zz,
                     Ψ_xx, Ψ_xy, Ψ_xz, Ψ_yy, Ψ_yz, Ψ_zz)

        tau_xx = similar(Ψ_xx); tau_xy = similar(Ψ_xx); tau_xz = similar(Ψ_xx)
        tau_yy = similar(Ψ_xx); tau_yz = similar(Ψ_xx); tau_zz = similar(Ψ_xx)
        compute_stress_from_logconf_3d!(tau_xx, tau_xy, tau_xz, tau_yy, tau_yz, tau_zz,
                                        Ψ_xx, Ψ_xy, Ψ_xz, Ψ_yy, Ψ_yz, Ψ_zz; G)

        Cxx_num = Array(C_xx)[ic, jc, kc]
        Cxy_num = Array(C_xy)[ic, jc, kc]
        Cxz_num = Array(C_xz)[ic, jc, kc]
        Cyy_num = Array(C_yy)[ic, jc, kc]
        Cyz_num = Array(C_yz)[ic, jc, kc]
        Czz_num = Array(C_zz)[ic, jc, kc]
        txy_num = Array(tau_xy)[ic, jc, kc]
        N1_num = Array(tau_xx)[ic, jc, kc] - Array(tau_yy)[ic, jc, kc]

        @test Cxy_num ≈ C_xy_an rtol=0.05
        @test Cxx_num ≈ C_xx_an rtol=0.05
        @test abs(Cxz_num) < 1e-3
        @test abs(Cyz_num) < 1e-3
        @test Cyy_num ≈ 1.0 atol=1e-3
        @test Czz_num ≈ 1.0 atol=1e-3
        @test txy_num ≈ τ_xy_an rtol=0.05
        @test N1_num ≈ N1_an rtol=0.08
    end
end
