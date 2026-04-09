using Test
using Kraken

@testset "Conformation TRT-LBM (Liu 2025)" begin

    @testset "Pure relaxation: C → I (no flow)" begin
        # No velocity, perturbed initial C: should relax exponentially to I
        # at rate 1/λ, AND diffuse spatially toward uniform.
        Nx, Ny = 16, 16
        lambda = 10.0
        tau_plus = 1.0  # κ = (1 - 0.5)/3 = 0.167

        ux = zeros(Float64, Nx, Ny)
        uy = zeros(Float64, Nx, Ny)
        is_solid = falses(Nx, Ny)

        # Uniform perturbation away from I (so diffusion does nothing)
        C_xx = fill(2.0, Nx, Ny)
        C_xy = fill(0.5, Nx, Ny)
        C_yy = fill(1.5, Nx, Ny)

        g_xx = zeros(Float64, Nx, Ny, 9)
        g_xy = zeros(Float64, Nx, Ny, 9)
        g_yy = zeros(Float64, Nx, Ny, 9)
        init_conformation_field_2d!(g_xx, C_xx, ux, uy)
        init_conformation_field_2d!(g_xy, C_xy, ux, uy)
        init_conformation_field_2d!(g_yy, C_yy, ux, uy)

        g_xx_buf = similar(g_xx); g_xy_buf = similar(g_xy); g_yy_buf = similar(g_yy)

        n_steps = 50
        for step in 1:n_steps
            stream_2d!(g_xx_buf, g_xx, Nx, Ny)
            stream_2d!(g_xy_buf, g_xy, Nx, Ny)
            stream_2d!(g_yy_buf, g_yy, Nx, Ny)
            g_xx, g_xx_buf = g_xx_buf, g_xx
            g_xy, g_xy_buf = g_xy_buf, g_xy
            g_yy, g_yy_buf = g_yy_buf, g_yy

            compute_conformation_macro_2d!(C_xx, g_xx)
            compute_conformation_macro_2d!(C_xy, g_xy)
            compute_conformation_macro_2d!(C_yy, g_yy)

            collide_conformation_2d!(g_xx, C_xx, ux, uy, C_xx, C_xy, C_yy, is_solid, tau_plus, lambda; component=1)
            collide_conformation_2d!(g_xy, C_xy, ux, uy, C_xx, C_xy, C_yy, is_solid, tau_plus, lambda; component=2)
            collide_conformation_2d!(g_yy, C_yy, ux, uy, C_xx, C_xy, C_yy, is_solid, tau_plus, lambda; component=3)
        end
        compute_conformation_macro_2d!(C_xx, g_xx)
        compute_conformation_macro_2d!(C_xy, g_xy)
        compute_conformation_macro_2d!(C_yy, g_yy)

        # Analytical (uniform field, just relaxation):
        #   C_xx(t) - 1 = (C_xx(0) - 1) · exp(-t/λ)
        i, j = Nx÷2, Ny÷2
        ana_xx = 1.0 + 1.0 * exp(-n_steps/lambda)
        ana_xy = 0.5 * exp(-n_steps/lambda)
        ana_yy = 1.0 + 0.5 * exp(-n_steps/lambda)

        @info "Pure relaxation" n_steps Cxx_num=round(C_xx[i,j],digits=4) Cxx_ana=round(ana_xx,digits=4) Cxy_num=round(C_xy[i,j],digits=4) Cxy_ana=round(ana_xy,digits=4) Cyy_num=round(C_yy[i,j],digits=4) Cyy_ana=round(ana_yy,digits=4)

        # Tolerance: TRT introduces some smoothing, but the relaxation
        # rate is captured to within ~25% on the small (~1e-3) Cxy value.
        @test C_xx[i,j] ≈ ana_xx rtol=0.05
        @test C_xy[i,j] ≈ ana_xy atol=0.001
        @test C_yy[i,j] ≈ ana_yy rtol=0.05
    end

    @testset "Pure shear: Oldroyd-B steady state (fully periodic)" begin
        # u_x = γ̇·y, prescribed velocity, FULLY PERIODIC streaming.
        # Note: bounce-back streaming pollutes the bulk for prescribed
        # velocity tests — must use stream_fully_periodic_2d!.
        # Analytical Oldroyd-B steady state:
        #   C_xx = 1 + 2(λγ̇)²
        #   C_xy = λγ̇
        #   C_yy = 1
        Nx, Ny = 16, 32
        lambda = 10.0
        tau_plus = 1.0  # κ = 1/6 — sweet spot for accuracy at low Wi
        is_solid = falses(Nx, Ny)

        for γ̇ in [0.01, 0.03, 0.05]
            Wi = lambda * γ̇

            ux = zeros(Float64, Nx, Ny)
            uy = zeros(Float64, Nx, Ny)
            for j in 1:Ny, i in 1:Nx
                ux[i,j] = γ̇ * (j - 0.5)
            end

            C_xx = ones(Float64, Nx, Ny)
            C_xy = zeros(Float64, Nx, Ny)
            C_yy = ones(Float64, Nx, Ny)

            g_xx = zeros(Float64, Nx, Ny, 9)
            g_xy = zeros(Float64, Nx, Ny, 9)
            g_yy = zeros(Float64, Nx, Ny, 9)
            init_conformation_field_2d!(g_xx, C_xx, ux, uy)
            init_conformation_field_2d!(g_xy, C_xy, ux, uy)
            init_conformation_field_2d!(g_yy, C_yy, ux, uy)

            g_xx_buf = similar(g_xx); g_xy_buf = similar(g_xy); g_yy_buf = similar(g_yy)

            for _ in 1:50_000
                stream_fully_periodic_2d!(g_xx_buf, g_xx, Nx, Ny)
                stream_fully_periodic_2d!(g_xy_buf, g_xy, Nx, Ny)
                stream_fully_periodic_2d!(g_yy_buf, g_yy, Nx, Ny)
                g_xx, g_xx_buf = g_xx_buf, g_xx
                g_xy, g_xy_buf = g_xy_buf, g_xy
                g_yy, g_yy_buf = g_yy_buf, g_yy

                compute_conformation_macro_2d!(C_xx, g_xx)
                compute_conformation_macro_2d!(C_xy, g_xy)
                compute_conformation_macro_2d!(C_yy, g_yy)

                collide_conformation_2d!(g_xx, C_xx, ux, uy, C_xx, C_xy, C_yy, is_solid, tau_plus, lambda; component=1)
                collide_conformation_2d!(g_xy, C_xy, ux, uy, C_xx, C_xy, C_yy, is_solid, tau_plus, lambda; component=2)
                collide_conformation_2d!(g_yy, C_yy, ux, uy, C_xx, C_xy, C_yy, is_solid, tau_plus, lambda; component=3)
            end
            compute_conformation_macro_2d!(C_xx, g_xx)
            compute_conformation_macro_2d!(C_xy, g_xy)
            compute_conformation_macro_2d!(C_yy, g_yy)

            i, j = Nx÷2, Ny÷2
            ana_xx = 1.0 + 2*Wi^2
            ana_xy = Wi
            err_xx = (C_xx[i,j] - ana_xx) / ana_xx * 100
            err_xy = (C_xy[i,j] - ana_xy) / ana_xy * 100

            @info "Pure shear" γ̇ Wi Cxx=round(C_xx[i,j],digits=6) ana_xx=round(ana_xx,digits=6) err_xx=round(err_xx,digits=3) Cxy=round(C_xy[i,j],digits=6) ana_xy=round(ana_xy,digits=6) err_xy=round(err_xy,digits=3)

            # Tight tolerance now that bounce-back is gone
            @test C_xx[i,j] ≈ ana_xx rtol=0.005
            @test C_xy[i,j] ≈ ana_xy rtol=0.005
            @test C_yy[i,j] ≈ 1.0 atol=1e-6
        end
    end

    @testset "Oldroyd-B Poiseuille channel (full LBM coupling)" begin
        # Body-force-driven channel: u_x(y) = (Fx/(2ν_total)) y(H-y)
        # Analytical N1(y) = 2 ν_p λ γ̇²(y) where γ̇(y) = (Fx/ν_total)(H/2 - y)
        Nx, Ny = 4, 64
        ν_s = 0.04
        ν_p = 0.06
        ν_total = ν_s + ν_p
        lambda = 50.0
        Fx_val = 1e-5
        max_steps = 100_000
        G = ν_p / lambda
        ω_s = 1.0 / (3.0 * ν_s + 0.5)
        tau_plus = 1.0

        f_in  = zeros(Float64, Nx, Ny, 9)
        f_out = zeros(Float64, Nx, Ny, 9)
        is_solid = falses(Nx, Ny)
        ux = zeros(Float64, Nx, Ny)
        uy = zeros(Float64, Nx, Ny)
        ρ  = ones(Float64, Nx, Ny)

        for j in 1:Ny, i in 1:Nx, q in 1:9
            f_in[i,j,q] = Kraken.equilibrium(D2Q9(), 1.0, 0.0, 0.0, q)
        end
        copy!(f_out, f_in)

        C_xx = ones(Float64, Nx, Ny)
        C_xy = zeros(Float64, Nx, Ny)
        C_yy = ones(Float64, Nx, Ny)
        g_xx = zeros(Float64, Nx, Ny, 9)
        g_xy = zeros(Float64, Nx, Ny, 9)
        g_yy = zeros(Float64, Nx, Ny, 9)
        init_conformation_field_2d!(g_xx, C_xx, ux, uy)
        init_conformation_field_2d!(g_xy, C_xy, ux, uy)
        init_conformation_field_2d!(g_yy, C_yy, ux, uy)
        g_xx_buf = similar(g_xx); g_xy_buf = similar(g_xy); g_yy_buf = similar(g_yy)

        tau_p_xx = zeros(Float64, Nx, Ny)
        tau_p_xy = zeros(Float64, Nx, Ny)
        tau_p_yy = zeros(Float64, Nx, Ny)

        for step in 1:max_steps
            stream_periodic_x_wall_y_2d!(f_out, f_in, Nx, Ny)
            collide_viscoelastic_source_guo_2d!(f_out, is_solid, ω_s,
                                                 Fx_val, 0.0,
                                                 tau_p_xx, tau_p_xy, tau_p_yy)
            f_in, f_out = f_out, f_in
            compute_macroscopic_2d!(ρ, ux, uy, f_in)

            stream_periodic_x_wall_y_2d!(g_xx_buf, g_xx, Nx, Ny)
            stream_periodic_x_wall_y_2d!(g_xy_buf, g_xy, Nx, Ny)
            stream_periodic_x_wall_y_2d!(g_yy_buf, g_yy, Nx, Ny)
            g_xx, g_xx_buf = g_xx_buf, g_xx
            g_xy, g_xy_buf = g_xy_buf, g_xy
            g_yy, g_yy_buf = g_yy_buf, g_yy

            compute_conformation_macro_2d!(C_xx, g_xx)
            compute_conformation_macro_2d!(C_xy, g_xy)
            compute_conformation_macro_2d!(C_yy, g_yy)

            collide_conformation_2d!(g_xx, C_xx, ux, uy, C_xx, C_xy, C_yy, is_solid, tau_plus, lambda; component=1)
            collide_conformation_2d!(g_xy, C_xy, ux, uy, C_xx, C_xy, C_yy, is_solid, tau_plus, lambda; component=2)
            collide_conformation_2d!(g_yy, C_yy, ux, uy, C_xx, C_xy, C_yy, is_solid, tau_plus, lambda; component=3)

            @. tau_p_xx = G * (C_xx - 1.0)
            @. tau_p_xy = G * C_xy
            @. tau_p_yy = G * (C_yy - 1.0)
        end

        H = Float64(Ny)
        u_ana = [Fx_val / (2 * ν_total) * (j - 0.5) * (H + 0.5 - j) for j in 1:Ny]
        u_num = ux[2, :]
        u_max_ana = maximum(u_ana)
        u_max_num = maximum(u_num)

        @info "Poiseuille velocity" u_max_num=round(u_max_num, digits=6) u_max_ana=round(u_max_ana, digits=6) ratio=round(u_max_num/u_max_ana, digits=4)

        errors_u = abs.(u_num[3:end-2] .- u_ana[3:end-2]) ./ u_max_ana
        @test maximum(errors_u) < 0.10

        N1_num = tau_p_xx[2, :] .- tau_p_yy[2, :]
        N1_ana = zeros(Ny)
        for j in 1:Ny
            γ_dot = Fx_val / ν_total * (H/2 - (j - 0.5))
            N1_ana[j] = 2 * ν_p * lambda * γ_dot^2
        end

        N1_center = N1_num[Ny÷2]
        N1_quart  = N1_num[Ny÷4]
        N1_quart_ana = N1_ana[Ny÷4]

        @info "Poiseuille N1" N1_center=round(N1_center, digits=8) N1_quart_num=round(N1_quart, digits=8) N1_quart_ana=round(N1_quart_ana, digits=8) ratio=round(N1_quart/N1_quart_ana, digits=4)

        @test N1_quart > 0
        @test N1_quart ≈ N1_quart_ana rtol=0.30
        @test abs(N1_center) < 0.1 * abs(N1_quart_ana)
    end
end
