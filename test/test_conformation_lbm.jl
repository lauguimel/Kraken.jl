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

    @testset "Pure shear: Oldroyd-B steady state" begin
        # u_x = γ̇·y, prescribed velocity
        # Analytical Oldroyd-B steady state:
        #   C_xx = 1 + 2(λγ̇)²
        #   C_xy = λγ̇
        #   C_yy = 1
        Nx, Ny = 16, 32
        ν_p = 0.05
        lambda = 10.0
        γ̇ = 0.01
        tau_plus = 1.0

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

        is_solid = falses(Nx, Ny)
        g_xx_buf = similar(g_xx); g_xy_buf = similar(g_xy); g_yy_buf = similar(g_yy)

        for step in 1:50_000
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

        i, j = Nx÷2, Ny÷2
        Wi = lambda*γ̇
        ana_xx = 1.0 + 2*Wi^2
        ana_xy = Wi
        ana_yy = 1.0

        err_xx = (C_xx[i,j] - ana_xx) / ana_xx * 100
        err_xy = (C_xy[i,j] - ana_xy) / ana_xy * 100
        err_yy = (C_yy[i,j] - ana_yy) / ana_yy * 100

        @info "Pure shear (TRT-LBM)" Wi Cxx=round(C_xx[i,j],digits=4) Cxx_ana=round(ana_xx,digits=4) err_xx=round(err_xx,digits=2) Cxy=round(C_xy[i,j],digits=4) Cxy_ana=round(ana_xy,digits=4) err_xy=round(err_xy,digits=2) Cyy=round(C_yy[i,j],digits=4) Cyy_ana=round(ana_yy,digits=4) err_yy=round(err_yy,digits=2)

        # Tolerance: 15% (artificial diffusion broadens the response,
        # but we want correct sign and magnitude)
        @test C_xx[i,j] ≈ ana_xx rtol=0.15
        @test C_xy[i,j] ≈ ana_xy rtol=0.15
        @test C_yy[i,j] ≈ ana_yy rtol=0.05
    end
end
