using Test
using Kraken

@testset "Viscoelastic coupling unit tests" begin

    # Shared Poiseuille setup: body-force channel, periodic x, walls y.
    # Uses collide_viscoelastic_source_guo_2d! with PRESCRIBED τ_p arrays
    # (no conformation evolution) to isolate the Hermite stress source coupling.

    function run_poiseuille_prescribed_stress(;
            Nx=4, Ny=64, ν_s=0.1, Fx=1e-5,
            tau_p_xx_val=0.0, tau_p_xy_fn=nothing, tau_p_yy_val=0.0,
            max_steps=50_000)
        ω_s = 1.0 / (3.0 * ν_s + 0.5)

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

        tau_p_xx = fill(Float64(tau_p_xx_val), Nx, Ny)
        tau_p_xy = zeros(Float64, Nx, Ny)
        tau_p_yy = fill(Float64(tau_p_yy_val), Nx, Ny)

        if !isnothing(tau_p_xy_fn)
            for j in 1:Ny, i in 1:Nx
                tau_p_xy[i, j] = tau_p_xy_fn(j)
            end
        end

        for step in 1:max_steps
            stream_periodic_x_wall_y_2d!(f_out, f_in, Nx, Ny)
            collide_viscoelastic_source_guo_2d!(f_out, is_solid, ω_s,
                                                  Fx, 0.0,
                                                  tau_p_xx, tau_p_xy, tau_p_yy)
            f_in, f_out = f_out, f_in
            compute_macroscopic_2d!(ρ, ux, uy, f_in)
        end

        return ux[2, :]
    end

    # Analytical Poiseuille at viscosity ν with body force Fx:
    #   u(j) = Fx/(2ν) · (j - 0.5)(H + 0.5 - j),  H = Ny
    poiseuille_ana(Fx, ν, Ny) =
        [Fx / (2ν) * (j - 0.5) * (Ny + 0.5 - j) for j in 1:Ny]


    @testset "1a: Poiseuille ν_s only, τ_p=0 → u_max = Fx H²/(8 ν_s)" begin
        # With zero polymer stress, the velocity profile must be
        # determined by the solvent viscosity ν_s alone.
        ν_s = 0.1
        Fx  = 1e-5
        Ny  = 64

        u_num = run_poiseuille_prescribed_stress(; ν_s=ν_s, Fx=Fx, Ny=Ny)
        u_ana = poiseuille_ana(Fx, ν_s, Ny)

        u_max_num = maximum(u_num)
        u_max_ana = maximum(u_ana)

        @info "Test 1a: τ_p=0" u_max_num=round(u_max_num, digits=6) u_max_ana=round(u_max_ana, digits=6) ratio=round(u_max_num/u_max_ana, digits=5)

        errors = abs.(u_num[3:end-2] .- u_ana[3:end-2]) ./ u_max_ana
        @test maximum(errors) < 0.02   # < 2 %
        @test isapprox(u_max_num, u_max_ana; rtol=0.01)
    end

    @testset "1b: Uniform τ_p_xy=const → no change in velocity" begin
        # A spatially uniform polymer stress has zero divergence,
        # so adding τ_p_xy = const must NOT change the velocity profile.
        # Compare to test 1a (τ_p = 0).
        ν_s = 0.1
        Fx  = 1e-5
        Ny  = 64

        u_ref = run_poiseuille_prescribed_stress(; ν_s=ν_s, Fx=Fx, Ny=Ny)
        u_cst = run_poiseuille_prescribed_stress(; ν_s=ν_s, Fx=Fx, Ny=Ny,
                    tau_p_xy_fn = _ -> 0.01)    # constant 0.01

        diff = maximum(abs.(u_ref .- u_cst))
        u_max = maximum(u_ref)

        @info "Test 1b: uniform τ_p_xy" max_diff=diff rel_diff=round(diff/u_max, digits=8)

        @test diff / u_max < 1e-4   # should be ~0 (machine precision noise)
    end

    @testset "1c: Linear τ_p_xy → modified u_max" begin
        # Impose τ_p_xy(y) = A · (H/2 - (y - 0.5)) = linear shear stress
        # like in an Oldroyd-B Poiseuille. The polymer carries part of the
        # applied load, so the velocity gradient is REDUCED:
        #   σ_xy = ν_s γ̇ + τ_p_xy  →  ∂_y σ_xy = -Fx
        #   ν_s ∂²u/∂y² + ∂_y τ_p_xy = -Fx
        #   ν_s ∂²u/∂y² = -(Fx - A)     (since ∂_y τ_p_xy = -A)
        # u_max = (Fx - A) · H² / (8 ν_s)  — LESS than the τ_p=0 case.
        #
        # This tests the QUANTITATIVE coupling of the Hermite source.
        ν_s = 0.1
        Fx  = 1e-5
        Ny  = 64
        H   = Float64(Ny)
        A   = 5e-6    # extra "body force" from polymer stress gradient

        tau_p_xy_fn(j) = A * (H/2 - (j - 0.5))

        u_num = run_poiseuille_prescribed_stress(; ν_s=ν_s, Fx=Fx, Ny=Ny,
                    tau_p_xy_fn=tau_p_xy_fn)
        u_ana = poiseuille_ana(Fx - A, ν_s, Ny)

        u_max_num = maximum(u_num)
        u_max_ana = maximum(u_ana)

        @info "Test 1c: linear τ_p_xy" A u_max_num=round(u_max_num, digits=6) u_max_ana=round(u_max_ana, digits=6) ratio=round(u_max_num/u_max_ana, digits=5)

        errors = abs.(u_num[3:end-2] .- u_ana[3:end-2]) ./ u_max_ana
        @test maximum(errors) < 0.03   # < 3 %
        @test isapprox(u_max_num, u_max_ana; rtol=0.02)
    end

    @testset "1d: Effective viscosity ν_total at varying β" begin
        # Full coupling (Oldroyd-B conformation + flow) at very low Wi.
        # At Wi→0, polymer acts as extra Newtonian viscosity ν_p.
        # So u_max → Fx·H²/(8·ν_total).
        #
        # Test at two different β (viscosity ratios) to detect a scaling bug
        # in the Hermite source normalization (factor ω, cs², etc.).
        Nx, Ny = 4, 64
        Fx     = 1e-5
        max_steps = 100_000

        for (ν_s, ν_p, label) in [(0.06, 0.04, "β=0.6"),
                                    (0.04, 0.06, "β=0.4")]
            ν_total = ν_s + ν_p
            lambda  = 50.0      # large λ but small Wi because Fx is small
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
                                                      Fx, 0.0,
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

            u_ana = poiseuille_ana(Fx, ν_total, Ny)
            u_num = ux[2, :]
            u_max_num = maximum(u_num)
            u_max_ana = maximum(u_ana)

            @info "Test 1d: $label" ν_s ν_p ν_total u_max_num=round(u_max_num, digits=6) u_max_ana=round(u_max_ana, digits=6) ratio=round(u_max_num/u_max_ana, digits=4)

            @test isapprox(u_max_num, u_max_ana; rtol=0.05)
        end
    end
end
