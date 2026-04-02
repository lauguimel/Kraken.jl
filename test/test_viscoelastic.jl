using Test
using Kraken

@testset "Viscoelastic" begin

    @testset "2×2 symmetric matrix algebra" begin
        # Identity matrix: eigenvalues = 1,1
        λ1, λ2, e1x, e1y, e2x, e2y = eigen_sym2x2(1.0, 0.0, 1.0)
        @test λ1 ≈ 1.0
        @test λ2 ≈ 1.0

        # Diagonal matrix [3,0;0,1]: eigenvalues = 3,1
        λ1, λ2, e1x, e1y, e2x, e2y = eigen_sym2x2(3.0, 0.0, 1.0)
        @test λ1 ≈ 3.0
        @test λ2 ≈ 1.0
        @test abs(e1x) ≈ 1.0 atol=1e-10  # eigenvec along x
        @test abs(e1y) ≈ 0.0 atol=1e-10

        # Symmetric matrix [2,1;1,2]: eigenvalues = 3,1
        λ1, λ2, e1x, e1y, e2x, e2y = eigen_sym2x2(2.0, 1.0, 2.0)
        @test λ1 ≈ 3.0
        @test λ2 ≈ 1.0

        # Matrix exponential of zero = identity
        e11, e12, e22 = mat_exp_sym2x2(0.0, 0.0, 0.0)
        @test e11 ≈ 1.0
        @test e12 ≈ 0.0 atol=1e-14
        @test e22 ≈ 1.0

        # Matrix log of identity = zero
        l11, l12, l22 = mat_log_sym2x2(1.0, 0.0, 1.0)
        @test l11 ≈ 0.0 atol=1e-14
        @test l12 ≈ 0.0 atol=1e-14
        @test l22 ≈ 0.0 atol=1e-14

        # exp(log(A)) = A roundtrip
        a11, a12, a22 = 3.0, 0.5, 2.0
        l11, l12, l22 = mat_log_sym2x2(a11, a12, a22)
        r11, r12, r22 = mat_exp_sym2x2(l11, l12, l22)
        @test r11 ≈ a11 atol=1e-12
        @test r12 ≈ a12 atol=1e-12
        @test r22 ≈ a22 atol=1e-12

        # log(exp(B)) = B roundtrip
        b11, b12, b22 = 1.5, -0.3, 0.8
        e11, e12, e22 = mat_exp_sym2x2(b11, b12, b22)
        r11, r12, r22 = mat_log_sym2x2(e11, e12, e22)
        @test r11 ≈ b11 atol=1e-12
        @test r12 ≈ b12 atol=1e-12
        @test r22 ≈ b22 atol=1e-12
    end

    @testset "Oldroyd-B channel flow (log-conformation)" begin
        # Oldroyd-B fully developed channel flow:
        # Analytical: velocity = Newtonian Poiseuille (total viscosity = ν_s + ν_p)
        # First normal stress difference: N1 = 2·ν_p·λ·γ̇²

        Nx, Ny = 4, 32
        ν_s = 0.08
        ν_p = 0.02
        ν_total = ν_s + ν_p
        lambda = 5.0
        G = ν_p / lambda  # elastic modulus
        Fx_val = 1e-5
        max_steps = 30000

        # Initialize LBM
        f_in  = zeros(Float64, Nx, Ny, 9)
        f_out = zeros(Float64, Nx, Ny, 9)
        is_solid = falses(Nx, Ny)
        ux = zeros(Float64, Nx, Ny)
        uy = zeros(Float64, Nx, Ny)
        ρ  = ones(Float64, Nx, Ny)

        for j in 1:Ny, i in 1:Nx, q in 1:9
            f_in[i,j,q] = equilibrium(D2Q9(), 1.0, 0.0, 0.0, q)
        end
        copy!(f_out, f_in)

        # Log-conformation arrays (Θ = log(C), initialized to log(I) = 0)
        Θ_xx = zeros(Float64, Nx, Ny)
        Θ_xy = zeros(Float64, Nx, Ny)
        Θ_yy = zeros(Float64, Nx, Ny)
        Θ_xx_new = zeros(Float64, Nx, Ny)
        Θ_xy_new = zeros(Float64, Nx, Ny)
        Θ_yy_new = zeros(Float64, Nx, Ny)

        # Polymeric stress arrays
        tau_p_xx = zeros(Float64, Nx, Ny)
        tau_p_xy = zeros(Float64, Nx, Ny)
        tau_p_yy = zeros(Float64, Nx, Ny)
        Fx_p = zeros(Float64, Nx, Ny)
        Fy_p = zeros(Float64, Nx, Ny)

        # Total force = body force + polymeric stress divergence
        Fx_total = fill(Float64(Fx_val), Nx, Ny)
        Fy_total = zeros(Float64, Nx, Ny)

        ω_s = 1.0 / (3.0 * ν_s + 0.5)

        for step in 1:max_steps
            # 1. Stream
            stream_periodic_x_wall_y_2d!(f_out, f_in, Nx, Ny)

            # 2. Collide with solvent viscosity + total force (body + polymeric)
            collide_guo_field_2d!(f_out, is_solid, Fx_total, Fy_total, Float64(ω_s))

            # 3. Macroscopic (force correction is small, skip for simplicity)
            compute_macroscopic_2d!(ρ, ux, uy, f_out)

            # 4. Evolve log-conformation
            evolve_logconf_2d!(Θ_xx_new, Θ_xy_new, Θ_yy_new,
                               Θ_xx, Θ_xy, Θ_yy,
                               ux, uy; lambda=lambda)
            copyto!(Θ_xx, Θ_xx_new)
            copyto!(Θ_xy, Θ_xy_new)
            copyto!(Θ_yy, Θ_yy_new)

            # 5. Compute polymeric stress from Θ
            compute_stress_from_logconf_2d!(tau_p_xx, tau_p_xy, tau_p_yy,
                                            Θ_xx, Θ_xy, Θ_yy; G=G)

            # 6. Polymeric force = divergence of τ_p
            compute_polymeric_force_2d!(Fx_p, Fy_p, tau_p_xx, tau_p_xy, tau_p_yy)

            # 7. Update total force
            Fx_total .= Fx_val .+ Fx_p
            Fy_total .= Fy_p

            # Swap
            f_in, f_out = f_out, f_in
        end

        # --- Check velocity profile ---
        # Analytical: same as Newtonian Poiseuille with ν = ν_s + ν_p
        # (the polymeric stress contributes to the total shear stress
        #  but doesn't change the velocity profile at steady state)
        H = Float64(Ny)
        u_analytical = [Fx_val / (2 * ν_total) * (j - 0.5) * (H + 0.5 - j) for j in 1:Ny]
        u_num = ux[2, :]
        u_max_ana = maximum(u_analytical)
        u_max_num = maximum(u_num)

        errors_u = abs.(u_num[3:end-2] .- u_analytical[3:end-2]) ./ u_max_ana
        max_err_u = maximum(errors_u)

        @info "Oldroyd-B channel: u_max error = $(round(max_err_u*100, digits=2))%, u_max_num=$(round(u_max_num, digits=6)), u_max_ana=$(round(u_max_ana, digits=6))"
        @test max_err_u < 0.15  # 15% tolerance (Ny=32, no Guo correction in macroscopic)

        # --- Check first normal stress difference ---
        # N1 = τ_xx - τ_yy = 2·ν_p·λ·γ̇²
        # At the center (y = Ny/2): γ̇ ≈ 0, so N1 ≈ 0
        # Near walls: γ̇ = Fx/(2ν)·H/2, so N1 should be positive
        N1_center = tau_p_xx[2, Ny÷2] - tau_p_yy[2, Ny÷2]
        N1_wall   = tau_p_xx[2, 3]     - tau_p_yy[2, 3]
        @info "Oldroyd-B: N1_center = $(round(N1_center, digits=8)), N1_wall = $(round(N1_wall, digits=8))"
        @test N1_wall > N1_center  # N1 should be larger near walls
        @test N1_center ≈ 0.0 atol=abs(N1_wall)*0.2  # N1 near zero at center
    end
end
