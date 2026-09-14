using Test
using Kraken

@testset "Viscoelastic equation transcription checks" begin
    @testset "Oldroyd-B stress closure matches rheoTool convention" begin
        C_xx = [1.2 0.9; 1.0 1.5]
        C_xy = [0.3 -0.2; 0.0 0.1]
        C_yy = [0.8 1.1; 1.0 0.7]
        τ_xx = similar(C_xx)
        τ_xy = similar(C_xy)
        τ_yy = similar(C_yy)
        model = OldroydB(G=4.1, λ=0.1)

        update_polymer_stress!(τ_xx, τ_xy, τ_yy, C_xx, C_xy, C_yy, model)

        @test τ_xx ≈ 4.1 .* (C_xx .- 1.0)
        @test τ_xy ≈ 4.1 .* C_xy
        @test τ_yy ≈ 4.1 .* (C_yy .- 1.0)
    end

    @testset "Conservative CDE source includes C div(u)" begin
        Nx, Ny = 7, 7
        ic, jc = 4, 4
        λ = 5.0
        tau_plus = 1.0
        cxx, cxy, cyy = 1.3, -0.2, 0.7
        dudx, dudy = 0.03, -0.04
        dvdx, dvdy = 0.02, -0.01
        divu = dudx + dvdy

        ux = zeros(Float64, Nx, Ny)
        uy = zeros(Float64, Nx, Ny)
        for j in 1:Ny, i in 1:Nx
            x = i - ic
            y = j - jc
            ux[i, j] = dudx * x + dudy * y
            uy[i, j] = dvdx * x + dvdy * y
        end
        is_solid = falses(Nx, Ny)
        C_xx = fill(cxx, Nx, Ny)
        C_xy = fill(cxy, Nx, Ny)
        C_yy = fill(cyy, Nx, Ny)
        g_xx = zeros(Float64, Nx, Ny, 9)
        g_xy = zeros(Float64, Nx, Ny, 9)
        g_yy = zeros(Float64, Nx, Ny, 9)
        init_conformation_field_2d!(g_xx, C_xx, ux, uy)
        init_conformation_field_2d!(g_xy, C_xy, ux, uy)
        init_conformation_field_2d!(g_yy, C_yy, ux, uy)

        collide_conformation_2d!(g_xx, C_xx, ux, uy, C_xx, C_xy, C_yy,
                                  is_solid, tau_plus, λ; component=1)
        collide_conformation_2d!(g_xy, C_xy, ux, uy, C_xx, C_xy, C_yy,
                                  is_solid, tau_plus, λ; component=2)
        collide_conformation_2d!(g_yy, C_yy, ux, uy, C_xx, C_xy, C_yy,
                                  is_solid, tau_plus, λ; component=3)

        C_xx_new = similar(C_xx)
        C_xy_new = similar(C_xy)
        C_yy_new = similar(C_yy)
        compute_conformation_macro_2d!(C_xx_new, g_xx)
        compute_conformation_macro_2d!(C_xy_new, g_xy)
        compute_conformation_macro_2d!(C_yy_new, g_yy)

        S_xx = -(cxx - 1.0) / λ + 2.0 * (cxx * dudx + cxy * dudy) + cxx * divu
        S_xy = -cxy / λ + cxx * dvdx + cyy * dudy + cxy * (dudx + dvdy) + cxy * divu
        S_yy = -(cyy - 1.0) / λ + 2.0 * (cxy * dvdx + cyy * dvdy) + cyy * divu

        @test C_xx_new[ic, jc] ≈ cxx + S_xx atol=1e-14
        @test C_xy_new[ic, jc] ≈ cxy + S_xy atol=1e-14
        @test C_yy_new[ic, jc] ≈ cyy + S_yy atol=1e-14
    end

    @testset "Conformation macro source shift applies local Oldroyd-B source" begin
        Nx, Ny = 7, 7
        ic, jc = 4, 4
        λ = 5.0
        shift = 0.5
        cxx, cxy, cyy = 1.3, -0.2, 0.7
        dudx, dudy = 0.03, -0.04
        dvdx, dvdy = 0.02, -0.01
        divu = dudx + dvdy

        ux = zeros(Float64, Nx, Ny)
        uy = zeros(Float64, Nx, Ny)
        for j in 1:Ny, i in 1:Nx
            x = i - ic
            y = j - jc
            ux[i, j] = dudx * x + dudy * y
            uy[i, j] = dvdx * x + dvdy * y
        end
        is_solid = falses(Nx, Ny)
        C_xx = fill(cxx, Nx, Ny)
        C_xy = fill(cxy, Nx, Ny)
        C_yy = fill(cyy, Nx, Ny)

        Kraken.apply_conformation_macro_source_shift_2d!(C_xx, C_xy, C_yy, ux, uy,
                                                          is_solid, λ, shift)

        S_xx = -(cxx - 1.0) / λ + 2.0 * (cxx * dudx + cxy * dudy) + cxx * divu
        S_xy = -cxy / λ + cxx * dvdx + cyy * dudy + cxy * (dudx + dvdy) + cxy * divu
        S_yy = -(cyy - 1.0) / λ + 2.0 * (cxy * dvdx + cyy * dvdy) + cyy * divu

        @test C_xx[ic, jc] ≈ cxx + shift * S_xx atol=1e-14
        @test C_xy[ic, jc] ≈ cxy + shift * S_xy atol=1e-14
        @test C_yy[ic, jc] ≈ cyy + shift * S_yy atol=1e-14
    end

    @testset "Conformation source includes Liu first-order Hermite moment" begin
        Nx, Ny = 7, 7
        ic, jc = 4, 4
        λ = 5.0
        tau_plus = 1.25
        ωp = 1.0 / tau_plus
        cxx, cxy, cyy = 1.3, -0.2, 0.7
        dudx, dudy = 0.03, -0.04
        dvdx, dvdy = 0.02, -0.01
        u0, v0 = 0.07, -0.03
        divu = dudx + dvdy

        ux = zeros(Float64, Nx, Ny)
        uy = zeros(Float64, Nx, Ny)
        for j in 1:Ny, i in 1:Nx
            x = i - ic
            y = j - jc
            ux[i, j] = u0 + dudx * x + dudy * y
            uy[i, j] = v0 + dvdx * x + dvdy * y
        end
        is_solid = falses(Nx, Ny)
        C_xx = fill(cxx, Nx, Ny)
        C_xy = fill(cxy, Nx, Ny)
        C_yy = fill(cyy, Nx, Ny)
        g_xx = zeros(Float64, Nx, Ny, 9)
        init_conformation_field_2d!(g_xx, C_xx, ux, uy)
        geq0 = copy(g_xx)

        collide_conformation_2d!(g_xx, C_xx, ux, uy, C_xx, C_xy, C_yy,
                                  is_solid, tau_plus, λ; component=1)

        cxv = Int.(Kraken.velocities_x(D2Q9()))
        cyv = Int.(Kraken.velocities_y(D2Q9()))
        Δmass = sum(g_xx[ic, jc, q] - geq0[ic, jc, q] for q in 1:9)
        Δmx = sum(cxv[q] * (g_xx[ic, jc, q] - geq0[ic, jc, q]) for q in 1:9)
        Δmy = sum(cyv[q] * (g_xx[ic, jc, q] - geq0[ic, jc, q]) for q in 1:9)
        S_xx = -(cxx - 1.0) / λ + 2.0 * (cxx * dudx + cxy * dudy) + cxx * divu

        @test Δmass ≈ S_xx atol=1e-14
        @test Δmx ≈ (1.0 - 0.5 * ωp) * u0 * S_xx atol=1e-14
        @test Δmy ≈ (1.0 - 0.5 * ωp) * v0 * S_xx atol=1e-14
    end

    @testset "Conformation source uses one-sided velocity gradient at walls" begin
        Nx, Ny = 7, 7
        i, j = 4, 1
        λ = 100.0
        tau_plus = 1.0
        γdot = 0.02

        ux = zeros(Float64, Nx, Ny)
        uy = zeros(Float64, Nx, Ny)
        for jj in 1:Ny, ii in 1:Nx
            ux[ii, jj] = γdot * (jj - 0.5)
        end
        is_solid = falses(Nx, Ny)
        C_xx = ones(Float64, Nx, Ny)
        C_xy = zeros(Float64, Nx, Ny)
        C_yy = ones(Float64, Nx, Ny)
        g_xy = zeros(Float64, Nx, Ny, 9)
        init_conformation_field_2d!(g_xy, C_xy, ux, uy)

        collide_conformation_2d!(g_xy, C_xy, ux, uy, C_xx, C_xy, C_yy,
                                  is_solid, tau_plus, λ; component=2)

        C_xy_new = similar(C_xy)
        compute_conformation_macro_2d!(C_xy_new, g_xy)

        @test C_xy_new[i, j] ≈ γdot atol=1e-14
    end

    @testset "Regularized conformation collision has same source moments at equilibrium" begin
        Nx, Ny = 7, 7
        ic, jc = 4, 4
        λ = 5.0
        tau_plus = 1.25
        ωp = 1.0 / tau_plus
        cxx, cxy, cyy = 1.3, -0.2, 0.7
        dudx, dudy = 0.03, -0.04
        dvdx, dvdy = 0.02, -0.01
        u0, v0 = 0.07, -0.03
        divu = dudx + dvdy

        ux = zeros(Float64, Nx, Ny)
        uy = zeros(Float64, Nx, Ny)
        for j in 1:Ny, i in 1:Nx
            x = i - ic
            y = j - jc
            ux[i, j] = u0 + dudx * x + dudy * y
            uy[i, j] = v0 + dvdx * x + dvdy * y
        end
        is_solid = falses(Nx, Ny)
        C_xx = fill(cxx, Nx, Ny)
        C_xy = fill(cxy, Nx, Ny)
        C_yy = fill(cyy, Nx, Ny)
        g_xx = zeros(Float64, Nx, Ny, 9)
        init_conformation_field_2d!(g_xx, C_xx, ux, uy)
        geq0 = copy(g_xx)

        collide_conformation_regularized_2d!(g_xx, C_xx, ux, uy, C_xx, C_xy, C_yy,
                                              is_solid, tau_plus, λ; component=1)

        cxv = Int.(Kraken.velocities_x(D2Q9()))
        cyv = Int.(Kraken.velocities_y(D2Q9()))
        Δmass = sum(g_xx[ic, jc, q] - geq0[ic, jc, q] for q in 1:9)
        Δmx = sum(cxv[q] * (g_xx[ic, jc, q] - geq0[ic, jc, q]) for q in 1:9)
        Δmy = sum(cyv[q] * (g_xx[ic, jc, q] - geq0[ic, jc, q]) for q in 1:9)
        S_xx = -(cxx - 1.0) / λ + 2.0 * (cxx * dudx + cxy * dudy) + cxx * divu

        @test Δmass ≈ S_xx atol=1e-14
        @test Δmx ≈ (1.0 - 0.5 * ωp) * u0 * S_xx atol=1e-14
        @test Δmy ≈ (1.0 - 0.5 * ωp) * v0 * S_xx atol=1e-14
    end

    @testset "Liu Eq26 adds finite-difference Fe time derivative" begin
        Nx, Ny = 7, 7
        ic, jc = 4, 4
        λ = 5.0
        tau_plus = 1.25
        ωp = 1.0 / tau_plus
        coeff = 1.0 - 0.5 * ωp
        cxx, cxy, cyy = 1.3, -0.2, 0.7
        dudx, dudy = 0.03, -0.04
        dvdx, dvdy = 0.02, -0.01
        u0, v0 = 0.07, -0.03
        divu = dudx + dvdy

        ux = zeros(Float64, Nx, Ny)
        uy = zeros(Float64, Nx, Ny)
        for j in 1:Ny, i in 1:Nx
            x = i - ic
            y = j - jc
            ux[i, j] = u0 + dudx * x + dudy * y
            uy[i, j] = v0 + dvdx * x + dvdy * y
        end
        ρ = ones(Float64, Nx, Ny)
        is_solid = falses(Nx, Ny)
        C_xx = fill(cxx, Nx, Ny)
        C_xy = fill(cxy, Nx, Ny)
        C_yy = fill(cyy, Nx, Ny)
        g_xx = zeros(Float64, Nx, Ny, 9)
        Fe_prev = zeros(Float64, Nx, Ny, 9)
        init_conformation_field_2d!(g_xx, C_xx, ux, uy)
        geq0 = copy(g_xx)

        collide_conformation_liu_eq26_2d!(g_xx, Fe_prev, C_xx, ux, uy, ρ,
                                           C_xx, C_xy, C_yy, is_solid,
                                           tau_plus, λ; component=1)

        cxv = Int.(Kraken.velocities_x(D2Q9()))
        cyv = Int.(Kraken.velocities_y(D2Q9()))
        Δmass = sum(g_xx[ic, jc, q] - geq0[ic, jc, q] for q in 1:9)
        Δmx = sum(cxv[q] * (g_xx[ic, jc, q] - geq0[ic, jc, q]) for q in 1:9)
        Δmy = sum(cyv[q] * (g_xx[ic, jc, q] - geq0[ic, jc, q]) for q in 1:9)
        Fe_mass = sum(Fe_prev[ic, jc, q] for q in 1:9)
        S_xx = -(cxx - 1.0) / λ + 2.0 * (cxx * dudx + cxy * dudy) + cxx * divu

        @test Fe_mass ≈ S_xx atol=1e-14
        @test Δmass ≈ 1.5 * S_xx atol=1e-14
        @test Δmx ≈ 1.5 * coeff * u0 * S_xx atol=1e-14
        @test Δmy ≈ 1.5 * coeff * v0 * S_xx atol=1e-14
    end

    @testset "Liu Eq26 Ge density-gradient correction has expected moments" begin
        Nx, Ny = 7, 7
        ic, jc = 4, 4
        λ = 5.0
        tau_plus = 1.25
        ωp = 1.0 / tau_plus
        coeff = 1.0 - 0.5 * ωp
        drhodx = 0.02

        ux = zeros(Float64, Nx, Ny)
        uy = zeros(Float64, Nx, Ny)
        ρ = ones(Float64, Nx, Ny)
        for j in 1:Ny, i in 1:Nx
            ρ[i, j] = 1.0 + drhodx * (i - ic)
        end
        is_solid = falses(Nx, Ny)
        C_xx = ones(Float64, Nx, Ny)
        C_xy = zeros(Float64, Nx, Ny)
        C_yy = ones(Float64, Nx, Ny)
        g_xx = zeros(Float64, Nx, Ny, 9)
        Fe_prev = zeros(Float64, Nx, Ny, 9)
        init_conformation_field_2d!(g_xx, C_xx, ux, uy)
        geq0 = copy(g_xx)

        collide_conformation_liu_eq26_2d!(g_xx, Fe_prev, C_xx, ux, uy, ρ,
                                           C_xx, C_xy, C_yy, is_solid,
                                           tau_plus, λ; component=1)

        cxv = Int.(Kraken.velocities_x(D2Q9()))
        cyv = Int.(Kraken.velocities_y(D2Q9()))
        Δmass = sum(g_xx[ic, jc, q] - geq0[ic, jc, q] for q in 1:9)
        Δmx = sum(cxv[q] * (g_xx[ic, jc, q] - geq0[ic, jc, q]) for q in 1:9)
        Δmy = sum(cyv[q] * (g_xx[ic, jc, q] - geq0[ic, jc, q]) for q in 1:9)

        @test Δmass ≈ 0.0 atol=1e-14
        @test Δmx ≈ -coeff * (1/3) * drhodx atol=1e-14
        @test Δmy ≈ 0.0 atol=1e-14
    end

    @testset "Liu Eq26 Bneq source correction changes low moments predictably" begin
        Nx, Ny = 7, 7
        ic, jc = 4, 4
        λ = 5.0
        tau_plus = 1.25
        ωp = 1.0 / tau_plus
        r1 = 1.0 - ωp
        coeff = 1.0 - 0.5 * ωp
        bscale = -0.5
        cxx, cxy, cyy = 1.3, -0.2, 0.7
        dudx, dudy = 0.03, -0.04
        dvdx, dvdy = 0.02, -0.01
        u0, v0 = 0.07, -0.03
        divu = dudx + dvdy

        ux = zeros(Float64, Nx, Ny)
        uy = zeros(Float64, Nx, Ny)
        for j in 1:Ny, i in 1:Nx
            x = i - ic
            y = j - jc
            ux[i, j] = u0 + dudx * x + dudy * y
            uy[i, j] = v0 + dvdx * x + dvdy * y
        end
        ρ = ones(Float64, Nx, Ny)
        is_solid = falses(Nx, Ny)
        C_xx = fill(cxx, Nx, Ny)
        C_xy = fill(cxy, Nx, Ny)
        C_yy = fill(cyy, Nx, Ny)
        g_xx = zeros(Float64, Nx, Ny, 9)
        Fe_prev = zeros(Float64, Nx, Ny, 9)
        init_conformation_field_2d!(g_xx, C_xx, ux, uy)
        geq0 = copy(g_xx)

        collide_conformation_liu_eq26_2d!(g_xx, Fe_prev, C_xx, ux, uy, ρ,
                                           C_xx, C_xy, C_yy, is_solid,
                                           tau_plus, λ; bneq_source_scale=bscale,
                                           component=1)

        cxv = Int.(Kraken.velocities_x(D2Q9()))
        cyv = Int.(Kraken.velocities_y(D2Q9()))
        Δmass = sum(g_xx[ic, jc, q] - geq0[ic, jc, q] for q in 1:9)
        Δmx = sum(cxv[q] * (g_xx[ic, jc, q] - geq0[ic, jc, q]) for q in 1:9)
        Δmy = sum(cyv[q] * (g_xx[ic, jc, q] - geq0[ic, jc, q]) for q in 1:9)
        S_xx = -(cxx - 1.0) / λ + 2.0 * (cxx * dudx + cxy * dudy) + cxx * divu

        expected_scale = 1.5 + r1 * bscale
        @test Δmass ≈ expected_scale * S_xx atol=1e-14
        @test Δmx ≈ expected_scale * coeff * u0 * S_xx atol=1e-14
        @test Δmy ≈ expected_scale * coeff * v0 * S_xx atol=1e-14
    end

    @testset "Liu Eq26 Bneq mass scale suppresses zeroth reconstruction" begin
        Nx, Ny = 7, 7
        ic, jc = 4, 4
        λ = 5.0
        tau_plus = 1.25
        ωp = 1.0 / tau_plus
        r1 = 1.0 - ωp
        coeff = 1.0 - 0.5 * ωp
        bscale = 0.5
        cxx, cxy, cyy = 1.3, -0.2, 0.7
        dudx, dudy = 0.03, -0.04
        dvdx, dvdy = 0.02, -0.01
        u0, v0 = 0.07, -0.03
        divu = dudx + dvdy

        ux = zeros(Float64, Nx, Ny)
        uy = zeros(Float64, Nx, Ny)
        for j in 1:Ny, i in 1:Nx
            x = i - ic
            y = j - jc
            ux[i, j] = u0 + dudx * x + dudy * y
            uy[i, j] = v0 + dvdx * x + dvdy * y
        end
        ρ = ones(Float64, Nx, Ny)
        is_solid = falses(Nx, Ny)
        C_xx = fill(cxx, Nx, Ny)
        C_xy = fill(cxy, Nx, Ny)
        C_yy = fill(cyy, Nx, Ny)
        g_xx = zeros(Float64, Nx, Ny, 9)
        Fe_prev = zeros(Float64, Nx, Ny, 9)
        init_conformation_field_2d!(g_xx, C_xx, ux, uy)
        geq0 = copy(g_xx)

        collide_conformation_liu_eq26_2d!(g_xx, Fe_prev, C_xx, ux, uy, ρ,
                                           C_xx, C_xy, C_yy, is_solid,
                                           tau_plus, λ; bneq_source_scale=bscale,
                                           bneq_mass_scale=0.0, component=1)

        cxv = Int.(Kraken.velocities_x(D2Q9()))
        cyv = Int.(Kraken.velocities_y(D2Q9()))
        Δmass = sum(g_xx[ic, jc, q] - geq0[ic, jc, q] for q in 1:9)
        Δmx = sum(cxv[q] * (g_xx[ic, jc, q] - geq0[ic, jc, q]) for q in 1:9)
        Δmy = sum(cyv[q] * (g_xx[ic, jc, q] - geq0[ic, jc, q]) for q in 1:9)
        S_xx = -(cxx - 1.0) / λ + 2.0 * (cxx * dudx + cxy * dudy) + cxx * divu

        expected_first_scale = 1.5 + r1 * bscale
        @test Δmass ≈ 1.5 * S_xx atol=1e-14
        @test Δmx ≈ expected_first_scale * coeff * u0 * S_xx atol=1e-14
        @test Δmy ≈ expected_first_scale * coeff * v0 * S_xx atol=1e-14
    end

    @testset "Liu Eq26 raw second Bneq moment changes trace response" begin
        Nx, Ny = 7, 7
        ic, jc = 4, 4
        λ = 5.0
        tau_plus = 1.25
        magic = 0.25
        tau_minus = magic / (tau_plus - 0.5) + 0.5
        r2 = 1.0 - 1.0 / tau_minus
        δ = 0.03

        ux = zeros(Float64, Nx, Ny)
        uy = zeros(Float64, Nx, Ny)
        ρ = ones(Float64, Nx, Ny)
        is_solid = falses(Nx, Ny)
        C_xx = ones(Float64, Nx, Ny)
        C_xy = zeros(Float64, Nx, Ny)
        C_yy = ones(Float64, Nx, Ny)

        g_h = zeros(Float64, Nx, Ny, 9)
        g_r = zeros(Float64, Nx, Ny, 9)
        Fe_prev_h = zeros(Float64, Nx, Ny, 9)
        Fe_prev_r = zeros(Float64, Nx, Ny, 9)
        init_conformation_field_2d!(g_h, C_xx, ux, uy)
        copyto!(g_r, g_h)
        geq0 = copy(g_h)
        g_h[ic, jc, 1] += δ
        g_r[ic, jc, 1] += δ

        collide_conformation_liu_eq26_2d!(g_h, Fe_prev_h, C_xx, ux, uy, ρ,
                                           C_xx, C_xy, C_yy, is_solid,
                                           tau_plus, λ; magic, component=1)
        collide_conformation_liu_eq26_2d!(g_r, Fe_prev_r, C_xx, ux, uy, ρ,
                                           C_xx, C_xy, C_yy, is_solid,
                                           tau_plus, λ; magic,
                                           bneq_second_moment_raw=true,
                                           component=1)

        cxv = Int.(Kraken.velocities_x(D2Q9()))
        Mxx_h = sum(cxv[q]^2 * (g_h[ic, jc, q] - geq0[ic, jc, q]) for q in 1:9)
        Mxx_r = sum(cxv[q]^2 * (g_r[ic, jc, q] - geq0[ic, jc, q]) for q in 1:9)

        @test Mxx_r - Mxx_h ≈ r2 * δ / 3 atol=1e-14
    end

    @testset "Log-conformation conservative source includes Ψ div(u)" begin
        Nx, Ny = 7, 7
        ic, jc = 4, 4
        λ = 6.0
        tau_plus = 1.0
        ψxx, ψxy, ψyy = 0.2, 0.0, -0.1
        dudx, dudy = 0.03, 0.0
        dvdx, dvdy = 0.0, -0.01
        divu = dudx + dvdy

        ux = zeros(Float64, Nx, Ny)
        uy = zeros(Float64, Nx, Ny)
        for j in 1:Ny, i in 1:Nx
            x = i - ic
            y = j - jc
            ux[i, j] = dudx * x + dudy * y
            uy[i, j] = dvdx * x + dvdy * y
        end
        is_solid = falses(Nx, Ny)
        Ψ_xx = fill(ψxx, Nx, Ny)
        Ψ_xy = fill(ψxy, Nx, Ny)
        Ψ_yy = fill(ψyy, Nx, Ny)
        g_xx = zeros(Float64, Nx, Ny, 9)
        g_xy = zeros(Float64, Nx, Ny, 9)
        g_yy = zeros(Float64, Nx, Ny, 9)
        init_conformation_field_2d!(g_xx, Ψ_xx, ux, uy)
        init_conformation_field_2d!(g_xy, Ψ_xy, ux, uy)
        init_conformation_field_2d!(g_yy, Ψ_yy, ux, uy)

        collide_logconf_2d!(g_xx, Ψ_xx, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy,
                            is_solid, tau_plus, λ; component=1)
        collide_logconf_2d!(g_xy, Ψ_xy, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy,
                            is_solid, tau_plus, λ; component=2)
        collide_logconf_2d!(g_yy, Ψ_yy, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy,
                            is_solid, tau_plus, λ; component=3)

        Ψ_xx_new = similar(Ψ_xx)
        Ψ_xy_new = similar(Ψ_xy)
        Ψ_yy_new = similar(Ψ_yy)
        compute_conformation_macro_2d!(Ψ_xx_new, g_xx)
        compute_conformation_macro_2d!(Ψ_xy_new, g_xy)
        compute_conformation_macro_2d!(Ψ_yy_new, g_yy)

        S_xx = 2.0 * dudx - (1.0 - exp(-ψxx)) / λ + ψxx * divu
        S_xy = 0.0
        S_yy = 2.0 * dvdy - (1.0 - exp(-ψyy)) / λ + ψyy * divu

        @test Ψ_xx_new[ic, jc] ≈ ψxx + S_xx atol=1e-14
        @test Ψ_xy_new[ic, jc] ≈ ψxy + S_xy atol=1e-14
        @test Ψ_yy_new[ic, jc] ≈ ψyy + S_yy atol=1e-14
    end

    @testset "Log-conformation gradient stencil collision matches bulk stencil" begin
        Nx, Ny = 7, 7
        ic, jc = 4, 4
        λ = 6.0
        tau_plus = 1.0
        magic = 1e-6
        dudx, dudy = 0.02, -0.03
        dvdx, dvdy = 0.04, -0.01
        ux = zeros(Float64, Nx, Ny)
        uy = zeros(Float64, Nx, Ny)
        for j in 1:Ny, i in 1:Nx
            x = i - ic
            y = j - jc
            ux[i, j] = dudx * x + dudy * y
            uy[i, j] = dvdx * x + dvdy * y
        end
        is_solid = falses(Nx, Ny)
        q_wall = zeros(Float64, Nx, Ny, 9)
        uw = zeros(Float64, Nx, Ny, 9)
        stencils = Kraken.precompute_conformation_gradient_stencils_2d(
            is_solid, q_wall; mode=:embedded_axis, max_terms=4)
        Ψ_xx = fill(0.2, Nx, Ny)
        Ψ_xy = fill(-0.03, Nx, Ny)
        Ψ_yy = fill(-0.1, Nx, Ny)

        for (component, Ψ_field) in ((1, Ψ_xx), (2, Ψ_xy), (3, Ψ_yy))
            g_ref = zeros(Float64, Nx, Ny, 9)
            g_stn = zeros(Float64, Nx, Ny, 9)
            init_conformation_field_2d!(g_ref, Ψ_field, ux, uy)
            copyto!(g_stn, g_ref)

            collide_logconf_2d!(g_ref, Ψ_field, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy,
                                is_solid, tau_plus, λ; magic, component)
            collide_logconf_2d_with_gradient_stencils!(
                g_stn, Ψ_field, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid,
                uw, uw, stencils, tau_plus, λ; magic, component)

            @test g_stn[ic, jc, :] ≈ g_ref[ic, jc, :] atol=1e-14
        end
    end

    @testset "CNEBB rest population preserves φ exactly" begin
        Nx, Ny = 3, 3
        i, j = 2, 2
        g_post = zeros(Float64, Nx, Ny, 9)
        g_pre = zeros(Float64, Nx, Ny, 9)
        for q in 1:9
            g_post[i, j, q] = 0.1 * q
            g_pre[i, j, q] = 1.0 + 0.1 * q
        end
        is_solid = falses(Nx, Ny)
        is_solid[i + 1, j] = true
        C = zeros(Float64, Nx, Ny)

        expected_φ = g_post[i, j, 1] + g_pre[i, j, 2] +
                     sum(g_post[i, j, q] for q in (2, 3, 5, 6, 7, 8, 9))

        apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, C)

        @test C[i, j] ≈ expected_φ atol=1e-14
        @test sum(g_post[i, j, q] for q in 1:9) ≈ expected_φ atol=1e-14
        @test g_post[i, j, 4] ≈ 0.2 atol=1e-14
        @test g_post[i, j, 1] ≈ expected_φ - sum(g_post[i, j, q] for q in 2:9) atol=1e-14
    end

    @testset "CNEBB single-link Eq39 uses the imposed local velocity" begin
        Nx, Ny = 5, 5
        i, j = 3, 3
        cxv = Int.(Kraken.velocities_x(D2Q9()))
        cyv = Int.(Kraken.velocities_y(D2Q9()))
        opp = Kraken.opposite(D2Q9())
        ub, vb = 0.07, -0.03

        for q_missing in 2:9
            si = i - cxv[q_missing]
            sj = j - cyv[q_missing]
            @test 1 <= si <= Nx
            @test 1 <= sj <= Ny

            g_post = zeros(Float64, Nx, Ny, 9)
            g_pre = zeros(Float64, Nx, Ny, 9)
            for q in 1:9
                g_post[i, j, q] = 0.05 + 0.01q + 0.001q_missing
                g_pre[i, j, q] = 0.13 + 0.02q + 0.003q_missing
            end
            g_post_before = copy(g_post)

            is_solid = falses(Nx, Ny)
            is_solid[si, sj] = true
            C = fill(-1.0, Nx, Ny)
            ux = zeros(Float64, Nx, Ny)
            uy = zeros(Float64, Nx, Ny)
            ux[i, j] = ub
            uy[i, j] = vb

            expected_φ = g_post_before[i, j, 1]
            for q in 2:9
                src_solid = (i - cxv[q], j - cyv[q]) == (si, sj)
                expected_φ += src_solid ? g_pre[i, j, opp[q]] : g_post_before[i, j, q]
            end
            expected_unknown = Kraken.equilibrium(D2Q9(), expected_φ, ub, vb, q_missing) +
                               (g_post_before[i, j, opp[q_missing]] -
                                Kraken.equilibrium(D2Q9(), expected_φ, ub, vb, opp[q_missing]))

            apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, C, ux, uy)

            @test C[i, j] ≈ expected_φ atol=1e-14
            @test sum(g_post[i, j, q] for q in 1:9) ≈ expected_φ atol=1e-14
            @test g_post[i, j, q_missing] ≈ expected_unknown atol=1e-14
            for q in 2:9
                q == q_missing && continue
                @test g_post[i, j, q] ≈ g_post_before[i, j, q] atol=1e-14
            end
            @test g_post[i, j, 1] ≈ expected_φ - sum(g_post[i, j, q] for q in 2:9) atol=1e-14
        end
    end

    @testset "CNEBB eq-gradient preserves linear equilibrium wall profiles" begin
        Nx, Ny = 8, 6
        C0 = zeros(Float64, Nx, Ny)
        ux = zeros(Float64, Nx, Ny)
        uy = zeros(Float64, Nx, Ny)
        for j in 1:Ny, i in 1:Nx
            C0[i, j] = 1.2 + 0.03i - 0.07j
            ux[i, j] = 0.04 + 0.002j
        end

        g_pre = zeros(Float64, Nx, Ny, 9)
        for q in 1:9, j in 1:Ny, i in 1:Nx
            g_pre[i, j, q] = Kraken.equilibrium(D2Q9(), C0[i, j], ux[i, j], uy[i, j], q)
        end
        g_post = similar(g_pre)
        stream_periodic_x_wall_y_2d!(g_post, g_pre, Nx, Ny)

        is_solid = falses(Nx, Ny)
        C_pre = copy(C0)
        C_eq_gradient = copy(C0)
        g_pre_opp = copy(g_post)
        g_eq_gradient = copy(g_post)

        apply_cnebb_conformation_2d!(g_pre_opp, g_pre, is_solid, C_pre, ux, uy;
                                     phi_mode=:pre_opp)
        apply_cnebb_conformation_2d!(g_eq_gradient, g_pre, is_solid, C_eq_gradient, ux, uy;
                                     phi_mode=:eq_gradient)

        bottom = CartesianIndex.(1:Nx, 1)
        top = CartesianIndex.(1:Nx, Ny)
        @test maximum(abs.(C_eq_gradient[bottom] .- C0[bottom])) < 1e-14
        @test maximum(abs.(C_eq_gradient[top] .- C0[top])) < 1e-14
        @test maximum(abs.(C_pre[bottom] .- C0[bottom])) > 1e-5
        @test maximum(abs.(C_pre[top] .- C0[top])) > 1e-5
    end

    @testset "CNEBB diagnostic φ modes are explicit" begin
        Nx, Ny = 3, 3
        i, j = 2, 2
        g_post0 = zeros(Float64, Nx, Ny, 9)
        g_pre = zeros(Float64, Nx, Ny, 9)
        for q in 1:9
            g_post0[i, j, q] = 0.1 * q
            g_pre[i, j, q] = 1.0 + 0.1 * q
        end
        is_solid = falses(Nx, Ny)
        is_solid[i + 1, j] = true

        g_post = copy(g_post0)
        C = fill(2.5, Nx, Ny)
        expected_post_opp = g_post0[i, j, 1] + g_post0[i, j, 2] +
                            sum(g_post0[i, j, q] for q in (2, 3, 5, 6, 7, 8, 9))
        apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, C;
                                     phi_mode=:post_opp)
        @test C[i, j] ≈ expected_post_opp atol=1e-14
        @test sum(g_post[i, j, q] for q in 1:9) ≈ expected_post_opp atol=1e-14

        g_post = copy(g_post0)
        C = fill(2.5, Nx, Ny)
        apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, C;
                                     phi_mode=:field)
        @test C[i, j] ≈ 2.5 atol=1e-14
        @test sum(g_post[i, j, q] for q in 1:9) ≈ 2.5 atol=1e-14
    end

    @testset "CNEBB also treats domain boundaries as walls" begin
        Nx, Ny = 3, 3
        i, j = 2, 1
        g_post = zeros(Float64, Nx, Ny, 9)
        g_pre = zeros(Float64, Nx, Ny, 9)
        for q in 1:9
            g_post[i, j, q] = 0.1 * q
            g_pre[i, j, q] = 1.0 + 0.1 * q
        end
        is_solid = falses(Nx, Ny)
        C = zeros(Float64, Nx, Ny)

        expected_φ = g_post[i, j, 1] +
                     sum(g_post[i, j, q] for q in (2, 4, 5, 8, 9)) +
                     g_pre[i, j, 5] + g_pre[i, j, 8] + g_pre[i, j, 9]

        apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, C)

        @test C[i, j] ≈ expected_φ atol=1e-14
        @test sum(g_post[i, j, q] for q in 1:9) ≈ expected_φ atol=1e-14
        @test g_post[i, j, 1] ≈ expected_φ - sum(g_post[i, j, q] for q in 2:9) atol=1e-14
    end

    @testset "rheoTool finite-Re cylinder targets are pinned" begin
        newtonian = read(joinpath(@__DIR__, "..", "bench", "rheotool",
                                  "cylinder_newtonian_re1", "RESULTS.md"), String)
        oldroydb = read(joinpath(@__DIR__, "..", "bench", "rheotool",
                                 "cylinder_oldroydb_log_re1_wi01", "RESULTS.md"), String)

        @test occursin("Re_R = 1", newtonian)
        @test occursin("Re_D = 2", newtonian)
        @test occursin("132.362236515", newtonian)
        @test occursin("Re_R = 1", oldroydb)
        @test occursin("Wi_R = 0.1", oldroydb)
        @test occursin("130.428774404", oldroydb)
    end
end
