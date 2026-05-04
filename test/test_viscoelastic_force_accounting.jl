using Test
using Kraken

@testset "Viscoelastic force accounting before Cd" begin
    CX2 = Int.(Kraken.velocities_x(D2Q9()))
    CY2 = Int.(Kraken.velocities_y(D2Q9()))

    function rest_f_2d(Nx, Ny)
        f = zeros(Float64, Nx, Ny, 9)
        for j in 1:Ny, i in 1:Nx, q in 1:9
            f[i, j, q] = Kraken.equilibrium(D2Q9(), 1.0, 0.0, 0.0, q)
        end
        return f
    end

    function second_moment_delta(f, f0, i, j)
        mass = sum(f[i, j, q] - f0[i, j, q] for q in 1:9)
        mx = sum(CX2[q] * (f[i, j, q] - f0[i, j, q]) for q in 1:9)
        my = sum(CY2[q] * (f[i, j, q] - f0[i, j, q]) for q in 1:9)
        pxx = sum(CX2[q]^2 * (f[i, j, q] - f0[i, j, q]) for q in 1:9)
        pxy = sum(CX2[q] * CY2[q] * (f[i, j, q] - f0[i, j, q]) for q in 1:9)
        pyy = sum(CY2[q]^2 * (f[i, j, q] - f0[i, j, q]) for q in 1:9)
        return (; mass, mx, my, pxx, pxy, pyy)
    end

    function hermite_delta_f(q, s_plus, txx, txy, tyy; corrected=true)
        cs2 = 1.0 / 3.0
        wq = q == 1 ? 4.0 / 9.0 : (q in (2, 3, 4, 5) ? 1.0 / 9.0 : 1.0 / 36.0)
        prefactor = -s_plus * 9.0 / 2.0
        corrected && (prefactor /= (1.0 - s_plus / 2.0))
        return prefactor * wq *
               ((CX2[q]^2 - cs2) * txx +
                (CY2[q]^2 - cs2) * tyy +
                2.0 * CX2[q] * CY2[q] * txy)
    end

    function square_obstacle_q_wall_only(geom)
        q_wall = zero(geom.q_wall)
        Nx, Ny = geom.Nx, geom.Ny
        for j in 1:Ny, i in 1:Nx, q in 2:9
            geom.q_wall[i, j, q] > 0.0 || continue
            ni = i + CX2[q]
            nj = j + CY2[q]
            if 1 <= ni <= Nx && 1 <= nj <= Ny && geom.is_solid[ni, nj]
                q_wall[i, j, q] = geom.q_wall[i, j, q]
            end
        end
        return q_wall
    end

    @testset "standalone Hermite source has exact bulk moment closure" begin
        Nx, Ny = 4, 4
        s = 1.25
        txx_val, txy_val, tyy_val = 1e-3, -2e-3, 3e-3
        f0 = rest_f_2d(Nx, Ny)
        f = copy(f0)
        is_solid = falses(Nx, Ny)
        txx = fill(txx_val, Nx, Ny)
        txy = fill(txy_val, Nx, Ny)
        tyy = fill(tyy_val, Nx, Ny)

        apply_hermite_source_2d!(f, is_solid, s, txx, txy, tyy)
        m = second_moment_delta(f, f0, 2, 2)
        gain = -s / (1.0 - s / 2.0)

        @test abs(m.mass) < 1e-15
        @test abs(m.mx) < 1e-15
        @test abs(m.my) < 1e-15
        @test isapprox(m.pxx, gain * txx_val; rtol=1e-12, atol=1e-15)
        @test isapprox(m.pxy, gain * txy_val; rtol=1e-12, atol=1e-15)
        @test isapprox(m.pyy, gain * tyy_val; rtol=1e-12, atol=1e-15)
    end

    @testset "standalone source is larger than in-collision Liu source by CE factor" begin
        Nx, Ny = 4, 4
        s = 1.25
        txy_val = 2e-3
        is_solid = falses(Nx, Ny)
        zero_field = zeros(Float64, Nx, Ny)
        txy = fill(txy_val, Nx, Ny)

        f0 = rest_f_2d(Nx, Ny)
        f_standalone = copy(f0)
        f_in_collision = copy(f0)

        apply_hermite_source_2d!(f_standalone, is_solid, s,
                                 zero_field, txy, zero_field)
        collide_viscoelastic_source_2d!(f_in_collision, is_solid, s,
                                        zero_field, txy, zero_field)
        f_standalone_liu = copy(f0)
        apply_hermite_source_2d!(f_standalone_liu, is_solid, s,
                                 zero_field, txy, zero_field;
                                 ce_correction=false)

        m_standalone = second_moment_delta(f_standalone, f0, 2, 2)
        m_in_collision = second_moment_delta(f_in_collision, f0, 2, 2)
        m_standalone_liu = second_moment_delta(f_standalone_liu, f0, 2, 2)

        @test isapprox(m_standalone.pxy, -s * txy_val / (1.0 - s / 2.0);
                       rtol=1e-12)
        @test isapprox(m_in_collision.pxy, -s * txy_val; rtol=1e-12)
        @test isapprox(m_standalone_liu.pxy, m_in_collision.pxy; rtol=1e-12)
        @test isapprox(m_standalone.pxy / m_in_collision.pxy,
                       1.0 / (1.0 - s / 2.0); rtol=1e-12)
    end

    @testset "integrated Hermite collision matches post-collision source in bulk" begin
        Nx, Ny = 5, 4
        ν = 1.0 / 6.0
        source_scale = 1.7
        f_in = rest_f_2d(Nx, Ny)
        f_post = similar(f_in)
        f_integrated = similar(f_in)
        ρ_post = zeros(Float64, Nx, Ny)
        ux_post = zeros(Float64, Nx, Ny)
        uy_post = zeros(Float64, Nx, Ny)
        ρ_integrated = zeros(Float64, Nx, Ny)
        ux_integrated = zeros(Float64, Nx, Ny)
        uy_integrated = zeros(Float64, Nx, Ny)
        is_solid = falses(Nx, Ny)
        q_wall = zeros(Float64, Nx, Ny, 9)
        uw_x = zeros(Float64, Nx, Ny, 9)
        uw_y = zeros(Float64, Nx, Ny, 9)
        tau_xx = [1e-4 * (i + 2j) for i in 1:Nx, j in 1:Ny]
        tau_xy = [-2e-4 * (2i - j) for i in 1:Nx, j in 1:Ny]
        tau_yy = [3e-4 * (i - j) for i in 1:Nx, j in 1:Ny]
        s_plus, = trt_rates(ν)

        fused_trt_libb_v2_step!(f_post, f_in, ρ_post, ux_post, uy_post,
                                is_solid, q_wall, uw_x, uw_y, Nx, Ny, ν)
        apply_hermite_source_2d!(f_post, is_solid, s_plus,
                                 tau_xx, tau_xy, tau_yy;
                                 ce_correction=false)

        fused_trt_libb_v2_hermite_step!(f_integrated, f_in, ρ_integrated,
                                        ux_integrated, uy_integrated,
                                        is_solid, q_wall, uw_x, uw_y,
                                        tau_xx, tau_xy, tau_yy, Nx, Ny, ν;
                                        source_scale=source_scale)

        fused_trt_libb_v2_step!(f_post, f_in, ρ_post, ux_post, uy_post,
                                is_solid, q_wall, uw_x, uw_y, Nx, Ny, ν)
        apply_hermite_source_2d!(f_post, is_solid, s_plus,
                                 source_scale .* tau_xx,
                                 source_scale .* tau_xy,
                                 source_scale .* tau_yy;
                                 ce_correction=false)

        @test isapprox(f_integrated, f_post; rtol=1e-13, atol=1e-15)
        @test isapprox(ρ_integrated, ρ_post; rtol=1e-13, atol=1e-15)
        @test isapprox(ux_integrated, ux_post; rtol=1e-13, atol=1e-15)
        @test isapprox(uy_integrated, uy_post; rtol=1e-13, atol=1e-15)
    end

    @testset "single cut-link post-source MEA increment is exactly source-driven" begin
        Nx, Ny = 5, 5
        s = 1.25
        is_solid = falses(Nx, Ny)
        uw_x = zeros(Float64, Nx, Ny, 9)
        uw_y = zeros(Float64, Nx, Ny, 9)

        stress_cases = (
            (txx=1e-3, txy=0.0, tyy=0.0),
            (txx=0.0, txy=1e-3, tyy=0.0),
            (txx=0.0, txy=0.0, tyy=1e-3),
        )

        for ce_correction in (true, false), q in 2:9, q_w in (0.3, 0.5, 0.7),
            stress in stress_cases
            f = rest_f_2d(Nx, Ny)
            q_wall = zeros(Float64, Nx, Ny, 9)
            q_wall[3, 3, q] = q_w
            drag_before = compute_drag_libb_mei_2d(f, q_wall, uw_x, uw_y, Nx, Ny)
            txx = fill(stress.txx, Nx, Ny)
            txy = fill(stress.txy, Nx, Ny)
            tyy = fill(stress.tyy, Nx, Ny)
            apply_hermite_source_2d!(f, is_solid, s, txx, txy, tyy;
                                     ce_correction)
            drag_after = compute_drag_libb_mei_2d(f, q_wall, uw_x, uw_y, Nx, Ny)

            expected_source = hermite_delta_f(q, s, stress.txx, stress.txy, stress.tyy;
                                             corrected=ce_correction)
            expected_dfx = 2.0 * CX2[q] * expected_source
            expected_dfy = 2.0 * CY2[q] * expected_source
            @test isapprox(drag_after.Fx - drag_before.Fx, expected_dfx;
                           rtol=1e-12, atol=1e-15)
            @test isapprox(drag_after.Fy - drag_before.Fy, expected_dfy;
                           rtol=1e-12, atol=1e-15)
        end
    end

    @testset "square obstacle q=0.5 MEA reduces to halfway over all links" begin
        geom = square_obstacle_channel_geometry_2d(; H=28, side=6, L_up=3, L_down=4)
        Nx, Ny = geom.Nx, geom.Ny
        q_wall = square_obstacle_q_wall_only(geom)
        @test count(>(0), q_wall) > 0

        f = zeros(Float64, Nx, Ny, 9)
        for j in 1:Ny, i in 1:Nx, q in 1:9
            f[i, j, q] = 0.2 + 0.003i - 0.002j + 0.017q + 0.0004q^2
        end
        uw_x = zeros(Float64, Nx, Ny, 9)
        uw_y = zeros(Float64, Nx, Ny, 9)

        expected_fx = 0.0
        expected_fy = 0.0
        for j in 1:Ny, i in 1:Nx, q in 2:9
            q_wall[i, j, q] > 0.0 || continue
            @test q_wall[i, j, q] ≈ 0.5
            expected_fx += 2.0 * CX2[q] * f[i, j, q]
            expected_fy += 2.0 * CY2[q] * f[i, j, q]
        end

        simple = compute_drag_libb_2d(f, q_wall, Nx, Ny)
        mei = compute_drag_libb_mei_2d(f, q_wall, uw_x, uw_y, Nx, Ny)
        @test isapprox(simple.Fx, expected_fx; rtol=1e-13, atol=1e-13)
        @test isapprox(simple.Fy, expected_fy; rtol=1e-13, atol=1e-13)
        @test isapprox(mei.Fx, expected_fx; rtol=1e-13, atol=1e-13)
        @test isapprox(mei.Fy, expected_fy; rtol=1e-13, atol=1e-13)

        f_pair = copy(f)
        for j in 1:Ny, i in 1:Nx, q in 2:9
            q_wall[i, j, q] > 0.0 || continue
            f_pair[i, j, Kraken._D2Q9_OPP[q]] = f_pair[i, j, q]
        end
        pair = compute_drag_libb_postpair_2d(f_pair, q_wall, Nx, Ny)
        @test isapprox(pair.Fx, expected_fx; rtol=1e-13, atol=1e-13)
        @test isapprox(pair.Fy, expected_fy; rtol=1e-13, atol=1e-13)
    end

    @testset "square obstacle q=0.5 post-source MEA increment is source-driven" begin
        geom = square_obstacle_channel_geometry_2d(; H=28, side=6, L_up=3, L_down=4)
        Nx, Ny = geom.Nx, geom.Ny
        q_wall = square_obstacle_q_wall_only(geom)
        uw_x = zeros(Float64, Nx, Ny, 9)
        uw_y = zeros(Float64, Nx, Ny, 9)
        s = 1.25
        txx = [1e-4 * (2i - j) for i in 1:Nx, j in 1:Ny]
        txy = [-2e-4 * (i + j) for i in 1:Nx, j in 1:Ny]
        tyy = [3e-4 * (-i + 2j) for i in 1:Nx, j in 1:Ny]

        for ce_correction in (true, false)
            f = rest_f_2d(Nx, Ny)
            before = compute_drag_libb_mei_2d(f, q_wall, uw_x, uw_y, Nx, Ny)
            apply_hermite_source_2d!(f, geom.is_solid, s, txx, txy, tyy;
                                     ce_correction)
            after = compute_drag_libb_mei_2d(f, q_wall, uw_x, uw_y, Nx, Ny)

            expected_fx = 0.0
            expected_fy = 0.0
            for j in 1:Ny, i in 1:Nx, q in 2:9
                q_wall[i, j, q] > 0.0 || continue
                expected_source = hermite_delta_f(
                    q, s, txx[i, j], txy[i, j], tyy[i, j];
                    corrected=ce_correction,
                )
                expected_fx += 2.0 * CX2[q] * expected_source
                expected_fy += 2.0 * CY2[q] * expected_source
            end

            @test isapprox(after.Fx - before.Fx, expected_fx; rtol=1e-12, atol=1e-14)
            @test isapprox(after.Fy - before.Fy, expected_fy; rtol=1e-12, atol=1e-14)
        end
    end

    @testset "post-LI-BB pair force is a distinct diagnostic" begin
        Nx, Ny = 5, 5
        i, j, q = 3, 3, 2
        qbar = Kraken._D2Q9_OPP[q]
        q_w = 0.75
        f_post_here = 1.2
        f_post_back = 0.8
        f_bar_post_here = 0.4
        reflected = Kraken._libb_branch(q_w, f_post_here, f_post_back,
                                        f_bar_post_here, 0.0)

        f_post = zeros(Float64, Nx, Ny, 9)
        q_wall = zeros(Float64, Nx, Ny, 9)
        uw_x = zeros(Float64, Nx, Ny, 9)
        uw_y = zeros(Float64, Nx, Ny, 9)
        q_wall[i, j, q] = q_w
        f_post[i, j, q] = f_post_here
        f_post[i, j, qbar] = reflected
        f_post[i - 1, j, q] = f_post_back

        pair = compute_drag_libb_postpair_2d(f_post, q_wall, Nx, Ny)
        reconstructed = compute_drag_libb_mei_2d(f_post, q_wall, uw_x, uw_y, Nx, Ny)
        liu_eq63 = compute_drag_libb_liu_eq63_2d(f_post, q_wall, uw_x, uw_y, Nx, Ny)

        @test isapprox(pair.Fx, f_post_here + reflected; rtol=1e-12)
        @test isapprox(liu_eq63.Fx, reconstructed.Fx; rtol=1e-12)
        @test isapprox(liu_eq63.Fy, reconstructed.Fy; rtol=1e-12, atol=1e-15)
        @test reconstructed.Fx > pair.Fx
        @test !isapprox(reconstructed.Fx, pair.Fx; rtol=1e-12)
    end

    @testset "post-source MEA is not a physical surface quadrature for analytic stress" begin
        s = 1.25
        for R in (20, 40)
            Nx, Ny = 30R, 4R
            cx0, cy0 = Nx / 4, Ny / 2
            q_wall, is_solid = precompute_q_wall_cylinder(Nx, Ny, cx0, cy0, R)
            f = rest_f_2d(Nx, Ny)
            f_liu = rest_f_2d(Nx, Ny)
            uw_x = zeros(Float64, Nx, Ny, 9)
            uw_y = zeros(Float64, Nx, Ny, 9)
            drag_before = compute_drag_libb_mei_2d(f, q_wall, uw_x, uw_y, Nx, Ny)
            drag_liu_before = compute_drag_libb_mei_2d(f_liu, q_wall, uw_x, uw_y, Nx, Ny)

            txx = [(i - 1) - cx0 for i in 1:Nx, j in 1:Ny]
            txy = zeros(Float64, Nx, Ny)
            tyy = zeros(Float64, Nx, Ny)
            apply_hermite_source_2d!(f, is_solid, s, txx, txy, tyy)
            apply_hermite_source_2d!(f_liu, is_solid, s, txx, txy, tyy;
                                     ce_correction=false)
            drag_after = compute_drag_libb_mei_2d(f, q_wall, uw_x, uw_y, Nx, Ny)
            drag_liu_after = compute_drag_libb_mei_2d(f_liu, q_wall, uw_x, uw_y, Nx, Ny)

            delta_fx = drag_after.Fx - drag_before.Fx
            delta_liu_fx = drag_liu_after.Fx - drag_liu_before.Fx
            exact_fx = π * R^2
            ratio = delta_fx / exact_fx
            ratio_liu = delta_liu_fx / exact_fx

            @info "post-source MEA analytic stress" R delta_fx delta_liu_fx exact_fx ratio ratio_liu
            @test 3.0 < ratio < 3.8
            @test 1.1 < ratio_liu < 1.4
            @test isapprox(ratio / ratio_liu, 1.0 / (1.0 - s / 2.0); rtol=1e-10)
            @test !isapprox(delta_fx, exact_fx; rtol=0.1)
        end
    end
end
