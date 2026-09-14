using Test
using Kraken

@testset "Polymeric drag geometry quadrature" begin
    @testset "cylinder q_wall is analytic for every D2Q9 cut direction" begin
        cxv = (0, 1, 0, -1, 0, 1, -1, -1, 1)
        cyv = (0, 0, 1, 0, -1, 1, 1, -1, -1)

        for (cx, cy, R) in ((43.0, 39.0, 13.0), (43.37, 38.79, 13.0))
            Nx, Ny = 96, 80
            q_wall, is_solid = precompute_q_wall_cylinder(Nx, Ny, cx, cy, R)
            counts = zeros(Int, 9)

            for j in 1:Ny, i in 1:Nx, q in 2:9
                ni = i + cxv[q]
                nj = j + cyv[q]
                neighbour_solid =
                    1 <= ni <= Nx && 1 <= nj <= Ny && is_solid[ni, nj]
                expected_cut = !is_solid[i, j] && neighbour_solid
                has_cut = q_wall[i, j, q] > 0.0
                @test has_cut == expected_cut

                has_cut || continue
                counts[q] += 1
                qw = q_wall[i, j, q]
                @test 0.0 < qw <= 1.0

                xf = Float64(i - 1)
                yf = Float64(j - 1)
                xw = xf + qw * cxv[q]
                yw = yf + qw * cyv[q]
                @test hypot(xw - cx, yw - cy) ≈ R atol=5e-13

                if qw > 1e-8
                    x_fluid = xf + (qw - 1e-8) * cxv[q]
                    y_fluid = yf + (qw - 1e-8) * cyv[q]
                    @test hypot(x_fluid - cx, y_fluid - cy) >= R - 1e-12
                end
                if qw < 1.0 - 1e-8
                    x_solid = xf + (qw + 1e-8) * cxv[q]
                    y_solid = yf + (qw + 1e-8) * cyv[q]
                    @test hypot(x_solid - cx, y_solid - cy) <= R + 1e-12
                end
            end

            for q in 2:9
                @test counts[q] > 0
            end
        end
    end

    @testset "wall-link reconstruction is exact for analytic polynomials" begin
        cxv = (0, 1, 0, -1, 0, 1, -1, -1, 1)
        cyv = (0, 0, 1, 0, -1, 1, 1, -1, -1)
        Nx, Ny = 14, 14
        i0, j0 = 7, 7
        qws = (0.1, 0.3, 0.5, 0.7, 0.9)

        linear = [1.7 + 0.2 * (i - 1) - 0.35 * (j - 1)
                  for i in 1:Nx, j in 1:Ny]
        for q in 2:9, qw in qws, location in (:cell, :cut)
            offset = location === :cell ? 0.0 : qw
            xw = (i0 - 1) + offset * cxv[q]
            yw = (j0 - 1) + offset * cyv[q]
            expected = 1.7 + 0.2 * xw - 0.35 * yw
            got = Kraken.reconstruct_wall_link_value_2d(linear, i0, j0, q, qw;
                                                        location, order=1)
            @test isapprox(got, expected; rtol=0, atol=1e-12)
        end

        for q in 2:9, qw in qws, location in (:cell, :cut)
            cxq, cyq = cxv[q], cyv[q]
            quadratic = [begin
                    s = (i - 1) * cxq + (j - 1) * cyq
                    2.0 + 0.1 * s + 0.03 * s^2
                end for i in 1:Nx, j in 1:Ny]
            offset = location === :cell ? 0.0 : qw
            xw = (i0 - 1) + offset * cxq
            yw = (j0 - 1) + offset * cyq
            sw = xw * cxq + yw * cyq
            expected = 2.0 + 0.1 * sw + 0.03 * sw^2
            got = Kraken.reconstruct_wall_link_value_2d(quadratic, i0, j0, q, qw;
                                                        location, order=2)
            @test isapprox(got, expected; rtol=0, atol=1e-12)
        end
    end

    @testset "solid-mask square traction uses face measure, not diagonal links" begin
        geom = square_obstacle_channel_geometry_2d(; H=28, side=6, L_up=3, L_down=4)
        Nx, Ny = geom.Nx, geom.Ny
        cx = geom.i_step + (geom.H_ref - 1) / 2
        cy = (Ny - 1) / 2
        x = [(i - 1) - cx for i in 1:Nx, j in 1:Ny]
        y = [(j - 1) - cy for i in 1:Nx, j in 1:Ny]
        z = zeros(Float64, Nx, Ny)
        exact = Float64(geom.H_ref^2)

        cases = (
            (txx=x, txy=z, tyy=z, Fx=exact, Fy=0.0),
            (txx=y, txy=z, tyy=z, Fx=0.0, Fy=0.0),
            (txx=z, txy=x, tyy=z, Fx=0.0, Fy=exact),
            (txx=z, txy=y, tyy=z, Fx=exact, Fy=0.0),
            (txx=z, txy=z, tyy=x, Fx=0.0, Fy=0.0),
            (txx=z, txy=z, tyy=y, Fx=0.0, Fy=exact),
        )

        for case in cases
            drag = Kraken.compute_polymeric_drag_2d(
                case.txx, case.txy, case.tyy, geom.is_solid, Nx, Ny;
                extrapolate=true,
            )
            @test isapprox(drag.Fx, case.Fx; rtol=0, atol=1e-12)
            @test isapprox(drag.Fy, case.Fy; rtol=0, atol=1e-12)
        end

        unit = ones(Float64, Nx, Ny)
        drag = Kraken.compute_polymeric_drag_2d(
            unit, z, unit, geom.is_solid, Nx, Ny; extrapolate=true,
        )
        @test abs(drag.Fx) < 1e-12
        @test abs(drag.Fy) < 1e-12
    end

    # Analytic check for a circular boundary:
    #   τ_xx = x - cx, τ_xy = τ_yy = 0
    #   F_x = ∮ τ_xx n_x ds = ∮ R cosθ · cosθ · R dθ = πR²
    #
    # This exercises the q_wall-aware curved-surface quadrature. The previous
    # staircase/link-counting diagnostic overestimated this integral by ~3.1×.
    for R in (20, 40)
        Nx = 30R
        Ny = 4R
        cx = Nx / 4
        cy = Ny / 2
        q_wall, _ = precompute_q_wall_cylinder(Nx, Ny, cx, cy, R)
        tau_xx = zeros(Float64, Nx, Ny)
        tau_xy = zeros(Float64, Nx, Ny)
        tau_yy = zeros(Float64, Nx, Ny)
        for j in 1:Ny, i in 1:Nx
            tau_xx[i, j] = (i - 1) - cx
        end

        drag = Kraken.compute_polymeric_drag_2d(tau_xx, tau_xy, tau_yy,
                                                q_wall, Nx, Ny;
                                                cx=cx, cy=cy, radius=R)
        exact = π * R^2
        ratio = drag.Fx / exact

        @info "Polymeric drag quadrature" R Fx_qwall=drag.Fx Fx_exact=exact ratio
        @test isapprox(drag.Fx, exact; rtol=0.05)
        @test abs(drag.Fy) < 1e-10 * exact
    end

    @testset "linear stress tensor components integrate with correct orientation" begin
        for R in (20, 40)
            Nx = 30R
            Ny = 4R
            cx = Nx / 4
            cy = Ny / 2
            q_wall, _ = precompute_q_wall_cylinder(Nx, Ny, cx, cy, R)
            x = [(i - 1) - cx for i in 1:Nx, j in 1:Ny]
            y = [(j - 1) - cy for i in 1:Nx, j in 1:Ny]
            z = zeros(Float64, Nx, Ny)
            exact = π * R^2
            cases = (
                (txx=x, txy=z, tyy=z, Fx=exact, Fy=0.0),
                (txx=y, txy=z, tyy=z, Fx=0.0, Fy=0.0),
                (txx=z, txy=x, tyy=z, Fx=0.0, Fy=exact),
                (txx=z, txy=y, tyy=z, Fx=exact, Fy=0.0),
                (txx=z, txy=z, tyy=x, Fx=0.0, Fy=0.0),
                (txx=z, txy=z, tyy=y, Fx=0.0, Fy=exact),
            )

            for case in cases
                drag = Kraken.compute_polymeric_drag_2d(
                    case.txx, case.txy, case.tyy, q_wall, Nx, Ny;
                    cx=cx, cy=cy, radius=R)
                @test isapprox(drag.Fx, case.Fx; rtol=1e-12, atol=1e-10 * exact)
                @test isapprox(drag.Fy, case.Fy; rtol=1e-12, atol=1e-10 * exact)
            end
        end
    end
end
