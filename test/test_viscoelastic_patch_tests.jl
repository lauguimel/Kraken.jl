using Test
using Kraken

@testset "Viscoelastic local patch tests" begin
    CX2 = Int.(Kraken.velocities_x(D2Q9()))
    CY2 = Int.(Kraken.velocities_y(D2Q9()))

    function _fill_equilibrium!(g, φ, ux, uy)
        Nx, Ny = size(φ)
        for q in 1:9, j in 1:Ny, i in 1:Nx
            g[i, j, q] = Kraken.equilibrium(D2Q9(), φ[i, j], ux[i, j], uy[i, j], q)
        end
        return g
    end

    function _rest_f_2d(Nx, Ny)
        f = zeros(Float64, Nx, Ny, 9)
        for q in 1:9, j in 1:Ny, i in 1:Nx
            f[i, j, q] = Kraken.equilibrium(D2Q9(), 1.0, 0.0, 0.0, q)
        end
        return f
    end

    _momentum_x(f, i, j) = sum(CX2[q] * f[i, j, q] for q in 1:9)
    _momentum_y(f, i, j) = sum(CY2[q] * f[i, j, q] for q in 1:9)

    function _couette_cnebb_error(; Nx=8, Ny=8, γ=0.02, λ=5.0)
        is_solid = falses(Nx, Ny)
        is_solid[:, 1] .= true
        is_solid[:, Ny] .= true

        ux = zeros(Float64, Nx, Ny)
        uy = zeros(Float64, Nx, Ny)
        for j in 1:Ny, i in 1:Nx
            ux[i, j] = γ * ((j - 1) - 1.5)
        end

        Wi = λ * γ
        Cxx = fill(1 + 2Wi^2, Nx, Ny)
        Cxy = fill(Wi, Nx, Ny)
        Cyy = fill(1.0, Nx, Ny)

        gxx = _fill_equilibrium!(zeros(Float64, Nx, Ny, 9), Cxx, ux, uy)
        gxy = _fill_equilibrium!(zeros(Float64, Nx, Ny, 9), Cxy, ux, uy)
        gyy = _fill_equilibrium!(zeros(Float64, Nx, Ny, 9), Cyy, ux, uy)

        collide_conformation_2d!(gxx, Cxx, ux, uy, Cxx, Cxy, Cyy, is_solid, 1.0, λ; component=1)
        collide_conformation_2d!(gxy, Cxy, ux, uy, Cxx, Cxy, Cyy, is_solid, 1.0, λ; component=2)
        collide_conformation_2d!(gyy, Cyy, ux, uy, Cxx, Cxy, Cyy, is_solid, 1.0, λ; component=3)

        pre_xx = copy(gxx)
        pre_xy = copy(gxy)
        pre_yy = copy(gyy)
        buf_xx = similar(gxx)
        buf_xy = similar(gxy)
        buf_yy = similar(gyy)
        stream_2d!(buf_xx, gxx, Nx, Ny; sync=true)
        stream_2d!(buf_xy, gxy, Nx, Ny; sync=true)
        stream_2d!(buf_yy, gyy, Nx, Ny; sync=true)

        Cxx_after = copy(Cxx)
        Cxy_after = copy(Cxy)
        Cyy_after = copy(Cyy)
        apply_cnebb_conformation_2d!(buf_xx, pre_xx, is_solid, Cxx_after, ux, uy)
        apply_cnebb_conformation_2d!(buf_xy, pre_xy, is_solid, Cxy_after, ux, uy)
        apply_cnebb_conformation_2d!(buf_yy, pre_yy, is_solid, Cyy_after, ux, uy)

        wall_fluid = [CartesianIndex(i, j) for i in 2:Nx-1 for j in (2, Ny-1)]
        err_xx = maximum(abs.(Cxx_after[wall_fluid] .- (1 + 2Wi^2)))
        err_xy = maximum(abs.(Cxy_after[wall_fluid] .- Wi))
        err_yy = maximum(abs.(Cyy_after[wall_fluid] .- 1.0))
        return max(err_xx, err_xy, err_yy)
    end

    @testset "P1 planar CNEBB preserves stationary Oldroyd-B Couette patch" begin
        @test _couette_cnebb_error() < 1e-14
    end

    function _single_link_qaware_patch(qw; q_missing=4, φ=1.0,
                                       residual_here_out=0.017,
                                       residual_back_out=-0.011,
                                       residual_here_missing=0.023)
        Nx, Ny = 5, 5
        i, j = 3, 3
        opp = Kraken.opposite(D2Q9())
        q_out = opp[q_missing]
        is_solid = falses(Nx, Ny)
        is_solid[i - CX2[q_missing], j - CY2[q_missing]] = true
        q_wall = zeros(Float64, Nx, Ny, 9)
        q_wall[i, j, q_out] = qw

        C0 = fill(φ, Nx, Ny)
        ux = zeros(Float64, Nx, Ny)
        uy = zeros(Float64, Nx, Ny)
        g_pre = _fill_equilibrium!(zeros(Float64, Nx, Ny, 9), C0, ux, uy)
        g_post = _fill_equilibrium!(zeros(Float64, Nx, Ny, 9), C0, ux, uy)

        g_pre[i, j, q_out] += residual_here_out
        g_post[i, j, q_out] += residual_back_out
        g_pre[i, j, q_missing] += residual_here_missing

        C_after = copy(C0)
        apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, q_wall,
                                     C_after, ux, uy; phi_mode=:field)
        return (; g_post, C_after, i, j, q_missing, q_out,
                residual_here_out, residual_back_out, residual_here_missing)
    end

    _ylw_chi(qw, tau_plus=1.0) = qw < 0.5 ? (2qw - 1) / (tau_plus - 2) :
                                             (2qw - 1) / tau_plus

    function _wall_fluxes(g_pre, g_post, is_solid, i, j)
        Nx, Ny = size(is_solid)
        opp = Kraken.opposite(D2Q9())
        outgoing = 0.0
        incoming = 0.0
        missing = Int[]
        valid_nonrest = 0.0
        for q in 2:9
            si = i - CX2[q]
            sj = j - CY2[q]
            src_solid = !(1 <= si <= Nx && 1 <= sj <= Ny) || is_solid[si, sj]
            if src_solid
                push!(missing, q)
                outgoing += g_pre[i, j, opp[q]]
                incoming += g_post[i, j, q]
            else
                valid_nonrest += g_post[i, j, q]
            end
        end
        return (; outgoing, incoming, missing, valid_nonrest)
    end

    @testset "CNEBB dispatch is Yu 2025 strict; CNEBBQAware owns q_wall hybrid" begin
        Nx, Ny = 5, 5
        i, j = 3, 3
        q_missing = 4
        q_out = 2
        is_solid = falses(Nx, Ny)
        is_solid[4, 3] = true
        q_wall = zeros(Float64, Nx, Ny, 9)
        q_wall[i, j, q_out] = 0.3
        C0 = fill(1.0, Nx, Ny)
        ux = zeros(Float64, Nx, Ny)
        uy = zeros(Float64, Nx, Ny)
        g_pre = _fill_equilibrium!(zeros(Float64, Nx, Ny, 9), C0, ux, uy)
        g_post = _fill_equilibrium!(zeros(Float64, Nx, Ny, 9), C0, ux, uy)
        g_pre[i, j, q_out] += 0.037
        g_post[i, j, q_out] -= 0.011
        g_pre[i, j, q_missing] += 0.023

        g_strict_ref = copy(g_post)
        C_strict_ref = copy(C0)
        apply_cnebb_conformation_2d!(g_strict_ref, g_pre, is_solid,
                                     C_strict_ref, ux, uy)

        g_strict = copy(g_post)
        C_strict = copy(C0)
        apply_polymer_wall_bc!(g_strict, g_pre, is_solid, q_wall,
                               C_strict, ux, uy, CNEBB())

        g_qaware_ref = copy(g_post)
        C_qaware_ref = copy(C0)
        apply_cnebb_conformation_2d!(g_qaware_ref, g_pre, is_solid, q_wall,
                                     C_qaware_ref, ux, uy)

        g_qaware = copy(g_post)
        C_qaware = copy(C0)
        apply_polymer_wall_bc!(g_qaware, g_pre, is_solid, q_wall,
                               C_qaware, ux, uy, CNEBBQAware())

        @test g_strict ≈ g_strict_ref atol=1e-14
        @test C_strict ≈ C_strict_ref atol=1e-14
        @test g_qaware ≈ g_qaware_ref atol=1e-14
        @test C_qaware ≈ C_qaware_ref atol=1e-14
        @test !isapprox(g_qaware[i, j, q_missing],
                        g_strict[i, j, q_missing]; atol=1e-14)
    end

    @testset "YLW A/B local conservation follows Yu-Li-Wen equations" begin
        Nx, Ny = 5, 5
        i, j = 3, 3
        opp = Kraken.opposite(D2Q9())
        is_solid = falses(Nx, Ny)
        is_solid[4, 3] = true
        is_solid[3, 4] = true
        q_wall = zeros(Float64, Nx, Ny, 9)
        q_wall[i, j, 2] = 0.3
        q_wall[i, j, 3] = 0.7
        C0 = fill(1.0, Nx, Ny)
        ux = zeros(Float64, Nx, Ny)
        uy = zeros(Float64, Nx, Ny)
        g_pre = _fill_equilibrium!(zeros(Float64, Nx, Ny, 9), C0, ux, uy)
        g_post = _fill_equilibrium!(zeros(Float64, Nx, Ny, 9), C0, ux, uy)
        for q in 1:9
            g_pre[i, j, q] += 0.003q
            g_post[i, j, q] -= 0.002q
        end

        g_a = copy(g_post)
        C_a = copy(C0)
        apply_polymer_wall_bc!(g_a, g_pre, is_solid, q_wall, C_a, ux, uy, YLW_A())
        flux_a = _wall_fluxes(g_pre, g_a, is_solid, i, j)
        target_total_a = g_post[i, j, 1] + flux_a.valid_nonrest + flux_a.outgoing
        @test sum(g_a[i, j, q] for q in 1:9) ≈ target_total_a atol=1e-14
        @test C_a[i, j] ≈ sum(g_a[i, j, q] for q in 1:9) atol=1e-14

        g_b = copy(g_post)
        C_b = copy(C0)
        apply_polymer_wall_bc!(g_b, g_pre, is_solid, q_wall, C_b, ux, uy, YLW_B())
        flux_b = _wall_fluxes(g_pre, g_b, is_solid, i, j)
        @test flux_b.incoming ≈ flux_b.outgoing atol=1e-14
        @test C_b[i, j] ≈ sum(g_b[i, j, q] for q in 1:9) atol=1e-14

        for q in flux_b.missing
            q_out = opp[q]
            qw = q_wall[i, j, q_out]
            χ = _ylw_chi(qw)
            @test χ >= 0
            @test χ <= 1
        end
    end

    @testset "YLW A/B reduce to halfway at q=0.5" begin
        Nx, Ny = 5, 5
        i, j = 3, 3
        q_missing = 4
        q_out = 2
        is_solid = falses(Nx, Ny)
        is_solid[4, 3] = true
        q_wall = zeros(Float64, Nx, Ny, 9)
        q_wall[i, j, q_out] = 0.5
        C0 = fill(1.0, Nx, Ny)
        ux = zeros(Float64, Nx, Ny)
        uy = zeros(Float64, Nx, Ny)
        g_pre = _fill_equilibrium!(zeros(Float64, Nx, Ny, 9), C0, ux, uy)
        g_post = _fill_equilibrium!(zeros(Float64, Nx, Ny, 9), C0, ux, uy)
        g_pre[i, j, q_out] += 0.031
        g_post[i, j, 1] -= 0.017

        for bc in (YLW_A(), YLW_B())
            g = copy(g_post)
            C = copy(C0)
            apply_polymer_wall_bc!(g, g_pre, is_solid, q_wall, C, ux, uy, bc)
            @test g[i, j, q_missing] ≈ g_pre[i, j, q_out] atol=1e-14
            @test g[i, j, 1] ≈ g_post[i, j, 1] atol=1e-14
            @test C[i, j] ≈ sum(g[i, j, q] for q in 1:9) atol=1e-14
        end
    end

    @testset "P2 q-aware CNEBB preserves compatibility and conservation" begin
        Nx, Ny = 5, 5
        i, j = 3, 3
        q_missing = 4
        q_out = 2
        is_solid = falses(Nx, Ny)
        is_solid[4, 3] = true
        q_wall = zeros(Float64, Nx, Ny, 9)
        q_wall[i, j, q_out] = 0.5
        C0 = [1.0 + 0.01i - 0.02j for i in 1:Nx, j in 1:Ny]
        ux = [0.01 + 0.001j for i in 1:Nx, j in 1:Ny]
        uy = zeros(Float64, Nx, Ny)
        g_pre = _fill_equilibrium!(zeros(Float64, Nx, Ny, 9), C0, ux, uy)
        g_post = copy(g_pre)
        for q in 1:9
            g_pre[i, j, q] += 0.001q
            g_post[i, j, q] -= 0.0007q
        end

        g_old = copy(g_post)
        C_old = copy(C0)
        apply_cnebb_conformation_2d!(g_old, g_pre, is_solid, C_old, ux, uy;
                                     phi_mode=:field)

        g_half = copy(g_post)
        C_half = copy(C0)
        apply_cnebb_conformation_2d!(g_half, g_pre, is_solid, q_wall,
                                     C_half, ux, uy; phi_mode=:field)

        @test g_half[i, j, q_missing] ≈ g_old[i, j, q_missing] atol=1e-14
        @test C_half[i, j] ≈ C_old[i, j] atol=1e-14
        @test sum(g_half[i, j, q] for q in 1:9) ≈ C_half[i, j] atol=1e-14

        for qw in (0.3, 0.7)
            out = _single_link_qaware_patch(qw; q_missing=q_missing)
            geq_missing = Kraken.equilibrium(D2Q9(), 1.0, 0.0, 0.0, q_missing)
            expected_residual = if qw <= 0.5
                2qw * out.residual_here_out +
                (1 - 2qw) * out.residual_back_out
            else
                inv_two_q = 1.0 / (2qw)
                inv_two_q * out.residual_here_out +
                (1 - inv_two_q) * out.residual_here_missing
            end

            @test out.g_post[out.i, out.j, q_missing] ≈
                  geq_missing + expected_residual atol=1e-14
            @test out.C_after[out.i, out.j] ≈ 1.0 atol=1e-14
            @test sum(out.g_post[out.i, out.j, q] for q in 1:9) ≈
                  out.C_after[out.i, out.j] atol=1e-14
        end
    end

    @testset "P2 q<0.5 branch is second-order on smooth residual interpolation" begin
        function q03_interpolation_error(N)
            qw = 0.3
            h = 1.0 / N
            exact_residual(x) = sin(1.7x)
            out = _single_link_qaware_patch(qw;
                q_missing=4,
                residual_here_out=exact_residual(0.0),
                residual_back_out=exact_residual(-h),
                residual_here_missing=0.0)
            geq_missing = Kraken.equilibrium(D2Q9(), 1.0, 0.0, 0.0,
                                             out.q_missing)
            numerical = out.g_post[out.i, out.j, out.q_missing] - geq_missing
            exact = exact_residual((2qw - 1) * h)
            return abs(numerical - exact)
        end

        err32 = q03_interpolation_error(32)
        err64 = q03_interpolation_error(64)
        err128 = q03_interpolation_error(128)
        @test err64 < 0.26 * err32
        @test err128 < 0.26 * err64
    end

    @testset "P2 interior reconstruction ignores polluted wall cell" begin
        Nx, Ny = 7, 5
        i, j = 5, 3
        q = 2
        q_w = 0.3

        function patch_field(f)
            a = zeros(Float64, Nx, Ny)
            for ii in 1:Nx, jj in 1:Ny
                x = ii - i
                a[ii, jj] = f(x)
            end
            a[i, j] = 999.0
            return a
        end

        constant = patch_field(x -> 2.5)
        linear = patch_field(x -> 1.2 - 0.4x)
        quadratic = patch_field(x -> 0.7 + 0.2x + 0.05x^2)

        @test Kraken.reconstruct_wall_link_value_2d(constant, i, j, q, q_w;
                                                    location=:cell, order=1) ≈ 2.5 atol=1e-14
        @test Kraken.reconstruct_wall_link_value_2d(constant, i, j, q, q_w;
                                                    location=:cut, order=2) ≈ 2.5 atol=1e-14

        @test Kraken.reconstruct_wall_link_value_2d(linear, i, j, q, q_w;
                                                    location=:cell, order=1) ≈ 1.2 atol=1e-14
        @test Kraken.reconstruct_wall_link_value_2d(linear, i, j, q, q_w;
                                                    location=:cut, order=1) ≈ 1.2 - 0.4q_w atol=1e-14

        @test Kraken.reconstruct_wall_link_value_2d(quadratic, i, j, q, q_w;
                                                    location=:cell, order=2) ≈ 0.7 atol=1e-14
        @test Kraken.reconstruct_wall_link_value_2d(quadratic, i, j, q, q_w;
                                                    location=:cut, order=2) ≈
              0.7 + 0.2q_w + 0.05q_w^2 atol=1e-14

        function sinus_error(N; order)
            q_w = 0.3
            h = 1.0 / N
            field = zeros(Float64, Nx, Ny)
            for ii in 1:Nx, jj in 1:Ny
                x = h * (ii - i)
                field[ii, jj] = sin(1.3x)
            end
            field[i, j] = 999.0
            numerical = Kraken.reconstruct_wall_link_value_2d(field, i, j, q, q_w;
                                                              location=:cut, order=order)
            exact = sin(1.3 * h * q_w)
            return abs(numerical - exact)
        end

        err32 = sinus_error(32; order=1)
        err64 = sinus_error(64; order=1)
        err128 = sinus_error(128; order=1)
        @test err64 < 0.26 * err32
        @test err128 < 0.26 * err64

        qerr32 = sinus_error(32; order=2)
        qerr64 = sinus_error(64; order=2)
        qerr128 = sinus_error(128; order=2)
        @test qerr64 < 0.14 * qerr32
        @test qerr128 < 0.14 * qerr64
    end

    @testset "P2 source stress reconstruction builds clean cell-centered tau" begin
        Nx, Ny = 7, 5
        i, j = 5, 3
        q = 2
        is_solid = falses(Nx, Ny)
        is_solid[i + 1, j] = true
        q_wall = zeros(Float64, Nx, Ny, 9)
        q_wall[i, j, q] = 0.3

        tau_xx = [1.2 - 0.4 * (ii - i) for ii in 1:Nx, jj in 1:Ny]
        tau_xy = [0.7 + 0.2 * (ii - i) + 0.05 * (ii - i)^2 for ii in 1:Nx, jj in 1:Ny]
        tau_yy = [2.0 + 0.1 * (jj - j) for ii in 1:Nx, jj in 1:Ny]
        tau_xx[i, j] = 999.0
        tau_xy[i, j] = 999.0
        tau_yy[i, j] = 999.0

        out_xx = similar(tau_xx)
        out_xy = similar(tau_xy)
        out_yy = similar(tau_yy)
        reconstruct_wall_cell_stress_from_interior_2d!(
            out_xx, out_xy, out_yy, tau_xx, tau_xy, tau_yy, q_wall, is_solid;
            order=2)

        @test out_xx[i, j] ≈ 1.2 atol=1e-14
        @test out_xy[i, j] ≈ 0.7 atol=1e-14
        @test out_yy[i, j] ≈ 2.0 atol=1e-14
        @test out_xx[i - 1, j] ≈ tau_xx[i - 1, j] atol=1e-14
        @test out_xy[i - 1, j] ≈ tau_xy[i - 1, j] atol=1e-14
        @test out_yy[i - 1, j] ≈ tau_yy[i - 1, j] atol=1e-14
    end

    @testset "P2 source stress reconstruction on square obstacle cut-links" begin
        geom = square_obstacle_channel_geometry_2d(; H=28, side=6, L_up=3, L_down=4)
        Nx, Ny = geom.Nx, geom.Ny
        cx0 = geom.i_step + (geom.H_ref - 1) / 2
        cy0 = (Ny - 1) / 2
        exact_xx = [
            1.0 + 0.03 * ((i - 1) - cx0) - 0.02 * ((j - 1) - cy0)
            for i in 1:Nx, j in 1:Ny
        ]
        exact_xy = [
            0.4 + 0.02 * ((i - 1) - cx0) +
            0.01 * ((j - 1) - cy0) +
            0.005 * ((i - 1) - cx0)^2
            for i in 1:Nx, j in 1:Ny
        ]
        exact_yy = [
            1.7 - 0.015 * ((i - 1) - cx0) +
            0.004 * ((j - 1) - cy0)^2
            for i in 1:Nx, j in 1:Ny
        ]
        tau_xx = copy(exact_xx)
        tau_xy = copy(exact_xy)
        tau_yy = copy(exact_yy)

        cut_cells = Tuple{Int,Int}[]
        for j in 1:Ny, i in 1:Nx
            obstacle_link = false
            for q in 2:9
                geom.q_wall[i, j, q] > 0.0 || continue
                ni = i + Int(Kraken.velocities_x(D2Q9())[q])
                nj = j + Int(Kraken.velocities_y(D2Q9())[q])
                if 1 <= ni <= Nx && 1 <= nj <= Ny && geom.is_solid[ni, nj]
                    obstacle_link = true
                    break
                end
            end
            if obstacle_link
                push!(cut_cells, (i, j))
                tau_xx[i, j] = 999.0
                tau_xy[i, j] = -777.0
                tau_yy[i, j] = 555.0
            end
        end
        @test !isempty(cut_cells)

        out_xx = similar(tau_xx)
        out_xy = similar(tau_xy)
        out_yy = similar(tau_yy)
        reconstruct_wall_cell_stress_from_interior_2d!(
            out_xx, out_xy, out_yy, tau_xx, tau_xy, tau_yy,
            geom.q_wall, geom.is_solid; order=2)

        for (i, j) in cut_cells
            @test out_xx[i, j] ≈ exact_xx[i, j] atol=1e-12
            @test out_xy[i, j] ≈ exact_xy[i, j] atol=1e-12
            @test out_yy[i, j] ≈ exact_yy[i, j] atol=1e-12
        end
        for j in 1:Ny, i in 1:Nx
            geom.is_solid[i, j] && continue
            any(q -> q > 0.0, view(geom.q_wall, i, j, :)) && continue
            @test out_xx[i, j] ≈ exact_xx[i, j] atol=1e-14
            @test out_xy[i, j] ≈ exact_xy[i, j] atol=1e-14
            @test out_yy[i, j] ≈ exact_yy[i, j] atol=1e-14
        end
    end

    @testset "P2 source stress reconstruction on actual cylinder cut-links" begin
        Nx, Ny = 56, 48
        cx, cy, R = 27.31, 23.67, 9.0
        q_wall, is_solid = precompute_q_wall_cylinder(Nx, Ny, cx, cy, R)
        exact_xx = [
            1.0 + 0.03 * ((i - 1) - cx) - 0.02 * ((j - 1) - cy)
            for i in 1:Nx, j in 1:Ny
        ]
        exact_xy = [
            0.4 + 0.02 * ((i - 1) - cx) +
            0.01 * ((j - 1) - cy) +
            0.005 * ((i - 1) - cx)^2
            for i in 1:Nx, j in 1:Ny
        ]
        exact_yy = [
            1.7 - 0.015 * ((i - 1) - cx) +
            0.004 * ((j - 1) - cy)^2
            for i in 1:Nx, j in 1:Ny
        ]
        tau_xx = copy(exact_xx)
        tau_xy = copy(exact_xy)
        tau_yy = copy(exact_yy)

        cut_cells = Tuple{Int,Int}[]
        for j in 1:Ny, i in 1:Nx
            if any(q -> q > 0.0, view(q_wall, i, j, :))
                push!(cut_cells, (i, j))
                tau_xx[i, j] = 999.0
                tau_xy[i, j] = -777.0
                tau_yy[i, j] = 555.0
            end
        end
        @test !isempty(cut_cells)

        out_xx = similar(tau_xx)
        out_xy = similar(tau_xy)
        out_yy = similar(tau_yy)
        reconstruct_wall_cell_stress_from_interior_2d!(
            out_xx, out_xy, out_yy, tau_xx, tau_xy, tau_yy, q_wall, is_solid;
            order=2)

        for (i, j) in cut_cells
            @test out_xx[i, j] ≈ exact_xx[i, j] atol=1e-12
            @test out_xy[i, j] ≈ exact_xy[i, j] atol=1e-12
            @test out_yy[i, j] ≈ exact_yy[i, j] atol=1e-12
        end
        for j in 1:Ny, i in 1:Nx
            is_solid[i, j] && continue
            any(q -> q > 0.0, view(q_wall, i, j, :)) && continue
            @test out_xx[i, j] ≈ exact_xx[i, j] atol=1e-14
            @test out_xy[i, j] ≈ exact_xy[i, j] atol=1e-14
            @test out_yy[i, j] ≈ exact_yy[i, j] atol=1e-14
        end
    end

    @testset "P2 curved-wall BCs still corrupt linear node-centered macro" begin
        function linear_macro_patch_error(bc, qw)
            Nx, Ny = 5, 5
            i, j = 3, 3
            q_out = 2
            is_solid = falses(Nx, Ny)
            is_solid[4, 3] = true
            q_wall = zeros(Float64, Nx, Ny, 9)
            q_wall[i, j, q_out] = qw
            C0 = [1.0 + 0.1 * (ii - i) for ii in 1:Nx, jj in 1:Ny]
            ux = zeros(Float64, Nx, Ny)
            uy = zeros(Float64, Nx, Ny)
            g_pre = _fill_equilibrium!(zeros(Float64, Nx, Ny, 9), C0, ux, uy)
            g_post = similar(g_pre)
            stream_2d!(g_post, g_pre, Nx, Ny; sync=true)

            C_after = copy(C0)
            apply_polymer_wall_bc!(g_post, g_pre, is_solid, q_wall,
                                   C_after, ux, uy, bc)
            sum_error = sum(g_post[i, j, q] for q in 1:9) - C_after[i, j]
            return C_after[i, j] - C0[i, j], sum_error
        end

        for bc in (CNEBB(), CNEBBQAware(), YLW_A(), YLW_B()), qw in (0.3, 0.7)
            macro_error, sum_error = linear_macro_patch_error(bc, qw)
            @test abs(sum_error) < 1e-14
            @test abs(macro_error) > 1e-6
        end
    end

    @testset "P3 Hermite stress source gives second-order divergence matrix after streaming" begin
        function source_divergence_error(N, case)
            Nx = Ny = N
            s_plus = 1.0
            kx = 2π / Nx
            ky = 2π / Ny
            f = _rest_f_2d(Nx, Ny)
            streamed = similar(f)
            is_solid = falses(Nx, Ny)
            τxx = zeros(Float64, Nx, Ny)
            τxy = zeros(Float64, Nx, Ny)
            τyy = zeros(Float64, Nx, Ny)

            if case === :txx_x
                τxx .= [sin(kx * (i - 1)) for i in 1:Nx, j in 1:Ny]
            elseif case === :txx_y
                τxx .= [sin(ky * (j - 1)) for i in 1:Nx, j in 1:Ny]
            elseif case === :txy_x
                τxy .= [sin(kx * (i - 1)) for i in 1:Nx, j in 1:Ny]
            elseif case === :txy_y
                τxy .= [sin(ky * (j - 1)) for i in 1:Nx, j in 1:Ny]
            elseif case === :tyy_x
                τyy .= [sin(kx * (i - 1)) for i in 1:Nx, j in 1:Ny]
            elseif case === :tyy_y
                τyy .= [sin(ky * (j - 1)) for i in 1:Nx, j in 1:Ny]
            else
                error("unknown source divergence case $case")
            end

            apply_hermite_source_2d!(f, is_solid, s_plus, τxx, τxy, τyy;
                                     ce_correction=false)
            stream_fully_periodic_2d!(streamed, f, Nx, Ny)

            err = 0.0
            norm = 0.0
            orthogonal = 0.0
            for i in 1:Nx, j in 1:Ny
                expected_x = if case === :txx_x
                    s_plus * kx * cos(kx * (i - 1))
                elseif case === :txy_y
                    s_plus * ky * cos(ky * (j - 1))
                else
                    0.0
                end
                expected_y = if case === :txy_x
                    s_plus * kx * cos(kx * (i - 1))
                elseif case === :tyy_y
                    s_plus * ky * cos(ky * (j - 1))
                else
                    0.0
                end
                mx = _momentum_x(streamed, i, j)
                my = _momentum_y(streamed, i, j)
                err = max(err, abs(mx - expected_x), abs(my - expected_y))
                norm = max(norm, abs(expected_x), abs(expected_y))
                if expected_x == 0.0
                    orthogonal = max(orthogonal, abs(mx))
                end
                if expected_y == 0.0
                    orthogonal = max(orthogonal, abs(my))
                end
            end
            if norm == 0.0
                return orthogonal
            end
            return err / norm
        end

        for case in (:txx_x, :txy_y, :txy_x, :tyy_y)
            err32 = source_divergence_error(32, case)
            err64 = source_divergence_error(64, case)
            err128 = source_divergence_error(128, case)

            @test err64 < 0.26 * err32
            @test err128 < 0.26 * err64
            @test err128 < 5e-4
        end

        for case in (:txx_y, :tyy_x)
            @test source_divergence_error(64, case) < 1e-13
        end
    end
end
