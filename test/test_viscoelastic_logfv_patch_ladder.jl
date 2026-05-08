using Test
using LinearAlgebra
using KernelAbstractions
using Kraken

const LOGFV_ATOL = 1e-12
const LOGFV_RTOL = 1e-12

_sym2_mat(a, b, d) = [a b; b d]

function _sym2_min_eig(a, b, d)
    return Kraken.logfv_min_eig_sym2_2d(a, b, d)
end

function _sym2_exp(a, b, d)
    return Kraken.logfv_exp_sym2_2d(a, b, d)
end

function _mat2_exp(a, b, c, d)
    return Kraken.logfv_exp_mat2_2d(a, b, c, d)
end

function _sym2_log(a, b, d)
    λmin = _sym2_min_eig(a, b, d)
    λmin > 0 || throw(DomainError(λmin, "symmetric 2x2 log requires SPD input"))
    return Kraken.logfv_log_spd_sym2_2d(a, b, d)
end

function _oldroydb_relax_c(cxx, cxy, cyy, λ, dt)
    return Kraken.logfv_oldroydb_relax_c_2d(cxx, cxy, cyy, λ, dt)
end

function _oldroydb_relax_log(ψxx, ψxy, ψyy, λ, dt)
    return Kraken.logfv_oldroydb_relax_log_2d(ψxx, ψxy, ψyy, λ, dt)
end

function _oldroydb_step_log(ψxx, ψxy, ψyy, dudx, dudy, dvdx, dvdy, λ, dt)
    return Kraken.logfv_oldroydb_step_log_2d(ψxx, ψxy, ψyy, dudx, dudy, dvdx, dvdy, λ, dt)
end

function _oldroydb_source_c(cxx, cxy, cyy, dudx, dudy, dvdx, dvdy, λ)
    return Kraken.logfv_oldroydb_source_c_2d(cxx, cxy, cyy, dudx, dudy, dvdx, dvdy, λ)
end

_oldroydb_simple_shear_stationary(γ, λ) = (1 + 2 * (λ * γ)^2, λ * γ, 1.0)

function _poiseuille_shear(y, height, umax)
    return 4 * umax / height * (1 - 2y / height)
end

function _poiseuille_ux(y, height, umax)
    η = y / height
    return 4 * umax * η * (1 - η)
end

function _couette_ux(y, height, uwall)
    return uwall * y / height
end

function _oldroydb_simple_shear_from_identity(γ, λ, t)
    e = exp(-t / λ)
    return (
        1 + 2 * λ^2 * γ^2 * (1 - e) - 2 * λ * γ^2 * t * e,
        λ * γ * (1 - e),
        1.0,
    )
end

function _upwind_scalar_advective_rhs(φ, ux_face, uy_face, i, j)
    return Kraken.logfv_upwind_scalar_advective_rhs_2d(φ, ux_face, uy_face, i, j)
end

function _upwind_tensor_advective_rhs(ψxx, ψxy, ψyy, ux_face, uy_face, i, j)
    return Kraken.logfv_upwind_tensor_advective_rhs_2d(ψxx, ψxy, ψyy, ux_face, uy_face, i, j)
end

_periodic(i, n) = mod1(i, n)

function _periodic_upwind_scalar_step(φ, u, v, dt)
    Nx, Ny = size(φ)
    out = similar(φ)
    for j in 1:Ny, i in 1:Nx
        im = _periodic(i - 1, Nx)
        ip = _periodic(i + 1, Nx)
        jm = _periodic(j - 1, Ny)
        jp = _periodic(j + 1, Ny)

        dφdx = ifelse(u >= 0, φ[i, j] - φ[im, j], φ[ip, j] - φ[i, j])
        dφdy = ifelse(v >= 0, φ[i, j] - φ[i, jm], φ[i, jp] - φ[i, j])
        out[i, j] = φ[i, j] - dt * (u * dφdx + v * dφdy)
    end
    return out
end

function _assert_sym2_close(actual, expected; atol=LOGFV_ATOL, rtol=LOGFV_RTOL)
    @test actual[1] ≈ expected[1] atol=atol rtol=rtol
    @test actual[2] ≈ expected[2] atol=atol rtol=rtol
    @test actual[3] ≈ expected[3] atol=atol rtol=rtol
end

@testset "Log-FV polymer patch ladder" begin
    @testset "M0 symmetric 2x2 exp/log algebra" begin
        _assert_sym2_close(_sym2_exp(0.0, 0.0, 0.0), (1.0, 0.0, 1.0))
        _assert_sym2_close(_sym2_log(1.0, 0.0, 1.0), (0.0, 0.0, 0.0))

        diagonal_cases = (
            (ψxx=0.3, ψxy=0.0, ψyy=-0.2),
            (ψxx=-1.5, ψxy=0.0, ψyy=0.7),
        )
        for case in diagonal_cases
            c = _sym2_exp(case.ψxx, case.ψxy, case.ψyy)
            _assert_sym2_close(c, (exp(case.ψxx), 0.0, exp(case.ψyy)))
            _assert_sym2_close(_sym2_log(c...), (case.ψxx, case.ψxy, case.ψyy))
        end

        ψ_cases = (
            (0.2, 0.05, -0.1),
            (1.4, -0.31, 0.6),
            (-2.0, 0.4, -0.7),
            (0.01, 1e-12, 0.01 + 1e-12),
            (0.0, 0.0, 0.0),
        )
        for ψ in ψ_cases
            c = _sym2_exp(ψ...)
            @test _sym2_min_eig(c...) > 0
            ψ_round = _sym2_log(c...)
            _assert_sym2_close(ψ_round, ψ; atol=2e-12, rtol=2e-12)

            c_ref = exp(_sym2_mat(ψ...))
            _assert_sym2_close(c, (c_ref[1, 1], c_ref[1, 2], c_ref[2, 2]); atol=2e-12, rtol=2e-12)
        end

        mat_cases = (
            (0.0, 0.4, 0.0, 0.0),
            (0.1, -0.3, 0.2, -0.05),
            (0.0, -0.7, 0.7, 0.0),
            (1e-12, 2e-12, -3e-12, -1e-12),
        )
        for m in mat_cases
            e = _mat2_exp(m...)
            e_ref = exp([m[1] m[2]; m[3] m[4]])
            @test e[1] ≈ e_ref[1, 1] atol=2e-12 rtol=2e-12
            @test e[2] ≈ e_ref[1, 2] atol=2e-12 rtol=2e-12
            @test e[3] ≈ e_ref[2, 1] atol=2e-12 rtol=2e-12
            @test e[4] ≈ e_ref[2, 2] atol=2e-12 rtol=2e-12
        end

        c_cases = (
            (1.0, 0.0, 1.0),
            (3.0, 0.5, 2.0),
            (0.4, -0.08, 0.9),
            (1.0 + 1e-12, 1e-13, 1.0 - 1e-12),
        )
        for c in c_cases
            @test _sym2_min_eig(c...) > 0
            ψ = _sym2_log(c...)
            c_round = _sym2_exp(ψ...)
            _assert_sym2_close(c_round, c; atol=2e-12, rtol=2e-12)
        end

        @test_throws DomainError _sym2_log(1.0, 1.2, 1.0)
    end

    @testset "M1 pure Oldroyd-B relaxation in log variables" begin
        λ_cases = (0.25, 2.0, 100.0)
        dt_cases = (0.0, 0.01, 0.4, 3.0)
        c0_cases = (
            (1.8, 0.0, 0.6),
            (3.0, 0.5, 2.0),
            (0.35, -0.04, 1.2),
        )

        for λ in λ_cases, dt in dt_cases, c0 in c0_cases
            @test _sym2_min_eig(c0...) > 0
            ψ0 = _sym2_log(c0...)
            ψ1 = _oldroydb_relax_log(ψ0..., λ, dt)
            c1 = _sym2_exp(ψ1...)
            c_exact = _oldroydb_relax_c(c0..., λ, dt)
            _assert_sym2_close(c1, c_exact; atol=2e-12, rtol=2e-12)
            @test _sym2_min_eig(c1...) > 0
        end
    end

    @testset "M1 pure relaxation composes exactly" begin
        λ = 7.0
        c0 = (2.4, -0.3, 1.1)
        ψ = _sym2_log(c0...)
        nsteps = 17
        dt = 0.13
        for _ in 1:nsteps
            ψ = _oldroydb_relax_log(ψ..., λ, dt)
        end
        c_many = _sym2_exp(ψ...)
        c_once = _oldroydb_relax_c(c0..., λ, nsteps * dt)
        _assert_sym2_close(c_many, c_once; atol=2e-12, rtol=2e-12)
    end

    @testset "M1 KA relaxation kernel matches exact local relaxation" begin
        Nx, Ny = 6, 5
        λ = 2.7
        dt = 0.31
        psixx = [0.1 + 0.02i - 0.01j for i in 1:Nx, j in 1:Ny]
        psixy = [0.01 * sin(i + 2j) for i in 1:Nx, j in 1:Ny]
        psiyy = [-0.05 + 0.015i + 0.02j for i in 1:Nx, j in 1:Ny]
        outxx = similar(psixx)
        outxy = similar(psixy)
        outyy = similar(psiyy)

        Kraken.logfv_relax_log_2d!(outxx, outxy, outyy, psixx, psixy, psiyy, λ, dt)
        KernelAbstractions.synchronize(KernelAbstractions.CPU())

        for j in 1:Ny, i in 1:Nx
            expected = _oldroydb_relax_log(psixx[i, j], psixy[i, j], psiyy[i, j], λ, dt)
            _assert_sym2_close((outxx[i, j], outxy[i, j], outyy[i, j]), expected; atol=2e-12, rtol=2e-12)
        end
    end

    @testset "M2 homogeneous simple shear source convention" begin
        γ_cases = (-0.03, -1e-4, 1e-4, 0.02)
        λ_cases = (0.5, 3.0, 600.0)

        for γ in γ_cases, λ in λ_cases
            source_i = _oldroydb_source_c(1.0, 0.0, 1.0, 0.0, γ, 0.0, 0.0, λ)
            _assert_sym2_close(source_i, (0.0, γ, 0.0))

            csteady = _oldroydb_simple_shear_stationary(γ, λ)
            source_steady = _oldroydb_source_c(csteady..., 0.0, γ, 0.0, 0.0, λ)
            _assert_sym2_close(source_steady, (0.0, 0.0, 0.0); atol=5e-12, rtol=5e-12)
            @test _sym2_min_eig(csteady...) > 0
        end
    end

    @testset "M2 homogeneous simple shear exact trajectory from identity" begin
        γ = 0.015
        λ = 8.0
        times = (0.0, 0.1, 1.0, 12.0, 80.0)

        previous_cxy = -Inf
        for t in times
            c = _oldroydb_simple_shear_from_identity(γ, λ, t)
            @test _sym2_min_eig(c...) > 0
            @test c[2] >= previous_cxy
            previous_cxy = c[2]

            if t > 0
                h = max(1e-7, 1e-7 * t)
                c_plus = _oldroydb_simple_shear_from_identity(γ, λ, t + h)
                c_minus = _oldroydb_simple_shear_from_identity(γ, λ, t - h)
                numeric_dt = (
                    (c_plus[1] - c_minus[1]) / (2h),
                    (c_plus[2] - c_minus[2]) / (2h),
                    (c_plus[3] - c_minus[3]) / (2h),
                )
                source = _oldroydb_source_c(c..., 0.0, γ, 0.0, 0.0, λ)
                _assert_sym2_close(numeric_dt, source; atol=2e-8, rtol=2e-8)
            end
        end

        c_long = _oldroydb_simple_shear_from_identity(γ, λ, 400λ)
        csteady = _oldroydb_simple_shear_stationary(γ, λ)
        _assert_sym2_close(c_long, csteady; atol=1e-12, rtol=1e-12)
    end

    @testset "M2b log source split gives exact pure deformation and preserves SPD" begin
        γ = 3.0
        dt = 0.4
        λ_huge = 1e30
        ψ1 = _oldroydb_step_log(0.0, 0.0, 0.0, 0.0, γ, 0.0, 0.0, λ_huge, dt)
        c1 = _sym2_exp(ψ1...)
        _assert_sym2_close(c1, (1 + (γ * dt)^2, γ * dt, 1.0); atol=2e-12, rtol=2e-12)

        λ = 2.0
        for Wi in (1.0, 5.0, 10.0, 30.0)
            γwi = Wi / λ
            ψ = (0.0, 0.0, 0.0)
            for _ in 1:400
                ψ = _oldroydb_step_log(ψ..., 0.0, γwi, 0.0, 0.0, λ, 0.01)
                c = _sym2_exp(ψ...)
                @test _sym2_min_eig(c...) > 0
                @test all(isfinite, c)
            end
            c = _sym2_exp(ψ...)
            @test c[1] > 1.0
            @test sign(c[2]) == sign(γwi)
            @test c[3] ≈ 1.0 atol=2e-12 rtol=2e-12
        end
    end

    @testset "M2b KA log source step matches local split operator" begin
        Nx, Ny = 5, 6
        λ = 3.0
        dt = 0.07
        psixx = [0.03i - 0.02j for i in 1:Nx, j in 1:Ny]
        psixy = [0.004 * (i + j) for i in 1:Nx, j in 1:Ny]
        psiyy = [-0.01i + 0.015j for i in 1:Nx, j in 1:Ny]
        dudx = [0.01 * (i - 2) for i in 1:Nx, j in 1:Ny]
        dudy = [0.03 - 0.002j for i in 1:Nx, j in 1:Ny]
        dvdx = [-0.02 + 0.001i for i in 1:Nx, j in 1:Ny]
        dvdy = [-dudx[i, j] for i in 1:Nx, j in 1:Ny]
        outxx = similar(psixx)
        outxy = similar(psixy)
        outyy = similar(psiyy)

        Kraken.logfv_step_oldroydb_log_2d!(
            outxx, outxy, outyy,
            psixx, psixy, psiyy,
            dudx, dudy, dvdx, dvdy,
            λ, dt,
        )
        KernelAbstractions.synchronize(KernelAbstractions.CPU())

        for j in 1:Ny, i in 1:Nx
            expected = _oldroydb_step_log(
                psixx[i, j], psixy[i, j], psiyy[i, j],
                dudx[i, j], dudy[i, j], dvdx[i, j], dvdy[i, j],
                λ, dt,
            )
            _assert_sym2_close((outxx[i, j], outxy[i, j], outyy[i, j]), expected; atol=2e-14, rtol=2e-14)
            @test _sym2_min_eig(_sym2_exp(outxx[i, j], outxy[i, j], outyy[i, j])...) > 0
        end
    end

    @testset "M2c velocity gradient kernel is exact on affine and Poiseuille fields" begin
        Nx, Ny = 9, 11
        dx, dy = 0.4, 0.25
        ax, ay = 0.03, -0.07
        bx, by = -0.02, 0.05
        ux = [0.2 + ax * ((i - 0.5) * dx) + ay * ((j - 0.5) * dy) for i in 1:Nx, j in 1:Ny]
        uy = [-0.1 + bx * ((i - 0.5) * dx) + by * ((j - 0.5) * dy) for i in 1:Nx, j in 1:Ny]
        dudx = similar(ux)
        dudy = similar(ux)
        dvdx = similar(ux)
        dvdy = similar(ux)

        Kraken.logfv_velocity_gradient_centered_2d!(dudx, dudy, dvdx, dvdy, ux, uy, dx, dy)
        KernelAbstractions.synchronize(KernelAbstractions.CPU())

        for j in 1:Ny, i in 1:Nx
            if i > 1 && i < Nx && j > 1 && j < Ny
                @test dudx[i, j] ≈ ax atol=2e-16 rtol=0.0
                @test dudy[i, j] ≈ ay atol=2e-16 rtol=0.0
                @test dvdx[i, j] ≈ bx atol=2e-16 rtol=0.0
                @test dvdy[i, j] ≈ by atol=2e-16 rtol=0.0
            else
                @test dudx[i, j] == 0.0
                @test dudy[i, j] == 0.0
                @test dvdx[i, j] == 0.0
                @test dvdy[i, j] == 0.0
            end
        end

        height = 1.0
        umax = 0.08
        ux_p = [_poiseuille_ux((j - 0.5) * height / Ny, height, umax) for i in 1:Nx, j in 1:Ny]
        uy_p = zeros(Float64, Nx, Ny)
        Kraken.logfv_velocity_gradient_centered_2d!(dudx, dudy, dvdx, dvdy, ux_p, uy_p, 1.0, height / Ny)
        KernelAbstractions.synchronize(KernelAbstractions.CPU())
        for j in 2:(Ny - 1), i in 2:(Nx - 1)
            y = (j - 0.5) * height / Ny
            @test dudy[i, j] ≈ _poiseuille_shear(y, height, umax) atol=2e-15 rtol=2e-15
            @test dudx[i, j] ≈ 0.0 atol=2e-15
            @test dvdx[i, j] ≈ 0.0 atol=2e-15
            @test dvdy[i, j] ≈ 0.0 atol=2e-15
        end
    end

    @testset "M2c channel velocity gradient is wall-exact for Poiseuille" begin
        Nx, Ny = 8, 13
        height = 1.0
        width = 2.0
        dx = width / Nx
        dy = height / Ny
        umax = 0.09
        ux = [_poiseuille_ux((j - 0.5) * dy, height, umax) for i in 1:Nx, j in 1:Ny]
        uy = zeros(Float64, Nx, Ny)
        dudx = similar(ux)
        dudy = similar(ux)
        dvdx = similar(ux)
        dvdy = similar(ux)

        Kraken.logfv_velocity_gradient_periodicx_wally_2d!(dudx, dudy, dvdx, dvdy, ux, uy, dx, dy)
        KernelAbstractions.synchronize(KernelAbstractions.CPU())

        for j in 1:Ny, i in 1:Nx
            y = (j - 0.5) * dy
            @test dudy[i, j] ≈ _poiseuille_shear(y, height, umax) atol=3e-15 rtol=3e-15
            @test dudx[i, j] ≈ 0.0 atol=3e-15
            @test dvdx[i, j] ≈ 0.0 atol=3e-15
            @test dvdy[i, j] ≈ 0.0 atol=3e-15
        end
    end

    @testset "M3 divergence-corrected upwind preserves constant Psi" begin
        Nx, Ny = 9, 8
        ux_face = [0.17 + 0.021 * (i - 1) - 0.013 * (j - 1) for i in 1:(Nx + 1), j in 1:Ny]
        uy_face = [-0.08 + 0.009 * (i - 1) + 0.017 * (j - 1) for i in 1:Nx, j in 1:(Ny + 1)]
        ψxx = fill(0.4, Nx, Ny)
        ψxy = fill(-0.07, Nx, Ny)
        ψyy = fill(0.2, Nx, Ny)

        for j in 2:(Ny - 1), i in 2:(Nx - 1)
            rhs = _upwind_tensor_advective_rhs(ψxx, ψxy, ψyy, ux_face, uy_face, i, j)
            _assert_sym2_close(rhs, (0.0, 0.0, 0.0); atol=2e-16, rtol=0.0)
        end
    end

    @testset "M3 constant velocity advects affine Psi exactly" begin
        Nx, Ny = 10, 11
        x(i) = i - 1
        y(j) = j - 1
        fields = (
            [0.3 + 0.04 * x(i) - 0.02 * y(j) for i in 1:Nx, j in 1:Ny],
            [-0.1 - 0.03 * x(i) + 0.05 * y(j) for i in 1:Nx, j in 1:Ny],
            [0.2 + 0.01 * x(i) + 0.07 * y(j) for i in 1:Nx, j in 1:Ny],
        )
        slopes = ((0.04, -0.02), (-0.03, 0.05), (0.01, 0.07))
        velocity_cases = ((0.13, -0.09), (-0.11, 0.21), (0.18, 0.06), (-0.05, -0.08))

        for (u, v) in velocity_cases
            ux_face = fill(u, Nx + 1, Ny)
            uy_face = fill(v, Nx, Ny + 1)
            for j in 2:(Ny - 1), i in 2:(Nx - 1)
                rhs = _upwind_tensor_advective_rhs(fields..., ux_face, uy_face, i, j)
                expected = ntuple(k -> -(u * slopes[k][1] + v * slopes[k][2]), 3)
                _assert_sym2_close(rhs, expected; atol=2e-16, rtol=0.0)
            end
        end
    end

    @testset "M3 CFL-one periodic upwind direction" begin
        Nx, Ny = 7, 6
        φ = [10i + j for i in 1:Nx, j in 1:Ny]

        step_east = _periodic_upwind_scalar_step(φ, 1.0, 0.0, 1.0)
        step_west = _periodic_upwind_scalar_step(φ, -1.0, 0.0, 1.0)
        step_north = _periodic_upwind_scalar_step(φ, 0.0, 1.0, 1.0)
        step_south = _periodic_upwind_scalar_step(φ, 0.0, -1.0, 1.0)

        for j in 1:Ny, i in 1:Nx
            @test step_east[i, j] == φ[_periodic(i - 1, Nx), j]
            @test step_west[i, j] == φ[_periodic(i + 1, Nx), j]
            @test step_north[i, j] == φ[i, _periodic(j - 1, Ny)]
            @test step_south[i, j] == φ[i, _periodic(j + 1, Ny)]
        end
    end

    @testset "M3 KA upwind advection kernel matches local operator" begin
        Nx, Ny = 9, 8
        dt = 0.2
        ux_face = [0.12 for _ in 1:(Nx + 1), _ in 1:Ny]
        uy_face = [-0.05 for _ in 1:Nx, _ in 1:(Ny + 1)]
        psixx = [0.3 + 0.04 * (i - 1) - 0.02 * (j - 1) for i in 1:Nx, j in 1:Ny]
        psixy = [-0.1 - 0.03 * (i - 1) + 0.05 * (j - 1) for i in 1:Nx, j in 1:Ny]
        psiyy = [0.2 + 0.01 * (i - 1) + 0.07 * (j - 1) for i in 1:Nx, j in 1:Ny]
        outxx = similar(psixx)
        outxy = similar(psixy)
        outyy = similar(psiyy)

        Kraken.logfv_advect_upwind_2d!(outxx, outxy, outyy, psixx, psixy, psiyy, ux_face, uy_face, dt)
        KernelAbstractions.synchronize(KernelAbstractions.CPU())

        for j in 1:Ny, i in 1:Nx
            if i > 1 && i < Nx && j > 1 && j < Ny
                rhs = _upwind_tensor_advective_rhs(psixx, psixy, psiyy, ux_face, uy_face, i, j)
                expected = (
                    psixx[i, j] + dt * rhs[1],
                    psixy[i, j] + dt * rhs[2],
                    psiyy[i, j] + dt * rhs[3],
                )
                _assert_sym2_close((outxx[i, j], outxy[i, j], outyy[i, j]), expected; atol=2e-16, rtol=0.0)
            else
                _assert_sym2_close(
                    (outxx[i, j], outxy[i, j], outyy[i, j]),
                    (psixx[i, j], psixy[i, j], psiyy[i, j]);
                    atol=0.0, rtol=0.0,
                )
            end
        end
    end

    @testset "M4 local Poiseuille conformation and stress are analytical" begin
        Nx, Ny = 5, 19
        height = 1.0
        umax = 0.08
        λ = 6.0
        prefactor = 0.14
        psixx = zeros(Float64, Nx, Ny)
        psixy = zeros(Float64, Nx, Ny)
        psiyy = zeros(Float64, Nx, Ny)
        expected_tau = Array{NTuple{3,Float64}}(undef, Nx, Ny)

        for j in 1:Ny, i in 1:Nx
            y = height * (j - 0.5) / Ny
            γ = _poiseuille_shear(y, height, umax)
            c = _oldroydb_simple_shear_stationary(γ, λ)
            ψ = _sym2_log(c...)
            psixx[i, j], psixy[i, j], psiyy[i, j] = ψ

            source = _oldroydb_source_c(c..., 0.0, γ, 0.0, 0.0, λ)
            _assert_sym2_close(source, (0.0, 0.0, 0.0); atol=4e-14, rtol=4e-14)

            expected_tau[i, j] = (
                prefactor * 2 * (λ * γ)^2,
                prefactor * λ * γ,
                0.0,
            )
        end

        tauxx = similar(psixx)
        tauxy = similar(psixy)
        tauyy = similar(psiyy)
        Kraken.logfv_stress_from_log_2d!(tauxx, tauxy, tauyy, psixx, psixy, psiyy, prefactor)
        KernelAbstractions.synchronize(KernelAbstractions.CPU())

        for j in 1:Ny, i in 1:Nx
            _assert_sym2_close(
                (tauxx[i, j], tauxy[i, j], tauyy[i, j]),
                expected_tau[i, j];
                atol=3e-14, rtol=3e-14,
            )
        end
    end

    @testset "M4 local Poiseuille polymer force is analytical" begin
        Nx, Ny = 7, 21
        height = 1.0
        dx = 1.0
        dy = height / Ny
        umax = 0.06
        λ = 5.0
        prefactor = 0.11
        psixx = zeros(Float64, Nx, Ny)
        psixy = zeros(Float64, Nx, Ny)
        psiyy = zeros(Float64, Nx, Ny)

        for j in 1:Ny, i in 1:Nx
            y = height * (j - 0.5) / Ny
            γ = _poiseuille_shear(y, height, umax)
            psixx[i, j], psixy[i, j], psiyy[i, j] = _sym2_log(_oldroydb_simple_shear_stationary(γ, λ)...)
        end

        tauxx = similar(psixx)
        tauxy = similar(psixy)
        tauyy = similar(psiyy)
        fx = similar(psixx)
        fy = similar(psixx)
        Kraken.logfv_stress_from_log_2d!(tauxx, tauxy, tauyy, psixx, psixy, psiyy, prefactor)
        Kraken.logfv_polymer_force_centered_2d!(fx, fy, tauxx, tauxy, tauyy, dx, dy)
        KernelAbstractions.synchronize(KernelAbstractions.CPU())

        expected_fx = prefactor * λ * (-8 * umax / height^2)
        for j in 2:(Ny - 1), i in 2:(Nx - 1)
            @test fx[i, j] ≈ expected_fx atol=8e-14 rtol=8e-14
            @test fy[i, j] ≈ 0.0 atol=8e-14
        end
        for i in 1:Nx
            @test fx[i, 1] == 0.0
            @test fy[i, 1] == 0.0
            @test fx[i, Ny] == 0.0
            @test fy[i, Ny] == 0.0
        end
    end

    @testset "M4 BSD force correction preserves continuum balance" begin
        Nx, Ny = 7, 21
        height = 1.0
        dx = 1.0
        dy = height / Ny
        umax = 0.06
        λ = 5.0
        prefactor = 0.11
        nu_p = prefactor * λ
        lapu = -8 * umax / height^2

        psixx = zeros(Float64, Nx, Ny)
        psixy = zeros(Float64, Nx, Ny)
        psiyy = zeros(Float64, Nx, Ny)
        ux = zeros(Float64, Nx, Ny)
        uy = zeros(Float64, Nx, Ny)
        for j in 1:Ny, i in 1:Nx
            y = height * (j - 0.5) / Ny
            γ = _poiseuille_shear(y, height, umax)
            psixx[i, j], psixy[i, j], psiyy[i, j] = _sym2_log(_oldroydb_simple_shear_stationary(γ, λ)...)
            ux[i, j] = _poiseuille_ux(y, height, umax)
        end

        tauxx = similar(psixx)
        tauxy = similar(psixy)
        tauyy = similar(psiyy)
        fx_poly = similar(psixx)
        fy_poly = similar(psixx)
        fx_total = similar(psixx)
        fy_total = similar(psixx)

        Kraken.logfv_stress_from_log_2d!(tauxx, tauxy, tauyy, psixx, psixy, psiyy, prefactor)
        Kraken.logfv_polymer_force_centered_2d!(fx_poly, fy_poly, tauxx, tauxy, tauyy, dx, dy)

        for zeta in (0.0, 0.5, 1.0)
            Kraken.logfv_bsd_correct_force_centered_2d!(
                fx_total, fy_total, fx_poly, fy_poly, ux, uy, zeta, nu_p, dx, dy,
            )
            KernelAbstractions.synchronize(KernelAbstractions.CPU())

            expected_fx = (1 - zeta) * nu_p * lapu
            for j in 2:(Ny - 1), i in 2:(Nx - 1)
                @test fx_total[i, j] ≈ expected_fx atol=1e-13 rtol=1e-13
                @test fy_total[i, j] ≈ 0.0 atol=1e-13
            end
        end
    end

    @testset "M4 force boundary fill copies nearest interior halo" begin
        Nx, Ny = 6, 5
        fx = [100i + j for i in 1:Nx, j in 1:Ny]
        fy = [-10i + 0.5j for i in 1:Nx, j in 1:Ny]
        fx0 = copy(fx)
        fy0 = copy(fy)

        Kraken.logfv_fill_nearest_boundary_2d!(fx, fy)
        KernelAbstractions.synchronize(KernelAbstractions.CPU())

        for j in 1:Ny, i in 1:Nx
            ii = clamp(i, 2, Nx - 1)
            jj = clamp(j, 2, Ny - 1)
            if i == 1 || i == Nx || j == 1 || j == Ny
                @test fx[i, j] == fx0[ii, jj]
                @test fy[i, j] == fy0[ii, jj]
            else
                @test fx[i, j] == fx0[i, j]
                @test fy[i, j] == fy0[i, j]
            end
        end
    end

    @testset "M5 local Couette conformation, stress, force, and BSD are analytical" begin
        Nx, Ny = 8, 18
        height = 1.0
        dx = 1.0
        dy = height / Ny
        uwall = 0.07
        γ = uwall / height
        λ = 9.0
        prefactor = 0.08
        nu_p = prefactor * λ
        psixx = zeros(Float64, Nx, Ny)
        psixy = zeros(Float64, Nx, Ny)
        psiyy = zeros(Float64, Nx, Ny)
        ux = zeros(Float64, Nx, Ny)
        uy = zeros(Float64, Nx, Ny)
        csteady = _oldroydb_simple_shear_stationary(γ, λ)
        ψsteady = _sym2_log(csteady...)

        for j in 1:Ny, i in 1:Nx
            y = height * (j - 0.5) / Ny
            psixx[i, j], psixy[i, j], psiyy[i, j] = ψsteady
            ux[i, j] = _couette_ux(y, height, uwall)
            source = _oldroydb_source_c(csteady..., 0.0, γ, 0.0, 0.0, λ)
            _assert_sym2_close(source, (0.0, 0.0, 0.0); atol=2e-14, rtol=2e-14)
        end

        tauxx = similar(psixx)
        tauxy = similar(psixy)
        tauyy = similar(psiyy)
        fx_poly = similar(psixx)
        fy_poly = similar(psixx)
        fx_total = similar(psixx)
        fy_total = similar(psixx)

        Kraken.logfv_stress_from_log_2d!(tauxx, tauxy, tauyy, psixx, psixy, psiyy, prefactor)
        Kraken.logfv_polymer_force_centered_2d!(fx_poly, fy_poly, tauxx, tauxy, tauyy, dx, dy)
        Kraken.logfv_bsd_correct_force_centered_2d!(
            fx_total, fy_total, fx_poly, fy_poly, ux, uy, 1.0, nu_p, dx, dy,
        )
        KernelAbstractions.synchronize(KernelAbstractions.CPU())

        expected_tau = (
            prefactor * 2 * (λ * γ)^2,
            prefactor * λ * γ,
            0.0,
        )
        for j in 2:(Ny - 1), i in 2:(Nx - 1)
            _assert_sym2_close((tauxx[i, j], tauxy[i, j], tauyy[i, j]), expected_tau; atol=2e-14, rtol=2e-14)
            @test fx_poly[i, j] ≈ 0.0 atol=2e-14
            @test fy_poly[i, j] ≈ 0.0 atol=2e-14
            @test fx_total[i, j] ≈ 0.0 atol=2e-14
            @test fy_total[i, j] ≈ 0.0 atol=2e-14
        end
    end

    @testset "M5 macro channel driver exercises log-FV pipeline" begin
        for flow in (:poiseuille, :couette)
            result = Kraken.run_viscoelastic_logfv_channel_2d(;
                Nx=13, Ny=17, flow=flow, height=1.0, width=2.0,
                umax=0.06, uwall=0.07, lambda=5.0, prefactor=0.11,
                bsd_fraction=flow === :poiseuille ? 1.0 : 0.5,
                backend=KernelAbstractions.CPU(), T=Float64,
            )
            @test result.flow === flow
            @test result.Nx == 13
            @test result.Ny == 17
            @test result.dx ≈ 2.0 / 13
            @test result.dy ≈ 1.0 / 17
            @test result.min_c_eig > 0
            @test result.max_tau_error < 5e-14
            @test result.max_poly_force_error < 5e-13
            @test result.max_total_force_error < 5e-13
            @test result.max_transverse_force < 5e-13
            @test all(isfinite, result.fx_total)
            @test all(isfinite, result.tauxx)
        end
    end

    @testset "M5b forced-field macroscopic velocity uses local Guo correction" begin
        Nx, Ny = 5, 4
        f = zeros(Float64, Nx, Ny, 9)
        for q in 1:9
            f[:, :, q] .= Kraken.equilibrium(D2Q9(), 1.0, 0.0, 0.0, q)
        end
        fx = [1e-4 * (i - 0.25j) for i in 1:Nx, j in 1:Ny]
        fy = [-7e-5 * (j + 0.1i) for i in 1:Nx, j in 1:Ny]
        rho = similar(fx)
        ux = similar(fx)
        uy = similar(fx)

        Kraken.logfv_compute_macroscopic_forced_field_2d!(rho, ux, uy, f, fx, fy)
        KernelAbstractions.synchronize(KernelAbstractions.CPU())

        for j in 1:Ny, i in 1:Nx
            @test rho[i, j] ≈ 1.0 atol=5e-16 rtol=0.0
            @test ux[i, j] ≈ 0.5 * fx[i, j] atol=2e-16 rtol=0.0
            @test uy[i, j] ≈ 0.5 * fy[i, j] atol=2e-16 rtol=0.0
        end
    end

    @testset "M5c frozen-force coupled Poiseuille recovers total viscosity" begin
        Nx, Ny = 6, 20
        nu_s = 0.04
        nu_p = 0.06
        Fx_body = 1e-5
        lambda = 5.0

        for zeta in (0.0, 0.5, 1.0)
            result = Kraken.run_viscoelastic_logfv_poiseuille_frozen_force_2d(;
                Nx=Nx, Ny=Ny, nu_s=nu_s, nu_p=nu_p, Fx_body=Fx_body,
                lambda=lambda, bsd_fraction=zeta, force_boundary_fill=:nearest,
                max_steps=8000, backend=KernelAbstractions.CPU(), T=Float64,
            )
            expected_force = Fx_body * (nu_s + zeta * nu_p) / (nu_s + nu_p)

            @test result.nu_lbm ≈ nu_s + zeta * nu_p
            @test result.polymer_channel.min_c_eig > 0
            @test result.polymer_channel.max_total_force_error < 5e-13
            @test result.max_rel_error < 5e-3
            @test result.max_uy < 1e-12
            @test all(isfinite, result.ux)
            @test all(isfinite, result.fx_total)
            for j in 1:Ny, i in 1:Nx
                @test result.fx_total[i, j] ≈ expected_force atol=2e-13 rtol=2e-13
                @test result.fy_total[i, j] ≈ 0.0 atol=2e-13
            end
        end
    end

    @testset "M5c missing force boundary fill is a visible split-mode defect" begin
        result = Kraken.run_viscoelastic_logfv_poiseuille_frozen_force_2d(;
            Nx=6, Ny=16, nu_s=0.04, nu_p=0.06, Fx_body=1e-5,
            lambda=5.0, bsd_fraction=0.0, force_boundary_fill=:none,
            max_steps=4000, backend=KernelAbstractions.CPU(), T=Float64,
        )
        @test result.max_rel_error > 0.1
    end
end
