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

function _oldroydb_source_c(cxx, cxy, cyy, dudx, dudy, dvdx, dvdy, λ)
    return Kraken.logfv_oldroydb_source_c_2d(cxx, cxy, cyy, dudx, dudy, dvdx, dvdy, λ)
end

_oldroydb_simple_shear_stationary(γ, λ) = (1 + 2 * (λ * γ)^2, λ * γ, 1.0)

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
end
