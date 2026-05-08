using Test
using LinearAlgebra

const LOGFV_ATOL = 1e-12
const LOGFV_RTOL = 1e-12

_sym2_mat(a, b, d) = [a b; b d]

function _sym2_min_eig(a, b, d)
    m = 0.5 * (a + d)
    h = 0.5 * (a - d)
    return m - hypot(h, b)
end

function _sym2_exp(a, b, d)
    m = 0.5 * (a + d)
    h = 0.5 * (a - d)
    δ = hypot(h, b)
    em = exp(m)
    scale = ifelse(δ < sqrt(eps(typeof(δ))), one(δ) + δ^2 / 6, sinh(δ) / δ)
    ch = cosh(δ)
    return (
        em * (ch + scale * h),
        em * scale * b,
        em * (ch - scale * h),
    )
end

function _sym2_log(a, b, d)
    λmin = _sym2_min_eig(a, b, d)
    λmin > 0 || throw(DomainError(λmin, "symmetric 2x2 log requires SPD input"))
    m = 0.5 * (a + d)
    h = 0.5 * (a - d)
    δ = hypot(h, b)
    α = 0.5 * (log(m + δ) + log(m - δ))
    β = if δ < sqrt(eps(typeof(δ))) * max(one(m), abs(m))
        inv(m) + δ^2 / (3 * m^3)
    else
        0.5 * (log(m + δ) - log(m - δ)) / δ
    end
    return (
        α + β * h,
        β * b,
        α - β * h,
    )
end

function _oldroydb_relax_c(cxx, cxy, cyy, λ, dt)
    decay = exp(-dt / λ)
    return (
        1 + (cxx - 1) * decay,
        cxy * decay,
        1 + (cyy - 1) * decay,
    )
end

function _oldroydb_relax_log(ψxx, ψxy, ψyy, λ, dt)
    cxx, cxy, cyy = _sym2_exp(ψxx, ψxy, ψyy)
    rxx, rxy, ryy = _oldroydb_relax_c(cxx, cxy, cyy, λ, dt)
    return _sym2_log(rxx, rxy, ryy)
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
end
