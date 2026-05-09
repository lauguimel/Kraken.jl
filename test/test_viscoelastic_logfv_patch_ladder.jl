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

function _fluid_x_neighbor(is_solid, i, j)
    Nx, _ = size(is_solid)
    return (i > 1 && !is_solid[i - 1, j]) || (i < Nx && !is_solid[i + 1, j])
end

function _fluid_y_neighbor(is_solid, i, j)
    _, Ny = size(is_solid)
    return (j > 1 && !is_solid[i, j - 1]) || (j < Ny && !is_solid[i, j + 1])
end

function _fluid_x_second(is_solid, i, j)
    Nx, _ = size(is_solid)
    return (i > 1 && !is_solid[i - 1, j] && i < Nx && !is_solid[i + 1, j]) ||
           (i + 2 <= Nx && !is_solid[i + 1, j] && !is_solid[i + 2, j]) ||
           (i - 2 >= 1 && !is_solid[i - 1, j] && !is_solid[i - 2, j])
end

function _fluid_y_second(is_solid, i, j)
    _, Ny = size(is_solid)
    return (j > 1 && !is_solid[i, j - 1] && j < Ny && !is_solid[i, j + 1]) ||
           (j + 2 <= Ny && !is_solid[i, j + 1] && !is_solid[i, j + 2]) ||
           (j - 2 >= 1 && !is_solid[i, j - 1] && !is_solid[i, j - 2])
end

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

function _run_uniform_shear_log_source_kernel(γ, λ, dt, nsteps; Nx=4, Ny=5)
    psixx = zeros(Float64, Nx, Ny)
    psixy = zeros(Float64, Nx, Ny)
    psiyy = zeros(Float64, Nx, Ny)
    outxx = similar(psixx)
    outxy = similar(psixy)
    outyy = similar(psiyy)
    dudx = zeros(Float64, Nx, Ny)
    dudy = fill(Float64(γ), Nx, Ny)
    dvdx = zeros(Float64, Nx, Ny)
    dvdy = zeros(Float64, Nx, Ny)

    for _ in 1:nsteps
        Kraken.logfv_step_oldroydb_log_2d!(
            outxx, outxy, outyy,
            psixx, psixy, psiyy,
            dudx, dudy, dvdx, dvdy,
            Float64(λ), Float64(dt);
            sync=false,
        )
        psixx, outxx = outxx, psixx
        psixy, outxy = outxy, psixy
        psiyy, outyy = outyy, psiyy
    end
    KernelAbstractions.synchronize(KernelAbstractions.CPU())

    return Kraken.logfv_exp_sym2_2d(psixx[2, 2], psixy[2, 2], psiyy[2, 2])
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

    @testset "M2b source subcycling estimator separates relaxation and deformation limits" begin
        z01 = Kraken.logfv_oldroydb_split_relax_increment(0.01)
        @test z01 > 0.019
        @test z01 < 0.021

        relax_limited = Kraken.logfv_oldroydb_subcycle_estimate(
            0.0, 5.0, 1.0;
            relative_tolerance=0.01,
            max_deformation_increment=0.05,
            max_substeps=64,
        )
        @test relax_limited.recommended == 10
        @test relax_limited.relax_substeps == 10
        @test relax_limited.deformation_substeps == 1
        @test !relax_limited.clamped

        deformation_limited = Kraken.logfv_oldroydb_subcycle_estimate(
            0.23, 1e6, 1.0;
            relative_tolerance=0.01,
            max_deformation_increment=0.05,
            max_substeps=64,
        )
        @test deformation_limited.recommended == 5
        @test deformation_limited.relax_substeps == 1
        @test deformation_limited.deformation_substeps == 5

        clamped = Kraken.logfv_oldroydb_subcycle_estimate(
            0.0, 0.01, 1.0;
            relative_tolerance=0.01,
            max_deformation_increment=0.05,
            max_substeps=8,
        )
        @test clamped.recommended == 8
        @test clamped.clamped

        @test_throws ArgumentError Kraken.logfv_oldroydb_split_relax_increment(0.0)
        @test_throws ArgumentError Kraken.logfv_oldroydb_subcycle_estimate(-1.0, 1.0, 1.0)
        @test_throws ArgumentError Kraken.logfv_oldroydb_subcycle_estimate(0.0, -1.0, 1.0)
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

    @testset "M2c solid-aware velocity gradient does not read through obstacles" begin
        Nx, Ny = 10, 9
        dx, dy = 0.3, 0.2
        ax, ay = 0.04, -0.03
        bx, by = -0.02, 0.05
        ux = [0.12 + ax * ((i - 0.5) * dx) + ay * ((j - 0.5) * dy) for i in 1:Nx, j in 1:Ny]
        uy = [-0.08 + bx * ((i - 0.5) * dx) + by * ((j - 0.5) * dy) for i in 1:Nx, j in 1:Ny]
        is_solid = falses(Nx, Ny)
        is_solid[4:6, 4:6] .= true
        dudx = similar(ux)
        dudy = similar(ux)
        dvdx = similar(ux)
        dvdy = similar(ux)

        Kraken.logfv_velocity_gradient_solid_aware_2d!(dudx, dudy, dvdx, dvdy, ux, uy, is_solid, dx, dy)
        KernelAbstractions.synchronize(KernelAbstractions.CPU())

        for j in 1:Ny, i in 1:Nx
            if is_solid[i, j]
                @test dudx[i, j] == 0.0
                @test dudy[i, j] == 0.0
                @test dvdx[i, j] == 0.0
                @test dvdy[i, j] == 0.0
                continue
            end

            x_has_neighbor = (i > 1 && !is_solid[i - 1, j]) || (i < Nx && !is_solid[i + 1, j])
            y_has_neighbor = (j > 1 && !is_solid[i, j - 1]) || (j < Ny && !is_solid[i, j + 1])
            @test dudx[i, j] ≈ (x_has_neighbor ? ax : 0.0) atol=3e-15 rtol=3e-15
            @test dvdx[i, j] ≈ (x_has_neighbor ? bx : 0.0) atol=3e-15 rtol=3e-15
            @test dudy[i, j] ≈ (y_has_neighbor ? ay : 0.0) atol=3e-15 rtol=3e-15
            @test dvdy[i, j] ≈ (y_has_neighbor ? by : 0.0) atol=3e-15 rtol=3e-15
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

    @testset "M3 solid-aware upwind preserves constant Psi around square obstacle" begin
        Nx, Ny = 12, 10
        is_solid = falses(Nx, Ny)
        is_solid[5:7, 4:6] .= true
        ux = [0.08 + 0.01 * sin(i + j) for i in 1:Nx, j in 1:Ny]
        uy = [0.02 * cos(i - 2j) for i in 1:Nx, j in 1:Ny]
        ux[is_solid] .= 0.0
        uy[is_solid] .= 0.0
        ux_face = zeros(Float64, Nx + 1, Ny)
        uy_face = zeros(Float64, Nx, Ny + 1)

        Kraken.logfv_cell_velocity_to_faces_solid_aware_2d!(ux_face, uy_face, ux, uy, is_solid)
        KernelAbstractions.synchronize(KernelAbstractions.CPU())

        for j in 1:Ny
            @test ux_face[1, j] == ux_face[Nx + 1, j]
        end
        for j in 4:6
            @test ux_face[5, j] == 0.0
            @test ux_face[8, j] == 0.0
        end
        for i in 5:7
            @test uy_face[i, 4] == 0.0
            @test uy_face[i, 7] == 0.0
        end

        psixx = fill(0.3, Nx, Ny)
        psixy = fill(-0.04, Nx, Ny)
        psiyy = fill(0.2, Nx, Ny)
        outxx = similar(psixx)
        outxy = similar(psixy)
        outyy = similar(psiyy)

        Kraken.logfv_advect_upwind_solid_aware_2d!(
            outxx, outxy, outyy,
            psixx, psixy, psiyy, ux_face, uy_face, is_solid, 0.2,
        )
        KernelAbstractions.synchronize(KernelAbstractions.CPU())

        for j in 1:Ny, i in 1:Nx
            if is_solid[i, j]
                _assert_sym2_close((outxx[i, j], outxy[i, j], outyy[i, j]), (0.0, 0.0, 0.0); atol=0.0, rtol=0.0)
            else
                _assert_sym2_close((outxx[i, j], outxy[i, j], outyy[i, j]), (0.3, -0.04, 0.2); atol=3e-16, rtol=0.0)
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

    @testset "M4 solid-aware polymer force does not read through obstacles" begin
        Nx, Ny = 10, 9
        dx, dy = 0.3, 0.2
        axx, axy = 0.07, -0.03
        bxy, byy = 0.02, 0.05
        tauxx = [0.1 + axx * ((i - 0.5) * dx) - 0.01 * ((j - 0.5) * dy) for i in 1:Nx, j in 1:Ny]
        tauxy = [-0.2 + axy * ((i - 0.5) * dx) + bxy * ((j - 0.5) * dy) for i in 1:Nx, j in 1:Ny]
        tauyy = [0.3 + 0.04 * ((i - 0.5) * dx) + byy * ((j - 0.5) * dy) for i in 1:Nx, j in 1:Ny]
        is_solid = falses(Nx, Ny)
        is_solid[4:6, 4:6] .= true
        fx = similar(tauxx)
        fy = similar(tauxx)

        Kraken.logfv_polymer_force_solid_aware_2d!(fx, fy, tauxx, tauxy, tauyy, is_solid, dx, dy)
        KernelAbstractions.synchronize(KernelAbstractions.CPU())

        for j in 1:Ny, i in 1:Nx
            if is_solid[i, j]
                @test fx[i, j] == 0.0
                @test fy[i, j] == 0.0
                continue
            end
            x_has_neighbor = (i > 1 && !is_solid[i - 1, j]) || (i < Nx && !is_solid[i + 1, j])
            y_has_neighbor = (j > 1 && !is_solid[i, j - 1]) || (j < Ny && !is_solid[i, j + 1])
            expected_fx = (x_has_neighbor ? axx : 0.0) + (y_has_neighbor ? bxy : 0.0)
            expected_fy = (x_has_neighbor ? axy : 0.0) + (y_has_neighbor ? byy : 0.0)
            @test fx[i, j] ≈ expected_fx atol=3e-15 rtol=3e-15
            @test fy[i, j] ≈ expected_fy atol=3e-15 rtol=3e-15
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

    @testset "M4b solid-aware BSD force correction is exact for quadratic patches" begin
        Nx, Ny = 10, 9
        dx, dy = 0.3, 0.2
        axx, axy = 0.06, -0.04
        bxx, bxy = -0.03, 0.05
        ayx, ayy = 0.02, -0.07
        byx, byy = 0.08, 0.01
        zeta = 0.6
        nu_p = 0.17

        is_solid = fill(false, Nx, Ny)
        is_solid[4:6, 4:6] .= true
        ux = [0.1 + axx * ((i - 0.5) * dx)^2 + bxx * ((j - 0.5) * dy)^2 +
              0.01 * (i - 0.5) * dx
              for i in 1:Nx, j in 1:Ny]
        uy = [-0.2 + ayy * ((i - 0.5) * dx)^2 + byy * ((j - 0.5) * dy)^2 +
              0.02 * (j - 0.5) * dy
              for i in 1:Nx, j in 1:Ny]
        fx_poly = [0.03 + axy * (i - 0.5) * dx + bxy * (j - 0.5) * dy
                   for i in 1:Nx, j in 1:Ny]
        fy_poly = [-0.02 + ayx * (i - 0.5) * dx + byx * (j - 0.5) * dy
                   for i in 1:Nx, j in 1:Ny]
        fx_total = similar(fx_poly)
        fy_total = similar(fy_poly)

        Kraken.logfv_bsd_correct_force_solid_aware_2d!(
            fx_total, fy_total, fx_poly, fy_poly, ux, uy, is_solid, zeta, nu_p, dx, dy,
        )
        KernelAbstractions.synchronize(KernelAbstractions.CPU())

        for j in 1:Ny, i in 1:Nx
            if is_solid[i, j]
                @test fx_total[i, j] == 0.0
                @test fy_total[i, j] == 0.0
                continue
            end

            x_second =
                (i > 1 && !is_solid[i - 1, j] && i < Nx && !is_solid[i + 1, j]) ||
                (i + 2 <= Nx && !is_solid[i + 1, j] && !is_solid[i + 2, j]) ||
                (i - 2 >= 1 && !is_solid[i - 1, j] && !is_solid[i - 2, j])
            y_second =
                (j > 1 && !is_solid[i, j - 1] && j < Ny && !is_solid[i, j + 1]) ||
                (j + 2 <= Ny && !is_solid[i, j + 1] && !is_solid[i, j + 2]) ||
                (j - 2 >= 1 && !is_solid[i, j - 1] && !is_solid[i, j - 2])
            expected_lap_ux = (x_second ? 2 * axx : 0.0) + (y_second ? 2 * bxx : 0.0)
            expected_lap_uy = (x_second ? 2 * ayy : 0.0) + (y_second ? 2 * byy : 0.0)
            @test fx_total[i, j] ≈ fx_poly[i, j] - zeta * nu_p * expected_lap_ux atol=5e-14 rtol=5e-14
            @test fy_total[i, j] ≈ fy_poly[i, j] - zeta * nu_p * expected_lap_uy atol=5e-14 rtol=5e-14
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

    @testset "M5a Couette log source increment matches analytical transient" begin
        γ = 0.04
        λ = 5.0
        dt = 0.002
        nsteps = 1
        t = nsteps * dt

        c_num = _run_uniform_shear_log_source_kernel(γ, λ, dt, nsteps)
        c_exact = _oldroydb_simple_shear_from_identity(γ, λ, t)

        @test _sym2_min_eig(c_num...) > 0
        _assert_sym2_close(c_num, c_exact; atol=5e-8, rtol=5e-8)
        @test c_num[3] ≈ 1.0 atol=2e-12 rtol=2e-12
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

    @testset "M5d coupled Poiseuille source-force loop recovers total viscosity" begin
        coarse = Kraken.run_viscoelastic_logfv_poiseuille_coupled_2d(;
            Nx=6, Ny=16, nu_s=0.04, nu_p=0.06, Fx_body=1e-5,
            lambda=5.0, bsd_fraction=0.0, polymer_substeps=1,
            max_steps=4000, backend=KernelAbstractions.CPU(), T=Float64,
        )
        fine = Kraken.run_viscoelastic_logfv_poiseuille_coupled_2d(;
            Nx=6, Ny=16, nu_s=0.04, nu_p=0.06, Fx_body=1e-5,
            lambda=5.0, bsd_fraction=0.0, polymer_substeps=:auto,
            max_steps=4000, backend=KernelAbstractions.CPU(), T=Float64,
        )

        @test coarse.max_rel_error > 0.02
        @test fine.polymer_substeps == 10
        @test fine.subcycle_estimate.recommended == 10
        @test fine.subcycle_estimate.relax_substeps == 10
        @test fine.max_rel_error < 0.01
        @test fine.max_rel_error < coarse.max_rel_error / 5
        @test fine.min_c_eig > 0
        @test fine.max_uy < 1e-12

        for zeta in (0.5, 1.0)
            result = Kraken.run_viscoelastic_logfv_poiseuille_coupled_2d(;
                Nx=6, Ny=16, nu_s=0.04, nu_p=0.06, Fx_body=1e-5,
                lambda=5.0, bsd_fraction=zeta, polymer_substeps=:auto,
                max_steps=4000, backend=KernelAbstractions.CPU(), T=Float64,
            )
            @test result.polymer_substeps == 10
            @test result.max_rel_error < 0.012
            @test result.min_c_eig > 0
            @test result.max_uy < 1e-12
            @test all(isfinite, result.ux)
            @test all(isfinite, result.psixx)
        end
    end

    @testset "M6 square periodic coarse macroflow stays SPD and bounded" begin
        result = Kraken.run_viscoelastic_logfv_square_periodic_2d(;
            Nx=28, Ny=14, side=4,
            nu_s=0.08, nu_p=0.02, Fx_body=1e-6,
            lambda=5.0, bsd_fraction=1.0, polymer_substeps=:auto,
            max_steps=150, backend=KernelAbstractions.CPU(), T=Float64,
        )

        @test result.bsd_fraction == 1.0
        @test result.nu_lbm ≈ 0.10
        @test result.polymer_substeps == 10
        @test result.subcycle_estimate.recommended == 10
        @test result.min_c_eig > 0.99
        @test result.max_speed > 1e-6
        @test result.max_speed < 1e-3
        @test result.rho_min > 0.99
        @test result.rho_max < 1.01
        @test all(isfinite, result.ux)
        @test all(isfinite, result.uy)
        @test all(isfinite, result.psixx)
        @test all(isfinite, result.fx_total)
        @test any(result.is_solid)
        @test !all(result.is_solid)
    end

    @testset "M7 square periodic low-beta coarse sweep stays SPD and bounded" begin
        cases = (
            (nu_s=0.005, nu_p=0.095, Fx_body=5e-7, lambda=10.0),
            (nu_s=0.002, nu_p=0.098, Fx_body=5e-6, lambda=50.0),
            (nu_s=0.002, nu_p=0.098, Fx_body=5e-6, lambda=200.0),
        )

        for case in cases
            result = Kraken.run_viscoelastic_logfv_square_periodic_2d(;
                Nx=28, Ny=14, side=4,
                nu_s=case.nu_s, nu_p=case.nu_p, Fx_body=case.Fx_body,
                lambda=case.lambda, bsd_fraction=1.0, polymer_substeps=:auto,
                max_steps=120, backend=KernelAbstractions.CPU(), T=Float64,
            )
            beta = result.nu_s / result.nu_total

            @test beta <= 0.05
            @test result.bsd_fraction == 1.0
            @test result.nu_lbm ≈ result.nu_total
            @test result.polymer_substeps >= 1
            @test !result.subcycle_estimate.clamped
            @test result.min_c_eig > 0.9
            @test result.max_speed > 1e-5
            @test result.max_speed < 3e-3
            @test result.rho_min > 0.995
            @test result.rho_max < 1.005
            @test all(isfinite, result.ux)
            @test all(isfinite, result.uy)
            @test all(isfinite, result.psixx)
            @test all(isfinite, result.psixy)
            @test all(isfinite, result.psiyy)
            @test all(isfinite, result.fx_total)
            @test all(isfinite, result.fy_total)
        end
    end

    @testset "M7b square periodic near-Newtonian limit matches total-viscosity hydro" begin
        hydro = Kraken.run_viscoelastic_logfv_square_periodic_2d(;
            Nx=28, Ny=14, side=4,
            nu_s=0.10, nu_p=0.0, Fx_body=1e-6,
            lambda=1.0, bsd_fraction=1.0,
            max_steps=80, backend=KernelAbstractions.CPU(), T=Float64,
        )
        visco = Kraken.run_viscoelastic_logfv_square_periodic_2d(;
            Nx=28, Ny=14, side=4,
            nu_s=0.08, nu_p=0.02, Fx_body=1e-6,
            lambda=1.0, bsd_fraction=1.0,
            max_steps=80, backend=KernelAbstractions.CPU(), T=Float64,
        )
        fluid = .!visco.is_solid

        @test visco.nu_lbm ≈ hydro.nu_s
        @test visco.polymer_substeps == 50
        @test !visco.subcycle_estimate.clamped
        @test visco.min_c_eig > 0.999
        @test maximum(abs.(visco.ux[fluid] .- hydro.ux[fluid])) < 1e-6
        @test maximum(abs.(visco.uy[fluid] .- hydro.uy[fluid])) < 1e-6
        @test maximum(abs.(visco.rho[fluid] .- hydro.rho[fluid])) < 1e-6
        @test all(isfinite, visco.ux)
        @test all(isfinite, visco.psixx)
        @test all(isfinite, visco.fx_total)
    end

    @testset "M7c square channel open-x coupled log-FV stays bounded" begin
        result = Kraken.run_viscoelastic_logfv_square_channel_coupled_2d(;
            H=12, side=4, L_up=2, L_down=3,
            nu_s=0.08, nu_p=0.02, lambda=5.0,
            u_mean=0.01, Fx_body=2e-7,
            bsd_fraction=1.0, max_steps=20,
            backend=KernelAbstractions.CPU(), T=Float64,
        )

        @test result.geometry.name === :square_obstacle
        @test any(result.is_solid)
        @test !all(result.is_solid)
        @test result.nu_lbm ≈ result.nu_total
        @test result.polymer_substeps == 10
        @test result.min_c_eig > 0.9
        @test result.max_abs_psi < 0.08
        @test result.max_abs_tau < 4e-4
        @test result.max_abs_poly_force > 0
        @test result.max_abs_total_force > 0
        @test result.max_speed > 1e-4
        @test result.max_speed < 0.05
        @test result.rho_min > 0.98
        @test result.rho_max < 1.03
        @test all(isfinite, result.ux)
        @test all(isfinite, result.psixx)
        @test all(isfinite, result.fx_total)
    end

    @testset "M7d square channel near-Newtonian limit matches total-viscosity hydro" begin
        hydro = Kraken.run_viscoelastic_logfv_square_channel_coupled_2d(;
            H=12, side=4, L_up=2, L_down=3,
            nu_s=0.10, nu_p=0.0, lambda=1.0,
            u_mean=0.01, Fx_body=2e-7,
            bsd_fraction=1.0, max_steps=20,
            backend=KernelAbstractions.CPU(), T=Float64,
        )
        visco = Kraken.run_viscoelastic_logfv_square_channel_coupled_2d(;
            H=12, side=4, L_up=2, L_down=3,
            nu_s=0.08, nu_p=0.02, lambda=1.0,
            u_mean=0.01, Fx_body=2e-7,
            bsd_fraction=1.0, max_steps=20,
            backend=KernelAbstractions.CPU(), T=Float64,
        )
        fluid = .!visco.is_solid

        @test visco.nu_lbm ≈ hydro.nu_s
        @test visco.polymer_substeps == 50
        @test !visco.subcycle_estimate.clamped
        @test visco.min_c_eig > 0.98
        @test visco.max_abs_psi < 0.02
        @test maximum(abs.(visco.ux[fluid] .- hydro.ux[fluid])) < 7e-4
        @test maximum(abs.(visco.uy[fluid] .- hydro.uy[fluid])) < 2e-4
        @test maximum(abs.(visco.rho[fluid] .- hydro.rho[fluid])) < 4e-4
        @test all(isfinite, visco.ux)
        @test all(isfinite, visco.psixx)
        @test all(isfinite, visco.fx_total)
    end

    @testset "M8-pre BFS mask solid-aware operators are analytical" begin
        geom = Kraken.backward_facing_step_geometry_2d(;
            H_in=4, expansion_ratio=2, L_up=2, L_down=3, FT=Float64,
        )
        is_solid = geom.is_solid
        Nx, Ny = geom.Nx, geom.Ny
        dx, dy = 0.4, 0.25
        ax, ay = 0.03, -0.02
        bx, by = -0.04, 0.05
        ux = [0.12 + ax * (i - 0.5) * dx + ay * (j - 0.5) * dy
              for i in 1:Nx, j in 1:Ny]
        uy = [-0.08 + bx * (i - 0.5) * dx + by * (j - 0.5) * dy
              for i in 1:Nx, j in 1:Ny]
        dudx = similar(ux)
        dudy = similar(ux)
        dvdx = similar(ux)
        dvdy = similar(ux)

        Kraken.logfv_velocity_gradient_solid_aware_2d!(dudx, dudy, dvdx, dvdy, ux, uy, is_solid, dx, dy)
        KernelAbstractions.synchronize(KernelAbstractions.CPU())

        for j in 1:Ny, i in 1:Nx
            if is_solid[i, j]
                @test dudx[i, j] == 0.0
                @test dudy[i, j] == 0.0
                @test dvdx[i, j] == 0.0
                @test dvdy[i, j] == 0.0
            else
                @test dudx[i, j] ≈ (_fluid_x_neighbor(is_solid, i, j) ? ax : 0.0) atol=2e-14 rtol=2e-14
                @test dvdx[i, j] ≈ (_fluid_x_neighbor(is_solid, i, j) ? bx : 0.0) atol=2e-14 rtol=2e-14
                @test dudy[i, j] ≈ (_fluid_y_neighbor(is_solid, i, j) ? ay : 0.0) atol=2e-14 rtol=2e-14
                @test dvdy[i, j] ≈ (_fluid_y_neighbor(is_solid, i, j) ? by : 0.0) atol=2e-14 rtol=2e-14
            end
        end

        ux_face = zeros(Float64, Nx + 1, Ny)
        uy_face = zeros(Float64, Nx, Ny + 1)
        psixx = fill(0.25, Nx, Ny)
        psixy = fill(-0.03, Nx, Ny)
        psiyy = fill(0.11, Nx, Ny)
        advxx = similar(psixx)
        advxy = similar(psixy)
        advyy = similar(psiyy)
        Kraken.logfv_cell_velocity_to_faces_solid_aware_2d!(ux_face, uy_face, ux, uy, is_solid)
        Kraken.logfv_advect_upwind_solid_aware_2d!(
            advxx, advxy, advyy, psixx, psixy, psiyy, ux_face, uy_face, is_solid, 0.2,
        )
        KernelAbstractions.synchronize(KernelAbstractions.CPU())

        for j in 1:Ny, i in 1:Nx
            if is_solid[i, j]
                @test advxx[i, j] == 0.0
                @test advxy[i, j] == 0.0
                @test advyy[i, j] == 0.0
            else
                @test advxx[i, j] ≈ 0.25 atol=2e-14 rtol=2e-14
                @test advxy[i, j] ≈ -0.03 atol=2e-14 rtol=2e-14
                @test advyy[i, j] ≈ 0.11 atol=2e-14 rtol=2e-14
            end
        end

        tx_xx, ty_xx = 0.07, -0.01
        tx_xy, ty_xy = -0.03, 0.02
        tx_yy, ty_yy = 0.04, 0.05
        tauxx = [0.1 + tx_xx * (i - 0.5) * dx + ty_xx * (j - 0.5) * dy for i in 1:Nx, j in 1:Ny]
        tauxy = [-0.2 + tx_xy * (i - 0.5) * dx + ty_xy * (j - 0.5) * dy for i in 1:Nx, j in 1:Ny]
        tauyy = [0.3 + tx_yy * (i - 0.5) * dx + ty_yy * (j - 0.5) * dy for i in 1:Nx, j in 1:Ny]
        fx_stress = similar(tauxx)
        fy_stress = similar(tauxx)

        Kraken.logfv_polymer_force_solid_aware_2d!(
            fx_stress, fy_stress, tauxx, tauxy, tauyy, is_solid, dx, dy,
        )
        KernelAbstractions.synchronize(KernelAbstractions.CPU())

        for j in 1:Ny, i in 1:Nx
            if is_solid[i, j]
                @test fx_stress[i, j] == 0.0
                @test fy_stress[i, j] == 0.0
            else
                expected_fx = (_fluid_x_neighbor(is_solid, i, j) ? tx_xx : 0.0) +
                              (_fluid_y_neighbor(is_solid, i, j) ? ty_xy : 0.0)
                expected_fy = (_fluid_x_neighbor(is_solid, i, j) ? tx_xy : 0.0) +
                              (_fluid_y_neighbor(is_solid, i, j) ? ty_yy : 0.0)
                @test fx_stress[i, j] ≈ expected_fx atol=5e-14 rtol=5e-14
                @test fy_stress[i, j] ≈ expected_fy atol=5e-14 rtol=5e-14
            end
        end

        qxx, qxy = 0.04, -0.01
        qyx, qyy = -0.06, 0.03
        ux_quad = [0.1 + qxx * ((i - 0.5) * dx)^2 + qxy * ((j - 0.5) * dy)^2
                   for i in 1:Nx, j in 1:Ny]
        uy_quad = [-0.2 + qyx * ((i - 0.5) * dx)^2 + qyy * ((j - 0.5) * dy)^2
                   for i in 1:Nx, j in 1:Ny]
        fx_poly = [0.02 + 0.01 * (i - 0.5) * dx for i in 1:Nx, j in 1:Ny]
        fy_poly = [-0.03 + 0.02 * (j - 0.5) * dy for i in 1:Nx, j in 1:Ny]
        fx_total = similar(fx_poly)
        fy_total = similar(fy_poly)
        zeta = 0.75
        nu_p = 0.09

        Kraken.logfv_bsd_correct_force_solid_aware_2d!(
            fx_total, fy_total, fx_poly, fy_poly, ux_quad, uy_quad, is_solid, zeta, nu_p, dx, dy,
        )
        KernelAbstractions.synchronize(KernelAbstractions.CPU())

        for j in 1:Ny, i in 1:Nx
            if is_solid[i, j]
                @test fx_total[i, j] == 0.0
                @test fy_total[i, j] == 0.0
            else
                lap_ux = (_fluid_x_second(is_solid, i, j) ? 2 * qxx : 0.0) +
                         (_fluid_y_second(is_solid, i, j) ? 2 * qxy : 0.0)
                lap_uy = (_fluid_x_second(is_solid, i, j) ? 2 * qyx : 0.0) +
                         (_fluid_y_second(is_solid, i, j) ? 2 * qyy : 0.0)
                @test fx_total[i, j] ≈ fx_poly[i, j] - zeta * nu_p * lap_ux atol=5e-14 rtol=5e-14
                @test fy_total[i, j] ≈ fy_poly[i, j] - zeta * nu_p * lap_uy atol=5e-14 rtol=5e-14
            end
        end

        field = [10i + j for i in 1:Nx, j in 1:Ny]
        profile = zeros(Float64, Ny)
        Kraken.logfv_copy_column_profile_2d!(profile, field, 3)
        @test profile == field[3, :]

        fx_fluid = zeros(Float64, Nx, Ny)
        fy_fluid = zeros(Float64, Nx, Ny)
        Kraken.logfv_add_constant_force_fluid_2d!(fx_fluid, fy_fluid, is_solid, 0.7, -0.2)
        for j in 1:Ny, i in 1:Nx
            if is_solid[i, j]
                @test fx_fluid[i, j] == 0.0
                @test fy_fluid[i, j] == 0.0
            else
                @test fx_fluid[i, j] == 0.7
                @test fy_fluid[i, j] == -0.2
            end
        end
    end

    @testset "M8a modular LI-BB V2 Guo field canaries" begin
        geom = Kraken.backward_facing_step_geometry_2d(;
            H_in=4, expansion_ratio=2, L_up=2, L_down=3, FT=Float64,
        )
        Nx, Ny = geom.Nx, geom.Ny
        ν = 0.08
        f_in = zeros(Float64, Nx, Ny, 9)
        for j in 1:Ny, i in 1:Nx, q in 1:9
            ux0 = geom.is_solid[i, j] ? 0.0 : 0.015
            f_in[i, j, q] = Kraken.equilibrium(D2Q9(), 1.0, ux0, 0.0, q)
        end

        f_ref = similar(f_in)
        f_force = similar(f_in)
        rho_ref = zeros(Float64, Nx, Ny)
        ux_ref = zeros(Float64, Nx, Ny)
        uy_ref = zeros(Float64, Nx, Ny)
        rho_force = similar(rho_ref)
        ux_force = similar(ux_ref)
        uy_force = similar(uy_ref)
        uwx = zeros(Float64, Nx, Ny, 9)
        uwy = zeros(Float64, Nx, Ny, 9)
        fx_zero = zeros(Float64, Nx, Ny)
        fy_zero = zeros(Float64, Nx, Ny)

        Kraken.fused_trt_libb_v2_step!(
            f_ref, f_in, rho_ref, ux_ref, uy_ref, geom.is_solid,
            geom.q_wall, uwx, uwy, Nx, Ny, ν,
        )
        Kraken.fused_trt_libb_v2_guo_field_step!(
            f_force, f_in, rho_force, ux_force, uy_force, geom.is_solid,
            geom.q_wall, uwx, uwy, fx_zero, fy_zero, Nx, Ny, ν,
        )
        KernelAbstractions.synchronize(KernelAbstractions.CPU())

        @test f_force == f_ref
        @test rho_force == rho_ref
        @test ux_force == ux_ref
        @test uy_force == uy_ref

        Nx2, Ny2 = 6, 5
        ν_bgk = 0.1
        ω = 1 / (3ν_bgk + 0.5)
        Λ_bgk = (1 / ω - 0.5)^2
        f0 = zeros(Float64, Nx2, Ny2, 9)
        for j in 1:Ny2, i in 1:Nx2, q in 1:9
            f0[i, j, q] = Kraken.equilibrium(D2Q9(), 1.0, 0.0, 0.0, q)
        end
        is_fluid = fill(false, Nx2, Ny2)
        q_wall = zeros(Float64, Nx2, Ny2, 9)
        uwx2 = zeros(Float64, Nx2, Ny2, 9)
        uwy2 = zeros(Float64, Nx2, Ny2, 9)
        fx = [1e-5 * (1 + 0.1i - 0.05j) for i in 1:Nx2, j in 1:Ny2]
        fy = [-7e-6 * (1 - 0.02i + 0.03j) for i in 1:Nx2, j in 1:Ny2]
        f_bgk = copy(f0)
        f_trt = similar(f0)
        rho = zeros(Float64, Nx2, Ny2)
        ux = zeros(Float64, Nx2, Ny2)
        uy = zeros(Float64, Nx2, Ny2)

        Kraken.collide_guo_field_2d!(f_bgk, is_fluid, fx, fy, ω)
        Kraken.fused_trt_libb_v2_guo_field_step!(
            f_trt, f0, rho, ux, uy, is_fluid, q_wall, uwx2, uwy2, fx, fy,
            Nx2, Ny2, ν_bgk; Λ=Λ_bgk,
        )
        KernelAbstractions.synchronize(KernelAbstractions.CPU())

        @test f_trt ≈ f_bgk atol=2e-15 rtol=2e-15
        @test all(isfinite, rho)
        @test all(isfinite, ux)
        @test all(isfinite, uy)
    end

    @testset "M8b BFS hydrodynamic Guo-field pipeline is bounded" begin
        geom = Kraken.backward_facing_step_geometry_2d(;
            H_in=4, expansion_ratio=2, L_up=2, L_down=4, FT=Float64,
        )
        Nx, Ny = geom.Nx, geom.Ny
        ν = 0.08
        u_mean = 0.01
        u_profile = Kraken.parabolic_face_profile_2d(geom; face=:west, mean_velocity=u_mean, FT=Float64)
        bcspec = Kraken.default_step_bcspec_2d(geom, u_profile, 1.0)

        f_in = zeros(Float64, Nx, Ny, 9)
        f_out = similar(f_in)
        for j in 1:Ny, i in 1:Nx, q in 1:9
            ux0 = geom.is_solid[i, j] ? 0.0 : u_profile[j]
            f_in[i, j, q] = Kraken.equilibrium(D2Q9(), 1.0, ux0, 0.0, q)
        end
        rho = ones(Float64, Nx, Ny)
        ux = zeros(Float64, Nx, Ny)
        uy = zeros(Float64, Nx, Ny)
        uwx = zeros(Float64, Nx, Ny, 9)
        uwy = zeros(Float64, Nx, Ny, 9)
        fx = [geom.is_solid[i, j] ? 0.0 : 2e-7 for i in 1:Nx, j in 1:Ny]
        fy = zeros(Float64, Nx, Ny)

        for _ in 1:60
            Kraken.fused_trt_libb_v2_guo_field_step!(
                f_out, f_in, rho, ux, uy, geom.is_solid, geom.q_wall,
                uwx, uwy, fx, fy, Nx, Ny, ν,
            )
            Kraken.apply_bc_rebuild_2d!(f_out, f_in, bcspec, ν, Nx, Ny)
            f_in, f_out = f_out, f_in
        end
        Kraken.logfv_compute_macroscopic_forced_field_2d!(rho, ux, uy, f_in, fx, fy)
        KernelAbstractions.synchronize(KernelAbstractions.CPU())

        fluid = .!geom.is_solid
        @test all(isfinite, rho[fluid])
        @test all(isfinite, ux[fluid])
        @test all(isfinite, uy[fluid])
        @test minimum(rho[fluid]) > 0.98
        @test maximum(rho[fluid]) < 1.02
        @test maximum(abs, ux[fluid]) > 1e-4
        @test maximum(abs, ux[fluid]) < 0.05
        @test maximum(abs, uy[fluid]) < 0.02
        outlet_probe = ux[Nx - 2, geom.outlet_open]
        @test sum(outlet_probe) / length(outlet_probe) > 1e-4
    end

    @testset "M8c open-x solid-aware Psi advection is analytical" begin
        Nx, Ny = 8, 6
        dt = 0.25
        u0 = 0.2
        ux = fill(u0, Nx, Ny)
        uy = zeros(Float64, Nx, Ny)
        is_solid = fill(false, Nx, Ny)
        ux_west = fill(u0, Ny)
        ux_east = fill(u0, Ny)
        ux_face = zeros(Float64, Nx + 1, Ny)
        uy_face = zeros(Float64, Nx, Ny + 1)

        Kraken.logfv_cell_velocity_to_faces_openx_solid_aware_2d!(
            ux_face, uy_face, ux, uy, is_solid, ux_west, ux_east,
        )
        @test all(ux_face .≈ u0)
        @test all(uy_face .≈ 0.0)

        axx, axy, ayy = 0.03, -0.02, 0.01
        bxx, bxy, byy = 0.2, -0.1, 0.05
        psixx = [bxx + axx * i for i in 1:Nx, j in 1:Ny]
        psixy = [bxy + axy * i for i in 1:Nx, j in 1:Ny]
        psiyy = [byy + ayy * i for i in 1:Nx, j in 1:Ny]
        west_xx = fill(bxx, Ny)
        west_xy = fill(bxy, Ny)
        west_yy = fill(byy, Ny)
        east_xx = fill(bxx + axx * (Nx + 1), Ny)
        east_xy = fill(bxy + axy * (Nx + 1), Ny)
        east_yy = fill(byy + ayy * (Nx + 1), Ny)
        outxx = similar(psixx)
        outxy = similar(psixy)
        outyy = similar(psiyy)

        Kraken.logfv_advect_upwind_openx_solid_aware_2d!(
            outxx, outxy, outyy,
            psixx, psixy, psiyy,
            west_xx, west_xy, west_yy,
            east_xx, east_xy, east_yy,
            ux_face, uy_face, is_solid, dt,
        )
        KernelAbstractions.synchronize(KernelAbstractions.CPU())

        for j in 2:(Ny - 1), i in 1:Nx
            @test outxx[i, j] ≈ psixx[i, j] - dt * u0 * axx atol=2e-14 rtol=2e-14
            @test outxy[i, j] ≈ psixy[i, j] - dt * u0 * axy atol=2e-14 rtol=2e-14
            @test outyy[i, j] ≈ psiyy[i, j] - dt * u0 * ayy atol=2e-14 rtol=2e-14
        end

        geom = Kraken.backward_facing_step_geometry_2d(;
            H_in=4, expansion_ratio=2, L_up=2, L_down=3, FT=Float64,
        )
        Nx2, Ny2 = geom.Nx, geom.Ny
        ux2 = [geom.is_solid[i, j] ? 0.0 : 0.015 for i in 1:Nx2, j in 1:Ny2]
        uy2 = zeros(Float64, Nx2, Ny2)
        ux_west2 = fill(0.015, Ny2)
        ux_east2 = fill(0.015, Ny2)
        ux_face2 = zeros(Float64, Nx2 + 1, Ny2)
        uy_face2 = zeros(Float64, Nx2, Ny2 + 1)
        Kraken.logfv_cell_velocity_to_faces_openx_solid_aware_2d!(
            ux_face2, uy_face2, ux2, uy2, geom.is_solid, ux_west2, ux_east2,
        )

        cxx, cxy, cyy = 0.25, -0.03, 0.11
        psixx2 = fill(cxx, Nx2, Ny2)
        psixy2 = fill(cxy, Nx2, Ny2)
        psiyy2 = fill(cyy, Nx2, Ny2)
        outxx2 = similar(psixx2)
        outxy2 = similar(psixy2)
        outyy2 = similar(psiyy2)
        Kraken.logfv_advect_upwind_openx_solid_aware_2d!(
            outxx2, outxy2, outyy2,
            psixx2, psixy2, psiyy2,
            fill(cxx, Ny2), fill(cxy, Ny2), fill(cyy, Ny2),
            fill(cxx, Ny2), fill(cxy, Ny2), fill(cyy, Ny2),
            ux_face2, uy_face2, geom.is_solid, 0.2,
        )
        KernelAbstractions.synchronize(KernelAbstractions.CPU())

        for j in 1:Ny2, i in 1:Nx2
            if geom.is_solid[i, j]
                @test outxx2[i, j] == 0.0
                @test outxy2[i, j] == 0.0
                @test outyy2[i, j] == 0.0
            else
                @test outxx2[i, j] ≈ cxx atol=2e-14 rtol=2e-14
                @test outxy2[i, j] ≈ cxy atol=2e-14 rtol=2e-14
                @test outyy2[i, j] ≈ cyy atol=2e-14 rtol=2e-14
            end
        end
    end

    @testset "M8d BFS passive log-FV polymer pipeline stays SPD" begin
        result = Kraken.run_viscoelastic_logfv_bfs_passive_2d(;
            H_in=4, expansion_ratio=2, L_up=2, L_down=4,
            nu_s=0.08, nu_p=0.02, lambda=5.0,
            u_mean=0.01, Fx_body=2e-7,
            hydro_steps=60, polymer_steps=20,
            backend=KernelAbstractions.CPU(), T=Float64,
        )

        @test result.min_c_eig > 0.9
        @test result.max_abs_psi > 0
        @test result.max_abs_psi < 0.12
        @test result.max_abs_tau > 0
        @test result.max_abs_tau < 1e-3
        @test result.max_speed > 1e-4
        @test result.rho_min > 0.98
        @test result.rho_max < 1.02
        @test all(isfinite, result.psixx)
        @test all(isfinite, result.psixy)
        @test all(isfinite, result.psiyy)
        @test all(isfinite, result.tauxx)
        @test all(isfinite, result.tauxy)
        @test all(isfinite, result.tauyy)
    end

    @testset "M8e BFS coupled log-FV feedback is hydro-consistent and bounded" begin
        hydro = Kraken.run_viscoelastic_logfv_bfs_passive_2d(;
            H_in=4, expansion_ratio=2, L_up=2, L_down=4,
            nu_s=0.08, nu_p=0.0, lambda=5.0,
            u_mean=0.01, Fx_body=2e-7,
            hydro_steps=40, polymer_steps=0,
            backend=KernelAbstractions.CPU(), T=Float64,
        )
        zero_poly = Kraken.run_viscoelastic_logfv_bfs_coupled_2d(;
            H_in=4, expansion_ratio=2, L_up=2, L_down=4,
            nu_s=0.08, nu_p=0.0, lambda=5.0,
            u_mean=0.01, Fx_body=2e-7,
            bsd_fraction=1.0, max_steps=40,
            backend=KernelAbstractions.CPU(), T=Float64,
        )
        fluid = .!zero_poly.is_solid
        @test zero_poly.nu_lbm ≈ zero_poly.nu_s
        @test zero_poly.max_abs_tau == 0.0
        @test maximum(abs.(zero_poly.ux[fluid] .- hydro.ux[fluid])) < 1e-12
        @test maximum(abs.(zero_poly.uy[fluid] .- hydro.uy[fluid])) < 1e-12
        @test maximum(abs.(zero_poly.rho[fluid] .- hydro.rho[fluid])) < 1e-12

        coupled = Kraken.run_viscoelastic_logfv_bfs_coupled_2d(;
            H_in=4, expansion_ratio=2, L_up=2, L_down=4,
            nu_s=0.08, nu_p=0.02, lambda=5.0,
            u_mean=0.01, Fx_body=2e-7,
            bsd_fraction=1.0, max_steps=40,
            backend=KernelAbstractions.CPU(), T=Float64,
        )

        @test coupled.bsd_fraction == 1.0
        @test coupled.nu_lbm ≈ 0.10
        @test coupled.max_abs_tau > 0
        @test coupled.max_abs_poly_force > 0
        @test coupled.max_abs_total_force > 0
        @test coupled.min_c_eig > 0.9
        @test coupled.max_abs_psi < 0.2
        @test coupled.max_abs_tau < 2e-3
        @test coupled.max_speed > 1e-4
        @test coupled.max_speed < 0.05
        @test coupled.rho_min > 0.98
        @test coupled.rho_max < 1.02
        @test all(isfinite, coupled.ux)
        @test all(isfinite, coupled.uy)
        @test all(isfinite, coupled.psixx)
        @test all(isfinite, coupled.tauxx)
        @test all(isfinite, coupled.fx_total)
    end

    @testset "M8f BFS coupled low-beta sweep stays SPD and bounded" begin
        cases = (
            (nu_s=0.005, nu_p=0.095, lambda=10.0, u_mean=0.005, Fx_body=1e-7),
            (nu_s=0.002, nu_p=0.098, lambda=50.0, u_mean=0.005, Fx_body=1e-7),
            (nu_s=0.002, nu_p=0.098, lambda=200.0, u_mean=0.003, Fx_body=5e-8),
        )

        for case in cases
            result = Kraken.run_viscoelastic_logfv_bfs_coupled_2d(;
                H_in=4, expansion_ratio=2, L_up=2, L_down=4,
                nu_s=case.nu_s, nu_p=case.nu_p, lambda=case.lambda,
                u_mean=case.u_mean, Fx_body=case.Fx_body,
                bsd_fraction=1.0, max_steps=30,
                backend=KernelAbstractions.CPU(), T=Float64,
            )
            beta = result.nu_s / result.nu_total

            @test beta <= 0.05
            @test result.bsd_fraction == 1.0
            @test result.nu_lbm ≈ result.nu_total
            @test result.polymer_substeps >= 1
            @test !result.subcycle_estimate.clamped
            @test result.min_c_eig > 0.8
            @test result.max_abs_psi < 0.25
            @test result.max_abs_tau < 1.5e-3
            @test result.max_speed > 1e-4
            @test result.max_speed < 0.02
            @test result.rho_min > 0.995
            @test result.rho_max < 1.01
            @test all(isfinite, result.ux)
            @test all(isfinite, result.uy)
            @test all(isfinite, result.psixx)
            @test all(isfinite, result.psixy)
            @test all(isfinite, result.psiyy)
            @test all(isfinite, result.fx_total)
            @test all(isfinite, result.fy_total)
        end
    end

    @testset "M8g BFS coupled low-beta duration ramp stays bounded" begin
        result = Kraken.run_viscoelastic_logfv_bfs_coupled_2d(;
            H_in=4, expansion_ratio=2, L_up=2, L_down=4,
            nu_s=0.002, nu_p=0.098, lambda=50.0,
            u_mean=0.003, Fx_body=5e-8,
            bsd_fraction=1.0, max_steps=80,
            backend=KernelAbstractions.CPU(), T=Float64,
        )

        @test result.nu_s / result.nu_total <= 0.02
        @test result.nu_lbm ≈ result.nu_total
        @test result.min_c_eig > 0.8
        @test result.max_abs_psi < 0.3
        @test result.max_abs_tau < 7e-4
        @test result.max_abs_poly_force > 0
        @test result.max_abs_total_force > 0
        @test result.max_speed > 1e-4
        @test result.max_speed < 0.02
        @test result.rho_min > 0.995
        @test result.rho_max < 1.01
        @test all(isfinite, result.ux)
        @test all(isfinite, result.uy)
        @test all(isfinite, result.psixx)
        @test all(isfinite, result.fx_total)
    end

    @testset "M8g2 BFS coupled low-beta long-relaxation ramp stays bounded" begin
        result = Kraken.run_viscoelastic_logfv_bfs_coupled_2d(;
            H_in=4, expansion_ratio=2, L_up=2, L_down=4,
            nu_s=0.002, nu_p=0.098, lambda=200.0,
            u_mean=0.003, Fx_body=5e-8,
            bsd_fraction=1.0, max_steps=60,
            backend=KernelAbstractions.CPU(), T=Float64,
        )

        @test result.nu_s / result.nu_total <= 0.02
        @test result.nu_lbm ≈ result.nu_total
        @test result.polymer_substeps == 1
        @test !result.subcycle_estimate.clamped
        @test result.min_c_eig > 0.75
        @test result.max_abs_psi < 0.3
        @test result.max_abs_tau < 2e-4
        @test result.max_abs_poly_force > 0
        @test result.max_abs_total_force > 0
        @test result.max_speed > 1e-4
        @test result.max_speed < 0.02
        @test result.rho_min > 0.995
        @test result.rho_max < 1.01
        @test all(isfinite, result.ux)
        @test all(isfinite, result.uy)
        @test all(isfinite, result.psixx)
        @test all(isfinite, result.fx_total)
    end

    @testset "M8h BFS coupled near-Newtonian limit matches total-viscosity hydro" begin
        hydro = Kraken.run_viscoelastic_logfv_bfs_passive_2d(;
            H_in=4, expansion_ratio=2, L_up=2, L_down=4,
            nu_s=0.10, nu_p=0.0, lambda=1.0,
            u_mean=0.01, Fx_body=2e-7,
            hydro_steps=30, polymer_steps=0,
            backend=KernelAbstractions.CPU(), T=Float64,
        )
        visco = Kraken.run_viscoelastic_logfv_bfs_coupled_2d(;
            H_in=4, expansion_ratio=2, L_up=2, L_down=4,
            nu_s=0.08, nu_p=0.02, lambda=1.0,
            u_mean=0.01, Fx_body=2e-7,
            bsd_fraction=1.0, max_steps=30,
            backend=KernelAbstractions.CPU(), T=Float64,
        )
        fluid = .!visco.is_solid

        @test visco.nu_lbm ≈ hydro.nu_s
        @test visco.polymer_substeps == 50
        @test !visco.subcycle_estimate.clamped
        @test visco.min_c_eig > 0.98
        @test visco.max_abs_psi < 0.03
        @test maximum(abs.(visco.ux[fluid] .- hydro.ux[fluid])) < 2e-4
        @test maximum(abs.(visco.uy[fluid] .- hydro.uy[fluid])) < 2e-4
        @test maximum(abs.(visco.rho[fluid] .- hydro.rho[fluid])) < 3e-4
        @test all(isfinite, visco.ux)
        @test all(isfinite, visco.psixx)
        @test all(isfinite, visco.fx_total)
    end

    @testset "M9 cylinder channel coupled log-FV coarse canary stays bounded" begin
        result = Kraken.run_viscoelastic_logfv_cylinder_coupled_2d(;
            radius=4.0, H=18, L_up=4, L_down=7,
            nu_s=0.08, nu_p=0.02, lambda=5.0,
            u_mean=0.006, Fx_body=1e-7,
            bsd_fraction=1.0, max_steps=50,
            backend=KernelAbstractions.CPU(), T=Float64,
        )
        fluid = .!result.is_solid

        @test result.geometry.name === :cylinder
        @test count(fluid) > 0
        @test result.nu_lbm ≈ result.nu_total
        @test result.polymer_substeps >= 1
        @test !result.subcycle_estimate.clamped
        @test result.min_c_eig > 0.7
        @test result.max_abs_psi < 0.4
        @test result.max_abs_tau < 5e-4
        @test result.max_abs_poly_force > 0
        @test result.max_abs_total_force > 0
        @test result.max_speed > 1e-4
        @test result.max_speed < 0.02
        @test result.rho_min > 0.995
        @test result.rho_max < 1.02
        @test all(isfinite, result.ux[fluid])
        @test all(isfinite, result.uy[fluid])
        @test all(isfinite, result.psixx[fluid])
        @test all(isfinite, result.fx_total[fluid])
    end
end
