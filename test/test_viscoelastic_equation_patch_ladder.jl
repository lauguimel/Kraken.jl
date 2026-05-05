using Test
using LinearAlgebra
using Statistics
using Kraken

const P0_ATOL = 1e-12

_direct_source_tuple(cxx, cxy, cyy, dudx, dudy, dvdx, dvdy, λ) = (
    conformation_source_2d(cxx, cxy, cyy, dudx, dudy, dvdx, dvdy, λ, 1),
    conformation_source_2d(cxx, cxy, cyy, dudx, dudy, dvdx, dvdy, λ, 2),
    conformation_source_2d(cxx, cxy, cyy, dudx, dudy, dvdx, dvdy, λ, 3),
)

_logconf_source_tuple(ψxx, ψxy, ψyy, dudx, dudy, dvdx, dvdy, λ) = (
    logconf_source_2d(ψxx, ψxy, ψyy, dudx, dudy, dvdx, dvdy, λ, 1),
    logconf_source_2d(ψxx, ψxy, ψyy, dudx, dudy, dvdx, dvdy, λ, 2),
    logconf_source_2d(ψxx, ψxy, ψyy, dudx, dudy, dvdx, dvdy, λ, 3),
)

function _log_spd_2x2(cxx, cxy, cyy)
    trace_c = cxx + cyy
    diff_c = cxx - cyy
    disc = sqrt(diff_c^2 + 4cxy^2)
    λ1 = 0.5 * (trace_c + disc)
    λ2 = 0.5 * (trace_c - disc)
    θ = 0.5 * atan(2cxy, diff_c)
    c = cos(θ)
    s = sin(θ)
    log_λ1 = log(λ1)
    log_λ2 = log(λ2)
    return (
        c^2 * log_λ1 + s^2 * log_λ2,
        c * s * (log_λ1 - log_λ2),
        s^2 * log_λ1 + c^2 * log_λ2,
    )
end

function _stationary_direct_conformation_incompressible(dudx, dudy, dvdx, dvdy, λ)
    @assert abs(dudx + dvdy) < 1e-14
    system = [
        1 / λ - 2dudx  -2dudy             0.0
        -dvdx          1 / λ - dudx - dvdy -dudy
        0.0            -2dvdx             1 / λ - 2dvdy
    ]
    rhs = [1 / λ, 0.0, 1 / λ]
    cxx, cxy, cyy = system \ rhs
    @assert cxx > 0
    @assert cxx * cyy - cxy^2 > 0
    return cxx, cxy, cyy
end

function _min_eig_spd_2x2(cxx, cxy, cyy)
    tr = cxx + cyy
    diff = cxx - cyy
    disc = sqrt(diff^2 + 4cxy^2)
    return 0.5 * (tr - disc)
end

@testset "P0 direct conformation source: no-flow relaxation" begin
    λ = 2.3
    cxx, cxy, cyy = 1.4, -0.2, 0.7
    source = _direct_source_tuple(cxx, cxy, cyy, 0.0, 0.0, 0.0, 0.0, λ)
    @test source[1] ≈ -(cxx - 1.0) / λ atol=P0_ATOL
    @test source[2] ≈ -cxy / λ atol=P0_ATOL
    @test source[3] ≈ -(cyy - 1.0) / λ atol=P0_ATOL
end

@testset "P0 conservative source: isotropic divergence term" begin
    λ = 7.0
    a = 0.013
    cxx, cxy, cyy = 1.3, -0.17, 0.8
    source = _direct_source_tuple(cxx, cxy, cyy, a, 0.0, 0.0, a, λ)
    @test source[1] ≈ -(cxx - 1.0) / λ + 4a * cxx atol=P0_ATOL
    @test source[2] ≈ -cxy / λ + 4a * cxy atol=P0_ATOL
    @test source[3] ≈ -(cyy - 1.0) / λ + 4a * cyy atol=P0_ATOL

    cxx_d, cyy_d = 1.3, 0.8
    ψxx, ψxy, ψyy = log(cxx_d), 0.0, log(cyy_d)
    log_source = _logconf_source_tuple(ψxx, ψxy, ψyy, a, 0.0, 0.0, a, λ)
    @test log_source[1] ≈ 2a - (1.0 - inv(cxx_d)) / λ + 2a * ψxx atol=P0_ATOL
    @test log_source[2] ≈ 0.0 atol=P0_ATOL
    @test log_source[3] ≈ 2a - (1.0 - inv(cyy_d)) / λ + 2a * ψyy atol=P0_ATOL
end

function _direct_source_euler(; λ, dudx, dudy, dvdx, dvdy, steps)
    cxx, cxy, cyy = 1.0, 0.0, 1.0
    min_eig = _min_eig_spd_2x2(cxx, cxy, cyy)
    for _ in 1:steps
        sxx, sxy, syy = _direct_source_tuple(cxx, cxy, cyy, dudx, dudy, dvdx, dvdy, λ)
        cxx += sxx
        cxy += sxy
        cyy += syy
        min_eig = min(min_eig, _min_eig_spd_2x2(cxx, cxy, cyy))
        isfinite(cxx) && isfinite(cxy) && isfinite(cyy) || break
    end
    return (; cxx, cxy, cyy, min_eig)
end

@testset "P0 direct conformation source: low-Wi long-time SPD canary" begin
    λ = 600.0
    cases = (
        (name=:shear_x_from_y, dudx=0.0, dudy=2.0e-4, dvdx=0.0, dvdy=0.0),
        (name=:shear_y_from_x, dudx=0.0, dudy=0.0, dvdx=2.0e-4, dvdy=0.0),
        (name=:extension_x, dudx=3.0e-4, dudy=0.0, dvdx=0.0, dvdy=-3.0e-4),
        (name=:extension_y, dudx=-3.0e-4, dudy=0.0, dvdx=0.0, dvdy=3.0e-4),
        (name=:mixed, dudx=1.0e-4, dudy=2.0e-4, dvdx=-1.0e-4, dvdy=-1.0e-4),
    )
    for case in cases
        result = _direct_source_euler(
            λ=λ, dudx=case.dudx, dudy=case.dudy,
            dvdx=case.dvdx, dvdy=case.dvdy, steps=20_000,
        )
        @test isfinite(result.cxx)
        @test isfinite(result.cxy)
        @test isfinite(result.cyy)
        @test result.min_eig > 0.0
        cxx, cxy, cyy = _stationary_direct_conformation_incompressible(
            case.dudx, case.dudy, case.dvdx, case.dvdy, λ,
        )
        @test result.cxx ≈ cxx rtol=1e-6 atol=1e-10
        @test result.cxy ≈ cxy rtol=1e-6 atol=1e-10
        @test result.cyy ≈ cyy rtol=1e-6 atol=1e-10
    end
end

@testset "P0 velocity-gradient stencils: centered, one-sided, and blocked" begin
    Nx, Ny = 7, 8
    α, β, c0 = 0.23, -0.17, 1.4
    field = [c0 + α * (i - 1) + β * (j - 1) for i in 1:Nx, j in 1:Ny]
    solid = falses(Nx, Ny)

    @test Kraken._wall_aware_dx_2d(field, solid, 4, 4, Nx, Float64) ≈ α atol=P0_ATOL
    @test Kraken._wall_aware_dy_2d(field, solid, 4, 4, Ny, Float64) ≈ β atol=P0_ATOL
    @test Kraken._wall_aware_dx_2d(field, solid, 1, 4, Nx, Float64) ≈ α atol=P0_ATOL
    @test Kraken._wall_aware_dx_2d(field, solid, Nx, 4, Nx, Float64) ≈ α atol=P0_ATOL
    @test Kraken._wall_aware_dy_2d(field, solid, 4, 1, Ny, Float64) ≈ β atol=P0_ATOL
    @test Kraken._wall_aware_dy_2d(field, solid, 4, Ny, Ny, Float64) ≈ β atol=P0_ATOL

    solid[5, 4] = true
    @test Kraken._wall_aware_dx_2d(field, solid, 4, 4, Nx, Float64) ≈ α atol=P0_ATOL
    solid[5, 4] = false
    solid[3, 4] = true
    @test Kraken._wall_aware_dx_2d(field, solid, 4, 4, Nx, Float64) ≈ α atol=P0_ATOL
    solid[5, 4] = true
    @test Kraken._wall_aware_dx_2d(field, solid, 4, 4, Nx, Float64) == 0.0

    solid .= false
    solid[4, 5] = true
    @test Kraken._wall_aware_dy_2d(field, solid, 4, 4, Ny, Float64) ≈ β atol=P0_ATOL
    solid[4, 5] = false
    solid[4, 3] = true
    @test Kraken._wall_aware_dy_2d(field, solid, 4, 4, Ny, Float64) ≈ β atol=P0_ATOL
    solid[4, 5] = true
    @test Kraken._wall_aware_dy_2d(field, solid, 4, 4, Ny, Float64) == 0.0
end

@testset "P0 straight-wall quadratic gradients use second-order one-sided stencils" begin
    Nx, Ny = 9, 10
    ax, bx = 0.11, -0.017
    ay, by = -0.07, 0.021
    field_x = [1.2 + ax * (i - 1) + bx * (i - 1)^2 for i in 1:Nx, j in 1:Ny]
    field_y = [0.4 + ay * (j - 1) + by * (j - 1)^2 for i in 1:Nx, j in 1:Ny]
    solid = falses(Nx, Ny)

    solid[3, 5] = true
    @test Kraken._wall_aware_dx_2d(field_x, solid, 4, 5, Nx, Float64) ≈
          ax + 2bx * (4 - 1) atol=P0_ATOL
    solid[3, 5] = false
    solid[5, 5] = true
    @test Kraken._wall_aware_dx_2d(field_x, solid, 4, 5, Nx, Float64) ≈
          ax + 2bx * (4 - 1) atol=P0_ATOL

    solid .= false
    solid[5, 3] = true
    @test Kraken._wall_aware_dy_2d(field_y, solid, 5, 4, Ny, Float64) ≈
          ay + 2by * (4 - 1) atol=P0_ATOL
    solid[5, 3] = false
    solid[5, 5] = true
    @test Kraken._wall_aware_dy_2d(field_y, solid, 5, 4, Ny, Float64) ≈
          ay + 2by * (4 - 1) atol=P0_ATOL
end

@testset "P0 direct conformation source: stationary simple shear x<-y" begin
    λ = 4.0
    γ = 0.03
    wi = λ * γ
    cxx, cxy, cyy = 1.0 + 2wi^2, wi, 1.0
    source = _direct_source_tuple(cxx, cxy, cyy, 0.0, γ, 0.0, 0.0, λ)
    @test source[1] ≈ 0.0 atol=P0_ATOL
    @test source[2] ≈ 0.0 atol=P0_ATOL
    @test source[3] ≈ 0.0 atol=P0_ATOL
end

@testset "P0 direct conformation source: stationary simple shear y<-x" begin
    λ = 4.0
    γ = 0.03
    wi = λ * γ
    cxx, cxy, cyy = 1.0, wi, 1.0 + 2wi^2
    source = _direct_source_tuple(cxx, cxy, cyy, 0.0, 0.0, γ, 0.0, λ)
    @test source[1] ≈ 0.0 atol=P0_ATOL
    @test source[2] ≈ 0.0 atol=P0_ATOL
    @test source[3] ≈ 0.0 atol=P0_ATOL
end

@testset "P0 direct conformation source: stationary planar extension" begin
    λ = 1.7
    ε = 0.08
    cxx = 1.0 / (1.0 - 2λ * ε)
    cyy = 1.0 / (1.0 + 2λ * ε)
    source = _direct_source_tuple(cxx, 0.0, cyy, ε, 0.0, 0.0, -ε, λ)
    @test source[1] ≈ 0.0 atol=P0_ATOL
    @test source[2] ≈ 0.0 atol=P0_ATOL
    @test source[3] ≈ 0.0 atol=P0_ATOL
end

@testset "P0 direct conformation source: pure rotation leaves identity unchanged" begin
    Ω = 0.07
    source = _direct_source_tuple(1.0, 0.0, 1.0, 0.0, -Ω, Ω, 0.0, 2.0)
    @test source[1] ≈ 0.0 atol=P0_ATOL
    @test source[2] ≈ 0.0 atol=P0_ATOL
    @test source[3] ≈ 0.0 atol=P0_ATOL
end

@testset "P0 direct conformation source: arbitrary incompressible stationary gradients" begin
    λ = 3.0
    gradients = (
        (0.012, 0.035, -0.018, -0.012),
        (-0.014, 0.022, 0.031, 0.014),
        (0.0, 0.04, -0.015, 0.0),
        (0.02, 0.015, 0.01, -0.02),
    )
    for (dudx, dudy, dvdx, dvdy) in gradients
        cxx, cxy, cyy = _stationary_direct_conformation_incompressible(
            dudx, dudy, dvdx, dvdy, λ,
        )
        source = _direct_source_tuple(cxx, cxy, cyy, dudx, dudy, dvdx, dvdy, λ)
        @test source[1] ≈ 0.0 atol=P0_ATOL
        @test source[2] ≈ 0.0 atol=P0_ATOL
        @test source[3] ≈ 0.0 atol=P0_ATOL
    end
end

@testset "P0 log-conformation source: identity limits" begin
    γ = 0.03
    ε = 0.04
    Ω = 0.07

    source_xy = _logconf_source_tuple(0.0, 0.0, 0.0, 0.0, γ, 0.0, 0.0, 2.0)
    @test source_xy[1] ≈ 0.0 atol=P0_ATOL
    @test source_xy[2] ≈ γ atol=P0_ATOL
    @test source_xy[3] ≈ 0.0 atol=P0_ATOL

    source_yx = _logconf_source_tuple(0.0, 0.0, 0.0, 0.0, 0.0, γ, 0.0, 2.0)
    @test source_yx[1] ≈ 0.0 atol=P0_ATOL
    @test source_yx[2] ≈ γ atol=P0_ATOL
    @test source_yx[3] ≈ 0.0 atol=P0_ATOL

    source_ext = _logconf_source_tuple(0.0, 0.0, 0.0, ε, 0.0, 0.0, -ε, 2.0)
    @test source_ext[1] ≈ 2ε atol=P0_ATOL
    @test source_ext[2] ≈ 0.0 atol=P0_ATOL
    @test source_ext[3] ≈ -2ε atol=P0_ATOL

    source_rot = _logconf_source_tuple(0.0, 0.0, 0.0, 0.0, -Ω, Ω, 0.0, 2.0)
    @test source_rot[1] ≈ 0.0 atol=P0_ATOL
    @test source_rot[2] ≈ 0.0 atol=P0_ATOL
    @test source_rot[3] ≈ 0.0 atol=P0_ATOL
end

@testset "P0 log-conformation source: stationary simple shear x<-y" begin
    λ = 4.0
    γ = 0.03
    wi = λ * γ
    ψxx, ψxy, ψyy = _log_spd_2x2(1.0 + 2wi^2, wi, 1.0)
    source = _logconf_source_tuple(ψxx, ψxy, ψyy, 0.0, γ, 0.0, 0.0, λ)
    @test source[1] ≈ 0.0 atol=P0_ATOL
    @test source[2] ≈ 0.0 atol=P0_ATOL
    @test source[3] ≈ 0.0 atol=P0_ATOL
end

@testset "P0 log-conformation source: stationary simple shear y<-x" begin
    λ = 4.0
    γ = 0.03
    wi = λ * γ
    ψxx, ψxy, ψyy = _log_spd_2x2(1.0, wi, 1.0 + 2wi^2)
    source = _logconf_source_tuple(ψxx, ψxy, ψyy, 0.0, 0.0, γ, 0.0, λ)
    @test source[1] ≈ 0.0 atol=P0_ATOL
    @test source[2] ≈ 0.0 atol=P0_ATOL
    @test source[3] ≈ 0.0 atol=P0_ATOL
end

@testset "P0 log-conformation source: stationary planar extension" begin
    λ = 1.7
    ε = 0.08
    ψxx = log(1.0 / (1.0 - 2λ * ε))
    ψyy = log(1.0 / (1.0 + 2λ * ε))
    source = _logconf_source_tuple(ψxx, 0.0, ψyy, ε, 0.0, 0.0, -ε, λ)
    @test source[1] ≈ 0.0 atol=P0_ATOL
    @test source[2] ≈ 0.0 atol=P0_ATOL
    @test source[3] ≈ 0.0 atol=P0_ATOL
end

@testset "P0 log-conformation source: stationary planar extension y-axis" begin
    λ = 1.7
    ε = 0.08
    ψxx = log(1.0 / (1.0 + 2λ * ε))
    ψyy = log(1.0 / (1.0 - 2λ * ε))
    source = _logconf_source_tuple(ψxx, 0.0, ψyy, -ε, 0.0, 0.0, ε, λ)
    @test source[1] ≈ 0.0 atol=P0_ATOL
    @test source[2] ≈ 0.0 atol=P0_ATOL
    @test source[3] ≈ 0.0 atol=P0_ATOL
end

@testset "P0 log-conformation source: arbitrary incompressible stationary gradients" begin
    λ = 3.0
    gradients = (
        (0.012, 0.035, -0.018, -0.012),
        (-0.014, 0.022, 0.031, 0.014),
        (0.0, 0.04, -0.015, 0.0),
        (0.02, 0.015, 0.01, -0.02),
    )
    for (dudx, dudy, dvdx, dvdy) in gradients
        cxx, cxy, cyy = _stationary_direct_conformation_incompressible(
            dudx, dudy, dvdx, dvdy, λ,
        )
        ψxx, ψxy, ψyy = _log_spd_2x2(cxx, cxy, cyy)
        source = _logconf_source_tuple(ψxx, ψxy, ψyy, dudx, dudy, dvdx, dvdy, λ)
        @test source[1] ≈ 0.0 atol=2e-12
        @test source[2] ≈ 0.0 atol=2e-12
        @test source[3] ≈ 0.0 atol=2e-12
    end
end

const D2Q9_CX = (0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0)
const D2Q9_CY = (0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0)
const D2Q9_W = (4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36)

function _hermite_delta_moments(txx, txy, tyy, s_plus; ce_correction=false)
    f = zeros(Float64, 1, 1, 9)
    is_solid = falses(1, 1)
    tau_xx = fill(txx, 1, 1)
    tau_xy = fill(txy, 1, 1)
    tau_yy = fill(tyy, 1, 1)
    apply_hermite_source_2d!(
        f, is_solid, s_plus, tau_xx, tau_xy, tau_yy;
        ce_correction,
    )
    mass = sum(f[1, 1, q] for q in 1:9)
    mom_x = sum(D2Q9_CX[q] * f[1, 1, q] for q in 1:9)
    mom_y = sum(D2Q9_CY[q] * f[1, 1, q] for q in 1:9)
    m_xx = sum(D2Q9_CX[q]^2 * f[1, 1, q] for q in 1:9)
    m_xy = sum(D2Q9_CX[q] * D2Q9_CY[q] * f[1, 1, q] for q in 1:9)
    m_yy = sum(D2Q9_CY[q]^2 * f[1, 1, q] for q in 1:9)
    return mass, mom_x, mom_y, m_xx, m_xy, m_yy
end

@testset "P1 Hermite polymer source moments: τxx orientation" begin
    s_plus = 0.8
    txx = 0.12
    mass, mom_x, mom_y, m_xx, m_xy, m_yy =
        _hermite_delta_moments(txx, 0.0, 0.0, s_plus)
    @test mass ≈ 0.0 atol=P0_ATOL
    @test mom_x ≈ 0.0 atol=P0_ATOL
    @test mom_y ≈ 0.0 atol=P0_ATOL
    @test m_xx ≈ -s_plus * txx atol=P0_ATOL
    @test m_xy ≈ 0.0 atol=P0_ATOL
    @test m_yy ≈ 0.0 atol=P0_ATOL
end

@testset "P1 Hermite polymer source moments: τyy orientation" begin
    s_plus = 0.8
    tyy = -0.05
    mass, mom_x, mom_y, m_xx, m_xy, m_yy =
        _hermite_delta_moments(0.0, 0.0, tyy, s_plus)
    @test mass ≈ 0.0 atol=P0_ATOL
    @test mom_x ≈ 0.0 atol=P0_ATOL
    @test mom_y ≈ 0.0 atol=P0_ATOL
    @test m_xx ≈ 0.0 atol=P0_ATOL
    @test m_xy ≈ 0.0 atol=P0_ATOL
    @test m_yy ≈ -s_plus * tyy atol=P0_ATOL
end

@testset "P1 Hermite polymer source moments: τxy orientation" begin
    s_plus = 0.8
    txy = 0.07
    mass, mom_x, mom_y, m_xx, m_xy, m_yy =
        _hermite_delta_moments(0.0, txy, 0.0, s_plus)
    @test mass ≈ 0.0 atol=P0_ATOL
    @test mom_x ≈ 0.0 atol=P0_ATOL
    @test mom_y ≈ 0.0 atol=P0_ATOL
    @test m_xx ≈ 0.0 atol=P0_ATOL
    @test m_xy ≈ -s_plus * txy atol=P0_ATOL
    @test m_yy ≈ 0.0 atol=P0_ATOL
end

@testset "P1 Hermite polymer source moments: CE scaling" begin
    s_plus = 0.7
    scale = inv(1.0 - s_plus / 2.0)
    txx, txy, tyy = 0.12, -0.04, -0.03
    mass, mom_x, mom_y, m_xx, m_xy, m_yy =
        _hermite_delta_moments(txx, txy, tyy, s_plus; ce_correction=true)
    @test mass ≈ 0.0 atol=P0_ATOL
    @test mom_x ≈ 0.0 atol=P0_ATOL
    @test mom_y ≈ 0.0 atol=P0_ATOL
    @test m_xx ≈ -s_plus * scale * txx atol=P0_ATOL
    @test m_xy ≈ -s_plus * scale * txy atol=P0_ATOL
    @test m_yy ≈ -s_plus * scale * tyy atol=P0_ATOL
end

@testset "P2 D2Q9 streaming: single interior link orientations" begin
    Nx, Ny = 7, 7
    i0, j0 = 4, 4
    for q in 2:9
        f_in = zeros(Float64, Nx, Ny, 9)
        f_out = similar(f_in)
        value = 10.0 + q
        f_in[i0, j0, q] = value
        stream_2d!(f_out, f_in, Nx, Ny; sync=true)

        i1 = i0 + Int(D2Q9_CX[q])
        j1 = j0 + Int(D2Q9_CY[q])
        @test f_out[i1, j1, q] == value
        @test sum(f_out) == value
    end
end

function _single_missing_link_state(q_missing)
    Nx, Ny = 7, 7
    i0, j0 = 4, 4
    g_pre = zeros(Float64, Nx, Ny, 9)
    g_post = zeros(Float64, Nx, Ny, 9)
    for i in 1:Nx, j in 1:Ny, q in 1:9
        g_pre[i, j, q] = 0.01i + 0.02j + 0.001q
        g_post[i, j, q] = 0.03i + 0.04j + 0.002q
    end
    is_solid = falses(Nx, Ny)
    is_solid[i0 - Int(D2Q9_CX[q_missing]), j0 - Int(D2Q9_CY[q_missing])] = true
    C_field = fill(1.25, Nx, Ny)
    return g_post, g_pre, is_solid, C_field, i0, j0
end

function _strict_single_link_expected(g_post0, g_pre, i0, j0, q_missing)
    q_out = if q_missing == 2
        4
    elseif q_missing == 3
        5
    elseif q_missing == 4
        2
    elseif q_missing == 5
        3
    elseif q_missing == 6
        8
    elseif q_missing == 7
        9
    elseif q_missing == 8
        6
    else
        7
    end
    φ = g_post0[i0, j0, 1]
    for q in 2:9
        φ += q == q_missing ? g_pre[i0, j0, q_out] : g_post0[i0, j0, q]
    end
    expected = [g_post0[i0, j0, q] for q in 1:9]
    expected[q_missing] = g_post0[i0, j0, q_out]
    expected[1] = φ - sum(expected[q] for q in 2:9)
    return φ, expected
end

function _opp_local(q)
    q == 1 && return 1
    q == 2 && return 4
    q == 3 && return 5
    q == 4 && return 2
    q == 5 && return 3
    q == 6 && return 8
    q == 7 && return 9
    q == 8 && return 6
    return 7
end

@testset "P3 CNEBB strict: single missing link orientations" begin
    for q_missing in 2:9
        g_post, g_pre, is_solid, C_field, i0, j0 =
            _single_missing_link_state(q_missing)
        g_post0 = copy(g_post)
        φ_expected, g_expected =
            _strict_single_link_expected(g_post0, g_pre, i0, j0, q_missing)

        apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, C_field)

        @test C_field[i0, j0] ≈ φ_expected atol=P0_ATOL
        for q in 1:9
            @test g_post[i0, j0, q] ≈ g_expected[q] atol=P0_ATOL
        end
    end
end

@testset "P3 CNEBB q-aware: q=0.5 single link equals strict" begin
    for q_missing in 2:9
        g_strict, g_pre, is_solid, C_strict, i0, j0 =
            _single_missing_link_state(q_missing)
        g_qaware = copy(g_strict)
        C_qaware = copy(C_strict)
        q_wall = zeros(Float64, size(g_strict))
        q_wall[i0, j0, _opp_local(q_missing)] = 0.5

        apply_cnebb_conformation_2d!(g_strict, g_pre, is_solid, C_strict)
        apply_cnebb_conformation_2d!(g_qaware, g_pre, is_solid, q_wall, C_qaware)

        @test C_qaware[i0, j0] ≈ C_strict[i0, j0] atol=P0_ATOL
        for q in 1:9
            @test g_qaware[i0, j0, q] ≈ g_strict[i0, j0, q] atol=P0_ATOL
        end
    end
end

function _fill_zero_velocity_equilibrium!(g, C)
    Nx, Ny = size(C)
    for i in 1:Nx, j in 1:Ny, q in 1:9
        g[i, j, q] = D2Q9_W[q] * C[i, j]
    end
    return g
end

function _fill_constant_velocity_equilibrium!(g, C, u, v)
    Nx, Ny = size(C)
    usq = u * u + v * v
    for i in 1:Nx, j in 1:Ny, q in 1:9
        eu = D2Q9_CX[q] * u + D2Q9_CY[q] * v
        g[i, j, q] = D2Q9_W[q] * C[i, j] *
                     (1.0 + 3.0 * eu + 4.5 * eu^2 - 1.5 * usq)
    end
    return g
end

function _linear_cut_link_macro_error(q_out, q_wall_value;
                                      qaware=false,
                                      orientation::Symbol=:normal,
                                      phi_mode::Symbol=:pre_opp,
                                      velocity=(0.0, 0.0))
    Nx, Ny = 7, 7
    i0, j0 = 4, 4
    cx = Int(D2Q9_CX[q_out])
    cy = Int(D2Q9_CY[q_out])
    tx = -cy
    ty = cx
    slope = 0.1

    is_solid = falses(Nx, Ny)
    is_solid[i0 + cx, j0 + cy] = true
    C0 = [
        orientation === :uniform ? 1.0 :
        orientation === :tangent ? 1.0 + slope * ((i - i0) * tx + (j - j0) * ty) :
        orientation === :normal ? 1.0 + slope * ((i - i0) * cx + (j - j0) * cy) :
        error("unknown orientation $(orientation)")
        for i in 1:Nx, j in 1:Ny
    ]
    ux = fill(velocity[1], Nx, Ny)
    uy = fill(velocity[2], Nx, Ny)
    g_pre = _fill_constant_velocity_equilibrium!(
        zeros(Float64, Nx, Ny, 9), C0, velocity[1], velocity[2],
    )
    g_post = similar(g_pre)
    stream_2d!(g_post, g_pre, Nx, Ny; sync=true)
    C_after = copy(C0)

    if qaware
        q_wall = zeros(Float64, Nx, Ny, 9)
        q_wall[i0, j0, q_out] = q_wall_value
        apply_cnebb_conformation_2d!(
            g_post, g_pre, is_solid, q_wall, C_after, ux, uy;
            phi_mode,
        )
    else
        apply_cnebb_conformation_2d!(
            g_post, g_pre, is_solid, C_after, ux, uy;
            phi_mode,
        )
    end

    sum_error = sum(g_post[i0, j0, q] for q in 1:9) - C_after[i0, j0]
    macro_error = C_after[i0, j0] - C0[i0, j0]
    return macro_error, sum_error
end

@testset "P4 CNEBB macro consistency: uniform and tangent fields" begin
    for q_out in 2:9, orientation in (:uniform, :tangent), qaware in (false, true)
        macro_error, sum_error =
            _linear_cut_link_macro_error(q_out, 0.3; qaware, orientation)
        @test abs(sum_error) < P0_ATOL
        @test abs(macro_error) < P0_ATOL
    end
end

@testset "P4 CNEBB macro consistency: normal linear field exposes pre_opp defect" begin
    for q_out in 2:9, qw in (0.3, 0.7)
        macro_error, sum_error = _linear_cut_link_macro_error(q_out, qw)
        @test abs(sum_error) < P0_ATOL
        @test abs(macro_error) > 1e-6
    end
end

@testset "P4 CNEBB q-aware macro consistency: normal linear field exposes pre_opp defect" begin
    for q_out in 2:9, qw in (0.3, 0.7)
        macro_error, sum_error = _linear_cut_link_macro_error(q_out, qw; qaware=true)
        @test abs(sum_error) < P0_ATOL
        @test abs(macro_error) > 1e-6
    end
end

@testset "P4 CNEBB eq-gradient macro consistency: normal linear field" begin
    for q_out in 2:9, qw in (0.3, 0.7), qaware in (false, true)
        macro_error, sum_error = _linear_cut_link_macro_error(
            q_out, qw; qaware, orientation=:normal, phi_mode=:eq_gradient,
        )
        @test abs(sum_error) < P0_ATOL
        @test abs(macro_error) < P0_ATOL
    end
end

function _multi_cut_link_macro_error(q_outs, gx, gy;
                                     qaware=false,
                                     phi_mode::Symbol=:pre_opp)
    Nx, Ny = 9, 9
    i0, j0 = 5, 5
    is_solid = falses(Nx, Ny)
    q_wall = zeros(Float64, Nx, Ny, 9)
    for q_out in q_outs
        cx = Int(D2Q9_CX[q_out])
        cy = Int(D2Q9_CY[q_out])
        is_solid[i0 + cx, j0 + cy] = true
        q_wall[i0, j0, q_out] = 0.3
    end
    C0 = [
        1.0 + gx * (i - i0) + gy * (j - j0)
        for i in 1:Nx, j in 1:Ny
    ]
    g_pre = _fill_zero_velocity_equilibrium!(zeros(Float64, Nx, Ny, 9), C0)
    g_post = similar(g_pre)
    stream_2d!(g_post, g_pre, Nx, Ny; sync=true)
    C_after = copy(C0)

    if qaware
        apply_cnebb_conformation_2d!(
            g_post, g_pre, is_solid, q_wall, C_after;
            phi_mode,
        )
    else
        apply_cnebb_conformation_2d!(
            g_post, g_pre, is_solid, C_after;
            phi_mode,
        )
    end

    sum_error = sum(g_post[i0, j0, q] for q in 1:9) - C_after[i0, j0]
    macro_error = C_after[i0, j0] - C0[i0, j0]
    return macro_error, sum_error
end

function _single_cut_link_macro_error_bc(q_out, bc;
                                         q_wall_value=0.3,
                                         orientation::Symbol=:normal,
                                         velocity=(0.03, -0.02))
    Nx, Ny = 7, 7
    i0, j0 = 4, 4
    cx = Int(D2Q9_CX[q_out])
    cy = Int(D2Q9_CY[q_out])
    tx = -cy
    ty = cx
    is_solid = falses(Nx, Ny)
    is_solid[i0 + cx, j0 + cy] = true
    q_wall = zeros(Float64, Nx, Ny, 9)
    q_wall[i0, j0, q_out] = q_wall_value
    slope = 0.1
    C0 = [
        orientation === :uniform ? 1.0 :
        orientation === :tangent ? 1.0 + slope * ((i - i0) * tx + (j - j0) * ty) :
        orientation === :normal ? 1.0 + slope * ((i - i0) * cx + (j - j0) * cy) :
        orientation === :mixed ? 1.0 + 0.1 * (i - i0) - 0.07 * (j - j0) :
        error("unknown orientation $(orientation)")
        for i in 1:Nx, j in 1:Ny
    ]
    ux = fill(velocity[1], Nx, Ny)
    uy = fill(velocity[2], Nx, Ny)
    g_pre = _fill_constant_velocity_equilibrium!(
        zeros(Float64, Nx, Ny, 9), C0, velocity[1], velocity[2],
    )
    g_post = similar(g_pre)
    stream_2d!(g_post, g_pre, Nx, Ny; sync=true)
    C_after = copy(C0)
    apply_polymer_wall_bc!(g_post, g_pre, is_solid, q_wall, C_after, ux, uy, bc)

    sum_error = sum(g_post[i0, j0, q] for q in 1:9) - C_after[i0, j0]
    macro_error = C_after[i0, j0] - C0[i0, j0]
    return macro_error, sum_error
end

_wall_bc_cases() = (
    (name=:cnebb, bc=CNEBB()),
    (name=:cnebb_qaware, bc=CNEBBQAware()),
    (name=:cnebb_field, bc=CNEBBField()),
    (name=:cnebb_field_equilibrium, bc=CNEBBFieldEquilibrium()),
    (name=:cnebb_eq_gradient, bc=CNEBBEqGradient()),
    (name=:cnebb_cutlink_eq_gradient, bc=CNEBBCutLinkEqGradient()),
    (name=:ylw_a, bc=YLW_A()),
    (name=:ylw_b, bc=YLW_B()),
    (name=:ylw_balance, bc=YLWBalanceOnly()),
)

function _active_bc_linear_macro_passes(name::Symbol, orientation::Symbol, velocity)
    name in (:cnebb_field, :cnebb_field_equilibrium) && return true
    name === :cnebb_eq_gradient && return true
    name === :cnebb_cutlink_eq_gradient && return false
    return velocity == (0.0, 0.0) && orientation in (:uniform, :tangent)
end

@testset "P5 CNEBB eq-gradient macro consistency: multi cut-link linear fields" begin
    cut_sets = (
        (2, 3),
        (2, 6),
        (2, 3, 6),
        (2, 3, 4),
        (2, 3, 6, 7),
    )
    gradients = (
        (0.1, 0.0),
        (0.0, 0.1),
        (0.1, -0.07),
    )
    for q_outs in cut_sets, (gx, gy) in gradients, qaware in (false, true)
        macro_error, sum_error = _multi_cut_link_macro_error(
            q_outs, gx, gy; qaware, phi_mode=:eq_gradient,
        )
        @test abs(sum_error) < P0_ATOL
        @test abs(macro_error) < P0_ATOL
    end
end

@testset "P6 CNEBB eq-gradient macro consistency: constant velocity equilibrium" begin
    velocities = (
        (0.03, 0.0),
        (0.0, -0.02),
        (0.03, -0.02),
    )
    for velocity in velocities, q_out in (2, 3, 6), orientation in (:normal, :tangent)
        macro_error, sum_error = _linear_cut_link_macro_error(
            q_out, 0.3; orientation, phi_mode=:eq_gradient, velocity,
        )
        @test abs(sum_error) < P0_ATOL
        @test abs(macro_error) < P0_ATOL
    end
end

@testset "P7 CNEBBEqGradient dispatch: linear cut-link patch" begin
    for q_out in 2:9
        macro_error, sum_error =
            _single_cut_link_macro_error_bc(q_out, CNEBBEqGradient(); orientation=:mixed)
        @test abs(sum_error) < P0_ATOL
        @test abs(macro_error) < P0_ATOL
    end
end

@testset "P8 NoPolymerWallBC: no-wall no-op and wall misuse detector" begin
    Nx, Ny = 5, 5
    g_pre = rand(Float64, Nx, Ny, 9)
    g_post = copy(g_pre)
    is_solid = falses(Nx, Ny)
    q_wall = zeros(Float64, Nx, Ny, 9)
    C = rand(Float64, Nx, Ny)
    C0 = copy(C)
    ux = zeros(Float64, Nx, Ny)
    uy = zeros(Float64, Nx, Ny)
    apply_polymer_wall_bc!(g_post, g_pre, is_solid, q_wall, C, ux, uy, NoPolymerWallBC())
    @test g_post == g_pre
    @test C == C0

    @test_throws ErrorException _single_cut_link_macro_error_bc(
        2, NoPolymerWallBC(); orientation=:normal, velocity=(0.03, -0.02),
    )
end

function _logspace_wall_bc_gap(q_out, bc; q_wall_value=0.3, amplitude=0.35)
    Nx, Ny = 7, 7
    i0, j0 = 4, 4
    cx = Int(D2Q9_CX[q_out])
    cy = Int(D2Q9_CY[q_out])
    is_solid = falses(Nx, Ny)
    is_solid[i0 + cx, j0 + cy] = true
    q_wall = zeros(Float64, Nx, Ny, 9)
    q_wall[i0, j0, q_out] = q_wall_value
    ux = zeros(Float64, Nx, Ny)
    uy = zeros(Float64, Nx, Ny)
    Ψ0 = [
        amplitude * ((i - i0) * cx + (j - j0) * cy)
        for i in 1:Nx, j in 1:Ny
    ]
    C0 = exp.(Ψ0)

    g_c_pre = _fill_zero_velocity_equilibrium!(zeros(Float64, Nx, Ny, 9), C0)
    g_c_post = similar(g_c_pre)
    stream_2d!(g_c_post, g_c_pre, Nx, Ny; sync=true)
    C_after = copy(C0)
    apply_polymer_wall_bc!(g_c_post, g_c_pre, is_solid, q_wall, C_after, ux, uy, bc)

    g_ψ_pre = _fill_zero_velocity_equilibrium!(zeros(Float64, Nx, Ny, 9), Ψ0)
    g_ψ_post = similar(g_ψ_pre)
    stream_2d!(g_ψ_post, g_ψ_pre, Nx, Ny; sync=true)
    Ψ_after = copy(Ψ0)
    apply_polymer_wall_bc!(g_ψ_post, g_ψ_pre, is_solid, q_wall, Ψ_after, ux, uy, bc)

    c_direct = C_after[i0, j0]
    c_from_log = exp(Ψ_after[i0, j0])
    return c_from_log - c_direct
end

@testset "P8b log-space wall BC is not equivalent to C-space BC" begin
    for case in _wall_bc_cases(),
        q_out in 2:9,
        qw in (0.3, 0.7)

        uniform_gap = _logspace_wall_bc_gap(q_out, case.bc; q_wall_value=qw,
                                            amplitude=0.0)
        nonlinear_gap = _logspace_wall_bc_gap(q_out, case.bc; q_wall_value=qw)
        @test abs(uniform_gap) < P0_ATOL
        if case.name in (:cnebb_field, :cnebb_field_equilibrium, :cnebb_eq_gradient)
            @test abs(nonlinear_gap) < P0_ATOL
        elseif case.name === :cnebb_cutlink_eq_gradient
            @test isfinite(nonlinear_gap)
        else
            @test abs(nonlinear_gap) > 1e-4
        end
    end
end

@testset "P9 all wall BCs: single cut-link conservation matrix" begin
    for case in _wall_bc_cases(),
        q_out in 2:9,
        qw in (0.3, 0.5, 0.7),
        orientation in (:uniform, :tangent, :normal),
        velocity in ((0.0, 0.0), (0.03, -0.02))

        _, sum_error = _single_cut_link_macro_error_bc(
            q_out, case.bc; q_wall_value=qw, orientation, velocity,
        )
        @test abs(sum_error) < P0_ATOL
    end
end

@testset "P10 all wall BCs: single cut-link macro consistency matrix" begin
    broken_macro_errors = Dict{Symbol,Vector{Float64}}(
        case.name => Float64[] for case in _wall_bc_cases()
    )
    for case in _wall_bc_cases(),
        q_out in 2:9,
        qw in (0.3, 0.5, 0.7),
        orientation in (:uniform, :tangent, :normal),
        velocity in ((0.0, 0.0), (0.03, -0.02))

        macro_error, _ = _single_cut_link_macro_error_bc(
            q_out, case.bc; q_wall_value=qw, orientation, velocity,
        )
        if _active_bc_linear_macro_passes(case.name, orientation, velocity)
            @test abs(macro_error) < P0_ATOL
        else
            push!(broken_macro_errors[case.name], abs(macro_error))
        end
    end
    for case in _wall_bc_cases()
        if !isempty(broken_macro_errors[case.name])
            @test maximum(broken_macro_errors[case.name]) > 1e-6
        end
    end
end

function _multi_cut_link_macro_error_bc(q_outs, gx, gy, bc; velocity=(0.0, 0.0))
    Nx, Ny = 9, 9
    i0, j0 = 5, 5
    is_solid = falses(Nx, Ny)
    q_wall = zeros(Float64, Nx, Ny, 9)
    for q_out in q_outs
        cx = Int(D2Q9_CX[q_out])
        cy = Int(D2Q9_CY[q_out])
        is_solid[i0 + cx, j0 + cy] = true
        q_wall[i0, j0, q_out] = 0.3
    end
    C0 = [
        1.0 + gx * (i - i0) + gy * (j - j0)
        for i in 1:Nx, j in 1:Ny
    ]
    ux = fill(velocity[1], Nx, Ny)
    uy = fill(velocity[2], Nx, Ny)
    g_pre = _fill_constant_velocity_equilibrium!(
        zeros(Float64, Nx, Ny, 9), C0, velocity[1], velocity[2],
    )
    g_post = similar(g_pre)
    stream_2d!(g_post, g_pre, Nx, Ny; sync=true)
    C_after = copy(C0)
    apply_polymer_wall_bc!(g_post, g_pre, is_solid, q_wall, C_after, ux, uy, bc)

    sum_error = sum(g_post[i0, j0, q] for q in 1:9) - C_after[i0, j0]
    macro_error = C_after[i0, j0] - C0[i0, j0]
    return macro_error, sum_error
end

@testset "P11 all wall BCs: multi cut-link conservation matrix" begin
    cut_sets = ((2, 3), (2, 6), (2, 3, 6), (2, 3, 4), (2, 3, 6, 7))
    gradients = ((0.1, 0.0), (0.0, 0.1), (0.1, -0.07))
    for case in _wall_bc_cases(),
        q_outs in cut_sets,
        (gx, gy) in gradients,
        velocity in ((0.0, 0.0), (0.03, -0.02))

        _, sum_error = _multi_cut_link_macro_error_bc(
            q_outs, gx, gy, case.bc; velocity,
        )
        @test abs(sum_error) < P0_ATOL
    end
end

@testset "P12 all wall BCs: multi cut-link macro consistency matrix" begin
    cut_sets = ((2, 3), (2, 6), (2, 3, 6), (2, 3, 4), (2, 3, 6, 7))
    gradients = ((0.1, 0.0), (0.0, 0.1), (0.1, -0.07))
    broken_macro_errors = Dict{Symbol,Vector{Float64}}(
        case.name => Float64[] for case in _wall_bc_cases()
    )
    for case in _wall_bc_cases(),
        q_outs in cut_sets,
        (gx, gy) in gradients,
        velocity in ((0.0, 0.0), (0.03, -0.02))

        macro_error, _ = _multi_cut_link_macro_error_bc(
            q_outs, gx, gy, case.bc; velocity,
        )
        if case.name in (:cnebb_field, :cnebb_field_equilibrium, :cnebb_eq_gradient)
            @test abs(macro_error) < P0_ATOL
        else
            push!(broken_macro_errors[case.name], abs(macro_error))
        end
    end
    for case in _wall_bc_cases()
        if !isempty(broken_macro_errors[case.name])
            @test maximum(broken_macro_errors[case.name]) > 1e-6
        end
    end
end

function _actual_square_obstacle_macro_errors(bc; gradient=(0.0, 0.0),
                                              velocity=(0.0, 0.0))
    geom = square_obstacle_channel_geometry_2d(; H=24, side=6, L_up=3, L_down=4)
    Nx, Ny = geom.Nx, geom.Ny
    cx0 = geom.i_step + (geom.H_ref - 1) / 2
    cy0 = (Ny - 1) / 2
    gx, gy = gradient
    C0 = [
        1.0 + gx * ((i - 1) - cx0) + gy * ((j - 1) - cy0)
        for i in 1:Nx, j in 1:Ny
    ]
    ux = fill(velocity[1], Nx, Ny)
    uy = fill(velocity[2], Nx, Ny)
    g_pre = _fill_constant_velocity_equilibrium!(
        zeros(Float64, Nx, Ny, 9), C0, velocity[1], velocity[2],
    )
    g_post = similar(g_pre)
    stream_2d!(g_post, g_pre, Nx, Ny; sync=true)
    C_after = copy(C0)
    apply_polymer_wall_bc!(
        g_post, g_pre, geom.is_solid, geom.q_wall, C_after, ux, uy, bc,
    )

    max_sum_error = 0.0
    max_macro_error = 0.0
    n_cut_cells = 0
    for j in 1:Ny, i in 1:Nx
        any_cut = any(q -> q > 0.0, view(geom.q_wall, i, j, :))
        any_cut || continue
        n_cut_cells += 1
        max_sum_error = max(
            max_sum_error,
            abs(sum(g_post[i, j, q] for q in 1:9) - C_after[i, j]),
        )
        max_macro_error = max(max_macro_error, abs(C_after[i, j] - C0[i, j]))
    end
    return (; max_sum_error, max_macro_error, n_cut_cells)
end

@testset "P12a all wall BCs: square obstacle q=0.5 matrix" begin
    for case in _wall_bc_cases()
        result = _actual_square_obstacle_macro_errors(case.bc; velocity=(0.0, 0.0))
        @test result.n_cut_cells > 0
        @test result.max_sum_error < P0_ATOL
        @test result.max_macro_error < P0_ATOL
    end

    broken_velocity_errors = Dict{Symbol,Vector{Float64}}(
        case.name => Float64[] for case in _wall_bc_cases()
    )
    for case in _wall_bc_cases()
        result = _actual_square_obstacle_macro_errors(case.bc; velocity=(0.03, -0.02))
        @test result.n_cut_cells > 0
        @test result.max_sum_error < P0_ATOL
        if case.name in (:cnebb_field, :cnebb_field_equilibrium, :cnebb_eq_gradient)
            @test result.max_macro_error < P0_ATOL
        else
            push!(broken_velocity_errors[case.name], result.max_macro_error)
        end
    end
    for case in _wall_bc_cases()
        case.name in (:cnebb_field, :cnebb_field_equilibrium, :cnebb_eq_gradient) && continue
        @test maximum(broken_velocity_errors[case.name]) > 1e-6
    end

    broken_gradient_errors = Dict{Symbol,Vector{Float64}}(
        case.name => Float64[] for case in _wall_bc_cases()
    )
    for case in _wall_bc_cases(),
        gradient in ((0.05, 0.0), (0.0, 0.04), (0.03, -0.02)),
        velocity in ((0.0, 0.0), (0.03, -0.02))

        result = _actual_square_obstacle_macro_errors(case.bc; gradient, velocity)
        @test result.n_cut_cells > 0
        @test result.max_sum_error < P0_ATOL
        if case.name in (:cnebb_field, :cnebb_field_equilibrium, :cnebb_eq_gradient)
            @test result.max_macro_error < P0_ATOL
        else
            push!(broken_gradient_errors[case.name], result.max_macro_error)
        end
    end
    for case in _wall_bc_cases()
        case.name in (:cnebb_field, :cnebb_field_equilibrium, :cnebb_eq_gradient) && continue
        @test maximum(broken_gradient_errors[case.name]) > 1e-6
    end
end

function _actual_cylinder_cutlink_macro_errors(bc; gradient=(0.0, 0.0),
                                               velocity=(0.0, 0.0))
    Nx, Ny = 48, 40
    cx, cy, R = 23.37, 19.79, 8.0
    q_wall, is_solid = precompute_q_wall_cylinder(Nx, Ny, cx, cy, R)
    gx, gy = gradient
    C0 = [
        1.0 + gx * ((i - 1) - cx) + gy * ((j - 1) - cy)
        for i in 1:Nx, j in 1:Ny
    ]
    ux = fill(velocity[1], Nx, Ny)
    uy = fill(velocity[2], Nx, Ny)
    g_pre = _fill_constant_velocity_equilibrium!(
        zeros(Float64, Nx, Ny, 9), C0, velocity[1], velocity[2],
    )
    g_post = similar(g_pre)
    stream_2d!(g_post, g_pre, Nx, Ny; sync=true)
    C_after = copy(C0)
    apply_polymer_wall_bc!(g_post, g_pre, is_solid, q_wall, C_after, ux, uy, bc)

    max_sum_error = 0.0
    max_macro_error = 0.0
    n_cut_cells = 0
    for j in 1:Ny, i in 1:Nx
        any_cut = any(q -> q > 0.0, view(q_wall, i, j, :))
        any_cut || continue
        n_cut_cells += 1
        max_sum_error = max(
            max_sum_error,
            abs(sum(g_post[i, j, q] for q in 1:9) - C_after[i, j]),
        )
        max_macro_error = max(max_macro_error, abs(C_after[i, j] - C0[i, j]))
    end
    return (; max_sum_error, max_macro_error, n_cut_cells)
end

@testset "P12b all wall BCs: actual cylinder cut-link matrix" begin
    for case in _wall_bc_cases()
        result = _actual_cylinder_cutlink_macro_errors(case.bc; velocity=(0.0, 0.0))
        @test result.n_cut_cells > 0
        @test result.max_sum_error < P0_ATOL
        @test result.max_macro_error < P0_ATOL
    end

    arbitrary_velocity_errors = Dict{Symbol,Vector{Float64}}(
        case.name => Float64[] for case in _wall_bc_cases()
    )
    for case in _wall_bc_cases()
        result = _actual_cylinder_cutlink_macro_errors(case.bc; velocity=(0.03, -0.02))
        @test result.n_cut_cells > 0
        @test result.max_sum_error < P0_ATOL
        if case.name in (:cnebb_field, :cnebb_field_equilibrium, :cnebb_eq_gradient, :cnebb_cutlink_eq_gradient)
            @test result.max_macro_error < P0_ATOL
        else
            push!(arbitrary_velocity_errors[case.name], result.max_macro_error)
        end
    end
    for case in _wall_bc_cases()
        case.name in (:cnebb_field, :cnebb_field_equilibrium, :cnebb_eq_gradient, :cnebb_cutlink_eq_gradient) &&
            continue
        @test maximum(arbitrary_velocity_errors[case.name]) > 1e-6
    end

    broken_macro_errors = Dict{Symbol,Vector{Float64}}(
        case.name => Float64[] for case in _wall_bc_cases()
    )
    for case in _wall_bc_cases(),
        gradient in ((0.05, 0.0), (0.0, 0.04), (0.03, -0.02)),
        velocity in ((0.0, 0.0), (0.03, -0.02))

        result = _actual_cylinder_cutlink_macro_errors(case.bc; gradient, velocity)
        @test result.n_cut_cells > 0
        @test result.max_sum_error < P0_ATOL
        if case.name in (:cnebb_field, :cnebb_field_equilibrium, :cnebb_eq_gradient, :cnebb_cutlink_eq_gradient)
            @test result.max_macro_error < P0_ATOL
        else
            push!(broken_macro_errors[case.name], result.max_macro_error)
        end
    end
    for case in _wall_bc_cases()
        case.name in (:cnebb_field, :cnebb_field_equilibrium, :cnebb_eq_gradient, :cnebb_cutlink_eq_gradient) &&
            continue
        @test maximum(broken_macro_errors[case.name]) > 1e-6
    end
end

@testset "P13 polymer stress closure: direct and log-conformation" begin
    cxx, cxy, cyy = 1.3, 0.2, 0.9
    ψxx, ψxy, ψyy = _log_spd_2x2(cxx, cxy, cyy)
    G = 0.7

    C_xx = fill(cxx, 1, 1)
    C_xy = fill(cxy, 1, 1)
    C_yy = fill(cyy, 1, 1)
    τ_xx = zeros(Float64, 1, 1)
    τ_xy = zeros(Float64, 1, 1)
    τ_yy = zeros(Float64, 1, 1)
    update_polymer_stress!(τ_xx, τ_xy, τ_yy, C_xx, C_xy, C_yy, OldroydB(G=G, λ=2.0))
    @test τ_xx[1, 1] ≈ G * (cxx - 1.0) atol=P0_ATOL
    @test τ_xy[1, 1] ≈ G * cxy atol=P0_ATOL
    @test τ_yy[1, 1] ≈ G * (cyy - 1.0) atol=P0_ATOL

    C_from_ψ_xx = zeros(Float64, 1, 1)
    C_from_ψ_xy = zeros(Float64, 1, 1)
    C_from_ψ_yy = zeros(Float64, 1, 1)
    psi_to_C_2d!(
        C_from_ψ_xx, C_from_ψ_xy, C_from_ψ_yy,
        fill(ψxx, 1, 1), fill(ψxy, 1, 1), fill(ψyy, 1, 1),
    )
    @test C_from_ψ_xx[1, 1] ≈ cxx atol=P0_ATOL
    @test C_from_ψ_xy[1, 1] ≈ cxy atol=P0_ATOL
    @test C_from_ψ_yy[1, 1] ≈ cyy atol=P0_ATOL

    fill!(τ_xx, 0.0)
    fill!(τ_xy, 0.0)
    fill!(τ_yy, 0.0)
    update_polymer_stress!(
        τ_xx, τ_xy, τ_yy,
        C_from_ψ_xx, C_from_ψ_xy, C_from_ψ_yy,
        LogConfOldroydB(G=G, λ=2.0),
    )
    @test τ_xx[1, 1] ≈ G * (cxx - 1.0) atol=P0_ATOL
    @test τ_xy[1, 1] ≈ G * cxy atol=P0_ATOL
    @test τ_yy[1, 1] ≈ G * (cyy - 1.0) atol=P0_ATOL

    fill!(τ_xx, NaN)
    fill!(τ_xy, NaN)
    fill!(τ_yy, NaN)
    C_bad_xx = fill(NaN, 1, 1)
    C_bad_xy = fill(NaN, 1, 1)
    C_bad_yy = fill(NaN, 1, 1)
    update_polymer_stress!(
        τ_xx, τ_xy, τ_yy,
        C_bad_xx, C_bad_xy, C_bad_yy,
        OldroydB(G=0.0, λ=2.0),
    )
    @test τ_xx[1, 1] == 0.0
    @test τ_xy[1, 1] == 0.0
    @test τ_yy[1, 1] == 0.0

    fill!(τ_xx, NaN)
    fill!(τ_xy, NaN)
    fill!(τ_yy, NaN)
    update_polymer_stress!(
        τ_xx, τ_xy, τ_yy,
        C_bad_xx, C_bad_xy, C_bad_yy,
        LogConfOldroydB(G=0.0, λ=2.0),
    )
    @test τ_xx[1, 1] == 0.0
    @test τ_xy[1, 1] == 0.0
    @test τ_yy[1, 1] == 0.0
end

function _linear_velocity_patch(; Nx=5, Ny=5, u0=0.02, v0=-0.01,
                                dudx=0.0, dudy=0.0, dvdx=0.0, dvdy=0.0)
    i0, j0 = (Nx + 1) ÷ 2, (Ny + 1) ÷ 2
    ux = [u0 + dudx * (i - i0) + dudy * (j - j0) for i in 1:Nx, j in 1:Ny]
    uy = [v0 + dvdx * (i - i0) + dvdy * (j - j0) for i in 1:Nx, j in 1:Ny]
    return ux, uy, i0, j0
end

function _collide_component_once!(collision, g, Fe_prev, C_field, ux, uy, ρ,
                                  Cxx, Cxy, Cyy, is_solid, tau_plus, λ,
                                  component; magic=2.5e-7,
                                  divergence_mode::Symbol=:numerical,
                                  stencils=nothing, uwx=nothing, uwy=nothing)
    if collision === :trt
        collide_conformation_2d!(
            g, C_field, ux, uy, Cxx, Cxy, Cyy, is_solid,
            tau_plus, λ; magic, component, divergence_mode,
        )
    elseif collision in (:trt_embedded_axis, :trt_wallfit4)
        Kraken.collide_conformation_2d_with_gradient_stencils!(
            g, C_field, ux, uy, Cxx, Cxy, Cyy, is_solid,
            uwx, uwy, stencils, tau_plus, λ; magic, component,
            divergence_mode,
        )
    elseif collision === :regularized
        collide_conformation_regularized_2d!(
            g, C_field, ux, uy, Cxx, Cxy, Cyy, is_solid,
            tau_plus, λ; magic, component, divergence_mode,
        )
    elseif collision === :liu_eq26
        collide_conformation_liu_eq26_2d!(
            g, Fe_prev, C_field, ux, uy, ρ, Cxx, Cxy, Cyy, is_solid,
            tau_plus, λ; magic, component, divergence_mode,
        )
    else
        error("unknown collision $collision")
    end
    return g
end

function _steady_source_liu_collision!(g, Fe_prev, C_field, ux, uy, ρ,
                                       Cxx, Cxy, Cyy, is_solid, tau_plus, λ,
                                       component; magic=2.5e-7,
                                       divergence_mode::Symbol=:numerical)
    init_conformation_field_2d!(g, C_field, ux, uy)
    _collide_component_once!(
        :liu_eq26, g, Fe_prev, C_field, ux, uy, ρ, Cxx, Cxy, Cyy,
        is_solid, tau_plus, λ, component; magic, divergence_mode,
    )
    init_conformation_field_2d!(g, C_field, ux, uy)
    _collide_component_once!(
        :liu_eq26, g, Fe_prev, C_field, ux, uy, ρ, Cxx, Cxy, Cyy,
        is_solid, tau_plus, λ, component; magic, divergence_mode,
    )
end

function _collision_patch_moments(collision; tau_plus=1.0, λ=2.0,
                                  dudx=0.0, dudy=0.0, dvdx=0.0, dvdy=0.0,
                                  cxx=1.0, cxy=0.0, cyy=1.0, component=1,
                                  divergence_mode::Symbol=:numerical)
    Nx = Ny = 5
    ux, uy, i0, j0 = _linear_velocity_patch(; Nx, Ny, dudx, dudy, dvdx, dvdy)
    is_solid = falses(Nx, Ny)
    ρ = ones(Float64, Nx, Ny)
    Cxx = fill(cxx, Nx, Ny)
    Cxy = fill(cxy, Nx, Ny)
    Cyy = fill(cyy, Nx, Ny)
    C_field = component == 1 ? Cxx : component == 2 ? Cxy : Cyy
    g = zeros(Float64, Nx, Ny, 9)
    Fe_prev = zeros(Float64, Nx, Ny, 9)
    init_conformation_field_2d!(g, C_field, ux, uy)
    g0 = copy(g)

    if collision === :liu_eq26
        _steady_source_liu_collision!(
            g, Fe_prev, C_field, ux, uy, ρ, Cxx, Cxy, Cyy, is_solid,
            tau_plus, λ, component; divergence_mode,
        )
    else
        _collide_component_once!(
            collision, g, Fe_prev, C_field, ux, uy, ρ, Cxx, Cxy, Cyy,
            is_solid, tau_plus, λ, component; divergence_mode,
        )
    end

    Δ = [g[i0, j0, q] - g0[i0, j0, q] for q in 1:9]
    return (
        mass = sum(Δ),
        mom_x = sum(D2Q9_CX[q] * Δ[q] for q in 1:9),
        mom_y = sum(D2Q9_CY[q] * Δ[q] for q in 1:9),
        max_pop_delta = maximum(abs, Δ),
    )
end

@testset "P14 conformation collisions: analytic stationary fixed points" begin
    λ = 2.0
    γ = 0.03
    ε = 0.08
    fixed_points = (
        (name=:shear_x_from_y, dudx=0.0, dudy=γ, dvdx=0.0, dvdy=0.0,
         cxx=1.0 + 2 * (λ * γ)^2, cxy=λ * γ, cyy=1.0),
        (name=:shear_y_from_x, dudx=0.0, dudy=0.0, dvdx=γ, dvdy=0.0,
         cxx=1.0, cxy=λ * γ, cyy=1.0 + 2 * (λ * γ)^2),
        (name=:extension, dudx=ε, dudy=0.0, dvdx=0.0, dvdy=-ε,
         cxx=1.0 / (1.0 - 2λ * ε), cxy=0.0, cyy=1.0 / (1.0 + 2λ * ε)),
    )
    for collision in (:trt, :regularized, :liu_eq26),
        fp in fixed_points,
        component in 1:3

        moments = _collision_patch_moments(
            collision; λ, dudx=fp.dudx, dudy=fp.dudy,
            dvdx=fp.dvdx, dvdy=fp.dvdy,
            cxx=fp.cxx, cxy=fp.cxy, cyy=fp.cyy, component,
        )
        @test abs(moments.mass) < 5e-13
        @test abs(moments.mom_x) < 5e-13
        @test abs(moments.mom_y) < 5e-13
        @test moments.max_pop_delta < 5e-13
    end
end

@testset "P15a conformation collisions: trace-free incompressible gradient projection" begin
    a = 0.01
    for collision in (:trt, :regularized, :liu_eq26),
        tau_plus in (1.0, 0.50001),
        component in (1, 3)

        numerical = _collision_patch_moments(
            collision; tau_plus, dudx=a, dvdy=a, component,
            divergence_mode=:numerical,
        )
        projected = _collision_patch_moments(
            collision; tau_plus, dudx=a, dvdy=a, component,
            divergence_mode=:trace_free,
        )
        @test abs(numerical.mass) > 0.01
        @test abs(projected.mass) < 5e-13
        @test abs(projected.mom_x) < 5e-13
        @test abs(projected.mom_y) < 5e-13
    end
end

@testset "P15 conformation collisions: source mass and momentum moments" begin
    λ = 2.0
    γ = 0.03
    ε = 0.04
    for collision in (:trt, :regularized, :liu_eq26),
        tau_plus in (1.0, 0.50001),
        case in (
            (dudx=0.0, dudy=γ, dvdx=0.0, dvdy=0.0, component=2, S=γ),
            (dudx=0.0, dudy=0.0, dvdx=γ, dvdy=0.0, component=2, S=γ),
            (dudx=ε, dudy=0.0, dvdx=0.0, dvdy=-ε, component=1, S=2ε),
            (dudx=ε, dudy=0.0, dvdx=0.0, dvdy=-ε, component=3, S=-2ε),
        )

        moments = _collision_patch_moments(
            collision; tau_plus, λ,
            dudx=case.dudx, dudy=case.dudy,
            dvdx=case.dvdx, dvdy=case.dvdy,
            component=case.component,
        )
        coeff = 1.0 - 0.5 / tau_plus
        @test moments.mass ≈ case.S atol=5e-13
        @test moments.mom_x ≈ coeff * case.S * 0.02 atol=5e-13
        @test moments.mom_y ≈ coeff * case.S * -0.01 atol=5e-13
    end
end

function _bulk_direct_collision_reaction(; λ, dudx, dudy, dvdx, dvdy,
                                         tau_plus=1.0, magic=1e-6,
                                         steps=20_000)
    Nx = Ny = 5
    ux, uy, i0, j0 = _linear_velocity_patch(; Nx, Ny, dudx, dudy, dvdx, dvdy)
    is_solid = falses(Nx, Ny)
    Cxx = ones(Float64, Nx, Ny)
    Cxy = zeros(Float64, Nx, Ny)
    Cyy = ones(Float64, Nx, Ny)
    gxx = zeros(Float64, Nx, Ny, 9)
    gxy = zeros(Float64, Nx, Ny, 9)
    gyy = zeros(Float64, Nx, Ny, 9)
    init_conformation_field_2d!(gxx, Cxx, ux, uy)
    init_conformation_field_2d!(gxy, Cxy, ux, uy)
    init_conformation_field_2d!(gyy, Cyy, ux, uy)

    min_eig = Inf
    for _ in 1:steps
        compute_conformation_macro_2d!(Cxx, gxx)
        compute_conformation_macro_2d!(Cxy, gxy)
        compute_conformation_macro_2d!(Cyy, gyy)
        min_eig = min(
            min_eig,
            _min_eig_spd_2x2(Cxx[i0, j0], Cxy[i0, j0], Cyy[i0, j0]),
        )
        collide_conformation_2d!(
            gxx, Cxx, ux, uy, Cxx, Cxy, Cyy, is_solid, tau_plus, λ;
            magic, component=1,
        )
        collide_conformation_2d!(
            gxy, Cxy, ux, uy, Cxx, Cxy, Cyy, is_solid, tau_plus, λ;
            magic, component=2,
        )
        collide_conformation_2d!(
            gyy, Cyy, ux, uy, Cxx, Cxy, Cyy, is_solid, tau_plus, λ;
            magic, component=3,
        )
    end
    compute_conformation_macro_2d!(Cxx, gxx)
    compute_conformation_macro_2d!(Cxy, gxy)
    compute_conformation_macro_2d!(Cyy, gyy)
    min_eig = min(min_eig, _min_eig_spd_2x2(Cxx[i0, j0], Cxy[i0, j0], Cyy[i0, j0]))
    return (; cxx=Cxx[i0, j0], cxy=Cxy[i0, j0], cyy=Cyy[i0, j0], min_eig)
end

@testset "P15b direct-C TRT bulk reaction: low-Wi long-time canary" begin
    λ = 600.0
    for case in (
        (dudx=0.0, dudy=2.0e-4, dvdx=0.0, dvdy=0.0),
        (dudx=0.0, dudy=0.0, dvdx=2.0e-4, dvdy=0.0),
        (dudx=3.0e-4, dudy=0.0, dvdx=0.0, dvdy=-3.0e-4),
        (dudx=-3.0e-4, dudy=0.0, dvdx=0.0, dvdy=3.0e-4),
        (dudx=1.0e-4, dudy=2.0e-4, dvdx=-1.0e-4, dvdy=-1.0e-4),
    )
        result = _bulk_direct_collision_reaction(
            λ=λ, dudx=case.dudx, dudy=case.dudy,
            dvdx=case.dvdx, dvdy=case.dvdy,
        )
        @test isfinite(result.cxx)
        @test isfinite(result.cxy)
        @test isfinite(result.cyy)
        @test result.min_eig > 0.0
        cxx, cxy, cyy = _stationary_direct_conformation_incompressible(
            case.dudx, case.dudy, case.dvdx, case.dvdy, λ,
        )
        @test result.cxx ≈ cxx rtol=1e-6 atol=1e-10
        @test result.cxy ≈ cxy rtol=1e-6 atol=1e-10
        @test result.cyy ≈ cyy rtol=1e-6 atol=1e-10
    end
end

@testset "P15c CDE inlet reset removes boundary-gradient source pollution" begin
    Nx, Ny = 5, 7
    λ = 600.0
    g = zeros(Float64, Nx, Ny, 9)
    C = ones(Float64, Nx, Ny)
    C_inlet = ones(Float64, Ny)
    u_profile = [0.01 * sinpi((j - 0.5) / Ny) for j in 1:Ny]
    ux = zeros(Float64, Nx, Ny)
    uy = zeros(Float64, Nx, Ny)
    solid = falses(Nx, Ny)
    for j in 1:Ny
        ux[1, j] = u_profile[j]
        ux[2, j] = -0.5 * u_profile[j]
    end

    reset_conformation_inlet_2d!(g, C_inlet, u_profile, Ny)
    collide_conformation_2d!(
        g, C, ux, uy, C, zeros(Float64, Nx, Ny), C, solid, 1.0, λ;
        magic=1e-6, component=1, divergence_mode=:trace_free,
    )
    polluted = maximum(abs(g[1, j, q] - equilibrium(D2Q9(), C_inlet[j], u_profile[j], 0.0, q))
                       for j in 1:Ny, q in 1:9)
    @test polluted > 1e-6

    reset_conformation_inlet_2d!(g, C_inlet, u_profile, Ny)
    clean = maximum(abs(g[1, j, q] - equilibrium(D2Q9(), C_inlet[j], u_profile[j], 0.0, q))
                    for j in 1:Ny, q in 1:9)
    @test clean < P0_ATOL
end

function _bulk_affine_shear_patch(; orientation=:x_shear_y)
    Nx = Ny = 48
    γ = 0.01
    λ = 3.0
    ux = zeros(Float64, Nx, Ny)
    uy = zeros(Float64, Nx, Ny)
    if orientation === :x_shear_y
        for j in 1:Ny, i in 1:Nx
            ux[i, j] = γ * ((j - 1.0) - 24.0)
        end
        cxx, cxy, cyy = _stationary_direct_conformation_incompressible(
            0.0, γ, 0.0, 0.0, λ,
        )
    elseif orientation === :y_shear_x
        for j in 1:Ny, i in 1:Nx
            uy[i, j] = γ * ((i - 1.0) - 24.0)
        end
        cxx, cxy, cyy = _stationary_direct_conformation_incompressible(
            0.0, 0.0, γ, 0.0, λ,
        )
    else
        error("unknown bulk affine shear orientation $(orientation)")
    end
    return (; Nx, Ny, λ, ux, uy, cxx, cxy, cyy, is_solid=falses(Nx, Ny))
end

function _bulk_affine_transport_error(; orientation=:x_shear_y, steps=4,
                                      constant_velocity=false)
    p = _bulk_affine_shear_patch(; orientation)
    C_ref = 1.234
    C = fill(C_ref, p.Nx, p.Ny)
    ux = constant_velocity ? fill(0.02, p.Nx, p.Ny) : p.ux
    uy = constant_velocity ? fill(-0.01, p.Nx, p.Ny) : p.uy
    g = zeros(Float64, p.Nx, p.Ny, 9)
    buf = similar(g)
    init_conformation_field_2d!(g, C, ux, uy)
    margin = steps + 4
    max_error = 0.0
    for _ in 1:steps
        stream_2d!(buf, g, p.Nx, p.Ny; sync=true)
        g, buf = buf, g
        compute_conformation_macro_2d!(C, g)
        for j in margin:p.Ny-margin+1, i in margin:p.Nx-margin+1
            max_error = max(max_error, abs(C[i, j] - C_ref))
        end
    end
    return max_error
end

function _bulk_affine_cde_stationary_error(; orientation=:x_shear_y, steps=4)
    p = _bulk_affine_shear_patch(; orientation)
    Cxx = fill(p.cxx, p.Nx, p.Ny)
    Cxy = fill(p.cxy, p.Nx, p.Ny)
    Cyy = fill(p.cyy, p.Nx, p.Ny)
    gxx = zeros(Float64, p.Nx, p.Ny, 9)
    gxy = similar(gxx)
    gyy = similar(gxx)
    bxx = similar(gxx)
    bxy = similar(gxy)
    byy = similar(gyy)
    Fe_xx = similar(gxx)
    Fe_xy = similar(gxy)
    Fe_yy = similar(gyy)
    ρ = ones(Float64, p.Nx, p.Ny)
    fill!(Fe_xx, 0.0)
    fill!(Fe_xy, 0.0)
    fill!(Fe_yy, 0.0)
    init_conformation_field_2d!(gxx, Cxx, p.ux, p.uy)
    init_conformation_field_2d!(gxy, Cxy, p.ux, p.uy)
    init_conformation_field_2d!(gyy, Cyy, p.ux, p.uy)

    margin = steps + 4
    max_error = 0.0
    for _ in 1:steps
        stream_2d!(bxx, gxx, p.Nx, p.Ny; sync=true)
        stream_2d!(bxy, gxy, p.Nx, p.Ny; sync=true)
        stream_2d!(byy, gyy, p.Nx, p.Ny; sync=true)
        gxx, bxx = bxx, gxx
        gxy, bxy = bxy, gxy
        gyy, byy = byy, gyy
        compute_conformation_macro_2d!(Cxx, gxx)
        compute_conformation_macro_2d!(Cxy, gxy)
        compute_conformation_macro_2d!(Cyy, gyy)
        _collide_component_once!(
            :trt, gxx, Fe_xx, Cxx, p.ux, p.uy, ρ, Cxx, Cxy, Cyy,
            p.is_solid, 1.0, p.λ, 1; magic=1e-6, divergence_mode=:trace_free,
        )
        _collide_component_once!(
            :trt, gxy, Fe_xy, Cxy, p.ux, p.uy, ρ, Cxx, Cxy, Cyy,
            p.is_solid, 1.0, p.λ, 2; magic=1e-6, divergence_mode=:trace_free,
        )
        _collide_component_once!(
            :trt, gyy, Fe_yy, Cyy, p.ux, p.uy, ρ, Cxx, Cxy, Cyy,
            p.is_solid, 1.0, p.λ, 3; magic=1e-6, divergence_mode=:trace_free,
        )
        compute_conformation_macro_2d!(Cxx, gxx)
        compute_conformation_macro_2d!(Cxy, gxy)
        compute_conformation_macro_2d!(Cyy, gyy)

        for j in margin:p.Ny-margin+1, i in margin:p.Nx-margin+1
            max_error = max(
                max_error,
                abs(Cxx[i, j] - p.cxx),
                abs(Cxy[i, j] - p.cxy),
                abs(Cyy[i, j] - p.cyy),
            )
        end
    end
    return max_error
end

@testset "P15d bulk affine CDE separates transport from wall defects" begin
    for orientation in (:x_shear_y, :y_shear_x)
        @test _bulk_affine_transport_error(
            ; orientation, steps=4, constant_velocity=true,
        ) < P0_ATOL
        @test _bulk_affine_transport_error(
            ; orientation, steps=4, constant_velocity=false,
        ) < P0_ATOL
        @test _bulk_affine_cde_stationary_error(; orientation, steps=4) < P0_ATOL
    end
end

function _stream_periodic_y_wall_x_2d!(f_out, f_in, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        jm = j > 1 ? j - 1 : Ny
        jp = j < Ny ? j + 1 : 1

        f_out[i, j, 1] = f_in[i, j, 1]
        f_out[i, j, 2] = i > 1  ? f_in[i - 1, j, 2] : f_in[i, j, 4]
        f_out[i, j, 3] = f_in[i, jm, 3]
        f_out[i, j, 4] = i < Nx ? f_in[i + 1, j, 4] : f_in[i, j, 2]
        f_out[i, j, 5] = f_in[i, jp, 5]
        f_out[i, j, 6] = i > 1  ? f_in[i - 1, jm, 6] : f_in[i, j, 8]
        f_out[i, j, 7] = i < Nx ? f_in[i + 1, jm, 7] : f_in[i, j, 9]
        f_out[i, j, 8] = i < Nx ? f_in[i + 1, jp, 8] : f_in[i, j, 6]
        f_out[i, j, 9] = i > 1  ? f_in[i - 1, jp, 9] : f_in[i, j, 7]
    end
    return nothing
end

function _poiseuille_patch_profiles(Nx, Ny, u_mean, λ; orientation=:horizontal)
    u_max = 1.5 * u_mean
    ux = zeros(Float64, Nx, Ny)
    uy = zeros(Float64, Nx, Ny)
    Cxx = ones(Float64, Nx, Ny)
    Cxy = zeros(Float64, Nx, Ny)
    Cyy = ones(Float64, Nx, Ny)

    if orientation === :horizontal
        H = Float64(Ny)
        ref_Cxy = zeros(Float64, Ny)
        ref_N1 = zeros(Float64, Ny)
        for j in 1:Ny
            y = Float64(j) - 0.5
            u = 4.0 * u_max * y * (H - y) / (H * H)
            shear = 4.0 * u_max * (H - 2.0 * y) / (H * H)
            ref_Cxy[j] = λ * shear
            ref_N1[j] = 2.0 * (λ * shear)^2
            for i in 1:Nx
                ux[i, j] = u
                Cxy[i, j] = ref_Cxy[j]
                Cxx[i, j] = 1.0 + ref_N1[j]
            end
        end
        return ux, uy, Cxx, Cxy, Cyy, ref_Cxy, ref_N1
    elseif orientation === :vertical
        H = Float64(Nx)
        ref_Cxy = zeros(Float64, Nx)
        ref_N1 = zeros(Float64, Nx)
        for i in 1:Nx
            x = Float64(i) - 0.5
            v = 4.0 * u_max * x * (H - x) / (H * H)
            shear = 4.0 * u_max * (H - 2.0 * x) / (H * H)
            ref_Cxy[i] = λ * shear
            ref_N1[i] = -2.0 * (λ * shear)^2
            for j in 1:Ny
                uy[i, j] = v
                Cxy[i, j] = ref_Cxy[i]
                Cyy[i, j] = 1.0 - ref_N1[i]
            end
        end
        return ux, uy, Cxx, Cxy, Cyy, ref_Cxy, ref_N1
    end
    error("unknown Poiseuille patch orientation $(orientation)")
end

function _rel_l2_profile(profile, ref)
    den = sum(abs2, ref)
    den == 0.0 && return sqrt(sum(abs2, profile .- ref))
    return sqrt(sum(abs2, profile .- ref) / den)
end

function _apply_poiseuille_patch_wall_bc!(g_post, g_pre, is_solid, q_wall,
                                          C_field, ux, uy, bc, phi_mode)
    if bc === nothing
        apply_cnebb_conformation_2d!(
            g_post, g_pre, is_solid, C_field, ux, uy; phi_mode,
        )
    else
        apply_polymer_wall_bc!(g_post, g_pre, is_solid, q_wall, C_field, ux, uy, bc)
    end
    return nothing
end

function _poiseuille_cde_patch_error(; collision=:trt, phi_mode=:pre_opp,
                                     bc=nothing,
                                     tau_plus=1.0, steps=200,
                                     orientation=:horizontal,
                                     magic=2.5e-7)
    Nx, Ny = orientation === :vertical ? (16, 4) : (4, 16)
    R = 4.0
    u_mean = 0.005
    Wi = 0.1
    λ = Wi * R / u_mean
    ux, uy, Cxx, Cxy, Cyy, ref_Cxy, ref_N1 =
        _poiseuille_patch_profiles(Nx, Ny, u_mean, λ; orientation)
    is_solid = falses(Nx, Ny)
    q_wall = zeros(Float64, Nx, Ny, 9)
    ρ = ones(Float64, Nx, Ny)

    g_xx = zeros(Float64, Nx, Ny, 9)
    g_xy = zeros(Float64, Nx, Ny, 9)
    g_yy = zeros(Float64, Nx, Ny, 9)
    init_conformation_field_2d!(g_xx, Cxx, ux, uy)
    init_conformation_field_2d!(g_xy, Cxy, ux, uy)
    init_conformation_field_2d!(g_yy, Cyy, ux, uy)
    g_xx_buf = similar(g_xx)
    g_xy_buf = similar(g_xy)
    g_yy_buf = similar(g_yy)
    Fe_xx = zeros(Float64, Nx, Ny, 9)
    Fe_xy = zeros(Float64, Nx, Ny, 9)
    Fe_yy = zeros(Float64, Nx, Ny, 9)
    prod_mode = collision === :trt_embedded_axis ? :embedded_axis :
                collision === :trt_wallfit4 ? :wallfit4 : nothing
    stencils = prod_mode === nothing ? nothing :
        Kraken.precompute_conformation_gradient_stencils_2d(
            is_solid, q_wall; mode=prod_mode,
            max_terms=prod_mode === :embedded_axis ? 4 : 64,
            FT=Float64,
        )
    uwx = prod_mode === nothing ? nothing : zeros(Float64, Nx, Ny, 9)
    uwy = prod_mode === nothing ? nothing : zeros(Float64, Nx, Ny, 9)

    for _ in 1:steps
        if orientation === :horizontal
            stream_periodic_x_wall_y_2d!(g_xx_buf, g_xx, Nx, Ny)
            stream_periodic_x_wall_y_2d!(g_xy_buf, g_xy, Nx, Ny)
            stream_periodic_x_wall_y_2d!(g_yy_buf, g_yy, Nx, Ny)
        else
            _stream_periodic_y_wall_x_2d!(g_xx_buf, g_xx, Nx, Ny)
            _stream_periodic_y_wall_x_2d!(g_xy_buf, g_xy, Nx, Ny)
            _stream_periodic_y_wall_x_2d!(g_yy_buf, g_yy, Nx, Ny)
        end

        _apply_poiseuille_patch_wall_bc!(
            g_xx_buf, g_xx, is_solid, q_wall, Cxx, ux, uy, bc, phi_mode,
        )
        _apply_poiseuille_patch_wall_bc!(
            g_xy_buf, g_xy, is_solid, q_wall, Cxy, ux, uy, bc, phi_mode,
        )
        _apply_poiseuille_patch_wall_bc!(
            g_yy_buf, g_yy, is_solid, q_wall, Cyy, ux, uy, bc, phi_mode,
        )

        g_xx, g_xx_buf = g_xx_buf, g_xx
        g_xy, g_xy_buf = g_xy_buf, g_xy
        g_yy, g_yy_buf = g_yy_buf, g_yy

        compute_conformation_macro_2d!(Cxx, g_xx)
        compute_conformation_macro_2d!(Cxy, g_xy)
        compute_conformation_macro_2d!(Cyy, g_yy)

        for (g, Fe, C_field, component) in
            ((g_xx, Fe_xx, Cxx, 1), (g_xy, Fe_xy, Cxy, 2), (g_yy, Fe_yy, Cyy, 3))
            _collide_component_once!(
                collision, g, Fe, C_field, ux, uy, ρ, Cxx, Cxy, Cyy,
                is_solid, tau_plus, λ, component; magic, stencils, uwx, uwy,
            )
        end
    end

    Cxy_profile = orientation === :horizontal ? vec(mean(Cxy, dims=1)) :
                                                 vec(mean(Cxy, dims=2))
    N1_profile = orientation === :horizontal ? vec(mean(Cxx .- Cyy, dims=1)) :
                                                vec(mean(Cxx .- Cyy, dims=2))
    return (
        Cxy_l2 = _rel_l2_profile(Cxy_profile, ref_Cxy),
        N1_l2 = _rel_l2_profile(N1_profile, ref_N1),
        min_Cyy = minimum(Cyy),
    )
end

function _poiseuille_prod_gradient_source_residual(; orientation=:horizontal,
                                                   mode::Symbol=:embedded_axis)
    Nx, Ny = orientation === :vertical ? (16, 4) : (4, 16)
    R = 4.0
    u_mean = 0.005
    Wi = 0.1
    λ = Wi * R / u_mean
    ux, uy, Cxx, Cxy, Cyy, _, _ =
        _poiseuille_patch_profiles(Nx, Ny, u_mean, λ; orientation)
    is_solid = falses(Nx, Ny)
    q_wall = zeros(Float64, Nx, Ny, 9)
    stencils = Kraken.precompute_conformation_gradient_stencils_2d(
        is_solid, q_wall; mode,
        max_terms=mode === :embedded_axis ? 4 : 64,
        FT=Float64,
    )
    uwx = zeros(Float64, Nx, Ny, 9)
    uwy = zeros(Float64, Nx, Ny, 9)
    stats = Kraken.conformation_gradient_stencil_stats_2d(stencils)

    max_source = 0.0
    max_gradient_gap = 0.0
    for j in 1:Ny, i in 1:Nx
        prod = Kraken.conformation_velocity_gradient_from_stencils_2d(
            ux, uy, uwx, uwy, stencils, i, j,
        )
        ref_dudx = Kraken._wall_aware_dx_2d(ux, is_solid, i, j, Nx, Float64)
        ref_dudy = Kraken._wall_aware_dy_2d(ux, is_solid, i, j, Ny, Float64)
        ref_dvdx = Kraken._wall_aware_dx_2d(uy, is_solid, i, j, Nx, Float64)
        ref_dvdy = Kraken._wall_aware_dy_2d(uy, is_solid, i, j, Ny, Float64)
        max_gradient_gap = max(max_gradient_gap,
            abs(prod.dudx - ref_dudx), abs(prod.dudy - ref_dudy),
            abs(prod.dvdx - ref_dvdx), abs(prod.dvdy - ref_dvdy),
        )
        source = _direct_source_tuple(
            Cxx[i, j], Cxy[i, j], Cyy[i, j],
            prod.dudx, prod.dudy, prod.dvdx, prod.dvdy, λ,
        )
        max_source = max(max_source, maximum(abs, source))
    end
    return (; max_source, max_gradient_gap, stats)
end

@testset "P16 Poiseuille CDE analytic patch: supported collision windows" begin
    for orientation in (:horizontal, :vertical)
        trt_tau1 = _poiseuille_cde_patch_error(
            collision=:trt, tau_plus=1.0; orientation,
        )
        @test trt_tau1.Cxy_l2 < 0.10
        @test trt_tau1.N1_l2 < 0.20
        @test trt_tau1.min_Cyy > 0.0

        for collision in (:regularized, :liu_eq26)
            high_sc = _poiseuille_cde_patch_error(
                collision=collision, tau_plus=0.50001; orientation,
            )
            @test high_sc.Cxy_l2 < 0.25
            @test high_sc.N1_l2 < 0.40
            @test high_sc.min_Cyy > 0.0
        end
    end
end

@testset "P17 Poiseuille CDE analytic patch: known unsupported/broken paths" begin
    broken_results = []
    for orientation in (:horizontal, :vertical)
        for collision in (:regularized, :liu_eq26)
            push!(
                broken_results,
                _poiseuille_cde_patch_error(
                    collision=collision, tau_plus=1.0; orientation,
                ),
            )
        end
        push!(
            broken_results,
            _poiseuille_cde_patch_error(
                collision=:trt, tau_plus=0.50001; orientation,
            ),
        )
        for collision in (:trt, :regularized, :liu_eq26), tau_plus in (1.0, 0.50001)
            push!(
                broken_results,
                _poiseuille_cde_patch_error(
                    collision=collision, tau_plus=tau_plus,
                    phi_mode=:eq_gradient; orientation,
                ),
            )
        end
    end
    for result in broken_results
        @test !(isfinite(result.Cxy_l2) &&
                isfinite(result.N1_l2) &&
                result.Cxy_l2 < 0.25 &&
                result.N1_l2 < 0.40 &&
                result.min_Cyy > 0.0)
    end
end

function _planar_wall_bc_cases_for_tau(tau_plus)
    return (
        (name=:cnebb, bc=CNEBB()),
        (name=:cnebb_qaware, bc=CNEBBQAware()),
        (name=:cnebb_field, bc=CNEBBField()),
        (name=:cnebb_field_equilibrium, bc=CNEBBFieldEquilibrium()),
        (name=:cnebb_eq_gradient, bc=CNEBBEqGradient()),
        (name=:cnebb_cutlink_eq_gradient, bc=CNEBBCutLinkEqGradient()),
        (name=:ylw_a, bc=YLW_A(tau_plus=tau_plus)),
        (name=:ylw_b, bc=YLW_B(tau_plus=tau_plus)),
        (name=:ylw_balance, bc=YLWBalanceOnly()),
        (name=:none, bc=NoPolymerWallBC()),
    )
end

@testset "P18 Poiseuille CDE analytic patch: planar wall BC matrix" begin
    for orientation in (:horizontal, :vertical),
        (collision, tau_plus) in ((:trt, 1.0), (:regularized, 0.50001), (:liu_eq26, 0.50001)),
        case in _planar_wall_bc_cases_for_tau(tau_plus)

        result = _poiseuille_cde_patch_error(
            collision=collision, tau_plus=tau_plus, bc=case.bc; orientation,
        )
        if case.name === :cnebb_eq_gradient
            @test !(isfinite(result.Cxy_l2) &&
                    isfinite(result.N1_l2) &&
                    result.Cxy_l2 < 0.25 &&
                    result.N1_l2 < 0.40 &&
                    result.min_Cyy > 0.0)
        else
            @test result.Cxy_l2 < 0.25
            @test result.N1_l2 < 0.40
            @test result.min_Cyy > 0.0
        end
    end
end

@testset "P18b Poiseuille CDE analytic patch: TRT magic parameter is part of validation" begin
    for orientation in (:horizontal, :vertical)
        liu_magic = _poiseuille_cde_patch_error(
            collision=:trt, tau_plus=1.0, bc=CNEBB();
            orientation, magic=1e-6,
        )
        historical_magic = _poiseuille_cde_patch_error(
            collision=:trt, tau_plus=1.0, bc=CNEBB();
            orientation, magic=0.25,
        )

        @test liu_magic.Cxy_l2 < 0.10
        @test liu_magic.N1_l2 < 0.20
        @test historical_magic.Cxy_l2 > 0.25
        @test historical_magic.N1_l2 > 0.40
    end
end

@testset "P18b2 Poiseuille CDE analytic patch: production gradient stencils preserve planar walls" begin
    for orientation in (:horizontal, :vertical),
        mode in (:embedded_axis, :wallfit4)

        source = _poiseuille_prod_gradient_source_residual(; orientation, mode)
        @test source.max_gradient_gap < P0_ATOL
        @test source.max_source < P0_ATOL
    end

    for orientation in (:horizontal, :vertical),
        collision in (:trt_embedded_axis, :trt_wallfit4)

        result = _poiseuille_cde_patch_error(
            collision=collision, tau_plus=1.0, bc=CNEBB();
            orientation, magic=1e-6,
        )
        @test result.Cxy_l2 < 0.10
        @test result.N1_l2 < 0.20
        @test result.min_Cyy > 0.0
    end
end

function _straight_embedded_wall_macro_errors(bc; orientation=:horizontal,
                                              field=:tangent,
                                              velocity=(0.0, 0.0))
    Nx, Ny = 14, 12
    is_solid = falses(Nx, Ny)
    if orientation === :horizontal
        is_solid[:, 1:4] .= true
        cut_range = ((3:Nx-2), (5:5))
        tangent = (1.0, 0.0)
        normal = (0.0, 1.0)
    elseif orientation === :vertical
        is_solid[1:4, :] .= true
        cut_range = ((5:5), (3:Ny-2))
        tangent = (0.0, 1.0)
        normal = (1.0, 0.0)
    else
        error("unknown straight wall orientation $(orientation)")
    end

    q_wall = zeros(Float64, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx
        is_solid[i, j] && continue
        for q in 2:9
            ni = i + Int(D2Q9_CX[q])
            nj = j + Int(D2Q9_CY[q])
            if 1 <= ni <= Nx && 1 <= nj <= Ny && is_solid[ni, nj]
                q_wall[i, j, q] = 0.5
            end
        end
    end

    cx0, cy0 = 6.5, 4.5
    slope = 0.08
    C0 = [
        field === :uniform ? 1.0 :
        field === :tangent ? 1.0 + slope * (((i - 1.0) - cx0) * tangent[1] +
                                             ((j - 1.0) - cy0) * tangent[2]) :
        field === :normal ? 1.0 + slope * (((i - 1.0) - cx0) * normal[1] +
                                            ((j - 1.0) - cy0) * normal[2]) :
        error("unknown straight wall field $(field)")
        for i in 1:Nx, j in 1:Ny
    ]
    ux = fill(velocity[1], Nx, Ny)
    uy = fill(velocity[2], Nx, Ny)
    g_pre = _fill_constant_velocity_equilibrium!(
        zeros(Float64, Nx, Ny, 9), C0, velocity[1], velocity[2],
    )
    g_post = similar(g_pre)
    stream_2d!(g_post, g_pre, Nx, Ny; sync=true)
    C_after = copy(C0)
    apply_polymer_wall_bc!(g_post, g_pre, is_solid, q_wall, C_after, ux, uy, bc)

    max_sum_error = 0.0
    max_macro_error = 0.0
    n_cut_cells = 0
    for j in cut_range[2], i in cut_range[1]
        any(q -> q > 0.0, view(q_wall, i, j, :)) || continue
        n_cut_cells += 1
        max_sum_error = max(
            max_sum_error,
            abs(sum(g_post[i, j, q] for q in 1:9) - C_after[i, j]),
        )
        max_macro_error = max(max_macro_error, abs(C_after[i, j] - C0[i, j]))
    end
    return (; max_sum_error, max_macro_error, n_cut_cells)
end

@testset "P18c straight embedded q=0.5 wall matrix before curved tests" begin
    cutlink_tangent_errors = Float64[]
    for case in _wall_bc_cases(), orientation in (:horizontal, :vertical)
        uniform = _straight_embedded_wall_macro_errors(
            case.bc; orientation, field=:uniform,
        )
        tangent = _straight_embedded_wall_macro_errors(
            case.bc; orientation, field=:tangent,
        )
        @test uniform.n_cut_cells > 0
        @test tangent.n_cut_cells > 0
        @test uniform.max_sum_error < P0_ATOL
        @test tangent.max_sum_error < P0_ATOL
        @test uniform.max_macro_error < P0_ATOL
        if case.name === :cnebb_cutlink_eq_gradient
            push!(cutlink_tangent_errors, tangent.max_macro_error)
        else
            @test tangent.max_macro_error < P0_ATOL
        end
    end
    @test maximum(cutlink_tangent_errors) > 1e-6

    cutlink_active_errors = Float64[]
    for case in _wall_bc_cases()
        horizontal = _straight_embedded_wall_macro_errors(
            case.bc; orientation=:horizontal, field=:uniform,
            velocity=(0.03, 0.0),
        )
        vertical = _straight_embedded_wall_macro_errors(
            case.bc; orientation=:vertical, field=:uniform,
            velocity=(0.0, -0.02),
        )
        @test horizontal.max_sum_error < P0_ATOL
        @test vertical.max_sum_error < P0_ATOL
        if case.name === :cnebb_cutlink_eq_gradient
            push!(cutlink_active_errors, horizontal.max_macro_error)
            push!(cutlink_active_errors, vertical.max_macro_error)
        else
            @test horizontal.max_macro_error < P0_ATOL
            @test vertical.max_macro_error < P0_ATOL
        end
    end
    @test maximum(cutlink_active_errors) > 1e-6
end

function _straight_couette_oldroydb_patch(; orientation=:horizontal)
    Nx, Ny = 32, 28
    γ = 0.01
    λ = 3.0
    is_solid = falses(Nx, Ny)
    ux = zeros(Float64, Nx, Ny)
    uy = zeros(Float64, Nx, Ny)
    if orientation === :horizontal
        is_solid[:, 1:4] .= true
        y_wall = 3.5
        for j in 1:Ny, i in 1:Nx
            ux[i, j] = is_solid[i, j] ? 0.0 : γ * ((j - 1.0) - y_wall)
        end
        cxx, cxy, cyy = _stationary_direct_conformation_incompressible(
            0.0, γ, 0.0, 0.0, λ,
        )
    elseif orientation === :vertical
        is_solid[1:4, :] .= true
        x_wall = 3.5
        for j in 1:Ny, i in 1:Nx
            uy[i, j] = is_solid[i, j] ? 0.0 : γ * ((i - 1.0) - x_wall)
        end
        cxx, cxy, cyy = _stationary_direct_conformation_incompressible(
            0.0, 0.0, γ, 0.0, λ,
        )
    else
        error("unknown straight Couette orientation $(orientation)")
    end

    q_wall = zeros(Float64, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx
        is_solid[i, j] && continue
        for q in 2:9
            ni = i + Int(D2Q9_CX[q])
            nj = j + Int(D2Q9_CY[q])
            if 1 <= ni <= Nx && 1 <= nj <= Ny && is_solid[ni, nj]
                q_wall[i, j, q] = 0.5
            end
        end
    end
    return (; Nx, Ny, λ, q_wall, is_solid, ux, uy, cxx, cxy, cyy)
end

function _collide_direct_cde_state!(gxx, gxy, gyy, Cxx, Cxy, Cyy,
                                    ux, uy, is_solid, λ)
    collide_conformation_2d!(gxx, Cxx, ux, uy, Cxx, Cxy, Cyy,
                             is_solid, 1.0, λ;
                             magic=1e-6, component=1, divergence_mode=:trace_free)
    collide_conformation_2d!(gxy, Cxy, ux, uy, Cxx, Cxy, Cyy,
                             is_solid, 1.0, λ;
                             magic=1e-6, component=2, divergence_mode=:trace_free)
    collide_conformation_2d!(gyy, Cyy, ux, uy, Cxx, Cxy, Cyy,
                             is_solid, 1.0, λ;
                             magic=1e-6, component=3, divergence_mode=:trace_free)
    compute_conformation_macro_2d!(Cxx, gxx)
    compute_conformation_macro_2d!(Cxy, gxy)
    compute_conformation_macro_2d!(Cyy, gyy)
    return nothing
end

function _straight_couette_cde_once(bc; orientation=:horizontal)
    p = _straight_couette_oldroydb_patch(; orientation)
    Cxx = fill(p.cxx, p.Nx, p.Ny)
    Cxy = fill(p.cxy, p.Nx, p.Ny)
    Cyy = fill(p.cyy, p.Nx, p.Ny)
    gxx = zeros(Float64, p.Nx, p.Ny, 9)
    gxy = similar(gxx)
    gyy = similar(gxx)
    init_conformation_field_2d!(gxx, Cxx, p.ux, p.uy)
    init_conformation_field_2d!(gxy, Cxy, p.ux, p.uy)
    init_conformation_field_2d!(gyy, Cyy, p.ux, p.uy)
    bxx = similar(gxx)
    bxy = similar(gxy)
    byy = similar(gyy)

    stream_2d!(bxx, gxx, p.Nx, p.Ny; sync=true)
    stream_2d!(bxy, gxy, p.Nx, p.Ny; sync=true)
    stream_2d!(byy, gyy, p.Nx, p.Ny; sync=true)
    apply_polymer_wall_bc!(bxx, gxx, p.is_solid, p.q_wall, Cxx, p.ux, p.uy, bc)
    apply_polymer_wall_bc!(bxy, gxy, p.is_solid, p.q_wall, Cxy, p.ux, p.uy, bc)
    apply_polymer_wall_bc!(byy, gyy, p.is_solid, p.q_wall, Cyy, p.ux, p.uy, bc)
    gxx, bxx = bxx, gxx
    gxy, bxy = bxy, gxy
    gyy, byy = byy, gyy

    compute_conformation_macro_2d!(Cxx, gxx)
    compute_conformation_macro_2d!(Cxy, gxy)
    compute_conformation_macro_2d!(Cyy, gyy)
    collide_conformation_2d!(gxx, Cxx, p.ux, p.uy, Cxx, Cxy, Cyy,
                             p.is_solid, 1.0, p.λ;
                             magic=1e-6, component=1, divergence_mode=:trace_free)
    collide_conformation_2d!(gxy, Cxy, p.ux, p.uy, Cxx, Cxy, Cyy,
                             p.is_solid, 1.0, p.λ;
                             magic=1e-6, component=2, divergence_mode=:trace_free)
    collide_conformation_2d!(gyy, Cyy, p.ux, p.uy, Cxx, Cxy, Cyy,
                             p.is_solid, 1.0, p.λ;
                             magic=1e-6, component=3, divergence_mode=:trace_free)
    compute_conformation_macro_2d!(Cxx, gxx)
    compute_conformation_macro_2d!(Cxy, gxy)
    compute_conformation_macro_2d!(Cyy, gyy)

    max_cut = 0.0
    max_far = 0.0
    for j in 3:p.Ny-2, i in 3:p.Nx-2
        p.is_solid[i, j] && continue
        err = max(abs(Cxx[i, j] - p.cxx), abs(Cxy[i, j] - p.cxy),
                  abs(Cyy[i, j] - p.cyy))
        if any(q -> q > 0.0, view(p.q_wall, i, j, :))
            max_cut = max(max_cut, err)
        else
            max_far = max(max_far, err)
        end
    end
    return (; max_cut, max_far)
end

function _straight_couette_cde_repeated(bc; orientation=:horizontal, steps=4)
    p = _straight_couette_oldroydb_patch(; orientation)
    Cxx = fill(p.cxx, p.Nx, p.Ny)
    Cxy = fill(p.cxy, p.Nx, p.Ny)
    Cyy = fill(p.cyy, p.Nx, p.Ny)
    gxx = zeros(Float64, p.Nx, p.Ny, 9)
    gxy = similar(gxx)
    gyy = similar(gxx)
    init_conformation_field_2d!(gxx, Cxx, p.ux, p.uy)
    init_conformation_field_2d!(gxy, Cxy, p.ux, p.uy)
    init_conformation_field_2d!(gyy, Cyy, p.ux, p.uy)
    bxx = similar(gxx)
    bxy = similar(gxy)
    byy = similar(gyy)

    max_cut = 0.0
    max_far = 0.0
    min_eig = Inf
    n_cut = 0
    n_far = 0
    margin = steps + 3
    for _ in 1:steps
        stream_2d!(bxx, gxx, p.Nx, p.Ny; sync=true)
        stream_2d!(bxy, gxy, p.Nx, p.Ny; sync=true)
        stream_2d!(byy, gyy, p.Nx, p.Ny; sync=true)
        apply_polymer_wall_bc!(bxx, gxx, p.is_solid, p.q_wall, Cxx, p.ux, p.uy, bc)
        apply_polymer_wall_bc!(bxy, gxy, p.is_solid, p.q_wall, Cxy, p.ux, p.uy, bc)
        apply_polymer_wall_bc!(byy, gyy, p.is_solid, p.q_wall, Cyy, p.ux, p.uy, bc)
        gxx, bxx = bxx, gxx
        gxy, bxy = bxy, gxy
        gyy, byy = byy, gyy

        compute_conformation_macro_2d!(Cxx, gxx)
        compute_conformation_macro_2d!(Cxy, gxy)
        compute_conformation_macro_2d!(Cyy, gyy)
        _collide_direct_cde_state!(gxx, gxy, gyy, Cxx, Cxy, Cyy,
                                   p.ux, p.uy, p.is_solid, p.λ)

        irange = orientation === :horizontal ? (margin:p.Nx-margin+1) :
                 orientation === :vertical ? (5:p.Nx-margin+1) :
                 error("unknown straight Couette orientation $(orientation)")
        jrange = orientation === :horizontal ? (5:p.Ny-margin+1) :
                 orientation === :vertical ? (margin:p.Ny-margin+1) :
                 error("unknown straight Couette orientation $(orientation)")
        for j in jrange, i in irange
            p.is_solid[i, j] && continue
            err = max(abs(Cxx[i, j] - p.cxx), abs(Cxy[i, j] - p.cxy),
                      abs(Cyy[i, j] - p.cyy))
            min_eig = min(min_eig, _min_eig_spd_2x2(Cxx[i, j], Cxy[i, j], Cyy[i, j]))
            if any(q -> q > 0.0, view(p.q_wall, i, j, :))
                n_cut += 1
                max_cut = max(max_cut, err)
            else
                n_far += 1
                max_far = max(max_far, err)
            end
        end
    end
    return (; max_cut, max_far, min_eig, n_cut, n_far)
end

@testset "P18d straight Couette CDE one-step exact before curved tests" begin
    cutlink_errors = Float64[]
    for case in _wall_bc_cases(), orientation in (:horizontal, :vertical)
        result = _straight_couette_cde_once(case.bc; orientation)
        if case.name === :cnebb_cutlink_eq_gradient
            push!(cutlink_errors, result.max_cut)
        else
            @test result.max_cut < P0_ATOL
            @test result.max_far < P0_ATOL
        end
    end
    @test maximum(cutlink_errors) > 1e-6
end

@testset "P18d2 straight Couette CDE repeated q=0.5 before cut-link tests" begin
    repeated_tol = 1e-6
    for case in _wall_bc_cases(), orientation in (:horizontal, :vertical)
        result = _straight_couette_cde_repeated(case.bc; orientation, steps=4)
        @test result.n_cut > 0
        @test result.n_far > 0
        @test result.min_eig > 0.9
        @test result.max_cut < repeated_tol
        @test result.max_far < repeated_tol
    end
end

function _single_cutlink_wall_aligned_couette_once(q_out, q_wall_value, bc)
    Nx = Ny = 9
    i0 = j0 = 5
    cx = D2Q9_CX[q_out]
    cy = D2Q9_CY[q_out]
    link_length = hypot(cx, cy)
    nx = cx / link_length
    ny = cy / link_length
    tx = -ny
    ty = nx
    γ = 0.01
    λ = 3.0
    wall_x = (i0 - 1.0) + q_wall_value * cx
    wall_y = (j0 - 1.0) + q_wall_value * cy

    is_solid = falses(Nx, Ny)
    is_solid[i0 + Int(cx), j0 + Int(cy)] = true
    q_wall = zeros(Float64, Nx, Ny, 9)
    q_wall[i0, j0, q_out] = q_wall_value
    ux = zeros(Float64, Nx, Ny)
    uy = zeros(Float64, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        is_solid[i, j] && continue
        distance = (wall_x - (i - 1.0)) * nx + (wall_y - (j - 1.0)) * ny
        ux[i, j] = γ * distance * tx
        uy[i, j] = γ * distance * ty
    end
    dudx = -γ * tx * nx
    dudy = -γ * tx * ny
    dvdx = -γ * ty * nx
    dvdy = -γ * ty * ny
    cxx, cxy, cyy = _stationary_direct_conformation_incompressible(
        dudx, dudy, dvdx, dvdy, λ,
    )

    Cxx = fill(cxx, Nx, Ny)
    Cxy = fill(cxy, Nx, Ny)
    Cyy = fill(cyy, Nx, Ny)
    gxx = zeros(Float64, Nx, Ny, 9)
    gxy = similar(gxx)
    gyy = similar(gxx)
    init_conformation_field_2d!(gxx, Cxx, ux, uy)
    init_conformation_field_2d!(gxy, Cxy, ux, uy)
    init_conformation_field_2d!(gyy, Cyy, ux, uy)
    bxx = similar(gxx)
    bxy = similar(gxy)
    byy = similar(gyy)

    stream_2d!(bxx, gxx, Nx, Ny; sync=true)
    stream_2d!(bxy, gxy, Nx, Ny; sync=true)
    stream_2d!(byy, gyy, Nx, Ny; sync=true)
    apply_polymer_wall_bc!(bxx, gxx, is_solid, q_wall, Cxx, ux, uy, bc)
    apply_polymer_wall_bc!(bxy, gxy, is_solid, q_wall, Cxy, ux, uy, bc)
    apply_polymer_wall_bc!(byy, gyy, is_solid, q_wall, Cyy, ux, uy, bc)
    gxx, bxx = bxx, gxx
    gxy, bxy = bxy, gxy
    gyy, byy = byy, gyy

    compute_conformation_macro_2d!(Cxx, gxx)
    compute_conformation_macro_2d!(Cxy, gxy)
    compute_conformation_macro_2d!(Cyy, gyy)
    wall_error = max(abs(Cxx[i0, j0] - cxx), abs(Cxy[i0, j0] - cxy),
                     abs(Cyy[i0, j0] - cyy))

    _collide_direct_cde_state!(gxx, gxy, gyy, Cxx, Cxy, Cyy,
                               ux, uy, is_solid, λ)
    cde_error = max(abs(Cxx[i0, j0] - cxx), abs(Cxy[i0, j0] - cxy),
                    abs(Cyy[i0, j0] - cyy))
    return (; wall_error, cde_error)
end

function _single_cutlink_wall_aligned_couette_repeated(q_out, q_wall_value, bc;
                                                       steps=20)
    Nx = Ny = 9
    i0 = j0 = 5
    cx = D2Q9_CX[q_out]
    cy = D2Q9_CY[q_out]
    link_length = hypot(cx, cy)
    nx = cx / link_length
    ny = cy / link_length
    tx = -ny
    ty = nx
    γ = 0.01
    λ = 3.0
    wall_x = (i0 - 1.0) + q_wall_value * cx
    wall_y = (j0 - 1.0) + q_wall_value * cy

    is_solid = falses(Nx, Ny)
    is_solid[i0 + Int(cx), j0 + Int(cy)] = true
    q_wall = zeros(Float64, Nx, Ny, 9)
    q_wall[i0, j0, q_out] = q_wall_value
    ux = zeros(Float64, Nx, Ny)
    uy = zeros(Float64, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        is_solid[i, j] && continue
        distance = (wall_x - (i - 1.0)) * nx + (wall_y - (j - 1.0)) * ny
        ux[i, j] = γ * distance * tx
        uy[i, j] = γ * distance * ty
    end
    dudx = -γ * tx * nx
    dudy = -γ * tx * ny
    dvdx = -γ * ty * nx
    dvdy = -γ * ty * ny
    cxx, cxy, cyy = _stationary_direct_conformation_incompressible(
        dudx, dudy, dvdx, dvdy, λ,
    )

    Cxx = fill(cxx, Nx, Ny)
    Cxy = fill(cxy, Nx, Ny)
    Cyy = fill(cyy, Nx, Ny)
    gxx = zeros(Float64, Nx, Ny, 9)
    gxy = similar(gxx)
    gyy = similar(gxx)
    init_conformation_field_2d!(gxx, Cxx, ux, uy)
    init_conformation_field_2d!(gxy, Cxy, ux, uy)
    init_conformation_field_2d!(gyy, Cyy, ux, uy)
    bxx = similar(gxx)
    bxy = similar(gxy)
    byy = similar(gyy)

    max_wall_error = 0.0
    max_cde_error = 0.0
    min_eig = Inf
    for _ in 1:steps
        stream_2d!(bxx, gxx, Nx, Ny; sync=true)
        stream_2d!(bxy, gxy, Nx, Ny; sync=true)
        stream_2d!(byy, gyy, Nx, Ny; sync=true)
        apply_polymer_wall_bc!(bxx, gxx, is_solid, q_wall, Cxx, ux, uy, bc)
        apply_polymer_wall_bc!(bxy, gxy, is_solid, q_wall, Cxy, ux, uy, bc)
        apply_polymer_wall_bc!(byy, gyy, is_solid, q_wall, Cyy, ux, uy, bc)
        gxx, bxx = bxx, gxx
        gxy, bxy = bxy, gxy
        gyy, byy = byy, gyy

        compute_conformation_macro_2d!(Cxx, gxx)
        compute_conformation_macro_2d!(Cxy, gxy)
        compute_conformation_macro_2d!(Cyy, gyy)
        wall_error = max(abs(Cxx[i0, j0] - cxx), abs(Cxy[i0, j0] - cxy),
                         abs(Cyy[i0, j0] - cyy))
        max_wall_error = max(max_wall_error, wall_error)

        _collide_direct_cde_state!(gxx, gxy, gyy, Cxx, Cxy, Cyy,
                                   ux, uy, is_solid, λ)
        cde_error = max(abs(Cxx[i0, j0] - cxx), abs(Cxy[i0, j0] - cxy),
                        abs(Cyy[i0, j0] - cyy))
        max_cde_error = max(max_cde_error, cde_error)
        min_eig = min(min_eig, _min_eig_spd_2x2(Cxx[i0, j0], Cxy[i0, j0], Cyy[i0, j0]))
    end
    return (; max_wall_error, max_cde_error, min_eig)
end

@testset "P18e single cut-link q-aware Couette canary before curved geometry" begin
    cutlink_errors = Float64[]
    for case in _wall_bc_cases(),
        q_out in 2:9,
        q_wall_value in (0.3, 0.5, 0.7)

        result = _single_cutlink_wall_aligned_couette_once(
            q_out, q_wall_value, case.bc,
        )
        if case.name in (:cnebb_field, :cnebb_field_equilibrium, :cnebb_eq_gradient)
            @test result.wall_error < P0_ATOL
            @test result.cde_error < P0_ATOL
        elseif case.name === :cnebb_cutlink_eq_gradient
            push!(cutlink_errors, result.wall_error)
        elseif q_wall_value == 0.5
            @test result.wall_error < P0_ATOL
            @test result.cde_error < P0_ATOL
        else
            @test result.wall_error > 1e-6
            @test result.cde_error > 1e-6
        end
    end
    @test maximum(cutlink_errors) > 1e-6
end

@testset "P18f repeated single cut-link Couette canary before curved geometry" begin
    broken_errors = Float64[]
    for case in _wall_bc_cases(),
        q_out in 2:9,
        q_wall_value in (0.3, 0.7)

        result = _single_cutlink_wall_aligned_couette_repeated(
            q_out, q_wall_value, case.bc; steps=12,
        )
        @test result.min_eig > 0.9
        if case.name in (:cnebb_field, :cnebb_field_equilibrium)
            @test result.max_wall_error < P0_ATOL
            @test result.max_cde_error < P0_ATOL
        else
            push!(broken_errors, result.max_wall_error)
            @test result.max_wall_error > 1e-6
            @test result.max_cde_error > 1e-6
        end
    end
    @test maximum(broken_errors) > 1e-6
end

_is_cut_cell(q_wall, i, j) = any(q -> q > 0.0, view(q_wall, i, j, :))

function _reset_cut_cells_to_equilibrium!(g, C, ux, uy, is_solid, q_wall)
    Nx, Ny = size(C)
    for j in 1:Ny, i in 1:Nx
        is_solid[i, j] && continue
        _is_cut_cell(q_wall, i, j) || continue
        ϕ = C[i, j]
        u = ux[i, j]
        v = uy[i, j]
        for q in 1:9
            g[i, j, q] = equilibrium(D2Q9(), ϕ, u, v, q)
        end
    end
    return nothing
end

function _inclined_straight_couette_patch(; normal=(3.0, 4.0), γ=0.01)
    Nx = Ny = 48
    norm_n = hypot(normal[1], normal[2])
    nx = normal[1] / norm_n
    ny = normal[2] / norm_n
    tx = -ny
    ty = nx
    λ = 3.0
    x0 = 22.37
    y0 = 19.81
    signed_distance(i, j) = nx * ((i - 1.0) - x0) + ny * ((j - 1.0) - y0)

    is_solid = falses(Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        is_solid[i, j] = signed_distance(i, j) < 0.0
    end

    q_wall = zeros(Float64, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx
        is_solid[i, j] && continue
        φ0 = signed_distance(i, j)
        for q in 2:9
            ni = i + Int(D2Q9_CX[q])
            nj = j + Int(D2Q9_CY[q])
            if 1 <= ni <= Nx && 1 <= nj <= Ny && is_solid[ni, nj]
                φ1 = signed_distance(ni, nj)
                q_wall[i, j, q] = φ0 / (φ0 - φ1)
            end
        end
    end

    ux = zeros(Float64, Nx, Ny)
    uy = zeros(Float64, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        is_solid[i, j] && continue
        distance = signed_distance(i, j)
        ux[i, j] = γ * distance * tx
        uy[i, j] = γ * distance * ty
    end

    dudx = γ * tx * nx
    dudy = γ * tx * ny
    dvdx = γ * ty * nx
    dvdy = γ * ty * ny
    cxx, cxy, cyy = _stationary_direct_conformation_incompressible(
        dudx, dudy, dvdx, dvdy, λ,
    )
    return (; Nx, Ny, λ, q_wall, is_solid, ux, uy, cxx, cxy, cyy)
end

function _inclined_straight_couette_repeated(bc; normal=(3.0, 4.0), γ=0.01, steps=4,
                                             reset_cut_equilibrium=false,
                                             reset_cut_after_collision=false)
    p = _inclined_straight_couette_patch(; normal, γ)
    Cxx = fill(p.cxx, p.Nx, p.Ny)
    Cxy = fill(p.cxy, p.Nx, p.Ny)
    Cyy = fill(p.cyy, p.Nx, p.Ny)
    gxx = zeros(Float64, p.Nx, p.Ny, 9)
    gxy = similar(gxx)
    gyy = similar(gxx)
    init_conformation_field_2d!(gxx, Cxx, p.ux, p.uy)
    init_conformation_field_2d!(gxy, Cxy, p.ux, p.uy)
    init_conformation_field_2d!(gyy, Cyy, p.ux, p.uy)
    bxx = similar(gxx)
    bxy = similar(gxy)
    byy = similar(gyy)

    margin = steps + 4
    max_cut = 0.0
    max_far = 0.0
    n_cut = 0
    n_far = 0
    min_eig = Inf
    for _ in 1:steps
        stream_2d!(bxx, gxx, p.Nx, p.Ny; sync=true)
        stream_2d!(bxy, gxy, p.Nx, p.Ny; sync=true)
        stream_2d!(byy, gyy, p.Nx, p.Ny; sync=true)
        apply_polymer_wall_bc!(bxx, gxx, p.is_solid, p.q_wall, Cxx, p.ux, p.uy, bc)
        apply_polymer_wall_bc!(bxy, gxy, p.is_solid, p.q_wall, Cxy, p.ux, p.uy, bc)
        apply_polymer_wall_bc!(byy, gyy, p.is_solid, p.q_wall, Cyy, p.ux, p.uy, bc)
        gxx, bxx = bxx, gxx
        gxy, bxy = bxy, gxy
        gyy, byy = byy, gyy

        if reset_cut_equilibrium
            _reset_cut_cells_to_equilibrium!(gxx, Cxx, p.ux, p.uy, p.is_solid, p.q_wall)
            _reset_cut_cells_to_equilibrium!(gxy, Cxy, p.ux, p.uy, p.is_solid, p.q_wall)
            _reset_cut_cells_to_equilibrium!(gyy, Cyy, p.ux, p.uy, p.is_solid, p.q_wall)
        end

        compute_conformation_macro_2d!(Cxx, gxx)
        compute_conformation_macro_2d!(Cxy, gxy)
        compute_conformation_macro_2d!(Cyy, gyy)
        _collide_direct_cde_state!(gxx, gxy, gyy, Cxx, Cxy, Cyy,
                                   p.ux, p.uy, p.is_solid, p.λ)
        if reset_cut_after_collision
            reset_cutlink_conformation_equilibrium_2d!(gxx, Cxx, p.ux, p.uy,
                                                       p.is_solid, p.q_wall)
            reset_cutlink_conformation_equilibrium_2d!(gxy, Cxy, p.ux, p.uy,
                                                       p.is_solid, p.q_wall)
            reset_cutlink_conformation_equilibrium_2d!(gyy, Cyy, p.ux, p.uy,
                                                       p.is_solid, p.q_wall)
            compute_conformation_macro_2d!(Cxx, gxx)
            compute_conformation_macro_2d!(Cxy, gxy)
            compute_conformation_macro_2d!(Cyy, gyy)
        end

        for j in margin:p.Ny-margin+1, i in margin:p.Nx-margin+1
            p.is_solid[i, j] && continue
            err = max(abs(Cxx[i, j] - p.cxx), abs(Cxy[i, j] - p.cxy),
                      abs(Cyy[i, j] - p.cyy))
            min_eig = min(min_eig, _min_eig_spd_2x2(Cxx[i, j], Cxy[i, j], Cyy[i, j]))
            if _is_cut_cell(p.q_wall, i, j)
                n_cut += 1
                max_cut = max(max_cut, err)
            else
                n_far += 1
                max_far = max(max_far, err)
            end
        end
    end
    return (; max_cut, max_far, min_eig, n_cut, n_far)
end

@testset "P18g inclined straight Couette q-variable canary before curved geometry" begin
    broken_errors = Float64[]
    for case in _wall_bc_cases(), normal in ((3.0, 4.0), (4.0, 3.0))
        result = _inclined_straight_couette_repeated(case.bc; normal, steps=4)
        @test result.n_cut > 0
        @test result.n_far > 0
        @test result.min_eig > 0.9
        if case.name === :cnebb_field
            @test result.max_cut < P0_ATOL
            @test result.max_far > 1e-4

            oracle = _inclined_straight_couette_repeated(
                case.bc; normal, steps=4, reset_cut_equilibrium=true,
            )
            @test oracle.max_cut < P0_ATOL
            @test oracle.max_far < 5e-5
            @test result.max_far > 50 * oracle.max_far
        elseif case.name === :cnebb_field_equilibrium
            @test result.max_cut < P0_ATOL
            @test result.max_far < 5e-5
        else
            push!(broken_errors, result.max_cut)
            @test result.max_cut > 1e-6
        end
    end
    @test maximum(broken_errors) > 1e-6
end

function _inclined_straight_source_residual(; normal=(3.0, 4.0), γ=0.01)
    p = _inclined_straight_couette_patch(; normal, γ)
    max_cut = 0.0
    max_far = 0.0
    for j in 3:p.Ny-2, i in 3:p.Nx-2
        p.is_solid[i, j] && continue
        source = _direct_source_tuple(
            p.cxx, p.cxy, p.cyy,
            Kraken._wall_aware_dx_2d(p.ux, p.is_solid, i, j, p.Nx, Float64),
            Kraken._wall_aware_dy_2d(p.ux, p.is_solid, i, j, p.Ny, Float64),
            Kraken._wall_aware_dx_2d(p.uy, p.is_solid, i, j, p.Nx, Float64),
            Kraken._wall_aware_dy_2d(p.uy, p.is_solid, i, j, p.Ny, Float64),
            p.λ,
        )
        residual = maximum(abs, source)
        if _is_cut_cell(p.q_wall, i, j)
            max_cut = max(max_cut, residual)
        else
            max_far = max(max_far, residual)
        end
    end
    return (; max_cut, max_far)
end

function _inclined_straight_transport_only(bc; normal=(3.0, 4.0), steps=4,
                                           constant_velocity=false, γ=0.01)
    p = _inclined_straight_couette_patch(; normal, γ)
    C_ref = 1.234
    C = fill(C_ref, p.Nx, p.Ny)
    ux = constant_velocity ? fill(0.02, p.Nx, p.Ny) : p.ux
    uy = constant_velocity ? fill(-0.01, p.Nx, p.Ny) : p.uy
    g = zeros(Float64, p.Nx, p.Ny, 9)
    buf = similar(g)
    init_conformation_field_2d!(g, C, ux, uy)

    margin = steps + 4
    max_cut = 0.0
    max_far = 0.0
    n_cut = 0
    n_far = 0
    for _ in 1:steps
        stream_2d!(buf, g, p.Nx, p.Ny; sync=true)
        apply_polymer_wall_bc!(buf, g, p.is_solid, p.q_wall, C, ux, uy, bc)
        g, buf = buf, g
        compute_conformation_macro_2d!(C, g)

        for j in margin:p.Ny-margin+1, i in margin:p.Nx-margin+1
            p.is_solid[i, j] && continue
            err = abs(C[i, j] - C_ref)
            if _is_cut_cell(p.q_wall, i, j)
                n_cut += 1
                max_cut = max(max_cut, err)
            else
                n_far += 1
                max_far = max(max_far, err)
            end
        end
    end
    return (; max_cut, max_far, n_cut, n_far)
end

@testset "P18h inclined straight Couette residual is not source-gradient error" begin
    for normal in ((3.0, 4.0), (4.0, 3.0))
        source = _inclined_straight_source_residual(; normal)
        pre_only = _inclined_straight_couette_repeated(
            CNEBBFieldEquilibrium(); normal, steps=4,
        )
        post_reset = _inclined_straight_couette_repeated(
            CNEBBFieldEquilibrium(); normal, steps=4,
            reset_cut_after_collision=true,
        )
        @test source.max_cut < P0_ATOL
        @test source.max_far < P0_ATOL
        @test pre_only.max_cut < P0_ATOL
        @test post_reset.max_cut < P0_ATOL
        @test pre_only.max_far > 1e-5
        @test post_reset.max_far < 5e-5
        @test pre_only.max_far < 5e-5
    end
end

@testset "P18i inclined straight pure transport isolates variable-velocity residual" begin
    for normal in ((3.0, 4.0), (4.0, 3.0))
        cnebb_variable = nothing
        field_variable = nothing
        field_equilibrium_variable = nothing
        for case in _wall_bc_cases()
            constant = _inclined_straight_transport_only(
                case.bc; normal, steps=4, constant_velocity=true,
            )
            variable = _inclined_straight_transport_only(
                case.bc; normal, steps=4, constant_velocity=false,
            )
            @test constant.n_cut > 0
            @test constant.n_far > 0
            @test variable.n_cut == constant.n_cut
            @test variable.n_far == constant.n_far
            @test isfinite(constant.max_cut)
            @test isfinite(constant.max_far)
            @test isfinite(variable.max_cut)
            @test isfinite(variable.max_far)

            if case.name in (:cnebb_field, :cnebb_field_equilibrium,
                             :cnebb_eq_gradient)
                @test constant.max_cut < P0_ATOL
                @test constant.max_far < P0_ATOL
            end
            if case.name === :cnebb_field
                field_variable = variable
                @test variable.max_cut < P0_ATOL
                @test 1e-3 < variable.max_far < 2e-3
            elseif case.name === :cnebb_field_equilibrium
                field_equilibrium_variable = variable
                @test variable.max_cut < P0_ATOL
                @test 1e-3 < variable.max_far < 2e-3
            elseif case.name === :cnebb
                cnebb_variable = variable
                @test variable.max_cut > 1e-3
            end
        end

        @test cnebb_variable.max_cut > 3 * field_equilibrium_variable.max_cut + 1e-3
        @test field_variable.max_far ≈ field_equilibrium_variable.max_far rtol=5e-3
    end
end

@testset "P18j inclined straight residual scales with wall-coupled velocity transport" begin
    for normal in ((3.0, 4.0), (4.0, 3.0))
        zero_transport = _inclined_straight_transport_only(
            CNEBBFieldEquilibrium(); normal, γ=0.0, steps=4,
        )
        zero_cde = _inclined_straight_couette_repeated(
            CNEBBFieldEquilibrium(); normal, γ=0.0, steps=4,
        )
        source = _inclined_straight_source_residual(; normal, γ=0.01)
        transport_low = _inclined_straight_transport_only(
            CNEBBFieldEquilibrium(); normal, γ=0.005, steps=4,
        )
        transport_high = _inclined_straight_transport_only(
            CNEBBFieldEquilibrium(); normal, γ=0.01, steps=4,
        )
        cde_low = _inclined_straight_couette_repeated(
            CNEBBFieldEquilibrium(); normal, γ=0.005, steps=4,
        )
        cde_high = _inclined_straight_couette_repeated(
            CNEBBFieldEquilibrium(); normal, γ=0.01, steps=4,
        )

        @test zero_transport.max_cut < 2e-15
        @test zero_transport.max_far < P0_ATOL
        @test zero_cde.max_cut < P0_ATOL
        @test zero_cde.max_far < P0_ATOL
        @test source.max_cut < P0_ATOL
        @test source.max_far < P0_ATOL

        @test 1.8 < transport_high.max_far / transport_low.max_far < 2.2
        @test 3.5 < cde_high.max_far / cde_low.max_far < 4.5
        @test cde_high.max_far < 0.02 * transport_high.max_far
    end
end

# ---------------------------------------------------------------------
# Population-level wall residual on the inclined straight wall.
#
# After ONE stream + apply_polymer_wall_bc!, classify each population
# g_post[i, j, q] and compare it to the closed-form expectation:
#
#   * cut cell, q is BC-filled (source is solid)         → local equilibrium
#                                                          feq(C_ref, ux[i,j], uy[i,j], q)
#   * cut cell, q == 1 (rest, BC rebalances)             → same local equilibrium
#   * cut cell, q has fluid source                       → pure-stream prediction
#   * non-cut cell, q == 1                               → local equilibrium
#   * non-cut cell, fluid source not on a cut cell       → pure-stream prediction
#   * non-cut cell, fluid source ON a cut cell           → pure-stream prediction
#
# The canary isolates which population class carries the residual that
# the macro-level P18i/P18j tests already report. The findings frozen
# below are surprising and must not regress:
#
#   1. CNEBB:  inclined-wall :pre_opp φ-recovery has an O(|u| · C_ref)
#      asymmetry residual *even at constant velocity*. The src_solid set
#      {2,3,6} (or any set without its opposite) makes the recovery sum
#      asymmetric, producing δφ ≈ -(5/6)·(u_x+u_y)·C_ref. This is the
#      irreducible inclined-wall defect of the strict CNEBB recovery and
#      the wall-coupled velocity transport residual seen in P18i/P18j.
#
#   2. CNEBBField + q_aware: with q_wall provided, the q_w ≤ 0.5 branch
#      simultaneously zeroes both non-equilibrium terms, giving
#      err_filled = 0 even with variable velocity. The BC's wall-fill
#      is bit-exact at one step.
#
#   3. CNEBBFieldEquilibrium: reset_cutlink_conformation_equilibrium_2d!
#      overwrites the pure-stream populations at cut cells with the
#      local equilibrium, breaking the bulk-affine moment cancellation
#      and creating an O(γ) population residual on src_fluid populations
#      at cut cells (err_cut_outgoing > 1e-4). The reset is therefore a
#      new pollution channel, not a fix.
# ---------------------------------------------------------------------

function _inclined_straight_population_residual(bc; normal=(3.0, 4.0), γ=0.01,
                                                constant_velocity=false)
    p = _inclined_straight_couette_patch(; normal, γ)
    C_ref = 1.234
    C = fill(C_ref, p.Nx, p.Ny)
    ux = constant_velocity ? fill(0.02, p.Nx, p.Ny) : copy(p.ux)
    uy = constant_velocity ? fill(-0.01, p.Nx, p.Ny) : copy(p.uy)

    g_pre = zeros(Float64, p.Nx, p.Ny, 9)
    init_conformation_field_2d!(g_pre, C, ux, uy)
    g_post = similar(g_pre)
    stream_2d!(g_post, g_pre, p.Nx, p.Ny; sync=true)
    apply_polymer_wall_bc!(g_post, g_pre, p.is_solid, p.q_wall, C, ux, uy, bc)

    margin = 3
    err_filled = 0.0; n_filled = 0
    err_rest = 0.0; n_rest = 0
    err_cut_outgoing = 0.0; n_cut_outgoing = 0
    err_far_pure = 0.0; n_far_pure = 0
    err_far_from_cut = 0.0; n_far_from_cut = 0
    domain_boundary_touched = false

    for j in margin:p.Ny-margin+1, i in margin:p.Nx-margin+1
        p.is_solid[i, j] && continue
        i_cut = _is_cut_cell(p.q_wall, i, j)
        for q in 1:9
            cx = Int(D2Q9_CX[q]); cy = Int(D2Q9_CY[q])
            si = i - cx; sj = j - cy
            in_dom = 1 <= si <= p.Nx && 1 <= sj <= p.Ny
            src_solid = !in_dom || p.is_solid[si, sj]
            local_eq = equilibrium(D2Q9(), C_ref, ux[i, j], uy[i, j], q)
            if i_cut && q == 1
                err_rest = max(err_rest, abs(g_post[i, j, 1] - local_eq))
                n_rest += 1
            elseif i_cut && src_solid
                err_filled = max(err_filled, abs(g_post[i, j, q] - local_eq))
                n_filled += 1
            elseif i_cut
                pure = equilibrium(D2Q9(), C_ref, ux[si, sj], uy[si, sj], q)
                err_cut_outgoing = max(err_cut_outgoing,
                                       abs(g_post[i, j, q] - pure))
                n_cut_outgoing += 1
            else
                if q == 1
                    err_far_pure = max(err_far_pure,
                                       abs(g_post[i, j, 1] - local_eq))
                    n_far_pure += 1
                elseif src_solid
                    domain_boundary_touched = true
                else
                    pure = equilibrium(D2Q9(), C_ref, ux[si, sj], uy[si, sj], q)
                    err = abs(g_post[i, j, q] - pure)
                    if _is_cut_cell(p.q_wall, si, sj)
                        err_far_from_cut = max(err_far_from_cut, err)
                        n_far_from_cut += 1
                    else
                        err_far_pure = max(err_far_pure, err)
                        n_far_pure += 1
                    end
                end
            end
        end
    end
    return (; err_filled, n_filled, err_rest, n_rest,
              err_cut_outgoing, n_cut_outgoing,
              err_far_pure, n_far_pure,
              err_far_from_cut, n_far_from_cut,
              domain_boundary_touched)
end

@testset "P18k inclined straight population residual localizes wall-fill" begin
    for normal in ((3.0, 4.0), (4.0, 3.0))
        # Common census: every BC must populate every class.
        for case in (CNEBB(), CNEBBField(), CNEBBFieldEquilibrium())
            const_pop = _inclined_straight_population_residual(
                case; normal, γ=0.01, constant_velocity=true,
            )
            @test !const_pop.domain_boundary_touched
            @test const_pop.n_filled > 0
            @test const_pop.n_rest > 0
            @test const_pop.n_cut_outgoing > 0
            @test const_pop.n_far_pure > 0
            @test const_pop.n_far_from_cut > 0
            # Far cells (no wall contact) are exact for every BC.
            @test const_pop.err_far_pure < P0_ATOL
            @test const_pop.err_far_from_cut < P0_ATOL
            # Outgoing fluid-source populations at cut cells are not
            # touched by CNEBB / CNEBBField (the BC only writes
            # src_solid populations + rest). CNEBBFieldEquilibrium
            # however resets all 9, so it is the only BC that
            # disturbs cut_outgoing at constant velocity — and even
            # the reset preserves them when velocity is uniform.
            @test const_pop.err_cut_outgoing < P0_ATOL
        end

        const_cnebb = _inclined_straight_population_residual(
            CNEBB(); normal, γ=0.01, constant_velocity=true,
        )
        const_field = _inclined_straight_population_residual(
            CNEBBField(); normal, γ=0.01, constant_velocity=true,
        )
        const_equil = _inclined_straight_population_residual(
            CNEBBFieldEquilibrium(); normal, γ=0.01, constant_velocity=true,
        )

        # CNEBB :pre_opp recovery on an inclined wall: asymmetry of the
        # src_solid set yields δφ ≈ -(5/6)(u_x+u_y)·C_ref ≈ 1e-2
        # *at constant velocity*. The defect rides on |u| × C_ref, not
        # on velocity gradients, and is the irreducible signature of the
        # strict CNEBB recovery on non-axis-aligned walls.
        @test const_cnebb.err_rest > 5e-3
        @test const_cnebb.err_filled > 5e-5

        # CNEBBField + q_aware pins φ to C_ref → constant velocity is
        # exact at eps everywhere (filled, rest, outgoing).
        @test const_field.err_filled < P0_ATOL
        @test const_field.err_rest < P0_ATOL

        # CNEBBFieldEquilibrium = CNEBBField + reset to local equilibrium.
        # At constant velocity, the reset writes the same value the BC
        # would have written, so all populations remain exact.
        @test const_equil.err_filled < P0_ATOL
        @test const_equil.err_rest < P0_ATOL

        var_cnebb = _inclined_straight_population_residual(
            CNEBB(); normal, γ=0.01, constant_velocity=false,
        )
        var_field = _inclined_straight_population_residual(
            CNEBBField(); normal, γ=0.01, constant_velocity=false,
        )
        var_equil = _inclined_straight_population_residual(
            CNEBBFieldEquilibrium(); normal, γ=0.01, constant_velocity=false,
        )

        @test !var_cnebb.domain_boundary_touched
        @test !var_field.domain_boundary_touched
        @test !var_equil.domain_boundary_touched

        # No BC may pollute cells away from the wall in a single step.
        @test var_cnebb.err_far_pure < P0_ATOL
        @test var_cnebb.err_far_from_cut < P0_ATOL
        @test var_field.err_far_pure < P0_ATOL
        @test var_field.err_far_from_cut < P0_ATOL
        @test var_equil.err_far_pure < P0_ATOL
        @test var_equil.err_far_from_cut < P0_ATOL

        # CNEBB / CNEBBField do not touch fluid-source populations at
        # cut cells — they retain the pure-stream prediction.
        @test var_cnebb.err_cut_outgoing < P0_ATOL
        @test var_field.err_cut_outgoing < P0_ATOL

        # CNEBBFieldEquilibrium's reset overwrites those pure-stream
        # populations with the cut-cell local equilibrium, breaking
        # bulk-affine moment cancellation. This is the new pollution
        # channel that the reset introduces.
        @test var_equil.err_cut_outgoing > 1e-4
        @test var_equil.err_filled < P0_ATOL
        @test var_equil.err_rest < P0_ATOL

        # Strict CNEBB still carries its inclined-wall asymmetry.
        @test var_cnebb.err_filled > 1e-5
        @test var_cnebb.err_rest > 1e-3

        # CNEBBField + q_aware fills cut-link populations exactly to
        # local equilibrium even with variable velocity (the q_w ≤ 0.5
        # branch zeroes both non-equilibria simultaneously). The rest
        # rebalance still carries an O(γ) residual because src_fluid
        # populations differ from local equilibrium by O(γ).
        @test var_field.err_filled < P0_ATOL
        @test var_field.err_rest > 1e-5
    end
end

# ---------------------------------------------------------------------
# Prototype wall BC: linear extrapolation of (C, u) to the virtual
# upstream position x_v = x_b - c_q for each src_solid q at a cut
# cell, then feq(C_v, u_v, q). No rest rebalance — the rest pop is
# left at its post-stream value local_eq_init[1]. Conservation is
# automatic via the bulk-affine canary (sum of pure-stream feq over
# all q evaluates to C(x_b) for affine fields).
#
# This is a TEST-ONLY helper. It exists in the test file so we can
# canary the scheme on patches before promoting any of it to src/.
# ---------------------------------------------------------------------

function _extrap_eq_wall_bc!(g_post, is_solid, q_wall, C, ux, uy)
    Nx, Ny = size(C)
    for j in 1:Ny, i in 1:Nx
        is_solid[i, j] && continue
        _is_cut_cell(q_wall, i, j) || continue
        dCdx = Kraken._wall_aware_dx_2d(C, is_solid, i, j, Nx, Float64)
        dCdy = Kraken._wall_aware_dy_2d(C, is_solid, i, j, Ny, Float64)
        dudx = Kraken._wall_aware_dx_2d(ux, is_solid, i, j, Nx, Float64)
        dudy = Kraken._wall_aware_dy_2d(ux, is_solid, i, j, Ny, Float64)
        dvdx = Kraken._wall_aware_dx_2d(uy, is_solid, i, j, Nx, Float64)
        dvdy = Kraken._wall_aware_dy_2d(uy, is_solid, i, j, Ny, Float64)
        for q in 2:9
            cx = Int(D2Q9_CX[q]); cy = Int(D2Q9_CY[q])
            si = i - cx; sj = j - cy
            in_dom = 1 <= si <= Nx && 1 <= sj <= Ny
            src_solid = !in_dom || is_solid[si, sj]
            src_solid || continue
            offx = -cx; offy = -cy
            C_v = C[i, j] + offx * dCdx + offy * dCdy
            u_v = ux[i, j] + offx * dudx + offy * dudy
            v_v = uy[i, j] + offx * dvdx + offy * dvdy
            g_post[i, j, q] = equilibrium(D2Q9(), C_v, u_v, v_v, q)
        end
    end
    return nothing
end

@inline function _wall_aware_d2x_2d(a, is_solid, i, j, Nx, ::Type{T}) where {T}
    plus_ok = i < Nx && !is_solid[i + 1, j]
    minus_ok = i > 1 && !is_solid[i - 1, j]
    plus2_ok = i < Nx - 1 && plus_ok && !is_solid[i + 2, j]
    minus2_ok = i > 2 && minus_ok && !is_solid[i - 2, j]
    if plus_ok && minus_ok
        return a[i + 1, j] - T(2) * a[i, j] + a[i - 1, j]
    elseif plus2_ok
        return a[i, j] - T(2) * a[i + 1, j] + a[i + 2, j]
    elseif minus2_ok
        return a[i, j] - T(2) * a[i - 1, j] + a[i - 2, j]
    else
        return zero(T)
    end
end

@inline function _wall_aware_d2y_2d(a, is_solid, i, j, Ny, ::Type{T}) where {T}
    plus_ok = j < Ny && !is_solid[i, j + 1]
    minus_ok = j > 1 && !is_solid[i, j - 1]
    plus2_ok = j < Ny - 1 && plus_ok && !is_solid[i, j + 2]
    minus2_ok = j > 2 && minus_ok && !is_solid[i, j - 2]
    if plus_ok && minus_ok
        return a[i, j + 1] - T(2) * a[i, j] + a[i, j - 1]
    elseif plus2_ok
        return a[i, j] - T(2) * a[i, j + 1] + a[i, j + 2]
    elseif minus2_ok
        return a[i, j] - T(2) * a[i, j - 1] + a[i, j - 2]
    else
        return zero(T)
    end
end

@inline function _wall_aware_dxdy_2d(a, is_solid, i, j, Nx, Ny,
                                    ::Type{T}) where {T}
    plus_ok = i < Nx && !is_solid[i + 1, j]
    minus_ok = i > 1 && !is_solid[i - 1, j]
    plus2_ok = i < Nx - 1 && plus_ok && !is_solid[i + 2, j]
    minus2_ok = i > 2 && minus_ok && !is_solid[i - 2, j]

    dy0 = Kraken._wall_aware_dy_2d(a, is_solid, i, j, Ny, T)
    if plus_ok && minus_ok
        dyp = Kraken._wall_aware_dy_2d(a, is_solid, i + 1, j, Ny, T)
        dym = Kraken._wall_aware_dy_2d(a, is_solid, i - 1, j, Ny, T)
        return (dyp - dym) / T(2)
    elseif plus2_ok
        dy1 = Kraken._wall_aware_dy_2d(a, is_solid, i + 1, j, Ny, T)
        dy2 = Kraken._wall_aware_dy_2d(a, is_solid, i + 2, j, Ny, T)
        return (-T(3) * dy0 + T(4) * dy1 - dy2) / T(2)
    elseif minus2_ok
        dy1 = Kraken._wall_aware_dy_2d(a, is_solid, i - 1, j, Ny, T)
        dy2 = Kraken._wall_aware_dy_2d(a, is_solid, i - 2, j, Ny, T)
        return (T(3) * dy0 - T(4) * dy1 + dy2) / T(2)
    elseif plus_ok
        dy1 = Kraken._wall_aware_dy_2d(a, is_solid, i + 1, j, Ny, T)
        return dy1 - dy0
    elseif minus_ok
        dy1 = Kraken._wall_aware_dy_2d(a, is_solid, i - 1, j, Ny, T)
        return dy0 - dy1
    else
        return zero(T)
    end
end

function _extrap_eq_wall_bc_quadratic!(g_post, is_solid, q_wall, C, ux, uy)
    Nx, Ny = size(C)
    for j in 1:Ny, i in 1:Nx
        is_solid[i, j] && continue
        _is_cut_cell(q_wall, i, j) || continue
        dCdx = Kraken._wall_aware_dx_2d(C, is_solid, i, j, Nx, Float64)
        dCdy = Kraken._wall_aware_dy_2d(C, is_solid, i, j, Ny, Float64)
        d2Cdx = _wall_aware_d2x_2d(C, is_solid, i, j, Nx, Float64)
        d2Cdy = _wall_aware_d2y_2d(C, is_solid, i, j, Ny, Float64)
        d2Cdxdy = _wall_aware_dxdy_2d(C, is_solid, i, j, Nx, Ny, Float64)

        dudx = Kraken._wall_aware_dx_2d(ux, is_solid, i, j, Nx, Float64)
        dudy = Kraken._wall_aware_dy_2d(ux, is_solid, i, j, Ny, Float64)
        d2udx = _wall_aware_d2x_2d(ux, is_solid, i, j, Nx, Float64)
        d2udy = _wall_aware_d2y_2d(ux, is_solid, i, j, Ny, Float64)
        d2udxdy = _wall_aware_dxdy_2d(ux, is_solid, i, j, Nx, Ny, Float64)

        dvdx = Kraken._wall_aware_dx_2d(uy, is_solid, i, j, Nx, Float64)
        dvdy = Kraken._wall_aware_dy_2d(uy, is_solid, i, j, Ny, Float64)
        d2vdx = _wall_aware_d2x_2d(uy, is_solid, i, j, Nx, Float64)
        d2vdy = _wall_aware_d2y_2d(uy, is_solid, i, j, Ny, Float64)
        d2vdxdy = _wall_aware_dxdy_2d(uy, is_solid, i, j, Nx, Ny, Float64)

        for q in 2:9
            cx = Int(D2Q9_CX[q]); cy = Int(D2Q9_CY[q])
            si = i - cx; sj = j - cy
            in_dom = 1 <= si <= Nx && 1 <= sj <= Ny
            src_solid = !in_dom || is_solid[si, sj]
            src_solid || continue
            offx = -cx; offy = -cy
            quad_C = offx^2 * d2Cdx + 2.0 * offx * offy * d2Cdxdy +
                     offy^2 * d2Cdy
            quad_u = offx^2 * d2udx + 2.0 * offx * offy * d2udxdy +
                     offy^2 * d2udy
            quad_v = offx^2 * d2vdx + 2.0 * offx * offy * d2vdxdy +
                     offy^2 * d2vdy
            C_v = C[i, j] + offx * dCdx + offy * dCdy + 0.5 * quad_C
            u_v = ux[i, j] + offx * dudx + offy * dudy + 0.5 * quad_u
            v_v = uy[i, j] + offx * dvdx + offy * dvdy + 0.5 * quad_v
            g_post[i, j, q] = equilibrium(D2Q9(), C_v, u_v, v_v, q)
        end
    end
    return nothing
end

function _rebalance_cut_cell_rest_to_field!(g_post, is_solid, q_wall, C)
    Nx, Ny = size(C)
    for j in 1:Ny, i in 1:Nx
        is_solid[i, j] && continue
        _is_cut_cell(q_wall, i, j) || continue
        nonrest = 0.0
        for q in 2:9
            nonrest += g_post[i, j, q]
        end
        g_post[i, j, 1] = C[i, j] - nonrest
    end
    return nothing
end

function _extrap_eq_wall_bc_rebalanced!(g_post, is_solid, q_wall, C, ux, uy)
    _extrap_eq_wall_bc!(g_post, is_solid, q_wall, C, ux, uy)
    _rebalance_cut_cell_rest_to_field!(g_post, is_solid, q_wall, C)
    return nothing
end

function _extrap_eq_wall_bc_quadratic_rebalanced!(g_post, is_solid, q_wall,
                                                  C, ux, uy)
    _extrap_eq_wall_bc_quadratic!(g_post, is_solid, q_wall, C, ux, uy)
    _rebalance_cut_cell_rest_to_field!(g_post, is_solid, q_wall, C)
    return nothing
end

@testset "P18l prototype extrap-eq BC: M0 quadratic Hessian stencils" begin
    Nx, Ny = 9, 10
    c0, ax, ay = 1.4, 0.11, -0.07
    bxx, bxy, byy = -0.017, 0.023, 0.021
    field = [
        c0 + ax * (i - 1) + ay * (j - 1) +
        0.5 * bxx * (i - 1)^2 + bxy * (i - 1) * (j - 1) +
        0.5 * byy * (j - 1)^2
        for i in 1:Nx, j in 1:Ny
    ]
    solid = falses(Nx, Ny)

    for (si, sj) in ((3, 5), (5, 5), (4, 3), (4, 5))
        solid .= false
        solid[si, sj] = true
        @test _wall_aware_d2x_2d(field, solid, 4, 4, Nx, Float64) ≈ bxx atol=P0_ATOL
        @test _wall_aware_d2y_2d(field, solid, 4, 4, Ny, Float64) ≈ byy atol=P0_ATOL
        @test _wall_aware_dxdy_2d(field, solid, 4, 4, Nx, Ny, Float64) ≈ bxy atol=P0_ATOL
    end
end

# Per-population residual against the n-step pure-transport
# prediction feq(field(x − n·c_q), q) (q≥2) and feq(field(x), 1) for
# the rest. For affine fields and the extrapolation BC, this must be
# machine zero everywhere — even after multiple steps — provided the
# inspection margin clears domain-edge halfway-BB pollution.
function _extrap_population_residual(; normal=(3.0, 4.0), γ=0.01,
                                     constant_velocity=false,
                                     steps::Int=1,
                                     wall_bc=_extrap_eq_wall_bc!)
    p = _inclined_straight_couette_patch(; normal, γ)
    C_const = 1.234
    norm_n = hypot(normal[1], normal[2])
    nx_an = normal[1] / norm_n; ny_an = normal[2] / norm_n
    tx_an = -ny_an; ty_an = nx_an
    x0_an = 22.37; y0_an = 19.81
    if constant_velocity
        u0x, u0y = 0.02, -0.01
        analytical_u = (i, j) -> u0x
        analytical_v = (i, j) -> u0y
    else
        analytical_u = (i, j) -> γ * (nx_an * ((i - 1.0) - x0_an) +
                                      ny_an * ((j - 1.0) - y0_an)) * tx_an
        analytical_v = (i, j) -> γ * (nx_an * ((i - 1.0) - x0_an) +
                                      ny_an * ((j - 1.0) - y0_an)) * ty_an
    end
    analytical_C = (i, j) -> C_const

    C = [analytical_C(i, j) for i in 1:p.Nx, j in 1:p.Ny]
    ux = [analytical_u(i, j) for i in 1:p.Nx, j in 1:p.Ny]
    uy = [analytical_v(i, j) for i in 1:p.Nx, j in 1:p.Ny]
    # Solid-cell velocity: leave at analytical extension (init writes
    # the populations there too, but the BC is what matters at the
    # fluid side).

    g = zeros(Float64, p.Nx, p.Ny, 9)
    init_conformation_field_2d!(g, C, ux, uy)
    buf = similar(g)
    for _ in 1:steps
        stream_2d!(buf, g, p.Nx, p.Ny; sync=true)
        wall_bc(buf, p.is_solid, p.q_wall, C, ux, uy)
        g, buf = buf, g
        compute_conformation_macro_2d!(C, g)
    end

    # Domain-edge halfway-BB in stream_2d! corrupts cells within
    # `n` rings of the boundary after `n` steps; keep the inspection
    # region clear of that pollution by enlarging the margin with the
    # step count (mirrors _bulk_affine_transport_error).
    margin = steps + 4
    err_filled = 0.0; n_filled = 0
    err_rest = 0.0; n_rest = 0
    err_cut_outgoing = 0.0; n_cut_outgoing = 0
    err_far_pure = 0.0; n_far_pure = 0
    err_far_from_cut = 0.0; n_far_from_cut = 0
    err_macro = 0.0; n_macro = 0
    domain_boundary_touched = false

    for j in margin:p.Ny-margin+1, i in margin:p.Nx-margin+1
        p.is_solid[i, j] && continue
        i_cut = _is_cut_cell(p.q_wall, i, j)
        err_macro = max(err_macro, abs(C[i, j] - analytical_C(i, j)))
        n_macro += 1
        for q in 1:9
            cx = Int(D2Q9_CX[q]); cy = Int(D2Q9_CY[q])
            si = i - steps * cx
            sj = j - steps * cy
            if q == 1
                si = i; sj = j
            end
            CC = analytical_C(si, sj)
            uxx = analytical_u(si, sj)
            uyy = analytical_v(si, sj)
            target = equilibrium(D2Q9(), CC, uxx, uyy, q)
            err = abs(g[i, j, q] - target)
            # Source-solid classification uses the 1-step neighbor
            # (the link the BC actually fills), independent of step
            # count for the n-step transport target.
            si1 = i - cx; sj1 = j - cy
            in_dom1 = 1 <= si1 <= p.Nx && 1 <= sj1 <= p.Ny
            src_solid_link = !in_dom1 || p.is_solid[si1, sj1]
            if i_cut && q == 1
                err_rest = max(err_rest, err); n_rest += 1
            elseif i_cut && src_solid_link
                err_filled = max(err_filled, err); n_filled += 1
            elseif i_cut
                err_cut_outgoing = max(err_cut_outgoing, err); n_cut_outgoing += 1
            else
                if q == 1
                    err_far_pure = max(err_far_pure, err); n_far_pure += 1
                elseif src_solid_link
                    domain_boundary_touched = true
                else
                    if _is_cut_cell(p.q_wall, si1, sj1)
                        err_far_from_cut = max(err_far_from_cut, err)
                        n_far_from_cut += 1
                    else
                        err_far_pure = max(err_far_pure, err)
                        n_far_pure += 1
                    end
                end
            end
        end
    end
    return (; err_filled, n_filled, err_rest, n_rest,
              err_cut_outgoing, n_cut_outgoing,
              err_far_pure, n_far_pure,
              err_far_from_cut, n_far_from_cut,
              err_macro, n_macro,
              domain_boundary_touched)
end

@testset "P18l prototype extrap-eq BC: M1 const velocity inclined wall" begin
    # Constant velocity, constant C, inclined straight wall.
    # Linear extrapolation of u (which is constant) is exact, so every
    # filled population must match the pure-stream feq at eps.
    for normal in ((3.0, 4.0), (4.0, 3.0))
        r = _extrap_population_residual(; normal, γ=0.01,
                                        constant_velocity=true, steps=1)
        @test !r.domain_boundary_touched
        @test r.n_filled > 0 && r.n_rest > 0 && r.n_cut_outgoing > 0
        @test r.n_far_pure > 0 && r.n_far_from_cut > 0
        @test r.err_filled < P0_ATOL
        @test r.err_rest < P0_ATOL
        @test r.err_cut_outgoing < P0_ATOL
        @test r.err_far_pure < P0_ATOL
        @test r.err_far_from_cut < P0_ATOL
        @test r.err_macro < P0_ATOL
    end
end

@testset "P18l prototype extrap-eq BC: M2 affine velocity inclined wall" begin
    # Variable velocity (γ ≠ 0), constant C, inclined straight wall.
    # Velocity is affine in space (γ × distance × tangent), so linear
    # extrapolation through the wall is exact. Every population must
    # match the pure-stream feq at eps.
    for normal in ((3.0, 4.0), (4.0, 3.0))
        r = _extrap_population_residual(; normal, γ=0.01,
                                        constant_velocity=false, steps=1)
        @test !r.domain_boundary_touched
        @test r.err_filled < P0_ATOL
        @test r.err_rest < P0_ATOL
        @test r.err_cut_outgoing < P0_ATOL
        @test r.err_far_pure < P0_ATOL
        @test r.err_far_from_cut < P0_ATOL
        @test r.err_macro < P0_ATOL
    end
end

@testset "P18l prototype extrap-eq BC: M3 transport-only repeated steps" begin
    # Pure transport (no collision) is a stress test for any local BC:
    # the cut cell's filled populations carry "1-step backward"
    # extrapolation while the rest of the lattice has advected by n
    # steps, so the two go out of phase. We measure the magnitude
    # rather than asserting eps — production runs use collision,
    # which resynchronises populations to local equilibrium each
    # step (M3b). The expectation here is that drift stays in the
    # 1e-3 band typical of CNEBBField on the same patch (P18i).
    for normal in ((3.0, 4.0), (4.0, 3.0))
        r = _extrap_population_residual(; normal, γ=0.01,
                                        constant_velocity=false, steps=4)
        @test !r.domain_boundary_touched
        @test r.err_macro < 5e-3
        @test r.err_far_pure < 5e-3
        @test r.err_far_from_cut < 5e-3
        @test r.err_rest < P0_ATOL
    end
end

# Full CDE pipeline test: stream → extrap-eq BC → macro recompute →
# TRT collide (τ_plus = 1) → next step. Mirrors
# _inclined_straight_couette_repeated. With collision active, every
# cell relaxes back to local equilibrium each step, so the BC's
# "1-step extrapolation" assumption is self-consistent at every step
# and the analytical stationary C field must be preserved at eps.
function _extrap_repeated_cde(; normal=(3.0, 4.0), γ=0.01, steps::Int=4,
                              wall_bc=_extrap_eq_wall_bc!)
    p = _inclined_straight_couette_patch(; normal, γ)
    Cxx = fill(p.cxx, p.Nx, p.Ny)
    Cxy = fill(p.cxy, p.Nx, p.Ny)
    Cyy = fill(p.cyy, p.Nx, p.Ny)
    gxx = zeros(Float64, p.Nx, p.Ny, 9)
    gxy = similar(gxx)
    gyy = similar(gxx)
    init_conformation_field_2d!(gxx, Cxx, p.ux, p.uy)
    init_conformation_field_2d!(gxy, Cxy, p.ux, p.uy)
    init_conformation_field_2d!(gyy, Cyy, p.ux, p.uy)
    bxx = similar(gxx); bxy = similar(gxy); byy = similar(gyy)

    margin = steps + 4
    max_cut = 0.0; max_far = 0.0
    n_cut = 0; n_far = 0
    min_eig = Inf

    for _ in 1:steps
        stream_2d!(bxx, gxx, p.Nx, p.Ny; sync=true)
        stream_2d!(bxy, gxy, p.Nx, p.Ny; sync=true)
        stream_2d!(byy, gyy, p.Nx, p.Ny; sync=true)
        wall_bc(bxx, p.is_solid, p.q_wall, Cxx, p.ux, p.uy)
        wall_bc(bxy, p.is_solid, p.q_wall, Cxy, p.ux, p.uy)
        wall_bc(byy, p.is_solid, p.q_wall, Cyy, p.ux, p.uy)
        gxx, bxx = bxx, gxx
        gxy, bxy = bxy, gxy
        gyy, byy = byy, gyy
        compute_conformation_macro_2d!(Cxx, gxx)
        compute_conformation_macro_2d!(Cxy, gxy)
        compute_conformation_macro_2d!(Cyy, gyy)
        _collide_direct_cde_state!(gxx, gxy, gyy, Cxx, Cxy, Cyy,
                                   p.ux, p.uy, p.is_solid, p.λ)

        for j in margin:p.Ny-margin+1, i in margin:p.Nx-margin+1
            p.is_solid[i, j] && continue
            err = max(abs(Cxx[i, j] - p.cxx),
                      abs(Cxy[i, j] - p.cxy),
                      abs(Cyy[i, j] - p.cyy))
            min_eig = min(min_eig,
                          _min_eig_spd_2x2(Cxx[i, j], Cxy[i, j], Cyy[i, j]))
            if _is_cut_cell(p.q_wall, i, j)
                n_cut += 1
                max_cut = max(max_cut, err)
            else
                n_far += 1
                max_far = max(max_far, err)
            end
        end
    end
    return (; max_cut, max_far, min_eig, n_cut, n_far)
end

@testset "P18l prototype extrap-eq BC: M3b repeated CDE inclined wall" begin
    # Stationary affine CDE on inclined wall: with TRT collision
    # (τ_plus = 1) the new BC stabilises at ~1e-5 residual through
    # step 16 — well below the 1e-3 ceiling of CNEBB / CNEBBField on
    # the same patch (compare in P18m). Stronger expectations would
    # need a physically motivated bound; for now we lock in the
    # observed magnitude as a non-regression guard.
    for normal in ((3.0, 4.0), (4.0, 3.0))
        r4 = _extrap_repeated_cde(; normal, γ=0.01, steps=4)
        @test r4.n_cut > 0 && r4.n_far > 0
        @test r4.min_eig > 0.9
        @test r4.max_cut < 2e-5
        @test r4.max_far < 1e-5

        r16 = _extrap_repeated_cde(; normal, γ=0.01, steps=16)
        @test r16.max_cut < 2e-5
        @test r16.max_far < 1e-5
    end
end

@testset "P18l prototype extrap-eq BC: M3c beats CNEBB on inclined wall" begin
    # Side-by-side comparison — frozen as a non-regression: the new
    # extrap-eq BC must dominate the strict CNEBB recovery by at
    # least two orders of magnitude on cut-cell error.
    for normal in ((3.0, 4.0), (4.0, 3.0))
        r_new = _extrap_repeated_cde(; normal, γ=0.01, steps=4)
        r_cnebb = _inclined_straight_couette_repeated(
            CNEBB(); normal, γ=0.01, steps=4,
        )
        @test r_cnebb.max_cut > 100 * r_new.max_cut
        @test r_cnebb.max_far > 10 * r_new.max_far
    end
end

function _curved_affine_oldroydb_patch()
    Nx, Ny = 56, 48
    cx, cy, R = 27.31, 23.67, 9.0
    q_wall, is_solid = precompute_q_wall_cylinder(Nx, Ny, cx, cy, R)
    dudx, dudy, dvdx, dvdy = 0.012, 0.035, -0.018, -0.012
    λ = 3.0
    G = 0.01025
    cxx, cxy, cyy = _stationary_direct_conformation_incompressible(
        dudx, dudy, dvdx, dvdy, λ,
    )
    ux = [
        0.02 + dudx * ((i - 1) - cx) + dudy * ((j - 1) - cy)
        for i in 1:Nx, j in 1:Ny
    ]
    uy = [
        -0.01 + dvdx * ((i - 1) - cx) + dvdy * ((j - 1) - cy)
        for i in 1:Nx, j in 1:Ny
    ]
    return (; Nx, Ny, cx, cy, R, q_wall, is_solid,
            dudx, dudy, dvdx, dvdy, λ, G, cxx, cxy, cyy, ux, uy)
end

function _max_curved_cut_error(q_wall, is_solid, Cxx, Cxy, Cyy, cxx, cxy, cyy)
    Nx, Ny = size(Cxx)
    max_cut = 0.0
    max_far = 0.0
    for j in 3:Ny-2, i in 3:Nx-2
        is_solid[i, j] && continue
        err = max(abs(Cxx[i, j] - cxx), abs(Cxy[i, j] - cxy),
                  abs(Cyy[i, j] - cyy))
        if _is_cut_cell(q_wall, i, j)
            max_cut = max(max_cut, err)
        else
            max_far = max(max_far, err)
        end
    end
    return (; max_cut, max_far)
end

function _curved_affine_bc_once_error(bc)
    p = _curved_affine_oldroydb_patch()
    Cxx = fill(p.cxx, p.Nx, p.Ny)
    Cxy = fill(p.cxy, p.Nx, p.Ny)
    Cyy = fill(p.cyy, p.Nx, p.Ny)
    gxx = zeros(Float64, p.Nx, p.Ny, 9)
    gxy = zeros(Float64, p.Nx, p.Ny, 9)
    gyy = zeros(Float64, p.Nx, p.Ny, 9)
    init_conformation_field_2d!(gxx, Cxx, p.ux, p.uy)
    init_conformation_field_2d!(gxy, Cxy, p.ux, p.uy)
    init_conformation_field_2d!(gyy, Cyy, p.ux, p.uy)
    bxx = similar(gxx)
    bxy = similar(gxy)
    byy = similar(gyy)

    stream_2d!(bxx, gxx, p.Nx, p.Ny; sync=true)
    stream_2d!(bxy, gxy, p.Nx, p.Ny; sync=true)
    stream_2d!(byy, gyy, p.Nx, p.Ny; sync=true)
    apply_polymer_wall_bc!(bxx, gxx, p.is_solid, p.q_wall, Cxx, p.ux, p.uy, bc)
    apply_polymer_wall_bc!(bxy, gxy, p.is_solid, p.q_wall, Cxy, p.ux, p.uy, bc)
    apply_polymer_wall_bc!(byy, gyy, p.is_solid, p.q_wall, Cyy, p.ux, p.uy, bc)
    return _max_curved_cut_error(
        p.q_wall, p.is_solid, Cxx, Cxy, Cyy, p.cxx, p.cxy, p.cyy,
    )
end

function _curved_affine_cde_once(bc)
    p = _curved_affine_oldroydb_patch()
    Cxx = fill(p.cxx, p.Nx, p.Ny)
    Cxy = fill(p.cxy, p.Nx, p.Ny)
    Cyy = fill(p.cyy, p.Nx, p.Ny)
    gxx = zeros(Float64, p.Nx, p.Ny, 9)
    gxy = zeros(Float64, p.Nx, p.Ny, 9)
    gyy = zeros(Float64, p.Nx, p.Ny, 9)
    init_conformation_field_2d!(gxx, Cxx, p.ux, p.uy)
    init_conformation_field_2d!(gxy, Cxy, p.ux, p.uy)
    init_conformation_field_2d!(gyy, Cyy, p.ux, p.uy)
    bxx = similar(gxx)
    bxy = similar(gxy)
    byy = similar(gyy)

    stream_2d!(bxx, gxx, p.Nx, p.Ny; sync=true)
    stream_2d!(bxy, gxy, p.Nx, p.Ny; sync=true)
    stream_2d!(byy, gyy, p.Nx, p.Ny; sync=true)
    apply_polymer_wall_bc!(bxx, gxx, p.is_solid, p.q_wall, Cxx, p.ux, p.uy, bc)
    apply_polymer_wall_bc!(bxy, gxy, p.is_solid, p.q_wall, Cxy, p.ux, p.uy, bc)
    apply_polymer_wall_bc!(byy, gyy, p.is_solid, p.q_wall, Cyy, p.ux, p.uy, bc)

    gxx, bxx = bxx, gxx
    gxy, bxy = bxy, gxy
    gyy, byy = byy, gyy
    compute_conformation_macro_2d!(Cxx, gxx)
    compute_conformation_macro_2d!(Cxy, gxy)
    compute_conformation_macro_2d!(Cyy, gyy)
    collide_conformation_2d!(
        gxx, Cxx, p.ux, p.uy, Cxx, Cxy, Cyy, p.is_solid, 1.0, p.λ;
        component=1,
    )
    collide_conformation_2d!(
        gxy, Cxy, p.ux, p.uy, Cxx, Cxy, Cyy, p.is_solid, 1.0, p.λ;
        component=2,
    )
    collide_conformation_2d!(
        gyy, Cyy, p.ux, p.uy, Cxx, Cxy, Cyy, p.is_solid, 1.0, p.λ;
        component=3,
    )
    return p, Cxx, Cxy, Cyy
end

function _hermite_source_delta_f(q, s_plus, txx, txy, tyy; ce_correction=false)
    cs2 = 1.0 / 3.0
    prefactor = -s_plus * 9.0 / 2.0
    ce_correction && (prefactor /= 1.0 - s_plus / 2.0)
    return prefactor * D2Q9_W[q] *
           ((D2Q9_CX[q]^2 - cs2) * txx +
            (D2Q9_CY[q]^2 - cs2) * tyy +
            2.0 * D2Q9_CX[q] * D2Q9_CY[q] * txy)
end

function _curved_expected_mea_increment_from_source(
        q_wall, tau_xx, tau_xy, tau_yy, s_plus; ce_correction=false)
    Nx, Ny = size(tau_xx)
    Fx = 0.0
    Fy = 0.0
    for j in 1:Ny, i in 1:Nx, q in 2:9
        qw = q_wall[i, j, q]
        qw > 0.0 || continue
        qbar = _opp_local(q)
        im = i - Int(D2Q9_CX[q])
        jm = j - Int(D2Q9_CY[q])
        δ_here = _hermite_source_delta_f(
            q, s_plus, tau_xx[i, j], tau_xy[i, j], tau_yy[i, j];
            ce_correction,
        )
        δ_qbar = _hermite_source_delta_f(
            qbar, s_plus, tau_xx[i, j], tau_xy[i, j], tau_yy[i, j];
            ce_correction,
        )
        δ_back = if 1 <= im <= Nx && 1 <= jm <= Ny
            _hermite_source_delta_f(
                q, s_plus, tau_xx[im, jm], tau_xy[im, jm], tau_yy[im, jm];
                ce_correction,
            )
        else
            δ_qbar
        end
        δ_arriving = if qw <= 0.5
            2.0 * qw * δ_here + (1.0 - 2.0 * qw) * δ_back
        else
            inv2q = 1.0 / (2.0 * qw)
            inv2q * δ_here + (1.0 - inv2q) * δ_qbar
        end
        δ_link = δ_here + δ_arriving
        Fx += D2Q9_CX[q] * δ_link
        Fy += D2Q9_CY[q] * δ_link
    end
    return (; Fx, Fy)
end

@testset "P19 curved cut-link affine velocity gradients are exact" begin
    p = _curved_affine_oldroydb_patch()
    n_cut = 0
    max_err = 0.0
    for j in 1:p.Ny, i in 1:p.Nx
        _is_cut_cell(p.q_wall, i, j) || continue
        n_cut += 1
        gradients = (
            Kraken._wall_aware_dx_2d(p.ux, p.is_solid, i, j, p.Nx, Float64),
            Kraken._wall_aware_dy_2d(p.ux, p.is_solid, i, j, p.Ny, Float64),
            Kraken._wall_aware_dx_2d(p.uy, p.is_solid, i, j, p.Nx, Float64),
            Kraken._wall_aware_dy_2d(p.uy, p.is_solid, i, j, p.Ny, Float64),
        )
        exact = (p.dudx, p.dudy, p.dvdx, p.dvdy)
        max_err = max(max_err, maximum(abs(gradients[k] - exact[k]) for k in 1:4))
    end
    @test n_cut > 0
    @test max_err < P0_ATOL
end

@testset "P20 curved cut-link Oldroyd-B local closure is exact" begin
    p = _curved_affine_oldroydb_patch()
    max_source = 0.0
    max_stress = 0.0
    τxx = zeros(Float64, p.Nx, p.Ny)
    τxy = zeros(Float64, p.Nx, p.Ny)
    τyy = zeros(Float64, p.Nx, p.Ny)
    update_polymer_stress!(
        τxx, τxy, τyy,
        fill(p.cxx, p.Nx, p.Ny), fill(p.cxy, p.Nx, p.Ny), fill(p.cyy, p.Nx, p.Ny),
        OldroydB(G=p.G, λ=p.λ),
    )
    for j in 1:p.Ny, i in 1:p.Nx
        _is_cut_cell(p.q_wall, i, j) || continue
        source = _direct_source_tuple(
            p.cxx, p.cxy, p.cyy,
            Kraken._wall_aware_dx_2d(p.ux, p.is_solid, i, j, p.Nx, Float64),
            Kraken._wall_aware_dy_2d(p.ux, p.is_solid, i, j, p.Ny, Float64),
            Kraken._wall_aware_dx_2d(p.uy, p.is_solid, i, j, p.Nx, Float64),
            Kraken._wall_aware_dy_2d(p.uy, p.is_solid, i, j, p.Ny, Float64),
            p.λ,
        )
        max_source = max(max_source, maximum(abs, source))
        max_stress = max(
            max_stress,
            abs(τxx[i, j] - p.G * (p.cxx - 1.0)),
            abs(τxy[i, j] - p.G * p.cxy),
            abs(τyy[i, j] - p.G * (p.cyy - 1.0)),
        )
    end
    @test max_source < P0_ATOL
    @test max_stress < P0_ATOL
end

@testset "P21 curved affine stream+BC localizes active-wall defect" begin
    for bc in (CNEBBField(), CNEBBFieldEquilibrium(), CNEBBEqGradient())
        exact = _curved_affine_bc_once_error(bc)
        @test exact.max_cut < P0_ATOL
    end

    for bc in (CNEBB(), CNEBBQAware(), YLWBalanceOnly())
        broken = _curved_affine_bc_once_error(bc)
        @test broken.max_cut > 1e-3
    end
end

@testset "P22 curved frozen affine CDE one-step localizes CNEBB pollution" begin
    for bc in (CNEBBField(), CNEBBFieldEquilibrium(), CNEBBEqGradient())
        p, Cxx, Cxy, Cyy = _curved_affine_cde_once(bc)
        exact = _max_curved_cut_error(p.q_wall, p.is_solid, Cxx, Cxy, Cyy,
                                      p.cxx, p.cxy, p.cyy)
        @test exact.max_cut < P0_ATOL
    end

    p, Cxx, Cxy, Cyy = _curved_affine_cde_once(CNEBB())
    broken = _max_curved_cut_error(p.q_wall, p.is_solid, Cxx, Cxy, Cyy,
                                   p.cxx, p.cxy, p.cyy)
    @test broken.max_cut > 1e-2
end

@testset "P23 curved Hermite source plus MEA matches link oracle" begin
    p = _curved_affine_oldroydb_patch()
    s_plus = 1.25
    tau_xx = [1e-4 * (2i - j) for i in 1:p.Nx, j in 1:p.Ny]
    tau_xy = [-2e-4 * (i + j) for i in 1:p.Nx, j in 1:p.Ny]
    tau_yy = [3e-4 * (-i + 2j) for i in 1:p.Nx, j in 1:p.Ny]
    f = zeros(Float64, p.Nx, p.Ny, 9)
    for j in 1:p.Ny, i in 1:p.Nx, q in 1:9
        f[i, j, q] = equilibrium(D2Q9(), 1.0, 0.0, 0.0, q)
    end
    uwx = zeros(Float64, p.Nx, p.Ny, 9)
    uwy = zeros(Float64, p.Nx, p.Ny, 9)
    before = compute_drag_libb_mei_2d(f, p.q_wall, uwx, uwy, p.Nx, p.Ny)
    apply_hermite_source_2d!(f, p.is_solid, s_plus, tau_xx, tau_xy, tau_yy;
                             ce_correction=false)
    after = compute_drag_libb_mei_2d(f, p.q_wall, uwx, uwy, p.Nx, p.Ny)
    expected = _curved_expected_mea_increment_from_source(
        p.q_wall, tau_xx, tau_xy, tau_yy, s_plus; ce_correction=false,
    )
    @test isapprox(after.Fx - before.Fx, expected.Fx; rtol=1e-12, atol=1e-14)
    @test isapprox(after.Fy - before.Fy, expected.Fy; rtol=1e-12, atol=1e-14)
end

@testset "P24 curved CDE-generated tau creates spurious source force unless BC is fixed" begin
    s_plus = 1.25
    results = Dict{Symbol,Any}()
    for (name, bc) in ((:cnebb, CNEBB()),
                       (:field, CNEBBField()),
                       (:field_equilibrium, CNEBBFieldEquilibrium()),
                       (:eq_gradient, CNEBBEqGradient()))
        p, Cxx, Cxy, Cyy = _curved_affine_cde_once(bc)
        tau_xx = p.G .* (Cxx .- 1.0)
        tau_xy = p.G .* Cxy
        tau_yy = p.G .* (Cyy .- 1.0)
        f = zeros(Float64, p.Nx, p.Ny, 9)
        for j in 1:p.Ny, i in 1:p.Nx, q in 1:9
            f[i, j, q] = equilibrium(D2Q9(), 1.0, 0.0, 0.0, q)
        end
        uwx = zeros(Float64, p.Nx, p.Ny, 9)
        uwy = zeros(Float64, p.Nx, p.Ny, 9)
        before = compute_drag_libb_mei_2d(f, p.q_wall, uwx, uwy, p.Nx, p.Ny)
        apply_hermite_source_2d!(f, p.is_solid, s_plus, tau_xx, tau_xy, tau_yy;
                                 ce_correction=false)
        after = compute_drag_libb_mei_2d(f, p.q_wall, uwx, uwy, p.Nx, p.Ny)
        results[name] = (Fx=after.Fx - before.Fx, Fy=after.Fy - before.Fy)
    end
    @test hypot(results[:field].Fx, results[:field].Fy) < 1e-5
    @test hypot(results[:field_equilibrium].Fx, results[:field_equilibrium].Fy) < 1e-5
    @test hypot(results[:eq_gradient].Fx, results[:eq_gradient].Fy) < 1e-5
    @test hypot(results[:cnebb].Fx, results[:cnebb].Fy) > 1e-3
end

@testset "P25 active curved-wall defect is present for every cut-link orientation" begin
    for q_out in 2:9, qw in (0.1, 0.3, 0.5, 0.7, 0.9)
        for bc in (CNEBBField(), CNEBBFieldEquilibrium(), CNEBBEqGradient())
            fixed_macro, fixed_sum = _single_cut_link_macro_error_bc(
                q_out, bc; q_wall_value=qw,
                orientation=:uniform, velocity=(0.03, -0.02),
            )
            @test abs(fixed_sum) < P0_ATOL
            @test abs(fixed_macro) < P0_ATOL
        end

        for bc in (CNEBB(), CNEBBQAware(), YLWBalanceOnly())
            broken_macro, broken_sum = _single_cut_link_macro_error_bc(
                q_out, bc; q_wall_value=qw,
                orientation=:uniform, velocity=(0.03, -0.02),
            )
            @test abs(broken_sum) < P0_ATOL
            @test abs(broken_macro) > 1e-6
        end
    end
end

@testset "P26 EqGradient is an equilibrium canary, not a stable wall BC" begin
    first_step = _poiseuille_cde_patch_error(
        collision=:trt, tau_plus=1.0, bc=CNEBBEqGradient();
        orientation=:horizontal, steps=1,
    )
    stable = _poiseuille_cde_patch_error(
        collision=:trt, tau_plus=1.0, bc=CNEBB();
        orientation=:horizontal, steps=20,
    )
    unstable = _poiseuille_cde_patch_error(
        collision=:trt, tau_plus=1.0, bc=CNEBBEqGradient();
        orientation=:horizontal, steps=20,
    )

    @test first_step.Cxy_l2 < P0_ATOL
    @test first_step.N1_l2 < 1e-2
    @test unstable.Cxy_l2 > 10 * stable.Cxy_l2
    @test unstable.N1_l2 > 10 * stable.N1_l2
end

@testset "P27 cut-link-only EqGradient is geometry-dependent and rejected" begin
    max_exhaustive = 0.0
    for q_out in 2:9, qw in (0.1, 0.3, 0.5, 0.7, 0.9),
        orientation in (:uniform, :tangent, :normal),
        velocity in ((0.0, 0.0), (0.03, -0.02))

        macro_error, sum_error = _single_cut_link_macro_error_bc(
            q_out, CNEBBCutLinkEqGradient(); q_wall_value=qw,
            orientation, velocity,
        )
        @test abs(sum_error) < P0_ATOL
        max_exhaustive = max(max_exhaustive, abs(macro_error))
    end

    actual = _actual_cylinder_cutlink_macro_errors(
        CNEBBCutLinkEqGradient(); gradient=(0.03, -0.02),
        velocity=(0.03, -0.02),
    )
    @test max_exhaustive > 1e-4
    @test actual.max_macro_error < P0_ATOL
end

@inline function _couette_circular_constants(Ri, Ro, Ω)
    A = -Ω * Ri^2 / (Ro^2 - Ri^2)
    B = Ω * Ri^2 * Ro^2 / (Ro^2 - Ri^2)
    return A, B
end

@inline function _couette_circular_velocity_gradient(x, y; cx, cy, Ri, Ro, Ω)
    xr = x - cx
    yr = y - cy
    r2 = xr^2 + yr^2
    A, B = _couette_circular_constants(Ri, Ro, Ω)
    F = A + B / r2
    ux = -F * yr
    uy = F * xr
    dudx = 2B * xr * yr / (r2^2)
    dudy = -F + 2B * yr^2 / (r2^2)
    dvdx = F - 2B * xr^2 / (r2^2)
    dvdy = -2B * xr * yr / (r2^2)
    return (; ux, uy, dudx, dudy, dvdx, dvdy)
end

@inline function _couette_circular_velocity_hessian(x, y; cx, cy, Ri, Ro, Ω)
    xr = x - cx
    yr = y - cy
    r2 = xr^2 + yr^2
    r6 = r2^3
    _, B = _couette_circular_constants(Ri, Ro, Ω)
    sx_xx = 2B * xr * (xr^2 - 3yr^2) / r6
    sx_xy = 2B * yr * (3xr^2 - yr^2) / r6
    sx_yy = -sx_xx
    sy_xx = sx_xy
    sy_xy = -sx_xx
    sy_yy = -sx_xy
    return (;
        ux_xx = -sy_xx,
        ux_xy = -sy_xy,
        ux_yy = -sy_yy,
        uy_xx = sx_xx,
        uy_xy = sx_xy,
        uy_yy = sx_yy,
    )
end

@inline function _couette_circular_conformation_tuple(x, y, p)
    vg = _couette_circular_velocity_gradient(x, y; cx=p.cx, cy=p.cy,
                                             Ri=p.Ri, Ro=p.Ro, Ω=p.Ω)
    return _stationary_direct_conformation_incompressible(
        vg.dudx, vg.dudy, vg.dvdx, vg.dvdy, p.λ,
    )
end

function _continuous_hessian_2d(f, x, y; h=1e-3)
    f00 = f(x, y)
    fpx = f(x + h, y)
    fmx = f(x - h, y)
    fpy = f(x, y + h)
    fmy = f(x, y - h)
    fpp = f(x + h, y + h)
    fpm = f(x + h, y - h)
    fmp = f(x - h, y + h)
    fmm = f(x - h, y - h)
    return (;
        xx = (fpx - 2 * f00 + fmx) / h^2,
        xy = (fpp - fpm - fmp + fmm) / (4h^2),
        yy = (fpy - 2 * f00 + fmy) / h^2,
    )
end

function _polyfit_features_2d(dx, dy, degree::Int)
    if degree == 3
        return [
            dx, dy,
            0.5 * dx^2, dx * dy, 0.5 * dy^2,
            dx^3 / 6, 0.5 * dx^2 * dy, 0.5 * dx * dy^2, dy^3 / 6,
        ]
    elseif degree == 4
        return [
            dx, dy,
            0.5 * dx^2, dx * dy, 0.5 * dy^2,
            dx^3 / 6, 0.5 * dx^2 * dy, 0.5 * dx * dy^2, dy^3 / 6,
            dx^4 / 24, dx^3 * dy / 6, 0.25 * dx^2 * dy^2,
            dx * dy^3 / 6, dy^4 / 24,
        ]
    else
        error("unsupported local polynomial degree $(degree)")
    end
end

function _fluid_polyfit_gradient_2d(a, is_solid, i, j, Nx, Ny;
                                    degree::Int=4, radius::Int=4)
    ncoef = length(_polyfit_features_2d(1.0, 0.0, degree))
    rows = Float64[]
    rhs = Float64[]
    a0 = a[i, j]
    for dj in -radius:radius, di in -radius:radius
        (di == 0 && dj == 0) && continue
        di^2 + dj^2 <= radius^2 || continue
        ii = i + di
        jj = j + dj
        1 <= ii <= Nx && 1 <= jj <= Ny || continue
        is_solid[ii, jj] && continue
        w = 1.0 / max(1.0, di^2 + dj^2)
        append!(rows, sqrt(w) .* _polyfit_features_2d(di, dj, degree))
        push!(rhs, sqrt(w) * (a[ii, jj] - a0))
    end
    nrows = length(rhs)
    if nrows < ncoef
        return (;
            dx = Kraken._wall_aware_dx_2d(a, is_solid, i, j, Nx, Float64),
            dy = Kraken._wall_aware_dy_2d(a, is_solid, i, j, Ny, Float64),
            points = nrows,
            rank = 0,
            fallback = true,
        )
    end
    A = transpose(reshape(rows, ncoef, nrows))
    Arank = rank(A)
    if Arank < ncoef
        return (;
            dx = Kraken._wall_aware_dx_2d(a, is_solid, i, j, Nx, Float64),
            dy = Kraken._wall_aware_dy_2d(a, is_solid, i, j, Ny, Float64),
            points = nrows,
            rank = Arank,
            fallback = true,
        )
    end
    coeffs = A \ rhs
    return (; dx = coeffs[1], dy = coeffs[2], points = nrows,
              rank = Arank, fallback = false)
end

@inline function _couette_wall_velocity(x, y, p)
    xr = x - p.cx
    yr = y - p.cy
    return (; ux = -p.Ω * yr, uy = p.Ω * xr)
end

@inline function _couette_wall_component(x, y, p, component::Symbol)
    uw = _couette_wall_velocity(x, y, p)
    return component === :ux ? uw.ux : uw.uy
end

@inline function _quadratic_derivative_at_zero(samples, a0)
    if length(samples) >= 2
        rows = Float64[]
        rhs = Float64[]
        for (s, val) in samples
            append!(rows, (s, s^2))
            push!(rhs, val - a0)
        end
        A = transpose(reshape(rows, 2, length(samples)))
        coeffs = A \ rhs
        return coeffs[1]
    elseif length(samples) == 1
        s, val = samples[1]
        return (val - a0) / s
    else
        return 0.0
    end
end

function _axis_ghost_value(a, component::Symbol, is_solid, q_wall,
                           i, j, q, Nx, Ny, p)
    cx = Int(D2Q9_CX[q])
    cy = Int(D2Q9_CY[q])
    ii = i + cx
    jj = j + cy
    if 1 <= ii <= Nx && 1 <= jj <= Ny && !is_solid[ii, jj]
        return a[ii, jj]
    end
    qw = q_wall[i, j, q]
    if qw > 0.0
        wx = (i - 1.0) + qw * cx
        wy = (j - 1.0) + qw * cy
        wall_value = _couette_wall_component(wx, wy, p, component)
        return (wall_value - (1.0 - qw) * a[i, j]) / qw
    end
    return a[i, j]
end

function _ghost_axis_gradient_2d(a, component::Symbol, is_solid, q_wall,
                                 i, j, Nx, Ny, p)
    xp = _axis_ghost_value(a, component, is_solid, q_wall, i, j, 2, Nx, Ny, p)
    xm = _axis_ghost_value(a, component, is_solid, q_wall, i, j, 4, Nx, Ny, p)
    yp = _axis_ghost_value(a, component, is_solid, q_wall, i, j, 3, Nx, Ny, p)
    ym = _axis_ghost_value(a, component, is_solid, q_wall, i, j, 5, Nx, Ny, p)
    return (; dx = 0.5 * (xp - xm), dy = 0.5 * (yp - ym), fallback = false)
end

function _embedded_axis_derivative_2d(a, component::Symbol, is_solid, q_wall,
                                      i, j, Nx, Ny, p, axis::Symbol)
    a0 = a[i, j]
    samples = Tuple{Float64,Float64}[]
    dirs = axis === :x ? ((2, 1.0), (4, -1.0)) : ((3, 1.0), (5, -1.0))
    for (q, sfluid) in dirs
        cx = Int(D2Q9_CX[q])
        cy = Int(D2Q9_CY[q])
        ii = i + cx
        jj = j + cy
        if 1 <= ii <= Nx && 1 <= jj <= Ny && !is_solid[ii, jj]
            push!(samples, (sfluid, a[ii, jj]))
        elseif q_wall[i, j, q] > 0.0
            qw = q_wall[i, j, q]
            wx = (i - 1.0) + qw * cx
            wy = (j - 1.0) + qw * cy
            push!(samples, (sfluid * qw,
                            _couette_wall_component(wx, wy, p, component)))
        end
    end
    return _quadratic_derivative_at_zero(samples, a0)
end

function _embedded_axis_gradient_2d(a, component::Symbol, is_solid, q_wall,
                                    i, j, Nx, Ny, p)
    return (;
        dx = _embedded_axis_derivative_2d(a, component, is_solid, q_wall,
                                          i, j, Nx, Ny, p, :x),
        dy = _embedded_axis_derivative_2d(a, component, is_solid, q_wall,
                                          i, j, Nx, Ny, p, :y),
        fallback = false,
    )
end

function _normal_tangent_gradient_2d(a, component::Symbol, is_solid, q_wall,
                                     i, j, Nx, Ny, p)
    if !_is_cut_cell(q_wall, i, j)
        return (;
            dx = Kraken._wall_aware_dx_2d(a, is_solid, i, j, Nx, Float64),
            dy = Kraken._wall_aware_dy_2d(a, is_solid, i, j, Ny, Float64),
            fallback = false,
        )
    end

    x0 = i - 1.0
    y0 = j - 1.0
    best_d = Inf
    best_wx = x0
    best_wy = y0
    for q in 2:9
        qw = q_wall[i, j, q]
        qw > 0.0 || continue
        wx = x0 + qw * D2Q9_CX[q]
        wy = y0 + qw * D2Q9_CY[q]
        d = hypot(x0 - wx, y0 - wy)
        if d < best_d
            best_d = d
            best_wx = wx
            best_wy = wy
        end
    end
    isfinite(best_d) && best_d > 0.0 || return (;
        dx = Kraken._wall_aware_dx_2d(a, is_solid, i, j, Nx, Float64),
        dy = Kraken._wall_aware_dy_2d(a, is_solid, i, j, Ny, Float64),
        fallback = true,
    )

    nx = (best_wx - p.cx) / p.Ri
    ny = (best_wy - p.cy) / p.Ri
    tx = -ny
    ty = nx
    wall_value = _couette_wall_component(best_wx, best_wy, p, component)
    dn = ((a[i, j] - wall_value) / best_d)
    # Rigid rotation wall velocity: grad(ux)=(-Ω y)_grad, grad(uy)=(Ω x)_grad.
    dt = component === :ux ? -p.Ω * ty : p.Ω * tx
    return (; dx = dn * nx + dt * tx, dy = dn * ny + dt * ty,
              fallback = false)
end


function _wall_constrained_polyfit_gradient_2d(a, component::Symbol,
                                               is_solid, q_wall, i, j,
                                               Nx, Ny, p;
                                               degree::Int=4, radius::Int=3,
                                               wall_weight::Float64=16.0)
    ncoef = length(_polyfit_features_2d(1.0, 0.0, degree))
    rows = Float64[]
    rhs = Float64[]
    a0 = a[i, j]
    x0 = i - 1.0
    y0 = j - 1.0
    nwall = 0

    for dj in -radius:radius, di in -radius:radius
        ii = i + di
        jj = j + dj
        1 <= ii <= Nx && 1 <= jj <= Ny || continue
        is_solid[ii, jj] && continue
        di^2 + dj^2 <= radius^2 || continue

        if !(di == 0 && dj == 0)
            dist2 = di^2 + dj^2
            w = 1.0 / max(1.0, dist2)
            append!(rows, sqrt(w) .* _polyfit_features_2d(di, dj, degree))
            push!(rhs, sqrt(w) * (a[ii, jj] - a0))
        end

        for q in 2:9
            qw = q_wall[ii, jj, q]
            qw > 0.0 || continue
            wx = (ii - 1.0) + qw * D2Q9_CX[q]
            wy = (jj - 1.0) + qw * D2Q9_CY[q]
            dx = wx - x0
            dy = wy - y0
            dx^2 + dy^2 <= (radius + 0.5)^2 || continue
            uw = _couette_wall_velocity(wx, wy, p)
            wall_value = component === :ux ? uw.ux : uw.uy
            w = wall_weight / max(0.25, dx^2 + dy^2)
            append!(rows, sqrt(w) .* _polyfit_features_2d(dx, dy, degree))
            push!(rhs, sqrt(w) * (wall_value - a0))
            nwall += 1
        end
    end

    nrows = length(rhs)
    if nrows < ncoef
        return (;
            dx = Kraken._wall_aware_dx_2d(a, is_solid, i, j, Nx, Float64),
            dy = Kraken._wall_aware_dy_2d(a, is_solid, i, j, Ny, Float64),
            points = nrows,
            rank = 0,
            wall_points = nwall,
            fallback = true,
        )
    end
    A = transpose(reshape(rows, ncoef, nrows))
    Arank = rank(A)
    if Arank < ncoef
        return (;
            dx = Kraken._wall_aware_dx_2d(a, is_solid, i, j, Nx, Float64),
            dy = Kraken._wall_aware_dy_2d(a, is_solid, i, j, Ny, Float64),
            points = nrows,
            rank = Arank,
            wall_points = nwall,
            fallback = true,
        )
    end
    coeffs = A \ rhs
    return (; dx = coeffs[1], dy = coeffs[2], points = nrows,
              rank = Arank, wall_points = nwall, fallback = false)
end

@inline function _push_stencil_fluid_term!(terms, coeff, di, dj)
    abs(coeff) > 0.0 && push!(terms, (Float64(coeff), Int(di), Int(dj)))
    return nothing
end

@inline function _push_stencil_wall_term!(terms, coeff, wx, wy)
    abs(coeff) > 0.0 && push!(terms, (Float64(coeff), Float64(wx), Float64(wy)))
    return nothing
end

function _derivative_rhs_weights_at_zero(positions)
    n = length(positions)
    if n >= 2
        rows = Float64[]
        for s in positions
            append!(rows, (s, s^2))
        end
        A = transpose(reshape(rows, 2, n))
        return vec((A \ Matrix{Float64}(I, n, n))[1, :])
    elseif n == 1
        return [1.0 / positions[1]]
    else
        return Float64[]
    end
end

function _eval_scalar_gradient_stencil_2d(a, component::Symbol, stencil,
                                          i, j, p)
    value = 0.0
    for (coeff, di, dj) in stencil.fluid_terms
        value += coeff * a[i + di, j + dj]
    end
    for (coeff, wx, wy) in stencil.wall_terms
        value += coeff * _couette_wall_component(wx, wy, p, component)
    end
    return value
end

function _embedded_axis_derivative_stencil_2d(is_solid, q_wall, i, j,
                                              Nx, Ny, axis::Symbol)
    samples = NamedTuple{(:s, :is_wall, :di, :dj, :wx, :wy),
                         Tuple{Float64,Bool,Int,Int,Float64,Float64}}[]
    dirs = axis === :x ? ((2, 1.0), (4, -1.0)) : ((3, 1.0), (5, -1.0))
    for (q, sfluid) in dirs
        cx = Int(D2Q9_CX[q])
        cy = Int(D2Q9_CY[q])
        ii = i + cx
        jj = j + cy
        if 1 <= ii <= Nx && 1 <= jj <= Ny && !is_solid[ii, jj]
            push!(samples, (s=Float64(sfluid), is_wall=false,
                            di=cx, dj=cy, wx=0.0, wy=0.0))
        elseif q_wall[i, j, q] > 0.0
            qw = q_wall[i, j, q]
            wx = (i - 1.0) + qw * cx
            wy = (j - 1.0) + qw * cy
            push!(samples, (s=Float64(sfluid * qw), is_wall=true,
                            di=0, dj=0, wx=wx, wy=wy))
        end
    end

    weights = _derivative_rhs_weights_at_zero([sample.s for sample in samples])
    fluid_terms = Tuple{Float64,Int,Int}[]
    wall_terms = Tuple{Float64,Float64,Float64}[]
    _push_stencil_fluid_term!(fluid_terms, -sum(weights), 0, 0)
    for (weight, sample) in zip(weights, samples)
        if sample.is_wall
            _push_stencil_wall_term!(wall_terms, weight, sample.wx, sample.wy)
        else
            _push_stencil_fluid_term!(fluid_terms, weight, sample.di, sample.dj)
        end
    end
    return (; fluid_terms, wall_terms, fallback=false, points=length(samples),
              rank=length(samples))
end

function _embedded_axis_coeff_gradient_2d(a, component::Symbol, is_solid,
                                          q_wall, i, j, Nx, Ny, p)
    sx = _embedded_axis_derivative_stencil_2d(
        is_solid, q_wall, i, j, Nx, Ny, :x,
    )
    sy = _embedded_axis_derivative_stencil_2d(
        is_solid, q_wall, i, j, Nx, Ny, :y,
    )
    return (;
        dx = _eval_scalar_gradient_stencil_2d(a, component, sx, i, j, p),
        dy = _eval_scalar_gradient_stencil_2d(a, component, sy, i, j, p),
        fallback = sx.fallback || sy.fallback,
        points = sx.points + sy.points,
        max_terms = max(length(sx.fluid_terms) + length(sx.wall_terms),
                        length(sy.fluid_terms) + length(sy.wall_terms)),
    )
end

function _wallfit_stencil_from_samples(rows, samples, ncoef)
    nrows = length(samples)
    if nrows < ncoef
        return (;
            dx = nothing, dy = nothing, fallback = true,
            points = nrows, rank = 0, wall_points = count(s -> s.is_wall, samples),
        )
    end
    A = transpose(reshape(rows, ncoef, nrows))
    Arank = rank(A)
    if Arank < ncoef
        return (;
            dx = nothing, dy = nothing, fallback = true,
            points = nrows, rank = Arank, wall_points = count(s -> s.is_wall, samples),
        )
    end

    W = A \ Matrix{Float64}(I, nrows, nrows)
    function build_stencil(row)
        fluid_terms = Tuple{Float64,Int,Int}[]
        wall_terms = Tuple{Float64,Float64,Float64}[]
        center_coeff = 0.0
        for (weight, sample) in zip(vec(W[row, :]), samples)
            coeff = weight * sample.rhs_scale
            center_coeff -= coeff
            if sample.is_wall
                _push_stencil_wall_term!(wall_terms, coeff, sample.wx, sample.wy)
            else
                _push_stencil_fluid_term!(fluid_terms, coeff, sample.di, sample.dj)
            end
        end
        _push_stencil_fluid_term!(fluid_terms, center_coeff, 0, 0)
        return (; fluid_terms, wall_terms, fallback=false,
                  points=nrows, rank=Arank,
                  wall_points=count(s -> s.is_wall, samples))
    end

    return (;
        dx = build_stencil(1),
        dy = build_stencil(2),
        fallback = false,
        points = nrows,
        rank = Arank,
        wall_points = count(s -> s.is_wall, samples),
    )
end

function _wall_constrained_polyfit_gradient_stencils_2d(
        is_solid, q_wall, i, j, Nx, Ny, p;
        degree::Int=4, radius::Int=3, wall_weight::Float64=16.0)
    ncoef = length(_polyfit_features_2d(1.0, 0.0, degree))
    rows = Float64[]
    samples = NamedTuple{(:is_wall, :di, :dj, :wx, :wy, :rhs_scale),
                         Tuple{Bool,Int,Int,Float64,Float64,Float64}}[]
    x0 = i - 1.0
    y0 = j - 1.0

    for dj in -radius:radius, di in -radius:radius
        ii = i + di
        jj = j + dj
        1 <= ii <= Nx && 1 <= jj <= Ny || continue
        is_solid[ii, jj] && continue
        di^2 + dj^2 <= radius^2 || continue

        if !(di == 0 && dj == 0)
            dist2 = di^2 + dj^2
            rhs_scale = sqrt(1.0 / max(1.0, dist2))
            append!(rows, rhs_scale .* _polyfit_features_2d(di, dj, degree))
            push!(samples, (is_wall=false, di=di, dj=dj,
                            wx=0.0, wy=0.0, rhs_scale=rhs_scale))
        end

        for q in 2:9
            qw = q_wall[ii, jj, q]
            qw > 0.0 || continue
            wx = (ii - 1.0) + qw * D2Q9_CX[q]
            wy = (jj - 1.0) + qw * D2Q9_CY[q]
            dx = wx - x0
            dy = wy - y0
            dx^2 + dy^2 <= (radius + 0.5)^2 || continue
            rhs_scale = sqrt(wall_weight / max(0.25, dx^2 + dy^2))
            append!(rows, rhs_scale .* _polyfit_features_2d(dx, dy, degree))
            push!(samples, (is_wall=true, di=0, dj=0,
                            wx=wx, wy=wy, rhs_scale=rhs_scale))
        end
    end

    return _wallfit_stencil_from_samples(rows, samples, ncoef)
end

function _wallfit_coeff_gradient_2d(a, component::Symbol, is_solid, q_wall,
                                    i, j, Nx, Ny, p)
    stencils = _wall_constrained_polyfit_gradient_stencils_2d(
        is_solid, q_wall, i, j, Nx, Ny, p;
        degree=4, radius=3, wall_weight=16.0,
    )
    if stencils.fallback
        fallback = _wall_constrained_polyfit_gradient_2d(
            a, component, is_solid, q_wall, i, j, Nx, Ny, p;
            degree=4, radius=3, wall_weight=16.0,
        )
        return (; dx=fallback.dx, dy=fallback.dy, fallback=true,
                  points=stencils.points, max_terms=0)
    end
    return (;
        dx = _eval_scalar_gradient_stencil_2d(a, component, stencils.dx, i, j, p),
        dy = _eval_scalar_gradient_stencil_2d(a, component, stencils.dy, i, j, p),
        fallback = false,
        points = stencils.points,
        max_terms = max(length(stencils.dx.fluid_terms) + length(stencils.dx.wall_terms),
                        length(stencils.dy.fluid_terms) + length(stencils.dy.wall_terms)),
    )
end

function _test_velocity_gradient(ux, uy, is_solid, i, j, Nx, Ny, p,
                                 gradient_mode::Symbol)
    if gradient_mode === :analytic_couette
        vg = _couette_circular_velocity_gradient(i - 1.0, j - 1.0;
                                                 cx=p.cx, cy=p.cy,
                                                 Ri=p.Ri, Ro=p.Ro, Ω=p.Ω)
        return (; dudx=vg.dudx, dudy=vg.dudy, dvdx=vg.dvdx, dvdy=vg.dvdy,
                  fallback=false)
    elseif gradient_mode === :polyfit4
        uxg = _fluid_polyfit_gradient_2d(ux, is_solid, i, j, Nx, Ny;
                                         degree=4, radius=4)
        uyg = _fluid_polyfit_gradient_2d(uy, is_solid, i, j, Nx, Ny;
                                         degree=4, radius=4)
        return (; dudx=uxg.dx, dudy=uxg.dy, dvdx=uyg.dx, dvdy=uyg.dy,
                  fallback=uxg.fallback || uyg.fallback)
    elseif gradient_mode === :ghost_axis
        uxg = _ghost_axis_gradient_2d(
            ux, :ux, is_solid, p.q_wall, i, j, Nx, Ny, p,
        )
        uyg = _ghost_axis_gradient_2d(
            uy, :uy, is_solid, p.q_wall, i, j, Nx, Ny, p,
        )
        return (; dudx=uxg.dx, dudy=uxg.dy, dvdx=uyg.dx, dvdy=uyg.dy,
                  fallback=uxg.fallback || uyg.fallback)
    elseif gradient_mode === :embedded_axis
        uxg = _embedded_axis_gradient_2d(
            ux, :ux, is_solid, p.q_wall, i, j, Nx, Ny, p,
        )
        uyg = _embedded_axis_gradient_2d(
            uy, :uy, is_solid, p.q_wall, i, j, Nx, Ny, p,
        )
        return (; dudx=uxg.dx, dudy=uxg.dy, dvdx=uyg.dx, dvdy=uyg.dy,
                  fallback=uxg.fallback || uyg.fallback)
    elseif gradient_mode === :embedded_axis_coeff
        uxg = _embedded_axis_coeff_gradient_2d(
            ux, :ux, is_solid, p.q_wall, i, j, Nx, Ny, p,
        )
        uyg = _embedded_axis_coeff_gradient_2d(
            uy, :uy, is_solid, p.q_wall, i, j, Nx, Ny, p,
        )
        return (; dudx=uxg.dx, dudy=uxg.dy, dvdx=uyg.dx, dvdy=uyg.dy,
                  fallback=uxg.fallback || uyg.fallback)
    elseif gradient_mode === :normal_tangent
        uxg = _normal_tangent_gradient_2d(
            ux, :ux, is_solid, p.q_wall, i, j, Nx, Ny, p,
        )
        uyg = _normal_tangent_gradient_2d(
            uy, :uy, is_solid, p.q_wall, i, j, Nx, Ny, p,
        )
        return (; dudx=uxg.dx, dudy=uxg.dy, dvdx=uyg.dx, dvdy=uyg.dy,
                  fallback=uxg.fallback || uyg.fallback)
    elseif gradient_mode === :wallfit4
        uxg = _wall_constrained_polyfit_gradient_2d(
            ux, :ux, is_solid, p.q_wall, i, j, Nx, Ny, p;
            degree=4, radius=3, wall_weight=16.0,
        )
        uyg = _wall_constrained_polyfit_gradient_2d(
            uy, :uy, is_solid, p.q_wall, i, j, Nx, Ny, p;
            degree=4, radius=3, wall_weight=16.0,
        )
        return (; dudx=uxg.dx, dudy=uxg.dy, dvdx=uyg.dx, dvdy=uyg.dy,
                  fallback=uxg.fallback || uyg.fallback)
    elseif gradient_mode === :wallfit4_coeff
        uxg = _wallfit_coeff_gradient_2d(
            ux, :ux, is_solid, p.q_wall, i, j, Nx, Ny, p,
        )
        uyg = _wallfit_coeff_gradient_2d(
            uy, :uy, is_solid, p.q_wall, i, j, Nx, Ny, p,
        )
        return (; dudx=uxg.dx, dudy=uxg.dy, dvdx=uyg.dx, dvdy=uyg.dy,
                  fallback=uxg.fallback || uyg.fallback)
    else
        return (;
            dudx = Kraken._wall_aware_dx_2d(ux, is_solid, i, j, Nx, Float64),
            dudy = Kraken._wall_aware_dy_2d(ux, is_solid, i, j, Ny, Float64),
            dvdx = Kraken._wall_aware_dx_2d(uy, is_solid, i, j, Nx, Float64),
            dvdy = Kraken._wall_aware_dy_2d(uy, is_solid, i, j, Ny, Float64),
            fallback = false,
        )
    end
end

function _gradient_coeff_stencil_shape_stats(mode::Symbol)
    p = _curved_couette_oldroydb_patch()
    max_terms = 0
    max_wall_terms = 0
    max_points = 0
    max_rank = 0
    fallback_count = 0
    fluid_cells = 0

    for j in 1:p.Ny, i in 1:p.Nx
        p.is_solid[i, j] && continue
        hypot((i - 1.0) - p.cx, (j - 1.0) - p.cy) < p.Ro || continue
        fluid_cells += 1
        if mode === :embedded_axis_coeff
            stencils = (
                _embedded_axis_derivative_stencil_2d(
                    p.is_solid, p.q_wall, i, j, p.Nx, p.Ny, :x,
                ),
                _embedded_axis_derivative_stencil_2d(
                    p.is_solid, p.q_wall, i, j, p.Nx, p.Ny, :y,
                ),
            )
        elseif mode === :wallfit4_coeff
            wallfit = _wall_constrained_polyfit_gradient_stencils_2d(
                p.is_solid, p.q_wall, i, j, p.Nx, p.Ny, p;
                degree=4, radius=3, wall_weight=16.0,
            )
            fallback_count += wallfit.fallback ? 1 : 0
            wallfit.fallback && continue
            stencils = (wallfit.dx, wallfit.dy)
        else
            error("unsupported coefficient stencil mode $(mode)")
        end

        for stencil in stencils
            total_terms = length(stencil.fluid_terms) + length(stencil.wall_terms)
            max_terms = max(max_terms, total_terms)
            max_wall_terms = max(max_wall_terms, length(stencil.wall_terms))
            max_points = max(max_points, stencil.points)
            max_rank = max(max_rank, stencil.rank)
            fallback_count += stencil.fallback ? 1 : 0
        end
    end

    return (; fluid_cells, max_terms, max_wall_terms, max_points,
              max_rank, fallback_count)
end

function _curved_couette_wall_velocity_arrays(p)
    uwx = zeros(Float64, p.Nx, p.Ny, 9)
    uwy = zeros(Float64, p.Nx, p.Ny, 9)
    for q in 2:9, j in 1:p.Ny, i in 1:p.Nx
        qw = p.q_wall[i, j, q]
        qw > 0.0 || continue
        wx = (i - 1.0) + qw * D2Q9_CX[q]
        wy = (j - 1.0) + qw * D2Q9_CY[q]
        uw = _couette_wall_velocity(wx, wy, p)
        uwx[i, j, q] = uw.ux
        uwy[i, j, q] = uw.uy
    end
    return uwx, uwy
end

function _collide_scalar_cde_state_test_gradient!(
        g, C_field, ux, uy, Cxx, Cxy, Cyy, is_solid, λ, component, p,
        gradient_mode::Symbol)
    Nx, Ny = size(C_field)
    tau_plus = 1.0
    tau_minus = 1e-6 / (tau_plus - 0.5) + 0.5
    ωp = 1.0 / tau_plus
    ωm = 1.0 / tau_minus
    half = 0.5
    source_linear_coeff = (1.0 - 0.5 * ωp) * 3.0
    pairs = ((2, 4), (3, 5), (6, 8), (7, 9))
    for j in 1:Ny, i in 1:Nx
        is_solid[i, j] && continue
        φ = C_field[i, j]
        u = ux[i, j]
        v = uy[i, j]
        grad = _test_velocity_gradient(ux, uy, is_solid, i, j, Nx, Ny, p,
                                       gradient_mode)
        div_half = 0.5 * (grad.dudx + grad.dvdy)
        dudx = grad.dudx - div_half
        dudy = grad.dudy
        dvdx = grad.dvdx
        dvdy = grad.dvdy - div_half
        S = conformation_source_2d(
            Cxx[i, j], Cxy[i, j], Cyy[i, j],
            dudx, dudy, dvdx, dvdy, λ, component,
        )

        old = [g[i, j, q] for q in 1:9]
        ge = [equilibrium(D2Q9(), φ, u, v, q) for q in 1:9]
        F = [
            D2Q9_W[q] * (S + source_linear_coeff * S *
                         (D2Q9_CX[q] * u + D2Q9_CY[q] * v))
            for q in 1:9
        ]
        new = similar(old)
        new[1] = old[1] - ωp * (old[1] - ge[1]) + F[1]
        for (q, oq) in pairs
            gp = (old[q] + old[oq]) * half
            gm = (old[q] - old[oq]) * half
            ep = (ge[q] + ge[oq]) * half
            em = (ge[q] - ge[oq]) * half
            new[q] = old[q] - ωp * (gp - ep) - ωm * (gm - em) + F[q]
            new[oq] = old[oq] - ωp * (gp - ep) + ωm * (gm - em) + F[oq]
        end
        for q in 1:9
            g[i, j, q] = new[q]
        end
    end
    return nothing
end

function _collide_direct_cde_state_test_gradient!(
        gxx, gxy, gyy, Cxx, Cxy, Cyy, ux, uy, is_solid, λ, p;
        gradient_mode::Symbol)
    if gradient_mode in (:embedded_axis_prod, :wallfit4_prod)
        prod_mode = gradient_mode === :embedded_axis_prod ? :embedded_axis : :wallfit4
        max_terms = gradient_mode === :embedded_axis_prod ? 4 : 64
        stencils = Kraken.precompute_conformation_gradient_stencils_2d(
            is_solid, p.q_wall; mode=prod_mode, max_terms, FT=Float64,
        )
        uwx, uwy = _curved_couette_wall_velocity_arrays(p)
        _collide_direct_cde_state_prod_gradient!(
            gxx, gxy, gyy, Cxx, Cxy, Cyy, ux, uy, is_solid, λ,
            stencils, uwx, uwy,
        )
        return nothing
    end
    _collide_scalar_cde_state_test_gradient!(
        gxx, Cxx, ux, uy, Cxx, Cxy, Cyy, is_solid, λ, 1, p, gradient_mode,
    )
    _collide_scalar_cde_state_test_gradient!(
        gxy, Cxy, ux, uy, Cxx, Cxy, Cyy, is_solid, λ, 2, p, gradient_mode,
    )
    _collide_scalar_cde_state_test_gradient!(
        gyy, Cyy, ux, uy, Cxx, Cxy, Cyy, is_solid, λ, 3, p, gradient_mode,
    )
    compute_conformation_macro_2d!(Cxx, gxx)
    compute_conformation_macro_2d!(Cxy, gxy)
    compute_conformation_macro_2d!(Cyy, gyy)
    return nothing
end

function _collide_direct_cde_state_prod_gradient!(
        gxx, gxy, gyy, Cxx, Cxy, Cyy, ux, uy, is_solid, λ,
        stencils, uwx, uwy)
    Kraken.collide_conformation_2d_with_gradient_stencils!(
        gxx, Cxx, ux, uy, Cxx, Cxy, Cyy, is_solid,
        uwx, uwy, stencils, 1.0, λ; magic=1e-6,
        component=1, divergence_mode=:trace_free,
    )
    Kraken.collide_conformation_2d_with_gradient_stencils!(
        gxy, Cxy, ux, uy, Cxx, Cxy, Cyy, is_solid,
        uwx, uwy, stencils, 1.0, λ; magic=1e-6,
        component=2, divergence_mode=:trace_free,
    )
    Kraken.collide_conformation_2d_with_gradient_stencils!(
        gyy, Cyy, ux, uy, Cxx, Cxy, Cyy, is_solid,
        uwx, uwy, stencils, 1.0, λ; magic=1e-6,
        component=3, divergence_mode=:trace_free,
    )
    compute_conformation_macro_2d!(Cxx, gxx)
    compute_conformation_macro_2d!(Cxy, gxy)
    compute_conformation_macro_2d!(Cyy, gyy)
    return nothing
end

function _curved_couette_oldroydb_patch(; Nx=72, Ny=72, Ri=12.0,
                                        Ro=28.0, Ω=0.006, λ=4.0)
    cx = (Nx - 1) / 2
    cy = (Ny - 1) / 2
    q_wall, is_solid = precompute_q_wall_cylinder(Nx, Ny, cx, cy, Ri)
    ux = zeros(Float64, Nx, Ny)
    uy = zeros(Float64, Nx, Ny)
    Cxx = ones(Float64, Nx, Ny)
    Cxy = zeros(Float64, Nx, Ny)
    Cyy = ones(Float64, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        if is_solid[i, j]
            continue
        end
        vg = _couette_circular_velocity_gradient(i - 1.0, j - 1.0;
                                                 cx, cy, Ri, Ro, Ω)
        ux[i, j] = vg.ux
        uy[i, j] = vg.uy
        Cxx[i, j], Cxy[i, j], Cyy[i, j] =
            _stationary_direct_conformation_incompressible(
                vg.dudx, vg.dudy, vg.dvdx, vg.dvdy, λ,
            )
    end
    return (; Nx, Ny, cx, cy, Ri, Ro, Ω, λ, q_wall, is_solid, ux, uy,
            Cxx, Cxy, Cyy)
end

function _curved_couette_error_stats(p, Axx, Axy, Ayy)
    max_cut = 0.0
    max_near = 0.0
    max_far = 0.0
    for j in 3:p.Ny-2, i in 3:p.Nx-2
        p.is_solid[i, j] && continue
        err = max(abs(Axx[i, j] - p.Cxx[i, j]),
                  abs(Axy[i, j] - p.Cxy[i, j]),
                  abs(Ayy[i, j] - p.Cyy[i, j]))
        r = hypot((i - 1.0) - p.cx, (j - 1.0) - p.cy)
        if _is_cut_cell(p.q_wall, i, j)
            max_cut = max(max_cut, err)
        elseif r < p.Ri + 3.5
            max_near = max(max_near, err)
        elseif r < p.Ro
            max_far = max(max_far, err)
        end
    end
    return (; max_cut, max_near, max_far)
end

function _curved_couette_hessian_error_stats(p)
    max_cut_velocity = 0.0
    max_near_velocity = 0.0
    max_far_velocity = 0.0
    max_cut_conformation = 0.0
    max_near_conformation = 0.0
    max_far_conformation = 0.0
    for j in 3:p.Ny-2, i in 3:p.Nx-2
        p.is_solid[i, j] && continue
        x = i - 1.0
        y = j - 1.0
        href = _couette_circular_velocity_hessian(x, y; cx=p.cx, cy=p.cy,
                                                  Ri=p.Ri, Ro=p.Ro, Ω=p.Ω)
        velocity_err = maximum(abs, (
            _wall_aware_d2x_2d(p.ux, p.is_solid, i, j, p.Nx, Float64) - href.ux_xx,
            _wall_aware_dxdy_2d(p.ux, p.is_solid, i, j, p.Nx, p.Ny, Float64) - href.ux_xy,
            _wall_aware_d2y_2d(p.ux, p.is_solid, i, j, p.Ny, Float64) - href.ux_yy,
            _wall_aware_d2x_2d(p.uy, p.is_solid, i, j, p.Nx, Float64) - href.uy_xx,
            _wall_aware_dxdy_2d(p.uy, p.is_solid, i, j, p.Nx, p.Ny, Float64) - href.uy_xy,
            _wall_aware_d2y_2d(p.uy, p.is_solid, i, j, p.Ny, Float64) - href.uy_yy,
        ))

        cxx_h = _continuous_hessian_2d(
            (xx, yy) -> _couette_circular_conformation_tuple(xx, yy, p)[1],
            x, y,
        )
        cxy_h = _continuous_hessian_2d(
            (xx, yy) -> _couette_circular_conformation_tuple(xx, yy, p)[2],
            x, y,
        )
        cyy_h = _continuous_hessian_2d(
            (xx, yy) -> _couette_circular_conformation_tuple(xx, yy, p)[3],
            x, y,
        )
        conformation_err = maximum(abs, (
            _wall_aware_d2x_2d(p.Cxx, p.is_solid, i, j, p.Nx, Float64) - cxx_h.xx,
            _wall_aware_dxdy_2d(p.Cxx, p.is_solid, i, j, p.Nx, p.Ny, Float64) - cxx_h.xy,
            _wall_aware_d2y_2d(p.Cxx, p.is_solid, i, j, p.Ny, Float64) - cxx_h.yy,
            _wall_aware_d2x_2d(p.Cxy, p.is_solid, i, j, p.Nx, Float64) - cxy_h.xx,
            _wall_aware_dxdy_2d(p.Cxy, p.is_solid, i, j, p.Nx, p.Ny, Float64) - cxy_h.xy,
            _wall_aware_d2y_2d(p.Cxy, p.is_solid, i, j, p.Ny, Float64) - cxy_h.yy,
            _wall_aware_d2x_2d(p.Cyy, p.is_solid, i, j, p.Nx, Float64) - cyy_h.xx,
            _wall_aware_dxdy_2d(p.Cyy, p.is_solid, i, j, p.Nx, p.Ny, Float64) - cyy_h.xy,
            _wall_aware_d2y_2d(p.Cyy, p.is_solid, i, j, p.Ny, Float64) - cyy_h.yy,
        ))

        r = hypot(x - p.cx, y - p.cy)
        if _is_cut_cell(p.q_wall, i, j)
            max_cut_velocity = max(max_cut_velocity, velocity_err)
            max_cut_conformation = max(max_cut_conformation, conformation_err)
        elseif r < p.Ri + 3.5
            max_near_velocity = max(max_near_velocity, velocity_err)
            max_near_conformation = max(max_near_conformation, conformation_err)
        elseif r < p.Ro
            max_far_velocity = max(max_far_velocity, velocity_err)
            max_far_conformation = max(max_far_conformation, conformation_err)
        end
    end
    return (; max_cut_velocity, max_near_velocity, max_far_velocity,
              max_cut_conformation, max_near_conformation,
              max_far_conformation)
end

@testset "P28 circular Couette Oldroyd-B analytic closure is exact" begin
    p = _curved_couette_oldroydb_patch()
    max_source = 0.0
    min_eig = Inf
    for j in 1:p.Ny, i in 1:p.Nx
        p.is_solid[i, j] && continue
        vg = _couette_circular_velocity_gradient(i - 1.0, j - 1.0;
                                                 cx=p.cx, cy=p.cy,
                                                 Ri=p.Ri, Ro=p.Ro, Ω=p.Ω)
        source = _direct_source_tuple(
            p.Cxx[i, j], p.Cxy[i, j], p.Cyy[i, j],
            vg.dudx, vg.dudy, vg.dvdx, vg.dvdy, p.λ,
        )
        max_source = max(max_source, maximum(abs, source))
        min_eig = min(min_eig, _min_eig_spd_2x2(p.Cxx[i, j], p.Cxy[i, j], p.Cyy[i, j]))
    end
    @test max_source < 5e-14
    @test min_eig > 0.9
end

@testset "P29 circular Couette numerical gradient is a separate curved-wall canary" begin
    p = _curved_couette_oldroydb_patch()
    max_cut_source = 0.0
    max_far_source = 0.0
    for j in 3:p.Ny-2, i in 3:p.Nx-2
        p.is_solid[i, j] && continue
        source = _direct_source_tuple(
            p.Cxx[i, j], p.Cxy[i, j], p.Cyy[i, j],
            Kraken._wall_aware_dx_2d(p.ux, p.is_solid, i, j, p.Nx, Float64),
            Kraken._wall_aware_dy_2d(p.ux, p.is_solid, i, j, p.Ny, Float64),
            Kraken._wall_aware_dx_2d(p.uy, p.is_solid, i, j, p.Nx, Float64),
            Kraken._wall_aware_dy_2d(p.uy, p.is_solid, i, j, p.Ny, Float64),
            p.λ,
        )
        residual = maximum(abs, source)
        if _is_cut_cell(p.q_wall, i, j)
            max_cut_source = max(max_cut_source, residual)
        elseif hypot((i - 1.0) - p.cx, (j - 1.0) - p.cy) > p.Ri + 6
            max_far_source = max(max_far_source, residual)
        end
    end
    @test max_cut_source < 5e-4
    @test max_cut_source > 5 * max_far_source
end

function _curved_couette_source_residual_stats(; gradient_mode::Symbol)
    p = _curved_couette_oldroydb_patch()
    max_cut_source = 0.0
    max_near_source = 0.0
    max_far_source = 0.0
    fallback_count = 0
    for j in 3:p.Ny-2, i in 3:p.Nx-2
        p.is_solid[i, j] && continue
        grad = _test_velocity_gradient(
            p.ux, p.uy, p.is_solid, i, j, p.Nx, p.Ny, p, gradient_mode,
        )
        fallback_count += grad.fallback ? 1 : 0
        half_trace = 0.5 * (grad.dudx + grad.dvdy)
        source = _direct_source_tuple(
            p.Cxx[i, j], p.Cxy[i, j], p.Cyy[i, j],
            grad.dudx - half_trace, grad.dudy,
            grad.dvdx, grad.dvdy - half_trace,
            p.λ,
        )
        residual = maximum(abs, source)
        r = hypot((i - 1.0) - p.cx, (j - 1.0) - p.cy)
        if _is_cut_cell(p.q_wall, i, j)
            max_cut_source = max(max_cut_source, residual)
        elseif r < p.Ri + 3.5
            max_near_source = max(max_near_source, residual)
        elseif r < p.Ro
            max_far_source = max(max_far_source, residual)
        end
    end
    return (; max_cut_source, max_near_source, max_far_source,
              fallback_count)
end

@testset "P29b circular Couette source-gradient floor has a gradient-only escape hatch" begin
    kernel = _curved_couette_source_residual_stats(
        ; gradient_mode=:wall_aware,
    )
    exact = _curved_couette_source_residual_stats(
        ; gradient_mode=:analytic_couette,
    )
    polyfit = _curved_couette_source_residual_stats(
        ; gradient_mode=:polyfit4,
    )
    wallfit = _curved_couette_source_residual_stats(
        ; gradient_mode=:wallfit4,
    )

    @test kernel.max_cut_source > 1e-4
    @test exact.max_cut_source < 5e-14
    @test exact.max_near_source < 5e-14
    @test exact.max_far_source < 5e-14
    @test polyfit.fallback_count == 0
    @test polyfit.max_cut_source < 5e-5
    @test polyfit.max_cut_source < 0.5 * kernel.max_cut_source
    @test polyfit.max_far_source < 5e-6
    @test wallfit.fallback_count == 0
    @test wallfit.max_cut_source < 1.5e-5
    @test wallfit.max_cut_source < 0.4 * polyfit.max_cut_source
    @test wallfit.max_far_source < 2e-6
end

function _curved_couette_collision_only_stats(; gradient_mode::Symbol=:kernel)
    p = _curved_couette_oldroydb_patch()
    gxx = zeros(Float64, p.Nx, p.Ny, 9)
    gxy = zeros(Float64, p.Nx, p.Ny, 9)
    gyy = zeros(Float64, p.Nx, p.Ny, 9)
    Cxx = copy(p.Cxx)
    Cxy = copy(p.Cxy)
    Cyy = copy(p.Cyy)
    init_conformation_field_2d!(gxx, Cxx, p.ux, p.uy)
    init_conformation_field_2d!(gxy, Cxy, p.ux, p.uy)
    init_conformation_field_2d!(gyy, Cyy, p.ux, p.uy)
    if gradient_mode === :kernel
        _collide_direct_cde_state!(gxx, gxy, gyy, Cxx, Cxy, Cyy,
                                   p.ux, p.uy, p.is_solid, p.λ)
    else
        _collide_direct_cde_state_test_gradient!(
            gxx, gxy, gyy, Cxx, Cxy, Cyy,
            p.ux, p.uy, p.is_solid, p.λ, p; gradient_mode,
        )
    end
    return _curved_couette_error_stats(p, Cxx, Cxy, Cyy)
end

@testset "P30 circular Couette collision-only drift localizes gradient/source error" begin
    stats = _curved_couette_collision_only_stats()
    @test stats.max_cut > 1e-5
    @test stats.max_cut > stats.max_far
end

@testset "P30b circular Couette collision-only drift is gradient-limited" begin
    kernel = _curved_couette_collision_only_stats()
    exact = _curved_couette_collision_only_stats(
        ; gradient_mode=:analytic_couette,
    )
    polyfit = _curved_couette_collision_only_stats(; gradient_mode=:polyfit4)
    wallfit = _curved_couette_collision_only_stats(; gradient_mode=:wallfit4)

    @test exact.max_cut < P0_ATOL
    @test exact.max_near < P0_ATOL
    @test exact.max_far < P0_ATOL
    @test polyfit.max_cut < 5e-5
    @test polyfit.max_cut < 0.5 * kernel.max_cut
    @test polyfit.max_far < 5e-6
    @test wallfit.max_cut < 1.5e-5
    @test wallfit.max_cut < 0.4 * polyfit.max_cut
    @test wallfit.max_far < 2e-6
end

@testset "P31 circular Couette stream+BC-only drift localizes wall transport error" begin
    p = _curved_couette_oldroydb_patch()
    stats_by_name = Dict{Symbol,Any}()
    for (name, bc) in ((:cnebb, CNEBB()),
                       (:field, CNEBBField()),
                       (:field_equilibrium, CNEBBFieldEquilibrium()),
                       (:eq_gradient, CNEBBEqGradient()))
        gxx = zeros(Float64, p.Nx, p.Ny, 9)
        gxy = zeros(Float64, p.Nx, p.Ny, 9)
        gyy = zeros(Float64, p.Nx, p.Ny, 9)
        Cxx = copy(p.Cxx)
        Cxy = copy(p.Cxy)
        Cyy = copy(p.Cyy)
        init_conformation_field_2d!(gxx, Cxx, p.ux, p.uy)
        init_conformation_field_2d!(gxy, Cxy, p.ux, p.uy)
        init_conformation_field_2d!(gyy, Cyy, p.ux, p.uy)
        bxx = similar(gxx)
        bxy = similar(gxy)
        byy = similar(gyy)
        stream_2d!(bxx, gxx, p.Nx, p.Ny; sync=true)
        stream_2d!(bxy, gxy, p.Nx, p.Ny; sync=true)
        stream_2d!(byy, gyy, p.Nx, p.Ny; sync=true)
        apply_polymer_wall_bc!(bxx, gxx, p.is_solid, p.q_wall, Cxx, p.ux, p.uy, bc)
        apply_polymer_wall_bc!(bxy, gxy, p.is_solid, p.q_wall, Cxy, p.ux, p.uy, bc)
        apply_polymer_wall_bc!(byy, gyy, p.is_solid, p.q_wall, Cyy, p.ux, p.uy, bc)
        compute_conformation_macro_2d!(Cxx, bxx)
        compute_conformation_macro_2d!(Cxy, bxy)
        compute_conformation_macro_2d!(Cyy, byy)
        stats_by_name[name] = _curved_couette_error_stats(p, Cxx, Cxy, Cyy)
    end

    @test stats_by_name[:field].max_cut < P0_ATOL
    @test stats_by_name[:field_equilibrium].max_cut < P0_ATOL
    @test stats_by_name[:eq_gradient].max_cut < P0_ATOL
    @test stats_by_name[:cnebb].max_cut > 1e-5
    @test stats_by_name[:cnebb].max_cut > 10 * stats_by_name[:cnebb].max_far
end

function _curved_couette_cde_repeated_stats(bc; steps=4)
    p = _curved_couette_oldroydb_patch()
    gxx = zeros(Float64, p.Nx, p.Ny, 9)
    gxy = zeros(Float64, p.Nx, p.Ny, 9)
    gyy = zeros(Float64, p.Nx, p.Ny, 9)
    Cxx = copy(p.Cxx)
    Cxy = copy(p.Cxy)
    Cyy = copy(p.Cyy)
    init_conformation_field_2d!(gxx, Cxx, p.ux, p.uy)
    init_conformation_field_2d!(gxy, Cxy, p.ux, p.uy)
    init_conformation_field_2d!(gyy, Cyy, p.ux, p.uy)
    bxx = similar(gxx)
    bxy = similar(gxy)
    byy = similar(gyy)

    max_cut = 0.0
    max_near = 0.0
    max_far = 0.0
    for _ in 1:steps
        stream_2d!(bxx, gxx, p.Nx, p.Ny; sync=true)
        stream_2d!(bxy, gxy, p.Nx, p.Ny; sync=true)
        stream_2d!(byy, gyy, p.Nx, p.Ny; sync=true)
        apply_polymer_wall_bc!(bxx, gxx, p.is_solid, p.q_wall, Cxx, p.ux, p.uy, bc)
        apply_polymer_wall_bc!(bxy, gxy, p.is_solid, p.q_wall, Cxy, p.ux, p.uy, bc)
        apply_polymer_wall_bc!(byy, gyy, p.is_solid, p.q_wall, Cyy, p.ux, p.uy, bc)
        gxx, bxx = bxx, gxx
        gxy, bxy = bxy, gxy
        gyy, byy = byy, gyy

        compute_conformation_macro_2d!(Cxx, gxx)
        compute_conformation_macro_2d!(Cxy, gxy)
        compute_conformation_macro_2d!(Cyy, gyy)
        _collide_direct_cde_state!(gxx, gxy, gyy, Cxx, Cxy, Cyy,
                                   p.ux, p.uy, p.is_solid, p.λ)
        stats = _curved_couette_error_stats(p, Cxx, Cxy, Cyy)
        max_cut = max(max_cut, stats.max_cut)
        max_near = max(max_near, stats.max_near)
        max_far = max(max_far, stats.max_far)
    end
    return (; max_cut, max_near, max_far)
end

@testset "P32 circular Couette repeated CDE separates wall and collision errors" begin
    stats = Dict{Symbol,Any}()
    for (name, bc) in ((:cnebb, CNEBB()),
                       (:field, CNEBBField()),
                       (:field_equilibrium, CNEBBFieldEquilibrium()),
                       (:eq_gradient, CNEBBEqGradient()))
        stats[name] = _curved_couette_cde_repeated_stats(bc; steps=4)
    end

    @test stats[:cnebb].max_cut > 1e-2
    @test stats[:field].max_cut < 1e-3
    @test stats[:field_equilibrium].max_cut < 1e-3
    @test stats[:eq_gradient].max_cut > stats[:field_equilibrium].max_cut
    @test stats[:field_equilibrium].max_near < stats[:field].max_near
    @test stats[:field_equilibrium].max_far < stats[:field].max_far
end

@testset "P18l prototype extrap-eq BC: M4-pre curved Couette Hessians" begin
    # The quadratic BC needs local second derivatives at cut cells.
    # This canary checks the wall-aware stencils against the continuous
    # circular-Couette solution before those derivatives are allowed to
    # enter a population fill.
    p = _curved_couette_oldroydb_patch()
    h = _curved_couette_hessian_error_stats(p)
    @test h.max_cut_velocity < 3.5e-4
    @test h.max_near_velocity < 1e-5
    @test h.max_far_velocity < 5e-6
    @test h.max_cut_conformation < 1e-3
    @test h.max_near_conformation < 3e-5
    @test h.max_far_conformation < 1e-5
end

# ---------------------------------------------------------------------
# M4: prototype extrap-eq BC on curved circular Couette. Linear
# extrapolation truncates at O(|c_q|² · ∇²u) when the analytical
# velocity field is not affine in space (circular Couette has 1/r²
# dependence). The canary measures the actual residual so we can
# judge whether quadratic extrapolation is required before the BC
# is generalisable. Setup mirrors P31/P32.
# ---------------------------------------------------------------------

function _extrap_repeated_curved_couette(; steps::Int=4,
                                         wall_bc=_extrap_eq_wall_bc!,
                                         gradient_mode::Symbol=:kernel)
    p = _curved_couette_oldroydb_patch()
    Cxx = copy(p.Cxx); Cxy = copy(p.Cxy); Cyy = copy(p.Cyy)
    gxx = zeros(Float64, p.Nx, p.Ny, 9)
    gxy = similar(gxx); gyy = similar(gxx)
    init_conformation_field_2d!(gxx, Cxx, p.ux, p.uy)
    init_conformation_field_2d!(gxy, Cxy, p.ux, p.uy)
    init_conformation_field_2d!(gyy, Cyy, p.ux, p.uy)
    bxx = similar(gxx); bxy = similar(gxy); byy = similar(gyy)

    max_cut = 0.0; max_near = 0.0; max_far = 0.0
    for _ in 1:steps
        stream_2d!(bxx, gxx, p.Nx, p.Ny; sync=true)
        stream_2d!(bxy, gxy, p.Nx, p.Ny; sync=true)
        stream_2d!(byy, gyy, p.Nx, p.Ny; sync=true)
        wall_bc(bxx, p.is_solid, p.q_wall, Cxx, p.ux, p.uy)
        wall_bc(bxy, p.is_solid, p.q_wall, Cxy, p.ux, p.uy)
        wall_bc(byy, p.is_solid, p.q_wall, Cyy, p.ux, p.uy)
        gxx, bxx = bxx, gxx
        gxy, bxy = bxy, gxy
        gyy, byy = byy, gyy
        compute_conformation_macro_2d!(Cxx, gxx)
        compute_conformation_macro_2d!(Cxy, gxy)
        compute_conformation_macro_2d!(Cyy, gyy)
        if gradient_mode === :kernel
            _collide_direct_cde_state!(gxx, gxy, gyy, Cxx, Cxy, Cyy,
                                       p.ux, p.uy, p.is_solid, p.λ)
        else
            _collide_direct_cde_state_test_gradient!(
                gxx, gxy, gyy, Cxx, Cxy, Cyy,
                p.ux, p.uy, p.is_solid, p.λ, p; gradient_mode,
            )
        end
        stats = _curved_couette_error_stats(p, Cxx, Cxy, Cyy)
        max_cut = max(max_cut, stats.max_cut)
        max_near = max(max_near, stats.max_near)
        max_far = max(max_far, stats.max_far)
    end
    return (; max_cut, max_near, max_far)
end

function _extrap_curved_couette_fill_residual(wall_bc)
    p = _curved_couette_oldroydb_patch()
    components = (
        (field=p.Cxx, ref=1),
        (field=p.Cxy, ref=2),
        (field=p.Cyy, ref=3),
    )
    max_filled = 0.0
    max_macro = 0.0
    n_filled = 0
    for comp in components
        C = copy(comp.field)
        g = zeros(Float64, p.Nx, p.Ny, 9)
        init_conformation_field_2d!(g, C, p.ux, p.uy)
        buf = similar(g)
        stream_2d!(buf, g, p.Nx, p.Ny; sync=true)
        wall_bc(buf, p.is_solid, p.q_wall, C, p.ux, p.uy)
        for j in 3:p.Ny-2, i in 3:p.Nx-2
            p.is_solid[i, j] && continue
            _is_cut_cell(p.q_wall, i, j) || continue
            φ = 0.0
            for q in 1:9
                φ += buf[i, j, q]
            end
            max_macro = max(max_macro, abs(φ - comp.field[i, j]))
            for q in 2:9
                cx = Int(D2Q9_CX[q]); cy = Int(D2Q9_CY[q])
                si = i - cx; sj = j - cy
                in_dom = 1 <= si <= p.Nx && 1 <= sj <= p.Ny
                src_solid = !in_dom || p.is_solid[si, sj]
                src_solid || continue
                cref = _couette_circular_conformation_tuple(
                    (i - 1.0) - cx, (j - 1.0) - cy, p,
                )[comp.ref]
                vg = _couette_circular_velocity_gradient(
                    (i - 1.0) - cx, (j - 1.0) - cy;
                    cx=p.cx, cy=p.cy, Ri=p.Ri, Ro=p.Ro, Ω=p.Ω,
                )
                target = equilibrium(D2Q9(), cref, vg.ux, vg.uy, q)
                max_filled = max(max_filled, abs(buf[i, j, q] - target))
                n_filled += 1
            end
        end
    end
    return (; max_filled, max_macro, n_filled)
end

@testset "P18l prototype extrap-eq BC: M4 curved Couette CDE" begin
    # Curved circular Couette: u = (-F·yr, F·xr) with F = A + B/r²
    # is NOT affine in space, so first-order linear extrapolation
    # truncates at O(|c_q|² · ∇²u). The canary records the actual
    # residual structure so we can decide whether quadratic
    # extrapolation (or a Filippova-Hänel-style q_w-aware scheme) is
    # the next development step.
    #
    # Frozen findings (steps = 4):
    #   * extrap-eq beats strict CNEBB by an order of magnitude on
    #     cut cells (14× in current measurements) — the inclined-wall
    #     asymmetry signature is gone.
    #   * extrap-eq is competitive with CNEBBField / FieldEq at near-
    #     wall and far-field cells (≤ 1× to 1.5×).
    #   * BUT linear extrapolation at the cut cells is ≈10× worse
    #     than Field-pinned variants at cut cells on curved walls.
    #     This is the open development item: curved-wall BC needs
    #     either quadratic extrapolation or a wall-position-aware
    #     fictitious-equilibrium step (FH / MLS-like).
    r = _extrap_repeated_curved_couette(; steps=4)
    cnebb = _curved_couette_cde_repeated_stats(CNEBB(); steps=4)
    field = _curved_couette_cde_repeated_stats(CNEBBField(); steps=4)
    field_eq = _curved_couette_cde_repeated_stats(
        CNEBBFieldEquilibrium(); steps=4,
    )
    @test r.max_cut > 1e-3            # truncation floor on curved wall
    @test r.max_cut < 1e-2            # but bounded
    @test r.max_cut < 0.1 * cnebb.max_cut
    @test r.max_near < 1.5 * field.max_near
    @test r.max_near < 1.5 * field_eq.max_near
    @test r.max_far < 1.5 * field_eq.max_far
end

@testset "P18l prototype extrap-eq BC: M5a quadratic fill is not enough" begin
    # Quadratic full-link extrapolation does improve the missing
    # populations against the pure-stream analytic oracle, but without
    # a conservative rest rebalance it worsens the cut-cell macro
    # moment. This freezes why the raw quadratic helper is not a
    # promotable wall BC by itself.
    linear_fill = _extrap_curved_couette_fill_residual(_extrap_eq_wall_bc!)
    quadratic_fill = _extrap_curved_couette_fill_residual(
        _extrap_eq_wall_bc_quadratic!,
    )
    @test linear_fill.n_filled > 0
    @test quadratic_fill.max_filled < 0.5 * linear_fill.max_filled
    @test quadratic_fill.max_macro > linear_fill.max_macro

    linear = _extrap_repeated_curved_couette(; steps=4)
    quadratic = _extrap_repeated_curved_couette(
        ; steps=4, wall_bc=_extrap_eq_wall_bc_quadratic!,
    )
    @test quadratic.max_cut > linear.max_cut
    @test quadratic.max_far < 1.1 * linear.max_far
end

@testset "P18l prototype extrap-eq BC: M5b rest rebalance closes curved cut floor" begin
    # The curved-wall channel left open by M4 is local conservation:
    # affine bulk cancellation is no longer exact for curved fields.
    # Rebalancing only the rest population restores the local macro
    # moment without overwriting outgoing pure-stream populations.
    for normal in ((3.0, 4.0), (4.0, 3.0))
        r_const = _extrap_population_residual(
            ; normal, γ=0.01, constant_velocity=true, steps=1,
            wall_bc=_extrap_eq_wall_bc_rebalanced!,
        )
        @test r_const.err_filled < P0_ATOL
        @test r_const.err_rest < P0_ATOL
        @test r_const.err_macro < P0_ATOL

        r_affine = _extrap_population_residual(
            ; normal, γ=0.01, constant_velocity=false, steps=1,
            wall_bc=_extrap_eq_wall_bc_rebalanced!,
        )
        @test r_affine.err_filled < P0_ATOL
        @test r_affine.err_rest < P0_ATOL
        @test r_affine.err_macro < P0_ATOL

        r_transport = _extrap_population_residual(
            ; normal, γ=0.01, constant_velocity=false, steps=4,
            wall_bc=_extrap_eq_wall_bc_rebalanced!,
        )
        @test r_transport.err_macro < 5e-3

        r_cde = _extrap_repeated_cde(
            ; normal, γ=0.01, steps=16,
            wall_bc=_extrap_eq_wall_bc_rebalanced!,
        )
        @test r_cde.max_cut < P0_ATOL
        @test r_cde.max_far < 1e-5
    end

    linear = _extrap_repeated_curved_couette(; steps=4)
    rebalanced = _extrap_repeated_curved_couette(
        ; steps=4, wall_bc=_extrap_eq_wall_bc_rebalanced!,
    )
    quadratic_rebalanced = _extrap_repeated_curved_couette(
        ; steps=4, wall_bc=_extrap_eq_wall_bc_quadratic_rebalanced!,
    )
    collision = _curved_couette_collision_only_stats()
    field_eq = _curved_couette_cde_repeated_stats(
        CNEBBFieldEquilibrium(); steps=4,
    )
    @test rebalanced.max_cut < 0.12 * linear.max_cut
    @test rebalanced.max_cut < 4e-4
    @test rebalanced.max_cut < 1.1 * field_eq.max_cut
    @test rebalanced.max_cut > 2.0 * collision.max_cut
    @test rebalanced.max_near < linear.max_near
    @test rebalanced.max_far < linear.max_far
    @test quadratic_rebalanced.max_cut < 4e-4
end

@testset "P18l prototype extrap-eq BC: M6 curved CDE is now gradient-limited" begin
    # With rest rebalance active, replacing only the velocity gradient
    # in the collision source by the analytic circular-Couette gradient
    # removes the cut-cell floor. A generic fluid-only quartic fit gets
    # close but still misses the <1e-4 promotion target over four CDE
    # steps. Adding q_wall intersection samples with the moving-wall
    # velocity closes the target, so the remaining production work is
    # to promote this geometric gradient path, not another population
    # fill tweak.
    kernel = _extrap_repeated_curved_couette(
        ; steps=4, wall_bc=_extrap_eq_wall_bc_rebalanced!,
    )
    exact = _extrap_repeated_curved_couette(
        ; steps=4, wall_bc=_extrap_eq_wall_bc_rebalanced!,
        gradient_mode=:analytic_couette,
    )
    polyfit = _extrap_repeated_curved_couette(
        ; steps=4, wall_bc=_extrap_eq_wall_bc_rebalanced!,
        gradient_mode=:polyfit4,
    )
    wallfit = _extrap_repeated_curved_couette(
        ; steps=4, wall_bc=_extrap_eq_wall_bc_rebalanced!,
        gradient_mode=:wallfit4,
    )

    @test exact.max_cut < P0_ATOL
    @test polyfit.max_cut < 0.4 * kernel.max_cut
    @test polyfit.max_cut < 1.5e-4
    @test polyfit.max_cut > 1e-4
    @test wallfit.max_cut < 5e-5
    @test wallfit.max_cut < 0.4 * polyfit.max_cut
    @test polyfit.max_near < 1.01 * kernel.max_near
    @test polyfit.max_far < 1.01 * kernel.max_far
    @test wallfit.max_near < 1.01 * kernel.max_near
    @test wallfit.max_far < 1.01 * kernel.max_far
end

@testset "P18l prototype extrap-eq BC: M7 compares curved gradient recovery strategies" begin
    modes = (:ghost_axis, :embedded_axis, :normal_tangent, :polyfit4, :wallfit4)
    source = Dict(
        mode => _curved_couette_source_residual_stats(; gradient_mode=mode)
        for mode in modes
    )
    cde = Dict(
        mode => _extrap_repeated_curved_couette(
            ; steps=4, wall_bc=_extrap_eq_wall_bc_rebalanced!,
            gradient_mode=mode,
        )
        for mode in modes
    )

    @test all(source[mode].fallback_count == 0 for mode in modes)

    @test source[:embedded_axis].max_cut_source < 4e-5
    @test source[:embedded_axis].max_cut_source < source[:polyfit4].max_cut_source
    @test source[:wallfit4].max_cut_source < 0.5 * source[:embedded_axis].max_cut_source

    @test cde[:embedded_axis].max_cut < 1e-4
    @test cde[:embedded_axis].max_cut < cde[:polyfit4].max_cut
    @test cde[:wallfit4].max_cut < 5e-5
    @test cde[:wallfit4].max_cut < 0.5 * cde[:embedded_axis].max_cut

    @test cde[:ghost_axis].max_cut > 1e-3
    @test cde[:ghost_axis].max_cut > 10.0 * cde[:embedded_axis].max_cut
    @test cde[:normal_tangent].max_cut > 5e-3
    @test cde[:normal_tangent].max_cut > 5.0 * cde[:ghost_axis].max_cut
end

@testset "P18l prototype extrap-eq BC: M8 embedded and wallfit gradients are coefficient paths" begin
    embedded_shape = _gradient_coeff_stencil_shape_stats(:embedded_axis_coeff)
    wallfit_shape = _gradient_coeff_stencil_shape_stats(:wallfit4_coeff)

    @test embedded_shape.fallback_count == 0
    @test embedded_shape.max_terms <= 3
    @test embedded_shape.max_wall_terms <= 2
    @test embedded_shape.max_points <= 2
    @test embedded_shape.max_rank <= 2

    @test wallfit_shape.fallback_count == 0
    @test wallfit_shape.max_terms <= 48
    @test wallfit_shape.max_wall_terms <= 24
    @test wallfit_shape.max_points <= 48
    @test wallfit_shape.max_rank == 14

    for (direct_mode, coeff_mode) in (
            (:embedded_axis, :embedded_axis_coeff),
            (:wallfit4, :wallfit4_coeff))
        direct_source = _curved_couette_source_residual_stats(
            ; gradient_mode=direct_mode,
        )
        coeff_source = _curved_couette_source_residual_stats(
            ; gradient_mode=coeff_mode,
        )
        direct_cde = _extrap_repeated_curved_couette(
            ; steps=4, wall_bc=_extrap_eq_wall_bc_rebalanced!,
            gradient_mode=direct_mode,
        )
        coeff_cde = _extrap_repeated_curved_couette(
            ; steps=4, wall_bc=_extrap_eq_wall_bc_rebalanced!,
            gradient_mode=coeff_mode,
        )

        @test coeff_source.fallback_count == direct_source.fallback_count
        @test isapprox(coeff_source.max_cut_source, direct_source.max_cut_source;
                       atol=1e-12, rtol=0.0)
        @test isapprox(coeff_source.max_near_source, direct_source.max_near_source;
                       atol=1e-12, rtol=0.0)
        @test isapprox(coeff_source.max_far_source, direct_source.max_far_source;
                       atol=1e-12, rtol=0.0)

        @test isapprox(coeff_cde.max_cut, direct_cde.max_cut; atol=1e-12, rtol=0.0)
        @test isapprox(coeff_cde.max_near, direct_cde.max_near; atol=1e-12, rtol=0.0)
        @test isapprox(coeff_cde.max_far, direct_cde.max_far; atol=1e-12, rtol=0.0)
    end
end

@testset "P18l prototype extrap-eq BC: M9 production gradient stencils reproduce prototypes" begin
    p = _curved_couette_oldroydb_patch()
    uwx, uwy = _curved_couette_wall_velocity_arrays(p)
    cases = (
        (; prod_mode=:embedded_axis, test_mode=:embedded_axis_coeff,
           max_terms=4, max_cut=1e-4),
        (; prod_mode=:wallfit4, test_mode=:wallfit4_coeff,
           max_terms=64, max_cut=5e-5),
    )

    for case in cases
        stencils = Kraken.precompute_conformation_gradient_stencils_2d(
            p.is_solid, p.q_wall; mode=case.prod_mode,
            max_terms=case.max_terms, FT=Float64,
        )
        stats = Kraken.conformation_gradient_stencil_stats_2d(stencils)
        source = _curved_couette_source_residual_stats(
            ; gradient_mode=case.test_mode,
        )

        max_dudx = 0.0
        max_dudy = 0.0
        max_dvdx = 0.0
        max_dvdy = 0.0
        active_fallback_count = 0
        for j in 1:p.Ny, i in 1:p.Nx
            p.is_solid[i, j] && continue
            hypot((i - 1.0) - p.cx, (j - 1.0) - p.cy) < p.Ro || continue
            active_fallback_count += stencils.fallback[i, j, 1] ? 1 : 0
            active_fallback_count += stencils.fallback[i, j, 2] ? 1 : 0
            prod = Kraken.conformation_velocity_gradient_from_stencils_2d(
                p.ux, p.uy, uwx, uwy, stencils, i, j,
            )
            ref = _test_velocity_gradient(
                p.ux, p.uy, p.is_solid, i, j, p.Nx, p.Ny, p, case.test_mode,
            )
            max_dudx = max(max_dudx, abs(prod.dudx - ref.dudx))
            max_dudy = max(max_dudy, abs(prod.dudy - ref.dudy))
            max_dvdx = max(max_dvdx, abs(prod.dvdx - ref.dvdx))
            max_dvdy = max(max_dvdy, abs(prod.dvdy - ref.dvdy))
        end

        @test active_fallback_count == 0
        @test stats.max_count <= case.max_terms
        @test source.max_cut_source < case.max_cut
        @test max_dudx < 1e-12
        @test max_dudy < 1e-12
        @test max_dvdx < 1e-12
        @test max_dvdy < 1e-12
    end
end

@testset "P18l prototype extrap-eq BC: M10 production stencil collision reproduces prototypes" begin
    cases = (
        (; prod_mode=:embedded_axis_prod, ref_mode=:embedded_axis_coeff,
           max_cut=1e-4),
        (; prod_mode=:wallfit4_prod, ref_mode=:wallfit4_coeff,
           max_cut=5e-5),
    )

    for case in cases
        ref_collision = _curved_couette_collision_only_stats(
            ; gradient_mode=case.ref_mode,
        )
        prod_collision = _curved_couette_collision_only_stats(
            ; gradient_mode=case.prod_mode,
        )
        ref_cde = _extrap_repeated_curved_couette(
            ; steps=4, wall_bc=_extrap_eq_wall_bc_rebalanced!,
            gradient_mode=case.ref_mode,
        )
        prod_cde = _extrap_repeated_curved_couette(
            ; steps=4, wall_bc=_extrap_eq_wall_bc_rebalanced!,
            gradient_mode=case.prod_mode,
        )

        @test prod_collision.max_cut < case.max_cut
        @test isapprox(prod_collision.max_cut, ref_collision.max_cut;
                       atol=1e-12, rtol=0.0)
        @test isapprox(prod_collision.max_near, ref_collision.max_near;
                       atol=1e-12, rtol=0.0)
        @test isapprox(prod_collision.max_far, ref_collision.max_far;
                       atol=1e-12, rtol=0.0)
        @test isapprox(prod_cde.max_cut, ref_cde.max_cut; atol=1e-12, rtol=0.0)
        @test isapprox(prod_cde.max_near, ref_cde.max_near; atol=1e-12, rtol=0.0)
        @test isapprox(prod_cde.max_far, ref_cde.max_far; atol=1e-12, rtol=0.0)
    end
end

function _square_obstacle_affine_velocity_patch(;
        dudx=0.002, dudy=0.003, dvdx=-0.001, dvdy=-0.002,
        u0=0.015, v0=-0.01)
    geom = square_obstacle_channel_geometry_2d(; H=24, side=6, L_up=3, L_down=4)
    Nx, Ny = geom.Nx, geom.Ny
    cx0 = geom.i_step + (geom.H_ref - 1) / 2
    cy0 = (Ny - 1) / 2
    ux = [
        u0 + dudx * ((i - 1) - cx0) + dudy * ((j - 1) - cy0)
        for i in 1:Nx, j in 1:Ny
    ]
    uy = [
        v0 + dvdx * ((i - 1) - cx0) + dvdy * ((j - 1) - cy0)
        for i in 1:Nx, j in 1:Ny
    ]
    uwx = zeros(Float64, Nx, Ny, 9)
    uwy = zeros(Float64, Nx, Ny, 9)
    for q in 2:9, j in 1:Ny, i in 1:Nx
        qw = geom.q_wall[i, j, q]
        qw > 0.0 || continue
        wx = (i - 1.0) + qw * D2Q9_CX[q]
        wy = (j - 1.0) + qw * D2Q9_CY[q]
        uwx[i, j, q] = u0 + dudx * (wx - cx0) + dudy * (wy - cy0)
        uwy[i, j, q] = v0 + dvdx * (wx - cx0) + dvdy * (wy - cy0)
    end
    return (; geom, ux, uy, uwx, uwy, dudx, dudy, dvdx, dvdy,
              cx0, cy0)
end

function _square_obstacle_prod_gradient_stats(mode::Symbol)
    p = _square_obstacle_affine_velocity_patch()
    stencils = Kraken.precompute_conformation_gradient_stencils_2d(
        p.geom.is_solid, p.geom.q_wall; mode,
        max_terms=mode === :embedded_axis ? 4 : 64,
        FT=Float64,
    )
    max_cut_gradient = 0.0
    max_near_gradient = 0.0
    active_fallback_count = 0
    n_cut = 0
    n_near = 0
    for j in 1:p.geom.Ny, i in 1:p.geom.Nx
        p.geom.is_solid[i, j] && continue
        is_cut = _is_cut_cell(p.geom.q_wall, i, j)
        near = is_cut || any(
            1 <= i + Int(D2Q9_CX[q]) <= p.geom.Nx &&
            1 <= j + Int(D2Q9_CY[q]) <= p.geom.Ny &&
            p.geom.is_solid[i + Int(D2Q9_CX[q]), j + Int(D2Q9_CY[q])]
            for q in 2:9
        )
        near || continue
        grad = Kraken.conformation_velocity_gradient_from_stencils_2d(
            p.ux, p.uy, p.uwx, p.uwy, stencils, i, j,
        )
        err = maximum(abs, (
            grad.dudx - p.dudx, grad.dudy - p.dudy,
            grad.dvdx - p.dvdx, grad.dvdy - p.dvdy,
        ))
        active_fallback_count += stencils.fallback[i, j, 1] ? 1 : 0
        active_fallback_count += stencils.fallback[i, j, 2] ? 1 : 0
        if is_cut
            n_cut += 1
            max_cut_gradient = max(max_cut_gradient, err)
        else
            n_near += 1
            max_near_gradient = max(max_near_gradient, err)
        end
    end
    return (; max_cut_gradient, max_near_gradient, active_fallback_count,
              n_cut, n_near,
              stats=Kraken.conformation_gradient_stencil_stats_2d(stencils))
end

function _square_obstacle_extrap_eq_affine_once_stats()
    p = _square_obstacle_affine_velocity_patch()
    Nx, Ny = p.geom.Nx, p.geom.Ny
    C = [
        1.0 + 0.02 * ((i - 1) - p.cx0) - 0.015 * ((j - 1) - p.cy0)
        for i in 1:Nx, j in 1:Ny
    ]
    g = zeros(Float64, Nx, Ny, 9)
    init_conformation_field_2d!(g, C, p.ux, p.uy)
    buf = similar(g)
    stream_2d!(buf, g, Nx, Ny; sync=true)
    _extrap_eq_wall_bc_rebalanced!(buf, p.geom.is_solid, p.geom.q_wall,
                                   C, p.ux, p.uy)

    max_macro = 0.0
    max_sum = 0.0
    n_cut = 0
    for j in 1:Ny, i in 1:Nx
        _is_cut_cell(p.geom.q_wall, i, j) || continue
        n_cut += 1
        macro_value = sum(buf[i, j, q] for q in 1:9)
        max_macro = max(max_macro, abs(macro_value - C[i, j]))
        max_sum = max(max_sum, abs(macro_value - sum(buf[i, j, :])))
    end
    return (; max_macro, max_sum, n_cut)
end

@testset "P18l prototype extrap-eq BC: M11 square obstacle remontée" begin
    for mode in (:embedded_axis, :wallfit4)
        stats = _square_obstacle_prod_gradient_stats(mode)
        @test stats.n_cut > 0
        @test stats.max_cut_gradient < 5e-14
        @test stats.max_near_gradient < 5e-14
        @test stats.active_fallback_count <= 4
        @test stats.stats.max_count <= (mode === :embedded_axis ? 4 : 64)
    end

    fill_stats = _square_obstacle_extrap_eq_affine_once_stats()
    @test fill_stats.n_cut > 0
    @test fill_stats.max_macro < P0_ATOL
    @test fill_stats.max_sum < P0_ATOL
end
