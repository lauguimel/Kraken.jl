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
    (name=:cnebb_eq_gradient, bc=CNEBBEqGradient()),
    (name=:cnebb_cutlink_eq_gradient, bc=CNEBBCutLinkEqGradient()),
    (name=:ylw_a, bc=YLW_A()),
    (name=:ylw_b, bc=YLW_B()),
    (name=:ylw_balance, bc=YLWBalanceOnly()),
)

function _active_bc_linear_macro_passes(name::Symbol, orientation::Symbol, velocity)
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
        if case.name === :cnebb_eq_gradient
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
        if case.name === :cnebb_eq_gradient
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
        if case.name === :cnebb_eq_gradient
            @test result.max_macro_error < P0_ATOL
        else
            push!(broken_velocity_errors[case.name], result.max_macro_error)
        end
    end
    for case in _wall_bc_cases()
        case.name === :cnebb_eq_gradient && continue
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
        if case.name === :cnebb_eq_gradient
            @test result.max_macro_error < P0_ATOL
        else
            push!(broken_gradient_errors[case.name], result.max_macro_error)
        end
    end
    for case in _wall_bc_cases()
        case.name === :cnebb_eq_gradient && continue
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
        if case.name in (:cnebb_eq_gradient, :cnebb_cutlink_eq_gradient)
            @test result.max_macro_error < P0_ATOL
        else
            push!(arbitrary_velocity_errors[case.name], result.max_macro_error)
        end
    end
    for case in _wall_bc_cases()
        case.name in (:cnebb_eq_gradient, :cnebb_cutlink_eq_gradient) && continue
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
        if case.name in (:cnebb_eq_gradient, :cnebb_cutlink_eq_gradient)
            @test result.max_macro_error < P0_ATOL
        else
            push!(broken_macro_errors[case.name], result.max_macro_error)
        end
    end
    for case in _wall_bc_cases()
        case.name in (:cnebb_eq_gradient, :cnebb_cutlink_eq_gradient) && continue
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
                                  divergence_mode::Symbol=:numerical)
    if collision === :trt
        collide_conformation_2d!(
            g, C_field, ux, uy, Cxx, Cxy, Cyy, is_solid,
            tau_plus, λ; magic, component, divergence_mode,
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
                is_solid, tau_plus, λ, component; magic,
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

_is_cut_cell(q_wall, i, j) = any(q -> q > 0.0, view(q_wall, i, j, :))

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
    exact = _curved_affine_bc_once_error(CNEBBEqGradient())
    @test exact.max_cut < P0_ATOL

    for bc in (CNEBB(), CNEBBQAware(), YLWBalanceOnly())
        broken = _curved_affine_bc_once_error(bc)
        @test broken.max_cut > 1e-3
    end
end

@testset "P22 curved frozen affine CDE one-step localizes CNEBB pollution" begin
    p, Cxx, Cxy, Cyy = _curved_affine_cde_once(CNEBBEqGradient())
    exact = _max_curved_cut_error(p.q_wall, p.is_solid, Cxx, Cxy, Cyy,
                                  p.cxx, p.cxy, p.cyy)
    @test exact.max_cut < P0_ATOL

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
    for (name, bc) in ((:cnebb, CNEBB()), (:eq_gradient, CNEBBEqGradient()))
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
    @test hypot(results[:eq_gradient].Fx, results[:eq_gradient].Fy) < 1e-5
    @test hypot(results[:cnebb].Fx, results[:cnebb].Fy) > 1e-3
end

@testset "P25 active curved-wall defect is present for every cut-link orientation" begin
    for q_out in 2:9, qw in (0.1, 0.3, 0.5, 0.7, 0.9)
        fixed_macro, fixed_sum = _single_cut_link_macro_error_bc(
            q_out, CNEBBEqGradient(); q_wall_value=qw,
            orientation=:uniform, velocity=(0.03, -0.02),
        )
        @test abs(fixed_sum) < P0_ATOL
        @test abs(fixed_macro) < P0_ATOL

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
    @test max_cut_source > 1e-5
    @test max_cut_source > 5 * max_far_source
end

@testset "P30 circular Couette collision-only drift localizes gradient/source error" begin
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
    stats = _curved_couette_error_stats(p, Cxx, Cxy, Cyy)
    @test stats.max_cut > 1e-5
    @test stats.max_cut > stats.max_far
end

@testset "P31 circular Couette stream+BC-only drift localizes wall transport error" begin
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
    stream_2d!(bxx, gxx, p.Nx, p.Ny; sync=true)
    stream_2d!(bxy, gxy, p.Nx, p.Ny; sync=true)
    stream_2d!(byy, gyy, p.Nx, p.Ny; sync=true)
    apply_polymer_wall_bc!(bxx, gxx, p.is_solid, p.q_wall, Cxx, p.ux, p.uy, CNEBB())
    apply_polymer_wall_bc!(bxy, gxy, p.is_solid, p.q_wall, Cxy, p.ux, p.uy, CNEBB())
    apply_polymer_wall_bc!(byy, gyy, p.is_solid, p.q_wall, Cyy, p.ux, p.uy, CNEBB())
    compute_conformation_macro_2d!(Cxx, bxx)
    compute_conformation_macro_2d!(Cxy, bxy)
    compute_conformation_macro_2d!(Cyy, byy)
    stats = _curved_couette_error_stats(p, Cxx, Cxy, Cyy)
    @test stats.max_cut > 1e-5
    @test stats.max_cut > 10 * stats.max_far
end
