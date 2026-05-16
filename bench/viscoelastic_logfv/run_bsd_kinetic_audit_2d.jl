#!/usr/bin/env julia

using Kraken
using Printf
using Test

const CXS = (0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0)
const CYS = (0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0)
const WS = (4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
            1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0)

struct BSDAuditResult
    N::Int
    rel_l2_kinetic_vs_fd_bsd::Float64
    rel_l2_fd_baseline::Float64
    l2_fd_bsd::Float64
    l2_delta::Float64
    max_abs_delta::Float64
    max_i::Int
    max_j::Int
    status::Bool
end

@inline function feq_host(q::Int, rho, ux, uy)
    cx = CXS[q]
    cy = CYS[q]
    cu = cx * ux + cy * uy
    usq = ux * ux + uy * uy
    return WS[q] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)
end

@inline function derivative_x(field, i::Int, j::Int, Nx::Int)
    if 1 < i < Nx
        return (field[i + 1, j] - field[i - 1, j]) * 0.5
    elseif i == 1
        return Nx >= 3 ? (-3.0 * field[i, j] + 4.0 * field[i + 1, j] - field[i + 2, j]) * 0.5 :
               field[i + 1, j] - field[i, j]
    else
        return Nx >= 3 ? (3.0 * field[i, j] - 4.0 * field[i - 1, j] + field[i - 2, j]) * 0.5 :
               field[i, j] - field[i - 1, j]
    end
end

@inline function derivative_y(field, i::Int, j::Int, Ny::Int)
    if 1 < j < Ny
        return (field[i, j + 1] - field[i, j - 1]) * 0.5
    elseif j == 1
        return Ny >= 3 ? (-3.0 * field[i, j] + 4.0 * field[i, j + 1] - field[i, j + 2]) * 0.5 :
               field[i, j + 1] - field[i, j]
    else
        return Ny >= 3 ? (3.0 * field[i, j] - 4.0 * field[i, j - 1] + field[i, j - 2]) * 0.5 :
               field[i, j] - field[i, j - 1]
    end
end

@inline function second_derivative_x(field, i::Int, j::Int, Nx::Int)
    if 1 < i < Nx
        return field[i + 1, j] - 2.0 * field[i, j] + field[i - 1, j]
    elseif i == 1 && Nx >= 3
        return field[i, j] - 2.0 * field[i + 1, j] + field[i + 2, j]
    elseif i == Nx && Nx >= 3
        return field[i, j] - 2.0 * field[i - 1, j] + field[i - 2, j]
    else
        return 0.0
    end
end

@inline function second_derivative_y(field, i::Int, j::Int, Ny::Int)
    if 1 < j < Ny
        return field[i, j + 1] - 2.0 * field[i, j] + field[i, j - 1]
    elseif j == 1 && Ny >= 3
        return field[i, j] - 2.0 * field[i, j + 1] + field[i, j + 2]
    elseif j == Ny && Ny >= 3
        return field[i, j] - 2.0 * field[i, j - 1] + field[i, j - 2]
    else
        return 0.0
    end
end

function build_fixture(N::Int; nu_s::Float64, nu_p::Float64, zeta::Float64)
    Nx = N
    Ny = N
    rho = ones(Float64, Nx, Ny)
    ux = zeros(Float64, Nx, Ny)
    uy = zeros(Float64, Nx, Ny)
    tauxx = zeros(Float64, Nx, Ny)
    tauxy = zeros(Float64, Nx, Ny)
    tauyy = zeros(Float64, Nx, Ny)

    amp_a = 5.0e-5
    amp_b = -3.0e-5
    center = 0.5 * (N + 1)
    for j in 1:Ny, i in 1:Nx
        x = i - center
        y = j - center
        ux[i, j] = amp_a * x * x + 2.0 * amp_b * x * y
        uy[i, j] = -2.0 * amp_a * x * y - amp_b * y * y

        xn = (i - 1) / (Nx - 1)
        yn = (j - 1) / (Ny - 1)
        tauxx[i, j] = 0.02 * sin(2.0 * pi * xn) * cos(pi * yn)
        tauxy[i, j] = 0.015 * cos(pi * xn) * sin(2.0 * pi * yn)
        tauyy[i, j] = -0.012 * sin(pi * xn) * sin(2.0 * pi * yn)
    end

    nu_lbm = nu_s + zeta * nu_p
    s_plus = trt_rates(nu_lbm)[1]
    f = zeros(Float64, Nx, Ny, 9)
    cs2 = 1.0 / 3.0
    for j in 1:Ny, i in 1:Nx
        dudx = derivative_x(ux, i, j, Nx)
        dudy = derivative_y(ux, i, j, Ny)
        dvdx = derivative_x(uy, i, j, Nx)
        dvdy = derivative_y(uy, i, j, Ny)
        sxx = dudx
        sxy = 0.5 * (dudy + dvdx)
        syy = dvdy
        for q in 1:9
            cx = CXS[q]
            cy = CYS[q]
            ce_projector_s = (cx * cx - cs2) * sxx + 2.0 * cx * cy * sxy + (cy * cy - cs2) * syy
            f_neq = -(WS[q] / cs2) * rho[i, j] * ce_projector_s / s_plus
            f[i, j, q] = feq_host(q, rho[i, j], ux[i, j], uy[i, j]) + f_neq
        end
    end

    return (; rho, ux, uy, tauxx, tauxy, tauyy, f, s_plus, nu_lbm)
end

function compute_fd_fields(fields, nu_p::Float64, zeta::Float64)
    Nx, Ny = size(fields.ux)
    fx_poly = zeros(Float64, Nx, Ny)
    fy_poly = zeros(Float64, Nx, Ny)
    fx_fd_bsd = zeros(Float64, Nx, Ny)
    fy_fd_bsd = zeros(Float64, Nx, Ny)
    lap_ux = zeros(Float64, Nx, Ny)
    lap_uy = zeros(Float64, Nx, Ny)
    zeta_nu_p = zeta * nu_p

    for j in 1:Ny, i in 1:Nx
        fx_poly[i, j] = derivative_x(fields.tauxx, i, j, Nx) + derivative_y(fields.tauxy, i, j, Ny)
        fy_poly[i, j] = derivative_x(fields.tauxy, i, j, Nx) + derivative_y(fields.tauyy, i, j, Ny)
        lap_ux[i, j] = second_derivative_x(fields.ux, i, j, Nx) + second_derivative_y(fields.ux, i, j, Ny)
        lap_uy[i, j] = second_derivative_x(fields.uy, i, j, Nx) + second_derivative_y(fields.uy, i, j, Ny)
        fx_fd_bsd[i, j] = fx_poly[i, j] - zeta_nu_p * lap_ux[i, j]
        fy_fd_bsd[i, j] = fy_poly[i, j] - zeta_nu_p * lap_uy[i, j]
    end

    return fx_poly, fy_poly, fx_fd_bsd, fy_fd_bsd, lap_ux, lap_uy
end

function reduce_interior(fx, fy, refx, refy)
    Nx, Ny = size(fx)
    ref_sum = 0.0
    delta_sum = 0.0
    max_abs_delta = -Inf
    max_i = 0
    max_j = 0
    for j in 2:(Ny - 1), i in 2:(Nx - 1)
        dx = fx[i, j] - refx[i, j]
        dy = fy[i, j] - refy[i, j]
        delta2 = dx * dx + dy * dy
        ref2 = refx[i, j] * refx[i, j] + refy[i, j] * refy[i, j]
        delta = sqrt(delta2)
        delta_sum += delta2
        ref_sum += ref2
        if delta > max_abs_delta
            max_abs_delta = delta
            max_i = i
            max_j = j
        end
    end
    l2_ref = sqrt(ref_sum)
    l2_ref > 0.0 || error("interior reference L2 norm is zero")
    l2_delta = sqrt(delta_sum)
    return l2_delta / l2_ref, l2_ref, l2_delta, max_abs_delta, max_i, max_j
end

function run_audit(; N::Int)
    nu_s = 0.1
    nu_p = 0.1
    zeta = 0.75
    fields = build_fixture(N; nu_s, nu_p, zeta)
    fx_poly, fy_poly, fx_fd_bsd, fy_fd_bsd = compute_fd_fields(fields, nu_p, zeta)[1:4]

    is_solid = falses(N, N)
    fx_kinetic = zeros(Float64, N, N)
    fy_kinetic = zeros(Float64, N, N)
    compute_bsd_force_kinetic_2d!(
        fx_kinetic, fy_kinetic, fx_poly, fy_poly,
        fields.f, fields.rho, fields.ux, fields.uy, is_solid,
        zeta, nu_p, fields.s_plus, 1.0, 1.0;
        sync=true,
    )

    rel, l2_fd_bsd, l2_delta, max_abs_delta, max_i, max_j =
        reduce_interior(fx_kinetic, fy_kinetic, fx_fd_bsd, fy_fd_bsd)
    rel_baseline = reduce_interior(fx_fd_bsd, fy_fd_bsd, fx_poly, fy_poly)[1]
    status = isfinite(rel) && isfinite(rel_baseline) && rel < 1.0e-6

    return BSDAuditResult(N, rel, rel_baseline, l2_fd_bsd, l2_delta,
                          max_abs_delta, max_i, max_j, status)
end

function print_result(result::BSDAuditResult; io::IO=stdout)
    @printf(io, "== M5-B kinetic BSD synthetic audit ==\n")
    @printf(io, "N                                      : %d\n", result.N)
    @printf(io, "relative L2(F_kinetic - F_FD_BSD)     : %.16e\n", result.rel_l2_kinetic_vs_fd_bsd)
    @printf(io, "relative L2(F_FD_BSD - F_poly_FD)     : %.16e\n", result.rel_l2_fd_baseline)
    @printf(io, "F_FD_BSD interior L2 norm             : %.16e\n", result.l2_fd_bsd)
    @printf(io, "max|F_kinetic - F_FD_BSD|             : %.16e at (i, j) = (%d, %d)\n",
            result.max_abs_delta, result.max_i, result.max_j)
    @printf(io, "RESULT bsd_kinetic_interior_L2_F64=%.16e  fd_baseline_interior_L2_F64=%.16e  status=%s\n",
            result.rel_l2_kinetic_vs_fd_bsd, result.rel_l2_fd_baseline,
            result.status ? "PASS" : "FAIL")
end

function run_mode(; N::Int)
    mktempdir() do _dir
        result = run_audit(; N)
        print_result(result)
        @assert result.status "bsd kinetic self-test failed"
        @testset "M5-B kinetic BSD synthetic audit" begin
            @test result.status
            @test isfinite(result.rel_l2_kinetic_vs_fd_bsd)
            @test isfinite(result.rel_l2_fd_baseline)
            @test result.rel_l2_kinetic_vs_fd_bsd < 1.0e-6
            @test result.rel_l2_fd_baseline > 0.0
            @test 2 <= result.max_i <= result.N - 1
            @test 2 <= result.max_j <= result.N - 1
        end
        return result
    end
end

function main(args::Vector{String}=ARGS)
    try
        if isempty(args) || first(args) == "--self-test" || first(args) == "-t"
            run_mode(; N=32)
        elseif length(args) == 1 && first(args) == "--full"
            run_mode(; N=64)
        else
            error("usage: julia --project=. $(PROGRAM_FILE) [--self-test|--full]")
        end
    catch err
        println(stderr, "BSD kinetic audit failed: ", sprint(showerror, err))
        exit(1)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
