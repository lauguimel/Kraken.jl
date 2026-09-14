#!/usr/bin/env julia

using Printf
using Statistics
using Kraken

const OUTDIR = joinpath(@__DIR__, "..", "..", "..", "scratch", "M49_canary")
const CSV = joinpath(OUTDIR, "wall_stencil_canary.csv")

struct CaseSpec
    name::String
    f::Function
    df::Function
    exact_poly::Bool
end

const CASES = CaseSpec[
    CaseSpec("P1", y -> y, y -> 1.0, true),
    CaseSpec("P2", y -> y^2, y -> 2.0y, true),
    CaseSpec("P3", y -> y + 2.0y^2, y -> 1.0 + 4.0y, true),
    CaseSpec("P4", y -> sin(0.3y), y -> 0.3cos(0.3y), false),
]

analytic_at(case::CaseSpec, wall::Symbol, L::Int) =
    wall in (:south, :west) ? case.df(0.0) : case.df(Float64(L))

pass_flag(case::CaseSpec, mode::Symbol, err::Float64) =
    case.exact_poly && mode == :halfway_quadratic ? (err < 1e-10 ? "PASS" : "FAIL") : "INFO"

function y_case(case::CaseSpec, mode::Symbol)
    Nx, Ny = 16, 8
    dx = dy = 1.0
    ux = zeros(Float64, Nx, Ny)
    solid = falses(Nx, Ny)
    bc = Kraken.fvfd_periodicx_wally_bcspec_2d()
    for j in 1:Ny, i in 1:Nx
        ux[i, j] = case.f(j - 0.5)
    end
    inv_dy, inv_2dy = inv(dy), inv(2dy)
    rows = NamedTuple[]
    for (wall, j) in ((:south, 1), (:north, Ny))
        vals = [
            Kraken._fvfd_solid_bc_derivative_y_2d(
                ux, solid, i, j, Ny, inv_dy, inv_2dy, bc.south, bc.north, Val(mode),
            ) for i in 1:Nx
        ]
        computed = mean(vals)
        analytic = analytic_at(case, wall, Ny)
        err = abs(computed - analytic)
        raw_mode = mode === :quadratic ? :quadratic_raw : :linear_raw
        push!(rows, (case=case.name, direction="y", wall_row=String(wall),
                    stencil_mode=raw_mode, computed=computed, analytic=analytic,
                    abs_error=err, rel_error=err / max(1e-12, abs(analytic)),
                    pass="INFO", stddev=std(vals)))
    end
    return rows
end

function x_case(case::CaseSpec, mode::Symbol)
    Nx, Ny = 8, 16
    dx = dy = 1.0
    ux = zeros(Float64, Nx, Ny)
    solid = falses(Nx, Ny)
    bc = Kraken.fvfd_wallxwally_bcspec_2d()
    for j in 1:Ny, i in 1:Nx
        ux[i, j] = case.f(i - 0.5)
    end
    inv_dx, inv_2dx = inv(dx), inv(2dx)
    rows = NamedTuple[]
    for (wall, i) in ((:west, 1), (:east, Nx))
        vals = [
            Kraken._fvfd_solid_bc_derivative_x_2d(
                ux, solid, i, j, Nx, inv_dx, inv_2dx, bc.west, bc.east, Val(mode),
            ) for j in 1:Ny
        ]
        computed = mean(vals)
        analytic = analytic_at(case, wall, Nx)
        err = abs(computed - analytic)
        raw_mode = mode === :quadratic ? :quadratic_raw : :linear_raw
        push!(rows, (case=case.name, direction="x", wall_row=String(wall),
                    stencil_mode=raw_mode, computed=computed, analytic=analytic,
                    abs_error=err, rel_error=err / max(1e-12, abs(analytic)),
                    pass="INFO", stddev=std(vals)))
    end
    return rows
end

function helper_y_case(case::CaseSpec, mode::Symbol)
    Nx, Ny = 16, 8
    dx = dy = 1.0
    ux = zeros(Float64, Nx, Ny)
    uy = zeros(Float64, Nx, Ny)
    is_solid = falses(Nx, Ny)
    dudx = zeros(Float64, Nx, Ny)
    dudy = zeros(Float64, Nx, Ny)
    dvdx = zeros(Float64, Nx, Ny)
    dvdy = zeros(Float64, Nx, Ny)
    bc = Kraken.fvfd_periodicx_wally_bcspec_2d()
    for j in 1:Ny, i in 1:Nx
        ux[i, j] = case.f(j - 0.5)
    end
    south = fill(case.f(0.0), Nx)
    north = fill(case.f(Float64(Ny)), Nx)
    sides = Kraken.WallGradientSides(south, north, nothing, nothing)
    Kraken.fvfd_velocity_gradient_2d!(dudx, dudy, dvdx, dvdy, ux, uy, is_solid, dx, dy, bc)
    Kraken.apply_halfway_wall_gradient_correction!(
        dudx, dudy, dvdx, dvdy, ux, uy, sides, dx, dy; order=mode,
    )
    rows = NamedTuple[]
    helper_mode = mode === :quadratic ? :halfway_quadratic : :halfway_linear
    for (wall, j) in ((:south, 1), (:north, Ny))
        vals = [dudy[i, j] for i in 1:Nx]
        computed = mean(vals)
        analytic = analytic_at(case, wall, Ny)
        err = abs(computed - analytic)
        push!(rows, (case=case.name, direction="y", wall_row=String(wall),
                    stencil_mode=helper_mode, computed=computed, analytic=analytic,
                    abs_error=err, rel_error=err / max(1e-12, abs(analytic)),
                    pass=pass_flag(case, helper_mode, err), stddev=std(vals)))
    end
    return rows
end

function helper_x_case(case::CaseSpec, mode::Symbol)
    Nx, Ny = 8, 16
    dx = dy = 1.0
    ux = zeros(Float64, Nx, Ny)
    uy = zeros(Float64, Nx, Ny)
    is_solid = falses(Nx, Ny)
    dudx = zeros(Float64, Nx, Ny)
    dudy = zeros(Float64, Nx, Ny)
    dvdx = zeros(Float64, Nx, Ny)
    dvdy = zeros(Float64, Nx, Ny)
    bc = Kraken.fvfd_wallxwally_bcspec_2d()
    for j in 1:Ny, i in 1:Nx
        uy[i, j] = case.f(i - 0.5)
    end
    west = fill(case.f(0.0), Ny)
    east = fill(case.f(Float64(Nx)), Ny)
    sides = Kraken.WallGradientSides(nothing, nothing, east, west)
    Kraken.fvfd_velocity_gradient_2d!(dudx, dudy, dvdx, dvdy, ux, uy, is_solid, dx, dy, bc)
    Kraken.apply_halfway_wall_gradient_correction!(
        dudx, dudy, dvdx, dvdy, ux, uy, sides, dx, dy; order=mode,
    )
    rows = NamedTuple[]
    helper_mode = mode === :quadratic ? :halfway_quadratic : :halfway_linear
    for (wall, i) in ((:west, 1), (:east, Nx))
        vals = [dvdx[i, j] for j in 1:Ny]
        computed = mean(vals)
        analytic = analytic_at(case, wall, Nx)
        err = abs(computed - analytic)
        push!(rows, (case=case.name, direction="x", wall_row=String(wall),
                    stencil_mode=helper_mode, computed=computed, analytic=analytic,
                    abs_error=err, rel_error=err / max(1e-12, abs(analytic)),
                    pass=pass_flag(case, helper_mode, err), stddev=std(vals)))
    end
    return rows
end

function write_csv(rows)
    mkpath(OUTDIR)
    open(CSV, "w") do io
        println(io, "case,direction,wall_row,stencil_mode,computed,analytic,abs_error,rel_error,pass")
        for r in rows
            @printf(io, "%s,%s,%s,%s,%.17g,%.17g,%.17g,%.17g,%s\n",
                    r.case, r.direction, r.wall_row, repr(r.stencil_mode),
                    r.computed, r.analytic, r.abs_error, r.rel_error, r.pass)
        end
    end
end

function main()
    t0 = time()
    rows = NamedTuple[]
    for case in CASES, mode in (:quadratic, :linear)
        append!(rows, y_case(case, mode))
        append!(rows, x_case(case, mode))
        append!(rows, helper_y_case(case, mode))
        append!(rows, helper_x_case(case, mode))
    end
    helper_quad_polys = filter(
        r -> r.stencil_mode == :halfway_quadratic && r.case in ("P1", "P2", "P3"),
        rows,
    )
    @assert all(r.abs_error < 1e-10 for r in helper_quad_polys) "M49 helper-quadratic FAIL"
    write_csv(rows)
    wall = time() - t0
    fails = filter(r -> r.pass == "FAIL", rows)
    verdict = isempty(fails) ? "PASS" : "FAIL"
    qP1 = maximum(r.abs_error for r in rows if r.case == "P1" && r.stencil_mode == :quadratic_raw)
    qP2 = maximum(r.abs_error for r in rows if r.case == "P2" && r.stencil_mode == :quadratic_raw)
    lP1 = maximum(r.abs_error for r in rows if r.case == "P1" && r.stencil_mode == :linear_raw)
    hqmax = maximum(r.abs_error for r in helper_quad_polys)
    @printf("[M49-WALL-STENCIL-CANARY] csv at %s\n", CSV)
    @printf("- TL;DR: %s\n- Quadratic raw P1 abs_err: %.17g\n- Quadratic raw P2 abs_err: %.17g\n- Linear raw P1 abs_err: %.17g\n- Helper quadratic P1-P3 max abs_error < 1e-10: %.17g\n- Wall time: %.6f s\n",
            verdict, qP1, qP2, lP1, hqmax, wall)
    return verdict == "PASS" ? 0 : 1
end

exit(main())
