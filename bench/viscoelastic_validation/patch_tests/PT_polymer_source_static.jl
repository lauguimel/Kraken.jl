#!/usr/bin/env julia

using Printf
using Kraken

const OUTDIR = joinpath("scratch", "M56_vv_ladder")
const CSV = joinpath(OUTDIR, "PT_polymer_source_static.csv")

function analytic_gradient(i, j)
    x = Float64(i - 1)
    y = Float64(j - 1)
    dudx = 0.011 + 0.0007 * y
    dudy = -0.017 + 0.0007 * x + 0.0002 * y
    dvdx = 0.023 - 0.0003 * y + 0.0004 * x
    dvdy = -0.005 - 0.0003 * x
    return dudx, dudy, dvdx, dvdy
end

function run_l2()
    mkpath(OUTDIR)
    Nx = Ny = 32
    cx = cy = 16.5
    R = 4.0
    _q_wall, is_solid = Kraken.precompute_q_wall_cylinder(Nx, Ny, cx, cy, R; FT=Float64)
    lambda = 7.0
    max_err = 0.0
    worst = (i=0, j=0, component="none", computed=0.0, expected=0.0)
    nfluid = 0

    open(CSV, "w") do io
        println(io, "i,j,component,computed,expected,abs_err")
        for j in 1:Ny, i in 1:Nx
            is_solid[i, j] && continue
            nfluid += 1
            dudx, dudy, dvdx, dvdy = analytic_gradient(i, j)
            got = Kraken.logfv_oldroydb_source_c_2d(
                1.0, 0.0, 1.0, dudx, dudy, dvdx, dvdy, lambda,
            )
            expected = (2.0 * dudx, dudy + dvdx, 2.0 * dvdy)
            for (name, g, e) in zip(("xx", "xy", "yy"), got, expected)
                err = abs(g - e)
                @printf(io, "%d,%d,%s,%.17g,%.17g,%.17g\n", i, j, name, g, e, err)
                if err > max_err
                    max_err = err
                    worst = (i=i, j=j, component=name, computed=g, expected=e)
                end
            end
        end
    end

    pass = max_err < 1.0e-12
    @printf("M56_L2 verdict=%s max_abs_error=%.17g fluid_cells=%d csv=%s\n",
            pass ? "PASS" : "FAIL", max_err, nfluid, CSV)
    @printf("M56_L2 worst cell=(%d,%d) component=%s computed=%.17g expected=%.17g\n",
            worst.i, worst.j, worst.component, worst.computed, worst.expected)
    return pass ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(run_l2())
end
