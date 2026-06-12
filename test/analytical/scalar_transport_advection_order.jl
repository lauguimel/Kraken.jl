# Analytical validation of scalar advection order.
#
# Manufactured 1D-in-x steady advection-diffusion on a 2D strip:
#   u dT/dx - DT d2T/dx2 = S(x)
# with T(x) = 1 + 0.3 sin(pi*x)^4. The zero boundary slope keeps the existing
# boundary fallback from dominating the interior error, while Pe_cell remains
# advection dominated on every grid below.

using Test

if !isdefined(@__MODULE__, :solve_scalar_transport)
    include(joinpath(@__DIR__, "..", "..", "src", "methods",
                     "scalar_transport", "thermal_transport.jl"))
end

const SCALAR_ADVECTION_ORDER_RESULTS = Dict{Symbol,Any}()

function manufactured_advection_case(nx::Integer; ny::Integer = 8,
                                     U::Real = 1.0, DT::Real = 1e-3,
                                     advection::Symbol = :linear_upwind,
                                     deferred_passes::Integer = 4)
    nx = Int(nx); ny = Int(ny)
    Lx = 1.0
    Ly = 1.0
    dx = Lx / nx
    dy = Ly / ny
    xcenters = [(i - 0.5) * dx for i in 1:nx]

    exact(x) = 1.0 + 0.3 * sin(pi * x)^4
    d_exact(x) = 1.2 * pi * sin(pi * x)^3 * cos(pi * x)
    dd_exact(x) = 1.2 * pi^2 * (3.0 * sin(pi * x)^2 * cos(pi * x)^2 -
                                 sin(pi * x)^4)

    uf = fill(Float64(U), nx, ny)
    vf = zeros(Float64, nx, ny)
    source = zeros(Float64, nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        x = xcenters[i]
        source[i, j] = Float64(U) * d_exact(x) - Float64(DT) * dd_exact(x)
    end

    bc = (west = (kind = :dirichlet, value = exact(0.0)),
          east = (kind = :dirichlet, value = exact(Lx)),
          south = (kind = :flux, value = 0.0),
          north = (kind = :flux, value = 0.0))

    res = solve_scalar_transport(; nx, ny, dx, dy, uf, vf, DT, bc, source,
                                 advection, deferred_passes)

    # Measure away from the two fallback bands so the test isolates the interior
    # advection stencil. The profile is y-independent.
    i0 = round(Int, 0.15 * nx)
    i1 = round(Int, 0.85 * nx)
    err2 = 0.0
    nerr = 0
    @inbounds for j in 1:ny, i in i0:i1
        err2 += (res.T[i, j] - exact(xcenters[i]))^2
        nerr += 1
    end
    l2 = sqrt(err2 / nerr)
    return (; res, l2, nx, ny)
end

function observed_order(advection::Symbol)
    coarse = manufactured_advection_case(40; advection)
    medium = manufactured_advection_case(80; advection)
    fine = manufactured_advection_case(160; advection)
    order_40_80 = log2(coarse.l2 / medium.l2)
    order_80_160 = log2(medium.l2 / fine.l2)
    return (; coarse, medium, fine, order_40_80, order_80_160)
end

@testset "Scalar transport: advection order" begin
    up = observed_order(:upwind)
    lu = observed_order(:linear_upwind)
    SCALAR_ADVECTION_ORDER_RESULTS[:upwind] = up
    SCALAR_ADVECTION_ORDER_RESULTS[:linear_upwind] = lu

    @test up.coarse.res.iters == 1
    @test up.fine.res.Pe_cell > 5.0
    @test 0.85 <= up.order_80_160 <= 1.15

    @test lu.coarse.res.iters <= 5
    @test lu.fine.res.Pe_cell > 5.0
    @test lu.order_40_80 >= 1.7
    @test lu.order_80_160 >= 1.7
end

if (abspath(PROGRAM_FILE) == (@__FILE__)) || (get(ENV, "ST_REPORT", "0") == "1")
    up = get(SCALAR_ADVECTION_ORDER_RESULTS, :upwind, nothing)
    lu = get(SCALAR_ADVECTION_ORDER_RESULTS, :linear_upwind, nothing)
    if up !== nothing && lu !== nothing
        @info "scalar advection order" upwind_40_80=up.order_40_80 upwind_80_160=up.order_80_160 linear_40_80=lu.order_40_80 linear_80_160=lu.order_80_160 Pe=lu.fine.res.Pe_cell
    end
end
