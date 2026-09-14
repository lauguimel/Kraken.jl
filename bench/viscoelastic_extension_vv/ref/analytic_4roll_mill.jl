using Printf

const M44_TWO_PI = 2.0 * pi

fourroll_x(i::Integer, n::Integer) = M44_TWO_PI * (i - 1) / n

function fourroll_velocity(x::T, y::T) where {T}
    return sin(x) * cos(y), -cos(x) * sin(y)
end

function fourroll_gradient(x::T, y::T) where {T}
    return (
        dudx=cos(x) * cos(y),
        dudy=-sin(x) * sin(y),
        dvdx=sin(x) * sin(y),
        dvdy=-cos(x) * cos(y),
    )
end

function extensional_stagnation_indices(n::Integer)
    h = div(n, 2) + 1
    return ((1, 1), (h, 1), (1, h), (h, h))
end

function brief_listed_stagnation_points()
    return ((pi / 2, pi / 2), (3pi / 2, pi / 2), (pi / 2, 3pi / 2), (3pi / 2, 3pi / 2))
end

function actual_extensional_stagnation_points()
    return ((0.0, 0.0), (pi, 0.0), (0.0, pi), (pi, pi))
end

function analytic_cxx_minus_one_stagnation(wi::Real, t::Real)
    wi > 0 || throw(ArgumentError("wi must be positive"))
    t >= 0 || throw(ArgumentError("t must be non-negative"))
    rate = inv(wi) - 2.0
    if abs(rate) < 1.0e-12
        return 2.0 * t
    elseif rate > 0
        steady = 2.0 / rate
        return steady * (1.0 - exp(-rate * t))
    else
        growth = -rate
        return 2.0 / growth * (exp(growth * t) - 1.0)
    end
end

analytic_psixx_stagnation(wi::Real, t::Real) =
    log1p(analytic_cxx_minus_one_stagnation(wi, t))

brief_steady_cxx_minus_one(wi::Real) = 2.0 * wi / (1.0 - 2.0 * wi)

function m44_case_tag(wi::Real, n::Integer, beta::Real)
    fmt(x) = replace(@sprintf("%.2f", x), "." => "p")
    return "wi$(fmt(wi))_n$(n)_beta$(fmt(beta))"
end
