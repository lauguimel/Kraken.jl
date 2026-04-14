# ===========================================================================
# Parametric mesh generators for the curvilinear LBM v0.2 path.
#
# Two generators ship in v0.2:
#   - polar_mesh: O-grid around a point (cylinder benchmark)
#   - stretched_box_mesh: orthogonal rectangular with optional tanh stretch
#                         (Poiseuille, natural-convection cavity)
#
# Gmsh import and C/H grids are deferred to v0.3.
# ===========================================================================

"""
    polar_mesh(; cx=0, cy=0, r_inner, r_outer, n_theta, n_r,
                 r_stretch=0, FT=Float64) -> CurvilinearMesh

O-grid around a point at `(cx, cy)`, with `n_theta` cells azimuthally
(periodic in ξ) and `n_r` nodes radially from `r_inner` to `r_outer`.

`r_stretch ≥ 0` controls clustering near the inner wall. `r_stretch = 0`
gives uniform radial spacing; larger values pack nodes near `r = r_inner`
following `tanh(r_stretch · t) / tanh(r_stretch)`.

Coordinate convention: `ξ ∈ [0, 1)` maps to `θ ∈ [0, 2π)` (periodic),
`η ∈ [0, 1]` maps to `r ∈ [r_inner, r_outer]`. The inner wall sits on
`j = 1`, the far-field on `j = n_r`.
"""
function polar_mesh(; cx::Real=0.0, cy::Real=0.0,
                      r_inner::Real, r_outer::Real,
                      n_theta::Int, n_r::Int,
                      r_stretch::Real=0.0,
                      FT::Type{<:AbstractFloat}=Float64)
    r_outer > r_inner > 0 || error("polar_mesh: require 0 < r_inner < r_outer")
    n_theta ≥ 8 || error("polar_mesh: n_theta must be ≥ 8")
    n_r ≥ 4 || error("polar_mesh: n_r must be ≥ 4")
    r_stretch ≥ 0 || error("polar_mesh: r_stretch must be ≥ 0")

    cxT, cyT = FT(cx), FT(cy)
    rinT, routT = FT(r_inner), FT(r_outer)
    sT = FT(r_stretch)
    twoπ = FT(2π)

    function mapping(ξ, η)
        θ = twoπ * ξ
        if iszero(sT)
            r = rinT + (routT - rinT) * η
        else
            # Convex map η ∈ [0,1] → [0,1] that clusters near η = 0,
            # so points pack near the inner wall (r_inner).
            t = one(η) - tanh(sT * (one(η) - η)) / tanh(sT)
            r = rinT + (routT - rinT) * t
        end
        return (cxT + r * cos(θ), cyT + r * sin(θ))
    end

    return build_mesh(mapping; Nξ=n_theta, Nη=n_r,
                      periodic_ξ=true, periodic_η=false,
                      type=:polar, FT=FT)
end

"""
    stretched_box_mesh(; x_min, x_max, y_min, y_max, Nx, Ny,
                         x_stretch=0, y_stretch=0,
                         x_stretch_dir=:none, y_stretch_dir=:none,
                         FT=Float64) -> CurvilinearMesh

Rectangular `Nx × Ny` mesh in `[x_min, x_max] × [y_min, y_max]` with
optional one-dimensional `tanh` stretching along each axis.

`x_stretch_dir ∈ (:none, :left, :right, :both)`: clustering toward the
`x_min` wall, the `x_max` wall, both walls, or none. Same for
`y_stretch_dir`. `x_stretch = 0` (or `:none` direction) gives uniform
spacing.

Useful for: Poiseuille (y_stretch_dir=:both), natural-convection cavity
(x_stretch_dir=:both, y_stretch_dir=:both), boundary layers.
"""
function stretched_box_mesh(; x_min::Real, x_max::Real,
                              y_min::Real, y_max::Real,
                              Nx::Int, Ny::Int,
                              x_stretch::Real=0.0, y_stretch::Real=0.0,
                              x_stretch_dir::Symbol=:none,
                              y_stretch_dir::Symbol=:none,
                              FT::Type{<:AbstractFloat}=Float64)
    x_max > x_min || error("stretched_box_mesh: require x_max > x_min")
    y_max > y_min || error("stretched_box_mesh: require y_max > y_min")
    Nx ≥ 2 && Ny ≥ 2 || error("stretched_box_mesh: Nx, Ny must be ≥ 2")
    x_stretch_dir ∈ (:none, :left, :right, :both) ||
        error("x_stretch_dir must be :none, :left, :right, or :both")
    y_stretch_dir ∈ (:none, :left, :right, :both) ||
        error("y_stretch_dir must be :none, :left, :right, or :both")

    xminT, xmaxT = FT(x_min), FT(x_max)
    yminT, ymaxT = FT(y_min), FT(y_max)
    sxT, syT = FT(x_stretch), FT(y_stretch)

    function mapping(ξ, η)
        tx = _stretch(ξ, sxT, x_stretch_dir)
        ty = _stretch(η, syT, y_stretch_dir)
        return (xminT + (xmaxT - xminT) * tx,
                yminT + (ymaxT - yminT) * ty)
    end

    return build_mesh(mapping; Nξ=Nx, Nη=Ny,
                      periodic_ξ=false, periodic_η=false,
                      type=:stretched_box, FT=FT)
end

"""
    cartesian_mesh(; x_min, x_max, y_min, y_max, Nx, Ny, FT=Float64)

Convenience: uniform Cartesian mesh exposed as a `CurvilinearMesh`. The
metric is the identity scaled by `(Δx, Δy)`. Useful for SLBM debugging
(should reproduce standard LBM exactly) and as the default when no
`Mesh` block is present in a `.krk` config.
"""
function cartesian_mesh(; x_min::Real, x_max::Real,
                          y_min::Real, y_max::Real,
                          Nx::Int, Ny::Int,
                          FT::Type{<:AbstractFloat}=Float64)
    return stretched_box_mesh(; x_min=x_min, x_max=x_max,
                                y_min=y_min, y_max=y_max,
                                Nx=Nx, Ny=Ny, FT=FT)
end

# Internal: 1D stretching map t ∈ [0,1] → [0,1] with optional tanh
# clustering. No type constraint on (t, s) so ForwardDiff Dual numbers
# mix freely with the (Float-typed) stretch parameter.
@inline function _stretch(t, s, dir::Symbol)
    iszero(s) && return t
    dir === :none && return t
    if dir === :right
        return tanh(s * t) / tanh(s)
    elseif dir === :left
        return one(t) - tanh(s * (one(t) - t)) / tanh(s)
    elseif dir === :both
        return (tanh(s * (2 * t - one(t))) / tanh(s) + one(t)) / 2
    end
    return t
end
