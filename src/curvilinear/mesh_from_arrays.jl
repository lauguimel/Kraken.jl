using Interpolations

# ===========================================================================
# Build a CurvilinearMesh (2D/3D) from node-coordinate arrays supplied by an
# external mesh generator (gmsh Transfinite, OpenFOAM blockMesh, VTK .vts, …).
#
# Strategy: fit a cubic B-spline to the discrete node positions so that the
# mapping (ξ, η[, ζ]) → (X, Y[, Z]) becomes a differentiable Julia function.
# Then the existing `build_mesh` / `build_mesh_3d` machinery (which evaluates
# the Jacobian via ForwardDiff) runs unchanged.
#
# Why cubic B-splines and not finite differences:
# - FD centred (order 2) gives O(Δξ²) ≈ 1e-4 errors on N=100, which pollute
#   every downstream SLBM departure and break reverse-mode AD for shape
#   derivatives (the errors accumulate through 10⁴ timesteps).
# - Cubic B-splines interpolate exactly on the nodes, C² between them, with
#   O(Δξ⁴) error on the off-node evaluation of the mapping. Combined with
#   ForwardDiff on the spline, the metric at each node is exact within the
#   interpolation accuracy of the spline at that node (i.e. very small).
# - The same path transparently supports Enzyme reverse-mode differentiation
#   through the spline w.r.t. the node positions, which unlocks shape
#   derivatives on gmsh-parameterised geometries.
# ===========================================================================

"""
    CurvilinearMesh(X::AbstractMatrix, Y::AbstractMatrix;
                    periodic_ξ=false, periodic_η=false,
                    type=:imported,
                    dx_ref=nothing, FT=Float64) -> CurvilinearMesh

Build a 2D `CurvilinearMesh` from node-coordinate arrays `X[i, j], Y[i, j]`
produced by an external mesh generator. The node ordering must correspond
to the logical grid directions ξ (first index) and η (second index); the
caller is responsible for reshaping if the source mesh iterates differently.

Cubic B-splines are fitted over the logical coordinates `(ξ, η) ∈ [0, 1]²`
(or `[0, 1)` on a periodic axis) and the metric tensor + Jacobian are
computed from the splines via ForwardDiff — i.e. the same path as the
analytic `build_mesh(mapping; …)`, so no accuracy is lost at the nodes.
"""
function CurvilinearMesh(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real};
                         periodic_ξ::Bool=false, periodic_η::Bool=false,
                         type::Symbol=:imported,
                         dx_ref::Union{Real, Nothing}=nothing,
                         skip_validate::Bool=false,
                         FT::Type{<:AbstractFloat}=Float64)
    size(X) == size(Y) || error("CurvilinearMesh: X and Y must share the same size, got $(size(X)) and $(size(Y))")
    Nξ, Nη = size(X)
    # Cubic B-spline with natural ("Line") boundary condition on non-periodic
    # axes. Interpolations.jl supports Periodic() BCs directly on the axis
    # when the last-index sample is NOT a duplicate of the first.
    bc_ξ = periodic_ξ ? Periodic(OnGrid()) : Line(OnGrid())
    bc_η = periodic_η ? Periodic(OnGrid()) : Line(OnGrid())
    itp_X = interpolate(convert(Matrix{FT}, X), (BSpline(Cubic(bc_ξ)), BSpline(Cubic(bc_η))))
    itp_Y = interpolate(convert(Matrix{FT}, Y), (BSpline(Cubic(bc_ξ)), BSpline(Cubic(bc_η))))
    # Map ξ, η ∈ [0, 1] (or [0, 1) periodic) to the spline index range.
    # Periodic → Nξ samples cover a full period, index 1:Nξ spans [0, 1).
    # Non-periodic → Nξ samples cover [0, 1], index 1:Nξ spans [0, 1].
    denom_ξ = FT(periodic_ξ ? Nξ : Nξ - 1)
    denom_η = FT(periodic_η ? Nη : Nη - 1)
    mapping = (ξ, η) -> begin
        ti = ξ * denom_ξ + one(ξ)       # ∈ [1, Nξ+1) periodic  or  [1, Nξ] non-periodic
        tj = η * denom_η + one(η)
        return (itp_X(ti, tj), itp_Y(ti, tj))
    end
    return build_mesh(mapping; Nξ=Nξ, Nη=Nη,
                       periodic_ξ=periodic_ξ, periodic_η=periodic_η,
                       type=type, dx_ref=dx_ref,
                       skip_validate=skip_validate, FT=FT)
end

"""
    CurvilinearMesh3D(X, Y, Z; periodic_ξ=false, periodic_η=false, periodic_ζ=false,
                     type=:imported, dx_ref=nothing, FT=Float64) -> CurvilinearMesh3D

3D counterpart. Same BSpline-cubic + ForwardDiff strategy; `X, Y, Z` are
`Nξ × Nη × Nζ` node coordinate arrays in logical order.
"""
function CurvilinearMesh3D(X::AbstractArray{<:Real, 3},
                            Y::AbstractArray{<:Real, 3},
                            Z::AbstractArray{<:Real, 3};
                            periodic_ξ::Bool=false, periodic_η::Bool=false,
                            periodic_ζ::Bool=false,
                            type::Symbol=:imported,
                            dx_ref::Union{Real, Nothing}=nothing,
                            FT::Type{<:AbstractFloat}=Float64)
    (size(X) == size(Y) == size(Z)) ||
        error("CurvilinearMesh3D: X, Y, Z must share the same size")
    Nξ, Nη, Nζ = size(X)
    bc_ξ = periodic_ξ ? Periodic(OnGrid()) : Line(OnGrid())
    bc_η = periodic_η ? Periodic(OnGrid()) : Line(OnGrid())
    bc_ζ = periodic_ζ ? Periodic(OnGrid()) : Line(OnGrid())
    itp_X = interpolate(convert(Array{FT, 3}, X),
                         (BSpline(Cubic(bc_ξ)), BSpline(Cubic(bc_η)), BSpline(Cubic(bc_ζ))))
    itp_Y = interpolate(convert(Array{FT, 3}, Y),
                         (BSpline(Cubic(bc_ξ)), BSpline(Cubic(bc_η)), BSpline(Cubic(bc_ζ))))
    itp_Z = interpolate(convert(Array{FT, 3}, Z),
                         (BSpline(Cubic(bc_ξ)), BSpline(Cubic(bc_η)), BSpline(Cubic(bc_ζ))))
    denom_ξ = FT(periodic_ξ ? Nξ : Nξ - 1)
    denom_η = FT(periodic_η ? Nη : Nη - 1)
    denom_ζ = FT(periodic_ζ ? Nζ : Nζ - 1)
    mapping = (ξ, η, ζ) -> begin
        ti = ξ * denom_ξ + one(ξ)
        tj = η * denom_η + one(η)
        tk = ζ * denom_ζ + one(ζ)
        return (itp_X(ti, tj, tk), itp_Y(ti, tj, tk), itp_Z(ti, tj, tk))
    end
    return build_mesh_3d(mapping; Nξ=Nξ, Nη=Nη, Nζ=Nζ,
                          periodic_ξ=periodic_ξ, periodic_η=periodic_η,
                          periodic_ζ=periodic_ζ,
                          type=type, dx_ref=dx_ref, FT=FT)
end
