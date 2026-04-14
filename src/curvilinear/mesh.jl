using ForwardDiff

# ===========================================================================
# Curvilinear (body-fitted) mesh infrastructure for v0.2 SLBM path.
#
# A CurvilinearMesh is a logically structured Nξ × Nη grid in computational
# coordinates (ξ, η) with a parametric mapping to physical coordinates
# (X, Y). Metric derivatives ∂X/∂ξ, ∂X/∂η, ∂Y/∂ξ, ∂Y/∂η and the Jacobian
# determinant J are computed once at build time via ForwardDiff on the
# user-supplied mapping function.
#
# References:
# - Krämer, Küllmer, Reith, Foysi, Steiner (2017) Phys. Rev. E 95, 023305
# - Wilde, Krämer, Reith, Foysi (2020) Comput. Fluids 204, 104519
# ===========================================================================

"""
    CurvilinearMesh{T, AT}

Logically structured `Nξ × Nη` mesh in computational space with a user-
supplied mapping to physical space. All metric arrays are precomputed
host-side; transfer to device happens at simulation start.

# Fields
- `type`: discriminator (`:polar`, `:stretched_box`, `:cartesian`, `:custom`)
- `Nξ, Nη`: logical grid extents
- `periodic_ξ, periodic_η`: wrap-around flags (true → ξ ∈ [0,1), false → [0,1])
- `X, Y`: physical node coordinates `[Nξ, Nη]`
- `dXdξ, dXdη, dYdξ, dYdη`: metric derivatives `[Nξ, Nη]`
- `J`: Jacobian determinant `J = ∂X/∂ξ · ∂Y/∂η − ∂X/∂η · ∂Y/∂ξ`

# Conventions
- For `:polar`: ξ is angular (periodic, n_θ samples in [0, 2π)),
  η is radial (non-periodic, n_r samples in [r_inner, r_outer]).
- For `:stretched_box`: ξ is x-like, η is y-like.
- Streaming runs on the logical (ξ, η) grid; physics quantities are
  carried in physical units.
"""
struct CurvilinearMesh{T<:AbstractFloat, AT<:AbstractMatrix{T}}
    type::Symbol
    Nξ::Int
    Nη::Int
    periodic_ξ::Bool
    periodic_η::Bool
    X::AT
    Y::AT
    dXdξ::AT
    dXdη::AT
    dYdξ::AT
    dYdη::AT
    J::AT
    # Physical distance that corresponds to one lattice unit. Used by
    # the semi-Lagrangian streaming step: departure point in physical
    # space is P − c_q · dx_ref. For a uniform Cartesian mesh with
    # isotropic spacing this equals the grid spacing, so SLBM reduces
    # to standard LBM (integer departures, zero interpolation error).
    dx_ref::T
end

"""
    compute_metric(mapping, ξ::T, η::T) -> (dXdξ, dXdη, dYdξ, dYdη, J)

Forward-mode AD evaluation of the mapping derivatives at one point.
The mapping must be a Julia function `(ξ, η) -> (X, Y)` returning a
`Tuple` or `AbstractVector` of length 2, written in pure Julia so that
`ForwardDiff.Dual` numbers propagate through it.
"""
function compute_metric(mapping, ξ::T, η::T) where {T<:AbstractFloat}
    f = p -> begin
        xy = mapping(p[1], p[2])
        return [xy[1], xy[2]]
    end
    Jmat = ForwardDiff.jacobian(f, T[ξ, η])
    dXdξ = Jmat[1, 1]
    dXdη = Jmat[1, 2]
    dYdξ = Jmat[2, 1]
    dYdη = Jmat[2, 2]
    detJ = dXdξ * dYdη - dXdη * dYdξ
    return dXdξ, dXdη, dYdξ, dYdη, detJ
end

"""
    build_mesh(mapping; Nξ, Nη, periodic_ξ=false, periodic_η=false,
               type=:custom, FT=Float64) -> CurvilinearMesh

Sample the mapping on the logical `Nξ × Nη` grid, computing physical
coordinates and metric derivatives at every node via ForwardDiff.

The computational coordinate `ξ` ranges over `[0, 1]` if `periodic_ξ` is
false, or `[0, 1)` (last sample omitted) if periodic. Same for `η`.

Validates: `J > 0` everywhere (asserts no mesh fold), warns on extreme
aspect ratios or near-degenerate orthogonality.
"""
function build_mesh(mapping;
                    Nξ::Int, Nη::Int,
                    periodic_ξ::Bool=false, periodic_η::Bool=false,
                    type::Symbol=:custom,
                    dx_ref::Union{Real, Nothing}=nothing,
                    FT::Type{<:AbstractFloat}=Float64)
    X = zeros(FT, Nξ, Nη)
    Y = zeros(FT, Nξ, Nη)
    dXdξ = zeros(FT, Nξ, Nη)
    dXdη = zeros(FT, Nξ, Nη)
    dYdξ = zeros(FT, Nξ, Nη)
    dYdη = zeros(FT, Nξ, Nη)
    J = zeros(FT, Nξ, Nη)

    denom_ξ = FT(periodic_ξ ? Nξ : Nξ - 1)
    denom_η = FT(periodic_η ? Nη : Nη - 1)

    for j in 1:Nη, i in 1:Nξ
        ξ = FT(i - 1) / denom_ξ
        η = FT(j - 1) / denom_η
        xy = mapping(ξ, η)
        X[i, j] = xy[1]
        Y[i, j] = xy[2]
        dXdξ_, dXdη_, dYdξ_, dYdη_, detJ = compute_metric(mapping, ξ, η)
        dXdξ[i, j] = dXdξ_
        dXdη[i, j] = dXdη_
        dYdξ[i, j] = dYdξ_
        dYdη[i, j] = dYdη_
        J[i, j] = detJ
    end

    dxr = dx_ref === nothing ? _default_dx_ref(J, Nξ, Nη, periodic_ξ, periodic_η) :
                                FT(dx_ref)

    mesh = CurvilinearMesh{FT, Matrix{FT}}(type, Nξ, Nη, periodic_ξ, periodic_η,
                                            X, Y, dXdξ, dXdη, dYdξ, dYdη, J, dxr)
    validate_mesh(mesh)
    return mesh
end

# Default physical reference spacing. Definition: average physical cell
# area is mean(|J|) · Δξ · Δη, so the geometric-mean cell edge is
# sqrt(mean(|J|) / (Nξ_eff · Nη_eff)). For a uniform Cartesian mesh
# this reduces to sqrt(Δx · Δy), which equals the isotropic grid
# spacing — so SLBM on a uniform mesh reproduces standard LBM exactly.
function _default_dx_ref(J::AbstractMatrix{T}, Nξ, Nη,
                         periodic_ξ, periodic_η) where {T}
    Nξ_eff = periodic_ξ ? Nξ : Nξ - 1
    Nη_eff = periodic_η ? Nη : Nη - 1
    # mean(|J|) avoids sign issues for left-handed mappings (polar).
    total_area = zero(T)
    @inbounds for k in eachindex(J)
        total_area += abs(J[k])
    end
    mean_abs_J = total_area / T(length(J))
    return sqrt(mean_abs_J / (T(Nξ_eff) * T(Nη_eff)))
end

"""
    validate_mesh(mesh; min_orthog_deg=45.0, max_aspect=100.0)

Assert positive Jacobian everywhere; warn on poor orthogonality or
extreme aspect ratio. Pitfalls documented in Mei & Shyy 1998 §3,
Imamura 2005 Appendix A, Budinski 2014 §4.
"""
function validate_mesh(mesh::CurvilinearMesh{T};
                       min_orthog_deg::Real=45.0,
                       max_aspect::Real=100.0) where {T}
    # Fold detection: J must be strictly non-zero and have a consistent
    # sign across the entire mesh. Sign itself is irrelevant — it only
    # encodes whether the (ξ, η) basis is right-handed in physical space.
    Jmin = minimum(mesh.J)
    Jmax = maximum(mesh.J)
    Jabs_min = min(abs(Jmin), abs(Jmax))
    if Jabs_min == zero(T) || (Jmin < zero(T) && Jmax > zero(T))
        i, j = Tuple(argmin(abs.(mesh.J)))
        error("CurvilinearMesh: degenerate Jacobian (sign change or zero) " *
              "at (i=$i, j=$j): min|J|=$(Jabs_min), Jmin=$(Jmin), Jmax=$(Jmax). " *
              "Mesh has a fold or degenerate cell.")
    end

    cos_min = one(T)
    aspect_max = zero(T)
    @inbounds for j in 1:mesh.Nη, i in 1:mesh.Nξ
        ξ_len = sqrt(mesh.dXdξ[i, j]^2 + mesh.dYdξ[i, j]^2)
        η_len = sqrt(mesh.dXdη[i, j]^2 + mesh.dYdη[i, j]^2)
        if ξ_len > zero(T) && η_len > zero(T)
            dot = (mesh.dXdξ[i, j] * mesh.dXdη[i, j] + mesh.dYdξ[i, j] * mesh.dYdη[i, j]) /
                  (ξ_len * η_len)
            cos_min = min(cos_min, abs(dot))
            ar = max(ξ_len, η_len) / min(ξ_len, η_len)
            aspect_max = max(aspect_max, ar)
        end
    end

    angle_min_deg = rad2deg(acos(min(cos_min, one(T))))
    if angle_min_deg < T(min_orthog_deg)
        @warn "CurvilinearMesh: minimum orthogonality angle $(angle_min_deg)° < $(min_orthog_deg)° " *
              "(Budinski 2014 reports >30% Cd error below 60°)."
    end
    if aspect_max > T(max_aspect)
        @warn "CurvilinearMesh: maximum cell aspect ratio $(aspect_max) > $(max_aspect) " *
              "(Dorschner 2016 warns that regularised collision degrades above this)."
    end
    return nothing
end

"""
    cell_area(mesh, i, j)

Physical area of the cell at logical index `(i, j)` (≈ `J[i,j]` times
the computational cell size).
"""
@inline function cell_area(mesh::CurvilinearMesh{T}, i::Int, j::Int) where {T}
    dξ = one(T) / T(mesh.periodic_ξ ? mesh.Nξ : mesh.Nξ - 1)
    dη = one(T) / T(mesh.periodic_η ? mesh.Nη : mesh.Nη - 1)
    return abs(mesh.J[i, j]) * dξ * dη
end

"""
    domain_extent(mesh) -> (xmin, xmax, ymin, ymax)

Bounding box of the physical mesh nodes.
"""
function domain_extent(mesh::CurvilinearMesh)
    return extrema(mesh.X)..., extrema(mesh.Y)...
end
