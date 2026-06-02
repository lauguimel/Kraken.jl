# =====================================================================
# Block mesh extension for the multi-block SLBM pipeline (Phase B.2.3).
#
# `BlockState2D` already allocates the populations on an extended grid
# of size `(Nξ + 2·Ng, Nη + 2·Ng, 9)` so the ghost rows can carry the
# neighbour's data or a halfway-BB reflection. For the SLBM step
# kernel to run on this extended grid, the `SLBMGeometry` (departure
# points + interpolation stencils + metric) must also be sized for
# `(Nξ + 2·Ng, Nη + 2·Ng)` — otherwise kernel launches crash on
# out-of-range index access.
#
# `extend_mesh_2d(mesh; n_ghost)` produces a new `CurvilinearMesh` of
# the extended size by LINEAR EXTRAPOLATION of the nearest node rows
# and columns. For smooth meshes this preserves the cell spacing at
# the block edges; corners are filled by two-pass extrapolation
# (ξ then η). The returned mesh is passed through the standard
# `CurvilinearMesh(X, Y; …)` constructor so the spline metric is
# recomputed consistently on the extended grid.
#
# `build_block_slbm_geometry_extended(block; n_ghost, local_cfl)` is a
# thin wrapper that:
#   1. builds the extended mesh
#   2. calls `build_slbm_geometry(extended_mesh; local_cfl)`
#   3. returns `(extended_mesh, slbm_geometry)`
#
# The user is responsible for also extending `is_solid`, `q_wall`,
# `uw_x`, `uw_y` arrays from `(Nξ, Nη)` to `(Nξ + 2Ng, Nη + 2Ng)`:
# those are naturally zero on the ghost layer (no solid outside the
# interior). A helper `extend_interior_field_2d!` copies an interior
# array into the interior of an extended array.
# =====================================================================

"""
    extend_mesh_2d(mesh::CurvilinearMesh; n_ghost=1) -> CurvilinearMesh

Linearly extrapolate `mesh.X, mesh.Y` by `n_ghost` rows/columns on all
four sides, returning a new `CurvilinearMesh` of size
`(mesh.Nξ + 2n_ghost, mesh.Nη + 2n_ghost)` whose interior matches the
input.

Extrapolation is piecewise-linear in the logical direction: the two
innermost interior nodes define the extrapolation slope, which is
stepped outward by `Δξ` (resp. `Δη`) per ghost layer. For a uniform
Cartesian mesh this reproduces the natural uniform extension. For a
stretched mesh the extension preserves the local cell spacing at the
boundary — cells outside grow at the same rate as the last interior
cell's spacing.

The metric (derivatives, Jacobian, `dx_ref`) is RECOMPUTED on the
extended grid by refitting cubic B-splines. Because the extrapolation
is purely linear the resulting metric remains well-defined, but the
extrapolated cells may be slightly distorted compared to an
analytically-defined mesh — safe for ghost-layer use where the kernel
writes but the exchange overwrites before the next read.

For meshes whose boundary follows a circular arc (e.g. an O-grid ring
block around a cylinder), Cartesian linear extrapolation folds the
Jacobian near the arc corners because the chord through two adjacent
interior nodes does NOT track the arc curvature. Pass
`curved_edges=(:south,)` (or any subset of `(:west, :east, :south,
:north)`) with `curved_center=(cx, cy)` to compute the ghost for that
edge via LINEAR EXTRAPOLATION IN POLAR COORDINATES `(r, θ)` relative
to `curved_center`. This produces ghost cells that ride the arc
instead of its tangent chord.
"""
function extend_mesh_2d(mesh::CurvilinearMesh{T, AT};
                         n_ghost::Int=1,
                         curved_edges::Tuple{Vararg{Symbol}}=(),
                         curved_center::Tuple{<:Real,<:Real}=(0, 0)) where {T, AT}
    n_ghost ≥ 0 || error("extend_mesh_2d: n_ghost must be ≥ 0")
    n_ghost == 0 && return mesh
    Nξ, Nη = mesh.Nξ, mesh.Nη
    Nξ_ext = Nξ + 2 * n_ghost
    Nη_ext = Nη + 2 * n_ghost
    X_ext = Matrix{T}(undef, Nξ_ext, Nη_ext)
    Y_ext = similar(X_ext)
    cx_c = T(curved_center[1]); cy_c = T(curved_center[2])

    curved_w = :west in curved_edges
    curved_e = :east in curved_edges
    curved_s = :south in curved_edges
    curved_n = :north in curved_edges

    # Helper: linear extrapolate (x, y) by k steps past ref1 using
    # (ref1 - ref2) displacement in POLAR coords about (cx_c, cy_c).
    function _polar_extrap(x1, y1, x2, y2, k::Int)
        r1 = sqrt((x1 - cx_c)^2 + (y1 - cy_c)^2)
        r2 = sqrt((x2 - cx_c)^2 + (y2 - cy_c)^2)
        θ1 = atan(y1 - cy_c, x1 - cx_c)
        θ2 = atan(y2 - cy_c, x2 - cx_c)
        Δθ = θ2 - θ1
        Δθ > π  && (Δθ -= 2π)
        Δθ < -π && (Δθ += 2π)
        r_g = r1 - k * (r2 - r1)
        θ_g = θ1 - k * Δθ
        return (cx_c + r_g * cos(θ_g), cy_c + r_g * sin(θ_g))
    end

    # Pass 1: interior (copy) + west/east ghost rows (extrapolate in ξ).
    @inbounds for j in 1:Nη
        j_ext = j + n_ghost
        # Interior
        for i in 1:Nξ
            i_ext = i + n_ghost
            X_ext[i_ext, j_ext] = mesh.X[i, j]
            Y_ext[i_ext, j_ext] = mesh.Y[i, j]
        end
        if curved_w
            for k in 1:n_ghost
                xg, yg = _polar_extrap(mesh.X[1, j], mesh.Y[1, j],
                                         mesh.X[2, j], mesh.Y[2, j], k)
                X_ext[n_ghost - k + 1, j_ext] = xg
                Y_ext[n_ghost - k + 1, j_ext] = yg
            end
        else
            dXw = mesh.X[2, j] - mesh.X[1, j]
            dYw = mesh.Y[2, j] - mesh.Y[1, j]
            for k in 1:n_ghost
                X_ext[n_ghost - k + 1, j_ext] = mesh.X[1, j] - k * dXw
                Y_ext[n_ghost - k + 1, j_ext] = mesh.Y[1, j] - k * dYw
            end
        end
        if curved_e
            for k in 1:n_ghost
                xg, yg = _polar_extrap(mesh.X[Nξ, j], mesh.Y[Nξ, j],
                                         mesh.X[Nξ - 1, j], mesh.Y[Nξ - 1, j], k)
                X_ext[Nξ + n_ghost + k, j_ext] = xg
                Y_ext[Nξ + n_ghost + k, j_ext] = yg
            end
        else
            dXe = mesh.X[Nξ, j] - mesh.X[Nξ - 1, j]
            dYe = mesh.Y[Nξ, j] - mesh.Y[Nξ - 1, j]
            for k in 1:n_ghost
                X_ext[Nξ + n_ghost + k, j_ext] = mesh.X[Nξ, j] + k * dXe
                Y_ext[Nξ + n_ghost + k, j_ext] = mesh.Y[Nξ, j] + k * dYe
            end
        end
    end

    # Pass 2: south/north ghost rows (extrapolate in η over ALL
    # columns, including the west/east ghosts filled by Pass 1).
    @inbounds for i_ext in 1:Nξ_ext
        j1 = n_ghost + 1
        j2 = n_ghost + 2
        if curved_s
            for k in 1:n_ghost
                xg, yg = _polar_extrap(X_ext[i_ext, j1], Y_ext[i_ext, j1],
                                         X_ext[i_ext, j2], Y_ext[i_ext, j2], k)
                X_ext[i_ext, n_ghost - k + 1] = xg
                Y_ext[i_ext, n_ghost - k + 1] = yg
            end
        else
            dXs = X_ext[i_ext, j2] - X_ext[i_ext, j1]
            dYs = Y_ext[i_ext, j2] - Y_ext[i_ext, j1]
            for k in 1:n_ghost
                X_ext[i_ext, n_ghost - k + 1] = X_ext[i_ext, j1] - k * dXs
                Y_ext[i_ext, n_ghost - k + 1] = Y_ext[i_ext, j1] - k * dYs
            end
        end
        jN  = Nη + n_ghost
        jNm = jN - 1
        if curved_n
            for k in 1:n_ghost
                xg, yg = _polar_extrap(X_ext[i_ext, jN], Y_ext[i_ext, jN],
                                         X_ext[i_ext, jNm], Y_ext[i_ext, jNm], k)
                X_ext[i_ext, jN + k] = xg
                Y_ext[i_ext, jN + k] = yg
            end
        else
            dXn = X_ext[i_ext, jN] - X_ext[i_ext, jNm]
            dYn = Y_ext[i_ext, jN] - Y_ext[i_ext, jNm]
            for k in 1:n_ghost
                X_ext[i_ext, jN + k] = X_ext[i_ext, jN] + k * dXn
                Y_ext[i_ext, jN + k] = Y_ext[i_ext, jN] + k * dYn
            end
        end
    end

    # Note on dx_ref: the extended mesh's default dx_ref is the
    # minimum physical edge across the FULL grid. On a stretched mesh
    # with the smallest cell at the block edge, the extension adds
    # cells of the same size → dx_ref unchanged. On a uniform mesh,
    # dx_ref is identical to the original.
    #
    # skip_validate=true: the ghost-corner cells produced by the
    # two-pass linear extrapolation can have ill-defined spline-fit
    # Jacobians at the extended (1,1), (1,Nye), (Nxe,1), (Nxe,Nye)
    # double-ghost corners — especially on curved-boundary blocks
    # (O-grid ring blocks). These cells are not read by the SLBM
    # kernel (the kernel only accesses physical interior + 1-ghost
    # rows adjacent to it). Skipping validation avoids false positives.
    # Preserve the original mesh's dx_ref: the extended mesh adds ghost-
    # corner cells whose spline-fit edge lengths can collapse to ~1e-9
    # (spurious cusps where ξ- and η-extrapolated ghost corners meet).
    # `dx_ref` is used downstream to scale τ and drag norms, so a
    # polluted dx_ref from a ghost corner would break the SLBM physics.
    return CurvilinearMesh(X_ext, Y_ext;
                             periodic_ξ=false, periodic_η=false,
                             type=Symbol(string(mesh.type) * "_ext"),
                             dx_ref=mesh.dx_ref,
                             skip_validate=true,
                             FT=T)
end

"""
    build_block_slbm_geometry_extended(block::Block; n_ghost=1,
                                        local_cfl::Bool=true)
        -> (mesh_ext::CurvilinearMesh, geom::SLBMGeometry)

Convenience: build the extended mesh for a `Block` (via
`extend_mesh_2d`) and its `SLBMGeometry`. The returned geometry has
shape-compatible arrays so `slbm_trt_libb_step!` /
`slbm_trt_libb_step_local_2d!` can run directly on the block's
extended `(Nξ + 2Ng, Nη + 2Ng, 9)` state array.
"""
function build_block_slbm_geometry_extended(block::Block;
                                              n_ghost::Int=1,
                                              local_cfl::Bool=true,
                                              curved_edges::Tuple{Vararg{Symbol}}=(),
                                              curved_center::Tuple{<:Real,<:Real}=(0, 0),
                                              dx_ref::Real=0)
    mesh_ext = extend_mesh_2d(block.mesh; n_ghost=n_ghost,
                                curved_edges=curved_edges,
                                curved_center=curved_center)
    geom = build_slbm_geometry(mesh_ext; local_cfl=local_cfl, dx_ref=dx_ref)
    return mesh_ext, geom
end

"""
    extend_interior_field_2d(field_interior::AbstractArray,
                              n_ghost::Int; pad_value=zero(eltype(field_interior)))
        -> AbstractArray

Copy a `(Nξ, Nη)` or `(Nξ, Nη, 9)` interior field into the interior of
an extended array of size `(Nξ + 2Ng, Nη + 2Ng)` or
`(Nξ + 2Ng, Nη + 2Ng, 9)`. Ghost cells are filled with `pad_value`
(default 0). Useful for lifting `is_solid`, `q_wall`, `uw_x`, `uw_y`
onto the extended block state.
"""
function extend_interior_field_2d(field::AbstractArray{T, N},
                                    n_ghost::Int;
                                    pad_value::T=zero(T)) where {T, N}
    n_ghost ≥ 0 || error("extend_interior_field_2d: n_ghost must be ≥ 0")
    if N == 2
        Nξ, Nη = size(field)
        Nξ_ext = Nξ + 2 * n_ghost
        Nη_ext = Nη + 2 * n_ghost
        out = fill(pad_value, Nξ_ext, Nη_ext)
        out[(n_ghost + 1):(n_ghost + Nξ), (n_ghost + 1):(n_ghost + Nη)] .= field
        return out
    elseif N == 3
        Nξ, Nη, Q = size(field)
        Nξ_ext = Nξ + 2 * n_ghost
        Nη_ext = Nη + 2 * n_ghost
        out = fill(pad_value, Nξ_ext, Nη_ext, Q)
        out[(n_ghost + 1):(n_ghost + Nξ), (n_ghost + 1):(n_ghost + Nη), :] .= field
        return out
    else
        error("extend_interior_field_2d: unsupported ndims $N (expected 2 or 3)")
    end
end
