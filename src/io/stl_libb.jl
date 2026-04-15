# =====================================================================
# STL → LI-BB cut-fraction fields.
#
# Given an STL mesh + a regular grid, precompute `(q_wall, is_solid)`
# arrays suitable for `fused_trt_libb_v2_step!` (2D) or
# `fused_trt_libb_v2_step_3d!` (3D).
#
# 2D: sub-cell q_w via ray-segment intersection on the mesh slice.
# 3D: halfway-BB (q_w = 0.5) default — sub-cell 3D via Möller-Trumbore
# against mesh triangles is tracked as future work.
# =====================================================================

"""
Smallest positive `t ∈ (0, 1]` such that `(x0 + t·dx, y0 + t·dy)` lies
on the segment `(p1, p2)`. Returns `-1` if the ray does not intersect
the segment in that interval.
"""
@inline function _ray_seg_intersect_t(x0::T, y0::T, dx::T, dy::T,
                                       p1::NTuple{2,T}, p2::NTuple{2,T}) where {T}
    sx = p2[1] - p1[1]
    sy = p2[2] - p1[2]
    denom = dx * sy - dy * sx
    abs(denom) < eps(T) && return -one(T)
    rx = p1[1] - x0
    ry = p1[2] - y0
    t = (rx * sy - ry * sx) / denom    # ray parameter
    u = (rx * dy - ry * dx) / denom    # segment parameter
    if t > zero(T) && t ≤ one(T) && u ≥ zero(T) && u ≤ one(T)
        return t
    end
    return -one(T)
end

"""
    precompute_q_wall_from_stl_2d(mesh, Nx, Ny, dx, dy;
                                   z_slice=0.0, FT=Float64,
                                   sub_cell=true)
        -> (q_wall, is_solid)

Voxelise `mesh` at z = z_slice and compute the LI-BB cut-fraction
field. Each fluid cell whose link q points into the solid mask gets
`q_wall[i,j,q] = q_w` where:

- `sub_cell = true`: `q_w` is the distance (in units of the link
  length) from the fluid node to the first intersection with the STL
  slice segments. Gives Bouzidi sub-cell precision.
- `sub_cell = false`: `q_w = 0.5` (halfway-BB default).

Returns a `(Nx, Ny, 9)` cut-fraction array + a `(Nx, Ny)` bool mask.
"""
function precompute_q_wall_from_stl_2d(mesh::STLMesh{T},
                                        Nx::Int, Ny::Int,
                                        dx::Real, dy::Real;
                                        z_slice::Real=0.0,
                                        FT::Type{<:AbstractFloat}=Float64,
                                        sub_cell::Bool=true) where {T}
    is_solid = voxelize_2d(mesh, Nx, Ny, dx, dy; z_slice=z_slice)
    q_wall = zeros(FT, Nx, Ny, 9)
    cxs = velocities_x(D2Q9())
    cys = velocities_y(D2Q9())
    half = FT(0.5)

    # Slice once for sub-cell queries.
    segments = sub_cell ? _slice_mesh_z(mesh, T(z_slice)) : nothing

    @inbounds for j in 1:Ny, i in 1:Nx
        is_solid[i, j] && continue
        # Fluid node position in the same coords as the voxelizer:
        # cell center at ((i - 0.5)*dx, (j - 0.5)*dy).
        xf = FT((i - 0.5) * dx)
        yf = FT((j - 0.5) * dy)
        for q in 2:9
            ni = i + Int(cxs[q])
            nj = j + Int(cys[q])
            (1 <= ni <= Nx && 1 <= nj <= Ny) || continue
            is_solid[ni, nj] || continue

            qw = half
            if sub_cell && !isnothing(segments) && !isempty(segments)
                # Ray from (xf, yf) in direction (cxs[q]*dx, cys[q]*dy).
                # Link length = |c_q| · √(dx² + dy²)... for integer
                # c_q with dx=dy: link length = |c_q|·dx in axis,
                # √2·dx in diagonals. q_w is the fraction of that
                # link inside the fluid ⇒ t from the ray parameter
                # (since the ray length is exactly one link) gives q_w.
                ddx = FT(Int(cxs[q]) * dx)
                ddy = FT(Int(cys[q]) * dy)
                t_best = FT(Inf)
                for seg in segments
                    # Seg stored as (NTuple{2,T}, NTuple{2,T})
                    p1 = (FT(seg[1][1]), FT(seg[1][2]))
                    p2 = (FT(seg[2][1]), FT(seg[2][2]))
                    t = _ray_seg_intersect_t(xf, yf, ddx, ddy, p1, p2)
                    (t > zero(FT) && t < t_best) && (t_best = t)
                end
                if t_best ≤ one(FT) && t_best > zero(FT)
                    qw = t_best
                end
            end
            q_wall[i, j, q] = qw
        end
    end
    return q_wall, is_solid
end

"""
    precompute_q_wall_from_stl_3d(mesh, Nx, Ny, Nz, dx, dy, dz;
                                   FT=Float64)
        -> (q_wall, is_solid)

3D version: voxelise `mesh` into a 3D boolean mask, flag D3Q19
fluid-cell links that cross into the mask with `q_w = 0.5`.
"""
function precompute_q_wall_from_stl_3d(mesh::STLMesh{T},
                                        Nx::Int, Ny::Int, Nz::Int,
                                        dx::Real, dy::Real, dz::Real;
                                        FT::Type{<:AbstractFloat}=Float64) where {T}
    is_solid = voxelize_3d(mesh, Nx, Ny, Nz, dx, dy, dz)
    q_wall = zeros(FT, Nx, Ny, Nz, 19)
    cxs = velocities_x(D3Q19())
    cys = velocities_y(D3Q19())
    czs = velocities_z(D3Q19())
    half = FT(0.5)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        is_solid[i, j, k] && continue
        for q in 2:19
            ni = i + Int(cxs[q])
            nj = j + Int(cys[q])
            nk = k + Int(czs[q])
            if 1 <= ni <= Nx && 1 <= nj <= Ny && 1 <= nk <= Nz &&
                is_solid[ni, nj, nk]
                q_wall[i, j, k, q] = half
            end
        end
    end
    return q_wall, is_solid
end
