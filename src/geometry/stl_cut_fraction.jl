# =====================================================================
# STL → LI-BB cut-fraction fields.
#
# Given an STL mesh + a regular grid, precompute `(q_wall, is_solid)`
# arrays suitable for `fused_trt_libb_v2_step!` (2D) or
# `fused_trt_libb_v2_step_3d!` (3D).
#
# 2D: sub-cell q_w via ray-segment intersection on the mesh slice.
# 3D: sub-cell q_w via Möller-Trumbore ray-triangle intersection; the
#     default remains halfway (q_w=0.5) for backward compatibility.
# =====================================================================

"""
Möller-Trumbore ray-triangle intersection. Returns the smallest `t ≥ 0`
such that `origin + t · dir` lies on the triangle `(v0, v1, v2)`, or a
sentinel `-1` if no intersection.
"""
@inline function _ray_tri_intersect_t(origin::NTuple{3,T}, dir::NTuple{3,T},
                                        v0::NTuple{3,T}, v1::NTuple{3,T},
                                        v2::NTuple{3,T}) where {T}
    eps_T = eps(T) * T(100)
    e1x = v1[1] - v0[1]; e1y = v1[2] - v0[2]; e1z = v1[3] - v0[3]
    e2x = v2[1] - v0[1]; e2y = v2[2] - v0[2]; e2z = v2[3] - v0[3]
    # h = dir × e2
    hx = dir[2]*e2z - dir[3]*e2y
    hy = dir[3]*e2x - dir[1]*e2z
    hz = dir[1]*e2y - dir[2]*e2x
    a  = e1x*hx + e1y*hy + e1z*hz
    abs(a) < eps_T && return -one(T)
    f = one(T) / a
    sx = origin[1] - v0[1]; sy = origin[2] - v0[2]; sz = origin[3] - v0[3]
    u = f * (sx*hx + sy*hy + sz*hz)
    (u < zero(T) || u > one(T)) && return -one(T)
    # q = s × e1
    qx = sy*e1z - sz*e1y
    qy = sz*e1x - sx*e1z
    qz = sx*e1y - sy*e1x
    v = f * (dir[1]*qx + dir[2]*qy + dir[3]*qz)
    (v < zero(T) || u + v > one(T)) && return -one(T)
    t = f * (e2x*qx + e2y*qy + e2z*qz)
    t > eps_T && return t
    return -one(T)
end

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
                                   FT=Float64, sub_cell=false)
        -> (q_wall, is_solid)

3D version. With `sub_cell=false` (default, halfway-BB), every cut
link gets `q_w = 0.5`. With `sub_cell=true`, each cut link is queried
against all STL triangles (Möller-Trumbore) and `q_w` is set to the
fraction of the link length from the fluid node to the first
intersection — full Bouzidi precision on arbitrary STL surfaces.

Complexity is `O(N_fluid_boundary · N_triangles)` which is fine for
moderate meshes (~thousands of triangles) but can be accelerated
with a BVH for very large meshes (future work).
"""
function precompute_q_wall_from_stl_3d(mesh::STLMesh{T},
                                        Nx::Int, Ny::Int, Nz::Int,
                                        dx::Real, dy::Real, dz::Real;
                                        FT::Type{<:AbstractFloat}=Float64,
                                        sub_cell::Bool=false) where {T}
    is_solid = voxelize_3d(mesh, Nx, Ny, Nz, dx, dy, dz)
    q_wall = zeros(FT, Nx, Ny, Nz, 19)
    cxs = velocities_x(D3Q19())
    cys = velocities_y(D3Q19())
    czs = velocities_z(D3Q19())
    half = FT(0.5)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        is_solid[i, j, k] && continue
        # Fluid node position — cell-centre convention matching voxelize_3d.
        xf = FT((i - 0.5) * dx)
        yf = FT((j - 0.5) * dy)
        zf = FT((k - 0.5) * dz)
        for q in 2:19
            ni = i + Int(cxs[q])
            nj = j + Int(cys[q])
            nk = k + Int(czs[q])
            (1 <= ni <= Nx && 1 <= nj <= Ny && 1 <= nk <= Nz) || continue
            is_solid[ni, nj, nk] || continue
            qw = half
            if sub_cell
                # Ray along the link. Link length = |c_q| · Δx (axis) or
                # √2·Δx (edges in D3Q19 — no body diagonals). Since we
                # solve for t with the ray parameter, q_w = t directly
                # when the ray direction equals the link vector scaled
                # to unit link length.
                ddx = FT(Int(cxs[q]) * dx)
                ddy = FT(Int(cys[q]) * dy)
                ddz = FT(Int(czs[q]) * dz)
                origin = (xf, yf, zf)
                dir    = (ddx, ddy, ddz)
                t_best = FT(Inf)
                for tri in mesh.triangles
                    t = _ray_tri_intersect_t(origin, dir,
                                              (FT(tri.v1[1]), FT(tri.v1[2]), FT(tri.v1[3])),
                                              (FT(tri.v2[1]), FT(tri.v2[2]), FT(tri.v2[3])),
                                              (FT(tri.v3[1]), FT(tri.v3[2]), FT(tri.v3[3])))
                    (t > zero(FT) && t < t_best) && (t_best = t)
                end
                if t_best > zero(FT) && t_best ≤ one(FT)
                    qw = t_best
                end
            end
            q_wall[i, j, k, q] = qw
        end
    end
    return q_wall, is_solid
end
