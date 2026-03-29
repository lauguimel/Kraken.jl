# --- Voxelization: STL mesh → is_solid boolean grid ---

"""
    voxelize_3d(mesh::STLMesh{T}, Nx, Ny, Nz, dx, dy, dz) -> Array{Bool, 3}

Convert an STL mesh to a 3D boolean solid mask using ray casting along the
z-axis (Möller–Trumbore algorithm).

Each grid node at cell-center `(x, y, z) = ((i-0.5)*dx, (j-0.5)*dy, (k-0.5)*dz)`
is classified as inside (true) or outside (false) the closed surface.

The mesh must be a closed (watertight) surface for correct results.
"""
function voxelize_3d(mesh::STLMesh{T}, Nx::Int, Ny::Int, Nz::Int,
                     dx::Real, dy::Real, dz::Real) where T
    is_solid = zeros(Bool, Nx, Ny, Nz)

    for j in 1:Ny
        y = T((j - 0.5) * dy)
        for i in 1:Nx
            x = T((i - 0.5) * dx)

            # Cast ray from (x, y, -∞) along +z, count intersections
            z_hits = T[]
            for tri in mesh.triangles
                t_hit = _ray_triangle_z(x, y, tri)
                t_hit !== nothing && push!(z_hits, t_hit)
            end

            # Sort and deduplicate close hits (shared edges cause double-counting)
            sort!(z_hits)
            _deduplicate_hits!(z_hits)
            inside = false
            hit_idx = 1
            for k in 1:Nz
                z = T((k - 0.5) * dz)
                # Advance through hits below this z
                while hit_idx <= length(z_hits) && z_hits[hit_idx] < z
                    inside = !inside
                    hit_idx += 1
                end
                is_solid[i, j, k] = inside
            end
        end
    end

    return is_solid
end

"""
    voxelize_2d(mesh::STLMesh{T}, Nx, Ny, dx, dy; z_slice=0.0) -> Matrix{Bool}

Convert an STL mesh to a 2D boolean solid mask by:
1. Slicing the mesh at `z = z_slice` to extract a 2D contour
2. Using 2D ray casting (along +x) for point-in-polygon classification

Each grid node at `(x, y) = ((i-0.5)*dx, (j-0.5)*dy)` is tested.
"""
function voxelize_2d(mesh::STLMesh{T}, Nx::Int, Ny::Int,
                     dx::Real, dy::Real;
                     z_slice::Real=0.0) where T
    # Step 1: Extract 2D segments from z-plane intersection
    segments = _slice_mesh_z(mesh, T(z_slice))

    isempty(segments) && return zeros(Bool, Nx, Ny)

    # Step 2: Ray casting in 2D for each grid point
    is_solid = zeros(Bool, Nx, Ny)

    for j in 1:Ny
        y = T((j - 0.5) * dy)
        for i in 1:Nx
            x = T((i - 0.5) * dx)
            # Cast ray from (x, y) along +x, count intersections with segments
            crossings = 0
            for (p1, p2) in segments
                if _ray_crosses_segment_x(x, y, p1, p2)
                    crossings += 1
                end
            end
            is_solid[i, j] = isodd(crossings)
        end
    end

    return is_solid
end

"""Remove duplicate hits that are within tolerance (shared edges/vertices)."""
function _deduplicate_hits!(hits::Vector{T}; tol::T=T(1e-10)) where T
    isempty(hits) && return hits
    write_idx = 1
    for i in 2:length(hits)
        if abs(hits[i] - hits[write_idx]) > tol
            write_idx += 1
            hits[write_idx] = hits[i]
        end
    end
    resize!(hits, write_idx)
    return hits
end

# --- Ray-triangle intersection (z-axis ray) ---

"""
Möller–Trumbore intersection of ray `(x, y, -∞) + t*(0,0,1)` with triangle.
Returns z-coordinate of hit, or nothing if no intersection.
"""
function _ray_triangle_z(x::T, y::T, tri::STLTriangle{T}) where T
    v1, v2, v3 = tri.v1, tri.v2, tri.v3

    # Edge vectors
    e1x = v2[1] - v1[1];  e1y = v2[2] - v1[2];  e1z = v2[3] - v1[3]
    e2x = v3[1] - v1[1];  e2y = v3[2] - v1[2];  e2z = v3[3] - v1[3]

    # Ray direction D = (0, 0, 1)
    # h = D × e2 = (−e2y, e2x, 0)
    hx = -e2y
    hy = e2x

    # a = e1 · h
    a = e1x * hx + e1y * hy
    abs(a) < eps(T) && return nothing  # ray parallel to triangle

    inv_a = one(T) / a

    # s = O − v1, with ray origin O = (x, y, 0)
    sx = x - v1[1]
    sy = y - v1[2]
    sz = -v1[3]  # ray origin at z=0

    # u = (s · h) / a  (hz = 0, so sz doesn't contribute)
    u = (sx * hx + sy * hy) * inv_a
    (u < zero(T) || u > one(T)) && return nothing

    # q = s × e1
    qx = sy * e1z - sz * e1y
    qy = sz * e1x - sx * e1z
    qz = sx * e1y - sy * e1x

    # v = (D · q) / a = qz / a  (since D = (0,0,1))
    v = qz * inv_a
    (v < zero(T) || u + v > one(T)) && return nothing

    # t = (e2 · q) / a  → z-coordinate of intersection (ray parameter)
    t = (e2x * qx + e2y * qy + e2z * qz) * inv_a

    return t
end

# --- 2D cross-section extraction ---

"""
Slice mesh with z = z0 plane. Returns a list of 2D line segments
`[(p1, p2), ...]` where each p is `(x, y)`.
"""
function _slice_mesh_z(mesh::STLMesh{T}, z0::T) where T
    segments = Tuple{NTuple{2,T}, NTuple{2,T}}[]

    for tri in mesh.triangles
        pts = _triangle_plane_intersection(tri, z0)
        pts !== nothing && push!(segments, pts)
    end

    return segments
end

"""
Intersect a triangle with the z = z0 plane.
Returns a 2D segment `((x1,y1), (x2,y2))` or nothing.
"""
function _triangle_plane_intersection(tri::STLTriangle{T}, z0::T) where T
    verts = (tri.v1, tri.v2, tri.v3)
    signs = ntuple(i -> verts[i][3] - z0, 3)

    # Count vertices above/below/on the plane
    pts = NTuple{2,T}[]

    # Check each edge for intersection
    for (ia, ib) in ((1,2), (2,3), (3,1))
        sa, sb = signs[ia], signs[ib]
        if sa * sb < zero(T)
            # Edge crosses the plane
            va, vb = verts[ia], verts[ib]
            t = sa / (sa - sb)
            x = va[1] + t * (vb[1] - va[1])
            y = va[2] + t * (vb[2] - va[2])
            push!(pts, (x, y))
        elseif abs(sa) < eps(T) * 100
            # Vertex on the plane
            push!(pts, (verts[ia][1], verts[ia][2]))
        end
    end

    length(pts) >= 2 || return nothing
    return (pts[1], pts[2])
end

# --- 2D ray casting for point-in-polygon ---

"""
Test if a ray from (x, y) along +x crosses the segment (p1, p2).
"""
function _ray_crosses_segment_x(x::T, y::T,
                                 p1::NTuple{2,T}, p2::NTuple{2,T}) where T
    y1, y2 = p1[2], p2[2]

    # Segment must straddle y
    (y1 <= y < y2 || y2 <= y < y1) || return false

    # Compute x-intersection
    t = (y - y1) / (y2 - y1)
    x_int = p1[1] + t * (p2[1] - p1[1])

    return x_int > x
end
