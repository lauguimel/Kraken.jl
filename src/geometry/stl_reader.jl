# --- Minimal STL file reader (binary + ASCII, zero dependencies) ---

"""
    STLTriangle{T}

A single triangle from an STL file: normal vector and three vertices.
"""
struct STLTriangle{T <: AbstractFloat}
    normal::NTuple{3, T}
    v1::NTuple{3, T}
    v2::NTuple{3, T}
    v3::NTuple{3, T}
end

"""
    STLMesh{T}

A triangulated surface mesh loaded from an STL file.

# Fields
- `triangles::Vector{STLTriangle{T}}`: list of triangles.
- `bbox_min::NTuple{3,T}`: bounding box minimum corner.
- `bbox_max::NTuple{3,T}`: bounding box maximum corner.
"""
struct STLMesh{T <: AbstractFloat}
    triangles::Vector{STLTriangle{T}}
    bbox_min::NTuple{3, T}
    bbox_max::NTuple{3, T}
end

"""
    read_stl(filename::String; T=Float64) -> STLMesh{T}

Read an STL file (binary or ASCII) and return an `STLMesh`.
Auto-detects format by checking the file header.
"""
function read_stl(filename::String; T::Type{<:AbstractFloat}=Float64)
    data = read(filename)

    if _is_ascii_stl(data)
        return _read_ascii_stl(filename, T)
    else
        return _read_binary_stl(data, T)
    end
end

"""Check if data looks like an ASCII STL (starts with 'solid' followed by a name/newline)."""
function _is_ascii_stl(data::Vector{UInt8})
    length(data) < 6 && return false
    header = String(data[1:min(80, length(data))])
    # ASCII STL starts with "solid" followed by space/newline
    # Binary STL also has an 80-byte header that could start with "solid"
    # Heuristic: if "facet" appears in the first 1000 bytes, it's ASCII
    startswith(header, "solid") || return false
    preview = String(data[1:min(1000, length(data))])
    return occursin("facet", preview)
end

# --- Binary STL reader ---

function _read_binary_stl(data::Vector{UInt8}, ::Type{T}) where T
    # Binary format: 80-byte header + 4-byte uint32 count + triangles
    length(data) < 84 && throw(ArgumentError("STL file too short for binary format"))

    ntri = reinterpret(UInt32, data[81:84])[1]
    expected = 84 + ntri * 50  # each triangle = 50 bytes
    length(data) < expected &&
        throw(ArgumentError("STL file truncated: expected $expected bytes, got $(length(data))"))

    triangles = Vector{STLTriangle{T}}(undef, ntri)
    bbox_min = (T(Inf), T(Inf), T(Inf))
    bbox_max = (T(-Inf), T(-Inf), T(-Inf))

    offset = 85  # 1-indexed, after header (80) + count (4)
    for i in 1:ntri
        # 12 floats (normal + 3 vertices) as Float32 = 48 bytes + 2 bytes attribute
        floats = reinterpret(Float32, data[offset:offset+47])

        nx, ny, nz = T(floats[1]), T(floats[2]), T(floats[3])
        v1 = (T(floats[4]),  T(floats[5]),  T(floats[6]))
        v2 = (T(floats[7]),  T(floats[8]),  T(floats[9]))
        v3 = (T(floats[10]), T(floats[11]), T(floats[12]))

        triangles[i] = STLTriangle{T}((nx, ny, nz), v1, v2, v3)

        # Update bounding box
        for v in (v1, v2, v3)
            bbox_min = (min(bbox_min[1], v[1]), min(bbox_min[2], v[2]), min(bbox_min[3], v[3]))
            bbox_max = (max(bbox_max[1], v[1]), max(bbox_max[2], v[2]), max(bbox_max[3], v[3]))
        end

        offset += 50
    end

    return STLMesh{T}(triangles, bbox_min, bbox_max)
end

# --- ASCII STL reader ---

function _read_ascii_stl(filename::String, ::Type{T}) where T
    triangles = STLTriangle{T}[]
    bbox_min = (T(Inf), T(Inf), T(Inf))
    bbox_max = (T(-Inf), T(-Inf), T(-Inf))

    lines = readlines(filename)
    i = 1
    while i <= length(lines)
        line = strip(lines[i])

        if startswith(line, "facet normal")
            # Parse normal
            parts = split(line)
            nx, ny, nz = parse(T, parts[3]), parse(T, parts[4]), parse(T, parts[5])

            # Skip "outer loop"
            i += 2

            # Parse 3 vertices
            verts = NTuple{3, T}[]
            for _ in 1:3
                vline = strip(lines[i])
                vparts = split(vline)
                push!(verts, (parse(T, vparts[2]), parse(T, vparts[3]), parse(T, vparts[4])))
                i += 1
            end

            push!(triangles, STLTriangle{T}((nx, ny, nz), verts[1], verts[2], verts[3]))

            for v in verts
                bbox_min = (min(bbox_min[1], v[1]), min(bbox_min[2], v[2]), min(bbox_min[3], v[3]))
                bbox_max = (max(bbox_max[1], v[1]), max(bbox_max[2], v[2]), max(bbox_max[3], v[3]))
            end

            # Skip "endloop" and "endfacet"
            i += 2
        else
            i += 1
        end
    end

    return STLMesh{T}(triangles, bbox_min, bbox_max)
end

"""
    transform_mesh(mesh::STLMesh{T}; scale=1.0, translate=(0,0,0)) -> STLMesh{T}

Apply scale and translation to an STL mesh.
Scaling is applied first (around origin), then translation.
"""
function transform_mesh(mesh::STLMesh{T};
                        scale::Real=1.0,
                        translate::NTuple{3,<:Real}=(0.0, 0.0, 0.0)) where T
    s = T(scale)
    tx, ty, tz = T(translate[1]), T(translate[2]), T(translate[3])

    _tr(v) = (v[1] * s + tx, v[2] * s + ty, v[3] * s + tz)

    new_tris = [STLTriangle{T}(tri.normal, _tr(tri.v1), _tr(tri.v2), _tr(tri.v3))
                for tri in mesh.triangles]

    new_min = _tr(mesh.bbox_min)
    new_max = _tr(mesh.bbox_max)

    return STLMesh{T}(new_tris, new_min, new_max)
end
