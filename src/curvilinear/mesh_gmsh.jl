using Gmsh

# ===========================================================================
# Load a 2D/3D structured mesh produced by gmsh (Transfinite block) into a
# CurvilinearMesh. Companion to mesh_from_arrays.jl: this file is the reader,
# the spline + metric pipeline lives there.
#
# Two layout-detection strategies are exposed:
#   :axis_aligned (default, Phase A)
#       Sort node coordinates lex (Y then X in 2D) and reshape into (Nξ, Nη).
#       Robust for any Transfinite block whose physical layout is axis-
#       aligned (Cartesian, stretched_box, channel with rectangular blocks).
#       Fails on O-grid / annulus where physical X is not monotone in ξ.
#
#   :topological (Phase B, TODO)
#       Edge-walk the quad/hex connectivity to recover (Nξ, Nη[, Nζ]) for
#       any Transfinite block, including O-grids. Heavier, ~150 lines.
#
# Physical groups are read so the caller can map BCSpec faces to gmsh tags
# without hand-coding indices. See `gmsh_physical_groups`.
# ===========================================================================

"""
    GmshPhysicalGroups

Per-tag mapping returned alongside a mesh: `name => tag` for each Physical
group, and `tag => geo_entities` (the ξ/η/ζ entity ids the user can match
to a face of the structured block).
"""
struct GmshPhysicalGroups
    by_name::Dict{String, Int}              # name → physical tag
    by_tag::Dict{Int, Vector{Int}}          # physical tag → geo entities
    dim::Dict{Int, Int}                     # physical tag → topological dim
end

"""
    load_gmsh_mesh_2d(path; surface_tag=nothing, layout=:axis_aligned, FT=Float64)
        -> (mesh::CurvilinearMesh, groups::GmshPhysicalGroups)

Open a gmsh `.msh` (or `.geo` followed by `gmsh.model.mesh.generate`) and
build a `CurvilinearMesh` from the nodes of the surface tagged
`surface_tag` (auto if exactly one surface is present). The mesh must
have been generated as a Transfinite quadrangular block.

`layout=:axis_aligned`: sort nodes lex (Y, X) and reshape; works for
any Transfinite block whose physical layout is axis-aligned.

`layout=:topological`: TODO — edge-walk the connectivity (needed for
O-grids, where physical X is not monotone in ξ).

Returns the `CurvilinearMesh` and a `GmshPhysicalGroups` struct mapping
physical names/tags to the geo entities the user can assign to BCSpec
faces.
"""
function load_gmsh_mesh_2d(path::AbstractString;
                            surface_tag::Union{Int, Nothing}=nothing,
                            layout::Symbol=:axis_aligned,
                            periodic_ξ::Bool=false, periodic_η::Bool=false,
                            FT::Type{<:AbstractFloat}=Float64)
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        if endswith(lowercase(path), ".geo")
            gmsh.open(path)
            gmsh.model.mesh.generate(2)
        else
            gmsh.open(path)
        end
        return _load_2d_open(; surface_tag=surface_tag, layout=layout,
                             periodic_ξ=periodic_ξ, periodic_η=periodic_η, FT=FT)
    finally
        gmsh.finalize()
    end
end

function _load_2d_open(; surface_tag, layout, periodic_ξ, periodic_η, FT)
    if surface_tag === nothing
        ents = gmsh.model.getEntities(2)
        length(ents) == 1 ||
            error("load_gmsh_mesh_2d: $(length(ents)) surfaces found; pass surface_tag=…")
        surface_tag = Int(ents[1][2])
    end

    # All nodes attached to (or referenced by) the requested surface.
    # includeBoundary=true → include nodes on the four bordering curves.
    node_tags_raw, coord_raw, _ = gmsh.model.mesh.getNodes(2, surface_tag, true, false)
    n_nodes = length(node_tags_raw)
    coords = reshape(coord_raw, 3, n_nodes)
    coord_x = FT.(coords[1, :])
    coord_y = FT.(coords[2, :])

    # Build (Nξ, Nη) layout. Phase A: lex sort on (y rounded to mesh tol, x).
    # Tolerance picked at 1e-9 of the bounding-box diagonal to merge a row
    # whose y-values differ by floating-point noise.
    if layout !== :axis_aligned
        error("load_gmsh_mesh_2d: layout=$layout not yet implemented (Phase B). " *
              "Use :axis_aligned for Transfinite blocks with axis-aligned outline.")
    end
    bbox_diag = sqrt((maximum(coord_x) - minimum(coord_x))^2 +
                     (maximum(coord_y) - minimum(coord_y))^2)
    tol = max(FT(1e-12), FT(1e-9) * bbox_diag)
    # Quantise y so equal rows compare equal under sort
    y_q = round.(coord_y ./ tol) .* tol
    perm = sortperm(collect(zip(y_q, coord_x)))
    coord_x_s = coord_x[perm]
    coord_y_s = coord_y[perm]
    y_q_s = y_q[perm]
    # Detect Nξ as the count of consecutive entries with the same y_q at the
    # start. Then verify the total is divisible.
    Nξ = 1
    while Nξ < n_nodes && y_q_s[Nξ + 1] == y_q_s[1]
        Nξ += 1
    end
    n_nodes % Nξ == 0 ||
        error("load_gmsh_mesh_2d: $n_nodes nodes not divisible by detected Nξ=$Nξ; " *
              "the mesh may not be a structured Transfinite block, or the y-tolerance " *
              "$tol is wrong (try a finer mesh).")
    Nη = n_nodes ÷ Nξ
    X = reshape(coord_x_s, Nξ, Nη)
    Y = reshape(coord_y_s, Nξ, Nη)

    mesh = CurvilinearMesh(X, Y;
                            periodic_ξ=periodic_ξ, periodic_η=periodic_η,
                            type=:gmsh_imported, FT=FT)
    groups = _read_physical_groups()
    return mesh, groups
end

function _read_physical_groups()
    by_name = Dict{String, Int}()
    by_tag  = Dict{Int, Vector{Int}}()
    dim_of  = Dict{Int, Int}()
    for (dim, tag) in gmsh.model.getPhysicalGroups()
        name = gmsh.model.getPhysicalName(dim, tag)
        ents = Int.(gmsh.model.getEntitiesForPhysicalGroup(dim, tag))
        if !isempty(name); by_name[name] = Int(tag); end
        by_tag[Int(tag)] = ents
        dim_of[Int(tag)] = Int(dim)
    end
    return GmshPhysicalGroups(by_name, by_tag, dim_of)
end

"""
    load_gmsh_mesh_3d(path; volume_tag=nothing, layout=:axis_aligned, FT=Float64)
        -> (mesh::CurvilinearMesh3D, groups::GmshPhysicalGroups)

3D counterpart. Same Phase A axis-aligned layout: lex sort on (z, y, x)
then reshape to `(Nξ, Nη, Nζ)`.
"""
function load_gmsh_mesh_3d(path::AbstractString;
                            volume_tag::Union{Int, Nothing}=nothing,
                            layout::Symbol=:axis_aligned,
                            periodic_ξ::Bool=false, periodic_η::Bool=false,
                            periodic_ζ::Bool=false,
                            FT::Type{<:AbstractFloat}=Float64)
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        if endswith(lowercase(path), ".geo")
            gmsh.open(path)
            gmsh.model.mesh.generate(3)
        else
            gmsh.open(path)
        end
        return _load_3d_open(; volume_tag=volume_tag, layout=layout,
                             periodic_ξ=periodic_ξ, periodic_η=periodic_η,
                             periodic_ζ=periodic_ζ, FT=FT)
    finally
        gmsh.finalize()
    end
end

function _load_3d_open(; volume_tag, layout, periodic_ξ, periodic_η, periodic_ζ, FT)
    if volume_tag === nothing
        ents = gmsh.model.getEntities(3)
        length(ents) == 1 ||
            error("load_gmsh_mesh_3d: $(length(ents)) volumes found; pass volume_tag=…")
        volume_tag = Int(ents[1][2])
    end
    layout === :axis_aligned ||
        error("load_gmsh_mesh_3d: only :axis_aligned supported (Phase A).")

    node_tags_raw, coord_raw, _ = gmsh.model.mesh.getNodes(3, volume_tag, true, false)
    n_nodes = length(node_tags_raw)
    coords = reshape(coord_raw, 3, n_nodes)
    cx = FT.(coords[1, :]); cy = FT.(coords[2, :]); cz = FT.(coords[3, :])

    bbox_diag = sqrt((maximum(cx) - minimum(cx))^2 +
                     (maximum(cy) - minimum(cy))^2 +
                     (maximum(cz) - minimum(cz))^2)
    tol = max(FT(1e-12), FT(1e-9) * bbox_diag)
    y_q = round.(cy ./ tol) .* tol
    z_q = round.(cz ./ tol) .* tol
    perm = sortperm(collect(zip(z_q, y_q, cx)))
    cx_s = cx[perm]; cy_s = cy[perm]; cz_s = cz[perm]
    z_q_s = z_q[perm]; y_q_s = y_q[perm]
    Nξ = 1
    while Nξ < n_nodes && y_q_s[Nξ+1] == y_q_s[1] && z_q_s[Nξ+1] == z_q_s[1]
        Nξ += 1
    end
    Nη = 1
    while Nη < (n_nodes ÷ Nξ) && z_q_s[1 + Nη*Nξ] == z_q_s[1]
        Nη += 1
    end
    n_nodes == Nξ * Nη * (n_nodes ÷ (Nξ*Nη)) ||
        error("load_gmsh_mesh_3d: cannot infer 3D layout, got Nξ=$Nξ Nη=$Nη")
    Nζ = n_nodes ÷ (Nξ * Nη)
    X = reshape(cx_s, Nξ, Nη, Nζ)
    Y = reshape(cy_s, Nξ, Nη, Nζ)
    Z = reshape(cz_s, Nξ, Nη, Nζ)

    mesh = CurvilinearMesh3D(X, Y, Z;
                              periodic_ξ=periodic_ξ, periodic_η=periodic_η,
                              periodic_ζ=periodic_ζ,
                              type=:gmsh_imported, FT=FT)
    groups = _read_physical_groups()
    return mesh, groups
end
