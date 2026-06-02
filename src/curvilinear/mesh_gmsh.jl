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

    if layout === :axis_aligned
        X, Y = _layout_axis_aligned_2d(surface_tag, FT)
    elseif layout === :topological
        X, Y = _layout_topological_2d(surface_tag, FT;
                                        periodic_ξ=periodic_ξ, periodic_η=periodic_η)
    else
        error("load_gmsh_mesh_2d: layout=$layout not supported")
    end

    mesh = CurvilinearMesh(X, Y;
                            periodic_ξ=periodic_ξ, periodic_η=periodic_η,
                            type=:gmsh_imported, FT=FT)
    groups = _read_physical_groups()
    return mesh, groups
end

# Phase A: lex sort on (y, x). Robust for any Transfinite block whose
# physical outline is axis-aligned (Cartesian, stretched_box, channel
# blocks). Cannot reconstruct O-grids / annuli — use :topological.
function _layout_axis_aligned_2d(surface_tag, FT)
    node_tags_raw, coord_raw, _ = gmsh.model.mesh.getNodes(2, surface_tag, true, false)
    n_nodes = length(node_tags_raw)
    coords = reshape(coord_raw, 3, n_nodes)
    coord_x = FT.(coords[1, :])
    coord_y = FT.(coords[2, :])
    bbox_diag = sqrt((maximum(coord_x) - minimum(coord_x))^2 +
                     (maximum(coord_y) - minimum(coord_y))^2)
    tol = max(FT(1e-12), FT(1e-9) * bbox_diag)
    y_q = round.(coord_y ./ tol) .* tol
    perm = sortperm(collect(zip(y_q, coord_x)))
    coord_x_s = coord_x[perm]; coord_y_s = coord_y[perm]; y_q_s = y_q[perm]
    Nξ = 1
    while Nξ < n_nodes && y_q_s[Nξ + 1] == y_q_s[1]; Nξ += 1; end
    n_nodes % Nξ == 0 ||
        error("layout=:axis_aligned: $n_nodes not divisible by detected Nξ=$Nξ; " *
              "the mesh outline may not be axis-aligned, switch to :topological.")
    Nη = n_nodes ÷ Nξ
    X = reshape(coord_x_s, Nξ, Nη)
    Y = reshape(coord_y_s, Nξ, Nη)
    return X, Y
end

# Phase B: topological edge-walking on the quad connectivity. Supports
# the open-canal (4 corner nodes) and the annulus (periodic-in-ξ, no
# corner nodes) cases. Multi-block / curved boundaries with no real
# corner are not yet handled.
function _layout_topological_2d(surface_tag, FT; periodic_ξ::Bool, periodic_η::Bool)
    # Get nodes (tag → coord)
    node_tags_raw, coord_raw, _ = gmsh.model.mesh.getNodes(2, surface_tag, true, false)
    n_nodes = length(node_tags_raw)
    coords = reshape(coord_raw, 3, n_nodes)
    node_xy = Dict{Int, NTuple{2, FT}}()
    for k in 1:n_nodes
        node_xy[Int(node_tags_raw[k])] = (FT(coords[1, k]), FT(coords[2, k]))
    end

    # Get quads (gmsh element type 3 = QUAD4)
    elem_types, _, conn_list = gmsh.model.mesh.getElements(2, surface_tag)
    qi = findfirst(==(3), elem_types)
    qi === nothing && error("layout=:topological: surface has no QUAD4 elements")
    conn = Int.(conn_list[qi])
    Nq = length(conn) ÷ 4
    quad_nodes = reshape(conn, 4, Nq)

    # Build adjacency
    node_quads = Dict{Int, Vector{Int}}()
    edge_quads = Dict{Tuple{Int, Int}, Vector{Int}}()
    for q in 1:Nq, k in 1:4
        n = quad_nodes[k, q]
        push!(get!(node_quads, n, Int[]), q)
        n_next = quad_nodes[mod1(k + 1, 4), q]
        e = n < n_next ? (n, n_next) : (n_next, n)
        push!(get!(edge_quads, e, Int[]), q)
    end

    boundary_edges = [e for (e, qs) in edge_quads if length(qs) == 1]
    node_b_edges = Dict{Int, Vector{Tuple{Int, Int}}}()
    for e in boundary_edges
        push!(get!(node_b_edges, e[1], Tuple{Int, Int}[]), e)
        push!(get!(node_b_edges, e[2], Tuple{Int, Int}[]), e)
    end

    corners = [n for (n, qs) in node_quads if length(qs) == 1]

    if length(corners) == 4 && !periodic_ξ && !periodic_η
        return _layout_quad_block_2d(node_xy, quad_nodes, node_quads, node_b_edges,
                                      corners, FT)
    elseif isempty(corners) && periodic_ξ && !periodic_η
        return _layout_annulus_periodic_xi_2d(node_xy, quad_nodes, node_quads,
                                                node_b_edges, FT)
    else
        error("layout=:topological: unsupported topology — $(length(corners)) " *
              "corners with periodic_ξ=$periodic_ξ, periodic_η=$periodic_η. " *
              "Currently supported: (4 corners, both non-periodic) or (0 corners, " *
              "periodic_ξ only).")
    end
end

# 4-corner open block: walk along 2 boundary curves from one corner to
# get Nξ and Nη, then propagate inward via shared quads.
function _layout_quad_block_2d(node_xy, quad_nodes, node_quads, node_b_edges,
                                 corners, FT)
    function walk_boundary(start::Int, prev::Int)
        path = [prev, start]
        while true
            current = path[end]
            length(node_quads[current]) == 1 && current != prev && break
            edges = node_b_edges[current]
            length(edges) == 2 ||
                error("non-corner boundary node $current has $(length(edges)) bdy edges")
            prev_node = path[end - 1]
            next_edge = edges[1] == (min(current, prev_node), max(current, prev_node)) ? edges[2] : edges[1]
            next_node = next_edge[1] == current ? next_edge[2] : next_edge[1]
            push!(path, next_node)
            length(node_quads[next_node]) == 1 && break
        end
        return path
    end

    corner_BL = corners[1]
    quad_BL = node_quads[corner_BL][1]
    pos = findfirst(==(corner_BL), quad_nodes[:, quad_BL])
    nbr_ξ = quad_nodes[mod1(pos + 1, 4), quad_BL]
    nbr_η = quad_nodes[mod1(pos - 1 + 4, 4), quad_BL]

    row1 = walk_boundary(nbr_ξ, corner_BL)
    col1 = walk_boundary(nbr_η, corner_BL)
    Nξ = length(row1); Nη = length(col1)
    n_total = length(node_quads)
    Nξ * Nη == n_total ||
        error("topo: Nξ=$Nξ × Nη=$Nη != n_nodes=$n_total — multi-block?")

    layout = zeros(Int, Nξ, Nη)
    layout[:, 1] .= row1
    layout[1, :] .= col1
    for j in 2:Nη, i in 2:Nξ
        # Find the unique quad containing (i-1, j-1), (i, j-1), (i-1, j); (i, j) is its 4th node.
        q_set = intersect(node_quads[layout[i-1, j-1]],
                           node_quads[layout[i,   j-1]],
                           node_quads[layout[i-1, j]])
        length(q_set) == 1 || error("topo: non-unique enclosing quad at ($i, $j)")
        q = q_set[1]
        for n in quad_nodes[:, q]
            if n != layout[i-1, j-1] && n != layout[i, j-1] && n != layout[i-1, j]
                layout[i, j] = n; break
            end
        end
    end

    X = [node_xy[layout[i, j]][1] for i in 1:Nξ, j in 1:Nη]
    Y = [node_xy[layout[i, j]][2] for i in 1:Nξ, j in 1:Nη]
    return X, Y
end

# Annulus periodic in ξ: 0 corners, the boundary consists of 2 disjoint
# loops (inner + outer ring). Walk one ring to get Nξ; the radial extent
# Nη is then n_total / Nξ. Inner-to-outer column built by following the
# quad chain perpendicular to each ring node.
function _layout_annulus_periodic_xi_2d(node_xy, quad_nodes, node_quads,
                                          node_b_edges, FT)
    # Identify the two boundary loops by connected components of boundary nodes
    bnd_nodes = collect(keys(node_b_edges))
    visited = Set{Int}()
    loops = Vector{Vector{Int}}()
    for seed in bnd_nodes
        seed in visited && continue
        loop = [seed]; push!(visited, seed)
        prev = seed
        cur  = node_b_edges[seed][1][1] == seed ? node_b_edges[seed][1][2] : node_b_edges[seed][1][1]
        while cur != seed
            push!(loop, cur); push!(visited, cur)
            edges = node_b_edges[cur]
            next_edge = edges[1] == (min(cur, prev), max(cur, prev)) ? edges[2] : edges[1]
            nxt = next_edge[1] == cur ? next_edge[2] : next_edge[1]
            prev = cur; cur = nxt
        end
        push!(loops, loop)
    end
    length(loops) == 2 ||
        error("annulus topo: expected 2 boundary loops, got $(length(loops))")
    # Inner loop = the smaller-radius one (judge by mean distance from centroid).
    centroid = let X = first.(values(node_xy)), Y = last.(values(node_xy))
        (sum(X) / length(X), sum(Y) / length(Y))
    end
    function loop_radius(loop)
        rs = [hypot(node_xy[n][1] - centroid[1], node_xy[n][2] - centroid[2]) for n in loop]
        sum(rs) / length(rs)
    end
    r1 = loop_radius(loops[1]); r2 = loop_radius(loops[2])
    inner_loop, outer_loop = r1 < r2 ? (loops[1], loops[2]) : (loops[2], loops[1])

    Nξ = length(inner_loop)
    length(outer_loop) == Nξ ||
        error("annulus topo: inner ring has $Nξ nodes but outer ring has $(length(outer_loop))")
    n_total = length(node_quads)
    n_total % Nξ == 0 ||
        error("annulus topo: total $n_total not divisible by Nξ=$Nξ")
    Nη = n_total ÷ Nξ

    # Build column 1 (radial walk from inner_loop[1] through interior to outer_loop)
    layout = zeros(Int, Nξ, Nη)
    # The inner ring is η=1
    for i in 1:Nξ
        layout[i, 1] = inner_loop[i]
    end
    # For each i, walk radially: the next node (i, j+1) is reached by
    # finding a quad containing (layout[i, j], layout[mod_next, j]) and
    # picking the 2 nodes that aren't on row j.
    inner_set = Set(inner_loop)
    for j in 1:(Nη - 1)
        for i in 1:Nξ
            i_next = mod1(i + 1, Nξ)
            n_a = layout[i, j]; n_b = layout[i_next, j]
            cands = intersect(node_quads[n_a], node_quads[n_b])
            # Pick the quad whose other 2 nodes are NOT on row j (avoid going back)
            chosen_quad = 0
            chosen_side = (0, 0)
            for q in cands
                others = Int[]
                for n in quad_nodes[:, q]
                    n != n_a && n != n_b && push!(others, n)
                end
                # Are they on row j? (for j=1 they would be on the inner ring)
                if j == 1
                    bad = (others[1] in inner_set) && (others[2] in inner_set)
                else
                    bad = (others[1] == layout[i, j-1] || others[1] == layout[i_next, j-1]) &&
                          (others[2] == layout[i, j-1] || others[2] == layout[i_next, j-1])
                end
                if !bad
                    chosen_quad = q; chosen_side = (others[1], others[2]); break
                end
            end
            chosen_quad == 0 && error("annulus topo: cannot advance at (i=$i, j=$j)")
            # Assign so that the "above-i" node is layout[i, j+1] and "above-i_next" is layout[i_next, j+1].
            # Disambiguate by choosing the one closer to (i, j) than to (i_next, j).
            xa, ya = node_xy[n_a]
            d1_to_a = hypot(node_xy[chosen_side[1]][1] - xa, node_xy[chosen_side[1]][2] - ya)
            d2_to_a = hypot(node_xy[chosen_side[2]][1] - xa, node_xy[chosen_side[2]][2] - ya)
            top_at_i, top_at_inext = d1_to_a < d2_to_a ? (chosen_side[1], chosen_side[2]) :
                                                          (chosen_side[2], chosen_side[1])
            if layout[i, j+1] == 0
                layout[i, j+1] = top_at_i
            elseif layout[i, j+1] != top_at_i
                error("annulus topo: inconsistent top at (i=$i, j=$j+1): " *
                      "old=$(layout[i, j+1]) new=$top_at_i")
            end
            if layout[i_next, j+1] == 0
                layout[i_next, j+1] = top_at_inext
            elseif layout[i_next, j+1] != top_at_inext
                error("annulus topo: inconsistent top at (i=$i_next, j=$j+1): " *
                      "old=$(layout[i_next, j+1]) new=$top_at_inext")
            end
        end
    end
    # Sanity
    all(layout .!= 0) || error("annulus topo: incomplete layout")

    X = [node_xy[layout[i, j]][1] for i in 1:Nξ, j in 1:Nη]
    Y = [node_xy[layout[i, j]][2] for i in 1:Nξ, j in 1:Nη]
    return X, Y
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
