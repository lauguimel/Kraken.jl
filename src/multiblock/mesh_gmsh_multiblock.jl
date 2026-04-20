# =====================================================================
# Multi-block mesh loader from gmsh (v0.3 Phase B.1).
#
# Reads ALL 2D physical surfaces of a `.geo` (which is then meshed with
# `gmsh.model.mesh.generate(2)`) or a pre-meshed `.msh` and produces a
# `MultiBlockMesh2D`: one `Block` per surface, with inter-block
# `Interface`s auto-detected from shared curves or from explicit
# `interface_names` physical-line tags.
#
# Edge tags
# ---------
# Each block has 4 edges (:west, :east, :south, :north) identified by
# their logical indices on the structured layout (ξ = 1, ξ = Nξ, η = 1,
# η = Nη). Each bounding gmsh curve is matched to one of those 4 edges
# by comparing its geometric endpoints to the 4 corner points of the
# loaded layout. The curve's physical group name (if any) becomes the
# edge tag.
#
# - If a curve carries a physical name listed in `interface_names`
#   (default: "interface"), the edge is tagged `:interface`.
# - If the curve is shared between two surfaces (gmsh's natural case
#   for a conforming multi-block mesh) and has no physical name, the
#   edge is tagged `:interface` too.
# - Otherwise the tag is `Symbol(physical_name)`; with no physical
#   name the tag defaults to `:wall`.
#
# Interface detection
# -------------------
# Every edge tagged `:interface` is paired with exactly one mate across
# a different block by spatial colocation of the edge coordinates.
# Both the shared-node (zero offset) and one-cell non-overlap
# topologies are accepted; the sanity check issues a `:warning` on
# shared-node (current `exchange_ghost_2d!` expects non-overlap — see
# `src/multiblock/exchange.jl`).
#
# Block ids
# ---------
# `Symbol(physical_surface_name)` if the surface is in exactly one
# physical surface group with a non-empty name; otherwise
# `Symbol("block_<gmsh_tag>")`.
# =====================================================================

"""
    load_gmsh_multiblock_2d(path; FT=Float64, layout=:auto,
                             interface_names=("interface",), tol=1e-9)
        -> (mbm::MultiBlockMesh2D, groups::GmshPhysicalGroups)

Load every 2D surface from a gmsh file into a `MultiBlockMesh2D`. Each
physical surface becomes a `Block`; inter-block `Interface`s are
auto-detected from shared curves or from curves tagged with a name in
`interface_names`.

Arguments:
- `path`: `.geo` (re-meshed on the fly) or `.msh` file path.
- `FT`: element type for the mesh coordinates (default Float64).
- `layout`: per-surface layout strategy. `:axis_aligned`, `:topological`,
  or `:auto` (try axis-aligned, fall back to topological).
- `interface_names`: tuple of physical-line names that mark an edge as
  a multi-block interface. Default: `("interface",)`.
- `tol`: absolute tolerance used by the corner-matching step.

Returns `(mbm, groups)` where `mbm` carries the blocks + interfaces and
`groups` is the `GmshPhysicalGroups` map of the whole model (same form
as `load_gmsh_mesh_2d`).

The returned `mbm` has NOT been passed through `sanity_check_multiblock`;
run it before launching a simulation.
"""
function load_gmsh_multiblock_2d(path::AbstractString;
                                   FT::Type{<:AbstractFloat}=Float64,
                                   layout::Symbol=:auto,
                                   interface_names=("interface",),
                                   tol::Real=1e-9)
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        if endswith(lowercase(path), ".geo")
            gmsh.open(path)
            gmsh.model.mesh.generate(2)
        else
            gmsh.open(path)
        end
        return _load_multiblock_2d_open(; FT=FT, layout=layout,
                                          interface_names=interface_names,
                                          tol=FT(tol))
    finally
        gmsh.finalize()
    end
end

function _load_multiblock_2d_open(; FT, layout, interface_names, tol)
    groups = _read_physical_groups()

    # curve_tag → physical name (Symbol)
    curve_phys_name = Dict{Int, Symbol}()
    for (name, tag) in groups.by_name
        get(groups.dim, tag, -1) == 1 || continue
        for ent in groups.by_tag[tag]
            curve_phys_name[ent] = Symbol(name)
        end
    end
    # surface_tag → physical name (Symbol)
    surface_phys_name = Dict{Int, Symbol}()
    for (name, tag) in groups.by_name
        get(groups.dim, tag, -1) == 2 || continue
        for ent in groups.by_tag[tag]
            surface_phys_name[ent] = Symbol(name)
        end
    end

    surface_ents = gmsh.model.getEntities(2)
    isempty(surface_ents) &&
        error("load_gmsh_multiblock_2d: no surfaces found in gmsh model")

    # curve_tag → list of surface tags using it (detects shared curves)
    curve_usage = Dict{Int, Vector{Int}}()
    surface_bdry_curves = Dict{Int, Vector{Int}}()
    for (_, stag_raw) in surface_ents
        stag = Int(stag_raw)
        bnd = gmsh.model.getBoundary([(2, stag)], false, false, false)
        curves = Int[Int(ct) for (dim, ct) in bnd if dim == 1]
        surface_bdry_curves[stag] = curves
        for ct in curves
            push!(get!(curve_usage, ct, Int[]), stag)
        end
    end

    interface_names_set = Set{Symbol}(Symbol.(interface_names))

    AT = Matrix{FT}
    blocks = Block{FT, AT}[]
    for (_, stag_raw) in surface_ents
        stag = Int(stag_raw)
        bid = get(surface_phys_name, stag, Symbol("block_$stag"))
        X, Y = _load_surface_layout(stag, layout, FT)
        mesh = CurvilinearMesh(X, Y; type=:gmsh_imported, FT=FT)

        Nξ, Nη = size(X)
        corner_WS = (X[1,  1 ],  Y[1,  1 ])
        corner_ES = (X[Nξ, 1 ],  Y[Nξ, 1 ])
        corner_WN = (X[1,  Nη],  Y[1,  Nη])
        corner_EN = (X[Nξ, Nη],  Y[Nξ, Nη])

        edge_tag = Dict{Symbol, Symbol}(
            :west => :wall, :east => :wall, :south => :wall, :north => :wall,
        )
        for ct in surface_bdry_curves[stag]
            p1, p2 = _curve_endpoint_coords_2d(ct, FT)
            edge_sym = _match_curve_to_edge(p1, p2,
                                              corner_WS, corner_ES,
                                              corner_WN, corner_EN, tol)
            edge_sym === :unknown && continue
            pname = get(curve_phys_name, ct, Symbol())  # empty Symbol if no name
            is_named = pname !== Symbol()
            is_shared = length(curve_usage[ct]) > 1
            tag = if is_named && (pname in interface_names_set)
                INTERFACE_TAG
            elseif !is_named && is_shared
                INTERFACE_TAG
            elseif is_named
                pname
            else
                :wall
            end
            edge_tag[edge_sym] = tag
        end

        blk = Block(bid, mesh;
                     west=edge_tag[:west], east=edge_tag[:east],
                     south=edge_tag[:south], north=edge_tag[:north])
        push!(blocks, blk)
    end

    interfaces = _autodetect_interfaces_2d(blocks, FT(tol))
    mbm = MultiBlockMesh2D(blocks; interfaces=interfaces)
    return mbm, groups
end

# Pick an X, Y array pair for the surface's structured layout.
function _load_surface_layout(surface_tag::Int, layout::Symbol, FT)
    if layout === :axis_aligned
        return _layout_axis_aligned_2d(surface_tag, FT)
    elseif layout === :topological
        return _layout_topological_2d(surface_tag, FT;
                                        periodic_ξ=false, periodic_η=false)
    elseif layout === :auto
        try
            return _layout_axis_aligned_2d(surface_tag, FT)
        catch
            return _layout_topological_2d(surface_tag, FT;
                                            periodic_ξ=false, periodic_η=false)
        end
    else
        error("load_gmsh_multiblock_2d: unknown layout $layout; " *
              "valid = :axis_aligned, :topological, :auto")
    end
end

# Geometric endpoints (2D projection) of curve `curve_tag`.
function _curve_endpoint_coords_2d(curve_tag::Int, FT)
    bnd = gmsh.model.getBoundary([(1, curve_tag)], false, false, false)
    length(bnd) == 2 ||
        error("_curve_endpoint_coords_2d: curve $curve_tag has " *
              "$(length(bnd)) endpoints, expected 2")
    function pt_coord(ptag::Int)
        # After meshing, the point entity has exactly one mesh node.
        _, coord, _ = gmsh.model.mesh.getNodes(0, ptag)
        if length(coord) >= 2
            return (FT(coord[1]), FT(coord[2]))
        end
        # Fallback to parametric geometric evaluation (covers the case of a
        # point that has not been included in the mesh — unusual but safe).
        xyz = gmsh.model.getValue(0, ptag, Float64[])
        return (FT(xyz[1]), FT(xyz[2]))
    end
    return pt_coord(Int(bnd[1][2])), pt_coord(Int(bnd[2][2]))
end

# Match a curve (given by its two endpoints) to one of the 4 logical edges
# of a structured block, defined by its 4 corners. Returns `:unknown` if
# neither endpoint matches any corner.
function _match_curve_to_edge(p1, p2,
                                corner_WS, corner_ES, corner_WN, corner_EN, tol)
    near(a, b) = hypot(a[1] - b[1], a[2] - b[2]) < tol
    pairs = (
        (:west,  corner_WS, corner_WN),
        (:east,  corner_ES, corner_EN),
        (:south, corner_WS, corner_ES),
        (:north, corner_WN, corner_EN),
    )
    for (edge_sym, c_a, c_b) in pairs
        if (near(p1, c_a) && near(p2, c_b)) || (near(p1, c_b) && near(p2, c_a))
            return edge_sym
        end
    end
    return :unknown
end

# Pair every edge tagged :interface with its mate across a different
# block. Matching criterion: geometric colocation (shared-node) or
# one-cell offset (non-overlap). Pair type (east↔west vs same-side
# etc.) depends on the per-block logical ξ/η orientation produced by
# the layout helper, which is fixed by Transfinite Surface corner
# ordering for axis-aligned blocks but arbitrary for topologically-
# walked O-grid blocks. The sanity check further constrains aligned
# orientations for the current `exchange_ghost_2d!` kernel.
function _autodetect_interfaces_2d(blocks, tol)
    edge_list = Tuple{Symbol, Symbol, Vector, Vector}[]
    for b in blocks
        for e in EDGE_SYMBOLS_2D
            getproperty(b.boundary_tags, e) === INTERFACE_TAG || continue
            xs, ys = edge_coords(b, e)
            push!(edge_list, (b.id, e, xs, ys))
        end
    end
    interfaces = Interface[]
    used = Set{Int}()
    for i in 1:length(edge_list)
        (i in used) && continue
        bid_i, edge_i, xs_i, ys_i = edge_list[i]
        best_j = 0
        best_err = Inf
        edge_len = hypot(xs_i[end] - xs_i[1], ys_i[end] - ys_i[1])
        n_cells_i = length(xs_i) - 1
        dx_est = n_cells_i > 0 ? edge_len / n_cells_i : edge_len
        for j in (i+1):length(edge_list)
            (j in used) && continue
            bid_j, edge_j, xs_j, ys_j = edge_list[j]
            bid_j === bid_i && continue
            length(xs_j) == length(xs_i) || continue
            err_aligned = maximum(hypot.(xs_i .- xs_j, ys_i .- ys_j))
            err_flip    = maximum(hypot.(xs_i .- reverse(xs_j),
                                           ys_i .- reverse(ys_j)))
            err = min(err_aligned, err_flip)
            if err < best_err
                best_err = err
                best_j = j
            end
        end
        # Accept shared-node (≈ 0) or up to one-cell offset (non-overlap).
        if best_j != 0 && best_err <= 2 * dx_est + tol
            bid_j, edge_j, _, _ = edge_list[best_j]
            push!(interfaces, Interface(; from=(bid_i, edge_i),
                                          to=(bid_j, edge_j)))
            push!(used, i); push!(used, best_j)
        end
    end
    return interfaces
end
