# =====================================================================
# Block reorientation (v0.3 Phase B.2.2).
#
# The topological layout walker in `load_gmsh_multiblock_2d` picks the
# starting corner of each surface arbitrarily, so two neighbouring
# blocks loaded from a shared-curve .geo may end up with flipped or
# same-side edge pairings at their interfaces (e.g. the shared curve is
# labelled `:east` in block A and `:east` in block B, or the η
# direction runs opposite-ways on each side). The current exchange
# kernels (both non-overlap and shared-node) only support aligned
# opposite-normal pairs — `:east↔:west` and `:north↔:south` — so we
# need a preprocessing pass that reshapes the (X, Y, metric) arrays
# and the edge-tag NamedTuple of each block so every declared
# `Interface` becomes canonical.
#
# Two helpers:
#
#   `reorient_block(block; flip_ξ=false, flip_η=false)` — returns a new
#     `Block` whose mesh arrays are flipped along one or both logical
#     axes. Edge tags are permuted accordingly
#     (flip_ξ: west↔east; flip_η: south↔north).
#
#   `autoreorient_blocks(mbm)` — BFS on the interface graph. Each block
#     is tried in the 8 {transpose × flip_ξ × flip_η} variants and the
#     one that makes the parent-interface canonical (opposite-normal,
#     aligned) is chosen. Boundary tags with physical direction
#     semantics (:inlet, :outlet, :wall_bot, :wall_top) are also kept on
#     their canonical logical edges, so an interface-only choice cannot
#     accidentally move an inlet from west to north.
#
# Transpose (swap ξ ↔ η) is not supported in the MVP — it changes the
# block dimensions and requires rebuilding the underlying spline grid.
# For the 4/8-block O-grid used in the paper, consistent Transfinite
# Surface corner ordering in the .geo guarantees that flip_ξ / flip_η
# are sufficient.
# =====================================================================

"""
    reorient_block(block::Block; flip_ξ=false, flip_η=false, transpose=false)
        -> Block

Return a new `Block` whose mesh has been transformed by the requested
combination of flips and/or a transpose (ξ↔η swap). Both mesh arrays
(`X, Y`, metric, Jacobian) and the `boundary_tags` NamedTuple are
updated consistently. The underlying spline-based metric is recomputed
on the transformed node coordinates.

Operations (applied in order: transpose → flip_ξ → flip_η):

- `transpose=true` : `X_new[i, j] = X[j, i]` — swap logical axes. Tags
  rotate `west ↔ south` and `east ↔ north`. Changes the block shape
  from `(Nξ, Nη)` to `(Nη, Nξ)`.
- `flip_ξ=true`    : mirror along ξ. Tags rotate `west ↔ east`.
- `flip_η=true`    : mirror along η. Tags rotate `south ↔ north`.

Combinations span the full dihedral group of 8 orientations, which
covers every per-block reorientation the multi-block loader can
produce from the gmsh topological walker.
"""
function reorient_block(block::Block{T, AT};
                         flip_ξ::Bool=false, flip_η::Bool=false,
                         transpose::Bool=false) where {T, AT}
    (!flip_ξ && !flip_η && !transpose) && return block
    mesh = block.mesh
    Nξ_src, Nη_src = mesh.Nξ, mesh.Nη
    if transpose
        Nξ_new, Nη_new = Nη_src, Nξ_src
    else
        Nξ_new, Nη_new = Nξ_src, Nη_src
    end
    X_new = Matrix{T}(undef, Nξ_new, Nη_new)
    Y_new = Matrix{T}(undef, Nξ_new, Nη_new)
    @inbounds for j in 1:Nη_new, i in 1:Nξ_new
        i0, j0 = transpose ? (j, i) : (i, j)
        ii = flip_ξ ? (Nξ_src + 1 - i0) : i0
        jj = flip_η ? (Nη_src + 1 - j0) : j0
        # After transpose, flip_ξ acts on the NEW ξ (= old η), so swap
        # the interpretation: when transpose=true, flip_ξ means flip the
        # ORIGINAL η direction, and flip_η the ORIGINAL ξ.
        if transpose
            ii_orig = flip_η ? (Nξ_src + 1 - i0) : i0
            jj_orig = flip_ξ ? (Nη_src + 1 - j0) : j0
            X_new[i, j] = mesh.X[ii_orig, jj_orig]
            Y_new[i, j] = mesh.Y[ii_orig, jj_orig]
        else
            X_new[i, j] = mesh.X[ii, jj]
            Y_new[i, j] = mesh.Y[ii, jj]
        end
    end
    new_mesh = CurvilinearMesh(X_new, Y_new;
                                 periodic_ξ=mesh.periodic_ξ,
                                 periodic_η=mesh.periodic_η,
                                 type=mesh.type, FT=T)
    # Permute tags: transpose first (swaps w↔s, e↔n), then flips.
    w, e, s, n = block.boundary_tags.west, block.boundary_tags.east,
                 block.boundary_tags.south, block.boundary_tags.north
    if transpose
        w, s = s, w
        e, n = n, e
    end
    if flip_ξ
        w, e = e, w
    end
    if flip_η
        s, n = n, s
    end
    return Block(block.id, new_mesh; west=w, east=e, south=s, north=n)
end

const _DEFAULT_PHYSICAL_TAG_EDGES = Dict{Symbol, Symbol}(
    :inlet => :west,
    :outlet => :east,
    :wall_bot => :south,
    :wall_bottom => :south,
    :bottom => :south,
    :wall_top => :north,
    :wall_upper => :north,
    :top => :north,
)

"""
    autoreorient_blocks(mbm::MultiBlockMesh2D; verbose=false,
                        respect_physical_tags=true)
        -> MultiBlockMesh2D

Walk the interface graph and orient each block so that every traversed
`Interface` ends up as an aligned opposite-normal pair
(`:east ↔ :west` or `:north ↔ :south`, with the edge coordinates
running in the same direction on both sides).

When `respect_physical_tags=true` (default), tags with established
global direction semantics are constrained to their canonical logical
edges:

- `:inlet` → `:west`
- `:outlet` → `:east`
- `:wall_bot`, `:wall_bottom`, `:bottom` → `:south`
- `:wall_top`, `:wall_upper`, `:top` → `:north`

This avoids a subtle failure mode where the interface is canonical but
a physical inlet/outlet migrated to a north/south logical edge, causing
the boundary-condition rebuild to impose the wrong velocity component.

The block indices and `id` symbols are preserved; only the underlying
mesh arrays and `boundary_tags` of non-root blocks may change.
`Interface` entries are left untouched — they still reference the
same `(block_id, edge_symbol)` pairs, but after reorientation those
edge symbols point to the newly-aligned edges.

Returns a brand-new `MultiBlockMesh2D` (the input is not mutated).
Raises `ErrorException` if no consistent flip assignment exists (the
topology is genuinely twisted and needs a transpose or human review).
"""
function autoreorient_blocks(mbm::MultiBlockMesh2D; verbose::Bool=false,
                              respect_physical_tags::Bool=true,
                              physical_tag_edges=_DEFAULT_PHYSICAL_TAG_EDGES)
    blocks = collect(mbm.blocks)
    n_blocks = length(blocks)
    n_blocks == 0 && return mbm
    orient_options = [(fξ, fη, tr) for tr in (false, true)
                                   for fη in (false, true)
                                   for fξ in (false, true)]
    # Per-block cumulative orientation state (flip_ξ, flip_η, transpose).
    # (false, false, false) = identity. Applied in order:
    # transpose → flip_ξ → flip_η, matching reorient_block.
    orient_state = fill((false, false, false), n_blocks)
    root_orient = _choose_root_orientation(blocks[1], orient_options;
                                           respect_physical_tags=respect_physical_tags,
                                           physical_tag_edges=physical_tag_edges)
    if root_orient !== (false, false, false)
        fξ, fη, tr = root_orient
        verbose && println("  autoreorient: root block :$(blocks[1].id) orient=$(root_orient)")
        blocks[1] = reorient_block(blocks[1]; flip_ξ=fξ, flip_η=fη, transpose=tr)
        orient_state[1] = root_orient
    end
    visited = falses(n_blocks)
    visited[1] = true
    queue = [1]
    while !isempty(queue)
        a_idx = popfirst!(queue)
        a_id = blocks[a_idx].id
        for iface in mbm.interfaces
            other = _other_endpoint(iface, a_id)
            other === nothing && continue
            b_id, b_edge_original = other
            b_idx = get(mbm.block_by_id, b_id, 0)
            b_idx == 0 && continue
            visited[b_idx] && continue
            a_edge_original = (iface.from[1] === a_id ? iface.from[2] : iface.to[2])
            a_edge_current = _apply_orient_to_edge(a_edge_original, orient_state[a_idx])
            best = nothing
            for (fξ, fη, tr) in orient_options
                candidate_b = reorient_block(blocks[b_idx];
                                               flip_ξ=fξ, flip_η=fη, transpose=tr)
                if respect_physical_tags &&
                   !_physical_tags_aligned(candidate_b, physical_tag_edges)
                    continue
                end
                b_edge_after = _apply_orient_to_edge(b_edge_original, (fξ, fη, tr))
                ok = _is_canonical_pair(blocks[a_idx], a_edge_current,
                                          candidate_b, b_edge_after)
                if ok
                    best = (candidate_b, (fξ, fη, tr), b_edge_after)
                    verbose && println("  autoreorient: $(b_id) orient=$((fξ, fη, tr)) " *
                                        "→ $(a_id).$a_edge_current ↔ $(b_id).$b_edge_after canonical")
                    break
                end
            end
            best === nothing &&
                error("autoreorient_blocks: no consistent orientation for block :$b_id " *
                      "relative to :$a_id via interface " *
                      "($a_id.$a_edge_original ↔ $b_id.$b_edge_original). " *
                      "Tried all 8 flips × transpose combinations.")
            blocks[b_idx] = best[1]
            orient_state[b_idx] = best[2]
            visited[b_idx] = true
            push!(queue, b_idx)
        end
    end
    any(.!visited) &&
        @warn "autoreorient_blocks: $(sum(.!visited)) block(s) not reached by BFS"
    updated_ifaces = Interface[]
    for iface in mbm.interfaces
        from_id, from_edge = iface.from
        to_id,   to_edge   = iface.to
        from_idx = mbm.block_by_id[from_id]
        to_idx   = mbm.block_by_id[to_id]
        push!(updated_ifaces,
              Interface(; from=(from_id, _apply_orient_to_edge(from_edge, orient_state[from_idx])),
                          to=(to_id,   _apply_orient_to_edge(to_edge,   orient_state[to_idx]))))
    end
    return MultiBlockMesh2D(blocks; interfaces=updated_ifaces)
end

function _physical_tags_aligned(block::Block, physical_tag_edges)
    for edge in EDGE_SYMBOLS_2D
        tag = getproperty(block.boundary_tags, edge)
        preferred = get(physical_tag_edges, tag, nothing)
        preferred === nothing && continue
        preferred === edge || return false
    end
    return true
end

function _has_physical_tag_constraints(block::Block, physical_tag_edges)
    for edge in EDGE_SYMBOLS_2D
        haskey(physical_tag_edges, getproperty(block.boundary_tags, edge)) && return true
    end
    return false
end

function _choose_root_orientation(block::Block, orient_options;
                                  respect_physical_tags::Bool,
                                  physical_tag_edges)
    if respect_physical_tags && _has_physical_tag_constraints(block, physical_tag_edges)
        for (fξ, fη, tr) in orient_options
            candidate = reorient_block(block; flip_ξ=fξ, flip_η=fη, transpose=tr)
            minimum(candidate.mesh.J) > 0 || continue
            _physical_tags_aligned(candidate, physical_tag_edges) || continue
            return (fξ, fη, tr)
        end
        error("autoreorient_blocks: no positive-J orientation of root block " *
              ":$(block.id) satisfies physical boundary tag directions")
    end

    minimum(block.mesh.J) > 0 && return (false, false, false)
    for (fξ, fη, tr) in orient_options
        candidate = reorient_block(block; flip_ξ=fξ, flip_η=fη, transpose=tr)
        minimum(candidate.mesh.J) > 0 && return (fξ, fη, tr)
    end
    error("autoreorient_blocks: no positive-J orientation for root block :$(block.id)")
end

# Apply a (flip_ξ, flip_η, transpose) transformation to an edge symbol.
# Operation order matches reorient_block: transpose first, then flips.
function _apply_orient_to_edge(edge::Symbol, orient::Tuple{Bool, Bool, Bool})
    fξ, fη, tr = orient
    if tr
        edge = edge === :west  ? :south :
               edge === :south ? :west  :
               edge === :east  ? :north :
               edge === :north ? :east  : edge
    end
    if fξ
        edge = edge === :west ? :east : (edge === :east ? :west : edge)
    end
    if fη
        edge = edge === :south ? :north : (edge === :north ? :south : edge)
    end
    return edge
end

# Check whether the edge pair (a.edge_a, b.edge_b) is opposite-normal
# AND has aligned (non-flipped) coordinates.
function _is_canonical_pair(a::Block, edge_a::Symbol, b::Block, edge_b::Symbol)
    opp = Dict(:east => :west, :west => :east, :south => :north, :north => :south)
    get(opp, edge_a, :_) === edge_b || return false
    xs_a, ys_a = edge_coords(a, edge_a)
    xs_b, ys_b = edge_coords(b, edge_b)
    length(xs_a) == length(xs_b) || return false
    err_aligned = maximum(@. sqrt((xs_a - xs_b)^2 + (ys_a - ys_b)^2))
    xs_r = reverse(xs_b); ys_r = reverse(ys_b)
    err_flipped = maximum(@. sqrt((xs_a - xs_r)^2 + (ys_a - ys_r)^2))
    return err_aligned ≤ err_flipped + 1e-12
end

# Return (bid, edge) of the other endpoint of `iface` if it touches
# block `query_id`, else nothing.
function _other_endpoint(iface::Interface, query_id::Symbol)
    if iface.from[1] === query_id
        return iface.to
    elseif iface.to[1] === query_id
        return iface.from
    end
    return nothing
end

const _TRANSPOSE_FACE = Dict(:west => :south, :east => :north,
                              :south => :west, :north => :east)

"""
    transpose_multiblock(mbm::MultiBlockMesh2D) -> MultiBlockMesh2D

Transpose every block (swap ξ↔η) and update interface face labels
accordingly (west↔south, east↔north). Useful after `autoreorient_blocks`
to move the outer boundary from north to east on O-grid topologies.
"""
function transpose_multiblock(mbm::MultiBlockMesh2D)
    new_blocks = [reorient_block(b; transpose=true) for b in mbm.blocks]
    new_ifaces = map(mbm.interfaces) do iface
        f1 = (iface.from[1], _TRANSPOSE_FACE[iface.from[2]])
        f2 = (iface.to[1],   _TRANSPOSE_FACE[iface.to[2]])
        Interface(from=f1, to=f2)
    end
    return MultiBlockMesh2D(new_blocks; interfaces=new_ifaces)
end
