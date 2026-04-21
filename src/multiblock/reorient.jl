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
#   `autoreorient_blocks(mbm)` — BFS on the interface graph: the first
#     block is kept as-is, each subsequent block is tried in the 4
#     {identity, flip_ξ, flip_η, flip_both} variants and the one that
#     makes the parent-interface canonical (opposite-normal, aligned)
#     is chosen. When several already-visited blocks constrain a
#     target, we pick the first consistent flip and leave downstream
#     mismatches for the sanity check to report.
#
# Transpose (swap ξ ↔ η) is not supported in the MVP — it changes the
# block dimensions and requires rebuilding the underlying spline grid.
# For the 4/8-block O-grid used in the paper, consistent Transfinite
# Surface corner ordering in the .geo guarantees that flip_ξ / flip_η
# are sufficient.
# =====================================================================

"""
    reorient_block(block::Block; flip_ξ=false, flip_η=false) -> Block

Return a new `Block` whose mesh has been flipped along the requested
logical axes. Both mesh arrays (`X, Y`, metric, Jacobian) and the
`boundary_tags` NamedTuple are updated consistently. The underlying
spline-based metric is recomputed on the flipped node coordinates so
the resulting mesh is fully self-consistent.

`flip_ξ=true` maps `X_new[i, j] = X[Nξ + 1 - i, j]`
(and similarly for `Y`); edge tags rotate `west ↔ east`.
`flip_η=true` maps `X_new[i, j] = X[i, Nη + 1 - j]`; edge tags rotate
`south ↔ north`.

Apply `flip_ξ=true, flip_η=true` for a 180° rotation.
"""
function reorient_block(block::Block{T, AT};
                         flip_ξ::Bool=false, flip_η::Bool=false) where {T, AT}
    (!flip_ξ && !flip_η) && return block
    mesh = block.mesh
    Nξ, Nη = mesh.Nξ, mesh.Nη
    # Start from the plain (X, Y) so the spline refit in
    # `CurvilinearMesh(X, Y; ...)` produces a consistent metric.
    X_new = Matrix{T}(undef, Nξ, Nη)
    Y_new = Matrix{T}(undef, Nξ, Nη)
    @inbounds for j in 1:Nη, i in 1:Nξ
        ii = flip_ξ ? (Nξ + 1 - i) : i
        jj = flip_η ? (Nη + 1 - j) : j
        X_new[i, j] = mesh.X[ii, jj]
        Y_new[i, j] = mesh.Y[ii, jj]
    end
    new_mesh = CurvilinearMesh(X_new, Y_new;
                                 periodic_ξ=mesh.periodic_ξ,
                                 periodic_η=mesh.periodic_η,
                                 type=mesh.type, FT=T)
    # Permute edge tags
    w, e, s, n = block.boundary_tags.west, block.boundary_tags.east,
                 block.boundary_tags.south, block.boundary_tags.north
    if flip_ξ
        w, e = e, w
    end
    if flip_η
        s, n = n, s
    end
    return Block(block.id, new_mesh; west=w, east=e, south=s, north=n)
end

"""
    autoreorient_blocks(mbm::MultiBlockMesh2D; verbose=false)
        -> MultiBlockMesh2D

Walk the interface graph starting from the first block (kept as-is)
and flip every subsequent block so that every traversed `Interface`
ends up as an aligned opposite-normal pair
(`:east ↔ :west` or `:north ↔ :south`, with the edge coordinates
running in the same direction on both sides).

The block indices and `id` symbols are preserved; only the underlying
mesh arrays and `boundary_tags` of non-root blocks may change.
`Interface` entries are left untouched — they still reference the
same `(block_id, edge_symbol)` pairs, but after reorientation those
edge symbols point to the newly-aligned edges.

Returns a brand-new `MultiBlockMesh2D` (the input is not mutated).
Raises `ErrorException` if no consistent flip assignment exists (the
topology is genuinely twisted and needs a transpose or human review).
"""
function autoreorient_blocks(mbm::MultiBlockMesh2D; verbose::Bool=false)
    blocks = collect(mbm.blocks)
    n_blocks = length(blocks)
    n_blocks == 0 && return mbm
    # Per-block cumulative flip state (false, false) = identity. When
    # a block is flipped, its edge labels in all interfaces must be
    # remapped through this state to resolve "which curve is this
    # interface talking about" after the flip.
    flip_state = fill((false, false), n_blocks)
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
            # Resolve a's edge through its cumulative flip state so we
            # know which edge of the CURRENT a_block the interface is on.
            a_edge_current = _apply_flip_to_edge(a_edge_original, flip_state[a_idx])
            # Try the 4 flips of b. The check must compare:
            #   * a's edge on CURRENT a (already flipped)
            #   * b's edge on CANDIDATE b (blocks[b_idx] already has any
            #     prior flip baked in — in a BFS tree it has none yet)
            # b_edge_original refers to the ORIGINAL b's labelling.
            best = nothing
            for (fξ, fη) in ((false, false), (true, false), (false, true), (true, true))
                candidate_b = reorient_block(blocks[b_idx]; flip_ξ=fξ, flip_η=fη)
                # After the tentative flip, the original edge label maps to:
                b_edge_after = _apply_flip_to_edge(b_edge_original, (fξ, fη))
                ok = _is_canonical_pair(blocks[a_idx], a_edge_current,
                                          candidate_b, b_edge_after)
                if ok
                    best = (candidate_b, (fξ, fη), b_edge_after)
                    verbose && println("  autoreorient: $(b_id) flip=$((fξ, fη)) " *
                                        "→ $(a_id).$a_edge_current ↔ $(b_id).$b_edge_after canonical")
                    break
                end
            end
            best === nothing &&
                error("autoreorient_blocks: no consistent flip for block :$b_id " *
                      "relative to :$a_id via interface " *
                      "($a_id.$a_edge_original ↔ $b_id.$b_edge_original). " *
                      "Mesh may need a transpose (not MVP-supported) or " *
                      "manual reorient_block.")
            blocks[b_idx] = best[1]
            flip_state[b_idx] = best[2]
            visited[b_idx] = true
            push!(queue, b_idx)
        end
    end
    any(.!visited) &&
        @warn "autoreorient_blocks: $(sum(.!visited)) block(s) not reached by BFS"
    # Update every interface's edge labels through the final flip_state.
    updated_ifaces = Interface[]
    for iface in mbm.interfaces
        from_id, from_edge = iface.from
        to_id,   to_edge   = iface.to
        from_idx = mbm.block_by_id[from_id]
        to_idx   = mbm.block_by_id[to_id]
        push!(updated_ifaces,
              Interface(; from=(from_id, _apply_flip_to_edge(from_edge, flip_state[from_idx])),
                          to=(to_id,   _apply_flip_to_edge(to_edge,   flip_state[to_idx]))))
    end
    return MultiBlockMesh2D(blocks; interfaces=updated_ifaces)
end

# Apply a (flip_ξ, flip_η) transformation to an edge symbol. flip_ξ
# swaps west ↔ east; flip_η swaps south ↔ north.
function _apply_flip_to_edge(edge::Symbol, flip::Tuple{Bool, Bool})
    fξ, fη = flip
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

