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
    # Per-block cumulative orientation state (flip_ξ, flip_η, transpose).
    # (false, false, false) = identity. Applied in order:
    # transpose → flip_ξ → flip_η, matching reorient_block.
    orient_state = fill((false, false, false), n_blocks)
    # Force root block to have a positive Jacobian so validate_mesh
    # accepts the extended mesh downstream. Left-handed parameterisations
    # (J<0 everywhere) would be CONSISTENT on the interior but cause
    # sign-change errors after mesh extension on curved boundaries. We
    # flip ξ if the root's interior Jacobian (via finite differences at
    # (i=2, j=2)) is negative; this orients the block so ξ-η is
    # right-handed, which then propagates through the BFS to every
    # downstream block via canonical interface pairing.
    _J_at_22(blk) = begin
        m = blk.mesh
        m.Nξ < 3 || m.Nη < 3 ? zero(eltype(m.X)) : begin
            dXξ = (m.X[3,2] - m.X[1,2]) / 2
            dYξ = (m.Y[3,2] - m.Y[1,2]) / 2
            dXη = (m.X[2,3] - m.X[2,1]) / 2
            dYη = (m.Y[2,3] - m.Y[2,1]) / 2
            dXξ * dYη - dXη * dYξ
        end
    end
    if _J_at_22(blocks[1]) < 0
        verbose && println("  autoreorient: root block :$(blocks[1].id) has J<0, " *
                             "applying transpose to get right-handed orientation " *
                             "(also migrates tags from south/north to west/east)")
        blocks[1] = reorient_block(blocks[1]; transpose=true)
        orient_state[1] = (false, false, true)
    end
    visited = falses(n_blocks)
    visited[1] = true
    queue = [1]
    orient_options = [(fξ, fη, tr) for tr in (false, true)
                                   for fη in (false, true)
                                   for fξ in (false, true)]
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

