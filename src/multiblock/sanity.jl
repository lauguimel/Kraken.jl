# =====================================================================
# Sanity check for MultiBlockMesh2D.
#
# Same pattern as `src/sanity_check.jl` (v0.2): each family of
# invariants returns zero or more `SanityIssue` entries; the runner
# aggregates them and the caller decides whether to error or warn.
#
# Families (in evaluation order):
#   1. NoDuplicateBlockIDs           — block ids are unique symbols
#   2. AllEdgeTagsKnown              — each edge tag is a plain Symbol
#   3. InterfaceRefsExist            — every Interface references
#                                       existing block ids and edges
#   4. InterfaceBothEdgesMarked      — both endpoints tagged :interface
#   5. InterfaceEdgesSameLength      — Nξ/Nη match at the two edges
#   6. InterfaceEdgesColocated       — physical (X,Y) coords agree
#                                       within tolerance
#   7. InterfaceOrientationTrivial   — MVP supports only index-aligned
#                                       same-direction exchange
#   8. InterfaceEveryMarkedEdgeUsed  — an edge tagged :interface must
#                                       appear in exactly one Interface
#   9. UniformElementType            — all blocks share the same FT
# =====================================================================

"""
    MultiBlockSanityIssue(severity::Symbol, code::Symbol, block_id::Union{Symbol,Nothing},
                          edge::Union{Symbol,Nothing}, message::String)

One invariant violation reported by `sanity_check_multiblock`.

- `severity`: `:error` (must fix before running) or `:warning`
- `code`: short symbol identifying the check family
- `block_id`, `edge`: locate the offence, `nothing` if global
- `message`: human-readable detail
"""
struct MultiBlockSanityIssue
    severity::Symbol
    code::Symbol
    block_id::Union{Symbol, Nothing}
    edge::Union{Symbol, Nothing}
    message::String
end

function Base.show(io::IO, iss::MultiBlockSanityIssue)
    loc = iss.block_id === nothing ? "" :
          iss.edge === nothing ? " [$(iss.block_id)]" :
          " [$(iss.block_id).$(iss.edge)]"
    print(io, "[$(iss.severity) $(iss.code)]$(loc) $(iss.message)")
end

"""
    sanity_check_multiblock(mbm::MultiBlockMesh2D; tol=1e-9, verbose=true)
        -> Vector{MultiBlockSanityIssue}

Run all invariants on `mbm` and return the collected issues. If
`verbose=true` each issue is printed as it is found. The caller
decides how to react:

```julia
issues = sanity_check_multiblock(mbm)
any(iss -> iss.severity === :error, issues) &&
    error("multi-block mesh has \$(length(issues)) validation issues")
```

`tol` is the absolute coordinate tolerance used by the colocation
check (`InterfaceEdgesColocated`).
"""
function sanity_check_multiblock(mbm::MultiBlockMesh2D;
                                  tol::Real=1e-9, verbose::Bool=true)
    issues = MultiBlockSanityIssue[]
    _check_no_duplicate_block_ids!(issues, mbm)
    _check_edge_tags_known!(issues, mbm)
    _check_uniform_element_type!(issues, mbm)
    _check_interface_refs_exist!(issues, mbm)
    _check_interface_both_edges_marked!(issues, mbm)
    _check_interface_edges_same_length!(issues, mbm)
    _check_interface_edges_colocated!(issues, mbm; tol=tol)
    _check_interface_orientation_trivial!(issues, mbm; tol=tol)
    _check_every_marked_edge_used!(issues, mbm)
    if verbose
        for iss in issues
            println(iss)
        end
        if isempty(issues)
            println("sanity_check_multiblock: OK ($(length(mbm.blocks)) blocks, $(length(mbm.interfaces)) interfaces)")
        end
    end
    return issues
end

# ---- invariant implementations (internal, prefixed _check_) ---------

function _check_no_duplicate_block_ids!(issues, mbm)
    seen = Dict{Symbol, Int}()
    for b in mbm.blocks
        seen[b.id] = get(seen, b.id, 0) + 1
    end
    for (id, n) in seen
        if n > 1
            push!(issues, MultiBlockSanityIssue(:error, :NoDuplicateBlockIDs,
                id, nothing, "block id appears $n times; ids must be unique"))
        end
    end
end

function _check_edge_tags_known!(issues, mbm)
    for b in mbm.blocks
        for e in EDGE_SYMBOLS_2D
            tag = getproperty(b.boundary_tags, e)
            if !(tag isa Symbol)
                push!(issues, MultiBlockSanityIssue(:error, :AllEdgeTagsKnown,
                    b.id, e, "edge tag must be a Symbol; got $(typeof(tag))"))
            end
        end
    end
end

function _check_uniform_element_type!(issues, mbm)
    length(mbm.blocks) ≤ 1 && return
    T0 = eltype(mbm.blocks[1].mesh.X)
    for b in mbm.blocks[2:end]
        T = eltype(b.mesh.X)
        if T !== T0
            push!(issues, MultiBlockSanityIssue(:error, :UniformElementType,
                b.id, nothing,
                "mesh element type $T differs from first block ($T0); " *
                "mixed-precision not supported in v0.3 MVP"))
        end
    end
end

function _check_interface_refs_exist!(issues, mbm)
    for (k, iface) in enumerate(mbm.interfaces)
        for (slot, endpt) in zip((:from, :to), (iface.from, iface.to))
            bid, edge = endpt
            if !haskey(mbm.block_by_id, bid)
                push!(issues, MultiBlockSanityIssue(:error, :InterfaceRefsExist,
                    bid, edge, "interface #$k.$slot references unknown block id $bid"))
            elseif !(edge in EDGE_SYMBOLS_2D)
                push!(issues, MultiBlockSanityIssue(:error, :InterfaceRefsExist,
                    bid, edge, "interface #$k.$slot: unknown edge symbol $edge; " *
                                "valid = $EDGE_SYMBOLS_2D"))
            end
        end
    end
end

function _check_interface_both_edges_marked!(issues, mbm)
    for (k, iface) in enumerate(mbm.interfaces)
        for (slot, endpt) in zip((:from, :to), (iface.from, iface.to))
            bid, edge = endpt
            haskey(mbm.block_by_id, bid) || continue
            (edge in EDGE_SYMBOLS_2D) || continue
            b = getblock(mbm, bid)
            tag = getproperty(b.boundary_tags, edge)
            if tag !== INTERFACE_TAG
                push!(issues, MultiBlockSanityIssue(:error, :InterfaceBothEdgesMarked,
                    bid, edge, "interface #$k.$slot: edge tag is $tag, expected $INTERFACE_TAG"))
            end
        end
    end
end

function _check_interface_edges_same_length!(issues, mbm)
    for (k, iface) in enumerate(mbm.interfaces)
        a_id, a_edge = iface.from
        b_id, b_edge = iface.to
        (haskey(mbm.block_by_id, a_id) && haskey(mbm.block_by_id, b_id)) || continue
        (a_edge in EDGE_SYMBOLS_2D && b_edge in EDGE_SYMBOLS_2D) || continue
        na = edge_length(getblock(mbm, a_id), a_edge)
        nb = edge_length(getblock(mbm, b_id), b_edge)
        if na != nb
            push!(issues, MultiBlockSanityIssue(:error, :InterfaceEdgesSameLength,
                a_id, a_edge,
                "interface #$k: edge length $na ($a_id.$a_edge) ≠ $nb ($b_id.$b_edge)"))
        end
    end
end

function _check_interface_edges_colocated!(issues, mbm; tol)
    # Two valid topologies at an interface:
    # (1) NON-OVERLAP (standard LBM multi-block): A's edge cells and
    #     B's edge cells are 1·dx apart — B's row lies one cell past A's
    #     row. Each block owns DISJOINT physical cells. This is what the
    #     ghost-layer exchange expects: A's east ghost row (one cell past
    #     A's east edge) ← B's west edge interior row.
    # (2) SHARED-NODE: A's east edge nodes and B's west edge nodes are
    #     colocated. The interface is a single line of nodes stored in
    #     both blocks. Exchange must then read from one cell INSIDE the
    #     neighbour (ng + 2), not at the neighbour's edge. NOT supported
    #     by `exchange_ghost_2d!` in its current form.
    # We detect which topology the user built. For (1), we check that the
    # offset equals ONE cell width in the direction normal to the edge;
    # for (2), zero offset (colocated). Anything else is flagged.
    for (k, iface) in enumerate(mbm.interfaces)
        a_id, a_edge = iface.from
        b_id, b_edge = iface.to
        (haskey(mbm.block_by_id, a_id) && haskey(mbm.block_by_id, b_id)) || continue
        (a_edge in EDGE_SYMBOLS_2D && b_edge in EDGE_SYMBOLS_2D) || continue
        Ba = getblock(mbm, a_id); Bb = getblock(mbm, b_id)
        edge_length(Ba, a_edge) == edge_length(Bb, b_edge) || continue
        xa, ya = edge_coords(Ba, a_edge)
        xb, yb = edge_coords(Bb, b_edge)
        # Evaluate both aligned and reversed orientations; the
        # InterfaceOrientationTrivial check handles the flip case
        # separately, but the colocation distance itself must use
        # whichever orientation actually lines up so a flipped-but-
        # shared edge is not reported as geometrically disjoint.
        xb_rev = reverse(xb); yb_rev = reverse(yb)
        err_aligned = maximum(@. sqrt((xa - xb)^2 + (ya - yb)^2))
        err_flip    = maximum(@. sqrt((xa - xb_rev)^2 + (ya - yb_rev)^2))
        max_err = min(err_aligned, err_flip)
        # Accept either 0 (shared-node — warning: exchange not supported)
        # or 1·dx (non-overlap — what exchange_ghost_2d! needs).
        dx_a = (a_edge in (:west, :east)) ?
               (Ba.mesh.X[2, 1] - Ba.mesh.X[1, 1]) :
               (Ba.mesh.Y[1, 2] - Ba.mesh.Y[1, 1])
        dx = abs(dx_a)
        colocated = max_err < tol
        one_cell  = isapprox(max_err, dx; atol=tol, rtol=1e-6)
        if !(colocated || one_cell)
            push!(issues, MultiBlockSanityIssue(:error, :InterfaceEdgesColocated,
                a_id, a_edge,
                "interface #$k: edge physical offset = $(max_err); expected " *
                "either 0 (shared node) or $(dx) (one-cell non-overlap). " *
                "Got neither — check block coordinates."))
        elseif colocated
            push!(issues, MultiBlockSanityIssue(:warning, :InterfaceEdgesColocated,
                a_id, a_edge,
                "interface #$k: shared-node topology (zero offset). " *
                "exchange_ghost_2d! assumes NON-OVERLAP (one-cell offset); " *
                "results will be incorrect unless you restructure the blocks."))
        end
    end
end

function _check_interface_orientation_trivial!(issues, mbm; tol)
    # MVP: we enforce that the two edges run in the same local direction.
    # For west/east: both edges are indexed by j=1..Nη; same direction
    # means ya[1] matches yb[1] (not yb[end]).
    # For south/north: both edges indexed by i=1..Nξ; same direction means
    # xa[1] matches xb[1].
    # If xa[1]/ya[1] matches the OPPOSITE end of edge b, we have a flipped
    # orientation which the current exchange kernel does not support.
    for (k, iface) in enumerate(mbm.interfaces)
        a_id, a_edge = iface.from
        b_id, b_edge = iface.to
        (haskey(mbm.block_by_id, a_id) && haskey(mbm.block_by_id, b_id)) || continue
        (a_edge in EDGE_SYMBOLS_2D && b_edge in EDGE_SYMBOLS_2D) || continue
        Ba = getblock(mbm, a_id); Bb = getblock(mbm, b_id)
        edge_length(Ba, a_edge) == edge_length(Bb, b_edge) || continue
        xa, ya = edge_coords(Ba, a_edge)
        xb, yb = edge_coords(Bb, b_edge)
        d_same = sqrt((xa[1] - xb[1])^2 + (ya[1] - yb[1])^2)
        d_flip = sqrt((xa[1] - xb[end])^2 + (ya[1] - yb[end])^2)
        if d_flip < d_same
            push!(issues, MultiBlockSanityIssue(:error, :InterfaceOrientationTrivial,
                a_id, a_edge,
                "interface #$k: edges appear to run in opposite directions " *
                "(flipped orientation). MVP exchange kernel only supports " *
                "aligned orientation; reverse $b_id.$b_edge or add support."))
        end
    end
end

function _check_every_marked_edge_used!(issues, mbm)
    # Collect every (block_id, edge) that appears in an Interface endpoint.
    used = Set{Tuple{Symbol, Symbol}}()
    for iface in mbm.interfaces
        push!(used, iface.from)
        push!(used, iface.to)
    end
    for b in mbm.blocks, e in EDGE_SYMBOLS_2D
        tag = getproperty(b.boundary_tags, e)
        if tag === INTERFACE_TAG && !((b.id, e) in used)
            push!(issues, MultiBlockSanityIssue(:error, :InterfaceEveryMarkedEdgeUsed,
                b.id, e,
                "edge tagged $INTERFACE_TAG but does not appear in any Interface"))
        end
    end
    # Reverse: an edge that appears in an Interface but is NOT marked
    # :interface is caught by `InterfaceBothEdgesMarked`, so we don't
    # duplicate that check here.
end
