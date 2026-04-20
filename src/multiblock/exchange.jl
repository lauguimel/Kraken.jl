# =====================================================================
# Ghost-cell exchange for a MultiBlockMesh2D (v0.3 MVP).
#
# Protocol (halo strict, no extra ghost rows in the f array):
#
# Each block stores f of size (Nξ × Nη × 9) and runs its normal
# single-block LBM step. At an interface edge, the step's pull for
# populations that would read OUTSIDE the block falls back to the
# usual boundary treatment (clamp / halfway-BB) — those outputs are
# WRONG on purpose. The exchange then immediately overwrites them
# with the valid post-step populations from the NEIGHBOUR block,
# where those same directions streamed from interior cells.
#
# For a west-east interface (block A on left, edge :east; block B on
# right, edge :west):
#
#   cqx > 0  (q ∈ {2, 6, 9}): valid in A (came from A's interior)
#                              → f_B[1, j, q]     ← f_A[Nξ_A, j, q]
#   cqx < 0  (q ∈ {4, 7, 8}): valid in B (came from B's interior)
#                              → f_A[Nξ_A, j, q]  ← f_B[1, j, q]
#   cqx = 0  (q ∈ {1, 3, 5}): both sides already agree by construction;
#                              we re-sync by assignment for robustness.
#
# Same shape for south-north interfaces with cqy instead of cqx.
#
# The API takes a MultiBlockMesh2D and a Vector of f arrays, one per
# block, ordered the same as mbm.blocks. We use `view(...) .= view(...)`
# so this works transparently on CPU, CUDA, and Metal backends.
# =====================================================================

# D2Q9 split by the sign of the x-velocity component
const _Q_CQX_POS = (2, 6, 9)
const _Q_CQX_NEG = (4, 7, 8)
const _Q_CQX_ZERO = (1, 3, 5)

# D2Q9 split by the sign of the y-velocity component
const _Q_CQY_POS = (3, 6, 7)
const _Q_CQY_NEG = (5, 8, 9)
const _Q_CQY_ZERO = (1, 2, 4)

"""
    exchange_ghost_2d!(mbm::MultiBlockMesh2D, f_arrays::Vector{<:AbstractArray})

Run the ghost exchange for every interface declared in `mbm`. Mutates
each block's `f` in place. `f_arrays[i]` must be the populations of
`mbm.blocks[i]`, a `(Nξ × Nη × 9)` array on the same backend as the
block's mesh (CPU, CUDA, Metal — the exchange is a broadcast).

Call sequence per timestep:

    for k in 1:length(mbm.blocks)
        step_kernel!(f_out[k], f_in[k], ...)       # each block's own step
    end
    exchange_ghost_2d!(mbm, f_out)                  # sync interface populations
    apply_physical_bcs!(mbm, f_out, f_in)           # BCs on non-interface edges
    f_in, f_out = f_out, f_in

Does not touch populations away from the interface edge. Does not
average the `c_qx = 0` populations — it copies block A's value into
block B (chosen deterministically for reproducibility).
"""
function exchange_ghost_2d!(mbm::MultiBlockMesh2D,
                             f_arrays::AbstractVector{<:AbstractArray})
    length(f_arrays) == length(mbm.blocks) ||
        error("exchange_ghost_2d!: f_arrays has $(length(f_arrays)) entries, " *
              "mbm has $(length(mbm.blocks)) blocks")
    for iface in mbm.interfaces
        a_idx = mbm.block_by_id[iface.from[1]]
        b_idx = mbm.block_by_id[iface.to[1]]
        a_edge = iface.from[2]
        b_edge = iface.to[2]
        _exchange_pair_2d!(f_arrays[a_idx], f_arrays[b_idx],
                            a_edge, b_edge,
                            mbm.blocks[a_idx].mesh, mbm.blocks[b_idx].mesh)
    end
    return mbm
end

function _exchange_pair_2d!(f_a, f_b,
                             a_edge::Symbol, b_edge::Symbol,
                             mesh_a, mesh_b)
    # Normalise so `left_f` is always the block whose :east edge is the
    # interface and `right_f` is the block whose :west edge is the
    # interface; analogous for south-north. This keeps the inner
    # copy logic in one place.
    if a_edge === :east && b_edge === :west
        _exchange_we_aligned!(f_a, f_b, mesh_a.Nξ, mesh_a.Nη)
    elseif a_edge === :west && b_edge === :east
        _exchange_we_aligned!(f_b, f_a, mesh_b.Nξ, mesh_b.Nη)
    elseif a_edge === :north && b_edge === :south
        _exchange_sn_aligned!(f_a, f_b, mesh_a.Nξ, mesh_a.Nη)
    elseif a_edge === :south && b_edge === :north
        _exchange_sn_aligned!(f_b, f_a, mesh_b.Nξ, mesh_b.Nη)
    else
        error("_exchange_pair_2d!: unsupported edge pair $a_edge → $b_edge. " *
              "MVP supports only opposite-normal aligned pairs " *
              "(east↔west, north↔south).")
    end
end

# west-east interface: `f_left` is the block on the LEFT (its :east
# edge at i=Nξ_left is the interface) and `f_right` is the block on
# the RIGHT (its :west edge at i=1 is the interface). Both blocks
# must have the same Nη at this interface.
function _exchange_we_aligned!(f_left, f_right, Nξ_l::Int, Nη_l::Int)
    # cqx > 0: valid in f_left, copy to f_right
    @inbounds for q in _Q_CQX_POS
        view(f_right, 1, :, q) .= view(f_left, Nξ_l, :, q)
    end
    # cqx < 0: valid in f_right, copy to f_left
    @inbounds for q in _Q_CQX_NEG
        view(f_left, Nξ_l, :, q) .= view(f_right, 1, :, q)
    end
    # cqx = 0: sync to A's value (deterministic)
    @inbounds for q in _Q_CQX_ZERO
        view(f_right, 1, :, q) .= view(f_left, Nξ_l, :, q)
    end
    return nothing
end

# south-north interface: `f_bot` is on the bottom (:north at j=Nη_b)
# and `f_top` is on top (:south at j=1). Both blocks must share Nξ.
function _exchange_sn_aligned!(f_bot, f_top, Nξ_b::Int, Nη_b::Int)
    @inbounds for q in _Q_CQY_POS
        view(f_top, :, 1, q) .= view(f_bot, :, Nη_b, q)
    end
    @inbounds for q in _Q_CQY_NEG
        view(f_bot, :, Nη_b, q) .= view(f_top, :, 1, q)
    end
    @inbounds for q in _Q_CQY_ZERO
        view(f_top, :, 1, q) .= view(f_bot, :, Nη_b, q)
    end
    return nothing
end
