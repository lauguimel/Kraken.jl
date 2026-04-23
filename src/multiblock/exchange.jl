# =====================================================================
# Ghost-cell exchange on extended block arrays (Phase A.5b).
#
# Each block stores f on an extended grid (Nξ_phys + 2Ng, Nη_phys +
# 2Ng, 9). The physical interior lives at indices (Ng+1 .. Ng+Nξ_phys,
# Ng+1 .. Ng+Nη_phys); ghost rows wrap it on all 4 sides.
#
# PROTOCOL — pre-step fill, bit-exact to single-block
# ---------------------------------------------------
# Before each timestep, `exchange_ghost_2d!` walks the interface list
# and copies 1 row/column of ghost data from the neighbour block's
# physical interior, one cell deep from the shared interface.
#
# For a west-east interface (A on left, :east; B on right, :west):
#   A's east ghost at i = Nξ_phys_A + Ng + 1 ← B's interior at i = Ng + 1
#   B's west ghost at i = Ng                 ← A's interior at i = Nξ_phys_A + Ng
#
# With Ng = 1 this simplifies to two rank-2 copies per interface:
#   A[Nξ_phys_A + 2, :, :] ← B[2, :, :]
#   B[1, :, :]             ← A[Nξ_phys_A + 1, :, :]
#
# Analogous for south-north with j instead of i.
#
# After the exchange, the step kernel reads from these ghost rows
# when its pull would cross the interface — no halfway-BB fallback
# fires on a physical interior-boundary cell. Post-step, ghost-row
# contents in `f_out` are garbage (the kernel's halfway-BB fired at
# the extended array's outer edges) but are irrelevant: the next
# iteration's `exchange_ghost_2d!` overwrites them before the next
# step reads them.
#
# The API takes a `MultiBlockMesh2D` and a vector of `BlockState2D`,
# one per block, ordered like `mbm.blocks`. All blocks must share the
# same n_ghost. Broadcasting via `view(...) .= view(...)` keeps this
# backend-agnostic (CPU / CUDA / Metal).
# =====================================================================

"""
    exchange_ghost_2d!(mbm::MultiBlockMesh2D,
                        states::AbstractVector{<:BlockState2D})

Fill the ghost rows of each block's `state.f` from the physical
interior of its neighbour across every declared interface. Must be
called BEFORE the step kernel (reads `state.f` as `f_in`).

Precondition: all blocks share the same `n_ghost`.

Call sequence per timestep:

    exchange_ghost_2d!(mbm, states)                   # fill ghosts
    for (k, blk) in enumerate(mbm.blocks)
        Nx_ext, Ny_ext = ext_dims(states[k])
        fused_bgk_step!(f_out[k], states[k].f, ...,   # step on extended size
                          states[k].ρ, states[k].ux, states[k].uy,
                          is_solid_ext[k], Nx_ext, Ny_ext, ω)
    end
    for (k, blk) in enumerate(mbm.blocks)
        apply_bc_rebuild_2d!(interior_f(f_out[k]),    # physical BCs on interior view
                              interior_f(states[k].f), bcspec[k], ν,
                              states[k].Nξ_phys, states[k].Nη_phys)
    end
    # swap states[k].f ↔ f_out[k]  (handled by caller)
"""
function exchange_ghost_2d!(mbm::MultiBlockMesh2D,
                             states::AbstractVector{<:BlockState2D})
    length(states) == length(mbm.blocks) ||
        error("exchange_ghost_2d!: states has $(length(states)) entries, " *
              "mbm has $(length(mbm.blocks)) blocks")
    # Consistency: all blocks must share n_ghost
    ng = states[1].n_ghost
    for k in 2:length(states)
        states[k].n_ghost == ng ||
            error("exchange_ghost_2d!: n_ghost mismatch (block 1: $ng, block $k: $(states[k].n_ghost))")
    end
    for iface in mbm.interfaces
        a_idx = mbm.block_by_id[iface.from[1]]
        b_idx = mbm.block_by_id[iface.to[1]]
        _exchange_pair_ghost_2d!(states[a_idx], states[b_idx],
                                  iface.from[2], iface.to[2])
    end
    return mbm
end

function _exchange_pair_ghost_2d!(state_a::BlockState2D, state_b::BlockState2D,
                                    edge_a::Symbol, edge_b::Symbol)
    if edge_a === :east && edge_b === :west
        _fill_ghost_we!(state_a, state_b)
    elseif edge_a === :west && edge_b === :east
        _fill_ghost_we!(state_b, state_a)
    elseif edge_a === :north && edge_b === :south
        _fill_ghost_sn!(state_a, state_b)
    elseif edge_a === :south && edge_b === :north
        _fill_ghost_sn!(state_b, state_a)
    else
        error("_exchange_pair_ghost_2d!: unsupported edge pair $edge_a → $edge_b. " *
              "MVP supports only opposite-normal aligned pairs " *
              "(east↔west, north↔south).")
    end
end

# West-east: state_left is the LEFT block (its :east edge is the
# interface), state_right is the RIGHT block (its :west edge is the
# interface). Both must share Nη_phys and n_ghost.
function _fill_ghost_we!(state_left::BlockState2D, state_right::BlockState2D)
    ng = state_left.n_ghost
    Nx_l = state_left.Nξ_phys
    Ny_phys = state_left.Nη_phys
    # East-ghost of LEFT ← west-interior of RIGHT
    #   LEFT[Nx_l + ng + k, :, :]  ←  RIGHT[ng + k, :, :]   for k = 1..ng
    # West-ghost of RIGHT ← east-interior of LEFT
    #   RIGHT[k, :, :]  ←  LEFT[Nx_l + k, :, :]              for k = 1..ng
    # Both blocks' physical rows in η are (ng+1 .. ng+Ny_phys).
    j_range = (ng + 1):(ng + Ny_phys)
    @inbounds for k in 1:ng
        view(state_left.f,  Nx_l + ng + k, j_range, :) .=
            view(state_right.f, ng + k,     j_range, :)
        view(state_right.f, k,              j_range, :) .=
            view(state_left.f,  Nx_l + k,   j_range, :)
    end
    return nothing
end

# South-north: state_bot is the BOTTOM block (:north edge is interface),
# state_top is the TOP block (:south edge is interface).
function _fill_ghost_sn!(state_bot::BlockState2D, state_top::BlockState2D)
    ng = state_bot.n_ghost
    Ny_b = state_bot.Nη_phys
    Nx_phys = state_bot.Nξ_phys
    i_range = (ng + 1):(ng + Nx_phys)
    @inbounds for k in 1:ng
        view(state_bot.f, i_range, Ny_b + ng + k, :) .=
            view(state_top.f, i_range, ng + k,    :)
        view(state_top.f, i_range, k,              :) .=
            view(state_bot.f, i_range, Ny_b + k,  :)
    end
    return nothing
end

# =====================================================================
# Shared-node variant (Phase B.2).
#
# Natural gmsh multi-block produces SHARED-NODE topology: adjacent
# surfaces share the boundary curve, so each block stores the same
# interface column. The non-overlap exchange above fetches the
# neighbour's interior row 1 cell past the shared edge; for shared-
# node, the neighbour's 1-cell-past-edge column is at interior index
# (ng + 2) rather than (ng + 1), because (ng + 1) is the duplicated
# shared column.
#
# Both blocks keep computing their step on the shared column; after
# each step the duplicated column values in A and B can drift
# slightly (they read different ghosts at the previous step). In
# practice this is tolerated at the interface — the shared column is
# essentially a redundant dof that is refreshed by both sides' steps
# plus ghost fill. For a bit-exact match vs. single-block you would
# additionally copy A's shared column into B's after the swap, but we
# don't enforce that here: the MVP target is convergence to a
# physically correct solution, not bit-exactness.
# =====================================================================

"""
    exchange_ghost_shared_node_2d!(mbm::MultiBlockMesh2D,
                                    states::AbstractVector{<:BlockState2D})

Ghost-fill for the SHARED-NODE topology (gmsh's natural output: two
adjacent blocks store the same interface column). Read the neighbour's
interior one column PAST the shared column (index `ng + 1 + k` instead
of `ng + k`), so the ghost row represents the physical cell 1·dx
beyond the interface in the normal direction, just as the non-overlap
exchange does. Call sequence is identical to `exchange_ghost_2d!`.
"""
function exchange_ghost_shared_node_2d!(mbm::MultiBlockMesh2D,
                                          states::AbstractVector{<:BlockState2D})
    length(states) == length(mbm.blocks) ||
        error("exchange_ghost_shared_node_2d!: states has $(length(states)) entries, " *
              "mbm has $(length(mbm.blocks)) blocks")
    ng = states[1].n_ghost
    for k in 2:length(states)
        states[k].n_ghost == ng ||
            error("exchange_ghost_shared_node_2d!: n_ghost mismatch")
    end
    for iface in mbm.interfaces
        a_idx = mbm.block_by_id[iface.from[1]]
        b_idx = mbm.block_by_id[iface.to[1]]
        _exchange_pair_ghost_shared_node_2d!(states[a_idx], states[b_idx],
                                               iface.from[2], iface.to[2])
    end
    return mbm
end

function _exchange_pair_ghost_shared_node_2d!(state_a::BlockState2D, state_b::BlockState2D,
                                                edge_a::Symbol, edge_b::Symbol)
    if edge_a === :east && edge_b === :west
        _fill_ghost_we_shared!(state_a, state_b)
    elseif edge_a === :west && edge_b === :east
        _fill_ghost_we_shared!(state_b, state_a)
    elseif edge_a === :north && edge_b === :south
        _fill_ghost_sn_shared!(state_a, state_b)
    elseif edge_a === :south && edge_b === :north
        _fill_ghost_sn_shared!(state_b, state_a)
    else
        error("_exchange_pair_ghost_shared_node_2d!: unsupported edge pair " *
              "$edge_a → $edge_b. MVP supports only opposite-normal aligned pairs.")
    end
end

function _fill_ghost_we_shared!(state_left::BlockState2D, state_right::BlockState2D)
    ng = state_left.n_ghost
    Nx_l = state_left.Nξ_phys
    Ny_phys = state_left.Nη_phys
    j_range = (ng + 1):(ng + Ny_phys)
    # East-ghost of LEFT ← right block one cell past the shared column
    # (skip the duplicate at ng + 1 → read from ng + 1 + k).
    # West-ghost of RIGHT ← left block one cell before the shared column
    # (skip the duplicate at Nx_l + ng → read from Nx_l + ng - 1 - (k-1)).
    @inbounds for k in 1:ng
        view(state_left.f,  Nx_l + ng + k, j_range, :) .=
            view(state_right.f, ng + 1 + k,     j_range, :)
        view(state_right.f, k,              j_range, :) .=
            view(state_left.f,  Nx_l + ng - 1 - (ng - k), j_range, :)
    end
    return nothing
end

function _fill_ghost_sn_shared!(state_bot::BlockState2D, state_top::BlockState2D)
    ng = state_bot.n_ghost
    Ny_b = state_bot.Nη_phys
    Nx_phys = state_bot.Nξ_phys
    i_range = (ng + 1):(ng + Nx_phys)
    @inbounds for k in 1:ng
        view(state_bot.f, i_range, Ny_b + ng + k, :) .=
            view(state_top.f, i_range, ng + 1 + k, :)
        view(state_top.f, i_range, k,              :) .=
            view(state_bot.f, i_range, Ny_b + ng - 1 - (ng - k), :)
    end
    return nothing
end

# =====================================================================
# Ghost-corner fill (Phase B.2+).
#
# After edge exchange, the 4 corner ghost cells of each block's extended
# array are still unfilled. For D2Q9, diagonal populations (q=6,7,8,9)
# at a physical corner cell pull from these ghost corners. Unfilled
# corners contain stale data from the previous step's kernel (which
# wrote BB garbage there), causing exponential growth at interface
# intersections.
#
# Fix: extrapolate each corner from the adjacent ghost edge cell
# (zero-th order copy). This is called AFTER edge exchange so the
# ghost edges already contain correct neighbor data.
# =====================================================================

"""
    fill_ghost_corners_2d!(mbm::MultiBlockMesh2D,
                            states::AbstractVector{<:BlockState2D})

Fill the 4 ghost-corner cells of each block by copying from the nearest
ghost-edge cell. Must be called AFTER edge exchange (both non-overlap
and shared-node variants) and BEFORE `fill_physical_wall_ghost_2d!`.

For a block of extended size `(Nxe, Nye)` with `ng=1`:
- SW corner `f[1, 1, :]`       ← `f[2, 1, :]`  (from west ghost row at south ghost j)
- SE corner `f[Nxe, 1, :]`     ← `f[Nxe-1, 1, :]`
- NW corner `f[1, Nye, :]`     ← `f[2, Nye, :]`
- NE corner `f[Nxe, Nye, :]`   ← `f[Nxe-1, Nye, :]`

Fills corners where at least one adjacent edge is `:interface`
(pure wall-wall corners are handled by `fill_physical_wall_ghost_2d!`).
"""
function fill_ghost_corners_2d!(mbm::MultiBlockMesh2D,
                                  states::AbstractVector{<:BlockState2D})
    for (k, blk) in enumerate(mbm.blocks)
        st = states[k]
        ng = st.n_ghost
        Nxe, Nye = ext_dims(st)
        tags = blk.boundary_tags
        iW = tags.west  === INTERFACE_TAG
        iE = tags.east  === INTERFACE_TAG
        iS = tags.south === INTERFACE_TAG
        iN = tags.north === INTERFACE_TAG
        # SW corner: at least one of west/south is interface
        if iW || iS
            @inbounds st.f[ng, ng, :] .= st.f[ng + 1, ng, :]
        end
        # SE corner
        if iE || iS
            @inbounds st.f[Nxe - ng + 1, ng, :] .= st.f[Nxe - ng, ng, :]
        end
        # NW corner
        if iW || iN
            @inbounds st.f[ng, Nye - ng + 1, :] .= st.f[ng + 1, Nye - ng + 1, :]
        end
        # NE corner
        if iE || iN
            @inbounds st.f[Nxe - ng + 1, Nye - ng + 1, :] .= st.f[Nxe - ng, Nye - ng + 1, :]
        end
    end
    return nothing
end
