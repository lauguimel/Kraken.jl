# =====================================================================
# Physical-wall ghost fill (Phase A.5c).
#
# For edges tagged as a physical BC (not :interface), the ghost row
# adjacent to the physical boundary is filled with a halfway-BB
# reflection of the boundary-row populations. The effect: when the
# step kernel pulls from ghost at the physical-boundary interior
# cell, it reads the same values as single-block's halfway-BB
# fallback would produce at its own i=1/j=1 edge.
#
# Derivation — west wall with Ng = 1, physical boundary at i = 2:
#   Pull at (2, j, q) with cqx > 0 reads ghost[1, j - cqy, q].
#   Halfway-BB expects this value to equal the opposite population
#   at the boundary row: interior[2, j, q_opposite].
#   Rearranging: ghost[1, j', q] = interior[2, j' + cqy, q_opposite]
#   for q ∈ {2, 6, 9} (cqx > 0).
#
# For corners (j'+cqy out of the physical-row range), we clamp to
# the nearest valid physical row. The resulting corner cells are
# not strictly halfway-BB but are "reflective enough" for a 1-cell-
# wide region that does not influence the bulk flow.
#
# This helper is called BEFORE the step, AFTER exchange_ghost_2d!, so
# that both interface-ghost and wall-ghost are filled before the
# kernel fires. Edges tagged :interface are skipped here (handled by
# the exchange).
#
# BC tags other than :wall (e.g., :inlet, :outlet, :cylinder) fall
# under this code as well — for those, the filled wall reflection
# is ignored by a post-step `apply_bc_rebuild_2d!` call on the
# physical-interior view, which overwrites the boundary row with
# the ZouHe / pressure / Bouzidi values. The wall-ghost fill is
# thus a safe default: harmless when overwritten, correct when it
# matters (halfway-BB walls).
# =====================================================================

"""
    fill_physical_wall_ghost_2d!(mbm::MultiBlockMesh2D,
                                  states::AbstractVector{<:BlockState2D})

Pre-fill ghost rows of every non-`:interface` edge with a halfway-BB
reflection of the physical boundary row. Skips edges already handled
by `exchange_ghost_2d!` (tag == `INTERFACE_TAG`).

Call sequence per timestep:

    exchange_ghost_2d!(mbm, states)           # interface ghosts
    fill_physical_wall_ghost_2d!(mbm, states) # physical-wall ghosts
    step_kernel!(...)                         # now reads valid ghosts
    apply_bc_rebuild_2d!(interior_view, ...)  # post-step BC overwrite
    swap f_in ↔ f_out
"""
function fill_physical_wall_ghost_2d!(mbm::MultiBlockMesh2D,
                                        states::AbstractVector{<:BlockState2D})
    length(states) == length(mbm.blocks) ||
        error("fill_physical_wall_ghost_2d!: states has $(length(states)) entries, " *
              "mbm has $(length(mbm.blocks)) blocks")
    for (k, blk) in enumerate(mbm.blocks)
        st = states[k]
        for edge in EDGE_SYMBOLS_2D
            tag = getproperty(blk.boundary_tags, edge)
            tag === INTERFACE_TAG && continue
            _fill_wall_ghost_edge_2d!(st, edge)
        end
    end
    return mbm
end

# D2Q9 velocity components and opposite indices (q=1..9)
const _CQX = (0,  1, 0, -1,  0,  1, -1, -1,  1)
const _CQY = (0,  0, 1,  0, -1,  1,  1, -1, -1)
const _Q_OPP = (1, 4, 5,  2,  3,  8,  9,  6,  7)

function _fill_wall_ghost_edge_2d!(st::BlockState2D, edge::Symbol)
    ng = st.n_ghost
    Nxp = st.Nξ_phys
    Nyp = st.Nη_phys
    if edge === :west
        # Ghost i = 1..ng; physical boundary at i = ng + 1.
        # Fill pops with cqx > 0: q ∈ {2, 6, 9}.
        _fill_wall_vertical_2d!(st, :west,  ng, Nxp, Nyp, (2, 6, 9))
    elseif edge === :east
        # Ghost i = ng+Nxp+1 .. ng+Nxp+ng; boundary at i = ng + Nxp.
        # Fill pops with cqx < 0: q ∈ {4, 7, 8}.
        _fill_wall_vertical_2d!(st, :east,  ng, Nxp, Nyp, (4, 7, 8))
    elseif edge === :south
        # Ghost j = 1..ng; boundary at j = ng + 1. Fill cqy > 0: q ∈ {3, 6, 7}.
        _fill_wall_horizontal_2d!(st, :south, ng, Nxp, Nyp, (3, 6, 7))
    elseif edge === :north
        _fill_wall_horizontal_2d!(st, :north, ng, Nxp, Nyp, (5, 8, 9))
    else
        error("_fill_wall_ghost_edge_2d!: unknown edge $edge")
    end
    return nothing
end

# Vertical walls (west / east). Ghost column at i = i_ghost, physical
# boundary column at i = i_bd. For each q with cqx crossing the wall,
# set ghost[i_ghost, j', q] so that the step's pull at (i_bd, j'+cqy)
# for direction q reads exactly the single-block halfway-BB value:
#
#   pull at (i_bd, j, q) reads ghost[i_ghost, j - cqy, q];
#   single-block halfway-BB expects this to equal interior[i_bd, j, q_opp];
#   setting j' = j - cqy:   ghost[i_ghost, j', q] = interior[i_bd, j' + cqy, q_opp].
#
# We loop j' over the FULL extended range so that doubly-ghost corners
# (j' = ng or j' = ng + Nyp + 1) also carry valid values, which diagonal
# pulls from the physical-boundary-row corner cells will read. jsrc out
# of the physical range is clamped to the boundary — for interior cells
# that never read those ghost slots, the clamped value is unused.
@inline function _fill_wall_vertical_2d!(st::BlockState2D, edge::Symbol,
                                           ng, Nxp, Nyp, pop_list)
    i_ghost = edge === :west ? ng      : ng + Nxp + 1
    i_bd    = edge === :west ? ng + 1  : ng + Nxp
    Nye = 2 * ng + Nyp
    @inbounds for j′ in 1:Nye, q in pop_list
        q_opp = _Q_OPP[q]
        cqy = _CQY[q]
        jsrc = clamp(j′ + cqy, ng + 1, ng + Nyp)
        st.f[i_ghost, j′, q] = st.f[i_bd, jsrc, q_opp]
    end
    return nothing
end

@inline function _fill_wall_horizontal_2d!(st::BlockState2D, edge::Symbol,
                                             ng, Nxp, Nyp, pop_list)
    j_ghost = edge === :south ? ng     : ng + Nyp + 1
    j_bd    = edge === :south ? ng + 1 : ng + Nyp
    Nxe = 2 * ng + Nxp
    @inbounds for i′ in 1:Nxe, q in pop_list
        q_opp = _Q_OPP[q]
        cqx = _CQX[q]
        isrc = clamp(i′ + cqx, ng + 1, ng + Nxp)
        st.f[i′, j_ghost, q] = st.f[isrc, j_bd, q_opp]
    end
    return nothing
end
