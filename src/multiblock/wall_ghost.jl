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
# KernelAbstractions kernels — broadcast-compatible on CPU and any GPU
# backend (CUDA, Metal). Each kernel fills one row/column × 3 populations
# so we launch once per edge with a (Nye, 3) or (Nxe, 3) work domain.

@kernel function _fill_wall_vertical_kernel_2d!(f, i_ghost::Int, i_bd::Int,
                                                   pop1::Int, pop2::Int, pop3::Int,
                                                   cqy1::Int, cqy2::Int, cqy3::Int,
                                                   q_opp1::Int, q_opp2::Int, q_opp3::Int,
                                                   j_lo::Int, j_hi::Int)
    idx, p = @index(Global, NTuple)
    # idx = 1..Nye (j'), p = 1..3 (which population)
    j′ = idx
    q, cqy, q_opp = if p == 1
        (pop1, cqy1, q_opp1)
    elseif p == 2
        (pop2, cqy2, q_opp2)
    else
        (pop3, cqy3, q_opp3)
    end
    # clamp j' + cqy to [j_lo, j_hi]
    jsrc = j′ + cqy
    jsrc = jsrc < j_lo ? j_lo : (jsrc > j_hi ? j_hi : jsrc)
    @inbounds f[i_ghost, j′, q] = f[i_bd, jsrc, q_opp]
end

@kernel function _fill_wall_horizontal_kernel_2d!(f, j_ghost::Int, j_bd::Int,
                                                     pop1::Int, pop2::Int, pop3::Int,
                                                     cqx1::Int, cqx2::Int, cqx3::Int,
                                                     q_opp1::Int, q_opp2::Int, q_opp3::Int,
                                                     i_lo::Int, i_hi::Int)
    idx, p = @index(Global, NTuple)
    i′ = idx
    q, cqx, q_opp = if p == 1
        (pop1, cqx1, q_opp1)
    elseif p == 2
        (pop2, cqx2, q_opp2)
    else
        (pop3, cqx3, q_opp3)
    end
    isrc = i′ + cqx
    isrc = isrc < i_lo ? i_lo : (isrc > i_hi ? i_hi : isrc)
    @inbounds f[i′, j_ghost, q] = f[isrc, j_bd, q_opp]
end

@inline function _fill_wall_vertical_2d!(st::BlockState2D, edge::Symbol,
                                           ng, Nxp, Nyp, pop_list)
    i_ghost = edge === :west ? ng      : ng + Nxp + 1
    i_bd    = edge === :west ? ng + 1  : ng + Nxp
    Nye = 2 * ng + Nyp
    p1, p2, p3 = pop_list
    cqy1, cqy2, cqy3 = _CQY[p1], _CQY[p2], _CQY[p3]
    qo1, qo2, qo3    = _Q_OPP[p1], _Q_OPP[p2], _Q_OPP[p3]
    backend = KernelAbstractions.get_backend(st.f)
    kernel = _fill_wall_vertical_kernel_2d!(backend)
    kernel(st.f, i_ghost, i_bd, p1, p2, p3, cqy1, cqy2, cqy3,
            qo1, qo2, qo3, ng + 1, ng + Nyp; ndrange=(Nye, 3))
    KernelAbstractions.synchronize(backend)
    return nothing
end

@inline function _fill_wall_horizontal_2d!(st::BlockState2D, edge::Symbol,
                                             ng, Nxp, Nyp, pop_list)
    j_ghost = edge === :south ? ng     : ng + Nyp + 1
    j_bd    = edge === :south ? ng + 1 : ng + Nyp
    Nxe = 2 * ng + Nxp
    p1, p2, p3 = pop_list
    cqx1, cqx2, cqx3 = _CQX[p1], _CQX[p2], _CQX[p3]
    qo1, qo2, qo3    = _Q_OPP[p1], _Q_OPP[p2], _Q_OPP[p3]
    backend = KernelAbstractions.get_backend(st.f)
    kernel = _fill_wall_horizontal_kernel_2d!(backend)
    kernel(st.f, j_ghost, j_bd, p1, p2, p3, cqx1, cqx2, cqx3,
            qo1, qo2, qo3, ng + 1, ng + Nxp; ndrange=(Nxe, 3))
    KernelAbstractions.synchronize(backend)
    return nothing
end

# =====================================================================
# Full pre-fill for SLBM multi-block (Phase B.2.3 fix).
#
# On a curvilinear mesh, SLBM departure points at cells near a physical
# wall can cross into the ghost region at OBLIQUE angles, reading
# populations that the standard 3-population wall-ghost fill did not
# set. This helper copies ALL 9 populations from the boundary row into
# the ghost row for every physical-wall edge, providing valid data for
# any departure direction. Called BEFORE `fill_physical_wall_ghost_2d!`,
# which then overwrites the 3 reflected populations with more accurate
# halfway-BB values.
# =====================================================================

@kernel function _copy_col_kernel_2d!(f, i_dst::Int, i_src::Int)
    j, q = @index(Global, NTuple)
    @inbounds f[i_dst, j, q] = f[i_src, j, q]
end

@kernel function _copy_row_kernel_2d!(f, j_dst::Int, j_src::Int)
    i, q = @index(Global, NTuple)
    @inbounds f[i, j_dst, q] = f[i, j_src, q]
end

@kernel function _extrap_col_kernel_2d!(f, i_dst::Int, i_bd::Int, i_inner::Int)
    j, q = @index(Global, NTuple)
    @inbounds f[i_dst, j, q] = 2 * f[i_bd, j, q] - f[i_inner, j, q]
end

@kernel function _extrap_row_kernel_2d!(f, j_dst::Int, j_bd::Int, j_inner::Int)
    i, q = @index(Global, NTuple)
    @inbounds f[i, j_dst, q] = 2 * f[i, j_bd, q] - f[i, j_inner, q]
end

"""
    fill_slbm_wall_ghost_2d!(mbm, states)

Pre-fill ALL 9 populations at each physical-wall ghost for SLBM on
curvilinear meshes, where oblique departure points may read any
population from the ghost region.

- **Wall/cylinder edges**: boundary-row copy (zeroth-order extrapolation).
- **Inlet/outlet edges**: linear extrapolation `ghost = 2·boundary − interior`
  to avoid a feedback loop through the BC-reconstructed boundary row.

Skips `:interface` edges (handled by the exchange). Must be called
BEFORE `fill_physical_wall_ghost_2d!`, which then overwrites the 3
reflected populations with more accurate halfway-BB values.
"""
function fill_slbm_wall_ghost_2d!(mbm::MultiBlockMesh2D,
                                    states::AbstractVector{<:BlockState2D})
    _EXTRAP_TAGS = (:inlet, :outlet)
    for (k, blk) in enumerate(mbm.blocks)
        st = states[k]
        ng = st.n_ghost; Nxp = st.Nξ_phys; Nyp = st.Nη_phys
        Nxe = Nxp + 2 * ng; Nye = Nyp + 2 * ng
        backend = KernelAbstractions.get_backend(st.f)
        for edge in EDGE_SYMBOLS_2D
            tag = getproperty(blk.boundary_tags, edge)
            tag === INTERFACE_TAG && continue
            use_extrap = tag in _EXTRAP_TAGS
            if edge === :west || edge === :east
                i_ghost = edge === :west ? ng      : ng + Nxp + 1
                i_bd    = edge === :west ? ng + 1  : ng + Nxp
                if use_extrap
                    i_inner = edge === :west ? ng + 2  : ng + Nxp - 1
                    kernel = _extrap_col_kernel_2d!(backend)
                    kernel(st.f, i_ghost, i_bd, i_inner; ndrange=(Nye, 9))
                else
                    kernel = _copy_col_kernel_2d!(backend)
                    kernel(st.f, i_ghost, i_bd; ndrange=(Nye, 9))
                end
            else
                j_ghost = edge === :south ? ng     : ng + Nyp + 1
                j_bd    = edge === :south ? ng + 1 : ng + Nyp
                if use_extrap
                    j_inner = edge === :south ? ng + 2  : ng + Nyp - 1
                    kernel = _extrap_row_kernel_2d!(backend)
                    kernel(st.f, j_ghost, j_bd, j_inner; ndrange=(Nxe, 9))
                else
                    kernel = _copy_row_kernel_2d!(backend)
                    kernel(st.f, j_ghost, j_bd; ndrange=(Nxe, 9))
                end
            end
        end
        KernelAbstractions.synchronize(backend)
    end
    return mbm
end
