using KernelAbstractions

# Metal silently returns 0 when indexing a runtime tuple inside a
# @kernel. Use explicit branches that the Metal AIR compiler can fold.

@inline function _d2q9_cx(q::Integer)
    q == 1 && return 0
    q == 2 && return 1
    q == 3 && return 0
    q == 4 && return -1
    q == 5 && return 0
    q == 6 && return 1
    q == 7 && return -1
    q == 8 && return -1
    q == 9 && return 1
    return 0
end
@inline function _d2q9_cy(q::Integer)
    q == 1 && return 0
    q == 2 && return 0
    q == 3 && return 1
    q == 4 && return 0
    q == 5 && return -1
    q == 6 && return 1
    q == 7 && return 1
    q == 8 && return -1
    q == 9 && return -1
    return 0
end
@inline function _d2q9_opp(q::Integer)
    q == 1 && return 1
    q == 2 && return 4
    q == 3 && return 5
    q == 4 && return 2
    q == 5 && return 3
    q == 6 && return 8
    q == 7 && return 9
    q == 8 && return 6
    q == 9 && return 7
    return 1
end

@inline _d3q19_cx(q::Integer) =
    q==1 ? 0 : q==2 ? 1 : q==3 ? -1 : q==4 ? 0 : q==5 ? 0 :
    q==6 ? 0 : q==7 ? 0 : q==8 ? 1 : q==9 ? -1 : q==10 ? 1 :
    q==11 ? -1 : q==12 ? 1 : q==13 ? -1 : q==14 ? 1 : q==15 ? -1 :
    q==16 ? 0 : q==17 ? 0 : q==18 ? 0 : q==19 ? 0 : 0
@inline _d3q19_cy(q::Integer) =
    q==1 ? 0 : q==2 ? 0 : q==3 ? 0 : q==4 ? 1 : q==5 ? -1 :
    q==6 ? 0 : q==7 ? 0 : q==8 ? 1 : q==9 ? 1 : q==10 ? -1 :
    q==11 ? -1 : q==12 ? 0 : q==13 ? 0 : q==14 ? 0 : q==15 ? 0 :
    q==16 ? 1 : q==17 ? -1 : q==18 ? 1 : q==19 ? -1 : 0
@inline _d3q19_cz(q::Integer) =
    q==1 ? 0 : q==2 ? 0 : q==3 ? 0 : q==4 ? 0 : q==5 ? 0 :
    q==6 ? 1 : q==7 ? -1 : q==8 ? 0 : q==9 ? 0 : q==10 ? 0 :
    q==11 ? 0 : q==12 ? 1 : q==13 ? 1 : q==14 ? -1 : q==15 ? -1 :
    q==16 ? 1 : q==17 ? 1 : q==18 ? -1 : q==19 ? -1 : 0

# =====================================================================
# GPU-native drag reduction for LI-BB — lightweight cut-link list.
#
# The legacy `compute_drag_libb_mei_2d(Array(f_pre), …)` copies the
# entire f array (41 MB at D=80 Float64) to host at every evaluation
# and computes the Mei MEA loop on CPU. With a per-step drag this
# serialises the GPU pipeline on every PCIe transfer and dominates
# H100 runtime (measured: 52 MLUPS at D=80, vs ~15 k MLUPS peak).
#
# Strategy: a cylinder with D=80 has only ~200 cut links out of the
# 577 k grid cells. Precompute a compact list of these links
# `(i, j, q, q_w)` once on host, upload to device, then at every
# drag evaluation launch a small kernel that reads only those cells
# and writes one float per link into a dense `F_link[1:Nlinks, 2]`
# device array. The CPU reduction — `sum(Array(F_link))` — transfers
# 2 × 4·Nlinks bytes per evaluation (~1-2 KB), negligible vs the
# kernel step.
#
# For the time-resolved ST 2D-2 case, stack results into
# `(Nlinks, n_steps, 2)`. One host transfer at the very end gives the
# full Fx(t), Fy(t) time series for FFT / Strouhal.
# =====================================================================

"""
    CutLinkList{AT}

Compact device-side representation of the cut links flagged by
`q_wall > 0` — `list_i`, `list_j`, `list_q`, `list_qw` are parallel
arrays of length `Nlinks`. Built once via `build_cut_link_list_2d(q_wall_h)`
on host, copied to the target backend.
"""
struct CutLinkList{IT<:AbstractVector{Int32}, WT<:AbstractVector}
    list_i::IT
    list_j::IT
    list_q::IT
    list_qw::WT
    Nlinks::Int
end

struct CutLinkList3D{IT<:AbstractVector{Int32}, WT<:AbstractVector}
    list_i::IT; list_j::IT; list_k::IT; list_q::IT
    list_qw::WT
    Nlinks::Int
end

"""
    build_cut_link_list_2d(q_wall_h::Array{T,3}; backend=CPU())
                          -> CutLinkList

Scan a host `q_wall_h[Nx, Ny, 9]` array, collect the coordinates of
every cut link, and upload to `backend`. O(Nx·Ny) once at setup.
"""
function build_cut_link_list_2d(q_wall_h::AbstractArray{T,3}; backend=CPU()) where {T}
    Nx, Ny, _ = size(q_wall_h)
    is_h = Int32[]; js_h = Int32[]; qs_h = Int32[]; qws_h = T[]
    @inbounds for j in 1:Ny, i in 1:Nx, q in 2:9
        if q_wall_h[i,j,q] > 0
            push!(is_h, i); push!(js_h, j); push!(qs_h, q)
            push!(qws_h, q_wall_h[i,j,q])
        end
    end
    N = length(is_h)
    li = KernelAbstractions.allocate(backend, Int32, N)
    lj = KernelAbstractions.allocate(backend, Int32, N)
    lq = KernelAbstractions.allocate(backend, Int32, N)
    lw = KernelAbstractions.allocate(backend, T,      N)
    copyto!(li, is_h); copyto!(lj, js_h)
    copyto!(lq, qs_h); copyto!(lw, qws_h)
    return CutLinkList(li, lj, lq, lw, N)
end

function build_cut_link_list_3d(q_wall_h::AbstractArray{T,4}; backend=CPU()) where {T}
    Nx, Ny, Nz, _ = size(q_wall_h)
    is_h = Int32[]; js_h = Int32[]; ks_h = Int32[]; qs_h = Int32[]; qws_h = T[]
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx, q in 2:19
        if q_wall_h[i,j,k,q] > 0
            push!(is_h, i); push!(js_h, j); push!(ks_h, k); push!(qs_h, q)
            push!(qws_h, q_wall_h[i,j,k,q])
        end
    end
    N = length(is_h)
    li = KernelAbstractions.allocate(backend, Int32, N)
    lj = KernelAbstractions.allocate(backend, Int32, N)
    lk = KernelAbstractions.allocate(backend, Int32, N)
    lq = KernelAbstractions.allocate(backend, Int32, N)
    lw = KernelAbstractions.allocate(backend, T,      N)
    copyto!(li, is_h); copyto!(lj, js_h); copyto!(lk, ks_h)
    copyto!(lq, qs_h); copyto!(lw, qws_h)
    return CutLinkList3D(li, lj, lk, lq, lw, N)
end

# --- 2D: Mei-Luo-Shyy MEA on cut-link list ---
@kernel function _drag_mei_2d_list_kernel!(Fx_link, Fy_link,
                                             @Const(list_i), @Const(list_j),
                                             @Const(list_q), @Const(list_qw),
                                             @Const(f),
                                             @Const(uw_x), @Const(uw_y),
                                             Nx, Ny)
    n = @index(Global)
    T = eltype(f)
    @inbounds begin
        i  = Int(list_i[n]); j = Int(list_j[n])
        qi = Int(list_q[n]); qw = list_qw[n]
        qbar = _d2q9_opp(qi)
        cx_int = _d2q9_cx(qi); cy_int = _d2q9_cy(qi)
        cx_q = T(cx_int); cy_q = T(cy_int)
        im = i - cx_int; jm = j - cy_int
        fp_q_back = (1 <= im <= Nx && 1 <= jm <= Ny) ?
                     f[im, jm, qi] : f[i, j, qbar]
        fq_here    = f[i, j, qi]
        fqbar_here = f[i, j, qbar]
        w_q = (qi == 2 || qi == 3 || qi == 4 || qi == 5) ? T(1/9) : T(1/36)
        cu  = cx_q * uw_x[i, j, qi] + cy_q * uw_y[i, j, qi]
        δ   = -T(6) * w_q * cu
        arriving = qw ≤ T(0.5) ?
            T(2)*qw*fq_here + (one(T) - T(2)*qw)*fp_q_back + δ :
            (one(T)/(T(2)*qw))*fq_here + (one(T) - one(T)/(T(2)*qw))*fqbar_here +
                (one(T)/(T(2)*qw))*δ
        F_link = fq_here + arriving
        Fx_link[n] = cx_q * F_link
        Fy_link[n] = cy_q * F_link
    end
end

"""
    compute_drag_libb_mei_2d_gpu!(Fx_link, Fy_link, links::CutLinkList,
                                     f, uw_x, uw_y, Nx, Ny)

Compute each cut link's Fx, Fy contribution into dense device arrays
`Fx_link`, `Fy_link` (length = `links.Nlinks`). No atomic operations;
each thread writes its own slot. The caller sums on host with
`sum(Array(Fx_link))` — transferring at most a few kilobytes.
"""
function compute_drag_libb_mei_2d_gpu!(Fx_link, Fy_link, links::CutLinkList,
                                         f, uw_x, uw_y,
                                         Nx::Integer, Ny::Integer)
    # Skip kernel launch for empty link list — KernelAbstractions on CUDA
    # with ndrange=(0,) triggers an integer division error in the
    # workgroup-size computation. An empty list can arise in multi-block
    # setups where some blocks do not contain any cut cells (the cylinder
    # sits entirely in another block).
    links.Nlinks == 0 && return nothing
    backend = KernelAbstractions.get_backend(f)
    _drag_mei_2d_list_kernel!(backend)(Fx_link, Fy_link,
                                         links.list_i, links.list_j,
                                         links.list_q, links.list_qw,
                                         f, uw_x, uw_y, Nx, Ny;
                                         ndrange=(links.Nlinks,))
    return nothing
end

# --- 3D: halfway-BB Ladd (Mei 3D port = future work) ---
@kernel function _drag_halfway_3d_list_kernel!(Fx_link, Fy_link, Fz_link,
                                                 @Const(list_i), @Const(list_j),
                                                 @Const(list_k), @Const(list_q),
                                                 @Const(f), Nx, Ny, Nz)
    n = @index(Global)
    T = eltype(f)
    @inbounds begin
        i  = Int(list_i[n]); j = Int(list_j[n]); k = Int(list_k[n])
        qi = Int(list_q[n])
        fv = f[i, j, k, qi]
        two_fv = T(2) * fv
        Fx_link[n] = T(_d3q19_cx(qi)) * two_fv
        Fy_link[n] = T(_d3q19_cy(qi)) * two_fv
        Fz_link[n] = T(_d3q19_cz(qi)) * two_fv
    end
end

"""
    compute_drag_libb_3d_gpu!(Fx_link, Fy_link, Fz_link,
                                links::CutLinkList3D, f, Nx, Ny, Nz)

GPU halfway-BB Ladd drag on cut-link list (no atomics). Sum host-side.
"""
function compute_drag_libb_3d_gpu!(Fx_link, Fy_link, Fz_link,
                                     links::CutLinkList3D, f,
                                     Nx::Integer, Ny::Integer, Nz::Integer)
    backend = KernelAbstractions.get_backend(f)
    _drag_halfway_3d_list_kernel!(backend)(Fx_link, Fy_link, Fz_link,
                                             links.list_i, links.list_j,
                                             links.list_k, links.list_q,
                                             f, Nx, Ny, Nz;
                                             ndrange=(links.Nlinks,))
    return nothing
end
