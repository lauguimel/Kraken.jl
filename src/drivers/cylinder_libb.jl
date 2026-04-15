using KernelAbstractions

# =====================================================================
# Schäfer-Turek-style cylinder drivers with Bouzidi LI-BB.
#
# Uses the Ginzburg-exact V2 LI-BB kernel (pre-phase Bouzidi + TRT),
# combined with Zou-He inlet (parabolic profile) and Zou-He pressure
# outlet. Top and bottom walls are handled by the fused kernel's
# default halfway-BB fallback at the y domain edges.
#
# Drag / lift measured via the MEA (momentum-exchange) formula
# adapted to LI-BB: for each fluid cell adjacent to the cylinder and
# each flagged cut link q, the momentum transferred per step is
# 2 · c_q · f_q(post-collision). We sum over all flagged fluid links
# and average over a window of the final steps.
# =====================================================================

"""
    compute_drag_libb_2d(f_pre, q_wall, Nx, Ny)
        -> (Fx, Fy)

Simple halfway-BB MEA drag (2·c_q·f_q at the fluid cell). Underestimates
Cd when q_w ≠ 0.5; kept for the old scaffold/test. Prefer
`compute_drag_libb_mei_2d` for quantitative Schäfer-Turek-style
benchmarks.
"""
function compute_drag_libb_2d(f_post, q_wall, Nx::Int, Ny::Int)
    f = Array(f_post)
    qw = Array(q_wall)
    cxv = (0, 1, 0, -1,  0, 1, -1, -1,  1)
    cyv = (0, 0, 1,  0, -1, 1,  1, -1, -1)
    Fx = 0.0; Fy = 0.0
    @inbounds for j in 1:Ny, i in 1:Nx
        for q in 2:9
            if qw[i, j, q] > 0
                Fx += 2.0 * Float64(cxv[q]) * Float64(f[i, j, q])
                Fy += 2.0 * Float64(cyv[q]) * Float64(f[i, j, q])
            end
        end
    end
    return (Fx = Fx, Fy = Fy)
end

# Opposite direction lookup for D2Q9 (q=1..9).
const _D2Q9_OPP = (1, 4, 5, 2, 3, 8, 9, 6, 7)

"""
    compute_drag_libb_mei_2d(f_pre, q_wall, uw_link_x, uw_link_y, Nx, Ny)
        -> (Fx, Fy)

Mei-Luo-Shyy 2002 (J. Comput. Phys. 161, 680) momentum-exchange drag
for LI-BB. Consistent with the V2 pre-phase Bouzidi substitution: for
each cut link q on a fluid cell, the force per link per step is

    F_link = c_q · (f_q_pre + f_q̄_bouzidi)

where `f_q_pre = f_pre[i,j,q]` (pop q pre-step at the fluid cell,
i.e. post-coll from the previous step) and `f_q̄_bouzidi` is the
arriving pop computed via the same `_libb_branch` used in the
kernel's ApplyLiBBPrePhase brick (lag-1 Bouzidi estimate). Reduces to
`2·c_q·f_q` at q_w = 0.5 stationary, matching the halfway formula.

`f_pre` must be the `f_in` array at step N (= `f_out` from step N-1
BEFORE the boundary patches of step N are applied; in the driver we
pass the full `f_in` each sampling step). `uw_link_x/y` are the per-
link wall-velocity arrays (zero for a stationary cylinder).
"""
function compute_drag_libb_mei_2d(f_pre, q_wall, uw_link_x, uw_link_y,
                                    Nx::Int, Ny::Int)
    f = Array(f_pre)
    qw = Array(q_wall)
    uwx = Array(uw_link_x); uwy = Array(uw_link_y)
    cxv = (0, 1, 0, -1,  0, 1, -1, -1,  1)
    cyv = (0, 0, 1,  0, -1, 1,  1, -1, -1)
    Fx = 0.0; Fy = 0.0
    # Helper: δ for the arriving pop q̄ given outgoing link q.
    # δ_{q̄} = -2 w_q (c_q · u_w) / c_s² = -6 w_q (c_q · u_w). For axis
    # links (w=1/9): δ = -(2/3)(c_q·u_w); for diagonal (w=1/36):
    # δ = -(1/6)(c_q·u_w).
    @inbounds for j in 1:Ny, i in 1:Nx
        for q in 2:9
            q_w = qw[i, j, q]
            q_w > 0 || continue
            qbar = _D2Q9_OPP[q]
            # Pulled value (f_post_back in Bouzidi): the pop q at the
            # opposite-to-wall neighbour, i.e. pulled along -c_q.
            im = i - cxv[q]; jm = j - cyv[q]
            # Clamp to domain (matches the fused kernel's fallback:
            # out-of-range → reflect to the same cell's opposite pop).
            fp_q_back = (1 <= im <= Nx && 1 <= jm <= Ny) ?
                Float64(f[im, jm, q]) : Float64(f[i, j, qbar])
            fq_here = Float64(f[i, j, q])
            fqbar_here = Float64(f[i, j, qbar])
            # δ_{q̄} from wall velocity at this link
            w_q = (q in (2, 3, 4, 5)) ? 1.0/9.0 : 1.0/36.0
            cu = Float64(cxv[q]) * uwx[i, j, q] +
                 Float64(cyv[q]) * uwy[i, j, q]
            δ = -6.0 * w_q * cu
            # Bouzidi branch
            arriving = if q_w ≤ 0.5
                2q_w * fq_here + (1 - 2q_w) * fp_q_back + δ
            else
                inv2q = 1.0 / (2q_w)
                inv2q * fq_here + (1 - inv2q) * fqbar_here + inv2q * δ
            end
            F_link = fq_here + arriving
            Fx += Float64(cxv[q]) * F_link
            Fy += Float64(cyv[q]) * F_link
        end
    end
    return (Fx = Fx, Fy = Fy)
end

# =====================================================================
# Boundary "rebuild" for Zou-He inlet/outlet compatible with V2 kernel.
#
# Problem: the fused V2 kernel applies halfway-BB fallback at i=1 and
# i=Nx (PullHalfwayBB brick), which corrupts the boundary density by
# bouncing back the +x-streamed pops. A post-hoc Zou-He pressure outlet
# then computes ux = −1 + (known)/ρ_out using these corrupted pops,
# producing a wrong velocity that streams back into the interior. The
# error grows exponentially and blows up in O(10) steps.
#
# Fix: after the kernel, OVERWRITE `f_out[1, j, :]` and
# `f_out[Nx, j, :]` for j = 2..Ny-1 by rebuilding the cell from:
#   (a) pre-step `f_in` values streamed from the interior along each
#       real lattice link — NOT from the corrupted bounce-back at the
#       boundary itself;
#   (b) Zou-He velocity / pressure closure for the three unknown pops
#       that would have streamed from outside the domain;
#   (c) local TRT collision on the nine reconstructed pre-collision
#       pops — same Λ = 3/16 as the kernel.
#
# Corners (i∈{1,Nx} ∧ j∈{1,Ny}) are left to the kernel's halfway-BB
# fallback, since the parabolic profile prescribes u = 0 there (the
# corner is a true no-slip point where inlet/outlet meets the
# channel wall).
# =====================================================================

@inline function _trt_collide_local(f1::T, f2::T, f3::T, f4::T, f5::T,
                                     f6::T, f7::T, f8::T, f9::T,
                                     s_p::T, s_m::T) where {T}
    ρ  = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
    ux = (f2 - f4 + f6 - f8 + f9 - f7) / ρ
    uy = (f3 - f5 + f6 - f8 + f7 - f9) / ρ
    usq = ux * ux + uy * uy
    fe1 = feq_2d(Val(1), ρ, ux, uy, usq)
    fe2 = feq_2d(Val(2), ρ, ux, uy, usq)
    fe3 = feq_2d(Val(3), ρ, ux, uy, usq)
    fe4 = feq_2d(Val(4), ρ, ux, uy, usq)
    fe5 = feq_2d(Val(5), ρ, ux, uy, usq)
    fe6 = feq_2d(Val(6), ρ, ux, uy, usq)
    fe7 = feq_2d(Val(7), ρ, ux, uy, usq)
    fe8 = feq_2d(Val(8), ρ, ux, uy, usq)
    fe9 = feq_2d(Val(9), ρ, ux, uy, usq)
    a = (s_p + s_m) * T(0.5)
    b = (s_p - s_m) * T(0.5)
    return (
        f1 - s_p * (f1 - fe1),
        f2 - a * (f2 - fe2) - b * (f4 - fe4),
        f3 - a * (f3 - fe3) - b * (f5 - fe5),
        f4 - a * (f4 - fe4) - b * (f2 - fe2),
        f5 - a * (f5 - fe5) - b * (f3 - fe3),
        f6 - a * (f6 - fe6) - b * (f8 - fe8),
        f7 - a * (f7 - fe7) - b * (f9 - fe9),
        f8 - a * (f8 - fe8) - b * (f6 - fe6),
        f9 - a * (f9 - fe9) - b * (f7 - fe7),
    )
end

"""
    rebuild_inlet_outlet_libb_2d!(f_out, f_in, u_profile, ρ_out, ν, Nx, Ny)

Overwrite `f_out[1, j, :]` and `f_out[Nx, j, :]` for j = 2..Ny-1 using
pre-collision Zou-He reconstruction + local TRT collision.

- Inlet (west, i=1): Zou-He velocity with `u = u_profile[j]`, v = 0.
- Outlet (east, i=Nx): Zou-He pressure with ρ = ρ_out, v = 0.

Call this *after* `fused_trt_libb_v2_step!` and *before* swapping
`f_in`/`f_out`. `f_in` must still hold the pre-step populations.
"""
@kernel function _rebuild_west_libb_2d_kernel!(f_out, f_in, u_profile, s_p, s_m)
    jm1 = @index(Global)
    j = jm1 + 1
    T = eltype(f_out)
    @inbounds begin
        fp1 = f_in[1, j,   1]
        fp3 = f_in[1, j-1, 3]
        fp4 = f_in[2, j,   4]
        fp5 = f_in[1, j+1, 5]
        fp7 = f_in[2, j-1, 7]
        fp8 = f_in[2, j+1, 8]
        u_in = u_profile[j]
        ρ_w  = (fp1 + fp3 + fp5 + T(2) * (fp4 + fp7 + fp8)) / (one(T) - u_in)
        fp2  = fp4 + T(2/3) * ρ_w * u_in
        fp6  = fp8 - T(0.5) * (fp3 - fp5) + T(1/6) * ρ_w * u_in
        fp9  = fp7 + T(0.5) * (fp3 - fp5) + T(1/6) * ρ_w * u_in
        F1,F2,F3,F4,F5,F6,F7,F8,F9 = _trt_collide_local(
            fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9, s_p, s_m)
        f_out[1, j, 1] = F1; f_out[1, j, 2] = F2; f_out[1, j, 3] = F3
        f_out[1, j, 4] = F4; f_out[1, j, 5] = F5; f_out[1, j, 6] = F6
        f_out[1, j, 7] = F7; f_out[1, j, 8] = F8; f_out[1, j, 9] = F9
    end
end

@kernel function _rebuild_east_libb_2d_kernel!(f_out, f_in, Nx, ρ_out, s_p, s_m)
    jm1 = @index(Global)
    j = jm1 + 1
    T = eltype(f_out)
    @inbounds begin
        fp1 = f_in[Nx,   j,   1]
        fp2 = f_in[Nx-1, j,   2]
        fp3 = f_in[Nx,   j-1, 3]
        fp5 = f_in[Nx,   j+1, 5]
        fp6 = f_in[Nx-1, j-1, 6]
        fp9 = f_in[Nx-1, j+1, 9]
        u_x = -one(T) + (fp1 + fp3 + fp5 + T(2) * (fp2 + fp6 + fp9)) / ρ_out
        fp4 = fp2 - T(2/3) * ρ_out * u_x
        fp7 = fp9 - T(0.5) * (fp3 - fp5) - T(1/6) * ρ_out * u_x
        fp8 = fp6 + T(0.5) * (fp3 - fp5) - T(1/6) * ρ_out * u_x
        F1,F2,F3,F4,F5,F6,F7,F8,F9 = _trt_collide_local(
            fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9, s_p, s_m)
        f_out[Nx, j, 1] = F1; f_out[Nx, j, 2] = F2; f_out[Nx, j, 3] = F3
        f_out[Nx, j, 4] = F4; f_out[Nx, j, 5] = F5; f_out[Nx, j, 6] = F6
        f_out[Nx, j, 7] = F7; f_out[Nx, j, 8] = F8; f_out[Nx, j, 9] = F9
    end
end

function rebuild_inlet_outlet_libb_2d!(f_out, f_in, u_profile, ρ_out::Real,
                                        ν::Real, Nx::Int, Ny::Int)
    backend = KernelAbstractions.get_backend(f_out)
    T = eltype(f_out)
    s_p_r, s_m_r = trt_rates(ν; Λ=3/16)
    s_p = T(s_p_r); s_m = T(s_m_r); ρ_o = T(ρ_out)
    kw! = _rebuild_west_libb_2d_kernel!(backend)
    ke! = _rebuild_east_libb_2d_kernel!(backend)
    kw!(f_out, f_in, u_profile, s_p, s_m; ndrange=(Ny - 2,))
    ke!(f_out, f_in, Nx, ρ_o, s_p, s_m; ndrange=(Ny - 2,))
    return nothing
end

"""
    run_cylinder_libb_2d(; Nx=300, Ny=80, cx=Nx÷4, cy=Ny÷2, radius=10,
                          u_in=0.04, ν=0.04, max_steps=50000,
                          avg_window=5000, inlet=:parabolic, ρ_out=1.0)

2D cylinder with LI-BB V2 boundary condition. Parabolic / uniform
Zou-He velocity inlet, Zou-He pressure outlet; both reconstructed from
pre-step populations to bypass the kernel's halfway-BB fallback
corruption at the non-wall boundaries. Top / bottom channel walls are
halfway-BB via the kernel's fallback at j=1 and j=Ny.

Arguments:
- `u_in`: centerline u_max for `:parabolic` inlet; uniform u for `:uniform`.
- `inlet`: `:parabolic` (Schäfer-Turek 2D convention) or `:uniform`.
- `ρ_out`: outlet density for the Zou-He pressure BC (default 1).

Returns a NamedTuple with:
- `ρ`, `ux`, `uy`, `is_solid`, `q_wall`, `u_ref`, `D`, `Fx`, `Fy`
- `Cd = 2·Fx / (u_ref² · D)` with `u_ref = u_mean = 2/3 · u_max`
  (parabolic) or `u_ref = u_in` (uniform).
- `Cl` analogously.

Reference for Schäfer-Turek 2D-1 (Re=20, blockage ≈ 25%): Cd ≈ 5.58.
"""
function run_cylinder_libb_2d(; Nx::Int=300, Ny::Int=80,
                                cx::Union{Nothing,Real}=nothing,
                                cy::Union{Nothing,Real}=nothing,
                                radius::Real=10,
                                u_in::Real=0.04, ν::Real=0.04,
                                max_steps::Int=50_000,
                                avg_window::Int=5_000,
                                inlet::Symbol=:parabolic,
                                ρ_out::Real=1.0,
                                backend=KernelAbstractions.CPU(),
                                T::Type{<:AbstractFloat}=Float64)
    cx = isnothing(cx) ? Nx ÷ 4 : Float64(cx)
    cy = isnothing(cy) ? Ny ÷ 2 : Float64(cy)

    q_wall_h, is_solid_h = precompute_q_wall_cylinder(Nx, Ny, cx, cy, radius; FT=T)

    u_prof_h = if inlet === :parabolic
        [T(4) * T(u_in) * T(j - 1) * T(Ny - j) / T(Ny - 1)^2 for j in 1:Ny]
    elseif inlet === :uniform
        fill(T(u_in), Ny)
    else
        error("unknown inlet $(inlet); expected :parabolic or :uniform")
    end
    u_ref = inlet === :parabolic ? (2 / 3) * Float64(u_in) : Float64(u_in)

    # Host-side initial populations
    f_in_h  = zeros(T, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx, q in 1:9
        f_in_h[i, j, q] = Kraken.equilibrium(D2Q9(), one(T), u_prof_h[j],
                                               zero(T), q)
    end

    # Device allocations
    q_wall    = KernelAbstractions.allocate(backend, T, Nx, Ny, 9)
    is_solid  = KernelAbstractions.allocate(backend, Bool, Nx, Ny)
    uw_x      = KernelAbstractions.allocate(backend, T, Nx, Ny, 9)
    uw_y      = KernelAbstractions.allocate(backend, T, Nx, Ny, 9)
    f_in      = KernelAbstractions.allocate(backend, T, Nx, Ny, 9)
    f_out     = KernelAbstractions.allocate(backend, T, Nx, Ny, 9)
    ρ         = KernelAbstractions.allocate(backend, T, Nx, Ny)
    ux        = KernelAbstractions.allocate(backend, T, Nx, Ny)
    uy        = KernelAbstractions.allocate(backend, T, Nx, Ny)
    u_profile = KernelAbstractions.allocate(backend, T, Ny)
    copyto!(q_wall, q_wall_h); copyto!(is_solid, is_solid_h)
    fill!(uw_x, zero(T)); fill!(uw_y, zero(T))
    copyto!(f_in, f_in_h); fill!(f_out, zero(T))
    fill!(ρ, one(T)); fill!(ux, zero(T)); fill!(uy, zero(T))
    copyto!(u_profile, u_prof_h)

    Fx_sum = 0.0; Fy_sum = 0.0; n_avg = 0

    for step in 1:max_steps
        fused_trt_libb_v2_step!(f_out, f_in, ρ, ux, uy, is_solid,
                                 q_wall, uw_x, uw_y, Nx, Ny, T(ν))
        # Pre-collision Zou-He rebuild at i=1 and i=Nx (j=2..Ny-1)
        rebuild_inlet_outlet_libb_2d!(f_out, f_in, u_profile, ρ_out, ν, Nx, Ny)

        if step > max_steps - avg_window
            drag = compute_drag_libb_mei_2d(f_out, q_wall, uw_x, uw_y, Nx, Ny)
            Fx_sum += drag.Fx; Fy_sum += drag.Fy
            n_avg  += 1
        end

        f_in, f_out = f_out, f_in
    end
    KernelAbstractions.synchronize(backend)

    Fx_avg = Fx_sum / n_avg
    Fy_avg = Fy_sum / n_avg
    D      = 2 * Float64(radius)
    Cd     = 2.0 * Fx_avg / (u_ref^2 * D)
    Cl     = 2.0 * Fy_avg / (u_ref^2 * D)

    return (; ρ = Array(ρ), ux = Array(ux), uy = Array(uy),
             Cd, Cl, Fx = Fx_avg, Fy = Fy_avg,
             q_wall = Array(q_wall), is_solid = Array(is_solid),
             u_ref, D, inlet)
end
