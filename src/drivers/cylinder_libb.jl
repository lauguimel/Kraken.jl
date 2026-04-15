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
    compute_drag_libb_2d(f_post, q_wall, Nx, Ny)
        -> (Fx, Fy)

MEA-style drag/lift integration for LI-BB boundaries (stationary wall).
`f_post[i, j, q]` must be the post-collision population at the fluid
node after the LI-BB V2 step (same array the kernel wrote). Sums
`2·c_q·f_q` over every fluid cell with `q_wall[i, j, q] > 0`.

For a moving wall the signature generalises to include a Ladd
correction; not needed for Schäfer-Turek (wall stationary).
"""
function compute_drag_libb_2d(f_post, q_wall, Nx::Int, Ny::Int)
    f = Array(f_post)
    qw = Array(q_wall)
    cxv = (0, 1, 0, -1,  0, 1, -1, -1,  1)
    cyv = (0, 0, 1,  0, -1, 1,  1, -1, -1)
    Fx = 0.0
    Fy = 0.0
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

"""
    run_cylinder_libb_2d(; Nx=300, Ny=80, cx=Nx÷4, cy=Ny÷2, radius=10,
                          u_in=0.04, ν=0.04, max_steps=50000,
                          avg_window=5000, inlet=:parabolic)

2D cylinder with LI-BB V2 boundary condition. Zou-He inlet (uniform
or parabolic), pressure outlet via Zou-He, halfway-BB top/bottom walls
(fused kernel fallback at j=1 and j=Ny).

Arguments:
- `u_in`: inlet velocity scale. Interpreted as centerline u_max for
  parabolic inlet; as uniform velocity for `:uniform` inlet.
- `inlet`: `:parabolic` (Schäfer-Turek 2D-1 convention) or `:uniform`.

Returns a NamedTuple with:
- `ρ`, `ux`, `uy`, `is_solid`, `q_wall`, `u_ref`, `D`, `Fx`, `Fy`
- `Cd = 2·Fx / (u_ref² · D)` with `u_ref = u_mean = 2/3 · u_max`
  (parabolic) or `u_ref = u_in` (uniform).
- `Cl` analogously.

Reference for Schäfer-Turek 2D-1 (Re=20, blockage 10%): Cd ≈ 5.58.
"""
function run_cylinder_libb_2d(; Nx::Int=300, Ny::Int=80,
                                cx::Union{Nothing,Real}=nothing,
                                cy::Union{Nothing,Real}=nothing,
                                radius::Real=10,
                                u_in::Real=0.04, ν::Real=0.04,
                                max_steps::Int=50_000,
                                avg_window::Int=5_000,
                                inlet::Symbol=:parabolic,
                                T::Type{<:AbstractFloat}=Float64)
    cx = isnothing(cx) ? Nx ÷ 4 : Float64(cx)
    cy = isnothing(cy) ? Ny ÷ 2 : Float64(cy)

    q_wall, is_solid = precompute_q_wall_cylinder(Nx, Ny, cx, cy, radius; FT=T)
    uw_x = zeros(T, Nx, Ny, 9)
    uw_y = zeros(T, Nx, Ny, 9)

    u_profile = if inlet === :parabolic
        # u(y) = 4·u_max·(j-1)·(Ny-j) / (Ny-1)²  → peaks at u_max at center
        [T(4) * T(u_in) * T(j - 1) * T(Ny - j) / T(Ny - 1)^2 for j in 1:Ny]
    elseif inlet === :uniform
        fill(T(u_in), Ny)
    else
        error("unknown inlet $(inlet); expected :parabolic or :uniform")
    end
    uy_profile = zeros(T, Ny)
    u_ref = inlet === :parabolic ? (2 / 3) * Float64(u_in) : Float64(u_in)

    # Equilibrium init at the inlet profile; system converges over
    # ν·max_steps time units.
    f_in  = zeros(T, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx, q in 1:9
        f_in[i, j, q] = Kraken.equilibrium(D2Q9(), one(T), u_profile[j],
                                            zero(T), q)
    end
    f_out = similar(f_in)
    ρ  = ones(T, Nx, Ny)
    ux = zeros(T, Nx, Ny)
    uy = zeros(T, Nx, Ny)

    Fx_sum = 0.0
    Fy_sum = 0.0
    n_avg  = 0

    for step in 1:max_steps
        fused_trt_libb_v2_step!(f_out, f_in, ρ, ux, uy, is_solid,
                                 q_wall, uw_x, uw_y, Nx, Ny, T(ν))

        # Inlet (west): equilibrium enforcement at ρ=1, u=u_profile[j].
        @inbounds for j in 1:Ny, q in 1:9
            f_out[1, j, q] = Kraken.equilibrium(D2Q9(), one(T),
                                                u_profile[j], zero(T), q)
        end
        # Outlet (east): Neumann (zero-gradient) on f — copy from i=Nx-1.
        @inbounds for j in 1:Ny, q in 1:9
            f_out[Nx, j, q] = f_out[Nx-1, j, q]
        end

        if step > max_steps - avg_window
            drag = compute_drag_libb_2d(f_out, q_wall, Nx, Ny)
            Fx_sum += drag.Fx
            Fy_sum += drag.Fy
            n_avg  += 1
        end

        f_in, f_out = f_out, f_in
    end

    Fx_avg = Fx_sum / n_avg
    Fy_avg = Fy_sum / n_avg
    D      = 2 * Float64(radius)
    Cd     = 2.0 * Fx_avg / (u_ref^2 * D)
    Cl     = 2.0 * Fy_avg / (u_ref^2 * D)

    return (; ρ, ux, uy, Cd, Cl, Fx = Fx_avg, Fy = Fy_avg,
            q_wall, is_solid, u_ref, D, inlet)
end
