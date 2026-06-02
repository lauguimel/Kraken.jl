# --- Viscoelastic simulation drivers (lbm branch, f[i,j,q] layout) ---

const _D2Q9_CX_VISCO = (0, 1, 0, -1,  0, 1, -1, -1,  1)
const _D2Q9_CY_VISCO = (0, 0, 1,  0, -1, 1,  1, -1, -1)

"""
    reconstruct_wall_link_value_2d(field, i, j, q, q_w; location=:cut, order=1)

Reconstruct a smooth physical field at a wall-adjacent cell or at the
cut-point of link `q`, using only interior fluid samples along `-c_q`.
`location=:cell` evaluates at the wall-adjacent cell center (`x=0`);
`location=:cut` evaluates at the cut point (`x=q_w`).  The samples are at
`x=-1,-2` for linear reconstruction and `x=-1,-2,-3` for quadratic
reconstruction. If the required interior samples are out of bounds, the
function falls back to the nearest available lower-order reconstruction.
"""
function reconstruct_wall_link_value_2d(field::AbstractMatrix, i::Integer, j::Integer,
                                        q::Integer, q_w::Real;
                                        location::Symbol=:cut,
                                        order::Integer=1)
    Nx, Ny = size(field)
    cxq = _D2Q9_CX_VISCO[q]
    cyq = _D2Q9_CY_VISCO[q]
    x = location === :cell ? 0.0 :
        (location === :cut ? Float64(q_w) :
         error("unknown reconstruction location $(location); expected :cell or :cut"))

    i1 = i - cxq; j1 = j - cyq
    1 <= i1 <= Nx && 1 <= j1 <= Ny || return Float64(field[i, j])
    y1 = Float64(field[i1, j1])

    i2 = i - 2cxq; j2 = j - 2cyq
    1 <= i2 <= Nx && 1 <= j2 <= Ny || return y1
    y2 = Float64(field[i2, j2])

    if order >= 2
        i3 = i - 3cxq; j3 = j - 3cyq
        if 1 <= i3 <= Nx && 1 <= j3 <= Ny
            y3 = Float64(field[i3, j3])
            return 0.5 * (x + 2.0) * (x + 3.0) * y1 -
                   (x + 1.0) * (x + 3.0) * y2 +
                   0.5 * (x + 1.0) * (x + 2.0) * y3
        end
    end

    return (x + 2.0) * y1 - (x + 1.0) * y2
end

"""
    compute_polymeric_drag_2d(tau_p_xx, tau_p_xy, tau_p_yy, q_wall, Nx, Ny; cx, cy)
        -> (Fx, Fy)

Polymeric stress contribution to the drag on a solid surface.

For a curved LI-BB surface, prefer the `q_wall` method. It evaluates tau at
the actual cut point `x_w = x_f + q_w c_q`, computes the local circle normal
from `(cx, cy)`, and integrates `tau * n ds` with arc-length weights over the
ordered cut points. This is a surface quadrature, unlike the older
solid-neighbour link count.

Sign is consistent with `compute_drag_mea_2d`: in the Newtonian limit
(Wi->0), tau_p is approximately 2 * nu_p * S, and Cd_p + Cd_solvent should
equal the total Cd of a Newtonian fluid with nu_total = nu_s + nu_p.
"""
function compute_polymeric_drag_2d(tau_p_xx, tau_p_xy, tau_p_yy,
                                     q_wall::AbstractArray{<:Real,3},
                                     Nx::Integer, Ny::Integer;
                                     cx::Real, cy::Real,
                                     radius::Union{Nothing,Real}=nothing,
                                     extrapolate::Bool=true,
                                     reconstruction_order::Integer=1,
                                     reconstruction_mode::Symbol=:interior)
    txx = Array(tau_p_xx)
    txy = Array(tau_p_xy)
    tyy = Array(tau_p_yy)
    qw = Array(q_wall)
    cxv = _D2Q9_CX_VISCO
    cyv = _D2Q9_CY_VISCO

    points = Vector{NTuple{7,Float64}}()
    @inbounds for j in 1:Ny, i in 1:Nx, q in 2:9
        q_w = Float64(qw[i, j, q])
        q_w > 0 || continue
        xw = Float64(i - 1) + q_w * Float64(cxv[q])
        yw = Float64(j - 1) + q_w * Float64(cyv[q])
        rx = xw - Float64(cx)
        ry = yw - Float64(cy)
        r = hypot(rx, ry)
        r > 0 || continue
        nx = rx / r
        ny = ry / r

        txx_w = Float64(txx[i, j])
        txy_w = Float64(txy[i, j])
        tyy_w = Float64(tyy[i, j])
        if extrapolate
            if reconstruction_mode === :interior
                txx_w = reconstruct_wall_link_value_2d(txx, i, j, q, q_w;
                                                       location=:cut,
                                                       order=reconstruction_order)
                txy_w = reconstruct_wall_link_value_2d(txy, i, j, q, q_w;
                                                       location=:cut,
                                                       order=reconstruction_order)
                tyy_w = reconstruct_wall_link_value_2d(tyy, i, j, q, q_w;
                                                       location=:cut,
                                                       order=reconstruction_order)
            elseif reconstruction_mode === :wall_cell
                ib = i - cxv[q]
                jb = j - cyv[q]
                if 1 <= ib <= Nx && 1 <= jb <= Ny
                    txx_w += q_w * (txx_w - Float64(txx[ib, jb]))
                    txy_w += q_w * (txy_w - Float64(txy[ib, jb]))
                    tyy_w += q_w * (tyy_w - Float64(tyy[ib, jb]))
                end
            else
                error("unknown reconstruction_mode $(reconstruction_mode); expected :interior or :wall_cell")
            end
        end

        theta = atan(ry, rx)
        push!(points, (theta, r, nx, ny, txx_w, txy_w, tyy_w))
    end

    isempty(points) && return (Fx=0.0, Fy=0.0)
    sort!(points; by=first)
    R = if isnothing(radius)
        r_sum = 0.0
        for p in points
            r_sum += p[2]
        end
        r_sum / length(points)
    else
        Float64(radius)
    end
    Fx_p = 0.0
    Fy_p = 0.0
    npts = length(points)
    @inbounds for k in 1:npts
        theta_prev = k == 1 ? points[end][1] - 2pi : points[k - 1][1]
        theta_next = k == npts ? points[1][1] + 2pi : points[k + 1][1]
        ds = R * 0.5 * (theta_next - theta_prev)
        _, _, nx, ny, txx_w, txy_w, tyy_w = points[k]
        Fx_p += (txx_w * nx + txy_w * ny) * ds
        Fy_p += (txy_w * nx + tyy_w * ny) * ds
    end
    return (Fx=Fx_p, Fy=Fy_p)
end

"""
    compute_polymeric_drag_2d(tau_p_xx, tau_p_xy, tau_p_yy, is_solid, Nx, Ny)
        → (Fx, Fy)

Polymeric stress contribution to the drag on a solid surface.

For each fluid cell adjacent to a solid neighbor (link q), accumulate the
traction by the fluid on the solid:
    dF = τ_p · n_solid_outward · dA
where n_solid_outward points from solid to fluid = -(cx[q], cy[q]). So:
    dF_x = -(τ_xx·cx + τ_xy·cy)
    dF_y = -(τ_xy·cx + τ_yy·cy)

Sign is consistent with `compute_drag_mea_2d`: in the Newtonian limit
(Wi→0), τ_p ≈ 2·ν_p·S, and Cd_p + Cd_solvent should equal the total
Cd of a Newtonian fluid with ν_total = ν_s + ν_p.

Each link is treated with unit face area (consistent with MEA convention).
"""
function compute_polymeric_drag_2d(tau_p_xx, tau_p_xy, tau_p_yy, is_solid, Nx, Ny;
                                     extrapolate=true)
    txx = Array(tau_p_xx)
    txy = Array(tau_p_xy)
    tyy = Array(tau_p_yy)
    solid = Array(is_solid)
    cxv = [0, 1, 0, -1,  0, 1, -1, -1,  1]
    cyv = [0, 0, 1,  0, -1, 1,  1, -1, -1]

    Fx_p = 0.0
    Fy_p = 0.0

    for j in 1:Ny, i in 1:Nx
        if !solid[i, j]
            for q in 2:9
                ni = i + cxv[q]
                nj = j + cyv[q]
                if 1 <= ni <= Nx && 1 <= nj <= Ny && solid[ni, nj]
                    cx = Float64(cxv[q])
                    cy = Float64(cyv[q])

                    # Extrapolate τ_p from cell center (0.5 dx from wall) to wall
                    # using neighbour 1 cell INTO the fluid (opposite direction)
                    if extrapolate
                        i2 = i - cxv[q]
                        j2 = j - cyv[q]
                        if 1 <= i2 <= Nx && 1 <= j2 <= Ny && !solid[i2, j2]
                            txx_w = 1.5 * txx[i,j] - 0.5 * txx[i2, j2]
                            txy_w = 1.5 * txy[i,j] - 0.5 * txy[i2, j2]
                            tyy_w = 1.5 * tyy[i,j] - 0.5 * tyy[i2, j2]
                        else
                            txx_w = txx[i,j]; txy_w = txy[i,j]; tyy_w = tyy[i,j]
                        end
                    else
                        txx_w = txx[i,j]; txy_w = txy[i,j]; tyy_w = tyy[i,j]
                    end

                    Fx_p -= txx_w * cx + txy_w * cy
                    Fy_p -= txy_w * cx + tyy_w * cy
                end
            end
        end
    end

    return (Fx=Fx_p, Fy=Fy_p)
end

"""
    run_viscoelastic_cylinder_2d(; Nx=400, Ny=80, radius=10, cx=nothing, cy=nothing,
                                   u_in=0.02, ν_s=0.08, ν_p=0.02, lambda=1.0,
                                   L_max=0.0, formulation=:stress,
                                   max_steps=50000, avg_window=5000,
                                   backend=CPU(), FT=Float64)

Flow past a confined cylinder with Oldroyd-B (L_max=0) or FENE-P (L_max>0)
viscoelastic fluid.

Standard benchmark: compare drag coefficient Cd vs Weissenberg number Wi
against literature (Alves et al. 2001, Hulsen et al. 2005, RheoTool).

# Setup
- Confined channel Nx × Ny with cylinder of given radius at (Nx/4, Ny/2)
- Zou-He velocity inlet (west), pressure outlet (east)
- Walls at north/south (bounce-back via streaming)
- Blockage ratio B = 2R/Ny

# Parameters
- `β = ν_s/(ν_s + ν_p)` : viscosity ratio (β=0.59 standard for confined cylinder)
- `Wi = lambda·u_in/radius` : Weissenberg number
- `Re = u_in·2·radius/(ν_s + ν_p)` : Reynolds number
- `formulation`: `:stress` (default) — evolves τ_p directly (UCM); accurate at
  low Wi (validated in `test/test_viscoelastic.jl`) but can blow up (NaN) at
  high Wi. `:logconf` evolves Θ=log(C); it currently has a known singular-source
  bug at isotropy (Θ=0, never activates from rest) and is tracked `@test_broken`
  in `test/test_viscoelastic.jl` — not yet reliable. Neither formulation is
  robust across the full Wi range; this driver's VE is not release-hardened.

Returns `(ux, uy, ρ, Cd, Fx_drag, Fy_drag, tau_p_xx, tau_p_xy, tau_p_yy,
          Theta_xx, Theta_xy, Theta_yy, Re, Wi, beta)`.
"""
function run_viscoelastic_cylinder_2d(;
        Nx=400, Ny=80, radius=10, cx=nothing, cy=nothing,
        u_in=0.02, nu_s=nothing, ν_s=0.08, nu_p=nothing, ν_p=0.02, lambda=1.0,
        L_max=0.0, formulation=:stress,
        max_steps=50000, avg_window=5000,
        backend=KernelAbstractions.CPU(), FT=Float64)
    !isnothing(nu_s) && (ν_s = nu_s)
    !isnothing(nu_p) && (ν_p = nu_p)

    cx = isnothing(cx) ? Nx ÷ 4 : cx
    cy = isnothing(cy) ? Ny ÷ 2 : cy
    D = 2 * radius
    ν_total = ν_s + ν_p
    Re = u_in * D / ν_total
    Wi = lambda * u_in / radius
    beta = ν_s / ν_total
    G = FT(ν_p / lambda)
    ω_s = FT(1.0 / (3.0 * ν_s + 0.5))

    @info "Viscoelastic cylinder" Nx Ny radius Re Wi beta formulation L_max

    # Initialize via Newtonian cylinder setup (uses solvent viscosity)
    state, config = initialize_cylinder_2d(; Nx=Nx, Ny=Ny, cx=cx, cy=cy,
                                            radius=radius, u_in=u_in, ν=ν_s,
                                            backend=backend, T=FT)
    f_in, f_out = state.f_in, state.f_out
    ρ, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid

    # Conformation tensor (log-conf or direct stress)
    Θ_xx     = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Θ_xy     = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Θ_yy     = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Θ_xx_new = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Θ_xy_new = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Θ_yy_new = KernelAbstractions.zeros(backend, FT, Nx, Ny)

    # Polymeric stress
    tau_p_xx = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    tau_p_xy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    tau_p_yy = KernelAbstractions.zeros(backend, FT, Nx, Ny)

    # Polymeric force (input to BGK collision)
    Fx_p = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Fy_p = KernelAbstractions.zeros(backend, FT, Nx, Ny)

    # Drag accumulators (solvent via MEA + polymeric via stress integral)
    Fx_s_sum = 0.0
    Fy_s_sum = 0.0
    Fx_p_sum = 0.0
    Fy_p_sum = 0.0
    n_avg = 0
    poly_drag_cache = build_polymeric_drag_cache_2d(tau_p_xx, is_solid, Nx, Ny)

    for step in 1:max_steps
        # 1. Stream
        stream_2d!(f_out, f_in, Nx, Ny)

        # 2. Boundary conditions
        apply_zou_he_west_2d!(f_out, FT(u_in), Nx, Ny)
        apply_zou_he_pressure_east_2d!(f_out, Nx, Ny)

        # 3a. Solvent drag via MEA (needs pre-stream f_in and post-stream f_out)
        # 3b. Polymeric drag via stress integral on cylinder surface
        if step > max_steps - avg_window
            drag_s = compute_drag_mea_2d(f_in, f_out, is_solid, Nx, Ny)
            drag_p = compute_polymeric_drag_2d_gpu_cached(poly_drag_cache,
                                                           tau_p_xx, tau_p_xy,
                                                           tau_p_yy)
            Fx_s_sum += drag_s.Fx;  Fy_s_sum += drag_s.Fy
            Fx_p_sum += drag_p.Fx;  Fy_p_sum += drag_p.Fy
            n_avg += 1
        end

        # 4. Collide with solvent viscosity + polymeric force
        collide_guo_field_2d!(f_out, is_solid, Fx_p, Fy_p, ω_s)

        # 5. Macroscopic
        compute_macroscopic_2d!(ρ, ux, uy, f_out)

        # 6. Evolve conformation tensor
        if formulation == :logconf
            evolve_logconf_2d!(Θ_xx_new, Θ_xy_new, Θ_yy_new,
                               Θ_xx, Θ_xy, Θ_yy,
                               ux, uy; lambda=Float64(lambda), L_max=Float64(L_max))
        else
            evolve_stress_2d!(Θ_xx_new, Θ_xy_new, Θ_yy_new,
                              Θ_xx, Θ_xy, Θ_yy,
                              ux, uy, Float64(ν_p), Float64(lambda))
        end
        copyto!(Θ_xx, Θ_xx_new); copyto!(Θ_xy, Θ_xy_new); copyto!(Θ_yy, Θ_yy_new)

        # 7. Polymeric stress + force divergence
        if formulation == :logconf
            compute_stress_from_logconf_2d!(tau_p_xx, tau_p_xy, tau_p_yy,
                                            Θ_xx, Θ_xy, Θ_yy;
                                            G=Float64(G), L_max=Float64(L_max))
        else
            copyto!(tau_p_xx, Θ_xx); copyto!(tau_p_xy, Θ_xy); copyto!(tau_p_yy, Θ_yy)
        end
        compute_polymeric_force_2d!(Fx_p, Fy_p, tau_p_xx, tau_p_xy, tau_p_yy)

        # 8. Swap
        f_in, f_out = f_out, f_in
    end

    compute_macroscopic_2d!(ρ, ux, uy, f_in)

    Fx_s = n_avg > 0 ? Fx_s_sum / n_avg : 0.0
    Fy_s = n_avg > 0 ? Fy_s_sum / n_avg : 0.0
    Fx_p = n_avg > 0 ? Fx_p_sum / n_avg : 0.0
    Fy_p = n_avg > 0 ? Fy_p_sum / n_avg : 0.0
    Fx_drag = Fx_s + Fx_p
    Fy_drag = Fy_s + Fy_p
    Cd = 2.0 * Fx_drag / (1.0 * u_in^2 * D)
    Cd_s = 2.0 * Fx_s / (1.0 * u_in^2 * D)
    Cd_p = 2.0 * Fx_p / (1.0 * u_in^2 * D)

    @info "Viscoelastic cylinder result" Cd Cd_s Cd_p Fx_s Fx_p

    return (ux=Array(ux), uy=Array(uy), ρ=Array(ρ),
            Cd=Cd, Cd_s=Cd_s, Cd_p=Cd_p,
            Fx_drag=Fx_drag, Fy_drag=Fy_drag,
            tau_p_xx=Array(tau_p_xx), tau_p_xy=Array(tau_p_xy), tau_p_yy=Array(tau_p_yy),
            Theta_xx=Array(Θ_xx), Theta_xy=Array(Θ_xy), Theta_yy=Array(Θ_yy),
            Re=Re, Wi=Wi, beta=beta)
end
