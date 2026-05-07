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
        → (Fx, Fy)

Polymeric stress contribution to the drag on a solid surface.

For a curved LI-BB surface, prefer the `q_wall` method. It evaluates τ at
the actual cut point `x_w = x_f + q_w c_q`, computes the local circle normal
from `(cx, cy)`, and integrates `τ·n ds` with arc-length weights over the
ordered cut points. This is a surface quadrature, unlike the older
solid-neighbour link count.

Sign is consistent with `compute_drag_mea_2d`: in the Newtonian limit
(Wi→0), τ_p ≈ 2·ν_p·S, and Cd_p + Cd_solvent should equal the total
Cd of a Newtonian fluid with ν_total = ν_s + ν_p.
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

        θ = atan(ry, rx)
        push!(points, (θ, r, nx, ny, txx_w, txy_w, tyy_w))
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
        θ_prev = k == 1 ? points[end][1] - 2π : points[k - 1][1]
        θ_next = k == npts ? points[1][1] + 2π : points[k + 1][1]
        ds = R * 0.5 * (θ_next - θ_prev)
        _, _, nx, ny, txx_w, txy_w, tyy_w = points[k]
        Fx_p += (txx_w * nx + txy_w * ny) * ds
        Fy_p += (txy_w * nx + tyy_w * ny) * ds
    end
    return (Fx=Fx_p, Fy=Fy_p)
end

function compute_polymeric_drag_2d(tau_p_xx, tau_p_xy, tau_p_yy,
                                     is_solid::AbstractArray{Bool,2},
                                     Nx::Integer, Ny::Integer;
                                     extrapolate=true)
    txx = Array(tau_p_xx)
    txy = Array(tau_p_xy)
    tyy = Array(tau_p_yy)
    solid = Array(is_solid)
    cxv = [0, 1, 0, -1,  0, 1, -1, -1,  1]
    cyv = [0, 0, 1,  0, -1, 1,  1, -1, -1]

    Fx_p = 0.0
    Fy_p = 0.0

    # Staircase/solid-mask fallback: integrate over axis-aligned cell faces
    # only. D2Q9 diagonal fluid-solid neighbours touch at a corner and have
    # zero surface measure; including them overcounts square-obstacle traction
    # by a factor three on linear analytic stress patches.
    for j in 1:Ny, i in 1:Nx
        if !solid[i, j]
            for q in 2:5
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
- `formulation`: `:logconf` (default, stable at high Wi) or `:stress`

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
    cy = isnothing(cy) ? (Ny - 1) / 2 : cy
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
            drag_p = compute_polymeric_drag_2d(tau_p_xx, tau_p_xy, tau_p_yy,
                                                is_solid, Nx, Ny)
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

"""
    run_conformation_cylinder_2d(; Nx=400, Ny=80, radius=10, cx=nothing, cy=nothing,
                                   u_in=0.02, ν_s=0.08, ν_p=0.02, lambda=10.0,
                                   tau_plus=1.0,
                                   max_steps=50_000, avg_window=5_000,
                                   backend=CPU(), FT=Float64)

Confined-cylinder Oldroyd-B benchmark using the TRT-LBM conformation tensor
solver (Liu et al. 2025) — same scheme validated on Poiseuille and pure shear.

Solvent: BGK + Hermite stress source (`collide_viscoelastic_source_guo_2d!`).
Polymer: 3 D2Q9 fields C_xx, C_xy, C_yy advected/diffused/relaxed by TRT.
Walls : standard `stream_2d!` (bounce-back) for both `f` and the `g_*`
        conformation distributions; CNEBB on solid C cells.

Drag :  Cd = Cd_s (MEA on f). Because the Hermite stress source injects
        τ_p directly into the populations f, the MEA drag already captures
        the full effective shear (solvent + polymer). The separate stress
        integral Cd_p is reported as a diagnostic only — adding it to Cd_s
        would double-count the polymer contribution.

Returns NamedTuple `(ux, uy, ρ, C_xx, C_xy, C_yy,
                     tau_p_xx, tau_p_xy, tau_p_yy,
                     Cd, Cd_s, Cd_p, Re, Wi, beta)`.
"""
function run_conformation_cylinder_2d(;
        Nx=400, Ny=80, radius=10, cx=nothing, cy=nothing,
        u_in=0.02, ν_s=0.08, ν_p=0.02, lambda=10.0,
        tau_plus=1.0,
        max_steps=50_000, avg_window=5_000,
        backend=KernelAbstractions.CPU(), FT=Float64)

    cx = isnothing(cx) ? Nx ÷ 4 : cx
    cy = isnothing(cy) ? (Ny - 1) / 2 : cy
    D = 2 * radius
    ν_total = ν_s + ν_p
    Re = u_in * D / ν_total
    Wi = lambda * u_in / radius
    beta = ν_s / ν_total
    G = FT(ν_p / lambda)
    ω_s = FT(1.0 / (3.0 * ν_s + 0.5))

    @info "Conformation cylinder" Nx Ny radius Re Wi beta lambda tau_plus

    # Newtonian initialization (uses solvent viscosity in collide later)
    state, _ = initialize_cylinder_2d(; Nx=Nx, Ny=Ny, cx=cx, cy=cy,
                                        radius=radius, u_in=u_in, ν=ν_s,
                                        backend=backend, T=FT)
    f_in, f_out = state.f_in, state.f_out
    ρ, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid

    # Conformation fields
    C_xx = KernelAbstractions.zeros(backend, FT, Nx, Ny); fill!(C_xx, FT(1))
    C_xy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    C_yy = KernelAbstractions.zeros(backend, FT, Nx, Ny); fill!(C_yy, FT(1))

    g_xx = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    g_xy = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    g_yy = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    init_conformation_field_2d!(g_xx, C_xx, ux, uy)
    init_conformation_field_2d!(g_xy, C_xy, ux, uy)
    init_conformation_field_2d!(g_yy, C_yy, ux, uy)
    g_xx_buf = similar(g_xx); g_xy_buf = similar(g_xy); g_yy_buf = similar(g_yy)

    # Polymeric stress
    tau_p_xx = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    tau_p_xy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    tau_p_yy = KernelAbstractions.zeros(backend, FT, Nx, Ny)

    Fx_s_sum = 0.0; Fy_s_sum = 0.0
    Fx_p_sum = 0.0; Fy_p_sum = 0.0
    n_avg = 0

    for step in 1:max_steps
        # --- Solvent LBM ---
        stream_2d!(f_out, f_in, Nx, Ny)
        apply_zou_he_west_2d!(f_out, FT(u_in), Nx, Ny)
        apply_zou_he_pressure_east_2d!(f_out, Nx, Ny)

        if step > max_steps - avg_window
            drag_s = compute_drag_mea_2d(f_in, f_out, is_solid, Nx, Ny)
            drag_p = compute_polymeric_drag_2d(tau_p_xx, tau_p_xy, tau_p_yy,
                                                is_solid, Nx, Ny)
            Fx_s_sum += drag_s.Fx;  Fy_s_sum += drag_s.Fy
            Fx_p_sum += drag_p.Fx;  Fy_p_sum += drag_p.Fy
            n_avg += 1
        end

        collide_viscoelastic_source_guo_2d!(f_out, is_solid, ω_s,
                                              FT(0), FT(0),
                                              tau_p_xx, tau_p_xy, tau_p_yy)
        compute_macroscopic_2d!(ρ, ux, uy, f_out)

        # --- Conformation LBM (TRT) ---
        # Stream into buffers; CNEBB needs BOTH pre- (g) and post-stream (g_buf).
        stream_2d!(g_xx_buf, g_xx, Nx, Ny)
        stream_2d!(g_xy_buf, g_xy, Nx, Ny)
        stream_2d!(g_yy_buf, g_yy, Nx, Ny)

        # Conservative non-equilibrium bounce-back at fluid cells adjacent
        # to solid (Liu et al. 2025, Eqs 38-39). Updates both g_*_buf
        # populations and the C_* macroscopic field at near-wall cells.
        apply_cnebb_conformation_2d!(g_xx_buf, g_xx, is_solid, C_xx, ux, uy)
        apply_cnebb_conformation_2d!(g_xy_buf, g_xy, is_solid, C_xy, ux, uy)
        apply_cnebb_conformation_2d!(g_yy_buf, g_yy, is_solid, C_yy, ux, uy)

        g_xx, g_xx_buf = g_xx_buf, g_xx
        g_xy, g_xy_buf = g_xy_buf, g_xy
        g_yy, g_yy_buf = g_yy_buf, g_yy

        compute_conformation_macro_2d!(C_xx, g_xx)
        compute_conformation_macro_2d!(C_xy, g_xy)
        compute_conformation_macro_2d!(C_yy, g_yy)

        collide_conformation_2d!(g_xx, C_xx, ux, uy, C_xx, C_xy, C_yy, is_solid, tau_plus, lambda; component=1)
        collide_conformation_2d!(g_xy, C_xy, ux, uy, C_xx, C_xy, C_yy, is_solid, tau_plus, lambda; component=2)
        collide_conformation_2d!(g_yy, C_yy, ux, uy, C_xx, C_xy, C_yy, is_solid, tau_plus, lambda; component=3)

        # Polymeric stress τ_p = G·(C - I)
        @. tau_p_xx = G * (C_xx - one(FT))
        @. tau_p_xy = G * C_xy
        @. tau_p_yy = G * (C_yy - one(FT))

        f_in, f_out = f_out, f_in
    end

    compute_macroscopic_2d!(ρ, ux, uy, f_in)

    Fx_s = n_avg > 0 ? Fx_s_sum / n_avg : 0.0
    Fy_s = n_avg > 0 ? Fy_s_sum / n_avg : 0.0
    Fx_p = n_avg > 0 ? Fx_p_sum / n_avg : 0.0
    Fy_p = n_avg > 0 ? Fy_p_sum / n_avg : 0.0
    # Hermite stress source already injects τ_p into f, so MEA on f
    # captures the full effective shear. Cd_p kept as diagnostic only.
    Fx_drag = Fx_s
    Cd   = 2.0 * Fx_s / (1.0 * u_in^2 * D)
    Cd_s = 2.0 * Fx_s / (1.0 * u_in^2 * D)
    Cd_p = 2.0 * Fx_p / (1.0 * u_in^2 * D)

    @info "Conformation cylinder result" Cd Cd_s Cd_p Re Wi

    return (ux=Array(ux), uy=Array(uy), ρ=Array(ρ),
            C_xx=Array(C_xx), C_xy=Array(C_xy), C_yy=Array(C_yy),
            tau_p_xx=Array(tau_p_xx), tau_p_xy=Array(tau_p_xy), tau_p_yy=Array(tau_p_yy),
            Cd=Cd, Cd_s=Cd_s, Cd_p=Cd_p,
            Re=Re, Wi=Wi, beta=beta)
end

"""
    run_conformation_cylinder_libb_2d(; Nx, Ny, radius, cx, cy,
                                         u_mean, ν_s, ν_p, lambda, tau_plus,
                                         inlet=:parabolic, ρ_out=1.0,
                                         max_steps, avg_window,
                                         drag_stride=1,
                                         drag_mode=:post_source_mea,
                                         hermite_source_mode=:liu_direct,
                                         conformation_magic=0.01,
                                         momentum_exchange_mode=:mei_reconstruct,
                                         backend, FT)

Confined-cylinder Oldroyd-B benchmark using:
- **Fused TRT + Bouzidi LI-BB V2** for the solvent flow (sub-cell accurate
  curved wall via precomputed `q_wall`; halfway-BB on N/S channel walls).
- **Modular BCSpec** with `ZouHeVelocity(Poiseuille profile)` at inlet,
  `ZouHePressure(ρ_out)` at outlet. Inlet is fully-developed Poiseuille
  if `inlet=:parabolic`, uniform plug flow if `:uniform`.
- **Mei-consistent MEA drag** (`compute_drag_libb_mei_2d`) for the total
  Liu/Yu-style force, sampled after the Hermite polymer source is injected.
  The explicit polymer surface-stress split is retained as a diagnostic.
- TRT conformation LBM (Liu 2025) + CNEBB for the polymer stress.
- Hermite source added post-collision on the fused-stepped f_out.

`u_mean` is the reference velocity for `Cd = 2·Fx/(u_mean²·D)`:
- `:parabolic` inlet: u_mean = 2/3·u_max (Schäfer-Turek / Liu convention)
- `:uniform` inlet:   u_mean = u_in  (plug flow)

`drag_mode` controls the reported `Cd`:
- `:post_source_mea` (default): raw MEA after source injection, matching the
  coupled discrete Liu/Yu force path. The low-Wi Newtonian-limit coarse canary
  converges with this mode as the cut-link cylinder is refined.
- `:explicit_split`: `Cd = Cd_s + Cd_p`, the explicit polymer surface-traction
  diagnostic. It is useful for stress quadrature audits but under-counts the
  coupled discrete force near cut-links on the Newtonian-limit canary.
- `:source_scaled_mea`: diagnostic cancellation path retained for audits only;
  callers must pass `allow_diagnostic_force_mode=true`.

`hermite_source_mode` controls the standalone post-collision stress source:
- `:liu_direct` (default): no `1/(1-s_plus/2)` denominator, matching Liu/Yu's
  in-collision source amplitude for force-accounting audits.
- `:ce_corrected`: applies the standard post-collision CE denominator for bulk
  coupling studies; do not pair it with `:source_scaled_mea` for validation.

`solvent_source_mode` controls where the Hermite stress source is inserted:
- `:post_collision` (default): current split path, `TRT+LI-BB` then
  `apply_hermite_source_2d!`.
- `:integrated_collision`: experimental path, source fused into the
  `TRT+LI-BB` collision kernel.

`conformation_magic` is the TRT magic parameter Λₚ for the conformation/log-
conformation populations, i.e. `tau_minus = Λₚ/(tau_plus-0.5)+0.5`. With the
validated `tau_plus=1.0` window, very small Λₚ puts the anti-symmetric
relaxation almost at its stability limit; the default `0.01` is the
high-Wi/corner-stable production value. Use `1e-6` only for Liu-parameter
audits where the canary explicitly freezes that behavior.
`conformation_collision` windows are guarded by the analytic CDE patch tests:
TRT is validated at `tau_plus=1.0`, while `:regularized` and `:liu_eq26` are
validated at `tau_plus=0.50001`. Other combinations require
`allow_diagnostic_conformation_collision=true`.
`conformation_gradient_mode` selects the velocity-gradient path used by the
direct-C TRT source:
- `:wall_aware` (default): existing in-kernel wall-aware finite differences.
- `:embedded_axis`: precomputed compact axis stencils with wall samples.
- `:wallfit4`: precomputed wall-constrained polynomial stencils.

`momentum_exchange_mode` selects how cut-link force is evaluated from the
post-boundary populations:
- `:mei_reconstruct` (default): existing LI-BB reconstruction path.
- `:liu_eq63`: explicit Liu/Yu Eq. 63 diagnostic; identical reconstruction
  path, but named to make the force equation selection auditable.
- `:simple_halfway`: old `2c_q f_q` diagnostic.
- `:postpair`: diagnostic direct pair form using the already-overwritten
  reflected population in `f_out`.

Returns `(ux, uy, ρ, C_xx, C_xy, C_yy, tau_p_*, Cd, Cd_s, Cd_p,
Cd_mea_post_source, Cd_mea_source_scaled, Cd_split_explicit, drag_mode,
hermite_source_mode, conformation_magic, momentum_exchange_mode, Re, Wi,
beta, u_ref, D)`.
`n_drag_samples` reports how many cut-link MEA samples were averaged.
"""
function run_conformation_cylinder_libb_2d(;
        Nx=600, Ny=120, radius=30, cx=nothing, cy=nothing,
        u_mean=0.02, ν_s=0.354, ν_p=nothing, lambda=10.0,
        polymer_model::Union{Nothing,AbstractPolymerModel}=nothing,
        polymer_bc::AbstractPolymerWallBC=CNEBB(),
        bcspec::Union{Nothing,BCSpec2D}=nothing,
        inlet::Symbol=:parabolic, ρ_out=1.0,
        tau_plus=1.0,
        max_steps=100_000, avg_window=10_000, drag_stride::Int=1,
        drag_mode::Symbol=:post_source_mea,
        hermite_source_mode::Symbol=:liu_direct,
        solvent_source_mode::Symbol=:post_collision,
        solvent_magic::Real=3/16,
        conformation_magic::Real=0.01,
        conformation_collision::Symbol=:trt,
        conformation_divergence_mode::Symbol=:trace_free,
        conformation_gradient_mode::Symbol=:wall_aware,
        conformation_gradient_max_terms::Union{Nothing,Integer}=nothing,
        conformation_gradient_degree::Integer=4,
        conformation_gradient_radius::Integer=3,
        conformation_gradient_wall_weight::Real=16.0,
        conformation_initial_condition::Symbol=:identity,
        wall_geometry::Symbol=:cutlink,
        momentum_exchange_mode::Symbol=:mei_reconstruct,
        source_stress_reconstruction::Symbol=:interior,
        source_stress_reconstruction_order::Integer=2,
        source_scale_dynamics::Union{Nothing,Real}=nothing,
        hydrodynamic_warmup_steps::Integer=0,
        solvent_source_on_domain_walls::Bool=false,
        diagnostic_interval::Integer=0,
        allow_diagnostic_polymer_bc::Bool=false,
        allow_diagnostic_force_mode::Bool=false,
        allow_diagnostic_conformation_collision::Bool=false,
        allow_diagnostic_log_wall_bc::Bool=false,
        backend=KernelAbstractions.CPU(), FT=Float64)
    drag_stride > 0 || error("drag_stride must be positive")
    drag_mode in (:source_scaled_mea, :post_source_mea, :explicit_split) ||
        error("unknown drag_mode $(drag_mode); expected :source_scaled_mea, :post_source_mea, or :explicit_split")
    hermite_source_mode in (:liu_direct, :ce_corrected) ||
        error("unknown hermite_source_mode $(hermite_source_mode); expected :liu_direct or :ce_corrected")
    solvent_source_mode in (:post_collision, :integrated_collision) ||
        error("unknown solvent_source_mode $(solvent_source_mode); expected :post_collision or :integrated_collision")
    conformation_collision in (:trt, :regularized, :liu_eq26) ||
        error("unknown conformation_collision $(conformation_collision); expected :trt, :regularized, or :liu_eq26")
    conformation_divergence_mode in (:numerical, :trace_free, :trace_free_conservative) ||
        error("unknown conformation_divergence_mode $(conformation_divergence_mode); expected :numerical, :trace_free, or :trace_free_conservative")
    conformation_gradient_mode in (:wall_aware, :embedded_axis, :wallfit4) ||
        error("unknown conformation_gradient_mode $(conformation_gradient_mode); expected :wall_aware, :embedded_axis, or :wallfit4")
    conformation_initial_condition in (:identity, :inlet_profile) ||
        error("unknown conformation_initial_condition $(conformation_initial_condition); expected :identity or :inlet_profile")
    wall_geometry in (:cutlink, :staircase) ||
        error("unknown wall_geometry $(wall_geometry); expected :cutlink or :staircase")
    momentum_exchange_mode in (:mei_reconstruct, :liu_eq63, :simple_halfway, :postpair) ||
        error("unknown momentum_exchange_mode $(momentum_exchange_mode); expected :mei_reconstruct, :liu_eq63, :simple_halfway, or :postpair")
    source_stress_reconstruction in (:raw, :interior) ||
        error("unknown source_stress_reconstruction $(source_stress_reconstruction); expected :raw or :interior")
    source_stress_reconstruction_order in (1, 2) ||
        error("source_stress_reconstruction_order must be 1 or 2")
    hydrodynamic_warmup_steps >= 0 ||
        error("hydrodynamic_warmup_steps must be non-negative")
    _assert_validation_polymer_wall_bc(polymer_bc;
                                       allow_diagnostic=allow_diagnostic_polymer_bc)
    if drag_mode === :source_scaled_mea && !allow_diagnostic_force_mode
        error("drag_mode=:source_scaled_mea is a diagnostic cancellation path, not a validation force law; use :post_source_mea or pass allow_diagnostic_force_mode=true in audit code.")
    end
    _assert_validation_conformation_collision_window(conformation_collision, tau_plus;
        allow_diagnostic=allow_diagnostic_conformation_collision)

    # Resolve polymer model: explicit `polymer_model` wins; else build
    # an Oldroyd-B from the scalar kwargs (`ν_p` + `lambda`).
    if polymer_model === nothing
        isnothing(ν_p) && error("run_conformation_cylinder_libb_2d: supply either `polymer_model` or (`ν_p`, `lambda`).")
        G_ = FT(ν_p / lambda)
        polymer_model = OldroydB(G=G_, λ=FT(lambda))
    end
    _assert_validation_log_wall_bc(polymer_model, polymer_bc;
        allow_diagnostic=allow_diagnostic_log_wall_bc)
    if conformation_gradient_mode !== :wall_aware &&
       (!uses_log_conformation(polymer_model) && conformation_collision !== :trt)
        error("conformation_gradient_mode=$(conformation_gradient_mode) is currently implemented only for TRT direct-C/log-conformation collisions")
    end
    λ_p = polymer_relaxation_time(polymer_model)
    ν_p_eff = polymer_modulus(polymer_model) * λ_p   # G·λ = ν_p (Oldroyd-B)

    cx_f = isnothing(cx) ? Nx / 4 : Float64(cx)
    cy_f = isnothing(cy) ? (Ny - 1) / 2 : Float64(cy)
    D = 2 * Float64(radius)
    ν_total = ν_s + ν_p_eff
    beta = ν_s / ν_total
    # s_plus for TRT = ω for BGK = 1/(3ν_s + 0.5)
    s_plus_s = 1.0 / (3.0 * ν_s + 0.5)

    # Inlet centreline velocity — u_max = 1.5·u_mean for Poiseuille so
    # that area-averaged velocity equals u_mean (Schäfer-Turek / Liu).
    u_max = inlet === :parabolic ? FT(1.5) * FT(u_mean) : FT(u_mean)
    u_ref = Float64(u_mean)
    Re_R = u_ref * Float64(radius) / ν_total
    Re_D = u_ref * D / ν_total
    Re = Re_R
    Wi = λ_p * u_ref / radius

    magic_p = FT(conformation_magic)

    source_scale_dynamics = isnothing(source_scale_dynamics) ?
        1.0 : Float64(source_scale_dynamics)

    @info "Conformation cylinder (LI-BB V2)" Nx Ny radius Re Re_R Re_D Wi beta λ_p tau_plus solvent_magic conformation_magic conformation_collision conformation_divergence_mode conformation_gradient_mode conformation_initial_condition hydrodynamic_warmup_steps wall_geometry inlet u_max u_ref drag_mode hermite_source_mode solvent_source_mode momentum_exchange_mode source_stress_reconstruction source_stress_reconstruction_order source_scale_dynamics solvent_source_on_domain_walls polymer_bc=typeof(polymer_bc) polymer_model=typeof(polymer_model)

    # Precompute cylinder cut-link geometry (q_wall ∈ [0,1] per link)
    q_wall_h, is_solid_h = precompute_q_wall_cylinder(Nx, Ny, cx_f, cy_f,
                                                        Float64(radius); FT=FT)
    if wall_geometry === :staircase
        @inbounds for q in 2:9, j in 1:Ny, i in 1:Nx
            q_wall_h[i, j, q] > 0 && (q_wall_h[i, j, q] = FT(0.5))
        end
    end
    conformation_gradient_stencils =
        conformation_gradient_mode === :wall_aware ? nothing :
        precompute_conformation_gradient_stencils_2d(
            is_solid_h, q_wall_h; mode=conformation_gradient_mode,
            max_terms=conformation_gradient_max_terms,
            degree=conformation_gradient_degree,
            radius=conformation_gradient_radius,
            wall_weight=conformation_gradient_wall_weight,
            backend=backend, FT=FT,
        )
    conformation_gradient_stats = conformation_gradient_stencils === nothing ?
        nothing : conformation_gradient_stencil_stats_2d(conformation_gradient_stencils)

    # Inlet velocity profile (only used if `bcspec === nothing`)
    u_prof_h = if inlet === :parabolic
        [FT(4) * u_max * FT(j - 1) * FT(Ny - j) / FT(Ny - 1)^2 for j in 1:Ny]
    elseif inlet === :uniform
        fill(FT(u_mean), Ny)
    else
        error("unknown inlet $(inlet); expected :parabolic or :uniform")
    end

    # Device allocations
    q_wall   = KernelAbstractions.allocate(backend, FT, Nx, Ny, 9)
    is_solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny)
    uw_x     = KernelAbstractions.allocate(backend, FT, Nx, Ny, 9)
    uw_y     = KernelAbstractions.allocate(backend, FT, Nx, Ny, 9)
    f_in     = KernelAbstractions.allocate(backend, FT, Nx, Ny, 9)
    f_out    = KernelAbstractions.allocate(backend, FT, Nx, Ny, 9)
    ρ        = KernelAbstractions.allocate(backend, FT, Nx, Ny)
    ux       = KernelAbstractions.allocate(backend, FT, Nx, Ny)
    uy       = KernelAbstractions.allocate(backend, FT, Nx, Ny)
    u_profile = KernelAbstractions.allocate(backend, FT, Ny)
    copyto!(q_wall, q_wall_h); copyto!(is_solid, is_solid_h)
    fill!(uw_x, zero(FT)); fill!(uw_y, zero(FT))
    fill!(ρ, one(FT)); fill!(ux, zero(FT)); fill!(uy, zero(FT))
    copyto!(u_profile, u_prof_h)

    # Build default BCSpec if user didn't pass one
    if bcspec === nothing
        bcspec = BCSpec2D(; west = ZouHeVelocity(u_profile),
                            east = ZouHePressure(FT(ρ_out)))
    end

    # Initialize f to equilibrium at the inlet profile velocity
    f_in_h = zeros(FT, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx, q in 1:9
        f_in_h[i,j,q] = Kraken.equilibrium(D2Q9(), one(FT), u_prof_h[j],
                                             zero(FT), q)
    end
    copyto!(f_in, f_in_h); fill!(f_out, zero(FT))

    ux_h = zeros(FT, Nx, Ny)
    uy_h = zeros(FT, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        if !is_solid_h[i, j]
            ux_h[i, j] = u_prof_h[j]
        end
    end
    copyto!(ux, ux_h)
    copyto!(uy, uy_h)

    if hydrodynamic_warmup_steps > 0
        for _ in 1:hydrodynamic_warmup_steps
            fused_trt_libb_v2_step!(f_out, f_in, ρ, ux, uy, is_solid,
                                      q_wall, uw_x, uw_y, Nx, Ny, FT(ν_s);
                                      Λ = solvent_magic)
            apply_bc_rebuild_2d!(f_out, f_in, bcspec, ν_s, Nx, Ny;
                                 Λ = solvent_magic)
            f_in, f_out = f_out, f_in
        end
        compute_macroscopic_2d!(ρ, ux, uy, f_in; sync=true)
    end

    # Evolved field: C (direct) or Ψ = log(C) (log-conformation).
    # At quiescent equilibrium C = I ⇒ Ψ = 0.
    use_logconf = uses_log_conformation(polymer_model)

    # Analytical Oldroyd-B inlet C profile (Liu Eq 62):
    #   C_xy(y) = λ · ∂u/∂y,  C_xx(y) = 1 + 2·(λ·∂u/∂y)²,  C_yy = 1
    H_chan = FT(Ny)
    C_xx_inlet_h = ones(FT, Ny)
    C_xy_inlet_h = zeros(FT, Ny)
    C_yy_inlet_h = ones(FT, Ny)
    for j in 1:Ny
        y = FT(j) - FT(0.5)
        dudy = u_max * FT(4) * (H_chan - FT(2)*y) / (H_chan * H_chan)
        C_xy_inlet_h[j] = FT(λ_p) * dudy
        C_xx_inlet_h[j] = FT(1) + FT(2) * (FT(λ_p) * dudy)^2
    end
    if use_logconf
        for j in 1:Ny
            cxx = C_xx_inlet_h[j]
            cxy = C_xy_inlet_h[j]
            cyy = C_yy_inlet_h[j]
            tr = cxx + cyy
            diff = cxx - cyy
            disc = sqrt(diff * diff + FT(4) * cxy * cxy)
            μ1 = FT(0.5) * (tr + disc)
            μ2 = FT(0.5) * (tr - disc)
            l1 = log(max(μ1, FT(1e-30)))
            l2 = log(max(μ2, FT(1e-30)))
            θ = FT(0.5) * atan(FT(2) * cxy, diff)
            c = cos(θ)
            s = sin(θ)
            C_xx_inlet_h[j] = c * c * l1 + s * s * l2
            C_xy_inlet_h[j] = c * s * (l1 - l2)
            C_yy_inlet_h[j] = s * s * l1 + c * c * l2
        end
    end
    C_xx_inlet = KernelAbstractions.allocate(backend, FT, Ny)
    C_xy_inlet = KernelAbstractions.allocate(backend, FT, Ny)
    C_yy_inlet = KernelAbstractions.allocate(backend, FT, Ny)
    copyto!(C_xx_inlet, C_xx_inlet_h)
    copyto!(C_xy_inlet, C_xy_inlet_h)
    copyto!(C_yy_inlet, C_yy_inlet_h)

    C_xx = KernelAbstractions.zeros(backend, FT, Nx, Ny); fill!(C_xx, FT(1))
    C_xy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    C_yy = KernelAbstractions.zeros(backend, FT, Nx, Ny); fill!(C_yy, FT(1))

    # `Ψ_*` alias `C_*` for direct conformation so one code path handles
    # both. For log-conformation we allocate separate Ψ fields.
    Ψ_xx = use_logconf ? KernelAbstractions.zeros(backend, FT, Nx, Ny) : C_xx
    Ψ_xy = use_logconf ? KernelAbstractions.zeros(backend, FT, Nx, Ny) : C_xy
    Ψ_yy = use_logconf ? KernelAbstractions.zeros(backend, FT, Nx, Ny) : C_yy
    if conformation_initial_condition === :inlet_profile
        Ψ_xx_h = zeros(FT, Nx, Ny)
        Ψ_xy_h = zeros(FT, Nx, Ny)
        Ψ_yy_h = zeros(FT, Nx, Ny)
        for j in 1:Ny, i in 1:Nx
            if is_solid_h[i, j]
                Ψ_xx_h[i, j] = use_logconf ? zero(FT) : one(FT)
                Ψ_xy_h[i, j] = zero(FT)
                Ψ_yy_h[i, j] = use_logconf ? zero(FT) : one(FT)
            else
                Ψ_xx_h[i, j] = C_xx_inlet_h[j]
                Ψ_xy_h[i, j] = C_xy_inlet_h[j]
                Ψ_yy_h[i, j] = C_yy_inlet_h[j]
            end
        end
        copyto!(Ψ_xx, Ψ_xx_h)
        copyto!(Ψ_xy, Ψ_xy_h)
        copyto!(Ψ_yy, Ψ_yy_h)
        use_logconf && psi_to_C_2d!(C_xx, C_xy, C_yy, Ψ_xx, Ψ_xy, Ψ_yy)
    end

    g_xx = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    g_xy = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    g_yy = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    init_conformation_field_2d!(g_xx, Ψ_xx, ux, uy)
    init_conformation_field_2d!(g_xy, Ψ_xy, ux, uy)
    init_conformation_field_2d!(g_yy, Ψ_yy, ux, uy)
    g_xx_buf = similar(g_xx); g_xy_buf = similar(g_xy); g_yy_buf = similar(g_yy)
    Fe_xx_prev = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    Fe_xy_prev = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    Fe_yy_prev = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)

    tau_p_xx = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    tau_p_xy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    tau_p_yy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    tau_p_xx_source = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    tau_p_xy_source = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    tau_p_yy_source = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    update_polymer_stress!(tau_p_xx, tau_p_xy, tau_p_yy,
                             C_xx, C_xy, C_yy, polymer_model)

    Fx_s_sum = 0.0; Fy_s_sum = 0.0
    Fx_mea_post_sum = 0.0; Fy_mea_post_sum = 0.0
    Fx_p_sum = 0.0; Fy_p_sum = 0.0
    n_avg = 0
    first_nonfinite_step = 0

    pre_source_available = solvent_source_mode === :post_collision

    for step in 1:max_steps
        # --- Solvent TRT + LI-BB V2 ---
        if solvent_source_mode === :integrated_collision
            fused_trt_libb_v2_hermite_step!(f_out, f_in, ρ, ux, uy, is_solid,
                                             q_wall, uw_x, uw_y,
                                             tau_p_xx, tau_p_xy, tau_p_yy,
                                             Nx, Ny, FT(ν_s);
                                             Λ = solvent_magic,
                                             source_scale = FT(source_scale_dynamics))
        else
            fused_trt_libb_v2_step!(f_out, f_in, ρ, ux, uy, is_solid,
                                      q_wall, uw_x, uw_y, Nx, Ny, FT(ν_s);
                                      Λ = solvent_magic)
        end

        # Pre-collision Zou-He rebuild at inlet/outlet (fixes halfway-BB corruption)
        apply_bc_rebuild_2d!(f_out, f_in, bcspec, ν_s, Nx, Ny; Λ = solvent_magic)

        sample_drag = step > max_steps - avg_window &&
           ((step - (max_steps - avg_window) - 1) % drag_stride == 0 ||
            step == max_steps)

        # Pre-source cut-link MEA measures the solvent/pressure contribution
        # before direct stress embedding. The explicit link stress integral is
        # retained only as a diagnostic because it uses staircase/link geometry.
        if sample_drag && pre_source_available
            drag_s = momentum_exchange_mode === :postpair ?
                compute_drag_libb_postpair_2d(f_out, q_wall, Nx, Ny) :
                (momentum_exchange_mode === :simple_halfway ?
                 compute_drag_libb_2d(f_out, q_wall, Nx, Ny) :
                 (momentum_exchange_mode === :liu_eq63 ?
                  compute_drag_libb_liu_eq63_2d(f_out, q_wall, uw_x, uw_y, Nx, Ny) :
                  compute_drag_libb_mei_2d(f_out, q_wall, uw_x, uw_y, Nx, Ny)))
            drag_p = compute_polymeric_drag_2d(tau_p_xx, tau_p_xy, tau_p_yy,
                                                q_wall, Nx, Ny;
                                                cx=cx_f, cy=cy_f,
                                                radius=Float64(radius))
            Fx_s_sum += drag_s.Fx;  Fy_s_sum += drag_s.Fy
            Fx_p_sum += drag_p.Fx;  Fy_p_sum += drag_p.Fy
        elseif sample_drag
            drag_p = compute_polymeric_drag_2d(tau_p_xx, tau_p_xy, tau_p_yy,
                                                q_wall, Nx, Ny;
                                                cx=cx_f, cy=cy_f,
                                                radius=Float64(radius))
            Fx_p_sum += drag_p.Fx;  Fy_p_sum += drag_p.Fy
        end

        # Inject Hermite polymer source on post-collision f_out. In the
        # Liu/Yu-style formulation, the total cylinder force is then measured
        # by MEA on these stress-embedded hydrodynamic populations.
        if solvent_source_mode === :post_collision
            source_tau_xx = tau_p_xx
            source_tau_xy = tau_p_xy
            source_tau_yy = tau_p_yy
            if source_stress_reconstruction === :interior
                reconstruct_wall_cell_stress_from_interior_2d!(
                    tau_p_xx_source, tau_p_xy_source, tau_p_yy_source,
                    tau_p_xx, tau_p_xy, tau_p_yy, q_wall, is_solid;
                    order=source_stress_reconstruction_order)
                source_tau_xx = tau_p_xx_source
                source_tau_xy = tau_p_xy_source
                source_tau_yy = tau_p_yy_source
            end
            apply_hermite_source_2d!(f_out, is_solid, s_plus_s,
                                       source_tau_xx, source_tau_xy, source_tau_yy;
                                       ce_correction = hermite_source_mode === :ce_corrected,
                                       source_scale = source_scale_dynamics,
                                       apply_y_domain_walls = solvent_source_on_domain_walls)
        end

        # Total MEA after direct stress embedding. This is the default Cd path
        # because Liu/Yu compute the total drag from hydrodynamic populations,
        # not from a separate explicit ∮τ_p·n ds quadrature.
        if sample_drag
            drag_post = momentum_exchange_mode === :postpair ?
                compute_drag_libb_postpair_2d(f_out, q_wall, Nx, Ny) :
                (momentum_exchange_mode === :simple_halfway ?
                 compute_drag_libb_2d(f_out, q_wall, Nx, Ny) :
                 (momentum_exchange_mode === :liu_eq63 ?
                  compute_drag_libb_liu_eq63_2d(f_out, q_wall, uw_x, uw_y, Nx, Ny) :
                  compute_drag_libb_mei_2d(f_out, q_wall, uw_x, uw_y, Nx, Ny)))
            Fx_mea_post_sum += drag_post.Fx
            Fy_mea_post_sum += drag_post.Fy
            n_avg += 1
        end

        # --- Conformation / log-conformation LBM (TRT) + polymer wall BC ---
        stream_2d!(g_xx_buf, g_xx, Nx, Ny)
        stream_2d!(g_xy_buf, g_xy, Nx, Ny)
        stream_2d!(g_yy_buf, g_yy, Nx, Ny)

        apply_polymer_wall_bc!(g_xx_buf, g_xx, is_solid, q_wall, Ψ_xx, ux, uy, polymer_bc)
        apply_polymer_wall_bc!(g_xy_buf, g_xy, is_solid, q_wall, Ψ_xy, ux, uy, polymer_bc)
        apply_polymer_wall_bc!(g_yy_buf, g_yy, is_solid, q_wall, Ψ_yy, ux, uy, polymer_bc)

        # Fix conformation at inlet (west, i=1) and outlet (east, i=Nx):
        # reset g-populations to equilibrium with analytical C and u_profile.
        # Without this, stream_2d bounce-back at domain edges corrupts C.
        reset_conformation_inlet_2d!(g_xx_buf, C_xx_inlet, u_profile, Ny)
        reset_conformation_inlet_2d!(g_xy_buf, C_xy_inlet, u_profile, Ny)
        reset_conformation_inlet_2d!(g_yy_buf, C_yy_inlet, u_profile, Ny)
        reset_conformation_outlet_2d!(g_xx_buf, Nx, Ny)
        reset_conformation_outlet_2d!(g_xy_buf, Nx, Ny)
        reset_conformation_outlet_2d!(g_yy_buf, Nx, Ny)

        g_xx, g_xx_buf = g_xx_buf, g_xx
        g_xy, g_xy_buf = g_xy_buf, g_xy
        g_yy, g_yy_buf = g_yy_buf, g_yy

        compute_conformation_macro_2d!(Ψ_xx, g_xx)
        compute_conformation_macro_2d!(Ψ_xy, g_xy)
        compute_conformation_macro_2d!(Ψ_yy, g_yy)

        if use_logconf
            conformation_collision in (:regularized, :liu_eq26) &&
                error("$(conformation_collision) conformation collision is only implemented for direct-C, not log-conformation")
            if conformation_gradient_stencils !== nothing
                collide_logconf_2d_with_gradient_stencils!(
                    g_xx, Ψ_xx, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid,
                    uw_x, uw_y, conformation_gradient_stencils,
                    tau_plus, λ_p; magic=magic_p, component=1,
                    divergence_mode=conformation_divergence_mode)
                collide_logconf_2d_with_gradient_stencils!(
                    g_xy, Ψ_xy, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid,
                    uw_x, uw_y, conformation_gradient_stencils,
                    tau_plus, λ_p; magic=magic_p, component=2,
                    divergence_mode=conformation_divergence_mode)
                collide_logconf_2d_with_gradient_stencils!(
                    g_yy, Ψ_yy, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid,
                    uw_x, uw_y, conformation_gradient_stencils,
                    tau_plus, λ_p; magic=magic_p, component=3,
                    divergence_mode=conformation_divergence_mode)
            else
                collide_logconf_2d!(g_xx, Ψ_xx, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid, tau_plus, λ_p; magic=magic_p, component=1, divergence_mode=conformation_divergence_mode)
                collide_logconf_2d!(g_xy, Ψ_xy, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid, tau_plus, λ_p; magic=magic_p, component=2, divergence_mode=conformation_divergence_mode)
                collide_logconf_2d!(g_yy, Ψ_yy, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid, tau_plus, λ_p; magic=magic_p, component=3, divergence_mode=conformation_divergence_mode)
            end
            # Reconstruct C = exp(Ψ) before computing τ_p
            psi_to_C_2d!(C_xx, C_xy, C_yy, Ψ_xx, Ψ_xy, Ψ_yy)
        elseif conformation_collision === :regularized
            collide_conformation_regularized_2d!(g_xx, Ψ_xx, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid, tau_plus, λ_p; magic=magic_p, component=1, divergence_mode=conformation_divergence_mode)
            collide_conformation_regularized_2d!(g_xy, Ψ_xy, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid, tau_plus, λ_p; magic=magic_p, component=2, divergence_mode=conformation_divergence_mode)
            collide_conformation_regularized_2d!(g_yy, Ψ_yy, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid, tau_plus, λ_p; magic=magic_p, component=3, divergence_mode=conformation_divergence_mode)
        elseif conformation_collision === :liu_eq26
            collide_conformation_liu_eq26_2d!(g_xx, Fe_xx_prev, Ψ_xx, ux, uy, ρ, Ψ_xx, Ψ_xy, Ψ_yy, is_solid, tau_plus, λ_p; magic=magic_p, component=1, divergence_mode=conformation_divergence_mode)
            collide_conformation_liu_eq26_2d!(g_xy, Fe_xy_prev, Ψ_xy, ux, uy, ρ, Ψ_xx, Ψ_xy, Ψ_yy, is_solid, tau_plus, λ_p; magic=magic_p, component=2, divergence_mode=conformation_divergence_mode)
            collide_conformation_liu_eq26_2d!(g_yy, Fe_yy_prev, Ψ_yy, ux, uy, ρ, Ψ_xx, Ψ_xy, Ψ_yy, is_solid, tau_plus, λ_p; magic=magic_p, component=3, divergence_mode=conformation_divergence_mode)
        elseif conformation_gradient_stencils !== nothing
            collide_conformation_2d_with_gradient_stencils!(
                g_xx, Ψ_xx, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid,
                uw_x, uw_y, conformation_gradient_stencils,
                tau_plus, λ_p; magic=magic_p, component=1,
                divergence_mode=conformation_divergence_mode)
            collide_conformation_2d_with_gradient_stencils!(
                g_xy, Ψ_xy, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid,
                uw_x, uw_y, conformation_gradient_stencils,
                tau_plus, λ_p; magic=magic_p, component=2,
                divergence_mode=conformation_divergence_mode)
            collide_conformation_2d_with_gradient_stencils!(
                g_yy, Ψ_yy, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid,
                uw_x, uw_y, conformation_gradient_stencils,
                tau_plus, λ_p; magic=magic_p, component=3,
                divergence_mode=conformation_divergence_mode)
        else
            collide_conformation_2d!(g_xx, Ψ_xx, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid, tau_plus, λ_p; magic=magic_p, component=1, divergence_mode=conformation_divergence_mode)
            collide_conformation_2d!(g_xy, Ψ_xy, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid, tau_plus, λ_p; magic=magic_p, component=2, divergence_mode=conformation_divergence_mode)
            collide_conformation_2d!(g_yy, Ψ_yy, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid, tau_plus, λ_p; magic=magic_p, component=3, divergence_mode=conformation_divergence_mode)
        end

        reset_conformation_inlet_2d!(g_xx, C_xx_inlet, u_profile, Ny)
        reset_conformation_inlet_2d!(g_xy, C_xy_inlet, u_profile, Ny)
        reset_conformation_inlet_2d!(g_yy, C_yy_inlet, u_profile, Ny)
        reset_conformation_outlet_2d!(g_xx, Nx, Ny)
        reset_conformation_outlet_2d!(g_xy, Nx, Ny)
        reset_conformation_outlet_2d!(g_yy, Nx, Ny)

        # The reset above fixes the populations used by the next transport
        # step. Refresh the macro fields before computing τ_p, otherwise the
        # hydrodynamic source sees the pre-reset inlet/outlet collision state.
        compute_conformation_macro_2d!(Ψ_xx, g_xx)
        compute_conformation_macro_2d!(Ψ_xy, g_xy)
        compute_conformation_macro_2d!(Ψ_yy, g_yy)
        use_logconf && psi_to_C_2d!(C_xx, C_xy, C_yy, Ψ_xx, Ψ_xy, Ψ_yy)

        update_polymer_stress!(tau_p_xx, tau_p_xy, tau_p_yy,
                                 C_xx, C_xy, C_yy, polymer_model)

        if diagnostic_interval > 0 &&
           (step == 1 || step % diagnostic_interval == 0 || step == max_steps)
            diag = conformation_field_diagnostics_2d(C_xx, C_xy, C_yy, ρ, ux, uy, is_solid)
            @info "Conformation field diagnostics" step finite=diag.finite min_eig_C=diag.min_eig min_i=diag.min_i min_j=diag.min_j min_C_xx=diag.min_C_xx min_C_xy=diag.min_C_xy min_C_yy=diag.min_C_yy max_abs_C=diag.max_abs_C maxC_i=diag.maxC_i maxC_j=diag.maxC_j max_abs_u=diag.max_abs_u max_abs_divu=diag.max_abs_divu maxDiv_i=diag.maxDiv_i maxDiv_j=diag.maxDiv_j max_strain_eig=diag.max_strain_eig maxStrain_i=diag.maxStrain_i maxStrain_j=diag.maxStrain_j min_dudx=diag.min_dudx min_dudy=diag.min_dudy min_dvdx=diag.min_dvdx min_dvdy=diag.min_dvdy min_strain_eig=diag.min_strain_eig first_bad_i=diag.first_bad_i first_bad_j=diag.first_bad_j n_fluid=diag.n_fluid
            if first_nonfinite_step == 0 && (!diag.finite || !(diag.min_eig > 0.0))
                first_nonfinite_step = step
            end
        end

        f_in, f_out = f_out, f_in
    end
    KernelAbstractions.synchronize(backend)

    Fx_s = (n_avg > 0 && pre_source_available) ? Fx_s_sum / n_avg : NaN
    Fy_s = (n_avg > 0 && pre_source_available) ? Fy_s_sum / n_avg : NaN
    Fx_p = n_avg > 0 ? Fx_p_sum / n_avg : 0.0
    Fy_p = n_avg > 0 ? Fy_p_sum / n_avg : 0.0
    Fx_mea_post = n_avg > 0 ? Fx_mea_post_sum / n_avg : 0.0
    Fy_mea_post = n_avg > 0 ? Fy_mea_post_sum / n_avg : 0.0

    Cd_s = 2.0 * Fx_s / (u_ref^2 * D)
    Cd_p = 2.0 * Fx_p / (u_ref^2 * D)
    Cd_mea_post_source = 2.0 * Fx_mea_post / (u_ref^2 * D)
    source_force_scale = (hermite_source_mode === :ce_corrected && pre_source_available) ?
        (1.0 - s_plus_s / 2.0) : 1.0
    Fx_mea_source_scaled = pre_source_available ?
        Fx_s + source_force_scale * (Fx_mea_post - Fx_s) : Fx_mea_post
    Fy_mea_source_scaled = pre_source_available ?
        Fy_s + source_force_scale * (Fy_mea_post - Fy_s) : Fy_mea_post
    Cd_mea_source_scaled = 2.0 * Fx_mea_source_scaled / (u_ref^2 * D)
    Cd_split_explicit = Cd_s + Cd_p
    Fx_split_explicit = Fx_s + Fx_p
    Fy_split_explicit = Fy_s + Fy_p
    Fx = drag_mode === :source_scaled_mea ? Fx_mea_source_scaled :
         (drag_mode === :post_source_mea ? Fx_mea_post : Fx_split_explicit)
    Fy = drag_mode === :source_scaled_mea ? Fy_mea_source_scaled :
         (drag_mode === :post_source_mea ? Fy_mea_post : Fy_split_explicit)
    Cd = drag_mode === :source_scaled_mea ? Cd_mea_source_scaled :
         (drag_mode === :post_source_mea ? Cd_mea_post_source : Cd_split_explicit)
    Cl = 2.0 * Fy / (u_ref^2 * D)

    @info "Conformation cylinder (LI-BB V2) result" Cd Cl drag_mode hermite_source_mode solvent_source_mode solvent_magic conformation_magic conformation_collision momentum_exchange_mode Cd_s Cd_p Cd_split_explicit Cd_mea_post_source Cd_mea_source_scaled source_force_scale Re Re_R Re_D Wi

    return (ux=Array(ux), uy=Array(uy), ρ=Array(ρ),
            C_xx=Array(C_xx), C_xy=Array(C_xy), C_yy=Array(C_yy),
            tau_p_xx=Array(tau_p_xx), tau_p_xy=Array(tau_p_xy), tau_p_yy=Array(tau_p_yy),
            q_wall=Array(q_wall), is_solid=Array(is_solid),
            Cd=Cd, Cl=Cl, Cd_s=Cd_s, Cd_p=Cd_p,
            Cd_mea_post_source=Cd_mea_post_source,
            Cd_mea_source_scaled=Cd_mea_source_scaled,
            Cd_split_explicit=Cd_split_explicit,
            drag_mode=drag_mode,
            hermite_source_mode=hermite_source_mode,
            solvent_source_mode=solvent_source_mode,
            source_stress_reconstruction=source_stress_reconstruction,
            source_stress_reconstruction_order=source_stress_reconstruction_order,
            solvent_source_on_domain_walls=solvent_source_on_domain_walls,
            solvent_magic=Float64(solvent_magic),
            conformation_magic=Float64(conformation_magic),
            conformation_collision=conformation_collision,
            conformation_divergence_mode=conformation_divergence_mode,
            conformation_gradient_mode=conformation_gradient_mode,
            conformation_initial_condition=conformation_initial_condition,
            hydrodynamic_warmup_steps=hydrodynamic_warmup_steps,
            conformation_gradient_stats=conformation_gradient_stats,
            wall_geometry=wall_geometry,
            momentum_exchange_mode=momentum_exchange_mode,
            Fx=Fx, Fy=Fy,
            Fx_s=Fx_s, Fy_s=Fy_s, Fx_p=Fx_p, Fy_p=Fy_p,
            Fx_split_explicit=Fx_split_explicit, Fy_split_explicit=Fy_split_explicit,
            Fx_mea_post_source=Fx_mea_post, Fy_mea_post_source=Fy_mea_post,
            Fx_mea_source_scaled=Fx_mea_source_scaled, Fy_mea_source_scaled=Fy_mea_source_scaled,
            source_scale_dynamics=source_scale_dynamics,
            source_force_scale=source_force_scale,
            first_nonfinite_step=first_nonfinite_step,
            Re=Re, Re_R=Re_R, Re_D=Re_D, Wi=Wi, beta=beta, u_ref=u_ref, D=D,
            n_drag_samples=n_avg)
end
