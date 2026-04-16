# --- Viscoelastic simulation drivers (lbm branch, f[i,j,q] layout) ---

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
    cy = isnothing(cy) ? Ny ÷ 2 : cy
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
        apply_cnebb_conformation_2d!(g_xx_buf, g_xx, is_solid, C_xx)
        apply_cnebb_conformation_2d!(g_xy_buf, g_xy, is_solid, C_xy)
        apply_cnebb_conformation_2d!(g_yy_buf, g_yy, is_solid, C_yy)

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
        @. tau_p_xx = G * (C_xx - 1.0)
        @. tau_p_xy = G * C_xy
        @. tau_p_yy = G * (C_yy - 1.0)

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
