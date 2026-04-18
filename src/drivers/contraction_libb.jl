# --- Planar 4:1 (and N:1) sudden contraction viscoelastic driver ---
#
# Geometry (Alves 2003, J. Non-Newt. Fluid Mech. 110): planar sudden
# contraction with contraction ratio β_c = H_in / H_out (full channel
# heights). Re-entrant sharp corners. Symmetric about cy.
#
#         ┌───────────────────────────────┐
#         │   inlet (full height = Ny)    │ ─── solid block above outlet
#         │                               │ ──┐
# (in)──> │                               │   │  outlet (height = h_out)
#         │                               │ ──┘
#         │                               │ ─── solid block below outlet
#         └───────────────────────────────┘
#         <----- L_up ----><----- L_down ------>
#
# Reuses the validated 2D modular pipeline from `cylinder_libb`:
# - Fused TRT + LI-BB V2 for the solvent flow (q_wall = 0.5 on
#   axis-aligned step walls; no curved walls so the LI-BB reduces to
#   halfway-BB at the step and to the kernel's halfway-BB fallback at
#   N/S domain edges in the upstream channel).
# - Modular `BCSpec2D`: ZouHeVelocity at west, ZouHePressure at east.
# - Conformation TRT-LBM (Liu 2025) + CNEBB at solid walls.
# - Hermite source post-collision on f_out (no double-counting drag).
#
# Outputs the macroscopic ψ/C/τ_p fields and the post-processing helpers
# needed for the canonical Alves 2003 metrics:
# - `vortex_length(...)`: salient corner vortex length X_R
# - `outlet_centerline_N1(...)`: first normal stress along outlet axis
#
# The driver is GPU/CPU agnostic (`backend` kwarg). 3D port (square
# duct contraction) will follow the same template but with D3Q19 + 6
# conformation components — left for a separate file.

"""
    run_conformation_contraction_libb_2d(; H_out, β_c=4, L_up=20, L_down=50,
                                           u_out_mean=0.02,
                                           ν_s=0.354, ν_p=nothing, lambda=10.0,
                                           polymer_model=nothing,
                                           polymer_bc=CNEBB(),
                                           bcspec=nothing,
                                           ρ_out=1.0, tau_plus=1.0,
                                           max_steps=200_000, avg_window=20_000,
                                           backend=CPU(), FT=Float64)

Planar sudden contraction (Alves 2003 4:1 family) for an Oldroyd-B,
log-conformation Oldroyd-B (or any `AbstractPolymerModel`) fluid.

# Geometry
- `H_out` : full downstream channel height (lattice cells, reference scale).
- `β_c`   : contraction ratio H_in / H_out (default 4 for the 4:1 benchmark).
- `L_up`  : upstream length in units of `H_out` (default 20).
- `L_down`: downstream length in units of `H_out` (default 50; use ≥80
            for accurate vortex length at high Wi).

The lattice domain is `Nx = (L_up + L_down) · H_out` by `Ny = β_c · H_out`,
with the contraction step located at `i_step = L_up · H_out + 1`. The
downstream outlet occupies `j ∈ [j_low, j_high]` with `j_low =
(β_c - 1)/2 · H_out + 1` and `j_high = j_low + H_out - 1`.

# Flow
- `u_out_mean` : mean axial velocity in the downstream outlet
  (reference for Re and Wi). The inlet parabolic profile is scaled
  by mass conservation: `u_in_mean = u_out_mean / β_c`.
- `Re = u_out_mean · H_out / (ν_s + ν_p_eff)`.
- `Wi = λ · u_out_mean / (H_out / 2)` (uses outlet half-height).

# Polymer
- `polymer_model` : explicit `AbstractPolymerModel` (preferred). If
  `nothing`, builds an `OldroydB(G=ν_p/λ, λ=lambda)` from `ν_p`+`lambda`.
- `polymer_bc`    : wall BC on conformation populations (default `CNEBB()`).

# Returns
NamedTuple with `ux, uy, ρ, C_xx, C_xy, C_yy, tau_p_*, is_solid,
                 i_step, j_low, j_high, Re, Wi, beta, u_ref, H_out`.
"""
function run_conformation_contraction_libb_2d(;
        H_out::Int=20, β_c::Int=4, L_up::Int=20, L_down::Int=50,
        u_out_mean=0.02, ν_s=0.354, ν_p=nothing, lambda=10.0,
        polymer_model::Union{Nothing,AbstractPolymerModel}=nothing,
        polymer_bc::AbstractPolymerWallBC=CNEBB(),
        bcspec::Union{Nothing,BCSpec2D}=nothing,
        ρ_out=1.0, tau_plus=1.0,
        max_steps=200_000, avg_window=20_000,
        backend=KernelAbstractions.CPU(), FT=Float64)

    # Resolve polymer model
    if polymer_model === nothing
        isnothing(ν_p) && error("supply either `polymer_model` or (`ν_p`, `lambda`).")
        G_ = FT(ν_p / lambda)
        polymer_model = OldroydB(G=G_, λ=FT(lambda))
    end
    λ_p     = polymer_relaxation_time(polymer_model)
    ν_p_eff = polymer_modulus(polymer_model) * λ_p   # G·λ for Oldroyd-B / FENE

    # Geometry in lattice cells. The outlet is centred on the upstream
    # channel: j_low / j_high are placed so that the gap above and below
    # the outlet is identical (within 1 cell when (Ny − H_out) is odd).
    Nx     = (L_up + L_down) * H_out
    Ny     = β_c * H_out
    i_step = L_up * H_out + 1
    j_low  = (Ny - H_out) ÷ 2 + 1
    j_high = j_low + H_out - 1
    isodd(Ny - H_out) && @warn "(Ny − H_out) odd → 1-cell asymmetric outlet" Ny H_out j_low j_high

    # Mass-conservation inlet mean (parabolic on full Ny)
    u_in_mean = FT(u_out_mean) * FT(H_out) / FT(Ny)
    u_in_max  = FT(1.5) * u_in_mean

    ν_total = ν_s + ν_p_eff
    beta    = ν_s / ν_total
    Re      = Float64(u_out_mean) * H_out / ν_total
    Wi      = λ_p * Float64(u_out_mean) / (H_out / 2)

    s_plus_s = 1.0 / (3.0 * ν_s + 0.5)

    @info "Conformation contraction (LI-BB V2)" Nx Ny H_out β_c L_up L_down i_step j_low j_high Re Wi beta λ_p tau_plus polymer_bc=typeof(polymer_bc) polymer_model=typeof(polymer_model) u_out_mean u_in_mean

    # Geometry: solid mask + axis-aligned q_wall (= 0.5 on cut links)
    q_wall_h, is_solid_h = precompute_q_wall_contraction_2d(Nx, Ny;
                              i_step=i_step, j_low=j_low, j_high=j_high, FT=FT)

    # Inlet u-profile: parabolic on full upstream width [1, Ny]
    H_chan_in = FT(Ny)
    u_prof_in_h = [FT(4) * u_in_max * FT(j - 1) * FT(Ny - j) / FT(Ny - 1)^2
                   for j in 1:Ny]

    # Inlet conformation profile (analytical Oldroyd-B, Liu Eq 62
    # generalised for the upstream parabolic shear field)
    C_xx_inlet_h = ones(FT, Ny)
    C_xy_inlet_h = zeros(FT, Ny)
    C_yy_inlet_h = ones(FT, Ny)
    for j in 1:Ny
        y = FT(j) - FT(0.5)
        dudy = u_in_max * FT(4) * (H_chan_in - FT(2)*y) / (H_chan_in * H_chan_in)
        C_xy_inlet_h[j] = FT(λ_p) * dudy
        C_xx_inlet_h[j] = FT(1) + FT(2) * (FT(λ_p) * dudy)^2
    end

    # Device allocations
    q_wall   = KernelAbstractions.allocate(backend, FT,   Nx, Ny, 9)
    is_solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny)
    uw_x     = KernelAbstractions.allocate(backend, FT,   Nx, Ny, 9)
    uw_y     = KernelAbstractions.allocate(backend, FT,   Nx, Ny, 9)
    f_in     = KernelAbstractions.allocate(backend, FT,   Nx, Ny, 9)
    f_out    = KernelAbstractions.allocate(backend, FT,   Nx, Ny, 9)
    ρ        = KernelAbstractions.allocate(backend, FT,   Nx, Ny)
    ux       = KernelAbstractions.allocate(backend, FT,   Nx, Ny)
    uy       = KernelAbstractions.allocate(backend, FT,   Nx, Ny)
    u_profile  = KernelAbstractions.allocate(backend, FT, Ny)
    C_xx_inlet = KernelAbstractions.allocate(backend, FT, Ny)
    C_xy_inlet = KernelAbstractions.allocate(backend, FT, Ny)
    C_yy_inlet = KernelAbstractions.allocate(backend, FT, Ny)
    copyto!(q_wall, q_wall_h); copyto!(is_solid, is_solid_h)
    fill!(uw_x, zero(FT));     fill!(uw_y, zero(FT))
    fill!(ρ, one(FT));         fill!(ux, zero(FT)); fill!(uy, zero(FT))
    copyto!(u_profile, u_prof_in_h)
    copyto!(C_xx_inlet, C_xx_inlet_h)
    copyto!(C_xy_inlet, C_xy_inlet_h)
    copyto!(C_yy_inlet, C_yy_inlet_h)

    # Default BCSpec: ZouHe velocity inlet (parabolic), ZouHe pressure outlet
    if bcspec === nothing
        bcspec = BCSpec2D(; west = ZouHeVelocity(u_profile),
                            east = ZouHePressure(FT(ρ_out)))
    end

    # Initialize f to equilibrium at the inlet velocity profile (extends
    # the upstream parabolic shear to all i — outlet cells then relax to
    # their own parabolic profile during the warm-up).
    f_in_h = zeros(FT, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx, q in 1:9
        u0 = is_solid_h[i, j] ? zero(FT) : u_prof_in_h[j]
        f_in_h[i, j, q] = Kraken.equilibrium(D2Q9(), one(FT), u0, zero(FT), q)
    end
    copyto!(f_in, f_in_h); fill!(f_out, zero(FT))

    # Conformation fields (Ψ aliases C for direct stress; separate for log-conf)
    use_logconf = uses_log_conformation(polymer_model)
    C_xx = KernelAbstractions.zeros(backend, FT, Nx, Ny); fill!(C_xx, FT(1))
    C_xy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    C_yy = KernelAbstractions.zeros(backend, FT, Nx, Ny); fill!(C_yy, FT(1))
    Ψ_xx = use_logconf ? KernelAbstractions.zeros(backend, FT, Nx, Ny) : C_xx
    Ψ_xy = use_logconf ? KernelAbstractions.zeros(backend, FT, Nx, Ny) : C_xy
    Ψ_yy = use_logconf ? KernelAbstractions.zeros(backend, FT, Nx, Ny) : C_yy

    g_xx = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    g_xy = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    g_yy = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    init_conformation_field_2d!(g_xx, Ψ_xx, ux, uy)
    init_conformation_field_2d!(g_xy, Ψ_xy, ux, uy)
    init_conformation_field_2d!(g_yy, Ψ_yy, ux, uy)
    g_xx_buf = similar(g_xx); g_xy_buf = similar(g_xy); g_yy_buf = similar(g_yy)

    tau_p_xx = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    tau_p_xy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    tau_p_yy = KernelAbstractions.zeros(backend, FT, Nx, Ny)

    for step in 1:max_steps
        # --- Solvent TRT + LI-BB V2 ---
        fused_trt_libb_v2_step!(f_out, f_in, ρ, ux, uy, is_solid,
                                  q_wall, uw_x, uw_y, Nx, Ny, FT(ν_s))

        # Pre-collision Zou-He rebuild at inlet/outlet
        apply_bc_rebuild_2d!(f_out, f_in, bcspec, ν_s, Nx, Ny)

        # Inject Hermite polymer source on f_out
        apply_hermite_source_2d!(f_out, is_solid, s_plus_s,
                                   tau_p_xx, tau_p_xy, tau_p_yy)

        # --- Conformation / log-conformation LBM (TRT) + polymer wall BC ---
        stream_2d!(g_xx_buf, g_xx, Nx, Ny)
        stream_2d!(g_xy_buf, g_xy, Nx, Ny)
        stream_2d!(g_yy_buf, g_yy, Nx, Ny)

        apply_polymer_wall_bc!(g_xx_buf, g_xx, is_solid, Ψ_xx, polymer_bc)
        apply_polymer_wall_bc!(g_xy_buf, g_xy, is_solid, Ψ_xy, polymer_bc)
        apply_polymer_wall_bc!(g_yy_buf, g_yy, is_solid, Ψ_yy, polymer_bc)

        # Inlet/outlet conformation fix (analytical at i=1, equilibrium-extrapol
        # at i=Nx). This is the same anti-corruption fix as in the cylinder
        # driver — without it stream_2d's bounce-back at domain edges pollutes C.
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
            collide_logconf_2d!(g_xx, Ψ_xx, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid, tau_plus, λ_p; component=1)
            collide_logconf_2d!(g_xy, Ψ_xy, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid, tau_plus, λ_p; component=2)
            collide_logconf_2d!(g_yy, Ψ_yy, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid, tau_plus, λ_p; component=3)
            psi_to_C_2d!(C_xx, C_xy, C_yy, Ψ_xx, Ψ_xy, Ψ_yy)
        else
            collide_conformation_2d!(g_xx, Ψ_xx, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid, tau_plus, λ_p; component=1)
            collide_conformation_2d!(g_xy, Ψ_xy, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid, tau_plus, λ_p; component=2)
            collide_conformation_2d!(g_yy, Ψ_yy, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid, tau_plus, λ_p; component=3)
        end

        update_polymer_stress!(tau_p_xx, tau_p_xy, tau_p_yy,
                                 C_xx, C_xy, C_yy, polymer_model)

        f_in, f_out = f_out, f_in
    end
    KernelAbstractions.synchronize(backend)

    @info "Conformation contraction (LI-BB V2) result" Re Wi β_c

    return (ux=Array(ux), uy=Array(uy), ρ=Array(ρ),
            C_xx=Array(C_xx), C_xy=Array(C_xy), C_yy=Array(C_yy),
            tau_p_xx=Array(tau_p_xx), tau_p_xy=Array(tau_p_xy), tau_p_yy=Array(tau_p_yy),
            q_wall=Array(q_wall), is_solid=Array(is_solid),
            Nx=Nx, Ny=Ny, i_step=i_step, j_low=j_low, j_high=j_high,
            Re=Re, Wi=Wi, beta=beta, u_ref=Float64(u_out_mean), H_out=H_out)
end

"""
    vortex_length_contraction_2d(ux, uy, is_solid; i_step, j_low,
                                   side=:south) -> (X_R, j_detach)

Estimate the salient-corner vortex length `X_R` (in lattice cells) from
the contraction step at `(i_step, j_low or j_high)`. Walks upstream from
the re-entrant corner along the wall and returns the distance to the
first sign change of `ux` evaluated 1 cell into the fluid (j_low−1 for
the south salient corner, j_high+1 for north).

Convention: `X_R` is reported in cells. Divide by `H_out` for the
dimensionless `X_R / H_out` value (Alves 2003 Table 4).
"""
function vortex_length_contraction_2d(ux::AbstractArray, uy::AbstractArray,
                                       is_solid::AbstractArray;
                                       i_step::Int, j_low::Int, j_high::Int,
                                       side::Symbol=:south)
    j_probe = side === :south ? j_low - 1 :
              side === :north ? j_high + 1 :
              error("side ∈ (:south, :north)")
    1 ≤ j_probe ≤ size(ux, 2) || error("j_probe $(j_probe) out of domain")
    Nx = size(ux, 1)
    # Walk upstream from i = i_step − 1, stop at first ux > 0 (re-attachment)
    sign_at_corner = sign(ux[i_step - 1, j_probe])
    sign_at_corner == 0 && return (0.0, j_probe)
    X_R = 0
    for di in 1:(i_step - 2)
        i = i_step - 1 - di
        if sign(ux[i, j_probe]) != sign_at_corner
            X_R = di
            break
        end
    end
    return (Float64(X_R), j_probe)
end

"""
    outlet_centerline_N1_contraction_2d(tau_p_xx, tau_p_yy;
                                          i_step, j_low, j_high) -> N1::Vector

First normal stress difference `N1(x) = τ_p_xx − τ_p_yy` along the
downstream centreline `j = (j_low + j_high) ÷ 2`, from the contraction
step (i = i_step) to the outlet (i = Nx). Used as a benchmark metric in
Alves 2003 Fig. 9.
"""
function outlet_centerline_N1_contraction_2d(tau_p_xx::AbstractArray,
                                                tau_p_yy::AbstractArray;
                                                i_step::Int, j_low::Int, j_high::Int)
    j_c = (j_low + j_high) ÷ 2
    Nx = size(tau_p_xx, 1)
    return [tau_p_xx[i, j_c] - tau_p_yy[i, j_c] for i in i_step:Nx]
end
