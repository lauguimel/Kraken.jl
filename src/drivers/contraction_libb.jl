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
    run_conformation_step_libb_2d(; geometry, u_ref_mean=0.02,
                                    ν_s=0.354, ν_p=nothing, lambda=10.0,
                                    polymer_model=nothing,
                                    polymer_bc=CNEBB(),
                                    bcspec=nothing,
                                    ρ_out=1.0, tau_plus=1.0,
                                    hermite_source_mode=:liu_direct,
                                    conformation_magic=1e-6,
                                    conformation_divergence_mode=:trace_free,
                                    max_steps=200_000, avg_window=20_000,
                                    backend=CPU(), FT=Float64)

Generic axis-aligned step/channel driver for an Oldroyd-B,
log-conformation Oldroyd-B (or any `AbstractPolymerModel`) fluid.
Concrete cases are supplied as `StepChannelGeometry2D` specs, e.g.
`contraction_step_geometry_2d(...)` or `backward_facing_step_geometry_2d(...)`.

# Geometry
- `geometry` : a `StepChannelGeometry2D` containing solid mask, `q_wall`,
  open inlet/outlet ranges, and masked hydrodynamic/conformation BC faces.

# Flow
- `u_ref_mean` : reference mean velocity for `Re` and `Wi`.
- The west inlet parabolic profile is scaled by mass conservation:
  `u_in_mean = u_ref_mean * geometry.H_out / geometry.H_in`.
- `Re = u_ref_mean · geometry.H_ref / (ν_s + ν_p_eff)`.
- `Wi = λ · u_ref_mean / (geometry.H_ref / 2)`.

# Polymer
- `polymer_model` : explicit `AbstractPolymerModel` (preferred). If
  `nothing`, builds an `OldroydB(G=ν_p/λ, λ=lambda)` from `ν_p`+`lambda`.
- `polymer_bc`    : wall BC on conformation populations (default `CNEBB()`).
- `hermite_source_mode` : `:liu_direct` or `:ce_corrected`, matching the
  validated cylinder driver. This is intentionally exposed for wall/step
  isolation runs.
- `conformation_magic` : TRT magic parameter Λₚ for conformation/log-conf.
- `conformation_divergence_mode` : velocity-gradient trace handling for the
  conformation source. Defaults to `:trace_free`, matching the validated
  cylinder driver and the analytic Poiseuille CDE ladder.

# Returns
NamedTuple with `ux, uy, ρ, C_xx, C_xy, C_yy, tau_p_*, is_solid,
                 geometry, i_step, j_low, j_high, Re, Wi, beta, u_ref`.
"""
function run_conformation_step_libb_2d(;
        geometry::StepChannelGeometry2D,
        u_ref_mean=0.02, ν_s=0.354, ν_p=nothing, lambda=10.0,
        polymer_model::Union{Nothing,AbstractPolymerModel}=nothing,
        polymer_bc::AbstractPolymerWallBC=CNEBB(),
        bcspec::Union{Nothing,BCSpec2D}=nothing,
        ρ_out=1.0, tau_plus=1.0,
        hermite_source_mode::Symbol=:liu_direct,
        conformation_magic::Real=1e-6,
        conformation_divergence_mode::Symbol=:trace_free,
        allow_diagnostic_polymer_bc::Bool=false,
        allow_diagnostic_conformation_collision::Bool=false,
        allow_diagnostic_log_wall_bc::Bool=false,
        max_steps=200_000, avg_window=20_000,
        backend=KernelAbstractions.CPU(), FT=Float64)
    hermite_source_mode in (:ce_corrected, :liu_direct) ||
        error("unknown hermite_source_mode $(hermite_source_mode); expected :ce_corrected or :liu_direct")
    conformation_divergence_mode in (:numerical, :trace_free) ||
        error("unknown conformation_divergence_mode $(conformation_divergence_mode); expected :numerical or :trace_free")
    _assert_validation_polymer_wall_bc(polymer_bc;
                                       allow_diagnostic=allow_diagnostic_polymer_bc)
    _assert_validation_conformation_collision_window(:trt, tau_plus;
        allow_diagnostic=allow_diagnostic_conformation_collision)

    # Resolve polymer model
    if polymer_model === nothing
        isnothing(ν_p) && error("supply either `polymer_model` or (`ν_p`, `lambda`).")
        G_ = FT(ν_p / lambda)
        polymer_model = OldroydB(G=G_, λ=FT(lambda))
    end
    _assert_validation_log_wall_bc(polymer_model, polymer_bc;
        allow_diagnostic=allow_diagnostic_log_wall_bc)
    λ_p     = polymer_relaxation_time(polymer_model)
    ν_p_eff = polymer_modulus(polymer_model) * λ_p   # G·λ for Oldroyd-B / FENE

    geom_h = geometry
    Nx     = geom_h.Nx
    Ny     = geom_h.Ny
    i_step = geom_h.i_step
    j_low  = first(geom_h.outlet_open)
    j_high = last(geom_h.outlet_open)

    # Mass-conservation inlet mean from the requested reference mean velocity.
    u_in_mean = FT(u_ref_mean) * FT(geom_h.H_out) / FT(geom_h.H_in)

    ν_total = ν_s + ν_p_eff
    beta    = ν_s / ν_total
    Re      = Float64(u_ref_mean) * geom_h.H_ref / ν_total
    Wi      = λ_p * Float64(u_ref_mean) / (geom_h.H_ref / 2)

    s_plus_s = 1.0 / (3.0 * ν_s + 0.5)

    @info "Conformation step-channel (LI-BB V2)" geometry=geom_h.name Nx Ny H_ref=geom_h.H_ref H_in=geom_h.H_in H_out=geom_h.H_out i_step j_low j_high Re Wi beta λ_p tau_plus hermite_source_mode conformation_magic conformation_divergence_mode polymer_bc=typeof(polymer_bc) polymer_model=typeof(polymer_model) u_ref_mean u_in_mean

    # Geometry/spec: host object for initialization, device object for kernels.
    geom = transfer_step_geometry_2d(geom_h, backend)

    # Inlet u-profile: parabolic on the geometry-defined west opening.
    u_prof_in_h = parabolic_face_profile_2d(geom_h; face=:west,
                                            mean_velocity=u_in_mean, FT=FT)

    # Inlet conformation profile (analytical Oldroyd-B, Liu Eq 62
    # generalised for the upstream parabolic shear field)
    use_logconf = uses_log_conformation(polymer_model)
    C_xx_inlet_h, C_xy_inlet_h, C_yy_inlet_h =
        oldroydb_inlet_conformation_profile_2d(geom_h; face=:west,
            mean_velocity=u_in_mean, λ=λ_p, log_formulation=use_logconf, FT=FT)

    # Device allocations
    q_wall   = geom.q_wall
    is_solid = geom.is_solid
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
    fill!(uw_x, zero(FT));     fill!(uw_y, zero(FT))
    fill!(ρ, one(FT));         fill!(ux, zero(FT)); fill!(uy, zero(FT))
    copyto!(u_profile, u_prof_in_h)
    copyto!(C_xx_inlet, C_xx_inlet_h)
    copyto!(C_xy_inlet, C_xy_inlet_h)
    copyto!(C_yy_inlet, C_yy_inlet_h)

    # Default BCSpec: ZouHe velocity inlet (parabolic), ZouHe pressure outlet
    if bcspec === nothing
        bcspec = default_step_bcspec_2d(geom, u_profile, FT(ρ_out))
    end

    # Initialize f to equilibrium at the inlet velocity profile (extends
    # the upstream parabolic shear to all i — outlet cells then relax to
    # their own parabolic profile during the warm-up).
    f_in_h = zeros(FT, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx, q in 1:9
        u0 = geom_h.is_solid[i, j] ? zero(FT) : u_prof_in_h[j]
        f_in_h[i, j, q] = Kraken.equilibrium(D2Q9(), one(FT), u0, zero(FT), q)
    end
    copyto!(f_in, f_in_h); fill!(f_out, zero(FT))

    # Conformation fields (Ψ aliases C for direct stress; separate for log-conf)
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
                                   tau_p_xx, tau_p_xy, tau_p_yy;
                                   ce_correction = hermite_source_mode === :ce_corrected)

        # --- Conformation / log-conformation LBM (TRT) + polymer wall BC ---
        stream_2d!(g_xx_buf, g_xx, Nx, Ny)
        stream_2d!(g_xy_buf, g_xy, Nx, Ny)
        stream_2d!(g_yy_buf, g_yy, Nx, Ny)

        apply_polymer_wall_bc!(g_xx_buf, g_xx, is_solid, q_wall, Ψ_xx, ux, uy, polymer_bc)
        apply_polymer_wall_bc!(g_xy_buf, g_xy, is_solid, q_wall, Ψ_xy, ux, uy, polymer_bc)
        apply_polymer_wall_bc!(g_yy_buf, g_yy, is_solid, q_wall, Ψ_yy, ux, uy, polymer_bc)

        # Inlet/outlet conformation fix (analytical at i=1, equilibrium-extrapol
        # at i=Nx). This is the same anti-corruption fix as in the cylinder
        # driver — without it stream_2d's bounce-back at domain edges pollutes C.
        reset_conformation_inlet_masked_2d!(g_xx_buf, C_xx_inlet, u_profile, geom.west_conformation_mask, Ny)
        reset_conformation_inlet_masked_2d!(g_xy_buf, C_xy_inlet, u_profile, geom.west_conformation_mask, Ny)
        reset_conformation_inlet_masked_2d!(g_yy_buf, C_yy_inlet, u_profile, geom.west_conformation_mask, Ny)
        reset_conformation_outlet_masked_2d!(g_xx_buf, Nx, Ny, geom.east_conformation_mask)
        reset_conformation_outlet_masked_2d!(g_xy_buf, Nx, Ny, geom.east_conformation_mask)
        reset_conformation_outlet_masked_2d!(g_yy_buf, Nx, Ny, geom.east_conformation_mask)

        g_xx, g_xx_buf = g_xx_buf, g_xx
        g_xy, g_xy_buf = g_xy_buf, g_xy
        g_yy, g_yy_buf = g_yy_buf, g_yy

        compute_conformation_macro_2d!(Ψ_xx, g_xx)
        compute_conformation_macro_2d!(Ψ_xy, g_xy)
        compute_conformation_macro_2d!(Ψ_yy, g_yy)

        if use_logconf
            collide_logconf_2d!(g_xx, Ψ_xx, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid, tau_plus, λ_p; magic=conformation_magic, component=1, divergence_mode=conformation_divergence_mode)
            collide_logconf_2d!(g_xy, Ψ_xy, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid, tau_plus, λ_p; magic=conformation_magic, component=2, divergence_mode=conformation_divergence_mode)
            collide_logconf_2d!(g_yy, Ψ_yy, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid, tau_plus, λ_p; magic=conformation_magic, component=3, divergence_mode=conformation_divergence_mode)
            psi_to_C_2d!(C_xx, C_xy, C_yy, Ψ_xx, Ψ_xy, Ψ_yy)
        else
            collide_conformation_2d!(g_xx, Ψ_xx, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid, tau_plus, λ_p; magic=conformation_magic, component=1, divergence_mode=conformation_divergence_mode)
            collide_conformation_2d!(g_xy, Ψ_xy, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid, tau_plus, λ_p; magic=conformation_magic, component=2, divergence_mode=conformation_divergence_mode)
            collide_conformation_2d!(g_yy, Ψ_yy, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid, tau_plus, λ_p; magic=conformation_magic, component=3, divergence_mode=conformation_divergence_mode)
        end

        update_polymer_stress!(tau_p_xx, tau_p_xy, tau_p_yy,
                                 C_xx, C_xy, C_yy, polymer_model)

        f_in, f_out = f_out, f_in
    end
    KernelAbstractions.synchronize(backend)

    @info "Conformation step-channel (LI-BB V2) result" geometry=geom_h.name Re Wi

    return (ux=Array(ux), uy=Array(uy), ρ=Array(ρ),
            C_xx=Array(C_xx), C_xy=Array(C_xy), C_yy=Array(C_yy),
            tau_p_xx=Array(tau_p_xx), tau_p_xy=Array(tau_p_xy), tau_p_yy=Array(tau_p_yy),
            q_wall=Array(q_wall), is_solid=Array(is_solid),
            west_hydro_mask=Array(geom.west_hydro_mask),
            east_hydro_mask=Array(geom.east_hydro_mask),
            west_conformation_mask=Array(geom.west_conformation_mask),
            east_conformation_mask=Array(geom.east_conformation_mask),
            geometry=geom_h.name,
            Nx=Nx, Ny=Ny, i_step=i_step, j_low=j_low, j_high=j_high,
            hermite_source_mode=hermite_source_mode,
            conformation_magic=Float64(conformation_magic),
            conformation_divergence_mode=conformation_divergence_mode,
            Re=Re, Wi=Wi, beta=beta, u_ref=Float64(u_ref_mean),
            H_ref=geom_h.H_ref, H_out=geom_h.H_out)
end

function run_conformation_contraction_libb_2d(;
        H_out::Int=20, β_c::Int=4, L_up::Int=20, L_down::Int=50,
        u_out_mean=0.02, ν_s=0.354, ν_p=nothing, lambda=10.0,
        polymer_model::Union{Nothing,AbstractPolymerModel}=nothing,
        polymer_bc::AbstractPolymerWallBC=CNEBB(),
        bcspec::Union{Nothing,BCSpec2D}=nothing,
        ρ_out=1.0, tau_plus=1.0,
        hermite_source_mode::Symbol=:liu_direct,
        conformation_magic::Real=1e-6,
        conformation_divergence_mode::Symbol=:trace_free,
        allow_diagnostic_polymer_bc::Bool=false,
        allow_diagnostic_conformation_collision::Bool=false,
        allow_diagnostic_log_wall_bc::Bool=false,
        max_steps=200_000, avg_window=20_000,
        backend=KernelAbstractions.CPU(), FT=Float64)
    geometry = contraction_step_geometry_2d(; H_out=H_out, β_c=β_c,
                                            L_up=L_up, L_down=L_down, FT=FT)
    return run_conformation_step_libb_2d(;
        geometry, u_ref_mean=u_out_mean, ν_s, ν_p, lambda,
        polymer_model, polymer_bc, bcspec, ρ_out, tau_plus,
        hermite_source_mode, conformation_magic, conformation_divergence_mode,
        allow_diagnostic_polymer_bc,
        allow_diagnostic_conformation_collision,
        allow_diagnostic_log_wall_bc,
        max_steps, avg_window,
        backend, FT)
end

function _sign_with_tolerance(value, tolerance)
    value > tolerance && return 1
    value < -tolerance && return -1
    return 0
end

"""
    vortex_length_contraction_2d(ux, uy, is_solid; i_step, j_low,
                                   side=:south, velocity_tolerance=nothing)

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
                                       side::Symbol=:south,
                                       velocity_tolerance=nothing)
    j_probe = side === :south ? j_low - 1 :
              side === :north ? j_high + 1 :
              error("side ∈ (:south, :north)")
    1 ≤ j_probe ≤ size(ux, 2) || error("j_probe $(j_probe) out of domain")

    max_abs_ux = maximum(abs, ux[.!is_solid])
    tolerance = isnothing(velocity_tolerance) ?
        sqrt(eps(float(eltype(ux)))) * max_abs_ux :
        Float64(velocity_tolerance)

    # The main flow is left-to-right. Near the re-entrant corner the sampled
    # velocity may be exactly zero or noise-dominated, especially in Float32
    # GPU runs. Establish the recirculation sign from the first significant
    # upstream sample, then return the first positive sample as reattachment.
    corner_sign = 0
    first_significant_distance = 0
    for distance in 0:(i_step - 2)
        i_probe = i_step - 1 - distance
        current_sign = _sign_with_tolerance(Float64(ux[i_probe, j_probe]), tolerance)
        if current_sign != 0
            corner_sign = current_sign
            first_significant_distance = distance
            break
        end
    end
    corner_sign == 0 && return (0.0, j_probe)
    corner_sign > 0 && return (0.0, j_probe)

    for distance in (first_significant_distance + 1):(i_step - 2)
        i_probe = i_step - 1 - distance
        current_sign = _sign_with_tolerance(Float64(ux[i_probe, j_probe]), tolerance)
        if current_sign > 0
            return (Float64(distance), j_probe)
        end
    end
    return (Float64(i_step - 2), j_probe)
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
