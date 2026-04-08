# --- Rayleigh-Bénard 2D (thermal convection) ---

"""
    run_rayleigh_benard_2d(; Nx=128, Ny=32, Ra=2000, Pr=1.0, T_hot=1.0, T_cold=0.0,
                            max_steps=20000, backend, T)

Rayleigh-Bénard convection: hot bottom wall, cold top wall, periodic x.
Ra = β·g·ΔT·H³/(ν·α), Pr = ν/α.
"""
function run_rayleigh_benard_2d(; Nx=128, Ny=32, Ra=2000.0, Pr=1.0,
                                 T_hot=1.0, T_cold=0.0, max_steps=20000,
                                 backend=KernelAbstractions.CPU(), FT=Float64)
    ΔT = T_hot - T_cold
    H = Ny  # channel height in lattice units (half-way BB)

    # Choose ν for stability (ω not too close to 2)
    ν = 0.05
    α = ν / Pr               # thermal diffusivity
    β_g = Ra * ν * α / (ΔT * H^3)  # β·g combined

    ω_f = FT(1.0 / (3.0 * ν + 0.5))     # flow relaxation
    ω_T = FT(1.0 / (3.0 * α + 0.5))     # thermal relaxation

    config = LBMConfig(D2Q9(); Nx=Nx, Ny=Ny, ν=ν, u_lid=0.0, max_steps=max_steps)
    state = initialize_2d(config, FT; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    ρ, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid

    # Thermal populations: initialize to linear temperature profile
    g_in  = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    g_out = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    Temp  = KernelAbstractions.zeros(backend, FT, Nx, Ny)

    w = weights(D2Q9())
    g_cpu = zeros(FT, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx
        # Linear profile from T_hot (j=1) to T_cold (j=Ny) + small perturbation
        T_init = FT(T_hot - ΔT * (j - 1) / (Ny - 1))
        T_init += FT(0.01 * ΔT) * sin(FT(2π * i / Nx)) * sin(FT(π * j / Ny))
        for q in 1:9
            g_cpu[i, j, q] = FT(w[q]) * T_init
        end
    end
    copyto!(g_in, g_cpu)
    copyto!(g_out, g_cpu)

    T_ref = FT((T_hot + T_cold) / 2)

    for step in 1:max_steps
        # 1. Stream flow (periodic x, wall y)
        stream_periodic_x_wall_y_2d!(f_out, f_in, Nx, Ny)

        # 2. Stream thermal (same kernel)
        stream_periodic_x_wall_y_2d!(g_out, g_in, Nx, Ny)

        # 3. Thermal BCs: fixed temperature at walls
        apply_fixed_temp_south_2d!(g_out, T_hot, Nx)
        apply_fixed_temp_north_2d!(g_out, T_cold, Nx, Ny)

        # 4. Compute temperature for Boussinesq coupling
        compute_temperature_2d!(Temp, g_out)

        # 5. Collide thermal (uses flow velocity)
        compute_macroscopic_2d!(ρ, ux, uy, f_out)
        collide_thermal_2d!(g_out, ux, uy, ω_T)

        # 6. Collide flow with per-node Boussinesq force: Fy(i,j) = β·g·(T(i,j) - T_ref)
        collide_boussinesq_2d!(f_out, Temp, is_solid, ω_f, β_g, T_ref)

        # 7. Swap
        f_in, f_out = f_out, f_in
        g_in, g_out = g_out, g_in
    end

    compute_macroscopic_2d!(ρ, ux, uy, f_in)
    compute_temperature_2d!(Temp, g_in)

    return (ρ=Array(ρ), ux=Array(ux), uy=Array(uy), Temp=Array(Temp),
            config=config, Ra=Ra, Pr=Pr, ν=ν, α=α)
end

# --- Natural convection in a differentially heated cavity ---

"""
    run_natural_convection_2d(; N=128, Ra=1e4, Pr=0.71, Rc=1.0,
                                T_hot=1.0, T_cold=0.0, max_steps=50000,
                                backend, FT)

Natural convection in a square cavity: hot left wall, cold right wall,
adiabatic top/bottom. Boussinesq approximation with optional
temperature-dependent viscosity via modified Arrhenius law.

Ra = β·g·ΔT·H³/(ν·α), Pr = ν/α, Rc = η_max/η_min (rheological contrast).
For Rc=1: constant viscosity. For Rc>1: ν(T) = ν_ref·exp(α_visc·(T - T_ref)).
"""
function run_natural_convection_2d(; N=128, Ra=1e4, Pr=0.71, Rc=1.0,
                                     T_hot=1.0, T_cold=0.0, max_steps=50000,
                                     backend=KernelAbstractions.CPU(), FT=Float64)
    Nx, Ny = N, N
    ΔT = T_hot - T_cold
    H = FT(N)

    # LBM parameters (NatConv convention: TRef=0, η=Pr, κ=1, β=Pr·Ra)
    ν = FT(0.05)
    α_thermal = ν / FT(Pr)
    β_g = FT(Ra) * ν * α_thermal / (FT(ΔT) * H^3)

    ω_f = FT(1.0 / (3.0 * ν + 0.5))
    ω_T = FT(1.0 / (3.0 * α_thermal + 0.5))

    # Variable viscosity: ν(T) = ν_ref · exp(α_visc · (T - T0_visc))
    # Rc = η_cold/η_hot: hot side is Rc times MORE FLUID
    # η_cold(T=0) = ν_ref, η_hot(T=1) = ν_ref/Rc → α = -ln(Rc)
    α_visc = FT(-log(Rc))
    T0_visc = FT(T_cold)
    # Buoyancy reference: F = β_g · (T - T_ref_buoy)
    T_ref_buoy = FT((T_hot + T_cold) / 2)

    config = LBMConfig(D2Q9(); Nx=Nx, Ny=Ny, ν=Float64(ν), u_lid=0.0, max_steps=max_steps)
    state = initialize_2d(config, FT; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    ρ, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid

    # Thermal populations: horizontal linear profile T(x)
    g_in  = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    g_out = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    Temp  = KernelAbstractions.zeros(backend, FT, Nx, Ny)

    w = weights(D2Q9())
    g_cpu = zeros(FT, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx
        T_init = FT(T_hot - ΔT * (i - 1) / (Nx - 1))
        T_init += FT(0.01 * ΔT) * sin(FT(2π * i / Nx)) * sin(FT(π * j / Ny))
        for q in 1:9
            g_cpu[i, j, q] = FT(w[q]) * T_init
        end
    end
    copyto!(g_in, g_cpu)
    copyto!(g_out, g_cpu)

    # Fused kernel: 1 GPU launch per step (stream + BC + macroscopic + collide)
    for step in 1:max_steps
        if Rc ≈ 1.0
            fused_natconv_step!(f_out, f_in, g_out, g_in, Temp, Nx, Ny,
                                 ω_f, ω_T, β_g, T_ref_buoy, FT(T_hot), FT(T_cold))
        else
            fused_natconv_vt_step!(f_out, f_in, g_out, g_in, Temp, Nx, Ny,
                                    ν, T0_visc, α_visc, ω_T, β_g, T_ref_buoy,
                                    FT(T_hot), FT(T_cold))
        end
        f_in, f_out = f_out, f_in
        g_in, g_out = g_out, g_in
    end

    compute_macroscopic_2d!(ρ, ux, uy, f_in)
    compute_temperature_2d!(Temp, g_in)

    # Compute Nusselt number at hot wall (second-order forward FD)
    T_cpu = Array(Temp)
    dx = FT(1.0)  # lattice spacing
    Nu_local = zeros(FT, Ny)
    for j in 2:Ny-1
        Nu_local[j] = -H * (-3*T_cpu[1,j] + 4*T_cpu[2,j] - T_cpu[3,j]) / (2*dx) / FT(ΔT)
    end
    Nu = sum(Nu_local[2:end-1]) / (Ny - 2)

    return (ρ=Array(ρ), ux=Array(ux), uy=Array(uy), Temp=T_cpu,
            Nu=Nu, config=config, Ra=Ra, Pr=Pr, Rc=Rc, ν=Float64(ν), α=Float64(α_thermal))
end

# --- Natural convection with wall-refined patches ---

"""
    run_natural_convection_refined_2d(; N=128, Ra=1e4, Pr=0.71, Rc=1.0,
                                        T_hot=1.0, T_cold=0.0, max_steps=50000,
                                        wall_fraction=0.2, ratio=2,
                                        backend, FT)

Natural convection in a square cavity with patch refinement near the hot (west)
and cold (east) walls. Patches cover `wall_fraction` of the domain width on
each side, refined by `ratio` (default 2:1).

Same physics as run_natural_convection_2d but with better boundary layer
resolution at lower total cost.
"""
function run_natural_convection_refined_2d(; N=128, Ra=1e4, Pr=0.71, Rc=1.0,
                                             T_hot=1.0, T_cold=0.0, max_steps=50000,
                                             wall_fraction=0.2, ratio=2,
                                             backend=KernelAbstractions.CPU(), FT=Float64)
    Nx, Ny = N, N
    ΔT = T_hot - T_cold
    H = FT(N)
    dx = FT(1.0)

    # LBM parameters
    ν = FT(0.05)
    α_thermal = ν / FT(Pr)
    β_g = FT(Ra) * ν * α_thermal / (FT(ΔT) * H^3)

    ω_f = FT(1.0 / (3.0 * ν + 0.5))
    ω_T = FT(1.0 / (3.0 * α_thermal + 0.5))

    α_visc = FT(-log(Rc))
    T0_visc = FT(T_cold)
    T_ref_buoy = FT((T_hot + T_cold) / 2)

    # --- Base grid ---
    config = LBMConfig(D2Q9(); Nx=Nx, Ny=Ny, ν=Float64(ν), u_lid=0.0, max_steps=max_steps)
    state = initialize_2d(config, FT; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    ρ, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid

    g_in  = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    g_out = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    Temp  = KernelAbstractions.zeros(backend, FT, Nx, Ny)

    w = weights(D2Q9())
    g_cpu = zeros(FT, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx
        T_init = FT(T_hot - ΔT * (i - 1) / (Nx - 1))
        T_init += FT(0.01 * ΔT) * sin(FT(2π * i / Nx)) * sin(FT(π * j / Ny))
        for q in 1:9
            g_cpu[i, j, q] = FT(w[q]) * T_init
        end
    end
    copyto!(g_in, g_cpu)
    copyto!(g_out, g_cpu)

    # --- Create refinement patches near walls ---
    L = FT(N * dx)
    wall_width = wall_fraction * L

    patch_west = create_patch("west_wall", 1, ratio,
        (0.0, 0.0, Float64(wall_width), Float64(L)),
        Nx, Ny, Float64(dx), Float64(ω_f), FT; backend=backend)
    patch_east = create_patch("east_wall", 1, ratio,
        (Float64(L - wall_width), 0.0, Float64(L), Float64(L)),
        Nx, Ny, Float64(dx), Float64(ω_f), FT; backend=backend)

    domain = create_refined_domain(Nx, Ny, Float64(dx), Float64(ω_f),
                                    [patch_west, patch_east])

    # Thermal arrays for patches
    thermal_west = create_thermal_patch_arrays(patch_west, Float64(ω_T);
                                                T_init=Float64((T_hot + T_cold) / 2),
                                                backend=backend)
    thermal_east = create_thermal_patch_arrays(patch_east, Float64(ω_T);
                                                T_init=Float64((T_hot + T_cold) / 2),
                                                backend=backend)
    thermals = [thermal_west, thermal_east]

    # Thermal BCs for patches at domain walls
    bc_thermal_patch_fns = Dict{Int, Function}(
        1 => (g, Nx_p, Ny_p) -> apply_fixed_temp_west_2d!(g, T_hot, Ny_p),
        2 => (g, Nx_p, Ny_p) -> apply_fixed_temp_east_2d!(g, T_cold, Nx_p, Ny_p),
    )

    # Flow no-slip BCs for patches: each patch spans the full vertical extent
    # so it touches south, north, AND its respective lateral wall.
    bc_flow_patch_fns = Dict{Int, Function}(
        1 => (f, Nx_p, Ny_p) -> begin
            apply_bounce_back_wall_2d!(f, Nx_p, Ny_p, :west)
            apply_bounce_back_wall_2d!(f, Nx_p, Ny_p, :south)
            apply_bounce_back_wall_2d!(f, Nx_p, Ny_p, :north)
        end,
        2 => (f, Nx_p, Ny_p) -> begin
            apply_bounce_back_wall_2d!(f, Nx_p, Ny_p, :east)
            apply_bounce_back_wall_2d!(f, Nx_p, Ny_p, :south)
            apply_bounce_back_wall_2d!(f, Nx_p, Ny_p, :north)
        end,
    )

    # Initialize patch interiors from coarse state (otherwise patches start
    # at uniform T=0.5 / rest and pollute the coarse grid via restriction).
    compute_macroscopic_2d!(ρ, ux, uy, f_in)
    compute_temperature_2d!(Temp, g_in)
    for (pidx, patch) in enumerate(domain.patches)
        Kraken.prolongate_f_rescaled_full_2d!(
            patch.f_in, f_in, ρ, ux, uy,
            patch.ratio, patch.Nx_inner, patch.Ny_inner,
            patch.n_ghost, first(patch.parent_i_range), first(patch.parent_j_range),
            Nx, Ny, Float64(ω_f), Float64(patch.omega))
        copyto!(patch.f_out, patch.f_in)
        compute_macroscopic_2d!(patch.rho, patch.ux, patch.uy, patch.f_in)
        fill_thermal_full!(patch, thermals[pidx], g_in, Nx, Ny)
    end

    # Fused step closure for base grid
    fused_step = if Rc ≈ 1.0
        (fo, fi, go, gi, Te, nx, ny) -> fused_natconv_step!(
            fo, fi, go, gi, Te, nx, ny,
            ω_f, ω_T, β_g, T_ref_buoy, FT(T_hot), FT(T_cold))
    else
        (fo, fi, go, gi, Te, nx, ny) -> fused_natconv_vt_step!(
            fo, fi, go, gi, Te, nx, ny,
            ν, T0_visc, α_visc, ω_T, β_g, T_ref_buoy,
            FT(T_hot), FT(T_cold))
    end

    # --- Time loop ---
    for step in 1:max_steps
        f_in, f_out, g_in, g_out = advance_thermal_refined_step!(
            domain, thermals,
            f_in, f_out, g_in, g_out, ρ, ux, uy, Temp, is_solid;
            fused_step_fn=fused_step,
            omega_T_coarse=Float64(ω_T),
            β_g=Float64(β_g),
            T_ref_buoy=Float64(T_ref_buoy),
            bc_thermal_patch_fns=bc_thermal_patch_fns,
            bc_flow_patch_fns=bc_flow_patch_fns
        )
    end

    compute_macroscopic_2d!(ρ, ux, uy, f_in)
    compute_temperature_2d!(Temp, g_in)

    # Nusselt at hot wall
    T_cpu = Array(Temp)
    Nu_local = zeros(FT, Ny)
    for j in 2:Ny-1
        Nu_local[j] = -H * (-3*T_cpu[1,j] + 4*T_cpu[2,j] - T_cpu[3,j]) / (2*dx) / FT(ΔT)
    end
    Nu = sum(Nu_local[2:end-1]) / (Ny - 2)

    return (ρ=Array(ρ), ux=Array(ux), uy=Array(uy), Temp=T_cpu,
            Nu=Nu, config=config, Ra=Ra, Pr=Pr, Rc=Rc, ν=Float64(ν), α=Float64(α_thermal),
            domain=domain, thermals=thermals)
end
