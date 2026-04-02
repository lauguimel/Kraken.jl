# --- Spinodal decomposition (Shan-Chen multiphase) ---

"""
    run_spinodal_2d(; N=128, ν=0.1, G=-5.5, ρ0=1.0, max_steps=5000, backend, T)

Spinodal decomposition: a random density field phase-separates into two phases.
G controls attraction strength (G < -4 for phase separation in D2Q9).
Fully periodic domain.
"""
function run_spinodal_2d(; N=128, ν=0.1, G=-5.5, ρ0=1.0, max_steps=5000,
                          backend=KernelAbstractions.CPU(), FT=Float64)
    Nx, Ny = N, N
    config = LBMConfig(D2Q9(); Nx=Nx, Ny=Ny, ν=ν, u_lid=0.0, max_steps=max_steps)
    ω = FT(omega(config))

    # Initialize with random density perturbation
    f_in  = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    f_out = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    ρ     = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    ux    = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    uy    = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    ψ     = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Fx_sc = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Fy_sc = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    is_solid = KernelAbstractions.zeros(backend, Bool, Nx, Ny)

    w = weights(D2Q9())
    f_cpu = zeros(FT, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx
        ρ_init = FT(ρ0 + 0.01 * (rand() - 0.5))
        for q in 1:9
            f_cpu[i, j, q] = FT(w[q]) * ρ_init
        end
    end
    copyto!(f_in, f_cpu)
    copyto!(f_out, f_cpu)

    for step in 1:max_steps
        # 1. Stream (fully periodic)
        stream_fully_periodic_2d!(f_out, f_in, Nx, Ny)

        # 2. Macroscopic (standard — SC force correction in collision)
        compute_macroscopic_2d!(ρ, ux, uy, f_out)

        # 3. Shan-Chen force
        compute_psi_2d!(ψ, ρ, FT(ρ0))
        compute_sc_force_2d!(Fx_sc, Fy_sc, ψ, FT(G), Nx, Ny)

        # 4. Collide with SC force
        collide_sc_2d!(f_out, Fx_sc, Fy_sc, is_solid, ω)

        # 5. Swap
        f_in, f_out = f_out, f_in
    end

    compute_macroscopic_2d!(ρ, ux, uy, f_in)
    return (ρ=Array(ρ), ux=Array(ux), uy=Array(uy), config=config, G=G)
end

# --- GPU benchmark utility ---

"""
    benchmark_mlups(; Ns=[64, 128, 256], steps=100, backend=CPU())

Measure Million Lattice Updates Per Second for 2D LBM.
Returns a vector of (N, mlups) tuples.
"""
function benchmark_mlups(; Ns=[64, 128, 256], steps=100,
                          backend=KernelAbstractions.CPU(), FT=Float64)
    results = Tuple{Int, Float64}[]

    for N in Ns
        config = LBMConfig(D2Q9(); Nx=N, Ny=N, ν=0.1, u_lid=0.05, max_steps=steps)
        state = initialize_2d(config, FT; backend=backend)
        f_in, f_out = state.f_in, state.f_out
        is_solid = state.is_solid
        ω = FT(omega(config))

        # Warmup
        for _ in 1:5
            stream_2d!(f_out, f_in, N, N)
            collide_2d!(f_out, is_solid, ω)
            f_in, f_out = f_out, f_in
        end

        # Timed run
        t_start = time_ns()
        for _ in 1:steps
            stream_2d!(f_out, f_in, N, N)
            collide_2d!(f_out, is_solid, ω)
            f_in, f_out = f_out, f_in
        end
        t_elapsed = (time_ns() - t_start) / 1e9  # seconds

        total_updates = N * N * steps
        mlups = total_updates / t_elapsed / 1e6
        push!(results, (N, mlups))
    end

    return results
end

# --- Static droplet (VOF-LBM validation) ---

"""
    run_static_droplet_2d(; N=128, R=20, σ=0.01, ν=0.1, ρ_l=1.0, ρ_g=0.001,
                           max_steps=5000, backend, T)

Static circular droplet in a periodic box. Validates Laplace pressure:
Δp = σ/R. Measures spurious currents.
"""
function run_static_droplet_2d(; N=128, R=20, σ=0.01, ν=0.1,
                                ρ_l=1.0, ρ_g=0.001, max_steps=5000,
                                backend=KernelAbstractions.CPU(), FT=Float64)
    Nx, Ny = N, N
    cx, cy = N ÷ 2, N ÷ 2

    config = LBMConfig(D2Q9(); Nx=Nx, Ny=Ny, ν=ν, u_lid=0.0, max_steps=max_steps)
    state = initialize_2d(config, FT; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    ρ, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid

    # VOF arrays
    C     = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    C_new = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    nx_n  = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    ny_n  = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    κ     = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Fx_st = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Fy_st = KernelAbstractions.zeros(backend, FT, Nx, Ny)

    # Initialize circular droplet (smooth tanh interface, width ~2 cells)
    C_cpu = zeros(FT, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        r = sqrt(FT((i - cx)^2 + (j - cy)^2))
        C_cpu[i, j] = FT(0.5) * (one(FT) - tanh((r - FT(R)) / FT(2)))
    end
    copyto!(C, C_cpu)

    # Initialize f to equilibrium with density from C
    w = weights(D2Q9())
    f_cpu = zeros(FT, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx
        ρ_init = C_cpu[i,j] * FT(ρ_l) + (1 - C_cpu[i,j]) * FT(ρ_g)
        for q in 1:9
            f_cpu[i, j, q] = FT(w[q]) * ρ_init
        end
    end
    copyto!(f_in, f_cpu)
    copyto!(f_out, f_cpu)

    for step in 1:max_steps
        # 1. Stream (fully periodic)
        stream_fully_periodic_2d!(f_out, f_in, Nx, Ny)

        # 2. Macroscopic
        compute_macroscopic_2d!(ρ, ux, uy, f_out)

        # 3. VOF advection + clamp
        advect_vof_step!(C, C_new, ux, uy, Nx, Ny)
        copyto!(C, C_new)

        # 4. Interface normal + curvature + surface tension
        compute_vof_normal_2d!(nx_n, ny_n, C, Nx, Ny)
        compute_hf_curvature_2d!(κ, C, nx_n, ny_n, Nx, Ny)
        compute_surface_tension_2d!(Fx_st, Fy_st, κ, C, σ, Nx, Ny)

        # 6. Two-phase collision
        collide_twophase_2d!(f_out, C, Fx_st, Fy_st, is_solid;
                             ρ_l=ρ_l, ρ_g=ρ_g, ν_l=ν, ν_g=ν)

        f_in, f_out = f_out, f_in
    end

    compute_macroscopic_2d!(ρ, ux, uy, f_in)

    ρ_cpu = Array(ρ)
    ux_cpu = Array(ux)
    uy_cpu = Array(uy)
    C_cpu = Array(C)

    # Measure spurious currents
    max_u = sqrt(maximum(ux_cpu .^ 2 .+ uy_cpu .^ 2))

    # Laplace pressure: Δp = p_inside - p_outside = σ/R
    # p = ρ·cs² = ρ/3
    inside_mask = [(i-cx)^2 + (j-cy)^2 < (R-3)^2 for i in 1:Nx, j in 1:Ny]
    outside_mask = [(i-cx)^2 + (j-cy)^2 > (R+3)^2 for i in 1:Nx, j in 1:Ny]
    p_in = sum(ρ_cpu[inside_mask]) / sum(inside_mask) / 3
    p_out = sum(ρ_cpu[outside_mask]) / sum(outside_mask) / 3
    Δp = p_in - p_out
    Δp_analytical = σ / R

    return (ρ=ρ_cpu, ux=ux_cpu, uy=uy_cpu, C=C_cpu,
            max_u_spurious=max_u, Δp=Δp, Δp_analytical=Δp_analytical,
            config=config, σ=σ, R=R)
end

# --- Rayleigh-Plateau pinch-off (2D liquid bridge) ---

"""
    run_plateau_pinch_2d(; Nx=256, Ny=64, R0=20, λ_ratio=4.5, ε=0.05,
                          σ=0.01, ν=0.05, ρ_l=1.0, ρ_g=0.01,
                          max_steps=10000, output_interval=500, backend, T)

Rayleigh-Plateau capillary instability: a 2D liquid bridge with sinusoidal
perturbation pinches off due to surface tension.

Setup (cf. Popinet 2009, Gerris plateau example):
- Liquid slab of half-width R0 centered at y = Ny/2
- Perturbation: R(x) = R0 * (1 - ε·cos(2π·x/λ)) where λ = λ_ratio·R0
- Periodic in x, walls in y (or periodic)
- Surface tension σ drives the instability
- λ > 2πR (Rayleigh criterion) → unstable → pinch-off

Returns snapshots of C and r_min (minimum bridge radius) vs time.
"""
function run_plateau_pinch_2d(; Nx=256, Ny=64, R0=20, λ_ratio=4.5, ε=0.05,
                               σ=0.01, ν=0.05, ρ_l=1.0, ρ_g=0.01,
                               max_steps=10000, output_interval=500,
                               backend=KernelAbstractions.CPU(), FT=Float64)
    config = LBMConfig(D2Q9(); Nx=Nx, Ny=Ny, ν=ν, u_lid=0.0, max_steps=max_steps)
    state = initialize_2d(config, FT; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    ρ, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid

    # VOF arrays
    C     = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    C_new = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    nx_n  = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    ny_n  = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    κ     = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Fx_st = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Fy_st = KernelAbstractions.zeros(backend, FT, Nx, Ny)

    # Wavelength
    λ = FT(λ_ratio * R0)
    cy = FT(Ny / 2)

    # Initialize: liquid slab with sinusoidal perturbation
    C_cpu = zeros(FT, Nx, Ny)
    w = weights(D2Q9())
    f_cpu = zeros(FT, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx
        # Perturbed bridge radius
        R_local = FT(R0) * (one(FT) - FT(ε) * cos(FT(2π) * FT(i - 1) / λ))
        # Distance from centerline
        dist = abs(FT(j) - cy)
        # Smooth interface (tanh)
        C_cpu[i, j] = FT(0.5) * (one(FT) - tanh((dist - R_local) / FT(2)))
        ρ_init = C_cpu[i,j] * FT(ρ_l) + (one(FT) - C_cpu[i,j]) * FT(ρ_g)
        for q in 1:9
            f_cpu[i, j, q] = FT(w[q]) * ρ_init
        end
    end
    copyto!(C, C_cpu)
    copyto!(f_in, f_cpu)
    copyto!(f_out, f_cpu)

    # Track minimum bridge radius over time
    r_min_history = FT[]
    times = Int[]

    ω_flow = FT(omega(config))

    for step in 1:max_steps
        # 1. Stream (fully periodic in x, periodic in y for simplicity)
        stream_fully_periodic_2d!(f_out, f_in, Nx, Ny)

        # 2. Macroscopic
        compute_macroscopic_2d!(ρ, ux, uy, f_out)

        # 3. VOF advection + clamp
        advect_vof_step!(C, C_new, ux, uy, Nx, Ny)
        copyto!(C, C_new)

        # 4. Interface + curvature + surface tension
        compute_vof_normal_2d!(nx_n, ny_n, C, Nx, Ny)
        compute_hf_curvature_2d!(κ, C, nx_n, ny_n, Nx, Ny)
        compute_surface_tension_2d!(Fx_st, Fy_st, κ, C, σ, Nx, Ny)

        # 5. Two-phase collision
        collide_twophase_2d!(f_out, C, Fx_st, Fy_st, is_solid;
                             ρ_l=ρ_l, ρ_g=ρ_g, ν_l=ν, ν_g=ν)

        f_in, f_out = f_out, f_in

        # Track r_min: minimum "bridge width" (sum of C along y at each x)
        if step % output_interval == 0 || step == 1
            C_cpu = Array(C)
            # For each x column, compute the half-width of the liquid bridge
            widths = FT[]
            for i in 1:Nx
                col_C = C_cpu[i, :]
                bridge_width = sum(col_C) / 2  # half the total liquid height
                push!(widths, bridge_width)
            end
            r_min_val = minimum(widths)
            push!(r_min_history, r_min_val)
            push!(times, step)
        end
    end

    compute_macroscopic_2d!(ρ, ux, uy, f_in)

    return (ρ=Array(ρ), ux=Array(ux), uy=Array(uy), C=Array(C),
            r_min=r_min_history, times=times, config=config,
            σ=σ, R0=R0, λ=λ, ε=ε)
end

# --- Dual-grid static droplet (fine VOF + coarse LBM) ---

"""
    run_static_droplet_dualgrid_2d(; N=64, R=15, σ=0.01, ν=0.1, refine=2, ...)

Static droplet with dual-grid VOF-LBM: VOF advection, curvature and surface
tension computed on a fine grid (refine·N)², LBM solved on the coarse grid N².

The fine grid resolves the interface with `refine` times more cells, giving
sharper curvature and reduced spurious currents without the cost of a full
fine-grid LBM solve.
"""
function run_static_droplet_dualgrid_2d(; N=64, R=15, σ=0.01, ν=0.1,
                                          ρ_l=1.0, ρ_g=0.001, max_steps=5000,
                                          refine=2,
                                          backend=KernelAbstractions.CPU(), FT=Float64)
    r = refine
    Nx_c, Ny_c = N, N
    Nx_f, Ny_f = r * N, r * N
    dx_f = one(FT) / FT(r)
    cx_c = FT(N ÷ 2)
    cy_c = FT(N ÷ 2)

    config = LBMConfig(D2Q9(); Nx=Nx_c, Ny=Ny_c, ν=ν, u_lid=0.0, max_steps=max_steps)
    state = initialize_2d(config, FT; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    ρ, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid

    # Coarse arrays (collision interface)
    C_coarse  = KernelAbstractions.zeros(backend, FT, Nx_c, Ny_c)
    Fx_coarse = KernelAbstractions.zeros(backend, FT, Nx_c, Ny_c)
    Fy_coarse = KernelAbstractions.zeros(backend, FT, Nx_c, Ny_c)

    # Fine arrays (VOF + geometry)
    C_fine     = KernelAbstractions.zeros(backend, FT, Nx_f, Ny_f)
    C_fine_new = KernelAbstractions.zeros(backend, FT, Nx_f, Ny_f)
    nx_fine    = KernelAbstractions.zeros(backend, FT, Nx_f, Ny_f)
    ny_fine    = KernelAbstractions.zeros(backend, FT, Nx_f, Ny_f)
    κ_fine     = KernelAbstractions.zeros(backend, FT, Nx_f, Ny_f)
    Fx_fine    = KernelAbstractions.zeros(backend, FT, Nx_f, Ny_f)
    Fy_fine    = KernelAbstractions.zeros(backend, FT, Nx_f, Ny_f)
    ux_fine    = KernelAbstractions.zeros(backend, FT, Nx_f, Ny_f)
    uy_fine    = KernelAbstractions.zeros(backend, FT, Nx_f, Ny_f)

    # Initialize C on fine grid (interface width scales with dx_f for sharp resolution)
    W_interface = FT(2) * dx_f   # 2 fine cells wide (sharp on fine grid)
    C_cpu = zeros(FT, Nx_f, Ny_f)
    for jf in 1:Ny_f, i_f in 1:Nx_f
        x = (FT(i_f) - FT(0.5)) * dx_f
        y = (FT(jf) - FT(0.5)) * dx_f
        dist = sqrt((x - cx_c)^2 + (y - cy_c)^2)
        C_cpu[i_f, jf] = FT(0.5) * (one(FT) - tanh((dist - FT(R)) / W_interface))
    end
    copyto!(C_fine, C_cpu)

    # Restrict to coarse for f initialization
    restrict_average_2d!(C_coarse, C_fine, r)

    w = weights(D2Q9())
    f_cpu = zeros(FT, Nx_c, Ny_c, 9)
    C_c_cpu = Array(C_coarse)
    for jc in 1:Ny_c, ic in 1:Nx_c
        ρ_init = C_c_cpu[ic, jc] * FT(ρ_l) + (one(FT) - C_c_cpu[ic, jc]) * FT(ρ_g)
        for q in 1:9
            f_cpu[ic, jc, q] = FT(w[q]) * ρ_init
        end
    end
    copyto!(f_in, f_cpu)
    copyto!(f_out, f_cpu)

    for step in 1:max_steps
        # 1. Stream (coarse, fully periodic)
        stream_fully_periodic_2d!(f_out, f_in, Nx_c, Ny_c)

        # 2. Macroscopic (coarse)
        compute_macroscopic_2d!(ρ, ux, uy, f_out)

        # 3. Prolongate velocity to fine grid (×r for CFL: dt/dx_f = r)
        prolongate_bilinear_2d!(ux_fine, ux, r; scale=FT(r))
        prolongate_bilinear_2d!(uy_fine, uy, r; scale=FT(r))

        # 4. VOF advection + clamp (fine grid)
        advect_vof_step!(C_fine, C_fine_new, ux_fine, uy_fine, Nx_f, Ny_f)
        copyto!(C_fine, C_fine_new)

        # 6. Interface geometry (fine grid, physical dx)
        compute_vof_normal_2d!(nx_fine, ny_fine, C_fine, Nx_f, Ny_f)
        compute_hf_curvature_dx_2d!(κ_fine, C_fine, nx_fine, ny_fine, Nx_f, Ny_f, dx_f; hw=2*r)
        compute_surface_tension_dx_2d!(Fx_fine, Fy_fine, κ_fine, C_fine, σ, Nx_f, Ny_f, dx_f)

        # 7. Restrict to coarse (block average preserves force integral)
        restrict_average_2d!(C_coarse, C_fine, r)
        restrict_average_2d!(Fx_coarse, Fx_fine, r)
        restrict_average_2d!(Fy_coarse, Fy_fine, r)

        # 8. Two-phase collision (coarse)
        collide_twophase_2d!(f_out, C_coarse, Fx_coarse, Fy_coarse, is_solid;
                             ρ_l=ρ_l, ρ_g=ρ_g, ν_l=ν, ν_g=ν)

        f_in, f_out = f_out, f_in
    end

    compute_macroscopic_2d!(ρ, ux, uy, f_in)

    ρ_cpu = Array(ρ)
    ux_cpu = Array(ux)
    uy_cpu = Array(uy)
    C_f_cpu = Array(C_fine)
    C_c_cpu = Array(C_coarse)

    # Laplace pressure diagnostic
    cx_i, cy_i = N ÷ 2, N ÷ 2
    max_u = sqrt(maximum(ux_cpu .^ 2 .+ uy_cpu .^ 2))
    inside_mask = [(i - cx_i)^2 + (j - cy_i)^2 < (R - 3)^2 for i in 1:Nx_c, j in 1:Ny_c]
    outside_mask = [(i - cx_i)^2 + (j - cy_i)^2 > (R + 3)^2 for i in 1:Nx_c, j in 1:Ny_c]
    p_in = sum(ρ_cpu[inside_mask]) / sum(inside_mask) / 3
    p_out = sum(ρ_cpu[outside_mask]) / sum(outside_mask) / 3
    Δp = p_in - p_out
    Δp_analytical = σ / R

    return (ρ=ρ_cpu, ux=ux_cpu, uy=uy_cpu, C_fine=C_f_cpu, C_coarse=C_c_cpu,
            max_u_spurious=max_u, Δp=Δp, Δp_analytical=Δp_analytical,
            config=config, σ=σ, R=R, refine=r)
end

# ===========================================================================
# Prescribed-velocity VOF advection (pure interface transport tests)
# ===========================================================================

"""
    run_advection_2d(; Nx=100, Ny=100, max_steps=628, dt=1.0,
                      velocity_fn, init_C_fn,
                      output_interval=0, output_dir="", backend, T)

Run pure VOF advection with a prescribed analytical velocity field.
No LBM solve — used for interface transport validation (Zalesak, reversed
vortex, shear tests).

# Arguments
- `velocity_fn(x, y, t) -> (vx, vy)`: analytical velocity at position and time
- `init_C_fn(x, y) -> C₀`: initial volume fraction (will be clamped to [0,1])
- `dt`: time step for velocity evaluation (lattice dt=1 by default)

# Returns
NamedTuple `(C, C0, ux, uy, mass_history, Nx, Ny, dx)`
"""
function run_advection_2d(; Nx=100, Ny=100, max_steps=628, dt=1.0,
                           velocity_fn,
                           init_C_fn,
                           output_interval=0,
                           output_dir="",
                           backend=KernelAbstractions.CPU(),
                           FT=Float64)
    dx = FT(1)

    # Allocate arrays
    C     = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    C_new = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    nx_n  = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    ny_n  = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    ux    = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    uy    = KernelAbstractions.zeros(backend, FT, Nx, Ny)

    # Initialize C
    init_vof_field!(C, init_C_fn, dx, FT)
    C0 = Array(C)  # save initial state

    # Output setup
    pvd = nothing
    do_output = !isempty(output_dir)
    if do_output
        output_dir = setup_output_dir(output_dir)
        pvd = create_pvd(joinpath(output_dir, "advection"))
    end

    mass_history = FT[sum(C0)]

    # Check if velocity is time-dependent
    is_time_dep = try
        velocity_fn(FT(0.5) * dx, FT(0.5) * dx, FT(0))
        velocity_fn(FT(0.5) * dx, FT(0.5) * dx, FT(1))
        true
    catch
        false
    end

    # Fill initial velocity
    fill_velocity_field!(ux, uy, velocity_fn, dx, FT(0), backend, FT)

    # Time loop
    for step in 1:max_steps
        t = FT(step - 1) * dt

        # Update velocity if time-dependent
        if is_time_dep && step > 1
            fill_velocity_field!(ux, uy, velocity_fn, dx, t, backend, FT)
        end

        # Advect VOF (MUSCL-Superbee TVD, 2nd order)
        advect_vof_step!(C, C_new, ux, uy, Nx, Ny)
        copyto!(C, C_new)

        # Track mass
        push!(mass_history, FT(sum(Array(C))))

        # Output
        if do_output && step % output_interval == 0
            fields = Dict("C" => Array(C), "ux" => Array(ux), "uy" => Array(uy))
            write_snapshot_2d!(output_dir, step, Nx, Ny, Float64(dx), fields;
                               pvd=pvd, time=Float64(t))
        end
    end

    if do_output && pvd !== nothing
        vtk_save(pvd)
    end

    return (C=Array(C), C0=C0, ux=Array(ux), uy=Array(uy),
            mass_history=mass_history, Nx=Nx, Ny=Ny, dx=dx)
end
