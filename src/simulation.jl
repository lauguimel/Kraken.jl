"""
    LBMConfig{L <: AbstractLattice}

Configuration for an LBM simulation.
"""
struct LBMConfig{L <: AbstractLattice}
    lattice::L
    Nx::Int
    Ny::Int
    Nz::Int          # 1 for 2D
    ν::Float64       # kinematic viscosity (lattice units)
    u_lid::Float64   # lid velocity (cavity)
    max_steps::Int
    output_interval::Int
end

function LBMConfig(::D2Q9; Nx, Ny, ν, u_lid=0.1, max_steps=10000, output_interval=1000)
    return LBMConfig(D2Q9(), Nx, Ny, 1, ν, u_lid, max_steps, output_interval)
end

function LBMConfig(::D3Q19; Nx, Ny, Nz, ν, u_lid=0.1, max_steps=10000, output_interval=1000)
    return LBMConfig(D3Q19(), Nx, Ny, Nz, ν, u_lid, max_steps, output_interval)
end

"""
    omega(config::LBMConfig) -> Float64

Compute BGK relaxation parameter from viscosity: ω = 1 / (3ν + 0.5).
"""
omega(config::LBMConfig) = 1.0 / (3.0 * config.ν + 0.5)

"""
    reynolds(config::LBMConfig) -> Float64

Effective Reynolds number: Re = u_lid · N / ν.
Uses Ny for 2D, Nz for 3D (cavity height).
"""
function reynolds(config::LBMConfig)
    L = lattice_dim(config.lattice) == 2 ? config.Ny : config.Nz
    return config.u_lid * L / config.ν
end

# --- Initialization ---

"""
    initialize_2d(config::LBMConfig{D2Q9}, T=Float64; backend=CPU())

Create initial LBM state for 2D simulation. Populations set to equilibrium
with ρ=1, u=0.
"""
function initialize_2d(config::LBMConfig{D2Q9}, ::Type{T}=Float64;
                        backend=KernelAbstractions.CPU()) where T
    Nx, Ny = config.Nx, config.Ny

    # Allocate on the desired backend
    f_in  = KernelAbstractions.zeros(backend, T, Nx, Ny, 9)
    f_out = KernelAbstractions.zeros(backend, T, Nx, Ny, 9)
    ρ     = KernelAbstractions.ones(backend, T, Nx, Ny)
    ux    = KernelAbstractions.zeros(backend, T, Nx, Ny)
    uy    = KernelAbstractions.zeros(backend, T, Nx, Ny)
    is_solid = KernelAbstractions.zeros(backend, Bool, Nx, Ny)

    # Initialize to equilibrium (ρ=1, u=0 → f_eq = w_q)
    w = weights(D2Q9())
    f_cpu = zeros(T, Nx, Ny, 9)
    for q in 1:9
        f_cpu[:, :, q] .= T(w[q])
    end
    copyto!(f_in, f_cpu)
    copyto!(f_out, f_cpu)

    return (f_in=f_in, f_out=f_out, ρ=ρ, ux=ux, uy=uy, is_solid=is_solid)
end

"""
    initialize_3d(config::LBMConfig{D3Q19}, T=Float64; backend=CPU())

Create initial LBM state for 3D simulation.
"""
function initialize_3d(config::LBMConfig{D3Q19}, ::Type{T}=Float64;
                        backend=KernelAbstractions.CPU()) where T
    Nx, Ny, Nz = config.Nx, config.Ny, config.Nz

    f_in  = KernelAbstractions.zeros(backend, T, Nx, Ny, Nz, 19)
    f_out = KernelAbstractions.zeros(backend, T, Nx, Ny, Nz, 19)
    ρ     = KernelAbstractions.ones(backend, T, Nx, Ny, Nz)
    ux    = KernelAbstractions.zeros(backend, T, Nx, Ny, Nz)
    uy    = KernelAbstractions.zeros(backend, T, Nx, Ny, Nz)
    uz    = KernelAbstractions.zeros(backend, T, Nx, Ny, Nz)
    is_solid = KernelAbstractions.zeros(backend, Bool, Nx, Ny, Nz)

    w = weights(D3Q19())
    f_cpu = zeros(T, Nx, Ny, Nz, 19)
    for q in 1:19
        f_cpu[:, :, :, q] .= T(w[q])
    end
    copyto!(f_in, f_cpu)
    copyto!(f_out, f_cpu)

    return (f_in=f_in, f_out=f_out, ρ=ρ, ux=ux, uy=uy, uz=uz, is_solid=is_solid)
end

# --- Main simulation loops ---

"""
    run_cavity_2d(config::LBMConfig{D2Q9}; backend=CPU(), T=Float64)

Run a 2D lid-driven cavity simulation using D2Q9 BGK-LBM.

Returns a NamedTuple with final fields (ρ, ux, uy) on CPU.
"""
function run_cavity_2d(config::LBMConfig{D2Q9};
                        backend=KernelAbstractions.CPU(), T=Float64)
    state = initialize_2d(config, T; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    ρ, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid
    Nx, Ny = config.Nx, config.Ny
    ω = T(omega(config))

    for step in 1:config.max_steps
        # 1. Stream (pull scheme, bounce-back at domain edges)
        stream_2d!(f_out, f_in, Nx, Ny)

        # 2. Apply Zou-He velocity BC on lid (post-stream, pre-collision)
        apply_zou_he_north_2d!(f_out, config.u_lid, Nx, Ny)

        # 3. BGK collision (in-place on f_out)
        collide_2d!(f_out, is_solid, ω)

        # 4. Compute macroscopic fields
        compute_macroscopic_2d!(ρ, ux, uy, f_out)

        # 5. Swap populations
        f_in, f_out = f_out, f_in
    end

    # Copy results to CPU
    ρ_cpu  = Array(ρ)
    ux_cpu = Array(ux)
    uy_cpu = Array(uy)

    return (ρ=ρ_cpu, ux=ux_cpu, uy=uy_cpu, config=config)
end

"""
    run_cavity_3d(config::LBMConfig{D3Q19}; backend=CPU(), T=Float64)

Run a 3D lid-driven cavity simulation using D3Q19 BGK-LBM.

Returns a NamedTuple with final fields (ρ, ux, uy, uz) on CPU.
"""
function run_cavity_3d(config::LBMConfig{D3Q19};
                        backend=KernelAbstractions.CPU(), T=Float64)
    state = initialize_3d(config, T; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    ρ, ux, uy, uz = state.ρ, state.ux, state.uy, state.uz
    is_solid = state.is_solid
    Nx, Ny, Nz = config.Nx, config.Ny, config.Nz
    ω = T(omega(config))

    for step in 1:config.max_steps
        stream_3d!(f_out, f_in, Nx, Ny, Nz)
        apply_zou_he_top_3d!(f_out, config.u_lid, Nx, Ny, Nz)
        collide_3d!(f_out, is_solid, ω)
        compute_macroscopic_3d!(ρ, ux, uy, uz, f_out)
        f_in, f_out = f_out, f_in
    end

    ρ_cpu  = Array(ρ)
    ux_cpu = Array(ux)
    uy_cpu = Array(uy)
    uz_cpu = Array(uz)

    return (ρ=ρ_cpu, ux=ux_cpu, uy=uy_cpu, uz=uz_cpu, config=config)
end

# --- Poiseuille 2D with body force ---

"""
    run_poiseuille_2d(; Nx=4, Ny=32, ν=0.1, Fx=1e-5, max_steps=10000, backend, T)

Channel flow driven by body force Fx. Periodic in x, walls at j=1 and j=Ny.
"""
function run_poiseuille_2d(; Nx=4, Ny=32, ν=0.1, Fx=1e-5, max_steps=10000,
                            backend=KernelAbstractions.CPU(), T=Float64)
    config = LBMConfig(D2Q9(); Nx=Nx, Ny=Ny, ν=ν, u_lid=0.0, max_steps=max_steps)
    state = initialize_2d(config, T; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    ρ, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid
    ω = T(omega(config))

    for step in 1:max_steps
        stream_periodic_x_wall_y_2d!(f_out, f_in, Nx, Ny)
        collide_guo_2d!(f_out, is_solid, ω, T(Fx), T(0))
        compute_macroscopic_forced_2d!(ρ, ux, uy, f_out, T(Fx), T(0))
        f_in, f_out = f_out, f_in
    end

    return (ρ=Array(ρ), ux=Array(ux), uy=Array(uy), config=config)
end

# --- Couette 2D ---

"""
    run_couette_2d(; Nx=4, Ny=32, ν=0.1, u_wall=0.05, max_steps=10000, backend, T)

Couette flow: bottom wall (j=1) moves at u_wall, top wall (j=Ny) stationary.
Periodic in x.
"""
function run_couette_2d(; Nx=4, Ny=32, ν=0.1, u_wall=0.05, max_steps=10000,
                         backend=KernelAbstractions.CPU(), T=Float64)
    config = LBMConfig(D2Q9(); Nx=Nx, Ny=Ny, ν=ν, u_lid=0.0, max_steps=max_steps)
    state = initialize_2d(config, T; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    ρ, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid
    ω = T(omega(config))

    for step in 1:max_steps
        stream_periodic_x_wall_y_2d!(f_out, f_in, Nx, Ny)
        apply_zou_he_south_2d!(f_out, u_wall, Nx)
        apply_zou_he_north_2d!(f_out, T(0), Nx, Ny)
        collide_2d!(f_out, is_solid, ω)
        compute_macroscopic_2d!(ρ, ux, uy, f_out)
        f_in, f_out = f_out, f_in
    end

    return (ρ=Array(ρ), ux=Array(ux), uy=Array(uy), config=config)
end

# --- Taylor-Green vortex 2D ---

"""
    initialize_taylor_green_2d(; N=64, ν=0.01, u0=0.01, backend, T)

Initialize populations to equilibrium with Taylor-Green velocity field.
"""
function initialize_taylor_green_2d(; N=64, ν=0.01, u0=0.01,
                                     backend=KernelAbstractions.CPU(), T=Float64)
    config = LBMConfig(D2Q9(); Nx=N, Ny=N, ν=ν, u_lid=0.0, max_steps=0)
    Nx, Ny = N, N
    k = T(2π / N)

    f_in  = KernelAbstractions.zeros(backend, T, Nx, Ny, 9)
    f_out = KernelAbstractions.zeros(backend, T, Nx, Ny, 9)
    ρ_arr = KernelAbstractions.ones(backend, T, Nx, Ny)
    ux    = KernelAbstractions.zeros(backend, T, Nx, Ny)
    uy    = KernelAbstractions.zeros(backend, T, Nx, Ny)
    is_solid = KernelAbstractions.zeros(backend, Bool, Nx, Ny)

    w = weights(D2Q9())
    f_cpu = zeros(T, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx
        x = T(i - 1)
        y = T(j - 1)
        ux_tg = -T(u0) * cos(k * x) * sin(k * y)
        uy_tg =  T(u0) * sin(k * x) * cos(k * y)
        ρ_tg = one(T) - T(3) * T(u0)^2 / T(4) * (cos(T(2) * k * x) + cos(T(2) * k * y))
        for q in 1:9
            f_cpu[i, j, q] = equilibrium(D2Q9(), ρ_tg, ux_tg, uy_tg, q)
        end
    end
    copyto!(f_in, f_cpu)
    copyto!(f_out, f_cpu)

    return (f_in=f_in, f_out=f_out, ρ=ρ_arr, ux=ux, uy=uy, is_solid=is_solid,
            config=config, u0=u0, k=k)
end

"""
    run_taylor_green_2d(; N=64, ν=0.01, u0=0.01, max_steps=1000, backend, T)

Taylor-Green vortex decay in a fully periodic domain.
Returns final macroscopic fields.
"""
function run_taylor_green_2d(; N=64, ν=0.01, u0=0.01, max_steps=1000,
                              backend=KernelAbstractions.CPU(), T=Float64)
    state = initialize_taylor_green_2d(; N=N, ν=ν, u0=u0, backend=backend, T=T)
    f_in, f_out = state.f_in, state.f_out
    ρ, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid
    ω = T(1.0 / (3.0 * ν + 0.5))

    for step in 1:max_steps
        stream_fully_periodic_2d!(f_out, f_in, N, N)
        collide_2d!(f_out, is_solid, ω)
        f_in, f_out = f_out, f_in
    end

    compute_macroscopic_2d!(ρ, ux, uy, f_in)
    return (ρ=Array(ρ), ux=Array(ux), uy=Array(uy),
            config=state.config, u0=u0, k=state.k, max_steps=max_steps)
end

# --- Cylinder 2D ---

"""
    initialize_cylinder_2d(; Nx=200, Ny=50, cx=Nx÷4, cy=Ny÷2, radius=10,
                            u_in=0.05, ν=0.05, backend, T)

Initialize a channel with a circular cylinder obstacle.
"""
function initialize_cylinder_2d(; Nx=200, Ny=50, cx=nothing, cy=nothing, radius=10,
                                 u_in=0.05, ν=0.05,
                                 backend=KernelAbstractions.CPU(), T=Float64)
    cx = isnothing(cx) ? Nx ÷ 4 : cx
    cy = isnothing(cy) ? Ny ÷ 2 : cy
    config = LBMConfig(D2Q9(); Nx=Nx, Ny=Ny, ν=ν, u_lid=0.0, max_steps=0)
    state = initialize_2d(config, T; backend=backend)

    # Set cylinder as solid
    solid_cpu = zeros(Bool, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        if (i - cx)^2 + (j - cy)^2 <= radius^2
            solid_cpu[i, j] = true
        end
    end
    copyto!(state.is_solid, solid_cpu)

    # Initialize to equilibrium with uniform inflow velocity
    w = weights(D2Q9())
    f_cpu = zeros(T, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx
        for q in 1:9
            f_cpu[i, j, q] = equilibrium(D2Q9(), one(T), T(u_in), zero(T), q)
        end
    end
    copyto!(state.f_in, f_cpu)
    copyto!(state.f_out, f_cpu)

    return state, config
end

"""
    compute_drag_mea_2d(f_pre, f_post, is_solid, Nx, Ny)

Compute drag and lift via two-time-level momentum exchange:
- `f_pre`: populations BEFORE streaming (= post-collision from previous step)
- `f_post`: populations AFTER streaming + BCs (= pre-collision current step)

The force on the solid for each boundary link q from fluid to solid is:
  F = c_q · f_q(pre-stream) + c_q · f_q̄(post-stream)
"""
function compute_drag_mea_2d(f_pre, f_post, is_solid, Nx, Ny)
    fpre = Array(f_pre)
    fpost = Array(f_post)
    solid = Array(is_solid)
    cxv = [0, 1, 0, -1,  0, 1, -1, -1,  1]
    cyv = [0, 0, 1,  0, -1, 1,  1, -1, -1]
    opp = [1, 4, 5, 2, 3, 8, 9, 6, 7]

    Fx_total = 0.0
    Fy_total = 0.0

    for j in 1:Ny, i in 1:Nx
        if !solid[i, j]
            for q in 2:9
                ni = i + cxv[q]
                nj = j + cyv[q]
                if 1 <= ni <= Nx && 1 <= nj <= Ny && solid[ni, nj]
                    oq = opp[q]
                    # f_q(pre-stream): population about to leave fluid toward solid
                    # f_opp(post-stream): population that just bounced back from solid
                    Fx_total += Float64(cxv[q]) * (fpre[i, j, q] + fpost[i, j, oq])
                    Fy_total += Float64(cyv[q]) * (fpre[i, j, q] + fpost[i, j, oq])
                end
            end
        end
    end

    return (Fx=Fx_total, Fy=Fy_total)
end

"""
    run_cylinder_2d(; Nx=200, Ny=50, radius=10, u_in=0.05, ν=0.05,
                     max_steps=20000, backend, T)

Flow around a cylinder at Re = u_in * 2*radius / ν.
Drag computed via momentum exchange on post-stream populations, averaged
over last `avg_window` steps.
"""
function run_cylinder_2d(; Nx=200, Ny=50, cx=nothing, cy=nothing, radius=10,
                          u_in=0.05, ν=0.05, max_steps=20000, avg_window=1000,
                          backend=KernelAbstractions.CPU(), T=Float64)
    state, config = initialize_cylinder_2d(; Nx=Nx, Ny=Ny, cx=cx, cy=cy,
                                            radius=radius, u_in=u_in, ν=ν,
                                            backend=backend, T=T)
    f_in, f_out = state.f_in, state.f_out
    ρ, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid
    ω = T(omega(config))

    Fx_sum = 0.0
    Fy_sum = 0.0
    n_avg = 0

    for step in 1:max_steps
        # 1. Stream (f_in = pre-stream, f_out = post-stream)
        stream_2d!(f_out, f_in, Nx, Ny)

        # 2. BCs
        apply_zou_he_west_2d!(f_out, u_in, Nx, Ny)
        apply_zou_he_pressure_east_2d!(f_out, Nx, Ny)

        # 3. MEA drag: f_pre=f_in (pre-stream), f_post=f_out (post-stream+BCs)
        if step > max_steps - avg_window
            drag = compute_drag_mea_2d(f_in, f_out, is_solid, Nx, Ny)
            Fx_sum += drag.Fx
            Fy_sum += drag.Fy
            n_avg += 1
        end

        # 4. Collide
        collide_2d!(f_out, is_solid, ω)

        # 5. Macroscopic + swap
        compute_macroscopic_2d!(ρ, ux, uy, f_out)
        f_in, f_out = f_out, f_in
    end

    Fx_avg = Fx_sum / n_avg
    Fy_avg = Fy_sum / n_avg
    D = 2 * radius
    Cd = 2.0 * Fx_avg / (1.0 * u_in^2 * D)

    return (ρ=Array(ρ), ux=Array(ux), uy=Array(uy), config=config,
            Cd=Cd, Fx=Fx_avg, Fy=Fy_avg)
end

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

# --- Axisymmetric D2Q9 (z-r coordinates) ---

"""
    collide_axisymmetric_2d!(f, is_solid, ω, Nx, Ny)

BGK collision with axisymmetric source terms for D2Q9 in (z,r) coordinates.
x-direction = z (axial), y-direction = r (radial).

Source term from Peng et al. (2003) / Zhou (2011):
Accounts for 1/r geometric terms in cylindrical Navier-Stokes.
r = j - 0.5 (half-way BB, r=0 axis at j=0.5).
"""
@kernel function collide_axisymmetric_2d_kernel!(f, @Const(is_solid), ω, Ny)
    i, j = @index(Global, NTuple)

    @inbounds begin
        if is_solid[i, j]
            tmp2 = f[i,j,2]; f[i,j,2] = f[i,j,4]; f[i,j,4] = tmp2
            tmp3 = f[i,j,3]; f[i,j,3] = f[i,j,5]; f[i,j,5] = tmp3
            tmp6 = f[i,j,6]; f[i,j,6] = f[i,j,8]; f[i,j,8] = tmp6
            tmp7 = f[i,j,7]; f[i,j,7] = f[i,j,9]; f[i,j,9] = tmp7
        else
            T = eltype(f)
            f1=f[i,j,1]; f2=f[i,j,2]; f3=f[i,j,3]; f4=f[i,j,4]
            f5=f[i,j,5]; f6=f[i,j,6]; f7=f[i,j,7]; f8=f[i,j,8]; f9=f[i,j,9]

            ρ = f1+f2+f3+f4+f5+f6+f7+f8+f9
            inv_ρ = one(T) / ρ
            uz = (f2 - f4 + f6 - f7 - f8 + f9) * inv_ρ  # axial (x=z)
            ur = (f3 - f5 + f6 + f7 - f8 - f9) * inv_ρ  # radial (y=r)
            usq = uz * uz + ur * ur

            # Radial position: r = j - 0.5 (half-way BB, axis at r=0)
            r = T(j) - T(0.5)
            inv_r = one(T) / r

            # Standard BGK collision
            cu = zero(T)
            feq = T(4.0/9.0) * ρ * (one(T) - T(1.5)*usq)
            f[i,j,1] = f1 - ω*(f1-feq)

            cu = uz
            feq = T(1.0/9.0) * ρ * (one(T) + T(3)*cu + T(4.5)*cu*cu - T(1.5)*usq)
            f[i,j,2] = f2 - ω*(f2-feq)

            cu = ur
            feq = T(1.0/9.0) * ρ * (one(T) + T(3)*cu + T(4.5)*cu*cu - T(1.5)*usq)
            f[i,j,3] = f3 - ω*(f3-feq)

            cu = -uz
            feq = T(1.0/9.0) * ρ * (one(T) + T(3)*cu + T(4.5)*cu*cu - T(1.5)*usq)
            f[i,j,4] = f4 - ω*(f4-feq)

            cu = -ur
            feq = T(1.0/9.0) * ρ * (one(T) + T(3)*cu + T(4.5)*cu*cu - T(1.5)*usq)
            f[i,j,5] = f5 - ω*(f5-feq)

            cu = uz + ur
            feq = T(1.0/36.0) * ρ * (one(T) + T(3)*cu + T(4.5)*cu*cu - T(1.5)*usq)
            f[i,j,6] = f6 - ω*(f6-feq)

            cu = -uz + ur
            feq = T(1.0/36.0) * ρ * (one(T) + T(3)*cu + T(4.5)*cu*cu - T(1.5)*usq)
            f[i,j,7] = f7 - ω*(f7-feq)

            cu = -uz - ur
            feq = T(1.0/36.0) * ρ * (one(T) + T(3)*cu + T(4.5)*cu*cu - T(1.5)*usq)
            f[i,j,8] = f8 - ω*(f8-feq)

            cu = uz - ur
            feq = T(1.0/36.0) * ρ * (one(T) + T(3)*cu + T(4.5)*cu*cu - T(1.5)*usq)
            f[i,j,9] = f9 - ω*(f9-feq)

            # Axisymmetric source: S_q = -w_q * ur/r * (1 + (c_qr*ur)*3 ... )
            # Simplified form (Zhou 2011): S_q = -f_eq_q * ur / r
            # This accounts for the ∂(r·ur)/r∂r - ur/r mass/momentum correction
            pref = -ur * inv_r

            f[i,j,1] = f[i,j,1] + pref * T(4.0/9.0) * ρ * (one(T) - T(1.5)*usq)
            f[i,j,2] = f[i,j,2] + pref * T(1.0/9.0) * ρ * (one(T) + T(3)*uz + T(4.5)*uz*uz - T(1.5)*usq)
            f[i,j,3] = f[i,j,3] + pref * T(1.0/9.0) * ρ * (one(T) + T(3)*ur + T(4.5)*ur*ur - T(1.5)*usq)
            f[i,j,4] = f[i,j,4] + pref * T(1.0/9.0) * ρ * (one(T) - T(3)*uz + T(4.5)*uz*uz - T(1.5)*usq)
            f[i,j,5] = f[i,j,5] + pref * T(1.0/9.0) * ρ * (one(T) - T(3)*ur + T(4.5)*ur*ur - T(1.5)*usq)
            f[i,j,6] = f[i,j,6] + pref * T(1.0/36.0) * ρ * (one(T) + T(3)*(uz+ur) + T(4.5)*(uz+ur)*(uz+ur) - T(1.5)*usq)
            f[i,j,7] = f[i,j,7] + pref * T(1.0/36.0) * ρ * (one(T) + T(3)*(-uz+ur) + T(4.5)*(-uz+ur)*(-uz+ur) - T(1.5)*usq)
            f[i,j,8] = f[i,j,8] + pref * T(1.0/36.0) * ρ * (one(T) + T(3)*(-uz-ur) + T(4.5)*(-uz-ur)*(-uz-ur) - T(1.5)*usq)
            f[i,j,9] = f[i,j,9] + pref * T(1.0/36.0) * ρ * (one(T) + T(3)*(uz-ur) + T(4.5)*(uz-ur)*(uz-ur) - T(1.5)*usq)
        end
    end
end

function collide_axisymmetric_2d!(f, is_solid, ω, Nx, Ny)
    backend = KernelAbstractions.get_backend(f)
    kernel! = collide_axisymmetric_2d_kernel!(backend)
    kernel!(f, is_solid, ω, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

# --- Li et al. (2010) axisymmetric LBM scheme ---
#
# From: "Improved axisymmetric lattice Boltzmann scheme", PRE 81, 056707
#
# Key formulas (Eqs. 12-13, 16-17):
# Source:    S_α = [(e_αi - u_i)·F_i / (ρ·cs²) - u_r/r] · f_α^eq
#            where F_r = -2μ·u_r/r²  (only radial component)
# Collision: f̂(x+e, t+1) = f̂(x,t) - ω_f·[f̂ - f^eq] + (1-0.5·ω_f)·S
#            where ω_f = [1 + τ·e_αr/r] / (τ + 0.5)  — direction-dependent!
# Velocity:  u_i = Σ e_αi·f̂_α / [Σ f̂_α + μ/r²·δ_ir]
# Density:   ρ = Σ f̂_α / [1 + 0.5·u_r/r]

# The driver uses a combined stream → collision approach.
# Since ω_f depends on direction, we implement the full collision inline.

@kernel function collide_li_axisym_2d_kernel!(f, @Const(is_solid), τ, Ny, Fz_body)
    i, j = @index(Global, NTuple)

    @inbounds begin
        if is_solid[i, j]
            tmp = f[i,j,2]; f[i,j,2] = f[i,j,4]; f[i,j,4] = tmp
            tmp = f[i,j,3]; f[i,j,3] = f[i,j,5]; f[i,j,5] = tmp
            tmp = f[i,j,6]; f[i,j,6] = f[i,j,8]; f[i,j,8] = tmp
            tmp = f[i,j,7]; f[i,j,7] = f[i,j,9]; f[i,j,9] = tmp
        else
            T = eltype(f)
            f1=f[i,j,1]; f2=f[i,j,2]; f3=f[i,j,3]; f4=f[i,j,4]
            f5=f[i,j,5]; f6=f[i,j,6]; f7=f[i,j,7]; f8=f[i,j,8]; f9=f[i,j,9]

            r = T(j) - T(0.5)
            inv_r = one(T) / r
            μ = τ * T(1.0/3.0)  # dynamic viscosity = τ·cs²

            # Macroscopic: Eq. (16) — u_i = Σ e_αi·f̂ / [Σ f̂ + μ/r²·δ_ir]
            sum_f = f1+f2+f3+f4+f5+f6+f7+f8+f9
            mom_z = f2-f4+f6-f7-f8+f9
            mom_r = f3-f5+f6+f7-f8-f9

            denom_z = sum_f  # no correction for z
            denom_r = sum_f + μ * inv_r * inv_r  # Eq. (16) with δ_ir

            uz = mom_z / denom_z
            ur = mom_r / denom_r

            # Density: Eq. (17) — ρ = Σ f̂ / [1 + 0.5·u_r/r]
            ρ = sum_f / (one(T) + T(0.5) * ur * inv_r)

            usq = uz*uz + ur*ur

            # Force: F_r = -2μ·u_r/r², F_z = Fz_body (external body force)
            Fr = -T(2) * μ * ur * inv_r * inv_r
            Fz = Fz_body

            # D2Q9 velocity components: e_αr for each direction
            # q: 1  2  3  4  5  6  7  8  9
            # cr: 0  0  1  0 -1  1  1 -1 -1
            # cz: 0  1  0 -1  0  1 -1 -1  1

            # For each direction α:
            # ω_f = [1 + τ·e_αr/r] / (τ + 0.5)          — Eq. (13)
            # S_α = [(e_αi - u_i)·F_geom/(ρcs²) - ur/r] · f_eq  — Eq. (12)
            # Guo_α = (1-0.5ω_f)·w_α·[(c-u)·F_ext/cs² + (c·u)(c·F_ext)/cs⁴]
            # f = f - ω_f·(f-feq) + (1-0.5ω_f)·S_α + Guo_α

            inv_cs2 = T(3)
            ur_over_r = ur * inv_r
            ω_std = one(T) / (τ + T(0.5))  # standard ω for Guo external force
            guo_ext_pref = one(T) - T(0.5) * ω_std  # (1-ω/2) for external force

            # Helper: compute collision for one direction
            # cz_q, cr_q = lattice velocity components
            # w_q = weight
            # For HP with ur=0: S_α=0, only ω_f asymmetry + Guo external force matter

            # Macro for each direction: BGK(ω_f) + axisym source + Guo external force
            # Inlined for all 9 directions:

            # q=1: rest (cz=0, cr=0), w=4/9
            feq = T(4.0/9.0)*ρ*(one(T) - T(1.5)*usq)
            ω_f = one(T) / (τ + T(0.5))
            # Geometric source: S = [(0-uz)*0 + (0-ur)*Fr]/(ρ·cs²) - ur/r] * feq
            Sq = ((-ur)*Fr * inv_cs2 / ρ - ur_over_r) * feq
            # Guo external: w*(c-u)·Fext*3 + w*(c·u)(c·Fext)*9
            Gq = T(4.0/9.0)*((-uz)*Fz*T(3))
            f[i,j,1] = f1 - ω_f*(f1-feq) + (one(T)-T(0.5)*ω_f)*Sq + guo_ext_pref*Gq

            # q=2: E (cz=1, cr=0), w=1/9
            cu=uz; feq=T(1.0/9.0)*ρ*(one(T)+T(3)*cu+T(4.5)*cu*cu-T(1.5)*usq)
            ω_f = one(T) / (τ + T(0.5))
            Sq = (((-ur)*Fr * inv_cs2 / ρ) - ur_over_r) * feq
            Gq = T(1.0/9.0)*((one(T)-uz)*Fz*T(3) + uz*Fz*T(9))
            f[i,j,2] = f2 - ω_f*(f2-feq) + (one(T)-T(0.5)*ω_f)*Sq + guo_ext_pref*Gq

            # q=3: N (cz=0, cr=+1), w=1/9
            cu=ur; feq=T(1.0/9.0)*ρ*(one(T)+T(3)*cu+T(4.5)*cu*cu-T(1.5)*usq)
            ω_f = (one(T) + τ*inv_r) / (τ + T(0.5))
            Sq = (((one(T)-ur)*Fr * inv_cs2 / ρ) - ur_over_r) * feq
            Gq = T(1.0/9.0)*((-uz)*Fz*T(3))
            f[i,j,3] = f3 - ω_f*(f3-feq) + (one(T)-T(0.5)*ω_f)*Sq + guo_ext_pref*Gq

            # q=4: W (cz=-1, cr=0), w=1/9
            cu=-uz; feq=T(1.0/9.0)*ρ*(one(T)+T(3)*cu+T(4.5)*cu*cu-T(1.5)*usq)
            ω_f = one(T) / (τ + T(0.5))
            Sq = (((-ur)*Fr * inv_cs2 / ρ) - ur_over_r) * feq
            Gq = T(1.0/9.0)*((-one(T)-uz)*Fz*T(3) + uz*Fz*T(9))
            f[i,j,4] = f4 - ω_f*(f4-feq) + (one(T)-T(0.5)*ω_f)*Sq + guo_ext_pref*Gq

            # q=5: S (cz=0, cr=-1), w=1/9
            cu=-ur; feq=T(1.0/9.0)*ρ*(one(T)+T(3)*cu+T(4.5)*cu*cu-T(1.5)*usq)
            ω_f = (one(T) - τ*inv_r) / (τ + T(0.5))
            Sq = (((-one(T)-ur)*Fr * inv_cs2 / ρ) - ur_over_r) * feq
            Gq = T(1.0/9.0)*((-uz)*Fz*T(3))
            f[i,j,5] = f5 - ω_f*(f5-feq) + (one(T)-T(0.5)*ω_f)*Sq + guo_ext_pref*Gq

            # q=6: NE (cz=1, cr=+1), w=1/36
            cu=uz+ur; feq=T(1.0/36.0)*ρ*(one(T)+T(3)*cu+T(4.5)*cu*cu-T(1.5)*usq)
            ω_f = (one(T) + τ*inv_r) / (τ + T(0.5))
            Sq = (((one(T)-ur)*Fr * inv_cs2 / ρ) - ur_over_r) * feq
            Gq = T(1.0/36.0)*((one(T)-uz)*Fz*T(3) + cu*Fz*T(9))
            f[i,j,6] = f6 - ω_f*(f6-feq) + (one(T)-T(0.5)*ω_f)*Sq + guo_ext_pref*Gq

            # q=7: NW (cz=-1, cr=+1), w=1/36
            cu=-uz+ur; feq=T(1.0/36.0)*ρ*(one(T)+T(3)*cu+T(4.5)*cu*cu-T(1.5)*usq)
            ω_f = (one(T) + τ*inv_r) / (τ + T(0.5))
            Sq = (((one(T)-ur)*Fr * inv_cs2 / ρ) - ur_over_r) * feq
            Gq = T(1.0/36.0)*((-one(T)-uz)*Fz*T(3) + cu*(-Fz)*T(9))
            f[i,j,7] = f7 - ω_f*(f7-feq) + (one(T)-T(0.5)*ω_f)*Sq + guo_ext_pref*Gq

            # q=8: SW (cz=-1, cr=-1), w=1/36
            cu=-uz-ur; feq=T(1.0/36.0)*ρ*(one(T)+T(3)*cu+T(4.5)*cu*cu-T(1.5)*usq)
            ω_f = (one(T) - τ*inv_r) / (τ + T(0.5))
            Sq = (((-one(T)-ur)*Fr * inv_cs2 / ρ) - ur_over_r) * feq
            Gq = T(1.0/36.0)*((-one(T)-uz)*Fz*T(3) + cu*(-Fz)*T(9))
            f[i,j,8] = f8 - ω_f*(f8-feq) + (one(T)-T(0.5)*ω_f)*Sq + guo_ext_pref*Gq

            # q=9: SE (cz=1, cr=-1), w=1/36
            cu=uz-ur; feq=T(1.0/36.0)*ρ*(one(T)+T(3)*cu+T(4.5)*cu*cu-T(1.5)*usq)
            ω_f = (one(T) - τ*inv_r) / (τ + T(0.5))
            Sq = (((-one(T)-ur)*Fr * inv_cs2 / ρ) - ur_over_r) * feq
            Gq = T(1.0/36.0)*((one(T)-uz)*Fz*T(3) + cu*Fz*T(9))
            f[i,j,9] = f9 - ω_f*(f9-feq) + (one(T)-T(0.5)*ω_f)*Sq + guo_ext_pref*Gq
        end
    end
end

function collide_li_axisym_2d!(f, is_solid, τ, Nx, Ny; Fz_body=0.0)
    backend = KernelAbstractions.get_backend(f)
    T = eltype(f)
    kernel! = collide_li_axisym_2d_kernel!(backend)
    kernel!(f, is_solid, T(τ), Ny, T(Fz_body); ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

"""
    run_hagen_poiseuille_2d(; Nz=4, Nr=32, ν=0.1, Fz=1e-5, max_steps=10000, backend, T)

Hagen-Poiseuille pipe flow (axisymmetric). Validates axisymmetric LBM.
Analytical: u_z(r) = Fz/(4ν) * (R² - r²) where R = Nr - 0.5.
"""
function run_hagen_poiseuille_2d(; Nz=4, Nr=32, ν=0.1, Fz=1e-5, max_steps=10000,
                                  backend=KernelAbstractions.CPU(), FT=Float64)
    Nx, Ny = Nz, Nr
    config = LBMConfig(D2Q9(); Nx=Nx, Ny=Ny, ν=ν, u_lid=0.0, max_steps=max_steps)
    state = initialize_2d(config, FT; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    ρ, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid
    ω = FT(omega(config))

    # Axisymmetric approach: standard BGK + Guo forcing with TWO forces:
    # 1. Body force Fz (drives the flow)
    # 2. Axisym viscous correction Fz_axi = ν/r · ∂uz/∂r (from macroscopic field)
    # Plus mass source -ρ·ur/r (negligible for HP since ur≈0)
    #
    # The correction is computed from the macroscopic velocity field at each step
    # using central FD on the (converging) uz field.

    # Pre-allocate force arrays
    Fz_total = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Fr_total = KernelAbstractions.zeros(backend, FT, Nx, Ny)

    for step in 1:max_steps
        stream_periodic_x_axisym_2d!(f_out, f_in, Nx, Ny)

        # Compute axisymmetric correction from current macroscopic field
        # (uses ux from PREVIOUS step — lagged but stable)
        if step > 1
            uz_cpu = Array(ux)  # ux = uz in axisym convention
            Fz_cpu = zeros(FT, Nx, Ny)
            for j in 2:Ny-1, i in 1:Nx
                r = FT(j) - FT(0.5)
                duz_dr = (uz_cpu[i, j+1] - uz_cpu[i, j-1]) / FT(2)
                Fz_cpu[i, j] = FT(Fz) + FT(ν) / r * duz_dr
            end
            # j=1 (near axis, r=0.5): L'Hôpital — lim(r→0) ν/r·∂u/∂r = ν·∂²u/∂r²
            # By symmetry: u(r=-Δr) = u(r=+Δr) → u(j=0) = u(j=2)
            # ∂²u/∂r² ≈ (u[j+1] - 2u[j] + u[j-1]) = 2*(u[2] - u[1])
            for i in 1:Nx
                d2uz_dr2 = FT(2) * (uz_cpu[i, 2] - uz_cpu[i, 1])
                Fz_cpu[i, 1] = FT(Fz) + FT(ν) * d2uz_dr2
            end
            # j=Ny (wall): just body force
            for i in 1:Nx
                Fz_cpu[i, Ny] = FT(Fz)
            end
            copyto!(Fz_total, Fz_cpu)
        else
            # First step: just body force
            fill!(Array(Fz_total), FT(Fz))
            Fz_cpu = fill(FT(Fz), Nx, Ny)
            copyto!(Fz_total, Fz_cpu)
        end

        # Collision with per-node Guo force field
        collide_guo_field_2d!(f_out, is_solid, Fz_total, Fr_total, ω)
        compute_macroscopic_2d!(ρ, ux, uy, f_out)
        f_in, f_out = f_out, f_in
    end

    return (ρ=Array(ρ), uz=Array(ux), ur=Array(uy), config=config)
end

# --- Rayleigh-Plateau pinch-off (axisymmetric VOF-LBM) ---

"""
    run_rp_axisym_2d(; Nz=256, Nr=40, R0=15, λ_ratio=7.0, ε=0.05,
                      σ=0.01, ν=0.05, ρ_l=1.0, ρ_g=0.01,
                      max_steps=10000, output_interval=500, backend, T)

Rayleigh-Plateau capillary instability in axisymmetric geometry.

Coordinates: x=z (axial, periodic), y=r (radial, axis at j=1, wall at j=Nr).
A liquid jet of radius R0 with sinusoidal perturbation R(z) = R0(1-ε·cos(2πz/λ)).
λ > 2πR0 → unstable → pinch-off.

The azimuthal curvature κ₂ = n_r/r drives the instability (absent in 2D planar).
"""
function run_rp_axisym_2d(; Nz=256, Nr=40, R0=15, λ_ratio=7.0, ε=0.05,
                           σ=0.01, ν=0.05, ρ_l=1.0, ρ_g=0.01,
                           max_steps=10000, output_interval=500,
                           backend=KernelAbstractions.CPU(), FT=Float64)
    Nx, Ny = Nz, Nr
    config = LBMConfig(D2Q9(); Nx=Nx, Ny=Ny, ν=ν, u_lid=0.0, max_steps=max_steps)
    state = initialize_2d(config, FT; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    ρ, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid
    ω = FT(omega(config))

    # VOF arrays
    C     = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    C_new = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    nx_n  = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    ny_n  = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    κ     = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Fx_st = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Fy_st = KernelAbstractions.zeros(backend, FT, Nx, Ny)

    # Axisymmetric force arrays
    Fz_field = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Fr_field = KernelAbstractions.zeros(backend, FT, Nx, Ny)

    λ = FT(λ_ratio * R0)

    # Initialize: axisymmetric jet with perturbation
    C_cpu = zeros(FT, Nx, Ny)
    w = weights(D2Q9())
    f_cpu = zeros(FT, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx
        r = FT(j) - FT(0.5)  # radial position
        R_local = FT(R0) * (one(FT) - FT(ε) * cos(FT(2π) * FT(i - 1) / λ))
        C_cpu[i, j] = FT(0.5) * (one(FT) - tanh((r - R_local) / FT(1.5)))
        ρ_init = C_cpu[i,j] * FT(ρ_l) + (one(FT) - C_cpu[i,j]) * FT(ρ_g)
        for q in 1:9
            f_cpu[i, j, q] = FT(w[q]) * ρ_init
        end
    end
    copyto!(C, C_cpu)
    copyto!(f_in, f_cpu)
    copyto!(f_out, f_cpu)

    r_min_history = FT[]
    times = Int[]

    for step in 1:max_steps
        # 1. Stream (axisym: periodic z, specular axis, wall at Nr)
        stream_periodic_x_axisym_2d!(f_out, f_in, Nx, Ny)

        # 2. Macroscopic
        compute_macroscopic_2d!(ρ, ux, uy, f_out)

        # 3. VOF advection (same as Cartesian for incompressible)
        advect_vof_2d!(C_new, C, ux, uy, Nx, Ny)
        copyto!(C, C_new)
        C_cpu = Array(C); clamp!(C_cpu, FT(0), FT(1)); copyto!(C, C_cpu)

        # 4. Curvature: meridional (κ₁) + azimuthal (κ₂ = n_r/r)
        compute_vof_normal_2d!(nx_n, ny_n, C, Nx, Ny)
        compute_hf_curvature_2d!(κ, C, nx_n, ny_n, Nx, Ny)
        add_azimuthal_curvature_2d!(κ, C, ny_n, Ny)

        # 5. Surface tension force
        compute_surface_tension_2d!(Fx_st, Fy_st, κ, C, σ, Nx, Ny)

        # 6. Axisymmetric viscous correction + surface tension
        uz_cpu = Array(ux)
        Fz_cpu = zeros(FT, Nx, Ny)
        Fr_cpu = Array(Fy_st)
        Fx_st_cpu = Array(Fx_st)
        for j in 1:Ny, i in 1:Nx
            r = FT(j) - FT(0.5)
            if j > 1 && j < Ny
                duz_dr = (uz_cpu[i,j+1] - uz_cpu[i,j-1]) / FT(2)
            elseif j == 1
                duz_dr = FT(2) * (uz_cpu[i,2] - uz_cpu[i,1])
                duz_dr = duz_dr  # L'Hôpital: ν/r·∂u/∂r → ν·∂²u/∂r²
            else
                duz_dr = zero(FT)
            end
            axisym_corr = j == 1 ? FT(ν) * duz_dr : FT(ν) / r * duz_dr
            Fz_cpu[i,j] = Fx_st_cpu[i,j] + axisym_corr
            Fr_cpu[i,j] = Fr_cpu[i,j]  # surface tension in r already there
        end
        copyto!(Fz_field, Fz_cpu)
        copyto!(Fr_field, Fr_cpu)

        # 7. Two-phase collision with per-node force
        collide_twophase_2d!(f_out, C, Fz_field, Fr_field, is_solid;
                             ρ_l=ρ_l, ρ_g=ρ_g, ν_l=ν, ν_g=ν)

        f_in, f_out = f_out, f_in

        # Track minimum jet radius
        if step % output_interval == 0 || step == 1
            C_cpu = Array(C)
            r_min_val = FT(Inf)
            for i in 1:Nx
                # Find interface position (where C ≈ 0.5)
                for j in 1:Ny-1
                    if C_cpu[i,j] > 0.5 && C_cpu[i,j+1] <= 0.5
                        # Linear interpolation
                        r_interf = (j - 0.5) + (C_cpu[i,j] - 0.5) / (C_cpu[i,j] - C_cpu[i,j+1])
                        r_min_val = min(r_min_val, r_interf)
                        break
                    end
                end
            end
            push!(r_min_history, r_min_val)
            push!(times, step)
        end
    end

    compute_macroscopic_2d!(ρ, ux, uy, f_in)

    return (ρ=Array(ρ), uz=Array(ux), ur=Array(uy), C=Array(C),
            r_min=r_min_history, times=times, config=config,
            σ=σ, R0=R0, λ=λ, ε=ε)
end

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

        # 3. VOF advection
        advect_vof_2d!(C_new, C, ux, uy, Nx, Ny)
        copyto!(C, C_new)

        # 4. Clamp C to [0,1]
        C_cpu = Array(C)
        clamp!(C_cpu, FT(0), FT(1))
        copyto!(C, C_cpu)

        # 5. Interface normal + curvature + surface tension
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

        # 3. VOF advection
        advect_vof_2d!(C_new, C, ux, uy, Nx, Ny)
        copyto!(C, C_new)

        # Clamp
        C_cpu = Array(C)
        clamp!(C_cpu, FT(0), FT(1))
        copyto!(C, C_cpu)

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

        # 4. VOF advection (fine grid, existing kernel with scaled velocity)
        advect_vof_2d!(C_fine_new, C_fine, ux_fine, uy_fine, Nx_f, Ny_f)
        copyto!(C_fine, C_fine_new)

        # 5. Clamp C ∈ [0,1]
        C_cpu = Array(C_fine)
        clamp!(C_cpu, FT(0), FT(1))
        copyto!(C_fine, C_cpu)

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

# --- CLSVOF static droplet (Level-Set curvature + VOF mass) ---

"""
    run_static_droplet_clsvof_2d(; N=128, R=20, σ=0.01, ν=0.1, ...)

Static droplet with CLSVOF: VOF advects C (conservative), level-set φ
provides smooth curvature κ = -∇·(∇φ/|∇φ|). The LS is reconstructed from
C at each step and redistanced to maintain |∇φ| ≈ 1.
"""
function run_static_droplet_clsvof_2d(; N=128, R=20, σ=0.01, ν=0.1,
                                        ρ_l=1.0, ρ_g=0.001, max_steps=5000,
                                        n_reinit=5, dtau_reinit=0.5, ε_delta=1.5,
                                        output_dir="", output_interval=100,
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
    Fx_st = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Fy_st = KernelAbstractions.zeros(backend, FT, Nx, Ny)

    # Level-set arrays
    phi      = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    phi_work = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    phi0     = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    κ_ls     = KernelAbstractions.zeros(backend, FT, Nx, Ny)

    # Initialize C (tanh) and φ (exact signed distance)
    C_cpu   = zeros(FT, Nx, Ny)
    phi_cpu = zeros(FT, Nx, Ny)
    w = weights(D2Q9())
    f_cpu = zeros(FT, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx
        r = sqrt(FT((i - cx)^2 + (j - cy)^2))
        C_cpu[i, j]   = FT(0.5) * (one(FT) - tanh((r - FT(R)) / FT(2)))
        phi_cpu[i, j] = FT(R) - r   # exact signed distance
        ρ_init = C_cpu[i,j] * FT(ρ_l) + (one(FT) - C_cpu[i,j]) * FT(ρ_g)
        for q in 1:9
            f_cpu[i, j, q] = FT(w[q]) * ρ_init
        end
    end
    copyto!(C, C_cpu)
    copyto!(phi, phi_cpu)
    copyto!(f_in, f_cpu)
    copyto!(f_out, f_cpu)

    # Output setup
    do_output = !isempty(output_dir)
    local pvd, diag_logger
    if do_output
        setup_output_dir(output_dir)
        pvd = create_pvd(joinpath(output_dir, "clsvof_droplet"))
        diag_logger = open_diagnostics(joinpath(output_dir, "diagnostics.csv"),
                                       ["step", "mass", "max_u"])
    end

    for step in 1:max_steps
        # 1. Stream
        stream_fully_periodic_2d!(f_out, f_in, Nx, Ny)

        # 2. Macroscopic
        compute_macroscopic_2d!(ρ, ux, uy, f_out)

        # 3. VOF advection (conservative mass transport)
        advect_vof_2d!(C_new, C, ux, uy, Nx, Ny)
        copyto!(C, C_new)
        C_cpu = Array(C); clamp!(C_cpu, FT(0), FT(1)); copyto!(C, C_cpu)

        # 4. Reconstruct φ from C + redistance
        ls_from_vof_2d!(phi, C, Nx, Ny)
        reinit_ls_2d!(phi, phi_work, phi0, Nx, Ny;
                       n_iter=n_reinit, dtau=dtau_reinit)

        # 5. Curvature from level-set (smooth!)
        curvature_ls_2d!(κ_ls, phi, Nx, Ny)

        # 6. Surface tension from LS
        surface_tension_clsvof_2d!(Fx_st, Fy_st, κ_ls, phi, σ, Nx, Ny;
                                    epsilon=ε_delta)

        # 7. Two-phase collision (C for density/viscosity, F from LS)
        collide_twophase_2d!(f_out, C, Fx_st, Fy_st, is_solid;
                             ρ_l=ρ_l, ρ_g=ρ_g, ν_l=ν, ν_g=ν)

        # Output snapshot
        if do_output && step % output_interval == 0
            C_out = Array(C)
            ux_out = Array(ux)
            uy_out = Array(uy)
            mass = sum(C_out)
            max_u_out = sqrt(maximum(ux_out .^ 2 .+ uy_out .^ 2))
            log_diagnostics!(diag_logger, step, mass, max_u_out)
            fields = Dict("rho" => Array(ρ), "ux" => ux_out, "uy" => uy_out,
                          "C" => C_out, "phi" => Array(phi), "kappa" => Array(κ_ls))
            write_snapshot_2d!(output_dir, step, Nx, Ny, 1.0, fields;
                               pvd=pvd, time=Float64(step))
        end

        f_in, f_out = f_out, f_in
    end

    # Close output
    if do_output
        close_diagnostics!(diag_logger)
        vtk_save(pvd)
    end

    compute_macroscopic_2d!(ρ, ux, uy, f_in)

    ρ_cpu = Array(ρ)
    ux_cpu = Array(ux)
    uy_cpu = Array(uy)
    C_cpu = Array(C)
    phi_cpu = Array(phi)

    max_u = sqrt(maximum(ux_cpu .^ 2 .+ uy_cpu .^ 2))
    inside_mask = [(i-cx)^2 + (j-cy)^2 < (R-3)^2 for i in 1:Nx, j in 1:Ny]
    outside_mask = [(i-cx)^2 + (j-cy)^2 > (R+3)^2 for i in 1:Nx, j in 1:Ny]
    p_in = sum(ρ_cpu[inside_mask]) / sum(inside_mask) / 3
    p_out = sum(ρ_cpu[outside_mask]) / sum(outside_mask) / 3
    Δp = p_in - p_out
    Δp_analytical = σ / R

    return (ρ=ρ_cpu, ux=ux_cpu, uy=uy_cpu, C=C_cpu, phi=phi_cpu,
            max_u_spurious=max_u, Δp=Δp, Δp_analytical=Δp_analytical,
            config=config, σ=σ, R=R)
end

# --- Rayleigh-Plateau capillary pinch-off with CLSVOF (axisymmetric) ---

"""
    run_rp_clsvof_2d(; Nz=256, Nr=40, R0=15, λ_ratio=7.0, ε=0.05, ...)

Rayleigh-Plateau capillary instability with CLSVOF in axisymmetric geometry.
VOF advects C (conservative), level-set φ provides smooth curvature.

Setup (cf. Popinet 2009, Gerris plateau example):
- Axisymmetric jet of radius R0 with perturbation R(z) = R0(1-ε·cos(2πz/λ))
- λ > 2πR0 → Rayleigh-unstable → pinch-off
- Total curvature: κ = κ_meridional(φ) + κ_azimuthal(φ) where κ₂ = -(1/r)·n_r

Analytical growth rate (inviscid linear theory):
  ω² = σ/(ρ·R0³) · x(1-x²) · I₁(x)/I₀(x)  where x = 2πR0/λ
"""
function run_rp_clsvof_2d(; Nz=256, Nr=40, R0=15, λ_ratio=7.0, ε=0.05,
                            σ=0.01, ν=0.05, ρ_l=1.0, ρ_g=0.01,
                            n_reinit=5, dtau_reinit=0.5, ε_delta=1.5,
                            max_steps=10000, output_interval=500,
                            output_dir="",
                            backend=KernelAbstractions.CPU(), FT=Float64)
    Nx, Ny = Nz, Nr
    config = LBMConfig(D2Q9(); Nx=Nx, Ny=Ny, ν=ν, u_lid=0.0, max_steps=max_steps)
    state = initialize_2d(config, FT; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    ρ, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid

    # VOF arrays
    C     = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    C_new = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Fx_st = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Fy_st = KernelAbstractions.zeros(backend, FT, Nx, Ny)

    # Level-set arrays
    phi      = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    phi_work = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    phi0     = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    κ_ls     = KernelAbstractions.zeros(backend, FT, Nx, Ny)

    # Axisymmetric force arrays
    Fz_field = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Fr_field = KernelAbstractions.zeros(backend, FT, Nx, Ny)

    λ = FT(λ_ratio * R0)

    # Initialize: axisymmetric jet with perturbation
    C_cpu   = zeros(FT, Nx, Ny)
    phi_cpu = zeros(FT, Nx, Ny)
    w = weights(D2Q9())
    f_cpu = zeros(FT, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx
        r = FT(j) - FT(0.5)  # radial position
        R_local = FT(R0) * (one(FT) - FT(ε) * cos(FT(2π) * FT(i - 1) / λ))
        C_cpu[i, j]   = FT(0.5) * (one(FT) - tanh((r - R_local) / FT(1.5)))
        phi_cpu[i, j] = R_local - r  # signed distance (positive inside jet)
        ρ_init = C_cpu[i,j] * FT(ρ_l) + (one(FT) - C_cpu[i,j]) * FT(ρ_g)
        for q in 1:9
            f_cpu[i, j, q] = FT(w[q]) * ρ_init
        end
    end
    copyto!(C, C_cpu)
    copyto!(phi, phi_cpu)
    copyto!(f_in, f_cpu)
    copyto!(f_out, f_cpu)

    r_min_history = FT[]
    times = Int[]

    # Output setup
    do_output = !isempty(output_dir)
    local pvd_rp, diag_logger_rp
    if do_output
        setup_output_dir(output_dir)
        pvd_rp = create_pvd(joinpath(output_dir, "clsvof_rp"))
        diag_logger_rp = open_diagnostics(joinpath(output_dir, "diagnostics.csv"),
                                          ["step", "mass", "max_u", "r_min"])
    end

    for step in 1:max_steps
        # 1. Stream (axisym: periodic z, specular axis, wall at Nr)
        stream_periodic_x_axisym_2d!(f_out, f_in, Nx, Ny)

        # 2. Macroscopic
        compute_macroscopic_2d!(ρ, ux, uy, f_out)

        # 3. VOF advection (conservative mass transport)
        advect_vof_2d!(C_new, C, ux, uy, Nx, Ny)
        copyto!(C, C_new)
        C_cpu = Array(C); clamp!(C_cpu, FT(0), FT(1)); copyto!(C, C_cpu)

        # 4. Reconstruct φ from C + redistance
        ls_from_vof_2d!(phi, C, Nx, Ny)
        reinit_ls_2d!(phi, phi_work, phi0, Nx, Ny;
                       n_iter=n_reinit, dtau=dtau_reinit)

        # 5. Curvature: meridional (LS) + azimuthal (LS)
        curvature_ls_2d!(κ_ls, phi, Nx, Ny)
        add_azimuthal_curvature_ls_2d!(κ_ls, phi, Ny)

        # 6. Surface tension: hybrid LS curvature + VOF gradient
        compute_surface_tension_2d!(Fx_st, Fy_st, κ_ls, C, σ, Nx, Ny)

        # 7. Axisymmetric viscous correction
        uz_cpu = Array(ux)
        Fz_cpu = zeros(FT, Nx, Ny)
        Fr_cpu = Array(Fy_st)
        Fx_st_cpu = Array(Fx_st)
        for j in 1:Ny, i in 1:Nx
            r = FT(j) - FT(0.5)
            if j > 1 && j < Ny
                duz_dr = (uz_cpu[i,j+1] - uz_cpu[i,j-1]) / FT(2)
            elseif j == 1
                duz_dr = FT(2) * (uz_cpu[i,2] - uz_cpu[i,1])
            else
                duz_dr = zero(FT)
            end
            axisym_corr = j == 1 ? FT(ν) * duz_dr : FT(ν) / r * duz_dr
            Fz_cpu[i,j] = Fx_st_cpu[i,j] + axisym_corr
            Fr_cpu[i,j] = Fr_cpu[i,j]
        end
        copyto!(Fz_field, Fz_cpu)
        copyto!(Fr_field, Fr_cpu)

        # 8. Two-phase collision
        collide_twophase_2d!(f_out, C, Fz_field, Fr_field, is_solid;
                             ρ_l=ρ_l, ρ_g=ρ_g, ν_l=ν, ν_g=ν)

        f_in, f_out = f_out, f_in

        # Track minimum jet radius
        if step % output_interval == 0 || step == 1
            C_cpu = Array(C)
            r_min_val = FT(Inf)
            for i in 1:Nx
                for j in 1:Ny-1
                    if C_cpu[i,j] > 0.5 && C_cpu[i,j+1] <= 0.5
                        r_interf = (j - 0.5) + (C_cpu[i,j] - 0.5) / (C_cpu[i,j] - C_cpu[i,j+1])
                        r_min_val = min(r_min_val, r_interf)
                        break
                    end
                end
            end
            push!(r_min_history, r_min_val)
            push!(times, step)

            # Output snapshot
            if do_output && step % output_interval == 0
                ux_out = Array(ux)
                uy_out = Array(uy)
                mass = sum(C_cpu)
                max_u_out = sqrt(maximum(ux_out .^ 2 .+ uy_out .^ 2))
                log_diagnostics!(diag_logger_rp, step, mass, max_u_out, r_min_val)
                fields = Dict("rho" => Array(ρ), "uz" => ux_out, "ur" => uy_out,
                              "C" => C_cpu, "phi" => Array(phi), "kappa" => Array(κ_ls))
                write_snapshot_2d!(output_dir, step, Nx, Ny, 1.0, fields;
                                   pvd=pvd_rp, time=Float64(step))
            end
        end
    end

    # Close output
    if do_output
        close_diagnostics!(diag_logger_rp)
        vtk_save(pvd_rp)
    end

    compute_macroscopic_2d!(ρ, ux, uy, f_in)

    return (ρ=Array(ρ), uz=Array(ux), ur=Array(uy), C=Array(C), phi=Array(phi),
            r_min=r_min_history, times=times, config=config,
            σ=σ, R0=R0, λ=λ, ε=ε)
end
