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

# --- Full axisymmetric source (Peng 2003 / Guo-Halliday formulation) ---
#
# In cylindrical (z,r) coords, the Navier-Stokes has extra terms vs Cartesian:
# - Continuity: +(ρ·ur)/r
# - z-momentum: +(ν/r)·∂uz/∂r  (extra viscous diffusion)
# - r-momentum: +(ν/r)·∂ur/∂r - ν·ur/r²  (extra viscous + geometric)
#
# These are implemented as effective body forces in the Guo scheme:
#   Fz_axi = (ν/r) · ∂uz/∂r  (estimated by finite difference)
#   Fr_axi = (ν/r) · ∂ur/∂r - ν·ur/r² - ur²/r  (viscous + convective)
# Plus a mass source: S_mass = -ρ·ur/r (applied to all f_q via equilibrium)

@kernel function apply_axisym_source_full_2d_kernel!(f, @Const(f_pre), ω, Ny)
    i, j = @index(Global, NTuple)

    @inbounds begin
        T = eltype(f)
        f1=f[i,j,1]; f2=f[i,j,2]; f3=f[i,j,3]; f4=f[i,j,4]
        f5=f[i,j,5]; f6=f[i,j,6]; f7=f[i,j,7]; f8=f[i,j,8]; f9=f[i,j,9]

        ρ = f1+f2+f3+f4+f5+f6+f7+f8+f9
        inv_ρ = one(T) / ρ
        uz = (f2-f4+f6-f7-f8+f9) * inv_ρ
        ur = (f3-f5+f6+f7-f8-f9) * inv_ρ
        usq = uz*uz + ur*ur

        r = T(j) - T(0.5)
        inv_r = one(T) / r
        nu_eff = (one(T)/ω - T(0.5)) / T(3)  # ν = (1/ω - 0.5) * cs²

        # Estimate ∂uz/∂r and ∂ur/∂r from pre-collision populations (central difference)
        if j > 1 && j < Ny
            # Macroscopic at j+1 and j-1 from pre-stream populations
            ρ_p = zero(T); uz_p = zero(T); ur_p = zero(T)
            ρ_m = zero(T); uz_m = zero(T); ur_m = zero(T)
            for q in 1:9
                ρ_p += f_pre[i, j+1, q]
                ρ_m += f_pre[i, j-1, q]
            end
            uz_p = (f_pre[i,j+1,2]-f_pre[i,j+1,4]+f_pre[i,j+1,6]-f_pre[i,j+1,7]-f_pre[i,j+1,8]+f_pre[i,j+1,9]) / ρ_p
            uz_m = (f_pre[i,j-1,2]-f_pre[i,j-1,4]+f_pre[i,j-1,6]-f_pre[i,j-1,7]-f_pre[i,j-1,8]+f_pre[i,j-1,9]) / ρ_m
            ur_p = (f_pre[i,j+1,3]-f_pre[i,j+1,5]+f_pre[i,j+1,6]+f_pre[i,j+1,7]-f_pre[i,j+1,8]-f_pre[i,j+1,9]) / ρ_p
            ur_m = (f_pre[i,j-1,3]-f_pre[i,j-1,5]+f_pre[i,j-1,6]+f_pre[i,j-1,7]-f_pre[i,j-1,8]-f_pre[i,j-1,9]) / ρ_m

            duz_dr = (uz_p - uz_m) / T(2)
            dur_dr = (ur_p - ur_m) / T(2)
        else
            duz_dr = zero(T)
            dur_dr = zero(T)
        end

        # Axisymmetric forces
        Fz_axi = nu_eff * inv_r * duz_dr
        Fr_axi = nu_eff * inv_r * dur_dr - nu_eff * ur * inv_r * inv_r - ur * ur * inv_r

        # Mass source: -ρ·ur/r (applied as -f_eq·ur/r)
        mass_src = -ur * inv_r

        # Apply Guo-style force + mass source
        guo_pref = one(T) - ω / T(2)

        # Combined: mass source via equilibrium + momentum via Guo
        for_q_w = T(4.0/9.0)
        feq_rest = for_q_w * ρ * (one(T) - T(1.5)*usq)
        Sq = for_q_w * ((-uz)*Fz_axi + (-ur)*Fr_axi)*T(3)
        f[i,j,1] = f1 + mass_src * feq_rest + guo_pref * Sq

        for_q_w = T(1.0/9.0)
        # q=2: E (cz=1, cr=0)
        feq = for_q_w*ρ*(one(T)+T(3)*uz+T(4.5)*uz*uz-T(1.5)*usq)
        Sq = for_q_w*((one(T)-uz)*Fz_axi+(-ur)*Fr_axi)*T(3) + for_q_w*uz*Fz_axi*T(9)
        f[i,j,2] = f2 + mass_src*feq + guo_pref*Sq

        # q=3: N (cz=0, cr=1)
        feq = for_q_w*ρ*(one(T)+T(3)*ur+T(4.5)*ur*ur-T(1.5)*usq)
        Sq = for_q_w*((-uz)*Fz_axi+(one(T)-ur)*Fr_axi)*T(3) + for_q_w*ur*Fr_axi*T(9)
        f[i,j,3] = f3 + mass_src*feq + guo_pref*Sq

        # q=4: W (cz=-1, cr=0)
        feq = for_q_w*ρ*(one(T)-T(3)*uz+T(4.5)*uz*uz-T(1.5)*usq)
        Sq = for_q_w*((-one(T)-uz)*Fz_axi+(-ur)*Fr_axi)*T(3) + for_q_w*uz*Fz_axi*T(9)
        f[i,j,4] = f4 + mass_src*feq + guo_pref*Sq

        # q=5: S (cz=0, cr=-1)
        feq = for_q_w*ρ*(one(T)-T(3)*ur+T(4.5)*ur*ur-T(1.5)*usq)
        Sq = for_q_w*((-uz)*Fz_axi+(-one(T)-ur)*Fr_axi)*T(3) + for_q_w*ur*Fr_axi*T(9)
        f[i,j,5] = f5 + mass_src*feq + guo_pref*Sq

        for_q_w = T(1.0/36.0)
        # q=6: NE (cz=1, cr=1)
        cu=uz+ur; feq=for_q_w*ρ*(one(T)+T(3)*cu+T(4.5)*cu*cu-T(1.5)*usq)
        Sq = for_q_w*((one(T)-uz)*Fz_axi+(one(T)-ur)*Fr_axi)*T(3) + for_q_w*cu*(Fz_axi+Fr_axi)*T(9)
        f[i,j,6] = f6 + mass_src*feq + guo_pref*Sq

        # q=7: NW (cz=-1, cr=1)
        cu=-uz+ur; feq=for_q_w*ρ*(one(T)+T(3)*cu+T(4.5)*cu*cu-T(1.5)*usq)
        Sq = for_q_w*((-one(T)-uz)*Fz_axi+(one(T)-ur)*Fr_axi)*T(3) + for_q_w*cu*(-Fz_axi+Fr_axi)*T(9)
        f[i,j,7] = f7 + mass_src*feq + guo_pref*Sq

        # q=8: SW (cz=-1, cr=-1)
        cu=-uz-ur; feq=for_q_w*ρ*(one(T)+T(3)*cu+T(4.5)*cu*cu-T(1.5)*usq)
        Sq = for_q_w*((-one(T)-uz)*Fz_axi+(-one(T)-ur)*Fr_axi)*T(3) + for_q_w*cu*(-Fz_axi-Fr_axi)*T(9)
        f[i,j,8] = f8 + mass_src*feq + guo_pref*Sq

        # q=9: SE (cz=1, cr=-1)
        cu=uz-ur; feq=for_q_w*ρ*(one(T)+T(3)*cu+T(4.5)*cu*cu-T(1.5)*usq)
        Sq = for_q_w*((one(T)-uz)*Fz_axi+(-one(T)-ur)*Fr_axi)*T(3) + for_q_w*cu*(Fz_axi-Fr_axi)*T(9)
        f[i,j,9] = f9 + mass_src*feq + guo_pref*Sq
    end
end

function apply_axisym_source_2d!(f, f_pre, ω, Nx, Ny)
    backend = KernelAbstractions.get_backend(f)
    kernel! = apply_axisym_source_full_2d_kernel!(backend)
    kernel!(f, f_pre, ω, Ny; ndrange=(Nx, Ny))
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

    # Axis at j=0.5 (y=r direction), wall at j=Nr+0.5
    # Streaming: periodic in z (x), wall bounce-back at r=Nr (j=Ny)
    # At j=1 (near axis): use symmetry (bounce-back mimics axis condition)

    for step in 1:max_steps
        stream_periodic_x_wall_y_2d!(f_out, f_in, Nx, Ny)
        collide_guo_2d!(f_out, is_solid, ω, FT(Fz), FT(0))
        # Full axisymmetric source (uses f_in as pre-collision for gradient estimation)
        apply_axisym_source_2d!(f_out, f_in, ω, Nx, Ny)
        compute_macroscopic_forced_2d!(ρ, ux, uy, f_out, FT(Fz), FT(0))
        f_in, f_out = f_out, f_in
    end

    return (ρ=Array(ρ), uz=Array(ux), ur=Array(uy), config=config)
end
