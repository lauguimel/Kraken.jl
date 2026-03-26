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
