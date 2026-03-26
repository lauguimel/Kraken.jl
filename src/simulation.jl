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
