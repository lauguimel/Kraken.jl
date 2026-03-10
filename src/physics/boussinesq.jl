using KernelAbstractions

"""
    buoyancy_kernel!(fu, fv, T, β, T_ref, gx, gy)

GPU-native kernel computing the Boussinesq buoyancy force: f = β·(T - T_ref)·g.
Operates on interior points (offset by 1 for boundary padding).
"""
@kernel function buoyancy_kernel!(fu, fv, @Const(T), β, T_ref, gx, gy)
    i, j = @index(Global, NTuple)
    ii = i + 1
    jj = j + 1
    @inbounds begin
        dT = T[ii, jj] - T_ref
        fu[ii, jj] = β * dT * gx
        fv[ii, jj] = β * dT * gy
    end
end

"""
    buoyancy_force!(fu, fv, T, β, T_ref, gx, gy; ndrange=nothing)

Compute Boussinesq buoyancy force f = β·(T - T_ref)·g for interior points.

Works on CPU and GPU arrays automatically via KernelAbstractions.

# Arguments
- `fu, fv`: output force arrays (N × N), modified in-place
- `T`: temperature field (N × N)
- `β`: thermal expansion coefficient (or β·g combined)
- `T_ref`: reference temperature
- `gx, gy`: gravity direction components

# Returns
- `(fu, fv)`: the buoyancy force arrays.

See also: [`advance_temperature!`](@ref), [`run_rayleigh_benard`](@ref)
"""
function buoyancy_force!(fu, fv, T, β, T_ref, gx, gy; ndrange=nothing)
    backend = KernelAbstractions.get_backend(T)
    n = size(T, 1) - 2
    if ndrange === nothing
        ndrange = (n, n)
    end
    FT = eltype(T)
    kernel! = buoyancy_kernel!(backend)
    kernel!(fu, fv, T, FT(β), FT(T_ref), FT(gx), FT(gy); ndrange=ndrange)
    KernelAbstractions.synchronize(backend)
    return fu, fv
end

"""
    advance_temperature!(T, u, v, κ, dx, dt;
                         adv_T=similar(T), lap_T=similar(T))

Advance the temperature field by one explicit Euler timestep:
    ∂T/∂t + u·∇T = κ·∇²T

Uses the existing `advect!` and `laplacian!` operators.

# Arguments
- `T`: temperature field (N × N), modified in-place
- `u, v`: velocity components (N × N)
- `κ`: thermal diffusivity
- `dx`: grid spacing
- `dt`: timestep

# Keyword Arguments
- `adv_T`: work array for advection term
- `lap_T`: work array for Laplacian term

# Returns
- `T`: the updated temperature field.

See also: [`buoyancy_force!`](@ref), [`run_rayleigh_benard`](@ref)
"""
function advance_temperature!(T, u, v, κ, dx, dt;
                               adv_T=similar(T), lap_T=similar(T))
    FT = eltype(T)

    fill!(adv_T, zero(FT))
    fill!(lap_T, zero(FT))

    advect!(adv_T, u, v, T, dx)
    laplacian!(lap_T, T, dx)

    # Explicit Euler: T = T + dt * (-advT + κ * lapT)
    T .= T .+ FT(dt) .* (.-adv_T .+ FT(κ) .* lap_T)

    return T
end

"""
    apply_temperature_bc_rb!(T, N, T_hot, T_cold)

Apply Rayleigh-Bénard temperature boundary conditions on an N×N grid:
- Bottom wall (j=1): T = T_hot
- Top wall (j=N): T = T_cold
- Left/Right walls: insulated (zero gradient, Neumann)
"""
function apply_temperature_bc_rb!(T, N, T_hot, T_cold)
    FT = eltype(T)
    # Bottom wall: T = T_hot
    T[:, 1] .= FT(T_hot)
    # Top wall: T = T_cold
    T[:, N] .= FT(T_cold)
    # Left wall: insulated (zero gradient)
    T[1, :] .= @view T[2, :]
    # Right wall: insulated (zero gradient)
    T[N, :] .= @view T[N-1, :]
    return T
end

"""
    apply_velocity_bc_rb!(u, v, N)

Apply no-slip velocity boundary conditions for Rayleigh-Bénard on an N×N grid:
all four walls have u=0, v=0.
"""
function apply_velocity_bc_rb!(u, v, N)
    FT_u = eltype(u)
    FT_v = eltype(v)
    # Bottom wall (j=1)
    u[:, 1] .= zero(FT_u)
    v[:, 1] .= zero(FT_v)
    # Top wall (j=N)
    u[:, N] .= zero(FT_u)
    v[:, N] .= zero(FT_v)
    # Left wall (i=1)
    u[1, :] .= zero(FT_u)
    v[1, :] .= zero(FT_v)
    # Right wall (i=N)
    u[N, :] .= zero(FT_u)
    v[N, :] .= zero(FT_v)
    return u, v
end

"""
    run_rayleigh_benard(; N=64, Ra=1e4, Pr=0.71, aspect=2.0,
                         max_steps=10000, tol=1e-6, cfl=0.5,
                         verbose=false, backend=CPU(), float_type=Float64)

Run Rayleigh-Bénard convection with Boussinesq approximation on an N×N grid.

Domain: [0, 1] × [0, 1] (square, uniform cells).
BCs: T=1 (hot) at bottom, T=0 (cold) at top, insulated sides, no-slip walls.

Uses explicit time-stepping with Chorin's projection method and buoyancy forcing.

# Keyword Arguments
- `N::Int`: grid points per dimension (default: 64)
- `Ra`: Rayleigh number (default: 1e4)
- `Pr`: Prandtl number (default: 0.71)
- `aspect`: aspect ratio (reserved for future rectangular grid support, currently unused)
- `max_steps::Int`: maximum timesteps (default: 10000)
- `tol`: convergence tolerance on max velocity change (default: 1e-6)
- `cfl`: CFL number (default: 0.5)
- `verbose::Bool`: print progress (default: false)
- `backend`: KernelAbstractions backend (default: CPU())
- `float_type`: floating-point type (default: Float64)

# Returns
- `(u, v, p, T, converged)`: velocity, pressure, temperature, convergence flag

# Example
```julia
u, v, p, T, converged = run_rayleigh_benard(N=32, Ra=1e4, max_steps=5000)
```

See also: [`advance_temperature!`](@ref), [`buoyancy_force!`](@ref), [`projection_step!`](@ref)
"""
function run_rayleigh_benard(; N=64, Ra=1e4, Pr=0.71, aspect=2.0,
                              max_steps=10000, tol=1e-6, cfl=0.5,
                              verbose=false, backend=KernelAbstractions.CPU(),
                              float_type=Float64)
    FT = float_type
    H = FT(1.0)
    ΔT = FT(1.0)  # T_hot - T_cold
    T_hot = FT(1.0)
    T_cold = FT(0.0)
    T_ref = FT(0.5)  # reference temperature

    dx = H / FT(N - 1)

    # Non-dimensionalization: Ra = β·g·ΔT·H³/(ν·κ), Pr = ν/κ
    # Choose ν = sqrt(Pr/Ra), κ = 1/sqrt(Pr*Ra) so that βg·ΔT = 1
    ν = FT(sqrt(Pr / Ra))
    κ = ν / FT(Pr)
    βg = FT(1.0)  # β·g·ΔT combined (non-dimensional)

    # Buoyancy direction: upward (+y) for T > T_ref
    gx_dir = FT(0.0)
    gy_dir = FT(1.0)

    # Timestep: limited by diffusive stability and advective CFL
    dt_diff = FT(cfl) * dx^2 / max(ν, κ)
    dt_adv = FT(cfl) * dx
    dt = min(dt_diff, dt_adv)

    # Allocate fields
    u = KernelAbstractions.zeros(backend, FT, N, N)
    v = KernelAbstractions.zeros(backend, FT, N, N)
    p = KernelAbstractions.zeros(backend, FT, N, N)
    T_field = KernelAbstractions.zeros(backend, FT, N, N)

    # Initialize temperature: linear profile + small perturbation to trigger convection
    T_init = Array{FT}(undef, N, N)
    for j in 1:N
        y = FT(j - 1) * dx
        for i in 1:N
            x = FT(i - 1) * dx
            T_init[i, j] = T_hot - ΔT * y / H +
                           FT(0.01) * sin(FT(π) * x / H) * sin(FT(π) * y / H)
        end
    end
    copyto!(T_field, T_init)

    # Apply initial BCs
    apply_velocity_bc_rb!(u, v, N)
    apply_temperature_bc_rb!(T_field, N, T_hot, T_cold)

    # Work arrays
    adv_u = KernelAbstractions.zeros(backend, FT, N, N)
    adv_v = KernelAbstractions.zeros(backend, FT, N, N)
    lap_u = KernelAbstractions.zeros(backend, FT, N, N)
    lap_v = KernelAbstractions.zeros(backend, FT, N, N)
    div_field = KernelAbstractions.zeros(backend, FT, N, N)
    gx_arr = KernelAbstractions.zeros(backend, FT, N, N)
    gy_arr = KernelAbstractions.zeros(backend, FT, N, N)
    fu = KernelAbstractions.zeros(backend, FT, N, N)
    fv = KernelAbstractions.zeros(backend, FT, N, N)
    adv_T = KernelAbstractions.zeros(backend, FT, N, N)
    lap_T = KernelAbstractions.zeros(backend, FT, N, N)

    u_old = similar(u)
    n = N - 2  # interior points

    converged = false
    for step in 1:max_steps
        copyto!(u_old, u)

        # --- Advance temperature ---
        advance_temperature!(T_field, u, v, κ, dx, dt;
                            adv_T=adv_T, lap_T=lap_T)
        apply_temperature_bc_rb!(T_field, N, T_hot, T_cold)

        # --- Compute buoyancy force ---
        fill!(fu, zero(FT))
        fill!(fv, zero(FT))
        buoyancy_force!(fu, fv, T_field, βg, T_ref, gx_dir, gy_dir)

        # --- Momentum: explicit projection with buoyancy ---
        # Step 1: intermediate velocity u* = u + dt*(-adv + ν·lap + f_buoy)
        fill!(adv_u, zero(FT))
        fill!(adv_v, zero(FT))
        advect!(adv_u, u, v, u, dx)
        advect!(adv_v, u, v, v, dx)

        fill!(lap_u, zero(FT))
        fill!(lap_v, zero(FT))
        laplacian!(lap_u, u, dx)
        laplacian!(lap_v, v, dx)

        # Update interior via broadcasting (GPU-compatible)
        u .= u .+ FT(dt) .* (.-adv_u .+ FT(ν) .* lap_u .+ fu)
        v .= v .+ FT(dt) .* (.-adv_v .+ FT(ν) .* lap_v .+ fv)

        apply_velocity_bc_rb!(u, v, N)

        # Step 2: pressure Poisson ∇²p = (1/dt)∇·u*
        fill!(div_field, zero(FT))
        divergence!(div_field, u, v, dx)
        div_field ./= dt

        solve_poisson_neumann!(p, div_field, dx; maxiter=2000, rtol=FT(1e-4), x0=p)
        apply_pressure_neumann_bc!(p, N)

        # Step 3: velocity correction u = u* - dt·∇p
        fill!(gx_arr, zero(FT))
        fill!(gy_arr, zero(FT))
        gradient!(gx_arr, gy_arr, p, dx)

        u .= u .- FT(dt) .* gx_arr
        v .= v .- FT(dt) .* gy_arr

        apply_velocity_bc_rb!(u, v, N)

        # Check convergence
        max_change = maximum(abs.(u .- u_old))
        if max_change < tol
            if verbose
                println("Converged at step $step (max_change = $max_change)")
            end
            converged = true
            break
        end

        if verbose && step % 500 == 0
            println("Step $step: max_change = $max_change, dt = $dt")
        end
    end

    return u, v, p, T_field, converged
end
