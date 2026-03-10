using Krylov
using LinearAlgebra
using KernelAbstractions
using FFTW

"""
    NeumannLaplacianOperator{T}

Matrix-free linear operator that applies the negative discrete Laplacian (-∇²)
with homogeneous Neumann boundary conditions on a 2D uniform grid.

Neumann BCs are implemented by ghost-point reflection: the value outside the
domain equals the value just inside (zero-gradient condition). The operator
is singular (constant is in the null space), so the system must be regularized
by fixing one pressure value (e.g., p[1,1] = 0).

# Fields
- `N::Int`: total grid size (including boundary points)
- `dx::T`: grid spacing
"""
struct NeumannLaplacianOperator{T}
    N::Int
    dx::T
end

Base.size(op::NeumannLaplacianOperator) = (op.N^2, op.N^2)
Base.size(op::NeumannLaplacianOperator, d::Int) = op.N^2
Base.eltype(::NeumannLaplacianOperator{T}) where {T} = T

@kernel function neumann_laplacian_kernel!(Y, @Const(X), inv_dx2, N)
    idx = @index(Global)
    @inbounds begin
        i = mod1(idx, N)
        j = div(idx - 1, N) + 1

        # Neumann BC: ghost values = interior neighbor (zero gradient)
        xim = i > 1 ? X[idx - 1] : X[idx + 1]
        xip = i < N ? X[idx + 1] : X[idx - 1]
        xjm = j > 1 ? X[idx - N] : X[idx + N]
        xjp = j < N ? X[idx + N] : X[idx - N]

        val = (4 * X[idx] - xim - xip - xjm - xjp) * inv_dx2

        # Pin first element to remove null space
        if i == 1 && j == 1
            val += X[1] * inv_dx2
        end

        Y[idx] = val
    end
end

function LinearAlgebra.mul!(y::AbstractVector, op::NeumannLaplacianOperator, x::AbstractVector)
    N = op.N
    inv_dx2 = one(op.dx) / (op.dx * op.dx)
    backend = KernelAbstractions.get_backend(x)

    kernel! = neumann_laplacian_kernel!(backend)
    kernel!(y, x, inv_dx2, N; ndrange=N * N)
    KernelAbstractions.synchronize(backend)

    return y
end

"""
    NeumannJacobiPreconditioner{T}

Jacobi preconditioner for the Neumann Laplacian. Diagonal is 4/dx² everywhere.

# Fields
- `inv_diag::T`: the value dx²/4
"""
struct NeumannJacobiPreconditioner{T}
    inv_diag::T
end

Base.size(p::NeumannJacobiPreconditioner) = error("Not needed")
Base.eltype(::NeumannJacobiPreconditioner{T}) where {T} = T

function LinearAlgebra.mul!(y::AbstractVector, M::NeumannJacobiPreconditioner, x::AbstractVector)
    y .= M.inv_diag .* x
    return y
end

"""
    solve_poisson_neumann!(phi, rhs, dx; maxiter=2000, rtol=1e-8)

Solve ∇²φ = rhs with Neumann BCs using CG. The solution is unique up to a
constant; we pin φ[1,1] = 0 to remove the null space.

Works on CPU arrays and GPU arrays (CUDA, Metal) automatically.

# Arguments
- `phi`: output array (N × N), will be overwritten with the solution.
- `rhs`: right-hand side array (N × N).
- `dx`: uniform grid spacing.

# Keyword Arguments
- `maxiter::Int`: maximum CG iterations (default: 2000).
- `rtol`: relative tolerance (default: 1e-8).

# Returns
- `(phi, niter)`: solution array and iteration count.

# Example
```julia
N = 32; dx = 1.0 / (N - 1)
phi = zeros(N, N)
rhs = zeros(N, N)
phi, niter = solve_poisson_neumann!(phi, rhs, dx)
```

See also: [`solve_poisson_fft!`](@ref), [`solve_poisson_cg!`](@ref)
"""
function solve_poisson_neumann_dct!(phi, rhs, dx; eigenvalues=nothing)
    N = size(rhs, 1)
    T = eltype(rhs)

    # Copy GPU→CPU if needed (FFTW is CPU-only)
    rhs_cpu = rhs isa Array ? rhs : Array(rhs)

    # Forward DCT-I (REDFT00) — matches Neumann BCs with values at grid points
    f_hat = FFTW.r2r(Float64.(rhs_cpu), FFTW.REDFT00)

    # Build or use cached eigenvalues of Neumann Laplacian under DCT-I
    if eigenvalues !== nothing
        eig = eigenvalues
    else
        eig = zeros(Float64, N, N)
        inv_dx2 = 1.0 / (Float64(dx) * Float64(dx))
        for l in 1:N, k in 1:N
            eig[k, l] = 2.0 * inv_dx2 * (cos(π * Float64(k - 1) / Float64(N - 1)) - 1.0) +
                         2.0 * inv_dx2 * (cos(π * Float64(l - 1) / Float64(N - 1)) - 1.0)
        end
    end

    # Divide in spectral space (skip zero mode)
    @inbounds for l in 1:N, k in 1:N
        if k == 1 && l == 1
            f_hat[k, l] = 0.0
        else
            f_hat[k, l] /= eig[k, l]
        end
    end

    # Inverse DCT-I (REDFT00 is self-inverse, normalization = 2*(N-1) per dim)
    norm_factor = 4.0 * Float64(N - 1) * Float64(N - 1)
    result_cpu = FFTW.r2r(f_hat, FFTW.REDFT00) ./ norm_factor

    # Pin phi[1,1] = 0
    p11 = result_cpu[1, 1]
    result_cpu .-= p11

    # Copy back CPU→GPU if needed
    if phi isa Array
        phi .= T.(result_cpu)
    else
        copyto!(phi, T.(result_cpu))
    end

    return phi, 0
end

function solve_poisson_neumann!(phi, rhs, dx; maxiter=2000, rtol=1e-8, x0=nothing,
                                solver=nothing, work_b=nothing)
    N = size(rhs, 1)
    T = eltype(rhs)

    # Use pre-allocated work_b or allocate
    b = work_b === nothing ? similar(rhs, N * N) : work_b
    # Negate for SPD: -∇²φ = -rhs (GPU-compatible)
    b .= vec(-rhs)

    A = NeumannLaplacianOperator{T}(N, dx)
    M = NeumannJacobiPreconditioner{T}(T(dx * dx / 4))

    niter = 0
    if solver !== nothing
        # In-place solve with pre-allocated workspace (avoids per-call allocation)
        solver.warm_start = false
        cg!(solver, A, b; M=M, ldiv=false, itmax=maxiter, rtol=T(rtol), atol=zero(T))
        phi .= reshape(solver.x, N, N)
        niter = solver.stats.niter
    else
        (x, stats) = cg(A, b; M=M, ldiv=false, itmax=maxiter, rtol=T(rtol), atol=zero(T))
        phi .= reshape(x, N, N)
        niter = stats.niter
    end

    # Ensure pin: shift so phi[1,1] = 0
    # Use Array() scalar extraction for GPU compatibility
    p11 = Array(phi[1:1, 1:1])[1]
    phi .-= p11

    return phi, niter
end

"""
    apply_velocity_bc!(u, v, N)

Apply boundary conditions for the lid-driven cavity:
- Top wall (j=N): u=1, v=0
- Bottom wall (j=1): u=0, v=0
- Left wall (i=1): u=0, v=0
- Right wall (i=N): u=0, v=0

Works on CPU and GPU arrays via slice broadcasting.

# Arguments
- `u`: x-velocity array (N × N), modified in-place.
- `v`: y-velocity array (N × N), modified in-place.
- `N::Int`: grid size.

# Returns
- `(u, v)`: the modified velocity arrays.

See also: [`apply_pressure_neumann_bc!`](@ref), [`projection_step!`](@ref)
"""
function apply_velocity_bc!(u, v, N)
    T_u = eltype(u)
    T_v = eltype(v)

    # Bottom wall (j=1)
    u[:, 1] .= zero(T_u)
    v[:, 1] .= zero(T_v)
    # Top wall (j=N): lid moves with u=1
    u[:, N] .= one(T_u)
    v[:, N] .= zero(T_v)
    # Left wall (i=1)
    u[1, :] .= zero(T_u)
    v[1, :] .= zero(T_v)
    # Right wall (i=N)
    u[N, :] .= zero(T_u)
    v[N, :] .= zero(T_v)

    return u, v
end

"""
    apply_pressure_neumann_bc!(p, N)

Apply homogeneous Neumann BCs for pressure: ∂p/∂n = 0 at all walls.
Implemented by copying the nearest interior value to the boundary.

Works on CPU and GPU arrays via slice broadcasting.

# Arguments
- `p`: pressure array (N × N), modified in-place.
- `N::Int`: grid size.

# Returns
- `p`: the modified pressure array.

See also: [`apply_velocity_bc!`](@ref), [`projection_step!`](@ref)
"""
function apply_pressure_neumann_bc!(p, N)
    p[:, 1] .= @view p[:, 2]       # bottom
    p[:, N] .= @view p[:, N-1]     # top
    p[1, :] .= @view p[2, :]       # left
    p[N, :] .= @view p[N-1, :]     # right
    return p
end

@kernel function velocity_update_kernel!(u, v, @Const(adv_u), @Const(adv_v),
                                          @Const(lap_u), @Const(lap_v), ν, dt)
    i, j = @index(Global, NTuple)
    ii = i + 1
    jj = j + 1
    @inbounds begin
        u[ii, jj] = u[ii, jj] + dt * (-adv_u[ii, jj] + ν * lap_u[ii, jj])
        v[ii, jj] = v[ii, jj] + dt * (-adv_v[ii, jj] + ν * lap_v[ii, jj])
    end
end

@kernel function pressure_correction_kernel!(u, v, @Const(gx), @Const(gy), dt)
    i, j = @index(Global, NTuple)
    ii = i + 1
    jj = j + 1
    @inbounds begin
        u[ii, jj] = u[ii, jj] - dt * gx[ii, jj]
        v[ii, jj] = v[ii, jj] - dt * gy[ii, jj]
    end
end

"""
    projection_step!(u, v, p, ν, dx, dt, N;
                     adv_u=similar(u), adv_v=similar(v),
                     lap_u=similar(u), lap_v=similar(v),
                     div_field=similar(u), gx=similar(u), gy=similar(v))

Perform one timestep of the Chorin projection method for the 2D incompressible
Navier-Stokes equations on a collocated grid.

1. Compute intermediate velocity: u* = u + dt*(−u·∇u + ν∇²u)
2. Solve pressure Poisson equation: ∇²p = (1/dt)∇·u*
3. Correct velocity: u^{n+1} = u* − dt*∇p

Boundary conditions are applied after each sub-step.
Works on CPU and GPU arrays automatically.

# Arguments
- `u, v`: velocity components (N × N), modified in-place
- `p`: pressure field (N × N), modified in-place
- `ν`: kinematic viscosity (= 1/Re)
- `dx`: grid spacing
- `dt`: timestep
- `N`: grid size

# Keyword Arguments
- Work arrays: `adv_u, adv_v, lap_u, lap_v, div_field, gx, gy`

# Returns
- `(u, v, p)`: the updated velocity and pressure fields.

See also: [`run_cavity`](@ref), [`solve_poisson_neumann!`](@ref), [`apply_velocity_bc!`](@ref)
"""
function projection_step!(u, v, p, ν, dx, dt, N;
                          adv_u=similar(u), adv_v=similar(v),
                          lap_u=similar(u), lap_v=similar(v),
                          div_field=similar(u), gx=similar(u), gy=similar(v),
                          poisson_solver=:cg,
                          poisson_eigenvalues=nothing)
    T = eltype(u)
    backend = KernelAbstractions.get_backend(u)
    n = N - 2  # interior points

    # --- Step 1: intermediate velocity ---
    # Advection: u·∇u, u·∇v
    fill!(adv_u, zero(T))
    fill!(adv_v, zero(T))
    advect!(adv_u, u, v, u, dx)
    advect!(adv_v, u, v, v, dx)

    # Diffusion: ν∇²u, ν∇²v
    fill!(lap_u, zero(T))
    fill!(lap_v, zero(T))
    laplacian!(lap_u, u, dx)
    laplacian!(lap_v, v, dx)

    # u* = u + dt * (-advection + ν * laplacian) — via kernel
    kernel_vel! = velocity_update_kernel!(backend)
    kernel_vel!(u, v, adv_u, adv_v, lap_u, lap_v, T(ν), T(dt); ndrange=(n, n))
    KernelAbstractions.synchronize(backend)

    # Apply velocity BCs to intermediate velocity
    apply_velocity_bc!(u, v, N)

    # --- Step 2: pressure Poisson equation ---
    # ∇²p = (1/dt) * ∇·u*
    fill!(div_field, zero(T))
    divergence!(div_field, u, v, dx)

    # RHS = (1/dt) * div(u*) — in-place
    div_field ./= dt

    if poisson_eigenvalues !== nothing
        # Fast DCT-based direct solver (no iterations)
        solve_poisson_neumann_dct!(p, div_field, dx; eigenvalues=poisson_eigenvalues)
    elseif poisson_solver == :mg
        solve_poisson_mg!(p, div_field, dx)
    else
        solve_poisson_neumann!(p, div_field, dx; maxiter=2000, rtol=eltype(u)(1e-4), x0=p)
    end
    apply_pressure_neumann_bc!(p, N)

    # --- Step 3: velocity correction ---
    fill!(gx, zero(T))
    fill!(gy, zero(T))
    gradient!(gx, gy, p, dx)

    kernel_corr! = pressure_correction_kernel!(backend)
    kernel_corr!(u, v, gx, gy, T(dt); ndrange=(n, n))
    KernelAbstractions.synchronize(backend)

    # Apply velocity BCs after correction
    apply_velocity_bc!(u, v, N)

    return u, v, p
end

@kernel function implicit_rhs_kernel!(rhs_u, rhs_v, @Const(u), @Const(v),
                                      @Const(adv_u), @Const(adv_v), dt)
    i, j = @index(Global, NTuple)
    ii = i + 1
    jj = j + 1
    @inbounds begin
        rhs_u[ii, jj] = u[ii, jj] - dt * adv_u[ii, jj]
        rhs_v[ii, jj] = v[ii, jj] - dt * adv_v[ii, jj]
    end
end

"""
    projection_step_implicit!(u, v, p, ν, dx, dt, N;
                              adv_u=similar(u), adv_v=similar(v),
                              rhs_u=similar(u), rhs_v=similar(v),
                              div_field=similar(u), gx=similar(u), gy=similar(v))

Perform one timestep of the semi-implicit projection method for the 2D
incompressible Navier-Stokes equations on a collocated grid.

Diffusion is treated implicitly via the Helmholtz equation, removing the
viscous stability constraint on the timestep. Only the advective CFL remains.

1. Compute advection explicitly
2. Build RHS: rhs = u - dt·advection
3. Solve Helmholtz: (I - dt·ν·∇²) u* = rhs
4. Solve pressure Poisson: ∇²p = (1/dt)∇·u*
5. Correct velocity: u^{n+1} = u* − dt·∇p

Works on CPU and GPU arrays automatically.

# Arguments
- `u, v`: velocity components (N × N), modified in-place
- `p`: pressure field (N × N), modified in-place
- `ν`: kinematic viscosity (= 1/Re)
- `dx`: grid spacing
- `dt`: timestep
- `N`: grid size

# Keyword Arguments
- Work arrays: `adv_u, adv_v, rhs_u, rhs_v, div_field, gx, gy`

# Returns
- `(u, v, p)`: the updated velocity and pressure fields.

See also: [`projection_step!`](@ref), [`solve_helmholtz!`](@ref), [`run_cavity`](@ref)
"""
function projection_step_implicit!(u, v, p, ν, dx, dt, N;
                                    adv_u=similar(u), adv_v=similar(v),
                                    rhs_u=similar(u), rhs_v=similar(v),
                                    div_field=similar(u), gx=similar(u), gy=similar(v),
                                    poisson_solver_obj=nothing,
                                    helmholtz_solver_u=nothing,
                                    helmholtz_solver_v=nothing,
                                    work_b=nothing,
                                    poisson_eigenvalues=nothing)
    T = eltype(u)
    backend = KernelAbstractions.get_backend(u)
    n = N - 2  # interior points

    # --- Step 1: Compute advection (explicit) ---
    fill!(adv_u, zero(T))
    fill!(adv_v, zero(T))
    advect!(adv_u, u, v, u, dx)
    advect!(adv_v, u, v, v, dx)

    # --- Step 2: Build RHS for Helmholtz ---
    # rhs = u - dt * advection (interior points only, boundary from BCs)
    copyto!(rhs_u, u)
    copyto!(rhs_v, v)
    kernel_rhs! = implicit_rhs_kernel!(backend)
    kernel_rhs!(rhs_u, rhs_v, u, v, adv_u, adv_v, T(dt); ndrange=(n, n))
    KernelAbstractions.synchronize(backend)

    # --- Step 3: Solve Helmholtz for u*, v* ---
    sigma = dt * ν
    if poisson_eigenvalues !== nothing
        # Fast DCT-based direct solver (no iterations)
        solve_helmholtz_dct!(u, rhs_u, dx, sigma;
                             poisson_eigenvalues=poisson_eigenvalues)
        solve_helmholtz_dct!(v, rhs_v, dx, sigma;
                             poisson_eigenvalues=poisson_eigenvalues)
    else
        solve_helmholtz!(u, rhs_u, dx, sigma;
                         solver=helmholtz_solver_u, work_b=work_b)
        solve_helmholtz!(v, rhs_v, dx, sigma;
                         solver=helmholtz_solver_v, work_b=work_b)
    end

    # Apply velocity BCs to intermediate velocity
    apply_velocity_bc!(u, v, N)

    # --- Step 4: Pressure Poisson equation ---
    fill!(div_field, zero(T))
    divergence!(div_field, u, v, dx)

    # RHS = (1/dt) * div(u*) — in-place
    div_field ./= dt

    if poisson_eigenvalues !== nothing
        # Fast DCT-based direct solver (no iterations)
        solve_poisson_neumann_dct!(p, div_field, dx; eigenvalues=poisson_eigenvalues)
    else
        solve_poisson_neumann!(p, div_field, dx; maxiter=2000, rtol=eltype(u)(1e-4), x0=p,
                               solver=poisson_solver_obj, work_b=work_b)
    end
    apply_pressure_neumann_bc!(p, N)

    # --- Step 5: Velocity correction ---
    fill!(gx, zero(T))
    fill!(gy, zero(T))
    gradient!(gx, gy, p, dx)

    kernel_corr! = pressure_correction_kernel!(backend)
    kernel_corr!(u, v, gx, gy, T(dt); ndrange=(n, n))
    KernelAbstractions.synchronize(backend)

    # Apply velocity BCs after correction
    apply_velocity_bc!(u, v, N)

    return u, v, p
end

"""
    available_backends()

Return a list of available KernelAbstractions backends.
Always includes CPU. Adds MetalBackend if Metal.jl is functional,
CUDABackend if CUDA.jl is functional.

# Returns
- `Vector{KernelAbstractions.Backend}`: list of available backends

# Example
```julia
backends = available_backends()  # [CPU(), ...Metal/CUDA if available]
```
"""
function available_backends()
    backends = Any[KernelAbstractions.CPU()]
    try
        @eval using Metal
        if Metal.functional()
            push!(backends, Metal.MetalBackend())
        end
    catch
    end
    try
        @eval using CUDA
        if CUDA.functional()
            push!(backends, CUDA.CUDABackend())
        end
    catch
    end
    return backends
end

"""
    run_cavity(; N=64, Re=100.0, cfl=0.2, max_steps=10000, tol=1e-6, verbose=false,
                 backend=KernelAbstractions.CPU(), float_type=Float64,
                 poisson_solver=:cg, time_scheme=:explicit)

Run the lid-driven cavity benchmark at given Reynolds number on an N×N grid.

Uses Chorin's projection method. Time stepping can be explicit (default) or
semi-implicit (diffusion treated implicitly via Helmholtz solver).

# Keyword Arguments
- `N::Int`: grid points per dimension (default: 64)
- `Re::Float64`: Reynolds number (default: 100)
- `cfl::Float64`: CFL number for timestep selection (default: 0.2)
- `max_steps::Int`: maximum number of timesteps (default: 10000)
- `tol::Float64`: convergence tolerance on max velocity change (default: 1e-6)
- `verbose::Bool`: print progress every 1000 steps (default: false)
- `backend`: KernelAbstractions backend (default: CPU())
- `float_type`: floating-point type (default: Float64, use Float32 for Metal)
- `poisson_solver::Symbol`: `:cg` (default) or `:mg` (geometric multigrid)
- `time_scheme::Symbol`: `:explicit` (default) or `:implicit` (semi-implicit diffusion)

# Returns
- `(u, v, p, converged)`: velocity components, pressure, and convergence flag

# Example
```julia
u, v, p, converged = run_cavity(N=32, Re=100.0, max_steps=500)
u, v, p, converged = run_cavity(N=32, Re=100.0, time_scheme=:implicit)
```

See also: [`projection_step!`](@ref), [`projection_step_implicit!`](@ref), [`available_backends`](@ref)
"""
function run_cavity(; N=64, Re=100.0, cfl=0.2, max_steps=10000, tol=1e-6,
                     verbose=false, backend=KernelAbstractions.CPU(), float_type=Float64,
                     poisson_solver=:cg, time_scheme=:explicit)
    T = float_type
    dx = T(1.0) / T(N - 1)
    ν = T(1.0) / T(Re)

    # Initialize fields using KernelAbstractions.zeros for backend support
    u = KernelAbstractions.zeros(backend, T, N, N)
    v = KernelAbstractions.zeros(backend, T, N, N)
    p = KernelAbstractions.zeros(backend, T, N, N)

    # Apply initial BCs
    apply_velocity_bc!(u, v, N)

    # Preallocate work arrays (shared between explicit/implicit)
    adv_u = KernelAbstractions.zeros(backend, T, N, N)
    adv_v = KernelAbstractions.zeros(backend, T, N, N)
    div_field = KernelAbstractions.zeros(backend, T, N, N)
    gx = KernelAbstractions.zeros(backend, T, N, N)
    gy = KernelAbstractions.zeros(backend, T, N, N)

    # Pre-compute DCT-I eigenvalues for direct solvers (always use DCT, even on GPU)
    # On GPU: DCT is computed on CPU with data copies (negligible for N≤512)
    inv_dx2 = one(T) / (dx * dx)
    # Eigenvalues are always Float64 on CPU for accuracy
    poisson_eigenvalues = zeros(Float64, N, N)
    for l in 1:N, k in 1:N
        poisson_eigenvalues[k, l] = 2.0 * Float64(inv_dx2) * (cos(π * Float64(k - 1) / Float64(N - 1)) - 1.0) +
                                     2.0 * Float64(inv_dx2) * (cos(π * Float64(l - 1) / Float64(N - 1)) - 1.0)
    end

    # Scheme-specific work arrays
    if time_scheme == :implicit
        rhs_u = KernelAbstractions.zeros(backend, T, N, N)
        rhs_v = KernelAbstractions.zeros(backend, T, N, N)
        # Pre-allocate CG workspaces for Helmholtz (fallback when DCT not available)
        n2 = N * N
        S = typeof(KernelAbstractions.zeros(backend, T, n2))
        helmholtz_solver_u = CgWorkspace(n2, n2, S)
        helmholtz_solver_v = CgWorkspace(n2, n2, S)
        work_b = KernelAbstractions.zeros(backend, T, n2)
    else
        lap_u = KernelAbstractions.zeros(backend, T, N, N)
        lap_v = KernelAbstractions.zeros(backend, T, N, N)
    end

    # Timestep selection
    if time_scheme == :implicit
        # No viscous stability restriction, only advective CFL
        dt = T(cfl) * dx
    else
        dt_adv = T(cfl) * dx  # CFL with max velocity ≈ 1
        dt_vis = T(cfl) * dx^2 / ν  # Viscous stability
        dt = min(dt_adv, dt_vis)
    end

    # Work array for convergence check
    u_old = similar(u)

    converged = false
    for step in 1:max_steps
        copyto!(u_old, u)

        if time_scheme == :implicit
            projection_step_implicit!(u, v, p, ν, dx, dt, N;
                                       adv_u=adv_u, adv_v=adv_v,
                                       rhs_u=rhs_u, rhs_v=rhs_v,
                                       div_field=div_field, gx=gx, gy=gy,
                                       helmholtz_solver_u=helmholtz_solver_u,
                                       helmholtz_solver_v=helmholtz_solver_v,
                                       work_b=work_b,
                                       poisson_eigenvalues=poisson_eigenvalues)
        else
            projection_step!(u, v, p, ν, dx, dt, N;
                             adv_u=adv_u, adv_v=adv_v,
                             lap_u=lap_u, lap_v=lap_v,
                             div_field=div_field, gx=gx, gy=gy,
                             poisson_solver=poisson_solver,
                             poisson_eigenvalues=poisson_eigenvalues)
        end

        # Check convergence (zero-allocation)
        u_old .-= u
        max_change = maximum(abs, u_old)
        if max_change < tol
            if verbose
                println("Converged at step $step (max_change = $max_change)")
            end
            converged = true
            break
        end

        if verbose && step % 1000 == 0
            println("Step $step: max_change = $max_change, dt = $dt")
        end
    end

    return u, v, p, converged
end
