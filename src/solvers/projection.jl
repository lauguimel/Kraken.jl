using Krylov
using LinearAlgebra

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

function LinearAlgebra.mul!(y::AbstractVector, op::NeumannLaplacianOperator, x::AbstractVector)
    N = op.N
    inv_dx2 = one(op.dx) / (op.dx * op.dx)
    X = reshape(x, N, N)
    Y = reshape(y, N, N)

    @inbounds for j in 1:N, i in 1:N
        # Neumann BC: ghost values = interior neighbor (zero gradient)
        xim = i > 1 ? X[i-1, j] : X[i+1, j]
        xip = i < N ? X[i+1, j] : X[i-1, j]
        xjm = j > 1 ? X[i, j-1] : X[i, j+1]
        xjp = j < N ? X[i, j+1] : X[i, j-1]

        Y[i, j] = (4 * X[i, j] - xim - xip - xjm - xjp) * inv_dx2
    end

    # Pin first element to remove null space
    Y[1, 1] += X[1, 1] * inv_dx2

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

# Returns
- `(phi, niter)`: solution and iteration count
"""
function solve_poisson_neumann!(phi, rhs, dx; maxiter=2000, rtol=1e-8)
    N = size(rhs, 1)
    T = eltype(rhs)

    # Negate for SPD: -∇²φ = -rhs
    b = vec(-copy(rhs))

    A = NeumannLaplacianOperator{T}(N, dx)
    M = NeumannJacobiPreconditioner{T}(T(dx * dx / 4))

    (x, stats) = cg(A, b; M=M, ldiv=false, itmax=maxiter, rtol=rtol, atol=zero(T))

    phi .= reshape(x, N, N)
    # Ensure pin: shift so phi[1,1] = 0
    phi .-= phi[1, 1]

    return phi, stats.niter
end

"""
    apply_velocity_bc!(u, v, N)

Apply boundary conditions for the lid-driven cavity:
- Top wall (j=N): u=1, v=0
- Bottom wall (j=1): u=0, v=0
- Left wall (i=1): u=0, v=0
- Right wall (i=N): u=0, v=0
"""
function apply_velocity_bc!(u, v, N)
    # Bottom wall (j=1)
    @inbounds for i in 1:N
        u[i, 1] = zero(eltype(u))
        v[i, 1] = zero(eltype(v))
    end
    # Top wall (j=N): lid moves with u=1
    @inbounds for i in 1:N
        u[i, N] = one(eltype(u))
        v[i, N] = zero(eltype(v))
    end
    # Left wall (i=1)
    @inbounds for j in 1:N
        u[1, j] = zero(eltype(u))
        v[1, j] = zero(eltype(v))
    end
    # Right wall (i=N)
    @inbounds for j in 1:N
        u[N, j] = zero(eltype(u))
        v[N, j] = zero(eltype(v))
    end
    return u, v
end

"""
    apply_pressure_neumann_bc!(p, N)

Apply homogeneous Neumann BCs for pressure: ∂p/∂n = 0 at all walls.
Implemented by copying the nearest interior value to the boundary.
"""
function apply_pressure_neumann_bc!(p, N)
    @inbounds for i in 1:N
        p[i, 1] = p[i, 2]       # bottom
        p[i, N] = p[i, N-1]     # top
    end
    @inbounds for j in 1:N
        p[1, j] = p[2, j]       # left
        p[N, j] = p[N-1, j]     # right
    end
    return p
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

# Arguments
- `u, v`: velocity components (N × N), modified in-place
- `p`: pressure field (N × N), modified in-place
- `ν`: kinematic viscosity (= 1/Re)
- `dx`: grid spacing
- `dt`: timestep
- `N`: grid size

# Keyword Arguments
- Work arrays: `adv_u, adv_v, lap_u, lap_v, div_field, gx, gy`
"""
function projection_step!(u, v, p, ν, dx, dt, N;
                          adv_u=similar(u), adv_v=similar(v),
                          lap_u=similar(u), lap_v=similar(v),
                          div_field=similar(u), gx=similar(u), gy=similar(v))
    T = eltype(u)

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

    # u* = u + dt * (-advection + ν * laplacian)
    @inbounds for j in 2:N-1, i in 2:N-1
        u[i, j] = u[i, j] + dt * (-adv_u[i, j] + ν * lap_u[i, j])
        v[i, j] = v[i, j] + dt * (-adv_v[i, j] + ν * lap_v[i, j])
    end

    # Apply velocity BCs to intermediate velocity
    apply_velocity_bc!(u, v, N)

    # --- Step 2: pressure Poisson equation ---
    # ∇²p = (1/dt) * ∇·u*
    fill!(div_field, zero(T))
    divergence!(div_field, u, v, dx)

    # RHS = (1/dt) * div(u*)
    rhs = div_field ./ dt

    solve_poisson_neumann!(p, rhs, dx; maxiter=2000, rtol=T(1e-6))
    apply_pressure_neumann_bc!(p, N)

    # --- Step 3: velocity correction ---
    fill!(gx, zero(T))
    fill!(gy, zero(T))
    gradient!(gx, gy, p, dx)

    @inbounds for j in 2:N-1, i in 2:N-1
        u[i, j] = u[i, j] - dt * gx[i, j]
        v[i, j] = v[i, j] - dt * gy[i, j]
    end

    # Apply velocity BCs after correction
    apply_velocity_bc!(u, v, N)

    return u, v, p
end

"""
    run_cavity(; N=64, Re=100.0, cfl=0.2, max_steps=10000, tol=1e-6, verbose=false)

Run the lid-driven cavity benchmark at given Reynolds number on an N×N grid.

Uses Chorin's projection method with explicit Euler time stepping. The simulation
runs until steady state (velocity change < `tol`) or `max_steps` is reached.

# Keyword Arguments
- `N::Int`: grid points per dimension (default: 64)
- `Re::Float64`: Reynolds number (default: 100)
- `cfl::Float64`: CFL number for timestep selection (default: 0.2)
- `max_steps::Int`: maximum number of timesteps (default: 10000)
- `tol::Float64`: convergence tolerance on max velocity change (default: 1e-6)
- `verbose::Bool`: print progress every 1000 steps (default: false)

# Returns
- `(u, v, p, converged)`: velocity components, pressure, and convergence flag
"""
function run_cavity(; N=64, Re=100.0, cfl=0.2, max_steps=10000, tol=1e-6, verbose=false)
    T = Float64
    dx = T(1.0) / T(N - 1)
    ν = T(1.0) / T(Re)

    # Initialize fields
    u = zeros(T, N, N)
    v = zeros(T, N, N)
    p = zeros(T, N, N)

    # Apply initial BCs
    apply_velocity_bc!(u, v, N)

    # Preallocate work arrays
    adv_u = zeros(T, N, N)
    adv_v = zeros(T, N, N)
    lap_u = zeros(T, N, N)
    lap_v = zeros(T, N, N)
    div_field = zeros(T, N, N)
    gx = zeros(T, N, N)
    gy = zeros(T, N, N)

    # Timestep from CFL and viscous stability
    dt_adv = cfl * dx  # CFL with max velocity ≈ 1
    dt_vis = cfl * dx^2 / ν  # Viscous stability
    dt = min(dt_adv, dt_vis)

    converged = false
    for step in 1:max_steps
        u_old = copy(u)

        projection_step!(u, v, p, ν, dx, dt, N;
                         adv_u=adv_u, adv_v=adv_v,
                         lap_u=lap_u, lap_v=lap_v,
                         div_field=div_field, gx=gx, gy=gy)

        # Check convergence
        max_change = maximum(abs.(u .- u_old))
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
