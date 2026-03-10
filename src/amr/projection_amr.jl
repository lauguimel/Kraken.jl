# ---------------------------------------------------------------------------
# Navier-Stokes projection method on QuadTree AMR
# ---------------------------------------------------------------------------

"""
    _is_boundary(level, i, j)

Check if cell (level, i, j) is at any domain boundary.
"""
function _is_boundary(level::Int, i::Int, j::Int)
    n = 2^level
    return i == 1 || i == n || j == 1 || j == n
end

"""
    apply_velocity_bc_amr!(tree, u_field, v_field)

Apply lid-driven cavity boundary conditions on quadtree velocity fields.

- Left/right/bottom walls: u=0, v=0
- Top wall (lid): u=1, v=0
- Corner cells get u=0, v=0 (walls dominate over lid)
"""
function apply_velocity_bc_amr!(tree::QuadTree{T}, u_field::Symbol, v_field::Symbol) where {T}
    foreach_leaf(tree) do level, i, j
        n = 2^level
        is_left   = (i == 1)
        is_right  = (i == n)
        is_bottom = (j == 1)
        is_top    = (j == n)

        if is_left || is_right || is_bottom || is_top
            if is_top && !is_left && !is_right && !is_bottom
                # Pure top wall (lid): u=1, v=0
                set_field!(tree, u_field, level, i, j, one(T))
                set_field!(tree, v_field, level, i, j, zero(T))
            else
                # All other boundary cells (including corners): u=0, v=0
                set_field!(tree, u_field, level, i, j, zero(T))
                set_field!(tree, v_field, level, i, j, zero(T))
            end
        end
    end
    nothing
end

"""
    projection_step_amr!(tree, ν, dt; n_vcycles=20, n_smooth=4, poisson_rtol=1e-6)

One timestep of explicit Chorin projection on a quadtree AMR grid.

Steps:
1. Compute advection (upwind)
2. Compute diffusion (Laplacian)
3. Update intermediate velocity (skip boundary cells)
4. Apply velocity BCs
5. Compute divergence / dt → RHS for Poisson
6. Solve Poisson for pressure
7. Compute pressure gradient
8. Correct velocity (skip boundary cells)
9. Apply velocity BCs
"""
function projection_step_amr!(tree::QuadTree{T}, ν::T, dt::T;
                               n_vcycles::Int=20, n_smooth::Int=4,
                               poisson_rtol::T=T(1e-6)) where {T}

    # Step 1: Advection
    advect_amr!(tree, :adv_u, :u, :v, :u)
    advect_amr!(tree, :adv_v, :u, :v, :v)

    # Step 2: Diffusion
    laplacian_amr!(tree, :lap_u, :u)
    laplacian_amr!(tree, :lap_v, :v)

    # Step 3: Update intermediate velocity (interior cells only)
    foreach_leaf(tree) do level, i, j
        if !_is_boundary(level, i, j)
            u_val = get_field(tree, :u, level, i, j)
            v_val = get_field(tree, :v, level, i, j)
            adv_u = get_field(tree, :adv_u, level, i, j)
            adv_v = get_field(tree, :adv_v, level, i, j)
            lap_u = get_field(tree, :lap_u, level, i, j)
            lap_v = get_field(tree, :lap_v, level, i, j)
            set_field!(tree, :u, level, i, j, u_val + dt * (-adv_u + ν * lap_u))
            set_field!(tree, :v, level, i, j, v_val + dt * (-adv_v + ν * lap_v))
        end
    end

    # Step 4: Apply velocity BCs
    apply_velocity_bc_amr!(tree, :u, :v)

    # Step 5: Compute divergence → divide by dt → copy to :rhs
    divergence_amr!(tree, :div, :u, :v)
    foreach_leaf(tree) do level, i, j
        d = get_field(tree, :div, level, i, j)
        set_field!(tree, :rhs, level, i, j, d / dt)
    end

    # Step 6: Solve Poisson ∇²p = rhs
    solve_poisson_amr!(tree, :p, :rhs; n_vcycles=n_vcycles, n_smooth=n_smooth, rtol=poisson_rtol)

    # Step 7: Compute pressure gradient
    gradient_amr!(tree, :gx, :gy, :p)

    # Step 8: Correct velocity (interior cells only)
    foreach_leaf(tree) do level, i, j
        if !_is_boundary(level, i, j)
            u_val = get_field(tree, :u, level, i, j)
            v_val = get_field(tree, :v, level, i, j)
            gx_val = get_field(tree, :gx, level, i, j)
            gy_val = get_field(tree, :gy, level, i, j)
            set_field!(tree, :u, level, i, j, u_val - dt * gx_val)
            set_field!(tree, :v, level, i, j, v_val - dt * gy_val)
        end
    end

    # Step 9: Apply velocity BCs
    apply_velocity_bc_amr!(tree, :u, :v)

    nothing
end

"""
    run_cavity_amr(; N=64, Re=100.0, cfl=0.2, max_steps=20000, tol=1e-6,
                     verbose=false, adapt_interval=0, adapt_field=:u,
                     adapt_threshold=0.1)

Run the lid-driven cavity benchmark on a quadtree AMR grid using explicit
Chorin projection.

# Returns
- `(tree, converged)`: the quadtree with solution fields and convergence flag.
"""
function run_cavity_amr(; N::Int=64, Re::Float64=100.0, cfl::Float64=0.2,
                          max_steps::Int=20000, tol::Float64=1e-6,
                          verbose::Bool=false, adapt_interval::Int=0,
                          adapt_field::Symbol=:u, adapt_threshold::Float64=0.1)
    T = Float64
    max_level = round(Int, log2(N))
    @assert 2^max_level == N "N must be a power of 2, got $N"

    # Create tree and refine uniformly
    tree = QuadTree(T, max_level)
    refine_uniformly!(tree, max_level)

    # Add all fields
    for name in (:u, :v, :p, :adv_u, :adv_v, :lap_u, :lap_v, :div, :gx, :gy, :rhs, :res)
        add_field!(tree, name)
    end

    # Physics
    ν = T(1.0 / Re)
    h_min = cell_size(tree, max_level)

    # Timestep: min of advective and viscous CFL
    dt_adv = T(cfl) * h_min
    dt_vis = T(cfl) * h_min^2 / ν
    dt = min(dt_adv, dt_vis)

    # Apply initial BCs
    apply_velocity_bc_amr!(tree, :u, :v)

    converged = false
    for step in 1:max_steps
        # Save old u for convergence check
        old_u = Dict{Tuple{Int,Int,Int}, T}()
        foreach_leaf(tree) do level, i, j
            old_u[(level, i, j)] = get_field(tree, :u, level, i, j)
        end

        # Projection step
        projection_step_amr!(tree, ν, dt)

        # Adaptive refinement
        if adapt_interval > 0 && step % adapt_interval == 0
            adapt!(tree; field=adapt_field, threshold=T(adapt_threshold))
            # Reapply BCs after adaptation
            apply_velocity_bc_amr!(tree, :u, :v)
            # Recompute dt based on finest level
            finest = maximum(l for (l, _, _) in tree.leaf_list)
            h_min_new = cell_size(tree, finest)
            dt_adv = T(cfl) * h_min_new
            dt_vis = T(cfl) * h_min_new^2 / ν
            dt = min(dt_adv, dt_vis)
        end

        # Check convergence: max |u_new - u_old|
        max_change = zero(T)
        foreach_leaf(tree) do level, i, j
            key = (level, i, j)
            if haskey(old_u, key)
                diff = abs(get_field(tree, :u, level, i, j) - old_u[key])
                max_change = max(max_change, diff)
            end
        end

        if max_change < tol
            if verbose
                println("AMR cavity converged at step $step (max_change = $max_change)")
            end
            converged = true
            break
        end

        if verbose && step % 1000 == 0
            println("AMR cavity step $step: max_change = $max_change, dt = $dt")
        end
    end

    return (tree, converged)
end
