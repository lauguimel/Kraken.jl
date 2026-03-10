"""
    Multigrid Poisson Solver on Quadtree AMR

V-cycle multigrid for solving ∇²φ = rhs on a QuadTree with Neumann BCs.
The quadtree level hierarchy IS the multigrid hierarchy.
CPU-only implementation.

Key insight: on a uniform grid at level L, leaves exist ONLY at level L.
The multigrid uses the underlying grid arrays at levels 0..L as the coarse
grid hierarchy, even though those cells are not active leaves.
"""

# ---------------------------------------------------------------------------
# Helper: refine uniformly to a target level
# ---------------------------------------------------------------------------

"""
    refine_uniformly!(tree, target_level)

Refine all leaf cells until every leaf is at `target_level`.
"""
function refine_uniformly!(tree::QuadTree{T}, target_level::Int) where {T}
    @assert target_level <= tree.max_level
    for lvl in 0:(target_level - 1)
        leaves = [(l, i, j) for (l, i, j) in tree.leaf_list if l == lvl]
        for (l, i, j) in leaves
            if tree.active[l+1][i, j]
                refine!(tree, l, i, j)
            end
        end
    end
    nothing
end

# ---------------------------------------------------------------------------
# Helper: get neighbor value on a full grid at a given level
# ---------------------------------------------------------------------------

"""
    _grid_neighbor_val(data, level_data, n, i, j, dir)

Get neighbor value on a full n×n grid with Neumann BCs (mirror at boundary).
"""
function _grid_neighbor_val(level_data::Matrix{T}, n::Int, i::Int, j::Int, dir::Symbol) where {T}
    if dir == :left
        ni, nj = max(i - 1, 1), j
    elseif dir == :right
        ni, nj = min(i + 1, n), j
    elseif dir == :bottom
        ni, nj = i, max(j - 1, 1)
    elseif dir == :top
        ni, nj = i, min(j + 1, n)
    else
        error("Invalid direction: $dir")
    end
    return level_data[ni, nj]
end

# ---------------------------------------------------------------------------
# Smooth: weighted Jacobi on a full grid at given level
# ---------------------------------------------------------------------------

"""
    _smooth_grid!(phi_data, rhs_data, n, h, omega, iterations)

Weighted Jacobi smoothing on a full n×n grid with Neumann BCs.
"""
function _smooth_grid!(phi_data::Matrix{T}, rhs_data::Matrix{T},
                        n::Int, h::T, omega::T, iterations::Int) where {T}
    h2 = h * h
    tmp = similar(phi_data)

    for _ in 1:iterations
        for j in 1:n, i in 1:n
            phi_l = i > 1 ? phi_data[i-1, j] : phi_data[i, j]
            phi_r = i < n ? phi_data[i+1, j] : phi_data[i, j]
            phi_b = j > 1 ? phi_data[i, j-1] : phi_data[i, j]
            phi_t = j < n ? phi_data[i, j+1] : phi_data[i, j]
            phi_gs = (phi_l + phi_r + phi_b + phi_t - h2 * rhs_data[i, j]) / T(4)
            tmp[i, j] = (one(T) - omega) * phi_data[i, j] + omega * phi_gs
        end
        copyto!(phi_data, tmp)
    end
    nothing
end

# ---------------------------------------------------------------------------
# Residual on a full grid
# ---------------------------------------------------------------------------

"""
    _compute_residual_grid!(res_data, phi_data, rhs_data, n, h)

Compute res = rhs - ∇²φ on a full n×n grid with Neumann BCs.
"""
function _compute_residual_grid!(res_data::Matrix{T}, phi_data::Matrix{T},
                                  rhs_data::Matrix{T}, n::Int, h::T) where {T}
    h2 = h * h
    for j in 1:n, i in 1:n
        phi_c = phi_data[i, j]
        phi_l = i > 1 ? phi_data[i-1, j] : phi_c
        phi_r = i < n ? phi_data[i+1, j] : phi_c
        phi_b = j > 1 ? phi_data[i, j-1] : phi_c
        phi_t = j < n ? phi_data[i, j+1] : phi_c
        lap = (phi_l + phi_r + phi_b + phi_t - T(4) * phi_c) / h2
        res_data[i, j] = rhs_data[i, j] - lap
    end
    nothing
end

# ---------------------------------------------------------------------------
# Restrict: fine grid → coarse grid (full-weighting)
# ---------------------------------------------------------------------------

"""
    _restrict_grid!(coarse, fine, nc, nf)

Full-weighting restriction from nf×nf fine grid to nc×nc coarse grid.
nc = nf/2, cell-centered.
"""
function _restrict_grid!(coarse::Matrix{T}, fine::Matrix{T}, nc::Int, nf::Int) where {T}
    for pj in 1:nc, pi in 1:nc
        ci, cj = 2 * pi - 1, 2 * pj - 1
        # Average of the 4 fine cells that map to this coarse cell
        coarse[pi, pj] = (fine[ci, cj] + fine[ci+1, cj] +
                          fine[ci, cj+1] + fine[ci+1, cj+1]) / T(4)
    end
    nothing
end

# ---------------------------------------------------------------------------
# Prolongate: coarse grid → fine grid (bilinear injection)
# ---------------------------------------------------------------------------

"""
    _prolongate_grid!(fine, coarse, nc, nf)

Prolongate correction from nc×nc coarse grid to nf×nf fine grid.
Uses injection: each coarse cell value is added to its 4 fine children.
"""
function _prolongate_grid!(fine::Matrix{T}, coarse::Matrix{T}, nc::Int, nf::Int) where {T}
    for pj in 1:nc, pi in 1:nc
        ci, cj = 2 * pi - 1, 2 * pj - 1
        fine[ci,   cj]   += coarse[pi, pj]
        fine[ci+1, cj]   += coarse[pi, pj]
        fine[ci,   cj+1] += coarse[pi, pj]
        fine[ci+1, cj+1] += coarse[pi, pj]
    end
    nothing
end

# ---------------------------------------------------------------------------
# V-cycle on the grid hierarchy
# ---------------------------------------------------------------------------

"""
    vcycle_amr!(tree, phi, rhs, res, n_smooth)

One V-cycle of multigrid using the quadtree level hierarchy.
Works on the raw grid data arrays at each level.

For a uniform grid at level L, this is a standard geometric V-cycle
using levels L, L-1, ..., 0 as the multigrid hierarchy.
"""
function vcycle_amr!(tree::QuadTree{T}, phi::Symbol, rhs::Symbol,
                      res::Symbol, n_smooth::Int) where {T}
    # Find the finest level with leaves
    max_level = 0
    for (l, _, _) in tree.leaf_list
        max_level = max(max_level, l)
    end

    if max_level == 0
        # Only root cell, nothing to do
        return nothing
    end

    omega = T(2) / T(3)

    # Access raw data arrays
    phi_arrays = tree.fields[phi]
    rhs_arrays = tree.fields[rhs]
    res_arrays = tree.fields[res]

    # Ensure coarse level data is initialized from fine levels
    # For a uniform grid, we need to restrict phi and rhs down to coarse levels
    # to initialize the MG hierarchy properly.

    # Step 1: Pre-smooth at finest level
    n = 2^max_level
    h = cell_size(tree, max_level)
    _smooth_grid!(phi_arrays[max_level+1], rhs_arrays[max_level+1], n, h, omega, n_smooth)

    # Step 2: Compute residual at finest level
    _compute_residual_grid!(res_arrays[max_level+1], phi_arrays[max_level+1],
                            rhs_arrays[max_level+1], n, h)

    # Step 3: Downward pass — restrict residual, solve error equation on coarse grids
    # The error equation: ∇²e = r, where e is the correction and r is the residual.
    # At coarse levels, we store the correction in phi_arrays and the restricted
    # residual in rhs_arrays. We need to save/restore the original rhs.

    # Use res as the restricted residual carrier going down.
    # We need a correction field at each coarse level.
    # Simple approach: zero out coarse phi (it's the correction), use restricted residual as rhs.

    # Save coarse-level rhs (will be overwritten by restricted residual)
    saved_rhs = Vector{Matrix{T}}(undef, max_level)
    for k in 0:(max_level - 1)
        saved_rhs[k+1] = copy(rhs_arrays[k+1])
    end

    # Downward pass
    for level in (max_level - 1):-1:0
        n_fine = 2^(level + 1)
        n_coarse = 2^level
        h_coarse = cell_size(tree, level)

        # Restrict residual from level+1 to level
        _restrict_grid!(rhs_arrays[level+1], res_arrays[level+2], n_coarse, n_fine)

        # Zero the correction at this level
        fill!(phi_arrays[level+1], zero(T))

        if level > 0
            # Smooth the error equation: ∇²e = r_restricted
            _smooth_grid!(phi_arrays[level+1], rhs_arrays[level+1],
                          n_coarse, h_coarse, omega, n_smooth)

            # Compute residual of the error equation at this level
            _compute_residual_grid!(res_arrays[level+1], phi_arrays[level+1],
                                    rhs_arrays[level+1], n_coarse, h_coarse)
        else
            # Coarsest level: extensive smoothing
            _smooth_grid!(phi_arrays[1], rhs_arrays[1],
                          n_coarse, h_coarse, omega, 4 * n_smooth)
        end
    end

    # Upward pass: prolongate correction and smooth
    for level in 1:max_level
        n_fine = 2^level
        n_coarse = 2^(level - 1)
        h_fine = cell_size(tree, level)

        if level < max_level
            # Prolongate correction from coarser level and add to this level's correction
            _prolongate_grid!(phi_arrays[level+1], phi_arrays[level],
                              n_coarse, n_fine)

            # Smooth the error equation at this level
            # rhs at this level still contains the restricted residual
            _smooth_grid!(phi_arrays[level+1], rhs_arrays[level+1],
                          n_fine, h_fine, omega, n_smooth)
        else
            # Finest level: prolongate correction and add to solution
            _prolongate_grid!(phi_arrays[level+1], phi_arrays[level],
                              n_coarse, n_fine)

            # Post-smooth the original equation
            # Restore original rhs at finest level (it wasn't overwritten)
            _smooth_grid!(phi_arrays[level+1], rhs_arrays[level+1],
                          n_fine, h_fine, omega, n_smooth)
        end
    end

    # Restore coarse-level rhs
    for k in 0:(max_level - 1)
        copyto!(rhs_arrays[k+1], saved_rhs[k+1])
    end

    nothing
end

# ---------------------------------------------------------------------------
# Helper: get neighbor value for leaf-based operations
# ---------------------------------------------------------------------------

"""
    _get_neighbor_val(tree, name, level, i, j, dir)

Get the neighbor value for the Laplacian stencil on leaf cells.
Handles same-level, coarser, finer neighbors, and Neumann BCs.
"""
function _get_neighbor_val(tree::QuadTree{T}, name::Symbol, level::Int,
                           i::Int, j::Int, dir::Symbol) where {T}
    nb = find_neighbor(tree, level, i, j, dir)
    if nb === nothing
        return get_field(tree, name, level, i, j)
    elseif nb isa Tuple{Int,Int,Int}
        nb_level, ni, nj = nb
        return get_field(tree, name, nb_level, ni, nj)
    elseif nb isa Vector
        val = zero(T)
        count = 0
        for (nb_level, ni, nj) in nb
            val += get_field(tree, name, nb_level, ni, nj)
            count += 1
        end
        return count > 0 ? val / T(count) : get_field(tree, name, level, i, j)
    end
    return get_field(tree, name, level, i, j)
end

# ---------------------------------------------------------------------------
# Compute residual on ALL leaf cells
# ---------------------------------------------------------------------------

"""
    compute_residual_all!(tree, res, phi, rhs)

Compute res = rhs - ∇²φ on all leaf cells.
"""
function compute_residual_all!(tree::QuadTree{T}, res::Symbol, phi::Symbol,
                                rhs::Symbol) where {T}
    for (level, i, j) in tree.leaf_list
        h = cell_size(tree, level)
        h2 = h * h
        phi_c = get_field(tree, phi, level, i, j)
        phi_l = _get_neighbor_val(tree, phi, level, i, j, :left)
        phi_r = _get_neighbor_val(tree, phi, level, i, j, :right)
        phi_b = _get_neighbor_val(tree, phi, level, i, j, :bottom)
        phi_t = _get_neighbor_val(tree, phi, level, i, j, :top)
        lap = (phi_l + phi_r + phi_b + phi_t - T(4) * phi_c) / h2
        rhs_val = get_field(tree, rhs, level, i, j)
        set_field!(tree, res, level, i, j, rhs_val - lap)
    end
    nothing
end

# ---------------------------------------------------------------------------
# Compute residual at one level (leaf cells only)
# ---------------------------------------------------------------------------

"""
    compute_residual_level!(tree, res, phi, rhs, level)

Compute res = rhs - ∇²φ for leaf cells at `level`.
"""
function compute_residual_level!(tree::QuadTree{T}, res::Symbol, phi::Symbol,
                                  rhs::Symbol, level::Int) where {T}
    h = cell_size(tree, level)
    h2 = h * h
    for (l, i, j) in tree.leaf_list
        l == level || continue
        phi_c = get_field(tree, phi, level, i, j)
        phi_l = _get_neighbor_val(tree, phi, level, i, j, :left)
        phi_r = _get_neighbor_val(tree, phi, level, i, j, :right)
        phi_b = _get_neighbor_val(tree, phi, level, i, j, :bottom)
        phi_t = _get_neighbor_val(tree, phi, level, i, j, :top)
        lap = (phi_l + phi_r + phi_b + phi_t - T(4) * phi_c) / h2
        rhs_val = get_field(tree, rhs, level, i, j)
        set_field!(tree, res, level, i, j, rhs_val - lap)
    end
    nothing
end

# ---------------------------------------------------------------------------
# Smooth on leaf cells at one level (for leaf-based operations)
# ---------------------------------------------------------------------------

"""
    smooth_level!(tree, phi, rhs, level)

One pass of weighted Jacobi smoothing on leaf cells at `level`.
"""
function smooth_level!(tree::QuadTree{T}, phi::Symbol, rhs::Symbol, level::Int) where {T}
    h = cell_size(tree, level)
    h2 = h * h
    omega = T(2) / T(3)

    updates = Vector{Tuple{Int,Int,T}}()
    for (l, i, j) in tree.leaf_list
        l == level || continue
        phi_c = get_field(tree, phi, level, i, j)
        phi_l = _get_neighbor_val(tree, phi, level, i, j, :left)
        phi_r = _get_neighbor_val(tree, phi, level, i, j, :right)
        phi_b = _get_neighbor_val(tree, phi, level, i, j, :bottom)
        phi_t = _get_neighbor_val(tree, phi, level, i, j, :top)
        rhs_val = get_field(tree, rhs, level, i, j)
        phi_gs = (phi_l + phi_r + phi_b + phi_t - h2 * rhs_val) / T(4)
        phi_new = (one(T) - omega) * phi_c + omega * phi_gs
        push!(updates, (i, j, phi_new))
    end
    for (i, j, val) in updates
        set_field!(tree, phi, level, i, j, val)
    end
    nothing
end

# ---------------------------------------------------------------------------
# Restrict: fine level → coarse level (for leaf-based)
# ---------------------------------------------------------------------------

"""
    restrict_level!(tree, field, level)

Restrict `field` from cells at `level` to parent cells at `level-1`.
"""
function restrict_level!(tree::QuadTree{T}, field::Symbol, level::Int) where {T}
    level >= 1 || return
    n_parent = 2^(level - 1)
    for pi in 1:n_parent, pj in 1:n_parent
        ci, cj = 2 * pi - 1, 2 * pj - 1
        n_child = 2^level
        if ci + 1 <= n_child && cj + 1 <= n_child
            vals = zero(T)
            for di in 0:1, dj in 0:1
                vals += get_field(tree, field, level, ci + di, cj + dj)
            end
            set_field!(tree, field, level - 1, pi, pj, vals / T(4))
        end
    end
    nothing
end

# ---------------------------------------------------------------------------
# Prolongate: coarse level → fine level
# ---------------------------------------------------------------------------

"""
    prolongate_level!(tree, phi, level)

Prolongate correction from level-1 to cells at `level` (injection).
"""
function prolongate_level!(tree::QuadTree{T}, phi::Symbol, level::Int) where {T}
    level >= 1 || return
    n_parent = 2^(level - 1)
    for pj in 1:n_parent, pi in 1:n_parent
        ci, cj = 2 * pi - 1, 2 * pj - 1
        coarse_val = get_field(tree, phi, level - 1, pi, pj)
        for di in 0:1, dj in 0:1
            old = get_field(tree, phi, level, ci + di, cj + dj)
            set_field!(tree, phi, level, ci + di, cj + dj, old + coarse_val)
        end
    end
    nothing
end

# ---------------------------------------------------------------------------
# Residual norm
# ---------------------------------------------------------------------------

"""
    residual_norm(tree, res)

Compute L2 norm of residual field weighted by cell area.
"""
function residual_norm(tree::QuadTree{T}, res::Symbol) where {T}
    norm_sq = zero(T)
    for (level, i, j) in tree.leaf_list
        h = cell_size(tree, level)
        r = get_field(tree, res, level, i, j)
        norm_sq += r * r * h * h
    end
    return sqrt(norm_sq)
end

# ---------------------------------------------------------------------------
# Remove mean (for Neumann BC compatibility)
# ---------------------------------------------------------------------------

"""
    _remove_mean!(tree, phi)

Subtract the volume-weighted mean of `phi` to enforce uniqueness for Neumann BCs.
"""
function _remove_mean!(tree::QuadTree{T}, phi::Symbol) where {T}
    total_val = zero(T)
    total_area = zero(T)
    for (level, i, j) in tree.leaf_list
        h = cell_size(tree, level)
        area = h * h
        total_val += get_field(tree, phi, level, i, j) * area
        total_area += area
    end
    mean_val = total_val / total_area
    for (level, i, j) in tree.leaf_list
        old = get_field(tree, phi, level, i, j)
        set_field!(tree, phi, level, i, j, old - mean_val)
    end
    nothing
end

# ---------------------------------------------------------------------------
# Remove mean on a raw grid array
# ---------------------------------------------------------------------------

function _remove_mean_grid!(data::Matrix{T}, n::Int) where {T}
    s = zero(T)
    for j in 1:n, i in 1:n
        s += data[i, j]
    end
    m = s / T(n * n)
    for j in 1:n, i in 1:n
        data[i, j] -= m
    end
    nothing
end

# ---------------------------------------------------------------------------
# Top-level solver
# ---------------------------------------------------------------------------

"""
    solve_poisson_amr!(tree, phi, rhs; n_vcycles=10, n_smooth=4, rtol=1e-6, verbose=false)

Solve ∇²φ = rhs on the quadtree using V-cycle multigrid with Neumann BCs.

# Returns
- `(converged::Bool, n_iters::Int, final_residual::Float64)`
"""
function solve_poisson_amr!(tree::QuadTree{T}, phi::Symbol, rhs::Symbol;
                             n_vcycles::Int=10, n_smooth::Int=4,
                             rtol::T=T(1e-6), verbose::Bool=false) where {T}

    # Ensure :res field exists
    if !haskey(tree.fields, :res)
        add_field!(tree, :res)
    end

    # Ensure RHS has zero mean (compatibility condition for Neumann BC)
    _remove_mean!(tree, rhs)

    # Initial residual
    compute_residual_all!(tree, :res, phi, rhs)
    r0 = residual_norm(tree, :res)
    if verbose
        println("AMR Poisson: initial residual = $r0")
    end

    if r0 < eps(T)
        return (true, 0, r0)
    end

    converged = false
    n_iter = 0

    for cycle in 1:n_vcycles
        vcycle_amr!(tree, phi, rhs, :res, n_smooth)

        # Remove mean to enforce uniqueness
        max_level = maximum(l for (l, _, _) in tree.leaf_list)
        _remove_mean_grid!(tree.fields[phi][max_level+1], 2^max_level)

        # Check convergence
        compute_residual_all!(tree, :res, phi, rhs)
        rn = residual_norm(tree, :res)
        n_iter = cycle

        if verbose
            println("  V-cycle $cycle: residual = $rn (ratio = $(rn/r0))")
        end

        if rn < rtol * r0
            converged = true
            break
        end
    end

    final_r = residual_norm(tree, :res)
    return (converged, n_iter, final_r)
end
