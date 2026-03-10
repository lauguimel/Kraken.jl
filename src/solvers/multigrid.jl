"""
    Geometric Multigrid Poisson Solver

V-cycle multigrid for solving ∇²φ = rhs with Neumann BCs on a 2D uniform grid.
CPU-only implementation using standard Julia arrays and broadcasting.
"""

"""
    apply_neumann_bc!(phi)

Apply homogeneous Neumann BCs by copying nearest interior values to boundaries.
"""
function apply_neumann_bc!(phi)
    N = size(phi, 1)
    phi[1, :]   .= @view phi[2, :]
    phi[N, :]   .= @view phi[N-1, :]
    phi[:, 1]   .= @view phi[:, 2]
    phi[:, N]   .= @view phi[:, N-1]
    return phi
end

"""
    restrict!(coarse, fine)

Full-weighting restriction from fine grid (2N-1 × 2N-1) to coarse grid (N × N).
Operates on interior points only; boundaries are set by Neumann BCs.
"""
function restrict!(coarse, fine)
    Nc = size(coarse, 1)
    for jc in 2:Nc-1
        jf = 2 * (jc - 1) + 1
        for ic in 2:Nc-1
            if_ = 2 * (ic - 1) + 1
            coarse[ic, jc] = (
                fine[if_, jf] * 0.25 +
                (fine[if_-1, jf] + fine[if_+1, jf] + fine[if_, jf-1] + fine[if_, jf+1]) * 0.125 +
                (fine[if_-1, jf-1] + fine[if_+1, jf-1] + fine[if_-1, jf+1] + fine[if_+1, jf+1]) * 0.0625
            )
        end
    end
    apply_neumann_bc!(coarse)
    return coarse
end

"""
    prolongate!(fine, coarse)

Bilinear prolongation from coarse grid (N × N) to fine grid (2N-1 × 2N-1).
Operates on interior points only; boundaries are set by Neumann BCs.
"""
function prolongate!(fine, coarse)
    Nc = size(coarse, 1)
    Nf = size(fine, 1)

    # Inject coarse points to fine grid
    for jc in 1:Nc
        jf = 2 * (jc - 1) + 1
        for ic in 1:Nc
            if_ = 2 * (ic - 1) + 1
            fine[if_, jf] = coarse[ic, jc]
        end
    end

    # Interpolate horizontally (odd rows, even columns)
    for jc in 1:Nc
        jf = 2 * (jc - 1) + 1
        for ic in 1:Nc-1
            if_ = 2 * (ic - 1) + 1
            fine[if_+1, jf] = 0.5 * (coarse[ic, jc] + coarse[ic+1, jc])
        end
    end

    # Interpolate vertically (even rows, all columns in fine)
    for jc in 1:Nc-1
        jf = 2 * (jc - 1) + 1
        for if_ in 1:Nf
            fine[if_, jf+1] = 0.5 * (fine[if_, jf] + fine[if_, jf+2])
        end
    end

    apply_neumann_bc!(fine)
    return fine
end

"""
    smooth_gs!(phi, rhs, dx, n_iter)

Red-Black Gauss-Seidel smoother for ∇²φ = rhs on interior points.
Much faster convergence than weighted Jacobi for Poisson problems.
"""
function smooth_gs!(phi, rhs, dx, n_iter)
    N = size(phi, 1)
    dx2 = dx * dx

    for _ in 1:n_iter
        # Red sweep (i+j even)
        for j in 2:N-1
            for i in 2:N-1
                if (i + j) % 2 == 0
                    neighbor_sum = phi[i-1, j] + phi[i+1, j] + phi[i, j-1] + phi[i, j+1]
                    phi[i, j] = 0.25 * (neighbor_sum - dx2 * rhs[i, j])
                end
            end
        end
        apply_neumann_bc!(phi)

        # Black sweep (i+j odd)
        for j in 2:N-1
            for i in 2:N-1
                if (i + j) % 2 == 1
                    neighbor_sum = phi[i-1, j] + phi[i+1, j] + phi[i, j-1] + phi[i, j+1]
                    phi[i, j] = 0.25 * (neighbor_sum - dx2 * rhs[i, j])
                end
            end
        end
        apply_neumann_bc!(phi)
    end
    return phi
end

"""
    compute_residual!(res, phi, rhs, dx)

Compute residual r = rhs - ∇²φ using 5-point stencil on interior points.
"""
function compute_residual!(res, phi, rhs, dx)
    N = size(phi, 1)
    inv_dx2 = 1.0 / (dx * dx)
    fill!(res, 0.0)
    for j in 2:N-1
        for i in 2:N-1
            lap = (phi[i-1, j] + phi[i+1, j] + phi[i, j-1] + phi[i, j+1] - 4.0 * phi[i, j]) * inv_dx2
            res[i, j] = rhs[i, j] - lap
        end
    end
    return res
end

"""
    vcycle!(phi, rhs, dx, levels; n_smooth=3)

Recursive V-cycle multigrid. At coarsest level (grid ≤ 4), do extra smoothing.
"""
function vcycle!(phi, rhs, dx, levels; n_smooth=3)
    N = size(phi, 1)

    # Base case: coarsest level — just smooth a lot
    if levels <= 1 || N <= 4
        smooth_gs!(phi, rhs, dx, 50)
        return phi
    end

    # Pre-smooth
    smooth_gs!(phi, rhs, dx, n_smooth)

    # Compute residual
    res = zeros(N, N)
    compute_residual!(res, phi, rhs, dx)

    # Restrict residual to coarse grid
    Nc = (N - 1) ÷ 2 + 1
    res_coarse = zeros(Nc, Nc)
    restrict!(res_coarse, res)

    # Solve on coarse grid (error equation: ∇²e = r)
    e_coarse = zeros(Nc, Nc)
    dx_coarse = 2.0 * dx
    vcycle!(e_coarse, res_coarse, dx_coarse, levels - 1; n_smooth=n_smooth)

    # Prolongate correction to fine grid and add
    e_fine = zeros(N, N)
    prolongate!(e_fine, e_coarse)
    for j in 2:N-1
        for i in 2:N-1
            phi[i, j] += e_fine[i, j]
        end
    end
    apply_neumann_bc!(phi)

    # Post-smooth
    smooth_gs!(phi, rhs, dx, n_smooth)

    return phi
end

"""
    solve_poisson_mg!(phi, rhs, dx; n_vcycles=100, n_smooth=3, rtol=1e-6)

Solve ∇²φ = rhs with Neumann BCs using geometric multigrid V-cycles.

The solution is unique up to a constant; we pin φ[1,1] = 0 to remove the null space.

# Arguments
- `phi`: output array (N × N), initial guess (modified in-place)
- `rhs`: right-hand side array (N × N)
- `dx`: uniform grid spacing

# Keyword Arguments
- `n_vcycles::Int`: maximum number of V-cycles (default: 100)
- `n_smooth::Int`: smoothing iterations per level (default: 3)
- `rtol`: relative tolerance on residual norm (default: 1e-6)

# Returns
- `(phi, n_vcycles_used)`: solution array and number of V-cycles performed.

See also: [`solve_poisson_neumann!`](@ref), [`solve_poisson_fft!`](@ref)
"""
function solve_poisson_mg!(phi, rhs, dx; n_vcycles=100, n_smooth=3, rtol=1e-6)
    N = size(phi, 1)

    # Determine number of multigrid levels from grid size
    # N must be 2^k + 1 for clean coarsening
    n = N - 1
    levels = 0
    temp = n
    while temp >= 2 && temp % 2 == 0
        levels += 1
        temp = temp ÷ 2
    end
    levels = max(levels, 1)

    # Ensure RHS compatibility for Neumann (zero mean on interior)
    rhs_mean = sum(rhs[2:N-1, 2:N-1]) / ((N-2)^2)
    rhs_work = copy(rhs)
    for j in 2:N-1
        for i in 2:N-1
            rhs_work[i, j] -= rhs_mean
        end
    end

    # Compute initial residual norm for relative tolerance
    res = zeros(N, N)
    compute_residual!(res, phi, rhs_work, dx)
    res_norm0 = sqrt(sum(res[2:N-1, 2:N-1] .^ 2))
    if res_norm0 == 0.0
        phi .-= phi[1, 1]
        return phi, 0
    end

    n_used = 0
    for cycle in 1:n_vcycles
        vcycle!(phi, rhs_work, dx, levels; n_smooth=n_smooth)

        # Remove mean to handle null space
        phi_mean = sum(phi[2:N-1, 2:N-1]) / ((N-2)^2)
        for j in 2:N-1
            for i in 2:N-1
                phi[i, j] -= phi_mean
            end
        end
        apply_neumann_bc!(phi)

        # Check convergence
        compute_residual!(res, phi, rhs_work, dx)
        res_norm = sqrt(sum(res[2:N-1, 2:N-1] .^ 2))
        n_used = cycle

        if res_norm / res_norm0 < rtol
            break
        end
    end

    # Pin phi[1,1] = 0 to remove null space
    phi .-= phi[1, 1]

    return phi, n_used
end
