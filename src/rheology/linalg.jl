# --- GPU-optimized 2×2 symmetric matrix operations ---
#
# All operations are @inline and branchless for efficient GPU execution.
# Used by log-conformation viscoelastic kernels.
#
# A symmetric 2×2 matrix is stored as (a11, a12, a22).

"""
    eigen_sym2x2(a11, a12, a22) → (λ1, λ2, e1x, e1y, e2x, e2y)

Analytical eigendecomposition of a symmetric 2×2 matrix.
Returns eigenvalues (λ1 ≥ λ2) and orthonormal eigenvectors.
Branchless: uses atan2 for robust angle computation.
"""
@inline function eigen_sym2x2(a11, a12, a22)
    T = typeof(a11)

    tr = a11 + a22
    diff = a11 - a22
    disc = sqrt(diff * diff + T(4) * a12 * a12)

    λ1 = (tr + disc) / T(2)
    λ2 = (tr - disc) / T(2)

    # Eigenvector angle via atan2 (robust, branchless)
    θ = atan(T(2) * a12, diff) / T(2)
    c = cos(θ)
    s = sin(θ)

    # Eigenvectors: e1 = (c, s), e2 = (-s, c)
    return λ1, λ2, c, s, -s, c
end

"""
    mat_exp_sym2x2(a11, a12, a22) → (e11, e12, e22)

Matrix exponential of a symmetric 2×2 matrix via eigendecomposition:
exp(A) = R · diag(exp(λ1), exp(λ2)) · Rᵀ
"""
@inline function mat_exp_sym2x2(a11, a12, a22)
    λ1, λ2, e1x, e1y, e2x, e2y = eigen_sym2x2(a11, a12, a22)

    eλ1 = exp(λ1)
    eλ2 = exp(λ2)

    # R · diag(eλ) · Rᵀ
    e11 = e1x * e1x * eλ1 + e2x * e2x * eλ2
    e12 = e1x * e1y * eλ1 + e2x * e2y * eλ2
    e22 = e1y * e1y * eλ1 + e2y * e2y * eλ2

    return e11, e12, e22
end

"""
    mat_log_sym2x2(a11, a12, a22) → (l11, l12, l22)

Matrix logarithm of a symmetric positive-definite 2×2 matrix:
log(A) = R · diag(log(λ1), log(λ2)) · Rᵀ

Eigenvalues are clamped to avoid log(0).
"""
@inline function mat_log_sym2x2(a11, a12, a22)
    T = typeof(a11)
    λ1, λ2, e1x, e1y, e2x, e2y = eigen_sym2x2(a11, a12, a22)

    lλ1 = log(max(λ1, T(1e-30)))
    lλ2 = log(max(λ2, T(1e-30)))

    l11 = e1x * e1x * lλ1 + e2x * e2x * lλ2
    l12 = e1x * e1y * lλ1 + e2x * e2y * lλ2
    l22 = e1y * e1y * lλ1 + e2y * e2y * lλ2

    return l11, l12, l22
end

"""
    decompose_velocity_gradient(dudx, dudy, dvdx, dvdy,
                                 e1x, e1y, e2x, e2y, λ1, λ2)
        → (Omega12, B11, B22)

Decompose the velocity gradient `(∇u)ᵀ` (transpose convention) in the
eigenbasis of `Θ = log(C)` according to Fattal & Kupferman (2004, §2.3).

The transpose of the velocity gradient is `L = (∇u)ᵀ`, with components
`L_ij = ∂u_i/∂x_j`. Project into the eigenbasis: `M = Rᵀ L R`.

The decomposition `M = Ω + B + N` gives:
- `B` : pure-extension part **commuting** with C — diag in eigenbasis
       (`B11 = M11`, `B22 = M22`, off-diag = 0)
- `Ω` : the EFFECTIVE rotation that keeps C symmetric positive-definite,
       built from the symmetric off-diagonal `Msym12 = (M12 + M21)/2`
       and the eigenvalues:
         Omega12 = ((eλ2·M12 + eλ1·M21) / (eλ2 - eλ1))   when λ1 ≠ λ2
                 = (M12 - M21)/2                          when λ1 = λ2 (limit)
- `N` : the part that does NOT commute with C, absorbed into Ω so that
       Θ stays diagonal in its own eigenbasis at all times.

In the log-conformation equation:
   ∂Θ/∂t + u·∇Θ = (ΩΘ - ΘΩ) + 2B + (1/λ)·(C⁻¹ - I)
the term `2B` directly drives the diagonal eigenvalues, while
`ΩΘ - ΘΩ` rotates the eigenvectors. With λ_i = log(eigenvalue of C).

This function returns:
- `Omega12` : the (1,2) component of the antisymmetric Ω matrix
- `B11`, `B22` : diagonal entries of B (note B12 = 0 by construction)
"""
@inline function decompose_velocity_gradient(dudx, dudy, dvdx, dvdy,
                                              e1x, e1y, e2x, e2y, λ1, λ2)
    T = typeof(dudx)
    # L = (∇u)ᵀ → L_ij = ∂u_i/∂x_j
    # In our 2D notation u_x = ux, u_y = uy:
    #   L11 = ∂ux/∂x = dudx
    #   L12 = ∂ux/∂y = dudy
    #   L21 = ∂uy/∂x = dvdx
    #   L22 = ∂uy/∂y = dvdy
    # M = Rᵀ L R where R columns are eigenvectors (e1, e2).
    # M_ij = e_i · L · e_j
    M11 = e1x*(dudx*e1x + dudy*e1y) + e1y*(dvdx*e1x + dvdy*e1y)
    M12 = e1x*(dudx*e2x + dudy*e2y) + e1y*(dvdx*e2x + dvdy*e2y)
    M21 = e2x*(dudx*e1x + dudy*e1y) + e2y*(dvdx*e1x + dvdy*e1y)
    M22 = e2x*(dudx*e2x + dudy*e2y) + e2y*(dvdx*e2x + dvdy*e2y)

    # Pure-extension part (commutes with C, diagonal in eigenbasis)
    B11 = M11
    B22 = M22

    # Effective rotation Ω that keeps Θ diagonal in its eigenbasis.
    # When λ1 ≠ λ2 (Fattal & Kupferman 2004, Eq. 2.6):
    #     Ω12 = (eλ2·M12 + eλ1·M21) / (eλ2 - eλ1)
    # When λ1 → λ2: degenerate limit reduces to pure rotation:
    #     Ω12 = (M12 - M21)/2
    eλ1 = exp(λ1)
    eλ2 = exp(λ2)
    Δ = eλ2 - eλ1
    pure_rot = (M12 - M21) / T(2)
    fk_rot   = (eλ2 * M12 + eλ1 * M21) / Δ
    # Switch to pure rotation when eigenvalues are nearly degenerate
    Omega12 = ifelse(abs(Δ) < T(1e-12), pure_rot, fk_rot)

    return Omega12, B11, B22
end
