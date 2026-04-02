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
    decompose_velocity_gradient(dudx, dudy, dvdx, dvdy, e1x, e1y, e2x, e2y)
        → (Ω12, B11, B22)

Decompose the velocity gradient ∇u in the eigenvector basis of C (or exp(Θ)):
- Ω: antisymmetric part (rotation)
- B: symmetric part with zero diagonal in eigenbasis (extension rates)

Used in the log-conformation evolution equation:
∂Θ/∂t + u·∇Θ = Ω·Θ - Θ·Ω + 2B + (1/λ)(e^{-Θ} - I)

The velocity gradient L = ∇u is decomposed as:
M = Rᵀ · L · R  (project into eigenbasis of C)
Ω12 = (M12 - M21) / 2  (antisymmetric = rotation)
B11 = M11, B22 = M22    (diagonal = extension rates)
"""
@inline function decompose_velocity_gradient(dudx, dudy, dvdx, dvdy,
                                              e1x, e1y, e2x, e2y)
    # M = Rᵀ · L · R where R = [e1 | e2]
    # L = [dudx dudy; dvdx dvdy]
    M11 = e1x * (dudx * e1x + dudy * e1y) + e1y * (dvdx * e1x + dvdy * e1y)
    M12 = e1x * (dudx * e2x + dudy * e2y) + e1y * (dvdx * e2x + dvdy * e2y)
    M21 = e2x * (dudx * e1x + dudy * e1y) + e2y * (dvdx * e1x + dvdy * e1y)
    M22 = e2x * (dudx * e2x + dudy * e2y) + e2y * (dvdx * e2x + dvdy * e2y)

    Omega12 = (M12 - M21) / typeof(dudx)(2)
    B11 = M11
    B22 = M22

    return Omega12, B11, B22
end
