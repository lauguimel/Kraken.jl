# Iterative smoothing of VOF field for pressure-based two-phase LBM.
#
# The sharp VOF field C (from PLIC) is smoothed into C_s for use in:
#   - ρ(C_s) in collision and macroscopic (smooth density → stable streaming)
#   - ∇C_s in CSF force (smooth gradient → bounded F/ρ)
#
# C_s is NEVER advected — it's recomputed from C at each timestep.
# Mass conservation comes from C (PLIC), stability from C_s (smooth).
#
# Filter: 1-2-1 in each direction (separable), iterated n_passes times.
# One pass: σ ≈ 0.7 cells. Three passes: σ ≈ 1.2 cells → W_eff ≈ 3 cells.

using KernelAbstractions

@kernel function smooth_vof_pass_2d_kernel!(C_out, @Const(C_in), Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds begin
        T = eltype(C_in)
        ip = min(i + 1, Nx); im = max(i - 1, 1)
        jp = min(j + 1, Ny); jm = max(j - 1, 1)

        # Separable 1-2-1 filter (normalized: /4 in each direction → /16 total)
        C_out[i,j] = (     C_in[im,jm] + T(2)*C_in[i,jm] +     C_in[ip,jm]
                     + T(2)*C_in[im,j]  + T(4)*C_in[i,j]  + T(2)*C_in[ip,j]
                     +      C_in[im,jp] + T(2)*C_in[i,jp] +     C_in[ip,jp]) / T(16)
    end
end

"""
    smooth_vof_2d!(C_s, C, C_tmp; n_passes=3)

Smooth sharp VOF field C into C_s using iterative 1-2-1 filtering.

The result C_s has a smooth transition of effective width W ≈ √(n_passes) × 1.4
lattice units. Default n_passes=3 gives W ≈ 2.4 cells (similar to phase-field W=3).

C_tmp is a work array of the same size as C.
C_s and C_tmp may alias each other but must not alias C.
"""
function smooth_vof_2d!(C_s, C, C_tmp; n_passes=3)
    backend = KernelAbstractions.get_backend(C)
    Nx, Ny = size(C)
    kernel! = smooth_vof_pass_2d_kernel!(backend)

    # First pass: C → C_s
    kernel!(C_s, C, Int32(Nx), Int32(Ny); ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)

    # Subsequent passes: ping-pong between C_s and C_tmp
    for pass in 2:n_passes
        if pass % 2 == 0
            kernel!(C_tmp, C_s, Int32(Nx), Int32(Ny); ndrange=(Nx, Ny))
            KernelAbstractions.synchronize(backend)
        else
            kernel!(C_s, C_tmp, Int32(Nx), Int32(Ny); ndrange=(Nx, Ny))
            KernelAbstractions.synchronize(backend)
        end
    end

    # Ensure result is in C_s (if n_passes even, result is in C_tmp)
    if n_passes > 1 && n_passes % 2 == 0
        copyto!(C_s, C_tmp)
    end
end

# --- Mass correction: redistribute PLIC mass error to interface cells ---

@kernel function correct_mass_2d_kernel!(C, correction_per_cell, Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds begin
        T = eltype(C)
        c = C[i,j]
        # Only correct at interface cells (0 < C < 1)
        if c > T(1e-6) && c < one(T) - T(1e-6)
            C[i,j] = clamp(c + correction_per_cell, zero(T), one(T))
        end
    end
end

"""
    correct_mass_2d!(C, mass_ref)

Redistribute mass error uniformly across interface cells to enforce exact
mass conservation after PLIC advection.

Corrects ΔM = mass_ref - sum(C) by distributing equally to cells where 0 < C < 1.
"""
function correct_mass_2d!(C, mass_ref)
    backend = KernelAbstractions.get_backend(C)
    Nx, Ny = size(C)
    T = eltype(C)

    mass_current = sum(C)
    δm = T(mass_ref) - mass_current

    # Count interface cells (CPU reduction — fast for small arrays, GPU reduction for large)
    C_host = Array(C)
    n_interface = count(c -> c > 1e-6 && c < 1.0 - 1e-6, C_host)

    if n_interface > 0
        correction = T(δm / n_interface)
        kernel! = correct_mass_2d_kernel!(backend)
        kernel!(C, correction, Int32(Nx), Int32(Ny); ndrange=(Nx, Ny))
        KernelAbstractions.synchronize(backend)
    end
end
