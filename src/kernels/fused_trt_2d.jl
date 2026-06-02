using KernelAbstractions

# =====================================================================
# Fused TRT (Two-Relaxation-Time) kernel: stream + bounce-back + collide
# + macroscopic. Single kernel launch per timestep, same pattern as
# fused_bgk_step!.
#
# TRT splits each opposite-pair population (q, q̄) into symmetric and
# antisymmetric modes:
#   f_s = (f_q + f_q̄) / 2,   f_a = (f_q − f_q̄) / 2
# and relaxes them with two independent rates s_plus and s_minus.
# BGK is recovered by s_plus = s_minus = ω. The kinematic viscosity is
# set by s_minus = 1/(3ν + 1/2); the "magic" parameter
#   Λ = (1/s_plus − 1/2)(1/s_minus − 1/2)
# controls boundary and bulk numerical accuracy. Setting Λ = 3/16 makes
# the half-way bounce-back error viscosity-independent (Ginzburg &
# d'Humières, 2003, Phys. Rev. E 68, 066614; Ginzburg et al., 2023,
# J. Comput. Phys. 473, 111711).
#
# Useful compact form for the collision, derived by reassembling f_s
# and f_a back to each population:
#
#   f_q* = f_q − a·(f_q − feq_q) − b·(f_q̄ − feq_q̄)
#
# with a = (s_plus + s_minus)/2  and  b = (s_plus − s_minus)/2.
# When s_plus = s_minus, a = s and b = 0 — pure BGK.
# =====================================================================

@kernel function fused_trt_step_kernel!(f_out, @Const(f_in),
                                         ρ_out, ux_out, uy_out,
                                         @Const(is_solid), Nx, Ny,
                                         s_plus, s_minus)
    i, j = @index(Global, NTuple)
    @inbounds begin
        # Pull-stream with halfway bounce-back at the domain boundary
        # (identical to fused_bgk_step_kernel!).
        fp1 = f_in[i, j, 1]
        fp2 = ifelse(i > 1,             f_in[i - 1, j,     2], f_in[i, j, 4])
        fp3 = ifelse(j > 1,             f_in[i,     j - 1, 3], f_in[i, j, 5])
        fp4 = ifelse(i < Nx,            f_in[i + 1, j,     4], f_in[i, j, 2])
        fp5 = ifelse(j < Ny,            f_in[i,     j + 1, 5], f_in[i, j, 3])
        fp6 = ifelse(i > 1  && j > 1,   f_in[i - 1, j - 1, 6], f_in[i, j, 8])
        fp7 = ifelse(i < Nx && j > 1,   f_in[i + 1, j - 1, 7], f_in[i, j, 9])
        fp8 = ifelse(i < Nx && j < Ny,  f_in[i + 1, j + 1, 8], f_in[i, j, 6])
        fp9 = ifelse(i > 1  && j < Ny,  f_in[i - 1, j + 1, 9], f_in[i, j, 7])

        if is_solid[i, j]
            T = eltype(f_out)
            f_out[i, j, 1] = fp1
            f_out[i, j, 2] = fp4; f_out[i, j, 4] = fp2
            f_out[i, j, 3] = fp5; f_out[i, j, 5] = fp3
            f_out[i, j, 6] = fp8; f_out[i, j, 8] = fp6
            f_out[i, j, 7] = fp9; f_out[i, j, 9] = fp7
            ρ_out[i, j] = one(T)
            ux_out[i, j] = zero(T)
            uy_out[i, j] = zero(T)
        else
            T = eltype(f_out)
            ρ, ux, uy = moments_2d(fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9)
            usq = ux * ux + uy * uy

            feq1 = feq_2d(Val(1), ρ, ux, uy, usq)
            feq2 = feq_2d(Val(2), ρ, ux, uy, usq)
            feq3 = feq_2d(Val(3), ρ, ux, uy, usq)
            feq4 = feq_2d(Val(4), ρ, ux, uy, usq)
            feq5 = feq_2d(Val(5), ρ, ux, uy, usq)
            feq6 = feq_2d(Val(6), ρ, ux, uy, usq)
            feq7 = feq_2d(Val(7), ρ, ux, uy, usq)
            feq8 = feq_2d(Val(8), ρ, ux, uy, usq)
            feq9 = feq_2d(Val(9), ρ, ux, uy, usq)

            a = (s_plus + s_minus) * T(0.5)
            b = (s_plus - s_minus) * T(0.5)

            # Rest direction (self-opposite): only symmetric mode.
            f_out[i, j, 1] = fp1 - s_plus * (fp1 - feq1)
            # Pair (2, 4) east-west
            f_out[i, j, 2] = fp2 - a * (fp2 - feq2) - b * (fp4 - feq4)
            f_out[i, j, 4] = fp4 - a * (fp4 - feq4) - b * (fp2 - feq2)
            # Pair (3, 5) north-south
            f_out[i, j, 3] = fp3 - a * (fp3 - feq3) - b * (fp5 - feq5)
            f_out[i, j, 5] = fp5 - a * (fp5 - feq5) - b * (fp3 - feq3)
            # Pair (6, 8) NE-SW
            f_out[i, j, 6] = fp6 - a * (fp6 - feq6) - b * (fp8 - feq8)
            f_out[i, j, 8] = fp8 - a * (fp8 - feq8) - b * (fp6 - feq6)
            # Pair (7, 9) NW-SE
            f_out[i, j, 7] = fp7 - a * (fp7 - feq7) - b * (fp9 - feq9)
            f_out[i, j, 9] = fp9 - a * (fp9 - feq9) - b * (fp7 - feq7)

            ρ_out[i, j] = ρ
            ux_out[i, j] = ux
            uy_out[i, j] = uy
        end
    end
end

"""
    trt_rates(ν; Λ=3/16) -> (s_plus, s_minus)

Compute the TRT relaxation rates from kinematic viscosity `ν` and the
magic parameter `Λ`. Following Ginzburg & d'Humières 2003 (PRE 68,
066614) and Krüger et al. 2017 (ch. 10.5):

- `s_plus` — rate of the SYMMETRIC (even) mode (f_q + f_q̄)/2. Sets
  the kinematic shear viscosity via `ν = (1/s_plus − 1/2) / 3`, so
  `s_plus = 1/(3ν + 1/2)`.
- `s_minus` — rate of the ANTISYMMETRIC (odd) mode (f_q − f_q̄)/2.
  Set by the magic parameter via `(1/s_plus − 1/2)(1/s_minus − 1/2) = Λ`,
  so `s_minus = 1/(Λ/(3ν) + 1/2)`.

The fused TRT/BGK collide in this file applies `s_plus` to the even
mode and `s_minus` to the odd mode — consistent with this convention.
Default `Λ = 3/16` makes the halfway bounce-back error
viscosity-independent (Ginzburg & d'Humières 2003).

Historical note: commits before 2026-04-15 returned the two rates with
labels swapped (`s_plus` set from Λ, `s_minus` from ν). The collide
formula then applied the Λ-derived rate to the even mode, giving an
effective viscosity `ν_eff = Λ/(9ν)` ≈ 13×ν at Λ=3/16, ν=0.04. This
inflated the real Re by ~13× and silently over-dragged every
benchmark that relied on `trt_rates`. Fixed here.
"""
function trt_rates(ν::Real; Λ::Real=3/16)
    s_plus  = 1.0 / (3ν + 0.5)
    s_minus = 1.0 / (Λ / (3ν) + 0.5)
    return s_plus, s_minus
end

"""
    fused_trt_step!(f_out, f_in, ρ, ux, uy, is_solid, Nx, Ny, ν; Λ=3/16)

Single fused GPU kernel: stream + halfway bounce-back + TRT collide +
macroscopic. Uses Λ = 3/16 by default (Ginzburg-magic bounce-back).
Pass `Λ = Inf` (or any large value) for pure BGK behaviour —
`s_plus → 0` yields `a = s_minus/2`, which is NOT BGK. To recover BGK
exactly, pass `Λ = (1/s − 0.5)²` with s = 1/(3ν+0.5), or equivalently
call `fused_bgk_step!`.
"""
function fused_trt_step!(f_out, f_in, ρ, ux, uy, is_solid, Nx, Ny, ν;
                          Λ::Real=3/16)
    backend = KernelAbstractions.get_backend(f_in)
    ET = eltype(f_in)
    s_plus, s_minus = trt_rates(ν; Λ=Λ)
    kernel! = fused_trt_step_kernel!(backend)
    kernel!(f_out, f_in, ρ, ux, uy, is_solid, Nx, Ny,
            ET(s_plus), ET(s_minus); ndrange=(Nx, Ny))
end
