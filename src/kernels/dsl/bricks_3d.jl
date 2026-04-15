# =====================================================================
# Kernel DSL — D3Q19 brick library.
#
# Same shared-vocabulary pattern as `bricks.jl` (D2Q9), extended to
# 19 populations and 3D indexing. Variable naming:
#   fp1..fp19            pulled populations
#   ρ, ux, uy, uz, usq   moments
#   feq1..feq19          equilibrium populations
#   T                    eltype(f_out)
#
# Bricks used by the LI-BB V2 spec (D3Q19):
#   PullHalfwayBB_3D, SolidInert_3D, ApplyLiBBPrePhase_3D,
#   Moments_3D, CollideTRTDirect_3D, WriteMoments_3D
#
# D3Q19 opposite pairs (for bounce-back swap + LI-BB):
#   (2, 3)   (4, 5)   (6, 7)
#   (8, 11)  (9, 10)
#   (12, 15) (13, 14)
#   (16, 19) (17, 18)
# =====================================================================

"Pull-stream D3Q19 with halfway-BB fallback at domain edges."
struct PullHalfwayBB_3D <: LBMBrick end
required_args(::PullHalfwayBB_3D) = (:f_in, :Nx, :Ny, :Nz)
phase(::PullHalfwayBB_3D) = :pre_solid
emit_code(::PullHalfwayBB_3D) = quote
    fp1  = f_in[i, j, k, 1]
    # Axis-aligned
    fp2  = ifelse(i > 1,              f_in[i-1, j,   k,   2],  f_in[i, j, k, 3])
    fp3  = ifelse(i < Nx,             f_in[i+1, j,   k,   3],  f_in[i, j, k, 2])
    fp4  = ifelse(j > 1,              f_in[i,   j-1, k,   4],  f_in[i, j, k, 5])
    fp5  = ifelse(j < Ny,             f_in[i,   j+1, k,   5],  f_in[i, j, k, 4])
    fp6  = ifelse(k > 1,              f_in[i,   j,   k-1, 6],  f_in[i, j, k, 7])
    fp7  = ifelse(k < Nz,             f_in[i,   j,   k+1, 7],  f_in[i, j, k, 6])
    # Edge xy: 8(+x+y), 9(-x+y), 10(+x-y), 11(-x-y)
    fp8  = ifelse(i > 1  && j > 1,    f_in[i-1, j-1, k,   8],  f_in[i, j, k, 11])
    fp9  = ifelse(i < Nx && j > 1,    f_in[i+1, j-1, k,   9],  f_in[i, j, k, 10])
    fp10 = ifelse(i > 1  && j < Ny,   f_in[i-1, j+1, k,   10], f_in[i, j, k, 9])
    fp11 = ifelse(i < Nx && j < Ny,   f_in[i+1, j+1, k,   11], f_in[i, j, k, 8])
    # Edge xz: 12(+x+z), 13(-x+z), 14(+x-z), 15(-x-z)
    fp12 = ifelse(i > 1  && k > 1,    f_in[i-1, j,   k-1, 12], f_in[i, j, k, 15])
    fp13 = ifelse(i < Nx && k > 1,    f_in[i+1, j,   k-1, 13], f_in[i, j, k, 14])
    fp14 = ifelse(i > 1  && k < Nz,   f_in[i-1, j,   k+1, 14], f_in[i, j, k, 13])
    fp15 = ifelse(i < Nx && k < Nz,   f_in[i+1, j,   k+1, 15], f_in[i, j, k, 12])
    # Edge yz: 16(+y+z), 17(-y+z), 18(+y-z), 19(-y-z)
    fp16 = ifelse(j > 1  && k > 1,    f_in[i,   j-1, k-1, 16], f_in[i, j, k, 19])
    fp17 = ifelse(j < Ny && k > 1,    f_in[i,   j+1, k-1, 17], f_in[i, j, k, 18])
    fp18 = ifelse(j > 1  && k < Nz,   f_in[i,   j-1, k+1, 18], f_in[i, j, k, 17])
    fp19 = ifelse(j < Ny && k < Nz,   f_in[i,   j+1, k+1, 19], f_in[i, j, k, 16])
end

"Solid cells are INERT: rest-equilibrium populations on D3Q19."
struct SolidInert_3D <: LBMBrick end
required_args(::SolidInert_3D) = (:is_solid, :f_out, :ρ_out, :ux_out, :uy_out, :uz_out)
phase(::SolidInert_3D) = :solid
emit_code(::SolidInert_3D) = quote
    f_out[i, j, k, 1]  = feq_3d(Val(1),  one(T), zero(T), zero(T), zero(T), zero(T))
    f_out[i, j, k, 2]  = feq_3d(Val(2),  one(T), zero(T), zero(T), zero(T), zero(T))
    f_out[i, j, k, 3]  = feq_3d(Val(3),  one(T), zero(T), zero(T), zero(T), zero(T))
    f_out[i, j, k, 4]  = feq_3d(Val(4),  one(T), zero(T), zero(T), zero(T), zero(T))
    f_out[i, j, k, 5]  = feq_3d(Val(5),  one(T), zero(T), zero(T), zero(T), zero(T))
    f_out[i, j, k, 6]  = feq_3d(Val(6),  one(T), zero(T), zero(T), zero(T), zero(T))
    f_out[i, j, k, 7]  = feq_3d(Val(7),  one(T), zero(T), zero(T), zero(T), zero(T))
    f_out[i, j, k, 8]  = feq_3d(Val(8),  one(T), zero(T), zero(T), zero(T), zero(T))
    f_out[i, j, k, 9]  = feq_3d(Val(9),  one(T), zero(T), zero(T), zero(T), zero(T))
    f_out[i, j, k, 10] = feq_3d(Val(10), one(T), zero(T), zero(T), zero(T), zero(T))
    f_out[i, j, k, 11] = feq_3d(Val(11), one(T), zero(T), zero(T), zero(T), zero(T))
    f_out[i, j, k, 12] = feq_3d(Val(12), one(T), zero(T), zero(T), zero(T), zero(T))
    f_out[i, j, k, 13] = feq_3d(Val(13), one(T), zero(T), zero(T), zero(T), zero(T))
    f_out[i, j, k, 14] = feq_3d(Val(14), one(T), zero(T), zero(T), zero(T), zero(T))
    f_out[i, j, k, 15] = feq_3d(Val(15), one(T), zero(T), zero(T), zero(T), zero(T))
    f_out[i, j, k, 16] = feq_3d(Val(16), one(T), zero(T), zero(T), zero(T), zero(T))
    f_out[i, j, k, 17] = feq_3d(Val(17), one(T), zero(T), zero(T), zero(T), zero(T))
    f_out[i, j, k, 18] = feq_3d(Val(18), one(T), zero(T), zero(T), zero(T), zero(T))
    f_out[i, j, k, 19] = feq_3d(Val(19), one(T), zero(T), zero(T), zero(T), zero(T))
    ρ_out[i, j, k] = one(T)
    ux_out[i, j, k] = zero(T)
    uy_out[i, j, k] = zero(T)
    uz_out[i, j, k] = zero(T)
end

"D3Q19 moments ρ, ux, uy, uz, usq from fp1..fp19."
struct Moments_3D <: LBMBrick end
required_args(::Moments_3D) = ()
emit_code(::Moments_3D) = quote
    ρ, ux, uy, uz = moments_3d(fp1, fp2, fp3, fp4, fp5, fp6, fp7,
                                fp8, fp9, fp10, fp11, fp12, fp13,
                                fp14, fp15, fp16, fp17, fp18, fp19)
    usq = ux * ux + uy * uy + uz * uz
end

"TRT collision on D3Q19, written directly to f_out[i, j, k, :]."
struct CollideTRTDirect_3D <: LBMBrick end
required_args(::CollideTRTDirect_3D) = (:f_out, :s_plus, :s_minus)
emit_code(::CollideTRTDirect_3D) = quote
    feq1  = feq_3d(Val(1),  ρ, ux, uy, uz, usq)
    feq2  = feq_3d(Val(2),  ρ, ux, uy, uz, usq)
    feq3  = feq_3d(Val(3),  ρ, ux, uy, uz, usq)
    feq4  = feq_3d(Val(4),  ρ, ux, uy, uz, usq)
    feq5  = feq_3d(Val(5),  ρ, ux, uy, uz, usq)
    feq6  = feq_3d(Val(6),  ρ, ux, uy, uz, usq)
    feq7  = feq_3d(Val(7),  ρ, ux, uy, uz, usq)
    feq8  = feq_3d(Val(8),  ρ, ux, uy, uz, usq)
    feq9  = feq_3d(Val(9),  ρ, ux, uy, uz, usq)
    feq10 = feq_3d(Val(10), ρ, ux, uy, uz, usq)
    feq11 = feq_3d(Val(11), ρ, ux, uy, uz, usq)
    feq12 = feq_3d(Val(12), ρ, ux, uy, uz, usq)
    feq13 = feq_3d(Val(13), ρ, ux, uy, uz, usq)
    feq14 = feq_3d(Val(14), ρ, ux, uy, uz, usq)
    feq15 = feq_3d(Val(15), ρ, ux, uy, uz, usq)
    feq16 = feq_3d(Val(16), ρ, ux, uy, uz, usq)
    feq17 = feq_3d(Val(17), ρ, ux, uy, uz, usq)
    feq18 = feq_3d(Val(18), ρ, ux, uy, uz, usq)
    feq19 = feq_3d(Val(19), ρ, ux, uy, uz, usq)
    a = (s_plus + s_minus) * T(0.5)
    b = (s_plus - s_minus) * T(0.5)
    # Rest
    f_out[i, j, k, 1]  = fp1  - s_plus * (fp1 - feq1)
    # Axis (2, 3) x
    f_out[i, j, k, 2]  = fp2  - a * (fp2  - feq2)  - b * (fp3  - feq3)
    f_out[i, j, k, 3]  = fp3  - a * (fp3  - feq3)  - b * (fp2  - feq2)
    # Axis (4, 5) y
    f_out[i, j, k, 4]  = fp4  - a * (fp4  - feq4)  - b * (fp5  - feq5)
    f_out[i, j, k, 5]  = fp5  - a * (fp5  - feq5)  - b * (fp4  - feq4)
    # Axis (6, 7) z
    f_out[i, j, k, 6]  = fp6  - a * (fp6  - feq6)  - b * (fp7  - feq7)
    f_out[i, j, k, 7]  = fp7  - a * (fp7  - feq7)  - b * (fp6  - feq6)
    # Edge xy (8, 11) and (9, 10)
    f_out[i, j, k, 8]  = fp8  - a * (fp8  - feq8)  - b * (fp11 - feq11)
    f_out[i, j, k, 11] = fp11 - a * (fp11 - feq11) - b * (fp8  - feq8)
    f_out[i, j, k, 9]  = fp9  - a * (fp9  - feq9)  - b * (fp10 - feq10)
    f_out[i, j, k, 10] = fp10 - a * (fp10 - feq10) - b * (fp9  - feq9)
    # Edge xz (12, 15) and (13, 14)
    f_out[i, j, k, 12] = fp12 - a * (fp12 - feq12) - b * (fp15 - feq15)
    f_out[i, j, k, 15] = fp15 - a * (fp15 - feq15) - b * (fp12 - feq12)
    f_out[i, j, k, 13] = fp13 - a * (fp13 - feq13) - b * (fp14 - feq14)
    f_out[i, j, k, 14] = fp14 - a * (fp14 - feq14) - b * (fp13 - feq13)
    # Edge yz (16, 19) and (17, 18)
    f_out[i, j, k, 16] = fp16 - a * (fp16 - feq16) - b * (fp19 - feq19)
    f_out[i, j, k, 19] = fp19 - a * (fp19 - feq19) - b * (fp16 - feq16)
    f_out[i, j, k, 17] = fp17 - a * (fp17 - feq17) - b * (fp18 - feq18)
    f_out[i, j, k, 18] = fp18 - a * (fp18 - feq18) - b * (fp17 - feq17)
end

"Bouzidi pre-phase (D3Q19). For each flagged link q, substitutes the corrupted pulled pop fp_{q̄} via `_libb_branch(q_w, f_in[i,j,k,q], fp_q, f_in[i,j,k,q̄], δ_{q̄})`. Handles arbitrary q_w ∈ (0, 1]."
struct ApplyLiBBPrePhase_3D <: LBMBrick end
required_args(::ApplyLiBBPrePhase_3D) = (:f_in, :q_wall, :uw_link_x, :uw_link_y, :uw_link_z)
emit_code(::ApplyLiBBPrePhase_3D) = quote
    # δ_{q̄} = -2 w_q (c_q · u_w) / c_s² = -6 w_q (c_q · u_w)
    # Axis links — w=1/18 → coefficient = -6/18 = -1/3
    # Edge links — w=1/36 → coefficient = -6/36 = -1/6

    # Pair (2, 3) x-axis: link q=2 flagged → substitute fp3 (=q̄ of q=2)
    qw2 = q_wall[i, j, k, 2]
    if qw2 > zero(T)
        δ3 = -T(1/3) * uw_link_x[i, j, k, 2]
        fp3 = _libb_branch(qw2, f_in[i, j, k, 2], fp2, f_in[i, j, k, 3], δ3)
    end
    qw3 = q_wall[i, j, k, 3]
    if qw3 > zero(T)
        δ2 =  T(1/3) * uw_link_x[i, j, k, 3]
        fp2 = _libb_branch(qw3, f_in[i, j, k, 3], fp3, f_in[i, j, k, 2], δ2)
    end
    # Pair (4, 5) y-axis
    qw4 = q_wall[i, j, k, 4]
    if qw4 > zero(T)
        δ5 = -T(1/3) * uw_link_y[i, j, k, 4]
        fp5 = _libb_branch(qw4, f_in[i, j, k, 4], fp4, f_in[i, j, k, 5], δ5)
    end
    qw5 = q_wall[i, j, k, 5]
    if qw5 > zero(T)
        δ4 =  T(1/3) * uw_link_y[i, j, k, 5]
        fp4 = _libb_branch(qw5, f_in[i, j, k, 5], fp5, f_in[i, j, k, 4], δ4)
    end
    # Pair (6, 7) z-axis
    qw6 = q_wall[i, j, k, 6]
    if qw6 > zero(T)
        δ7 = -T(1/3) * uw_link_z[i, j, k, 6]
        fp7 = _libb_branch(qw6, f_in[i, j, k, 6], fp6, f_in[i, j, k, 7], δ7)
    end
    qw7 = q_wall[i, j, k, 7]
    if qw7 > zero(T)
        δ6 =  T(1/3) * uw_link_z[i, j, k, 7]
        fp6 = _libb_branch(qw7, f_in[i, j, k, 7], fp7, f_in[i, j, k, 6], δ6)
    end
    # Pair (8, 11) xy-edge: c_8 = (+1, +1, 0); c_11 = (-1, -1, 0)
    qw8 = q_wall[i, j, k, 8]
    if qw8 > zero(T)
        δ11 = -T(1/6) * (uw_link_x[i, j, k, 8] + uw_link_y[i, j, k, 8])
        fp11 = _libb_branch(qw8, f_in[i, j, k, 8], fp8, f_in[i, j, k, 11], δ11)
    end
    qw11 = q_wall[i, j, k, 11]
    if qw11 > zero(T)
        δ8 =  T(1/6) * (uw_link_x[i, j, k, 11] + uw_link_y[i, j, k, 11])
        fp8 = _libb_branch(qw11, f_in[i, j, k, 11], fp11, f_in[i, j, k, 8], δ8)
    end
    # Pair (9, 10): c_9 = (-1, +1, 0); c_10 = (+1, -1, 0)
    qw9 = q_wall[i, j, k, 9]
    if qw9 > zero(T)
        δ10 = -T(1/6) * (-uw_link_x[i, j, k, 9] + uw_link_y[i, j, k, 9])
        fp10 = _libb_branch(qw9, f_in[i, j, k, 9], fp9, f_in[i, j, k, 10], δ10)
    end
    qw10 = q_wall[i, j, k, 10]
    if qw10 > zero(T)
        δ9 = -T(1/6) * (uw_link_x[i, j, k, 10] - uw_link_y[i, j, k, 10])
        fp9 = _libb_branch(qw10, f_in[i, j, k, 10], fp10, f_in[i, j, k, 9], δ9)
    end
    # Pair (12, 15): c_12 = (+1, 0, +1); c_15 = (-1, 0, -1)
    qw12 = q_wall[i, j, k, 12]
    if qw12 > zero(T)
        δ15 = -T(1/6) * (uw_link_x[i, j, k, 12] + uw_link_z[i, j, k, 12])
        fp15 = _libb_branch(qw12, f_in[i, j, k, 12], fp12, f_in[i, j, k, 15], δ15)
    end
    qw15 = q_wall[i, j, k, 15]
    if qw15 > zero(T)
        δ12 =  T(1/6) * (uw_link_x[i, j, k, 15] + uw_link_z[i, j, k, 15])
        fp12 = _libb_branch(qw15, f_in[i, j, k, 15], fp15, f_in[i, j, k, 12], δ12)
    end
    # Pair (13, 14): c_13 = (-1, 0, +1); c_14 = (+1, 0, -1)
    qw13 = q_wall[i, j, k, 13]
    if qw13 > zero(T)
        δ14 = -T(1/6) * (-uw_link_x[i, j, k, 13] + uw_link_z[i, j, k, 13])
        fp14 = _libb_branch(qw13, f_in[i, j, k, 13], fp13, f_in[i, j, k, 14], δ14)
    end
    qw14 = q_wall[i, j, k, 14]
    if qw14 > zero(T)
        δ13 = -T(1/6) * (uw_link_x[i, j, k, 14] - uw_link_z[i, j, k, 14])
        fp13 = _libb_branch(qw14, f_in[i, j, k, 14], fp14, f_in[i, j, k, 13], δ13)
    end
    # Pair (16, 19): c_16 = (0, +1, +1); c_19 = (0, -1, -1)
    qw16 = q_wall[i, j, k, 16]
    if qw16 > zero(T)
        δ19 = -T(1/6) * (uw_link_y[i, j, k, 16] + uw_link_z[i, j, k, 16])
        fp19 = _libb_branch(qw16, f_in[i, j, k, 16], fp16, f_in[i, j, k, 19], δ19)
    end
    qw19 = q_wall[i, j, k, 19]
    if qw19 > zero(T)
        δ16 =  T(1/6) * (uw_link_y[i, j, k, 19] + uw_link_z[i, j, k, 19])
        fp16 = _libb_branch(qw19, f_in[i, j, k, 19], fp19, f_in[i, j, k, 16], δ16)
    end
    # Pair (17, 18): c_17 = (0, -1, +1); c_18 = (0, +1, -1)
    qw17 = q_wall[i, j, k, 17]
    if qw17 > zero(T)
        δ18 = -T(1/6) * (-uw_link_y[i, j, k, 17] + uw_link_z[i, j, k, 17])
        fp18 = _libb_branch(qw17, f_in[i, j, k, 17], fp17, f_in[i, j, k, 18], δ18)
    end
    qw18 = q_wall[i, j, k, 18]
    if qw18 > zero(T)
        δ17 = -T(1/6) * (uw_link_y[i, j, k, 18] - uw_link_z[i, j, k, 18])
        fp17 = _libb_branch(qw18, f_in[i, j, k, 18], fp18, f_in[i, j, k, 17], δ17)
    end
end

"Write ρ, ux, uy, uz to the (i, j, k) cell of the output arrays."
struct WriteMoments_3D <: LBMBrick end
required_args(::WriteMoments_3D) = (:ρ_out, :ux_out, :uy_out, :uz_out)
emit_code(::WriteMoments_3D) = quote
    ρ_out[i, j, k]  = ρ
    ux_out[i, j, k] = ux
    uy_out[i, j, k] = uy
    uz_out[i, j, k] = uz
end
