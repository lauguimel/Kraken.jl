# =====================================================================
# Kernel DSL — brick library.
#
# Each brick is a singleton type + `required_args` + `emit_code`.
# Emitted `Expr` blocks assume the shared local-variable vocabulary:
#   fp1..fp9          pulled populations (post-stream, pre-collision)
#   ρ, ux, uy, usq    moments (pre-collision)
#   feq1..feq9        equilibrium populations
#   fp1c..fp9c        post-collision populations (intermediate)
#   fp2_new..fp9_new  post-BC-overwrite populations (LI-BB)
#   T                 eltype(f_out), set once at kernel entry
#
# Code fragments are COPIED VERBATIM from existing hand-written fused
# kernels so the generated code is bit-exact. Do not refactor for
# "clarity" — downstream tests assert equality under `.==`.
# =====================================================================

# ------------------------------------------------------------------
# Streaming
# ------------------------------------------------------------------

"Pull-stream D2Q9 with halfway-BB fallback at domain edges."
struct PullHalfwayBB <: LBMBrick end
required_args(::PullHalfwayBB) = (:f_in, :Nx, :Ny)
phase(::PullHalfwayBB) = :pre_solid
emit_code(::PullHalfwayBB) = quote
    fp1 = f_in[i, j, 1]
    fp2 = ifelse(i > 1,             f_in[i - 1, j,     2], f_in[i, j, 4])
    fp3 = ifelse(j > 1,             f_in[i,     j - 1, 3], f_in[i, j, 5])
    fp4 = ifelse(i < Nx,            f_in[i + 1, j,     4], f_in[i, j, 2])
    fp5 = ifelse(j < Ny,            f_in[i,     j + 1, 5], f_in[i, j, 3])
    fp6 = ifelse(i > 1  && j > 1,   f_in[i - 1, j - 1, 6], f_in[i, j, 8])
    fp7 = ifelse(i < Nx && j > 1,   f_in[i + 1, j - 1, 7], f_in[i, j, 9])
    fp8 = ifelse(i < Nx && j < Ny,  f_in[i + 1, j + 1, 8], f_in[i, j, 6])
    fp9 = ifelse(i > 1  && j < Ny,  f_in[i - 1, j + 1, 9], f_in[i, j, 7])
end

# ------------------------------------------------------------------
# Solid-cell handling
# ------------------------------------------------------------------

"Legacy: solid cells do bounce-back swap on pulled pops. Bug cause for LI-BB."
struct SolidSwapBB <: LBMBrick end
required_args(::SolidSwapBB) = (:is_solid, :f_out, :ρ_out, :ux_out, :uy_out)
phase(::SolidSwapBB) = :solid
emit_code(::SolidSwapBB) = quote
    f_out[i, j, 1] = fp1
    f_out[i, j, 2] = fp4; f_out[i, j, 4] = fp2
    f_out[i, j, 3] = fp5; f_out[i, j, 5] = fp3
    f_out[i, j, 6] = fp8; f_out[i, j, 8] = fp6
    f_out[i, j, 7] = fp9; f_out[i, j, 9] = fp7
    ρ_out[i, j] = one(T)
    ux_out[i, j] = zero(T)
    uy_out[i, j] = zero(T)
end

"Fix: solid cells carry REST-EQUILIBRIUM populations (ρ=1, u=0 → f_q = w_q). Paired with ApplyLiBB for cut-link BCs. Unlike bare-zero inertia, this keeps the mass sourced to fluid neighbours physically sensible (any fluid cell pulling a population from a solid cell reads a w_q, not 0, so the intermediate post-collision fp_qc used by ApplyLiBB's fallback branches stays well-scaled)."
struct SolidInert <: LBMBrick end
required_args(::SolidInert) = (:is_solid, :f_out, :ρ_out, :ux_out, :uy_out)
phase(::SolidInert) = :solid
emit_code(::SolidInert) = quote
    f_out[i, j, 1] = feq_2d(Val(1), one(T), zero(T), zero(T), zero(T))
    f_out[i, j, 2] = feq_2d(Val(2), one(T), zero(T), zero(T), zero(T))
    f_out[i, j, 3] = feq_2d(Val(3), one(T), zero(T), zero(T), zero(T))
    f_out[i, j, 4] = feq_2d(Val(4), one(T), zero(T), zero(T), zero(T))
    f_out[i, j, 5] = feq_2d(Val(5), one(T), zero(T), zero(T), zero(T))
    f_out[i, j, 6] = feq_2d(Val(6), one(T), zero(T), zero(T), zero(T))
    f_out[i, j, 7] = feq_2d(Val(7), one(T), zero(T), zero(T), zero(T))
    f_out[i, j, 8] = feq_2d(Val(8), one(T), zero(T), zero(T), zero(T))
    f_out[i, j, 9] = feq_2d(Val(9), one(T), zero(T), zero(T), zero(T))
    ρ_out[i, j] = one(T)
    ux_out[i, j] = zero(T)
    uy_out[i, j] = zero(T)
end

# ------------------------------------------------------------------
# Moments
# ------------------------------------------------------------------

"Compute ρ, ux, uy, usq from fp1..fp9."
struct Moments <: LBMBrick end
required_args(::Moments) = ()
emit_code(::Moments) = quote
    ρ, ux, uy = moments_2d(fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9)
    usq = ux * ux + uy * uy
end

"Recompute ρ, ux, uy from post-BC populations (fp1c + fp*_new for q=2..9). Used after ApplyLiBB."
struct RecomputeMoments <: LBMBrick end
required_args(::RecomputeMoments) = ()
emit_code(::RecomputeMoments) = quote
    ρ = fp1c + fp2_new + fp3_new + fp4_new + fp5_new +
        fp6_new + fp7_new + fp8_new + fp9_new
    ux = (fp2_new - fp4_new + fp6_new - fp8_new + fp9_new - fp7_new) / ρ
    uy = (fp3_new - fp5_new + fp6_new - fp8_new - fp9_new + fp7_new) / ρ
end

# ------------------------------------------------------------------
# Collisions — direct write (for fused_bgk / fused_trt oracles)
# ------------------------------------------------------------------

"BGK collision, written directly to f_out[i, j, :]. Matches fused_bgk_step_kernel!."
struct CollideBGKDirect <: LBMBrick end
required_args(::CollideBGKDirect) = (:f_out, :ω)
emit_code(::CollideBGKDirect) = quote
    f_out[i, j, 1] = fp1 - ω * (fp1 - feq_2d(Val(1), ρ, ux, uy, usq))
    f_out[i, j, 2] = fp2 - ω * (fp2 - feq_2d(Val(2), ρ, ux, uy, usq))
    f_out[i, j, 3] = fp3 - ω * (fp3 - feq_2d(Val(3), ρ, ux, uy, usq))
    f_out[i, j, 4] = fp4 - ω * (fp4 - feq_2d(Val(4), ρ, ux, uy, usq))
    f_out[i, j, 5] = fp5 - ω * (fp5 - feq_2d(Val(5), ρ, ux, uy, usq))
    f_out[i, j, 6] = fp6 - ω * (fp6 - feq_2d(Val(6), ρ, ux, uy, usq))
    f_out[i, j, 7] = fp7 - ω * (fp7 - feq_2d(Val(7), ρ, ux, uy, usq))
    f_out[i, j, 8] = fp8 - ω * (fp8 - feq_2d(Val(8), ρ, ux, uy, usq))
    f_out[i, j, 9] = fp9 - ω * (fp9 - feq_2d(Val(9), ρ, ux, uy, usq))
end

"TRT collision, written directly to f_out[i, j, :]. Matches fused_trt_step_kernel!."
struct CollideTRTDirect <: LBMBrick end
required_args(::CollideTRTDirect) = (:f_out, :s_plus, :s_minus)
emit_code(::CollideTRTDirect) = quote
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
    f_out[i, j, 1] = fp1 - s_plus * (fp1 - feq1)
    f_out[i, j, 2] = fp2 - a * (fp2 - feq2) - b * (fp4 - feq4)
    f_out[i, j, 4] = fp4 - a * (fp4 - feq4) - b * (fp2 - feq2)
    f_out[i, j, 3] = fp3 - a * (fp3 - feq3) - b * (fp5 - feq5)
    f_out[i, j, 5] = fp5 - a * (fp5 - feq5) - b * (fp3 - feq3)
    f_out[i, j, 6] = fp6 - a * (fp6 - feq6) - b * (fp8 - feq8)
    f_out[i, j, 8] = fp8 - a * (fp8 - feq8) - b * (fp6 - feq6)
    f_out[i, j, 7] = fp7 - a * (fp7 - feq7) - b * (fp9 - feq9)
    f_out[i, j, 9] = fp9 - a * (fp9 - feq9) - b * (fp7 - feq7)
end

# ------------------------------------------------------------------
# Collisions — intermediate write (for fused_trt_libb)
# ------------------------------------------------------------------

"TRT collision writing to intermediate fp1c..fp9c. Used before ApplyLiBB overwrite."
struct CollideTRT <: LBMBrick end
required_args(::CollideTRT) = (:s_plus, :s_minus)
emit_code(::CollideTRT) = quote
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
    fp1c = fp1 - s_plus * (fp1 - feq1)
    fp2c = fp2 - a * (fp2 - feq2) - b * (fp4 - feq4)
    fp4c = fp4 - a * (fp4 - feq4) - b * (fp2 - feq2)
    fp3c = fp3 - a * (fp3 - feq3) - b * (fp5 - feq5)
    fp5c = fp5 - a * (fp5 - feq5) - b * (fp3 - feq3)
    fp6c = fp6 - a * (fp6 - feq6) - b * (fp8 - feq8)
    fp8c = fp8 - a * (fp8 - feq8) - b * (fp6 - feq6)
    fp7c = fp7 - a * (fp7 - feq7) - b * (fp9 - feq9)
    fp9c = fp9 - a * (fp9 - feq9) - b * (fp7 - feq7)
end

# ------------------------------------------------------------------
# Boundary conditions
# ------------------------------------------------------------------

"Pre-collision halfway-BB substitution on flagged cut links. Replaces the pulled pop fp_{q̄} (which came from a solid cell and is junk) with `f_in[i,j,q] + δ_{q̄}` — a lag-1 halfway-BB estimate, consistent with the classical halfway-BB storage at solids (which my DSL replaces with SolidInert equilibrium). At q_w=0.5 this is Bouzidi for pre-collision. Used alongside ApplyLiBB (post-collision) to close the moment-consistency loop."
struct ApplyHalfwayBBPrePhase <: LBMBrick end
required_args(::ApplyHalfwayBBPrePhase) = (:f_in, :q_wall, :uw_link_x, :uw_link_y)
emit_code(::ApplyHalfwayBBPrePhase) = quote
    # Pair (2, 4): link q=2 flagged → corrupted pop is fp4.
    if q_wall[i, j, 2] > zero(T)
        δ4 = -T(2/3) * uw_link_x[i, j, 2]
        fp4 = f_in[i, j, 2] + δ4
    end
    if q_wall[i, j, 4] > zero(T)
        δ2 =  T(2/3) * uw_link_x[i, j, 4]
        fp2 = f_in[i, j, 4] + δ2
    end
    # Pair (3, 5)
    if q_wall[i, j, 3] > zero(T)
        δ5 = -T(2/3) * uw_link_y[i, j, 3]
        fp5 = f_in[i, j, 3] + δ5
    end
    if q_wall[i, j, 5] > zero(T)
        δ3 =  T(2/3) * uw_link_y[i, j, 5]
        fp3 = f_in[i, j, 5] + δ3
    end
    # Pair (6, 8)
    if q_wall[i, j, 6] > zero(T)
        δ8 = -T(1/6) * (uw_link_x[i, j, 6] + uw_link_y[i, j, 6])
        fp8 = f_in[i, j, 6] + δ8
    end
    if q_wall[i, j, 8] > zero(T)
        δ6 =  T(1/6) * (uw_link_x[i, j, 8] + uw_link_y[i, j, 8])
        fp6 = f_in[i, j, 8] + δ6
    end
    # Pair (7, 9)
    if q_wall[i, j, 7] > zero(T)
        δ9 = -T(1/6) * (-uw_link_x[i, j, 7] + uw_link_y[i, j, 7])
        fp9 = f_in[i, j, 7] + δ9
    end
    if q_wall[i, j, 9] > zero(T)
        δ7 =  T(1/6) * (-uw_link_x[i, j, 9] + uw_link_y[i, j, 9])
        fp7 = f_in[i, j, 9] + δ7
    end
end

"Pre-collision axes-only halfway-BB substitution. Like ApplyHalfwayBBPrePhase but skips diagonal pops (q=6..9) so that diagonals remain as rest-equilibrium from SolidInert. Hypothesis-testing brick for investigating the near-wall residual of ApplyHalfwayBBPrePhase on diagonal links."
struct ApplyHalfwayBBPrePhaseAxes <: LBMBrick end
required_args(::ApplyHalfwayBBPrePhaseAxes) = (:f_in, :q_wall, :uw_link_x, :uw_link_y)
emit_code(::ApplyHalfwayBBPrePhaseAxes) = quote
    if q_wall[i, j, 2] > zero(T)
        δ4 = -T(2/3) * uw_link_x[i, j, 2]
        fp4 = f_in[i, j, 2] + δ4
    end
    if q_wall[i, j, 4] > zero(T)
        δ2 =  T(2/3) * uw_link_x[i, j, 4]
        fp2 = f_in[i, j, 4] + δ2
    end
    if q_wall[i, j, 3] > zero(T)
        δ5 = -T(2/3) * uw_link_y[i, j, 3]
        fp5 = f_in[i, j, 3] + δ5
    end
    if q_wall[i, j, 5] > zero(T)
        δ3 =  T(2/3) * uw_link_y[i, j, 5]
        fp3 = f_in[i, j, 5] + δ3
    end
end

"Bouzidi interpolated bounce-back (LI-BB) overwrite on flagged cut links. Reads fp*c + fp*, writes fp*_new for q=2..9."
struct ApplyLiBB <: LBMBrick end
required_args(::ApplyLiBB) = (:q_wall, :uw_link_x, :uw_link_y)
emit_code(::ApplyLiBB) = quote
    # Pair (2, 4) east / west
    qw2 = q_wall[i, j, 2]
    if qw2 > zero(T)
        δ4 = -T(2/3) * uw_link_x[i, j, 2]
        fp4_new = _libb_branch(qw2, fp2c, fp2, fp4c, δ4)
    else
        fp4_new = fp4c
    end
    qw4 = q_wall[i, j, 4]
    if qw4 > zero(T)
        δ2 =  T(2/3) * uw_link_x[i, j, 4]
        fp2_new = _libb_branch(qw4, fp4c, fp4, fp2c, δ2)
    else
        fp2_new = fp2c
    end
    # Pair (3, 5) north / south
    qw3 = q_wall[i, j, 3]
    if qw3 > zero(T)
        δ5 = -T(2/3) * uw_link_y[i, j, 3]
        fp5_new = _libb_branch(qw3, fp3c, fp3, fp5c, δ5)
    else
        fp5_new = fp5c
    end
    qw5 = q_wall[i, j, 5]
    if qw5 > zero(T)
        δ3 =  T(2/3) * uw_link_y[i, j, 5]
        fp3_new = _libb_branch(qw5, fp5c, fp5, fp3c, δ3)
    else
        fp3_new = fp3c
    end
    # Pair (6, 8) NE / SW
    qw6 = q_wall[i, j, 6]
    if qw6 > zero(T)
        uxw6 = uw_link_x[i, j, 6]; uyw6 = uw_link_y[i, j, 6]
        δ8 = -T(1/6) * (uxw6 + uyw6)
        fp8_new = _libb_branch(qw6, fp6c, fp6, fp8c, δ8)
    else
        fp8_new = fp8c
    end
    qw8 = q_wall[i, j, 8]
    if qw8 > zero(T)
        uxw8 = uw_link_x[i, j, 8]; uyw8 = uw_link_y[i, j, 8]
        δ6 =  T(1/6) * (uxw8 + uyw8)
        fp6_new = _libb_branch(qw8, fp8c, fp8, fp6c, δ6)
    else
        fp6_new = fp6c
    end
    # Pair (7, 9) NW / SE
    qw7 = q_wall[i, j, 7]
    if qw7 > zero(T)
        uxw7 = uw_link_x[i, j, 7]; uyw7 = uw_link_y[i, j, 7]
        δ9 = -T(1/6) * (-uxw7 + uyw7)
        fp9_new = _libb_branch(qw7, fp7c, fp7, fp9c, δ9)
    else
        fp9_new = fp9c
    end
    qw9 = q_wall[i, j, 9]
    if qw9 > zero(T)
        uxw9 = uw_link_x[i, j, 9]; uyw9 = uw_link_y[i, j, 9]
        δ7 =  T(1/6) * (-uxw9 + uyw9)
        fp7_new = _libb_branch(qw9, fp9c, fp9, fp7c, δ7)
    else
        fp7_new = fp7c
    end
end

# ------------------------------------------------------------------
# Write-back
# ------------------------------------------------------------------

"Write raw fp1..fp9 to f_out[i, j, :]. Used for pull-only debug kernels."
struct WriteF <: LBMBrick end
required_args(::WriteF) = (:f_out,)
emit_code(::WriteF) = quote
    f_out[i, j, 1] = fp1
    f_out[i, j, 2] = fp2
    f_out[i, j, 3] = fp3
    f_out[i, j, 4] = fp4
    f_out[i, j, 5] = fp5
    f_out[i, j, 6] = fp6
    f_out[i, j, 7] = fp7
    f_out[i, j, 8] = fp8
    f_out[i, j, 9] = fp9
end

"Write f_out composed of fp1c (rest) + fp*_new for q=2..9. Used after ApplyLiBB."
struct WriteFLiBB <: LBMBrick end
required_args(::WriteFLiBB) = (:f_out,)
emit_code(::WriteFLiBB) = quote
    f_out[i, j, 1] = fp1c
    f_out[i, j, 2] = fp2_new; f_out[i, j, 4] = fp4_new
    f_out[i, j, 3] = fp3_new; f_out[i, j, 5] = fp5_new
    f_out[i, j, 6] = fp6_new; f_out[i, j, 8] = fp8_new
    f_out[i, j, 7] = fp7_new; f_out[i, j, 9] = fp9_new
end

"Write ρ, ux, uy to ρ_out, ux_out, uy_out at (i, j)."
struct WriteMoments <: LBMBrick end
required_args(::WriteMoments) = (:ρ_out, :ux_out, :uy_out)
emit_code(::WriteMoments) = quote
    ρ_out[i, j] = ρ
    ux_out[i, j] = ux
    uy_out[i, j] = uy
end
