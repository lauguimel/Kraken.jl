using KernelAbstractions

# =====================================================================
# Fused BGK kernel: stream + bounce-back + collide + macroscopic
# Single kernel launch per timestep for isothermal BGK-LBM.
# =====================================================================

@kernel function fused_bgk_step_kernel!(f_out, @Const(f_in), ρ_out, ux_out, uy_out,
                                         @Const(is_solid), Nx, Ny, ω)
    i, j = @index(Global, NTuple)
    @inbounds begin
        # 1. Pull-stream (same as stream_2d_kernel!)
        im = max(i - 1, 1); ip = min(i + 1, Nx)
        jm = max(j - 1, 1); jp = min(j + 1, Ny)

        fp1 = f_in[i, j, 1]
        fp2 = ifelse(i > 1,             f_in[im, j,  2], f_in[i, j, 4])
        fp3 = ifelse(j > 1,             f_in[i,  jm, 3], f_in[i, j, 5])
        fp4 = ifelse(i < Nx,            f_in[ip, j,  4], f_in[i, j, 2])
        fp5 = ifelse(j < Ny,            f_in[i,  jp, 5], f_in[i, j, 3])
        fp6 = ifelse(i > 1  && j > 1,   f_in[im, jm, 6], f_in[i, j, 8])
        fp7 = ifelse(i < Nx && j > 1,   f_in[ip, jm, 7], f_in[i, j, 9])
        fp8 = ifelse(i < Nx && j < Ny,  f_in[ip, jp, 8], f_in[i, j, 6])
        fp9 = ifelse(i > 1  && j < Ny,  f_in[im, jp, 9], f_in[i, j, 7])

        if is_solid[i, j]
            # Bounce-back on streamed populations
            f_out[i, j, 1] = fp1
            f_out[i, j, 2] = fp4; f_out[i, j, 4] = fp2
            f_out[i, j, 3] = fp5; f_out[i, j, 5] = fp3
            f_out[i, j, 6] = fp8; f_out[i, j, 8] = fp6
            f_out[i, j, 7] = fp9; f_out[i, j, 9] = fp7
            T = eltype(f_out)
            ρ_out[i, j] = one(T)
            ux_out[i, j] = zero(T)
            uy_out[i, j] = zero(T)
        else
            T = eltype(f_out)
            # 2. Macroscopic moments
            ρ, ux, uy = moments_2d(fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9)
            usq = ux * ux + uy * uy

            # 3. BGK collision
            f_out[i, j, 1] = fp1 - ω * (fp1 - feq_2d(Val(1), ρ, ux, uy, usq))
            f_out[i, j, 2] = fp2 - ω * (fp2 - feq_2d(Val(2), ρ, ux, uy, usq))
            f_out[i, j, 3] = fp3 - ω * (fp3 - feq_2d(Val(3), ρ, ux, uy, usq))
            f_out[i, j, 4] = fp4 - ω * (fp4 - feq_2d(Val(4), ρ, ux, uy, usq))
            f_out[i, j, 5] = fp5 - ω * (fp5 - feq_2d(Val(5), ρ, ux, uy, usq))
            f_out[i, j, 6] = fp6 - ω * (fp6 - feq_2d(Val(6), ρ, ux, uy, usq))
            f_out[i, j, 7] = fp7 - ω * (fp7 - feq_2d(Val(7), ρ, ux, uy, usq))
            f_out[i, j, 8] = fp8 - ω * (fp8 - feq_2d(Val(8), ρ, ux, uy, usq))
            f_out[i, j, 9] = fp9 - ω * (fp9 - feq_2d(Val(9), ρ, ux, uy, usq))

            ρ_out[i, j] = ρ
            ux_out[i, j] = ux
            uy_out[i, j] = uy
        end
    end
end

"""
    fused_bgk_step!(f_out, f_in, ρ, ux, uy, is_solid, Nx, Ny, ω)

Single fused kernel for isothermal BGK-LBM: stream + bounce-back + collide + macroscopic.
Reduces kernel launches from 3 to 1 per timestep.
"""
function fused_bgk_step!(f_out, f_in, ρ, ux, uy, is_solid, Nx, Ny, ω)
    backend = KernelAbstractions.get_backend(f_in)
    ET = eltype(f_in)
    kernel! = fused_bgk_step_kernel!(backend)
    kernel!(f_out, f_in, ρ, ux, uy, is_solid, Nx, Ny, ET(ω); ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end
