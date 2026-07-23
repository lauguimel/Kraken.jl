using KernelAbstractions

# =====================================================================
# AA-pattern BGK kernels: single-buffer alternating access scheme.
#
# Even step: pull from neighbors -> collide -> write LOCAL
# Odd step:  read LOCAL -> collide -> push to neighbors
#
# Halves memory usage (1 buffer instead of 2) and improves cache reuse.
# =====================================================================

# --- Even step: pull + collide + write local (same buffer) ---

@kernel function aa_even_kernel!(f, @Const(is_solid), Nx, Ny, ω)
    i, j = @index(Global, NTuple)
    @inbounds begin
        im = max(i - 1, 1); ip = min(i + 1, Nx)
        jm = max(j - 1, 1); jp = min(j + 1, Ny)

        # Pull from neighbors
        fp1 = f[i, j, 1]
        fp2 = ifelse(i > 1,             f[im, j,  2], f[i, j, 4])
        fp3 = ifelse(j > 1,             f[i,  jm, 3], f[i, j, 5])
        fp4 = ifelse(i < Nx,            f[ip, j,  4], f[i, j, 2])
        fp5 = ifelse(j < Ny,            f[i,  jp, 5], f[i, j, 3])
        fp6 = ifelse(i > 1  && j > 1,   f[im, jm, 6], f[i, j, 8])
        fp7 = ifelse(i < Nx && j > 1,   f[ip, jm, 7], f[i, j, 9])
        fp8 = ifelse(i < Nx && j < Ny,  f[ip, jp, 8], f[i, j, 6])
        fp9 = ifelse(i > 1  && j < Ny,  f[im, jp, 9], f[i, j, 7])

        if is_solid[i, j]
            # Bounce-back
            f[i, j, 2] = fp4; f[i, j, 4] = fp2
            f[i, j, 3] = fp5; f[i, j, 5] = fp3
            f[i, j, 6] = fp8; f[i, j, 8] = fp6
            f[i, j, 7] = fp9; f[i, j, 9] = fp7
        else
            # Macroscopic + collision
            ρ, ux, uy = moments_2d(fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9)
            usq = ux * ux + uy * uy

            f[i, j, 1] = fp1 - ω * (fp1 - feq_2d(Val(1), ρ, ux, uy, usq))
            f[i, j, 2] = fp2 - ω * (fp2 - feq_2d(Val(2), ρ, ux, uy, usq))
            f[i, j, 3] = fp3 - ω * (fp3 - feq_2d(Val(3), ρ, ux, uy, usq))
            f[i, j, 4] = fp4 - ω * (fp4 - feq_2d(Val(4), ρ, ux, uy, usq))
            f[i, j, 5] = fp5 - ω * (fp5 - feq_2d(Val(5), ρ, ux, uy, usq))
            f[i, j, 6] = fp6 - ω * (fp6 - feq_2d(Val(6), ρ, ux, uy, usq))
            f[i, j, 7] = fp7 - ω * (fp7 - feq_2d(Val(7), ρ, ux, uy, usq))
            f[i, j, 8] = fp8 - ω * (fp8 - feq_2d(Val(8), ρ, ux, uy, usq))
            f[i, j, 9] = fp9 - ω * (fp9 - feq_2d(Val(9), ρ, ux, uy, usq))
        end
    end
end

# --- Odd step: read local -> collide -> push to neighbors ---

@kernel function aa_odd_kernel!(f, @Const(is_solid), Nx, Ny, ω)
    i, j = @index(Global, NTuple)
    @inbounds begin
        # Read LOCAL populations (post-collision from even step)
        f1 = f[i, j, 1]; f2 = f[i, j, 2]; f3 = f[i, j, 3]
        f4 = f[i, j, 4]; f5 = f[i, j, 5]; f6 = f[i, j, 6]
        f7 = f[i, j, 7]; f8 = f[i, j, 8]; f9 = f[i, j, 9]

        if is_solid[i, j]
            # Bounce-back: swap opposite pairs locally
            f[i, j, 2] = f4; f[i, j, 4] = f2
            f[i, j, 3] = f5; f[i, j, 5] = f3
            f[i, j, 6] = f8; f[i, j, 8] = f6
            f[i, j, 7] = f9; f[i, j, 9] = f7
        else
            ρ, ux, uy = moments_2d(f1, f2, f3, f4, f5, f6, f7, f8, f9)
            usq = ux * ux + uy * uy

            # Collide
            c1 = f1 - ω * (f1 - feq_2d(Val(1), ρ, ux, uy, usq))
            c2 = f2 - ω * (f2 - feq_2d(Val(2), ρ, ux, uy, usq))
            c3 = f3 - ω * (f3 - feq_2d(Val(3), ρ, ux, uy, usq))
            c4 = f4 - ω * (f4 - feq_2d(Val(4), ρ, ux, uy, usq))
            c5 = f5 - ω * (f5 - feq_2d(Val(5), ρ, ux, uy, usq))
            c6 = f6 - ω * (f6 - feq_2d(Val(6), ρ, ux, uy, usq))
            c7 = f7 - ω * (f7 - feq_2d(Val(7), ρ, ux, uy, usq))
            c8 = f8 - ω * (f8 - feq_2d(Val(8), ρ, ux, uy, usq))
            c9 = f9 - ω * (f9 - feq_2d(Val(9), ρ, ux, uy, usq))

            # Push to neighbors (streaming in push direction)
            im = max(i - 1, 1); ip = min(i + 1, Nx)
            jm = max(j - 1, 1); jp = min(j + 1, Ny)

            f[i, j, 1] = c1

            # Pop 2: velocity (+1,0) -> push to (i+1,j); bounce-back at east wall
            f[ifelse(i < Nx, ip, i), j, 2] = ifelse(i < Nx, c2, c4)
            # Pop 3: velocity (0,+1) -> push to (i,j+1); bounce-back at north wall
            f[i, ifelse(j < Ny, jp, j), 3] = ifelse(j < Ny, c3, c5)
            # Pop 4: velocity (-1,0) -> push to (i-1,j); bounce-back at west wall
            f[ifelse(i > 1, im, i), j, 4] = ifelse(i > 1, c4, c2)
            # Pop 5: velocity (0,-1) -> push to (i,j-1); bounce-back at south wall
            f[i, ifelse(j > 1, jm, j), 5] = ifelse(j > 1, c5, c3)

            # Pop 6: velocity (+1,+1) -> push to (i+1,j+1)
            cond6 = i < Nx && j < Ny
            f[ifelse(cond6, ip, i), ifelse(cond6, jp, j), 6] = ifelse(cond6, c6, c8)
            # Pop 7: velocity (-1,+1) -> push to (i-1,j+1)
            cond7 = i > 1 && j < Ny
            f[ifelse(cond7, im, i), ifelse(cond7, jp, j), 7] = ifelse(cond7, c7, c9)
            # Pop 8: velocity (-1,-1) -> push to (i-1,j-1)
            cond8 = i > 1 && j > 1
            f[ifelse(cond8, im, i), ifelse(cond8, jm, j), 8] = ifelse(cond8, c8, c6)
            # Pop 9: velocity (+1,-1) -> push to (i+1,j-1)
            cond9 = i < Nx && j > 1
            f[ifelse(cond9, ip, i), ifelse(cond9, jm, j), 9] = ifelse(cond9, c9, c7)
        end
    end
end

# --- Public API ---

"""
    aa_even_step!(f, is_solid, Nx, Ny, ω)

AA-pattern even step: pull from neighbors, collide, write locally (single buffer).
Must alternate with `aa_odd_step!`.
"""
function aa_even_step!(f, is_solid, Nx, Ny, ω)
    backend = KernelAbstractions.get_backend(f)
    ET = eltype(f)
    kernel! = aa_even_kernel!(backend)
    kernel!(f, is_solid, Nx, Ny, ET(ω); ndrange=(Nx, Ny))
end

"""
    aa_odd_step!(f, is_solid, Nx, Ny, ω)

AA-pattern odd step: read local, collide, push to neighbors (single buffer).
Must alternate with `aa_even_step!`.
"""
function aa_odd_step!(f, is_solid, Nx, Ny, ω)
    backend = KernelAbstractions.get_backend(f)
    ET = eltype(f)
    kernel! = aa_odd_kernel!(backend)
    kernel!(f, is_solid, Nx, Ny, ET(ω); ndrange=(Nx, Ny))
end
