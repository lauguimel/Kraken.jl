using KernelAbstractions

# --- Stream kernel with periodic x, bounce-back walls at y boundaries ---

@kernel function stream_periodic_x_wall_y_2d_kernel!(f_out, @Const(f_in), Nx, Ny)
    i, j = @index(Global, NTuple)

    @inbounds begin
        # Rest population (q=1): no movement
        fp1 = f_in[i, j, 1]

        # q=2: E (+1,0) — pull from (i-1, j), periodic in x
        i_src = ifelse(i > 1, i - 1, Nx)
        fp2 = f_in[i_src, j, 2]

        # q=3: N (0,+1) — pull from (i, j-1), bounce-back at bottom wall
        fp3 = ifelse(j > 1, f_in[i, j-1, 3], f_in[i, j, 5])

        # q=4: W (-1,0) — pull from (i+1, j), periodic in x
        i_src = ifelse(i < Nx, i + 1, 1)
        fp4 = f_in[i_src, j, 4]

        # q=5: S (0,-1) — pull from (i, j+1), bounce-back at top wall
        fp5 = ifelse(j < Ny, f_in[i, j+1, 5], f_in[i, j, 3])

        # q=6: NE (+1,+1) — pull from (i-1, j-1)
        i_src = ifelse(i > 1, i - 1, Nx)
        fp6 = ifelse(j > 1, f_in[i_src, j-1, 6], f_in[i, j, 8])

        # q=7: NW (-1,+1) — pull from (i+1, j-1)
        i_src = ifelse(i < Nx, i + 1, 1)
        fp7 = ifelse(j > 1, f_in[i_src, j-1, 7], f_in[i, j, 9])

        # q=8: SW (-1,-1) — pull from (i+1, j+1)
        i_src = ifelse(i < Nx, i + 1, 1)
        fp8 = ifelse(j < Ny, f_in[i_src, j+1, 8], f_in[i, j, 6])

        # q=9: SE (+1,-1) — pull from (i-1, j+1)
        i_src = ifelse(i > 1, i - 1, Nx)
        fp9 = ifelse(j < Ny, f_in[i_src, j+1, 9], f_in[i, j, 7])

        f_out[i,j,1] = fp1
        f_out[i,j,2] = fp2; f_out[i,j,3] = fp3
        f_out[i,j,4] = fp4; f_out[i,j,5] = fp5
        f_out[i,j,6] = fp6; f_out[i,j,7] = fp7
        f_out[i,j,8] = fp8; f_out[i,j,9] = fp9
    end
end

# --- Public API ---

function stream_periodic_x_wall_y_2d!(f_out, f_in, Nx, Ny)
    backend = KernelAbstractions.get_backend(f_in)
    kernel! = stream_periodic_x_wall_y_2d_kernel!(backend)
    kernel!(f_out, f_in, Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

# --- Fully periodic streaming (for Taylor-Green) ---

@kernel function stream_fully_periodic_2d_kernel!(f_out, @Const(f_in), Nx, Ny)
    i, j = @index(Global, NTuple)

    @inbounds begin
        ip = ifelse(i > 1,  i - 1, Nx)
        im = ifelse(i < Nx, i + 1, 1)
        jp = ifelse(j > 1,  j - 1, Ny)
        jm = ifelse(j < Ny, j + 1, 1)

        f_out[i,j,1] = f_in[i,  j,  1]
        f_out[i,j,2] = f_in[ip, j,  2]
        f_out[i,j,3] = f_in[i,  jp, 3]
        f_out[i,j,4] = f_in[im, j,  4]
        f_out[i,j,5] = f_in[i,  jm, 5]
        f_out[i,j,6] = f_in[ip, jp, 6]
        f_out[i,j,7] = f_in[im, jp, 7]
        f_out[i,j,8] = f_in[im, jm, 8]
        f_out[i,j,9] = f_in[ip, jm, 9]
    end
end

function stream_fully_periodic_2d!(f_out, f_in, Nx, Ny)
    backend = KernelAbstractions.get_backend(f_in)
    kernel! = stream_fully_periodic_2d_kernel!(backend)
    kernel!(f_out, f_in, Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end
