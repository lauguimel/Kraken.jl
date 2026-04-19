using KernelAbstractions

# --- Fully periodic D3Q19 streaming ---
#
# Pull-stream version: f_out[i,j,k,q] = f_in[(i,j,k) − c_q (mod N), q]
# Used for homogeneous-shear / Lees-Edwards-like tests where we want no
# wall pollution. Naming: im1 = i−1 wrap, ip1 = i+1 wrap.

@kernel function stream_fully_periodic_3d_kernel!(f_out, @Const(f_in),
                                                    Nx, Ny, Nz)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        im1 = ifelse(i > 1,  i - 1, Nx)
        ip1 = ifelse(i < Nx, i + 1, 1)
        jm1 = ifelse(j > 1,  j - 1, Ny)
        jp1 = ifelse(j < Ny, j + 1, 1)
        km1 = ifelse(k > 1,  k - 1, Nz)
        kp1 = ifelse(k < Nz, k + 1, 1)

        f_out[i,j,k,1]  = f_in[i,   j,   k,   1]   # rest
        # Axial — pull from the side opposite to direction
        f_out[i,j,k,2]  = f_in[im1, j,   k,   2]   # +x
        f_out[i,j,k,3]  = f_in[ip1, j,   k,   3]   # −x
        f_out[i,j,k,4]  = f_in[i,   jm1, k,   4]   # +y
        f_out[i,j,k,5]  = f_in[i,   jp1, k,   5]   # −y
        f_out[i,j,k,6]  = f_in[i,   j,   km1, 6]   # +z
        f_out[i,j,k,7]  = f_in[i,   j,   kp1, 7]   # −z
        # xy edges
        f_out[i,j,k,8]  = f_in[im1, jm1, k,   8]
        f_out[i,j,k,9]  = f_in[ip1, jm1, k,   9]
        f_out[i,j,k,10] = f_in[im1, jp1, k,   10]
        f_out[i,j,k,11] = f_in[ip1, jp1, k,   11]
        # xz edges
        f_out[i,j,k,12] = f_in[im1, j,   km1, 12]
        f_out[i,j,k,13] = f_in[ip1, j,   km1, 13]
        f_out[i,j,k,14] = f_in[im1, j,   kp1, 14]
        f_out[i,j,k,15] = f_in[ip1, j,   kp1, 15]
        # yz edges
        f_out[i,j,k,16] = f_in[i,   jm1, km1, 16]
        f_out[i,j,k,17] = f_in[i,   jp1, km1, 17]
        f_out[i,j,k,18] = f_in[i,   jm1, kp1, 18]
        f_out[i,j,k,19] = f_in[i,   jp1, kp1, 19]
    end
end

"""
    stream_fully_periodic_3d!(f_out, f_in, Nx, Ny, Nz)

Pull-streaming for D3Q19 with periodic wrap on all six faces. Used for
homogeneous tests (Taylor-Green 3D, Lees-Edwards shear, conformation
calibration) where halfway-BB at domain edges would pollute the field.
"""
function stream_fully_periodic_3d!(f_out, f_in, Nx, Ny, Nz)
    backend = KernelAbstractions.get_backend(f_in)
    kernel! = stream_fully_periodic_3d_kernel!(backend)
    kernel!(f_out, f_in, Nx, Ny, Nz; ndrange=(Nx, Ny, Nz))
    KernelAbstractions.synchronize(backend)
end
