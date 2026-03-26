using KernelAbstractions

# --- Zou-He velocity BC on top face (k = Nz) for 3D cavity ---

"""
    zou_he_velocity_top_3d!(f, u_wall_x, Nx, Ny, Nz)

Apply Zou-He velocity BC on the top face (k = Nz).
Imposes velocity (u_wall_x, 0, 0).
"""
@kernel function zou_he_velocity_top_3d_kernel!(f, u_wall_x, Nz)
    i, j = @index(Global, NTuple)
    k = Nz

    @inbounds begin
        T = eltype(f)

        # Known: populations not pointing into the domain from k=Nz
        # rest + axis-aligned in x,y + edges in xy plane + populations with +z
        f1  = f[i,j,k,1]
        f2  = f[i,j,k,2]   # +x
        f3  = f[i,j,k,3]   # -x
        f4  = f[i,j,k,4]   # +y
        f5  = f[i,j,k,5]   # -y
        f6  = f[i,j,k,6]   # +z (known, pointing outward)
        f8  = f[i,j,k,8]   # +x,+y
        f9  = f[i,j,k,9]   # -x,+y
        f10 = f[i,j,k,10]  # +x,-y
        f11 = f[i,j,k,11]  # -x,-y
        f12 = f[i,j,k,12]  # +x,+z (known)
        f13 = f[i,j,k,13]  # -x,+z (known)
        f16 = f[i,j,k,16]  # +y,+z (known)
        f17 = f[i,j,k,17]  # -y,+z (known)

        # Unknown: f7 (-z), f14 (+x,-z), f15 (-x,-z), f18 (+y,-z), f19 (-y,-z)

        # Density from known populations (uz_wall = 0)
        ρ_wall = (f1 + f2 + f3 + f4 + f5 + f8 + f9 + f10 + f11 +
                  T(2) * (f6 + f12 + f13 + f16 + f17))

        # Zou-He corrections
        f[i,j,k,7]  = f6   # -z <- +z

        f[i,j,k,14] = f13 - T(0.5) * (f2 - f3) + T(0.5) * ρ_wall * u_wall_x
        f[i,j,k,15] = f12 + T(0.5) * (f2 - f3) - T(0.5) * ρ_wall * u_wall_x

        f[i,j,k,18] = f17 - T(0.5) * (f4 - f5)
        f[i,j,k,19] = f16 + T(0.5) * (f4 - f5)
    end
end

function apply_zou_he_top_3d!(f, u_wall_x, Nx, Ny, Nz)
    backend = KernelAbstractions.get_backend(f)
    kernel! = zou_he_velocity_top_3d_kernel!(backend)
    kernel!(f, eltype(f)(u_wall_x), Nz; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

# --- Bounce-back on 5 walls of 3D cavity (all except top) ---

@kernel function bounce_back_face_3d_kernel!(f, N1, N2, Nfull, face_id)
    a, b = @index(Global, NTuple)
    @inbounds begin
        if face_id == 1  # Bottom (k=1): reflect populations pointing +z
            i, j, k = a, b, 1
            f[i,j,k,6]  = f[i,j,k,7]    # +z <- -z
            f[i,j,k,12] = f[i,j,k,15]   # +x,+z <- -x,-z
            f[i,j,k,13] = f[i,j,k,14]   # -x,+z <- +x,-z
            f[i,j,k,16] = f[i,j,k,19]   # +y,+z <- -y,-z
            f[i,j,k,17] = f[i,j,k,18]   # -y,+z <- +y,-z
        elseif face_id == 2  # South (j=1)
            i, k = a, b
            j = 1
            f[i,j,k,4]  = f[i,j,k,5]    # +y <- -y
            f[i,j,k,8]  = f[i,j,k,11]   # +x,+y <- -x,-y
            f[i,j,k,9]  = f[i,j,k,10]   # -x,+y <- +x,-y
            f[i,j,k,16] = f[i,j,k,19]   # +y,+z <- -y,-z
            f[i,j,k,18] = f[i,j,k,17]   # +y,-z <- -y,+z
        elseif face_id == 3  # North (j=Ny)
            i, k = a, b
            j = N1  # Ny
            f[i,j,k,5]  = f[i,j,k,4]
            f[i,j,k,10] = f[i,j,k,9]
            f[i,j,k,11] = f[i,j,k,8]
            f[i,j,k,17] = f[i,j,k,18]
            f[i,j,k,19] = f[i,j,k,16]
        elseif face_id == 4  # West (i=1)
            j, k = a, b
            i = 1
            f[i,j,k,2]  = f[i,j,k,3]    # +x <- -x
            f[i,j,k,8]  = f[i,j,k,11]   # +x,+y <- -x,-y
            f[i,j,k,10] = f[i,j,k,9]    # +x,-y <- -x,+y
            f[i,j,k,12] = f[i,j,k,15]   # +x,+z <- -x,-z
            f[i,j,k,14] = f[i,j,k,13]   # +x,-z <- -x,+z
        elseif face_id == 5  # East (i=Nx)
            j, k = a, b
            i = Nfull  # Nx
            f[i,j,k,3]  = f[i,j,k,2]
            f[i,j,k,9]  = f[i,j,k,10]
            f[i,j,k,11] = f[i,j,k,8]
            f[i,j,k,13] = f[i,j,k,14]
            f[i,j,k,15] = f[i,j,k,12]
        end
    end
end

"""
    apply_bounce_back_walls_3d!(f, Nx, Ny, Nz)

Apply bounce-back on bottom, south, north, west, east faces.
Top face (k=Nz) is handled by Zou-He.
"""
function apply_bounce_back_walls_3d!(f, Nx, Ny, Nz)
    backend = KernelAbstractions.get_backend(f)
    kernel! = bounce_back_face_3d_kernel!(backend)

    # Bottom (k=1): ndrange = (Nx, Ny)
    kernel!(f, Nx, Ny, Nx, Int32(1); ndrange=(Nx, Ny))
    # South (j=1): ndrange = (Nx, Nz)
    kernel!(f, Ny, Nz, Nx, Int32(2); ndrange=(Nx, Nz))
    # North (j=Ny): ndrange = (Nx, Nz)
    kernel!(f, Ny, Nz, Nx, Int32(3); ndrange=(Nx, Nz))
    # West (i=1): ndrange = (Ny, Nz)
    kernel!(f, Ny, Nz, Nx, Int32(4); ndrange=(Ny, Nz))
    # East (i=Nx): ndrange = (Ny, Nz)
    kernel!(f, Ny, Nz, Nx, Int32(5); ndrange=(Ny, Nz))

    KernelAbstractions.synchronize(backend)
end
