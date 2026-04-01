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

# --- Zou-He velocity BC on bottom face (k = 1) ---

"""
    zou_he_velocity_bottom_3d!(f, ux_w, uy_w, Nx, Ny)

Apply Zou-He velocity BC on the bottom face (k = 1).
Imposes velocity (ux_w, uy_w, 0). Unknown: f6(+z), f12(+x,+z), f13(-x,+z), f16(+y,+z), f17(-y,+z).
"""
@kernel function zou_he_velocity_bottom_3d_kernel!(f, ux_w, uy_w)
    i, j = @index(Global, NTuple)
    k = 1

    @inbounds begin
        T = eltype(f)

        f1  = f[i,j,k,1]
        f2  = f[i,j,k,2]   # +x
        f3  = f[i,j,k,3]   # -x
        f4  = f[i,j,k,4]   # +y
        f5  = f[i,j,k,5]   # -y
        f7  = f[i,j,k,7]   # -z (opposing normal)
        f8  = f[i,j,k,8]   # +x,+y
        f9  = f[i,j,k,9]   # -x,+y
        f10 = f[i,j,k,10]  # +x,-y
        f11 = f[i,j,k,11]  # -x,-y
        f14 = f[i,j,k,14]  # +x,-z (known)
        f15 = f[i,j,k,15]  # -x,-z (known)
        f18 = f[i,j,k,18]  # +y,-z (known)
        f19 = f[i,j,k,19]  # -y,-z (known)

        # uz_w = 0 => rho = sum_parallel + 2*sum_opposing_z
        ρ_wall = (f1 + f2 + f3 + f4 + f5 + f8 + f9 + f10 + f11 +
                  T(2) * (f7 + f14 + f15 + f18 + f19))

        f[i,j,k,6]  = f7   # +z <- -z

        f[i,j,k,12] = f15 - T(0.5) * (f2 - f3) + T(0.5) * ρ_wall * ux_w
        f[i,j,k,13] = f14 + T(0.5) * (f2 - f3) - T(0.5) * ρ_wall * ux_w

        f[i,j,k,16] = f19 - T(0.5) * (f4 - f5) + T(0.5) * ρ_wall * uy_w
        f[i,j,k,17] = f18 + T(0.5) * (f4 - f5) - T(0.5) * ρ_wall * uy_w
    end
end

function apply_zou_he_bottom_3d!(f, ux_w, uy_w, Nx, Ny)
    backend = KernelAbstractions.get_backend(f)
    kernel! = zou_he_velocity_bottom_3d_kernel!(backend)
    kernel!(f, eltype(f)(ux_w), eltype(f)(uy_w); ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

# --- Zou-He velocity BC on west face (i = 1) ---

"""
    zou_he_velocity_west_3d!(f, ux_w, uy_w, uz_w, Ny, Nz)

Apply Zou-He velocity BC on the west face (i = 1).
Unknown populations (cx=+1): f2(+x), f8(+x,+y), f10(+x,-y), f12(+x,+z), f14(+x,-z).
"""
@kernel function zou_he_velocity_west_3d_kernel!(f, ux_w, uy_w, uz_w)
    j, k = @index(Global, NTuple)
    i = 1

    @inbounds begin
        T = eltype(f)

        f1  = f[i,j,k,1]
        f3  = f[i,j,k,3]   # -x (opposing)
        f4  = f[i,j,k,4]   # +y
        f5  = f[i,j,k,5]   # -y
        f6  = f[i,j,k,6]   # +z
        f7  = f[i,j,k,7]   # -z
        f9  = f[i,j,k,9]   # -x,+y (opposing)
        f11 = f[i,j,k,11]  # -x,-y (opposing)
        f13 = f[i,j,k,13]  # -x,+z (opposing)
        f15 = f[i,j,k,15]  # -x,-z (opposing)
        f16 = f[i,j,k,16]  # +y,+z
        f17 = f[i,j,k,17]  # -y,+z
        f18 = f[i,j,k,18]  # +y,-z
        f19 = f[i,j,k,19]  # -y,-z

        ρ_wall = (f1 + f4 + f5 + f6 + f7 + f16 + f17 + f18 + f19 +
                  T(2) * (f3 + f9 + f11 + f13 + f15)) / (one(T) - ux_w)

        f[i,j,k,2]  = f3 + T(2.0/3.0) * ρ_wall * ux_w

        f[i,j,k,8]  = f11 - T(0.5) * (f4 - f5) + T(0.5) * ρ_wall * uy_w +
                       T(1.0/6.0) * ρ_wall * ux_w
        f[i,j,k,10] = f9  + T(0.5) * (f4 - f5) - T(0.5) * ρ_wall * uy_w +
                       T(1.0/6.0) * ρ_wall * ux_w

        f[i,j,k,12] = f15 - T(0.5) * (f6 - f7) + T(0.5) * ρ_wall * uz_w +
                       T(1.0/6.0) * ρ_wall * ux_w
        f[i,j,k,14] = f13 + T(0.5) * (f6 - f7) - T(0.5) * ρ_wall * uz_w +
                       T(1.0/6.0) * ρ_wall * ux_w
    end
end

function apply_zou_he_west_3d!(f, ux_w, uy_w, uz_w, Ny, Nz)
    backend = KernelAbstractions.get_backend(f)
    kernel! = zou_he_velocity_west_3d_kernel!(backend)
    kernel!(f, eltype(f)(ux_w), eltype(f)(uy_w), eltype(f)(uz_w); ndrange=(Ny, Nz))
    KernelAbstractions.synchronize(backend)
end

# --- Zou-He velocity BC on east face (i = Nx) ---

"""
    zou_he_velocity_east_3d!(f, ux_w, uy_w, uz_w, Nx, Ny, Nz)

Apply Zou-He velocity BC on the east face (i = Nx).
Unknown populations (cx=-1): f3(-x), f9(-x,+y), f11(-x,-y), f13(-x,+z), f15(-x,-z).
"""
@kernel function zou_he_velocity_east_3d_kernel!(f, ux_w, uy_w, uz_w, Nx)
    j, k = @index(Global, NTuple)
    i = Nx

    @inbounds begin
        T = eltype(f)

        f1  = f[i,j,k,1]
        f2  = f[i,j,k,2]   # +x (opposing)
        f4  = f[i,j,k,4]   # +y
        f5  = f[i,j,k,5]   # -y
        f6  = f[i,j,k,6]   # +z
        f7  = f[i,j,k,7]   # -z
        f8  = f[i,j,k,8]   # +x,+y (opposing)
        f10 = f[i,j,k,10]  # +x,-y (opposing)
        f12 = f[i,j,k,12]  # +x,+z (opposing)
        f14 = f[i,j,k,14]  # +x,-z (opposing)
        f16 = f[i,j,k,16]  # +y,+z
        f17 = f[i,j,k,17]  # -y,+z
        f18 = f[i,j,k,18]  # +y,-z
        f19 = f[i,j,k,19]  # -y,-z

        ρ_wall = (f1 + f4 + f5 + f6 + f7 + f16 + f17 + f18 + f19 +
                  T(2) * (f2 + f8 + f10 + f12 + f14)) / (one(T) + ux_w)

        f[i,j,k,3]  = f2 - T(2.0/3.0) * ρ_wall * ux_w

        f[i,j,k,9]  = f10 - T(0.5) * (f4 - f5) + T(0.5) * ρ_wall * uy_w -
                       T(1.0/6.0) * ρ_wall * ux_w
        f[i,j,k,11] = f8  + T(0.5) * (f4 - f5) - T(0.5) * ρ_wall * uy_w -
                       T(1.0/6.0) * ρ_wall * ux_w

        f[i,j,k,13] = f14 - T(0.5) * (f6 - f7) + T(0.5) * ρ_wall * uz_w -
                       T(1.0/6.0) * ρ_wall * ux_w
        f[i,j,k,15] = f12 + T(0.5) * (f6 - f7) - T(0.5) * ρ_wall * uz_w -
                       T(1.0/6.0) * ρ_wall * ux_w
    end
end

function apply_zou_he_east_3d!(f, ux_w, uy_w, uz_w, Nx, Ny, Nz)
    backend = KernelAbstractions.get_backend(f)
    kernel! = zou_he_velocity_east_3d_kernel!(backend)
    kernel!(f, eltype(f)(ux_w), eltype(f)(uy_w), eltype(f)(uz_w), Nx; ndrange=(Ny, Nz))
    KernelAbstractions.synchronize(backend)
end

# --- Zou-He velocity BC on south face (j = 1) ---

"""
    zou_he_velocity_south_3d!(f, ux_w, uy_w, uz_w, Nx, Nz)

Apply Zou-He velocity BC on the south face (j = 1).
Unknown populations (cy=+1): f4(+y), f8(+x,+y), f9(-x,+y), f16(+y,+z), f18(+y,-z).
"""
@kernel function zou_he_velocity_south_3d_kernel!(f, ux_w, uy_w, uz_w)
    i, k = @index(Global, NTuple)
    j = 1

    @inbounds begin
        T = eltype(f)

        f1  = f[i,j,k,1]
        f2  = f[i,j,k,2]   # +x
        f3  = f[i,j,k,3]   # -x
        f5  = f[i,j,k,5]   # -y (opposing)
        f6  = f[i,j,k,6]   # +z
        f7  = f[i,j,k,7]   # -z
        f10 = f[i,j,k,10]  # +x,-y (opposing)
        f11 = f[i,j,k,11]  # -x,-y (opposing)
        f12 = f[i,j,k,12]  # +x,+z
        f13 = f[i,j,k,13]  # -x,+z
        f14 = f[i,j,k,14]  # +x,-z
        f15 = f[i,j,k,15]  # -x,-z
        f17 = f[i,j,k,17]  # -y,+z (opposing)
        f19 = f[i,j,k,19]  # -y,-z (opposing)

        ρ_wall = (f1 + f2 + f3 + f6 + f7 + f12 + f13 + f14 + f15 +
                  T(2) * (f5 + f10 + f11 + f17 + f19)) / (one(T) - uy_w)

        f[i,j,k,4]  = f5 + T(2.0/3.0) * ρ_wall * uy_w

        f[i,j,k,8]  = f11 - T(0.5) * (f2 - f3) + T(0.5) * ρ_wall * ux_w +
                       T(1.0/6.0) * ρ_wall * uy_w
        f[i,j,k,9]  = f10 + T(0.5) * (f2 - f3) - T(0.5) * ρ_wall * ux_w +
                       T(1.0/6.0) * ρ_wall * uy_w

        f[i,j,k,16] = f19 - T(0.5) * (f6 - f7) + T(0.5) * ρ_wall * uz_w +
                       T(1.0/6.0) * ρ_wall * uy_w
        f[i,j,k,18] = f17 + T(0.5) * (f6 - f7) - T(0.5) * ρ_wall * uz_w +
                       T(1.0/6.0) * ρ_wall * uy_w
    end
end

function apply_zou_he_south_3d!(f, ux_w, uy_w, uz_w, Nx, Nz)
    backend = KernelAbstractions.get_backend(f)
    kernel! = zou_he_velocity_south_3d_kernel!(backend)
    kernel!(f, eltype(f)(ux_w), eltype(f)(uy_w), eltype(f)(uz_w); ndrange=(Nx, Nz))
    KernelAbstractions.synchronize(backend)
end

# --- Zou-He velocity BC on north face (j = Ny) ---

"""
    zou_he_velocity_north_3d!(f, ux_w, uy_w, uz_w, Nx, Ny, Nz)

Apply Zou-He velocity BC on the north face (j = Ny).
Unknown populations (cy=-1): f5(-y), f10(+x,-y), f11(-x,-y), f17(-y,+z), f19(-y,-z).
"""
@kernel function zou_he_velocity_north_3d_kernel!(f, ux_w, uy_w, uz_w, Ny)
    i, k = @index(Global, NTuple)
    j = Ny

    @inbounds begin
        T = eltype(f)

        f1  = f[i,j,k,1]
        f2  = f[i,j,k,2]   # +x
        f3  = f[i,j,k,3]   # -x
        f4  = f[i,j,k,4]   # +y (opposing)
        f6  = f[i,j,k,6]   # +z
        f7  = f[i,j,k,7]   # -z
        f8  = f[i,j,k,8]   # +x,+y (opposing)
        f9  = f[i,j,k,9]   # -x,+y (opposing)
        f12 = f[i,j,k,12]  # +x,+z
        f13 = f[i,j,k,13]  # -x,+z
        f14 = f[i,j,k,14]  # +x,-z
        f15 = f[i,j,k,15]  # -x,-z
        f16 = f[i,j,k,16]  # +y,+z (opposing)
        f18 = f[i,j,k,18]  # +y,-z (opposing)

        ρ_wall = (f1 + f2 + f3 + f6 + f7 + f12 + f13 + f14 + f15 +
                  T(2) * (f4 + f8 + f9 + f16 + f18)) / (one(T) + uy_w)

        f[i,j,k,5]  = f4 - T(2.0/3.0) * ρ_wall * uy_w

        f[i,j,k,10] = f9  - T(0.5) * (f2 - f3) + T(0.5) * ρ_wall * ux_w -
                       T(1.0/6.0) * ρ_wall * uy_w
        f[i,j,k,11] = f8  + T(0.5) * (f2 - f3) - T(0.5) * ρ_wall * ux_w -
                       T(1.0/6.0) * ρ_wall * uy_w

        f[i,j,k,17] = f18 - T(0.5) * (f6 - f7) + T(0.5) * ρ_wall * uz_w -
                       T(1.0/6.0) * ρ_wall * uy_w
        f[i,j,k,19] = f16 + T(0.5) * (f6 - f7) - T(0.5) * ρ_wall * uz_w -
                       T(1.0/6.0) * ρ_wall * uy_w
    end
end

function apply_zou_he_north_3d!(f, ux_w, uy_w, uz_w, Nx, Ny, Nz)
    backend = KernelAbstractions.get_backend(f)
    kernel! = zou_he_velocity_north_3d_kernel!(backend)
    kernel!(f, eltype(f)(ux_w), eltype(f)(uy_w), eltype(f)(uz_w), Ny; ndrange=(Nx, Nz))
    KernelAbstractions.synchronize(backend)
end

# --- Zou-He pressure outlet on east face (i = Nx) ---

"""
    zou_he_pressure_east_3d!(f, Nx, Ny, Nz; ρ_out=1.0)

Apply Zou-He pressure outlet BC on the east face (i = Nx).
Fixes density ρ_out, computes ux from known populations.
Unknown: f3(-x), f9(-x,+y), f11(-x,-y), f13(-x,+z), f15(-x,-z).
"""
@kernel function zou_he_pressure_east_3d_kernel!(f, Nx, ρ_out)
    j, k = @index(Global, NTuple)
    i = Nx

    @inbounds begin
        T = eltype(f)

        f1  = f[i,j,k,1]
        f2  = f[i,j,k,2]   # +x
        f4  = f[i,j,k,4]   # +y
        f5  = f[i,j,k,5]   # -y
        f6  = f[i,j,k,6]   # +z
        f7  = f[i,j,k,7]   # -z
        f8  = f[i,j,k,8]   # +x,+y
        f10 = f[i,j,k,10]  # +x,-y
        f12 = f[i,j,k,12]  # +x,+z
        f14 = f[i,j,k,14]  # +x,-z
        f16 = f[i,j,k,16]  # +y,+z
        f17 = f[i,j,k,17]  # -y,+z
        f18 = f[i,j,k,18]  # +y,-z
        f19 = f[i,j,k,19]  # -y,-z

        ux = -one(T) + (f1 + f4 + f5 + f6 + f7 + f16 + f17 + f18 + f19 +
              T(2) * (f2 + f8 + f10 + f12 + f14)) / ρ_out

        f[i,j,k,3]  = f2 - T(2.0/3.0) * ρ_out * ux

        f[i,j,k,9]  = f10 - T(0.5) * (f4 - f5) - T(1.0/6.0) * ρ_out * ux
        f[i,j,k,11] = f8  + T(0.5) * (f4 - f5) - T(1.0/6.0) * ρ_out * ux

        f[i,j,k,13] = f14 - T(0.5) * (f6 - f7) - T(1.0/6.0) * ρ_out * ux
        f[i,j,k,15] = f12 + T(0.5) * (f6 - f7) - T(1.0/6.0) * ρ_out * ux
    end
end

function apply_zou_he_pressure_east_3d!(f, Nx, Ny, Nz; ρ_out=1.0)
    backend = KernelAbstractions.get_backend(f)
    kernel! = zou_he_pressure_east_3d_kernel!(backend)
    kernel!(f, Nx, eltype(f)(ρ_out); ndrange=(Ny, Nz))
    KernelAbstractions.synchronize(backend)
end

# --- Zou-He pressure outlet on top face (k = Nz) ---

"""
    zou_he_pressure_top_3d!(f, Nx, Ny, Nz; ρ_out=1.0)

Apply Zou-He pressure outlet BC on the top face (k = Nz).
Fixes density ρ_out, computes uz from known populations.
Unknown: f7(-z), f14(+x,-z), f15(-x,-z), f18(+y,-z), f19(-y,-z).
"""
@kernel function zou_he_pressure_top_3d_kernel!(f, Nz, ρ_out)
    i, j = @index(Global, NTuple)
    k = Nz

    @inbounds begin
        T = eltype(f)

        f1  = f[i,j,k,1]
        f2  = f[i,j,k,2]
        f3  = f[i,j,k,3]
        f4  = f[i,j,k,4]
        f5  = f[i,j,k,5]
        f6  = f[i,j,k,6]   # +z
        f8  = f[i,j,k,8]
        f9  = f[i,j,k,9]
        f10 = f[i,j,k,10]
        f11 = f[i,j,k,11]
        f12 = f[i,j,k,12]  # +x,+z
        f13 = f[i,j,k,13]  # -x,+z
        f16 = f[i,j,k,16]  # +y,+z
        f17 = f[i,j,k,17]  # -y,+z

        uz = -one(T) + (f1 + f2 + f3 + f4 + f5 + f8 + f9 + f10 + f11 +
              T(2) * (f6 + f12 + f13 + f16 + f17)) / ρ_out

        f[i,j,k,7]  = f6 - T(2.0/3.0) * ρ_out * uz

        f[i,j,k,14] = f13 - T(0.5) * (f2 - f3) - T(1.0/6.0) * ρ_out * uz
        f[i,j,k,15] = f12 + T(0.5) * (f2 - f3) - T(1.0/6.0) * ρ_out * uz

        f[i,j,k,18] = f17 - T(0.5) * (f4 - f5) - T(1.0/6.0) * ρ_out * uz
        f[i,j,k,19] = f16 + T(0.5) * (f4 - f5) - T(1.0/6.0) * ρ_out * uz
    end
end

function apply_zou_he_pressure_top_3d!(f, Nx, Ny, Nz; ρ_out=1.0)
    backend = KernelAbstractions.get_backend(f)
    kernel! = zou_he_pressure_top_3d_kernel!(backend)
    kernel!(f, Nz, eltype(f)(ρ_out); ndrange=(Nx, Ny))
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
