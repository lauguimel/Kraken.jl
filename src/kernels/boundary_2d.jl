using KernelAbstractions

# --- Zou-He velocity boundary condition (2D) ---

"""
    zou_he_velocity_north_2d!(f, ρ, u_wall_x, Nx)

Apply Zou-He velocity BC on the north boundary (j = Ny).
Imposes velocity (u_wall_x, 0) on the top row.

For the lid-driven cavity: u_wall_x = u_lid, applied at j = Ny.
"""
@kernel function zou_he_velocity_north_2d_kernel!(f, u_wall_x, Ny)
    i = @index(Global)
    j = Ny

    @inbounds begin
        T = eltype(f)

        # Known populations (coming from interior or along boundary):
        # f1 (rest), f2 (+x), f4 (-x), f6 (+x,+y), f7 (-x,+y), f3 (+y)
        # Unknown: f5 (0,-1), f8 (-x,-y), f9 (+x,-1)

        f1 = f[i, j, 1]
        f2 = f[i, j, 2]
        f3 = f[i, j, 3]
        f4 = f[i, j, 4]
        f6 = f[i, j, 6]
        f7 = f[i, j, 7]

        # Compute density from equilibrium assumption
        ρ_wall = (f1 + f2 + f4 + T(2) * (f3 + f6 + f7)) / (one(T) + u_wall_x * T(0))
        # For uy_wall = 0: ρ = (f1 + f2 + f4 + 2*(f3 + f6 + f7))

        # Unknown populations (Zou-He)
        f[i, j, 5] = f3 # - T(2.0/3.0) * ρ_wall * uy_wall  (uy_wall = 0)

        f[i, j, 9] = f7 - T(0.5) * (f2 - f4) + T(0.5) * ρ_wall * u_wall_x

        f[i, j, 8] = f6 + T(0.5) * (f2 - f4) - T(0.5) * ρ_wall * u_wall_x
    end
end

function apply_zou_he_north_2d!(f, u_wall_x, Nx, Ny)
    backend = KernelAbstractions.get_backend(f)
    kernel! = zou_he_velocity_north_2d_kernel!(backend)
    kernel!(f, eltype(f)(u_wall_x), Ny; ndrange=(Nx,))
    KernelAbstractions.synchronize(backend)
end

# --- Bounce-back on all 4 walls (2D cavity, no-slip) ---

@kernel function bounce_back_walls_2d_kernel!(f, Nx, Ny, wall_id)
    idx = @index(Global)
    @inbounds begin
        if wall_id == 1  # South (j=1)
            i = idx
            # Populations pointing inward (north): 3,6,7 are reflected
            f[i, 1, 3] = f[i, 1, 5]   # (0,+1) <- (0,-1)
            f[i, 1, 6] = f[i, 1, 8]   # (+1,+1) <- (-1,-1)
            f[i, 1, 7] = f[i, 1, 9]   # (-1,+1) <- (+1,-1)
        elseif wall_id == 2  # West (i=1)
            j = idx
            f[1, j, 2] = f[1, j, 4]   # (+1,0) <- (-1,0)
            f[1, j, 6] = f[1, j, 8]   # (+1,+1) <- (-1,-1)
            f[1, j, 9] = f[1, j, 7]   # (+1,-1) <- (-1,+1)
        elseif wall_id == 3  # East (i=Nx)
            j = idx
            f[Nx, j, 4] = f[Nx, j, 2]
            f[Nx, j, 7] = f[Nx, j, 9]
            f[Nx, j, 8] = f[Nx, j, 6]
        elseif wall_id == 4  # South already handled, this is unused
        end
    end
end

"""
    apply_bounce_back_walls_2d!(f, Nx, Ny)

Apply simple bounce-back on south, west, and east walls of a 2D cavity.
North wall is handled by Zou-He (lid velocity).
"""
function apply_bounce_back_walls_2d!(f, Nx, Ny)
    backend = KernelAbstractions.get_backend(f)
    kernel! = bounce_back_walls_2d_kernel!(backend)

    # South wall (j=1)
    kernel!(f, Nx, Ny, Int32(1); ndrange=(Nx,))
    # West wall (i=1)
    kernel!(f, Nx, Ny, Int32(2); ndrange=(Ny,))
    # East wall (i=Nx)
    kernel!(f, Nx, Ny, Int32(3); ndrange=(Ny,))

    KernelAbstractions.synchronize(backend)
end
