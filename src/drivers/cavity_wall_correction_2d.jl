using KernelAbstractions

# =====================================================================
# Closed lid-driven cavity (axis-aligned all-walls)
#
# This is a CLOSED-domain coupled driver distinct from the open-x
# `_run_viscoelastic_logfv_step_channel_coupled_2d` core. The four
# sides are walls for the LBM solvent (3 fixed + 1 moving lid Zou-He)
# and for the log-FV polymer (`logfv_wallxwally_bcspec_2d`).
#
# The moving lid drives `du/dy` at the top row; the standard solid-aware
# velocity-gradient kernel only uses interior cells, so we explicitly
# overwrite the top-row `du/dy` (and side-wall `du/dx`, `dv/dx`,
# `dv/dy`) using a half-cell finite difference against the Dirichlet
# wall velocity. This is the only piece of the cavity pipeline that
# differs operator-wise from the step_channel core.
# =====================================================================

@kernel function _logfv_cavity_lid_profile_kernel!(
    profile, t_phys, u_max, ramp_start, ramp_steepness, Nx,
)
    i = @index(Global)
    T = eltype(profile)
    @inbounds if 1 <= i <= Nx
        x_phys = (T(i) - T(0.5)) / T(Nx)
        ramp = one(T) + tanh(ramp_steepness * (t_phys - ramp_start))
        shape = x_phys * x_phys * (one(T) - x_phys) * (one(T) - x_phys)
        profile[i] = T(8) * u_max * ramp * shape
    end
end

function _logfv_cavity_update_lid_profile!(
    profile, t_phys::Real, u_max::Real, ramp_start::Real, ramp_steepness::Real;
    sync::Bool=true,
)
    backend = KernelAbstractions.get_backend(profile)
    T = eltype(profile)
    Nx = length(profile)
    kernel! = _logfv_cavity_lid_profile_kernel!(backend)
    kernel!(
        profile, T(t_phys), T(u_max), T(ramp_start), T(ramp_steepness), Nx;
        ndrange=(Nx,),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function _logfv_cavity_wall_gradient_correction_kernel!(
    dudx, dudy, dvdx, dvdy,
    @Const(ux), @Const(uy), @Const(u_lid_profile),
    inv_dx_half, inv_dy_half, Nx, Ny, skip_top_corners,
)
    i, j = @index(Global, NTuple)
    T = eltype(dudx)
    @inbounds begin
        if i <= Nx && j <= Ny
            # North wall (j == Ny): moving lid, tangential ux = u_lid(x), v = 0
            # Pattern A: opt out only the moving-lid correction at the two top corner cells.
            if j == Ny && !(skip_top_corners && (i == 1 || i == Nx))
                dudy[i, j] = (u_lid_profile[i] - ux[i, j]) * inv_dy_half
                dvdy[i, j] = (zero(T) - uy[i, j]) * inv_dy_half
            end
            # South wall (j == 1): fixed, ux = v = 0
            if j == 1
                dudy[i, j] = (ux[i, j] - zero(T)) * inv_dy_half
                dvdy[i, j] = (uy[i, j] - zero(T)) * inv_dy_half
            end
            # West wall (i == 1): fixed
            if i == 1
                dudx[i, j] = (ux[i, j] - zero(T)) * inv_dx_half
                dvdx[i, j] = (uy[i, j] - zero(T)) * inv_dx_half
            end
            # East wall (i == Nx): fixed
            if i == Nx
                dudx[i, j] = (zero(T) - ux[i, j]) * inv_dx_half
                dvdx[i, j] = (zero(T) - uy[i, j]) * inv_dx_half
            end
        end
    end
end

function _logfv_cavity_apply_wall_gradient_correction!(
    dudx, dudy, dvdx, dvdy, ux, uy, u_lid_profile, dx::Real, dy::Real;
    skip_top_corners::Bool=false,
    sync::Bool=true,
)
    backend = KernelAbstractions.get_backend(dudx)
    Nx, Ny = size(dudx)
    T = eltype(dudx)
    kernel! = _logfv_cavity_wall_gradient_correction_kernel!(backend)
    kernel!(
        dudx, dudy, dvdx, dvdy, ux, uy, u_lid_profile,
        T(2) / T(dx), T(2) / T(dy), Nx, Ny, skip_top_corners;
        ndrange=(Nx, Ny),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end
