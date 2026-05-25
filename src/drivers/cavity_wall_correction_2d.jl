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

function _logfv_cavity_apply_wall_gradient_correction!(
    dudx, dudy, dvdx, dvdy, ux, uy, u_lid_profile, dx::Real, dy::Real;
    skip_top_corners::Bool=false,
    sync::Bool=true,
)
    backend = KernelAbstractions.get_backend(dudx)
    Nx, Ny = size(dudx)
    T = eltype(dudx)
    zero_x = KernelAbstractions.zeros(backend, T, Nx)
    zero_y = KernelAbstractions.zeros(backend, T, Ny)
    sides = WallGradientSides(zero_x, u_lid_profile, zero_y, zero_y)
    apply_halfway_wall_gradient_correction!(
        dudx, dudy, dvdx, dvdy, ux, uy, sides, dx, dy;
        order=:linear, sync=sync, skip_top_corners=skip_top_corners,
    )
    return nothing
end
