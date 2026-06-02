"""
    run_viscoelastic_logfv_frozen_circle_tangential_shear_cde_2d(; kwargs...)

Run a curved-wall numerical-gradient canary on a coherent FVFD circle.

The imposed field is tangential shear around a stationary circle,
`u = shear_rate * (r - radius) * e_theta`, so it is exactly zero on the
embedded wall. The driver uses the FVFD embedded velocity-gradient operator,
compares it with the analytical Cartesian gradient at the FV control-volume
point, then initializes the local Oldroyd-B steady conformation for the
numerical gradient and checks that the log-C source and stress reconstruction
recover the corresponding `tau`. Cut-cell values are evaluated at the sampled
fluid centroid used by the circle lowering.

The conformation is spatially varying, so this is intentionally a source and
gradient canary, not an advection-preservation canary.
"""
function run_viscoelastic_logfv_frozen_circle_tangential_shear_cde_2d(;
    Nx::Integer=64,
    Ny::Integer=64,
    cx::Real=Nx / 2,
    cy::Real=Ny / 2,
    radius::Real=min(Nx, Ny) / 6,
    shear_rate::Real=0.006,
    lambda::Real=2.0,
    prefactor::Real=0.02,
    dt::Real=0.001,
    samples::Integer=32,
    backend=KernelAbstractions.CPU(),
    T=Float64,
)
    Nx >= 16 || throw(ArgumentError("Nx must be >= 16"))
    Ny >= 16 || throw(ArgumentError("Ny must be >= 16"))
    samples > 0 || throw(ArgumentError("samples must be positive"))
    radius > 0 || throw(ArgumentError("radius must be positive"))
    lambda > 0 || throw(ArgumentError("lambda must be positive"))
    dt > 0 || throw(ArgumentError("dt must be positive"))

    Nx_i = Int(Nx)
    Ny_i = Int(Ny)
    cx_t = T(cx)
    cy_t = T(cy)
    radius_t = T(radius)
    shear_t = T(shear_rate)
    lambda_t = T(lambda)
    prefactor_t = T(prefactor)
    dt_t = T(dt)

    bc = FVFDDomainBC2D(;
        west=:open, east=:open, south=:open, north=:open,
    )
    embedded_h = fvfd_embedded_boundary_from_circle_2d(
        Nx_i, Ny_i, cx_t, cy_t, radius_t; FT=T, samples=samples,
    )
    is_solid_h = falses(Nx_i, Ny_i)
    ux_h = zeros(T, Nx_i, Ny_i)
    uy_h = zeros(T, Nx_i, Ny_i)
    dudx_ref = zeros(T, Nx_i, Ny_i)
    dudy_ref = zeros(T, Nx_i, Ny_i)
    dvdx_ref = zeros(T, Nx_i, Ny_i)
    dvdy_ref = zeros(T, Nx_i, Ny_i)

    solid_tol = sqrt(eps(T))
    @inbounds for j in 1:Ny_i, i in 1:Nx_i
        x_center_abs = T(i) - T(0.5)
        y_center_abs = T(j) - T(0.5)
        x_eval_abs = x_center_abs
        y_eval_abs = y_center_abs
        if embedded_h.cut_count[i, j] > 0
            sampled = _fvfd_circle_cell_fluid_moments_sampled_2d(
                T(i - 1), T(j - 1), cx_t, cy_t, radius_t, samples, T,
            )
            x_eval_abs = sampled.centroid_x
            y_eval_abs = sampled.centroid_y
        end
        x = x_eval_abs - cx_t
        y = y_eval_abs - cy_t
        r = hypot(x, y)
        is_solid = embedded_h.cell_fraction[i, j] <= solid_tol
        is_solid_h[i, j] = is_solid
        if r > eps(T)
            h = one(T) - radius_t / r
            if !is_solid
                ux_h[i, j] = -shear_t * y * h
                uy_h[i, j] = shear_t * x * h
            end
            dudx_ref[i, j], dudy_ref[i, j], dvdx_ref[i, j], dvdy_ref[i, j] =
                _logfv_circle_tangential_shear_gradient_2d(x, y, radius_t, shear_t)
        end
    end

    geometry_h = FVFDGeometry2D(
        is_solid_h, embedded_h, FVFDPatch2D(one(T), one(T)), bc,
    )
    geometry = fvfd_transfer_geometry_2d(geometry_h, backend, T)

    ux = KernelAbstractions.allocate(backend, T, Nx_i, Ny_i)
    uy = KernelAbstractions.allocate(backend, T, Nx_i, Ny_i)
    copyto!(ux, ux_h)
    copyto!(uy, uy_h)
    dudx = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    dudy = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    dvdx = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    dvdy = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    fvfd_velocity_gradient_embedded_2d!(
        dudx, dudy, dvdx, dvdy, ux, uy, geometry; sync=true,
    )

    dudx_cpu = Array(dudx)
    dudy_cpu = Array(dudy)
    dvdx_cpu = Array(dvdx)
    dvdy_cpu = Array(dvdy)

    psixx_h = zeros(T, Nx_i, Ny_i)
    psixy_h = zeros(T, Nx_i, Ny_i)
    psiyy_h = zeros(T, Nx_i, Ny_i)
    cxx_h = ones(T, Nx_i, Ny_i)
    cxy_h = zeros(T, Nx_i, Ny_i)
    cyy_h = ones(T, Nx_i, Ny_i)
    tauxx_ref = zeros(T, Nx_i, Ny_i)
    tauxy_ref = zeros(T, Nx_i, Ny_i)
    tauyy_ref = zeros(T, Nx_i, Ny_i)

    max_velocity_gradient_error = 0.0
    max_cut_velocity_gradient_error = 0.0
    max_bulk_velocity_gradient_error = 0.0
    min_c_eig = Inf
    max_c_trace = 0.0
    fluid_cells = 0
    cut_cells = 0
    @inbounds for j in 1:Ny_i, i in 1:Nx_i
        is_solid_h[i, j] && continue
        fluid_cells += 1
        is_cut = embedded_h.cut_count[i, j] > 0
        cut_cells += is_cut ? 1 : 0
        grad_error = max(
            abs(dudx_cpu[i, j] - dudx_ref[i, j]),
            abs(dudy_cpu[i, j] - dudy_ref[i, j]),
            abs(dvdx_cpu[i, j] - dvdx_ref[i, j]),
            abs(dvdy_cpu[i, j] - dvdy_ref[i, j]),
        )
        if 1 < i < Nx_i && 1 < j < Ny_i
            max_velocity_gradient_error = max(max_velocity_gradient_error, grad_error)
            if is_cut
                max_cut_velocity_gradient_error = max(max_cut_velocity_gradient_error, grad_error)
            else
                max_bulk_velocity_gradient_error = max(max_bulk_velocity_gradient_error, grad_error)
            end
        end

        cxx, cxy, cyy = _logfv_oldroydb_steady_conformation_from_gradient_2d(
            dudx_cpu[i, j], dudy_cpu[i, j], dvdx_cpu[i, j], dvdy_cpu[i, j],
            lambda_t,
        )
        cxx_h[i, j] = cxx
        cxy_h[i, j] = cxy
        cyy_h[i, j] = cyy
        psixx_h[i, j], psixy_h[i, j], psiyy_h[i, j] =
            logfv_log_spd_sym2_2d(cxx, cxy, cyy)
        tauxx_ref[i, j] = prefactor_t * (cxx - one(T))
        tauxy_ref[i, j] = prefactor_t * cxy
        tauyy_ref[i, j] = prefactor_t * (cyy - one(T))
        min_c_eig = min(min_c_eig, logfv_min_eig_sym2_2d(cxx, cxy, cyy))
        max_c_trace = max(max_c_trace, Float64(cxx + cyy))
    end

    psixx = KernelAbstractions.allocate(backend, T, Nx_i, Ny_i)
    psixy = KernelAbstractions.allocate(backend, T, Nx_i, Ny_i)
    psiyy = KernelAbstractions.allocate(backend, T, Nx_i, Ny_i)
    copyto!(psixx, psixx_h)
    copyto!(psixy, psixy_h)
    copyto!(psiyy, psiyy_h)
    psixx_next = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    psixy_next = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    psiyy_next = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    tauxx = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    tauxy = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    tauyy = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)

    logfv_step_constitutive_log_2d!(
        psixx_next, psixy_next, psiyy_next,
        psixx, psixy, psiyy,
        dudx, dudy, dvdx, dvdy,
        lambda_t, dt_t, LOGFV_MODEL_OLDROYDB, zero(T); sync=false,
    )
    logfv_stress_from_log_2d!(
        tauxx, tauxy, tauyy, psixx_next, psixy_next, psiyy_next, prefactor_t;
        sync=false,
    )
    KernelAbstractions.synchronize(backend)

    psixx_cpu = Array(psixx_next)
    psixy_cpu = Array(psixy_next)
    psiyy_cpu = Array(psiyy_next)
    tauxx_cpu = Array(tauxx)
    tauxy_cpu = Array(tauxy)
    tauyy_cpu = Array(tauyy)
    cxx_cpu = similar(cxx_h)
    cxy_cpu = similar(cxy_h)
    cyy_cpu = similar(cyy_h)
    max_c_error = 0.0
    max_psi_error = 0.0
    max_tau_error = 0.0
    @inbounds for j in 1:Ny_i, i in 1:Nx_i
        cxx, cxy, cyy = logfv_exp_sym2_2d(
            psixx_cpu[i, j], psixy_cpu[i, j], psiyy_cpu[i, j],
        )
        cxx_cpu[i, j] = cxx
        cxy_cpu[i, j] = cxy
        cyy_cpu[i, j] = cyy
        is_solid_h[i, j] && continue
        max_c_error = max(
            max_c_error,
            abs(cxx - cxx_h[i, j]),
            abs(cxy - cxy_h[i, j]),
            abs(cyy - cyy_h[i, j]),
        )
        max_psi_error = max(
            max_psi_error,
            abs(psixx_cpu[i, j] - psixx_h[i, j]),
            abs(psixy_cpu[i, j] - psixy_h[i, j]),
            abs(psiyy_cpu[i, j] - psiyy_h[i, j]),
        )
        max_tau_error = max(
            max_tau_error,
            abs(tauxx_cpu[i, j] - tauxx_ref[i, j]),
            abs(tauxy_cpu[i, j] - tauxy_ref[i, j]),
            abs(tauyy_cpu[i, j] - tauyy_ref[i, j]),
        )
    end

    return (;
        flow=:circle_tangential_shear,
        Nx=Nx_i,
        Ny=Ny_i,
        cx=Float64(cx_t),
        cy=Float64(cy_t),
        radius=Float64(radius_t),
        shear_rate=Float64(shear_t),
        lambda=Float64(lambda_t),
        prefactor=Float64(prefactor_t),
        dt=Float64(dt_t),
        samples=Int(samples),
        fluid_cells,
        cut_cells,
        geometry=geometry_h,
        ux=Array(ux),
        uy=Array(uy),
        dudx=dudx_cpu,
        dudy=dudy_cpu,
        dvdx=dvdx_cpu,
        dvdy=dvdy_cpu,
        reference_gradient=(;
            dudx=dudx_ref, dudy=dudy_ref, dvdx=dvdx_ref, dvdy=dvdy_ref,
        ),
        psixx=psixx_cpu,
        psixy=psixy_cpu,
        psiyy=psiyy_cpu,
        cxx=cxx_cpu,
        cxy=cxy_cpu,
        cyy=cyy_cpu,
        tauxx=tauxx_cpu,
        tauxy=tauxy_cpu,
        tauyy=tauyy_cpu,
        reference=(;
            cxx=cxx_h,
            cxy=cxy_h,
            cyy=cyy_h,
            psixx=psixx_h,
            psixy=psixy_h,
            psiyy=psiyy_h,
            tauxx=tauxx_ref,
            tauxy=tauxy_ref,
            tauyy=tauyy_ref,
        ),
        max_velocity_gradient_error,
        max_cut_velocity_gradient_error,
        max_bulk_velocity_gradient_error,
        max_c_error,
        max_psi_error,
        max_tau_error,
        min_c_eig,
        max_c_trace,
    )
end

