using KernelAbstractions

@inline function _logfv_channel_shear(flow::Symbol, y, height, umax, uwall)
    if flow === :poiseuille
        return 4 * umax / height * (1 - 2 * y / height)
    elseif flow === :couette
        return uwall / height
    else
        error("unsupported log-FV channel flow $(flow); expected :poiseuille or :couette")
    end
end

function _logfv_compute_bsd_drag_2d(
    dudx, dudy, dvdx, dvdy, q_wall, Nx::Integer, Ny::Integer;
    cx::Real,
    cy::Real,
    radius::Real,
    zeta_nu_p::Real,
    reconstruction_order::Integer=2,
)
    zeta_nu_p_f = Float64(zeta_nu_p)
    zeta_nu_p_f == 0.0 && return (Fx=0.0, Fy=0.0)

    dudx_h = Array(dudx)
    dudy_h = Array(dudy)
    dvdx_h = Array(dvdx)
    dvdy_h = Array(dvdy)
    Nx_i = Int(Nx)
    Ny_i = Int(Ny)
    tau_bsd_xx = Matrix{Float64}(undef, Nx_i, Ny_i)
    tau_bsd_xy = Matrix{Float64}(undef, Nx_i, Ny_i)
    tau_bsd_yy = Matrix{Float64}(undef, Nx_i, Ny_i)
    @inbounds for j in 1:Ny_i, i in 1:Nx_i
        tau_bsd_xx[i, j] = 2.0 * zeta_nu_p_f * Float64(dudx_h[i, j])
        tau_bsd_xy[i, j] = zeta_nu_p_f * (Float64(dudy_h[i, j]) + Float64(dvdx_h[i, j]))
        tau_bsd_yy[i, j] = 2.0 * zeta_nu_p_f * Float64(dvdy_h[i, j])
    end

    return compute_polymeric_drag_2d(
        tau_bsd_xx, tau_bsd_xy, tau_bsd_yy, q_wall, Nx_i, Ny_i;
        cx=Float64(cx),
        cy=Float64(cy),
        radius=Float64(radius),
        extrapolate=true,
        reconstruction_order,
    )
end

function _logfv_first_nonfinite_field_2d(is_solid_h, fields::Pair{Symbol,<:Any}...)
    for pair in fields
        name = pair.first
        values = Array(pair.second)
        @inbounds for j in axes(values, 2), i in axes(values, 1)
            is_solid_h[i, j] && continue
            value = Float64(values[i, j])
            if !isfinite(value)
                return (finite=false, field=name, i=i, j=j)
            end
        end
    end
    return (finite=true, field=:none, i=0, j=0)
end

function _logfv_bsd_dual_path_relative_l2_2d(
    fx_active, fy_active, fx_alt, fy_alt, is_solid_h, backend,
)
    KernelAbstractions.synchronize(backend)
    fx_active_h = Array(fx_active)
    fy_active_h = Array(fy_active)
    fx_alt_h = Array(fx_alt)
    fy_alt_h = Array(fy_alt)
    Nx, Ny = size(fx_active_h)
    active_sum = 0.0
    delta_sum = 0.0
    @inbounds for j in 2:(Ny - 1), i in 2:(Nx - 1)
        is_solid_h[i, j] && continue
        ax = Float64(fx_active_h[i, j])
        ay = Float64(fy_active_h[i, j])
        dx = ax - Float64(fx_alt_h[i, j])
        dy = ay - Float64(fy_alt_h[i, j])
        active_sum += ax * ax + ay * ay
        delta_sum += dx * dx + dy * dy
    end
    active_l2 = sqrt(active_sum)
    delta_l2 = sqrt(delta_sum)
    return active_l2 > 0.0 ? delta_l2 / active_l2 : (delta_l2 == 0.0 ? 0.0 : Inf)
end

function _logfv_normalize_polymer_symbol(polymer_model)
    raw = lowercase(String(polymer_model))
    normalized = Symbol(replace(raw, '-' => '_'))
    normalized in (:oldroydb, :oldroyd_b, :oldroyd_benchmark, :ob) && return :oldroydb
    normalized in (:fenep, :fene_p, :fene_peterlin) && return :fenep
    throw(ArgumentError("unsupported log-FV polymer_model=$(polymer_model); expected :oldroydb or :fenep"))
end

function _logfv_polymer_model_config(polymer_model, L_max, ::Type{T}) where {T<:AbstractFloat}
    model_symbol = if polymer_model isa Symbol || polymer_model isa AbstractString
        _logfv_normalize_polymer_symbol(polymer_model)
    elseif polymer_model isa FENEPPolymer
        L_max = polymer_model.L_max
        :fenep
    elseif polymer_model isa AbstractPolymerModel
        :oldroydb
    elseif hasproperty(polymer_model, :L_max)
        L_max = getproperty(polymer_model, :L_max)
        :fenep
    elseif hasproperty(polymer_model, :lambda)
        :oldroydb
    else
        throw(ArgumentError("unsupported log-FV polymer_model=$(polymer_model); expected Symbol/String or polymer model object"))
    end

    model_code = logfv_constitutive_model_code(model_symbol)
    if model_symbol === :fenep
        L_max_t = T(L_max)
        isfinite(Float64(L_max_t)) || throw(ArgumentError("FENE-P requires finite L_max"))
        L_max_t > zero(T) || throw(ArgumentError("FENE-P requires positive L_max"))
        L2_t = L_max_t * L_max_t
        L2_t > T(2) || throw(ArgumentError("FENE-P requires L_max^2 > 2 in 2D"))
        return (; polymer_model=model_symbol, model_code, L_max=Float64(L_max_t), L2=L2_t)
    end
    return (; polymer_model=model_symbol, model_code, L_max=0.0, L2=zero(T))
end

function _logfv_conformation_diagnostics_2d(psixx, psixy, psiyy, is_solid_h, model_code, L2)
    Nx, Ny = size(psixx)
    min_c_eig = Inf
    max_c_trace = 0.0
    min_fene_denom = model_code == LOGFV_MODEL_FENEP ? Inf : NaN
    max_fene_factor = model_code == LOGFV_MODEL_FENEP ? -Inf : 1.0
    for j in 1:Ny, i in 1:Nx
        if !is_solid_h[i, j]
            cxx, cxy, cyy = logfv_exp_sym2_2d(psixx[i, j], psixy[i, j], psiyy[i, j])
            trc = Float64(cxx + cyy)
            min_c_eig = min(min_c_eig, logfv_min_eig_sym2_2d(cxx, cxy, cyy))
            max_c_trace = max(max_c_trace, trc)
            if model_code == LOGFV_MODEL_FENEP
                denom = Float64(L2) - trc
                min_fene_denom = min(min_fene_denom, denom)
                max_fene_factor = max(max_fene_factor, (Float64(L2) - 2.0) / denom)
            end
        end
    end
    return (; min_c_eig, max_c_trace, min_fene_denom, max_fene_factor)
end

function _logfv_embedded_circle_normal_alignment_2d(embedded, cx::Real, cy::Real)
    min_alignment = Inf
    sum_alignment = 0.0
    samples = 0
    cx_f = Float64(cx)
    cy_f = Float64(cy)
    @inbounds for idx in CartesianIndices(embedded.cut_count)
        embedded.cut_count[idx] > 0 || continue
        i, j = Tuple(idx)
        x = (Float64(i) - 0.5) - cx_f
        y = (Float64(j) - 0.5) - cy_f
        r = hypot(x, y)
        r > 0 || continue
        alignment = Float64(embedded.wall_nx[idx]) * x / r +
                    Float64(embedded.wall_ny[idx]) * y / r
        min_alignment = min(min_alignment, alignment)
        sum_alignment += alignment
        samples += 1
    end
    if samples == 0
        return (min=NaN, mean=NaN, samples=0)
    end
    return (min=min_alignment, mean=sum_alignment / samples, samples)
end

function _run_viscoelastic_logfv_step_channel_coupled_2d(
    geom_h;
    shear_length::Real,
    nu_s::Real=0.08,
    nu_p::Real=0.02,
    lambda::Real=5.0,
    polymer_model=:oldroydb,
    L_max::Real=10.0,
    u_mean::Real=0.01,
    Fx_body::Real=2e-7,
    bsd_fraction::Real=1.0,
    polymer_substeps=:auto,
    subcycle_relative_tolerance::Real=0.01,
    max_deformation_increment::Real=0.05,
    max_memory_deformation_increment::Real=0.07,
    max_polymer_substeps::Integer=64,
    max_steps::Integer=60,
    avg_window::Union{Nothing,Integer}=nothing,
    drag_stride::Integer=1,
    diagnostic_stride::Integer=0,
    embedded_gradient::Bool=false,
    embedded_advection::Bool=false,
    embedded_force::Bool=false,
    embedded_drag::Bool=false,
    embedded_geometry=:qwall,
    embedded_circle_samples::Integer=32,
    force_boundary_fill::Symbol=:bc_aware,
    drag_cx::Union{Nothing,Real}=nothing,
    drag_cy::Union{Nothing,Real}=nothing,
    drag_radius::Union{Nothing,Real}=nothing,
    drag_u_ref::Union{Nothing,Real}=nothing,
    backend=KernelAbstractions.CPU(),
    T=Float64,
)
    shear_length > 0 || throw(ArgumentError("shear_length must be positive"))
    nu_s > 0 || throw(ArgumentError("nu_s must be positive"))
    nu_p >= 0 || throw(ArgumentError("nu_p must be non-negative"))
    lambda > 0 || throw(ArgumentError("lambda must be positive"))
    0 <= bsd_fraction <= 1 || throw(ArgumentError("bsd_fraction must be in [0, 1]"))
    max_steps >= 0 || throw(ArgumentError("max_steps must be non-negative"))
    drag_stride > 0 || throw(ArgumentError("drag_stride must be positive"))
    diagnostic_stride >= 0 || throw(ArgumentError("diagnostic_stride must be non-negative"))
    embedded_geometry_symbol = Symbol(embedded_geometry)
    embedded_geometry_symbol in (:qwall, :circle) ||
        throw(ArgumentError("embedded_geometry must be :qwall or :circle"))
    embedded_circle_samples > 0 ||
        throw(ArgumentError("embedded_circle_samples must be positive"))
    force_boundary_fill in (:bc_aware, :nearest, :none) ||
        throw(ArgumentError("force_boundary_fill must be :bc_aware, :nearest, or :none"))
    if embedded_geometry_symbol === :circle &&
       (isnothing(drag_cx) || isnothing(drag_cy) || isnothing(drag_radius))
        throw(ArgumentError(
            "embedded_geometry=:circle requires drag_cx, drag_cy, and drag_radius",
        ))
    end

    geom = transfer_step_geometry_2d(geom_h, backend)
    Nx, Ny = geom_h.Nx, geom_h.Ny
    is_solid = geom.is_solid
    q_wall = geom.q_wall
    is_solid_h = geom_h.is_solid
    dx = one(T)
    dy = one(T)
    embedded_circle_cx = isnothing(drag_cx) ? T(NaN) : T(drag_cx) + dx / T(2)
    embedded_circle_cy = isnothing(drag_cy) ? T(NaN) : T(drag_cy) + dy / T(2)
    embedded_h = if embedded_geometry_symbol === :circle
        # The FVFD circle lowering samples control volumes centered at
        # (i-0.5,j-0.5), while the LBM q_wall/is_solid mask is node-centered
        # at (i-1,j-1).  Shift the FVFD circle into that coordinate frame so
        # every LBM-fluid cell has a positive FVFD fluid volume.
        fvfd_embedded_boundary_from_circle_2d(
            Nx, Ny, embedded_circle_cx, embedded_circle_cy, T(drag_radius);
            FT=T, samples=embedded_circle_samples,
        )
    else
        fvfd_embedded_boundary_from_qwall_2d(geom_h.q_wall; FT=T)
    end
    embedded = fvfd_transfer_embedded_boundary_2d(embedded_h, backend, T)

    nu_s_t = T(nu_s)
    nu_p_t = T(nu_p)
    nu_total_t = nu_s_t + nu_p_t
    bsd_t = T(bsd_fraction)
    lambda_t = T(lambda)
    model_cfg = _logfv_polymer_model_config(polymer_model, L_max, T)
    model_code = model_cfg.model_code
    L2_t = model_cfg.L2
    prefactor_t = nu_p_t / lambda_t
    Fx_body_t = T(Fx_body)
    nu_lbm_t = nu_s_t + bsd_t * nu_p_t
    nu_lbm_t > zero(T) || throw(ArgumentError("nu_s + bsd_fraction * nu_p must be positive"))
    shear_length_t = T(shear_length)

    u_profile_h = parabolic_face_profile_2d(geom_h; face=:west, mean_velocity=T(u_mean), FT=T)
    u_profile = KernelAbstractions.allocate(backend, T, Ny)
    copyto!(u_profile, u_profile_h)
    bcspec = default_step_bcspec_2d(geom, u_profile, one(T))

    f_in = KernelAbstractions.allocate(backend, T, Nx, Ny, 9)
    f_out = KernelAbstractions.allocate(backend, T, Nx, Ny, 9)
    f_in_h = zeros(T, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx, q in 1:9
        ux0 = (is_solid_h[i, j] || embedded_h.cut_count[i, j] > 0) ?
              zero(T) : u_profile_h[j]
        f_in_h[i, j, q] = equilibrium(D2Q9(), one(T), ux0, zero(T), q)
    end
    copyto!(f_in, f_in_h)
    fill!(f_out, zero(T))

    rho = KernelAbstractions.zeros(backend, T, Nx, Ny)
    ux = KernelAbstractions.zeros(backend, T, Nx, Ny)
    uy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    uwx = KernelAbstractions.zeros(backend, T, Nx, Ny, 9)
    uwy = KernelAbstractions.zeros(backend, T, Nx, Ny, 9)

    psixx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psiyy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixx_adv = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixy_adv = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psiyy_adv = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixx_next = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixy_next = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psiyy_next = KernelAbstractions.zeros(backend, T, Nx, Ny)
    west_xx = KernelAbstractions.zeros(backend, T, Ny)
    west_xy = KernelAbstractions.zeros(backend, T, Ny)
    west_yy = KernelAbstractions.zeros(backend, T, Ny)
    east_xx = KernelAbstractions.zeros(backend, T, Ny)
    east_xy = KernelAbstractions.zeros(backend, T, Ny)
    east_yy = KernelAbstractions.zeros(backend, T, Ny)
    south_xx = KernelAbstractions.zeros(backend, T, Nx)
    south_xy = KernelAbstractions.zeros(backend, T, Nx)
    south_yy = KernelAbstractions.zeros(backend, T, Nx)
    north_xx = KernelAbstractions.zeros(backend, T, Nx)
    north_xy = KernelAbstractions.zeros(backend, T, Nx)
    north_yy = KernelAbstractions.zeros(backend, T, Nx)
    ux_east = KernelAbstractions.zeros(backend, T, Ny)
    uy_south = KernelAbstractions.zeros(backend, T, Nx)
    uy_north = KernelAbstractions.zeros(backend, T, Nx)
    ux_face = KernelAbstractions.zeros(backend, T, Nx + 1, Ny)
    uy_face = KernelAbstractions.zeros(backend, T, Nx, Ny + 1)
    dudx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    dudy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    dvdx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    dvdy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tauxx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tauxy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tauyy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    fx_poly = KernelAbstractions.zeros(backend, T, Nx, Ny)
    fy_poly = KernelAbstractions.zeros(backend, T, Nx, Ny)
    fx_total = KernelAbstractions.zeros(backend, T, Nx, Ny)
    fy_total = KernelAbstractions.zeros(backend, T, Nx, Ny)
    drag_tx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    drag_ty = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tau_bsd_xx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tau_bsd_xy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tau_bsd_yy = KernelAbstractions.zeros(backend, T, Nx, Ny)

    inlet_shear_estimate = T(4) * abs(T(u_mean)) / shear_length_t
    body_shear_estimate = abs(Fx_body_t) * T(Ny) / (T(2) * max(nu_total_t, eps(T)))
    max_grad_norm_estimate = max(inlet_shear_estimate, body_shear_estimate)
    subcycle_estimate = logfv_oldroydb_subcycle_estimate(
        Float64(max_grad_norm_estimate),
        Float64(lambda_t),
        1.0;
        relative_tolerance=Float64(subcycle_relative_tolerance),
        max_deformation_increment=Float64(max_deformation_increment),
        max_memory_deformation_increment=Float64(max_memory_deformation_increment),
        min_substeps=1,
        max_substeps=max_polymer_substeps,
    )
    selected_polymer_substeps = if polymer_substeps === :auto
        subcycle_estimate.recommended
    elseif polymer_substeps isa Integer
        polymer_substeps >= 1 || throw(ArgumentError("polymer_substeps must be >= 1"))
        polymer_substeps
    else
        throw(ArgumentError("polymer_substeps must be an integer or :auto"))
    end
    dt_poly = one(T) / T(selected_polymer_substeps)
    drag_enabled = !isnothing(drag_cx) && !isnothing(drag_cy) &&
                   !isnothing(drag_radius) && !isnothing(drag_u_ref)
    avg_window_i = isnothing(avg_window) ? max(1, max_steps) : Int(avg_window)
    avg_window_i >= 1 || throw(ArgumentError("avg_window must be >= 1"))
    drag_start = max_steps - min(max_steps, avg_window_i)
    Fx_s_sum = 0.0
    Fy_s_sum = 0.0
    Fx_p_sum = 0.0
    Fy_p_sum = 0.0
    Fx_bsd_sum = 0.0
    Fy_bsd_sum = 0.0
    n_drag = 0
    completed_steps = 0
    first_nonfinite_step = 0
    first_nonfinite_field = :none
    first_nonfinite_i = 0
    first_nonfinite_j = 0
    logfv_bc = fvfd_openx_wally_bcspec_2d()
    fvfd_geometry = FVFDGeometry2D(
        is_solid, embedded, FVFDPatch2D(dx, dy), logfv_bc,
    )
    ux_face_bc = FVFDFieldBC2D(u_profile, ux_east, uy_south, uy_north)
    uy_face_bc = FVFDFieldBC2D(u_profile, ux_east, uy_south, uy_north)

    logfv_add_constant_force_fluid_2d!(fx_total, fy_total, is_solid, Fx_body_t, zero(T); sync=false)
    logfv_compute_macroscopic_forced_field_2d!(rho, ux, uy, f_in, fx_total, fy_total; sync=false)

    for step in 1:max_steps
        completed_steps = step
        logfv_copy_column_profile_2d!(ux_east, ux, Nx; sync=false)
        logfv_copy_column_profile_2d!(east_xx, psixx, Nx; sync=false)
        logfv_copy_column_profile_2d!(east_xy, psixy, Nx; sync=false)
        logfv_copy_column_profile_2d!(east_yy, psiyy, Nx; sync=false)
        if embedded_advection
            logfv_cell_velocity_to_faces_embedded_2d!(
                ux_face, uy_face, ux, uy, fvfd_geometry, ux_face_bc, uy_face_bc;
                sync=false,
            )
        else
            logfv_cell_velocity_to_faces_bc_aware_2d!(
                ux_face, uy_face, ux, uy, is_solid,
                u_profile, ux_east, uy_south, uy_north, logfv_bc;
                sync=false,
            )
        end
        logfv_advect_upwind_bc_aware_2d!(
            psixx_adv, psixy_adv, psiyy_adv,
            psixx, psixy, psiyy,
            west_xx, west_xy, west_yy,
            east_xx, east_xy, east_yy,
            south_xx, south_xy, south_yy,
            north_xx, north_xy, north_yy,
            ux_face, uy_face, is_solid, dx, dy, logfv_bc, one(T);
            sync=false,
        )
        if embedded_gradient
            fvfd_velocity_gradient_embedded_2d!(
                dudx, dudy, dvdx, dvdy, ux, uy, is_solid, dx, dy, logfv_bc, embedded;
                sync=false,
            )
        else
            fvfd_velocity_gradient_2d!(
                dudx, dudy, dvdx, dvdy, ux, uy, is_solid, dx, dy, logfv_bc;
                sync=false,
            )
        end

        psixx_work, psixy_work, psiyy_work = psixx_adv, psixy_adv, psiyy_adv
        for _ in 1:selected_polymer_substeps
            logfv_step_constitutive_log_2d!(
                psixx_next, psixy_next, psiyy_next,
                psixx_work, psixy_work, psiyy_work,
                dudx, dudy, dvdx, dvdy,
                lambda_t, dt_poly, model_code, L2_t;
                sync=false,
            )
            psixx_work, psixx_next = psixx_next, psixx_work
            psixy_work, psixy_next = psixy_next, psixy_work
            psiyy_work, psiyy_next = psiyy_next, psiyy_work
        end
        psixx, psixx_adv = psixx_work, psixx
        psixy, psixy_adv = psixy_work, psixy
        psiyy, psiyy_adv = psiyy_work, psiyy

        logfv_stress_from_log_2d!(
            tauxx, tauxy, tauyy, psixx, psixy, psiyy, prefactor_t;
            model_code, L2=L2_t, sync=false,
        )
        if embedded_force
            logfv_polymer_force_embedded_bc_aware_2d!(
                fx_poly, fy_poly, tauxx, tauxy, tauyy, fvfd_geometry;
                sync=false,
            )
        else
            logfv_polymer_force_bc_aware_2d!(
                fx_poly, fy_poly, tauxx, tauxy, tauyy, is_solid, dx, dy, logfv_bc;
                sync=false,
            )
        end
        logfv_bsd_correct_force_bc_aware_2d!(
            fx_total, fy_total, fx_poly, fy_poly, ux, uy, is_solid, bsd_t, nu_p_t, dx, dy,
            logfv_bc;
            sync=false,
        )
        if force_boundary_fill === :nearest
            logfv_fill_nearest_boundary_2d!(fx_total, fy_total; sync=false)
        end
        logfv_add_constant_force_fluid_2d!(fx_total, fy_total, is_solid, Fx_body_t, zero(T); sync=false)

        fused_trt_libb_v2_guo_field_step!(
            f_out, f_in, rho, ux, uy, is_solid, q_wall, uwx, uwy, fx_total, fy_total,
            Nx, Ny, nu_lbm_t;
        )
        apply_bc_rebuild_2d!(f_out, f_in, bcspec, nu_lbm_t, Nx, Ny)
        if drag_enabled && step > drag_start &&
           ((step - drag_start - 1) % drag_stride == 0 || step == max_steps)
            drag_s = compute_drag_libb_mei_2d(f_out, q_wall, uwx, uwy, Nx, Ny)
            drag_p = if embedded_drag
                logfv_embedded_wall_traction_2d!(
                    drag_tx, drag_ty, tauxx, tauxy, tauyy, fvfd_geometry; sync=true,
                )
                (Fx=Float64(sum(Array(drag_tx))), Fy=Float64(sum(Array(drag_ty))))
            else
                compute_polymeric_drag_2d(
                    tauxx, tauxy, tauyy, q_wall, Nx, Ny;
                    cx=Float64(drag_cx),
                    cy=Float64(drag_cy),
                    radius=Float64(drag_radius),
                    extrapolate=true,
                    reconstruction_order=2,
                )
            end
            drag_bsd = if embedded_drag
                logfv_bsd_stress_from_gradient_2d!(
                    tau_bsd_xx, tau_bsd_xy, tau_bsd_yy,
                    dudx, dudy, dvdx, dvdy, bsd_t * nu_p_t; sync=false,
                )
                logfv_embedded_wall_traction_2d!(
                    drag_tx, drag_ty, tau_bsd_xx, tau_bsd_xy, tau_bsd_yy,
                    fvfd_geometry; sync=true,
                )
                (Fx=Float64(sum(Array(drag_tx))), Fy=Float64(sum(Array(drag_ty))))
            else
                _logfv_compute_bsd_drag_2d(
                    dudx, dudy, dvdx, dvdy, q_wall, Nx, Ny;
                    cx=Float64(drag_cx),
                    cy=Float64(drag_cy),
                    radius=Float64(drag_radius),
                    zeta_nu_p=Float64(bsd_t * nu_p_t),
                    reconstruction_order=2,
                )
            end
            Fx_s_sum += drag_s.Fx
            Fy_s_sum += drag_s.Fy
            Fx_p_sum += drag_p.Fx
            Fy_p_sum += drag_p.Fy
            Fx_bsd_sum += drag_bsd.Fx
            Fy_bsd_sum += drag_bsd.Fy
            n_drag += 1
        end
        logfv_compute_macroscopic_forced_field_2d!(rho, ux, uy, f_out, fx_total, fy_total; sync=false)
        if diagnostic_stride > 0 &&
           (step == 1 || step % diagnostic_stride == 0 || step == max_steps)
            KernelAbstractions.synchronize(backend)
            finite_diag = _logfv_first_nonfinite_field_2d(
                is_solid_h,
                :rho => rho,
                :ux => ux,
                :uy => uy,
                :psixx => psixx,
                :psixy => psixy,
                :psiyy => psiyy,
                :tauxx => tauxx,
                :tauxy => tauxy,
                :tauyy => tauyy,
                :fx_poly => fx_poly,
                :fy_poly => fy_poly,
                :fx_total => fx_total,
                :fy_total => fy_total,
            )
            if !finite_diag.finite
                first_nonfinite_step = step
                first_nonfinite_field = finite_diag.field
                first_nonfinite_i = finite_diag.i
                first_nonfinite_j = finite_diag.j
                break
            end
        end
        f_in, f_out = f_out, f_in
    end
    logfv_stress_from_log_2d!(
        tauxx, tauxy, tauyy, psixx, psixy, psiyy, prefactor_t;
        model_code, L2=L2_t,
    )
    KernelAbstractions.synchronize(backend)

    rho_cpu = Array(rho)
    ux_cpu = Array(ux)
    uy_cpu = Array(uy)
    psixx_cpu = Array(psixx)
    psixy_cpu = Array(psixy)
    psiyy_cpu = Array(psiyy)
    tauxx_cpu = Array(tauxx)
    tauxy_cpu = Array(tauxy)
    tauyy_cpu = Array(tauyy)
    fx_poly_cpu = Array(fx_poly)
    fy_poly_cpu = Array(fy_poly)
    fx_total_cpu = Array(fx_total)
    fy_total_cpu = Array(fy_total)

    conf_diag = _logfv_conformation_diagnostics_2d(
        psixx_cpu, psixy_cpu, psiyy_cpu, is_solid_h, model_code, L2_t,
    )
    min_c_eig = conf_diag.min_c_eig
    max_c_trace = conf_diag.max_c_trace
    min_fene_denom = conf_diag.min_fene_denom
    max_fene_factor = conf_diag.max_fene_factor
    max_speed = 0.0
    for j in 1:Ny, i in 1:Nx
        if !is_solid_h[i, j]
            max_speed = max(max_speed, hypot(Float64(ux_cpu[i, j]), Float64(uy_cpu[i, j])))
        end
    end
    fluid_mask = .!is_solid_h
    max_abs_psi = max(maximum(abs, psixx_cpu), maximum(abs, psixy_cpu), maximum(abs, psiyy_cpu))
    max_abs_tau = max(maximum(abs, tauxx_cpu), maximum(abs, tauxy_cpu), maximum(abs, tauyy_cpu))
    max_abs_poly_force = max(maximum(abs, fx_poly_cpu), maximum(abs, fy_poly_cpu))
    max_abs_total_force = max(maximum(abs, fx_total_cpu), maximum(abs, fy_total_cpu))
    Fx_s = n_drag > 0 ? Fx_s_sum / n_drag : NaN
    Fy_s = n_drag > 0 ? Fy_s_sum / n_drag : NaN
    Fx_p = n_drag > 0 ? Fx_p_sum / n_drag : NaN
    Fy_p = n_drag > 0 ? Fy_p_sum / n_drag : NaN
    Fx_bsd = n_drag > 0 ? Fx_bsd_sum / n_drag : NaN
    Fy_bsd = n_drag > 0 ? Fy_bsd_sum / n_drag : NaN
    Fx_drag = n_drag > 0 ? Fx_s + Fx_p - Fx_bsd : NaN
    Fy_drag = n_drag > 0 ? Fy_s + Fy_p - Fy_bsd : NaN
    drag_diameter = isnothing(drag_radius) ? NaN : 2.0 * Float64(drag_radius)
    drag_speed = isnothing(drag_u_ref) ? NaN : Float64(drag_u_ref)
    Cd_s = n_drag > 0 ? 2.0 * Fx_s / (drag_speed^2 * drag_diameter) : NaN
    Cd_p = n_drag > 0 ? 2.0 * Fx_p / (drag_speed^2 * drag_diameter) : NaN
    Cd_bsd = n_drag > 0 ? 2.0 * Fx_bsd / (drag_speed^2 * drag_diameter) : NaN
    Cd = n_drag > 0 ? Cd_s + Cd_p - Cd_bsd : NaN
    fluid_cell_fractions = embedded_h.cell_fraction[.!is_solid_h]
    embedded_min_fluid_cell_fraction = isempty(fluid_cell_fractions) ?
        NaN : Float64(minimum(fluid_cell_fractions))
    embedded_zero_fluid_cell_fraction_count = count(
        <=(sqrt(eps(T))), fluid_cell_fractions,
    )
    embedded_circle_normal_alignment = if embedded_geometry_symbol === :circle
        _logfv_embedded_circle_normal_alignment_2d(
            embedded_h, Float64(embedded_circle_cx), Float64(embedded_circle_cy),
        )
    else
        (min=NaN, mean=NaN, samples=0)
    end

    return (;
        geometry=geom_h,
        Nx,
        Ny,
        nu_s=Float64(nu_s_t),
        nu_p=Float64(nu_p_t),
        nu_total=Float64(nu_total_t),
        nu_lbm=Float64(nu_lbm_t),
        lambda=Float64(lambda_t),
        polymer_model=model_cfg.polymer_model,
        L_max=model_cfg.L_max,
        u_mean=Float64(u_mean),
        Fx_body=Float64(Fx_body_t),
        bsd_fraction=Float64(bsd_t),
        max_steps,
        completed_steps,
        polymer_substeps=selected_polymer_substeps,
        requested_polymer_substeps=polymer_substeps,
        diagnostic_stride,
        embedded_gradient,
        embedded_advection,
        embedded_force,
        embedded_drag,
        embedded_geometry=embedded_geometry_symbol,
        embedded_circle_samples,
        embedded_cut_count=count(embedded_h.cut_count .> 0),
        embedded_wall_length=Float64(sum(embedded_h.wall_fraction)),
        embedded_min_fluid_cell_fraction,
        embedded_zero_fluid_cell_fraction_count,
        embedded_normal_radial_min=embedded_circle_normal_alignment.min,
        embedded_normal_radial_mean=embedded_circle_normal_alignment.mean,
        embedded_normal_radial_samples=embedded_circle_normal_alignment.samples,
        force_boundary_fill,
        first_nonfinite_step,
        first_nonfinite_field,
        first_nonfinite_i,
        first_nonfinite_j,
        subcycle_estimate,
        max_grad_norm_estimate=Float64(max_grad_norm_estimate),
        rho=rho_cpu,
        ux=ux_cpu,
        uy=uy_cpu,
        psixx=psixx_cpu,
        psixy=psixy_cpu,
        psiyy=psiyy_cpu,
        tauxx=tauxx_cpu,
        tauxy=tauxy_cpu,
        tauyy=tauyy_cpu,
        fx_poly=fx_poly_cpu,
        fy_poly=fy_poly_cpu,
        fx_total=fx_total_cpu,
        fy_total=fy_total_cpu,
        is_solid=is_solid_h,
        min_c_eig,
        max_c_trace,
        min_fene_denom,
        max_fene_factor,
        max_speed,
        max_abs_psi,
        max_abs_tau,
        max_abs_poly_force,
        max_abs_total_force,
        Fx_s,
        Fy_s,
        Fx_p,
        Fy_p,
        Fx_bsd,
        Fy_bsd,
        Fx_drag,
        Fy_drag,
        Cd_s,
        Cd_p,
        Cd_bsd,
        Cd,
        n_drag_samples=n_drag,
        rho_min=minimum(rho_cpu[fluid_mask]),
        rho_max=maximum(rho_cpu[fluid_mask]),
    )
end

"""
    run_viscoelastic_logfv_bfs_coupled_2d(; kwargs...)

Run a coarse coupled log-FV polymer canary on a backward-facing-step geometry.

This is an open-x `StepChannelGeometry2D` path with feedback:

```text
LBM u -> open-x solid-aware log-FV polymer step
      -> tau_p -> div(tau_p) + BSD -> LI-BB V2 Guo-field solvent step
```

The dynamic outlet profiles are copied on device, so no host copy is needed
inside the time loop.
"""
function run_viscoelastic_logfv_bfs_coupled_2d(;
    H_in::Integer=4,
    expansion_ratio::Integer=2,
    L_up::Integer=2,
    L_down::Integer=4,
    kwargs...,
)
    H_in >= 3 || throw(ArgumentError("H_in must be >= 3"))
    expansion_ratio >= 2 || throw(ArgumentError("expansion_ratio must be >= 2"))
    T = get(kwargs, :T, Float64)
    geom_h = backward_facing_step_geometry_2d(;
        H_in=Int(H_in),
        expansion_ratio=Int(expansion_ratio),
        L_up=Int(L_up),
        L_down=Int(L_down),
        FT=T,
    )
    return _run_viscoelastic_logfv_step_channel_coupled_2d(
        geom_h; shear_length=H_in, kwargs...,
    )
end

"""
    run_viscoelastic_logfv_contraction_coupled_2d(; kwargs...)

Run the open-x coupled log-FV polymer path on a symmetric axis-aligned
contraction geometry.

This uses the same `StepChannelGeometry2D` core as the BFS and square-channel
drivers:

```text
LBM u -> open-x solid-aware log-FV polymer step
      -> tau_p -> div(tau_p) + BSD -> LI-BB V2 Guo-field solvent step
```
"""
function run_viscoelastic_logfv_contraction_coupled_2d(;
    H_out::Integer=4,
    β_c::Integer=4,
    L_up::Integer=4,
    L_down::Integer=4,
    kwargs...,
)
    H_out >= 3 || throw(ArgumentError("H_out must be >= 3"))
    β_c >= 2 || throw(ArgumentError("β_c must be >= 2"))
    T = get(kwargs, :T, Float64)
    geom_h = contraction_step_geometry_2d(;
        H_out=Int(H_out),
        β_c=Int(β_c),
        L_up=Int(L_up),
        L_down=Int(L_down),
        FT=T,
    )
    return _run_viscoelastic_logfv_step_channel_coupled_2d(
        geom_h; shear_length=H_out, kwargs...,
    )
end


"""
    run_viscoelastic_logfv_square_channel_coupled_2d(; kwargs...)

Run the same open-x coupled log-FV polymer path on a centered square obstacle
channel. This is the Cartesian-obstacle macro canary between periodic square
tests and curved cylinder validation.
"""
function run_viscoelastic_logfv_square_channel_coupled_2d(;
    H::Integer=12,
    side::Integer=4,
    L_up::Integer=2,
    L_down::Integer=3,
    kwargs...,
)
    H >= side + 4 || throw(ArgumentError("H must leave at least two fluid rows on each side"))
    side >= 2 || throw(ArgumentError("side must be >= 2"))
    T = get(kwargs, :T, Float64)
    geom_h = square_obstacle_channel_geometry_2d(;
        H=Int(H),
        side=Int(side),
        L_up=Int(L_up),
        L_down=Int(L_down),
        FT=T,
    )
    return _run_viscoelastic_logfv_step_channel_coupled_2d(
        geom_h; shear_length=H, kwargs...,
    )
end

function _logfv_cylinder_channel_geometry_2d(;
    radius::Real=6,
    H::Integer=max(ceil(Int, 4 * radius), ceil(Int, 2 * radius + 4)),
    L_up::Real=4,
    L_down::Real=8,
    FT::Type{<:AbstractFloat}=Float64,
)
    radius > 1 || throw(ArgumentError("radius must be > 1"))
    H >= ceil(Int, 2 * radius + 4) ||
        throw(ArgumentError("H must leave at least two fluid rows around the cylinder"))
    L_up > 1 || throw(ArgumentError("L_up must leave upstream clearance"))
    L_down > 1 || throw(ArgumentError("L_down must leave downstream clearance"))

    Nx = ceil(Int, (L_up + L_down) * radius)
    Ny = Int(H)
    cx = FT(L_up * radius)
    cy = FT((Ny - 1) / 2)
    q_wall, is_solid = precompute_q_wall_cylinder(Nx, Ny, cx, cy, radius; FT=FT)
    D = max(1, round(Int, 2 * radius))

    hydro_mask = fill(false, Ny)
    if Ny > 2
        hydro_mask[2:(Ny - 1)] .= true
    end
    conformation_mask = fill(true, Ny)

    return StepChannelGeometry2D{FT,Array{FT,3},Matrix{Bool},Vector{Bool}}(
        :cylinder,
        Nx,
        Ny,
        round(Int, cx) + 1,
        1:Ny,
        1:Ny,
        D,
        Ny,
        Ny,
        q_wall,
        Matrix{Bool}(is_solid),
        hydro_mask,
        copy(hydro_mask),
        conformation_mask,
        copy(conformation_mask),
    )
end

"""
    run_viscoelastic_logfv_cylinder_coupled_2d(; kwargs...)

Run the open-x coupled log-FV polymer path on a circular cylinder with
precomputed cut-link geometry. This is the curved-wall macro canary above BFS
and square-obstacle tests; benchmark Cd convergence still belongs in a
separate harness after the lower ladder is green.
"""
function run_viscoelastic_logfv_cylinder_coupled_2d(;
    radius::Real=6,
    H::Integer=max(ceil(Int, 4 * radius), ceil(Int, 2 * radius + 4)),
    L_up::Real=4,
    L_down::Real=8,
    kwargs...,
)
    T = get(kwargs, :T, Float64)
    geom_h = _logfv_cylinder_channel_geometry_2d(;
        radius,
        H,
        L_up,
        L_down,
        FT=T,
    )
    return _run_viscoelastic_logfv_step_channel_coupled_2d(
        geom_h;
        shear_length=H,
        drag_cx=Float64(L_up * radius),
        drag_cy=Float64((H - 1) / 2),
        drag_radius=Float64(radius),
        drag_u_ref=Float64(get(kwargs, :u_mean, 0.01)),
        kwargs...,
    )
end

@inline function _logfv_channel_ux(flow::Symbol, y, height, umax, uwall)
    if flow === :poiseuille
        eta = y / height
        return 4 * umax * eta * (1 - eta)
    elseif flow === :couette
        return uwall * y / height
    else
        error("unsupported log-FV channel flow $(flow); expected :poiseuille or :couette")
    end
end

@inline function _logfv_channel_lapu(flow::Symbol, height, umax)
    if flow === :poiseuille
        return -8 * umax / (height * height)
    elseif flow === :couette
        return 0.0
    else
        error("unsupported log-FV channel flow $(flow); expected :poiseuille or :couette")
    end
end

function _logfv_channel_reference_errors(
    flow::Symbol, tauxx, tauxy, tauyy, fx_poly, fy_poly, fx_total, fy_total,
    height, umax, uwall, lambda, prefactor, bsd_fraction,
)
    Nx, Ny = size(tauxx)
    max_tau_error = 0.0
    max_poly_force_error = 0.0
    max_total_force_error = 0.0
    max_transverse_force = 0.0
    min_c_eig = Inf
    nu_p = prefactor * lambda
    lapu = _logfv_channel_lapu(flow, height, umax)
    dy = height / Ny

    for j in 1:Ny, i in 1:Nx
        y = (j - 0.5) * dy
        gamma = _logfv_channel_shear(flow, y, height, umax, uwall)
        cxx = 1 + 2 * (lambda * gamma)^2
        cxy = lambda * gamma
        cyy = 1.0
        min_c_eig = min(min_c_eig, logfv_min_eig_sym2_2d(cxx, cxy, cyy))
        expected_tau = (
            prefactor * (cxx - 1),
            prefactor * cxy,
            prefactor * (cyy - 1),
        )
        max_tau_error = max(
            max_tau_error,
            abs(tauxx[i, j] - expected_tau[1]),
            abs(tauxy[i, j] - expected_tau[2]),
            abs(tauyy[i, j] - expected_tau[3]),
        )

        if i > 1 && i < Nx && j > 1 && j < Ny
            expected_poly_fx = nu_p * lapu
            expected_total_fx = (1 - bsd_fraction) * nu_p * lapu
            max_poly_force_error = max(max_poly_force_error, abs(fx_poly[i, j] - expected_poly_fx))
            max_total_force_error = max(max_total_force_error, abs(fx_total[i, j] - expected_total_fx))
            max_transverse_force = max(
                max_transverse_force,
                abs(fy_poly[i, j]),
                abs(fy_total[i, j]),
            )
        end
    end

    return (;
        max_tau_error,
        max_poly_force_error,
        max_total_force_error,
        max_transverse_force,
        min_c_eig,
    )
end

"""
    run_viscoelastic_logfv_channel_2d(; kwargs...)

Run the first patch-local macro canary for the cell-centered log-FV
polymer backend on a prescribed channel flow.

This is not a coupled fluid solve. It exercises the macro-domain polymer
pipeline on one uniform patch:

```text
analytic channel u -> analytic steady Psi -> tau_p -> div(tau_p) -> BSD force
```

The function keeps `dx`, `dy`, and patch fields explicit so the same
kernel path can later be wrapped by Basilisk-style quadtree AMR patch
exchange/prolongation/restriction.
"""
function run_viscoelastic_logfv_channel_2d(;
    Nx::Integer=32,
    Ny::Integer=32,
    flow::Symbol=:poiseuille,
    height::Real=1.0,
    width::Real=1.0,
    umax::Real=0.05,
    uwall::Real=0.05,
    lambda::Real=5.0,
    beta::Real=0.5,
    Wi::Real=1.0,
    prefactor::Union{Nothing,Real}=nothing,
    bsd_fraction::Real=0.0,
    backend=KernelAbstractions.CPU(),
    T=Float64,
)
    Nx >= 3 || throw(ArgumentError("Nx must be >= 3 for centered force checks"))
    Ny >= 3 || throw(ArgumentError("Ny must be >= 3 for centered force checks"))
    flow in (:poiseuille, :couette) || throw(ArgumentError("flow must be :poiseuille or :couette"))

    height_t = T(height)
    width_t = T(width)
    lambda_t = T(lambda)
    umax_t = T(umax)
    uwall_t = T(uwall)
    prefactor_t = isnothing(prefactor) ? (one(T) - T(beta)) / T(Wi) : T(prefactor)
    bsd_t = T(bsd_fraction)
    dx = width_t / T(Nx)
    dy = height_t / T(Ny)
    nu_p = prefactor_t * lambda_t

    psixx_h = zeros(T, Nx, Ny)
    psixy_h = zeros(T, Nx, Ny)
    psiyy_h = zeros(T, Nx, Ny)
    ux_h = zeros(T, Nx, Ny)
    uy_h = zeros(T, Nx, Ny)

    for j in 1:Ny, i in 1:Nx
        y = (T(j) - T(0.5)) * dy
        gamma = T(_logfv_channel_shear(flow, y, height_t, umax_t, uwall_t))
        ux_h[i, j] = T(_logfv_channel_ux(flow, y, height_t, umax_t, uwall_t))
        cxx = one(T) + T(2) * (lambda_t * gamma)^2
        cxy = lambda_t * gamma
        cyy = one(T)
        psixx_h[i, j], psixy_h[i, j], psiyy_h[i, j] = logfv_log_spd_sym2_2d(cxx, cxy, cyy)
    end

    psixx = KernelAbstractions.allocate(backend, T, Nx, Ny)
    psixy = KernelAbstractions.allocate(backend, T, Nx, Ny)
    psiyy = KernelAbstractions.allocate(backend, T, Nx, Ny)
    ux = KernelAbstractions.allocate(backend, T, Nx, Ny)
    uy = KernelAbstractions.allocate(backend, T, Nx, Ny)
    is_solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny)
    copyto!(psixx, psixx_h)
    copyto!(psixy, psixy_h)
    copyto!(psiyy, psiyy_h)
    copyto!(ux, ux_h)
    copyto!(uy, uy_h)
    fill!(is_solid, false)

    tauxx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tauxy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tauyy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    fx_poly = KernelAbstractions.zeros(backend, T, Nx, Ny)
    fy_poly = KernelAbstractions.zeros(backend, T, Nx, Ny)
    fx_total = KernelAbstractions.zeros(backend, T, Nx, Ny)
    fy_total = KernelAbstractions.zeros(backend, T, Nx, Ny)
    bc = logfv_periodicx_wally_bcspec_2d()

    logfv_stress_from_log_2d!(tauxx, tauxy, tauyy, psixx, psixy, psiyy, prefactor_t)
    logfv_polymer_force_bc_aware_2d!(fx_poly, fy_poly, tauxx, tauxy, tauyy, is_solid, dx, dy, bc)
    logfv_bsd_correct_force_bc_aware_2d!(
        fx_total, fy_total, fx_poly, fy_poly, ux, uy, is_solid, bsd_t, nu_p, dx, dy, bc,
    )
    KernelAbstractions.synchronize(backend)

    tauxx_cpu = Array(tauxx)
    tauxy_cpu = Array(tauxy)
    tauyy_cpu = Array(tauyy)
    fx_poly_cpu = Array(fx_poly)
    fy_poly_cpu = Array(fy_poly)
    fx_total_cpu = Array(fx_total)
    fy_total_cpu = Array(fy_total)
    errors = _logfv_channel_reference_errors(
        flow, tauxx_cpu, tauxy_cpu, tauyy_cpu, fx_poly_cpu, fy_poly_cpu,
        fx_total_cpu, fy_total_cpu, Float64(height_t), Float64(umax_t),
        Float64(uwall_t), Float64(lambda_t), Float64(prefactor_t), Float64(bsd_t),
    )

    return (;
        flow,
        Nx,
        Ny,
        dx=Float64(dx),
        dy=Float64(dy),
        height=Float64(height_t),
        width=Float64(width_t),
        lambda=Float64(lambda_t),
        prefactor=Float64(prefactor_t),
        beta=Float64(beta),
        Wi=Float64(Wi),
        bsd_fraction=Float64(bsd_t),
        nu_p=Float64(nu_p),
        ux=Array(ux),
        uy=Array(uy),
        psixx=Array(psixx),
        psixy=Array(psixy),
        psiyy=Array(psiyy),
        tauxx=tauxx_cpu,
        tauxy=tauxy_cpu,
        tauyy=tauyy_cpu,
        fx_poly=fx_poly_cpu,
        fy_poly=fy_poly_cpu,
        fx_total=fx_total_cpu,
        fy_total=fy_total_cpu,
        errors...,
    )
end

function _logfv_channel_reference_fields(
    flow::Symbol,
    Nx::Integer,
    Ny::Integer,
    height,
    width,
    umax,
    uwall,
    lambda,
    prefactor,
    ::Type{T},
) where {T}
    height_t = T(height)
    width_t = T(width)
    lambda_t = T(lambda)
    umax_t = T(umax)
    uwall_t = T(uwall)
    prefactor_t = T(prefactor)
    dx = width_t / T(Nx)
    dy = height_t / T(Ny)

    ux_h = zeros(T, Nx, Ny)
    uy_h = zeros(T, Nx, Ny)
    cxx_h = zeros(T, Nx, Ny)
    cxy_h = zeros(T, Nx, Ny)
    cyy_h = zeros(T, Nx, Ny)
    psixx_h = zeros(T, Nx, Ny)
    psixy_h = zeros(T, Nx, Ny)
    psiyy_h = zeros(T, Nx, Ny)
    tauxx_ref = zeros(T, Nx, Ny)
    tauxy_ref = zeros(T, Nx, Ny)
    tauyy_ref = zeros(T, Nx, Ny)
    dudx_ref = zeros(T, Nx, Ny)
    dudy_ref = zeros(T, Nx, Ny)
    dvdx_ref = zeros(T, Nx, Ny)
    dvdy_ref = zeros(T, Nx, Ny)

    for j in 1:Ny, i in 1:Nx
        y = (T(j) - T(0.5)) * dy
        gamma = T(_logfv_channel_shear(flow, y, height_t, umax_t, uwall_t))
        ux_h[i, j] = T(_logfv_channel_ux(flow, y, height_t, umax_t, uwall_t))
        dudy_ref[i, j] = gamma
        cxx = one(T) + T(2) * (lambda_t * gamma)^2
        cxy = lambda_t * gamma
        cyy = one(T)
        cxx_h[i, j] = cxx
        cxy_h[i, j] = cxy
        cyy_h[i, j] = cyy
        psixx_h[i, j], psixy_h[i, j], psiyy_h[i, j] =
            logfv_log_spd_sym2_2d(cxx, cxy, cyy)
        tauxx_ref[i, j] = prefactor_t * (cxx - one(T))
        tauxy_ref[i, j] = prefactor_t * cxy
        tauyy_ref[i, j] = prefactor_t * (cyy - one(T))
    end

    return (;
        dx,
        dy,
        ux=ux_h,
        uy=uy_h,
        cxx=cxx_h,
        cxy=cxy_h,
        cyy=cyy_h,
        psixx=psixx_h,
        psixy=psixy_h,
        psiyy=psiyy_h,
        tauxx=tauxx_ref,
        tauxy=tauxy_ref,
        tauyy=tauyy_ref,
        dudx=dudx_ref,
        dudy=dudy_ref,
        dvdx=dvdx_ref,
        dvdy=dvdy_ref,
    )
end

"""
    run_viscoelastic_logfv_frozen_channel_cde_2d(; kwargs...)

Run the FVFD/log-FV polymer CDE on a prescribed analytical channel velocity
field without feeding polymer force back into the LBM solvent.

This is the quantitative frozen-flow gate between local analytical operators
and obstacle/RheoTool comparisons:

```text
frozen u -> FVFD face velocities -> FVFD advection -> log-C source
         -> tau_p -> div(tau_p) + BSD diagnostics
```

Use `initial=:steady` to check that the production operator path preserves the
analytical Oldroyd-B channel solution up to source-splitting error. Use
`initial=:identity` to measure transient convergence toward the same reference.
"""
function run_viscoelastic_logfv_frozen_channel_cde_2d(;
    Nx::Integer=16,
    Ny::Integer=32,
    flow::Symbol=:poiseuille,
    height::Real=1.0,
    width::Real=1.0,
    umax::Real=0.02,
    uwall::Real=0.02,
    lambda::Real=2.0,
    beta::Real=0.5,
    Wi::Real=1.0,
    prefactor::Union{Nothing,Real}=nothing,
    bsd_fraction::Real=1.0,
    initial::Symbol=:steady,
    max_steps::Integer=1,
    polymer_substeps=:auto,
    subcycle_relative_tolerance::Real=0.01,
    max_deformation_increment::Real=0.05,
    max_memory_deformation_increment::Real=0.07,
    max_polymer_substeps::Integer=256,
    backend=KernelAbstractions.CPU(),
    T=Float64,
)
    Nx >= 3 || throw(ArgumentError("Nx must be >= 3"))
    Ny >= 5 || throw(ArgumentError("Ny must be >= 5"))
    flow in (:poiseuille, :couette) || throw(ArgumentError("flow must be :poiseuille or :couette"))
    lambda > 0 || throw(ArgumentError("lambda must be positive"))
    max_steps >= 0 || throw(ArgumentError("max_steps must be non-negative"))
    initial in (:steady, :identity) ||
        throw(ArgumentError("initial must be :steady or :identity"))

    prefactor_t = isnothing(prefactor) ? (one(T) - T(beta)) / T(Wi) : T(prefactor)
    bsd_t = T(bsd_fraction)
    lambda_t = T(lambda)
    height_t = T(height)
    width_t = T(width)
    umax_t = T(umax)
    uwall_t = T(uwall)
    max_grad_norm = flow === :poiseuille ?
        abs(T(4) * umax_t / height_t) :
        abs(uwall_t / height_t)
    subcycle_estimate = logfv_oldroydb_subcycle_estimate(
        Float64(max_grad_norm),
        Float64(lambda_t),
        1.0;
        relative_tolerance=Float64(subcycle_relative_tolerance),
        max_deformation_increment=Float64(max_deformation_increment),
        max_memory_deformation_increment=Float64(max_memory_deformation_increment),
        min_substeps=1,
        max_substeps=max_polymer_substeps,
    )
    selected_polymer_substeps = if polymer_substeps === :auto
        subcycle_estimate.recommended
    elseif polymer_substeps isa Integer
        polymer_substeps >= 1 || throw(ArgumentError("polymer_substeps must be >= 1"))
        polymer_substeps
    else
        throw(ArgumentError("polymer_substeps must be an integer or :auto"))
    end
    dt_poly = one(T) / T(selected_polymer_substeps)

    ref = _logfv_channel_reference_fields(
        flow, Nx, Ny, height_t, width_t, umax_t, uwall_t, lambda_t, prefactor_t, T,
    )
    psixx_h = initial === :steady ? copy(ref.psixx) : zeros(T, Nx, Ny)
    psixy_h = initial === :steady ? copy(ref.psixy) : zeros(T, Nx, Ny)
    psiyy_h = initial === :steady ? copy(ref.psiyy) : zeros(T, Nx, Ny)

    ux = KernelAbstractions.allocate(backend, T, Nx, Ny)
    uy = KernelAbstractions.allocate(backend, T, Nx, Ny)
    is_solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny)
    psixx = KernelAbstractions.allocate(backend, T, Nx, Ny)
    psixy = KernelAbstractions.allocate(backend, T, Nx, Ny)
    psiyy = KernelAbstractions.allocate(backend, T, Nx, Ny)
    psixx_adv = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixy_adv = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psiyy_adv = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixx_next = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixy_next = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psiyy_next = KernelAbstractions.zeros(backend, T, Nx, Ny)
    ux_face = KernelAbstractions.zeros(backend, T, Nx + 1, Ny)
    uy_face = KernelAbstractions.zeros(backend, T, Nx, Ny + 1)
    dudx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    dudy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    dvdx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    dvdy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tauxx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tauxy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tauyy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    fx_poly = KernelAbstractions.zeros(backend, T, Nx, Ny)
    fy_poly = KernelAbstractions.zeros(backend, T, Nx, Ny)
    fx_total = KernelAbstractions.zeros(backend, T, Nx, Ny)
    fy_total = KernelAbstractions.zeros(backend, T, Nx, Ny)

    copyto!(ux, ref.ux)
    copyto!(uy, ref.uy)
    fill!(is_solid, false)
    copyto!(psixx, psixx_h)
    copyto!(psixy, psixy_h)
    copyto!(psiyy, psiyy_h)

    bc = logfv_periodicx_wally_bcspec_2d()
    logfv_cell_velocity_to_faces_bc_aware_2d!(
        ux_face, uy_face, ux, uy, is_solid,
        ux, ux, uy, uy, bc; sync=false,
    )
    logfv_velocity_gradient_bc_aware_2d!(
        dudx, dudy, dvdx, dvdy, ux, uy, is_solid, ref.dx, ref.dy, bc;
        sync=false,
    )

    for _ in 1:max_steps
        logfv_advect_upwind_bc_aware_2d!(
            psixx_adv, psixy_adv, psiyy_adv,
            psixx, psixy, psiyy,
            psixx, psixy, psiyy,
            psixx, psixy, psiyy,
            psixx, psixy, psiyy,
            psixx, psixy, psiyy,
            ux_face, uy_face, is_solid, ref.dx, ref.dy, bc, one(T);
            sync=false,
        )
        psixx_work, psixy_work, psiyy_work = psixx_adv, psixy_adv, psiyy_adv
        for _ in 1:selected_polymer_substeps
            logfv_step_oldroydb_log_2d!(
                psixx_next, psixy_next, psiyy_next,
                psixx_work, psixy_work, psiyy_work,
                dudx, dudy, dvdx, dvdy,
                lambda_t, dt_poly;
                sync=false,
            )
            psixx_work, psixx_next = psixx_next, psixx_work
            psixy_work, psixy_next = psixy_next, psixy_work
            psiyy_work, psiyy_next = psiyy_next, psiyy_work
        end
        psixx, psixx_adv = psixx_work, psixx
        psixy, psixy_adv = psixy_work, psixy
        psiyy, psiyy_adv = psiyy_work, psiyy
    end

    logfv_stress_from_log_2d!(tauxx, tauxy, tauyy, psixx, psixy, psiyy, prefactor_t)
    logfv_polymer_force_bc_aware_2d!(
        fx_poly, fy_poly, tauxx, tauxy, tauyy, is_solid, ref.dx, ref.dy, bc;
        sync=false,
    )
    logfv_bsd_correct_force_bc_aware_2d!(
        fx_total, fy_total, fx_poly, fy_poly, ux, uy, is_solid,
        bsd_t, prefactor_t * lambda_t, ref.dx, ref.dy, bc;
        sync=false,
    )
    KernelAbstractions.synchronize(backend)

    psixx_cpu = Array(psixx)
    psixy_cpu = Array(psixy)
    psiyy_cpu = Array(psiyy)
    dudx_cpu = Array(dudx)
    dudy_cpu = Array(dudy)
    dvdx_cpu = Array(dvdx)
    dvdy_cpu = Array(dvdy)
    tauxx_cpu = Array(tauxx)
    tauxy_cpu = Array(tauxy)
    tauyy_cpu = Array(tauyy)
    fx_poly_cpu = Array(fx_poly)
    fy_poly_cpu = Array(fy_poly)
    fx_total_cpu = Array(fx_total)
    fy_total_cpu = Array(fy_total)

    cxx_cpu = similar(psixx_cpu)
    cxy_cpu = similar(psixy_cpu)
    cyy_cpu = similar(psiyy_cpu)
    max_c_error = 0.0
    max_psi_error = 0.0
    max_velocity_gradient_error = 0.0
    max_gradient_component_error = (dudx=0.0, dudy=0.0, dvdx=0.0, dvdy=0.0)
    min_c_eig = Inf
    for j in 1:Ny, i in 1:Nx
        cxx, cxy, cyy = logfv_exp_sym2_2d(psixx_cpu[i, j], psixy_cpu[i, j], psiyy_cpu[i, j])
        cxx_cpu[i, j] = cxx
        cxy_cpu[i, j] = cxy
        cyy_cpu[i, j] = cyy
        max_c_error = max(
            max_c_error,
            abs(cxx - ref.cxx[i, j]),
            abs(cxy - ref.cxy[i, j]),
            abs(cyy - ref.cyy[i, j]),
        )
        max_psi_error = max(
            max_psi_error,
            abs(psixx_cpu[i, j] - ref.psixx[i, j]),
            abs(psixy_cpu[i, j] - ref.psixy[i, j]),
            abs(psiyy_cpu[i, j] - ref.psiyy[i, j]),
        )
        dudx_error = abs(dudx_cpu[i, j] - ref.dudx[i, j])
        dudy_error = abs(dudy_cpu[i, j] - ref.dudy[i, j])
        dvdx_error = abs(dvdx_cpu[i, j] - ref.dvdx[i, j])
        dvdy_error = abs(dvdy_cpu[i, j] - ref.dvdy[i, j])
        max_gradient_component_error = (
            dudx=max(max_gradient_component_error.dudx, dudx_error),
            dudy=max(max_gradient_component_error.dudy, dudy_error),
            dvdx=max(max_gradient_component_error.dvdx, dvdx_error),
            dvdy=max(max_gradient_component_error.dvdy, dvdy_error),
        )
        max_velocity_gradient_error = max(
            max_velocity_gradient_error,
            dudx_error, dudy_error, dvdx_error, dvdy_error,
        )
        min_c_eig = min(min_c_eig, logfv_min_eig_sym2_2d(cxx, cxy, cyy))
    end
    errors = _logfv_channel_reference_errors(
        flow, tauxx_cpu, tauxy_cpu, tauyy_cpu, fx_poly_cpu, fy_poly_cpu,
        fx_total_cpu, fy_total_cpu, Float64(height_t), Float64(umax_t),
        Float64(uwall_t), Float64(lambda_t), Float64(prefactor_t), Float64(bsd_t),
    )

    return (;
        flow,
        Nx,
        Ny,
        dx=Float64(ref.dx),
        dy=Float64(ref.dy),
        height=Float64(height_t),
        width=Float64(width_t),
        umax=Float64(umax_t),
        uwall=Float64(uwall_t),
        lambda=Float64(lambda_t),
        prefactor=Float64(prefactor_t),
        beta=Float64(beta),
        Wi=Float64(Wi),
        bsd_fraction=Float64(bsd_t),
        initial,
        max_steps,
        polymer_substeps=selected_polymer_substeps,
        requested_polymer_substeps=polymer_substeps,
        subcycle_estimate,
        max_grad_norm_estimate=Float64(max_grad_norm),
        ux=Array(ux),
        uy=Array(uy),
        dudx=dudx_cpu,
        dudy=dudy_cpu,
        dvdx=dvdx_cpu,
        dvdy=dvdy_cpu,
        psixx=psixx_cpu,
        psixy=psixy_cpu,
        psiyy=psiyy_cpu,
        cxx=cxx_cpu,
        cxy=cxy_cpu,
        cyy=cyy_cpu,
        tauxx=tauxx_cpu,
        tauxy=tauxy_cpu,
        tauyy=tauyy_cpu,
        fx_poly=fx_poly_cpu,
        fy_poly=fy_poly_cpu,
        fx_total=fx_total_cpu,
        fy_total=fy_total_cpu,
        reference=ref,
        max_c_error,
        max_psi_error,
        max_velocity_gradient_error,
        max_gradient_component_error,
        min_c_eig,
        errors...,
    )
end

function _logfv_fenep_simple_shear_factor_2d(a, L2, ::Type{T}) where {T<:AbstractFloat}
    L2 > T(2) || throw(ArgumentError("FENE-P requires L_max^2 > 2 in 2D"))
    rhs = T(2) * a * a / L2
    f = one(T) + rhs
    for _ in 1:32
        g = f * f * (f - one(T)) - rhs
        gp = T(3) * f * f - T(2) * f
        f_next = max(one(T), f - g / gp)
        if abs(f_next - f) <= T(8) * eps(T) * max(one(T), abs(f_next))
            return f_next
        end
        f = f_next
    end
    return f
end

@inline function _logfv_solve3_2d(
    a11, a12, a13,
    a21, a22, a23,
    a31, a32, a33,
    b1, b2, b3,
)
    det = a11 * (a22 * a33 - a23 * a32) -
          a12 * (a21 * a33 - a23 * a31) +
          a13 * (a21 * a32 - a22 * a31)
    x = (
        b1 * (a22 * a33 - a23 * a32) -
        a12 * (b2 * a33 - a23 * b3) +
        a13 * (b2 * a32 - a22 * b3)
    ) / det
    y = (
        a11 * (b2 * a33 - a23 * b3) -
        b1 * (a21 * a33 - a23 * a31) +
        a13 * (a21 * b3 - b2 * a31)
    ) / det
    z = (
        a11 * (a22 * b3 - b2 * a32) -
        a12 * (a21 * b3 - b2 * a31) +
        b1 * (a21 * a32 - a22 * a31)
    ) / det
    return x, y, z
end

@inline function _logfv_oldroydb_steady_conformation_from_gradient_2d(
    dudx, dudy, dvdx, dvdy, lambda,
)
    T = typeof(dudx + dudy + dvdx + dvdy + lambda)
    inv_lambda = inv(lambda)
    return _logfv_solve3_2d(
        inv_lambda - T(2) * dudx, -T(2) * dudy, zero(T),
        -dvdx, inv_lambda - dudx - dvdy, -dudy,
        zero(T), -T(2) * dvdx, inv_lambda - T(2) * dvdy,
        inv_lambda, zero(T), inv_lambda,
    )
end

@inline function _logfv_circle_tangential_shear_gradient_2d(x, y, radius, shear_rate)
    T = typeof(x + y + radius + shear_rate)
    r = hypot(x, y)
    if r <= eps(T)
        return zero(T), zero(T), zero(T), zero(T)
    end
    h = one(T) - radius / r
    hx = radius * x / (r * r * r)
    hy = radius * y / (r * r * r)
    return (
        -shear_rate * y * hx,
        -shear_rate * (h + y * hy),
        shear_rate * (h + x * hx),
        shear_rate * x * hy,
    )
end

"""
    run_viscoelastic_logfv_frozen_circle_shear_cde_2d(; kwargs...)

Run a standalone analytical log-FV CDE canary on a coherent embedded circle.

The velocity is imposed as simple shear, `u_x = shear_rate * (y - cy)`,
`u_y = 0`, and the velocity gradient supplied to the constitutive source is
the corresponding analytical constant gradient. The initial conformation is
the steady simple-shear solution for `polymer_model=:oldroydb` or
`polymer_model=:fenep`:

```text
Oldroyd-B: Cxx = 1 + 2a^2, Cxy = a, Cyy = 1
FENE-P:    f^3 - f^2 = 2a^2 / L_max^2, Cyy = 1/f,
           Cxy = a/f^2, Cxx = 1/f + 2a^2/f^3
a = lambda * shear_rate
```

The canary exercises:

```text
imposed u -> FVFD embedded face velocities -> FVFD log-field advection
          -> log-C source -> tau_p
```

It intentionally does not use the embedded velocity-gradient operator: a global
affine shear field does not satisfy stationary no-slip on the internal circle.
"""
function run_viscoelastic_logfv_frozen_circle_shear_cde_2d(;
    Nx::Integer=32,
    Ny::Integer=32,
    cx::Real=Nx / 2,
    cy::Real=Ny / 2,
    radius::Real=min(Nx, Ny) / 5,
    shear_rate::Real=0.012,
    lambda::Real=3.0,
    prefactor::Real=0.02,
    polymer_model=:oldroydb,
    L_max::Real=10.0,
    dt::Real=0.01,
    samples::Integer=32,
    backend=KernelAbstractions.CPU(),
    T=Float64,
)
    Nx >= 8 || throw(ArgumentError("Nx must be >= 8"))
    Ny >= 8 || throw(ArgumentError("Ny must be >= 8"))
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
    model_cfg = _logfv_polymer_model_config(polymer_model, L_max, T)
    model_code = model_cfg.model_code
    L2_t = model_cfg.L2

    bc = FVFDDomainBC2D(;
        west=:periodic, east=:periodic, south=:periodic, north=:periodic,
    )
    geometry_h = fvfd_geometry_from_circle_2d(
        Nx_i, Ny_i, one(T), one(T), bc, cx_t, cy_t, radius_t;
        FT=T, samples=samples,
    )
    geometry = fvfd_transfer_geometry_2d(geometry_h, backend, T)

    wi_shear = lambda_t * shear_t
    f_ref = if model_code == LOGFV_MODEL_FENEP
        _logfv_fenep_simple_shear_factor_2d(wi_shear, L2_t, T)
    else
        one(T)
    end
    cxx_ref = if model_code == LOGFV_MODEL_FENEP
        inv(f_ref) + T(2) * wi_shear * wi_shear / (f_ref * f_ref * f_ref)
    else
        one(T) + T(2) * wi_shear * wi_shear
    end
    cxy_ref = wi_shear / (f_ref * f_ref)
    cyy_ref = inv(f_ref)
    psixx_ref, psixy_ref, psiyy_ref = logfv_log_spd_sym2_2d(
        cxx_ref, cxy_ref, cyy_ref,
    )
    tauxx_ref = prefactor_t * (f_ref * cxx_ref - one(T))
    tauxy_ref = prefactor_t * f_ref * cxy_ref
    tauyy_ref = prefactor_t * (f_ref * cyy_ref - one(T))

    ux_h = Matrix{T}(undef, Nx_i, Ny_i)
    uy_h = zeros(T, Nx_i, Ny_i)
    @inbounds for j in 1:Ny_i, i in 1:Nx_i
        y = T(j) - T(0.5)
        ux_h[i, j] = shear_t * (y - cy_t)
    end

    psixx_h = fill(psixx_ref, Nx_i, Ny_i)
    psixy_h = fill(psixy_ref, Nx_i, Ny_i)
    psiyy_h = fill(psiyy_ref, Nx_i, Ny_i)
    dudy_h = fill(shear_t, Nx_i, Ny_i)

    ux = KernelAbstractions.allocate(backend, T, Nx_i, Ny_i)
    uy = KernelAbstractions.allocate(backend, T, Nx_i, Ny_i)
    psixx = KernelAbstractions.allocate(backend, T, Nx_i, Ny_i)
    psixy = KernelAbstractions.allocate(backend, T, Nx_i, Ny_i)
    psiyy = KernelAbstractions.allocate(backend, T, Nx_i, Ny_i)
    dudy = KernelAbstractions.allocate(backend, T, Nx_i, Ny_i)
    copyto!(ux, ux_h)
    copyto!(uy, uy_h)
    copyto!(psixx, psixx_h)
    copyto!(psixy, psixy_h)
    copyto!(psiyy, psiyy_h)
    copyto!(dudy, dudy_h)

    psixx_adv = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    psixy_adv = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    psiyy_adv = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    psixx_next = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    psixy_next = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    psiyy_next = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    ux_face = KernelAbstractions.zeros(backend, T, Nx_i + 1, Ny_i)
    uy_face = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i + 1)
    dudx = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    dvdx = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    dvdy = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    tauxx = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    tauxy = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    tauyy = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)

    zero_bc_h = FVFDFieldBC2D(
        zeros(T, Ny_i), zeros(T, Ny_i), zeros(T, Nx_i), zeros(T, Nx_i),
    )
    psixx_bc_h = FVFDFieldBC2D(
        fill(psixx_ref, Ny_i), fill(psixx_ref, Ny_i),
        fill(psixx_ref, Nx_i), fill(psixx_ref, Nx_i),
    )
    psixy_bc_h = FVFDFieldBC2D(
        fill(psixy_ref, Ny_i), fill(psixy_ref, Ny_i),
        fill(psixy_ref, Nx_i), fill(psixy_ref, Nx_i),
    )
    psiyy_bc_h = FVFDFieldBC2D(
        fill(psiyy_ref, Ny_i), fill(psiyy_ref, Ny_i),
        fill(psiyy_ref, Nx_i), fill(psiyy_ref, Nx_i),
    )
    ux_bc = fvfd_transfer_field_bc_2d(zero_bc_h, backend, T, Nx_i, Ny_i, bc; name=:ux_bc)
    uy_bc = fvfd_transfer_field_bc_2d(zero_bc_h, backend, T, Nx_i, Ny_i, bc; name=:uy_bc)
    psixx_bc = fvfd_transfer_field_bc_2d(
        psixx_bc_h, backend, T, Nx_i, Ny_i, bc; name=:psixx_bc, default=psixx_ref,
    )
    psixy_bc = fvfd_transfer_field_bc_2d(
        psixy_bc_h, backend, T, Nx_i, Ny_i, bc; name=:psixy_bc, default=psixy_ref,
    )
    psiyy_bc = fvfd_transfer_field_bc_2d(
        psiyy_bc_h, backend, T, Nx_i, Ny_i, bc; name=:psiyy_bc, default=psiyy_ref,
    )

    fvfd_sym2_advect_upwind_embedded_2d!(
        psixx_adv, psixy_adv, psiyy_adv,
        psixx, psixy, psiyy,
        psixx_bc, psixy_bc, psiyy_bc,
        ux_face, uy_face, ux, uy, geometry, ux_bc, uy_bc, dt_t; sync=false,
    )
    logfv_step_constitutive_log_2d!(
        psixx_next, psixy_next, psiyy_next,
        psixx_adv, psixy_adv, psiyy_adv,
        dudx, dudy, dvdx, dvdy,
        lambda_t, dt_t, model_code, L2_t; sync=false,
    )
    logfv_stress_from_log_2d!(
        tauxx, tauxy, tauyy,
        psixx_next, psixy_next, psiyy_next, prefactor_t;
        model_code, L2=L2_t, sync=false,
    )
    KernelAbstractions.synchronize(backend)

    psixx_adv_cpu = Array(psixx_adv)
    psixy_adv_cpu = Array(psixy_adv)
    psiyy_adv_cpu = Array(psiyy_adv)
    psixx_cpu = Array(psixx_next)
    psixy_cpu = Array(psixy_next)
    psiyy_cpu = Array(psiyy_next)
    tauxx_cpu = Array(tauxx)
    tauxy_cpu = Array(tauxy)
    tauyy_cpu = Array(tauyy)
    cxx_cpu = similar(psixx_cpu)
    cxy_cpu = similar(psixy_cpu)
    cyy_cpu = similar(psiyy_cpu)

    max_adv_psi_error = 0.0
    max_psi_error = 0.0
    max_c_error = 0.0
    max_tau_error = 0.0
    max_tauxx_error = 0.0
    max_tauxy_error = 0.0
    max_tauyy_error = 0.0
    min_c_eig = Inf
    fluid_cells = 0
    @inbounds for j in 1:Ny_i, i in 1:Nx_i
        cxx, cxy, cyy = logfv_exp_sym2_2d(
            psixx_cpu[i, j], psixy_cpu[i, j], psiyy_cpu[i, j],
        )
        cxx_cpu[i, j] = cxx
        cxy_cpu[i, j] = cxy
        cyy_cpu[i, j] = cyy
        if geometry_h.is_solid[i, j]
            continue
        end
        fluid_cells += 1
        max_adv_psi_error = max(
            max_adv_psi_error,
            abs(psixx_adv_cpu[i, j] - psixx_ref),
            abs(psixy_adv_cpu[i, j] - psixy_ref),
            abs(psiyy_adv_cpu[i, j] - psiyy_ref),
        )
        max_psi_error = max(
            max_psi_error,
            abs(psixx_cpu[i, j] - psixx_ref),
            abs(psixy_cpu[i, j] - psixy_ref),
            abs(psiyy_cpu[i, j] - psiyy_ref),
        )
        max_c_error = max(
            max_c_error,
            abs(cxx - cxx_ref),
            abs(cxy - cxy_ref),
            abs(cyy - cyy_ref),
        )
        tauxx_error = abs(tauxx_cpu[i, j] - tauxx_ref)
        tauxy_error = abs(tauxy_cpu[i, j] - tauxy_ref)
        tauyy_error = abs(tauyy_cpu[i, j] - tauyy_ref)
        max_tauxx_error = max(max_tauxx_error, tauxx_error)
        max_tauxy_error = max(max_tauxy_error, tauxy_error)
        max_tauyy_error = max(max_tauyy_error, tauyy_error)
        max_tau_error = max(max_tau_error, tauxx_error, tauxy_error, tauyy_error)
        min_c_eig = min(min_c_eig, logfv_min_eig_sym2_2d(cxx, cxy, cyy))
    end

    wall_length = Float64(sum(Array(geometry_h.embedded.wall_fraction)))
    expected_wall_length = 2 * pi * Float64(radius_t)
    cut_cells = count(!iszero, Array(geometry_h.embedded.cut_count))

    return (;
        flow=:imposed_shear_circle,
        Nx=Nx_i,
        Ny=Ny_i,
        cx=Float64(cx_t),
        cy=Float64(cy_t),
        radius=Float64(radius_t),
        shear_rate=Float64(shear_t),
        lambda=Float64(lambda_t),
        prefactor=Float64(prefactor_t),
        polymer_model=model_cfg.polymer_model,
        L_max=model_cfg.L_max,
        fene_factor=Float64(f_ref),
        dt=Float64(dt_t),
        samples=Int(samples),
        fluid_cells,
        cut_cells,
        wall_length,
        expected_wall_length,
        wall_length_error=abs(wall_length - expected_wall_length),
        geometry=geometry_h,
        ux=Array(ux),
        uy=Array(uy),
        ux_face=Array(ux_face),
        uy_face=Array(uy_face),
        psixx_adv=psixx_adv_cpu,
        psixy_adv=psixy_adv_cpu,
        psiyy_adv=psiyy_adv_cpu,
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
            cxx=Float64(cxx_ref),
            cxy=Float64(cxy_ref),
            cyy=Float64(cyy_ref),
            psixx=Float64(psixx_ref),
            psixy=Float64(psixy_ref),
            psiyy=Float64(psiyy_ref),
            tauxx=Float64(tauxx_ref),
            tauxy=Float64(tauxy_ref),
            tauyy=Float64(tauyy_ref),
        ),
        max_adv_psi_error,
        max_psi_error,
        max_c_error,
        max_tau_error,
        max_tau_component_error=(;
            tauxx=max_tauxx_error,
            tauxy=max_tauxy_error,
            tauyy=max_tauyy_error,
        ),
        min_c_eig,
    )
end

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

function _logfv_lbm_poiseuille_reference(Fx_body, nu_total, Ny)
    return [Fx_body / (2 * nu_total) * (j - 0.5) * (Ny + 0.5 - j) for j in 1:Ny]
end

"""
    run_viscoelastic_logfv_poiseuille_frozen_force_2d(; kwargs...)

Run the first coupled LBM/log-FV macro canary on a periodic channel.

The polymer field is frozen at the analytical Oldroyd-B Poiseuille solution,
then the production log-FV kernels reconstruct `tau_p`, `div(tau_p)`, and the
BSD-corrected force. The solvent LBM is advanced with that force field. This
isolates the momentum-coupling contract:

```text
body force + log-FV polymer force + BSD correction -> total-viscosity profile
```

It does not validate polymer advection or near-wall polymer boundary
conditions. Those stay in lower canaries before square/obstacle flows.
"""
function run_viscoelastic_logfv_poiseuille_frozen_force_2d(;
    Nx::Integer=4,
    Ny::Integer=32,
    nu_s::Real=0.04,
    nu_p::Real=0.06,
    Fx_body::Real=1e-5,
    lambda::Real=5.0,
    bsd_fraction::Real=0.0,
    force_boundary_fill::Symbol=:bc_aware,
    max_steps::Integer=12000,
    backend=KernelAbstractions.CPU(),
    T=Float64,
)
    Nx >= 3 || throw(ArgumentError("Nx must be >= 3"))
    Ny >= 5 || throw(ArgumentError("Ny must be >= 5"))
    nu_s > 0 || throw(ArgumentError("nu_s must be positive"))
    nu_p >= 0 || throw(ArgumentError("nu_p must be non-negative"))
    lambda > 0 || throw(ArgumentError("lambda must be positive"))

    nu_s_t = T(nu_s)
    nu_p_t = T(nu_p)
    nu_total_t = nu_s_t + nu_p_t
    bsd_t = T(bsd_fraction)
    Fx_body_t = T(Fx_body)
    lambda_t = T(lambda)

    nu_lbm_t = nu_s_t + bsd_t * nu_p_t
    nu_lbm_t > zero(T) || throw(ArgumentError("nu_s + bsd_fraction * nu_p must be positive"))
    force_boundary_fill in (:bc_aware, :nearest, :none) ||
        throw(ArgumentError("force_boundary_fill must be :bc_aware, :nearest, or :none"))

    height_t = T(Ny)
    width_t = T(Nx)
    umax_t = Fx_body_t * height_t * height_t / (T(8) * nu_total_t)
    prefactor_t = iszero(lambda_t) ? zero(T) : nu_p_t / lambda_t

    channel = run_viscoelastic_logfv_channel_2d(;
        Nx=Nx,
        Ny=Ny,
        flow=:poiseuille,
        height=height_t,
        width=width_t,
        umax=umax_t,
        uwall=zero(T),
        lambda=lambda_t,
        prefactor=prefactor_t,
        bsd_fraction=bsd_t,
        backend=backend,
        T=T,
    )

    fx_total_h = T.(channel.fx_total)
    fy_total_h = T.(channel.fy_total)
    for j in 1:Ny, i in 1:Nx
        fx_total_h[i, j] += Fx_body_t
    end

    fx_total = KernelAbstractions.allocate(backend, T, Nx, Ny)
    fy_total = KernelAbstractions.allocate(backend, T, Nx, Ny)
    copyto!(fx_total, fx_total_h)
    copyto!(fy_total, fy_total_h)
    if force_boundary_fill === :nearest
        logfv_fill_nearest_boundary_2d!(fx_total, fy_total)
    end

    config = LBMConfig(D2Q9(); Nx=Nx, Ny=Ny, ν=Float64(nu_lbm_t), u_lid=0.0, max_steps=max_steps)
    state = initialize_2d(config, T; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    rho, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid
    omega_t = T(omega(config))

    for _ in 1:max_steps
        stream_periodic_x_wall_y_2d!(f_out, f_in, Nx, Ny)
        collide_guo_field_2d!(f_out, is_solid, fx_total, fy_total, omega_t)
        logfv_compute_macroscopic_forced_field_2d!(rho, ux, uy, f_out, fx_total, fy_total; sync=false)
        f_in, f_out = f_out, f_in
    end
    KernelAbstractions.synchronize(backend)

    ux_cpu = Array(ux)
    uy_cpu = Array(uy)
    rho_cpu = Array(rho)
    reference_u = _logfv_lbm_poiseuille_reference(Float64(Fx_body_t), Float64(nu_total_t), Ny)
    mean_ux = [sum(@view ux_cpu[:, j]) / Nx for j in 1:Ny]
    interior = 3:(Ny - 2)
    max_abs_error = maximum(abs.(mean_ux[interior] .- reference_u[interior]))
    max_ref = maximum(abs.(reference_u[interior]))
    max_rel_error = max_abs_error / max(max_ref, eps(Float64))
    max_uy = maximum(abs, uy_cpu[:, interior])

    return (;
        Nx,
        Ny,
        nu_s=Float64(nu_s_t),
        nu_p=Float64(nu_p_t),
        nu_total=Float64(nu_total_t),
        nu_lbm=Float64(nu_lbm_t),
        Fx_body=Float64(Fx_body_t),
        lambda=Float64(lambda_t),
        bsd_fraction=Float64(bsd_t),
        force_boundary_fill,
        max_steps,
        rho=rho_cpu,
        ux=ux_cpu,
        uy=uy_cpu,
        ux_mean=mean_ux,
        reference_ux=reference_u,
        fx_total=Array(fx_total),
        fy_total=Array(fy_total),
        polymer_channel=channel,
        max_abs_error,
        max_rel_error,
        max_uy,
    )
end

"""
    run_viscoelastic_logfv_poiseuille_coupled_2d(; kwargs...)

Run a coarse coupled channel canary with dynamic log-FV polymer stress.

This keeps the flow fully developed and periodic in `x`, so polymer advection is
identically zero. The canary exercises the local coupled loop without obstacle
or curved-wall complications:

```text
LBM u -> wall-exact channel grad(u) -> log-C Oldroyd-B source
      -> tau_p -> div(tau_p) + BSD -> Guo force -> LBM u
```

`polymer_substeps` is a time-integration convergence control for the current
Lie source split. It is not a physical parameter and must not be fitted to a
benchmark. Use `:auto` to choose a global patch value from the source
subcycling estimator. The estimator limits per-step relaxation, per-step
deformation, and memory-time deformation `lambda * ||grad(u)||`; future Strang
or local affine source solves should reduce this requirement.
"""
function run_viscoelastic_logfv_poiseuille_coupled_2d(;
    Nx::Integer=6,
    Ny::Integer=20,
    nu_s::Real=0.04,
    nu_p::Real=0.06,
    Fx_body::Real=1e-5,
    lambda::Real=5.0,
    bsd_fraction::Real=1.0,
    polymer_substeps=:auto,
    subcycle_relative_tolerance::Real=0.01,
    max_deformation_increment::Real=0.05,
    max_memory_deformation_increment::Real=0.07,
    max_polymer_substeps::Integer=64,
    force_boundary_fill::Symbol=:bc_aware,
    max_steps::Integer=10000,
    backend=KernelAbstractions.CPU(),
    T=Float64,
)
    Nx >= 3 || throw(ArgumentError("Nx must be >= 3"))
    Ny >= 5 || throw(ArgumentError("Ny must be >= 5"))
    nu_s > 0 || throw(ArgumentError("nu_s must be positive"))
    nu_p >= 0 || throw(ArgumentError("nu_p must be non-negative"))
    lambda > 0 || throw(ArgumentError("lambda must be positive"))
    force_boundary_fill in (:bc_aware, :nearest, :none) ||
        throw(ArgumentError("force_boundary_fill must be :bc_aware, :nearest, or :none"))

    nu_s_t = T(nu_s)
    nu_p_t = T(nu_p)
    nu_total_t = nu_s_t + nu_p_t
    bsd_t = T(bsd_fraction)
    Fx_body_t = T(Fx_body)
    lambda_t = T(lambda)
    prefactor_t = nu_p_t / lambda_t
    nu_lbm_t = nu_s_t + bsd_t * nu_p_t
    nu_lbm_t > zero(T) || throw(ArgumentError("nu_s + bsd_fraction * nu_p must be positive"))
    dx = one(T)
    dy = one(T)
    max_grad_norm_estimate = abs(Fx_body_t) * T(Ny) / (T(2) * nu_total_t)
    subcycle_estimate = logfv_oldroydb_subcycle_estimate(
        Float64(max_grad_norm_estimate),
        Float64(lambda_t),
        1.0;
        relative_tolerance=Float64(subcycle_relative_tolerance),
        max_deformation_increment=Float64(max_deformation_increment),
        max_memory_deformation_increment=Float64(max_memory_deformation_increment),
        min_substeps=1,
        max_substeps=max_polymer_substeps,
    )
    selected_polymer_substeps = if polymer_substeps === :auto
        subcycle_estimate.recommended
    elseif polymer_substeps isa Integer
        polymer_substeps >= 1 || throw(ArgumentError("polymer_substeps must be >= 1"))
        polymer_substeps
    else
        throw(ArgumentError("polymer_substeps must be an integer or :auto"))
    end
    dt_poly = one(T) / T(selected_polymer_substeps)

    config = LBMConfig(D2Q9(); Nx=Nx, Ny=Ny, ν=Float64(nu_lbm_t), u_lid=0.0, max_steps=max_steps)
    state = initialize_2d(config, T; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    rho, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid
    omega_t = T(omega(config))

    psixx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psiyy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixx_next = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixy_next = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psiyy_next = KernelAbstractions.zeros(backend, T, Nx, Ny)
    dudx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    dudy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    dvdx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    dvdy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tauxx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tauxy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tauyy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    fx_poly = KernelAbstractions.zeros(backend, T, Nx, Ny)
    fy_poly = KernelAbstractions.zeros(backend, T, Nx, Ny)
    fx_total = KernelAbstractions.zeros(backend, T, Nx, Ny)
    fy_total = KernelAbstractions.zeros(backend, T, Nx, Ny)
    logfv_bc = logfv_periodicx_wally_bcspec_2d()

    for _ in 1:max_steps
        logfv_velocity_gradient_bc_aware_2d!(
            dudx, dudy, dvdx, dvdy, ux, uy, is_solid, dx, dy, logfv_bc;
            sync=false,
        )
        for _ in 1:selected_polymer_substeps
            logfv_step_oldroydb_log_2d!(
                psixx_next, psixy_next, psiyy_next,
                psixx, psixy, psiyy,
                dudx, dudy, dvdx, dvdy,
                lambda_t, dt_poly;
                sync=false,
            )
            psixx, psixx_next = psixx_next, psixx
            psixy, psixy_next = psixy_next, psixy
            psiyy, psiyy_next = psiyy_next, psiyy
        end
        logfv_stress_from_log_2d!(tauxx, tauxy, tauyy, psixx, psixy, psiyy, prefactor_t; sync=false)
        logfv_polymer_force_bc_aware_2d!(
            fx_poly, fy_poly, tauxx, tauxy, tauyy, is_solid, dx, dy, logfv_bc;
            sync=false,
        )
        logfv_bsd_correct_force_bc_aware_2d!(
            fx_total, fy_total, fx_poly, fy_poly, ux, uy, is_solid, bsd_t, nu_p_t, dx, dy,
            logfv_bc;
            sync=false,
        )
        if force_boundary_fill === :nearest
            logfv_fill_nearest_boundary_2d!(fx_total, fy_total; sync=false)
        end
        logfv_add_constant_force_2d!(fx_total, fy_total, Fx_body_t, zero(T); sync=false)

        stream_periodic_x_wall_y_2d!(f_out, f_in, Nx, Ny)
        collide_guo_field_2d!(f_out, is_solid, fx_total, fy_total, omega_t)
        logfv_compute_macroscopic_forced_field_2d!(rho, ux, uy, f_out, fx_total, fy_total; sync=false)
        f_in, f_out = f_out, f_in
    end
    KernelAbstractions.synchronize(backend)

    ux_cpu = Array(ux)
    uy_cpu = Array(uy)
    reference_u = _logfv_lbm_poiseuille_reference(Float64(Fx_body_t), Float64(nu_total_t), Ny)
    mean_ux = [sum(@view ux_cpu[:, j]) / Nx for j in 1:Ny]
    interior = 3:(Ny - 2)
    max_abs_error = maximum(abs.(mean_ux[interior] .- reference_u[interior]))
    max_ref = maximum(abs.(reference_u[interior]))
    max_rel_error = max_abs_error / max(max_ref, eps(Float64))
    max_uy = maximum(abs, uy_cpu[:, interior])

    psixx_cpu = Array(psixx)
    psixy_cpu = Array(psixy)
    psiyy_cpu = Array(psiyy)
    min_c_eig = Inf
    for j in 1:Ny, i in 1:Nx
        cxx, cxy, cyy = logfv_exp_sym2_2d(psixx_cpu[i, j], psixy_cpu[i, j], psiyy_cpu[i, j])
        min_c_eig = min(min_c_eig, logfv_min_eig_sym2_2d(cxx, cxy, cyy))
    end

    return (;
        Nx,
        Ny,
        nu_s=Float64(nu_s_t),
        nu_p=Float64(nu_p_t),
        nu_total=Float64(nu_total_t),
        nu_lbm=Float64(nu_lbm_t),
        Fx_body=Float64(Fx_body_t),
        lambda=Float64(lambda_t),
        bsd_fraction=Float64(bsd_t),
        polymer_substeps=selected_polymer_substeps,
        requested_polymer_substeps=polymer_substeps,
        subcycle_estimate,
        max_grad_norm_estimate=Float64(max_grad_norm_estimate),
        force_boundary_fill,
        max_steps,
        rho=Array(rho),
        ux=ux_cpu,
        uy=uy_cpu,
        ux_mean=mean_ux,
        reference_ux=reference_u,
        psixx=psixx_cpu,
        psixy=psixy_cpu,
        psiyy=psiyy_cpu,
        fx_total=Array(fx_total),
        fy_total=Array(fy_total),
        min_c_eig,
        max_abs_error,
        max_rel_error,
        max_uy,
    )
end

function _logfv_square_obstacle_mask(Nx::Int, Ny::Int, side::Int, cx::Int, cy::Int)
    side >= 2 || throw(ArgumentError("side must be >= 2"))
    3 <= cx <= Nx - 2 || throw(ArgumentError("cx must leave fluid columns around the square"))
    3 <= cy <= Ny - 2 || throw(ArgumentError("cy must leave fluid rows around the square"))
    half_lo = (side - 1) ÷ 2
    half_hi = side ÷ 2
    i1 = cx - half_lo
    i2 = cx + half_hi
    j1 = cy - half_lo
    j2 = cy + half_hi
    2 <= i1 <= i2 <= Nx - 1 || throw(ArgumentError("square obstacle must leave one fluid column on each side"))
    2 <= j1 <= j2 <= Ny - 1 || throw(ArgumentError("square obstacle must leave one fluid row on each side"))
    is_solid = fill(false, Nx, Ny)
    is_solid[i1:i2, j1:j2] .= true
    return is_solid
end

"""
    run_viscoelastic_logfv_square_periodic_2d(; kwargs...)

Run the first coarse macro-flow canary for the log-FV backend around an
axis-aligned square obstacle. The domain is periodic in `x`, has halfway walls
at `y`, and is driven by a uniform body force.

This is a stability/coupling canary, not a drag benchmark. It exercises
solid-aware velocity gradients, solid-aware `Psi` advection, local
log-conformation source update, solid-aware polymer force, and Guo coupling.
BSD uses the same solid-aware compact stencil as the polymer force path, so
low-beta checks exercise the operator used by the coarse obstacle flow.
"""
function run_viscoelastic_logfv_square_periodic_2d(;
    Nx::Integer=48,
    Ny::Integer=24,
    side::Integer=6,
    cx::Integer=Nx ÷ 3,
    cy::Integer=Ny ÷ 2,
    nu_s::Real=0.08,
    nu_p::Real=0.02,
    Fx_body::Real=1e-6,
    lambda::Real=2.0,
    bsd_fraction::Real=0.0,
    polymer_substeps=:auto,
    subcycle_relative_tolerance::Real=0.01,
    max_deformation_increment::Real=0.05,
    max_memory_deformation_increment::Real=0.07,
    max_polymer_substeps::Integer=64,
    max_steps::Integer=500,
    backend=KernelAbstractions.CPU(),
    T=Float64,
)
    Nx >= 8 || throw(ArgumentError("Nx must be >= 8"))
    Ny >= 8 || throw(ArgumentError("Ny must be >= 8"))
    nu_s > 0 || throw(ArgumentError("nu_s must be positive"))
    nu_p >= 0 || throw(ArgumentError("nu_p must be non-negative"))
    lambda > 0 || throw(ArgumentError("lambda must be positive"))
    0 <= bsd_fraction <= 1 || throw(ArgumentError("bsd_fraction must be in [0, 1]"))

    Nx_i = Int(Nx)
    Ny_i = Int(Ny)
    side_i = Int(side)
    cx_i = Int(cx)
    cy_i = Int(cy)
    is_solid_h = _logfv_square_obstacle_mask(Nx_i, Ny_i, side_i, cx_i, cy_i)

    nu_s_t = T(nu_s)
    nu_p_t = T(nu_p)
    nu_total_t = nu_s_t + nu_p_t
    bsd_t = T(bsd_fraction)
    Fx_body_t = T(Fx_body)
    lambda_t = T(lambda)
    prefactor_t = nu_p_t / lambda_t
    nu_lbm_t = nu_s_t + bsd_t * nu_p_t
    nu_lbm_t > zero(T) || throw(ArgumentError("nu_s + bsd_fraction * nu_p must be positive"))
    dx = one(T)
    dy = one(T)

    max_grad_norm_estimate = abs(Fx_body_t) * T(Ny_i) / (T(2) * max(nu_total_t, eps(T)))
    subcycle_estimate = logfv_oldroydb_subcycle_estimate(
        Float64(max_grad_norm_estimate),
        Float64(lambda_t),
        1.0;
        relative_tolerance=Float64(subcycle_relative_tolerance),
        max_deformation_increment=Float64(max_deformation_increment),
        max_memory_deformation_increment=Float64(max_memory_deformation_increment),
        min_substeps=1,
        max_substeps=max_polymer_substeps,
    )
    selected_polymer_substeps = if polymer_substeps === :auto
        subcycle_estimate.recommended
    elseif polymer_substeps isa Integer
        polymer_substeps >= 1 || throw(ArgumentError("polymer_substeps must be >= 1"))
        polymer_substeps
    else
        throw(ArgumentError("polymer_substeps must be an integer or :auto"))
    end
    dt_poly = one(T) / T(selected_polymer_substeps)

    config = LBMConfig(D2Q9(); Nx=Nx_i, Ny=Ny_i, ν=Float64(nu_lbm_t), u_lid=0.0, max_steps=max_steps)
    state = initialize_2d(config, T; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    rho, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid
    copyto!(is_solid, is_solid_h)
    omega_t = T(omega(config))

    psixx = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    psixy = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    psiyy = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    psixx_adv = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    psixy_adv = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    psiyy_adv = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    psixx_next = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    psixy_next = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    psiyy_next = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    dudx = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    dudy = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    dvdx = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    dvdy = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    ux_west = KernelAbstractions.zeros(backend, T, Ny_i)
    ux_east = KernelAbstractions.zeros(backend, T, Ny_i)
    uy_south = KernelAbstractions.zeros(backend, T, Nx_i)
    uy_north = KernelAbstractions.zeros(backend, T, Nx_i)
    west_xx = KernelAbstractions.zeros(backend, T, Ny_i)
    west_xy = KernelAbstractions.zeros(backend, T, Ny_i)
    west_yy = KernelAbstractions.zeros(backend, T, Ny_i)
    east_xx = KernelAbstractions.zeros(backend, T, Ny_i)
    east_xy = KernelAbstractions.zeros(backend, T, Ny_i)
    east_yy = KernelAbstractions.zeros(backend, T, Ny_i)
    south_xx = KernelAbstractions.zeros(backend, T, Nx_i)
    south_xy = KernelAbstractions.zeros(backend, T, Nx_i)
    south_yy = KernelAbstractions.zeros(backend, T, Nx_i)
    north_xx = KernelAbstractions.zeros(backend, T, Nx_i)
    north_xy = KernelAbstractions.zeros(backend, T, Nx_i)
    north_yy = KernelAbstractions.zeros(backend, T, Nx_i)
    ux_face = KernelAbstractions.zeros(backend, T, Nx_i + 1, Ny_i)
    uy_face = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i + 1)
    tauxx = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    tauxy = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    tauyy = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    fx_poly = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    fy_poly = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    fx_total = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    fy_total = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    logfv_bc = logfv_periodicx_wally_bcspec_2d()

    for _ in 1:max_steps
        logfv_cell_velocity_to_faces_bc_aware_2d!(
            ux_face, uy_face, ux, uy, is_solid,
            ux_west, ux_east, uy_south, uy_north, logfv_bc;
            sync=false,
        )
        logfv_advect_upwind_bc_aware_2d!(
            psixx_adv, psixy_adv, psiyy_adv,
            psixx, psixy, psiyy,
            west_xx, west_xy, west_yy,
            east_xx, east_xy, east_yy,
            south_xx, south_xy, south_yy,
            north_xx, north_xy, north_yy,
            ux_face, uy_face, is_solid, dx, dy, logfv_bc, one(T);
            sync=false,
        )
        logfv_velocity_gradient_bc_aware_2d!(
            dudx, dudy, dvdx, dvdy, ux, uy, is_solid, dx, dy, logfv_bc;
            sync=false,
        )

        psixx_work, psixy_work, psiyy_work = psixx_adv, psixy_adv, psiyy_adv
        for _ in 1:selected_polymer_substeps
            logfv_step_oldroydb_log_2d!(
                psixx_next, psixy_next, psiyy_next,
                psixx_work, psixy_work, psiyy_work,
                dudx, dudy, dvdx, dvdy,
                lambda_t, dt_poly;
                sync=false,
            )
            psixx_work, psixx_next = psixx_next, psixx_work
            psixy_work, psixy_next = psixy_next, psixy_work
            psiyy_work, psiyy_next = psiyy_next, psiyy_work
        end
        psixx, psixx_adv = psixx_work, psixx
        psixy, psixy_adv = psixy_work, psixy
        psiyy, psiyy_adv = psiyy_work, psiyy

        logfv_stress_from_log_2d!(tauxx, tauxy, tauyy, psixx, psixy, psiyy, prefactor_t; sync=false)
        logfv_polymer_force_bc_aware_2d!(
            fx_poly, fy_poly, tauxx, tauxy, tauyy, is_solid, dx, dy, logfv_bc;
            sync=false,
        )
        logfv_bsd_correct_force_bc_aware_2d!(
            fx_total, fy_total, fx_poly, fy_poly, ux, uy, is_solid, bsd_t, nu_p_t, dx, dy,
            logfv_bc;
            sync=false,
        )
        logfv_add_constant_force_2d!(fx_total, fy_total, Fx_body_t, zero(T); sync=false)

        stream_periodic_x_wall_y_2d!(f_out, f_in, Nx_i, Ny_i)
        collide_guo_field_2d!(f_out, is_solid, fx_total, fy_total, omega_t)
        logfv_compute_macroscopic_forced_field_2d!(rho, ux, uy, f_out, fx_total, fy_total; sync=false)
        f_in, f_out = f_out, f_in
    end
    KernelAbstractions.synchronize(backend)

    ux_cpu = Array(ux)
    uy_cpu = Array(uy)
    rho_cpu = Array(rho)
    psixx_cpu = Array(psixx)
    psixy_cpu = Array(psixy)
    psiyy_cpu = Array(psiyy)
    min_c_eig = Inf
    max_speed = 0.0
    for j in 1:Ny_i, i in 1:Nx_i
        if !is_solid_h[i, j]
            cxx, cxy, cyy = logfv_exp_sym2_2d(psixx_cpu[i, j], psixy_cpu[i, j], psiyy_cpu[i, j])
            min_c_eig = min(min_c_eig, logfv_min_eig_sym2_2d(cxx, cxy, cyy))
            max_speed = max(max_speed, hypot(Float64(ux_cpu[i, j]), Float64(uy_cpu[i, j])))
        end
    end

    return (;
        Nx=Nx_i,
        Ny=Ny_i,
        side=side_i,
        cx=cx_i,
        cy=cy_i,
        nu_s=Float64(nu_s_t),
        nu_p=Float64(nu_p_t),
        nu_total=Float64(nu_total_t),
        nu_lbm=Float64(nu_lbm_t),
        Fx_body=Float64(Fx_body_t),
        lambda=Float64(lambda_t),
        bsd_fraction=Float64(bsd_t),
        polymer_substeps=selected_polymer_substeps,
        requested_polymer_substeps=polymer_substeps,
        subcycle_estimate,
        max_steps,
        rho=rho_cpu,
        ux=ux_cpu,
        uy=uy_cpu,
        psixx=psixx_cpu,
        psixy=psixy_cpu,
        psiyy=psiyy_cpu,
        fx_total=Array(fx_total),
        fy_total=Array(fy_total),
        is_solid=is_solid_h,
        min_c_eig,
        max_speed,
        rho_min=minimum(rho_cpu),
        rho_max=maximum(rho_cpu),
    )
end

"""
    run_viscoelastic_logfv_bfs_passive_2d(; kwargs...)

Run a passive log-FV polymer canary on a backward-facing-step geometry.

The hydrodynamic BFS field is first advanced with the modular LI-BB V2 +
Guo-field solvent step. The resulting velocity is then frozen while the
cell-centered log-conformation polymer equation is advanced with open-x
advection, solid-aware gradients, and the local Oldroyd-B source. No polymer
force is fed back into the solvent in this canary.
"""
function run_viscoelastic_logfv_bfs_passive_2d(;
    H_in::Integer=4,
    expansion_ratio::Integer=2,
    L_up::Integer=2,
    L_down::Integer=4,
    nu_s::Real=0.08,
    nu_p::Real=0.02,
    lambda::Real=5.0,
    u_mean::Real=0.01,
    Fx_body::Real=2e-7,
    hydro_steps::Integer=60,
    polymer_steps::Integer=20,
    polymer_substeps=:auto,
    max_memory_deformation_increment::Real=0.07,
    max_polymer_substeps::Integer=64,
    backend=KernelAbstractions.CPU(),
    T=Float64,
)
    H_in >= 3 || throw(ArgumentError("H_in must be >= 3"))
    expansion_ratio >= 2 || throw(ArgumentError("expansion_ratio must be >= 2"))
    nu_s > 0 || throw(ArgumentError("nu_s must be positive"))
    nu_p >= 0 || throw(ArgumentError("nu_p must be non-negative"))
    lambda > 0 || throw(ArgumentError("lambda must be positive"))
    hydro_steps >= 0 || throw(ArgumentError("hydro_steps must be non-negative"))
    polymer_steps >= 0 || throw(ArgumentError("polymer_steps must be non-negative"))

    geom_h = backward_facing_step_geometry_2d(;
        H_in=Int(H_in),
        expansion_ratio=Int(expansion_ratio),
        L_up=Int(L_up),
        L_down=Int(L_down),
        FT=T,
    )
    geom = transfer_step_geometry_2d(geom_h, backend)
    Nx, Ny = geom_h.Nx, geom_h.Ny
    is_solid = geom.is_solid
    q_wall = geom.q_wall
    is_solid_h = geom_h.is_solid

    nu_s_t = T(nu_s)
    nu_p_t = T(nu_p)
    lambda_t = T(lambda)
    prefactor_t = nu_p_t / lambda_t
    Fx_body_t = T(Fx_body)
    u_profile_h = parabolic_face_profile_2d(geom_h; face=:west, mean_velocity=T(u_mean), FT=T)
    u_profile = KernelAbstractions.allocate(backend, T, Ny)
    copyto!(u_profile, u_profile_h)
    bcspec = default_step_bcspec_2d(geom, u_profile, one(T))

    f_in = KernelAbstractions.allocate(backend, T, Nx, Ny, 9)
    f_out = KernelAbstractions.allocate(backend, T, Nx, Ny, 9)
    f_in_h = zeros(T, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx, q in 1:9
        ux0 = is_solid_h[i, j] ? zero(T) : u_profile_h[j]
        f_in_h[i, j, q] = equilibrium(D2Q9(), one(T), ux0, zero(T), q)
    end
    copyto!(f_in, f_in_h)
    fill!(f_out, zero(T))

    rho = KernelAbstractions.zeros(backend, T, Nx, Ny)
    ux = KernelAbstractions.zeros(backend, T, Nx, Ny)
    uy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    uwx = KernelAbstractions.zeros(backend, T, Nx, Ny, 9)
    uwy = KernelAbstractions.zeros(backend, T, Nx, Ny, 9)
    fx_h = [is_solid_h[i, j] ? zero(T) : Fx_body_t for i in 1:Nx, j in 1:Ny]
    fy_h = zeros(T, Nx, Ny)
    fx = KernelAbstractions.allocate(backend, T, Nx, Ny)
    fy = KernelAbstractions.allocate(backend, T, Nx, Ny)
    copyto!(fx, fx_h)
    copyto!(fy, fy_h)

    for _ in 1:hydro_steps
        fused_trt_libb_v2_guo_field_step!(
            f_out, f_in, rho, ux, uy, is_solid, q_wall, uwx, uwy, fx, fy,
            Nx, Ny, nu_s_t;
        )
        apply_bc_rebuild_2d!(f_out, f_in, bcspec, nu_s_t, Nx, Ny)
        f_in, f_out = f_out, f_in
    end
    logfv_compute_macroscopic_forced_field_2d!(rho, ux, uy, f_in, fx, fy)
    KernelAbstractions.synchronize(backend)

    ux_cpu_after_hydro = Array(ux)
    ux_east_h = copy(@view ux_cpu_after_hydro[Nx, :])
    ux_east = KernelAbstractions.allocate(backend, T, Ny)
    copyto!(ux_east, ux_east_h)

    subcycle_estimate = logfv_oldroydb_subcycle_estimate(
        0.0,
        Float64(lambda_t),
        1.0;
        max_memory_deformation_increment=Float64(max_memory_deformation_increment),
        min_substeps=1,
        max_substeps=max_polymer_substeps,
    )
    selected_polymer_substeps = if polymer_substeps === :auto
        subcycle_estimate.recommended
    elseif polymer_substeps isa Integer
        polymer_substeps >= 1 || throw(ArgumentError("polymer_substeps must be >= 1"))
        polymer_substeps
    else
        throw(ArgumentError("polymer_substeps must be an integer or :auto"))
    end
    dt_poly = one(T) / T(selected_polymer_substeps)

    psixx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psiyy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixx_adv = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixy_adv = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psiyy_adv = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixx_next = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixy_next = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psiyy_next = KernelAbstractions.zeros(backend, T, Nx, Ny)
    west_xx = KernelAbstractions.zeros(backend, T, Ny)
    west_xy = KernelAbstractions.zeros(backend, T, Ny)
    west_yy = KernelAbstractions.zeros(backend, T, Ny)
    east_xx = KernelAbstractions.zeros(backend, T, Ny)
    east_xy = KernelAbstractions.zeros(backend, T, Ny)
    east_yy = KernelAbstractions.zeros(backend, T, Ny)
    south_xx = KernelAbstractions.zeros(backend, T, Nx)
    south_xy = KernelAbstractions.zeros(backend, T, Nx)
    south_yy = KernelAbstractions.zeros(backend, T, Nx)
    north_xx = KernelAbstractions.zeros(backend, T, Nx)
    north_xy = KernelAbstractions.zeros(backend, T, Nx)
    north_yy = KernelAbstractions.zeros(backend, T, Nx)
    uy_south = KernelAbstractions.zeros(backend, T, Nx)
    uy_north = KernelAbstractions.zeros(backend, T, Nx)
    ux_face = KernelAbstractions.zeros(backend, T, Nx + 1, Ny)
    uy_face = KernelAbstractions.zeros(backend, T, Nx, Ny + 1)
    dudx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    dudy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    dvdx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    dvdy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tauxx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tauxy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tauyy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    logfv_bc = logfv_openx_wally_bcspec_2d()

    for _ in 1:polymer_steps
        logfv_cell_velocity_to_faces_bc_aware_2d!(
            ux_face, uy_face, ux, uy, is_solid,
            u_profile, ux_east, uy_south, uy_north, logfv_bc;
            sync=false,
        )
        logfv_advect_upwind_bc_aware_2d!(
            psixx_adv, psixy_adv, psiyy_adv,
            psixx, psixy, psiyy,
            west_xx, west_xy, west_yy,
            east_xx, east_xy, east_yy,
            south_xx, south_xy, south_yy,
            north_xx, north_xy, north_yy,
            ux_face, uy_face, is_solid, one(T), one(T), logfv_bc, one(T);
            sync=false,
        )
        logfv_velocity_gradient_bc_aware_2d!(
            dudx, dudy, dvdx, dvdy, ux, uy, is_solid, one(T), one(T), logfv_bc;
            sync=false,
        )

        psixx_work, psixy_work, psiyy_work = psixx_adv, psixy_adv, psiyy_adv
        for _ in 1:selected_polymer_substeps
            logfv_step_oldroydb_log_2d!(
                psixx_next, psixy_next, psiyy_next,
                psixx_work, psixy_work, psiyy_work,
                dudx, dudy, dvdx, dvdy,
                lambda_t, dt_poly;
                sync=false,
            )
            psixx_work, psixx_next = psixx_next, psixx_work
            psixy_work, psixy_next = psixy_next, psixy_work
            psiyy_work, psiyy_next = psiyy_next, psiyy_work
        end
        psixx, psixx_adv = psixx_work, psixx
        psixy, psixy_adv = psixy_work, psixy
        psiyy, psiyy_adv = psiyy_work, psiyy
    end
    logfv_stress_from_log_2d!(tauxx, tauxy, tauyy, psixx, psixy, psiyy, prefactor_t)
    KernelAbstractions.synchronize(backend)

    rho_cpu = Array(rho)
    ux_cpu = Array(ux)
    uy_cpu = Array(uy)
    psixx_cpu = Array(psixx)
    psixy_cpu = Array(psixy)
    psiyy_cpu = Array(psiyy)
    tauxx_cpu = Array(tauxx)
    tauxy_cpu = Array(tauxy)
    tauyy_cpu = Array(tauyy)

    min_c_eig = Inf
    max_speed = 0.0
    for j in 1:Ny, i in 1:Nx
        if !is_solid_h[i, j]
            cxx, cxy, cyy = logfv_exp_sym2_2d(psixx_cpu[i, j], psixy_cpu[i, j], psiyy_cpu[i, j])
            min_c_eig = min(min_c_eig, logfv_min_eig_sym2_2d(cxx, cxy, cyy))
            max_speed = max(max_speed, hypot(Float64(ux_cpu[i, j]), Float64(uy_cpu[i, j])))
        end
    end
    fluid_mask = .!is_solid_h
    max_abs_psi = max(maximum(abs, psixx_cpu), maximum(abs, psixy_cpu), maximum(abs, psiyy_cpu))
    max_abs_tau = max(maximum(abs, tauxx_cpu), maximum(abs, tauxy_cpu), maximum(abs, tauyy_cpu))

    return (;
        geometry=geom_h,
        Nx,
        Ny,
        nu_s=Float64(nu_s_t),
        nu_p=Float64(nu_p_t),
        lambda=Float64(lambda_t),
        u_mean=Float64(u_mean),
        Fx_body=Float64(Fx_body_t),
        hydro_steps,
        polymer_steps,
        polymer_substeps=selected_polymer_substeps,
        subcycle_estimate,
        rho=rho_cpu,
        ux=ux_cpu,
        uy=uy_cpu,
        psixx=psixx_cpu,
        psixy=psixy_cpu,
        psiyy=psiyy_cpu,
        tauxx=tauxx_cpu,
        tauxy=tauxy_cpu,
        tauyy=tauyy_cpu,
        is_solid=is_solid_h,
        min_c_eig,
        max_speed,
        max_abs_psi,
        max_abs_tau,
        rho_min=minimum(rho_cpu[fluid_mask]),
        rho_max=maximum(rho_cpu[fluid_mask]),
    )
end
