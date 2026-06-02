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

