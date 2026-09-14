#!/usr/bin/env julia

using Printf
using KernelAbstractions
using Kraken

try
    @eval using Metal
catch
end

const OUTDIR = joinpath("scratch", "M57")
const CSV = joinpath(OUTDIR, "L5b_poiseuille_steady.csv")

function backend_choice()
    requested = lowercase(get(ENV, "KRAKEN_BACKEND", "metal"))
    if requested == "metal" && isdefined(Main, :Metal)
        try
            metal = getfield(Main, :Metal)
            @eval KernelAbstractions.allocate(::Metal.MetalBackend, ::Type{T}, dims::Tuple;
                                              unified=nothing) where {T} =
                Metal.MtlArray{T}(undef, dims)
            @eval KernelAbstractions.zeros(::Metal.MetalBackend, ::Type{T}, dims::Tuple;
                                           unified=nothing) where {T} =
                Metal.zeros(T, dims)
            return metal.MetalBackend(), Float32, "metal"
        catch err
            @printf("M57_L5b backend_warning=metal_unavailable error=%s\n", sprint(showerror, err))
        end
    end
    return KernelAbstractions.CPU(), Float64, "cpu"
end

function channel_qwall(Nx, Ny, ::Type{T}) where {T}
    q = zeros(T, Nx, Ny, 9)
    solid = fill(false, Nx, Ny)
    solid[:, 1] .= true
    solid[:, Ny] .= true
    for i in 1:Nx
        q[i, 2, 5] = T(0.5)
        q[i, 2, 8] = T(0.5)
        q[i, 2, 9] = T(0.5)
        q[i, Ny - 1, 3] = T(0.5)
        q[i, Ny - 1, 6] = T(0.5)
        q[i, Ny - 1, 7] = T(0.5)
    end
    return q, solid
end

function first_nonfinite(ux, uy, psixx, psixy, psiyy)
    arrays = (:ux => Array(ux), :uy => Array(uy), :psixx => Array(psixx),
              :psixy => Array(psixy), :psiyy => Array(psiyy))
    for (name, a) in arrays, j in axes(a, 2), i in axes(a, 1)
        isfinite(Float64(a[i, j])) || return (field=name, i=i, j=j)
    end
    return nothing
end

function qwall_channel_reference(Fx_body, nu_total, Ny)
    return [Fx_body / (2 * nu_total) * (j - 1.5) * (Ny - 0.5 - j) for j in 1:Ny]
end

function run_l5b()
    backend, FT, backend_name = backend_choice()
    Nx = parse(Int, get(ENV, "M57_L5B_NX", "60"))
    Ny = parse(Int, get(ENV, "M57_L5B_NY", "32"))
    steps = parse(Int, get(ENV, "M57_L5B_STEPS", "5000"))
    log_every = parse(Int, get(ENV, "M57_L5B_LOG_EVERY", "250"))
    u_mean = FT(0.005)
    beta = FT(0.59)
    nu_total = u_mean * FT(Ny - 2)
    nu_s = beta * nu_total
    nu_p = (one(FT) - beta) * nu_total
    bsd = one(FT)
    lambda = FT(200)
    Fx_body = FT(8) * nu_total * u_mean / (FT(Ny - 2) ^ 2)
    nu_lbm = nu_s + bsd * nu_p
    prefactor = nu_p / lambda

    q_h, solid_h = channel_qwall(Nx, Ny, FT)
    embedded_h = Kraken.fvfd_embedded_boundary_from_qwall_2d(
        q_h; FT=FT, include_axis_aligned=true, include_halfway=true,
    )
    if count(>(zero(FT)), embedded_h.wall_q) == 0
        mkpath(OUTDIR)
        open(CSV, "w") do io
            println(io, "STOP,qwall_halfplane_did_not_populate_wall_q")
        end
        println("M57_L5b verdict=STOP reason=qwall_halfplane_did_not_populate_wall_q csv=$CSV")
        return 2
    end

    config = Kraken.LBMConfig(Kraken.D2Q9(); Nx, Ny, ν=Float64(nu_lbm), u_lid=0.0, max_steps=steps)
    state = Kraken.initialize_2d(config, FT; backend)
    f_in, f_out = state.f_in, state.f_out
    rho, ux, uy, is_solid = state.ρ, state.ux, state.uy, state.is_solid
    copyto!(is_solid, solid_h)
    ref0 = qwall_channel_reference(Float64(Fx_body), Float64(nu_total), Ny)
    ux0 = zeros(FT, Nx, Ny)
    f0 = zeros(FT, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx
        u = solid_h[i, j] ? zero(FT) : FT(ref0[j])
        ux0[i, j] = u
        for q in 1:9
            f0[i, j, q] = Kraken.equilibrium(Kraken.D2Q9(), one(FT), u, zero(FT), q)
        end
    end
    copyto!(ux, ux0)
    copyto!(f_in, f0)
    copyto!(f_out, f0)
    omega_t = FT(Kraken.omega(config))
    bc = Kraken.logfv_periodicx_wally_bcspec_2d()
    geom_h = Kraken.FVFDGeometry2D(solid_h, embedded_h, Kraken.FVFDPatch2D(one(FT), one(FT)), bc)
    geom = Kraken.fvfd_transfer_geometry_2d(geom_h, backend, FT)

    z2() = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    psixx, psixy, psiyy = z2(), z2(), z2()
    psixx_next, psixy_next, psiyy_next = z2(), z2(), z2()
    dudx, dudy, dvdx, dvdy = z2(), z2(), z2(), z2()
    tauxx, tauxy, tauyy = z2(), z2(), z2()
    fx_poly, fy_poly, fx_total, fy_total = z2(), z2(), z2(), z2()

    max_grad_est = abs(Fx_body) * FT(Ny) / (FT(2) * max(nu_total, eps(FT)))
    est = Kraken.logfv_oldroydb_subcycle_estimate(
        Float64(max_grad_est), Float64(lambda), 1.0;
        relative_tolerance=0.01, max_deformation_increment=0.05,
        max_memory_deformation_increment=0.07, min_substeps=1, max_substeps=64,
    )
    substeps = est.recommended
    dt_poly = one(FT) / FT(substeps)
    nan_step = 0

    for step in 1:steps
        Kraken.fvfd_velocity_gradient_embedded_2d!(
            dudx, dudy, dvdx, dvdy, ux, uy, geom; sync=false,
        )
        for _ in 1:substeps
            Kraken.logfv_step_oldroydb_log_2d!(
                psixx_next, psixy_next, psiyy_next, psixx, psixy, psiyy,
                dudx, dudy, dvdx, dvdy, lambda, dt_poly; sync=false,
            )
            psixx, psixx_next = psixx_next, psixx
            psixy, psixy_next = psixy_next, psixy
            psiyy, psiyy_next = psiyy_next, psiyy
        end
        Kraken.logfv_stress_from_log_2d!(tauxx, tauxy, tauyy, psixx, psixy, psiyy, prefactor; sync=false)
        Kraken.logfv_polymer_force_bc_aware_2d!(
            fx_poly, fy_poly, tauxx, tauxy, tauyy, is_solid, one(FT), one(FT), bc; sync=false,
        )
        Kraken.logfv_bsd_correct_force_bc_aware_2d!(
            fx_total, fy_total, fx_poly, fy_poly, ux, uy, is_solid, bsd, nu_p,
            one(FT), one(FT), bc; sync=false,
        )
        Kraken.logfv_add_constant_force_fluid_2d!(
            fx_total, fy_total, is_solid, Fx_body, zero(FT); sync=false,
        )
        Kraken.stream_periodic_x_wall_y_2d!(f_out, f_in, Nx, Ny)
        Kraken.collide_guo_field_2d!(f_out, is_solid, fx_total, fy_total, omega_t)
        Kraken.logfv_compute_macroscopic_forced_field_2d!(
            rho, ux, uy, f_out, fx_total, fy_total; sync=false,
        )
        f_in, f_out = f_out, f_in
        if step == 1 || step % log_every == 0 || step == steps
            KernelAbstractions.synchronize(backend)
            bad = first_nonfinite(ux, uy, psixx, psixy, psiyy)
            if bad !== nothing
                nan_step = step
                @printf("M57_L5b first_nonfinite step=%d field=%s cell=(%d,%d)\n",
                        step, bad.field, bad.i, bad.j)
                break
            end
        end
    end
    KernelAbstractions.synchronize(backend)

    ux_cpu = Array(ux)
    uy_cpu = Array(uy)
    mean_ux = [sum(@view ux_cpu[:, j]) / Nx for j in 1:Ny]
    ref = ref0
    interior = 3:(Ny - 2)
    max_abs_error = maximum(abs.(mean_ux[interior] .- ref[interior]))
    max_ref = maximum(abs.(ref[interior]))
    rel = max_abs_error / max(max_ref, eps(Float64))
    max_uy = maximum(abs, uy_cpu[:, interior])

    mkpath(OUTDIR)
    open(CSV, "w") do io
        println(io, "j,ux_mean,u_analytic,abs_error")
        for j in 1:Ny
            @printf(io, "%d,%.17g,%.17g,%.17g\n", j, mean_ux[j], ref[j], abs(mean_ux[j] - ref[j]))
        end
    end
    verdict = nan_step > 0 ? "RED" : rel < 0.05 ? "GREEN" : "RED"
    measured = nan_step > 0 ? @sprintf("nan_step=%d", nan_step) :
               @sprintf("rel_err=%.17g max_uy=%.17g", rel, max_uy)
    @printf("M57_L5b backend=%s substeps=%d verdict=%s %s qwall_entries=%d csv=%s\n",
            backend_name, substeps, verdict, measured, count(>(zero(FT)), embedded_h.wall_q), CSV)
    return verdict == "GREEN" ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(run_l5b())
end
