#!/usr/bin/env julia

using Printf
using Serialization
using Kraken
using KernelAbstractions

try
    @eval using Metal
catch
end

const OUTDIR = joinpath("scratch", "M56_vv_ladder")
const CSV = joinpath(OUTDIR, "L3_frozen_u_cylinder.csv")
const FIELD_JLS = joinpath(OUTDIR, "L3_frozen_u_cylinder_fields.jls")

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
            @printf("M56_L3 backend_warning=metal_unavailable error=%s\n", sprint(showerror, err))
        end
    end
    return KernelAbstractions.CPU(), Float64, "cpu"
end

function first_scan(psixx, psixy, psiyy, fx, fy, is_solid_h)
    axx = Array(psixx)
    axy = Array(psixy)
    ayy = Array(psiyy)
    afx = Array(fx)
    afy = Array(fy)
    max_trace = -Inf
    max_force = -Inf
    force_loc = (i=0, j=0)
    first_bad = nothing
    for j in axes(is_solid_h, 2), i in axes(is_solid_h, 1)
        is_solid_h[i, j] && continue
        vals = (axx[i, j], axy[i, j], ayy[i, j], afx[i, j], afy[i, j])
        if first_bad === nothing && any(v -> !isfinite(Float64(v)), vals)
            first_bad = (i=i, j=j)
            continue
        end
        cxx, _cxy, cyy = Kraken.logfv_exp_sym2_2d(
            Float64(axx[i, j]), Float64(axy[i, j]), Float64(ayy[i, j]),
        )
        trc = cxx + cyy
        if first_bad === nothing && !isfinite(trc)
            first_bad = (i=i, j=j)
            continue
        end
        max_trace = max(max_trace, trc)
        fmag = hypot(Float64(afx[i, j]), Float64(afy[i, j]))
        if fmag > max_force
            max_force = fmag
            force_loc = (i=i, j=j)
        end
    end
    return (; max_trace, max_force, force_loc, first_bad)
end

function warmup_newtonian(backend, FT, warmup_steps)
    R = 10
    Umean = 0.005
    nu_total = Umean * R
    t0 = time()
    result = Kraken.run_viscoelastic_logfv_cylinder_coupled_2d(;
        radius=R, H=4R, L_up=15.0, L_down=15.0,
        nu_s=FT(nu_total), nu_p=FT(0), lambda=FT(1),
        polymer_model=:oldroydb, L_max=FT(10), u_mean=FT(Umean), Fx_body=FT(0),
        bsd_fraction=FT(0), polymer_substeps=1, max_polymer_substeps=1,
        max_steps=warmup_steps, avg_window=max(1, min(warmup_steps, 100)),
        drag_stride=max(1, warmup_steps), diagnostic_stride=0,
        embedded_geometry=:qwall, embedded_gradient=true,
        embedded_advection=false, embedded_force=false, embedded_drag=false,
        advection_scheme=:muscl_superbee, wall_bc=:halfwayBB,
        force_boundary_fill=:bc_aware, backend, T=FT,
    )
    return result, time() - t0
end

function run_replay(backend, FT, warm)
    R = 10
    lambda = FT(2000)
    prefactor = FT((1.0 - 0.59) * 0.05 / 2000.0)
    steps = parse(Int, get(ENV, "M56_L3_STEPS", "5000"))
    log_every = parse(Int, get(ENV, "M56_L3_LOG_EVERY", "100"))
    Nx, Ny = size(warm.ux)
    cx = 15.0 * R
    cy = (Ny - 1) / 2
    q_wall, is_solid_h = Kraken.precompute_q_wall_cylinder(Nx, Ny, cx, cy, R; FT=FT)
    embedded_h = Kraken.fvfd_embedded_boundary_from_qwall_2d(q_wall; FT=FT)
    bc = Kraken.logfv_openx_wally_bcspec_2d()
    geom_h = Kraken.FVFDGeometry2D(is_solid_h, embedded_h, Kraken.FVFDPatch2D(one(FT), one(FT)), bc)
    geom = Kraken.fvfd_transfer_geometry_2d(geom_h, backend, FT)

    ux = KernelAbstractions.allocate(backend, FT, Nx, Ny)
    uy = KernelAbstractions.allocate(backend, FT, Nx, Ny)
    copyto!(ux, FT.(warm.ux))
    copyto!(uy, FT.(warm.uy))
    ux_west = KernelAbstractions.allocate(backend, FT, Ny)
    ux_east = KernelAbstractions.allocate(backend, FT, Ny)
    uy_south = KernelAbstractions.allocate(backend, FT, Nx)
    uy_north = KernelAbstractions.allocate(backend, FT, Nx)
    copyto!(ux_west, FT.(warm.ux[1, :]))
    copyto!(ux_east, FT.(warm.ux[end, :]))
    copyto!(uy_south, FT.(warm.uy[:, 1]))
    copyto!(uy_north, FT.(warm.uy[:, end]))

    psixx = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    psixy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    psiyy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    psixx_adv = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    psixy_adv = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    psiyy_adv = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    psixx_next = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    psixy_next = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    psiyy_next = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    zero_w = KernelAbstractions.zeros(backend, FT, Ny)
    east_xx = KernelAbstractions.zeros(backend, FT, Ny)
    east_xy = KernelAbstractions.zeros(backend, FT, Ny)
    east_yy = KernelAbstractions.zeros(backend, FT, Ny)
    zero_s = KernelAbstractions.zeros(backend, FT, Nx)
    ux_face = KernelAbstractions.zeros(backend, FT, Nx + 1, Ny)
    uy_face = KernelAbstractions.zeros(backend, FT, Nx, Ny + 1)
    dudx = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    dudy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    dvdx = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    dvdy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    tauxx = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    tauxy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    tauyy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    fx = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    fy = KernelAbstractions.zeros(backend, FT, Nx, Ny)

    Kraken.fvfd_velocity_gradient_embedded_2d!(dudx, dudy, dvdx, dvdy, ux, uy, geom; sync=true)
    grad_arrays = (Array(dudx), Array(dudy), Array(dvdx), Array(dvdy))
    max_grad = maximum(maximum(abs, a) for a in grad_arrays)
    est = Kraken.logfv_oldroydb_subcycle_estimate(
        Float64(max_grad), Float64(lambda), 1.0;
        relative_tolerance=0.01, max_deformation_increment=0.05,
        max_memory_deformation_increment=0.07, min_substeps=1, max_substeps=64,
    )
    substeps = est.recommended
    dt_poly = one(FT) / FT(substeps)

    mkpath(OUTDIR)
    open(CSV, "w") do io
        println(io, "step,trace_C_max,F_poly_max,F_poly_i,F_poly_j,first_nan_i,first_nan_j")
        first_bad = nothing
        final_scan = nothing
        for step in 1:steps
            Kraken.logfv_copy_column_profile_2d!(east_xx, psixx, Nx; sync=false)
            Kraken.logfv_copy_column_profile_2d!(east_xy, psixy, Nx; sync=false)
            Kraken.logfv_copy_column_profile_2d!(east_yy, psiyy, Nx; sync=false)
            Kraken.logfv_cell_velocity_to_faces_bc_aware_2d!(
                ux_face, uy_face, ux, uy, geom.is_solid,
                ux_west, ux_east, uy_south, uy_north, bc; sync=false,
            )
            Kraken.logfv_advect_upwind_bc_aware_2d!(
                psixx_adv, psixy_adv, psiyy_adv, psixx, psixy, psiyy,
                zero_w, zero_w, zero_w, east_xx, east_xy, east_yy,
                zero_s, zero_s, zero_s, zero_s, zero_s, zero_s,
                ux_face, uy_face, geom.is_solid, one(FT), one(FT), bc, one(FT);
                sync=false, advection_scheme=:muscl_superbee,
            )
            px, py, pz = psixx_adv, psixy_adv, psiyy_adv
            for _ in 1:substeps
                Kraken.logfv_step_constitutive_log_2d!(
                    psixx_next, psixy_next, psiyy_next, px, py, pz,
                    dudx, dudy, dvdx, dvdy, lambda, dt_poly,
                    Kraken.LOGFV_MODEL_OLDROYDB, zero(FT); sync=false,
                )
                px, psixx_next = psixx_next, px
                py, psixy_next = psixy_next, py
                pz, psiyy_next = psiyy_next, pz
            end
            psixx, psixx_adv = px, psixx
            psixy, psixy_adv = py, psixy
            psiyy, psiyy_adv = pz, psiyy
            Kraken.logfv_stress_from_log_2d!(tauxx, tauxy, tauyy, psixx, psixy, psiyy,
                                             prefactor; sync=false)
            Kraken.fvfd_tensor_divergence_2d!(fx, fy, tauxx, tauxy, tauyy,
                                              geom.is_solid, one(FT), one(FT), bc; sync=false)
            if step == 1 || step % log_every == 0 || step == steps
                KernelAbstractions.synchronize(backend)
                s = first_scan(psixx, psixy, psiyy, fx, fy, is_solid_h)
                final_scan = s
                first_bad = first_bad === nothing ? s.first_bad : first_bad
                bad_i = first_bad === nothing ? 0 : first_bad.i
                bad_j = first_bad === nothing ? 0 : first_bad.j
                @printf(io, "%d,%.17g,%.17g,%d,%d,%d,%d\n",
                        step, s.max_trace, s.max_force, s.force_loc.i, s.force_loc.j, bad_i, bad_j)
                flush(io)
                if first_bad !== nothing || s.max_trace >= 1.0e4
                    break
                end
            end
        end
        serialize(FIELD_JLS, (; ux=warm.ux, uy=warm.uy, psixx=Array(psixx),
                              psixy=Array(psixy), psiyy=Array(psiyy), q_wall=Array(q_wall),
                              is_solid=is_solid_h))
        return final_scan, first_bad, substeps, max_grad
    end
end

function run_l3()
    backend, FT, backend_name = backend_choice()
    warmup_steps = parse(Int, get(ENV, "M56_L3_WARMUP_STEPS", "10000"))
    t0 = time()
    warm, warm_time = warmup_newtonian(backend, FT, warmup_steps)
    scan, first_bad, substeps, max_grad = run_replay(backend, FT, warm)
    walltime = time() - t0
    pass = first_bad === nothing && scan !== nothing && scan.max_trace < 1.0e4
    bad_i = first_bad === nothing ? 0 : first_bad.i
    bad_j = first_bad === nothing ? 0 : first_bad.j
    @printf("M56_L3 backend=%s warmup_steps=%d warmup_s=%.3f replay_substeps=%d max_grad=%.17g walltime_s=%.3f\n",
            backend_name, warmup_steps, warm_time, substeps, max_grad, walltime)
    @printf("M56_L3 verdict=%s trace_C_max=%.17g F_poly_max=%.17g F_poly_loc=(%d,%d) first_nan=(%d,%d) csv=%s fields=%s\n",
            pass ? "PASS" : "FAIL", scan.max_trace, scan.max_force,
            scan.force_loc.i, scan.force_loc.j, bad_i, bad_j, CSV, FIELD_JLS)
    return pass ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(run_l3())
end
