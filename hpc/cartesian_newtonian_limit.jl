# Cartesian obstacle Newtonian-limit force canary.
#
# This isolates the macro force path from curved cut-links:
# - square obstacle, axis-aligned halfway links only
# - same LI-BB hydrodynamic solver for Newtonian and viscoelastic runs
# - force q_wall contains obstacle links only, not top/bottom channel walls
# - compares post-source MEA against explicit solvent+polymer split

include(joinpath(@__DIR__, "..", "src", "Kraken.jl"))

using .Kraken
using Dates
using KernelAbstractions
using Printf

const CUDA_MOD = try
    @eval using CUDA
    getfield(Main, :CUDA)
catch
    nothing
end

const METAL_MOD = if Sys.isapple()
    try
        @eval using Metal
        getfield(Main, :Metal)
    catch
        nothing
    end
else
    nothing
end

function select_backend()
    requested = lowercase(get(ENV, "KRAKEN_BACKEND", "auto"))
    if requested in ("auto", "cuda") && CUDA_MOD !== nothing
        try
            if Base.invokelatest(getfield(CUDA_MOD, :functional))
                backend = Base.invokelatest(getfield(CUDA_MOD, :CUDABackend))
                device = Base.invokelatest(getfield(CUDA_MOD, :device))
                name = Base.invokelatest(getfield(CUDA_MOD, :name), device)
                return backend, Float64, "CUDA $name"
            end
        catch err
            requested == "cuda" && rethrow(err)
        end
    end
    if requested in ("auto", "metal") && METAL_MOD !== nothing
        try
            if Base.invokelatest(getfield(METAL_MOD, :functional))
                backend = Base.invokelatest(getfield(METAL_MOD, :MetalBackend))
                return backend, Float32, "Metal"
            end
        catch err
            requested == "metal" && rethrow(err)
        end
    end
    return KernelAbstractions.CPU(), Float64, "CPU"
end

function parse_items(value::AbstractString)
    return [Symbol(strip(x)) for x in split(value, r"[,;]") if !isempty(strip(x))]
end

function parse_bool(value::AbstractString)
    v = lowercase(strip(value))
    v in ("1", "true", "yes", "on") && return true
    v in ("0", "false", "no", "off") && return false
    error("invalid boolean value $value")
end

function polymer_bc_from_symbol(name::Symbol)
    name === :cnebb && return CNEBB()
    name in (:extrap_eq, :extrapeq) && return ExtrapEqWallBC()
    name in (:none, :no_polymer_wall) && return NoPolymerWallBC()
    error("unknown polymer BC $name; expected cnebb, extrap_eq, or none")
end

function obstacle_only_q_wall(geom::StepChannelGeometry2D, ::Type{FT};
                              link_mode::Symbol=:cardinal) where {FT}
    link_mode in (:cardinal, :all) ||
        error("unknown force link mode $link_mode; expected :cardinal or :all")
    cxv = (0, 1, 0, -1, 0, 1, -1, -1, 1)
    cyv = (0, 0, 1, 0, -1, 1, 1, -1, -1)
    q_force = zeros(FT, geom.Nx, geom.Ny, 9)
    @inbounds for j in 1:geom.Ny, i in 1:geom.Nx
        geom.is_solid[i, j] && continue
        for q in 2:9
            link_mode === :cardinal && q > 5 && continue
            ni = i + cxv[q]
            nj = j + cyv[q]
            if 1 <= ni <= geom.Nx && 1 <= nj <= geom.Ny && geom.is_solid[ni, nj]
                q_force[i, j, q] = geom.q_wall[i, j, q] > 0 ? FT(geom.q_wall[i, j, q]) : FT(0.5)
            end
        end
    end
    return q_force
end

function drag_from_mode(f_out, q_force, uw_x, uw_y, Nx, Ny, mode::Symbol)
    mode === :postpair && return compute_drag_libb_postpair_2d(f_out, q_force, Nx, Ny)
    mode === :simple_halfway && return compute_drag_libb_2d(f_out, q_force, Nx, Ny)
    mode === :liu_eq63 && return compute_drag_libb_liu_eq63_2d(f_out, q_force, uw_x, uw_y, Nx, Ny)
    mode === :mei_reconstruct && return compute_drag_libb_mei_2d(f_out, q_force, uw_x, uw_y, Nx, Ny)
    error("unknown momentum_exchange_mode $mode")
end

function sample_drag(step, max_steps, avg_window, drag_stride)
    step > max_steps - avg_window || return false
    return ((step - (max_steps - avg_window) - 1) % drag_stride == 0) || step == max_steps
end

function initialize_populations(geom, u_profile_h, ::Type{FT}) where {FT}
    f_in_h = zeros(FT, geom.Nx, geom.Ny, 9)
    @inbounds for j in 1:geom.Ny, i in 1:geom.Nx, q in 1:9
        u0 = geom.is_solid[i, j] ? zero(FT) : u_profile_h[j]
        f_in_h[i, j, q] = equilibrium(D2Q9(), one(FT), u0, zero(FT), q)
    end
    return f_in_h
end

function run_square_newtonian(; geom, backend, FT, u_ref_mean, nu,
                              max_steps, avg_window, drag_stride,
                              momentum_exchange_mode, solvent_magic,
                              force_link_mode)
    Nx, Ny = geom.Nx, geom.Ny
    geom_d = transfer_step_geometry_2d(geom, backend)
    u_profile_h = parabolic_face_profile_2d(geom; face=:west,
                                            mean_velocity=u_ref_mean, FT=FT)
    f_in_h = initialize_populations(geom, u_profile_h, FT)
    q_force_h = obstacle_only_q_wall(geom, FT; link_mode=force_link_mode)

    q_wall = geom_d.q_wall
    is_solid = geom_d.is_solid
    q_force = KernelAbstractions.allocate(backend, FT, Nx, Ny, 9)
    uw_x = KernelAbstractions.allocate(backend, FT, Nx, Ny, 9)
    uw_y = KernelAbstractions.allocate(backend, FT, Nx, Ny, 9)
    f_in = KernelAbstractions.allocate(backend, FT, Nx, Ny, 9)
    f_out = KernelAbstractions.allocate(backend, FT, Nx, Ny, 9)
    rho = KernelAbstractions.allocate(backend, FT, Nx, Ny)
    ux = KernelAbstractions.allocate(backend, FT, Nx, Ny)
    uy = KernelAbstractions.allocate(backend, FT, Nx, Ny)
    u_profile = KernelAbstractions.allocate(backend, FT, Ny)

    copyto!(q_force, q_force_h)
    fill!(uw_x, zero(FT)); fill!(uw_y, zero(FT))
    copyto!(f_in, f_in_h); fill!(f_out, zero(FT))
    fill!(rho, one(FT)); fill!(ux, zero(FT)); fill!(uy, zero(FT))
    copyto!(u_profile, u_profile_h)
    bcspec = default_step_bcspec_2d(geom_d, u_profile, one(FT))

    fx_sum = 0.0
    fy_sum = 0.0
    n_avg = 0
    for step in 1:max_steps
        fused_trt_libb_v2_step!(f_out, f_in, rho, ux, uy, is_solid,
                                 q_wall, uw_x, uw_y, Nx, Ny, FT(nu);
                                 Λ=solvent_magic)
        apply_bc_rebuild_2d!(f_out, f_in, bcspec, FT(nu), Nx, Ny;
                             Λ=solvent_magic)
        if sample_drag(step, max_steps, avg_window, drag_stride)
            drag = drag_from_mode(f_out, q_force, uw_x, uw_y, Nx, Ny,
                                  momentum_exchange_mode)
            fx_sum += drag.Fx
            fy_sum += drag.Fy
            n_avg += 1
        end
        f_in, f_out = f_out, f_in
    end
    KernelAbstractions.synchronize(backend)

    fx = fx_sum / n_avg
    fy = fy_sum / n_avg
    d = Float64(geom.H_ref)
    uref = Float64(u_ref_mean)
    cd = 2.0 * fx / (uref^2 * d)
    cl = 2.0 * fy / (uref^2 * d)
    return (; Cd=cd, Cl=cl, Fx=fx, Fy=fy, Cd_s=cd, Cd_p=0.0,
            Cd_split=cd, Cd_post=cd, n_drag_samples=n_avg,
            Re=uref * d / Float64(nu), u_ref=uref, D=d)
end

function run_square_visco(; geom, backend, FT, u_ref_mean, nu_s, nu_p,
                          lambda_val, polymer_bc, max_steps, avg_window,
                          drag_stride, momentum_exchange_mode, solvent_magic,
                          conformation_magic, hermite_source_mode,
                          force_link_mode, conformation_divergence_mode,
                          source_scale_dynamics, solvent_source_on_cutlinks)
    Nx, Ny = geom.Nx, geom.Ny
    geom_d = transfer_step_geometry_2d(geom, backend)
    u_profile_h = parabolic_face_profile_2d(geom; face=:west,
                                            mean_velocity=u_ref_mean, FT=FT)
    cxx_in_h, cxy_in_h, cyy_in_h =
        oldroydb_inlet_conformation_profile_2d(geom; face=:west,
            mean_velocity=u_ref_mean, λ=FT(lambda_val),
            log_formulation=false, FT=FT)
    f_in_h = initialize_populations(geom, u_profile_h, FT)
    q_force_h = obstacle_only_q_wall(geom, FT; link_mode=force_link_mode)

    q_wall = geom_d.q_wall
    is_solid = geom_d.is_solid
    q_force = KernelAbstractions.allocate(backend, FT, Nx, Ny, 9)
    uw_x = KernelAbstractions.allocate(backend, FT, Nx, Ny, 9)
    uw_y = KernelAbstractions.allocate(backend, FT, Nx, Ny, 9)
    f_in = KernelAbstractions.allocate(backend, FT, Nx, Ny, 9)
    f_out = KernelAbstractions.allocate(backend, FT, Nx, Ny, 9)
    rho = KernelAbstractions.allocate(backend, FT, Nx, Ny)
    ux = KernelAbstractions.allocate(backend, FT, Nx, Ny)
    uy = KernelAbstractions.allocate(backend, FT, Nx, Ny)
    u_profile = KernelAbstractions.allocate(backend, FT, Ny)
    cxx_in = KernelAbstractions.allocate(backend, FT, Ny)
    cxy_in = KernelAbstractions.allocate(backend, FT, Ny)
    cyy_in = KernelAbstractions.allocate(backend, FT, Ny)

    copyto!(q_force, q_force_h)
    fill!(uw_x, zero(FT)); fill!(uw_y, zero(FT))
    copyto!(f_in, f_in_h); fill!(f_out, zero(FT))
    fill!(rho, one(FT)); fill!(ux, zero(FT)); fill!(uy, zero(FT))
    copyto!(u_profile, u_profile_h)
    copyto!(cxx_in, cxx_in_h); copyto!(cxy_in, cxy_in_h); copyto!(cyy_in, cyy_in_h)
    bcspec = default_step_bcspec_2d(geom_d, u_profile, one(FT))

    cxx = KernelAbstractions.zeros(backend, FT, Nx, Ny); fill!(cxx, one(FT))
    cxy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    cyy = KernelAbstractions.zeros(backend, FT, Nx, Ny); fill!(cyy, one(FT))
    gxx = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    gxy = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    gyy = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    init_conformation_field_2d!(gxx, cxx, ux, uy)
    init_conformation_field_2d!(gxy, cxy, ux, uy)
    init_conformation_field_2d!(gyy, cyy, ux, uy)
    gxx_buf = similar(gxx)
    gxy_buf = similar(gxy)
    gyy_buf = similar(gyy)
    tau_xx = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    tau_xy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    tau_yy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    model = OldroydB(G=FT(nu_p / lambda_val), λ=FT(lambda_val))
    s_plus_s = one(FT) / (FT(3) * FT(nu_s) + FT(0.5))

    fx_s_sum = 0.0
    fy_s_sum = 0.0
    fx_p_sum = 0.0
    fy_p_sum = 0.0
    fx_post_sum = 0.0
    fy_post_sum = 0.0
    n_avg = 0

    for step in 1:max_steps
        fused_trt_libb_v2_step!(f_out, f_in, rho, ux, uy, is_solid,
                                 q_wall, uw_x, uw_y, Nx, Ny, FT(nu_s);
                                 Λ=solvent_magic)
        apply_bc_rebuild_2d!(f_out, f_in, bcspec, FT(nu_s), Nx, Ny;
                             Λ=solvent_magic)

        do_sample = sample_drag(step, max_steps, avg_window, drag_stride)
        if do_sample
            drag_s = drag_from_mode(f_out, q_force, uw_x, uw_y, Nx, Ny,
                                    momentum_exchange_mode)
            drag_p = Kraken.compute_polymeric_drag_2d(tau_xx, tau_xy, tau_yy,
                                                      is_solid, Nx, Ny;
                                                      extrapolate=true)
            fx_s_sum += drag_s.Fx
            fy_s_sum += drag_s.Fy
            fx_p_sum += drag_p.Fx
            fy_p_sum += drag_p.Fy
        end

        if solvent_source_on_cutlinks
            apply_hermite_source_2d!(f_out, is_solid, s_plus_s,
                                      tau_xx, tau_xy, tau_yy;
                                      ce_correction = hermite_source_mode === :ce_corrected,
                                      source_scale = source_scale_dynamics)
        else
            apply_hermite_source_full_fluid_2d!(
                f_out, is_solid, q_wall, s_plus_s,
                tau_xx, tau_xy, tau_yy;
                ce_correction = hermite_source_mode === :ce_corrected,
                source_scale = source_scale_dynamics,
                apply_y_domain_walls = false,
            )
        end

        if do_sample
            drag_post = drag_from_mode(f_out, q_force, uw_x, uw_y, Nx, Ny,
                                       momentum_exchange_mode)
            fx_post_sum += drag_post.Fx
            fy_post_sum += drag_post.Fy
            n_avg += 1
        end

        stream_2d!(gxx_buf, gxx, Nx, Ny)
        stream_2d!(gxy_buf, gxy, Nx, Ny)
        stream_2d!(gyy_buf, gyy, Nx, Ny)
        apply_polymer_wall_bc!(gxx_buf, gxx, is_solid, q_wall, cxx, ux, uy, polymer_bc)
        apply_polymer_wall_bc!(gxy_buf, gxy, is_solid, q_wall, cxy, ux, uy, polymer_bc)
        apply_polymer_wall_bc!(gyy_buf, gyy, is_solid, q_wall, cyy, ux, uy, polymer_bc)
        reset_conformation_inlet_masked_2d!(gxx_buf, cxx_in, u_profile,
                                            geom_d.west_conformation_mask, Ny)
        reset_conformation_inlet_masked_2d!(gxy_buf, cxy_in, u_profile,
                                            geom_d.west_conformation_mask, Ny)
        reset_conformation_inlet_masked_2d!(gyy_buf, cyy_in, u_profile,
                                            geom_d.west_conformation_mask, Ny)
        reset_conformation_outlet_masked_2d!(gxx_buf, Nx, Ny,
                                             geom_d.east_conformation_mask)
        reset_conformation_outlet_masked_2d!(gxy_buf, Nx, Ny,
                                             geom_d.east_conformation_mask)
        reset_conformation_outlet_masked_2d!(gyy_buf, Nx, Ny,
                                             geom_d.east_conformation_mask)
        gxx, gxx_buf = gxx_buf, gxx
        gxy, gxy_buf = gxy_buf, gxy
        gyy, gyy_buf = gyy_buf, gyy

        compute_conformation_macro_2d!(cxx, gxx)
        compute_conformation_macro_2d!(cxy, gxy)
        compute_conformation_macro_2d!(cyy, gyy)
        collide_conformation_2d!(gxx, cxx, ux, uy, cxx, cxy, cyy,
                                  is_solid, one(FT), FT(lambda_val);
                                  magic=conformation_magic, component=1,
                                  divergence_mode=conformation_divergence_mode)
        collide_conformation_2d!(gxy, cxy, ux, uy, cxx, cxy, cyy,
                                  is_solid, one(FT), FT(lambda_val);
                                  magic=conformation_magic, component=2,
                                  divergence_mode=conformation_divergence_mode)
        collide_conformation_2d!(gyy, cyy, ux, uy, cxx, cxy, cyy,
                                  is_solid, one(FT), FT(lambda_val);
                                  magic=conformation_magic, component=3,
                                  divergence_mode=conformation_divergence_mode)
        update_polymer_stress!(tau_xx, tau_xy, tau_yy, cxx, cxy, cyy, model)

        f_in, f_out = f_out, f_in
    end
    KernelAbstractions.synchronize(backend)

    fx_s = fx_s_sum / n_avg
    fy_s = fy_s_sum / n_avg
    fx_p = fx_p_sum / n_avg
    fy_p = fy_p_sum / n_avg
    fx_post = fx_post_sum / n_avg
    fy_post = fy_post_sum / n_avg
    d = Float64(geom.H_ref)
    uref = Float64(u_ref_mean)
    scale = 2.0 / (uref^2 * d)
    cd_s = scale * fx_s
    cd_p = scale * fx_p
    cd_post = scale * fx_post
    cl_post = scale * fy_post
    return (; Cd=cd_post, Cl=cl_post, Fx=fx_post, Fy=fy_post,
            Cd_s=cd_s, Cd_p=cd_p, Cd_split=cd_s + cd_p, Cd_post=cd_post,
            Fx_s=fx_s, Fy_s=fy_s, Fx_p=fx_p, Fy_p=fy_p,
            n_drag_samples=n_avg,
            Re=uref * d / Float64(nu_s + nu_p),
            Wi=Float64(lambda_val) * uref / (d / 2),
            beta=Float64(nu_s / (nu_s + nu_p)),
            u_ref=uref, D=d)
end

backend, FT, backend_label = select_backend()

side = parse(Int, get(ENV, "KRAKEN_SIDE", "40"))
height = parse(Int, get(ENV, "KRAKEN_H", string(2 * side)))
l_up = parse(Int, get(ENV, "KRAKEN_L_UP", "7"))
l_down = parse(Int, get(ENV, "KRAKEN_L_DOWN", "8"))
u_ref_mean = parse(Float64, get(ENV, "KRAKEN_U_REF_MEAN", "0.02"))
re_target = parse(Float64, get(ENV, "KRAKEN_RE", "1.0"))
beta = parse(Float64, get(ENV, "KRAKEN_BETA", "0.59"))
wi = parse(Float64, get(ENV, "KRAKEN_WI", "0.001"))
max_steps = parse(Int, get(ENV, "KRAKEN_STEPS", "30000"))
avg_divisor = parse(Int, get(ENV, "KRAKEN_AVG_DIVISOR", "5"))
avg_window = min(max_steps, max(1, max_steps ÷ avg_divisor))
drag_stride = parse(Int, get(ENV, "KRAKEN_DRAG_STRIDE", "50"))
momentum_exchange_mode = Symbol(get(ENV, "KRAKEN_MOMENTUM_EXCHANGE_MODE", "mei_reconstruct"))
force_link_mode = Symbol(get(ENV, "KRAKEN_FORCE_LINK_MODE", "cardinal"))
solvent_magic = parse(Float64, get(ENV, "KRAKEN_SOLVENT_MAGIC", string(3 / 16)))
conformation_magic = parse(Float64, get(ENV, "KRAKEN_CONFORMATION_MAGIC", "1e-6"))
conformation_divergence_mode =
    Symbol(get(ENV, "KRAKEN_CONFORMATION_DIVERGENCE_MODE", "trace_free"))
hermite_source_mode = Symbol(get(ENV, "KRAKEN_HERMITE_SOURCE_MODE", "liu_direct"))
source_scale_dynamics = parse(Float64, get(ENV, "KRAKEN_SOURCE_SCALE_DYNAMICS", "1.0"))
solvent_source_on_cutlinks =
    parse_bool(get(ENV, "KRAKEN_SOLVENT_SOURCE_ON_CUTLINKS", "true"))
bc_names = parse_items(get(ENV, "KRAKEN_POLYMER_BCS", "cnebb,extrap_eq"))

geom = square_obstacle_channel_geometry_2d(; H=height, side=side,
                                           L_up=l_up, L_down=l_down, FT=FT)
d = Float64(geom.H_ref)
nu_total = u_ref_mean * d / re_target
lambda_val = wi * (d / 2) / u_ref_mean
nu_s = beta * nu_total
nu_p = (1 - beta) * nu_total

println("=" ^ 96)
println("Cartesian square obstacle Newtonian-limit force canary")
println("time                 = ", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))
println("backend              = ", backend_label)
println("geometry             = square, Nx=$(geom.Nx), Ny=$(geom.Ny), side=$side, blockage=$(side / height)")
println("flow                 = Re=$re_target, u_ref_mean=$u_ref_mean, nu_total=$nu_total")
println("visco                = beta=$beta, Wi=$wi, lambda=$lambda_val, nu_s=$nu_s, nu_p=$nu_p")
println("steps                = $max_steps, avg_window=$avg_window, drag_stride=$drag_stride")
println("force_link_mode      = ", force_link_mode)
println("divergence_mode      = ", conformation_divergence_mode)
println("source_scale         = ", source_scale_dynamics)
println("source_on_cutlinks   = ", solvent_source_on_cutlinks)
println("force links          = ",
        count(>(0), obstacle_only_q_wall(geom, FT; link_mode=force_link_mode)))
println("=" ^ 96)

newt = run_square_newtonian(; geom, backend, FT, u_ref_mean,
    nu=nu_total, max_steps, avg_window, drag_stride,
    momentum_exchange_mode, solvent_magic, force_link_mode)

@printf("%-18s %-12s %-9s %-12s %-12s %-12s %-12s %-12s %-12s\n",
        "case", "poly_bc", "Wi", "Cd_post", "ratio", "Cd_s", "Cd_p",
        "Cd_split", "Cl")
@printf("%-18s %-12s %-9s %-12.6f %-12.6f %-12.6f %-12.6f %-12.6f %-12.6f\n",
        "newtonian", "-", "-", newt.Cd, 1.0, newt.Cd_s, newt.Cd_p,
        newt.Cd_split, newt.Cl)

for bc_name in bc_names
    polymer_bc = polymer_bc_from_symbol(bc_name)
    zero_poly = run_square_visco(; geom, backend, FT, u_ref_mean,
        nu_s=nu_total, nu_p=0.0, lambda_val=max(lambda_val, eps(Float64)),
        polymer_bc, max_steps, avg_window, drag_stride,
        momentum_exchange_mode, solvent_magic, conformation_magic,
        hermite_source_mode, force_link_mode, conformation_divergence_mode,
        source_scale_dynamics, solvent_source_on_cutlinks)
    @printf("%-18s %-12s %-9s %-12.6f %-12.6f %-12.6f %-12.6f %-12.6f %-12.6f\n",
            "visco_nup0", string(bc_name), "0",
            zero_poly.Cd_post, zero_poly.Cd_post / newt.Cd,
            zero_poly.Cd_s, zero_poly.Cd_p, zero_poly.Cd_split, zero_poly.Cl)

    low_wi = run_square_visco(; geom, backend, FT, u_ref_mean,
        nu_s, nu_p, lambda_val, polymer_bc, max_steps, avg_window,
        drag_stride, momentum_exchange_mode, solvent_magic,
        conformation_magic, hermite_source_mode, force_link_mode,
        conformation_divergence_mode, source_scale_dynamics,
        solvent_source_on_cutlinks)
    @printf("%-18s %-12s %-9.3g %-12.6f %-12.6f %-12.6f %-12.6f %-12.6f %-12.6f\n",
            "visco_lowwi", string(bc_name), wi,
            low_wi.Cd_post, low_wi.Cd_post / newt.Cd,
            low_wi.Cd_s, low_wi.Cd_p, low_wi.Cd_split, low_wi.Cl)
end

println("=" ^ 96)
println("Interpretation:")
println("  visco_nup0 ratio must be ~1.0. If not, the visco driver plumbing differs from Newtonian.")
println("  visco_lowwi Cd_split tests explicit solvent+polymer stress accounting.")
println("  visco_lowwi Cd_post tests the post-source MEA path used for cylinder validation.")
println("=" ^ 96)
