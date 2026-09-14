# Frozen-velocity curved-wall patch for the viscoelastic conformation CDE.
#
# Purpose:
#   isolate whether the conformation solver/BC generates the correct
#   Newtonian-limit polymer stress near a curved wall, before any Cd/MEA
#   post-processing is involved.
#
# Geometry:
#   solid inner cylinder, fluid outside.  The imposed velocity is circular
#   Couette, u_r=0, u_θ=A r + B/r, so S and τ_target=2νpS are analytic.

include(joinpath(@__DIR__, "..", "src", "Kraken.jl"))

using .Kraken
using Dates
using KernelAbstractions
using Printf

const _CUDA_MOD = try
    @eval using CUDA
    getfield(Main, :CUDA)
catch
    nothing
end

const _METAL_MOD = if Sys.isapple()
    try
        @eval using Metal
        getfield(Main, :Metal)
    catch
        nothing
    end
else
    nothing
end

function _select_backend()
    requested = lowercase(get(ENV, "KRAKEN_BACKEND", "cpu"))
    if requested in ("auto", "cuda") && _CUDA_MOD !== nothing
        try
            if Base.invokelatest(getfield(_CUDA_MOD, :functional))
                return Base.invokelatest(getfield(_CUDA_MOD, :CUDABackend)),
                       Float64,
                       "CUDA"
            end
        catch err
            requested == "cuda" && rethrow(err)
        end
    end
    if requested in ("auto", "metal") && _METAL_MOD !== nothing
        try
            if Base.invokelatest(getfield(_METAL_MOD, :functional))
                return Base.invokelatest(getfield(_METAL_MOD, :MetalBackend)),
                       Float32,
                       "Metal"
            end
        catch err
            requested == "metal" && rethrow(err)
        end
    end
    return KernelAbstractions.CPU(), Float64, "CPU"
end

_parse_list(::Type{T}, raw::AbstractString) where {T} =
    [parse(T, strip(x)) for x in split(raw, ',') if !isempty(strip(x))]

_parse_symbol_list(raw::AbstractString) =
    [Symbol(strip(x)) for x in split(raw, ',') if !isempty(strip(x))]

function _polymer_bc(name::Symbol, tau_plus)
    name === :cnebb && return CNEBB()
    name === :cnebb_qaware && return CNEBBQAware()
    name in (:cnebb_eq_gradient, :cnebb_eqgrad, :eq_gradient) && return CNEBBEqGradient()
    name === :ylw_a && return YLW_A(tau_plus=tau_plus)
    name === :ylw_b && return YLW_B(tau_plus=tau_plus)
    name === :ylw_balance && return YLWBalanceOnly()
    name === :none && return NoPolymerWallBC()
    error("unknown polymer BC $(name); expected cnebb, cnebb_qaware, cnebb_eq_gradient, ylw_a, ylw_b, ylw_balance, or none")
end

@inline function _couette_constants(Ri, Ro, Ω)
    A = -Ω * Ri^2 / (Ro^2 - Ri^2)
    B = Ω * Ri^2 * Ro^2 / (Ro^2 - Ri^2)
    return A, B
end

function _velocity_and_target_stress(x, y; cx, cy, Ri, Ro, Ω, νp)
    dx = x - cx
    dy = y - cy
    r = hypot(dx, dy)
    r <= Ri && return (ux=0.0, uy=0.0, txx=0.0, txy=0.0, tyy=0.0)
    A, B = _couette_constants(Ri, Ro, Ω)
    c = dx / r
    s = dy / r
    uθ = A * r + B / r
    ux = -uθ * s
    uy =  uθ * c

    Srθ = -B / (r * r)
    Sxx = Srθ * (-2.0 * c * s)
    Sxy = Srθ * (c * c - s * s)
    Syy = Srθ * (2.0 * c * s)
    return (ux=ux, uy=uy,
            txx=2.0 * νp * Sxx,
            txy=2.0 * νp * Sxy,
            tyy=2.0 * νp * Syy)
end

function _fill_frozen_velocity!(ux_h, uy_h, is_solid_h; cx, cy, Ri, Ro, Ω, νp)
    Nx, Ny = size(ux_h)
    for j in 1:Ny, i in 1:Nx
        if is_solid_h[i, j]
            ux_h[i, j] = 0.0
            uy_h[i, j] = 0.0
        else
            x = i - 1.0
            y = j - 1.0
            v = _velocity_and_target_stress(x, y; cx, cy, Ri, Ro, Ω, νp)
            ux_h[i, j] = v.ux
            uy_h[i, j] = v.uy
        end
    end
    return nothing
end

function _nearest_cutlink_angle(q_wall_h, cx, cy, θ)
    cxv = (0, 1, 0, -1,  0, 1, -1, -1,  1)
    cyv = (0, 0, 1,  0, -1, 1,  1, -1, -1)
    Nx, Ny, _ = size(q_wall_h)
    best = (dist=Inf, qw=NaN, i=0, j=0, q=0, θw=NaN)
    for j in 1:Ny, i in 1:Nx, q in 2:9
        qw = Float64(q_wall_h[i, j, q])
        qw > 0 || continue
        xw = i - 1.0 + qw * cxv[q]
        yw = j - 1.0 + qw * cyv[q]
        θw = atan(yw - cy, xw - cx)
        δ = abs(atan(sin(θw - θ), cos(θw - θ)))
        if δ < best.dist
            best = (dist=δ, qw=qw, i=i, j=j, q=q, θw=θw)
        end
    end
    return best
end

function _sample_cell(cx, cy, Ri, θ, d, Nx, Ny, is_solid_h)
    target_r = Ri + d
    best = (score=Inf, i=1, j=1)
    for jj in 1:Ny, ii in 1:Nx
        is_solid_h[ii, jj] && continue
        x = ii - 1.0
        y = jj - 1.0
        dx = x - cx
        dy = y - cy
        r = hypot(dx, dy)
        abs(r - target_r) <= 1.5 || continue
        θi = atan(dy, dx)
        δθ = atan(sin(θi - θ), cos(θi - θ))
        score = (r - target_r)^2 + (Ri * δθ)^2
        if score < best.score
            best = (score=score, i=ii, j=jj)
        end
    end
    return best.i, best.j
end

function _target_stress_fields(Nx, Ny; cx, cy, Ri, Ro, Ω, νp)
    txx = zeros(Float64, Nx, Ny)
    txy = zeros(Float64, Nx, Ny)
    tyy = zeros(Float64, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        target = _velocity_and_target_stress(i - 1.0, j - 1.0;
                                             cx, cy, Ri, Ro, Ω, νp)
        txx[i, j] = target.txx
        txy[i, j] = target.txy
        tyy[i, j] = target.tyy
    end
    return txx, txy, tyy
end

function _cutlink_abs_force_2d(tau_xx, tau_xy, tau_yy, q_wall,
                               Nx::Integer, Ny::Integer; cx, cy, radius,
                               reconstruction_order::Integer=1,
                               reconstruction_mode::Symbol=:interior)
    cxv = (0, 1, 0, -1,  0, 1, -1, -1,  1)
    cyv = (0, 0, 1,  0, -1, 1,  1, -1, -1)
    points = Vector{NTuple{6,Float64}}()
    @inbounds for j in 1:Ny, i in 1:Nx, q in 2:9
        q_w = Float64(q_wall[i, j, q])
        q_w > 0 || continue
        xw = i - 1.0 + q_w * cxv[q]
        yw = j - 1.0 + q_w * cyv[q]
        rx = xw - cx
        ry = yw - cy
        r = hypot(rx, ry)
        r > 0 || continue
        nx = rx / r
        ny = ry / r

        txx_w = Float64(tau_xx[i, j])
        txy_w = Float64(tau_xy[i, j])
        tyy_w = Float64(tau_yy[i, j])
        if reconstruction_mode === :interior
            txx_w = Kraken.reconstruct_wall_link_value_2d(tau_xx, i, j, q,
                                                          q_w; location=:cut,
                                                          order=reconstruction_order)
            txy_w = Kraken.reconstruct_wall_link_value_2d(tau_xy, i, j, q,
                                                          q_w; location=:cut,
                                                          order=reconstruction_order)
            tyy_w = Kraken.reconstruct_wall_link_value_2d(tau_yy, i, j, q,
                                                          q_w; location=:cut,
                                                          order=reconstruction_order)
        elseif reconstruction_mode === :wall_cell
            ib = i - cxv[q]
            jb = j - cyv[q]
            if 1 <= ib <= Nx && 1 <= jb <= Ny
                txx_w += q_w * (txx_w - Float64(tau_xx[ib, jb]))
                txy_w += q_w * (txy_w - Float64(tau_xy[ib, jb]))
                tyy_w += q_w * (tyy_w - Float64(tau_yy[ib, jb]))
            end
        else
            error("unknown reconstruction_mode $(reconstruction_mode)")
        end
        push!(points, (atan(ry, rx), nx, ny, txx_w, txy_w, tyy_w))
    end

    isempty(points) && return (Fx_abs=0.0, Fy_abs=0.0, n_cutlinks=0)
    sort!(points; by=first)
    Fx_abs = 0.0
    Fy_abs = 0.0
    npts = length(points)
    @inbounds for k in 1:npts
        θ_prev = k == 1 ? points[end][1] - 2π : points[k - 1][1]
        θ_next = k == npts ? points[1][1] + 2π : points[k + 1][1]
        ds = Float64(radius) * 0.5 * (θ_next - θ_prev)
        _, nx, ny, txx_w, txy_w, tyy_w = points[k]
        Fx_abs += abs(txx_w * nx + txy_w * ny) * ds
        Fy_abs += abs(txy_w * nx + tyy_w * ny) * ds
    end
    return (Fx_abs=Fx_abs, Fy_abs=Fy_abs, n_cutlinks=npts)
end

function _run_case(; backend, FT, Nx, Ny, Ri, Ro, Ω, Wi, νp, tau_plus,
                   magic, max_steps, sample_interval, convergence_tol,
                   model_name, bc_name, results_io, cutlink_io, θ_list, d_list)
    cx = (Nx - 1) / 2
    cy = (Ny - 1) / 2
    λ = Wi * Ri / max(abs(Ω) * Ri, eps(Float64))
    G = νp / λ
    polymer_model = model_name === :logconf ?
        LogConfOldroydB(G=FT(G), λ=FT(λ)) :
        OldroydB(G=FT(G), λ=FT(λ))
    polymer_bc = _polymer_bc(bc_name, tau_plus)
    use_logconf = uses_log_conformation(polymer_model)

    q_wall_h, is_solid_h = precompute_q_wall_cylinder(Nx, Ny, cx, cy, Ri; FT=FT)
    ux_h = zeros(FT, Nx, Ny)
    uy_h = zeros(FT, Nx, Ny)
    _fill_frozen_velocity!(ux_h, uy_h, is_solid_h; cx, cy, Ri, Ro, Ω, νp)

    q_wall = KernelAbstractions.allocate(backend, FT, Nx, Ny, 9)
    is_solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny)
    ux = KernelAbstractions.allocate(backend, FT, Nx, Ny)
    uy = KernelAbstractions.allocate(backend, FT, Nx, Ny)
    copyto!(q_wall, q_wall_h)
    copyto!(is_solid, is_solid_h)
    copyto!(ux, ux_h)
    copyto!(uy, uy_h)

    C_xx = KernelAbstractions.zeros(backend, FT, Nx, Ny); fill!(C_xx, one(FT))
    C_xy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    C_yy = KernelAbstractions.zeros(backend, FT, Nx, Ny); fill!(C_yy, one(FT))
    Ψ_xx = use_logconf ? KernelAbstractions.zeros(backend, FT, Nx, Ny) : C_xx
    Ψ_xy = use_logconf ? KernelAbstractions.zeros(backend, FT, Nx, Ny) : C_xy
    Ψ_yy = use_logconf ? KernelAbstractions.zeros(backend, FT, Nx, Ny) : C_yy

    g_xx = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    g_xy = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    g_yy = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    init_conformation_field_2d!(g_xx, Ψ_xx, ux, uy)
    init_conformation_field_2d!(g_xy, Ψ_xy, ux, uy)
    init_conformation_field_2d!(g_yy, Ψ_yy, ux, uy)
    g_xx_buf = similar(g_xx)
    g_xy_buf = similar(g_xy)
    g_yy_buf = similar(g_yy)

    last_Cxx = Array(C_xx)
    last_Cxy = Array(C_xy)
    last_Cyy = Array(C_yy)
    converged_step = max_steps
    max_delta = Inf

    for step in 1:max_steps
        stream_2d!(g_xx_buf, g_xx, Nx, Ny)
        stream_2d!(g_xy_buf, g_xy, Nx, Ny)
        stream_2d!(g_yy_buf, g_yy, Nx, Ny)

        apply_polymer_wall_bc!(g_xx_buf, g_xx, is_solid, q_wall, Ψ_xx, ux, uy, polymer_bc)
        apply_polymer_wall_bc!(g_xy_buf, g_xy, is_solid, q_wall, Ψ_xy, ux, uy, polymer_bc)
        apply_polymer_wall_bc!(g_yy_buf, g_yy, is_solid, q_wall, Ψ_yy, ux, uy, polymer_bc)

        g_xx, g_xx_buf = g_xx_buf, g_xx
        g_xy, g_xy_buf = g_xy_buf, g_xy
        g_yy, g_yy_buf = g_yy_buf, g_yy

        compute_conformation_macro_2d!(Ψ_xx, g_xx)
        compute_conformation_macro_2d!(Ψ_xy, g_xy)
        compute_conformation_macro_2d!(Ψ_yy, g_yy)

        if use_logconf
            collide_logconf_2d!(g_xx, Ψ_xx, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid,
                                tau_plus, λ; magic=magic, component=1)
            collide_logconf_2d!(g_xy, Ψ_xy, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid,
                                tau_plus, λ; magic=magic, component=2)
            collide_logconf_2d!(g_yy, Ψ_yy, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid,
                                tau_plus, λ; magic=magic, component=3)
            psi_to_C_2d!(C_xx, C_xy, C_yy, Ψ_xx, Ψ_xy, Ψ_yy)
        else
            collide_conformation_2d!(g_xx, C_xx, ux, uy, C_xx, C_xy, C_yy,
                                     is_solid, tau_plus, λ; magic=magic, component=1)
            collide_conformation_2d!(g_xy, C_xy, ux, uy, C_xx, C_xy, C_yy,
                                     is_solid, tau_plus, λ; magic=magic, component=2)
            collide_conformation_2d!(g_yy, C_yy, ux, uy, C_xx, C_xy, C_yy,
                                     is_solid, tau_plus, λ; magic=magic, component=3)
        end

        if step % sample_interval == 0 || step == max_steps
            Cxx_h = Array(C_xx)
            Cxy_h = Array(C_xy)
            Cyy_h = Array(C_yy)
            max_delta = max(maximum(abs.(Cxx_h .- last_Cxx)),
                            maximum(abs.(Cxy_h .- last_Cxy)),
                            maximum(abs.(Cyy_h .- last_Cyy)))
            last_Cxx = Cxx_h
            last_Cxy = Cxy_h
            last_Cyy = Cyy_h
            if max_delta < convergence_tol
                converged_step = step
                break
            end
        end
    end

    Cxx_h = Array(C_xx)
    Cxy_h = Array(C_xy)
    Cyy_h = Array(C_yy)
    tau_xx = G .* (Cxx_h .- 1.0)
    tau_xy = G .* Cxy_h
    tau_yy = G .* (Cyy_h .- 1.0)
    target_xx, target_xy, target_yy = _target_stress_fields(Nx, Ny;
                                                            cx, cy, Ri, Ro,
                                                            Ω, νp)
    diff_xx = tau_xx .- target_xx
    diff_xy = tau_xy .- target_xy
    diff_yy = tau_yy .- target_yy
    sim_force = Kraken.compute_polymeric_drag_2d(tau_xx, tau_xy, tau_yy,
                                                 q_wall_h, Nx, Ny; cx, cy,
                                                 radius=Ri,
                                                 reconstruction_order=1,
                                                 reconstruction_mode=:interior)
    target_force = Kraken.compute_polymeric_drag_2d(target_xx, target_xy,
                                                    target_yy, q_wall_h, Nx,
                                                    Ny; cx, cy, radius=Ri,
                                                    reconstruction_order=1,
                                                    reconstruction_mode=:interior)
    err_force = Kraken.compute_polymeric_drag_2d(diff_xx, diff_xy, diff_yy,
                                                 q_wall_h, Nx, Ny; cx, cy,
                                                 radius=Ri,
                                                 reconstruction_order=1,
                                                 reconstruction_mode=:interior)
    err_abs = _cutlink_abs_force_2d(diff_xx, diff_xy, diff_yy, q_wall_h,
                                    Nx, Ny; cx, cy, radius=Ri,
                                    reconstruction_order=1,
                                    reconstruction_mode=:interior)
    cancel_x = abs(err_force.Fx) / max(err_abs.Fx_abs, eps(Float64))
    cancel_y = abs(err_force.Fy) / max(err_abs.Fy_abs, eps(Float64))
    @printf(cutlink_io,
            "%s,%s,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%.17g,%d,%.17g\n",
            model_name, bc_name, Nx, Ny, Ri, Ro, Ω, Wi, νp,
            sim_force.Fx, sim_force.Fy, target_force.Fx, target_force.Fy,
            err_force.Fx, err_force.Fy, err_abs.Fx_abs, err_abs.Fy_abs,
            cancel_x, cancel_y, err_abs.n_cutlinks, max_delta,
            converged_step, λ)
    flush(cutlink_io)

    for θ in θ_list
        cut = _nearest_cutlink_angle(q_wall_h, cx, cy, θ)
        for d in d_list
            i, j = _sample_cell(cx, cy, Ri, θ, d, Nx, Ny, is_solid_h)
            x = i - 1.0
            y = j - 1.0
            target = _velocity_and_target_stress(x, y; cx, cy, Ri, Ro, Ω, νp)
            τxx = tau_xx[i, j]
            τxy = tau_xy[i, j]
            τyy = tau_yy[i, j]
            err = sqrt((τxx - target.txx)^2 +
                       (τxy - target.txy)^2 +
                       (τyy - target.tyy)^2)
            norm = sqrt(target.txx^2 + target.txy^2 + target.tyy^2)
            rel = err / max(norm, eps(Float64))
            r = hypot(x - cx, y - cy)
            @printf(results_io,
                    "%s,%s,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%.17g\n",
                    model_name, bc_name, Nx, Ny, Ri, Ro, Ω, Wi, νp,
                    round(Int, rad2deg(θ)), d, cut.q, cut.qw, r - Ri,
                    τxx, τxy, τyy, target.txx, target.txy, target.tyy,
                    err, norm, rel, max_delta, converged_step, λ)
        end
    end
    flush(results_io)
    return nothing
end

backend, FT, backend_label = _select_backend()

R = parse(Int, get(ENV, "KRAKEN_R", "15"))
Nx = parse(Int, get(ENV, "KRAKEN_NX", string(8R)))
Ny = parse(Int, get(ENV, "KRAKEN_NY", string(8R)))
Ro = parse(Float64, get(ENV, "KRAKEN_RO", string(3R)))
Ω = parse(Float64, get(ENV, "KRAKEN_OMEGA", "0.0005"))
Wi = parse(Float64, get(ENV, "KRAKEN_WI", "0.001"))
νp = parse(Float64, get(ENV, "KRAKEN_NU_P", "0.041"))
tau_plus = parse(Float64, get(ENV, "KRAKEN_TAU_PLUS", "0.51"))
magic = parse(Float64, get(ENV, "KRAKEN_CONFORMATION_MAGIC", "1e-6"))
max_steps = parse(Int, get(ENV, "KRAKEN_STEPS", "2000"))
sample_interval = parse(Int, get(ENV, "KRAKEN_SAMPLE_INTERVAL", "100"))
convergence_tol = parse(Float64, get(ENV, "KRAKEN_CONVERGENCE_TOL", "1e-11"))
models = _parse_symbol_list(get(ENV, "KRAKEN_MODELS", "direct"))
polymer_bcs = _parse_symbol_list(get(ENV, "KRAKEN_POLYMER_BCS", "cnebb,cnebb_qaware,ylw_a,ylw_b"))
θ_deg_list = _parse_list(Float64, get(ENV, "KRAKEN_THETA_DEG_LIST", "0,45,90,135,180,225,270,315"))
d_list = _parse_list(Int, get(ENV, "KRAKEN_D_LIST", "1,2,3"))
θ_list = deg2rad.(θ_deg_list)

for model in models
    model in (:direct, :logconf) || error("unknown model $(model); expected direct or logconf")
end
foreach(bc -> _polymer_bc(bc, tau_plus), polymer_bcs)

results_dir = get(ENV, "KRAKEN_RESULTS_DIR",
    joinpath("tmp", "cde_frozen_curved_wall_" * Dates.format(now(), "yyyymmdd_HHMMSS")))
mkpath(results_dir)
csv_path = joinpath(results_dir, "cde_frozen_curved_wall_patch.csv")
cutlink_csv_path = joinpath(results_dir, "cutlink_force_error.csv")

println("="^120)
println("Frozen CDE curved-wall patch")
println("Backend: $backend_label, FT=$FT")
println("Nx=$Nx Ny=$Ny R=$R Ro=$Ro Ω=$Ω Wi=$Wi νp=$νp λ=$(Wi * R / max(abs(Ω) * R, eps(Float64)))")
println("tau_plus=$tau_plus magic=$magic steps=$max_steps sample_interval=$sample_interval")
println("models=$(join(models, ",")) bcs=$(join(polymer_bcs, ","))")
println("CSV: $csv_path")
println("Cut-link CSV: $cutlink_csv_path")
println("="^120)

open(csv_path, "w") do io
open(cutlink_csv_path, "w") do cutlink_io
    println(io, join(("model", "polymer_bc", "Nx", "Ny", "R", "Ro", "Omega",
                      "Wi", "nu_p", "theta_deg", "d_requested", "cut_q",
                      "q_wall", "actual_d", "tau_xx", "tau_xy", "tau_yy",
                      "target_xx", "target_xy", "target_yy", "abs_err",
                      "target_norm", "rel_err", "max_delta", "converged_step",
                      "lambda"), ","))
    println(cutlink_io, join(("model", "polymer_bc", "Nx", "Ny", "R", "Ro",
                              "Omega", "Wi", "nu_p", "Fx_sim", "Fy_sim",
                              "Fx_target", "Fy_target", "Fx_err_signed",
                              "Fy_err_signed", "Fx_err_abs", "Fy_err_abs",
                              "cancel_x", "cancel_y", "n_cutlinks",
                              "max_delta", "converged_step", "lambda"), ","))
    for model in models, bc in polymer_bcs
        t0 = time()
        _run_case(; backend, FT, Nx, Ny, Ri=R, Ro, Ω, Wi, νp, tau_plus,
                  magic, max_steps, sample_interval, convergence_tol,
                  model_name=model, bc_name=bc, results_io=io,
                  cutlink_io, θ_list, d_list)
        @printf("%-8s %-14s %.2fs\n", string(model), string(bc), time() - t0)
        flush(stdout)
    end
end
end

println("Done: $csv_path")
println("Done: $cutlink_csv_path")
