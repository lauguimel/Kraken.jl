using Kraken
using KernelAbstractions
using Printf

# M21 Poiseuille BSD path matrix. The V7 gradient path is intentionally a
# regression check: the log-FV wrapper is documented as a thin pass-through to
# the FVFD gradient operator, so V7 should match the baseline row means exactly.

const OUT_DIR = joinpath(@__DIR__, "..", "scratch")
const CSV_COLUMNS = [
    "y_idx",
    "y",
    "ux_mean",
    "F_poly_wide",
    "F_BSD_kind",
    "F_total_nb",
    "F_poly_target",
    "F_total_target",
    "rel_resid_F_poly",
    "rel_resid_F_total",
]
const SELFTEST_VARIANTS = (:baseline, :no_bsd, :fd_v2)
const FULL_VARIANTS = (
    :baseline,
    :no_bsd,
    :fd_v2,
    :fd_v2_unc,
    :kinetic,
    :epsilon_force,
    :baseline_fvfd_grad,
)

struct Case
    label::String
    lambda::Float64
end

function high_wi_lambda(nu_s::Float64, nu_p::Float64, Fx_body::Float64, Ny::Int)
    nu_total = nu_s + nu_p
    gamma_dot_max = abs(Fx_body) * Ny / (2.0 * nu_total)
    gamma_dot_max > 0.0 || throw(ArgumentError("gamma_dot_max must be positive"))
    return 1.0 / gamma_dot_max
end

function build_cases(Ny, Fx_body, nu_s, nu_p)
    return [
        Case("A", 1.0),
        Case("A_high_Wi", high_wi_lambda(nu_s, nu_p, Fx_body, Ny)),
    ]
end

function row_means(a)
    Nx, Ny = size(a)
    out = Vector{Float64}(undef, Ny)
    @inbounds for j in 1:Ny
        s = 0.0
        for i in 1:Nx
            s += Float64(a[i, j])
        end
        out[j] = s / Nx
    end
    return out
end

rel_resid(value::Float64, target::Float64) =
    abs(value - target) / max(abs(target), eps(Float64))

function targets(nu_s::Float64, nu_p::Float64, Fx_body::Float64, zeta::Float64)
    nu_total = nu_s + nu_p
    F_poly_tgt = -nu_p * Fx_body / nu_total
    F_total_tgt = -(1.0 - zeta) * nu_p * Fx_body / nu_total
    return F_poly_tgt, F_total_tgt
end

variant_zeta(variant::Symbol) = variant === :no_bsd ? 0.0 : 0.75

function check_nan_state(step, ux, uy, psixx, psixy, psiyy, backend)
    if step % 5000 != 0
        return -1
    end
    KernelAbstractions.synchronize(backend)
    if any(isnan, ux) || any(isnan, uy) ||
       any(isnan, psixx) || any(isnan, psixy) || any(isnan, psiyy)
        return step
    end
    return -1
end

function copy_field!(dst, src, Nx::Int, Ny::Int)
    @inbounds for j in 1:Ny, i in 1:Nx
        dst[i, j] = src[i, j]
    end
    return nothing
end

function subtract_force!(fx_total, fy_total, fx_poly, fy_poly, fx_bsd, fy_bsd, Nx::Int, Ny::Int)
    @inbounds for j in 1:Ny, i in 1:Nx
        fx_total[i, j] = fx_poly[i, j] - fx_bsd[i, j]
        fy_total[i, j] = fy_poly[i, j] - fy_bsd[i, j]
    end
    return nothing
end

function epsilon_force!(
    fx_total, fy_total, fx_poly, fy_poly, ux, uy, dudx, dudy, dvdx, dvdy,
    tau_newton_xx, tau_newton_xy, tau_newton_yy, fx_poly_newton, fy_poly_newton,
    fx_poly_elastic, fy_poly_elastic, is_solid, logfv_bc, nu_p_t, bsd_t, dx, dy,
    Nx::Int, Ny::Int,
)
    @inbounds for j in 1:Ny, i in 1:Nx
        tau_newton_xx[i, j] = 2 * nu_p_t * dudx[i, j]
        tau_newton_xy[i, j] = nu_p_t * (dudy[i, j] + dvdx[i, j])
        tau_newton_yy[i, j] = 2 * nu_p_t * dvdy[i, j]
    end
    Kraken.logfv_polymer_force_bc_aware_2d!(
        fx_poly_newton, fy_poly_newton, tau_newton_xx, tau_newton_xy, tau_newton_yy,
        is_solid, dx, dy, logfv_bc; sync=false,
    )
    @inbounds for j in 1:Ny, i in 1:Nx
        fx_poly_elastic[i, j] = fx_poly[i, j] - fx_poly_newton[i, j]
        fy_poly_elastic[i, j] = fy_poly[i, j] - fy_poly_newton[i, j]
    end
    # Halfway bounce-back walls sit half a cell outside the first and last
    # fluid rows, so the mirror ghost has the opposite wall-tangent velocity.
    @inbounds for j in 1:Ny, i in 1:Nx
        ip = i == Nx ? 1 : i + 1
        im = i == 1 ? Nx : i - 1
        if j == 1
            u_south_x = -ux[i, 1]
            u_south_y = -uy[i, 1]
            u_north_x = ux[i, 2]
            u_north_y = uy[i, 2]
        elseif j == Ny
            u_south_x = ux[i, Ny - 1]
            u_south_y = uy[i, Ny - 1]
            u_north_x = -ux[i, Ny]
            u_north_y = -uy[i, Ny]
        else
            u_south_x = ux[i, j - 1]
            u_south_y = uy[i, j - 1]
            u_north_x = ux[i, j + 1]
            u_north_y = uy[i, j + 1]
        end
        lap_ux = ux[ip, j] + ux[im, j] + u_north_x + u_south_x - 4 * ux[i, j]
        lap_uy = uy[ip, j] + uy[im, j] + u_north_y + u_south_y - 4 * uy[i, j]
        fx_total[i, j] = (1 - bsd_t) * nu_p_t * lap_ux + fx_poly_elastic[i, j]
        fy_total[i, j] = (1 - bsd_t) * nu_p_t * lap_uy + fy_poly_elastic[i, j]
    end
    return nothing
end

function run_variant(
    variant::Symbol, c::Case, Nx::Int, Ny::Int, Fx_body::Float64,
    nu_s::Float64, nu_p::Float64, max_steps::Int,
)
    T = Float64
    backend = KernelAbstractions.CPU()
    bsd_t = T(variant_zeta(variant))
    nu_s_t, nu_p_t = T(nu_s), T(nu_p)
    Fx_body_t, lambda_t = T(Fx_body), T(c.lambda)
    nu_lbm_t = nu_s_t + bsd_t * nu_p_t
    prefactor_t = nu_p_t / lambda_t
    dx = one(T)
    dy = one(T)
    logfv_bc = Kraken.logfv_periodicx_wally_bcspec_2d()
    max_grad_norm_estimate = abs(Fx_body_t) * T(Ny) / (T(2) * (nu_s_t + nu_p_t))
    sub = Kraken.logfv_oldroydb_subcycle_estimate(
        Float64(max_grad_norm_estimate), Float64(lambda_t), 1.0;
        relative_tolerance=0.01,
        max_deformation_increment=0.05,
        max_memory_deformation_increment=0.07,
        min_substeps=1, max_substeps=64,
    )
    selected_polymer_substeps = sub.recommended
    dt_poly = one(T) / T(selected_polymer_substeps)

    config = Kraken.LBMConfig(Kraken.D2Q9(); Nx=Nx, Ny=Ny, ν=Float64(nu_lbm_t), u_lid=0.0, max_steps=max_steps)
    state = Kraken.initialize_2d(config, T; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    rho, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid
    omega_t = T(Kraken.omega(config))

    z = KernelAbstractions.zeros
    psixx = z(backend, T, Nx, Ny); psixy = z(backend, T, Nx, Ny); psiyy = z(backend, T, Nx, Ny)
    psixx_next = z(backend, T, Nx, Ny); psixy_next = z(backend, T, Nx, Ny); psiyy_next = z(backend, T, Nx, Ny)
    dudx = z(backend, T, Nx, Ny); dudy = z(backend, T, Nx, Ny); dvdx = z(backend, T, Nx, Ny); dvdy = z(backend, T, Nx, Ny)
    tauxx = z(backend, T, Nx, Ny); tauxy = z(backend, T, Nx, Ny); tauyy = z(backend, T, Nx, Ny)
    fx_poly = z(backend, T, Nx, Ny); fy_poly = z(backend, T, Nx, Ny)
    fx_total = z(backend, T, Nx, Ny); fy_total = z(backend, T, Nx, Ny)
    tau_bsd_xx = z(backend, T, Nx, Ny); tau_bsd_xy = z(backend, T, Nx, Ny); tau_bsd_yy = z(backend, T, Nx, Ny)
    fx_bsd = z(backend, T, Nx, Ny); fy_bsd = z(backend, T, Nx, Ny)
    dudx_unc = z(backend, T, Nx, Ny); dudy_unc = z(backend, T, Nx, Ny); dvdx_unc = z(backend, T, Nx, Ny); dvdy_unc = z(backend, T, Nx, Ny)
    tau_newton_xx = z(backend, T, Nx, Ny); tau_newton_xy = z(backend, T, Nx, Ny); tau_newton_yy = z(backend, T, Nx, Ny)
    fx_poly_newton = z(backend, T, Nx, Ny); fy_poly_newton = z(backend, T, Nx, Ny)
    fx_poly_elastic = z(backend, T, Nx, Ny); fy_poly_elastic = z(backend, T, Nx, Ny)

    nan_step = -1
    for step in 1:max_steps
        if variant === :baseline_fvfd_grad
            Kraken.fvfd_velocity_gradient_2d!(dudx, dudy, dvdx, dvdy, ux, uy, is_solid, dx, dy, logfv_bc; sync=false)
        else
            Kraken.logfv_velocity_gradient_bc_aware_2d!(dudx, dudy, dvdx, dvdy, ux, uy, is_solid, dx, dy, logfv_bc; sync=false)
        end
        for _ in 1:selected_polymer_substeps
            Kraken.logfv_step_oldroydb_log_2d!(
                psixx_next, psixy_next, psiyy_next, psixx, psixy, psiyy,
                dudx, dudy, dvdx, dvdy, lambda_t, dt_poly; sync=false,
            )
            psixx, psixx_next = psixx_next, psixx
            psixy, psixy_next = psixy_next, psixy
            psiyy, psiyy_next = psiyy_next, psiyy
        end
        Kraken.logfv_stress_from_log_2d!(tauxx, tauxy, tauyy, psixx, psixy, psiyy, prefactor_t; sync=false)
        Kraken.logfv_polymer_force_bc_aware_2d!(fx_poly, fy_poly, tauxx, tauxy, tauyy, is_solid, dx, dy, logfv_bc; sync=false)

        if variant === :baseline || variant === :baseline_fvfd_grad
            Kraken.logfv_bsd_correct_force_bc_aware_2d!(fx_total, fy_total, fx_poly, fy_poly, ux, uy, is_solid, bsd_t, nu_p_t, dx, dy, logfv_bc; sync=false)
        elseif variant === :no_bsd
            copy_field!(fx_total, fx_poly, Nx, Ny)
            copy_field!(fy_total, fy_poly, Nx, Ny)
        elseif variant === :fd_v2
            Kraken.logfv_bsd_stress_from_gradient_2d!(tau_bsd_xx, tau_bsd_xy, tau_bsd_yy, dudx, dudy, dvdx, dvdy, bsd_t * nu_p_t; sync=false)
            Kraken.logfv_polymer_force_bc_aware_2d!(fx_bsd, fy_bsd, tau_bsd_xx, tau_bsd_xy, tau_bsd_yy, is_solid, dx, dy, logfv_bc; sync=false)
            subtract_force!(fx_total, fy_total, fx_poly, fy_poly, fx_bsd, fy_bsd, Nx, Ny)
        elseif variant === :fd_v2_unc
            Kraken.logfv_velocity_gradient_centered_2d!(dudx_unc, dudy_unc, dvdx_unc, dvdy_unc, ux, uy, dx, dy; sync=false)
            Kraken.logfv_bsd_stress_from_gradient_2d!(tau_bsd_xx, tau_bsd_xy, tau_bsd_yy, dudx_unc, dudy_unc, dvdx_unc, dvdy_unc, bsd_t * nu_p_t; sync=false)
            Kraken.logfv_polymer_force_bc_aware_2d!(fx_bsd, fy_bsd, tau_bsd_xx, tau_bsd_xy, tau_bsd_yy, is_solid, dx, dy, logfv_bc; sync=false)
            subtract_force!(fx_total, fy_total, fx_poly, fy_poly, fx_bsd, fy_bsd, Nx, Ny)
        elseif variant === :kinetic
            Kraken.compute_bsd_force_kinetic_2d!(fx_total, fy_total, fx_poly, fy_poly, f_in, rho, ux, uy, is_solid, bsd_t, nu_p_t, omega_t, dx, dy; sync=false)
        elseif variant === :epsilon_force
            epsilon_force!(
                fx_total, fy_total, fx_poly, fy_poly, ux, uy, dudx, dudy, dvdx, dvdy,
                tau_newton_xx, tau_newton_xy, tau_newton_yy, fx_poly_newton, fy_poly_newton,
                fx_poly_elastic, fy_poly_elastic, is_solid, logfv_bc, nu_p_t, bsd_t, dx, dy, Nx, Ny,
            )
        else
            throw(ArgumentError("unknown variant $(variant)"))
        end

        Kraken.logfv_add_constant_force_2d!(fx_total, fy_total, Fx_body_t, zero(T); sync=false)
        Kraken.stream_periodic_x_wall_y_2d!(f_out, f_in, Nx, Ny)
        Kraken.collide_guo_field_2d!(f_out, is_solid, fx_total, fy_total, omega_t)
        Kraken.logfv_compute_macroscopic_forced_field_2d!(rho, ux, uy, f_out, fx_total, fy_total; sync=false)
        f_in, f_out = f_out, f_in

        nan_step = check_nan_state(step, ux, uy, psixx, psixy, psiyy, backend)
        nan_step == -1 || break
    end
    KernelAbstractions.synchronize(backend)
    return summarise_case(variant, c, Nx, Ny, Fx_body, nu_s, nu_p, nan_step, ux, uy, psixx, psixy, psiyy, tauxx, tauxy, tauyy, fx_poly, fx_total)
end

function summarise_case(
    variant::Symbol, c::Case, Nx::Int, Ny::Int, Fx_body::Float64, nu_s::Float64,
    nu_p::Float64, nan_step::Int, ux, uy, psixx, psixy, psiyy, tauxx, tauxy, tauyy,
    fx_poly, fx_total,
)
    if nan_step != -1
        return (; nan_step, completed=false, rows=NamedTuple[], metrics=nan_metrics(), F_total_nb=Float64[])
    end
    nu_total = nu_s + nu_p
    u_analytical = [Fx_body / (2 * nu_total) * (j - 0.5) * (Ny + 0.5 - j) for j in 1:Ny]
    gamma_dot_analytical = [Fx_body / (2 * nu_total) * (Ny + 1 - 2 * j) for j in 1:Ny]
    mean_ux = row_means(ux)
    mean_tauxx = row_means(tauxx)
    mean_tauxy = row_means(tauxy)
    mean_tauyy = row_means(tauyy)
    F_poly = row_means(fx_poly)
    F_total_nb = row_means(fx_total .- Fx_body)
    F_BSD = F_poly .- F_total_nb
    F_poly_tgt, F_total_tgt = targets(nu_s, nu_p, Fx_body, variant_zeta(variant))

    interior = 3:(Ny - 2)
    walls = (1, 2, Ny - 1, Ny)
    u_max_ref = maximum(abs.(u_analytical[interior]))
    u_int_L2 = sqrt(sum((mean_ux[j] - u_analytical[j])^2 for j in interior) / length(interior)) / u_max_ref
    u_wall_max = maximum(abs(mean_ux[j] - u_analytical[j]) / u_max_ref for j in walls)
    tauxy_max_ref = maximum(abs(nu_p * gamma_dot_analytical[j]) for j in 1:Ny)
    tauxy_int_L2 = sqrt(sum((mean_tauxy[j] - nu_p * gamma_dot_analytical[j])^2 for j in interior) / length(interior)) / max(tauxy_max_ref, eps(Float64))
    if c.label == "A_high_Wi"
        target_xx = [2 * nu_p * c.lambda * gamma_dot_analytical[j]^2 for j in 1:Ny]
        tauxx_int_L2 = sqrt(sum((mean_tauxx[j] - target_xx[j])^2 for j in interior) / length(interior)) /
                       max(maximum(abs.(target_xx[interior])), eps(Float64))
    else
        tauxx_int_L2 = maximum(abs.(mean_tauxx))
    end
    tauyy_max_abs = maximum(abs.(mean_tauyy))
    min_c_eig = Inf
    @inbounds for j in 1:Ny, i in 1:Nx
        cxx, cxy, cyy = Kraken.logfv_exp_sym2_2d(psixx[i, j], psixy[i, j], psiyy[i, j])
        min_c_eig = min(min_c_eig, Kraken.logfv_min_eig_sym2_2d(cxx, cxy, cyy))
    end
    F_total_resid = [rel_resid(Float64(F_total_nb[j]), F_total_tgt) for j in 1:Ny]
    F_total_int_L2 = sqrt(sum(F_total_resid[j]^2 for j in interior) / length(interior))
    F_total_wall_max = maximum(F_total_resid[j] for j in walls)
    rows = [
        (;
            y_idx=j, y=j - 0.5, ux_mean=mean_ux[j], F_poly_wide=F_poly[j],
            F_BSD_kind=F_BSD[j], F_total_nb=F_total_nb[j],
            F_poly_target=F_poly_tgt, F_total_target=F_total_tgt,
            rel_resid_F_poly=rel_resid(Float64(F_poly[j]), F_poly_tgt),
            rel_resid_F_total=rel_resid(Float64(F_total_nb[j]), F_total_tgt),
        ) for j in 1:Ny
    ]
    metrics = (; u_int_L2, u_wall_max, tauxy_int_L2, tauxx_int_L2, tauyy_max_abs,
        F_total_int_L2, F_total_wall_max, min_C_eig=min_c_eig)
    return (; nan_step, completed=true, rows, metrics, F_total_nb)
end

nan_metrics() = (;
    u_int_L2=NaN, u_wall_max=NaN, tauxy_int_L2=NaN, tauxx_int_L2=NaN,
    tauyy_max_abs=NaN, F_total_int_L2=NaN, F_total_wall_max=NaN, min_C_eig=NaN,
)

csv_path(variant::Symbol, c::Case) =
    joinpath(OUT_DIR, "poiseuille_pathmatrix_$(variant)_$(c.label).csv")

function write_csv_for(variant::Symbol, c::Case, result)
    mkpath(OUT_DIR)
    open(csv_path(variant, c), "w") do io
        println(io, join(CSV_COLUMNS, ","))
        if !result.completed
            println(io, "# NaN at step $(result.nan_step)")
            return
        end
        for r in result.rows
            println(
                io,
                @sprintf(
                    "%d,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e",
                    r.y_idx, r.y, r.ux_mean, r.F_poly_wide, r.F_BSD_kind,
                    r.F_total_nb, r.F_poly_target, r.F_total_target,
                    r.rel_resid_F_poly, r.rel_resid_F_total,
                ),
            )
        end
    end
    return nothing
end

function print_summary_line(variant::Symbol, c::Case, result)
    m = result.metrics
    println(@sprintf(
        "SUMMARY %s %s nan_step=%d u_int_L2=%.4e u_wall_max=%.4e tauxy_int_L2=%.4e tauxx_int_L2=%.4e tauyy_max_abs=%.4e F_total_int_L2=%.4e F_total_wall_max=%.4e min_C_eig=%.4e",
        String(variant), c.label, result.nan_step, m.u_int_L2, m.u_wall_max,
        m.tauxy_int_L2, m.tauxx_int_L2, m.tauyy_max_abs, m.F_total_int_L2,
        m.F_total_wall_max, m.min_C_eig,
    ))
end

function parse_args(args)
    allowed = Set(["--full", "--self-test"])
    unknown = setdiff(args, allowed)
    isempty(unknown) || throw(ArgumentError("unknown args: $(join(unknown, ", "))"))
    return !("--full" in args)
end

function assert_selftest_outputs()
    expected_header = join(CSV_COLUMNS, ",")
    for variant in SELFTEST_VARIANTS
        path = csv_path(variant, Case("A", 1.0))
        @assert isfile(path) "missing self-test CSV at $(path)"
        lines = readlines(path)
        @assert length(lines) == 17 "self-test CSV line count mismatch at $(path)"
        @assert lines[1] == expected_header "self-test CSV header mismatch at $(path)"
        vals = Float64[parse(Float64, split(line, ",")[7]) for line in lines[2:end]]
        mean_val = sum(vals) / length(vals)
        denom = max(abs(mean_val), eps(Float64))
        @assert maximum(abs(v - mean_val) / denom for v in vals) <= 0.05 "F_poly_target drift at $(path)"
    end
    return nothing
end

function warn_v7_delta(results)
    key_a = (:baseline, "A")
    key_b = (:baseline_fvfd_grad, "A")
    if haskey(results, key_a) && haskey(results, key_b)
        a = results[key_a].F_total_nb
        b = results[key_b].F_total_nb
        if length(a) == length(b) && !all(isequal(a[j], b[j]) for j in eachindex(a))
            println(stderr, "WARNING baseline and baseline_fvfd_grad case A F_total row means differ")
        end
    end
    return nothing
end

function main(args)
    self_test = parse_args(args)
    if self_test
        Nx, Ny, max_steps = 8, 16, 1000
        variants = SELFTEST_VARIANTS
        cases = [Case("A", 1.0)]
    else
        Nx, Ny, max_steps = 8, 32, 100_000
        variants = FULL_VARIANTS
        cases = build_cases(Ny, 1.0e-5, 0.1, 0.1)
    end
    Fx_body = 1.0e-5
    nu_s, nu_p = 0.1, 0.1
    results = Dict{Tuple{Symbol,String},Any}()
    for variant in variants, c in cases
        result = try
            run_variant(variant, c, Nx, Ny, Fx_body, nu_s, nu_p, max_steps)
        catch err
            # Crash inside a constitutive substep (e.g. DomainError from
            # log(SPD) when C becomes non-SPD) escapes the per-step NaN
            # watcher. Report as nan_step=0 (sentinel: crashed before the
            # first NaN-watcher tick) so the sweep continues over the
            # remaining (variant, case) combinations.
            println(stderr, "CAUGHT variant=", variant, " case=", c.label,
                    " err=", typeof(err))
            (; nan_step=0, completed=false, rows=NamedTuple[],
               metrics=nan_metrics(), F_total_nb=Float64[])
        end
        results[(variant, c.label)] = result
        write_csv_for(variant, c, result)
        print_summary_line(variant, c, result)
    end
    self_test ? assert_selftest_outputs() : warn_v7_delta(results)
    return nothing
end

main(ARGS)
