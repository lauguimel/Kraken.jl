using Kraken
using Printf

const U0 = 1.0
const NU_P = 0.1
const ZETA = 0.75
const NS = (32, 64, 128)
const L2_N = 64
const TWO_PI = 2.0 * pi
const OUT_MD = joinpath(@__DIR__, "BSD_ANALYTICAL_LADDER_20260517.md")

function fill_tg!(ux, uy, N)
    @inbounds for j in 1:N, i in 1:N
        x = (i - 0.5) / N
        y = (j - 0.5) / N
        sx = sin(TWO_PI * x)
        cx = cos(TWO_PI * x)
        sy = sin(TWO_PI * y)
        cy = cos(TWO_PI * y)
        ux[i, j] = U0 * sx * cy
        uy[i, j] = -U0 * cx * sy
    end
    return nothing
end

function fill_tau_p_analytical!(tauxx, tauxy, tauyy, N)
    @inbounds for j in 1:N, i in 1:N
        x = (i - 0.5) / N
        y = (j - 0.5) / N
        dxx = TWO_PI * U0 * cos(TWO_PI * x) * cos(TWO_PI * y)
        tauxx[i, j] = 2.0 * NU_P * dxx
        tauxy[i, j] = 0.0
        tauyy[i, j] = -2.0 * NU_P * dxx
    end
    return nothing
end

function fill_force_ref!(fx_ref, fy_ref, ux, uy, scale)
    coeff = -8.0 * pi^2 * scale
    @inbounds for idx in eachindex(ux)
        fx_ref[idx] = coeff * ux[idx]
        fy_ref[idx] = coeff * uy[idx]
    end
    return nothing
end

function tg_fields(N)
    ux = zeros(Float64, N, N)
    uy = zeros(Float64, N, N)
    fill_tg!(ux, uy, N)
    return ux, uy
end

function tau_p_fields(N)
    tauxx = zeros(Float64, N, N)
    tauxy = zeros(Float64, N, N)
    tauyy = zeros(Float64, N, N)
    fill_tau_p_analytical!(tauxx, tauxy, tauyy, N)
    return tauxx, tauxy, tauyy
end

periodic_bc() =
    Kraken.FVFDDomainBC2D(;
        west=:periodic, east=:periodic, south=:periodic, north=:periodic,
    )

function polymer_force_from_tau(N, tauxx, tauxy, tauyy, is_solid, bc)
    dx = inv(N)
    dy = inv(N)
    fx = zeros(Float64, N, N)
    fy = zeros(Float64, N, N)
    Kraken.logfv_polymer_force_bc_aware_2d!(
        fx, fy, tauxx, tauxy, tauyy, is_solid, dx, dy, bc;
        sync=true, polymer_wall_extrap=:quadratic,
    )
    return fx, fy
end

function rel_l2(fx, fy, fx_ref, fy_ref)
    num = 0.0
    den = 0.0
    @inbounds for idx in eachindex(fx)
        dx = fx[idx] - fx_ref[idx]
        dy = fy[idx] - fy_ref[idx]
        num += dx * dx + dy * dy
        den += fx_ref[idx] * fx_ref[idx] + fy_ref[idx] * fy_ref[idx]
    end
    return sqrt(num / den)
end

function rel_l2_delta(fx_a, fy_a, fx_b, fy_b, fx_ref, fy_ref)
    num = 0.0
    den = 0.0
    @inbounds for idx in eachindex(fx_a)
        dx = fx_a[idx] - fx_b[idx]
        dy = fy_a[idx] - fy_b[idx]
        num += dx * dx + dy * dy
        den += fx_ref[idx] * fx_ref[idx] + fy_ref[idx] * fy_ref[idx]
    end
    return sqrt(num / den)
end

interior_mask(i, j, N) = 3 <= i <= N - 2 && 3 <= j <= N - 2
wall_band_mask(i, j, N) = i <= 2 || i >= N - 1 || j <= 2 || j >= N - 1

function rel_l2_masked(fx, fy, fx_ref, fy_ref, mask)
    N = size(fx, 1)
    num = 0.0
    den = 0.0
    @inbounds for j in 1:N, i in 1:N
        if mask(i, j, N)
            dx = fx[i, j] - fx_ref[i, j]
            dy = fy[i, j] - fy_ref[i, j]
            num += dx * dx + dy * dy
            den += fx_ref[i, j] * fx_ref[i, j] + fy_ref[i, j] * fy_ref[i, j]
        end
    end
    return sqrt(num / den)
end

function max_mag_masked(fx, fy, mask)
    N = size(fx, 1)
    out = 0.0
    @inbounds for j in 1:N, i in 1:N
        if mask(i, j, N)
            out = max(out, hypot(fx[i, j], fy[i, j]))
        end
    end
    return out
end

function convergence_order(coarse, fine)
    return coarse > 0.0 && fine > 0.0 ? log2(coarse / fine) : NaN
end

function bsd_v2_force(N, ux, uy, is_solid, bc; apply_wall_correction=false)
    dx = inv(N)
    dy = inv(N)
    dudx = zeros(Float64, N, N)
    dudy = zeros(Float64, N, N)
    dvdx = zeros(Float64, N, N)
    dvdy = zeros(Float64, N, N)
    Kraken.fvfd_velocity_gradient_2d!(
        dudx, dudy, dvdx, dvdy, ux, uy, is_solid, dx, dy, bc; sync=true,
    )
    if apply_wall_correction
        u_lid_profile = [
            U0 * sin(TWO_PI * (i - 0.5) / N) * cos(TWO_PI * (N - 0.5) / N)
            for i in 1:N
        ]
        Kraken._logfv_cavity_apply_wall_gradient_correction!(
            dudx, dudy, dvdx, dvdy, ux, uy, u_lid_profile, dx, dy;
            skip_top_corners=false, sync=true,
        )
    end
    tau_bsd_xx = zeros(Float64, N, N)
    tau_bsd_xy = zeros(Float64, N, N)
    tau_bsd_yy = zeros(Float64, N, N)
    Kraken.logfv_bsd_stress_from_gradient_2d!(
        tau_bsd_xx, tau_bsd_xy, tau_bsd_yy,
        dudx, dudy, dvdx, dvdy, ZETA * NU_P; sync=true,
    )
    return polymer_force_from_tau(N, tau_bsd_xx, tau_bsd_xy, tau_bsd_yy, is_solid, bc)
end

function run_L0(N)
    ux, uy = tg_fields(N)
    tauxx, tauxy, tauyy = tau_p_fields(N)
    is_solid = falses(N, N)
    fx_poly, fy_poly = polymer_force_from_tau(N, tauxx, tauxy, tauyy, is_solid, periodic_bc())
    fx_ref = zeros(Float64, N, N)
    fy_ref = zeros(Float64, N, N)
    fill_force_ref!(fx_ref, fy_ref, ux, uy, NU_P)
    return (N=N, rel=rel_l2(fx_poly, fy_poly, fx_ref, fy_ref))
end

function run_L1(N)
    ux, uy = tg_fields(N)
    tauxx, tauxy, tauyy = tau_p_fields(N)
    is_solid = falses(N, N)
    bc = periodic_bc()
    dx = inv(N)
    dy = inv(N)
    fx_poly, fy_poly = polymer_force_from_tau(N, tauxx, tauxy, tauyy, is_solid, bc)
    fx_ref = zeros(Float64, N, N)
    fy_ref = zeros(Float64, N, N)
    fill_force_ref!(fx_ref, fy_ref, ux, uy, (1.0 - ZETA) * NU_P)

    fx_total_fd = zeros(Float64, N, N)
    fy_total_fd = zeros(Float64, N, N)
    Kraken.logfv_bsd_correct_force_bc_aware_2d!(
        fx_total_fd, fy_total_fd, fx_poly, fy_poly, ux, uy,
        is_solid, ZETA, NU_P, dx, dy, bc; sync=true,
    )

    fx_bsd_v2, fy_bsd_v2 = bsd_v2_force(N, ux, uy, is_solid, bc)
    fx_total_v2 = fx_poly .- fx_bsd_v2
    fy_total_v2 = fy_poly .- fy_bsd_v2

    return (
        N=N,
        fd=rel_l2(fx_total_fd, fy_total_fd, fx_ref, fy_ref),
        fd_v2=rel_l2(fx_total_v2, fy_total_v2, fx_ref, fy_ref),
        delta=rel_l2_delta(fx_total_fd, fy_total_fd, fx_total_v2, fy_total_v2, fx_ref, fy_ref),
    )
end

function run_L2_kind_fd(N, ux, uy, is_solid, bc, fx_ref, fy_ref)
    dx = inv(N)
    dy = inv(N)
    tauxx, tauxy, tauyy = tau_p_fields(N)
    fx_poly, fy_poly = polymer_force_from_tau(N, tauxx, tauxy, tauyy, is_solid, bc)
    fx_total = zeros(Float64, N, N)
    fy_total = zeros(Float64, N, N)
    Kraken.logfv_bsd_correct_force_bc_aware_2d!(
        fx_total, fy_total, fx_poly, fy_poly, ux, uy,
        is_solid, ZETA, NU_P, dx, dy, bc; sync=true,
    )
    fx_bsd = fx_poly .- fx_total
    fy_bsd = fy_poly .- fy_total
    return (
        interior=rel_l2_masked(fx_total, fy_total, fx_ref, fy_ref, interior_mask),
        wall=rel_l2_masked(fx_total, fy_total, fx_ref, fy_ref, wall_band_mask),
        ratio=max_mag_masked(fx_bsd, fy_bsd, wall_band_mask) /
              max_mag_masked(fx_bsd, fy_bsd, interior_mask),
    )
end

function run_L2_kind_v2(N, ux, uy, is_solid, bc, fx_ref, fy_ref)
    tauxx, tauxy, tauyy = tau_p_fields(N)
    fx_poly, fy_poly = polymer_force_from_tau(N, tauxx, tauxy, tauyy, is_solid, bc)
    fx_bsd, fy_bsd = bsd_v2_force(N, ux, uy, is_solid, bc; apply_wall_correction=true)
    fx_total = fx_poly .- fx_bsd
    fy_total = fy_poly .- fy_bsd
    return (
        interior=rel_l2_masked(fx_total, fy_total, fx_ref, fy_ref, interior_mask),
        wall=rel_l2_masked(fx_total, fy_total, fx_ref, fy_ref, wall_band_mask),
        ratio=max_mag_masked(fx_bsd, fy_bsd, wall_band_mask) /
              max_mag_masked(fx_bsd, fy_bsd, interior_mask),
    )
end

function run_L2(N)
    ux, uy = tg_fields(N)
    is_solid = falses(N, N)
    bc = Kraken.fvfd_wallxwally_bcspec_2d()
    fx_ref = zeros(Float64, N, N)
    fy_ref = zeros(Float64, N, N)
    fill_force_ref!(fx_ref, fy_ref, ux, uy, (1.0 - ZETA) * NU_P)
    return (N=N, fd=run_L2_kind_fd(N, ux, uy, is_solid, bc, fx_ref, fy_ref),
            fd_v2=run_L2_kind_v2(N, ux, uy, is_solid, bc, fx_ref, fy_ref))
end

function choose_verdict(l1_by_n, l2)
    fd64 = l1_by_n[64].fd
    v264 = l1_by_n[64].fd_v2
    periodic = v264 <= fd64 ? ":fd_v2 gives the tighter periodic cancellation" :
               ":fd gives the tighter periodic cancellation"
    wall = l2.fd_v2.ratio <= l2.fd.ratio ? ":fd_v2 has the lower wall-band BSD spike" :
           ":fd has the lower wall-band BSD spike"
    return string(
        periodic, ", while ", wall,
        "; the M17 implication is to favor Option A D_uncorrected wall handling ",
        "unless a separate kinetic default is validated.",
    )
end

function build_table(l0, l1, l2)
    l0_order = convergence_order(l0[64].rel, l0[128].rel)
    fd_order = convergence_order(l1[64].fd, l1[128].fd)
    v2_order = convergence_order(l1[64].fd_v2, l1[128].fd_v2)
    verdict = choose_verdict(l1, l2)
    lines = String[]
    push!(lines, "=========================================================")
    push!(lines, "BSD analytical ladder - Taylor-Green vortex, CPU F64")
    push!(lines, "nu_p = 0.1, zeta = 0.75, U_0 = 1.0")
    push!(lines, "=========================================================")
    push!(lines, "")
    push!(lines, "L0 - div(tau_p) vs nu_p * lap(U) (periodic, no walls)")
    push!(lines, @sprintf("  N=%-3d  rel_L2 = %.3e", 32, l0[32].rel))
    push!(lines, @sprintf("  N=%-3d  rel_L2 = %.3e", 64, l0[64].rel))
    push!(lines, @sprintf("  N=%-3d  rel_L2 = %.3e    (order ~= %.2f)",
                          128, l0[128].rel, l0_order))
    push!(lines, "")
    push!(lines, "L1 - F_total = F_poly - F_BSD vs (1 - zeta) * nu_p * lap(U) (periodic)")
    push!(lines, @sprintf("  kind=:fd     N=32 rel_L2 = %.3e", l1[32].fd))
    push!(lines, @sprintf("               N=64 rel_L2 = %.3e  (order ~= %.2f)",
                          l1[64].fd, fd_order))
    push!(lines, @sprintf("               N=128 rel_L2 = %.3e", l1[128].fd))
    push!(lines, @sprintf("  kind=:fd_v2  N=32 rel_L2 = %.3e", l1[32].fd_v2))
    push!(lines, @sprintf("               N=64 rel_L2 = %.3e  (order ~= %.2f)",
                          l1[64].fd_v2, v2_order))
    push!(lines, @sprintf("               N=128 rel_L2 = %.3e", l1[128].fd_v2))
    push!(lines, @sprintf("  Delta(:fd vs :fd_v2)  N=64 rel_L2 = %.3e", l1[64].delta))
    push!(lines, "")
    push!(lines, "L2 - F_total at walls (closed box, lid profile matches TG)")
    push!(lines, @sprintf("  kind=:fd     interior rel_L2 = %.2e | wall rel_L2 = %.2e |",
                          l2.fd.interior, l2.fd.wall))
    push!(lines, @sprintf("               max|F_BSD| ratio wall/interior = %.2e",
                          l2.fd.ratio))
    push!(lines, @sprintf("  kind=:fd_v2  interior rel_L2 = %.2e | wall rel_L2 = %.2e |",
                          l2.fd_v2.interior, l2.fd_v2.wall))
    push!(lines, @sprintf("               max|F_BSD| ratio wall/interior = %.2e",
                          l2.fd_v2.ratio))
    push!(lines, "")
    push!(lines, "=========================================================")
    push!(lines, "VERDICT: $(verdict)")
    push!(lines, "=========================================================")
    return join(lines, "\n") * "\n"
end

function build_markdown(table, l2)
    interpretation = string(
        "At N=$(l2.N), the periodic ladder identifies which BSD cancellation ",
        "strategy matches the analytical Newtonian-limit force, while the closed-box ",
        "wall band quantifies the penalty from applying cavity wall gradients to the ",
        "BSD stress. The wall/interior BSD maxima are ",
        @sprintf("%.2e", l2.fd.ratio), " for :fd and ",
        @sprintf("%.2e", l2.fd_v2.ratio), " for :fd_v2, so the M17 architecture ",
        "should keep the BSD stabilizer on an uncorrected D path at walls unless a ",
        "separately validated kinetic default replaces it.",
    )
    return string(
        "# BSD Analytical Ladder - 2026-05-17\n\n",
        "Mission: M17-canary-analytical\n\n",
        "Params: U_0 = 1.0, nu_p = 0.1, zeta = 0.75, N = (32, 64, 128).\n\n",
        "Taylor-Green velocity: `Ux = U_0 sin(2*pi*x) cos(2*pi*y)`, ",
        "`Uy = -U_0 cos(2*pi*x) sin(2*pi*y)` on cell centers of `[0, 1]^2`. ",
        "The references are `lap(U) = -8*pi^2 U`, ",
        "`tau_p = 2*nu_p*D`, `F_poly = nu_p*lap(U)`, and ",
        "`F_total = (1 - zeta)*nu_p*lap(U)`.\n\n",
        "```text\n", table, "```\n\n",
        "Interpretation: ", interpretation, "\n",
    )
end

function main()
    l0 = Dict(N => run_L0(N) for N in NS)
    l1 = Dict(N => run_L1(N) for N in NS)
    l2 = run_L2(L2_N)
    table = build_table(l0, l1, l2)
    print(table)
    write(OUT_MD, build_markdown(table, l2))
    return nothing
end

main()
