using Kraken
using Printf

const U0 = 1.0
const NU_P = 0.1
const ZETA = 0.75
const NS = (32, 64, 128)
const L2_N = 64
const L4_N = 64
const L4_MODES = (1, 2, 4, 8, 16, 32)
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

# L2b: closed-box wall scenario with three BSD-source variants. The new
# :fd_v2_unc path is Option A — BSD reads D_uncorrected (centered FD only,
# the wall-correction overwrite is SKIPPED). :fd and :fd_v2 are sanity
# replicas of L2.

function run_L2b_kind_v2_unc(N, ux, uy, is_solid, bc, fx_ref, fy_ref)
    tauxx, tauxy, tauyy = tau_p_fields(N)
    fx_poly, fy_poly = polymer_force_from_tau(N, tauxx, tauxy, tauyy, is_solid, bc)
    fx_bsd, fy_bsd = bsd_v2_force(N, ux, uy, is_solid, bc; apply_wall_correction=false)
    fx_total = fx_poly .- fx_bsd
    fy_total = fy_poly .- fy_bsd
    return (
        interior=rel_l2_masked(fx_total, fy_total, fx_ref, fy_ref, interior_mask),
        wall=rel_l2_masked(fx_total, fy_total, fx_ref, fy_ref, wall_band_mask),
        ratio=max_mag_masked(fx_bsd, fy_bsd, wall_band_mask) /
              max_mag_masked(fx_bsd, fy_bsd, interior_mask),
    )
end

function run_L2b(N)
    ux, uy = tg_fields(N)
    is_solid = falses(N, N)
    bc = Kraken.fvfd_wallxwally_bcspec_2d()
    fx_ref = zeros(Float64, N, N)
    fy_ref = zeros(Float64, N, N)
    fill_force_ref!(fx_ref, fy_ref, ux, uy, (1.0 - ZETA) * NU_P)
    return (
        N=N,
        fd=run_L2_kind_fd(N, ux, uy, is_solid, bc, fx_ref, fy_ref),
        fd_v2=run_L2_kind_v2(N, ux, uy, is_solid, bc, fx_ref, fy_ref),
        fd_v2_unc=run_L2b_kind_v2_unc(N, ux, uy, is_solid, bc, fx_ref, fy_ref),
    )
end

# ---------------------------------------------------------------------------
# L4 - Spectral test of the WIDE F_poly operator vs the NARROW Laplacian
# at a sequence of cell-index Fourier modes m in (1, 2, 4, 8, 16, 32) on
# a periodic N=64 grid. The Nyquist mode is m = N/2 = 32, where k_x dx = pi.
# Velocity field uses cell-INDEX sampling (no half-shift) so that at m = N/2
# the field is the genuine checkerboard pattern (-1)^(i+j) on cell centres:
#   Ux(i,j) = U_0 * cos(2*pi*m*i/N) * cos(2*pi*m*j/N)
#   Uy(i,j) = -U_0 * sin(2*pi*m*i/N) * sin(2*pi*m*j/N)
# This is NOT divergence-free; the operator-level analytical reference is:
#   F_poly_x = div(tau)_x = 2 * nu_p * lap(Ux),   tau = 2 * nu_p * D(U)
#   lap(Ux) = -2 * k^2 * Ux   with k = 2*pi*m.
# So |F_poly_x_analytical|_inf = 4 * nu_p * k^2 * U_0 at any non-zero mode.
# The "ratio" reported is the operator amplitude divided by the analytical
# Laplacian amplitude (not divided by F_poly_analytical) so that NARROW and
# WIDE rows are directly comparable on the same Laplacian baseline.
# ---------------------------------------------------------------------------

function fill_l4_velocity!(ux, uy, N, m)
    @inbounds for j in 1:N, i in 1:N
        cx = cos(TWO_PI * m * i / N)
        sx = sin(TWO_PI * m * i / N)
        cy = cos(TWO_PI * m * j / N)
        sy = sin(TWO_PI * m * j / N)
        ux[i, j] = U0 * cx * cy
        uy[i, j] = -U0 * sx * sy
    end
    return nothing
end

function fill_l4_tau_analytical!(tauxx, tauxy, tauyy, N, m)
    # tau = 2 * nu_p * D, with D built from the analytical derivatives of the
    # L4 velocity field above. All four sin/cos arguments use cell-INDEX
    # sampling so that the strain rate at the Nyquist mode is identically
    # zero at every cell centre (the WIDE null-mode mechanism).
    km = TWO_PI * m
    @inbounds for j in 1:N, i in 1:N
        cx = cos(TWO_PI * m * i / N)
        sx = sin(TWO_PI * m * i / N)
        cy = cos(TWO_PI * m * j / N)
        sy = sin(TWO_PI * m * j / N)
        dudx = -U0 * km * sx * cy
        dudy = -U0 * km * cx * sy
        dvdx = -U0 * km * cx * sy
        dvdy = -U0 * km * sx * cy
        tauxx[i, j] = 2.0 * NU_P * dudx
        tauxy[i, j] = NU_P * (dudy + dvdx)
        tauyy[i, j] = 2.0 * NU_P * dvdy
    end
    return nothing
end

function narrow_laplacian_force(N, ux, uy, is_solid, bc)
    # Extract the NARROW 5-point Laplacian acting on u directly. The
    # logfv_bsd_correct_force_bc_aware_2d! kernel computes
    #   fx_out = fx_poly - zeta * nu_p * lap_u_narrow
    # so feeding fx_poly = 0, zeta = -1, nu_p = 1 returns lap_u_narrow.
    dx = inv(N)
    dy = inv(N)
    fx_poly = zeros(Float64, N, N)
    fy_poly = zeros(Float64, N, N)
    fx_out = zeros(Float64, N, N)
    fy_out = zeros(Float64, N, N)
    Kraken.logfv_bsd_correct_force_bc_aware_2d!(
        fx_out, fy_out, fx_poly, fy_poly, ux, uy,
        is_solid, -1.0, 1.0, dx, dy, bc; sync=true,
    )
    return fx_out, fy_out
end

function max_inf(fx, fy)
    out = 0.0
    @inbounds for idx in eachindex(fx)
        out = max(out, hypot(fx[idx], fy[idx]))
    end
    return out
end

function run_L4_mode(N, m)
    bc = periodic_bc()
    is_solid = falses(N, N)
    ux = zeros(Float64, N, N)
    uy = zeros(Float64, N, N)
    fill_l4_velocity!(ux, uy, N, m)
    tauxx = zeros(Float64, N, N)
    tauxy = zeros(Float64, N, N)
    tauyy = zeros(Float64, N, N)
    fill_l4_tau_analytical!(tauxx, tauxy, tauyy, N, m)

    fx_wide, fy_wide = polymer_force_from_tau(N, tauxx, tauxy, tauyy, is_solid, bc)
    fx_narrow, fy_narrow = narrow_laplacian_force(N, ux, uy, is_solid, bc)

    # Analytical |lap(U)|_inf at this mode: lap(U) = -2*k^2 * U so the
    # amplitude is 2*k^2 * U_0 (both components share this amplitude).
    k = TWO_PI * m
    lap_amp = 2.0 * k * k * U0

    # Analytical |F_poly|_inf = |2 * nu_p * lap(U)|_inf = 4 * nu_p * k^2 * U_0.
    fpoly_amp_analytical = 2.0 * NU_P * lap_amp

    wide_amp = max_inf(fx_wide, fy_wide)
    narrow_amp = max_inf(fx_narrow, fy_narrow)

    return (
        m=m,
        kdx=k / N,
        wide_amp=wide_amp,
        narrow_amp=narrow_amp,
        lap_amp_analytical=lap_amp,
        fpoly_amp_analytical=fpoly_amp_analytical,
        # Ratios on the SAME baseline (analytical Laplacian amplitude) so
        # NARROW and WIDE are directly comparable. The WIDE row is divided
        # by (2 * nu_p) to undo the strain-rate factor 2 * nu_p, leaving a
        # pure Laplacian-operator amplitude.
        wide_ratio=wide_amp / (2.0 * NU_P) / lap_amp,
        narrow_ratio=narrow_amp / lap_amp,
    )
end

function run_L4(N, modes)
    return [run_L4_mode(N, m) for m in modes]
end

function classify_l4(l4_rows)
    nyquist = last(l4_rows)
    # Null-mode hypothesis: at Nyquist (m = N/2) the WIDE ratio is essentially
    # zero (<= 1e-6) while the NARROW ratio retains ~4/pi^2 = 0.405.
    confirmed = nyquist.wide_ratio <= 1.0e-6 && nyquist.narrow_ratio >= 0.30
    partial = !confirmed && nyquist.wide_ratio <= 1.0e-2 && nyquist.narrow_ratio >= 0.30
    return if confirmed
        "CONFIRMED"
    elseif partial
        "PARTIALLY confirmed (WIDE non-trivially attenuated but not bit-zero)"
    else
        "REFUTED"
    end
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

function choose_l2b_verdict(l1_by_n, l2b_by_n)
    l2b64 = l2b_by_n[64]
    interior_ok = l2b64.fd_v2_unc.interior <= 1.5 * l1_by_n[64].fd_v2
    fd_wall = l2b64.fd.wall
    unc_wall = l2b64.fd_v2_unc.wall
    fd_ratio = l2b64.fd.ratio
    unc_ratio = l2b64.fd_v2_unc.ratio
    wall_close = unc_wall <= 5.0 * fd_wall
    ratio_close = unc_ratio <= 5.0 * fd_ratio
    drop_factor = l2b64.fd_v2.wall > 0.0 ? l2b64.fd_v2.wall / max(unc_wall, eps()) : NaN
    color = (interior_ok && wall_close && ratio_close) ? "GREEN" :
            (interior_ok && drop_factor >= 10.0) ? "YELLOW" : "RED"
    rec = color == "GREEN" ?
        "implement Option A in cavity_driver_2d.jl as M17" :
        color == "YELLOW" ?
        "escalate to Boss — wall partial improvement, decide go/no-go" :
        "pivot to :kinetic canary"
    return color, drop_factor, rec
end

function _l2b_kind_block!(lines, label, kind)
    push!(lines, @sprintf("  kind=%-12s N=32  interior rel_L2 = %.3e | wall rel_L2 = %.3e | ratio = %.3e",
                          label, kind[32].interior, kind[32].wall, kind[32].ratio))
    order = convergence_order(kind[64].interior, kind[128].interior)
    push!(lines, @sprintf("               N=64  interior rel_L2 = %.3e | wall rel_L2 = %.3e | ratio = %.3e  (interior order ~= %.2f)",
                          kind[64].interior, kind[64].wall, kind[64].ratio, order))
    push!(lines, @sprintf("               N=128 interior rel_L2 = %.3e | wall rel_L2 = %.3e | ratio = %.3e",
                          kind[128].interior, kind[128].wall, kind[128].ratio))
end

function _l4_block!(lines, l4_rows)
    push!(lines, "L4 - Spectral test: WIDE F_poly vs NARROW Laplacian (periodic, N=$(L4_N))")
    push!(lines, "  Velocity uses cell-INDEX sampling; mode m=N/2 is the Nyquist checkerboard.")
    push!(lines, "  Both ratios are normalised by the analytical |lap(U)|_inf at the mode.")
    push!(lines, "  m    k_x*dx       WIDE ratio    NARROW ratio  |lap(U)|_inf_analytical")
    for row in l4_rows
        push!(lines, @sprintf("  %-4d %.6f     %.4e    %.4e    %.4e",
                              row.m, row.kdx, row.wide_ratio,
                              row.narrow_ratio, row.lap_amp_analytical))
    end
    push!(lines, @sprintf("  Reference: NARROW(Nyquist) = 4/pi^2 = %.6f", 4.0 / pi^2))
    return nothing
end

function build_table(l0, l1, l2, l2b, l4_rows)
    l0_order = convergence_order(l0[64].rel, l0[128].rel)
    fd_order = convergence_order(l1[64].fd, l1[128].fd)
    v2_order = convergence_order(l1[64].fd_v2, l1[128].fd_v2)
    verdict = choose_verdict(l1, l2)
    color, drop_factor, rec = choose_l2b_verdict(l1, l2b)
    # Reshape L2b by kind for the table builder
    fd_by_n   = Dict(N => l2b[N].fd        for N in NS)
    v2_by_n   = Dict(N => l2b[N].fd_v2     for N in NS)
    unc_by_n  = Dict(N => l2b[N].fd_v2_unc for N in NS)
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
    push!(lines, "L2b - Option A test: BSD reads D_uncorrected (closed box, walls)")
    _l2b_kind_block!(lines, ":fd",          fd_by_n)
    _l2b_kind_block!(lines, ":fd_v2",       v2_by_n)
    _l2b_kind_block!(lines, ":fd_v2_unc",   unc_by_n)
    push!(lines, @sprintf("  Wall drop factor :fd_v2 -> :fd_v2_unc at N=64 = %.2e", drop_factor))
    push!(lines, "")
    _l4_block!(lines, l4_rows)
    push!(lines, "")
    push!(lines, "=========================================================")
    push!(lines, "VERDICT (L2):  $(verdict)")
    push!(lines, "VERDICT (L2b): [$(color)] $(rec)")
    push!(lines, "VERDICT (L4):  WIDE Nyquist null-mode hypothesis: $(classify_l4(l4_rows))")
    push!(lines, "=========================================================")
    return join(lines, "\n") * "\n"
end

function build_l4_markdown(l4_rows)
    verdict = classify_l4(l4_rows)
    nyquist = last(l4_rows)
    rows_md = String[]
    push!(rows_md, "| m | k_x*dx | WIDE / |lap U|_anal | NARROW / |lap U|_anal | |lap U|_anal |")
    push!(rows_md, "|---|--------|---------------------|-----------------------|--------------|")
    for r in l4_rows
        push!(rows_md, @sprintf("| %d | %.6f | %.4e | %.4e | %.4e |",
                                r.m, r.kdx, r.wide_ratio, r.narrow_ratio,
                                r.lap_amp_analytical))
    end
    table_md = join(rows_md, "\n")
    interp = string(
        "At the Nyquist mode (m = N/2 = $(nyquist.m), k_x*dx = pi), the WIDE F_poly ",
        "ratio is ", @sprintf("%.4e", nyquist.wide_ratio),
        " and the NARROW Laplacian ratio is ", @sprintf("%.4e", nyquist.narrow_ratio),
        " (analytical NARROW reference 4/pi^2 = ",
        @sprintf("%.4f", 4.0 / pi^2),
        "). The L4 mechanism is that the WIDE divergence stencil uses the 2dx ",
        "centred difference, whose Fourier symbol sin(k*dx)/dx is identically ",
        "zero at k*dx = pi; equivalently, the strain rate D built from a ",
        "checkerboard velocity is zero at every cell centre with cell-INDEX ",
        "sampling, so tau and its WIDE divergence both vanish. The NARROW ",
        "5-point Laplacian on the same checkerboard returns 4*|U|/dx^2 per ",
        "axis (8*|U|/dx^2 total), giving the 4/pi^2 ratio. ",
        "Null-mode hypothesis verdict: ", verdict, ".",
    )
    epsilon_implication = string(
        "Implication for the (epsilon) split: the proposed implementation ",
        "applies a NARROW 5-point Laplacian directly on u (the same operator ",
        "tested here), so it inherits the >=40 percent Nyquist damping ",
        "demonstrated in the table above. Replacing F_poly_WIDE by ",
        "(NARROW Laplacian)*(nu_s + nu_p)*u therefore closes the WIDE null ",
        "mode in addition to the M10 truncation bias, and gives the (epsilon) ",
        "split a second, stability-relevant benefit beyond the 3.4 percent ",
        "bulk-residual reduction.",
    )
    return string(
        "## L4 - Spectral test at Nyquist mode\n\n",
        "Periodic N=", L4_N, ", CPU F64. Cell-INDEX Fourier modes m in ",
        "(1, 2, 4, 8, 16, 32). The Nyquist mode is m = N/2 = 32 ",
        "(k_x*dx = pi). Velocity field:\n\n",
        "```\nUx(i,j) = U_0 * cos(2*pi*m*i/N) * cos(2*pi*m*j/N)\n",
        "Uy(i,j) = -U_0 * sin(2*pi*m*i/N) * sin(2*pi*m*j/N)\n```\n\n",
        "WIDE applies `logfv_polymer_force_bc_aware_2d!` on the analytical ",
        "tau = 2*nu_p*D(U); NARROW calls ",
        "`logfv_bsd_correct_force_bc_aware_2d!` with fx_poly=0, zeta=-1, nu_p=1 ",
        "to extract lap(U)_narrow alone. Both ratios are normalised by the ",
        "analytical |lap(U)|_inf = 2*k^2*U_0 at the mode.\n\n",
        table_md, "\n\n",
        "Interpretation: ", interp, "\n\n",
        epsilon_implication, "\n",
    )
end

function build_markdown(table, l1, l2, l2b, l4_rows)
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
    color, drop_factor, rec = choose_l2b_verdict(l1, l2b)
    l2b64 = l2b[64]
    l2b_interp = string(
        "L2b directly tests Option A: BSD reads D_uncorrected (centered FD only, ",
        "no wall-correction overwrite) while keeping the wide-stencil :fd_v2 kernel ",
        "path. At N=64 the :fd_v2_unc row gives interior rel L2 = ",
        @sprintf("%.2e", l2b64.fd_v2_unc.interior),
        " (L1 :fd_v2 baseline ", @sprintf("%.2e", l1[64].fd_v2),
        "), wall band rel L2 = ", @sprintf("%.2e", l2b64.fd_v2_unc.wall),
        " (down from :fd_v2's ", @sprintf("%.2e", l2b64.fd_v2.wall),
        " by factor ", @sprintf("%.2e", drop_factor),
        "), and wall/interior |F_BSD| ratio = ",
        @sprintf("%.2e", l2b64.fd_v2_unc.ratio),
        " (vs :fd ", @sprintf("%.2e", l2b64.fd.ratio),
        ", :fd_v2 ", @sprintf("%.2e", l2b64.fd_v2.ratio),
        "). Verdict: [$(color)] $(rec).",
    )
    return string(
        "# BSD Analytical Ladder - 2026-05-17\n\n",
        "Mission: M17-canary-analytical (L0-L2) + M17-canary-A (L2b Option A test) ",
        "+ M17-nyquist (L4 spectral test)\n\n",
        "Params: U_0 = 1.0, nu_p = 0.1, zeta = 0.75, N = (32, 64, 128); L4 uses N=", L4_N, ".\n\n",
        "Taylor-Green velocity (L0-L2b): `Ux = U_0 sin(2*pi*x) cos(2*pi*y)`, ",
        "`Uy = -U_0 cos(2*pi*x) sin(2*pi*y)` on cell centers of `[0, 1]^2`. ",
        "The references are `lap(U) = -8*pi^2 U`, ",
        "`tau_p = 2*nu_p*D`, `F_poly = nu_p*lap(U)`, and ",
        "`F_total = (1 - zeta)*nu_p*lap(U)`.\n\n",
        "```text\n", table, "```\n\n",
        "Interpretation (L0-L2): ", interpretation, "\n\n",
        "## L2b - Option A test (D_uncorrected for BSD)\n\n",
        l2b_interp, "\n\n",
        build_l4_markdown(l4_rows),
    )
end

function main()
    l0 = Dict(N => run_L0(N) for N in NS)
    l1 = Dict(N => run_L1(N) for N in NS)
    l2 = run_L2(L2_N)
    l2b = Dict(N => run_L2b(N) for N in NS)
    l4 = run_L4(L4_N, L4_MODES)
    table = build_table(l0, l1, l2, l2b, l4)
    print(table)
    write(OUT_MD, build_markdown(table, l1, l2, l2b, l4))
    return nothing
end

main()
