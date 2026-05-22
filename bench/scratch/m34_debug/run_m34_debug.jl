#!/usr/bin/env julia
# M34 debug — Cd decomposition + NaN spatial fingerprint
# ----------------------------------------------------------------------------
# Mission: localize the bug in :bouzidi_fl_twopass that makes
#   - R=30 Wi=0.1 finite but Cd=117.59 instead of ~111 expected
#   - R=40 Wi=0.1 + R={30,40} Wi=1.0 all NaN
#
# Adapted from bench/scratch/m32_phase4_wi1_walldecomp/run_walldecomp.jl
# (Cd wall decomposition on the :idx frame, per [[feedback_wall_ring_idx_frame]]).

using Printf
using Serialization
using Statistics

const M34DIR = abspath(joinpath(@__DIR__, "..", "..", "..",
    "tmp", "m34_aqua_results", "matrix"))
const BASELINE = abspath(joinpath(@__DIR__, "..", "..", "..",
    "tmp", "m30_rho_metal", "run01"))

const M34_FILES = Dict(
    "R30_Wi0p1" => joinpath(M34DIR,
        "cyl_bigsweep_v2_beta0p59_wi0p1_re1_R30_bsd1_Lup15_Ldn15_eg0_ea0_ef0_ed0_geomqwall_fields.jls"),
    "R40_Wi0p1" => joinpath(M34DIR,
        "cyl_bigsweep_v2_beta0p59_wi0p1_re1_R40_bsd1_Lup15_Ldn15_eg0_ea0_ef0_ed0_geomqwall_fields.jls"),
    "R30_Wi1"   => joinpath(M34DIR,
        "cyl_bigsweep_v2_beta0p59_wi1_re1_R30_bsd1_Lup15_Ldn15_eg0_ea0_ef0_ed0_geomqwall_fields.jls"),
    "R40_Wi1"   => joinpath(M34DIR,
        "cyl_bigsweep_v2_beta0p59_wi1_re1_R40_bsd1_Lup15_Ldn15_eg0_ea0_ef0_ed0_geomqwall_fields.jls"),
)
const BASELINE_FILE = joinpath(BASELINE,
    "cyl_bigsweep_v2_beta0p59_wi1_re1_R30_bsd1_Lup15_Ldn15_eg0_ea0_ef0_ed0_geomqwall_fields.jls")

function load_snap(path)
    isfile(path) || error("missing $path")
    open(path) do io
        return deserialize(io)
    end
end

# ---- inspect schema -----------------------------------------------------------
function describe_snap(label, snap)
    println("\n=== $label ===")
    pn = propertynames(snap)
    println("propertynames: ", pn)
    for f in pn
        v = getproperty(snap, f)
        sz = try size(v); catch; nothing; end
        et = try eltype(v); catch; typeof(v); end
        if sz === nothing
            println("  $f :: $(typeof(v)) = $v")
        else
            isfin = if v isa AbstractArray && et <: Number
                anyna = any(isnan, v)
                anyin = any(!isfinite, v)
                "any(NaN)=$anyna, any(!finite)=$anyin"
            else
                ""
            end
            println("  $f :: $(typeof(v)) size=$sz  $isfin")
        end
    end
end

# ---- Cd wall decomposition (theta-binned, :idx frame) -------------------------
function lattice_velocity_gradient(ux, uy, solid)
    Nx, Ny = size(ux)
    T = eltype(ux)
    dudx = zeros(T, Nx, Ny); dudy = zeros(T, Nx, Ny)
    dvdx = zeros(T, Nx, Ny); dvdy = zeros(T, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        solid[i, j] && continue
        west_ok = (i > 1) && !solid[i-1, j]
        east_ok = (i < Nx) && !solid[i+1, j]
        if west_ok && east_ok
            dudx[i, j] = T(0.5) * (ux[i+1, j] - ux[i-1, j])
            dvdx[i, j] = T(0.5) * (uy[i+1, j] - uy[i-1, j])
        elseif east_ok
            dudx[i, j] = ux[i+1, j] - ux[i, j]
            dvdx[i, j] = uy[i+1, j] - uy[i, j]
        elseif west_ok
            dudx[i, j] = ux[i, j] - ux[i-1, j]
            dvdx[i, j] = uy[i, j] - uy[i-1, j]
        end
        south_ok = (j > 1) && !solid[i, j-1]
        north_ok = (j < Ny) && !solid[i, j+1]
        if south_ok && north_ok
            dudy[i, j] = T(0.5) * (ux[i, j+1] - ux[i, j-1])
            dvdy[i, j] = T(0.5) * (uy[i, j+1] - uy[i, j-1])
        elseif north_ok
            dudy[i, j] = ux[i, j+1] - ux[i, j]
            dvdy[i, j] = uy[i, j+1] - uy[i, j]
        elseif south_ok
            dudy[i, j] = ux[i, j] - ux[i, j-1]
            dvdy[i, j] = uy[i, j] - uy[i, j-1]
        end
    end
    return dudx, dudy, dvdx, dvdy
end

function kraken_wall_decomp(snap; N_az::Int=36, label::String="")
    Nx = snap.Nx; Ny = snap.Ny
    cx_phys = Float64(snap.cylinder_x_lbm)
    cy_phys = Float64(snap.cylinder_y_lbm)
    R_lu = Float64(snap.radius_lbm)
    cx_lu = cx_phys + 1.0
    cy_lu = cy_phys + 1.0
    u_mean = Float64(snap.u_mean)
    solid = snap.is_solid
    rho = snap.rho
    ux = snap.ux; uy = snap.uy
    txx_p = snap.tauxx; txy_p = snap.tauxy; tyy_p = snap.tauyy
    mu_s = Float64(snap.nu_s)
    cs2 = 1.0/3.0
    norm = u_mean^2 * R_lu
    dtheta = 2π / N_az
    arc_dl = R_lu * dtheta
    dudx, dudy, dvdx, dvdy = lattice_velocity_gradient(Float64.(ux), Float64.(uy), solid)

    bins_pres = zeros(Float64, N_az)
    bins_solv = zeros(Float64, N_az)
    bins_poly = zeros(Float64, N_az)
    counts = zeros(Int, N_az)

    n_nan_ring = 0
    @inbounds for j in 1:Ny, i in 1:Nx
        solid[i, j] && continue
        has_solid_n = false
        for (di, dj) in ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1))
            ii = i + di; jj = j + dj
            if 1 <= ii <= Nx && 1 <= jj <= Ny && solid[ii, jj]
                has_solid_n = true; break
            end
        end
        has_solid_n || continue
        dx = Float64(i) - cx_lu
        dy = Float64(j) - cy_lu
        r = hypot(dx, dy)
        r > 0 || continue
        theta = atan(dy, dx)
        nx_o = dx / r; ny_o = dy / r
        rho_ij = Float64(rho[i, j])
        ux_ij = Float64(ux[i, j])
        # NaN check on ring cell
        if !isfinite(rho_ij) || !isfinite(ux_ij)
            n_nan_ring += 1
            continue
        end
        p_lbm = cs2 * (rho_ij - 1.0)
        tx_pres = -p_lbm * nx_o
        sxx_solv = 2.0 * mu_s * dudx[i, j]
        sxy_solv = mu_s * (dudy[i, j] + dvdx[i, j])
        tx_solv = sxx_solv * nx_o + sxy_solv * ny_o
        txx_pij = Float64(txx_p[i, j]); txy_pij = Float64(txy_p[i, j])
        tx_poly = (isfinite(txx_pij) ? txx_pij : 0.0) * nx_o +
                  (isfinite(txy_pij) ? txy_pij : 0.0) * ny_o
        bin = mod(floor(Int, (theta + π) / dtheta), N_az) + 1
        bins_pres[bin] += tx_pres
        bins_solv[bin] += tx_solv
        bins_poly[bin] += tx_poly
        counts[bin] += 1
    end
    dCd_pres = zeros(N_az); dCd_solv = zeros(N_az); dCd_poly = zeros(N_az)
    for k in 1:N_az
        if counts[k] > 0
            dCd_pres[k] = (bins_pres[k]/counts[k]) * arc_dl / norm
            dCd_solv[k] = (bins_solv[k]/counts[k]) * arc_dl / norm
            dCd_poly[k] = (bins_poly[k]/counts[k]) * arc_dl / norm
        end
    end
    theta_centres = [-π + (k - 0.5)*dtheta for k in 1:N_az]
    return (;
        label, N_az, theta_centres, counts, dCd_pres, dCd_solv, dCd_poly,
        Cd_pres = sum(dCd_pres),
        Cd_solv = sum(dCd_solv),
        Cd_poly = sum(dCd_poly),
        Cd_total = sum(dCd_pres) + sum(dCd_solv) + sum(dCd_poly),
        n_nan_ring,
    )
end

# ---- NaN spatial fingerprint --------------------------------------------------
function nan_fingerprint(snap; label::String="")
    Nx = snap.Nx; Ny = snap.Ny
    cx_phys = Float64(snap.cylinder_x_lbm)
    cy_phys = Float64(snap.cylinder_y_lbm)
    R_lu = Float64(snap.radius_lbm)
    cx_lu = cx_phys + 1.0; cy_lu = cy_phys + 1.0
    fields = (:rho, :ux, :uy, :tauxx, :tauxy, :tauyy)
    println("\n=== NaN fingerprint $label ===")
    for f in fields
        if hasproperty(snap, f)
            v = getproperty(snap, f)
            if v isa AbstractArray
                nn = count(isnan, v)
                ni = count(!isfinite, v) - nn
                println(@sprintf("  %-6s : NaN=%d, Inf=%d, finite_count=%d / %d",
                    f, nn, ni, count(isfinite, v), length(v)))
                if nn > 0
                    # spatial map: list cells, compute (r, θ) bins
                    inds = findall(isnan, v)
                    rs = Float64[]; thetas = Float64[]
                    for I in inds
                        i, j = Tuple(I)
                        dx = Float64(i) - cx_lu; dy = Float64(j) - cy_lu
                        r = hypot(dx, dy)
                        push!(rs, r)
                        push!(thetas, atan(dy, dx))
                    end
                    n_in_ring = count(r -> r <= 1.5*R_lu, rs)
                    n_in_wake = count(t -> abs(t) < π/4, thetas)
                    n_in_front = count(t -> abs(t) > 3π/4, thetas)
                    n_in_shoulder = length(thetas) - n_in_wake - n_in_front
                    @printf("    r range [%.2f, %.2f] (R=%.1f) ; in-ring (r≤1.5R): %d/%d\n",
                        minimum(rs), maximum(rs), R_lu, n_in_ring, length(rs))
                    @printf("    θ bins: front (|θ|>3π/4)=%d, shoulder=%d, wake (|θ|<π/4)=%d\n",
                        n_in_front, n_in_shoulder, n_in_wake)
                    # first few cells
                    show_n = min(5, length(inds))
                    println("    first $show_n cells (i, j, dx, dy, r/R, θ):")
                    for k in 1:show_n
                        I = inds[k]; i, j = Tuple(I)
                        dx = Float64(i) - cx_lu; dy = Float64(j) - cy_lu
                        r = hypot(dx, dy); θ = atan(dy, dx)
                        @printf("      (%d, %d)  dx=%+0.2f dy=%+0.2f  r/R=%.3f  θ=%+.2f rad (%.0f°)\n",
                            i, j, dx, dy, r/R_lu, θ, θ*180/π)
                    end
                end
            end
        end
    end
end

# ---- Summary printer ---------------------------------------------------------
function print_decomp(d; gap=nothing)
    println("\n--- $(d.label) ---")
    if d.n_nan_ring > 0
        println("  NaN ring cells skipped: $(d.n_nan_ring)")
    end
    @printf("  Cd_pres = %+9.4f\n", d.Cd_pres)
    @printf("  Cd_solv = %+9.4f\n", d.Cd_solv)
    @printf("  Cd_poly = %+9.4f\n", d.Cd_poly)
    @printf("  Cd_tot  = %+9.4f\n", d.Cd_total)
    # 3-region split
    front = falses(d.N_az); shoulder = falses(d.N_az); wake = falses(d.N_az)
    for (k, θ) in enumerate(d.theta_centres)
        if abs(θ) < π/4
            wake[k] = true
        elseif abs(θ) > 3π/4
            front[k] = true
        else
            shoulder[k] = true
        end
    end
    println("  Region split (3-bucket):")
    @printf("    front   Cd_pres=%+7.3f Cd_solv=%+7.3f Cd_poly=%+7.3f\n",
        sum(d.dCd_pres[front]), sum(d.dCd_solv[front]), sum(d.dCd_poly[front]))
    @printf("    should  Cd_pres=%+7.3f Cd_solv=%+7.3f Cd_poly=%+7.3f\n",
        sum(d.dCd_pres[shoulder]), sum(d.dCd_solv[shoulder]), sum(d.dCd_poly[shoulder]))
    @printf("    wake    Cd_pres=%+7.3f Cd_solv=%+7.3f Cd_poly=%+7.3f\n",
        sum(d.dCd_pres[wake]), sum(d.dCd_solv[wake]), sum(d.dCd_poly[wake]))
end

# ---- Main --------------------------------------------------------------------
println("="^72)
println("M34 debug — finite case Cd decomposition + NaN spatial fingerprint")
println("="^72)

# Inspect M34 R30 Wi0.1 (finite) first
snap_m34_fin = load_snap(M34_FILES["R30_Wi0p1"])
describe_snap("M34 R=30 Wi=0.1 (FINITE, Cd=117.59)", snap_m34_fin)
d_m34_fin = kraken_wall_decomp(snap_m34_fin; label="M34 R30 Wi0.1")
print_decomp(d_m34_fin)

# Compare with halfwayBB baseline (Metal F32 R=30 Wi=1.0)
println("\nLoading halfwayBB baseline (Metal F32) R=30 Wi=1.0 ...")
snap_bb = load_snap(BASELINE_FILE)
d_bb = kraken_wall_decomp(snap_bb; label="halfwayBB R30 Wi1.0 (baseline)")
print_decomp(d_bb)

# NaN cases
for key in ("R40_Wi0p1", "R30_Wi1", "R40_Wi1")
    snap = load_snap(M34_FILES[key])
    nan_fingerprint(snap; label="M34 $key")
end

println("\n=== END ===")
