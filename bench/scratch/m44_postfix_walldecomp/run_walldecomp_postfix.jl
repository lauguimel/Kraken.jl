#!/usr/bin/env julia
# M44 / M45-B post-fix per-θ wall decomposition across R={30,40,50,60}
# at Wi=1 β=0.59. Reuses kraken_wall_decomp from M32 P4 D1 to localize
# WHERE the Cd_s decrease with R lives (front_pole vs shoulder vs wake).
#
# Inputs: tmp/m44_postfix_sweep/21827394.aqua/*_R{R}_*_fields.jls
# Output: bench/scratch/m44_postfix_walldecomp/M44P_kraken_R{R}_bins_idx.csv
#         + M44P_decomp_RvsR.csv (aggregated 3-region per R)
#         + M44P_summary.md

using Dates
using Printf
using Serialization

function region_of(theta::Float64)
    aθ = abs(theta)
    if aθ < π/4
        return :wake
    elseif aθ > 3π/4
        return :front_pole
    else
        return :shoulder
    end
end

function lattice_velocity_gradient(ux::AbstractMatrix, uy::AbstractMatrix,
                                    solid::AbstractMatrix{Bool})
    Nx, Ny = size(ux)
    dudx = zeros(Float64, Nx, Ny); dudy = zeros(Float64, Nx, Ny)
    dvdx = zeros(Float64, Nx, Ny); dvdy = zeros(Float64, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        solid[i, j] && continue
        west_ok = (i > 1)  && !solid[i-1, j]
        east_ok = (i < Nx) && !solid[i+1, j]
        if west_ok && east_ok
            dudx[i,j] = 0.5 * (ux[i+1,j] - ux[i-1,j])
            dvdx[i,j] = 0.5 * (uy[i+1,j] - uy[i-1,j])
        elseif east_ok
            dudx[i,j] = ux[i+1,j] - ux[i,j]
            dvdx[i,j] = uy[i+1,j] - uy[i,j]
        elseif west_ok
            dudx[i,j] = ux[i,j] - ux[i-1,j]
            dvdx[i,j] = uy[i,j] - uy[i-1,j]
        end
        south_ok = (j > 1)  && !solid[i, j-1]
        north_ok = (j < Ny) && !solid[i, j+1]
        if south_ok && north_ok
            dudy[i,j] = 0.5 * (ux[i,j+1] - ux[i,j-1])
            dvdy[i,j] = 0.5 * (uy[i,j+1] - uy[i,j-1])
        elseif north_ok
            dudy[i,j] = ux[i,j+1] - ux[i,j]
            dvdy[i,j] = uy[i,j+1] - uy[i,j]
        elseif south_ok
            dudy[i,j] = ux[i,j] - ux[i,j-1]
            dvdy[i,j] = uy[i,j] - uy[i,j-1]
        end
    end
    return dudx, dudy, dvdx, dvdy
end

function kraken_wall_decomp(snap; N_az::Int=36)
    Nx = snap.Nx; Ny = snap.Ny
    cx_phys = Float64(snap.cylinder_x_lbm)
    cy_phys = Float64(snap.cylinder_y_lbm)
    R_lu    = Float64(snap.radius_lbm)
    cx_lu = cx_phys + 1.0
    cy_lu = cy_phys + 1.0
    u_mean = Float64(snap.u_mean)
    solid  = snap.is_solid
    rho    = snap.rho
    ux = snap.ux; uy = snap.uy
    txx_p = snap.tauxx; txy_p = snap.tauxy; tyy_p = snap.tauyy
    mu_s = Float64(snap.nu_s)

    cs2 = 1.0/3.0
    norm = u_mean^2 * R_lu
    dtheta = 2π / N_az
    arc_dl = R_lu * dtheta

    dudx, dudy, dvdx, dvdy = lattice_velocity_gradient(ux, uy, solid)

    bins = (
        tx_pres = zeros(Float64, N_az), tx_solv = zeros(Float64, N_az),
        tx_poly = zeros(Float64, N_az), count   = zeros(Int,     N_az),
    )

    @inbounds for j in 1:Ny, i in 1:Nx
        solid[i, j] && continue
        has_solid_n = false
        for (di, dj) in ((-1,0),(1,0),(0,-1),(0,1),
                         (-1,-1),(-1,1),(1,-1),(1,1))
            ii = i + di; jj = j + dj
            if 1 <= ii <= Nx && 1 <= jj <= Ny && solid[ii, jj]
                has_solid_n = true
                break
            end
        end
        has_solid_n || continue
        dx = Float64(i) - cx_lu
        dy = Float64(j) - cy_lu
        r  = hypot(dx, dy)
        r > 0 || continue
        theta = atan(dy, dx)
        nx_o  = dx / r
        ny_o  = dy / r
        rho_ij = rho[i, j]
        p_lbm  = cs2 * (rho_ij - 1.0)
        tx_pres = -p_lbm * nx_o
        sxx_solv = 2.0 * mu_s * dudx[i,j]
        sxy_solv =       mu_s * (dudy[i,j] + dvdx[i,j])
        tx_solv = sxx_solv * nx_o + sxy_solv * ny_o
        tx_poly = txx_p[i,j] * nx_o + txy_p[i,j] * ny_o

        bin = mod(floor(Int, (theta + π) / dtheta), N_az) + 1
        bins.tx_pres[bin] += tx_pres
        bins.tx_solv[bin] += tx_solv
        bins.tx_poly[bin] += tx_poly
        bins.count[bin]   += 1
    end

    dCd_pres = zeros(Float64, N_az)
    dCd_solv = zeros(Float64, N_az)
    dCd_poly = zeros(Float64, N_az)
    for k in 1:N_az
        if bins.count[k] > 0
            dCd_pres[k] = (bins.tx_pres[k] / bins.count[k]) * arc_dl / norm
            dCd_solv[k] = (bins.tx_solv[k] / bins.count[k]) * arc_dl / norm
            dCd_poly[k] = (bins.tx_poly[k] / bins.count[k]) * arc_dl / norm
        end
    end
    theta_centres = [(-π + (k - 0.5)*dtheta) for k in 1:N_az]
    return (;
        N_az, dtheta, arc_dl, theta_centres, bin_count=bins.count,
        dCd_pres, dCd_solv, dCd_poly,
        Cd_pres = sum(dCd_pres),
        Cd_solv = sum(dCd_solv),
        Cd_poly = sum(dCd_poly),
        Cd_total = sum(dCd_pres) + sum(dCd_solv) + sum(dCd_poly),
        u_mean, R_lu, norm,
    )
end

function aggregate_regions(theta_centres, dCd_pres, dCd_solv, dCd_poly)
    regs = Dict(:front_pole => (pres=0.0, solv=0.0, poly=0.0),
                :shoulder   => (pres=0.0, solv=0.0, poly=0.0),
                :wake       => (pres=0.0, solv=0.0, poly=0.0))
    for k in eachindex(theta_centres)
        r = region_of(theta_centres[k])
        old = regs[r]
        regs[r] = (
            pres = old.pres + dCd_pres[k],
            solv = old.solv + dCd_solv[k],
            poly = old.poly + dCd_poly[k],
        )
    end
    return regs
end

function main()
    base = "tmp/m44_postfix_sweep/21827394.aqua"
    out_dir = "bench/scratch/m44_postfix_walldecomp"
    mkpath(out_dir)
    Rs = [30, 40, 50, 60]
    rows = Tuple{Int, Float64, Float64, Float64, Float64,
                 NamedTuple, NamedTuple, NamedTuple}[]
    for R in Rs
        fname = "cyl_bigsweep_v2_beta0p59_wi1_re1_R$(R)_bsd1_Lup15_Ldn15_eg0_ea0_ef0_ed0_geomqwall_fields.jls"
        path  = joinpath(base, fname)
        if !isfile(path)
            @warn "missing snapshot" path
            continue
        end
        snap = open(deserialize, path)
        K = kraken_wall_decomp(snap; N_az=36)
        regs = aggregate_regions(K.theta_centres, K.dCd_pres, K.dCd_solv, K.dCd_poly)
        push!(rows, (R, K.Cd_pres, K.Cd_solv, K.Cd_poly, K.Cd_total,
                     regs[:front_pole], regs[:shoulder], regs[:wake]))
        # per-bin CSV
        open(joinpath(out_dir, "M44P_kraken_R$(R)_bins_idx.csv"), "w") do io
            println(io, "bin,theta_deg,dCd_pres,dCd_solv,dCd_poly")
            for k in eachindex(K.theta_centres)
                @printf(io, "%d,%.2f,%.6e,%.6e,%.6e\n",
                        k, rad2deg(K.theta_centres[k]),
                        K.dCd_pres[k], K.dCd_solv[k], K.dCd_poly[k])
            end
        end
    end

    # Aggregated CSV
    open(joinpath(out_dir, "M44P_decomp_RvsR.csv"), "w") do io
        println(io, "R,Cd_pres_total,Cd_solv_total,Cd_poly_total,Cd_total,",
                "pres_front,pres_shoulder,pres_wake,",
                "solv_front,solv_shoulder,solv_wake,",
                "poly_front,poly_shoulder,poly_wake")
        for r in rows
            R, Cd_p, Cd_s, Cd_po, Cd_t, fp, sh, wk = r
            @printf(io, "%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                    R, Cd_p, Cd_s, Cd_po, Cd_t,
                    fp.pres, sh.pres, wk.pres,
                    fp.solv, sh.solv, wk.solv,
                    fp.poly, sh.poly, wk.poly)
        end
    end

    # Pretty print to stdout
    println("\n=== M44 POSTFIX WALL DECOMPOSITION Wi=1 β=0.59 ===\n")
    println("Total Cd per R (sum from ring):")
    @printf("  %3s  %10s  %10s  %10s  %10s\n", "R", "Cd_pres", "Cd_solv", "Cd_poly", "Cd_total")
    for r in rows
        R, Cd_p, Cd_s, Cd_po, Cd_t, _, _, _ = r
        @printf("  %3d  %10.4f  %10.4f  %10.4f  %10.4f\n", R, Cd_p, Cd_s, Cd_po, Cd_t)
    end

    println("\nPer-region Cd_pressure (the residual signal):")
    @printf("  %3s  %12s  %12s  %12s\n", "R", "front_pole", "shoulder", "wake")
    for r in rows
        R = r[1]; fp = r[6]; sh = r[7]; wk = r[8]
        @printf("  %3d  %12.4f  %12.4f  %12.4f\n", R, fp.pres, sh.pres, wk.pres)
    end

    println("\nPer-region Cd_solvent (the dominant gap component):")
    @printf("  %3s  %12s  %12s  %12s\n", "R", "front_pole", "shoulder", "wake")
    for r in rows
        R = r[1]; fp = r[6]; sh = r[7]; wk = r[8]
        @printf("  %3d  %12.4f  %12.4f  %12.4f\n", R, fp.solv, sh.solv, wk.solv)
    end

    println("\nPer-region Cd_polymer:")
    @printf("  %3s  %12s  %12s  %12s\n", "R", "front_pole", "shoulder", "wake")
    for r in rows
        R = r[1]; fp = r[6]; sh = r[7]; wk = r[8]
        @printf("  %3d  %12.4f  %12.4f  %12.4f\n", R, fp.poly, sh.poly, wk.poly)
    end

    # ΔR = R=60 - R=30
    if length(rows) >= 2
        first_row = rows[1]; last_row = rows[end]
        R0 = first_row[1]; R1 = last_row[1]
        fp0 = first_row[6]; sh0 = first_row[7]; wk0 = first_row[8]
        fp1 = last_row[6];  sh1 = last_row[7];  wk1 = last_row[8]
        @printf("\nΔCd_pressure R=%d → R=%d:\n", R0, R1)
        @printf("  front_pole: %+.4f\n", fp1.pres - fp0.pres)
        @printf("  shoulder  : %+.4f\n", sh1.pres - sh0.pres)
        @printf("  wake      : %+.4f\n", wk1.pres - wk0.pres)
        @printf("  TOTAL     : %+.4f\n", (fp1.pres+sh1.pres+wk1.pres)-(fp0.pres+sh0.pres+wk0.pres))
        @printf("\nΔCd_solvent R=%d → R=%d:\n", R0, R1)
        @printf("  front_pole: %+.4f\n", fp1.solv - fp0.solv)
        @printf("  shoulder  : %+.4f\n", sh1.solv - sh0.solv)
        @printf("  wake      : %+.4f\n", wk1.solv - wk0.solv)
        @printf("  TOTAL     : %+.4f\n", (fp1.solv+sh1.solv+wk1.solv)-(fp0.solv+sh0.solv+wk0.solv))
    end

    println("\nwrote $(joinpath(out_dir, "M44P_decomp_RvsR.csv"))")
    println("[", now(), "] done.")
end

main()
