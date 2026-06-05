#!/usr/bin/env julia
# Generate SVG figures for examples 01–09
# Run: julia --project=docs docs/generate_figures_01_09.jl

using Kraken
using CairoMakie

const KA = Kraken.KernelAbstractions
const OUTDIR = joinpath(@__DIR__, "src", "examples")

# --- Dark docs theme (#1b1b1f Vitepress page bg, light text/ticks/spines/grid) ---
# Every Figure()/Axis()/Colorbar()/Legend() below inherits this so the exported
# SVGs sit seamlessly on the dark Vitepress page. Mirrors the explicit dark block
# that already produces cylinder_umag.png. Blocks that pass their own
# backgroundcolor (e.g. cylinder_umag) still win locally.
const KRAKEN_DARK = "#1b1b1f"
set_theme!(Theme(
    backgroundcolor = KRAKEN_DARK,
    textcolor = "gray92",
    Axis = (
        backgroundcolor = KRAKEN_DARK,
        titlecolor = "gray92",
        xlabelcolor = "gray92", ylabelcolor = "gray92",
        xticklabelcolor = "gray85", yticklabelcolor = "gray85",
        xtickcolor = "gray70", ytickcolor = "gray70",
        xgridcolor = ("gray60", 0.18), ygridcolor = ("gray60", 0.18),
        leftspinecolor = "gray55", rightspinecolor = "gray55",
        topspinecolor = "gray55", bottomspinecolor = "gray55",
    ),
    Axis3 = (
        backgroundcolor = KRAKEN_DARK,
        titlecolor = "gray92",
        xlabelcolor = "gray92", ylabelcolor = "gray92", zlabelcolor = "gray92",
        xticklabelcolor = "gray85", yticklabelcolor = "gray85", zticklabelcolor = "gray85",
    ),
    Colorbar = (
        labelcolor = "gray92", ticklabelcolor = "gray85", tickcolor = "gray70",
        leftspinecolor = "gray55", rightspinecolor = "gray55",
        topspinecolor = "gray55", bottomspinecolor = "gray55",
    ),
    Legend = (
        backgroundcolor = KRAKEN_DARK, labelcolor = "gray92",
        titlecolor = "gray92", framecolor = "gray45",
    ),
))

# ============================================================================
# === 1. Poiseuille 2D ======================================================
# ============================================================================
println("=== 1. Poiseuille 2D ===")

# --- 1a. Geometry schematic ---
let
    fig = Figure(size=(600, 300))
    ax = Axis(fig[1, 1]; title="Poiseuille 2D — geometry",
              xlabel="x", ylabel="y", aspect=DataAspect(),
              limits=(-1, 6, -1, 5))

    # Walls
    poly!(ax, Point2f[(0, 0), (5, 0), (5, -0.3), (0, -0.3)]; color=:gray70, strokewidth=1)
    poly!(ax, Point2f[(0, 4), (5, 4), (5, 4.3), (0, 4.3)]; color=:gray70, strokewidth=1)
    text!(ax, 2.5, -0.7; text="Bottom wall (bounce-back)", align=(:center, :top), fontsize=11)
    text!(ax, 2.5, 4.7; text="Top wall (bounce-back)", align=(:center, :bottom), fontsize=11)

    # Body force arrow
    arrows!(ax, [0.5], [2.0], [1.5], [0.0]; color=:red, linewidth=3, arrowsize=15)
    text!(ax, 1.2, 2.4; text="Fx (body force)", color=:red, fontsize=12)

    # Periodic labels
    text!(ax, -0.5, 2.0; text="periodic", rotation=pi/2, align=(:center, :center), fontsize=11, color=:blue)
    text!(ax, 5.5, 2.0; text="periodic", rotation=pi/2, align=(:center, :center), fontsize=11, color=:blue)

    hidespines!(ax)
    hidedecorations!(ax)
    save(joinpath(OUTDIR, "poiseuille_geometry.svg"), fig)
    println("  ✓ poiseuille_geometry.svg")
end

# --- 1b. Velocity profile ---
let
    Ny = 32; ν = 0.1; Fx = 1e-5
    ρ, ux, uy, config = run_poiseuille_2d(; Nx=4, Ny=Ny, ν=ν, Fx=Fx, max_steps=20000)

    # Full-way bounce-back (stream_periodic_x_wall_y_2d!) places the no-slip wall
    # HALFWAY between the last fluid node and the ghost layer: at y = 0.5 (below
    # j=1) and y = Ny + 0.5 (above j=Ny). The effective channel height is H = Ny,
    # the physical coordinate of fluid node j is y = j - 0.5, and ALL nodes j=1..Ny
    # are fluid. With these wall-aware coordinates the D2Q9 parabola is exact.
    H = Ny
    j_fluid = 1:Ny
    y_phys = [j - 0.5 for j in j_fluid]
    u_ana  = [Fx / (2ν) * y * (H - y) for y in y_phys]
    u_num  = [ux[2, j] for j in j_fluid]

    fig = Figure(size=(600, 420))
    ax = Axis(fig[1, 1]; xlabel="u_x (lattice units)", ylabel="y (lattice units)",
              title="Poiseuille flow — Ny = $Ny")
    lines!(ax, u_ana, y_phys; label="Analytical", linewidth=2)
    scatter!(ax, u_num, y_phys; label="Kraken", markersize=8)
    axislegend(ax; position=:rb)
    save(joinpath(OUTDIR, "poiseuille_profile.svg"), fig)
    println("  ✓ poiseuille_profile.svg")
end

# --- 1c. Convergence ---
let
    ν = 0.1; Fx = 1e-5
    Ny_list = [16, 32, 64, 128]
    errors = Float64[]

    for Ny_i in Ny_list
        H_i = Ny_i                       # halfway-wall channel height (see 1b)
        # Diffusive time to steady state scales as H²/ν, so the step budget MUST
        # grow with resolution — a fixed budget leaves the fine grids transient
        # and corrupts the slope (spurious error blow-up at Ny≥64).
        nsteps = max(30_000, ceil(Int, 8 * H_i^2 / ν))
        ρ_i, ux_i, _, _ = run_poiseuille_2d(; Nx=4, Ny=Ny_i, ν=ν, Fx=Fx, max_steps=nsteps)
        jf  = 1:Ny_i                     # all nodes are fluid
        u_a = [Fx / (2ν) * (j - 0.5) * (H_i - (j - 0.5)) for j in jf]
        u_n = [ux_i[2, j] for j in jf]
        L2  = sqrt(sum((u_n .- u_a).^2) / sum(u_a.^2))
        push!(errors, L2)
    end

    fig = Figure(size=(500, 400))
    ax = Axis(fig[1, 1]; xlabel="Ny", ylabel="Relative L2 error",
              title="Convergence — Poiseuille flow", xscale=log10, yscale=log10)
    scatterlines!(ax, Float64.(Ny_list), errors; linewidth=2, markersize=10, label="Kraken")
    ref = errors[1] .* (Ny_list[1] ./ Ny_list).^2
    lines!(ax, Float64.(Ny_list), ref; linestyle=:dash, color=:gray, label="slope 2")
    axislegend(ax; position=:lb)
    save(joinpath(OUTDIR, "poiseuille_convergence.svg"), fig)
    println("  ✓ poiseuille_convergence.svg")
end

# ============================================================================
# === 2. Couette 2D =========================================================
# ============================================================================
println("=== 2. Couette 2D ===")

# --- 2a. Geometry schematic ---
let
    fig = Figure(size=(600, 300))
    ax = Axis(fig[1, 1]; title="Couette 2D — geometry",
              xlabel="x", ylabel="y", aspect=DataAspect(),
              limits=(-1, 6, -1, 5))

    # Bottom wall (moving)
    poly!(ax, Point2f[(0, 0), (5, 0), (5, -0.3), (0, -0.3)]; color=:tomato, strokewidth=1)
    arrows!(ax, [0.3], [-0.15], [1.5], [0.0]; color=:red, linewidth=3, arrowsize=15)
    text!(ax, 2.5, -0.7; text="Moving wall (u_wall, Zou-He)", align=(:center, :top), fontsize=11)

    # Top wall (stationary)
    poly!(ax, Point2f[(0, 4), (5, 4), (5, 4.3), (0, 4.3)]; color=:gray70, strokewidth=1)
    text!(ax, 2.5, 4.7; text="Stationary wall (Zou-He)", align=(:center, :bottom), fontsize=11)

    # Periodic labels
    text!(ax, -0.5, 2.0; text="periodic", rotation=pi/2, align=(:center, :center), fontsize=11, color=:blue)
    text!(ax, 5.5, 2.0; text="periodic", rotation=pi/2, align=(:center, :center), fontsize=11, color=:blue)

    hidespines!(ax)
    hidedecorations!(ax)
    save(joinpath(OUTDIR, "couette_geometry.svg"), fig)
    println("  ✓ couette_geometry.svg")
end

# --- 2b. Velocity profile ---
let
    Ny = 32; ν = 0.1; u_wall = 0.05
    ρ, ux, uy, config = run_couette_2d(; Nx=4, Ny=Ny, ν=ν, u_wall=u_wall, max_steps=20000)

    H = Ny - 1
    j_fluid = 2:Ny-1
    y_phys = [j - 1 for j in j_fluid]
    u_ana  = [u_wall * (1 - y / H) for y in y_phys]
    u_num  = [ux[2, j] for j in j_fluid]

    fig = Figure(size=(600, 420))
    ax = Axis(fig[1, 1]; xlabel="u_x (lattice units)", ylabel="y (lattice units)",
              title="Couette flow — Ny = $Ny")
    lines!(ax, u_ana, y_phys; label="Analytical", linewidth=2)
    scatter!(ax, u_num, y_phys; label="Kraken", markersize=8)
    axislegend(ax; position=:rt)
    save(joinpath(OUTDIR, "couette_profile.svg"), fig)
    println("  ✓ couette_profile.svg")
end

# --- 2c. Convergence ---
let
    ν = 0.1; u_wall = 0.05
    Ny_list = [16, 32, 64, 128]
    errors = Float64[]

    for Ny_i in Ny_list
        H_i = Ny_i - 1
        nsteps = max(10_000, ceil(Int, 3 * H_i^2 / ν))
        ρ_i, ux_i, _, _ = run_couette_2d(; Nx=4, Ny=Ny_i, ν=ν, u_wall=u_wall, max_steps=nsteps)
        # Zou-He pins the wall velocity exactly ON nodes j=1 and j=Ny, so the
        # linear profile is reproduced over EVERY node. Measure over all of them.
        jf  = 1:Ny_i
        u_a = [u_wall * (1 - (j - 1) / H_i) for j in jf]
        u_n = [ux_i[2, j] for j in jf]
        L2  = sqrt(sum((u_n .- u_a).^2) / sum(u_a.^2))
        push!(errors, L2)
    end

    # Couette is EXACT for D2Q9 (linear profile is a 2nd-order-exact lattice
    # solution): the error is pure floating-point roundoff (~1e-14…1e-12) and has
    # NO convergence trend. Plotting a "slope 2" reference here would be a lie, so
    # we show the machine-precision floor honestly with a flat reference band.
    fig = Figure(size=(500, 400))
    ax = Axis(fig[1, 1]; xlabel="Ny", ylabel="Relative L2 error",
              title="Couette — exact to machine precision", xscale=log10, yscale=log10)
    scatterlines!(ax, Float64.(Ny_list), errors; linewidth=2, markersize=10, label="Kraken")
    hlines!(ax, [eps(Float64)]; linestyle=:dash, color=:gray, label="machine ε (Float64)")
    ylims!(ax, 1e-16, 1e-9)
    axislegend(ax; position=:lt)
    save(joinpath(OUTDIR, "couette_convergence.svg"), fig)
    println("  ✓ couette_convergence.svg")
end

# ============================================================================
# === 3. Taylor-Green 2D ====================================================
# ============================================================================
println("=== 3. Taylor-Green 2D ===")

# --- 3a. Initial vorticity field ---
let
    N = 64; u0 = 0.04; ν = 0.01
    # Compute initial vorticity analytically
    k = 2pi / N
    ωz_init = zeros(N, N)
    for j in 1:N, i in 1:N
        x = i - 1.0
        y = j - 1.0
        # ω_z = ∂uy/∂x - ∂ux/∂y = u0*k*cos(kx)*cos(ky) + u0*k*cos(kx)*cos(ky)
        ωz_init[i, j] = 2.0 * u0 * k * cos(k * x) * cos(k * y)
    end

    fig = Figure(size=(500, 450))
    ax = Axis(fig[1, 1]; title="Initial vorticity field", aspect=DataAspect())
    hm = heatmap!(ax, 1:N, 1:N, ωz_init; colormap=:balance)
    Colorbar(fig[1, 2], hm; label="omega_z")
    save(joinpath(OUTDIR, "taylor_green_geometry.svg"), fig)
    println("  ✓ taylor_green_geometry.svg")
end

# --- 3b. Energy decay ---
let
    N = 64; u0 = 0.04; ν = 0.01
    k = 2pi / N
    # The velocity field u = u0·(−cos kx sin ky, sin kx cos ky) decays as
    # exp(−2νk²t), so the kinetic energy ∝ u² decays as exp(−4νk²t). The MEAN
    # kinetic-energy density of that field is E0 = ⟨½(ux²+uy²)⟩ = u0²/4 (the cos²/sin²
    # spatial averages each give ¼), NOT the peak ½u0². Using ½u0² over-predicts the
    # baseline by 2× and exp(−2νk²t) halves the slope — both were wrong before.
    E0 = u0^2 / 4

    steps_list = 0:500:5000
    E_num = Float64[]
    E_ana = Float64[]

    for s in steps_list
        if s == 0
            push!(E_num, E0)
        else
            res_s = run_taylor_green_2d(; N=N, ν=ν, u0=u0, max_steps=s)
            ux_s = res_s.ux; uy_s = res_s.uy
            KE = 0.0
            for j in 1:N, i in 1:N
                KE += 0.5 * (ux_s[i, j]^2 + uy_s[i, j]^2)
            end
            push!(E_num, KE / (N * N))
        end
        push!(E_ana, E0 * exp(-4ν * k^2 * s))
    end

    fig = Figure(size=(600, 420))
    ax = Axis(fig[1, 1]; xlabel="Time step", ylabel="Mean kinetic energy",
              title="Taylor-Green vortex decay — N = $N")
    lines!(ax, collect(steps_list), E_ana; label="Analytical", linewidth=2)
    scatter!(ax, collect(steps_list), E_num; label="Kraken", markersize=10)
    axislegend(ax; position=:rt)
    save(joinpath(OUTDIR, "taylor_green_decay.svg"), fig)
    println("  ✓ taylor_green_decay.svg")
end

# --- 3c. Vorticity at final time ---
let
    N = 64; u0 = 0.04; ν = 0.01
    ρ, ux, uy, config, u0_out, k, max_steps = run_taylor_green_2d(;
        N=N, ν=ν, u0=u0, max_steps=5000)

    ωz = zeros(N, N)
    for j in 1:N, i in 1:N
        ip = mod1(i + 1, N); im = mod1(i - 1, N)
        jp = mod1(j + 1, N); jm = mod1(j - 1, N)
        ωz[i, j] = 0.5 * (uy[ip, j] - uy[im, j]) - 0.5 * (ux[i, jp] - ux[i, jm])
    end

    fig = Figure(size=(500, 450))
    ax = Axis(fig[1, 1]; title="Vorticity at t = $max_steps", aspect=DataAspect())
    hm = heatmap!(ax, 1:N, 1:N, ωz; colormap=:balance)
    Colorbar(fig[1, 2], hm; label="omega_z")
    save(joinpath(OUTDIR, "taylor_green_vorticity.svg"), fig)
    println("  ✓ taylor_green_vorticity.svg")
end

# ============================================================================
# === 4. Cavity 2D ==========================================================
# ============================================================================
println("=== 4. Cavity 2D ===")

# --- 4a. Geometry schematic ---
let
    fig = Figure(size=(500, 500))
    ax = Axis(fig[1, 1]; title="Lid-driven cavity — geometry",
              aspect=DataAspect(), limits=(-1, 6, -1, 6))

    # Walls
    poly!(ax, Point2f[(0, 0), (5, 0), (5, -0.3), (0, -0.3)]; color=:gray70, strokewidth=1)
    poly!(ax, Point2f[(0, 0), (-0.3, 0), (-0.3, 5), (0, 5)]; color=:gray70, strokewidth=1)
    poly!(ax, Point2f[(5, 0), (5.3, 0), (5.3, 5), (5, 5)]; color=:gray70, strokewidth=1)

    # Lid (top)
    poly!(ax, Point2f[(0, 5), (5, 5), (5, 5.3), (0, 5.3)]; color=:tomato, strokewidth=1)
    arrows!(ax, [0.5], [5.15], [2.0], [0.0]; color=:red, linewidth=3, arrowsize=15)
    text!(ax, 2.5, 5.7; text="u_lid", align=(:center, :bottom), fontsize=13, color=:red)

    # Wall labels
    text!(ax, 2.5, -0.7; text="bottom wall", align=(:center, :top), fontsize=11)
    text!(ax, -0.7, 2.5; text="left wall", rotation=pi/2, align=(:center, :center), fontsize=11)
    text!(ax, 5.7, 2.5; text="right wall", rotation=-pi/2, align=(:center, :center), fontsize=11)

    hidespines!(ax)
    hidedecorations!(ax)
    save(joinpath(OUTDIR, "cavity_geometry.svg"), fig)
    println("  ✓ cavity_geometry.svg")
end

# --- 4b. Centerline profiles vs Ghia ---
let
    N = 128; Re = 100; u_lid = 0.1
    ν = u_lid * N / Re
    config = LBMConfig(D2Q9(); Nx=N, Ny=N, ν=ν, u_lid=u_lid,
                       max_steps=60000, output_interval=10000)
    ρ, ux, uy, _ = run_cavity_2d(config)

    # Ghia et al. data for Re=100
    y_ghia  = [0.0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
               0.4531, 0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609,
               0.9688, 0.9766, 1.0]
    ux_ghia = [0.0, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
              -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151,
               0.68717, 0.73722, 0.78871, 0.84123, 1.0]

    mid = N ÷ 2 + 1
    ux_profile = [ux[mid, j] / u_lid for j in 1:N]
    y_norm     = [(j - 0.5) / N for j in 1:N]
    uy_profile = [uy[i, mid] / u_lid for i in 1:N]
    x_norm     = [(i - 0.5) / N for i in 1:N]

    fig = Figure(size=(900, 420))
    ax1 = Axis(fig[1, 1]; xlabel="u_x / u_lid", ylabel="y / N",
               title="Vertical centreline")
    lines!(ax1, ux_profile, y_norm; label="Kraken (N=$N)", linewidth=2)
    scatter!(ax1, ux_ghia, y_ghia; label="Ghia et al.", color=:red, markersize=8)
    axislegend(ax1; position=:lb)

    ax2 = Axis(fig[1, 2]; xlabel="x / N", ylabel="u_y / u_lid",
               title="Horizontal centreline")
    lines!(ax2, x_norm, uy_profile; label="Kraken (N=$N)", linewidth=2)
    axislegend(ax2; position=:rt)
    save(joinpath(OUTDIR, "cavity_centerlines.svg"), fig)
    println("  ✓ cavity_centerlines.svg")

    # --- 4c. Velocity magnitude ---
    umag = @. sqrt(ux^2 + uy^2) / u_lid
    fig2 = Figure(size=(500, 480))
    ax3 = Axis(fig2[1, 1]; title="Velocity magnitude — Re=$Re", aspect=DataAspect())
    hm = heatmap!(ax3, 1:N, 1:N, umag; colormap=:viridis)
    Colorbar(fig2[1, 2], hm; label="|u| / u_lid")
    save(joinpath(OUTDIR, "cavity_umag.svg"), fig2)
    println("  ✓ cavity_umag.svg")
end

# ============================================================================
# === 5. Cavity 3D ==========================================================
# ============================================================================
println("=== 5. Cavity 3D ===")

let
    # N=32 triggers a CPU segfault in stream_3d! on Apple Silicon; use N=24
    N = 24; Re = 100; u_lid = 0.05
    ν = u_lid * N / Re
    config = LBMConfig(D3Q19(); Nx=N, Ny=N, Nz=N, ν=ν, u_lid=u_lid,
                       max_steps=20000, output_interval=10000)
    ρ, ux, uy, uz, _ = run_cavity_3d(config)

    mid = N ÷ 2
    umag = zeros(N, N)
    for j in 1:N, i in 1:N
        umag[i, j] = sqrt(ux[i, j, mid]^2 + uy[i, j, mid]^2 + uz[i, j, mid]^2)
    end
    umag ./= u_lid

    fig = Figure(size=(550, 480))
    ax = Axis(fig[1, 1]; title="Velocity magnitude — mid-plane z=$mid",
              xlabel="x", ylabel="y", aspect=DataAspect())
    hm = heatmap!(ax, 1:N, 1:N, umag; colormap=:viridis)
    Colorbar(fig[1, 2], hm; label="|u| / u_lid")
    save(joinpath(OUTDIR, "cavity_3d_umag.svg"), fig)
    println("  ✓ cavity_3d_umag.svg")
end

# ============================================================================
# === 6. Cylinder 2D ========================================================
# ============================================================================
println("=== 6. Cylinder 2D ===")

# --- 6a. Geometry schematic ---
let
    fig = Figure(size=(800, 300))
    ax = Axis(fig[1, 1]; title="Flow around a cylinder — geometry",
              aspect=DataAspect(), limits=(-20, 420, -15, 115))

    # Domain boundary
    lines!(ax, [0, 400, 400, 0, 0], [0, 0, 100, 100, 0]; color="gray80", linewidth=1.5)

    # Cylinder
    θ = range(0, 2pi, length=60)
    cx, cy, R = 80.0, 50.0, 10.0
    poly!(ax, [Point2f(cx + R * cos(t), cy + R * sin(t)) for t in θ]; color=:gray50, strokewidth=2)

    # Inlet arrows
    for yy in 20:20:80
        arrows!(ax, [-15.0], [Float64(yy)], [12.0], [0.0]; color=:blue, linewidth=2, arrowsize=10)
    end
    text!(ax, -15, 95; text="inlet\nu_in", fontsize=11, color=:blue)

    # Outlet label
    text!(ax, 410, 50; text="outlet", fontsize=11, align=(:left, :center))

    # Top/bottom labels
    text!(ax, 200, -8; text="wall (free-slip)", align=(:center, :top), fontsize=11)
    text!(ax, 200, 108; text="wall (free-slip)", align=(:center, :bottom), fontsize=11)

    hidespines!(ax)
    hidedecorations!(ax)
    save(joinpath(OUTDIR, "cylinder_geometry.svg"), fig)
    println("  ✓ cylinder_geometry.svg")
end

# --- 6b. Velocity magnitude & drag ---
let
    Re = 20; radius = 10; u_in = 0.04
    D = 2 * radius
    ν = u_in * D / Re

    result = run_cylinder_2d(; Nx=400, Ny=100, radius=radius, u_in=u_in, ν=ν,
                               max_steps=40000, avg_window=2000)
    ux = result.ux; uy = result.uy; Cd = result.Cd
    Nx, Ny = size(ux)
    umag = @. sqrt(ux^2 + uy^2)

    # Dark docs styling (#1b1b1f Vitepress page bg, magma sequential field, light
    # text) to match the dark page. Saved as PNG — the per-cell heatmap
    # exports to a multi-MB SVG, so raster it directly.
    DARK = "#1b1b1f"
    fig = Figure(size=(800, 350), backgroundcolor=DARK)
    ax = Axis(fig[1, 1]; backgroundcolor=DARK,
              title="Velocity magnitude — Re=$Re", titlecolor="gray92",
              xlabel="x", ylabel="y", xlabelcolor="gray92", ylabelcolor="gray92",
              xticklabelcolor="gray85", yticklabelcolor="gray85",
              xtickcolor="gray70", ytickcolor="gray70",
              leftspinecolor="gray55", rightspinecolor="gray55",
              topspinecolor="gray55", bottomspinecolor="gray55",
              aspect=DataAspect())
    hm = heatmap!(ax, 1:Nx, 1:Ny, umag; colormap=:magma,
                  colorrange=(0, 1.5 * u_in))
    Colorbar(fig[1, 2], hm; label="|u|", labelcolor="gray92",
             ticklabelcolor="gray85", tickcolor="gray70")
    save(joinpath(OUTDIR, "cylinder_umag.png"), fig, px_per_unit=2)
    println("  ✓ cylinder_umag.png")

    # --- 6c. Drag comparison bar chart ---
    Cd_ref = 5.58
    fig2 = Figure(size=(400, 300))
    ax2 = Axis(fig2[1, 1]; title="Drag comparison at Re = $Re")
    barplot!(ax2, [1, 2], [Cd, Cd_ref]; color=[:steelblue, :tomato],
             bar_labels=[string(round(Cd; digits=3)), string(Cd_ref)],
             label_color="gray92")
    ax2.xticks = ([1, 2], ["Kraken", "Schafer-Turek"])
    ax2.ylabel = "Cd"
    save(joinpath(OUTDIR, "cylinder_drag.svg"), fig2)
    println("  ✓ cylinder_drag.svg")
end

# ============================================================================
# === 7. Heat Conduction =====================================================
# ============================================================================
println("=== 7. Heat Conduction ===")

# --- 7a. Geometry schematic ---
let
    fig = Figure(size=(600, 300))
    ax = Axis(fig[1, 1]; title="Heat conduction — geometry",
              aspect=DataAspect(), limits=(-1, 8, -1, 5))

    # Hot bottom wall
    poly!(ax, Point2f[(0, 0), (7, 0), (7, -0.3), (0, -0.3)]; color=:red, strokewidth=1)
    text!(ax, 3.5, -0.7; text="T_hot (bottom wall)", align=(:center, :top), fontsize=12, color=:red)

    # Cold top wall
    poly!(ax, Point2f[(0, 4), (7, 4), (7, 4.3), (0, 4.3)]; color=:blue, strokewidth=1)
    text!(ax, 3.5, 4.7; text="T_cold (top wall)", align=(:center, :bottom), fontsize=12, color=:blue)

    # Periodic labels
    text!(ax, -0.5, 2.0; text="periodic", rotation=pi/2, align=(:center, :center), fontsize=11, color=:gray50)
    text!(ax, 7.5, 2.0; text="periodic", rotation=pi/2, align=(:center, :center), fontsize=11, color=:gray50)

    # Temperature gradient arrows (downward = cold, upward direction)
    for xx in 1.0:1.5:6.0
        arrows!(ax, [xx], [3.2], [0.0], [-1.5]; color=:orange, linewidth=1.5, arrowsize=10)
    end
    text!(ax, 4.5, 2.0; text="heat flux", fontsize=11, color=:orange)

    hidespines!(ax)
    hidedecorations!(ax)
    save(joinpath(OUTDIR, "heat_geometry.svg"), fig)
    println("  ✓ heat_geometry.svg")
end

# --- 7b. Temperature profile ---
let
    Ra = 100.0; Pr = 1.0; T_hot = 1.0; T_cold = 0.0
    ρ, ux, uy, Temp, config, Ra_out, Pr_out, ν, α = run_rayleigh_benard_2d(;
        Nx=128, Ny=32, Ra=Ra, Pr=Pr, T_hot=T_hot, T_cold=T_cold, max_steps=20000)

    Ny = size(Temp, 2)
    H = Ny - 1
    j_fluid = 2:Ny-1
    y_phys = [(j - 1.5) / H for j in j_fluid]
    T_ana  = [T_hot - (T_hot - T_cold) * y for y in y_phys]
    T_num  = [Temp[64, j] for j in j_fluid]

    fig = Figure(size=(600, 420))
    ax = Axis(fig[1, 1]; xlabel="Temperature", ylabel="y / H",
              title="Heat conduction — Ra = $Ra (sub-critical)")
    lines!(ax, T_ana, y_phys; label="Analytical (linear)", linewidth=2)
    scatter!(ax, T_num, y_phys; label="Kraken", markersize=8)
    axislegend(ax; position=:rt)
    save(joinpath(OUTDIR, "heat_profile.svg"), fig)
    println("  ✓ heat_profile.svg")
end

# ============================================================================
# === 8. Rayleigh-Benard =====================================================
# ============================================================================
println("=== 8. Rayleigh-Benard ===")

# --- 8a. Geometry schematic ---
let
    fig = Figure(size=(700, 350))
    ax = Axis(fig[1, 1]; title="Rayleigh-Benard convection — geometry",
              aspect=DataAspect(), limits=(-1, 10, -1, 5))

    # Hot bottom
    poly!(ax, Point2f[(0, 0), (9, 0), (9, -0.3), (0, -0.3)]; color=:red, strokewidth=1)
    text!(ax, 4.5, -0.7; text="T_hot (bottom)", align=(:center, :top), fontsize=12, color=:red)

    # Cold top
    poly!(ax, Point2f[(0, 4), (9, 4), (9, 4.3), (0, 4.3)]; color=:blue, strokewidth=1)
    text!(ax, 4.5, 4.7; text="T_cold (top)", align=(:center, :bottom), fontsize=12, color=:blue)

    # Convection roll arrows (two counter-rotating cells)
    # Left roll (clockwise)
    arrows!(ax, [1.5], [1.0], [0.0], [2.0]; color=:orange, linewidth=2, arrowsize=12)
    arrows!(ax, [1.5], [3.0], [1.5], [0.0]; color=:orange, linewidth=2, arrowsize=12)
    arrows!(ax, [3.0], [3.0], [0.0], [-2.0]; color=:orange, linewidth=2, arrowsize=12)
    arrows!(ax, [3.0], [1.0], [-1.5], [0.0]; color=:orange, linewidth=2, arrowsize=12)

    # Right roll (counter-clockwise)
    arrows!(ax, [6.0], [1.0], [0.0], [2.0]; color=:purple, linewidth=2, arrowsize=12)
    arrows!(ax, [6.0], [3.0], [1.5], [0.0]; color=:purple, linewidth=2, arrowsize=12)
    arrows!(ax, [7.5], [3.0], [0.0], [-2.0]; color=:purple, linewidth=2, arrowsize=12)
    arrows!(ax, [7.5], [1.0], [-1.5], [0.0]; color=:purple, linewidth=2, arrowsize=12)

    text!(ax, 2.25, 2.0; text="convection\nrolls", align=(:center, :center), fontsize=11, color=:orange)

    # Periodic
    text!(ax, -0.5, 2.0; text="periodic", rotation=pi/2, align=(:center, :center), fontsize=11, color=:gray50)
    text!(ax, 9.5, 2.0; text="periodic", rotation=pi/2, align=(:center, :center), fontsize=11, color=:gray50)

    hidespines!(ax)
    hidedecorations!(ax)
    save(joinpath(OUTDIR, "rayleigh_benard_geometry.svg"), fig)
    println("  ✓ rayleigh_benard_geometry.svg")
end

# --- 8b. Temperature field & velocity magnitude ---
let
    Ra = 5000.0; Pr = 1.0; T_hot = 1.0; T_cold = 0.0
    ρ, ux, uy, Temp, config, Ra_out, Pr_out, ν, α = run_rayleigh_benard_2d(;
        Nx=128, Ny=32, Ra=Ra, Pr=Pr, T_hot=T_hot, T_cold=T_cold, max_steps=30000)

    Nx, Ny = size(Temp)

    fig = Figure(size=(800, 350))
    ax = Axis(fig[1, 1]; title="Temperature — Ra = $Ra, Pr = $Pr",
              xlabel="x", ylabel="y", aspect=DataAspect())
    hm = heatmap!(ax, 1:Nx, 1:Ny, Temp; colormap=:thermal, colorrange=(T_cold, T_hot))
    Colorbar(fig[1, 2], hm; label="T")
    save(joinpath(OUTDIR, "rayleigh_benard_temperature.svg"), fig)
    println("  ✓ rayleigh_benard_temperature.svg")

    umag = @. sqrt(ux^2 + uy^2)
    fig2 = Figure(size=(800, 350))
    ax2 = Axis(fig2[1, 1]; title="Velocity magnitude — Ra = $Ra",
               xlabel="x", ylabel="y", aspect=DataAspect())
    hm2 = heatmap!(ax2, 1:Nx, 1:Ny, umag; colormap=:viridis)
    Colorbar(fig2[1, 2], hm2; label="|u|")
    save(joinpath(OUTDIR, "rayleigh_benard_velocity.svg"), fig2)
    println("  ✓ rayleigh_benard_velocity.svg")
end

# ============================================================================
# === 9. Hagen-Poiseuille ===================================================
# ============================================================================
println("=== 9. Hagen-Poiseuille ===")

# --- 9a. Geometry schematic ---
let
    fig = Figure(size=(600, 450))
    ax = Axis(fig[1, 1]; title="Hagen-Poiseuille — pipe cross-section (z-r plane)",
              aspect=DataAspect(), limits=(-1, 7, -2, 5))

    # Pipe walls
    poly!(ax, Point2f[(0, 4), (6, 4), (6, 4.3), (0, 4.3)]; color=:gray70, strokewidth=1)
    poly!(ax, Point2f[(0, 0), (6, 0), (6, -0.3), (0, -0.3)]; color=:gray70, strokewidth=1)
    text!(ax, 3.0, 4.7; text="wall (bounce-back, r = R)", align=(:center, :bottom), fontsize=11)
    text!(ax, 3.0, -0.7; text="axis of symmetry (r = 0)", align=(:center, :top), fontsize=11, color=:blue)

    # Dashed symmetry line
    lines!(ax, [0, 6], [0, 0]; color=:blue, linestyle=:dash, linewidth=1.5)

    # Body force arrow
    arrows!(ax, [0.5], [2.0], [2.0], [0.0]; color=:red, linewidth=3, arrowsize=15)
    text!(ax, 1.5, 2.5; text="Fz (body force)", color=:red, fontsize=12)

    # Axis labels
    text!(ax, 6.5, 0.0; text="z", fontsize=14, align=(:left, :center))
    text!(ax, 0.0, 4.8; text="r", fontsize=14, align=(:center, :bottom))

    # Periodic arrows
    text!(ax, -0.5, 2.0; text="periodic", rotation=pi/2, align=(:center, :center), fontsize=11, color=:gray50)
    text!(ax, 6.5, 2.0; text="periodic", rotation=pi/2, align=(:center, :center), fontsize=11, color=:gray50)

    hidespines!(ax)
    hidedecorations!(ax)
    save(joinpath(OUTDIR, "hagen_poiseuille_geometry.svg"), fig)
    println("  ✓ hagen_poiseuille_geometry.svg")
end

# --- 9b. Velocity profile ---
let
    Nr = 32; ν = 0.1; Fz = 1e-5
    ρ, uz, ur, config = run_hagen_poiseuille_2d(; Nz=4, Nr=Nr, ν=ν, Fz=Fz, max_steps=20000)

    R_eff  = Nr - 0.5
    j_fluid = 1:Nr
    r_phys = [j - 0.5 for j in j_fluid]
    u_ana  = [Fz / (4ν) * (R_eff^2 - r^2) for r in r_phys]
    u_num  = [uz[2, j] for j in j_fluid]

    fig = Figure(size=(600, 420))
    ax = Axis(fig[1, 1]; xlabel="u_z (lattice units)", ylabel="r (lattice units)",
              title="Hagen-Poiseuille flow — Nr = $Nr")
    lines!(ax, u_ana, r_phys; label="Analytical", linewidth=2)
    scatter!(ax, u_num, r_phys; label="Kraken (axisymmetric)", markersize=8)
    axislegend(ax; position=:rt)
    save(joinpath(OUTDIR, "hagen_poiseuille_profile.svg"), fig)
    println("  ✓ hagen_poiseuille_profile.svg")
end

println("\n=== All figures generated! ===")
