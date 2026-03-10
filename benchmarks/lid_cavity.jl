"""
    run_lid_cavity(; save_figures=true, figdir="docs/src/assets/figures")

Lid-driven cavity Re=100 benchmark with Ghia et al. 1982 comparison.
"""
function run_lid_cavity(; save_figures=true, figdir="docs/src/assets/figures")
    println("=" ^ 60)
    println("Benchmark: Lid-Driven Cavity Re=100")
    println("=" ^ 60)

    N = 64

    # Ghia et al. 1982 reference data
    y_ghia = [1.0, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344,
              0.6172, 0.5, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703,
              0.0625, 0.0547, 0.0]
    u_ghia = [1.0, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332,
              -0.13641, -0.20581, -0.21090, -0.15662, -0.10150, -0.06434,
              -0.04775, -0.04192, -0.03717, 0.0]

    # --- CPU run (explicit) ---
    t_cpu = @elapsed begin
        u, v, p, converged = run_cavity(N=N, Re=100.0, cfl=0.2,
                                         max_steps=20000, tol=1e-7, verbose=false)
    end
    println("  Explicit — Converged: $converged")

    # --- CPU run (implicit) ---
    t_cpu_implicit = @elapsed begin
        u_impl, v_impl, p_impl, conv_impl = run_cavity(N=N, Re=100.0, cfl=0.5,
                                                         max_steps=20000, tol=1e-7,
                                                         verbose=false, time_scheme=:implicit)
    end
    println("  Implicit — Converged: $conv_impl")

    # Extract u(y) at x=0.5
    dx = 1.0 / (N - 1)
    i_mid = round(Int, 0.5 / dx) + 1
    y_grid = range(0.0, 1.0, length=N)
    u_profile = Array(u[i_mid, :])

    # Interpolate to Ghia locations
    u_interp = zeros(length(y_ghia))
    for (k, yg) in enumerate(y_ghia)
        if yg <= 0.0
            u_interp[k] = u_profile[1]
        elseif yg >= 1.0
            u_interp[k] = u_profile[N]
        else
            j_float = yg / dx + 1.0
            j_lo = clamp(floor(Int, j_float), 1, N)
            j_hi = clamp(j_lo + 1, 1, N)
            w = j_float - floor(j_float)
            u_interp[k] = (1 - w) * u_profile[j_lo] + w * u_profile[j_hi]
        end
    end

    l2_error = norm(u_interp .- u_ghia) / norm(u_ghia)
    @printf("  L2 relative error vs Ghia: %.2f%%\n", l2_error * 100)

    # --- Metal timing ---
    has_metal = false
    t_metal = NaN
    try
        @eval using Metal
        if Metal.functional()
            has_metal = true
            # Warm up
            run_cavity(N=16, Re=100.0, max_steps=10, backend=Metal.MetalBackend(), float_type=Float32)
            t_metal = @elapsed begin
                run_cavity(N=N, Re=100.0, cfl=0.2, max_steps=20000, tol=1e-7f0,
                           verbose=false, backend=Metal.MetalBackend(), float_type=Float32)
            end
        end
    catch e
        @warn "Metal not available: $e"
    end

    speedup = has_metal ? round(t_cpu / t_metal, digits=1) : NaN
    @printf("  Timing (N=%d): CPU-explicit=%.3fs, CPU-implicit=%.3fs, Metal=%s, Speedup=%s\n",
            N, t_cpu, t_cpu_implicit,
            has_metal ? @sprintf("%.3fs", t_metal) : "N/A",
            has_metal ? @sprintf("%.1fx", speedup) : "N/A")

    # --- Figures ---
    if save_figures
        mkpath(figdir)

        u_arr = Array(u)
        v_arr = Array(v)

        # (a) u(y) vs Ghia
        fig1 = Figure(size=(800, 600))
        ax1 = Axis(fig1[1, 1], xlabel="u", ylabel="y",
                    title="Lid-Driven Cavity Re=100 — u(y) at x=0.5")
        lines!(ax1, u_profile, collect(y_grid), linewidth=2, label="Kraken (N=$N)")
        scatter!(ax1, u_ghia, y_ghia, markersize=8, color=:red, label="Ghia et al. 1982")
        axislegend(ax1, position=:lb)
        save(joinpath(figdir, "lid_cavity_u_profile.png"), fig1)

        # (b) Velocity magnitude contour
        vel_mag = sqrt.(u_arr .^ 2 .+ v_arr .^ 2)
        x_plot = range(0.0, 1.0, length=N)
        fig2 = Figure(size=(800, 600))
        ax2 = Axis(fig2[1, 1], xlabel="x", ylabel="y",
                    title="Lid-Driven Cavity — Velocity magnitude", aspect=DataAspect())
        hm = heatmap!(ax2, x_plot, x_plot, vel_mag, colormap=:viridis)
        Colorbar(fig2[1, 2], hm)
        save(joinpath(figdir, "lid_cavity_velocity.png"), fig2)

        # (c) Velocity vectors (subsampled)
        fig3 = Figure(size=(800, 600))
        ax3 = Axis(fig3[1, 1], xlabel="x", ylabel="y",
                    title="Lid-Driven Cavity — Velocity vectors", aspect=DataAspect())
        heatmap!(ax3, x_plot, x_plot, vel_mag, colormap=:viridis, colorrange=(0, maximum(vel_mag)))
        # Subsample for arrows
        skip = max(1, N ÷ 16)
        xs = collect(x_plot)[1:skip:N]
        ys = collect(x_plot)[1:skip:N]
        us = u_arr[1:skip:N, 1:skip:N]
        vs = v_arr[1:skip:N, 1:skip:N]
        arrows!(ax3, repeat(xs, outer=length(ys)),
                repeat(ys, inner=length(xs)),
                vec(us), vec(vs),
                arrowsize=8, lengthscale=0.05, color=:white)
        save(joinpath(figdir, "lid_cavity_vectors.png"), fig3)

        println("  Figures saved to $figdir/lid_cavity_*.png")
    end

    return Dict(
        "name" => "lid_cavity",
        "l2_error" => l2_error,
        "converged" => converged,
        "t_cpu" => t_cpu,
        "t_cpu_implicit" => t_cpu_implicit,
        "t_metal" => t_metal,
        "has_metal" => has_metal
    )
end
