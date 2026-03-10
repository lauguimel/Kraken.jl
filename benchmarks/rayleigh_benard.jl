"""
    run_rayleigh_benard_benchmark(; save_figures=true, figdir="docs/src/assets/figures")

Rayleigh-Bénard convection benchmark: Ra=1e4, Pr=0.71, N=64.
Computes Nusselt number and generates temperature contour figure.
"""
function run_rayleigh_benard_benchmark(; save_figures=true, figdir="docs/src/assets/figures")
    println("=" ^ 60)
    println("Benchmark: Rayleigh-Bénard Convection Ra=1e4")
    println("=" ^ 60)

    N = 64
    Ra = 1e4
    Pr = 0.71

    # --- Timed run ---
    local u, v, p, T_field, converged
    t_cpu = NaN
    try
        t_cpu = @elapsed begin
            u, v, p, T_field, converged = run_rayleigh_benard(
                N=N, Ra=Ra, Pr=Pr, max_steps=20000, tol=1e-7, cfl=0.1, verbose=false)
        end
        println("  Converged: $converged")
    catch e
        println("  ERROR: $e")
        println("  Rayleigh-Bénard failed — returning empty results")
        return Dict(
            "name" => "rayleigh_benard",
            "Nu_bottom" => NaN, "Nu_top" => NaN, "Nu_avg" => NaN,
            "converged" => false, "t_cpu" => NaN, "t_metal" => NaN, "has_metal" => false
        )
    end

    # --- Compute Nusselt number ---
    # Nu = -dT/dy at bottom wall (averaged over x), normalized by conductive flux
    # Conductive flux: ΔT/H = 1.0
    dx = 1.0 / (N - 1)
    T_arr = Array(T_field)

    # dT/dy at j=1 (bottom wall): forward difference (T[i,2] - T[i,1]) / dx
    dTdy_bottom = zeros(N)
    for i in 1:N
        dTdy_bottom[i] = (T_arr[i, 2] - T_arr[i, 1]) / dx
    end
    # Average over interior x-points (exclude wall corners)
    Nu_bottom = -sum(dTdy_bottom[2:N-1]) / (N - 2)  # negative because T decreases upward

    # dT/dy at j=N (top wall): backward difference
    dTdy_top = zeros(N)
    for i in 1:N
        dTdy_top[i] = (T_arr[i, N] - T_arr[i, N-1]) / dx
    end
    Nu_top = -sum(dTdy_top[2:N-1]) / (N - 2)

    Nu_avg = (Nu_bottom + Nu_top) / 2.0

    @printf("  Nusselt number (bottom): %.3f\n", Nu_bottom)
    @printf("  Nusselt number (top):    %.3f\n", Nu_top)
    @printf("  Nusselt number (avg):    %.3f\n", Nu_avg)
    @printf("  Reference (Ra=1e4):      ~2.24 (de Vahl Davis 1983)\n")
    @printf("  Timing (N=%d): CPU=%.3fs\n", N, t_cpu)

    # --- Figures ---
    if save_figures
        mkpath(figdir)

        u_arr = Array(u)
        v_arr = Array(v)

        x_plot = range(0.0, 1.0, length=N)
        y_plot = range(0.0, 1.0, length=N)

        # (a) Temperature contour
        fig1 = Figure(size=(900, 600))
        ax1 = Axis(fig1[1, 1], xlabel="x", ylabel="y",
                    title="Rayleigh-Bénard Ra=10⁴ — Temperature field",
                    aspect=DataAspect())
        hm = heatmap!(ax1, x_plot, y_plot, T_arr, colormap=:thermal)
        Colorbar(fig1[1, 2], hm, label="T")
        save(joinpath(figdir, "rayleigh_benard_temperature.png"), fig1)

        # (b) Temperature + velocity vectors
        fig2 = Figure(size=(900, 600))
        ax2 = Axis(fig2[1, 1], xlabel="x", ylabel="y",
                    title="Rayleigh-Bénard Ra=10⁴ — Temperature + velocity",
                    aspect=DataAspect())
        heatmap!(ax2, x_plot, y_plot, T_arr, colormap=:thermal)
        skip = max(1, N ÷ 16)
        xs = collect(x_plot)[1:skip:N]
        ys = collect(y_plot)[1:skip:N]
        us = u_arr[1:skip:N, 1:skip:N]
        vs = v_arr[1:skip:N, 1:skip:N]
        vel_scale = maximum(sqrt.(us .^ 2 .+ vs .^ 2))
        if vel_scale > 0
            arrows!(ax2, repeat(xs, outer=length(ys)),
                    repeat(ys, inner=length(xs)),
                    vec(us), vec(vs),
                    arrowsize=8, lengthscale=0.03, color=:white)
        end
        save(joinpath(figdir, "rayleigh_benard_velocity.png"), fig2)

        println("  Figures saved to $figdir/rayleigh_benard_*.png")
    end

    return Dict(
        "name" => "rayleigh_benard",
        "Nu_bottom" => Nu_bottom,
        "Nu_top" => Nu_top,
        "Nu_avg" => Nu_avg,
        "converged" => converged,
        "t_cpu" => t_cpu,
        "t_metal" => NaN,
        "has_metal" => false
    )
end
