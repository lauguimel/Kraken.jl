"""
    run_couette(; save_figures=true, figdir="docs/src/assets/figures")

Couette flow benchmark: shear flow between two plates.
Domain [0,1]², bottom u=0, top u=1. Steady state: u(y) = y.
"""
function run_couette(; save_figures=true, figdir="docs/src/assets/figures")
    println("=" ^ 60)
    println("Benchmark: Couette Flow")
    println("=" ^ 60)

    N = 64
    dx = 1.0 / (N - 1)
    ν = 0.01

    # IC
    u = zeros(N, N)
    v = zeros(N, N)
    lap = zeros(N, N)

    # BCs: bottom u=0, top u=1, walls v=0, periodic-like x (copy edges)
    function apply_couette_bc!(u, v, N)
        u[:, 1] .= 0.0   # bottom wall: u=0
        u[:, N] .= 1.0   # top wall: u=1
        v[:, 1] .= 0.0
        v[:, N] .= 0.0
        # Copy x-boundaries (zero-gradient in x)
        u[1, :] .= @view u[2, :]
        u[N, :] .= @view u[N-1, :]
        v[1, :] .= @view v[2, :]
        v[N, :] .= @view v[N-1, :]
        return u, v
    end

    apply_couette_bc!(u, v, N)

    dt = 0.2 * dx^2 / ν
    t_final = 50.0  # diffusion timescale H²/ν = 100s, need ~5τ for steady state
    nsteps = ceil(Int, t_final / dt)
    dt = t_final / nsteps

    # Record snapshots
    y_grid = collect(range(0.0, 1.0, length=N))
    i_mid = N ÷ 2
    snapshot_times = [0.5, 2.0, 5.0, 15.0, 50.0]
    snapshots = Dict{Float64, Vector{Float64}}()

    t_cpu = @elapsed begin
        t = 0.0
        for step in 1:nsteps
            fill!(lap, 0.0)
            laplacian!(lap, u, dx)
            for j in 2:N-1, i in 2:N-1
                u[i, j] += dt * ν * lap[i, j]
            end
            apply_couette_bc!(u, v, N)
            t += dt

            # Save snapshots
            for ts in snapshot_times
                if abs(t - ts) < dt / 2 && !haskey(snapshots, ts)
                    snapshots[ts] = copy(u[i_mid, :])
                end
            end
        end
    end

    # Exact steady state
    u_exact = y_grid  # u(y) = y

    u_profile = u[i_mid, :]
    err = norm(u_profile .- u_exact) / norm(u_exact)
    @printf("  L2 relative error vs exact (steady): %.2e\n", err)

    # Metal timing
    has_metal = false
    t_metal = NaN
    try
        @eval using Metal
        if Metal.functional()
            has_metal = true
            backend = Metal.MetalBackend()

            u_g = KernelAbstractions.zeros(backend, Float32, N, N)
            lap_g = KernelAbstractions.zeros(backend, Float32, N, N)
            dx32 = Float32(dx)
            dt32 = Float32(dt)
            ν32 = Float32(ν)

            # Apply BC on CPU then copy
            u_init = zeros(Float32, N, N)
            u_init[:, N] .= 1f0
            u_init[1, :] .= @view u_init[2, :]
            u_init[N, :] .= @view u_init[N-1, :]
            copyto!(u_g, u_init)

            laplacian!(lap_g, u_g, dx32)  # warm up

            t_metal = @elapsed begin
                for step in 1:nsteps
                    fill!(lap_g, Float32(0))
                    laplacian!(lap_g, u_g, dx32)
                    u_c = Array(u_g)
                    lap_c = Array(lap_g)
                    for j in 2:N-1, i in 2:N-1
                        u_c[i, j] += dt32 * ν32 * lap_c[i, j]
                    end
                    u_c[:, 1] .= 0f0; u_c[:, N] .= 1f0
                    u_c[1, :] .= @view u_c[2, :]; u_c[N, :] .= @view u_c[N-1, :]
                    copyto!(u_g, u_c)
                end
                KernelAbstractions.synchronize(backend)
            end
        end
    catch e
        @warn "Metal not available: $e"
    end

    speedup = has_metal ? round(t_cpu / t_metal, digits=1) : NaN
    @printf("  Timing (N=%d): CPU=%.3fs, Metal=%s, Speedup=%s\n",
            N, t_cpu,
            has_metal ? @sprintf("%.3fs", t_metal) : "N/A",
            has_metal ? @sprintf("%.1fx", speedup) : "N/A")

    # --- Figures ---
    if save_figures
        mkpath(figdir)

        # (a) u(y) profiles at several times
        fig1 = Figure(size=(800, 600))
        ax1 = Axis(fig1[1, 1], xlabel="u", ylabel="y",
                    title="Couette Flow — u(y) at various times")
        colors = [:blue, :cyan, :green, :orange, :red]
        for (idx, ts) in enumerate(sort(collect(keys(snapshots))))
            lines!(ax1, snapshots[ts], y_grid, linewidth=2,
                   color=colors[min(idx, length(colors))],
                   label="t=$(ts)")
        end
        lines!(ax1, u_exact, y_grid, linestyle=:dash, color=:black, linewidth=2,
               label="Exact (steady)")
        axislegend(ax1, position=:lt)
        save(joinpath(figdir, "couette_profiles.png"), fig1)

        # (b) Convergence to steady state (error vs time)
        # Re-run quickly tracking error
        u_track = zeros(N, N)
        apply_couette_bc!(u_track, zeros(N, N), N)
        lap_track = zeros(N, N)
        errs_time = Float64[]
        times_track = Float64[]
        for step in 1:nsteps
            fill!(lap_track, 0.0)
            laplacian!(lap_track, u_track, dx)
            for j in 2:N-1, i in 2:N-1
                u_track[i, j] += dt * ν * lap_track[i, j]
            end
            u_track[:, 1] .= 0.0; u_track[:, N] .= 1.0
            u_track[1, :] .= @view u_track[2, :]; u_track[N, :] .= @view u_track[N-1, :]

            if step % max(1, nsteps ÷ 100) == 0
                push!(times_track, step * dt)
                push!(errs_time, norm(u_track[i_mid, :] .- u_exact) / norm(u_exact))
            end
        end

        fig2 = Figure(size=(800, 600))
        ax2 = Axis(fig2[1, 1], xlabel="t", ylabel="L2 relative error",
                    title="Couette Flow — Convergence to steady state",
                    yscale=log10)
        lines!(ax2, times_track, errs_time, linewidth=2)
        save(joinpath(figdir, "couette_convergence.png"), fig2)

        println("  Figures saved to $figdir/couette_*.png")
    end

    return Dict(
        "name" => "couette",
        "l2_error" => err,
        "t_cpu" => t_cpu,
        "t_metal" => t_metal,
        "has_metal" => has_metal
    )
end
