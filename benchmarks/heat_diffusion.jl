"""
    run_heat_diffusion(; save_figures=true, figdir="docs/src/assets/figures")

Pure 2D heat diffusion benchmark with exact solution comparison.
Domain [0,1]², Dirichlet BC T=0, IC = sin(πx)sin(πy), κ=0.01.
"""
function run_heat_diffusion(; save_figures=true, figdir="docs/src/assets/figures")
    println("=" ^ 60)
    println("Benchmark: Heat Diffusion 2D")
    println("=" ^ 60)

    κ = 0.01
    t_final = 0.1
    grids = [32, 64, 128]
    errors = Float64[]

    # --- Convergence study ---
    local T_field_last, T_exact_last, N_last, dx_last
    for N in grids
        dx = 1.0 / (N - 1)
        dt = 0.2 * dx^2 / κ  # stability: dt < dx²/(4κ)
        nsteps = ceil(Int, t_final / dt)
        dt = t_final / nsteps  # exact final time

        # Grid coordinates
        x = range(0.0, 1.0, length=N)
        y = range(0.0, 1.0, length=N)

        # IC: sin(πx)sin(πy)
        T_field = [sin(π * x[i]) * sin(π * y[j]) for i in 1:N, j in 1:N]
        lap = zeros(N, N)

        for _ in 1:nsteps
            fill!(lap, 0.0)
            laplacian!(lap, T_field, dx)
            # Euler explicit update (interior only updated by laplacian!)
            T_field .+= dt * κ .* lap
            # Dirichlet BC: T=0 on boundary
            T_field[1, :] .= 0.0
            T_field[N, :] .= 0.0
            T_field[:, 1] .= 0.0
            T_field[:, N] .= 0.0
        end

        # Exact solution
        T_exact = [exp(-2π^2 * κ * t_final) * sin(π * x[i]) * sin(π * y[j])
                   for i in 1:N, j in 1:N]

        err = norm(T_field .- T_exact) / norm(T_exact)
        push!(errors, err)
        @printf("  N=%3d: L2 relative error = %.2e\n", N, err)

        if N == grids[end]
            T_field_last = T_field
            T_exact_last = T_exact
            N_last = N
            dx_last = dx
        end
    end

    # --- Timing on 128×128 ---
    N = 128
    dx = 1.0 / (N - 1)
    dt = 0.2 * dx^2 / κ
    nsteps = ceil(Int, t_final / dt)
    dt = t_final / nsteps

    t_cpu = @elapsed begin
        x = range(0.0, 1.0, length=N)
        y = range(0.0, 1.0, length=N)
        T_run = [sin(π * x[i]) * sin(π * y[j]) for i in 1:N, j in 1:N]
        lap_run = zeros(N, N)
        for _ in 1:nsteps
            fill!(lap_run, 0.0)
            laplacian!(lap_run, T_run, dx)
            T_run .+= dt * κ .* lap_run
            T_run[1, :] .= 0.0; T_run[N, :] .= 0.0
            T_run[:, 1] .= 0.0; T_run[:, N] .= 0.0
        end
    end

    has_metal = false
    t_metal = NaN
    try
        @eval using Metal
        if Metal.functional()
            has_metal = true
            backend = Metal.MetalBackend()
            # Warm up
            T_gpu = KernelAbstractions.zeros(backend, Float32, N, N)
            lap_gpu = KernelAbstractions.zeros(backend, Float32, N, N)
            laplacian!(lap_gpu, T_gpu, Float32(dx))

            x32 = range(Float32(0), Float32(1), length=N)
            y32 = range(Float32(0), Float32(1), length=N)
            T_gpu_run = KernelAbstractions.allocate(backend, Float32, N, N)
            copyto!(T_gpu_run, Float32[sin(Float32(π) * x32[i]) * sin(Float32(π) * y32[j]) for i in 1:N, j in 1:N])
            lap_gpu_run = KernelAbstractions.zeros(backend, Float32, N, N)

            t_metal = @elapsed begin
                for _ in 1:nsteps
                    fill!(lap_gpu_run, Float32(0))
                    laplacian!(lap_gpu_run, T_gpu_run, Float32(dx))
                    T_gpu_run .+= Float32(dt * κ) .* lap_gpu_run
                    T_gpu_run[1, :] .= Float32(0); T_gpu_run[N, :] .= Float32(0)
                    T_gpu_run[:, 1] .= Float32(0); T_gpu_run[:, N] .= Float32(0)
                end
                KernelAbstractions.synchronize(backend)
            end
        end
    catch e
        @warn "Metal not available: $e"
    end

    speedup = has_metal ? round(t_cpu / t_metal, digits=1) : NaN
    @printf("  Timing (128×128): CPU=%.3fs, Metal=%s, Speedup=%s\n",
            t_cpu,
            has_metal ? @sprintf("%.3fs", t_metal) : "N/A",
            has_metal ? @sprintf("%.1fx", speedup) : "N/A")

    # --- Figures ---
    if save_figures
        mkpath(figdir)

        # (a) Convergence plot
        fig1 = Figure(size=(800, 600))
        ax1 = Axis(fig1[1, 1], xlabel="N", ylabel="L2 relative error",
                    title="Heat Diffusion — Convergence",
                    xscale=log10, yscale=log10)
        scatterlines!(ax1, Float64.(grids), errors, linewidth=2, markersize=10, label="Kraken")
        # Reference slope (2nd order)
        ref_errors = errors[1] .* (grids[1] ./ grids) .^ 2
        lines!(ax1, Float64.(grids), ref_errors, linestyle=:dash, color=:gray, label="O(dx²)")
        axislegend(ax1)
        save(joinpath(figdir, "heat_diffusion_convergence.png"), fig1)

        # (b) T field at t_final
        x_plot = range(0.0, 1.0, length=N_last)
        fig2 = Figure(size=(800, 600))
        ax2 = Axis(fig2[1, 1], xlabel="x", ylabel="y",
                    title="Heat Diffusion — T field at t=$(t_final)", aspect=DataAspect())
        hm = heatmap!(ax2, x_plot, x_plot, T_field_last, colormap=:viridis)
        Colorbar(fig2[1, 2], hm)
        save(joinpath(figdir, "heat_diffusion_field.png"), fig2)

        # (c) Error field
        fig3 = Figure(size=(800, 600))
        ax3 = Axis(fig3[1, 1], xlabel="x", ylabel="y",
                    title="Heat Diffusion — Error field", aspect=DataAspect())
        hm3 = heatmap!(ax3, x_plot, x_plot, abs.(T_field_last .- T_exact_last), colormap=:viridis)
        Colorbar(fig3[1, 2], hm3)
        save(joinpath(figdir, "heat_diffusion_error.png"), fig3)

        println("  Figures saved to $figdir/heat_diffusion_*.png")
    end

    return Dict(
        "name" => "heat_diffusion",
        "errors" => errors,
        "grids" => grids,
        "t_cpu" => t_cpu,
        "t_metal" => t_metal,
        "has_metal" => has_metal
    )
end
