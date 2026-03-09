"""
    run_advection_diffusion(; save_figures=true, figdir="docs/src/assets/figures")

Combined advection-diffusion benchmark. Domain [0,1]² periodic.
Uniform advection u=1, v=0 with diffusion κ=0.01.
IC: Gaussian bump at (0.25, 0.5), σ=0.1.
Exact: Gaussian centered at (0.25+t mod 1, 0.5) with σ(t) = √(σ₀² + 2κt).
"""
function run_advection_diffusion(; save_figures=true, figdir="docs/src/assets/figures")
    println("=" ^ 60)
    println("Benchmark: Advection-Diffusion")
    println("=" ^ 60)

    N = 128
    Nt = N + 2  # ghost cells
    L = 1.0
    dx = L / N
    κ = 0.01
    σ0 = 0.1
    t_final = 0.2

    # Grid
    x = [(i - 1.5) * dx for i in 1:Nt]
    y = [(j - 1.5) * dx for j in 1:Nt]

    # Uniform velocity
    u_vel = ones(Nt, Nt)
    v_vel = zeros(Nt, Nt)

    # IC: Gaussian at (0.25, 0.5)
    x0, y0 = 0.25, 0.5
    φ = [exp(-((x[i] - x0)^2 + (y[j] - y0)^2) / (2σ0^2)) for i in 1:Nt, j in 1:Nt]

    function apply_periodic!(f, Ntot)
        f[1, :] .= @view f[Ntot-1, :]
        f[Ntot, :] .= @view f[2, :]
        f[:, 1] .= @view f[:, Ntot-1]
        f[:, Ntot] .= @view f[:, 2]
    end

    # Time stepping
    dt_adv = 0.5 * dx / 1.0  # CFL for u=1
    dt_vis = 0.2 * dx^2 / κ
    dt = min(dt_adv, dt_vis)
    nsteps = ceil(Int, t_final / dt)
    dt = t_final / nsteps

    adv_out = zeros(Nt, Nt)
    lap_out = zeros(Nt, Nt)

    apply_periodic!(φ, Nt)
    apply_periodic!(u_vel, Nt)
    apply_periodic!(v_vel, Nt)

    t_cpu = @elapsed begin
        for _ in 1:nsteps
            fill!(adv_out, 0.0)
            fill!(lap_out, 0.0)
            advect!(adv_out, u_vel, v_vel, φ, dx)
            laplacian!(lap_out, φ, dx)
            # φ^{n+1} = φ^n + dt*(-u·∇φ + κ∇²φ)
            for j in 2:Nt-1, i in 2:Nt-1
                φ[i, j] += dt * (-adv_out[i, j] + κ * lap_out[i, j])
            end
            apply_periodic!(φ, Nt)
        end
    end

    # Exact solution: Gaussian at (x0 + t_final, y0) with σ(t) = √(σ0² + 2κt)
    # With periodic wrapping
    xc = x0 + t_final  # center after advection (might need mod)
    σ_t = sqrt(σ0^2 + 2κ * t_final)
    # Periodic distance
    function periodic_dist(a, b, L)
        d = abs(a - b)
        return min(d, L - d)
    end
    φ_exact = [(σ0 / σ_t) * exp(-(periodic_dist(x[i], xc, L)^2 + periodic_dist(y[j], y0, L)^2) / (2σ_t^2))
               for i in 1:Nt, j in 1:Nt]
    # Normalize: the 2D Gaussian amplitude scales as (σ0/σ_t)² but we used 1D scaling per dim
    # Actually: A(t)/A(0) = (σ0/σ_t)² for 2D Gaussian, but our IC peak is 1
    # Correct: peak at t = (σ0²/σ_t²) = σ0²/(σ0²+2κt)
    φ_exact_corrected = [(σ0^2 / σ_t^2) * exp(-(periodic_dist(x[i], xc, L)^2 + periodic_dist(y[j], y0, L)^2) / (2σ_t^2))
                         for i in 1:Nt, j in 1:Nt]

    φ_int = φ[2:Nt-1, 2:Nt-1]
    φe_int = φ_exact_corrected[2:Nt-1, 2:Nt-1]
    err = norm(φ_int .- φe_int) / norm(φe_int)
    @printf("  L2 relative error vs exact: %.2e\n", err)

    # Metal timing
    has_metal = false
    t_metal = NaN
    try
        @eval using Metal
        if Metal.functional()
            has_metal = true
            backend = Metal.MetalBackend()

            φ_g = KernelAbstractions.allocate(backend, Float32, Nt, Nt)
            φ_init_f32 = Float32[exp(-((x[i] - x0)^2 + (y[j] - y0)^2) / (2σ0^2)) for i in 1:Nt, j in 1:Nt]
            copyto!(φ_g, φ_init_f32)
            u_g = KernelAbstractions.allocate(backend, Float32, Nt, Nt)
            copyto!(u_g, Float32.(u_vel))
            v_g = KernelAbstractions.zeros(backend, Float32, Nt, Nt)
            adv_g = KernelAbstractions.zeros(backend, Float32, Nt, Nt)
            lap_g = KernelAbstractions.zeros(backend, Float32, Nt, Nt)

            # Warm up
            advect!(adv_g, u_g, v_g, φ_g, Float32(dx))
            laplacian!(lap_g, φ_g, Float32(dx))

            # Reset
            copyto!(φ_g, φ_init_f32)

            t_metal = @elapsed begin
                for _ in 1:nsteps
                    fill!(adv_g, Float32(0)); fill!(lap_g, Float32(0))
                    advect!(adv_g, u_g, v_g, φ_g, Float32(dx))
                    laplacian!(lap_g, φ_g, Float32(dx))
                    φ_c = Array(φ_g); adv_c = Array(adv_g); lap_c = Array(lap_g)
                    for j in 2:Nt-1, i in 2:Nt-1
                        φ_c[i, j] += Float32(dt) * (-adv_c[i, j] + Float32(κ) * lap_c[i, j])
                    end
                    φ_c[1, :] .= @view φ_c[Nt-1, :]; φ_c[Nt, :] .= @view φ_c[2, :]
                    φ_c[:, 1] .= @view φ_c[:, Nt-1]; φ_c[:, Nt] .= @view φ_c[:, 2]
                    copyto!(φ_g, φ_c)
                end
                KernelAbstractions.synchronize(backend)
            end
        end
    catch e
        @warn "Metal not available: $e"
    end

    speedup = has_metal ? round(t_cpu / t_metal, digits=1) : NaN
    @printf("  Timing (%d+2 grid): CPU=%.3fs, Metal=%s, Speedup=%s\n",
            N, t_cpu,
            has_metal ? @sprintf("%.3fs", t_metal) : "N/A",
            has_metal ? @sprintf("%.1fx", speedup) : "N/A")

    # --- Figures ---
    if save_figures
        mkpath(figdir)
        x_int = x[2:Nt-1]

        # (a) Field at t_final vs exact
        fig1 = Figure(size=(800, 300))
        ax1 = Axis(fig1[1, 1], xlabel="x", ylabel="y",
                    title="Advection-Diffusion — φ at t=$t_final (Kraken)", aspect=DataAspect())
        hm1 = heatmap!(ax1, x_int, x_int, φ_int, colormap=:viridis)
        Colorbar(fig1[1, 2], hm1)
        ax2 = Axis(fig1[1, 3], xlabel="x", ylabel="y",
                    title="Exact", aspect=DataAspect())
        hm2 = heatmap!(ax2, x_int, x_int, φe_int, colormap=:viridis)
        Colorbar(fig1[1, 4], hm2)
        save(joinpath(figdir, "advection_diffusion_field.png"), fig1)

        # (b) Cross-section at y=0.5
        j_mid = N ÷ 2
        fig2 = Figure(size=(800, 300))
        ax3 = Axis(fig2[1, 1], xlabel="x", ylabel="φ",
                    title="Advection-Diffusion — Cross-section at y=0.5")
        lines!(ax3, x_int, φ_int[:, j_mid], linewidth=2, label="Kraken")
        lines!(ax3, x_int, φe_int[:, j_mid], linestyle=:dash, color=:red,
               linewidth=2, label="Exact")
        axislegend(ax3)
        save(joinpath(figdir, "advection_diffusion_cross.png"), fig2)

        println("  Figures saved to $figdir/advection_diffusion_*.png")
    end

    return Dict(
        "name" => "advection_diffusion",
        "l2_error" => err,
        "t_cpu" => t_cpu,
        "t_metal" => t_metal,
        "has_metal" => has_metal
    )
end
