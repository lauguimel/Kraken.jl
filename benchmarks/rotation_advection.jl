"""
    run_rotation_advection(; save_figures=true, figdir="docs/src/assets/figures")

Solid body rotation of a Gaussian bump. Domain [0,1]² periodic.
Velocity: u = -(y-0.5), v = (x-0.5). After t=2π, bump returns to IC.
"""
function run_rotation_advection(; save_figures=true, figdir="docs/src/assets/figures")
    println("=" ^ 60)
    println("Benchmark: Rotation Advection")
    println("=" ^ 60)

    N = 128
    Nt = N + 2  # with ghost cells for periodic
    L = 1.0
    dx = L / N
    σ = 0.05

    # Grid (interior at indices 2:Nt-1)
    x = [(i - 1.5) * dx for i in 1:Nt]
    y = [(j - 1.5) * dx for j in 1:Nt]

    # Velocity field: solid body rotation around (0.5, 0.5)
    u_vel = [-(y[j] - 0.5) for i in 1:Nt, j in 1:Nt]
    v_vel = [(x[i] - 0.5) for i in 1:Nt, j in 1:Nt]

    # IC: Gaussian bump at (0.25, 0.5)
    x0, y0 = 0.25, 0.5
    φ_init = [exp(-((x[i] - x0)^2 + (y[j] - y0)^2) / (2σ^2)) for i in 1:Nt, j in 1:Nt]

    function apply_periodic!(f, Ntot)
        f[1, :] .= @view f[Ntot-1, :]
        f[Ntot, :] .= @view f[2, :]
        f[:, 1] .= @view f[:, Ntot-1]
        f[:, Ntot] .= @view f[:, 2]
    end

    # CFL condition
    u_max = 0.5  # max velocity ~ 0.5 (at radius 0.5 from center)
    dt = 0.5 * dx / u_max  # CFL < 1
    t_final = 2π
    nsteps = ceil(Int, t_final / dt)
    dt = t_final / nsteps

    φ = copy(φ_init)
    adv_out = zeros(Nt, Nt)

    apply_periodic!(φ, Nt)
    apply_periodic!(u_vel, Nt)
    apply_periodic!(v_vel, Nt)

    t_cpu = @elapsed begin
        for _ in 1:nsteps
            fill!(adv_out, 0.0)
            advect!(adv_out, u_vel, v_vel, φ, dx)
            # Euler: φ^{n+1} = φ^n - dt * (u·∇φ)
            for j in 2:Nt-1, i in 2:Nt-1
                φ[i, j] -= dt * adv_out[i, j]
            end
            apply_periodic!(φ, Nt)
        end
    end

    # Error: compare with IC
    φ_int = φ[2:Nt-1, 2:Nt-1]
    φ0_int = φ_init[2:Nt-1, 2:Nt-1]
    err = norm(φ_int .- φ0_int) / norm(φ0_int)
    @printf("  L2 relative error after 1 rotation: %.2e\n", err)

    # Metal timing
    has_metal = false
    t_metal = NaN
    try
        @eval using Metal
        if Metal.functional()
            has_metal = true
            backend = Metal.MetalBackend()

            φ_g = KernelAbstractions.allocate(backend, Float32, Nt, Nt)
            copyto!(φ_g, Float32.(φ_init))
            u_g = KernelAbstractions.allocate(backend, Float32, Nt, Nt)
            copyto!(u_g, Float32.(u_vel))
            v_g = KernelAbstractions.allocate(backend, Float32, Nt, Nt)
            copyto!(v_g, Float32.(v_vel))
            adv_g = KernelAbstractions.zeros(backend, Float32, Nt, Nt)

            # Warm up
            advect!(adv_g, u_g, v_g, φ_g, Float32(dx))

            # Reset
            copyto!(φ_g, Float32.(φ_init))

            t_metal = @elapsed begin
                for _ in 1:nsteps
                    fill!(adv_g, Float32(0))
                    advect!(adv_g, u_g, v_g, φ_g, Float32(dx))
                    # Update on CPU (small overhead but correct)
                    φ_c = Array(φ_g)
                    adv_c = Array(adv_g)
                    for j in 2:Nt-1, i in 2:Nt-1
                        φ_c[i, j] -= Float32(dt) * adv_c[i, j]
                    end
                    φ_c[1, :] .= @view φ_c[Nt-1, :]
                    φ_c[Nt, :] .= @view φ_c[2, :]
                    φ_c[:, 1] .= @view φ_c[:, Nt-1]
                    φ_c[:, Nt] .= @view φ_c[:, 2]
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

        # (a) Initial condition
        fig1 = Figure(size=(800, 600))
        ax1 = Axis(fig1[1, 1], xlabel="x", ylabel="y",
                    title="Rotation Advection — Initial condition", aspect=DataAspect())
        hm1 = heatmap!(ax1, x_int, x_int, φ0_int, colormap=:viridis)
        Colorbar(fig1[1, 2], hm1)
        save(joinpath(figdir, "rotation_advection_ic.png"), fig1)

        # (b) After 1 rotation
        fig2 = Figure(size=(800, 600))
        ax2 = Axis(fig2[1, 1], xlabel="x", ylabel="y",
                    title="Rotation Advection — After 1 rotation (t=2π)", aspect=DataAspect())
        hm2 = heatmap!(ax2, x_int, x_int, φ_int, colormap=:viridis)
        Colorbar(fig2[1, 2], hm2)
        save(joinpath(figdir, "rotation_advection_final.png"), fig2)

        # (c) Error field
        fig3 = Figure(size=(800, 600))
        ax3 = Axis(fig3[1, 1], xlabel="x", ylabel="y",
                    title="Rotation Advection — Error", aspect=DataAspect())
        hm3 = heatmap!(ax3, x_int, x_int, abs.(φ_int .- φ0_int), colormap=:viridis)
        Colorbar(fig3[1, 2], hm3)
        save(joinpath(figdir, "rotation_advection_error.png"), fig3)

        println("  Figures saved to $figdir/rotation_advection_*.png")
    end

    return Dict(
        "name" => "rotation_advection",
        "l2_error" => err,
        "t_cpu" => t_cpu,
        "t_metal" => t_metal,
        "has_metal" => has_metal
    )
end
