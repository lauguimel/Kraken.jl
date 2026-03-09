"""
    run_poiseuille(; save_figures=true, figdir="docs/src/assets/figures")

Poiseuille channel flow benchmark. Domain [0,L]×[0,H], L=4, H=1.
Periodic in x, walls at y=0 and y=H.
Body force f_x = 8ν drives flow to exact parabolic profile u(y) = 4*y*(1-y).
"""
function run_poiseuille(; save_figures=true, figdir="docs/src/assets/figures")
    println("=" ^ 60)
    println("Benchmark: Poiseuille Flow")
    println("=" ^ 60)

    H = 1.0
    ν = 0.01
    U_max = 1.0
    f_x = 8.0 * ν * U_max / H^2  # body force to get U_max=1

    # Grid: Nx × Ny with ghost-style layout for operators
    # Use Ny points across H, Nx = 4*Ny for L=4
    Ny = 64
    Nx = 4 * Ny  # but our operators assume square grids, so we use Ny×Ny on H×H
    # Simpler approach: use a square domain H×H with periodic x, wall y
    N = Ny
    dx = H / (N - 1)

    # IC
    u = zeros(N, N)
    v = zeros(N, N)
    lap = zeros(N, N)

    # Time stepping: pure diffusion + body force (no advection needed for Poiseuille)
    dt = 0.2 * dx^2 / ν
    t_final = 50.0  # diffusion timescale H²/ν = 100s, need ~5τ for steady state
    nsteps = ceil(Int, t_final / dt)
    dt = t_final / nsteps

    """Apply Poiseuille BCs: periodic x, no-slip walls y=0 and y=H."""
    function apply_poiseuille_bc!(u, v, N)
        # No-slip walls
        u[:, 1] .= 0.0   # bottom
        u[:, N] .= 0.0   # top
        v[:, 1] .= 0.0
        v[:, N] .= 0.0
        # Periodic in x
        u[1, :] .= @view u[N-1, :]
        u[N, :] .= @view u[2, :]
        v[1, :] .= @view v[N-1, :]
        v[N, :] .= @view v[2, :]
        return u, v
    end

    apply_poiseuille_bc!(u, v, N)

    t_cpu = @elapsed begin
        for step in 1:nsteps
            fill!(lap, 0.0)
            laplacian!(lap, u, dx)
            # u^{n+1} = u^n + dt*(ν*∇²u + f_x)
            for j in 2:N-1, i in 2:N-1
                u[i, j] += dt * (ν * lap[i, j] + f_x)
            end
            apply_poiseuille_bc!(u, v, N)
        end
    end

    # Exact solution: u(y) = 4*U_max*y*(H-y)/H²
    y_grid = range(0.0, H, length=N)
    u_exact = [4.0 * U_max * y * (H - y) / H^2 for y in y_grid]

    # Extract profile at mid-x
    i_mid = N ÷ 2
    u_profile = u[i_mid, :]

    err = norm(u_profile .- u_exact) / norm(u_exact)
    @printf("  L2 relative error vs exact: %.2e\n", err)

    # Metal timing
    has_metal = false
    t_metal = NaN
    try
        @eval using Metal
        if Metal.functional()
            has_metal = true
            backend = Metal.MetalBackend()

            u_g = KernelAbstractions.zeros(backend, Float32, N, N)
            v_g = KernelAbstractions.zeros(backend, Float32, N, N)
            lap_g = KernelAbstractions.zeros(backend, Float32, N, N)
            dx32 = Float32(dx)
            dt32 = Float32(dt)
            fx32 = Float32(f_x)
            ν32 = Float32(ν)

            # Warm up
            laplacian!(lap_g, u_g, dx32)

            t_metal = @elapsed begin
                for step in 1:nsteps
                    fill!(lap_g, Float32(0))
                    laplacian!(lap_g, u_g, dx32)
                    # Manual interior update on CPU since we can't easily do it on Metal
                    u_cpu = Array(u_g)
                    lap_cpu = Array(lap_g)
                    for j in 2:N-1, i in 2:N-1
                        u_cpu[i, j] += dt32 * (ν32 * lap_cpu[i, j] + fx32)
                    end
                    u_cpu[:, 1] .= 0f0; u_cpu[:, N] .= 0f0
                    u_cpu[1, :] .= @view u_cpu[N-1, :]; u_cpu[N, :] .= @view u_cpu[2, :]
                    copyto!(u_g, u_cpu)
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

        # (a) u(y) profile vs exact parabola
        fig1 = Figure(size=(800, 300))
        ax1 = Axis(fig1[1, 1], xlabel="u", ylabel="y",
                    title="Poiseuille Flow — u(y) profile")
        lines!(ax1, u_profile, collect(y_grid), linewidth=2, label="Kraken")
        lines!(ax1, u_exact, collect(y_grid), linestyle=:dash, color=:red,
               linewidth=2, label="Exact")
        axislegend(ax1, position=:rb)
        save(joinpath(figdir, "poiseuille_profile.png"), fig1)

        # (b) Error profile
        fig2 = Figure(size=(800, 300))
        ax2 = Axis(fig2[1, 1], xlabel="Error", ylabel="y",
                    title="Poiseuille Flow — Error")
        lines!(ax2, abs.(u_profile .- u_exact), collect(y_grid), linewidth=2, color=:red)
        save(joinpath(figdir, "poiseuille_error.png"), fig2)

        println("  Figures saved to $figdir/poiseuille_*.png")
    end

    return Dict(
        "name" => "poiseuille",
        "l2_error" => err,
        "t_cpu" => t_cpu,
        "t_metal" => t_metal,
        "has_metal" => has_metal
    )
end
