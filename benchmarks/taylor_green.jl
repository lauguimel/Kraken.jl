"""
    run_taylor_green(; save_figures=true, figdir="docs/src/assets/figures")

Taylor-Green vortex 2D benchmark with periodic BCs and exact solution.
Domain [0,2π]², ν=0.01, t_final=1.0.
"""
function run_taylor_green(; save_figures=true, figdir="docs/src/assets/figures")
    println("=" ^ 60)
    println("Benchmark: Taylor-Green Vortex 2D")
    println("=" ^ 60)

    ν = 0.01
    t_final = 1.0
    grids = [32, 64, 128]
    errors = Float64[]

    """Apply periodic BCs by copying opposite interior edges."""
    function apply_periodic_bc!(f, N)
        # f is N×N, with points 1..N representing a periodic domain
        # Copy: f[1,:] = f[N-1,:], f[N,:] = f[2,:]  (ghost = opposite interior)
        # Actually for periodic: first and last are the same point conceptually,
        # but with our operators expecting padding, we wrap:
        f[1, :] .= @view f[N-1, :]
        f[N, :] .= @view f[2, :]
        f[:, 1] .= @view f[:, N-1]
        f[:, N] .= @view f[:, 2]
        return f
    end

    local u_last, v_last, N_last, x_last

    for N in grids
        # Periodic domain [0, 2π] with N points (N-2 interior + 2 ghost)
        # Actually: use N interior points + 2 ghost = N+2 total
        Nt = N + 2  # total grid size with ghost cells
        L = 2π
        dx = L / N  # spacing between interior points

        # Grid: ghost cells at index 1 and Nt, interior at 2:Nt-1
        x = [(i - 1.5) * dx for i in 1:Nt]  # centered at ghost
        y = [(j - 1.5) * dx for j in 1:Nt]

        # IC
        u = [cos(x[i]) * sin(y[j]) for i in 1:Nt, j in 1:Nt]
        v = [-sin(x[i]) * cos(y[j]) for i in 1:Nt, j in 1:Nt]
        p = zeros(Nt, Nt)

        # Work arrays
        adv_u = zeros(Nt, Nt)
        adv_v = zeros(Nt, Nt)
        lap_u = zeros(Nt, Nt)
        lap_v = zeros(Nt, Nt)
        div_f = zeros(Nt, Nt)
        gx = zeros(Nt, Nt)
        gy = zeros(Nt, Nt)
        phi = zeros(N, N)  # FFT solver works on interior

        # Time stepping
        dt_adv = 0.2 * dx
        dt_vis = 0.2 * dx^2 / ν
        dt = min(dt_adv, dt_vis)
        nsteps = ceil(Int, t_final / dt)
        dt = t_final / nsteps

        apply_periodic_bc!(u, Nt)
        apply_periodic_bc!(v, Nt)

        for _ in 1:nsteps
            # Advection
            fill!(adv_u, 0.0)
            fill!(adv_v, 0.0)
            advect!(adv_u, u, v, u, dx)
            advect!(adv_v, u, v, v, dx)

            # Diffusion
            fill!(lap_u, 0.0)
            fill!(lap_v, 0.0)
            laplacian!(lap_u, u, dx)
            laplacian!(lap_v, v, dx)

            # Euler explicit: u* = u + dt*(-adv + ν*lap)
            for j in 2:Nt-1, i in 2:Nt-1
                u[i, j] += dt * (-adv_u[i, j] + ν * lap_u[i, j])
                v[i, j] += dt * (-adv_v[i, j] + ν * lap_v[i, j])
            end

            apply_periodic_bc!(u, Nt)
            apply_periodic_bc!(v, Nt)

            # Pressure correction (periodic Poisson)
            fill!(div_f, 0.0)
            divergence!(div_f, u, v, dx)

            # Extract interior for FFT solver
            rhs_interior = div_f[2:Nt-1, 2:Nt-1] ./ dt
            fill!(phi, 0.0)
            solve_poisson_fft!(phi, rhs_interior, dx)

            # Write back to padded array and compute gradient
            p[2:Nt-1, 2:Nt-1] .= phi
            apply_periodic_bc!(p, Nt)

            fill!(gx, 0.0)
            fill!(gy, 0.0)
            gradient!(gx, gy, p, dx)

            # Velocity correction
            for j in 2:Nt-1, i in 2:Nt-1
                u[i, j] -= dt * gx[i, j]
                v[i, j] -= dt * gy[i, j]
            end

            apply_periodic_bc!(u, Nt)
            apply_periodic_bc!(v, Nt)
        end

        # Exact solution at t_final
        decay = exp(-2ν * t_final)
        u_exact = [cos(x[i]) * sin(y[j]) * decay for i in 1:Nt, j in 1:Nt]
        v_exact = [-sin(x[i]) * cos(y[j]) * decay for i in 1:Nt, j in 1:Nt]

        # Error on interior
        u_int = u[2:Nt-1, 2:Nt-1]
        ue_int = u_exact[2:Nt-1, 2:Nt-1]
        v_int = v[2:Nt-1, 2:Nt-1]
        ve_int = v_exact[2:Nt-1, 2:Nt-1]
        err = sqrt(norm(u_int .- ue_int)^2 + norm(v_int .- ve_int)^2) /
              sqrt(norm(ue_int)^2 + norm(ve_int)^2)
        push!(errors, err)
        @printf("  N=%3d: L2 relative error = %.2e\n", N, err)

        if N == grids[end]
            u_last = u
            v_last = v
            N_last = Nt
            x_last = x
        end
    end

    # --- Timing on 128 ---
    N = 128
    Nt = N + 2
    L = 2π
    dx = L / N

    function apply_periodic!(f, Ntot)
        f[1, :] .= @view f[Ntot-1, :]
        f[Ntot, :] .= @view f[2, :]
        f[:, 1] .= @view f[:, Ntot-1]
        f[:, Ntot] .= @view f[:, 2]
    end

    t_cpu = @elapsed begin
        x = [(i - 1.5) * dx for i in 1:Nt]
        y = [(j - 1.5) * dx for j in 1:Nt]
        u_t = [cos(x[i]) * sin(y[j]) for i in 1:Nt, j in 1:Nt]
        v_t = [-sin(x[i]) * cos(y[j]) for i in 1:Nt, j in 1:Nt]
        p_t = zeros(Nt, Nt)
        au = zeros(Nt, Nt); av = zeros(Nt, Nt)
        lu = zeros(Nt, Nt); lv = zeros(Nt, Nt)
        df = zeros(Nt, Nt); gx_t = zeros(Nt, Nt); gy_t = zeros(Nt, Nt)
        phi_t = zeros(N, N)

        dt = min(0.2 * dx, 0.2 * dx^2 / ν)
        nsteps = ceil(Int, t_final / dt)
        dt = t_final / nsteps
        apply_periodic!(u_t, Nt); apply_periodic!(v_t, Nt)

        for _ in 1:nsteps
            fill!(au, 0.0); fill!(av, 0.0)
            advect!(au, u_t, v_t, u_t, dx); advect!(av, u_t, v_t, v_t, dx)
            fill!(lu, 0.0); fill!(lv, 0.0)
            laplacian!(lu, u_t, dx); laplacian!(lv, v_t, dx)
            for j in 2:Nt-1, i in 2:Nt-1
                u_t[i, j] += dt * (-au[i, j] + ν * lu[i, j])
                v_t[i, j] += dt * (-av[i, j] + ν * lv[i, j])
            end
            apply_periodic!(u_t, Nt); apply_periodic!(v_t, Nt)
            fill!(df, 0.0); divergence!(df, u_t, v_t, dx)
            rhs = df[2:Nt-1, 2:Nt-1] ./ dt
            fill!(phi_t, 0.0); solve_poisson_fft!(phi_t, rhs, dx)
            p_t[2:Nt-1, 2:Nt-1] .= phi_t; apply_periodic!(p_t, Nt)
            fill!(gx_t, 0.0); fill!(gy_t, 0.0); gradient!(gx_t, gy_t, p_t, dx)
            for j in 2:Nt-1, i in 2:Nt-1
                u_t[i, j] -= dt * gx_t[i, j]; v_t[i, j] -= dt * gy_t[i, j]
            end
            apply_periodic!(u_t, Nt); apply_periodic!(v_t, Nt)
        end
    end

    has_metal = false
    t_metal = NaN
    # Metal timing would require rewriting the loop with GPU arrays — skip for this benchmark
    # since the FFT solver falls back to CPU for Metal anyway

    @printf("  Timing (128+2 grid): CPU=%.3fs, Metal=N/A (FFT CPU fallback)\n", t_cpu)

    # --- Figures ---
    if save_figures
        mkpath(figdir)

        Nt_plot = N_last
        x_int = x_last[2:Nt_plot-1]

        # (a) Vorticity field
        u_int = u_last[2:Nt_plot-1, 2:Nt_plot-1]
        v_int = v_last[2:Nt_plot-1, 2:Nt_plot-1]
        Ni = Nt_plot - 2
        dx_p = x_int[2] - x_int[1]
        # Compute vorticity = dv/dx - du/dy (central diff on interior)
        ω = zeros(Ni, Ni)
        for j in 2:Ni-1, i in 2:Ni-1
            dvdx = (v_int[i+1, j] - v_int[i-1, j]) / (2dx_p)
            dudy = (u_int[i, j+1] - u_int[i, j-1]) / (2dx_p)
            ω[i, j] = dvdx - dudy
        end

        fig1 = Figure(size=(800, 600))
        ax1 = Axis(fig1[1, 1], xlabel="x", ylabel="y",
                    title="Taylor-Green — Vorticity at t=$t_final", aspect=DataAspect())
        hm = heatmap!(ax1, x_int, x_int, ω, colormap=:RdBu)
        Colorbar(fig1[1, 2], hm)
        save(joinpath(figdir, "taylor_green_vorticity.png"), fig1)

        # (b) Convergence plot
        fig2 = Figure(size=(800, 600))
        ax2 = Axis(fig2[1, 1], xlabel="N", ylabel="L2 relative error",
                    title="Taylor-Green — Convergence",
                    xscale=log10, yscale=log10)
        scatterlines!(ax2, Float64.(grids), errors, linewidth=2, markersize=10, label="Kraken")
        ref = errors[1] .* (grids[1] ./ grids) .^ 1  # first-order upwind
        lines!(ax2, Float64.(grids), ref, linestyle=:dash, color=:gray, label="O(dx)")
        axislegend(ax2)
        save(joinpath(figdir, "taylor_green_convergence.png"), fig2)

        # (c) Velocity decay at center vs exact
        # Re-run with tracking
        N_track = 64
        Nt_tr = N_track + 2
        dx_tr = L / N_track
        x_tr = [(i - 1.5) * dx_tr for i in 1:Nt_tr]
        y_tr = [(j - 1.5) * dx_tr for j in 1:Nt_tr]
        u_tr = [cos(x_tr[i]) * sin(y_tr[j]) for i in 1:Nt_tr, j in 1:Nt_tr]
        v_tr = [-sin(x_tr[i]) * cos(y_tr[j]) for i in 1:Nt_tr, j in 1:Nt_tr]
        p_tr = zeros(Nt_tr, Nt_tr)
        au_tr = zeros(Nt_tr, Nt_tr); av_tr = zeros(Nt_tr, Nt_tr)
        lu_tr = zeros(Nt_tr, Nt_tr); lv_tr = zeros(Nt_tr, Nt_tr)
        df_tr = zeros(Nt_tr, Nt_tr); gx_tr = zeros(Nt_tr, Nt_tr); gy_tr = zeros(Nt_tr, Nt_tr)
        phi_tr = zeros(N_track, N_track)

        dt_tr = min(0.2 * dx_tr, 0.2 * dx_tr^2 / ν)
        nsteps_tr = ceil(Int, t_final / dt_tr)
        dt_tr = t_final / nsteps_tr
        apply_periodic!(u_tr, Nt_tr); apply_periodic!(v_tr, Nt_tr)

        record_every = max(1, nsteps_tr ÷ 50)
        times_rec = Float64[]
        umax_rec = Float64[]

        for step in 1:nsteps_tr
            fill!(au_tr, 0.0); fill!(av_tr, 0.0)
            advect!(au_tr, u_tr, v_tr, u_tr, dx_tr); advect!(av_tr, u_tr, v_tr, v_tr, dx_tr)
            fill!(lu_tr, 0.0); fill!(lv_tr, 0.0)
            laplacian!(lu_tr, u_tr, dx_tr); laplacian!(lv_tr, v_tr, dx_tr)
            for j in 2:Nt_tr-1, i in 2:Nt_tr-1
                u_tr[i, j] += dt_tr * (-au_tr[i, j] + ν * lu_tr[i, j])
                v_tr[i, j] += dt_tr * (-av_tr[i, j] + ν * lv_tr[i, j])
            end
            apply_periodic!(u_tr, Nt_tr); apply_periodic!(v_tr, Nt_tr)
            fill!(df_tr, 0.0); divergence!(df_tr, u_tr, v_tr, dx_tr)
            rhs_tr = df_tr[2:Nt_tr-1, 2:Nt_tr-1] ./ dt_tr
            fill!(phi_tr, 0.0); solve_poisson_fft!(phi_tr, rhs_tr, dx_tr)
            p_tr[2:Nt_tr-1, 2:Nt_tr-1] .= phi_tr; apply_periodic!(p_tr, Nt_tr)
            fill!(gx_tr, 0.0); fill!(gy_tr, 0.0); gradient!(gx_tr, gy_tr, p_tr, dx_tr)
            for j in 2:Nt_tr-1, i in 2:Nt_tr-1
                u_tr[i, j] -= dt_tr * gx_tr[i, j]; v_tr[i, j] -= dt_tr * gy_tr[i, j]
            end
            apply_periodic!(u_tr, Nt_tr); apply_periodic!(v_tr, Nt_tr)

            if step % record_every == 0
                push!(times_rec, step * dt_tr)
                push!(umax_rec, maximum(abs.(u_tr[2:Nt_tr-1, 2:Nt_tr-1])))
            end
        end

        fig3 = Figure(size=(800, 600))
        ax3 = Axis(fig3[1, 1], xlabel="t", ylabel="max |u|",
                    title="Taylor-Green — Velocity decay")
        lines!(ax3, times_rec, umax_rec, linewidth=2, label="Kraken")
        t_exact = range(0, t_final, length=100)
        lines!(ax3, collect(t_exact), exp.(-2ν .* t_exact), linestyle=:dash,
               color=:red, linewidth=2, label="Exact: exp(-2νt)")
        axislegend(ax3)
        save(joinpath(figdir, "taylor_green_decay.png"), fig3)

        println("  Figures saved to $figdir/taylor_green_*.png")
    end

    return Dict(
        "name" => "taylor_green",
        "errors" => errors,
        "grids" => grids,
        "t_cpu" => t_cpu,
        "t_metal" => t_metal,
        "has_metal" => has_metal
    )
end
