# Newtonian cylinder K with Poiseuille inlet profile vs plug flow.
# Verifies that K converges to Hulsen 2005 value (132.36) when using
# developed Poiseuille inflow instead of uniform plug flow.
#
# Poiseuille profile: u(y) = (3/2) u_mean · 4y(H-y)/H²
# with u_mean = u_in, so Re = u_mean·D/ν is preserved.
#
# Usage: julia --project=. hpc/newtonian_K_poiseuille_inlet.jl

using Kraken, Printf, CUDA, KernelAbstractions

backend = CUDABackend()
FT = Float64

function run_cylinder_poiseuille_inlet(; Nx, Ny, radius, u_mean, ν,
                                         max_steps, avg_window,
                                         backend, FT)
    cx = Nx ÷ 4
    cy = Ny ÷ 2
    D = 2 * radius

    state, config = initialize_cylinder_2d(; Nx=Nx, Ny=Ny, cx=cx, cy=cy,
                                             radius=radius, u_in=u_mean, ν=ν,
                                             backend=backend, T=FT)
    f_in, f_out = state.f_in, state.f_out
    ρ, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid
    ω = FT(1.0 / (3.0 * ν + 0.5))

    # Poiseuille velocity profile: u(j) = (3/2)·u_mean · 4·y·(H-y)/H²
    # with y = j - 0.5 (half-way BB convention), H = Ny
    H = FT(Ny)
    u_profile_cpu = zeros(FT, Ny)
    for j in 1:Ny
        y = FT(j) - FT(0.5)
        u_profile_cpu[j] = FT(1.5) * u_mean * FT(4) * y * (H - y) / (H * H)
    end
    u_profile = KernelAbstractions.allocate(backend, FT, Ny)
    copyto!(u_profile, u_profile_cpu)

    # Re-init f to equilibrium with Poiseuille profile
    f_cpu = zeros(FT, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx
        u_loc = u_profile_cpu[j]
        for q in 1:9
            f_cpu[i,j,q] = Kraken.equilibrium(D2Q9(), one(FT), FT(u_loc), zero(FT), q)
        end
    end
    copyto!(f_in, f_cpu); copyto!(f_out, f_cpu)

    Fx_sum = 0.0; Fy_sum = 0.0; n_avg = 0

    for step in 1:max_steps
        stream_2d!(f_out, f_in, Nx, Ny)
        apply_zou_he_west_profile_2d!(f_out, u_profile, Nx, Ny)
        apply_zou_he_pressure_east_2d!(f_out, Nx, Ny)

        if step > max_steps - avg_window
            drag = compute_drag_mea_2d(f_in, f_out, is_solid, Nx, Ny)
            Fx_sum += drag.Fx; Fy_sum += drag.Fy; n_avg += 1
        end

        collide_2d!(f_out, is_solid, ω)
        compute_macroscopic_2d!(ρ, ux, uy, f_out)
        f_in, f_out = f_out, f_in
    end

    Fx_avg = Fx_sum / n_avg
    Re = u_mean * D / ν
    Cd = 2.0 * Fx_avg / (1.0 * u_mean^2 * D)
    K = Cd * Re / 2
    return (; Cd, K, Re, Fx=Fx_avg)
end

println("="^70)
println("Newtonian K: Poiseuille inlet vs plug flow (B=0.5)")
println("Backend: $(typeof(backend)), GPU: $(CUDA.name(CUDA.device()))")
println("="^70)

u_in = 0.02; Re = 1.0

@printf("\n%-6s %-5s %-6s %-10s %-10s %-10s %-10s\n",
        "R", "Ny", "Nx", "K_plug", "K_pois", "K_lit", "err_pois%")
println("-"^70)

for R in [8, 16, 24, 32, 48]
    D = 2R; Ny = 4R; Nx = 20*D
    ν = u_in * D / Re
    steps = min(500_000, max(100_000, round(Int, 100_000 * (R/8)^2)))
    avg = steps ÷ 5

    # Plug flow reference
    t0 = time()
    r_plug = run_cylinder_2d(; Nx=Nx, Ny=Ny, radius=R, u_in=u_in, ν=ν,
                               max_steps=steps, avg_window=avg,
                               backend=backend, T=FT)
    K_plug = r_plug.Cd * Re / 2

    # Poiseuille inlet
    r_pois = run_cylinder_poiseuille_inlet(; Nx=Nx, Ny=Ny, radius=R, u_mean=u_in, ν=ν,
                                              max_steps=steps, avg_window=avg,
                                              backend=backend, FT=FT)
    dt = time() - t0

    K_lit = 132.36
    err = (r_pois.K - K_lit) / K_lit * 100

    @printf("%-6d %-5d %-6d %-10.3f %-10.3f %-10.3f %-10.2f  (%.0fs)\n",
            R, Ny, Nx, K_plug, r_pois.K, K_lit, err, dt)
end

println("\nLiterature: K(Re→0, B=0.5) = 132.36 (Hulsen 2005)")
println("Done.")
