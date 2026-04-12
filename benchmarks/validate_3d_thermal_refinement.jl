# ==========================================================================
# Validation suite: 3D thermal + generic thermal refinement + 3D refinement
#
# Part A — Natural convection 2D: uniform vs refined vs generic-refined (.krk)
#           Reference: De Vahl Davis (1983), Ra=1e3 → Nu=1.118
# Part B — Natural convection 3D: uniform convergence
#           Reference: Fusegi et al. (1991), Ra=1e3 → Nu=1.085
# Part C — Cylinder 2D: uniform Cd vs refined (stability + flow quality)
#           Reference: Schäfer & Turek (1996), Re=20, Cd=5.58
# Part D — 3D refinement: lid-driven cavity with center patch (stability)
# ==========================================================================

using Kraken
using Printf

const NU_REF_2D = 1.118     # De Vahl Davis Ra=1e3
const NU_REF_3D = 1.085     # Fusegi Ra=1e3
const CD_REF    = 5.58       # Schäfer & Turek Re=20

nsteps_natconv(N) = max(5000, round(Int, 8 * N^2))
nsteps_cylinder(Ny) = max(10000, round(Int, 12 * Ny^2))

header(title) = begin
    println("\n" * "="^70)
    println(title)
    println("="^70)
end

# ==========================================================================
# Part A — Natural Convection 2D
# ==========================================================================

function validate_natconv_2d(; Ns=[32, 64, 128])
    header("PART A — NATURAL CONVECTION 2D  (Ra=1e3, Pr=0.71, Nu_ref=$NU_REF_2D)")

    # A.1 Global refinement (uniform grids)
    println("\n  A.1 Uniform (global refinement)")
    @printf("  %5s  %8s  %8s  %7s  %5s\n", "N", "steps", "Nu", "err%", "order")
    errs = Float64[]
    for N in Ns
        r = run_natural_convection_2d(; N=N, Ra=1e3, Pr=0.71,
                                        max_steps=nsteps_natconv(N))
        e = abs(r.Nu - NU_REF_2D) / NU_REF_2D
        push!(errs, e)
        ord = length(errs) > 1 ? log2(errs[end-1] / errs[end]) : NaN
        @printf("  %5d  %8d  %8.4f  %6.2f%%  %5s\n",
                N, nsteps_natconv(N), r.Nu, e*100,
                isnan(ord) ? "  -" : @sprintf("%.2f", ord))
    end

    # A.2 Local refinement (dedicated driver, wall patches)
    println("\n  A.2 Refined — dedicated driver (wall_fraction=0.2, ratio=2)")
    @printf("  %5s  %8s  %8s  %7s  %5s\n", "N_base", "steps", "Nu", "err%", "order")
    errs_r = Float64[]
    for N in Ns[1:min(2, end)]  # 32, 64 only (refined is more expensive)
        r = run_natural_convection_refined_2d(; N=N, Ra=1e3, Pr=0.71,
                max_steps=nsteps_natconv(N), ratio=2, wall_fraction=0.2)
        e = abs(r.Nu - NU_REF_2D) / NU_REF_2D
        push!(errs_r, e)
        ord = length(errs_r) > 1 ? log2(errs_r[end-1] / errs_r[end]) : NaN
        @printf("  %5d  %8d  %8.4f  %6.2f%%  %5s\n",
                N, nsteps_natconv(N), r.Nu, e*100,
                isnan(ord) ? "  -" : @sprintf("%.2f", ord))
    end

    # A.3 Generic thermal refined (.krk path)
    println("\n  A.3 Refined — generic .krk path (Module thermal + Refine)")
    @printf("  %5s  %8s  %8s  %7s  %12s\n", "N_base", "steps", "Nu", "err%", "T_range")
    for N in Ns[1:min(2, end)]
        steps = nsteps_natconv(N)
        setup = Kraken.parse_kraken("""
        Simulation natconv_generic D2Q9
        Module thermal
        Domain L = 1.0 x 1.0  N = $N x $N
        Setup rayleigh = 1e3, prandtl = 0.71
        Boundary west  wall T = 1.0
        Boundary east  wall T = 0.0
        Boundary south wall
        Boundary north wall
        Refine west_wall { region = [0.0, 0.0, 0.2, 1.0], ratio = 2, parent = base }
        Refine east_wall { region = [0.8, 0.0, 1.0, 1.0], ratio = 2, parent = base }
        Run $steps steps
        """)
        r = run_simulation(setup)
        e = abs(r.Nu - NU_REF_2D) / NU_REF_2D
        Tmin, Tmax = extrema(r.Temp)
        @printf("  %5d  %8d  %8.4f  %6.2f%%  [%.3f,%.3f]\n",
                N, steps, r.Nu, e*100, Tmin, Tmax)
    end
end

# ==========================================================================
# Part B — Natural Convection 3D
# ==========================================================================

function validate_natconv_3d(; Ns=[16, 24])
    header("PART B — NATURAL CONVECTION 3D  (Ra=1e3, Pr=0.71, Nu_ref=$NU_REF_3D)")

    println("\n  B.1 Uniform (global refinement)")
    @printf("  %5s  %8s  %8s  %7s  %5s\n", "N", "steps", "Nu", "err%", "order")
    errs = Float64[]
    for N in Ns
        steps = max(5000, round(Int, 12 * N^2))
        r = run_natural_convection_3d(; N=N, Ra=1e3, Pr=0.71, max_steps=steps)
        e = abs(r.Nu - NU_REF_3D) / NU_REF_3D
        push!(errs, e)
        ord = length(errs) > 1 ? log2(errs[end-1] / errs[end]) : NaN
        @printf("  %5d  %8d  %8.4f  %6.2f%%  %5s\n",
                N, steps, r.Nu, e*100,
                isnan(ord) ? "  -" : @sprintf("%.2f", ord))
    end
end

# ==========================================================================
# Part C — Cylinder in channel 2D
# ==========================================================================

function validate_cylinder_2d(; Nys=[40, 80], u_in=0.05)
    header("PART C — CYLINDER IN CHANNEL 2D  (Re=20, D/H≈0.25, Cd_ref=$CD_REF)")

    # C.1 Uniform — quantitative Cd
    println("\n  C.1 Uniform (MEA drag)")
    @printf("  %5s  %5s  %8s  %8s  %7s\n", "Ny", "Nx", "steps", "Cd", "err%")
    for Ny in Nys
        Nx = 4 * Ny
        R = round(Int, 0.125 * Ny)
        nu = u_in * 2R / 20.0
        steps = nsteps_cylinder(Ny)
        r = run_cylinder_2d(; Nx=Nx, Ny=Ny, radius=R, u_in=u_in, ν=nu,
                              max_steps=steps, avg_window=min(2000, steps÷5))
        e = abs(r.Cd - CD_REF) / CD_REF * 100
        @printf("  %5d  %5d  %8d  %8.4f  %6.2f%%\n", Ny, Nx, steps, r.Cd, e)
    end

    # C.2 Refined via .krk (stability + flow quality)
    println("\n  C.2 Refined via .krk (ratio=2 around cylinder)")
    @printf("  %5s  %5s  %8s  %8s  %8s  %12s\n",
            "Ny", "Nx", "steps", "|ux|max", "|uy|max", "ρ_range")
    for Ny in Nys
        Nx = 4 * Ny
        R = round(Int, 0.125 * Ny)
        nu = u_in * 2R / 20.0
        steps = nsteps_cylinder(Ny)
        Lx = Float64(Nx); Ly = Float64(Ny)
        R_phys = Float64(R); cx = 0.2 * Lx; cy = 0.5 * Ly
        margin = 3.0 * R_phys

        setup = Kraken.parse_kraken("""
        Simulation cyl_refined D2Q9
        Domain L = $Lx x $Ly  N = $Nx x $Ny
        Define R = $R_phys
        Define cx = $cx
        Define cy = $cy
        Physics nu = $nu
        Obstacle cylinder { (x - cx)^2 + (y - cy)^2 <= R^2 }
        Boundary west velocity(ux = $u_in, uy = 0)
        Boundary east pressure(rho = 1.0)
        Boundary south wall
        Boundary north wall
        Refine cyl_zone { region = [$(max(0,cx-margin)), $(max(0,cy-margin)), $(min(Lx,cx+2margin)), $(min(Ly,cy+margin))], ratio = 2 }
        Run $steps steps
        """)
        r = run_simulation(setup)
        @printf("  %5d  %5d  %8d  %8.4f  %8.4f  [%.4f,%.4f]\n",
                Ny, Nx, steps,
                maximum(abs, r.ux), maximum(abs, r.uy),
                minimum(r.ρ), maximum(r.ρ))
    end
end

# ==========================================================================
# Part D — 3D Refinement: lid-driven cavity stability
# ==========================================================================

function validate_refinement_3d(; N=16, max_steps=500)
    header("PART D — 3D REFINEMENT  (lid-driven cavity N=$N, ratio=2)")

    ν = 0.1; ω = 1.0 / (3.0 * ν + 0.5); dx = 1.0

    config = LBMConfig(D3Q19(); Nx=N, Ny=N, Nz=N, ν=ν, u_lid=0.1, max_steps=max_steps)
    state = initialize_3d(config, Float64)
    f_in, f_out = state.f_in, state.f_out
    ρ, ux, uy, uz = state.ρ, state.ux, state.uy, state.uz
    is_solid = state.is_solid

    # Center patch (away from walls — no patch BC needed)
    patch = create_patch_3d("center", 1, 2,
        (4.0, 4.0, 4.0, 12.0, 12.0, 12.0), N, N, N, dx, ω, Float64)
    domain = create_refined_domain_3d(N, N, N, dx, ω, [patch])

    compute_macroscopic_3d!(ρ, ux, uy, uz, f_in)
    prolongate_f_rescaled_full_3d!(
        patch.f_in, f_in, ρ, ux, uy, uz,
        patch.ratio, patch.Nx_inner, patch.Ny_inner, patch.Nz_inner,
        patch.n_ghost,
        first(patch.parent_i_range), first(patch.parent_j_range),
        first(patch.parent_k_range), N, N, N, ω, Float64(patch.omega))
    copyto!(patch.f_out, patch.f_in)
    compute_macroscopic_3d!(patch.rho, patch.ux, patch.uy, patch.uz, patch.f_in)

    @printf("\n  Patch: %d×%d×%d (inner %d×%d×%d), ω_fine=%.4f\n",
            patch.Nx, patch.Ny, patch.Nz,
            patch.Nx_inner, patch.Ny_inner, patch.Nz_inner, patch.omega)
    @printf("  %5s  %10s  %10s  %12s  %6s\n",
            "step", "ux_max", "uy_max", "ρ_range", "ok")

    for step in 1:max_steps
        f_in, f_out = advance_refined_step_3d!(domain, f_in, f_out, ρ, ux, uy, uz, is_solid;
            stream_fn=stream_3d!,
            collide_fn=(f, is_s) -> collide_3d!(f, is_s, ω),
            macro_fn=compute_macroscopic_3d!,
            bc_base_fn=(f) -> apply_zou_he_top_3d!(f, 0.1, N, N, N))
        apply_zou_he_top_3d!(f_in, 0.1, N, N, N)
        compute_macroscopic_3d!(ρ, ux, uy, uz, f_in)

        if step % (max_steps ÷ 5) == 0 || step == 1
            ρ_cpu = Array(ρ); ux_cpu = Array(ux); uy_cpu = Array(uy)
            ok = !any(isnan, ρ_cpu) && minimum(ρ_cpu) > 0.5
            @printf("  %5d  %10.6f  %10.6f  [%.4f,%.4f]  %6s\n",
                    step, maximum(abs, ux_cpu), maximum(abs, uy_cpu),
                    minimum(ρ_cpu), maximum(ρ_cpu), ok ? "✓" : "✗")
            if !ok
                println("  *** UNSTABLE — stopping early")
                return false
            end
        end
    end

    println("  → PASSED: $max_steps steps stable")
    return true
end

# ==========================================================================
# Main
# ==========================================================================

function run_validation(; quick=false)
    println("Kraken.jl — Validation: 3D thermal + refinement")
    println("CPU-only, Float64\n")

    if quick
        validate_natconv_2d(; Ns=[32, 64])
        validate_natconv_3d(; Ns=[16])
        validate_cylinder_2d(; Nys=[40])
        validate_refinement_3d(; N=16, max_steps=200)
    else
        validate_natconv_2d(; Ns=[32, 64, 128])
        validate_natconv_3d(; Ns=[16, 24])
        validate_cylinder_2d(; Nys=[40, 80])
        validate_refinement_3d(; N=16, max_steps=500)
    end

    println("\n" * "="^70)
    println("VALIDATION COMPLETE")
    println("="^70)
end

if abspath(PROGRAM_FILE) == @__FILE__
    quick = "--quick" in ARGS
    run_validation(; quick=quick)
end
