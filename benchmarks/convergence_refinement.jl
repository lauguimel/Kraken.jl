# Grid refinement convergence benchmarks
# 1) Natural convection: Nu vs De Vahl Davis — uniform vs refined
# 2) Cylinder in channel: Cd vs Schäfer-Turek — uniform vs refined
using Kraken
using Printf

# De Vahl Davis (1983) — Nu at Ra=1e3, Pr=0.71
const NU_REF_RA1E3 = 1.118

# Schäfer & Turek (1996) — Cd at Re=20, D/H=0.25
const CD_REF_RE20 = 5.58

"""Diffusive scaling: steps ~ N² to reach steady state."""
nsteps_natconv(N) = max(5000, round(Int, 8 * N^2))

"""Cylinder: need more steps at higher resolution."""
nsteps_cylinder(Ny) = max(10000, round(Int, 12 * Ny^2))

# ===========================================================================
# Part 1 — Natural Convection
# ===========================================================================

function run_natconv_convergence()
    println("\n" * "="^70)
    println("NATURAL CONVECTION CONVERGENCE — Ra=1e3, Pr=0.71")
    println("Reference: De Vahl Davis (1983) Nu = $NU_REF_RA1E3")
    println("="^70)

    # --- Uniform cases ---
    println("\n--- Uniform ---")
    @printf("  %5s   %8s   %8s   %8s   %8s\n",
            "N", "steps", "Nu", "err%", "order")
    @printf("  %5s   %8s   %8s   %8s   %8s\n",
            "-----", "------", "------", "------", "-----")

    Ns_uniform = [32, 64, 128]
    errs_uniform = Float64[]

    for N in Ns_uniform
        steps = nsteps_natconv(N)
        result = run_natural_convection_2d(; N=N, Ra=1e3, Pr=0.71, max_steps=steps)
        err = abs(result.Nu - NU_REF_RA1E3) / NU_REF_RA1E3
        push!(errs_uniform, err)
        order = length(errs_uniform) > 1 ?
            log2(errs_uniform[end-1] / errs_uniform[end]) : NaN
        @printf("  %5d   %8d   %8.4f   %7.2f%%   %5s\n",
                N, steps, result.Nu, err * 100,
                isnan(order) ? "  -" : @sprintf("%.2f", order))
    end

    # --- Refined cases (base N + ratio=2 near walls) ---
    println("\n--- Refined (wall_fraction=0.2, ratio=2) ---")
    @printf("  %5s   %8s   %8s   %8s   %8s   %8s\n",
            "N_base", "N_eff", "steps", "Nu", "err%", "order")
    @printf("  %5s   %8s   %8s   %8s   %8s   %8s\n",
            "------", "------", "------", "------", "------", "-----")

    Ns_refined = [32, 64, 128]
    errs_refined = Float64[]

    for N in Ns_refined
        steps = nsteps_natconv(N)
        result = run_natural_convection_refined_2d(;
            N=N, Ra=1e3, Pr=0.71, max_steps=steps,
            ratio=2, wall_fraction=0.2)
        err = abs(result.Nu - NU_REF_RA1E3) / NU_REF_RA1E3
        push!(errs_refined, err)
        N_eff = N  # base grid; effective is higher near walls
        order = length(errs_refined) > 1 ?
            log2(errs_refined[end-1] / errs_refined[end]) : NaN
        @printf("  %5d   %5d+2x  %8d   %8.4f   %7.2f%%   %5s\n",
                N, N, steps, result.Nu, err * 100,
                isnan(order) ? "  -" : @sprintf("%.2f", order))
    end

    return (Ns_uniform, errs_uniform, Ns_refined, errs_refined)
end

# ===========================================================================
# Part 2 — Cylinder in Channel
# ===========================================================================

function run_cylinder_convergence()
    println("\n" * "="^70)
    println("CYLINDER IN CHANNEL — Re=20, D/H=0.25")
    println("Reference: Schäfer & Turek (1996) Cd ≈ $CD_REF_RE20")
    println("="^70)

    # Parameters: Re = u_in * D / nu, D/H = 0.25
    # With u_in = 0.05, D = 2*R, H = Ny*dx:
    #   R/Ny = 0.125, cx/Lx = 0.2
    u_in = 0.05

    # --- Uniform cases ---
    println("\n--- Uniform ---")
    @printf("  %5s   %5s   %8s   %8s   %8s   %8s\n",
            "Ny", "Nx", "steps", "Cd", "err%", "order")
    @printf("  %5s   %5s   %8s   %8s   %8s   %8s\n",
            "-----", "-----", "------", "------", "------", "-----")

    Nys_uniform = [40, 80, 160]
    errs_cyl_u = Float64[]

    for Ny in Nys_uniform
        Nx = 4 * Ny
        R = round(Int, 0.125 * Ny)
        D = 2 * R
        nu = u_in * D / 20.0  # Re = 20
        steps = nsteps_cylinder(Ny)

        result = run_cylinder_2d(; Nx=Nx, Ny=Ny, radius=R, u_in=u_in, ν=nu,
                                   max_steps=steps, avg_window=min(1000, steps ÷ 5))
        err = abs(result.Cd - CD_REF_RE20) / CD_REF_RE20
        push!(errs_cyl_u, err)
        order = length(errs_cyl_u) > 1 ?
            log2(errs_cyl_u[end-1] / errs_cyl_u[end]) : NaN
        @printf("  %5d   %5d   %8d   %8.4f   %7.2f%%   %5s\n",
                Ny, Nx, steps, result.Cd, err * 100,
                isnan(order) ? "  -" : @sprintf("%.2f", order))
    end

    # --- Refined cases (patch around cylinder) ---
    println("\n--- Refined (patch around cylinder, ratio=2) ---")
    @printf("  %5s   %5s   %8s   %8s   %8s   %8s\n",
            "Ny", "Nx", "steps", "Cd", "err%", "order")
    @printf("  %5s   %5s   %8s   %8s   %8s   %8s\n",
            "-----", "-----", "------", "------", "------", "-----")

    Nys_refined = [40, 80]
    errs_cyl_r = Float64[]

    for Ny in Nys_refined
        Nx = 4 * Ny
        R_cells = round(Int, 0.125 * Ny)
        D = 2 * R_cells
        nu = u_in * D / 20.0
        steps = nsteps_cylinder(Ny)

        # Physical coordinates
        Lx = Float64(Nx) * 1.0  # dx = 1 in lattice units
        Ly = Float64(Ny) * 1.0
        R_phys = Float64(R_cells)
        cx = 0.2 * Lx
        cy = 0.5 * Ly

        # Refine zone: 3D around cylinder
        margin = 4.0 * R_phys
        x0 = max(0.0, cx - margin)
        x1 = min(Lx, cx + 2.0 * margin)
        y0 = max(0.0, cy - margin)
        y1 = min(Ly, cy + margin)

        krk_text = """
        Simulation cylinder_refined D2Q9
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
        Refine cyl_zone { region = [$x0, $y0, $x1, $y1], ratio = 2 }
        Run $steps steps
        """

        setup = Kraken.parse_kraken(krk_text)
        result = run_simulation(setup)

        # Compute drag on coarse grid (restriction updates it)
        ux_cpu = result.ux
        Cd = _estimate_drag_from_velocity(ux_cpu, u_in, D, Nx, Ny)

        err = abs(Cd - CD_REF_RE20) / CD_REF_RE20
        push!(errs_cyl_r, err)
        order = length(errs_cyl_r) > 1 ?
            log2(errs_cyl_r[end-1] / errs_cyl_r[end]) : NaN
        @printf("  %5d   %5d   %8d   %8.3f   %7.2f%%   %5s\n",
                Ny, Nx, steps, Cd, err * 100,
                isnan(order) ? "  -" : @sprintf("%.2f", order))
    end

    return (Nys_uniform, errs_cyl_u, Nys_refined, errs_cyl_r)
end

"""Rough drag estimate from velocity deficit (for cases without MEA access)."""
function _estimate_drag_from_velocity(ux, u_in, D, Nx, Ny)
    # Use the existing cylinder driver for uniform cases;
    # for refined cases we don't have easy MEA access, so we note
    # the Cd from the uniform driver and compare flow patterns.
    return NaN  # placeholder — use run_cylinder_2d for quantitative Cd
end

# ===========================================================================
# Main
# ===========================================================================

function run_refinement_convergence()
    natconv = run_natconv_convergence()
    # cylinder = run_cylinder_convergence()  # enable when drag extraction is ready
    println("\nDone.")
    return natconv
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_refinement_convergence()
end
