# WP-MESH-5 — Schäfer-Turek 2D-2 (Re=100, vortex shedding) matrix.
#
# This is the SENSITIVITY-CRITICAL benchmark: at Re=100 the cylinder
# wake is unsteady, and lift Cl(t) oscillates at the Strouhal frequency.
# Halfway-BB on a staircase boundary introduces gradient noise that
# inflates Cl_RMS and skews the shedding frequency — quantities that
# do NOT vanish with refinement at fixed mesh quality.
#
# Reference (Schäfer & Turek 1996):
#   Cd_mean ∈ [3.22, 3.24]
#   Cl_max  ∈ [0.99, 1.01]    (peak)  — Cl_RMS ≈ 0.7 - 0.71
#   St      ∈ [0.295, 0.305]
#
# Three baselines on each of three resolutions (D_lu = 20, 40):
#   (A) Cartesian + halfway-BB
#   (B) Cartesian + LI-BB v2
#   (C) gmsh + SLBM + LI-BB v2
#
# Per run we record:
#   - Cd_mean over the last ~5 shedding periods
#   - Cl_RMS  over the same window
#   - St      = peak frequency of Cl(t) FFT × dt / D
using Kraken, KernelAbstractions, Gmsh
using FFTW: rfft, rfftfreq

const Lx, Ly         = 2.2, 0.41
const cx_p, cy_p, R_p = 0.2, 0.2, 0.05
const Re_target       = 100.0
const u_max           = 0.04
const u_mean          = (2/3) * u_max
const Cd_ref          = 3.23
const Cl_ref          = 0.706       # RMS expected from peak ≈ 1.0 / sqrt(2)
const St_ref          = 0.30
T = Float64

backend = KernelAbstractions.CPU()

function _setup_lattice(D_lu)
    dx_ref = 2 * R_p / D_lu
    Nx = round(Int, Lx / dx_ref) + 1
    Ny = round(Int, Ly / dx_ref) + 1
    cx_lu = cx_p / dx_ref
    cy_lu = cy_p / dx_ref
    R_lu  = R_p / dx_ref
    ν     = u_mean * D_lu / Re_target
    return (; Nx, Ny, dx_ref, cx_lu, cy_lu, R_lu, ν)
end

function _inlet_profile(Ny)
    u_prof_h = T[T(u_max) * 4 * (T(j-1) * (T(Ny-1) - T(j-1))) / T(Ny-1)^2 for j in 1:Ny]
    u_prof = KernelAbstractions.allocate(backend, T, Ny); copyto!(u_prof, u_prof_h)
    return u_prof, u_prof_h
end

function _init_f(Nx, Ny, u_prof_h, is_solid_h)
    f_h = zeros(T, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx, q in 1:9
        u = is_solid_h[i, j] ? 0.0 : Float64(u_prof_h[j])
        f_h[i, j, q] = T(Kraken.equilibrium(D2Q9(), 1.0, u, 0.0, q))
    end
    return f_h
end

function _strouhal(cl_history, dt_lu, D_lu)
    # Strip mean, FFT, find dominant frequency
    cl = cl_history .- (sum(cl_history) / length(cl_history))
    F  = abs.(rfft(cl))
    n  = length(cl)
    fs = rfftfreq(n, 1 / dt_lu)
    # Skip DC bin (already removed but be safe)
    idx = argmax(F[2:end]) + 1
    f_peak = fs[idx]
    return f_peak * D_lu / u_mean
end

function _cd_cl_rms(cd_history, cl_history)
    cd_mean = sum(cd_history) / length(cd_history)
    cl_mean = sum(cl_history) / length(cl_history)
    cl_rms  = sqrt(sum((cl_history .- cl_mean).^2) / length(cl_history))
    return (cd_mean, cl_rms)
end

# ------------------------- (A) Cartesian + halfway-BB -------------------------
function run_cart_halfway_bb(D_lu; steps=80_000, sample_every=10, sample_window=40_000)
    s = _setup_lattice(D_lu)
    println("\n[A D=$D_lu] Cart+halfBB  $(s.Nx)×$(s.Ny) ν=$(round(s.ν, digits=4))")
    is_solid_h = zeros(Bool, s.Nx, s.Ny)
    for j in 1:s.Ny, i in 1:s.Nx
        if (i - 1.0 - s.cx_lu)^2 + (j - 1.0 - s.cy_lu)^2 ≤ s.R_lu^2
            is_solid_h[i, j] = true
        end
    end
    q_wall_h = zeros(T, s.Nx, s.Ny, 9)
    cxs = Kraken.velocities_x(D2Q9()); cys = Kraken.velocities_y(D2Q9())
    for j in 1:s.Ny, i in 1:s.Nx
        is_solid_h[i, j] && continue
        for q in 2:9
            in_, jn_ = i + cxs[q], j + cys[q]
            (in_ < 1 || in_ > s.Nx || jn_ < 1 || jn_ > s.Ny) && continue
            is_solid_h[in_, jn_] && (q_wall_h[i, j, q] = 0.5)
        end
    end
    return _step_loop_cart(s, is_solid_h, q_wall_h, steps, sample_every, sample_window)
end

# ------------------------- (B) Cartesian + LI-BB v2 -------------------------
function run_cart_libb(D_lu; steps=80_000, sample_every=10, sample_window=40_000)
    s = _setup_lattice(D_lu)
    println("\n[B D=$D_lu] Cart+LIBB    $(s.Nx)×$(s.Ny) ν=$(round(s.ν, digits=4))")
    q_wall_h, is_solid_h = precompute_q_wall_cylinder(s.Nx, s.Ny, s.cx_lu, s.cy_lu, s.R_lu)
    return _step_loop_cart(s, is_solid_h, q_wall_h, steps, sample_every, sample_window)
end

function _step_loop_cart(s, is_solid_h, q_wall_h, steps, sample_every, sample_window)
    is_solid = KernelAbstractions.allocate(backend, Bool, s.Nx, s.Ny); copyto!(is_solid, is_solid_h)
    q_wall   = KernelAbstractions.allocate(backend, T, s.Nx, s.Ny, 9); copyto!(q_wall, q_wall_h)
    uw_x     = KernelAbstractions.allocate(backend, T, s.Nx, s.Ny, 9); fill!(uw_x, zero(T))
    uw_y     = KernelAbstractions.allocate(backend, T, s.Nx, s.Ny, 9); fill!(uw_y, zero(T))
    u_prof, u_prof_h = _inlet_profile(s.Ny)
    bcspec = BCSpec2D(; west=ZouHeVelocity(u_prof), east=ZouHePressure(one(T)),
                        south=HalfwayBB(), north=HalfwayBB())
    fa = KernelAbstractions.allocate(backend, T, s.Nx, s.Ny, 9); copyto!(fa, _init_f(s.Nx, s.Ny, u_prof_h, is_solid_h))
    fb = similar(fa); fill!(fb, zero(T))
    ρ  = KernelAbstractions.allocate(backend, T, s.Nx, s.Ny); fill!(ρ, one(T))
    ux = KernelAbstractions.allocate(backend, T, s.Nx, s.Ny); fill!(ux, zero(T))
    uy = KernelAbstractions.allocate(backend, T, s.Nx, s.Ny); fill!(uy, zero(T))

    cd_h = Float64[]; cl_h = Float64[]
    sizehint!(cd_h, sample_window ÷ sample_every)
    sizehint!(cl_h, sample_window ÷ sample_every)
    t0 = time()
    norm = u_mean^2 * (s.R_lu * 2)
    for step in 1:steps
        fused_trt_libb_v2_step!(fb, fa, ρ, ux, uy, is_solid, q_wall, uw_x, uw_y, s.Nx, s.Ny, s.ν)
        apply_bc_rebuild_2d!(fb, fa, bcspec, s.ν, s.Nx, s.Ny)
        if step > steps - sample_window && step % sample_every == 0
            Fx, Fy = compute_drag_libb_mei_2d(fb, q_wall, uw_x, uw_y, s.Nx, s.Ny)
            push!(cd_h, 2 * Fx / norm)
            push!(cl_h, 2 * Fy / norm)
        end
        fa, fb = fb, fa
    end
    KernelAbstractions.synchronize(backend)
    elapsed = time() - t0
    cd_mean, cl_rms = _cd_cl_rms(cd_h, cl_h)
    St = _strouhal(cl_h, sample_every, 2 * s.R_lu)
    mlups = s.Nx * s.Ny * steps / elapsed / 1e6
    cells = s.Nx * s.Ny
    err_cd  = 100 * abs(cd_mean - Cd_ref) / Cd_ref
    err_clr = 100 * abs(cl_rms  - Cl_ref) / Cl_ref
    err_st  = 100 * abs(St      - St_ref) / St_ref
    println("  cells=$cells  Cd=$(round(cd_mean,digits=3)) (err $(round(err_cd,digits=2))%)  Cl_RMS=$(round(cl_rms,digits=3)) (err $(round(err_clr,digits=2))%)  St=$(round(St,digits=3)) (err $(round(err_st,digits=2))%)  $(round(mlups,digits=0)) MLUPS")
    return (; cells, cd=cd_mean, cl_rms, St, err_cd, err_clr, err_st, mlups, elapsed)
end

# ------------------------- (C) gmsh + SLBM + LI-BB v2 -------------------------
function run_gmsh_slbm_libb(D_lu; steps=80_000, sample_every=10, sample_window=40_000)
    s = _setup_lattice(D_lu)
    println("\n[C D=$D_lu] gmsh+SLBM+LIBB  $(s.Nx)×$(s.Ny) ν=$(round(s.ν, digits=4))")
    fpath = tempname() * ".msh"
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("st_block")
        gmsh.model.geo.addPoint(0.0, 0.0, 0.0, 0.1, 1)
        gmsh.model.geo.addPoint(Lx,  0.0, 0.0, 0.1, 2)
        gmsh.model.geo.addPoint(Lx,  Ly,  0.0, 0.1, 3)
        gmsh.model.geo.addPoint(0.0, Ly,  0.0, 0.1, 4)
        gmsh.model.geo.addLine(1, 2, 1); gmsh.model.geo.addLine(2, 3, 2)
        gmsh.model.geo.addLine(3, 4, 3); gmsh.model.geo.addLine(4, 1, 4)
        gmsh.model.geo.addCurveLoop([1,2,3,4], 1); gmsh.model.geo.addPlaneSurface([1], 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(1, s.Nx); gmsh.model.geo.mesh.setTransfiniteCurve(3, s.Nx)
        gmsh.model.geo.mesh.setTransfiniteCurve(2, s.Ny); gmsh.model.geo.mesh.setTransfiniteCurve(4, s.Ny)
        gmsh.model.geo.mesh.setTransfiniteSurface(1); gmsh.model.geo.mesh.setRecombine(2, 1)
        gmsh.model.geo.synchronize(); gmsh.model.mesh.generate(2); gmsh.write(fpath)
    finally
        gmsh.finalize()
    end

    mesh, _ = load_gmsh_mesh_2d(fpath)
    geom_h = build_slbm_geometry(mesh)
    geom = transfer_slbm_geometry(geom_h, backend)

    is_solid_h = zeros(Bool, mesh.Nξ, mesh.Nη)
    for j in 1:mesh.Nη, i in 1:mesh.Nξ
        x = mesh.X[i, j]; y = mesh.Y[i, j]
        (x - cx_p)^2 + (y - cy_p)^2 ≤ R_p^2 && (is_solid_h[i, j] = true)
    end
    qw_h, uwx_h, uwy_h = precompute_q_wall_slbm_cylinder_2d(mesh, is_solid_h, cx_p, cy_p, R_p)
    is_solid = KernelAbstractions.allocate(backend, Bool, mesh.Nξ, mesh.Nη); copyto!(is_solid, is_solid_h)
    q_wall   = KernelAbstractions.allocate(backend, T, mesh.Nξ, mesh.Nη, 9); copyto!(q_wall, qw_h)
    uw_x     = KernelAbstractions.allocate(backend, T, mesh.Nξ, mesh.Nη, 9); copyto!(uw_x, uwx_h)
    uw_y     = KernelAbstractions.allocate(backend, T, mesh.Nξ, mesh.Nη, 9); copyto!(uw_y, uwy_h)
    u_prof, u_prof_h = _inlet_profile(mesh.Nη)
    bcspec = BCSpec2D(; west=ZouHeVelocity(u_prof), east=ZouHePressure(one(T)),
                        south=HalfwayBB(), north=HalfwayBB())
    fa = KernelAbstractions.allocate(backend, T, mesh.Nξ, mesh.Nη, 9); copyto!(fa, _init_f(mesh.Nξ, mesh.Nη, u_prof_h, is_solid_h))
    fb = similar(fa); fill!(fb, zero(T))
    ρ  = KernelAbstractions.allocate(backend, T, mesh.Nξ, mesh.Nη); fill!(ρ, one(T))
    ux = KernelAbstractions.allocate(backend, T, mesh.Nξ, mesh.Nη); fill!(ux, zero(T))
    uy = KernelAbstractions.allocate(backend, T, mesh.Nξ, mesh.Nη); fill!(uy, zero(T))

    cd_h = Float64[]; cl_h = Float64[]
    sizehint!(cd_h, sample_window ÷ sample_every)
    sizehint!(cl_h, sample_window ÷ sample_every)
    t0 = time()
    norm = u_mean^2 * (s.R_lu * 2)
    for step in 1:steps
        slbm_trt_libb_step!(fb, fa, ρ, ux, uy, is_solid, q_wall, uw_x, uw_y, geom, s.ν)
        apply_bc_rebuild_2d!(fb, fa, bcspec, s.ν, mesh.Nξ, mesh.Nη)
        if step > steps - sample_window && step % sample_every == 0
            Fx, Fy = compute_drag_libb_mei_2d(fb, q_wall, uw_x, uw_y, mesh.Nξ, mesh.Nη)
            push!(cd_h, 2 * Fx / norm)
            push!(cl_h, 2 * Fy / norm)
        end
        fa, fb = fb, fa
    end
    KernelAbstractions.synchronize(backend)
    elapsed = time() - t0
    cd_mean, cl_rms = _cd_cl_rms(cd_h, cl_h)
    St = _strouhal(cl_h, sample_every, 2 * s.R_lu)
    cells = mesh.Nξ * mesh.Nη
    mlups = cells * steps / elapsed / 1e6
    err_cd  = 100 * abs(cd_mean - Cd_ref) / Cd_ref
    err_clr = 100 * abs(cl_rms  - Cl_ref) / Cl_ref
    err_st  = 100 * abs(St      - St_ref) / St_ref
    println("  cells=$cells  Cd=$(round(cd_mean,digits=3)) (err $(round(err_cd,digits=2))%)  Cl_RMS=$(round(cl_rms,digits=3)) (err $(round(err_clr,digits=2))%)  St=$(round(St,digits=3)) (err $(round(err_st,digits=2))%)  $(round(mlups,digits=0)) MLUPS")
    return (; cells, cd=cd_mean, cl_rms, St, err_cd, err_clr, err_st, mlups, elapsed)
end

# ====================================================================
# Smoke test on the smallest D=20 to verify shedding is captured.
# Full matrix (D=20/40/80 × 3 methods) goes to Aqua.
# ====================================================================
function _print_summary(label, r)
    println(rpad(label, 26),
            " cells=", lpad(r.cells, 8),
            " Cd=", lpad(round(r.cd, digits=3), 6),    " err=", lpad(round(r.err_cd, digits=2), 6), "%",
            " Cl_RMS=", lpad(round(r.cl_rms, digits=3), 6), " err=", lpad(round(r.err_clr, digits=2), 6), "%",
            " St=", lpad(round(r.St, digits=3), 6),    " err=", lpad(round(r.err_st, digits=2), 6), "%",
            " MLUPS=", lpad(round(Int, r.mlups), 4))
end

println("=== WP-MESH-5 — Schäfer-Turek 2D-2 (Re=100) ===")
println("Reference: Cd≈3.23, Cl_RMS≈0.706 (peak≈1.0), St≈0.30\n")

results = Dict{Tuple{String,Int}, NamedTuple}()
for D_lu in (20, 40, 80)
    # Steps scale with resolution: more cells → finer wall layer →
    # the diffusive transient over the cylinder D² grows like D².
    # Sample window kept at 50 % of total to capture ≥ 5 shedding periods.
    steps = D_lu == 20 ? 80_000 :
            D_lu == 40 ? 160_000 :
                          240_000
    sample_window = steps ÷ 2
    sample_every = D_lu == 80 ? 20 : 10
    println("\n--- D_lu = $D_lu  (steps = $steps, sample window = $sample_window) ---")
    try; results[("A", D_lu)] = run_cart_halfway_bb(D_lu; steps=steps, sample_every=sample_every, sample_window=sample_window); catch e; @warn "(A) D=$D_lu failed" exception=e; end
    try; results[("B", D_lu)] = run_cart_libb(D_lu;       steps=steps, sample_every=sample_every, sample_window=sample_window); catch e; @warn "(B) D=$D_lu failed" exception=e; end
    try; results[("C", D_lu)] = run_gmsh_slbm_libb(D_lu;  steps=steps, sample_every=sample_every, sample_window=sample_window); catch e; @warn "(C) D=$D_lu failed" exception=e; end
end

println("\n========= WP-MESH-5 SUMMARY =========")
for D_lu in (20, 40, 80)
    println("\n=== D_lu = $D_lu ===")
    for (k, lab) in (("A","(A) Cart+halfwayBB"),
                     ("B","(B) Cart+LIBB"),
                     ("C","(C) gmsh+SLBM+LIBB"))
        haskey(results, (k, D_lu)) ? _print_summary(lab, results[(k, D_lu)]) :
                                      println(rpad(lab, 26), "  FAILED")
    end
end
