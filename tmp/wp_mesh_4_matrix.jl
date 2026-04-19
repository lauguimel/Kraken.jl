# WP-MESH-4 — comparative matrix on Schäfer-Turek 2D-1 (Re=20):
#   (A) Cartesian uniform + halfway-BB only (cylinder = staircase solid)
#   (B) Cartesian uniform + LI-BB v2 (cylinder = sub-cell q_w)
#   (C) gmsh-imported Cartesian + SLBM + LI-BB (same q_w, body via SLBM)
# Same physical setup, same ν. Reports Cd vs Schäfer-Turek 5.58 ± 0.01,
# wall-clock per 1000 steps and per-cell MLUPS.
using Kraken, KernelAbstractions, Gmsh

const Lx, Ly = 2.2, 0.41
const cx_p, cy_p, R_p = 0.2, 0.2, 0.05
const D_lu = 20.0
const dx_ref = 2 * R_p / D_lu      # 0.005
const u_max = 0.04
const u_mean = (2/3) * u_max
const ν_phys = u_mean * D_lu / 20.0     # Re=20 → ν_lu
const ν = ν_phys
const steps = 30_000
const avg_window = 5_000
T = Float64

# Lattice grid sizes
const Nx = round(Int, Lx / dx_ref) + 1   # 441
const Ny = round(Int, Ly / dx_ref) + 1   # 83
const cx_lu = cx_p / dx_ref
const cy_lu = cy_p / dx_ref
const R_lu = R_p / dx_ref

println("Setup: $Nx × $Ny, D_lu=$D_lu, ν=$ν, u_max=$u_max")
println("Cylinder centre = ($cx_lu, $cy_lu)  R = $R_lu lu")

backend = KernelAbstractions.CPU()

function _init_inlet_profile()
    u_prof_h = T[T(u_max) * 4 * (T(j-1) * (T(Ny-1) - T(j-1))) / T(Ny-1)^2 for j in 1:Ny]
    u_prof = KernelAbstractions.allocate(backend, T, Ny); copyto!(u_prof, u_prof_h)
    return u_prof, u_prof_h
end

function _init_f(u_prof_h, is_solid_h)
    f_h = zeros(T, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx, q in 1:9
        u = is_solid_h[i, j] ? 0.0 : Float64(u_prof_h[j])
        f_h[i, j, q] = T(Kraken.equilibrium(D2Q9(), 1.0, u, 0.0, q))
    end
    return f_h
end

function _compute_cd_from_drag(Fx)
    return 2.0 * Fx / (u_mean^2 * D_lu)
end

# ---------- (A) Cartesian uniform + halfway-BB only ----------
function run_A()
    println("\n[A] Cartesian uniform + halfway-BB (no LI-BB)")
    is_solid_h = zeros(Bool, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        if (i-1.0 - cx_lu)^2 + (j-1.0 - cy_lu)^2 ≤ R_lu^2
            is_solid_h[i, j] = true
        end
    end
    q_wall_h = zeros(T, Nx, Ny, 9)   # all zero → halfway-BB fallback
    uwx_h = zeros(T, Nx, Ny, 9); uwy_h = zeros(T, Nx, Ny, 9)
    is_solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny); copyto!(is_solid, is_solid_h)
    q_wall   = KernelAbstractions.allocate(backend, T, Nx, Ny, 9); copyto!(q_wall, q_wall_h)
    uw_x     = KernelAbstractions.allocate(backend, T, Nx, Ny, 9); copyto!(uw_x, uwx_h)
    uw_y     = KernelAbstractions.allocate(backend, T, Nx, Ny, 9); copyto!(uw_y, uwy_h)

    u_prof, u_prof_h = _init_inlet_profile()
    bcspec = BCSpec2D(; west=ZouHeVelocity(u_prof), east=ZouHePressure(one(T)),
                        south=HalfwayBB(), north=HalfwayBB())
    f_h = _init_f(u_prof_h, is_solid_h)
    fa = KernelAbstractions.allocate(backend, T, Nx, Ny, 9); copyto!(fa, f_h)
    fb = similar(fa); fill!(fb, zero(T))
    ρ  = KernelAbstractions.allocate(backend, T, Nx, Ny); fill!(ρ, one(T))
    ux = KernelAbstractions.allocate(backend, T, Nx, Ny); fill!(ux, zero(T))
    uy = KernelAbstractions.allocate(backend, T, Nx, Ny); fill!(uy, zero(T))

    # For halfway-BB we need to flag all cut-link sites in q_wall (=0.5
    # implicitly through the kernel's halfway-BB fallback). The cleanest
    # way is to set q_wall = 0.5 for every link from a fluid cell to a
    # solid cell (equivalent to plain halfway-BB).
    cxs = Kraken.velocities_x(D2Q9()); cys = Kraken.velocities_y(D2Q9())
    for j in 1:Ny, i in 1:Nx
        is_solid_h[i, j] && continue
        for q in 2:9
            in_, jn_ = i + cxs[q], j + cys[q]
            (in_ < 1 || in_ > Nx || jn_ < 1 || jn_ > Ny) && continue
            if is_solid_h[in_, jn_]
                q_wall_h[i, j, q] = 0.5
            end
        end
    end
    copyto!(q_wall, q_wall_h)
    n_cut = count(q_wall_h .> 0)

    Fx_sum = 0.0; n_avg = 0
    t0 = time()
    for step in 1:steps
        fused_trt_libb_v2_step!(fb, fa, ρ, ux, uy, is_solid,
                                  q_wall, uw_x, uw_y, Nx, Ny, ν)
        apply_bc_rebuild_2d!(fb, fa, bcspec, ν, Nx, Ny)
        if step > steps - avg_window
            Fx, _ = compute_drag_libb_2d(fb, q_wall, Nx, Ny)
            Fx_sum += Fx; n_avg += 1
        end
        fa, fb = fb, fa
    end
    KernelAbstractions.synchronize(backend)
    elapsed = time() - t0
    Fx_avg = Fx_sum / n_avg
    Cd = _compute_cd_from_drag(Fx_avg)
    mlups = Nx * Ny * steps / elapsed / 1e6
    err = 100 * abs(Cd - 5.58) / 5.58
    println("  cells = $(Nx*Ny)   n_cut = $n_cut   Cd = $(round(Cd, digits=4))   err = $(round(err, digits=2))%")
    println("  $(round(elapsed, digits=1))s   $(round(mlups, digits=0)) MLUPS")
    return (Cd=Cd, err=err, cells=Nx*Ny, mlups=mlups, elapsed=elapsed)
end

# ---------- (B) Cartesian uniform + LI-BB ----------
function run_B()
    println("\n[B] Cartesian uniform + LI-BB v2 (sub-cell q_w)")
    q_wall_h, is_solid_h = precompute_q_wall_cylinder(Nx, Ny, cx_lu, cy_lu, R_lu)
    n_cut = count(q_wall_h .> 0)
    uwx_h = zeros(T, Nx, Ny, 9); uwy_h = zeros(T, Nx, Ny, 9)
    is_solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny); copyto!(is_solid, is_solid_h)
    q_wall   = KernelAbstractions.allocate(backend, T, Nx, Ny, 9); copyto!(q_wall, q_wall_h)
    uw_x     = KernelAbstractions.allocate(backend, T, Nx, Ny, 9); copyto!(uw_x, uwx_h)
    uw_y     = KernelAbstractions.allocate(backend, T, Nx, Ny, 9); copyto!(uw_y, uwy_h)

    u_prof, u_prof_h = _init_inlet_profile()
    bcspec = BCSpec2D(; west=ZouHeVelocity(u_prof), east=ZouHePressure(one(T)),
                        south=HalfwayBB(), north=HalfwayBB())
    f_h = _init_f(u_prof_h, is_solid_h)
    fa = KernelAbstractions.allocate(backend, T, Nx, Ny, 9); copyto!(fa, f_h)
    fb = similar(fa); fill!(fb, zero(T))
    ρ  = KernelAbstractions.allocate(backend, T, Nx, Ny); fill!(ρ, one(T))
    ux = KernelAbstractions.allocate(backend, T, Nx, Ny); fill!(ux, zero(T))
    uy = KernelAbstractions.allocate(backend, T, Nx, Ny); fill!(uy, zero(T))

    Fx_sum = 0.0; n_avg = 0
    t0 = time()
    for step in 1:steps
        fused_trt_libb_v2_step!(fb, fa, ρ, ux, uy, is_solid,
                                  q_wall, uw_x, uw_y, Nx, Ny, ν)
        apply_bc_rebuild_2d!(fb, fa, bcspec, ν, Nx, Ny)
        if step > steps - avg_window
            Fx, _ = compute_drag_libb_mei_2d(fb, q_wall, uw_x, uw_y, Nx, Ny)
            Fx_sum += Fx; n_avg += 1
        end
        fa, fb = fb, fa
    end
    KernelAbstractions.synchronize(backend)
    elapsed = time() - t0
    Fx_avg = Fx_sum / n_avg
    Cd = _compute_cd_from_drag(Fx_avg)
    mlups = Nx * Ny * steps / elapsed / 1e6
    err = 100 * abs(Cd - 5.58) / 5.58
    println("  cells = $(Nx*Ny)   n_cut = $n_cut   Cd = $(round(Cd, digits=4))   err = $(round(err, digits=2))%   [Mei drag]")
    println("  $(round(elapsed, digits=1))s   $(round(mlups, digits=0)) MLUPS")
    return (Cd=Cd, err=err, cells=Nx*Ny, mlups=mlups, elapsed=elapsed)
end

# ---------- (C) gmsh-imported Cartesian + SLBM + LI-BB ----------
function run_C()
    println("\n[C] gmsh Cartesian + SLBM + LI-BB v2 (q_w in physical space)")
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
        gmsh.model.geo.addCurveLoop([1,2,3,4], 1)
        gmsh.model.geo.addPlaneSurface([1], 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(1, Nx)
        gmsh.model.geo.mesh.setTransfiniteCurve(3, Nx)
        gmsh.model.geo.mesh.setTransfiniteCurve(2, Ny)
        gmsh.model.geo.mesh.setTransfiniteCurve(4, Ny)
        gmsh.model.geo.mesh.setTransfiniteSurface(1)
        gmsh.model.geo.mesh.setRecombine(2, 1)
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.write(fpath)
    finally
        gmsh.finalize()
    end

    mesh, _ = load_gmsh_mesh_2d(fpath)
    geom_h = build_slbm_geometry(mesh)
    geom = transfer_slbm_geometry(geom_h, backend)

    # Tag cylinder cells in physical coords
    is_solid_h = zeros(Bool, mesh.Nξ, mesh.Nη)
    for j in 1:mesh.Nη, i in 1:mesh.Nξ
        x = mesh.X[i, j]; y = mesh.Y[i, j]
        if (x - cx_p)^2 + (y - cy_p)^2 ≤ R_p^2
            is_solid_h[i, j] = true
        end
    end
    qw_h, uwx_h, uwy_h = precompute_q_wall_slbm_cylinder_2d(mesh, is_solid_h,
                                                              cx_p, cy_p, R_p)
    n_cut = count(qw_h .> 0)
    is_solid = KernelAbstractions.allocate(backend, Bool, mesh.Nξ, mesh.Nη); copyto!(is_solid, is_solid_h)
    q_wall   = KernelAbstractions.allocate(backend, T, mesh.Nξ, mesh.Nη, 9); copyto!(q_wall, qw_h)
    uw_x     = KernelAbstractions.allocate(backend, T, mesh.Nξ, mesh.Nη, 9); copyto!(uw_x, uwx_h)
    uw_y     = KernelAbstractions.allocate(backend, T, mesh.Nξ, mesh.Nη, 9); copyto!(uw_y, uwy_h)

    u_prof, u_prof_h = _init_inlet_profile()
    bcspec = BCSpec2D(; west=ZouHeVelocity(u_prof), east=ZouHePressure(one(T)),
                        south=HalfwayBB(), north=HalfwayBB())

    f_h = _init_f(u_prof_h, is_solid_h)
    fa = KernelAbstractions.allocate(backend, T, mesh.Nξ, mesh.Nη, 9); copyto!(fa, f_h)
    fb = similar(fa); fill!(fb, zero(T))
    ρ  = KernelAbstractions.allocate(backend, T, mesh.Nξ, mesh.Nη); fill!(ρ, one(T))
    ux = KernelAbstractions.allocate(backend, T, mesh.Nξ, mesh.Nη); fill!(ux, zero(T))
    uy = KernelAbstractions.allocate(backend, T, mesh.Nξ, mesh.Nη); fill!(uy, zero(T))

    Fx_sum = 0.0; n_avg = 0
    t0 = time()
    for step in 1:steps
        slbm_trt_libb_step!(fb, fa, ρ, ux, uy, is_solid,
                             q_wall, uw_x, uw_y, geom, ν)
        apply_bc_rebuild_2d!(fb, fa, bcspec, ν, mesh.Nξ, mesh.Nη)
        if step > steps - avg_window
            Fx, _ = compute_drag_libb_mei_2d(fb, q_wall, uw_x, uw_y, mesh.Nξ, mesh.Nη)
            Fx_sum += Fx; n_avg += 1
        end
        fa, fb = fb, fa
    end
    KernelAbstractions.synchronize(backend)
    elapsed = time() - t0
    cells = mesh.Nξ * mesh.Nη
    Fx_avg = Fx_sum / n_avg
    # For SLBM with this Cartesian mesh, dx_ref = dx_phys, so the lattice
    # drag formula works as-is on f post-step.
    Cd = 2.0 * Fx_avg / (u_mean^2 * (R_p * 2 / dx_ref))
    mlups = cells * steps / elapsed / 1e6
    err = 100 * abs(Cd - 5.58) / 5.58
    println("  cells = $cells   n_cut = $n_cut   Cd = $(round(Cd, digits=4))   err = $(round(err, digits=2))%")
    println("  $(round(elapsed, digits=1))s   $(round(mlups, digits=0)) MLUPS")
    return (Cd=Cd, err=err, cells=cells, mlups=mlups, elapsed=elapsed)
end

resA = run_A()
resB = run_B()
resC = run_C()

println("\n=== WP-MESH-4 summary (Schäfer-Turek 2D-1, Re=20, ref Cd=5.58) ===")
println("Approach                                Cells   Cd      err     MLUPS  walltime")
println("(A) Cartesian + halfway-BB              $(resA.cells)  $(round(resA.Cd, digits=3))  $(round(resA.err, digits=2))%  $(round(resA.mlups, digits=0))   $(round(resA.elapsed, digits=1))s")
println("(B) Cartesian + LI-BB                   $(resB.cells)  $(round(resB.Cd, digits=3))  $(round(resB.err, digits=2))%  $(round(resB.mlups, digits=0))   $(round(resB.elapsed, digits=1))s")
println("(C) gmsh Cartesian + SLBM + LI-BB       $(resC.cells)  $(round(resC.Cd, digits=3))  $(round(resC.err, digits=2))%  $(round(resC.mlups, digits=0))   $(round(resC.elapsed, digits=1))s")
println("\nNote: (B) vs (C) should match closely since gmsh produces the same uniform")
println("Cartesian mesh; the difference reports the SLBM trilinear interpolation overhead.")
