# WP-MESH-3 — full pipeline validation: gmsh half-annulus → SLBM TRT+LI-BB
# Compare with the analytic Taylor-Couette flow between R_in (rotating)
# and R_out (fixed). The 2D periodic-ξ annulus is approximated here by
# a 4-corner half-block (radial cuts at θ=0 and θ=π); we close the
# half-annulus with halfway-BB on the south/north logical walls. Solver
# accuracy is judged on the velocity profile u_θ(r) at θ = π/2.
using Kraken, KernelAbstractions, Gmsh

const R_in   = 1.0
const R_out  = 2.0
const N_θ_h  = 32      # nodes on each half-arc
const N_r    = 17
const u_w    = 0.01     # inner wall tangential velocity (lattice units)
const ν      = 0.05     # kinematic viscosity
const steps  = 5_000
T = Float64

# Analytic Taylor-Couette profile u_θ(r) for inner-rotating, outer-fixed:
#   u_θ(r) = A·r + B/r,  with A = -Ω·R_in²/(R_out²-R_in²),
#                              B =  Ω·R_in²·R_out²/(R_out²-R_in²)
#   where Ω = u_w / R_in.
function uθ_analytic(r)
    Ω = u_w / R_in
    den = R_out^2 - R_in^2
    A = -Ω * R_in^2 / den
    B =  Ω * R_in^2 * R_out^2 / den
    return A * r + B / r
end

mktemp() do path, _io
    fpath = path * ".msh"
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("half_annulus")
        gmsh.model.geo.addPoint(R_in,   0, 0, 0.1, 1)
        gmsh.model.geo.addPoint(-R_in,  0, 0, 0.1, 2)
        gmsh.model.geo.addPoint(R_out,  0, 0, 0.1, 3)
        gmsh.model.geo.addPoint(-R_out, 0, 0, 0.1, 4)
        gmsh.model.geo.addPoint(0, 0, 0, 0.1, 5)
        gmsh.model.geo.addCircleArc(1, 5, 2, 1)    # inner top
        gmsh.model.geo.addCircleArc(3, 5, 4, 2)    # outer top
        gmsh.model.geo.addLine(1, 3, 3)            # radial right
        gmsh.model.geo.addLine(2, 4, 4)            # radial left
        gmsh.model.geo.addCurveLoop([3, 2, -4, -1], 1)
        gmsh.model.geo.addPlaneSurface([1], 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(1, N_θ_h)
        gmsh.model.geo.mesh.setTransfiniteCurve(2, N_θ_h)
        gmsh.model.geo.mesh.setTransfiniteCurve(3, N_r)
        gmsh.model.geo.mesh.setTransfiniteCurve(4, N_r)
        gmsh.model.geo.mesh.setTransfiniteSurface(1)
        gmsh.model.geo.mesh.setRecombine(2, 1)
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(1, [1], 100, "inner_wall")
        gmsh.model.addPhysicalGroup(1, [2], 200, "outer_wall")
        gmsh.model.addPhysicalGroup(1, [3], 300, "radial_seam_right")
        gmsh.model.addPhysicalGroup(1, [4], 400, "radial_seam_left")
        gmsh.model.addPhysicalGroup(2, [1], 1000, "fluid")
        gmsh.model.mesh.generate(2)
        gmsh.write(fpath)
    finally
        gmsh.finalize()
    end

    println("== load mesh ==")
    mesh, groups = load_gmsh_mesh_2d(fpath; layout=:topological,
                                       periodic_ξ=false, periodic_η=false)
    Nξ, Nη = mesh.Nξ, mesh.Nη
    println("Nξ × Nη = $Nξ × $Nη  (expect $(N_r)×$(N_θ_h) or $(N_θ_h)×$(N_r))")
    # Detect orientation: find which axis varies along the inner wall
    # (η=1 by topo construction)
    geom_h = build_slbm_geometry(mesh)
    backend = KernelAbstractions.CPU()
    geom = transfer_slbm_geometry(geom_h, backend)

    # Identify walls by physical radius (NOT by logical index, because the
    # topological loader fixes the (ξ, η) orientation arbitrarily):
    #   r ≈ R_in  → inner ring (rotating wall)
    #   r ≈ R_out → outer ring (fixed wall)
    is_solid_h = zeros(Bool, Nξ, Nη)
    uw_x_h = zeros(T, Nξ, Nη); uw_y_h = zeros(T, Nξ, Nη)
    tol = 0.02 * (R_out - R_in)
    n_inner = 0; n_outer = 0
    for j in 1:Nη, i in 1:Nξ
        x = mesh.X[i, j]; y = mesh.Y[i, j]
        r = hypot(x, y)
        if abs(r - R_in) < tol
            is_solid_h[i, j] = true
            uw_x_h[i, j] = -u_w * y / r
            uw_y_h[i, j] =  u_w * x / r
            n_inner += 1
        elseif abs(r - R_out) < tol
            is_solid_h[i, j] = true
            n_outer += 1
        end
    end
    println("  tagged $n_inner inner-wall nodes (rotating) + $n_outer outer-wall nodes (fixed)")
    is_solid = KernelAbstractions.allocate(backend, Bool, Nξ, Nη); copyto!(is_solid, is_solid_h)
    uw_x = KernelAbstractions.allocate(backend, T, Nξ, Nη); copyto!(uw_x, uw_x_h)
    uw_y = KernelAbstractions.allocate(backend, T, Nξ, Nη); copyto!(uw_y, uw_y_h)

    # Init equilibrium at rest
    f_h = zeros(T, Nξ, Nη, 9)
    for j in 1:Nη, i in 1:Nξ, q in 1:9
        f_h[i, j, q] = Kraken.equilibrium(D2Q9(), one(T), zero(T), zero(T), q)
    end
    fa = KernelAbstractions.allocate(backend, T, Nξ, Nη, 9); copyto!(fa, f_h)
    fb = similar(fa); fill!(fb, zero(T))
    ρ  = KernelAbstractions.allocate(backend, T, Nξ, Nη); fill!(ρ, one(T))
    ux = KernelAbstractions.allocate(backend, T, Nξ, Nη); fill!(ux, zero(T))
    uy = KernelAbstractions.allocate(backend, T, Nξ, Nη); fill!(uy, zero(T))
    ω  = T(1.0 / (3 * ν + 0.5))

    # Step with the moving-wall BGK SLBM (no LI-BB needed: walls coincide
    # with mesh edges, so q_w = 1 on all crossings)
    for _ in 1:steps
        slbm_bgk_moving_step!(fb, fa, ρ, ux, uy, is_solid, uw_x, uw_y, geom, ω)
        KernelAbstractions.synchronize(backend)
        fa, fb = fb, fa
    end

    println("\n== compare to analytic Taylor-Couette u_θ(r) ==")
    # Take all interior fluid nodes whose y > 0 and away from the seams.
    ux_h = Array(ux); uy_h = Array(uy)
    rs = Float64[]; us_num = Float64[]; us_ref = Float64[]
    for j in 1:Nη, i in 1:Nξ
        is_solid_h[i, j] && continue
        x = mesh.X[i, j]; y = mesh.Y[i, j]
        # Sample only in the interior of the half-annulus (away from the
        # two radial seams). The half is upper, so keep y > 0.2·R_in.
        y < 0.3 * R_in && continue
        r = hypot(x, y)
        # u_θ = (-y·ux + x·uy) / r
        u_t = (-y * ux_h[i, j] + x * uy_h[i, j]) / r
        push!(rs, r); push!(us_num, u_t); push!(us_ref, uθ_analytic(r))
    end
    perm = sortperm(rs); rs = rs[perm]; us_num = us_num[perm]; us_ref = us_ref[perm]
    err_inf = maximum(abs.(us_num .- us_ref))
    err_rel = err_inf / maximum(abs.(us_ref))
    println("samples = $(length(rs))   Linf err = $(round(err_inf, digits=5))   relative = $(round(err_rel*100, digits=2))%")
    println("first 5 (r, u_num, u_ref, err):")
    for k in 1:min(5, length(rs))
        println("  r=", lpad(round(rs[k], digits=4), 8),
                "  num=", lpad(round(us_num[k], digits=6), 10),
                "  ref=", lpad(round(us_ref[k], digits=6), 10),
                "  err=", round(us_num[k] - us_ref[k], digits=5))
    end
    println("last 5:")
    for k in max(1, length(rs)-4):length(rs)
        println("  r=", lpad(round(rs[k], digits=4), 8),
                "  num=", lpad(round(us_num[k], digits=6), 10),
                "  ref=", lpad(round(us_ref[k], digits=6), 10),
                "  err=", round(us_num[k] - us_ref[k], digits=5))
    end
end
