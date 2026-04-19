# WP-MESH-3 — pipeline validation on a CANONICAL case where the analytic
# answer is exact: 2D Poiseuille channel, mesh imported from gmsh
# (Transfinite axis-aligned block) → SLBM TRT+LI-BB step → compare
# centreline u(y) with the analytic parabolic profile.
using Kraken, KernelAbstractions, Gmsh

const Lx, Ly = 4.0, 1.0
const Nx_n, Ny_n = 81, 41          # gmsh nodes
const u_max = 0.04
const ν = 0.05
const steps = 4_000
T = Float64

# Analytic Poiseuille u(y) for 0 ≤ y ≤ Ly with no-slip walls and peak u_max:
u_poiseuille(y) = u_max * 4 * (y / Ly) * (1 - y / Ly)

mktemp() do path, _io
    fpath = path * ".msh"
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("poiseuille_block")
        gmsh.model.geo.addPoint(0.0, 0.0, 0.0, 0.1, 1)
        gmsh.model.geo.addPoint(Lx,  0.0, 0.0, 0.1, 2)
        gmsh.model.geo.addPoint(Lx,  Ly,  0.0, 0.1, 3)
        gmsh.model.geo.addPoint(0.0, Ly,  0.0, 0.1, 4)
        gmsh.model.geo.addLine(1, 2, 1)   # south wall
        gmsh.model.geo.addLine(2, 3, 2)   # outlet (east)
        gmsh.model.geo.addLine(3, 4, 3)   # north wall
        gmsh.model.geo.addLine(4, 1, 4)   # inlet (west)
        gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
        gmsh.model.geo.addPlaneSurface([1], 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(1, Nx_n)
        gmsh.model.geo.mesh.setTransfiniteCurve(3, Nx_n)
        gmsh.model.geo.mesh.setTransfiniteCurve(2, Ny_n)
        gmsh.model.geo.mesh.setTransfiniteCurve(4, Ny_n)
        gmsh.model.geo.mesh.setTransfiniteSurface(1)
        gmsh.model.geo.mesh.setRecombine(2, 1)
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(1, [4], 100, "inlet")
        gmsh.model.addPhysicalGroup(1, [2], 200, "outlet")
        gmsh.model.addPhysicalGroup(1, [1], 300, "south")
        gmsh.model.addPhysicalGroup(1, [3], 400, "north")
        gmsh.model.addPhysicalGroup(2, [1], 1000, "fluid")
        gmsh.model.mesh.generate(2)
        gmsh.write(fpath)
    finally
        gmsh.finalize()
    end

    println("== load gmsh mesh (axis-aligned) ==")
    mesh, groups = load_gmsh_mesh_2d(fpath; layout=:axis_aligned)
    Nξ, Nη = mesh.Nξ, mesh.Nη
    println("Nξ × Nη = $Nξ × $Nη")
    println("groups: ", collect(keys(groups.by_name)))

    geom_h = build_slbm_geometry(mesh)
    backend = KernelAbstractions.CPU()
    geom = transfer_slbm_geometry(geom_h, backend)

    is_solid = KernelAbstractions.allocate(backend, Bool, Nξ, Nη); fill!(is_solid, false)
    q_wall = KernelAbstractions.allocate(backend, T, Nξ, Nη, 9); fill!(q_wall, zero(T))
    uw_x   = KernelAbstractions.allocate(backend, T, Nξ, Nη, 9); fill!(uw_x, zero(T))
    uw_y   = KernelAbstractions.allocate(backend, T, Nξ, Nη, 9); fill!(uw_y, zero(T))

    # Inlet profile via gmsh tag → logical face. Phase A sorts (y, x), so
    # the inlet (west, x=0) corresponds to i=1.
    u_prof_h = T[u_poiseuille(mesh.Y[1, j]) for j in 1:Nη]
    u_prof = KernelAbstractions.allocate(backend, T, Nη); copyto!(u_prof, u_prof_h)
    bcspec = BCSpec2D(; west=ZouHeVelocity(u_prof),
                        east=ZouHePressure(one(T)),
                        south=HalfwayBB(),
                        north=HalfwayBB())

    f_h = zeros(T, Nξ, Nη, 9)
    for j in 1:Nη, i in 1:Nξ, q in 1:9
        f_h[i, j, q] = Kraken.equilibrium(D2Q9(), one(T), u_poiseuille(mesh.Y[i, j]), zero(T), q)
    end
    fa = KernelAbstractions.allocate(backend, T, Nξ, Nη, 9); copyto!(fa, f_h)
    fb = similar(fa); fill!(fb, zero(T))
    ρ  = KernelAbstractions.allocate(backend, T, Nξ, Nη); fill!(ρ, one(T))
    ux = KernelAbstractions.allocate(backend, T, Nξ, Nη); fill!(ux, zero(T))
    uy = KernelAbstractions.allocate(backend, T, Nξ, Nη); fill!(uy, zero(T))

    for _ in 1:steps
        slbm_trt_libb_step!(fb, fa, ρ, ux, uy, is_solid, q_wall, uw_x, uw_y, geom, ν)
        apply_bc_rebuild_2d!(fb, fa, bcspec, ν, Nξ, Nη)
        KernelAbstractions.synchronize(backend)
        fa, fb = fb, fa
    end

    println("\n== compare with analytic u(y) at the channel midplane (i=Nξ÷2) ==")
    ux_h = Array(ux)
    i_mid = Nξ ÷ 2
    err_inf = 0.0; err_l2 = 0.0; n = 0
    for j in 2:(Nη - 1)
        y = mesh.Y[i_mid, j]
        u_num = ux_h[i_mid, j]; u_ref = u_poiseuille(y)
        err = abs(u_num - u_ref)
        err_inf = max(err_inf, err)
        err_l2 += err^2; n += 1
    end
    err_l2 = sqrt(err_l2 / n)
    println("samples = $n  Linf = $(round(err_inf, sigdigits=3))  L2 = $(round(err_l2, sigdigits=3))")
    println("relative Linf vs u_max = $(round(err_inf / u_max * 100, digits=2))%")
    println("first 5 (y, u_num, u_ref):")
    for j in 2:6
        y = mesh.Y[i_mid, j]
        println("  y=", round(y, digits=4),
                "  num=", round(ux_h[i_mid, j], digits=6),
                "  ref=", round(u_poiseuille(y), digits=6))
    end
end
