include("bench/viscoelastic_logfv/run_poiseuille_imposed_stress_2d.jl")

cfg = default_config(Float64)
backend = KernelAbstractions.CPU()
ref = poiseuille_reference_fields(cfg, Float64)

# Build only the body force, no LBM loop
Nx, Ny = cfg.Nx, cfg.Ny
is_solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny); fill!(is_solid, false)
tauxx = KernelAbstractions.allocate(backend, Float64, Nx, Ny); copyto!(tauxx, ref.tauxx)
tauxy = KernelAbstractions.allocate(backend, Float64, Nx, Ny); copyto!(tauxy, ref.tauxy)
tauyy = KernelAbstractions.allocate(backend, Float64, Nx, Ny); copyto!(tauyy, ref.tauyy)
fx_poly = KernelAbstractions.zeros(backend, Float64, Nx, Ny)
fy_poly = KernelAbstractions.zeros(backend, Float64, Nx, Ny)

fvfd_bc = Kraken.fvfd_periodicx_wally_bcspec_2d()
Kraken.logfv_polymer_force_bc_aware_2d!(fx_poly, fy_poly, tauxx, tauxy, tauyy,
    is_solid, cfg.dx, cfg.dy, fvfd_bc; polymer_wall_extrap=:quadratic, sync=true)

fx = Array(fx_poly)
fy = Array(fy_poly)

target = cfg.Fx_anal  # = -8·η_p·U_max/H² = constant
println("target Fx (constant) = ", target)
println("\ny    fx[1,j]               fy[1,j]               (fx-target)/target")
for j in 1:Ny
    err = (fx[1, j] - target) / target
    println(rpad(string(j), 4), "  ", rpad(string(fx[1, j]), 22),
            "  ", rpad(string(fy[1, j]), 22), "  ", err)
end
println("\nmax abs fy = ", maximum(abs.(fy)))
println("mean fx  = ", sum(fx) / length(fx))
println("max abs (fx - target) = ", maximum(abs.(fx .- target)))
