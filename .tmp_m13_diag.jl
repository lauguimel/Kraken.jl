# Quick diagnostic to characterise the M13 5e-3 residual
include("bench/viscoelastic_logfv/run_poiseuille_imposed_stress_2d.jl")

cfg = default_config(Float64)
backend = KernelAbstractions.CPU()
ref = poiseuille_reference_fields(cfg, Float64)
result = run_inverse_pipeline(ref, cfg, backend, Float64)

ux_prof = y_profile_average(result.ux)
ux_ref  = y_profile_average(ref.ux_ref)

println("\ny (j-0.5)   u_LBM            u_anal           rel_err")
for j in 1:cfg.Ny
    err = (ux_prof[j] - ux_ref[j]) / cfg.U_max
    println(rpad(string(j), 4), "  ", rpad(string(ux_prof[j]), 22),
            "  ", rpad(string(ux_ref[j]), 22),
            "  ", err)
end

println("\nrel L2 full     = ", interior_rel_l2(ux_prof, ux_ref, 1, cfg.Ny))
println("rel L2 interior = ", interior_rel_l2(ux_prof, ux_ref, 2, cfg.Ny - 1))
println("rel L2 deep int = ", interior_rel_l2(ux_prof, ux_ref, 4, cfg.Ny - 3))

# Look at spatial variation in x: is it well periodic-equivalent?
ux2 = result.ux
println("\nrange across x at j=Ny/2 = (", minimum(ux2[:, cfg.Ny ÷ 2]),
        ", ", maximum(ux2[:, cfg.Ny ÷ 2]), ")")
println("ux at corners:")
println("  (1, 1) = ", ux2[1, 1], "  (Nx, 1) = ", ux2[cfg.Nx, 1])
println("  (1, Ny) = ", ux2[1, cfg.Ny], "  (Nx, Ny) = ", ux2[cfg.Nx, cfg.Ny])
println("ux center line vs x (at j=16):")
for i in 1:4:cfg.Nx
    println("  i=", i, "  u=", ux2[i, 16])
end
