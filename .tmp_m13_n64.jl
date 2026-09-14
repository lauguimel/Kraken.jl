include("bench/viscoelastic_logfv/run_poiseuille_imposed_stress_2d.jl")

function cfg_n(::Type{T}, N::Int, nsteps::Int) where {T}
    base = default_config(T)
    return merge(base, (; Nx=N, Ny=N, H=T(N), Fx_anal=derive_Fx(base.nu_s, base.U_max, T(N)),
                          nsteps=nsteps))
end

for (N, nsteps) in [(32, 60000), (64, 120000), (128, 240000)]
    cfg = cfg_n(Float64, N, nsteps)
    backend = KernelAbstractions.CPU()
    ref = poiseuille_reference_fields(cfg, Float64)
    result = run_inverse_pipeline(ref, cfg, backend, Float64)
    ux_prof = y_profile_average(result.ux)
    ux_ref  = y_profile_average(ref.ux_ref)
    rel_l2 = interior_rel_l2(ux_prof, ux_ref, 2, cfg.Ny - 1)
    println(@sprintf("N=%3d  nsteps=%6d  rel L2 = %.6e", N, nsteps, rel_l2))
end
