
"""Dispatch thermal cases (non-refined) to the appropriate thermal driver."""
function _run_thermal(setup::SimulationSetup;
                      backend=KernelAbstractions.CPU(), T=Float64)
    name   = lowercase(setup.name)
    dom    = setup.domain
    params = setup.physics.params
    Ra = Float64(get(params, :Ra, 1e4))
    Pr = Float64(get(params, :Pr, 0.71))

    if occursin("rayleigh_benard", name) || occursin("rayleigh-benard", name)
        result = run_rayleigh_benard_2d(; Nx=dom.Nx, Ny=dom.Ny, Ra=Ra, Pr=Pr,
                                         max_steps=setup.max_steps,
                                         backend=backend, FT=T)
        return merge(result, (setup=setup,))
    elseif occursin("natural_convection", name)
        result = run_natural_convection_2d(; N=dom.Nx, Ra=Ra, Pr=Pr,
                                            max_steps=setup.max_steps,
                                            backend=backend, FT=T)
        return merge(result, (setup=setup,))
    elseif occursin("heat_conduction", name) || occursin("conduction", name)
        # No dedicated conduction driver: run Rayleigh-Bénard with Ra≈0
        # so buoyancy is negligible and diffusion dominates. Documented
        # as a pragmatic fallback — the resulting temperature field
        # matches a 1D diffusive profile once steady state is reached.
        result = run_rayleigh_benard_2d(; Nx=dom.Nx, Ny=dom.Ny,
                                         Ra=1e-8, Pr=Pr,
                                         max_steps=setup.max_steps,
                                         backend=backend, FT=T)
        return merge(result, (setup=setup,))
    else
        throw(ArgumentError(
            "thermal dispatch: unrecognized case name '$(setup.name)'. " *
            "Known cases: rayleigh_benard, natural_convection, heat_conduction."))
    end
end
