

# ===========================================================================
# Dispatch helpers for 3D / axisymmetric / refined / thermal cases
# ===========================================================================

"""Dispatch D3Q19 cases to the appropriate 3D driver."""
function _run_d3q19(setup::SimulationSetup;
                    backend=KernelAbstractions.CPU(), T=Float64)
    name = lowercase(setup.name)
    dom  = setup.domain
    ν    = setup.physics.params[:nu]
    if _has_stl_libb_obstacle(setup)
        result = run_obstacle_libb_3d(setup; backend=backend, T=T)
        return merge(result, (setup=setup,))
    elseif occursin("cavity_3d", name) || occursin("cavity3d", name)
        # Look for a velocity BC on top/north for u_lid; default to 0.1.
        u_lid = 0.1
        for b in setup.boundaries
            if b.type == :velocity && haskey(b.values, :ux)
                try
                    u_lid = Float64(evaluate(b.values[:ux]))
                catch
                end
                break
            end
        end
        config = LBMConfig(D3Q19(); Nx=dom.Nx, Ny=dom.Ny, Nz=dom.Nz,
                           ν=Float64(ν), u_lid=u_lid,
                           max_steps=setup.max_steps)
        result = run_cavity_3d(config; backend=backend, T=T)
        return merge(result, (setup=setup,))
    elseif occursin("natural_convection", name) || occursin("fusegi", name)
        params = setup.physics.params
        Ra = Float64(get(params, :Ra, 1e4))
        Pr = Float64(get(params, :Pr, 0.71))
        result = run_natural_convection_3d(; N=dom.Nx, Ra=Ra, Pr=Pr,
                                            max_steps=setup.max_steps,
                                            backend=backend, FT=T)
        return merge(result, (setup=setup,))
    else
        throw(ArgumentError(
            "D3Q19 dispatch: only `cavity_3d`, `natural_convection`, and " *
            "`fusegi` are supported in v0.1.0 " *
            "(got case name: $(setup.name)). Use the Julia API for other 3D cases."))
    end
end

"""Dispatch axisymmetric cases. Only Hagen-Poiseuille is supported in v0.1.0."""
function _run_axisymmetric(setup::SimulationSetup;
                           backend=KernelAbstractions.CPU(), T=Float64)
    name = lowercase(setup.name)
    dom  = setup.domain
    params = setup.physics.params
    ν  = Float64(params[:nu])
    Fz = haskey(setup.physics.body_force, :Fz) ?
         Float64(evaluate(setup.physics.body_force[:Fz])) : 1e-5
    if occursin("hagen_poiseuille", name) || occursin("hagen-poiseuille", name)
        result = run_hagen_poiseuille_2d(; Nz=dom.Nx, Nr=dom.Ny, ν=ν, Fz=Fz,
                                          max_steps=setup.max_steps,
                                          backend=backend, FT=T)
        return merge(result, (setup=setup,))
    else
        throw(ArgumentError(
            "axisymmetric dispatch: only `hagen_poiseuille` is supported in " *
            "v0.1.0 (got case name: $(setup.name))."))
    end
end
