
# ===========================================================================
# Prescribed-velocity advection runner (Zalesak, reversed vortex, shear)
# ===========================================================================

"""
    _run_advection_only(setup; backend, T)

Run pure VOF advection with prescribed velocity from `Velocity { ux=... uy=... }`.
No LBM solve — used for interface transport validation tests.
"""
function _run_advection_only(setup::SimulationSetup;
                             backend=KernelAbstractions.CPU(), T=Float64)
    dom = setup.domain
    Nx, Ny = dom.Nx, dom.Ny
    dx = T(dom.Lx / Nx)
    Lx, Ly = T(dom.Lx), T(dom.Ly)

    setup.velocity_field === nothing &&
        throw(ArgumentError("Module advection_only requires a Velocity { ux=... uy=... } block"))
    setup.initial === nothing || !haskey(setup.initial.fields, :C) &&
        throw(ArgumentError("Module advection_only requires Initial { C = ... }"))

    # Build velocity function from KrakenExpr
    vf = setup.velocity_field
    ux_expr = get(vf.fields, :ux, nothing)
    uy_expr = get(vf.fields, :uy, nothing)

    function velocity_fn(x, y, t)
        kw = (; x=x, y=y, t=t, Lx=Float64(Lx), Ly=Float64(Ly),
                Nx=Float64(Nx), Ny=Float64(Ny), dx=Float64(dx))
        vx = ux_expr !== nothing ? evaluate(ux_expr; kw...) : 0.0
        vy = uy_expr !== nothing ? evaluate(uy_expr; kw...) : 0.0
        return (vx, vy)
    end

    # Build init function from Initial { C = ... }
    C_expr = setup.initial.fields[:C]
    function init_C_fn(x, y)
        kw = (; x=x, y=y, Lx=Float64(Lx), Ly=Float64(Ly),
                Nx=Float64(Nx), Ny=Float64(Ny), dx=Float64(dx))
        return evaluate(C_expr; kw...)
    end

    _adv_vtk = _find_output(setup, :vtk)
    output_interval = _adv_vtk !== nothing ? _adv_vtk.interval : 0
    output_dir = _adv_vtk !== nothing ? _adv_vtk.directory : ""

    adv_result = run_advection_2d(; Nx=Nx, Ny=Ny, max_steps=setup.max_steps,
                                   velocity_fn=velocity_fn,
                                   init_C_fn=init_C_fn,
                                   output_interval=output_interval,
                                   output_dir=output_dir,
                                   backend=backend, FT=T)

    # Wrap with setup and ρ for postprocess helper compatibility
    ρ_from_C = adv_result.C  # use C as a proxy for ρ in advection-only mode
    return merge(adv_result, (ρ=ρ_from_C, setup=setup))
end
