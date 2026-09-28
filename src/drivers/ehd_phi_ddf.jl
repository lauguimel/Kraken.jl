# ============================================================================
# src/drivers/ehd_phi_ddf.jl
# The DDF (pseudo-time lattice Boltzmann) solve of the EHD electric potential,
# shared by the hydrostatic driver (`run_ehd_hydrostatic_2d`, periodic sides)
# and the electroconvection state (`advance!(::ECState)`, Neumann box).
#
# One iteration: collide with the charge source, stream, rebuild the plate
# populations by non-equilibrium extrapolation, take the zeroth moment. The
# adaptive solve repeats it until its stopping rule accepts. Not exported.
# ============================================================================

"""
Check cadence of the adaptive potential solve in `advance!(::ECState)`: the
stopping rule is evaluated every `EHD_EC_PHI_CHECK_EVERY` iterations and at
`phi_max_iter`, at the cost of one device-to-host copy per check.
"""
const EHD_EC_PHI_CHECK_EVERY = 8

"""
Check cadence of the adaptive potential solve in `run_ehd_hydrostatic_2d`: the
stopping rule is evaluated after every iteration.
"""
const EHD_HYDROSTATIC_PHI_CHECK_EVERY = 1

function _ehd_phi_ddf_check_xbc(xbc::Symbol)
    xbc in (:neumann, :periodic) ||
        throw(ArgumentError("xbc must be :neumann or :periodic, got :$(xbc)."))
    return nothing
end

"""
    ehd_phi_ddf_step!(f_in, f_out, phi, qfield, p, xbc; phi_bottom, phi_top) -> (f_in, f_out)

One pseudo-time iteration of the DDF potential solve: collide `f_in` in place with
the source `qfield`, stream into `f_out`, rebuild the plate populations by
non-equilibrium extrapolation (`phi_bottom` at `j = 1`, `phi_top` at `j = Ny`),
and leave the zeroth moment of `f_out` in `phi`.

Returns the swapped pair: the returned `f_in` is the `f_out` just filled. Callers
rebind, `f_in, f_out = ehd_phi_ddf_step!(f_in, f_out, ...)`.

`xbc` selects the lateral treatment:
- `:neumann`: wall streaming on all four sides and extrapolation on the whole box
  (`advance!(::ECState)`);
- `:periodic`: periodic in x, walls in y, extrapolation on the plates only
  (`run_ehd_hydrostatic_2d`).

`p` is any object with the fields `eps`, `omega_U` and `nu_U`.
"""
function ehd_phi_ddf_step!(f_in, f_out, phi, qfield, p, xbc::Symbol; phi_bottom, phi_top)
    _ehd_phi_ddf_check_xbc(xbc)
    Nx, Ny = size(phi)
    collide_electric_potential_2d!(f_in, qfield, p.eps, p.omega_U, p.nu_U)
    if xbc === :neumann
        stream_wall_x_wall_y_2d!(f_out, f_in, Nx, Ny)
        compute_ehd_scalar_2d!(phi, f_out)
        apply_phi_nee_box_2d!(f_out, phi, phi_bottom, phi_top, Nx, Ny)
    else
        stream_periodic_x_wall_y_2d!(f_out, f_in, Nx, Ny)
        compute_ehd_scalar_2d!(phi, f_out)
        apply_phi_nee_walls_2d!(f_out, phi, phi_bottom, phi_top, Nx, Ny)
    end
    compute_ehd_scalar_2d!(phi, f_out)
    return f_out, f_in
end

"""
    ehd_phi_ddf_workspace(phi) -> NamedTuple

Scratch buffers of [`ehd_phi_ddf_solve!`](@ref), allocated once on the backend and
with the element type of `phi`: `phi_prev` (the shape of `phi`), `diag` (length 2,
on the device) and `diag_host` (length 2, on the host). Their contents on entry do
not matter: each buffer is overwritten before it is read.
"""
function ehd_phi_ddf_workspace(phi)
    return (phi_prev=similar(phi), diag=similar(phi, 2),
            diag_host=Vector{eltype(phi)}(undef, 2))
end

"""
    ehd_phi_ddf_solve!(f_in, f_out, phi, qfield, p, xbc, ws; phi_tol, field_tol,
                       max_iter, check_every, phi_bottom, phi_top) -> (f_in, f_out, stats)

Adaptive DDF solve of the electric potential for a frozen charge `qfield`: repeat
[`ehd_phi_ddf_step!`](@ref) until the stopping rule accepts, or until `max_iter`
iterations have run.

The rule is evaluated at iteration `iter` when `iter % check_every == 0` or
`iter == max_iter`, with one device-to-host copy per check. It accepts when
`phi_rel <= phi_tol`, where `phi_rel = max|phi - phi_prev| / max|phi|` is the
relative change of the zeroth moment over the last iteration
(`ehd_rel_change_2d!`). `phi_tol` is compared as given, without conversion to the
element type.

`field_tol` must be `Inf`: no check on the electric field, which is the rule the
drivers have always applied.

Returns the swapped population pair (the solution is the returned `f_in`; callers
rebind) and `stats = (; iters, phi_rel, field_rel, converged)`:
- `iters`: iterations run;
- `phi_rel`: `phi_rel` at the last check (`Inf` when no check ran);
- `field_rel`: `NaN`, since the field is not checked;
- `converged`: `true` when a check accepted. `false` after `max_iter` iterations
  without acceptance; no exception is raised, each driver raises its own error.

On exit `phi` holds the zeroth moment of the returned `f_in` (unless no iteration
ran). `ws` is an [`ehd_phi_ddf_workspace`](@ref) or any `NamedTuple` with the same
fields; `p` is any object with the fields `eps`, `omega_U` and `nu_U`.
"""
function ehd_phi_ddf_solve!(f_in, f_out, phi, qfield, p, xbc::Symbol, ws;
                            phi_tol, field_tol, max_iter, check_every, phi_bottom, phi_top)
    _ehd_phi_ddf_check_xbc(xbc)
    check_every >= 1 ||
        throw(ArgumentError("check_every must be at least 1, got $(check_every)."))
    field_tol == Inf ||
        throw(ArgumentError("field_tol must be Inf (no field check), got $(field_tol)."))
    FT = eltype(phi)
    Nx, Ny = size(phi)
    iters = 0
    phi_rel = FT(Inf)
    converged = false
    for iter in 1:max_iter
        copyto!(ws.phi_prev, phi)
        f_in, f_out = ehd_phi_ddf_step!(f_in, f_out, phi, qfield, p, xbc;
                                        phi_bottom=phi_bottom, phi_top=phi_top)
        iters = iter
        if iter % check_every == 0 || iter == max_iter
            ehd_rel_change_2d!(ws.diag, phi, ws.phi_prev, Nx, Ny)
            copyto!(ws.diag_host, ws.diag)
            phi_rel = ws.diag_host[1]
            if phi_rel <= phi_tol
                converged = true
                break
            end
        end
    end
    return f_in, f_out, (; iters, phi_rel, field_rel=FT(NaN), converged)
end

# The drivers' non-convergence error. A solve that ran no iteration
# (`max_iter < 1`) has no failed check and raises nothing, as the drivers'
# loops never did.
function _ehd_phi_ddf_require_converged(stats, max_iter)
    (stats.converged || stats.iters == 0) && return nothing
    error("Electric potential solve did not converge within $(max_iter) iterations. " *
          "Last relative change: $(stats.phi_rel).")
end
