# Setup helpers — non-dimensional parameters

The `Setup` directive lets you specify a run by its dimensionless numbers
(Reynolds, Rayleigh, Prandtl) and let the parser back-compute the
corresponding lattice `nu`, `alpha`, `gbeta_DT`. This is implemented in
`_apply_setup_helpers!` (`src/io/kraken_parser.jl`, ~line 1096).

Known keys (see `_parse_setup`):

```julia
known = (:reynolds, :rayleigh, :prandtl, :L_ref, :U_ref)
```

Unknown keys trigger a Levenshtein suggestion (`Reynolds` → `reynolds`).

## Reference scales

- **`L_ref`** — characteristic length. Default: `min(Nx, Ny)` in lattice
  units.
- **`U_ref`** — characteristic velocity. Default: the magnitude of the first
  velocity BC found (`_probe_U_ref`), falling back to `0.1` if none exists.

You can override either one explicitly:

```text
Setup reynolds = 1000 L_ref = 128 U_ref = 0.1
```

## `Setup reynolds = Re`

Back-compute `nu` from

```
ν = U_ref · L_ref / Re
```

and write it into `physics.params[:nu]`. Errors if `Physics nu = ...` is
also specified (conflict).

**Example**

```text
Simulation cavity D2Q9
Domain L = 1.0 x 1.0  N = 128 x 128
Setup reynolds = 1000        # U_ref picked up from the lid velocity
Boundary north velocity(ux = 0.1, uy = 0)
Boundary south wall
Boundary east wall
Boundary west wall
Run 20000 steps
```

Source excerpt:

```julia
if haskey(helpers, :reynolds)
    Re = helpers[:reynolds]
    if haskey(physics_params, :nu)
        throw(ArgumentError(
            "Setup reynolds conflicts with Physics nu (both specified). ..."))
    end
    ν = U_ref * L_ref / Re
    physics_params[:nu] = ν
    physics_params[:Re] = Re
end
```

## `Setup rayleigh = Ra prandtl = Pr`

For buoyancy-driven thermal runs, back-compute `nu`, `alpha`, and the
buoyancy coefficient `gβΔT` from:

```
ν      = sqrt(Pr / Ra) · U_ref · L_ref
α      = ν / Pr
gβΔT   = Ra · ν · α / L_ref³
```

`prandtl` defaults to `Physics Pr = ...` if present, else `0.71`.

**Example**

```text
Module thermal
Domain L = 2.0 x 1.0  N = 256 x 128
Setup rayleigh = 1e6 prandtl = 0.71
Boundary south wall T = 1.0
Boundary north wall T = 0.0
Boundary x periodic
Run 50000 steps
```

Source excerpt:

```julia
if haskey(helpers, :rayleigh)
    Ra = helpers[:rayleigh]
    Pr = get(helpers, :prandtl, get(physics_params, :Pr, 0.71))
    ν_ra = sqrt(Pr / Ra) * U_ref * L_ref
    α_ra = ν_ra / Pr
    gβΔT = Ra * ν_ra * α_ra / L_ref^3
    physics_params[:nu]       = ν_ra
    physics_params[:alpha]    = α_ra
    physics_params[:gbeta_DT] = gβΔT
    physics_params[:Ra]       = Ra
    physics_params[:Pr]       = Pr
end
```

## `Setup prandtl = Pr`

Bare — used in combination with `rayleigh`; has no effect on its own.

## `Setup L_ref = ... U_ref = ...`

Override the reference scales used by `reynolds` and `rayleigh` computations.
Useful when the "natural" defaults are wrong (e.g. a channel where the
relevant length is the half-height, not `min(Nx, Ny)`).

```text
Setup reynolds = 400 L_ref = 64 U_ref = 0.05
```

## Interaction with sanity checks

Back-computed `nu` is fed through the same tau and Mach sanity checks as a
directly-specified `Physics nu = ...`. See [Sanity](sanity.md).
