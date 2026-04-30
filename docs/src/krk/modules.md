# Modules

A `Module <name>` directive toggles an optional parser/runner path. In this
branch, only `thermal` is public through `run_simulation`.

The parser stores module names as symbols. It may accept names that the
v0.1.0 runner later rejects. Treat the runner behavior as authoritative for
this branch.

## `thermal`

**Status:** supported.

```text
Module thermal
```

What it activates:

- allocation of a second distribution function for temperature transport;
- scalar fixed-temperature wall values such as `Boundary west wall T = 1.0`;
- thermal drivers for heat conduction, Rayleigh-Benard and natural convection;
- Boussinesq coupling when the selected thermal case requires it.

Example:

```text
Simulation rb D2Q9
Domain L = 1.0 x 1.0  N = 64 x 64
Module thermal
Physics nu = 0.05 Pr = 0.71 Ra = 1e3
Boundary west wall T = 1.0
Boundary east wall T = 0.0
Boundary north wall
Boundary south wall
Run 30000 steps
```

## Parser-visible but not public here

The parser can collect other module names, but `run_simulation` rejects the
advanced physics modules in this branch:

- `axisymmetric`
- `advection_only`
- `twophase_vof`
- `rheology`
- `viscoelastic`
- `species`

For example, `Module axisymmetric` may parse, but the runner raises an error
instead of silently running an unsupported solver. This avoids advertising
development-branch work as a v0.1.0 feature.

## Adding a module

Adding a public module requires three pieces to agree:

1. parser syntax in `src/io/kraken_parser.jl`;
2. runtime dispatch in `src/simulation_runner.jl`;
3. tests, examples, and documentation in this branch.

Do not document a module as supported until all three are present.
