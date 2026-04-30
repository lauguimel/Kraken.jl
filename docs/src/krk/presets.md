# Presets

Presets expand into plain .krk lines **before** anything else is parsed.
This means a preset can be overridden line-by-line by placing directives
*after* the `Preset` line: later `Physics` / `Domain` / `Boundary` entries
shadow the preset's defaults.

Source: `_expand_preset` + `_preset_lines` in `src/io/kraken_parser.jl`.

```julia
for line in lines
    if _first_word(line) == "Preset"
        append!(expanded, _expand_preset(line))
    else
        push!(expanded, line)
    end
end
```

Known presets:

- `cavity_2d`
- `poiseuille_2d`
- `couette_2d`
- `taylor_green_2d`
- `rayleigh_benard_2d`

Typos trigger a Levenshtein suggestion:
`ArgumentError: Unknown Preset 'cavity2d' (did you mean: cavity_2d?)`.

## `cavity_2d`

Lid-driven cavity at `Re = 10` (`U = 0.1`, `L = 128`, `nu = 0.01`).

```text
Simulation cavity_2d D2Q9
Domain L = 1.0 x 1.0  N = 128 x 128
Physics nu = 0.01
Boundary north velocity(ux = 0.1, uy = 0)
Boundary south wall
Boundary east wall
Boundary west wall
Run 10000 steps
```

**Override example**

```text
Preset cavity_2d
Physics nu = 0.001      # shadows the preset's nu = 0.01 → Re = 100
Run 50000 steps         # shadows the preset's Run
```

## `poiseuille_2d`

Periodic channel driven by a body force `Fx = 1e-5`.

```text
Simulation poiseuille_2d D2Q9
Domain L = 4.0 x 1.0  N = 64 x 32
Physics nu = 0.1 Fx = 1e-5
Boundary x periodic
Boundary south wall
Boundary north wall
Run 10000 steps
```

## `couette_2d`

Plane Couette flow, top plate moving at `ux = 0.05`.

```text
Simulation couette_2d D2Q9
Domain L = 1.0 x 1.0  N = 32 x 64
Physics nu = 0.1
Boundary x periodic
Boundary south wall
Boundary north velocity(ux = 0.05, uy = 0)
Run 5000 steps
```

## `taylor_green_2d`

Doubly periodic decaying Taylor–Green vortex with analytic initial condition.

```text
Simulation taylor_green_2d D2Q9
Domain L = 1.0 x 1.0  N = 64 x 64
Physics nu = 0.01
Boundary x periodic
Boundary y periodic
Initial { ux = 0.05*sin(2*pi*x)*cos(2*pi*y) uy = -0.05*cos(2*pi*x)*sin(2*pi*y) }
Run 5000 steps
```

## `rayleigh_benard_2d`

Rayleigh–Bénard convection at `Ra = 1e5`, `Pr = 0.71`, thermal module active,
hot bottom / cold top.

```text
Simulation rayleigh_benard_2d D2Q9
Domain L = 2.0 x 1.0  N = 128 x 64
Physics nu = 0.02 Pr = 0.71 Ra = 1e5
Module thermal
Boundary x periodic
Boundary south wall T = 1.0
Boundary north wall T = 0.0
Run 20000 steps
```

## Overriding a preset

Because preset expansion is a textual pre-pass, any directive placed after
`Preset <name>` is appended to the expanded block and therefore overrides
earlier entries for the **same** directive kind (second `Physics nu = ...`
wins, second `Run N steps` wins, etc.). Use this to build parametric
variants:

```text
Preset rayleigh_benard_2d
Physics nu = 0.01 Pr = 0.71 Ra = 1e6    # harder case
Diagnostics every 100 [step, Nu, KE]    # add diagnostics absent from preset
```
