# Cookbook

Copy-pasteable recipes for the most common `.krk` extensions. Each recipe
is a minimal snippet plus a one-line explanation of what it does. Combine
them to build richer simulations without reading the full [DSL reference](../../krk/overview.md).

## Add an analytic obstacle

A cylinder at `(cx, cy)` with radius `R`:

```krk
Define cx = 2.5
Define cy = 1.25
Define R  = 0.5

Obstacle cylinder { (x - cx)^2 + (y - cy)^2 <= R^2 }
```

Any expression returning a boolean on `(x, y)` works. A step:
`{ x < 1.0 && y > 0.5 }`. A triangle: `{ y < x && y > -x && x < 1.0 }`.
See [Obstacle / Fluid](../../krk/directives.md#obstacle-fluid) for the
full grammar.

## Load an STL body

```krk
Obstacle wing stl(file = "assets/wing.stl", scale = 1e-3, translate = [1.0, 0.5, 0.0])
```

- `scale` multiplies all vertex coordinates before insertion.
- `translate` shifts the body after scaling.
- STL files are rasterised onto the grid at simulation init.

## Set a parabolic inlet

Replace the inlet `Boundary` with a spatial expression:

```krk
Define U = 0.05
Define H = 1.0

Boundary west velocity(ux = 4*U*y*(H - y)/H^2, uy = 0)
```

The expression uses the DSL's spatial variables `x`, `y`, `z`, `t` and
standard math functions (`sin`, `cos`, `exp`, `sqrt`, …). See
[Expressions](../../krk/expressions.md).

## Turn on thermal coupling

```krk
Module thermal
Physics nu = 0.01  Pr = 0.71  Ra = 1e5

Boundary south wall T = 1.0
Boundary north wall T = 0.0
Boundary x     periodic
```

`Module thermal` enables the temperature DDF and unlocks the `T = ...`
kwarg on `wall` boundaries. Scalar T only in v0.1.0 — a spatial
temperature profile on a wall is on the roadmap.

## Start from a preset, override a few bits

Presets expand into canonical directive stacks. Run Rayleigh-Bénard at a
different Rayleigh number without rewriting the domain:

```krk
Preset rayleigh_benard_2d
Physics nu = 0.005  Pr = 0.71  Ra = 1e6

Run 50000 steps
Output vtk every 5000 [rho, ux, uy, T]
```

Later directives override earlier ones (so `Physics` and `Run` here
replace what the preset wrote). The five presets are `cavity_2d`,
`poiseuille_2d`, `couette_2d`, `taylor_green_2d`, `rayleigh_benard_2d`.

## Drive flow with a body force

Useful for periodic channels and natural convection side-cases:

```krk
Physics nu = 0.1  Fx = 1e-5
Physics nu = 0.1  Fx = 0  Fy = -9.81*beta*(T - T_ref)
```

`Fx`, `Fy`, `Fz` accept any expression — constants, functions of
`(x, y, T, …)`, etc.

## Add a grid refinement patch

```krk
Refine nearwall { region = [0.0, 0.0, 1.0, 0.1], ratio = 2 }
Refine tip      { region = [0.4, 0.0, 0.6, 0.1], ratio = 2, parent = nearwall }
```

`region = [x0, y0, x1, y1]` is axis-aligned. `ratio = 2` doubles
resolution; `parent` chains nested levels. See
[Theory → Grid refinement](../../theory/18_grid_refinement.md) for the
Filippova-Hanel rescaling details.

## Initial condition with an expression

```krk
Initial { ux = 0.05*sin(2*pi*x) uy = -0.05*cos(2*pi*y) T = 0.5 }
```

Fields not listed inherit their zero/uniform default.

## Log diagnostics during the run

```krk
Diagnostics every 100 [step, time, KE, uMax, drag, lift]
```

Writes a CSV next to the VTK outputs. Column names must match the
runner's registered diagnostics — see [Postprocess](../../api/postprocess.md).

## Use dimensionless helpers

Instead of hand-picking `nu`, express the setup physically:

```krk
Setup reynolds = 1000  L_ref = 1.0  U_ref = 0.1
```

Kraken back-computes `nu = U_ref * L_ref / reynolds`. For thermal:

```krk
Setup rayleigh = 1e5  prandtl = 0.71  L_ref = 1.0
```

Yields `nu`, `alpha`, `gbeta_DT` consistent with the requested Ra-Pr.
See [Helpers](../../krk/helpers.md) for the formulas.

## Sanity-check before running

```bash
kraken info examples/cavity.krk
```

Prints the fully-resolved setup and flags `τ < 0.55` (instability) or
`Mach > 0.1` (incompressibility breakdown). Cheaper than discovering the
issue after 60 000 steps.

## Run a parameter sweep

```krk
Sweep nu = [0.1, 0.05, 0.02, 0.01]
```

One full simulation per value. Outputs go to `output/<name>_nu=<value>/`.
Combine with `Diagnostics` to build convergence curves automatically.

## Overriding from the CLI

Any `Physics` scalar (and `Nx`, `Ny`, `max_steps`, …) is overridable at
launch:

```bash
kraken run examples/cavity.krk --nu=0.01 --Nx=256 --max_steps=100000
```

Takes precedence over the file's values without editing the file.

---

Need a recipe that isn't here? Open the [.krk DSL reference](../../krk/overview.md)
for the exhaustive syntax, or check the [Examples](../../examples/01_poiseuille_2d.md)
for full-worked simulations.
