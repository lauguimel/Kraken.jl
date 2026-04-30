# Hello KRK

Your first Kraken simulation, running in under two minutes. No Julia script,
no `run_*_2d(...)` call — just a plain text `.krk` file and a shell command.

## What is a `.krk` file?

A Kraken config file (`.krk`) is a declarative description of a simulation —
domain, physics, boundary conditions, run length, outputs. It looks like
this:

```text
Simulation cavity D2Q9
Domain  L = 1.0 x 1.0  N = 128 x 128
Physics nu = 0.128

Boundary north velocity(ux = 0.1, uy = 0)
Boundary south wall
Boundary east  wall
Boundary west  wall

Run 60000 steps
Output vtk every 10000 [rho, ux, uy]
```

That's the lid-driven cavity at Re = 100. Every `.krk` file has the same
shape: one `Simulation`, one `Domain`, one `Physics`, a handful of
`Boundary` lines, a `Run` duration, and one or more `Output` lines.

## Run it

The repository ships a CLI wrapper at `bin/kraken`. Symlink it once:

```bash
ln -s "$(pwd)/bin/kraken" ~/.local/bin/kraken
```

Then from anywhere in the project:

```bash
kraken run examples/cavity.krk
```

That's it. The solver picks up the file, runs on GPU if available (CUDA
auto-detected, Metal/ROCm selectable), and writes VTK snapshots to
`output/cavity_*.vtr` every 10 000 steps.

Open the result in ParaView:

```bash
paraview output/cavity.pvd
```

## Inspect without running

Use `kraken info` to parse the file and see how Kraken understands it,
including the sanity check on `τ` and Mach number:

```bash
kraken info examples/cavity.krk
```

This prints the resolved parameters and flags any instability risk
(`τ < 0.55` or `Ma > 0.1`) without launching the simulation.

## Override parameters from the CLI

Any scalar in `Physics` can be overridden at launch time. Want Re = 1000
instead of Re = 100? Lower the viscosity without editing the file:

```bash
kraken run examples/cavity.krk --nu=0.0128 --max_steps=200000
```

The override syntax is `--key=value`, matching the keyword arguments
accepted by the underlying Julia driver.

## Pick a backend

By default the CLI tries CUDA and falls back to CPU. Force one explicitly:

```bash
kraken run examples/cavity.krk --backend=cpu
```

For Metal (Apple Silicon) and ROCm (AMD GPU) backends, see the
[Installation guide](../../installation.md).

## What did we just do?

The `kraken run` command is equivalent to:

```julia
using Kraken, CUDA
result = run_simulation("examples/cavity.krk"; backend = CUDABackend())
```

…but without needing a Julia REPL, a project activation, or a wrapper
script. The `.krk` file is the **single source of truth** for the
simulation setup.

## Where next?

- **[Build a KRK](02_build_a_krk.md)** — construct a `.krk` from scratch,
  block by block, with an explanation of every directive.
- **[Cookbook](03_cookbook.md)** — recipes for obstacles, STL bodies,
  spatial boundary profiles, thermal coupling, grid refinement, and more.
- **[.krk DSL reference](../../krk/overview.md)** — full directive
  reference when you need the exhaustive syntax.
