# Visualization guide

Kraken.jl writes standard **VTK** files (`.vtr` rectilinear grids) and
**PVD** time-series collections via
[WriteVTK.jl](https://github.com/jipolanco/WriteVTK.jl). The recommended
viewer is [ParaView](https://www.paraview.org/) — the same tool used by
OpenFOAM, Basilisk, SU2, and most CFD solvers.

For programmatic figure generation (docs, papers, benchmarks), see the
[`KrakenView`](#batch-figures-with-krakenview) section below.

## Quick start

```julia
using Kraken

# Run a case with VTK output
setup = load_kraken("examples/cavity.krk")
run_simulation(setup)

# Open the results in ParaView
open_paraview("output/")
```

The `Output` directive in the `.krk` file controls what gets written:

```
Output vtk every 1000 [rho, ux, uy]
```

This writes a `.vtr` snapshot every 1000 steps, and a `.pvd` file that
ParaView uses to play back the time series.

## `open_paraview` helper

```julia
open_paraview("output/")               # picks the first .pvd found
open_paraview("output/", name="cavity") # opens output/cavity.pvd
```

On macOS, the helper auto-detects ParaView in `/Applications/`. On
Linux/HPC, `paraview` must be on `PATH`.

## What ParaView gives you

| Feature | How |
|---------|-----|
| Scalar heatmaps (ρ, T, C, φ) | Color by field in the pipeline |
| Velocity vectors | Glyph filter |
| Streamlines | Stream Tracer filter |
| Iso-surfaces (VOF interface) | Contour filter on C = 0.5 |
| 2D / 3D slices | Slice or Clip filter |
| Time animation | Play button on the PVD time-series |
| Derived quantities (vorticity, ∇p) | Calculator or Python filter |
| Multi-block (grid refinement) | Native `.vtm` support |
| Export images / videos | File → Save Screenshot / Save Animation |

## Following an HPC run (Aqua)

Typical workflow when running on the QUT Aqua cluster:

### 1. Submit the job

```bash
ssh aqua
cd ~/runs/cavity_fine
qsub run.pbs
```

### 2. Monitor output remotely

```bash
# On your local machine — sync the VTK output periodically
rsync -avz --include='*.pvd' --include='*.vtr' --include='*.vtm' \
      --exclude='*' \
      aqua:~/runs/cavity_fine/output/ ./output_cavity/

# Or watch it live with a loop
watch -n 30 'rsync -avz aqua:~/runs/cavity_fine/output/ ./output_cavity/'
```

### 3. View locally in ParaView

```julia
using Kraken
open_paraview("output_cavity/")
```

ParaView can reload the PVD as new snapshots arrive — use
**File → Reload Files** or enable auto-reload via the Properties panel.

### 4. Rsync workflow (recommended)

For large runs, add this to your PBS script to sync results periodically:

```bash
# In run.pbs — sync every 10 minutes while the job runs
while kill -0 $JULIA_PID 2>/dev/null; do
    rsync -avz output/ $LOCAL:~/results/$(basename $PWD)/output/
    sleep 600
done &
```

Or use the post-job rsync pattern from the Kraken HPC workflow:

```bash
# After job completes
rsync -avz aqua:~/runs/$CASE/output/ output/
julia -e 'using Kraken; open_paraview("output/")'
```

## VTK output API

| Function | Purpose |
|----------|---------|
| `write_vtk(file, Nx, Ny, dx, fields)` | Single 2D snapshot |
| `write_snapshot_2d!(dir, step, ...)` | Numbered 2D snapshot |
| `write_snapshot_3d!(dir, step, ...)` | Numbered 3D snapshot |
| `create_pvd(file)` | Start a PVD time-series |
| `write_vtk_to_pvd(pvd, ...)` | Add snapshot to PVD |
| `write_vtk_multiblock(file, blocks)` | Multi-block for grid refinement |
| `write_snapshot_refined_2d!(...)` | Refined domain snapshot |
| `open_paraview(dir; name="")` | Launch ParaView on output |

## Batch figures with KrakenView

For non-interactive, programmatic figures (documentation, papers,
benchmarks), the `view/` sub-package provides Makie-based figure
generators. This is separate from ParaView — it produces PNGs/SVGs
directly from Julia.

```julia
using Pkg; Pkg.activate("view")
using CairoMakie   # headless backend
using KrakenView

spec = [
    (case        = "cavity Re=100",
     figure_type = :heatmap,
     output      = "cavity_umag.png",
     options     = (colormap=:viridis, title="|u|"),
     krk         = "examples/cavity.krk"),
]

generate_figures(spec; output_dir="figures/")
```

Supported figure types: `:heatmap`, `:profile`, `:convergence`,
`:streamlines`. See the KrakenView API docs for details.
