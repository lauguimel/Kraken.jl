# 3D STL sphere drag (Re = 20)

Validation of the generic 3D STL-obstacle flow path (`run_simulation` →
`run_obstacle_libb_3d`, D3Q19 + interpolated bounce-back). A blockage sweep
extrapolated to the unbounded limit gives free-stream **`C_d ≈ 2.84`, +8.9 % vs the
Clift et al. (1978)** reference `C_d ≈ 2.61` at `Re = 20` — a resolution-limited
geometry gate, not a flaw in the drag path.

![sphere drag convergence](sphere-drag-3d.png)

The confined `C_d` decreases monotonically as the lateral walls recede; a quadratic
least-squares extrapolation of five `R = 16` blockage points (`D/W` from 20 % to 6 %,
R² = 0.9998) gives the free-stream limit `C_d ≈ 2.84`. The drag coefficient is
`C_d = 2 F_x / (u_in² A)`, with `A` the projected frontal area, measured by the
cut-link momentum-exchange integrator on the STL wall.

## Why the residual is resolution-limited

- At fixed 20 % blockage, refining `R = 8 → 16` already moved `C_d` −2.5 %
  (5.58 → 5.44) toward the reference; a resolution extrapolation `R → ∞` would close
  the gap further.
- The `D/W = 6 %` case uses a shorter box (`Nx = 256` vs `384`) to fit Float64 in
  80 GB, biasing its `C_d` marginally high.
- A cylindrical-tube wall correction (Haberman–Sayre) over-corrects for this square
  duct (per-point estimates 2.96–3.24), so the true limit lies below the tube model,
  consistent with the ~2.8 fit.

A combined blockage + resolution extrapolation (future work) is expected to close the
residual to a few percent.

## Validation path

1. **Self-consistency (local, `test/test_sphere_stl_drag_krk.jl`).** The STL `.krk`
   sphere drag reproduces the validated analytic-sphere driver `run_sphere_libb_3d`
   (itself asserted against Clift) to **0.4 %** at matched lattice registration. The
   STL voxelizer is cell-centred (`(i-0.5)·dx`) while the analytic q-wall is
   node-centred (`i-1`); the analytic reference is placed at `cx = 29.5` so the two
   q-wall arrays compare the same lattice obstacle (correlate at 1.0). GPU run; CPU is
   a parse+voxelize smoke.
2. **Physical reference (Aqua, CUDA Float64).** The blockage sweep at fixed resolution
   `R_LU = 16` (`D = 32`), `Re = 20`, extrapolated to `D/W → 0`.

## Reproduce

```bash
# local self-consistency twin (GPU):
julia --project=. test/test_sphere_stl_drag_krk.jl

# Aqua blockage sweep (CUDA F64) — generates the .krk cases + CSV:
qsub bench/geometry_stl/sphere_drag_convergence.pbs      # R=16: 20/14/10% (+R=8 smoke)
qsub bench/geometry_stl/sphere_drag_lowblock.pbs         # R=16: 8/6% (needs an 80 GB GPU)
conda run -n kraken-v0-3-figures python bench/geometry_stl/plot_sphere_drag.py
```

Data: `bench/geometry_stl/sphere_drag_conv*.csv`. Reference: Clift, Grace & Weber,
*Bubbles, Drops, and Particles* (1978), drag correlation
`C_d = (24/Re)(1 + 0.15 Re^0.687)`.
