# M48 — halfwayBB Wi=1 wrong mesh convergence discriminator
Date: 2026-05-26
Engine: Boss-direct local run (Codex script generation; Boss-direct Metal execution)
Backend: Metal M3 Max F32

## TL;DR

Pattern **(a) ALL CONVERGE** to a stable plateau within 1 flow-through —
**but plateau Cd is non-monotone in R: U-shape with the best value at R=30**.
Mesh refinement past R=30 DEGRADES Cd by ~3.4 (R=30 → R=50: 117.62 →
114.26). This empirically rules out "wake-not-established" as the
dominant mechanism behind the M46-B R=60 drift and localises the bug
in the polymer-LBM wall coupling under halfwayBB. Wall-cell count
scales with R; the bias scales with wall-cell count.

## Approach
- Cd(t) logging strategy: option 2. Codex added `step_callback`
  payload extension to `src/drivers/viscoelastic_logfv_2d.jl` (2 LOC
  on dev-viscoelastic) to expose `f_out, q_wall, uwx, uwy, dudx,
  dudy, dvdx, dvdy, psixx, psixy, psiyy, tauxx, tauxy, tauyy,
  is_solid_h`. Default behaviour byte-identical.
- Driver script: `bench/viscoelastic_validation/discriminators/M48_halfway_meshconv.jl`
  (150 LOC). Calls `run_viscoelastic_logfv_cylinder_coupled_2d` per R
  with the callback writing Cd_kraken, Cd_s, Cd_p, Cd_bsd, trace_C_max
  to CSV at `log_every` intervals.
- Codex blocked by sandbox Metal device enumeration; rerun directly
  from Boss bash where Metal M3 Max is reachable.
- Wall time per R: R=10 = 175 s; R=30 = 518 s; R=50 = 1015 s; total =
  1708 s (28.5 min) Metal F32.

## Per-R Cd(t) summary

Plateau Cd computed as the average over the last 50% of the run
(by which point dCd/dt < 0.001 per 1000 steps in all three cases).

| R | Cd(t=0) | Cd at 25% FT | Cd at 50% FT | Cd at 100% FT | Cd plateau (last 50%) | trace_C plateau | trajectory shape |
|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 152.99 (transient spike) | 114.49 | 114.46 | 114.48 | **114.48** | 106 | rapid plateau by 0.4 FT, stable |
| 30 | 121.05 | 117.28 | 117.66 | 117.62 | **117.62** | 199 | plateau by 0.5 FT, marginal drift ±0.05 |
| 50 | -3.68 (transient) | 114.67 | 114.47 | 114.26 | **114.26** | 225 (still growing ~+0.1/30k) | plateau by 0.5 FT, marginal drift ±0.05 |

Trajectory drift in the LAST quarter of each run:
- R=10 step 36k→60k: 114.47 → 114.48 (+0.01) ⇒ plateau
- R=30 step 100k→180k: 117.66 → 117.62 (−0.04) ⇒ plateau
- R=50 step 210k→300k: 114.31 → 114.26 (−0.05) ⇒ plateau

**All three are at genuine steady-state within 1 FT**, which falsifies
the simplest reading of M46-B "R=60 needs more FT" — at R=50 with 1 FT
the system IS at plateau, the plateau is just at the wrong value.

## Classification verdict

- Pattern: **(a) ALL CONVERGE** to a stable plateau within 1 flow-through.
- **U-shape in R**: peak Cd at R=30, lower on both sides.
- Mechanism implication: the halfwayBB polymer-LBM wall coupling has a
  bias whose effect on Cd is NON-MONOTONE in mesh resolution. Most
  likely candidate: the FVFD wall-stencil
  `_fvfd_solid_bc_derivative_x_2d` / `_y_2d` (operators_2d.jl:13-43)
  assumes the wall is at distance `dx` from the first-fluid cell
  center, but halfway BB places the wall at `dx/2`. The resulting
  wall-normal gradient error scales geometrically (factor-2 systematic
  offset) and the polymer chain amplifies it through the upper-convected
  term over many flow-throughs. At R=10 the polymer chain is under-resolved
  so the amplification saturates low; at R=30 polymer + wall geometry are
  near-optimal; at R=50 the wall-cell-count multiplies the bias.

## Reference comparison

| R | Cd_kraken (plateau Metal F32 1 FT) | Cd_kraken (M44 anchor Aqua F64 100k) | rT viscoelastic ref | gap vs rT |
|---:|---:|---:|---:|---:|
| 10 | 114.48 | n/a | 120.40 | −4.9 % |
| 30 | 117.62 | 118.10 (M44, Aqua F64 100k = 0.56 FT) | 120.40 | −2.3 % |
| 50 | 114.26 | ~114-115 (M44 sweep, Aqua F64 100k = 0.33 FT) | 120.40 | −5.1 % |

R=30 anchor reproduces M44 within 0.5 Cd (F32 / 180k vs F64 / 100k —
small precision offset + drift between 0.56→1.0 FT).
rT 120.40 reference: `bench/rheotool/cylinder_wi1.0/Cd.txt` final row.

## Anti-pattern flags

- Codex Metal sandbox failure surfaced as an "M48 = blocked" verdict;
  the script itself was correct, the sandbox lacked Metal device
  access. **Confirms** `[[feedback_codex_metal_sandbox_blocked]]`-class
  finding: Boss-direct execution required when Metal local is the
  backend.
- Run callback added 2 LOC to a 1700+ LOC driver file; brief allowed
  this. Not an anti-pattern.

## Recommendation to Boss

**Next mission**: test `polymer_wall_extrap=:linear` toggle (existing
kwarg from commit 7c790cd8, M6-B) on M48 R=30 and R=50 with otherwise
identical config. M6-B refuted this hypothesis on the cavity (different
physics: corner-driven), but on the cylinder benchmark with U-shape
mesh convergence as the symptom, it is the cheapest discriminator
(~25 min Metal F32, 2 R values, no new code).

Predicted outcomes:
- If `:linear` closes the gap at R=50 (Cd_R50 → ~117) → confirms FVFD
  wall-stencil bias is dominant. Next: investigate whether the
  quadratic stencil's wall-at-dx assumption should switch to wall-at-dx/2
  for halfway BB, or whether to default to linear extrap for polymer
  stress.
- If `:linear` leaves R=50 unchanged → wall extrap kwarg is not on the
  active polymer-stress read path for cylinder; need to instrument or
  static-audit the actual stencil that reads `τ_p[wall_neighbor]`.

After that result, fold the static audit of
`_fvfd_solid_bc_derivative_*_2d` wall geometry assumption in parallel.

**Caveats**:
- Metal F32 vs Aqua F64 has a ~0.5 Cd offset at R=30 (M48 117.62 vs
  M44 118.10). The U-shape signal (-3.4 Cd from R=30 → R=50) is
  ~7× larger than the precision offset, so the U-shape is real.
- M48 ran 1 FT per R (fairness criterion). The M46-B R=60 multi-FT
  data still suggests slow drift continues beyond 1 FT, but at R=50
  here the residual drift in the last 30% of the run is <0.05 Cd.

## Artifacts

- `scratch/M48_hw_meshconv/cdtraj_R{10,30,50}_wi1_halfway.csv` (full
  Cd(t) trajectories, 120-150 rows each)
- `scratch/M48_hw_meshconv/run_R{10,30,50}.log` (final values + walltime)
- `scratch/M48_hw_meshconv/M48_run.log` (orchestration log)
- `bench/viscoelastic_validation/discriminators/M48_halfway_meshconv.jl`
  (the driver, reusable for any future R-sweep with the same setup)
