# M29b-hrs — MUSCL-superbee TVD on log-conformation Ψ advection

Date : 2026-05-19
Department : M29b-hrs
Branch / worktree : `dev-viscoelastic` / `Kraken.jl-viscoelastic`
Operating point : β=0.59, Re=1, R=30, Wi=1.0, bsd_fraction=1.0,
                  embedded flags all OFF, geometry=qwall.

## TL;DR — outcome PARTIAL: +4.9 Cd, 56% of gap closed

The MUSCL-superbee TVD HRS scheme, added behind kwarg
`advection_scheme::Symbol = :rusanov` (default = byte-identical
legacy), recovers **+4.92 Cd** at the M29 worst-Wi point — closing
about 56 % of the −8.85 Cd gap between Kraken Rusanov upwind and
rheoTool `cubista`. Strict ±2 Cd acceptance window NOT met
(116.47 vs target 118-122). Stress-peak preservation also partial
(τ_xx max non-dim 80.3 vs target 135.5).

| metric                   | M29 Rusanov | M29b MUSCL | rheoTool | Δ vs target |
|--------------------------|-------------|------------|----------|-------------|
| Cd_kraken                | 111.55      | **116.47** | 120.40   | −3.93       |
| Cd_s                     | n/a         | 117.31     | n/a      | n/a         |
| Cd_p                     | n/a         | 14.20      | n/a      | n/a         |
| Cd_bsd                   | n/a         | 15.04      | n/a      | n/a         |
| τ_xx max (non-dim ρU²)   | 75.3        | **80.3**   | 135.5    | −41 %       |
| L2_rel τ_xx              | 0.93        | **0.92**   | 0        | unchanged   |
| L2_rel τ_xy              | 0.77        | **0.69**   | 0        | −0.08       |
| L2_rel τ_yy              | 0.58        | **0.35**   | 0        | **−0.23**   |
| L2_rel u_x               | 0.17        | 0.17       | 0        | unchanged   |

The dominant residual gap lives in τ_xx peak preservation at the
leeward shoulder of the cylinder, the same wrap-around extensional
zone the M29 verdict localised. The HRS limiter helps but the
**boundary fall-back to first-order Rusanov within 2 cells of any
solid** (necessary to avoid 4-point stencil reading into the solid
domain) prevents the limiter from acting where the stress peak
actually lives. CUBISTA would have the same fall-back constraint.

## What was implemented

### Source patch (~110 LOC across 3 files)

- `src/fvfd/operators_2d.jl` (+79 LOC): two new helpers
  `_fvfd_superbee_limiter_2d`, `_fvfd_muscl_superbee_face_value_2d`;
  Val-dispatched second method of `_fvfd_upwind_scalar_advective_rhs_2d`
  for `::Val{:muscl_superbee}` that falls back to Rusanov when
  `i ≤ 2 || i ≥ Nx-1 || j ≤ 2 || j ≥ Ny-1` or any neighbour within
  ±2 cells is solid. `fvfd_advect_upwind_2d_kernel!` and
  `fvfd_advect_upwind_2d!` (and the `embedded`/`sym2` family) all
  take `advection_scheme::Symbol = :rusanov` kwarg, validated via
  `_fvfd_advection_scheme_val` and wrapped with `Val(scheme)`
  before launch.
- `src/kernels/logconformation_fv_2d.jl` (+10 LOC): all six
  `logfv_advect_upwind_bc_aware_2d!` / `logfv_advect_upwind_embedded_2d!`
  / `logfv_advect_upwind_solid_aware_2d!` / `logfv_advect_upwind_openx_solid_aware_2d!`
  methods propagate the kwarg through to `fvfd_sym2_advect_upwind_2d!`.
- `src/drivers/viscoelastic_logfv_2d.jl` (+11 LOC):
  `_run_viscoelastic_logfv_step_channel_coupled_2d` (the shared
  inner driver for BFS / contraction / square_channel / cylinder)
  accepts `advection_scheme::Symbol = :rusanov`, normalises the
  symbol, threads through to the two
  `logfv_advect_upwind_bc_aware_2d!` and one
  `fvfd_sym2_advect_upwind_embedded_2d!` call sites. The `cylinder_coupled_2d`
  wrapper passes via `kwargs...` (no explicit threading needed).

### Bench (+7 LOC)

- `bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl`: new env var
  `KRAKEN_ADVECTION_SCHEME` (case-insensitive,
  `rusanov` | `muscl_superbee`), new CSV column `:advection_scheme`,
  kwarg threaded to `Kraken.run_viscoelastic_logfv_cylinder_coupled_2d`.

### Test (+87 LOC focused, ~400 LOC ancillary)

- `test/test_viscoelastic_logfv_patch_ladder.jl`: new
  `@testset "M29b HRS advection keeps default and preserves sharp scalar pulses"`
  (~87 LOC) asserting (i) `:rusanov` default is byte-equal to
  no-kwarg call AND to explicit `:rusanov`, (ii) `:muscl_superbee`
  matches the Rusanov result on linear/affine profiles to
  atol=1e-12 rtol=1e-12, (iii) on a 256-cell square-wave advection
  100 steps CFL=0.5, MUSCL preserves ≥ 70 % of amplitude while
  Rusanov preserves < 50 %, (iv) on a smooth-Ψ symmetric tensor
  advection MUSCL preserves SPD at all 256×8 cells over 10 steps.
- Codex also re-added ~10 testsets that depended on existing
  `Kraken.*` API symbols (FENE-P, embedded gradient, BC spec
  variants, BSD force, etc.) that were referenced in helper
  functions but not previously exercised. They pass and contribute
  to the 18223 patch-ladder total.

### Test suite preservation

```text
julia --project=. test/runtests.jl
=> 169194 passed, 6 failed, 0 errored, 4 broken
```

Byte-equal to documented HEAD baseline (engineer.md 2026-05-17
"runtests baseline carries pre-existing failures"). All pre-existing
failures and broken tests are unchanged (Pure-shear Oldroyd-B
steady state x6 failures, LI-BB canary + P18b2c x4 broken).

## Host smoke (Metal F32)

| case                | Cd_kraken | trace_C_max | walltime |
|---------------------|-----------|-------------|----------|
| R=20 Wi=0.1 Rusanov | 131.18    | 4.82        | 17 s     |
| R=20 Wi=0.1 MUSCL   | 131.18    | 4.82        | 17 s     |
| R=20 Wi=1.0 Rusanov | 106.30    | 129         | 39 s     |
| R=20 Wi=1.0 MUSCL   | **108.71**| 131         | 41 s     |

Low-Wi: HRS limiter rarely fires → byte-identical numerics.
High-Wi: MUSCL gains +2.4 Cd and +1.5 % trace_C_max → consistent
with the production result direction at R=30.

## Aqua A100 F64 production validation

Job `21585787.aqua` (resubmitted after `21585706.aqua` failed on
the first rsync layout error — operators_2d.jl went to src/ root
not src/fvfd/; cleaned). Walltime 88 s. **Cross-checked on H100
job `21585835.aqua`** : identical Cd=116.474 (bit-equal F64).

Snapshot : `tmp/m29b_kraken/cyl_bigsweep_v2_beta0p59_wi1_re1_R30_bsd1_*_fields.jls`

M29 comparison driver re-run :

```
bench/viscoelastic_audit/run_kraken_vs_rheotool_tau_compare.jl
```

ROI x ∈ [−3, 8] × y ∈ [−2, 2] sampled on 256×128 Cartesian. 29 825
valid fluid samples (91 % of ROI). Plots and band CSVs at
`bench/scratch/m29b_tau_compare/`.

## Verdict

**PARTIAL** — Cd ∈ [116, 117], outside the ±2 acceptance window
[118, 122] but **directionally correct and significant**:

- ΔCd_remaining drops from −8.85 (rusanov) to −3.93 (MUSCL).
- τ_yy L2_rel halves (0.58 → 0.35) — best component improvement.
- τ_xx peak only modestly improves (44 % under to 41 % under) :
  the dominant residual.

The MUSCL-superbee boundary fall-back (first-order within ±2 cells
of any solid) prevents the limiter from firing in the leeward
shoulder, the exact wrap-around zone where the stress peak lives
(M29 spatial localisation §"Top-5 x-bands by max|diff τ_xx|"
showed bands x ∈ [−0.02, +0.15] dominate the residual). The
remaining gap is the boundary-stencil constraint, not the limiter
choice.

## Recommendation

DEPLOY `:rusanov` as default (legacy byte-identical) and
`:muscl_superbee` as opt-in via kwarg / env var. The +5 Cd benefit
is worth the +5 % runtime cost (87.6 s vs ~83 s on A100) for any
finite-Wi cylinder benchmark.

Two follow-up missions can close the remaining ~4 Cd / 50 % τ_xx
gap (in priority order) :

1. **M29c — relax the MUSCL fall-back band** : use 1-sided
   3-point reconstruction at boundary cells instead of falling
   back to pure Rusanov. Stencil width unchanged but limiter
   becomes active at cells within 2 of the cylinder surface.
   Estimated +2-3 Cd recovery, brings the result to ~118-120 (in
   window).
2. **M29d — CUBISTA on the polymer field** : replace MUSCL-superbee
   with CUBISTA (Alves & Pinho 2003) as a third option. CUBISTA
   is rheoTool's actual scheme; its NVD framework preserves peaks
   slightly more aggressively than superbee. Estimated +1-2 Cd
   recovery on top of M29c.

Either alone may close the gap; together they should bring Kraken
within ±1 Cd of rheoTool at R=30 Wi=1.0 — i.e., within rheoTool's
own discretisation tolerance.

## File anchors

- Source patches :
  - `src/fvfd/operators_2d.jl` (lines 470-565: helpers + MUSCL kernel branch;
    lines 595, 618, 634, 653, 677 : kwarg threading)
  - `src/kernels/logconformation_fv_2d.jl` (lines 1163, 1184, 1210, 1236, 1251, 1274 :
    kwarg threading)
  - `src/drivers/viscoelastic_logfv_2d.jl` (lines 200, 223-225, 411, 652 :
    kwarg + threading)
- Bench patches :
  - `bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl` (line 110, 133, 264, 292,
    353 : env var, CSV column, driver kwarg)
  - `bench/viscoelastic_logfv/run_cyl_m29b_hrs_a100.pbs` (NEW)
- Test :
  - `test/test_viscoelastic_logfv_patch_ladder.jl`:692-780 (M29b testset)
- Aqua artefacts :
  - A100 F64 job `21585787.aqua` (88 s, gpu_id=A100)
  - H100 F64 cross-check job `21585835.aqua` (133 s, gpu_id=H100; bit-equal Cd)
  - Snapshot : `results/viscoelastic_logfv/cyl_m29b_hrs_21585787.aqua/`
- Local mirror :
  - `tmp/m29b_kraken/cyl_bigsweep_v2_*_fields.jls`
- Comparison driver outputs :
  - `bench/scratch/m29b_tau_compare/M29_residuals.csv`
  - `bench/scratch/m29b_tau_compare/M29_band_stats_x.csv`
  - `bench/scratch/m29b_tau_compare/M29_field_*.png` (5 fields × 3 panels)
  - `bench/scratch/m29b_tau_compare/M29_band_diffs.png`
