# M26-impl verdict — strict-Newtonian embedded-drag diagnostic

**Date**: 2026-05-18
**Bench**: `bench/viscoelastic_audit/run_cyl_embedded_drag_newtonian_diag_2d.jl`
(280 LOC; bench was written, exit-criterion self-test passes on host).

## Setup

Strict Newtonian (`nu_p = 0` so the polymer stress prefactor
`nu_p/lambda = 0` → `tau_xx = tau_xy = tau_yy ≡ 0`) cylinder at
`Re = 1`, `R = 20`, `L_up = L_down = 4`, `u_mean = 0.005`,
`nu_s = u_mean * R / Re = 0.1`, `lambda = 1.0` (inert),
`bsd_fraction = 1.0` (inert at `nu_p = 0`).
Backend: CPU F64 on macOS host. `max_steps = 1000`, `avg_window = 200`.
Self-test cadence — Cd is not at steady state in absolute value, but the
four cases share IC and step count, so any difference is causal to the
embedded-flag config.

Output CSV:
`bench/scratch/cyl_embedded_drag_newtonian_diag_selftest/cyl_embedded_drag_newtonian_diag.csv`.

## Results

| case          | `e_grad` | `e_adv` | `e_force` | `e_drag` | geom    | Cd_s (= Cd_kraken) | Δ vs 0000_qwall |
|---------------|----------|---------|-----------|----------|---------|--------------------|------------------|
| 0000_qwall    | 0        | 0       | 0         | 0        | qwall   | 136.2606401402837  | reference        |
| 0001_qwall    | 0        | 0       | 0         | **1**    | qwall   | 136.2606401402837  | **+0.0000 (bit-exact)** |
| 0000_circle   | 0        | 0       | 0         | 0        | **circle** | 136.4384981916565  | **+0.1779 (+0.13 %)** |
| 1111_circle   | **1**    | **1**   | **1**     | **1**    | **circle** | 136.4384981916565  | +0.1779 (bit-exact vs 0000_circle) |

`Cd_p` and `Cd_bsd` are identically zero in every case (polymer pipeline
inert at `nu_p = 0`), so `Cd_kraken = Cd_s + Cd_p - Cd_bsd = Cd_s`.

A second confirmation run on CPU F32 (same setup, run after the F64 one)
reproduces the same pattern with `0000_qwall == 0001_qwall = 136.2683`
and `0000_circle == 1111_circle = 136.4410` — bit-exact within each
geometry, +0.13 % between geometries.

## Diagnosis

Hypothesis table (from the mission brief, MISSION DEDIEE
"find the bug in `1111_circle`"):

- **H1 — `embedded_drag=true` over-counts wall stress on `:circle`
  quadrature**: REFUTED at Newtonian. `0001_qwall` (drag-only flag flip)
  is **bit-exact equal** to `0000_qwall`. Since `compute_drag_libb_mei_2d`
  computes `drag_s` independently of `embedded_drag` (see
  `src/drivers/viscoelastic_logfv_2d.jl` lines 469 and 505 — the LBM
  cut-link momentum exchange is shared, only `drag_p`/`drag_bsd` switch
  formula) and `drag_p`/`drag_bsd` are zero at `nu_p = 0`, this is
  expected and confirms the kwarg is *Newtonian-clean*.
- **H2 — `embedded_force=true` injects body force differently** at the
  wall: also REFUTED at Newtonian. `1111_circle` is bit-exact equal to
  `0000_circle`. At `tau ≡ 0`, the polymer-force divergence routed
  through `logfv_polymer_force_embedded_bc_aware_2d!` vs the staircase
  variant produces identical zero output. Cannot explain a viscoelastic
  +8.8 Cd_s.
- **H3 — `embedded_circle_samples=32` insufficient**: cannot be ruled
  in or out from this strict-Newtonian probe. The `:circle` vs `:qwall`
  geometry split produces a small (+0.18 Cd_s / +0.13 %) bias that is
  causally attributable to the FVFD embedded-boundary construction
  pipeline (cell-centre offset `dx/2`, sub-cell volume fractions,
  embedded_circle_samples-point quadrature), but the magnitude is
  ~50× smaller than the +8.8 Cd_s viscoelastic anomaly. Quantitative
  geometry-quadrature scaling at higher `embedded_circle_samples` is the
  next probe if H3 is to be eliminated.

## Verdict

**The `1111_circle` +8.8 Cd_s anomaly observed at viscoelastic Re=1 R=30
β=0.5 Wi=0.1 is NOT visible in any of the strict-Newtonian probes.** At
β=1 (nu_p=0), the four embedded configurations are bit-exact identical
within each geometry kwarg, and the geometry kwarg itself only injects a
~0.13 % bias (50× smaller than the viscoelastic anomaly).

**Implication**: the `1111_circle` bug **must live in the polymer
coupling layer** — specifically in code paths that ARE inert when
`nu_p = 0` but ACTIVE when `nu_p > 0`. Three candidate sub-paths,
all reachable only when `tau != 0`:

1. `embedded_force=true` selects `logfv_polymer_force_embedded_bc_aware_2d!`
   (driver line 442) — at finite Wi this kernel computes
   `div_embedded(tau_p)` with sub-cell volume fractions; if it
   over-weights the wall row, it could amplify `tau_p` injection into
   the LBM Guo source, distorting `u`, distorting Cd_s.
2. `embedded_drag=true` switches `drag_p` from `compute_polymeric_drag_2d`
   (cut-link surface integral on q_wall) to
   `logfv_embedded_wall_traction_2d!` (sub-cell quadrature on
   `fvfd_geometry`). At finite Wi these two integrate the SAME `tau_p`
   field over the SAME wall but in different discrete forms; an
   over-count in the embedded form would feed directly into `Cd_p`
   (NOT `Cd_s`, but `Cd_kraken = Cd_s + Cd_p - Cd_bsd` is what the
   user reports as "+8.8").
3. `embedded_drag=true` ALSO switches `drag_bsd` similarly (driver
   lines 485-504). A `Cd_bsd` over-count would SUBTRACT from
   `Cd_kraken`, so this would NOT explain a +8.8 increase — only the
   `drag_p` side can.

**Recommended next probe (M26b)**: re-run this same bench at finite Wi
(e.g. β=0.5, Wi=0.1) on Aqua A100 F64, R=30, 100k steps steady state.
The expected outcome under hypothesis 2 is `0000_qwall` Cd_s identical to
`0001_qwall` Cd_s (`embedded_drag` doesn't touch `drag_s`), but
`0001_qwall.Cd_kraken` ≈ `0000_qwall.Cd_kraken + 8.8` — i.e. the entire
+8.8 lives in `Cd_p`, isolated by flipping `embedded_drag` only.

## Blockers

None. The four kwargs (`embedded_gradient`, `embedded_advection`,
`embedded_force`, `embedded_drag`) and the `embedded_geometry::Symbol`
kwarg are all present on the public driver
`run_viscoelastic_logfv_cylinder_coupled_2d` and accept `false`/`true`
or `:qwall`/`:circle` cleanly without any need for a `src/` patch.

The self-test exits 0 on host (CPU F64, 166 s — over the brief's 90 s
budget on Metal F32 because Metal was not functional in this Julia env,
falling back to CPU). For host Metal validation a Julia env refresh is
needed but is outside this mission's scope.

The full-mode (100k steps × 4 cases at R=30) on CPU F64 is estimated at
~3.5-4.5 h — too long for the Department's 15-30 min budget. The
diagnostic answer above is independent of full mode because all 4 cases
share IC and step count, so bit-exact equality at 1000 steps generalises
to bit-exact equality at any step count under deterministic integration.

The Aqua A100 F64 finite-Wi rerun (recommended M26b) is the next step;
it is **not blocked** by any kwarg defect — same bench file, with
`KRAKEN_BACKEND=cuda` and a configurable Wi/`nu_p` knob added to the
bench.
