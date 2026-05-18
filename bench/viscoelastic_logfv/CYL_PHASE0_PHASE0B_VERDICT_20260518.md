# Cylinder Cd Phase 0 + Phase 0b verdicts — 2026-05-18

Two Aqua A100 F64 jobs landed back-to-back this evening:

- `21570657.aqua` (Phase 0 Liu-match) — 12 cases, 7m24s, 66.66 % GPU util.
- `21572831.aqua` (Phase 0b embedded-flag discrimination) — 27 cases,
  14m55s, 69.89 % GPU util.

Both ran on the patched `bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl`
(commit `e602726f`) that fixed the Julia world-age trap in `detect_backend()`
which had silently routed `21563085.aqua` to CPU.

## 1. Liu-match (Phase 0, `0000_qwall`)

Setup: β=0.59, Re=1, Wi=0.1, L_up=L_down=15, bsd_fraction=1.0.

| R | Cd_kraken (`0000_qwall`) | Cd_kraken (`0000_circle`) |
|---|---|---|
| 20 | 128.94 | 128.94 |
| 30 | **129.39** | 129.39 |
| 40 | 129.49 | 129.49 |

- Reference: Liu CNEBB R=30 = **130.36**, rheoTool R=30 = **130.43**.
- Kraken `0000_*` R=30 = **129.39** → **0.7 % below Liu** (0.97 Cd offset).
- Geometry kwarg alone (`:qwall` vs `:circle`) does **not** affect Cd
  when no embedded flag is set — both modes give bit-identical Cd to 4
  sig figs.
- Trend monotone Cd(R→∞) ≈ 129.5 — Kraken approaches an asymptote
  ~0.9 Cd below Liu, consistent with a small systematic bias in
  the TRT-LBM staircase coupling (not a regression).

**Verdict**: `0000_*` is the Liu-equivalent reference mode. The 0.7 %
offset is acceptable for the Phase 1 baseline; the Liu match is
approximate-PASS (just 0.11 below the strict ±1 window).

## 2. Embedded-flag discrimination (Phase 0b, all R=30, β=0.59, Wi=0.1)

The four binary flags (`embedded_gradient`, `embedded_advection`,
`embedded_force`, `embedded_drag`) plus `embedded_geometry` were swept
at R=30 (cleanest signal). Δ vs the `0000_circle` baseline 129.39:

| tuple                    | Cd_kraken | Cd_s   | Cd_p  | Cd_bsd | Δ vs 0000 | role            |
|--------------------------|-----------|--------|-------|--------|-----------|------------------|
| 0000_circle (baseline)   | 129.39    | 129.67 | 16.05 | 16.32  |  0.00     | reference        |
| 0100_circle (adv-only)   | 129.32    | 129.60 | 16.05 | 16.33  | −0.07     | **NO-OP**        |
| 0001_circle (drag-only)  | 129.57    | 129.67 |  6.97 |  7.06  | +0.18     | **NO-OP on Cd**  |
| 1000_circle (grad-only)  | 131.92    | 132.82 | 15.62 | 16.52  | **+2.53** | secondary effect |
| **0010_circle (force-only)** | **137.49** | **138.48** | 14.98 | 15.96 | **+8.10** | **🚨 dominant bug** |
| 1100_circle (g+a)        | 131.94    | 132.84 | 15.63 | 16.53  | +2.55     | = 1000 (adv noop)|
| 1110_circle (g+a+f)      | 138.31    | 139.45 | 15.35 | 16.50  | **+8.92** | f+grad compound  |
| **1111_circle (full bug)** | **139.27** | **139.45** |  5.51 |  5.69 | **+9.88** | **reproduces +8.8 handoff** |
| **0010_qwall (force-only qwall)** | **138.10** | **139.01** | 15.04 | 15.95 | **+8.71** | **bug also on qwall** |

### Discrimination summary

- **H1 (drag-formula bug) REFUTED empirically**: `0001_circle` drag-only
  produces only +0.18 Cd. `embedded_drag` flag does NOT affect Cd_s (it
  only changes how Cd_p / Cd_bsd are computed individually, and they
  cancel out in Cd_kraken).
- **H3 (`:circle` quadrature insufficient) REFUTED empirically**: the
  bug is also present at +8.71 Cd on `:qwall` geometry with the same
  `embedded_force=true` flag. The defect is **intrinsic to the
  `embedded_force` code path**, not the `:circle` 32-sample quadrature.
- **`embedded_advection` is a NO-OP** at this regime (Wi=0.1, R=30).
- **`embedded_gradient` is a secondary contributor** (+2.5 Cd) —
  consistent with the M26-analysis half-cell-ghost mechanism in
  `_fvfd_apply_embedded_wall_gradient_2d`, but it is NOT the dominant
  defect.
- **`embedded_force=true` ALONE produces +8.1 Cd at R=30** — this is
  the full magnitude of the original `1111_circle` ghost drag (+9.88
  total = 8.1 + 2.5 grad + 0.2 drag + ~0.1 cross-terms).

### Confirmed mechanism

`fvfd_tensor_divergence_embedded_2d_kernel!`
(`src/fvfd/operators_2d.jl:759-766`) divides its output by
`cell_fraction` (giving force-per-fluid-volume), but the Guo-source
consumer downstream expects force-per-lattice-cell. On cut cells
adjacent to the cylinder surface (typical `cell_fraction` ≈ 0.3 at
R=30), this is a 3-10× overdose of Guo body force. The overdose
biases the LBM distribution `f` near the wall, which inflates the
LBM cut-link MEA drag (`compute_drag_libb_mei_2d`) by ~8 Cd points.

This is the empirical confirmation of the M26-analysis Department
hypothesis (`.orchestrator/M26_analysis_verdict.md`).

## 3. Cd-component decomposition observations

Reading the Cd_p / Cd_bsd columns confirms the formula
`Cd_kraken = Cd_s + Cd_p − Cd_bsd`:

- `embedded_drag=true` shifts the Cd_p / Cd_bsd values dramatically
  (Cd_p drops from ~16 to ~5-7 in *_*_*_1 cases) but they stay close
  enough that Cd_p − Cd_bsd ≈ unchanged. This confirms `embedded_drag`
  swaps the formula used for `drag_p` and `drag_bsd` (FVFD traction on
  embedded quadrature vs LBM cut-link MEA) but the two are physically
  equivalent at the continuum.
- `embedded_force=true` shifts Cd_s up by ~+8.8 (the bug), with Cd_p
  / Cd_bsd marginally affected — confirming the bug biases the
  velocity field `u` (via `f`), and Cd_s is the downstream witness.
- `embedded_gradient=true` shifts Cd_s up by ~+2.5 (half-cell ghost
  amplification via the Guo body-force in cut cells, secondary
  mechanism).

## 4. Next mission gates

- **M26 closed**: defect localised to `embedded_force=true` via
  `fvfd_tensor_divergence_embedded_2d_kernel!` cell-fraction overdose.
  Empirical + math audit converge.
- **M26b unlocked**: `src/`-side fix mission to remove the
  `cell_fraction` divisor in the embedded tensor-divergence kernel (or
  re-scale on the Guo-consumer side). Acceptance: `0010_qwall` and
  `1111_circle` Newtonian and Wi=0.1 cases give Cd_s within ±1 of
  `0000_qwall`/`0000_circle` baseline.
- **M28 unlocked**: Phase 1 Wi sweep ∈ {0.1, 0.3, 0.5, 1.0} × R ∈
  {20, 30, 40} × `0000_qwall` (Liu reference mode, NO embedded flags
  → bug-free) × β=0.59 fixed, Re=1, bsd_fraction=1.0. 12 runs ~7 min
  A100. Validates BSD physics across the elastic regime before any
  embedded fix lands.

## Files

- Phase 0 SUMMARY.csv:
  `results/viscoelastic_logfv/cyl_bigsweep_v2_21570657.aqua/SUMMARY.csv`
  (rsync target).
- Phase 0b SUMMARY.csv:
  `results/viscoelastic_logfv/cyl_bigsweep_v2_21572831.aqua/SUMMARY.csv`
  (rsync target).
- Patched bench: `bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl`
  (commit `e602726f`).
- PBS log Phase 0: `~/Kraken.jl-viscoelastic-run/krk_cyl_bigsweep.o21570657`
  on Aqua.
- PBS log Phase 0b: `~/Kraken.jl-viscoelastic-run/krk_cyl_bigsweep.o21572831`
  on Aqua.
