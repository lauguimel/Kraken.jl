# M26b — Option-A cell-fraction rescale verdict — 2026-05-19

Branch `dev-viscoelastic`, worktree `Kraken.jl-viscoelastic`.
Department M26b-fix; Codex Engineer delivered the patch.

## TL;DR

**Status**: PARTIAL FAIL. Option-A consumer-side rescale by
`cell_fraction` was applied as designed but closes only **~8 %** of
the empirical +8 Cd over-dose (`+7.92 → +7.27` Δ at R=20 Wi=0.1 β=0.5
Metal F32). The kernel-side `1/cell_fraction` divisor is therefore
**not the dominant amplifier** at this regime — the M26-analysis
verdict §6 had estimated Fix-B alone would close 30–50 %; observation
is ~8 %.

The patch is **safe to ship** (small, surgical, kernel semantics
unchanged) but it is **insufficient** as the production-ready M26b
solution. A follow-up M26c must address the remaining +7 Cd
residual.

## Patch summary

| File                                                                                | Δ LOC | Concern                       |
|-------------------------------------------------------------------------------------|-------|-------------------------------|
| `src/fvfd/operators_2d.jl` (lines 825–847)                                          | +23   | new helper kernel + wrapper   |
| `src/drivers/viscoelastic_logfv_2d.jl` (lines 446–450, inside `embedded_force` arm) |  +5   | call rescale post divergence  |
| `bench/scratch/m26b_smoke_2d.jl` (NEW)                                              | +148  | smoke runner (Metal/CUDA/CPU) |

The driver patch (only tracked file):

```
diff --git a/src/drivers/viscoelastic_logfv_2d.jl b/src/drivers/viscoelastic_logfv_2d.jl
@@ -443,6 +443,11 @@ function _run_viscoelastic_logfv_step_channel_coupled_2d(
                 fx_poly, fy_poly, tauxx, tauxy, tauyy, fvfd_geometry;
                 sync=false,
             )
+            # Option-A rescale: the embedded kernel returns force per
+            # fluid volume; the LBM Guo source consumes force per lattice cell.
+            fvfd_scale_by_cell_fraction_2d!(
+                fx_poly, fy_poly, embedded, is_solid; sync=false,
+            )
         else
```

SHA-256 of the driver diff: `50551d6d5d50676a5f779b6fe73f9ad0946cb806445f46c5faac4f103269f0ae`.

`src/fvfd/operators_2d.jl` is currently UNTRACKED in this worktree
(the `src/fvfd/` directory has never been committed on
`dev-viscoelastic`); the patch + new helper therefore land together
when the Boss commits later. `grep -n fvfd_scale_by_cell_fraction
src/fvfd/operators_2d.jl` → lines 825/838/843 (kernel + wrapper).

## Smoke test (host Metal F32)

Smoke at `radius=20, H=80, L_up=L_down=4, u_mean=0.005, Re=1, β=0.5,
Wi=0.1, λ=400, max_steps=2000, avg_window=400`,
`embedded_geometry=:circle`.

| Case                                 | Cd_kraken | Cd_s     | Δ Cd_s vs A |
|--------------------------------------|-----------|----------|-------------|
| A: no embedded flags (baseline)      | 138.78    | 140.72   | 0.00        |
| B: `embedded_force=1` PATCHED        | 145.52    | 147.99   | **+7.27**   |
| B': `embedded_force=1` UNPATCHED     | 146.06    | 148.64   | **+7.92**   |

Department control measurement (B' obtained by commenting out the
`fvfd_scale_by_cell_fraction_2d!` call in the driver, running, then
restoring).

**Headline numbers**:

- Patched delta `7.27` vs target `±0.5` ⇒ **FAIL** by 14×.
- Patched-vs-unpatched difference `+7.92 − +7.27 = 0.65 Cd` = the
  fix's effective contribution at this regime.
- Therefore Option A closes only ~**8 %** of the empirical over-dose
  (`0.65 / 7.92`), not the 30-50 % the M26 verdict §6 predicted.

## Why the Option-A fix under-performed

The M26-analysis verdict §6 §"Fix B" hypothesis assumed the
`1/cell_fraction` overdose was THE amplifier. The smoke shows it
contributes ~0.65 Cd of the +7.92. The remaining +7.27 must come
from a different mechanism active in the `embedded_force=true`
branch even after the cell-fraction rescale. Candidate explanations
(none isolated yet):

1. **The wall-segment term** `wall_x_length * tauxx[i,j]` (resp.
   `wall_y_length * tauxy[i,j], ...`) in the kernel
   `fvfd_tensor_divergence_embedded_2d_kernel!` lines 759-766 adds a
   surface-flux contribution **at the cell center** with magnitude
   proportional to `(west_fraction - east_fraction)`. This is a
   singular surface delta in cut cells — it survives the
   cell-fraction rescale because the rescale multiplies the whole
   `(F_volume + F_wall)/V_cell` sum by `V_cell/V_lattice`, but the
   `F_wall` term geometrically belongs to the LBM cut-link MEA
   integral, not to the FVFD body-force source. **Hypothesis**:
   double-counting between the FVFD wall-traction term and the LBM
   Mei MEA. Test: set `wall_x_length = wall_y_length = 0` in the
   kernel ⇒ if Δ drops to <1, this is the residual amplifier.

2. **Implicit interaction with the gradient extrapolation at solid
   neighbours**: the polymer FD stencil
   (`logfv_polymer_force_bc_aware_2d!`) and the embedded variant use
   DIFFERENT extrapolation rules at solid neighbours. The `qwall`
   geometry baseline already has `force_bc_aware` extrapolating
   τ across the `is_solid` mask via the quadratic 3-point helper
   (engineer.md 2026-05-16); the embedded path uses zero-extension
   on solid faces and a wall-segment term instead. The mismatch
   itself produces a small bias.

3. **Tangential vs normal split of the cell-fraction factor**: the
   East/West face flux is rescaled by `min(east_fraction[i,j],
   west_fraction[i+1,j])` BEFORE the volume divisor. If the face
   fraction is, say, 0.7 while the cell fraction is 0.3, the
   post-volume-divisor flux is over-counted by a factor 0.7/0.3 ≈ 2.3
   that the Option-A scale-back by `cell_fraction = 0.3` cannot
   reverse (it multiplies the WHOLE flux including the `0.7` part).

## What was NOT touched

- `_fvfd_apply_embedded_wall_gradient_2d` (`src/fvfd/operators_2d.jl:127-140`)
  — the half-cell ghost write that the M26-analysis flagged as the
  secondary defect (+2.5 Cd via `embedded_gradient=true` per Phase 0b).
  This was deferred per the original brief; it remains a known
  contributor that would compound with the force-path residual if
  both flags are enabled together. Since the smoke isolates
  `embedded_force=true` only, the +7.27 residual is **entirely**
  inside the force-divergence path — the half-cell ghost is not
  active here.

- The Liu-mode bench-side production path
  (`bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl`,
  commit `e602726f`). Out of scope per the brief.

## Risk note for the Boss

- **Forward-compatibility**: the helper `fvfd_scale_by_cell_fraction_2d!`
  is internal (not exported) and only called once in the cylinder
  driver. No risk of silent regression in non-embedded paths
  (cavity, channel, Poiseuille).

- **Drivers NOT exercised by the smoke**: 3D viscoelastic
  (`src/drivers/viscoelastic_3d.jl`), generic viscoelastic
  (`src/drivers/viscoelastic_spec.jl`), and the cavity driver in the
  same file (`src/drivers/viscoelastic_logfv_2d.jl`,
  `run_viscoelastic_logfv_cavity_coupled_2d`). None of them currently
  enable `embedded_force` (grep shows only the cylinder driver does),
  so the patch is inert there.

- **Numerical mode**: Metal F32 (host). Production is CUDA F64; the
  ~0.65 Cd fix contribution should reproduce on F64 (the underlying
  physics is not F32-sensitive — Cd_s differs by <0.05 between F32
  Metal and F64 CPU per the Engineer's CPU control).

- **What might silently break if shipped**: nothing identified.
  The patched and unpatched embedded-force paths both produce finite
  Cd; this is a quantitative correctness shift, not a stability
  regression.

## Recommended next mission (M26c)

Test hypothesis #1 above as a follow-up:

1. Add a kwarg `_drop_wall_segment_term::Bool=false` to
   `fvfd_tensor_divergence_embedded_2d_kernel!`. When `true`, set
   the `wall_x_length` and `wall_y_length` summands to zero.
2. Re-run the M26b smoke with this knob; if Δ drops below 1, the
   wall-segment term is the residual amplifier (likely double-
   counted with LBM MEA).
3. If Δ stays high, test hypothesis #3 (face-fraction vs
   cell-fraction ratio).

Production gate: `0010_circle` and `1111_circle` Cd_s within ±1 of
`0000_circle` baseline at R=30 Wi=0.1 β=0.59 (consistent with the
Phase 0b reference table).

Until M26c lands, **`embedded_force=true` should NOT be used in
production**. The Boss's existing Phase-1 plan (β=0.59 Wi sweep on
`0000_qwall` Liu-mode) is unaffected — that path does not touch
the embedded force kernel.

## Memory candidates for `engineer.md`

- "M26b 2026-05-19: Option-A consumer-side rescale by `cell_fraction`
  in the cylinder `embedded_force` branch closes only ~8 % of the
  +8 Cd over-dose at R=20 Wi=0.1 β=0.5 (delta `+7.92 → +7.27`). The
  `1/cell_fraction` divisor inside
  `fvfd_tensor_divergence_embedded_2d_kernel!` is NOT the dominant
  amplifier — the wall-segment term `wall_x_length * tauxx[i,j]` (and
  the face-fraction-weighted East/West flux) likely contribute a
  much larger bias. Any future embedded-force fix must isolate
  these two contributions separately."

- "Detection helper for Metal backend inside a function trips
  `MethodError` on `MetalBackend()` when constructed via
  `Base.invokelatest(getfield(MetalMod, :MetalBackend))()` — the
  recovered DataType is in the right world but `__init__` triggers
  a world-mismatch. Top-level `@eval using Metal` followed by
  `MetalMod.MetalBackend()` inside the function works. Update the
  M26-impl-era detect_backend pattern accordingly."

## Files

- Patched: `src/fvfd/operators_2d.jl` (lines 825-847 added),
  `src/drivers/viscoelastic_logfv_2d.jl` (lines 446-450 added).
- New: `bench/scratch/m26b_smoke_2d.jl` (148 LOC; Metal/CUDA/CPU
  auto-detect, two-case Δ check, exit 0/1 on pass/fail).
- Engineer brief: `.engineer_brief_M26b.md` (consumed; can be deleted
  by the Boss after merge).
- Engineer log: `.engineer_logs/M26b_<ts>.log`.
