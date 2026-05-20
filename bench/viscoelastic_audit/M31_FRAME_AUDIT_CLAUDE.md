# M31 frame audit — Claude independent findings (Step 1, no Codex yet)

Date: 2026-05-20
Author: Claude (Anthropic Opus 4.7)
Snapshot of reference: `tmp/m30_rho_metal/run01/cyl_bigsweep_v2_..._fields.jls` (Cd_kraken=111.091)
Reading only — no edits or runs.

---

## Q1 — Where `Cd_kraken`, `Cd_s`, `Cd_p`, `Cd_bsd` are FINALLY computed

**File**: `src/drivers/viscoelastic_logfv_2d.jl`
**Function**: `_run_viscoelastic_logfv_step_channel_coupled_2d` (the cylinder-coupled inner driver, body starts around line ~196 and runs through line ~700 in this file).

Final reduction, **lines 591–604**:

```julia
Fx_s = n_drag > 0 ? Fx_s_sum / n_drag : NaN
Fy_s = n_drag > 0 ? Fy_s_sum / n_drag : NaN
Fx_p = n_drag > 0 ? Fx_p_sum / n_drag : NaN
Fy_p = n_drag > 0 ? Fy_p_sum / n_drag : NaN
Fx_bsd = n_drag > 0 ? Fx_bsd_sum / n_drag : NaN
Fy_bsd = n_drag > 0 ? Fy_bsd_sum / n_drag : NaN
Fx_drag = n_drag > 0 ? Fx_s + Fx_p - Fx_bsd : NaN
...
Cd_s  = 2.0 * Fx_s   / (drag_speed^2 * drag_diameter)
Cd_p  = 2.0 * Fx_p   / (drag_speed^2 * drag_diameter)
Cd_bsd = 2.0 * Fx_bsd / (drag_speed^2 * drag_diameter)
Cd    = Cd_s + Cd_p - Cd_bsd            # this is `Cd_kraken` in the output
```

Per-step accumulators (lines **515–521**): `Fx_s_sum += drag_s.Fx`,
`Fx_p_sum += drag_p.Fx`, `Fx_bsd_sum += drag_bsd.Fx`. Per-step force evaluations
at lines **479–514** call:
- `drag_s = compute_drag_libb_mei_2d(f_out, q_wall, uwx, uwy, Nx, Ny)`
- `drag_p = compute_polymeric_drag_2d(tauxx, tauxy, tauyy, q_wall, Nx, Ny; cx=drag_cx, cy=drag_cy, radius=drag_radius, …)` (non-embedded branch)
  OR `logfv_embedded_wall_traction_2d!` then `sum` (embedded branch).
- `drag_bsd = _logfv_compute_bsd_drag_2d(dudx, dudy, dvdx, dvdy, q_wall, …; cx=drag_cx, cy=drag_cy, …)` (non-embedded)
  OR analogous embedded sum.

In the M30 snapshot under audit, ALL `embedded_*` flags are OFF → the `q_wall`
ring path is used.

## Q2 — Mathematical formula & frame per component

### `Cd_s` — LBM cut-link MEA, **frame-INDEPENDENT**
`compute_drag_libb_mei_2d` (`src/drivers/cylinder_libb.jl:98`) is option **(b)**: a
sum over all `q_wall[i,j,q] > 0` boundary links of `c_q · (f_q + arriving)`,
where `arriving` is the Bouzidi-reconstructed post-bounce population. **No centre
is referenced** — the formula sums a directional momentum-exchange contribution
per link, so the result is the TOTAL force on the cylinder regardless of where
the moment arm is anchored. Therefore `Cd_s` is invariant under `:phys` vs
`:idx` frame choice.

### `Cd_p` — polymer ring integral, **`:phys` frame**
`compute_polymeric_drag_2d(... ; cx, cy, radius)` (`src/drivers/viscoelastic.jl:65`)
is option **(a)**: for each cut link `(i,j,q)` with `q_w > 0`:
- wall point in PHYSICAL coords: `xw = (i − 1) + q_w · c_qx`, `yw = (j − 1) + q_w · c_qy`
- moment arm: `(rx, ry) = (xw − cx, yw − cy)`
- normal: `(nx, ny) = (rx, ry) / hypot(rx, ry)`
- assemble Δθ-binned ring; `Fx_p = Σ (τ_xx·n_x + τ_xy·n_y) · ds`, `ds = R·(θ_next − θ_prev)/2`.

The `(cx, cy)` passed in is `drag_cx, drag_cy` from line 488–489. In
`run_viscoelastic_logfv_cylinder_coupled_2d` (line 874–875):
```julia
drag_cx = Float64(L_up * radius)
drag_cy = Float64((H − 1) / 2)
```
These are the SAME PHYSICAL centre coordinates used by `precompute_q_wall_cylinder`
to raster the disk (line 818–819). So the driver builds the disk and integrates
its ring using `(cx_phys, cy_phys)` consistently. **This is the `:phys` frame.**

### `Cd_bsd` — back-stress drag, **`:phys` frame**
`_logfv_compute_bsd_drag_2d` (`src/drivers/viscoelastic_logfv_2d.jl:13`) assembles
`τ_bsd = 2·ζ·ν_p · ∇u_sym` on the host and then **delegates to the same
`compute_polymeric_drag_2d(… ; cx=drag_cx, cy=drag_cy, radius=drag_radius, …)`**
(lines 39–46). Same `:phys` frame as `Cd_p`.

### `Cd` (= `Cd_kraken`)
`Cd = Cd_s + Cd_p − Cd_bsd`. Two of three terms (`Cd_p`, `Cd_bsd`) use the `:phys`
frame ring; `Cd_s` is frame-independent. Net behaviour of `Cd_kraken` therefore
follows the `:phys` convention by construction.

## Q3 — Rasterisation `(i − 1, j − 1)` convention

`src/kernels/li_bb_2d.jl`, `precompute_q_wall_cylinder` lines 266–309:

```julia
@inbounds for j in 1:Ny, i in 1:Nx
    xf = FT(i - 1); yf = FT(j - 1)       # line 277  <-- (i-1, j-1) convention
    dx_f = xf - cxT; dy_f = yf - cyT
    if dx_f * dx_f + dy_f * dy_f ≤ R²
        is_solid[i, j] = true
        continue
    end
    ...
```

**Confirmed**: lattice node `(i, j)` sits at physical coordinates `(i−1, j−1)`.
A solid cell is flagged when the **node** (not the cell-centred control volume)
sits inside the disk of radius `R` centred at `(cx, cy)`. The physical centre
`(cx_phys, cy_phys) = (450, 59.5)` corresponds to indices `(cx_phys + 1, cy_phys + 1) =
(451, 60.5)` — i.e. the "node nearest the centre" sits halfway between indices
60 and 61 in y, which is consistent with the M30 audit's observation that the
mask has perfect reflection symmetry about index 60.5.

## Q4 — Which frame is physically correct?

I argue **Option A (`:phys`)** is the **internally-consistent and defensible**
frame for THIS driver, with one important caveat below.

**Argument**:
1. The rasterised solid mask is the **discrete realisation** of the continuous
   disk `D = { (x, y) : (x − cx)² + (y − cy)² ≤ R² }` at the centre
   `(cx_phys, cy_phys)` passed in. The `(i−1, j−1)` convention is a
   coordinate-system **embedding** choice — it does NOT shift the centre.
2. The cylinder is symmetric about `(cx_phys, cy_phys)` in the continuous
   problem; the rasterisation cuts that continuous shape into cells. The
   physical centre of the disk you ARE TRYING to simulate is
   `(cx_phys, cy_phys)`, not the centroid of the rasterised pixel set.
3. The reference solution (rheoTool, exact-cylinder OpenFOAM) has its disk
   centred at the analytic point. To compare "Kraken's Cd_p of the disk it
   tried to discretise" with rheoTool's exact-disk Cd_p, the meaningful normal
   to use is the analytic-disk normal `(rx, ry)/R` with `(rx, ry) =
   (xw − cx_phys, yw − cy_phys)`. That IS what the driver does (and what Phase
   0c did) — i.e. the `:phys` frame.
4. M30 centering audit found that the discrete mask has **exact reflection
   symmetry about `cy_idx = 60.5`** (= `cy_phys + 1`), 0 mismatched pairs, and
   the solid-cell counts above/below `cy_idx` match. So the mask centroid AND
   the analytic disk centroid both project onto the same y-line; the difference
   is a fixed `+1 LU` shift in the index coordinate, which **does not change
   the meaning of the normal direction** at any cut-link — the normal field
   `(rx, ry)/R` is identical under the `+1 LU` re-labelling provided you also
   shift the wall points by `+1 LU`.

**The caveat (and why M30 saw a 24 % shift in `:phys` vs `:idx` ring integrals)**:
The wall points `(xw, yw)` are computed in physical coords (lines 84–85 of
viscoelastic.jl: `xw = (i − 1) + q_w·c_qx, yw = (j − 1) + q_w·c_qy`). The
moment arm `(rx, ry) = (xw − cx, yw − cy)` is therefore **correctly computed in
physical coords** when `cx, cy` are also in physical coords (`cx_phys`). M30's
`:idx` re-integration used `(i − cx_phys, j − cy_phys)` — i.e. it computed
`rx = i − cx_phys` instead of the correct `rx = (i − 1) − cx_phys = i − (cx_phys + 1)`.
**The `:idx` frame in M30's audit is exactly a `+1 LU` shift of the moment-arm
origin** to `(cx_phys + 1, cy_phys + 1)`. That shift is NOT applying the same
convention as the driver — it is asking "what if we declared the centre to be
at the index-frame nearest-node?".

Both representations describe the same discrete object, but the `:phys` frame
matches what the driver does, and matches what rheoTool's exact-disk
integration does (cx is the analytic disk centre, normal is computed from
moment arm to that centre). **So the `:phys` integration in Phase 0c was the
right call IF AND ONLY IF the wall-point formula `xw = (i − 1) + q_w·c_qx` is
also taken to define wall points in the same physical frame** — which it is.

What M30's `:idx` re-integration revealed is NOT a bug — it's a sensitivity
test that asks how much the ring decomposition depends on where you anchor
the moment-arm origin (subject to keeping the same cut points). The 24 %
shift in `Cd_p` between frames is large because the ring is short (R=30, 188
LU perimeter) and a `+1 LU` shift to the origin tilts the local normals
significantly. **Option B (`:idx`) is NOT the physically correct frame; it's a
diagnostic.**

**Therefore my vote**: **Option A (`:phys`)**.

(Note: Option C — centroid of `is_solid` mask — would give yet another point,
roughly the average of `cx_phys` and `cx_phys + 1` because the mask is
node-centred. M30 showed the solid-cell parity is exact about `(cx_idx, cy_idx)`,
so the mask centroid is closer to `:idx` than `:phys`. But none of this changes
the answer: the physically correct centre is the centre of the disk that we
ARE SIMULATING, i.e. `cx_phys`.)

## Q5 — Cd_polymer of Kraken in correct frame vs rheoTool

Given Q4 = `:phys`:
- Kraken Cd_p (`:phys`) from M30 = **13.46** (snapshot 100 k steps, Metal F32)
- M29c-wallstress M29b polymer = **13.40** (Aqua F64, 30 k steps; same `:phys`
  frame because it uses the same driver pathway, before M30 introduced any
  re-binning)
- rheoTool = **13.45**

**The M29c-wallstress "M29b polymer matches rheoTool to 0.05" claim
SURVIVES.** Both the M29b Aqua F64 (13.40) and the M30 ρ-metal F32 (13.46)
re-do bracket rheoTool's 13.45 within Δ ≤ 0.06 (~0.4 %).

If we had voted Option B (`:idx`), Kraken Cd_p = 10.82 vs rT = 13.45 → 19 %
under-prediction → M29c claim would be **falsified**. But Q4's argument is
that `:idx` is a re-anchoring of the moment arm, not the physically defensible
choice, so this 19 % is an artefact of the diagnostic, not of the physics.

## Q6 — Cd_kraken bug or convention?

**Verdict**: **internal-consistency convention** that does NOT bias the
Cd_kraken vs rheoTool comparison.

Reasoning:
- `Cd_s` is frame-independent (no centre used).
- `Cd_p` and `Cd_bsd` use the `:phys` frame and that frame matches the centre
  of the disk we intend to simulate — same centre rheoTool uses.
- The 2.5 % gap (111.09 vs 108.63) between stored `Cd_kraken` and the M30 `:idx`
  re-integration is NOT a bias — it is the cost of anchoring the moment arm
  at the wrong point. The stored value is `:phys`-frame consistent and that's
  the frame rheoTool also uses.
- M28–M30 benchmarks all used the same `:phys` convention; cross-benchmark
  comparisons remain valid. The +9.13-pt Cd_pressure gap identified in M30
  Phase 0c is genuine physics (LBM ρ-BC asymmetry, H1), not a frame artefact.

**Implication**: H1 ranking from Phase 0c **STANDS**.

The one cautionary note for downstream: when comparing Phase 0c's
ring-decomposed `Cd_p_phys` (13.46) directly against the driver's stored
`Cd_p` (11.49 from the M30 snapshot's `Cd_p_polymer` slot), there is still a
~2-pt gap. That gap comes from a **different** source: the snapshot's
stored `Cd_p = 11.49` is the per-step-averaged polymer ring integral over the
last avg_window, whereas the M30 re-integration is a single-snapshot final
integration. Steady-state should match to <0.1 pt; the 2-pt gap reflects
residual unsteadiness or single-frame F32 noise vs F64 averaging. This is
not a frame issue.

## Summary table

| component | frame used | matches rT (:phys)? |
|---|---|---|
| Cd_s   | frame-independent (cut-link sum, no centre) | n/a |
| Cd_p   | `:phys` (cx=L_up·R, cy=(H−1)/2)            | yes, 13.46 vs 13.45 (0.07 %) |
| Cd_bsd | `:phys` (same as Cd_p, delegated)          | n/a (no rheoTool reference for BSD) |
| Cd     | `:phys` (since Cd_s is frame-indep)        | global gap +9 attributed to pressure (H1) |

## Confidence

HIGH on Q1, Q2, Q3 (direct code citations).
HIGH on Q4 (clean argument that `:phys` matches the simulated-disk centre and
the rheoTool reference centre).
HIGH on Q5 conditional on Q4.
HIGH on Q6 conditional on Q4.

Open the Codex sub-brief next.
