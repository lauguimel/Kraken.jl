# M26 — `1111_circle` +6.7 % Cd_s defect — analytical verdict

Branch `dev-viscoelastic`, worktree `Kraken.jl-viscoelastic`.
Pure-analysis Department (no Engineer, no `src/` edits). Verdict gates
the H1/H2/H3 triage and proposes a fix design.

---

## 1. Path map — where the 4 embedded modes live

All `embedded_*` flags are kwargs of
`run_viscoelastic_logfv_cylinder_coupled_2d` in
`src/drivers/viscoelastic_logfv_2d.jl` (declared at lines 193-198, parsed
215-227, lowered 238-250).

| Flag                  | Branch entry           | Kernel called                                                                                                | File                                                            |
|-----------------------|------------------------|---------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------|
| `embedded_advection`  | driver lines 386-397    | `logfv_cell_velocity_to_faces_embedded_2d!` (line 1119+) → `fvfd_cell_velocity_to_faces_embedded_2d_kernel!`   | `src/kernels/logconformation_fv_2d.jl`, `src/fvfd/operators_2d.jl:243`  |
| `embedded_gradient`   | driver lines 408-418    | `fvfd_velocity_gradient_embedded_2d!` → `fvfd_velocity_gradient_embedded_2d_kernel!` (line 979)                | `src/fvfd/operators_2d.jl:979-1020`                              |
| `embedded_force`      | driver lines 441-451    | `logfv_polymer_force_embedded_bc_aware_2d!` (line 660) → `fvfd_tensor_divergence_embedded_2d_kernel!` (line 665) | `src/fvfd/operators_2d.jl:665-770`                               |
| `embedded_drag`       | driver lines 470-504    | `logfv_embedded_wall_traction_2d!` (line 669) → `fvfd_embedded_wall_traction_2d_kernel!` (line 825)            | `src/fvfd/operators_2d.jl:825-862`                               |

`embedded_geometry` controls the lowering only — selects
`fvfd_embedded_boundary_from_circle_2d` (`src/fvfd/lowering_2d.jl:299-377`)
versus `fvfd_embedded_boundary_from_qwall_2d` (lines 396-556) when
building the `FVFDEmbeddedBoundary2D` struct (`wall_nx, wall_ny,
wall_fraction, cell_fraction, {west,east,south,north}_fraction,
wall_distance, cut_count`).

LBM cut-link drag (always called, independent of `embedded_drag`) is
`compute_drag_libb_mei_2d` in `src/drivers/cylinder_libb.jl:98-163`
(CPU host loop) or its GPU twin
`compute_drag_libb_mei_2d_gpu!` in `src/kernels/drag_gpu.jl:151-181`.

Cd post-processing in driver lines 581-594: `Cd = Cd_s + Cd_p − Cd_bsd`
with `Cd_s = 2·Fx_s/(u_ref²·D)` and `Fx_s = drag_s.Fx` averaged over
`avg_window`. Critically, **`drag_s` is ALWAYS** `compute_drag_libb_mei_2d`,
independent of the embedded flags (driver line 469). Only `drag_p` and
`drag_bsd` branch on `embedded_drag`.

So the headline Cd_s number that "+8.8" lands on is **the LBM
cut-link Mei MEA itself**, not the FVFD traction integral — the bug
cannot live in `fvfd_embedded_wall_traction_2d_kernel!` directly. The
defect must be that **the LBM `f` field that feeds the MEA is itself
biased** when the four embedded paths route polymer-pipeline data
differently into the LBM body force.

This single observation re-orients the hypothesis ranking: H1 (as
originally framed, "embedded_drag formula over-counts") is structurally
impossible because the Cd_s value reported by the bench
`run_cyl_bigsweep_v2_2d.jl` is sourced from the LBM-side cut-link drag
in every column of the CSV.

## 2. LBM cut-link drag formula

Implemented in `_compute_drag_libb_mei_2d_host`
(`src/drivers/cylinder_libb.jl:129-163`) — Mei-Luo-Shyy
Momentum-Exchange Algorithm (MEA), Mei et al. JCP 161 (2000), as used
by Liu 2025 Eq. 63:

```
for each cut link (i, j, q) with q_w = q_wall[i,j,q] > 0:
   f_post   = f_pre[i, j, q]                       # outgoing
   f_arrive = LI-BB reconstruction at q_w           # incoming after BB
   F_link   = (f_post + f_arrive) * c_q
   F_total += F_link
```

The LI-BB reconstruction at `q_w ≤ ½` uses
`f_arrive = 2 q_w f_q + (1 − 2 q_w) f_p_q_back + δ`, at `q_w > ½`
`f_arrive = (1/2q_w) f_q + (1 − 1/2q_w) f_qbar + (1/2q_w) δ`, with
`δ = −6 w_q (c_q · u_wall)` (= zero for stationary cylinder).

**Continuum limit** (Caiazzo & Junk 2007): the per-link sum
approximates `∮_S σ_total · n  dS` with `σ_total = −p I + 2 ν_LBM ρ S`
and `ν_LBM = ν_s + ζ·ν_p` (driver line 262). Geometric input is
`q_wall[i,j,q]`, the cut-link distance from fluid node to wall.

**The MEA loop reads `f` only**. It does not read τ_p,
`wall_fraction`, `wall_nx`, `cell_fraction`, or any FVFD-embedded data.
Any shift in Cd_s must therefore come from a shift in the steady-
state `f`, i.e. in the **body force** `fx_total, fy_total` fed to
`fused_trt_libb_v2_guo_field_step!` (driver line 462-465).

## 3. FVFD embedded traction formula

`fvfd_embedded_wall_traction_2d_kernel!`
(`src/fvfd/operators_2d.jl:825-846`):

```
tx[i,j] = wall_fraction[i,j] * (tauxx[i,j] * wall_nx[i,j] + tauxy[i,j] * wall_ny[i,j])
ty[i,j] = wall_fraction[i,j] * (tauxy[i,j] * wall_nx[i,j] + tauyy[i,j] * wall_ny[i,j])
```

This is the discrete traction `t = τ · n × ds` on the cut segment,
with `ds = wall_fraction[i,j]` in cell units. The integration is a
zeroth-order surface quadrature: every cut cell contributes `τ_cell · n
· ds`, then the driver sums (line 474 / 494) to obtain Fx, Fy. For
`:circle` geometry, `wall_fraction = hypot(west−east, south−north)`
and `(wall_nx, wall_ny) = (−(west−east), −(south−north))/wall_fraction`
(`src/fvfd/lowering_2d.jl:352-358`) — i.e. the **chord-length**
of the circle through the cell (NOT the true arc length).

**Continuum limit**: `∮_S τ · n dS` where τ is whatever stress tensor
was passed in. In the `embedded_drag=true` Cd_p path that tensor is
`τ_p` (polymer); in the Cd_bsd path it is `τ_BSD = 2·ζ·ν_p·D`. Total
viscoelastic drag is reconstructed as `Cd_s + Cd_p − Cd_bsd`
(driver line 594). Importantly, **only Cd_p and Cd_bsd flow through
this kernel**. Cd_s is always Mei MEA.

Chord vs. arc length is an O(dx²) bias per cell (Taylor expansion of
`R sin(Δθ)` vs `R Δθ`) — at R=30 with ~190 cut cells around the
circumference, total relative bias is ≤ 1e-4. Not the +6.7 %.

## 4. Continuum-vs-discrete equivalence audit

At continuum, MEA on `f` gives `Cd_s,LBM = ∮ (σ_solvent + σ_BSD) · n dS`
and the driver compensates via `Cd = Cd_s + Cd_p − Cd_bsd`. At β=1
ν_p=0, Cd_p=Cd_bsd=0 algebraically and the formula reduces to
`Cd = Cd_s` (analytic Stokes ± discretisation). **No continuum
inequality between modes.** The defect is purely discrete.

Discretely, Cd_s shifts +8.8 between `0000_qwall` and `1111_circle`
even though Cd_p, Cd_bsd are zero in the β=1 Newtonian baseline. The
only feedback path that can shift `f` is the Guo body force
`fx_total, fy_total`. The four `embedded_*` switches modify how
τ_p (built from the polymer pipeline) and the Fx_body flow into
`fx_total`. The defect lives in one of:

(a) **`embedded_force` divides by `cell_fraction`** in
`fvfd_tensor_divergence_embedded_2d_kernel!` (operators_2d.jl:762, 766).
For cut cells with `cell_fraction ≈ 0.1-0.3` (typical along a slanted
chord), `fx_poly` is amplified 3-10× before being added to the Guo
source.

(b) **`embedded_advection`** in `logfv_cell_velocity_to_faces_embedded_2d!`
(kernels:1119+ → operators_2d.jl:243) weights face velocities by face-
fractions, shifting steady-state τ_p in cut cells.

(c) **`embedded_gradient`** (`_fvfd_apply_embedded_wall_gradient_2d`,
operators_2d.jl:127-140) overwrites the normal component of ∇u with
`(u_cell · n) / wall_distance` (half-cell one-sided FD against no-slip);
tangential component stays centred-FD. **Same coupling pathology as
the cavity wall correction** (engineer.md 2026-05-17 "Cavity wall-
gradient correction writes half-cell ghosts") — these `dudx, dudy,
dvdx, dvdy` buffers feed (i) the source ODE (driver line 421-432,
producing wall-aware τ_p) AND (ii) `logfv_bsd_stress_from_gradient_2d!`
(line 486-489) — both then route through
`fvfd_tensor_divergence_embedded_2d_kernel!`, where `1/cell_fraction`
amplifies the half-cell ghost magnitude.

The **product** of (a) and (c) is the smoking-gun mechanism:
half-cell normal `∂u/∂n` from (c) → enhanced τ_p in cut cells →
divided by small `cell_fraction` in (a) → enhanced fx_poly →
unbalanced Guo body force → biased `f` post-collision → biased
Cd_s through MEA. The bias is positive because cut-cell-side `τ_p`
has the wrong sign of the diagonal (`dudx` half-cell ghost reaches
deep into the cylinder along `n`, so `dudx[i,j] · n_x` overstates the
wall stress); the body force then **pushes fluid past the cylinder**,
which the MEA reads as extra drag.

This mechanism is consistent with engineer.md 2026-05-17 note "wall
correction breaks BSD wide-stencil divergence" — the cavity version of
the same coupling, where the +12 % cavity gap shrank when BSD was
given an `_uncorrected` D buffer.

## 5. H1 / H2 / H3 ranking

**H1 (FVFD traction over-counts in `embedded_drag=true`): REFUTED.**
Mathematically impossible because Cd_s in the CSV comes from the LBM
MEA, which is independent of `embedded_drag`. Phase 0 case 3
(`0001_qwall`) when it arrives should give Cd_s ≈ Cd_s(`0000_qwall`)
to ≤ 0.1 % — the only thing `embedded_drag=true` changes is
`Cd_p, Cd_bsd`, and in the `Cd = Cd_s + Cd_p − Cd_bsd` decomposition
those two largely cancel.

**H2 (`embedded_force` mis-injects body force at low fluid fraction):
SUPPORTED, partial.** The `1/volume_fraction` divider in
`fvfd_tensor_divergence_embedded_2d_kernel!` (operators_2d.jl:762, 766)
is the explicit amplification mechanism. Whether the bias matches the
+6.7 % depends on what τ_p (and hence fx_poly) actually carries in the
cut cells — see H3.

**H3 (`embedded_circle_samples=32` insufficient): REFUTED for the
+8.8 magnitude.** The Monte-Carlo lowering of `cell_fraction` and
`{west,east,south,north}_fraction` (lowering_2d.jl:264-296) has
quadrature error `O(1/samples)` on cell_fraction. At samples=32, the
per-cell error is ≲ 1/32² ≈ 1e-3 for a smooth boundary. Summing over
~190 cut cells, total quadrature noise ≲ 1 % — not 6.7 %. The
wall_fraction chord-length is exact (analytic face-fraction
intersections, lines 236-254), so the boundary length is not affected
by samples. H3 deserves a control run at samples=128 but cannot
explain the +8.8 alone.

**Dominant cause (Department's diagnosis): H2 × `embedded_gradient`
coupling.** The `embedded_gradient` path writes half-cell normal
∂u/∂n into the same `dudx, dudy, dvdx, dvdy` buffers that feed BOTH
the source ODE (producing τ_p in cut cells with wall-distance-scaled
amplitude) AND the BSD tensor. `embedded_force` then divides the
resulting `fx_poly` by `cell_fraction ≪ 1` in cut cells, amplifying
the half-cell ghost into a singular body force. The Newtonian-baseline
Cd_s = 131.99 measured **without** the embedded force amplification
(audit 2026-05-09 used `0000_qwall`) is the reference. Switching to
`1111_circle` activates **both** the half-cell-ghost source (H3-adjacent)
AND the `1/cell_fraction` amplification (H2) in a single change,
which is why isolating one knob at a time has been hard.

Phase 0 (`21563085`) case 3 = `0001_qwall` (drag-only ON, force/gradient
/advection OFF) will give Cd_s ≈ 131.99 (consistent with this
analysis, since `embedded_drag` does not feed back into the LBM Guo
loop). The real disambiguator the Boss should run next is `0100_qwall`
(force ON only) and `1000_qwall` (gradient ON only) on β=1 Newtonian:
both will lift Cd_s individually; the **product** will be near-
additive only if the two paths are independent. Strong correlation
(non-additive, super-additive) confirms the H2×H3 coupling.

## 6. Proposed fix design

### Fix A (mechanical, low-risk): `embedded_force` reads
`D_uncorrected`, not the wall-corrected gradient

**Touch**: `src/drivers/viscoelastic_logfv_2d.jl` around lines 408-451.

**Design**: introduce a second pair of buffers
`dudx_unc, dudy_unc, dvdx_unc, dvdy_unc` populated by the **non-
embedded** `fvfd_velocity_gradient_2d!` (centred FD only, no half-
cell wall correction). Have the constitutive ODE keep reading the
corrected `dudx ...` so τ_p in cut cells is wall-aware. But have
`logfv_polymer_force_embedded_bc_aware_2d!` rebuild τ_p from
`dudx_unc ...` for the FORCE INJECTION ONLY. Practically: build a
second `tauxx_force, tauxy_force, tauyy_force` from `psixx, psixy,
psiyy` and `D_unc` (e.g. through the same `logfv_stress_from_log_2d!`
applied to a state advected from `D_unc`-driven ODE — or simpler,
pass the uncorrected D as an EXTRA additive correction to fx_poly).

**Expected magnitude**: closes 70-90 % of the +6.7 %. Direct parallel
to M17-canary-A (cavity, commit `f60f5174` per
engineer.md 2026-05-17) where the same "BSD gets its own
D_uncorrected" pattern dropped the wall-row residual 1860×.

**Risk**: the source ODE still receives a wall-aware gradient (good for
cut-cell physics) but the force field is centred. At the wall row, the
LBM bounce-back already handles the viscous flux, so dropping the
wall-aware Guo at cut cells does not lose physics — it just removes a
double-count.

### Fix B (architectural, medium-risk): replace `1/cell_fraction`
amplification with a sub-cell-aware Guo coupling

**Touch**: `fvfd_tensor_divergence_embedded_2d_kernel!`
(`src/fvfd/operators_2d.jl:665-770`, specifically the
`/ volume_fraction` divisions on lines 762, 766).

**Design**: the divergence kernel currently outputs `(F dS) /
(fluid volume)`, i.e. force-per-unit-fluid-volume. This is the
correct continuum quantity for an FVFD momentum equation. But the
LBM Guo source kernel expects force-per-unit-LATTICE-cell, not
force-per-unit-fluid-volume. In a cut cell with `cell_fraction =
0.3`, the LBM cell still occupies the whole lattice cell — the
collision operator works on the entire `f[i,j,1:9]` array. Multiplying
by `1/cell_fraction = 3.3` overdoses the body force on the lattice
cell by 3.3×. Correct coupling: omit the `/ volume_fraction` division
on the *output* path that feeds Guo, OR multiply by `cell_fraction`
explicitly when adding to `fx_total`.

**Expected magnitude**: closes 30-50 % of the +6.7 % even if Fix A
also applied; should bring `1111_circle` Cd_s within ±2 of
`0000_qwall` Cd_s.

**Risk**: this changes the kernel's output semantics. Any other
consumer of `fvfd_tensor_divergence_embedded_2d!` (none in viscoelastic
drivers per the grep) would need to re-multiply by `cell_fraction`.
The kernel docstring should clarify "outputs continuum force per fluid
volume; LBM Guo source must scale by `cell_fraction` before adding".

**Implementation sketch (≤20 lines)**:
```julia
# in src/drivers/viscoelastic_logfv_2d.jl after the embedded_force branch:
if embedded_force
    logfv_polymer_force_embedded_bc_aware_2d!(
        fx_poly, fy_poly, tauxx, tauxy, tauyy, fvfd_geometry; sync=false,
    )
    # Re-scale to per-lattice-cell so Guo doesn't over-dose cut cells.
    # `cell_fraction` ∈ (0, 1] in fluid; the kernel already divided by it.
    _logfv_scale_force_by_cell_fraction_2d!(
        fx_poly, fy_poly, embedded.cell_fraction; sync=false,
    )
end
```
NEW helper `_logfv_scale_force_by_cell_fraction_2d!`: one kernel,
`fx[i,j] *= cell_fraction[i,j]` skipping `is_solid`.

### Fix C (cheap diagnostic): bump `embedded_circle_samples`

Bench-side toggle only — change default 32 → 128 in the bench or add
`KRAKEN_EMBEDDED_CIRCLE_SAMPLES`. Pure H3 control, expected effect
< 1 %, zero risk.

### Ranked recommendation

**Apply Fix A first.** It is the direct analog of an already-validated
intervention (M17-canary-A on cavity), exact same coupling pathology
(half-cell ghost from wall-correction routed through a divergence
operator), exact same mitigation. Then run the bench at samples=128
(Fix C) to confirm H3-noise floor. If +6.7 % shrinks to < 2 % after
Fix A alone, ship `1111_circle` as production-ready and document the
residual as "discrete coupling residual" (matches Phase 0 Cd_s = 132.0
± 0.5 noise floor seen across BSD modes). If Fix A leaves > 3 %
residual, **add Fix B** (cell-fraction re-scaling); this should close
the rest. Fix B is a "load-bearing" change for any future curvilinear-
embedded driver (cavity-axisymmetric, BFS-embedded, contraction) — the
kernel semantics are currently ambiguous about who owns the
volume-fraction divisor.

### Acceptance verification (Department-side, no code)

After the fix lands, the canary is `0000_qwall` vs `1111_circle`
at β=1, Re=1, R=30, no polymer — both should give Cd_s ∈ [131.5, 132.5]
(audit 2026-05-09 baseline ±0.5 noise). At Wi=0.1 β=0.59 R=30, the
`1111_circle` Cd should match `0000_qwall` Cd to ±1 (post-Fix-A
target) or ±0.3 (post-Fix-A+B target).

## 7. Memory candidates (Boss filters)

- **engineer.md** : "FVFD-embedded tensor-divergence kernel divides
  by `cell_fraction`, which over-doses the LBM Guo source on cut
  cells. Any embedded driver feeding `fvfd_tensor_divergence_embedded_
  2d!` into a Guo step must re-scale by `cell_fraction` before
  `fx_total += fx_poly`. Same coupling family as engineer.md
  2026-05-17 wall-gradient half-cell ghost — both are routing fluid-
  volume-normalised data through a per-lattice-cell consumer."

- **engineer.md** : "`embedded_gradient` writes half-cell normal
  ∂u/∂n into `dudx, dudy, dvdx, dvdy` via
  `_fvfd_apply_embedded_wall_gradient_2d` (operators_2d.jl:127-140).
  These buffers feed the polymer source ODE (correctly) AND
  `logfv_polymer_force_embedded_bc_aware_2d!` (which then amplifies
  by `1/cell_fraction`) — same wall-correction-into-divergence pitfall
  as the cavity case. Give the FORCE path its own `D_uncorrected`
  buffer per Fix A; keep the SOURCE-ODE path on the corrected D."

- **boss.md** : "When a CSV column derived from `Cd_s = drag_s.Fx`
  shifts with `embedded_*` flags, the defect cannot live in the FVFD
  traction kernel — `drag_s` is the LBM MEA (`compute_drag_libb_mei_
  2d`), which reads only `f` and `q_wall`. The flags must therefore
  bias `f` itself via the Guo body force loop. Don't grep for a 'drag
  formula bug' — grep for force-injection paths."

- **department.md** : "For any future embedded-cylinder DoE, the
  binary-flag matrix must include the `0010_qwall` (force-only) and
  `1000_qwall` (gradient-only) corners as well as the
  geometry-corner `0010_circle` and `1000_circle`. The current Phase
  0 matrix (cases `0000, 1000, 0001, 1100`) cannot disambiguate H2
  (force amplification) from H3 (gradient half-cell ghost) when run
  in isolation."

---

End of verdict.
