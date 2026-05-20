# M31 frame audit - Codex independent read-only audit

Snapshot context: M30 R30/Wi1/Re1/beta0.59/qwall snapshot, stored
`Cd_kraken = 111.091`. I did not run the driver or simulations.

## Q1. Final computation sites

The M30/M31 log-FV cylinder path is
`run_viscoelastic_logfv_cylinder_coupled_2d`, which builds the cylinder
geometry at `cx = L_up * radius`, `cy = (H - 1) / 2` and forwards those as
`drag_cx`, `drag_cy`, `drag_radius`, and `drag_u_ref` to
`_run_viscoelastic_logfv_step_channel_coupled_2d`
(`src/drivers/viscoelastic_logfv_2d.jl:856-879`).

In `_run_viscoelastic_logfv_step_channel_coupled_2d`
(`src/drivers/viscoelastic_logfv_2d.jl:173-207`), the per-sample drag
accumulation is inside the time loop at
`src/drivers/viscoelastic_logfv_2d.jl:477-521`:

- `drag_s = compute_drag_libb_mei_2d(...)` at lines 477-479.
- `drag_p` is either `logfv_embedded_wall_traction_2d!` or
  `compute_polymeric_drag_2d(...)` at lines 480-493.
- `drag_bsd` is either embedded wall traction on a synthetic BSD stress, or
  `_logfv_compute_bsd_drag_2d(...)`, at lines 495-513.
- the running sums are updated at lines 515-521.

The final reduction that creates the values returned in `result` is at
`src/drivers/viscoelastic_logfv_2d.jl:591-604`:

```text
Fx_s   = Fx_s_sum   / n_drag
Fx_p   = Fx_p_sum   / n_drag
Fx_bsd = Fx_bsd_sum / n_drag
Fx_drag = Fx_s + Fx_p - Fx_bsd
Cd_s   = 2 * Fx_s   / (drag_speed^2 * drag_diameter)
Cd_p   = 2 * Fx_p   / (drag_speed^2 * drag_diameter)
Cd_bsd = 2 * Fx_bsd / (drag_speed^2 * drag_diameter)
Cd     = Cd_s + Cd_p - Cd_bsd
```

Those returned values are placed in the result NamedTuple at
`src/drivers/viscoelastic_logfv_2d.jl:682-694`.

For the big-sweep `SUMMARY.csv`, `Cd_kraken` is not a separately computed
quantity. It is `Float64(result.Cd)`, while `Cd_s`, `Cd_p`, and `Cd_bsd` are
copied from the same result object (`bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl:129-136`
and `:312-315`).

Legacy direct-conformation code in `src/drivers/viscoelastic.jl` has another
final reduction at `src/drivers/viscoelastic.jl:1139-1164`, but that path has
no `Cd_bsd` and is not the M30 log-FV `Cd_kraken` path.

## Q2. Formula and frame for each reported component

### `Cd_s`

Formula: cut-link LI-BB momentum exchange, not a wall ring integral.
`compute_drag_libb_mei_2d` is documented as Mei-Luo-Shyy momentum exchange
where each cut link contributes

```text
F_link = c_q * (f_q_pre + f_qbar_bouzidi)
```

(`src/drivers/cylinder_libb.jl:82-90`). The host implementation reconstructs
the arriving Bouzidi population from `q_wall`, `f`, and optional wall velocity,
then sums

```text
Fx += c_x(q) * (f_q_here + arriving)
Fy += c_y(q) * (f_q_here + arriving)
```

over `q_wall[i,j,q] > 0` (`src/drivers/cylinder_libb.jl:129-162`). The GPU
list kernel is the same formula (`src/kernels/drag_gpu.jl:151-179`).

There is no center coordinate in the final `Cd_s` summation. Its geometry is
only the `q_wall` cut-link set, which was generated from the analytic cylinder
center. It is therefore not a `:phys` or `:idx` ring integral.

### `Cd_p`

Formula: q-wall surface quadrature over cut links, not a fluid-cell ring
integral. In the non-embedded M30 snapshot path (`embedded_drag = false`),
the driver calls `compute_polymeric_drag_2d(tauxx,tauxy,tauyy,q_wall,...)`
with `cx=drag_cx`, `cy=drag_cy`, `radius=drag_radius`,
`extrapolate=true`, `reconstruction_order=2`
(`src/drivers/viscoelastic_logfv_2d.jl:486-493`).

Inside `compute_polymeric_drag_2d`, each cut-link wall point is

```text
xw = (i - 1) + q_w * c_x(q)
yw = (j - 1) + q_w * c_y(q)
```

and the normal is radial from the supplied center:

```text
rx = xw - cx
ry = yw - cy
n = (rx, ry) / hypot(rx, ry)
```

(`src/drivers/viscoelastic.jl:80-91`). The stress is optionally reconstructed
to the cut point (`src/drivers/viscoelastic.jl:93-118`), the points are sorted
by `theta`, and the force is

```text
Fx_p += (tau_xx * n_x + tau_xy * n_y) * ds
Fy_p += (tau_xy * n_x + tau_yy * n_y) * ds
ds = R * 0.5 * (theta_next - theta_prev)
```

(`src/drivers/viscoelastic.jl:120-146`).

The center passed by the log-FV cylinder wrapper is
`drag_cx = L_up * radius`, `drag_cy = (H - 1) / 2`
(`src/drivers/viscoelastic_logfv_2d.jl:871-877`). Because the wall point uses
`i - 1`, this is the physical coordinate frame. If the same operation is
re-expressed in raw Julia array indices `(i,j)`, the equivalent center is
`(cx + 1, cy + 1)`.

### `Cd_bsd`

Formula: synthetic BSD stress followed by the same wall traction machinery as
`Cd_p`. In the q-wall path, `_logfv_compute_bsd_drag_2d` forms

```text
tau_bsd_xx = 2 * zeta_nu_p * dudx
tau_bsd_xy = zeta_nu_p * (dudy + dvdx)
tau_bsd_yy = 2 * zeta_nu_p * dvdy
```

and delegates to `compute_polymeric_drag_2d` with the same `cx`, `cy`, and
`radius` (`src/drivers/viscoelastic_logfv_2d.jl:13-46`). The time-loop call is
at `src/drivers/viscoelastic_logfv_2d.jl:506-513`.

If `embedded_drag = true`, both polymer and BSD traction instead use
`logfv_embedded_wall_traction_2d!` (`src/drivers/viscoelastic_logfv_2d.jl:480-504`),
which delegates to `fvfd_embedded_wall_traction_2d!`
(`src/kernels/logconformation_fv_2d.jl:669-675`). That kernel computes
`length * (tau * n)` from `embedded.wall_nx`, `embedded.wall_ny`, and
`embedded.wall_fraction` (`src/fvfd/operators_2d.jl:928-945`).

For the M30 snapshot, `embedded_drag = false`, so the relevant `Cd_bsd` path is
the q-wall `compute_polymeric_drag_2d` path above.

### `Cd_kraken`

`Cd_kraken` in `SUMMARY.csv` is `result.Cd`, and `result.Cd` is

```text
Cd_s + Cd_p - Cd_bsd
```

(`src/drivers/viscoelastic_logfv_2d.jl:601-604`;
`bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl:312-315`).

This is a mixed force accounting: solvent/pressure/viscous contribution by
LI-BB cut-link MEA, plus explicit polymer q-wall traction, minus explicit BSD
q-wall traction. It is not the same object as the M30 cell-ring decomposition.

## Q3. `precompute_q_wall_cylinder` convention

Confirmed. The docstring says node `(i,j)` sits at physical coordinates
`(i-1,j-1)` (`src/kernels/li_bb_2d.jl:250-252`). The implementation does exactly
that:

```text
xf = FT(i - 1)
yf = FT(j - 1)
dx_f = xf - cx
dy_f = yf - cy
is_solid[i,j] = true if dx_f^2 + dy_f^2 <= R^2
```

(`src/kernels/li_bb_2d.jl:276-281`). Neighbor and wall-cut testing also uses
`xn = xf + cqx`, `yn = yf + cqy` (`src/kernels/li_bb_2d.jl:283-305`).

Therefore the physical disk center is `(cx_phys, cy_phys)`, but the raw
1-based array-index locus of that center is `(cx_phys + 1, cy_phys + 1)`.
For the M30 snapshot, the driver convention gives `cx_phys = L_up*R` and
`cy_phys = (Ny-1)/2` (`src/drivers/viscoelastic_logfv_2d.jl:816-820`), i.e.
`450.0, 59.5` for `R=30, L_up=15, Ny=120`; the index-frame center is
`451.0, 60.5`. M30 reports exactly that (`bench/viscoelastic_audit/M30_CENTERING_AUDIT_VERDICT.md:60-66`).

## Q4. Physically correct frame for rheoTool comparison

Answer: **Option B in the M30/raw-index nomenclature**, with one important
clarification.

The physically correct continuous coordinate system in the Kraken q-wall code
is not raw `(i,j)`. It is

```text
x = i - 1
y = j - 1
center = (cx_lbm, cy_lbm)
```

That is the coordinate system used by `precompute_q_wall_cylinder` and by
`compute_polymeric_drag_2d` wall points. If an audit instead uses raw array
indices as coordinates, then it must also shift the center to
`(cx_lbm + 1, cy_lbm + 1)`. That raw-index version is M30's `:idx` frame.

So the statement "use `cx_lbm,cy_lbm`" is only correct if every wall/ring point
has first been converted to physical coordinates `(i-1,j-1)`. The Phase 0c
style `dx = i - cx_lbm`, `dy = j - cy_lbm` mixes raw indices with a physical
center. That is a one-lattice-unit off-center postprocessing frame, not a
physical frame.

The moment-arm wording is a trap: drag itself has no moment arm. For a surface
traction integral, the normal/angle should be tied to the wall geometry that
exchanged the traction. Here that wall geometry is the analytic circle encoded
in `q_wall`, not the center of mass of the rasterized `is_solid` pixels. The
`is_solid` COM is a diagnostic of raster symmetry; it is not the definition of
the cylinder that rheoTool is solving. If one deliberately switches to a pure
staircase-face integral, normals should be face/link normals at the exchanged
faces, not radial normals from a pixel COM.

Therefore, for the M30 ring files as written, the physically defensible frame
is `:idx`, not `:phys`.

## Q5. Correct-frame Kraken `Cd_polymer` vs rheoTool

Using the M30 cell-ring decomposition in the physically correct raw-index
frame, Kraken has

```text
Cd_polymer_idx = 10.8226
```

while rheoTool has `Cd_polymer = 13.45`
(`bench/viscoelastic_audit/M30_CENTERING_AUDIT_VERDICT.md:30-37`;
`bench/viscoelastic_audit/M29C_WALLSTRESS_VERDICT.md:51-58`).

The deficit is

```text
13.45 - 10.8226 = 2.6274
2.6274 / 13.45 = 19.5 %
```

The Phase 0c/M29c-wallstress `:phys` number, `13.4611`, matches rheoTool only
because the ring was evaluated in the wrong raw-index frame
(`bench/viscoelastic_audit/M30_CENTERING_AUDIT_VERDICT.md:19-26`).

For completeness, the actual driver-reported q-wall polymer component in the
M30 snapshot is `Cd_p = 11.4895`, not `13.4611`
(`bench/viscoelastic_audit/M30_CENTERING_AUDIT_VERDICT.md:45-49`). That is
still low against rheoTool by

```text
13.45 - 11.4895 = 1.9605 = 14.6 % of rheoTool
```

Verdict: the M29c-wallstress claim "M29b polymer matches rheoTool to 0.05"
does **not** survive. It was a coordinate-frame artefact in the wall-ring
postprocessor, amplified by staircase sampling. The real statement is:
Kraken M29b/log-FV polymer wall drag is under-predicted by roughly 15-20%,
depending on whether one compares the corrected cell-ring decomposition or the
driver's q-wall `Cd_p`.

## Q6. Reconciling stored `Cd_kraken = 111.09`

The tempting inference is:

```text
stored Cd_kraken ~= wrong-frame ring total 111.20
stored Cd_kraken != correct-frame ring total 108.63
therefore the driver uses the wrong frame
```

I do **not** buy that inference. The premise compares different numerical
objects.

Source says the driver computes:

```text
Cd_kraken = Cd_s(MEI cut-link) + Cd_p(q-wall surface) - Cd_bsd(q-wall surface)
```

The M30 ring audit computes:

```text
Cd_ring = Cd_pressure(cell-ring rho) + Cd_solvent(cell-ring gradients)
          + Cd_polymer(cell-ring tau)
```

Those are not the same quadrature, not the same stencil, and not even the same
component split. M30 itself says the stored `Cd_s` bundles pressure and viscous
solvent at the cut-link level and that the ring split is a separate
cell-centered decomposition (`bench/viscoelastic_audit/M30_CENTERING_AUDIT_VERDICT.md:51-55`).

The match between stored `111.091` and the wrong-frame ring total `111.2005`
is therefore best treated as an accident, not as evidence that the production
driver is literally using `dx = i - cx_lbm`. In fact, the source for
`compute_polymeric_drag_2d` uses `xw = i - 1 + q_w c_q` and center
`cx = drag_cx`, which is the correct physical/q-wall frame
(`src/drivers/viscoelastic.jl:80-91`;
`src/drivers/viscoelastic_logfv_2d.jl:486-493`).

There are still two load-bearing problems:

1. The component audits that used `dx = i - cx_lbm` are wrong for polymer wall
   drag. This invalidates the M29c-wallstress "polymer matches" conclusion.
2. The stored field snapshot metadata is easy to misread: the big-sweep file
   save says "LBM node centers at (i,j)" while writing
   `cylinder_x_lbm = L_up*R`, `cylinder_y_lbm = (Ny-1)/2`
   (`bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl:350-358`). The source
   q-wall convention is actually `(i-1,j-1)`. That metadata/comment mismatch
   is a reproducible way to regenerate the bad `:phys` postprocessing frame.

What I do **not** see is proof of a universal production `Cd_kraken` frame bug
affecting every benchmark since M28. The total `Cd_kraken` values are generated
by the q-wall/MEA driver path, not by the M30 cell-ring frame. They may still
hide serious component cancellation, and they absolutely should not be used as
proof of polymer correctness, but the 2.5% ring discrepancy is not by itself a
driver-center smoking gun.

One more blunt point: `bench/viscoelastic_audit/M30_CENTERING_AUDIT_VERDICT.md:55-56`
says both ring totals reconcile with `Cd_kraken` within about 0.1%. That is
numerically false for the `:idx` total printed four lines earlier:
`108.6308 - 111.0910 = -2.4602`, i.e. `-2.21%`
(`bench/viscoelastic_audit/M30_CENTERING_AUDIT_VERDICT.md:41-49`). The report
text should be corrected before downstream decisions quote it.

## Confidence

High on the source-level geometry convention and on Q1/Q2 formulas. High that
the M30 raw-index `:idx` frame is the physically consistent cell-ring frame.
High that the M29c-wallstress polymer-match claim is falsified in that frame.

Medium on interpreting the exact 111.09 vs 108.63 total discrepancy, because
that comparison mixes MEI cut-link force accounting with a separate cell-ring
decomposition. It is enough to reject the old component claim; it is not enough
to convict the production driver of a single center-offset bug.

Process caveat: during source search, a broad `rg` command accidentally matched
the forbidden `M31_FRAME_AUDIT_CLAUDE.md` file and printed snippets. I did not
open it or intentionally use it further. Treat this as a process contamination
note; the conclusions above are sourced from the code and M30/M29 inputs cited
inline.
