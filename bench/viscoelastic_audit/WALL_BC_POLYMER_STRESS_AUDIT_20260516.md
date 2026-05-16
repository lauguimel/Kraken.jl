# Wall BC audit — polymer stress and log-conformation
#### Cavity Oldroyd-B, Kraken (log-FV) vs rheoTool — 2026-05-16

---

## Problem statement

Mission M4 (audit done 2026-05-16, mandate §5/M4) measured a
relative L2 discrepancy of **53.5 % – 53.8 %** between Kraken's Guo
body force (`f^Guo = ∇·τ − ζ·ν_p·∇²u`, formed in
`logfv_polymer_force_bc_aware_2d!` then BSD-corrected in
`logfv_bsd_correct_force_bc_aware_2d!`) and an independent FD
estimate of `f^FD = ∇·τ` on saved N=64 cavity snapshots. The maximum
absolute difference sits at cell **(16, 63)** — the second row below
the moving lid in the right-wall recirculation corner — and the
discrepancy is structurally invariant across `u_max` (M1 verdict
file: `bench/viscoelastic_logfv/CAVITY_REMISMATCH_M1_VERDICT_20260515.md`).

The mission's working hypothesis (M5-A design,
`BSD_KINETIC_MOMENT_DESIGN_20260516.md`) is that the discrepancy
stems from a discrete-operator mismatch in the BSD `−ζ·ν_p·∇²u`
correction (LBM-consistent vs FD laplacian).

**Alternative hypothesis investigated here (M6-A)**: the 54 % gap is
dominated NOT by the BSD operator mismatch but by an **inconsistency
between Kraken's wall BC on `τ` (via `Ψ`) and rheoTool's
`linearExtrapolation` convention**. Because the max-diff cell lies
one row below the wall, the FD divergence of `τ` at that cell reads
into the wall-adjacent row. Any difference in how that row's `τ` is
populated propagates into ‖F_FD − F_Guo‖₂ even if the bulk operator
is perfectly consistent.

The verdict below is built from code reading only; the predicted
quantitative impact is qualitative.

---

## rheoTool wall BC convention

Source files in
`bench/rheotool/cavity_oldroydb_log_re001_de1_b05/0/`:

**`theta`** (log-conformation `Ψ = log(C)`, symmTensor):

```
dimensions      [0 0 0 0 0 0 0];

internalField   uniform (0 0 0 0 0 0);

boundaryField
{
    movingLid
    {
        type            zeroGradient;
    }
    fixedWalls
    {
        type            zeroGradient;
    }
    frontAndBack
    {
        type            empty;
    }
}
```

**`tau`** (polymer extra stress `τ`, symmTensor):

```
dimensions      [1 -1 -2 0 0 0 0];

internalField   uniform (0 0 0 0 0 0);

boundaryField
{
    movingLid
    {
        type            linearExtrapolation;
        value           uniform (0 0 0 0 0 0);
    }
    fixedWalls
    {
        type            linearExtrapolation;
        value           uniform (0 0 0 0 0 0);
    }
    frontAndBack
    {
        type            empty;
    }
}
```

`constant/constitutiveProperties`: `type Oldroyd-BLog`,
`stabilization coupling` (no log-conformation correction beyond the
default), `etaS = etaP = 0.5`, `lambda = 1`, `rho = 0.01`.

`system/fvSchemes`: `div(tau) Gauss linear`. The
divergence of `τ` on internal faces is therefore the standard
linearly-interpolated face value times the face area, summed; on
boundary faces the value used is exactly what
`linearExtrapolation` writes.

**rheoTool BC interpretation**:

- `theta` carries **Neumann (zeroGradient)** at all 4 walls — the
  log-conformation flux through the wall is zero; the wall ghost is
  identical to the wall-adjacent cell value.
- `tau` carries **linearExtrapolation** at all 4 walls — this is the
  rheoTool-specific BC `linearExtrapolationFvPatchField` (see
  rheoTool source `src/libs/openFoamExtend/finiteVolume/fields/`)
  which evaluates the wall-face value of `τ` as a **linear extrapolation
  from the two nearest internal cells along the face-normal**:
  `τ_face = τ_C + (τ_C − τ_CC) · (d_face / d_cellcell)` where `C` is
  the first interior cell and `CC` the next one. The `value uniform
  (0 0 0 0 0 0)` is only the initial guess; rheoTool overwrites it
  every time-step from the linear extrapolation.

So rheoTool uses TWO different conventions for the same physical
wall: a Neumann on `Ψ`, and a linear extrapolation on `τ` directly.
The polymer body force `∇·τ` in the momentum equation reads the
extrapolated `τ`-face value, never recomputes `τ` from `Ψ` at the
boundary.

---

## Kraken current wall BC

Kraken stores only `Ψ` (`psixx`, `psixy`, `psiyy` arrays); `τ` is a
**derived field** reconstructed cell-wise at every step from `Ψ`,
then differentiated by an FD divergence operator.

**Driver wiring** (cavity coupled core,
`src/drivers/viscoelastic_logfv_2d.jl`):

- L972 — `logfv_bc = logfv_wallxwally_bcspec_2d()` → equivalent to
  `FVFDDomainBC2D(:wall, :wall, :wall, :wall)` from
  `src/fvfd/specs.jl:94`. The trinary enum
  (`FVFD_BC_PERIODIC=1`, `FVFD_BC_OPEN=2`, `FVFD_BC_WALL=3`,
  `specs.jl:1-3`) has **no `linearExtrapolation` variant**.

- L943-945 — `psixx, psixy, psiyy` allocated as `zeros(T, Nx, Ny)`
  → initial value matches rheoTool's `internalField uniform
  (0 0 0 0 0 0)` on `theta`.

- L1030-1036 — log-conformation advection. The wall BC is consumed
  inside `fvfd_sym2_advect_upwind_2d!`
  (`src/fvfd/operators_2d.jl:557` → upwind helper at
  `operators_2d.jl:408-454`). At a wall face the helper
  `_fvfd_bc_east_scalar_2d` etc. returns `phi[i, j]` (the current
  cell value) — this is exactly **Neumann / zeroGradient on `Ψ`**.
  **MATCHES rheoTool's `theta` BC.**

- L1063-1068 — `logfv_stress_from_log_2d!`
  (`src/kernels/logconformation_fv_2d.jl:477-494`). Pure cell-wise
  transform: `τ_{i,j} = (η_p/λ)·(f·C(Ψ_{i,j}) − I)` with NO
  ghost / no BC. The wall-cell `τ` is therefore whatever
  `exp(Ψ_wall_cell) − I` produces, where `Ψ_wall_cell` is the
  result of the upwind advection + constitutive source.

- L1069-1073 — `logfv_polymer_force_bc_aware_2d!` →
  `fvfd_tensor_divergence_2d!` (`src/fvfd/operators_2d.jl:619-645`).
  At a wall-adjacent cell the helper `_fvfd_solid_bc_derivative_x_2d`
  (`operators_2d.jl:13-36`, used identically along `y` at L38-61)
  detects that one side (the wall) returns `_fvfd_bc_index_1d = 0`
  for `WALL` (`operators_2d.jl:1-11`), then falls back to the
  one-sided **second-order extrapolation**
  `(−3·τ_i + 4·τ_{i+1} − τ_{i+2}) / (2·dx)` (operators_2d.jl:24-26).
  This is **algebraically equivalent to a quadratic extrapolation
  of τ to the wall face followed by a centered difference** — but
  the polynomial is fitted to (i, i+1, i+2), not (i, i+1) — i.e. a
  **quadratic, not linear, extrapolation**.

- L1074-1087 — `logfv_bsd_correct_force_bc_aware_2d!` applies the
  same one-sided second-derivative stencil to `u` via
  `_fvfd_solid_bc_second_derivative_x_2d` (operators_2d.jl:63-86).
  At wall cells the laplacian collapses to a one-sided 3-point
  stencil. This is independent of the τ-BC question but is the
  ‘BSD operator mismatch’ flagged by M4 / M5-A.

**Synthesis**: Kraken has NO explicit wall BC on `τ`. The `τ` array
is computed cell-wise from `Ψ` everywhere (including the wall row),
and the FD divergence then does its own one-sided **quadratic**
extrapolation at the boundary. rheoTool, by contrast, computes a
**linear** face value of `τ` from the two nearest cells (skipping
the third) and uses that directly in the Gauss divergence.

---

## Identified differences

1. **Linear vs quadratic extrapolation of τ at the wall face**
   - rheoTool: `τ_face = τ_C + (τ_C − τ_CC)·(d_face/d_CC)` — 2-point
     linear extrapolation onto the wall face.
   - Kraken: implicit 3-point quadratic extrapolation built into the
     one-sided stencil `(−3·τ_i + 4·τ_{i+1} − τ_{i+2})/(2·dx)`.
   - Downstream operator affected: **`logfv_polymer_force_bc_aware_2d!`**
     (= ∇·τ in the Guo body force). Cells reading this row enter the
     M4 audit denominator at the wall and one cell inside.

2. **`τ` recomputed at wall vs extrapolated at wall**
   - rheoTool: `τ_face` is purely an extrapolation of interior `τ`
     values — it carries no information from `Ψ_wall_cell` directly.
   - Kraken: `τ_wall_cell` is `exp(Ψ_wall_cell) − I`. Because `Ψ`
     itself has a Neumann wall BC (matches rheoTool), the
     wall-cell `Ψ` reproduces the first interior `Ψ`. The wall-cell
     `τ` is therefore `exp(Ψ_{i,2}) − I` (using j=1 ↔ j=2 ghost)
     and NOT a linear extrapolation from `τ_{i,2}` and `τ_{i,3}`.
   - For Oldroyd-B with `exp(Ψ) = C` SPD, this difference is small
     for small `Ψ` but grows as `|Ψ|` grows in the recirculation
     corners.
   - Downstream operator affected: **`logfv_stress_from_log_2d!`**
     (the cell-wise τ-from-Ψ map) and **the divergence kernel**
     which reads this wall-row `τ`.

3. **Velocity-gradient BC at the moving lid is explicit in Kraken**
   - rheoTool's `Gauss linear` divergence implicitly handles the
     moving-lid `U`-BC.
   - Kraken applies `_logfv_cavity_apply_wall_gradient_correction!`
     (`src/drivers/viscoelastic_logfv_2d.jl:816-832`, L1042-1045 in
     the main loop) to overwrite `du/dy` at j=Ny using a half-cell
     FD against the imposed lid profile. This affects the polymer
     **source** `L·Ψ + Ψ·Lᵀ` at the top row, not the τ-divergence,
     but it changes `Ψ_wall_row` and hence the wall-row `τ` that
     the divergence reads.
   - Downstream operator affected: **`logfv_step_constitutive_log_2d!`**
     at j=Ny, then `logfv_stress_from_log_2d!` at j=Ny.

---

## Proposed Kraken modification

The minimal change is **not** to add a new enum to
`fvfd_wallxwally_bcspec_2d` (the BC is consumed inside the FD
divergence kernel, not by an explicit ghost fill on `τ`). The
modification should change the **stencil** used at wall-adjacent
cells in `_fvfd_solid_bc_derivative_x_2d` and `_y` to match rheoTool's
linear extrapolation.

**Concrete proposal** — replace the 3-point one-sided derivative at
wall-adjacent cells with a **linear-extrapolation-equivalent**
stencil. For the east-wall case (i = Nx):

- Current (`src/fvfd/operators_2d.jl:24-26`):
  ```julia
  (-T(3) * field[i, j] + T(4) * field[ri, j] - field[r2i, j]) * inv_2dx
  ```
  ⇔ quadratic extrapolation `τ_face = (3·τ_C − 3·τ_CC + τ_CCC)/?` and
  centered diff.

- Proposed:
  ```julia
  # Linear extrapolation to wall face: τ_face = 1.5·τ_C - 0.5·τ_CC
  # then derivative at cell C = (τ_face - τ_C)/(0.5·dx) = (τ_C - τ_CC)/dx
  (field[i, j] - field[ri, j]) * inv_dx  # NB sign: east wall, ri = i-1
  ```
  (and the analogous form for west/north/south).

This is a **2-point one-sided FD**, equivalent to rheoTool's linear
extrapolation followed by Gauss divergence.

**File:line of the proposed edits**:

- `src/fvfd/operators_2d.jl:24-27` (east-wall branch of
  `_fvfd_solid_bc_derivative_x_2d`)
- `src/fvfd/operators_2d.jl:30-32` (west-wall branch, same helper)
- `src/fvfd/operators_2d.jl:50-52` (`_y` analogue, north)
- `src/fvfd/operators_2d.jl:55-57` (`_y` analogue, south)

A safe roll-out is to add a kwarg `polymer_wall_extrap::Symbol =
:quadratic` to `fvfd_tensor_divergence_2d!`
(`src/fvfd/operators_2d.jl:754`) plumbed through
`logfv_polymer_force_bc_aware_2d!`
(`src/kernels/logconformation_fv_2d.jl:649-656`), accepting
`:quadratic` (current) and `:linear` (rheoTool-aligned), and have
the cavity driver opt into `:linear` at the call site
(`viscoelastic_logfv_2d.jl:1071`).

**Budget**: 4 stencil edits (≤ 4 LOC each) + 1 new kwarg through 3
function signatures + 1 driver opt-in = **≤ 25 LOC across 3 files**.

Note: the second derivative used in the BSD correction
(`_fvfd_solid_bc_second_derivative_*_2d`,
`operators_2d.jl:63-86`) is a SEPARATE question (this is the M5-A
target). The current audit only touches the first-derivative stencil
that consumes `τ`. The BSD second-derivative stencil should stay
under M5-A scope.

---

## Predicted impact on M4 audit

The M4 audit script `bench/viscoelastic_logfv/analyse_cavity_guo_vs_fd_2d.jl`
compares Kraken's `f^Guo` (Kraken's own `logfv_polymer_force_bc_aware_2d!`
output) against an **external** FD reconstruction `f^FD = ∇·τ`
implemented in the analysis script. If the analysis script uses a
**centered 2nd-order divergence** with periodic / open boundary
treatment, then at cell (16, 63):

- Current Kraken (quadratic) and the analysis-script (centered) will
  disagree by an `O(τ_wall − τ_interior)` term that does NOT vanish
  in the corner where the polymer is most stretched.
- After the proposed change (linear extrapolation, equivalent to a
  one-sided 2-point FD), Kraken's stencil at (16, 63) becomes
  `(τ_{16,63} − τ_{16,62})/dy` along `y`. If the analysis script
  uses a centered 2-point at the same cell — i.e.
  `(τ_{16,64} − τ_{16,62})/(2·dy)` — they still differ at the wall
  but by an `O(dy)` rather than `O(1)` term.

**Plausible drop**: from 54 % to roughly **15-30 %** in interior L2.
The residual would then reflect:
- the BSD second-derivative mismatch (M5-A target);
- the corner-row `Ψ` discrepancy from
  `_logfv_cavity_apply_wall_gradient_correction!` (M2 territory).

**Interior cells far from walls (e.g. (32, 32))**: the
proposed change is **a no-op** there — the centered stencil
`(τ_{i+1} − τ_{i-1})/(2·dx)` is identical with or without wall
extrapolation since neither side hits the wall. M4's audit at
interior cells should therefore be unchanged. If the M4 paper reports
interior cells as <1 % already, the wall-BC hypothesis is **likely
NOT the dominant story** and M5-B should proceed. If the M4 paper
reports interior cells as 5-10 %+, the wall hypothesis is still
plausible and M6-B (implement this change) is worth trying first.

---

## Validation strategy

Single-run, low-risk validation:

1. Implement the kwarg-gated linear-extrapolation stencil per
   §Proposed Kraken modification, defaulted to `:quadratic` (current
   behaviour preserved).
2. Re-run the cavity at `N=64, t=8, De=1, β=0.5, u_max=0.005,
   bsd_fraction=0.75` with `polymer_wall_extrap=:linear`. Use the
   existing PBS wrapper
   `bench/viscoelastic_logfv/run_cavity_oldroydb_vs_rheotool_anygpu.pbs`
   with a single env-var override.
3. Re-run `analyse_cavity_guo_vs_fd_2d.jl` on the new snapshot. Bar:
   `‖F_FD − F_Guo‖₂ / ‖F_FD‖₂ < 5 %` on the interior set excluding
   the 2-row halo against each wall.
4. Re-run the L2 profile comparison
   (`run_cavity_oldroydb_vs_rheotool.jl` post-processing block):
   centerline `u` and horizontal `psi_xy` against rheoTool. Bar:
   ≤ 10 % L2 on either (vs current 18-24 %).

If bar 3 passes but bar 4 does not, the wall-BC change closed only
the audit metric, not the physical mismatch — proceed with M5-B
(BSD kinetic-moment). If both bars pass, M5 can be deferred.

---

## Interaction with M4b sweep and M5-B prototype

**M4b** (BSD fraction sweep, mandate §5/M4b) probes the SAME
hypothesis as M5-A from the opposite direction: if `bsd_fraction → 0`
removes the BSD correction and the cavity L2 collapses, the BSD
laplacian (M5-A territory) is the dominant source. If it does NOT
collapse, the residual gap must lie elsewhere — and the wall-BC
hypothesis (this audit) becomes the leading candidate. Decision flow:

- **M4b shows L2 → 0 monotonically as bsd_fraction → 0**: commit to
  **M5-B (kinetic-moment BSD)**. The wall-BC change here is a 2nd-order
  cleanup, worth doing afterwards but not on the critical path.
- **M4b shows L2 flat or non-monotonic**: prioritise **M6-B
  (this proposal, linear wall extrapolation on τ)** over M5-B.
  The 25-LOC change is much cheaper than the M5-A scope estimate
  (5-8 h Codex + validation) and tests a complementary hypothesis.
- **M4b shows partial drop (~30-50 %)**: BOTH effects are real.
  Implement M6-B first (cheapest), re-measure M4 audit, then commit
  to M5-B if interior L2 still >5 %.

Recommendation: run M4b first (per mandate §6), then route to either
M5-B or M6-B based on its slope. Do NOT implement M5-B and M6-B in
parallel — they touch overlapping operators in
`src/fvfd/operators_2d.jl` and will conflict.
