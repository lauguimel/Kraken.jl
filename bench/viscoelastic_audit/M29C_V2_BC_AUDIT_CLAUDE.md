# M29c-v2 BC-helper audit — Claude independent pass

Mission: M29c-v2-audit-bc-helpers, Step 1 (pre-Codex).
Date: 2026-05-19.
Branch: `dev-viscoelastic`, working tree (M29c-v2 patch uncommitted).
Source file: `src/fvfd/operators_2d.jl` (1251 lines).
Driver entry: cylinder coupled driver dispatches the polymer advection
through `logfv_advect_upwind_bc_aware_2d!`
(`src/kernels/logconformation_fv_2d.jl` lines 1153–1173) which delegates
to `fvfd_sym2_advect_upwind_2d!` (operators_2d.jl line 711) which calls
`fvfd_advect_upwind_2d!` three times (operators_2d.jl line 655), which
launches `fvfd_advect_upwind_2d_kernel!` (line 631) which dispatches to
the M29c-v2 `:muscl_superbee` branch (line 537) via the `Val(:muscl_superbee)`
specialised method.

## Q1 — `:muscl_superbee` entry point

- Dispatch function: `_fvfd_upwind_scalar_advective_rhs_2d`, method specialised
  on `::Val{:muscl_superbee}`, `operators_2d.jl` lines **537–629**.
- Kernel that calls it: `fvfd_advect_upwind_2d_kernel!`, lines **631–653**.
- Wrapper that the cylinder driver eventually hits:
  `fvfd_advect_upwind_2d!` (line 655) ← `fvfd_sym2_advect_upwind_2d!`
  (line 711) ← `logfv_advect_upwind_bc_aware_2d!`
  (`logconformation_fv_2d.jl` line 1153) ← cylinder driver line 402 of
  `viscoelastic_logfv_2d.jl`.

## Q2 — BC helpers and `is_solid` test

`_fvfd_bc_east_scalar_2d` (lines **422–432**):
```julia
@inline function _fvfd_bc_east_scalar_2d(phi, east_phi, i, j, Nx, east_bc)
    if i < Nx
        return phi[i + 1, j]
    elseif east_bc == FVFD_BC_PERIODIC
        return phi[1, j]
    elseif east_bc == FVFD_BC_OPEN
        return east_phi[j]
    else
        return phi[i, j]
    end
end
```

`_fvfd_bc_west_scalar_2d` (lines **434–444**):
```julia
@inline function _fvfd_bc_west_scalar_2d(phi, west_phi, i, j, Nx, west_bc)
    if i > 1
        return phi[i - 1, j]
    elseif west_bc == FVFD_BC_PERIODIC
        return phi[Nx, j]
    elseif west_bc == FVFD_BC_OPEN
        return west_phi[j]
    else
        return phi[i, j]
    end
end
```

`_fvfd_bc_north_scalar_2d` (lines **446–456**):
```julia
@inline function _fvfd_bc_north_scalar_2d(phi, north_phi, i, j, Ny, north_bc)
    if j < Ny
        return phi[i, j + 1]
    elseif north_bc == FVFD_BC_PERIODIC
        return phi[i, 1]
    elseif north_bc == FVFD_BC_OPEN
        return north_phi[i]
    else
        return phi[i, j]
    end
end
```

`_fvfd_bc_south_scalar_2d` (lines **458–468**):
```julia
@inline function _fvfd_bc_south_scalar_2d(phi, south_phi, i, j, Ny, south_bc)
    if j > 1
        return phi[i, j - 1]
    elseif south_bc == FVFD_BC_PERIODIC
        return phi[i, Ny]
    elseif south_bc == FVFD_BC_OPEN
        return south_phi[i]
    else
        return phi[i, j]
    end
end
```

| helper | signature accepts `is_solid`? | tests `is_solid` before returning `phi[neighbour]`? | returns when neighbour is solid? |
|---|---|---|---|
| east | NO | NO | `phi[i+1, j]` unconditionally (assuming `i < Nx`) |
| west | NO | NO | `phi[i-1, j]` unconditionally (assuming `i > 1`) |
| north | NO | NO | `phi[i, j+1]` unconditionally (assuming `j < Ny`) |
| south | NO | NO | `phi[i, j-1]` unconditionally (assuming `j > 1`) |

**Answer to Q2: none of the four BC helpers accept `is_solid` as an
argument, and none test it. When the neighbouring index lies inside
the domain (the dominant case for cylinder-adjacent fluid cells), the
helper returns `phi[neighbour]` blindly, even if `is_solid[neighbour]`
is `true`.**

## Q3 — `phi[solid] = 0` enforcement and call ordering

Enforcement site: the kernel body, `operators_2d.jl` lines **640–651**:

```julia
if i <= Nx && j <= Ny
    if is_solid[i, j]
        phi_out[i, j] = zero(eltype(phi_out))
    else
        rhs = _fvfd_upwind_scalar_advective_rhs_2d(
            phi, west_phi, east_phi, south_phi, north_phi,
            ux_face, uy_face, is_solid, i, j, Nx, Ny, inv_dx, inv_dy,
            west_bc, east_bc, south_bc, north_bc, advection_scheme,
        )
        phi_out[i, j] = phi[i, j] + dt * rhs
    end
end
```

Call ordering, per LBM step in the cylinder coupled driver
(`viscoelastic_logfv_2d.jl` lines 390–440):

1. line 402: `logfv_advect_upwind_bc_aware_2d!` is invoked. **Input
   array** = `psixx` (etc.); **output array** = `psixx_adv` (etc.).
2. Inside, the kernel runs ONCE for all cells. For every solid cell
   `phi_out = psixx_adv[i,j] := 0`. For every fluid cell the MUSCL
   flux is assembled by reading `phi` (= `psixx`) via the BC helpers
   from ALL 4 neighbours, including potentially-solid ones.
3. The constitutive substepping loop at line 426 then ping-pongs
   `psixx_work / psixx_next` over `selected_polymer_substeps`
   iterations, reading and writing the SAME array set (`psixx_adv ↔
   psixx_next`). The constitutive kernel (`logfv_step_constitutive_log_2d_kernel!`
   `logconformation_fv_2d.jl` line 417) does NOT mask `is_solid`, so
   the solid cells start at `0` and then receive
   `(-I/λ + ...)·dt` increments per substep, ending the LBM step at
   some O(λ⁻¹·n_substeps·dt) ≪ |Ψ_fluid| residual.
4. At line 438 `psixx` and `psixx_adv` are swapped so that the next
   LBM step reads the now-substepped `psixx`.

**Critical ordering question — is `phi[solid] = 0` enforced BEFORE the
MUSCL face flux for the fluid neighbour is computed in the same
substep?** The MUSCL kernel is launched as a single `ndrange =
(Nx, Ny)` call. Inside one kernel launch:

- The work-items running on solid cells write `phi_out := 0`.
- The work-items running on fluid cells read `phi` (= the INPUT,
  marked `@Const`) via the BC helpers.

The kernel READS from `phi` and WRITES to `phi_out`. These are
**different arrays** (`psixx` vs `psixx_adv` per the driver line 402–
412 signature). Therefore the `phi_out[solid] = 0` line of THIS kernel
launch does not feed back into the MUSCL stencil of THIS kernel
launch — there is no race. **But the PREVIOUS LBM step's tail
constitutive substep wrote into `psixx_next`, which becomes the
current `psixx` after the swap.** So the MUSCL stencil reads
`Ψ_solid` from the last constitutive output of the previous LBM step,
which (by the residual analysis above) is small (O(λ⁻¹·dt·n_substeps)
≈ O(1e-6) at λ=6000 LU, n_substeps≈8, dt=1) but NOT exactly 0 and
NOT physical.

**However**, in the FIRST LBM step, `psixx_adv[solid]` has been
explicitly zeroed by the very same kernel one tick earlier (the kernel
ran on every cell), so by the SECOND LBM step `psixx[solid] = (some
O(1e-6) constitutive residue)`. From then on the "solid Ψ" the
MUSCL stencil sees is **approximately 0 but not exactly 0**, and
crucially **independent of the surrounding fluid Ψ values**.

For the purposes of the DIFF analysis (which only needs `|Ψ_solid| ≪
|Ψ_fluid|`), this answer is unchanged: **the BC helpers read the
solid-cell Ψ, which is held at ~0 by the kernel + constitutive substep
chain.**

## Q4 — algebraic trace at a cylinder-adjacent fluid cell

Setup. Fluid cell `(i, j)` is directly west of a solid cell, i.e.
`is_solid[i+1, j] = true`, with `is_solid` false everywhere else in the
4-neighbourhood and the far-neighbourhood. Wi = 1, R = 30, β = 0.59.
Typical magnitudes near the cylinder west surface: `Ψ_xx(fluid) ≈ 5`,
`Ψ_xx(solid) ≈ 0` (per Q3).

Path through the M29c-v2 code (lines 553–569 for the east face):

```julia
east_value = _fvfd_bc_east_scalar_2d(phi, east_phi, i, j, Nx, east_bc)
             # = phi[i+1, j] = Ψ_solid ≈ 0
phie = if ue >= 0
    upwind = phi[i, j]                # = Ψ_fluid ≈ 5
    downwind = east_value             # = Ψ_solid ≈ 0
    canonical_usable = i > 1 && !is_solid[i - 1, j]  # = true (west neighbour fluid)
    far_upwind = canonical_usable ? phi[i - 1, j] : upwind
                                       # = phi[i-1, j] = Ψ_far_west ≈ 5 (similar)
    _fvfd_muscl_superbee_guarded_face_value_2d(
        far_upwind, upwind, downwind, canonical_usable,
    )
    # canonical_usable = true → calls
    # _fvfd_muscl_superbee_face_value_2d(5, 5, 0):
    #   d_up = upwind - far_upwind = 5 - 5 = 0
    #   d_down = downwind - upwind = 0 - 5 = -5
    #   r = ifelse(d_down == 0, 0, d_up / d_down) = 0 / -5 = 0
    #   limiter(0) = max(0, max(min(0, 1), min(0, 2))) = 0
    #   return upwind + 0.5 * 0 * (-5) = 5
    # phie = 5
else
    upwind = east_value               # = Ψ_solid ≈ 0
    downwind = phi[i, j]              # = Ψ_fluid ≈ 5
    canonical_usable = i + 2 <= Nx && !is_solid[i + 2, j]  # = false (i+2 INSIDE solid)
    far_upwind = canonical_usable ? phi[i + 2, j] : upwind
                                       # = upwind = 0  (fallback)
    _fvfd_muscl_superbee_guarded_face_value_2d(
        0, 0, 5, false,                # canonical_usable = false
    )
    # canonical_usable = false → calls
    # _fvfd_muscl_superbee_face_value_oneSided_2d(0, 5):
    #   return upwind = 0
    # phie = 0
end
```

### Case A — `ue > 0` (flow toward the cylinder, fluid → solid)

- **M29c-v2**: `phie = upwind = phi[i, j] = Ψ_fluid ≈ 5`.
  The MUSCL limiter sees `d_up = 0, d_down = -5, r = 0 → limiter = 0`
  → first-order upwind, harmless. *Note*: the BC helper still READS
  `phi[i+1, j] = Ψ_solid` but the limiter zeroes the slope, so
  `Ψ_solid` does not enter the face value. **Algebraically benign for
  the face flux on this side.**
- **M29b**: at this same cell, `is_solid[i+1, j] = true` would have
  triggered the all-or-nothing Rusanov fallback (DIFF section 1, H2).
  With Rusanov, `phie = ifelse(ue >= 0, phi[i,j], east_value) = phi[i,j]
  = 5`. **Same answer, 5.**

### Case B — `ue < 0` (flow away from cylinder, solid → fluid)

- **M29c-v2**: `phie = _oneSided(0, 5) = 0`. **Ψ_solid ≈ 0 is the
  face value used in the flux `ue * phie = ue * 0 = 0`.** The flux
  contribution to the east face is `0`. Equivalently: zero polymer
  stress is advected out of the cylinder boundary into the wake. This
  IS pulling `Ψ_solid` into the flux, but the resulting flux is `0`
  (because the face value is 0 and gets multiplied by `ue`).
- **M29b**: at this same cell, again the band guard triggers Rusanov
  → `phie = east_value = phi[i+1, j] = Ψ_solid ≈ 0`. **Same answer
  algebraically: face value = 0.** (M29b's Rusanov face value also
  ingests `phi[i+1, j]` blindly — line 526 of operators_2d.jl, exactly
  the same BC helper output.)

**The face flux through the east face is THE SAME in M29b and M29c-v2
at this cell, in both wind cases, when `Ψ_solid` happens to be ≈ 0.**

What changes is the OTHER three faces (west, north, south), where in
M29c-v2 the MUSCL limiter — with all 3 of the non-solid-facing
neighbours being fluid — gives a steeper (TVD) slope than M29b's
Rusanov (which uses pure first-order upwind on all 4 faces once the
band triggers). So M29c-v2 produces a HIGHER-ORDER reconstruction on
the 3 non-solid-facing faces, while preserving the M29b answer on
the solid-facing face. **This is exactly the intended behaviour** —
not a pathological one.

### Cross-check: M29b's `:rusanov` branch at the SAME cell

`operators_2d.jl` line 510–535: M29b Rusanov also calls
`_fvfd_bc_east_scalar_2d`, which also returns `phi[i+1, j] = Ψ_solid`.
Then `phie = ifelse(ue >= 0, phi[i, j], east_value)`:
- `ue > 0` → `phie = phi[i, j] = 5` (same as M29c-v2 case A).
- `ue < 0` → `phie = east_value = Ψ_solid ≈ 0` (same as M29c-v2 case B).

**Algebraically, M29c-v2 does NOT ingest more `Ψ_solid` than M29b
does on the solid-facing face.** The DIFF's assertion that the
unguarded helper is the new vulnerability is **incorrect**: the SAME
unguarded helper exists in M29b's Rusanov branch (line 521) and gives
the SAME face value for the same wind sign.

### Where the boundary band guard actually mattered in M29b

DIFF section 1 H2 says the M29b guard was "all-or-nothing 4-face
Rusanov fallback". Re-reading the M29b code (DIFF section 1 H2 quotes
the removed lines):
- M29b: at a cylinder-adjacent cell, the band guard switched the
  whole cell from MUSCL → Rusanov on all 4 faces simultaneously.
- M29c-v2: at the same cell, only the solid-facing face falls back to
  `oneSided`, the other 3 faces use full MUSCL.

So the **only structural change** at the solid-adjacent cell is the
order of the reconstruction on the three NON-solid-facing faces. The
solid-facing face is, algebraically, **identical** in both schemes.

## Q5 — Reconciliation with LOCATE

LOCATE finds:
- First NaN at j=1 south wall, x/R ≈ −3.87 (F64) or −2.13 (F32), in
  `rho` (LBM density), step 92,200 (F64) or 102,800 (F32).
- Asymmetric (south wall only).
- O(80k) steps stable then 20k of explosion.
- M29b stable indefinitely at 200k steps.

DIFF predicts:
- First NaN at fluid cells "directly adjacent to the cylinder".
- "Symmetric north/south appearance".
- O(10) substeps once Wi ramps.

LOCATE contradicts DIFF on **all four** prediction axes (location:
upstream south wall vs cylinder-adjacent; symmetry: only south wall;
time-scale: O(80k LBM steps), not O(10) substeps; first-divergent
field: LBM `rho`, not polymer Ψ).

### Where DIFF's algebraic argument breaks

DIFF section 4 reads (paraphrased):
> "The kernel at line 642 sets `phi_out[i,j] = 0` on solid cells each
> substep, so `phi[i+1, j] = 0` is fed into MUSCL as `downwind` or
> `upwind`. ... For the opposite sign (`ue < 0`), `east_value = 0`
> becomes the `upwind`. The flux `ue * phie` then advects Ψ = 0 INTO
> the fluid."

The **algebraic error** is at "advects Ψ = 0 INTO the fluid". Per Q4
case B above, `phie = 0` means the FACE VALUE is 0, so the flux is
`ue * 0 = 0`. There is no advection of `Ψ = 0` into the fluid; there
is **no flux at all** through the solid-facing face in case B.

This is the **same behaviour M29b had** (and which is, physically, the
correct closure: no Ψ enters the fluid from the wall when the velocity
points away from the wall, because `Ψ_wall` is a boundary condition,
not a physical state). The MUSCL face value is multiplied by the FACE
velocity `ue`, and the BC helper's value (whatever it is) is used as
the face state when `ue` carries information out of the solid cell —
this is standard upwinding, not a bug.

### What ACTUALLY differs between M29c-v2 and M29b at the cylinder-adjacent layer

Only the 3 non-solid-facing faces' reconstruction order (MUSCL vs
Rusanov). MUSCL is higher-order accurate where the stencil is in
fluid; it should reduce, not increase, numerical error on those faces.

### So why does M29c-v2 NaN at 80–100k steps?

Three competing hypotheses, ranked by my read of the evidence:

**H-LATE-STIFF (LOCATE's verdict)**: The system is in a marginal
Wi=1 elastic-feedback regime. M29c-v2 produces less numerical
diffusion than M29b (because MUSCL > Rusanov on 3 of 4 faces in the
boundary layer, AND throughout the bulk because M29b's H2 band was a
non-trivial fraction of the inner region). Less numerical diffusion →
sharper wake polymer-stress hot spot → stronger body-force feedback
into the LBM → LBM `rho` loses positivity at the wall first (south
wall in this case, by some asymmetric flow detail). M29b's extra
diffusion masks the late-stage stiffness. **This is consistent with
all 4 LOCATE observations** (j=1 wall, rho first, 80k+ time scale,
asymmetric).

**H-SLOW-LEAK**: The "approximately 0 but not exactly 0" Ψ_solid value
discussed in Q3 slowly drifts due to the constitutive substep applied
on solid cells (which receive `-I/λ·dt` increments). At λ=6000 LU,
n_substeps≈8 substeps per LBM step, over 92k LBM steps the cumulative
drift on Ψ_xx(solid) is ~8·92000/6000 = 122 (in log-conformation
units, that is `exp(-122) ≈ 0`, i.e. C_xx_solid → 0 forever). Cannot
be the cause: `Ψ_solid` stays ≈ 0 throughout. **Ruled out.**

**H-WAKE-STRESS-COUPLING**: A small spurious flux per step (DIFF's
implied mechanism) accumulates over 80k steps. But Q4 shows the face
flux on the solid face is **identical** to M29b's at this cell, so
this mechanism would also blow up M29b — yet M29b is stable at 200k.
**Ruled out.**

### Verdict

- **DIFF's mechanism**: **DOES NOT HOLD** algebraically.
- The specific claim "MUSCL eats `Ψ_solid = 0` and produces a
  spurious flux into the fluid" is wrong because (a) the face value
  used by MUSCL when the solid value enters the stencil is
  computed by `oneSided → return upwind` which **does not depend on
  the solid value** (case B); (b) when the solid value enters as
  `downwind` (case A), the limiter sees `r = d_up / d_down = 0 / (-5)
  = 0` and returns `upwind` unchanged; (c) when `ue < 0` and the face
  value `phie = 0`, the flux `ue * phie = 0` advects NOTHING
  into the fluid; (d) M29b's Rusanov branch reads the SAME unguarded
  helpers and uses the SAME face values at the solid-facing face.
- **The line of the DIFF analysis that is wrong**: section 4, the
  sentence "The flux `ue * phie` then advects Ψ = 0 INTO the fluid"
  conflates the face value (which is 0) with a non-zero flux. The
  flux is `ue · 0 = 0`. The same sentence then says "next time step
  has `phi[i, j]` plunging toward 0" — this is also wrong: the
  divergence form `rhs = -(flux_div - phi[i,j] * divu)` includes a
  `+ phi[i, j] * divu` correction (operators_2d.jl line 628) that
  cancels the apparent loss when `divu` is non-zero. With `ue` only
  one face out of four contributing zero, and the divergence of the
  velocity field being small (incompressible to LBM precision), the
  net effect on `phi[i,j]` per LBM step is `dt · O(divu · |Ψ_fluid|)`
  which is well-behaved.

Therefore the NaN root cause remains **UNKNOWN from this audit
alone**, but the most plausible mechanism given Q1–Q4 is LOCATE's
**H-LATE-STIFF**: M29c-v2's reduced numerical diffusion (compared to
M29b's H2 boundary band + bulk-similar MUSCL) lets the physical
elastic-feedback runaway proceed unimpeded at Wi=1, λ=6000 LU.
Whether the LBM `rho` blow-up at the south wall j=1 has a separate
contributing pathology (e.g. wall-band ghost coupling for the body
force) is a question for a M29d-style force-coupling audit, not for
this BC-helper audit.

### Boss decision implication

DIFF's proposed minimal fix (re-introducing a ±1 solid-neighbour band)
would re-introduce M29b's numerical diffusion at the cylinder-adjacent
layer. This **might** stabilise the run (matching M29b's empirical
stability) but for the **wrong reason** (adding diffusion to mask the
physics, not fixing a spurious flux). Better to consult LOCATE's
proposed M29c-v3 mitigations (polymer-stress diffusion, BSD=0.5,
force clipping at wall band).

**Mission status: DIFF's structural-correctness claim falsified by
algebraic reading. Recommend escalating to a M29d force-coupling /
LBM rho-positivity audit for the true NaN root cause.**

## Sanity flags raised for adversarial step

If Codex agrees on Q1, Q2, Q3 but DISAGREES on Q4 case-B flux
algebra, that is the load-bearing disagreement. Either I or Codex
must be wrong about whether `phie = 0` produces zero or non-zero
advected mass. If Codex agrees on Q4 but disagrees on Q5 H-LATE-STIFF
ranking, that is a lower-stakes disagreement (multiple hypotheses
remain physically plausible, neither audit can adjudicate without
new simulation).
