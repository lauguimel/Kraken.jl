# M29c-v2 vs M29b structural diff — VERDICT

Department: M29c-v2-postmortem-diff (Layer 1, no Engineer spawn).
Date: 2026-05-19.
Branch: `dev-viscoelastic`.
Files compared:
- M29b: `git show 42d2177a:src/fvfd/operators_2d.jl` → `/tmp/m29b_operators_2d.jl` (1187 lines).
- M29c-v2: working tree `src/fvfd/operators_2d.jl` (1251 lines).
Diff: `/tmp/m29c_v2_vs_m29b.diff` (150 lines, +88 −24).

## 1. Summary table of structural changes

| # | Hunk lines | Change | Classification | Notes |
|---|-----------|--------|----------------|-------|
| H1 | diff L 3–27 (file +488..+508) | Added `_fvfd_muscl_superbee_face_value_oneSided_2d` (returns `upwind`) and dispatch `_fvfd_muscl_superbee_guarded_face_value_2d(far_upwind, upwind, downwind, canonical_usable)` | **Structural** (new code path) | The "M29c-v2" 1-line fix lives here: `oneSided` returns `upwind` instead of `(upwind+downwind)/2`. TVD with slope=0. |
| H2 | diff L 30–44 | Removed the M29b boundary band guard `i ≤ 2 ∥ i ≥ Nx−1 ∥ j ≤ 2 ∥ j ≥ Ny−1 ∥ is_solid[i±1..i±2, j±1..j±2] → Rusanov` | **Structural** (algorithm change) | The all-or-nothing 4-face Rusanov fallback is GONE. MUSCL is now always attempted. |
| H3 | diff L 45–70 → +71..+146 | Rewrote the 4 face ifelse blocks (phie/phiw/phin/phis) from `phi[i±k,j]` indexing to a per-face / per-sign closure using `_fvfd_bc_*_scalar_2d` BC helpers + `canonical_usable` per-side guard + zero-slope fallback | **Structural** (algorithm change AND new BC code path) | M29b read `phi[i+1,j]`, `phi[i+2,j]`, etc. directly (assumed bulk because of H2 guard). M29c-v2 routes EVERY downstream cell through `east_value/west_value/north_value/south_value`, the BC helpers. |

Total hunks: **3** (one combined chunk in diff but logically 3 distinct algorithmic changes). All three are **structural**, **0 cosmetic**. The diff is +88 −24 net.

## 2. Boundary band width before / after

| | M29b (42d2177a) | M29c-v2 (working tree) |
|---|----------------|-----------------------|
| Domain band | i ≤ 2 ∥ i ≥ Nx − 1 ∥ j ≤ 2 ∥ j ≥ Ny − 1 → **±2 cells** | **±0 cells** (MUSCL fires up to and including i=1, i=Nx, j=1, j=Ny) |
| Solid neighbour band | `is_solid[i±1, j], is_solid[i±2, j], is_solid[i, j±1], is_solid[i, j±2]` → **±2 cells** | per-face, per-side `canonical_usable = !is_solid[far_upwind cell]`; **0** on the downwind side (no solid check) |
| Fallback on collapse | full 4-face Rusanov 1st-order | per-face: zero-slope MUSCL on upwind branch only |

**The boundary band width collapsed from ±2 to ±0 cells. This matches structural cause hypothesis (S1) in the mission brief.**

## 3. Canonical_usable guard — detailed audit

The 8 per-sign guards in M29c-v2 are:

| Face | Sign | upwind cell | downwind cell | far_upwind cell | guard | comments |
|------|------|-------------|---------------|-----------------|-------|---------|
| east | ue ≥ 0 | phi[i,j] | east_value (= phi[i+1,j] in bulk) | phi[i−1,j] | i > 1 ∧ ¬is_solid[i−1,j] | downwind `east_value` UNGUARDED against solid `phi[i+1,j]` |
| east | ue < 0 | east_value | phi[i,j] | phi[i+2,j] | i+2 ≤ Nx ∧ ¬is_solid[i+2,j] | **upwind** `east_value` UNGUARDED against solid `phi[i+1,j]` |
| west | uw ≥ 0 | west_value | phi[i,j] | phi[i−2,j] | i > 2 ∧ ¬is_solid[i−2,j] | **upwind** `west_value` UNGUARDED against solid `phi[i−1,j]` |
| west | uw < 0 | phi[i,j] | west_value | phi[i+1,j] | i < Nx ∧ ¬is_solid[i+1,j] | downwind `west_value` UNGUARDED against solid `phi[i−1,j]` |
| north | vn ≥ 0 | phi[i,j] | north_value | phi[i,j−1] | j > 1 ∧ ¬is_solid[i,j−1] | downwind `north_value` UNGUARDED against solid `phi[i,j+1]` |
| north | vn < 0 | north_value | phi[i,j] | phi[i,j+2] | j+2 ≤ Ny ∧ ¬is_solid[i,j+2] | **upwind** `north_value` UNGUARDED against solid `phi[i,j+1]` |
| south | vs ≥ 0 | south_value | phi[i,j] | phi[i,j−2] | j > 2 ∧ ¬is_solid[i,j−2] | **upwind** `south_value` UNGUARDED against solid `phi[i,j−1]` |
| south | vs < 0 | phi[i,j] | south_value | phi[i,j+1] | j < Ny ∧ ¬is_solid[i,j+1] | downwind `south_value` UNGUARDED against solid `phi[i,j+1]` |

The guards are **symmetric east/west, north/south**: no S2-style off-by-one asymmetry, no S4-style `≤ vs <` typo (all upper-bound guards use `i + 2 ≤ Nx`, `j + 2 ≤ Ny`, which is correct since fluid indices run 1..Nx). The flux-divergence assembly is identical to M29b.

So **(S2), (S3), (S4) are NOT the root cause**.

## 4. Most likely root cause of the NaN — (S1+) BC HELPER READS SOLID Ψ

(S1) is correct ("boundary band collapsed → MUSCL fires at i=1, j=1, etc."), **but the killer is sharper**: the BC helpers `_fvfd_bc_east_scalar_2d` (lines 422–468 of the same file) only check **domain bounds** (`i < Nx`), they do **NOT** check `is_solid`. Concretely:

```
function _fvfd_bc_east_scalar_2d(phi, east_phi, i, j, Nx, east_bc)
    if i < Nx
        return phi[i + 1, j]   # ← reads phi blindly, even if (i+1,j) is solid
    elseif east_bc == FVFD_BC_PERIODIC
        ...
```

Cylinder geometry: at a fluid cell `(i, j)` adjacent to the cylinder west surface, `is_solid[i + 1, j] = true`. In M29b the boundary-band guard caught `is_solid[i+1,j] ∥ is_solid[i+2,j]` and switched all 4 faces to Rusanov, where for `ue ≥ 0` we have `phie = phi[i, j]` — the solid Ψ is **never** referenced.

In M29c-v2 the same cell calls `east_value = _fvfd_bc_east_scalar_2d(...) = phi[i + 1, j]`, which is the solid cell's Ψ. The kernel at line 642 sets `phi_out[i, j] = 0` on solid cells **on each substep**, so `phi[i+1, j] = 0` is fed into MUSCL as `downwind` (when `ue ≥ 0`) or as `upwind` (when `ue < 0`).

For log-conformation Ψ, the physical fluid values at Wi = 1 R = 30 reach Ψ_xx of order 5-10 at the cylinder front/back stagnation. Feeding Ψ_solid = 0 into the MUSCL limiter then evaluates:
- `d_up = upwind − far_upwind` = O(1)
- `d_down = downwind − upwind` = 0 − 5 = −5
- `r = d_up / d_down` finite, limiter ≤ 2.
- The face flux uses `downwind = 0` directly, producing a face value O(2.5) (mid-cell extrapolation of upwind toward a phantom Ψ = 0).

For the **opposite sign** (`ue < 0`, flow into the wall), `east_value = 0` becomes the `upwind`. The flux `ue * phie` then advects Ψ = 0 INTO the fluid. The next time step has `phi[i, j]` plunging toward 0, then `exp(Ψ)` of an over-corrected gradient picks up huge values. After a few sub-steps log-conformation **exponentiates** the artefact and the run NaNs.

This was hidden in M29c (CD2 form `(upwind+downwind)/2`) by the **anti-TVD** behaviour (which collapsed to a softer ~ 50 % linear mix of solid-0 + fluid-Ψ, giving a negative Cd = −1571 but not NaN). Switching to **strict upwind** `oneSided := upwind` in M29c-v2 makes the kernel return `upwind` (good for TVD) but the flux `ue * downwind` (when `ue < 0`) still pulls `Ψ_solid = 0` in as upwind, AND on the opposite face configuration the MUSCL stencil still reads `Ψ_solid` as downwind. The TVD fix only protects the **face value** computation, not the **downwind / upwind selection** from BC helpers.

**Verdict: the root cause is (S1) extended: removal of the ±2 boundary band exposed an unguarded read of `phi[solid neighbour]` via the `_fvfd_bc_*_scalar_2d` helpers. M29c-v2's `oneSided ← upwind` mitigates the limiter side but not the unguarded BC read.**

Predicted spatial pattern (to be compared with postmortem-locate): NaN should emerge at fluid cells *directly adjacent to the cylinder*, especially front and rear stagnation where |Ψ| is largest, within a handful of substeps (≤ 10) once Wi ramps. Symmetric north/south appearance.

## 5. Proposed minimal fix

Two viable 1-line family fixes; preferred is (A) because it preserves the M29c-v2 TVD intent without re-introducing the M29b heavy band:

**(A) Guard the BC helpers for solid (preferred — surgical).**

Augment each per-side path so that when `is_solid[downwind cell]` is true, fall back to the same `oneSided` form using `upwind` as both upwind and downwind:

For the east face, replace
```
canonical_usable = i > 1 && !is_solid[i - 1, j]
```
by
```
downwind_solid = i < Nx && is_solid[i + 1, j]
canonical_usable = i > 1 && !is_solid[i - 1, j] && !downwind_solid
downwind = ifelse(downwind_solid, upwind, east_value)
```
and analogously for the other 7 cases (8 lines, not 1, but minimal and localised). Returning `upwind` for both face value AND BC value at a solid-neighbour face is exactly equivalent to a first-order upwind Rusanov face there — i.e. the M29b behaviour restricted to the offending face only (no need to fall back all 4 faces).

**(B) Re-introduce a ±1 solid neighbour band (least diff to M29b, safest).**

Insert immediately before `ue = ux_face[i + 1, j]` in the `:muscl_superbee` branch:
```
if is_solid[max(i-1,1), j] || is_solid[min(i+1,Nx), j] ||
   is_solid[i, max(j-1,1)] || is_solid[i, min(j+1,Ny)]
    return _fvfd_upwind_scalar_advective_rhs_2d(
        phi, west_phi, east_phi, south_phi, north_phi,
        ux_face, uy_face, is_solid, i, j, Nx, Ny, inv_dx, inv_dy,
        west_bc, east_bc, south_bc, north_bc, Val(:rusanov),
    )
end
```
This is a **single block** patch that restores M29b behaviour at the 1-cell-from-solid layer (less aggressive than M29b's ±2, but enough since the unguarded read is at i+1, not i+2). Keeps M29c-v2 zero-slope MUSCL benefits in the bulk and near domain edges where M29b was needlessly degraded.

**Recommended**: ship (B) first to unblock production (Cd ladder R = 30 Wi = 1.0), then explore (A) as an optimisation if convergence shows the ±1 Rusanov band is too dissipative on the cylinder.

## 6. Cross-reference with empirical partner

When the postmortem-locate partner reports the spatial / temporal NaN pattern, expect:
- First NaN at a fluid cell whose 4-neighbourhood contains a solid cell (cylinder west/east/north/south surface).
- Likely on the **upstream** side (large |Ψ| from front stagnation accumulation).
- Time-to-NaN: O(10) substeps once Wi ramp reaches ~ 0.3-0.5, because the spurious flux into the wall cell grows linearly in |Ψ|.

If the partner instead finds NaN deep in the bulk or at the outlet, my prediction is falsified and (S2)/(S3) need a re-examination.

## 7. Files

- This verdict: `bench/viscoelastic_audit/M29C_V2_DIFF_VERDICT.md`.
- Working diff: `/tmp/m29c_v2_vs_m29b.diff` (150 lines).
- M29b kernel snapshot: `/tmp/m29b_operators_2d.jl` (1187 lines).
- M29c-v2 kernel under analysis: `src/fvfd/operators_2d.jl` (1251 lines, uncommitted patch).
