# First-principles derivation of the Bouzidi q ≤ 0.5 branch

## Setting

- D2Q9 lattice. `c_q` velocity vector for direction `q`, `q̄` opposite direction.
- Fluid cell `x_f` at index `(i, j)`. Solid cell across link `q`: `x_s = x_f + c_q`.
- Wall hit point `x_w = x_f + q · c_q` with `q ∈ (0, 1]` (Lallemand–Luo / BFL
  convention: `q = ‖x_w − x_f‖ / ‖x_s − x_f‖`).
- Half-way back: `q = 0.5`. `q < 0.5` ⇒ wall is "near" `x_f` (closer than the
  midpoint). `q > 0.5` ⇒ wall is "far" from `x_f` (closer to the solid).
- "Far-fluid neighbour" of `x_f` along link `q` is `x_ff = x_f − c_q` (always
  in the fluid sub-domain when defined).

## Algorithm in canonical form (Bouzidi, Firdaouss, Lallemand 2001, eq. 18-19)

Notation (Bouzidi-FL): `f̃_q(x, t)` = post-collision population at cell `x`
before streaming (we call this `f_post[x, q]`). After streaming, the population
arriving at `x_f` in direction `q̄` (back from the wall) is what we want to
overwrite.

```
if q ≤ 0.5:
    f_q̄(x_f, t+dt) = 2q · f̃_q(x_f, t) + (1 − 2q) · f̃_q(x_ff, t)
                     − 2 W_q · ρ_w · (c_q · u_w) / c_s²            … (A)
if q > 0.5:
    f_q̄(x_f, t+dt) = (1/(2q)) · f̃_q(x_f, t)
                     + ((2q − 1)/(2q)) · f̃_q̄(x_f, t)
                     − (1/q) · W_q · ρ_w · (c_q · u_w) / c_s²      … (B)
```

Both branches read `f̃` AT TIME `t` (post-collision of the CURRENT step,
which then streams to the next step). The wall correction reads `ρ_w` which
in the standard Ladd convention is the LOCAL fluid density at `x_f` at the
**current** macroscopic step (so it is also lag-0 — the value computed by
the same step's collision input).

## Where each read comes from (population locations)

| Read | What is it? | Time level | Storage in Kraken |
|---|---|---|---|
| `f̃_q(x_f, t)`  | post-collision pop `q` at the cell that owns the cut link | **current step, post-collision** (= "lag-0") | `f_out[i, j, q]` after `CollideTRTDirectGuoField` writes it |
| `f̃_q(x_ff, t)` | post-collision pop `q` at the upstream-of-link fluid neighbour | **current step, post-collision** (= "lag-0") | `f_out[i_ff, j_ff, q]` |
| `f̃_q̄(x_f, t)` | post-collision pop `q̄` at the SAME cell | lag-0 | `f_out[i, j, q̄]` |
| `ρ_w` | wall density (Ladd: copy the local fluid ρ) | lag-0 | `ρ_out[i, j]` after `WriteMoments` |

## Critical observations about the two-pass spec

The M30 P2b "Proposed fix" says: pass-1 = the existing
`_TRT_LIBB_V2_GUO_FIELD_SPEC`, pass-2 = the new
`ApplyBouzidiFLPostCollideTwoPass` brick reading `f_out` everywhere.

### Observation 1 — pass-1 ALREADY contains a Bouzidi correction

`_TRT_LIBB_V2_GUO_FIELD_SPEC` (li_bb_2d_v2.jl:49-54) is:

```
PullHalfwayBB, SolidInert, ApplyLiBBPrePhase, Moments,
CollideTRTDirectGuoField, WriteMoments
```

`ApplyLiBBPrePhase` (bricks.jl:355-402) is "FULL Bouzidi interpolated
bounce-back at the PRE-COLLISION phase. For arbitrary q_w ∈ (0, 1]" with the
lag-1 storage convention.

That means after pass-1, at every cut link the corrupted-pulled-pop `fp_q̄`
has ALREADY been replaced by a lag-1 Bouzidi estimate, and the collision was
performed on populations that include this Bouzidi-pre-corrected pop.
**So `f_out[i, j, q̄]` already contains a (lag-1) Bouzidi-FL-corrected pop**
when pass-2 reads it.

### Observation 2 — pass-2 then OVERWRITES that same `f_out[i, j, q̄]`

Pass-2's emit_code (bricks.jl:567-691) writes
`f_out[i, j, 4] = _bouzidi_fl_post_value(qw2, …)`, etc., for all 8 cut
directions. So pass-2 unconditionally replaces the value that pass-1 wrote
into `f_out[i, j, q̄]`. The pass-1 pre-phase substitution work is **DISCARDED**
on every cut link.

### Observation 3 — `f_out[i_ff, j_ff, q]` is what?

After pass-1, at the far-fluid neighbour `x_ff = x_f − c_q`:

- If `x_ff` is a regular fluid cell (no cut link adjacent), then
  `f_out[x_ff, q]` is the collision output for pop `q` at `x_ff`, which is
  exactly `f̃_q(x_ff, t)`. ✓ (matches canonical formula)
- If `x_ff` is itself a cut-link cell (rare, possible for q ≤ 0.5 on the
  cylinder front shoulder), then `f_out[x_ff, q]` is the collision output
  for pop `q`, and pass-2 does NOT overwrite `f_out[x_ff, q]` for cell
  `x_ff` (pass-2 only overwrites `q̄`-direction pops on the cell that owns
  the cut link — pop `q` is untouched).
  ✓ — still equals `f̃_q(x_ff, t)`.

So `f_out[i_ff, j_ff, q]` after pass-1 = `f̃_q(x_ff, t)` ≡ canonical lag-0.
**Match with canonical.** ✓

### Observation 4 — `f_out[i, j, q̄]` read on the q > 0.5 branch

Pass-2 emit_code reads `f_qbar_here` from the snapshot `f2_here..f9_here =
f_out[i, j, *]` at the top (bricks.jl:571-578). After pass-1, `f_out[i, j, q̄]`
is the collision output for pop `q̄` at `x_f`. ✓ canonical.

BUT — for cut directions where pass-1's `ApplyLiBBPrePhase` substituted
`fp_q̄` with a Bouzidi expression BEFORE collision, the collision output
`f_out[i, j, q̄]` is the post-collision evolution of that substituted pop.

The canonical formula reads `f̃_q̄(x_f, t)` which means "the post-collision
output for direction `q̄` at `x_f` at step `t`". As long as pass-1's
collision is "the" current-step collision, this is by definition lag-0. ✓

**However the q̄ pop that pass-1's collision processes was based on a LAG-1
Bouzidi-FL substitution for `fp_q̄`**. This means `f̃_q̄(x_f, t)` itself
already encodes a Bouzidi-FL correction. Pass-2 then applies a SECOND
Bouzidi-FL correction (on the q̄ output). **This is the same DOUBLE-BC
class as the bug that motivated the V2 refactor in the first place**
(see li_bb_2d_v2.jl:12-20 header comment).

## Wall density rho_w lag check

Pass-2 reads `rho_w = ρ_out[i, j]` (bricks.jl:568). After pass-1's
`WriteMoments`, this is the macroscopic ρ from pass-1's `Moments` brick,
which was computed from `fp1..fp9` (the pulled, pre-collision pops). For
cut-link cells, pass-1's `ApplyLiBBPrePhase` substituted some of those
`fp_q̄`, so `ρ_out[i, j]` is built from Bouzidi-pre-corrected pops, not raw
pulled pops.

This is NOT the canonical `ρ_w` (which is the "local fluid density" — there's
ambiguity here, but the standard interpretation is the density from the
raw post-stream pops at `x_f`). It's a close approximation; the dominant
defect is the double-Bouzidi, not the ρ_w.

## Summary

The M30 P2b "Proposed fix" naively re-uses `_TRT_LIBB_V2_GUO_FIELD_SPEC` as
pass-1, but that spec contains `ApplyLiBBPrePhase` which is itself a Bouzidi
correction. The result is a DOUBLE Bouzidi-FL application on every cut link
(once pre-collision lag-1, once post-collision lag-0). This is the same
double-BC pathology that the V2 refactor was designed to eliminate (see
li_bb_2d_v2.jl header lines 12-20).

For pass-2 to read a CANONICAL `f̃_q̄(x_f, t)`, pass-1 must NOT have
already applied a Bouzidi correction in any form. Pass-1 should be a "raw"
pull+collide+write spec WITHOUT the pre-phase substitution.
