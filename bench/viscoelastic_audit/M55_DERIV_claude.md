# M55 DERIVATION — Claude variant

Wall-normal derivative `a = ∂u/∂n|_wall` from a quadratic interpolant
through one Dirichlet wall sample at `s = 0` and two fluid cell-center
samples at `s₁ = q_w·dx` and `s₂ = (q_w+1)·dx`.

## 1. Setup

Let `s` be the wall-normal coordinate, `s = 0` at the wall. We are given
three samples:
- `u(0) = u_wall` (Dirichlet datum at the wall);
- `u(s₁) = u₁` at `s₁ = q_w·dx`, the first fluid cell-center;
- `u(s₂) = u₂` at `s₂ = (q_w+1)·dx`, the second fluid cell-center.

Here `0 < q_w ≤ 1` is the fractional wall offset and `dx > 0` is the
grid spacing along the normal. We seek the quadratic interpolant
`u(s) = u_wall + a·s + b·s²` passing through all three samples, and in
particular its slope at the wall, `a = ∂u/∂n|_wall`.

## 2. System

By construction `u(0) = u_wall` is satisfied automatically. The
remaining two constraints read:

    a·s₁ + b·s₁² = u₁ − u_wall
    a·s₂ + b·s₂² = u₂ − u_wall

In matrix form, with unknown vector `(a, b)ᵀ`:

    [ s₁   s₁² ] [ a ]   [ u₁ − u_wall ]
    [ s₂   s₂² ] [ b ] = [ u₂ − u_wall ]

Introduce shorthand `d₁ = u₁ − u_wall`, `d₂ = u₂ − u_wall`. Substitute
`s₁ = q_w·dx`, `s₂ = (q_w+1)·dx`:

    [ q_w·dx        q_w²·dx²       ] [ a ]   [ d₁ ]
    [ (q_w+1)·dx    (q_w+1)²·dx²   ] [ b ] = [ d₂ ]

## 3. Solution

Determinant of the 2×2 matrix:

    D = s₁·s₂² − s₂·s₁²
      = s₁·s₂·(s₂ − s₁)
      = (q_w·dx) · ((q_w+1)·dx) · ((q_w+1 − q_w)·dx)
      = q_w·(q_w+1)·dx³.

Since `s₂ − s₁ = dx` always, the determinant is `D = q_w·(q_w+1)·dx³`.
It vanishes only at `q_w = 0` (wall coincides with first cell-center),
which is the geometric degeneracy.

Cramer's rule for `a`:

    a = ( d₁·s₂² − d₂·s₁² ) / D
      = ( d₁·(q_w+1)²·dx² − d₂·q_w²·dx² ) / ( q_w·(q_w+1)·dx³ )
      = ( d₁·(q_w+1)² − d₂·q_w² ) / ( q_w·(q_w+1)·dx ).

(For completeness, `b = (d₂·s₁ − d₁·s₂)/D
= (d₂·q_w − d₁·(q_w+1)) / (q_w·(q_w+1)·dx²)`,
but we only need `a`.)

Substituting back `d₁ = u₁ − u_wall`, `d₂ = u₂ − u_wall`:

    a = [ (q_w+1)²·(u₁ − u_wall) − q_w²·(u₂ − u_wall) ]
        / [ q_w·(q_w+1)·dx ].

The `u_wall` coefficient simplifies. The numerator's contribution from
`u_wall` is `−[(q_w+1)² − q_w²]·u_wall = −(2·q_w + 1)·u_wall`, since
`(q_w+1)² − q_w² = 2·q_w + 1`.

## 4. Closed form

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   a = [ (q_w+1)²·u₁ − q_w²·u₂ − (2·q_w + 1)·u_wall ]                │
│       ─────────────────────────────────────────────                 │
│                  q_w·(q_w+1)·dx                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Equivalently, with coefficients explicit:

    a = [ C_w · u_wall + C₁ · u₁ + C₂ · u₂ ] / dx

where

    C₁ = (q_w + 1) / q_w                       =  1 + 1/q_w
    C₂ = − q_w / (q_w + 1)                     = −1 + 1/(q_w+1)
    C_w = − (2·q_w + 1) / ( q_w·(q_w+1) )      = −(C₁ + C₂)   ← partition

The identity `C_w + C₁ + C₂ = 0` is the standard consistency condition
for a derivative operator (it annihilates constants).

## 5. Sanity check at q_w = 1/2

Substitute `q_w = 1/2` into the boxed formula.

Coefficients:
- `(q_w+1)² = (3/2)² = 9/4`;
- `q_w² = 1/4`;
- `(2·q_w + 1) = 2`;
- Denominator: `q_w·(q_w+1)·dx = (1/2)·(3/2)·dx = (3/4)·dx`.

Numerator: `(9/4)·u₁ − (1/4)·u₂ − 2·u_wall`.

Therefore:

    a = [ (9/4)·u₁ − (1/4)·u₂ − 2·u_wall ] / ( (3/4)·dx )
      = (4 / (3·dx)) · [ (9/4)·u₁ − (1/4)·u₂ − 2·u_wall ]
      = [ 9·u₁ − u₂ − 8·u_wall ] / (3·dx)
      = [ 3·u₁ − u₂/3 − (8/3)·u_wall ] / dx.   ✓

This matches the M51 reference formula exactly. ALGEBRA CONFIRMED.

## 6. Limiting behavior

**`q_w → 0` (wall approaches first fluid cell-center):**
The denominator `q_w·(q_w+1)·dx → 0`, so the formula has a 1/q_w
singularity. Specifically `C₁ = 1 + 1/q_w → ∞` while `C₂ → 0` and
`C_w → −1/q_w + O(1)`. The singularity reflects the geometric
degeneracy: when the wall hits the first cell-center, samples `u_wall`
and `u₁` coincide in position, so the quadratic fit through three
co-located/colinear data is ill-posed. The leading divergent piece is
`a ≈ (u₁ − u_wall)/(q_w·dx)`, which is itself just the centered
finite difference over the vanishing distance — consistent and benign
if `(u₁ − u_wall) = O(q_w)`, but a numerical hazard otherwise.

**`q_w → 1` (wall sits at a cell-face, samples on a uniform grid):**
The cell-centers fall on the standard grid `s₁ = dx`, `s₂ = 2·dx`.
Coefficients become:
- `C₁ = (1+1)/1 = 2`;
- `C₂ = −1/2`;
- `C_w = −(2+1)/(1·2) = −3/2`.

The formula reduces to

    a = [ 2·u₁ − u₂/2 − (3/2)·u_wall ] / dx
      = [ 4·u₁ − u₂ − 3·u_wall ] / (2·dx).

This is the textbook **3-point one-sided second-order forward
difference** at a grid node (Fornberg / Taylor expansion). Confirms
consistency at the regular (non-cut) limit.

## 7. Flags

- **Conditioning:** `1/q_w` divergence dominates as `q_w → 0`. For
  q_w below ~0.1, expect catastrophic cancellation in the numerator
  when `(u₁ − u_wall)` is itself small (typical near no-slip wall).
  A safeguard `q_w_min ≈ 0.05–0.1` or a fallback to a halfway-BB
  treatment is advisable.
- **Coefficient partition:** `C_w + C₁ + C₂ = 0` (annihilates constants
  → consistent stencil) and `C₁·s₁ + C₂·s₂ = 1` can be verified by
  direct substitution; this is the order-2 consistency condition
  (reproduces linear functions exactly), hence the formula is
  formally second-order accurate `O(dx²)` in the truncation error of
  `∂u/∂n|_wall` for smooth `u`.
- **q_w = 1/2 collapse:** at q_w=1/2 the wall sits exactly between
  two cell-centers, recovering the M51 formula — useful unit test.
- **q_w = 1 collapse:** at q_w=1 we recover the classical 3-point
  forward FD; this is a second useful unit test.
- **Sign convention:** `n` is the outward normal from the wall into
  the fluid, i.e. positive `s` points into the fluid. If the caller
  needs `∂u/∂n` with `n` pointing INTO the wall, flip the sign.
- **Wall-tangent vs wall-normal:** `u` here is whichever velocity
  component the caller wants differentiated normal to the wall; the
  algebra is component-agnostic. Apply it once per component.
