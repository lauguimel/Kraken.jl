# M55 DERIV Boss Derivation

Derived offline in Claude (Boss) conversation, prior to adversarial
3-way comparison. Written here AFTER both engineers completed their
independent derivations.

## 1. Setup

Coordinate `s` along the wall outward normal, with the wall at `s = 0`.
Quadratic ansatz `u(s) = u_wall + a·s + b·s²` automatically satisfies
`u(0) = u_wall`. Two remaining unknowns `(a, b)` fit by two fluid cell
center samples at `s₁ = q_w·dx` (value `u₁`) and `s₂ = (q_w+1)·dx`
(value `u₂`). We want `a = ∂u/∂n|_wall`.

## 2. System

Let `f₁ = u₁ − u_wall`, `f₂ = u₂ − u_wall`. Then

```
[s₁  s₁²] [a]   [f₁]
[s₂  s₂²] [b] = [f₂]
```

## 3. Solution

Determinant: `D = s₁·s₂² − s₂·s₁² = s₁·s₂·(s₂ − s₁) = s₁·s₂·dx`.

By Cramer:
- `a = (f₁·s₂² − f₂·s₁²) / D`
- `b = (s₁·f₂ − s₂·f₁) / D`

## 4. Closed form

Substituting `s₁ = q_w·dx`, `s₂ = (q_w+1)·dx`:

```
∂u/∂n|_wall = a = [(q_w+1)²·u₁ − q_w²·u₂ − (2q_w+1)·u_wall]
                / [q_w·(q_w+1)·dx]
```

(Using `(q_w+1)² − q_w² = 2q_w+1` for the `u_wall` coefficient.)

## 5. Sanity check at q_w = 1/2

`(q_w+1)² = 9/4`, `q_w² = 1/4`, `2q_w+1 = 2`, `q_w(q_w+1) = 3/4`.

```
a = [(9/4)·u₁ − (1/4)·u₂ − 2·u_wall] / [(3/4)·dx]
  = [9·u₁ − u₂ − 8·u_wall] / (3·dx)
  = [3·u₁ − u₂/3 − (8/3)·u_wall] / dx     ✓ M51
```

## 6. Limiting behavior

- `q_w → 0`: denominator `q_w·(q_w+1)·dx → 0`, numerator
  `→ 1·u₁ − 0·u₂ − 1·u_wall = u₁ − u_wall` (finite, nonzero in general)
  ⇒ formula diverges as `1/q_w`. Physical interpretation: wall is at
  the cell center, no fluid samples on the "fluid" side — the
  quadratic ansatz is degenerate. Fallback to first-order or
  halfway-BB is required.
- `q_w → 1`: `s₁ = dx`, `s₂ = 2·dx`. Substituting:
  `a = (4·u₁ − u₂ − 3·u_wall) / (2·dx)`
  ⇒ standard textbook 3-point forward FD with wall-Dirichlet. ✓

## 7. Flags

- Numerator coefficient sum is zero (`(q_w+1)² − q_w² − (2q_w+1) = 0`),
  so a constant field `u ≡ const` produces `a = 0` as expected.
- Conditioning near `q_w → 0` is the only delicate region; cap with
  `q_w_min` floor.
