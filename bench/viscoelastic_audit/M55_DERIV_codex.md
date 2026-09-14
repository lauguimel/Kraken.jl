# M55 DERIV Codex Derivation

## 1. Setup

Let `s` be distance along the wall normal, with the wall at `s = 0`.
The wall Dirichlet value is `u_wall`, and the two fluid cell-center
samples are `u_1` at `s_1 = q_w dx` and `u_2` at
`s_2 = (q_w + 1) dx`, with `0 < q_w <= 1`. Fit
`u(s) = u_wall + a s + b s^2`; the wall constraint is built in, so
the unknowns are `a` and `b`. The desired wall-normal derivative is
`a = du/dn|_wall`.

## 2. System

Define

```text
q = q_w
h = dx
d_1 = u_1 - u_wall
d_2 = u_2 - u_wall
```

The two fluid constraints are

```text
a (q h)       + b (q h)^2       = d_1
a ((q+1) h)  + b ((q+1) h)^2   = d_2
```

Equivalently,

```text
[ q h        q^2 h^2       ] [ a ] = [ d_1 ]
[ (q+1) h    (q+1)^2 h^2   ] [ b ]   [ d_2 ]
```

## 3. Solution

The matrix determinant is

```text
D = (q h)((q+1)^2 h^2) - ((q+1) h)(q^2 h^2)
  = h^3 [ q(q+1)^2 - q^2(q+1) ]
  = h^3 q(q+1) [ (q+1) - q ]
  = h^3 q(q+1)
```

For `a`, Cramer's rule replaces the first column with the right-hand
side:

```text
D_a = d_1 ((q+1)^2 h^2) - d_2 (q^2 h^2)
    = h^2 [ (q+1)^2 d_1 - q^2 d_2 ]
```

Therefore,

```text
a = D_a / D
  = [ (q+1)^2 d_1 - q^2 d_2 ] / [ h q(q+1) ]
```

Substitute `d_1 = u_1 - u_wall` and `d_2 = u_2 - u_wall`:

```text
a = [ (q+1)^2 (u_1 - u_wall) - q^2 (u_2 - u_wall) ]
    / [ h q(q+1) ]

  = [ (q+1)^2 u_1 - q^2 u_2 - ((q+1)^2 - q^2) u_wall ]
    / [ h q(q+1) ]

  = [ (q+1)^2 u_1 - q^2 u_2 - (2q+1) u_wall ]
    / [ h q(q+1) ]
```

For completeness, `b` is

```text
D_b = (q h) d_2 - ((q+1) h) d_1
    = h [ q d_2 - (q+1) d_1 ]

b = D_b / D
  = [ q d_2 - (q+1) d_1 ] / [ h^2 q(q+1) ]
  = [ q u_2 - (q+1) u_1 + u_wall ] / [ h^2 q(q+1) ]
```

## 4. Closed form

Restoring `q_w` and `dx`,

```text
boxed:
du/dn|_wall = a =
[(q_w+1)^2 u_1 - q_w^2 u_2 - (2 q_w + 1) u_wall]
/ [dx q_w (q_w+1)]
```

## 5. Sanity check at q_w = 1/2

Set `q_w = 1/2`. Then `q_w + 1 = 3/2`,
`q_w^2 = 1/4`, `2 q_w + 1 = 2`, and
`q_w(q_w+1) = (1/2)(3/2) = 3/4`.

```text
a = [ (3/2)^2 u_1 - (1/2)^2 u_2 - 2 u_wall ]
    / [ dx (3/4) ]

  = [ (9/4) u_1 - (1/4) u_2 - 2 u_wall ]
    / [ (3/4) dx ]

  = [ 9 u_1 - u_2 - 8 u_wall ] / [ 3 dx ]
```

This is exactly the M51 formula:

```text
du/dn|_wall = (9 u_1 - u_2 - 8 u_wall) / (3 dx)
```

Equivalently,

```text
du/dn|_wall = (3 u_1 - u_2/3 - (8/3) u_wall) / dx
```

## 6. Limiting behavior

As `q_w -> 0`, the denominator `dx q_w(q_w+1)` goes to zero, and the
coefficients on `u_1` and `u_wall` diverge like `1/q_w`. This is the
expected degeneracy: the wall point and first fluid-center sample
coalesce, so two interpolation nodes occupy the same location. A finite
limit requires consistent smooth data with `u_1 - u_wall = O(q_w dx)`;
otherwise any mismatch is amplified singularly.

As `q_w -> 1`, the samples are at `dx` and `2 dx`. The formula becomes

```text
a = [ 4 u_1 - u_2 - 3 u_wall ] / [ 2 dx ]
```

This is the standard second-order one-sided derivative at the boundary
using values at `0`, `dx`, and `2 dx`:

```text
u'(0) = (-3 u(0) + 4 u(dx) - u(2dx)) / (2 dx)
```

## 7. Flags

The formula is algebraically second-order for distinct nodes, but it is
poorly conditioned for very small `q_w` because the first fluid sample
nearly coincides with the wall value. Implementations should avoid
using it at `q_w = 0` and should treat extremely small `q_w` as a
conditioning-sensitive case.
