# M55 DERIV — 3-way adversarial concordance verdict

## Summary

CONCORDANT-HIGH. Three independent derivations (Boss, Codex, Claude
subagent), each forbidden from reading the others, produce the same
closed-form formula character-for-character (modulo formatting).

## Formula (agreed)

```
∂u/∂n|_wall = [(q_w+1)²·u₁ − q_w²·u₂ − (2q_w+1)·u_wall]
            / [q_w·(q_w+1)·dx]
```

where:
- `u_wall` = Dirichlet wall datum (e.g. 0 for no-slip)
- `u₁` = velocity at first fluid cell center, at distance `q_w·dx`
  along the wall outward normal
- `u₂` = velocity at second fluid sample, at distance `(q_w+1)·dx`
  along the wall outward normal
- `dx` = lattice spacing (assumed uniform; for `q_w·dx, q_w·dy` along
  arbitrary normals, this generalises to `q_w·h_n`)

## Sanity checks (all three derivations independently verified)

- **q_w = 1/2** (axis-aligned halfway-BB): reduces to M51 formula
  `(3·u₁ − u₂/3 − (8/3)·u_wall) / dx = (9·u₁ − u₂ − 8·u_wall) / (3·dx)` ✓
- **q_w = 1** (wall one cell from first fluid center): reduces to
  standard textbook 3-point forward FD `(4·u₁ − u₂ − 3·u_wall) / (2·dx)` ✓
- **constant field** (`u ≡ const`): numerator coefficient sum is zero
  `(q_w+1)² − q_w² − (2q_w+1) = 0`, so the formula returns 0 ✓

## Limiting behavior (q_w → 0)

Denominator vanishes linearly in `q_w`, numerator stays finite at
`u₁ − u_wall` ⇒ singularity scales as `1/q_w`. Both engineers
independently recommended a `q_w_min ∈ [0.05, 0.1]` floor with
fallback to first-order or halfway-BB for `q_w < q_w_min`.

## Concordance metrics

- Formula: 3/3 identical (Boss, Codex, Claude).
- Sanity check q_w=1/2: 3/3 confirmed reduces to M51.
- q_w → 0 hazard: 3/3 flagged the same way.
- q_w → 1 reduction to standard FD: 2/3 explicitly noted (Claude + Boss);
  Codex consistent but did not call it out.

## Independence audit

- Boss derivation: written in conversation, NOT visible to either
  engineer.
- Codex (M55_DERIV_codex.md): brief required no source-code reads,
  no sibling file reads. Reasoning effort medium. Wall time ~5 min.
- Claude (M55_DERIV_claude.md): same brief, separate worker process,
  cross-validated independence at runtime.

## Phase A.1 verdict

CLOSE. Formula validated for Phase A.2 implementation. No further
algebraic derivation needed. Move to design + implementation +
empirical canary.

## Phase A.2 follow-up requirements

Implementation must:
1. Use the agreed formula at cut-cells with `q_w ≥ q_w_min`.
2. Fall back to first-order or halfway-BB at `q_w < q_w_min`.
3. Empirically tune `q_w_min` on M53a canary (PT_cylinder_adjacent_stencil.jl).
4. Target M53a embedded mean abs_err < 1e-3, max < 5e-3.
5. Preserve M49 (axis-aligned canary), 953/953 FVFD tests, and
   18213/18213 patch ladder tests (no driver consumer changes — see
   M53d lesson on wall-position vs cell-center semantics).
6. Address the geometric challenge of accessing `u₂` at
   `(q_w+1)·dx` along the wall normal: for general cut-cells the
   sample point is off-grid and requires bilinear interpolation.
