# M55b FIX verdict

## Summary
GREEN with backend caveat: the Taylor `u2` goalpost was replaced by a bilinear off-grid sample and the L3 frozen-U cylinder discriminator is back near baseline on the available CPU path. L3 CPU produced `max_grad=7.10676228984448e-3`, `trace_C_max=101.5211761487642`, and `F_poly_max=1.8060940245903171e-4`, all inside the M55b gates. The requested Metal run could not execute locally because Metal exposed zero `MTLDevice` instances; rerun G3 on a Metal-capable host if Boss requires backend identity.

## Design changes
- Bilinear `u2` access: for valid q-wall links, the helper samples `u2` at `cell_center + H*n_wall` with a 4-cell bilinear stencil. Links whose bilinear stencil exits the array or touches any solid cell are invalid and are skipped; cells with no valid q-aware link fall back to `_fvfd_apply_embedded_wall_gradient_2d`.
- q_w_min: raised from `0.01` to `0.1`, with an inline comment documenting the singular M55 denominator.
- Link spacing: `H = hypot(cx*dx, cy*dy)` is used for the second normal sample and denominator. Because the M55 equation is wall-normal, the first-sample distance is the projected normal distance `d = q_w*H*cos(theta)`; best-aligned D2Q9 links are used for the scalar normal-derivative LSQ average.
- Dropped previous Taylor/Hessian reconstruction, mixed-derivative helper use, full 2D link-direction LSQ, and 30/70 best-pair blend. Synthetic q-wall fixtures without adjacent solid mask cells keep the first-order embedded fallback.

## G1 FVFD tests
- Result: `953/953`
- Interpretation: FVFD operator tests pass with the required signature.

## G2 M53a canary
- Embedded mean abs_err: `6.1308673105335883e-3`; max abs_err: `2.1551510596578116e-2`
- Current default path in the same run: mean `7.1109500066542444e-2`; max `1.4125809929537925e-1`
- Verdict: PASS (`mean < 1e-2`).

## G3 L3 frozen-U cylinder
- Metal note: `KRAKEN_BACKEND=metal` failed before metrics because the local Metal package saw zero `MTLDevice` instances.
- CPU result: `max_grad=7.10676228984448e-3` vs baseline `6.5e-3` (ratio `1.09x`)
- `trace_C_max=101.5211761487642` vs baseline `100.74` (ratio `1.01x`)
- `F_poly_max=1.8060940245903171e-4` at `(142,25)`; vs context baseline `1.36e-4` (ratio `1.33x`)
- Verdict: PASS (`max_grad < 0.013`, trace and force inside 2x gates) on CPU; Metal still needs a capable host.

## G4 patch ladder
- Result: `18213/18213`
- Interpretation: Log-FV patch ladder is unchanged after preserving first-order behavior for synthetic q-wall/no-solid fixtures.

## Recommendation
Ship the M55b helper fix as the active branch result, with the explicit note that local Metal validation was blocked by environment rather than by the code path. Next step: rerun G3 on a Metal-capable machine if needed, then run the M48 R=30 smoke for 10k steps.
