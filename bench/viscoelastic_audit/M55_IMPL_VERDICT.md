# M55 IMPL verdict

## Summary

GREEN. The q_w-aware embedded velocity-gradient path now passes the
M52b/M53a cylinder-adjacent canary with embedded mean abs_err
`0.00083178041370409275` and max abs_err `0.0049172788096850661`,
below the M55 targets of `< 1e-3` and `< 5e-3`, with M49, FVFD
operators, and the Log-FV patch ladder still green.

## Design choices

- `u2` access strategy: store included q-wall link values in
  `FVFDEmbeddedBoundary2D.wall_q`, then reconstruct `u2` in-kernel with
  a local quadratic Taylor estimate from the cell value, seed gradient,
  and fixed-stencil Hessian. Rationale: direct bilinear `u2` was not
  exact enough for the quadratic C1 canary, while the Taylor access keeps
  the helper fixed-stencil and allocation-free.
- `q_w_min`: `0.01`. The canary q_w range is `0.21611781858498924` to
  `0.8542486889354093`, so trial values `0.01`, `0.05`, and `0.1` all
  exercise the same q-aware branch; the smallest value is retained.
- Fallback below `q_w_min`: the existing embedded first-order wall-normal
  projection via `_fvfd_apply_embedded_wall_gradient_2d`. The same fallback
  is used for q-wall-less embedded geometries and degenerate local seed
  gradients.

## Validation

- M52b/M53a canary:
  - default mean abs_err: `0.071109500066542444`
  - default max abs_err: `0.14125809929537925`
  - embedded mean abs_err: `0.00083178041370409275`
  - embedded max abs_err: `0.0049172788096850661`
  - corr(q_w, abs_err): `-0.31416492343515962`
  - improvement factor: `85.490712326197865`
  - wall time: `1.383934 s`
- M49 halfway wall stencil: PASS; helper quadratic P1-P3 max abs_error
  `4.9737991503207013e-14`.
- FVFD operators 2D: PASS `953/953`.
- Log-FV patch ladder: PASS `18213/18213`.
- Strict awk exit gate: PASS.
- `git diff --check`: PASS.

## What next

Phase A.3 can proceed to the Metal R-sweep. The new helper is still a
local cut-link gradient closure, not a driver-side default change; keep
the M53d wall-position vs cell-center consumer boundary intact when
promoting this into coupled cylinder runs.
