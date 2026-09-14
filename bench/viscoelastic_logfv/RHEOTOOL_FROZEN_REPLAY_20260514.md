# RheoTool Frozen Replay — 2026-05-14

## Purpose

This diagnostic stops the embedded-slice ratchet and replays a converged
RheoTool velocity field through Kraken's frozen log-FV polymer pipeline.

Input field used here:

- temporary local RheoTool rerun: `/private/tmp/kraken_rheotool_replay_case`
- time: `0.8`
- drag at `t=0.8`: `Cd = 130.428494971`
- preserved replay input copy:
  `tmp/rheotool_frozen_replay_20260514/rheotool_input_0.8/`

The replay output is in:

```text
tmp/rheotool_frozen_replay_20260514/
```

Key files:

- `summary.csv`
- `dashboard.html`
- `fields_R{10,20,30}.jls`
- `profiles_R{10,20,30}.csv`

## Replay Setup

- RheoTool fields parsed from OpenFOAM ASCII `U` and `tau`.
- OpenFOAM cell centers reconstructed from `polyMesh`.
- Fields resampled to Kraken Cartesian grids with local affine weighted
  interpolation.
- Domain: `x in [-5, 15]`, `y in [-2, 2]`.
- Resolutions: `R = 10, 20, 30`.
- Frozen pipeline:
  `U -> embedded face velocities -> log-C advection -> log-C source -> tau_p -> div(tau_p) -> embedded wall traction`.
- No LBM population update and no momentum feedback.

## Results

| R | tau_xx rel L∞ | tau_xy rel L∞ | tau_yy rel L∞ | Cd polymer | Cd ref traction | Cd error |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 3.2422e+00 | 1.1851e+00 | 8.0453e-01 | 0.7889779 | 5.2914694 | -4.5025e+00 |
| 20 | 5.4114e+04 | 2.1421e+04 | 5.5457e+03 | 0.96102686 | 5.4376194 | -4.4766e+00 |
| 30 | 9.6434e+10 | 1.2755e+11 | 3.2389e+10 | NaN | 6.7347287 | NaN |

Control:

- `KRAKEN_REPLAY_INITIAL=rheotool`, `KRAKEN_REPLAY_PHYSICAL_TIME=0`, `R=10`
  gives `Cd_polymer == Cd_ref_traction == 5.291469387359019`.
- That control found `12` non-SPD cells after interpolating RheoTool `tau` to
  `C = I + tau / ((1-beta)/Wi)`, so raw tau interpolation near the wall is not
  perfectly SPD-preserving. It does not explain the main replay blow-up because
  the main run starts from identity.

## Verdict

The tau fields do not match within 1%, and the disagreement grows
catastrophically with refinement. R=30 goes nonfinite during frozen replay.

Decision-tree branch:

```text
tau fields disagree near the wall / with refinement
=> bug is in the polymer CDE pipeline on cut cells
=> isolate advection vs source vs wall closure under frozen U
```

This result does **not** support continuing with another embedded-coverage
slice or another Aqua R-sweep. The next concrete action is a frozen-U toggle
matrix:

1. Replay with RheoTool `U`, but disable log-C advection and apply only the
   source step from Kraken's numerical embedded gradient.
2. Replay with advection enabled but source disabled from RheoTool-initialized
   `Psi`.
3. Replay source-only using an externally differentiated RheoTool gradient,
   then compare against Kraken's embedded-gradient source.

The fastest discriminator is item 1: it tests whether the numerical
`grad(U)`/source step alone can preserve a sane Oldroyd-B response on the
curved cut-cell band.
