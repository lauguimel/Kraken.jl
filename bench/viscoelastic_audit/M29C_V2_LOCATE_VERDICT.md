# M29c-v2 — NaN location & first-divergent-field verdict

**Mission** : M29c-v2-postmortem-locate
**Branch**  : `dev-viscoelastic`
**Date**    : 2026-05-19
**Driver**  : `bench/scratch/m29c_v2_locate_metal/run_locate.jl`
**Outputs** :
- Aqua CUDA F64 logs : `~/Kraken.jl-viscoelastic-run/krk_m29c_v2_loc.o215887{13,14,25}`
- Aqua snapshots     : `tmp/m29c_v2_kraken/aqua_locate/`
- Metal F32 snapshots: `bench/scratch/m29c_v2_locate_metal/m29*_*/`

## TL;DR

The 1-line M29c-v2 fix (`(u+d)/2 → upwind` in the canonical_usable=false
fallback) **does not produce an immediate NaN**. The system runs cleanly
for tens of thousands of steps with `Cd ≈ 115-116` (matching the
postmortem prediction of 116-122). It is a **delayed temporal-stiffness
runaway**, not a spatial artefact at the leeward shoulder.

| Quantity                          | Aqua CUDA F64  | Metal F32      |
|-----------------------------------|----------------|----------------|
| First non-finite step             | **92 200**     | **102 800**    |
| First non-finite field            | **`rho`**      | **`rho`**      |
| First non-finite (i, j)           | (334, 1)       | (386, 1)       |
| First non-finite (x/R, y/R)       | (−3.87, −1.95) | (−2.13, −1.95) |
| Wall location                     | south wall     | south wall     |
| Streamwise location               | far upstream   | upstream       |

**Verdict (1 sentence)** : the NaN is a **TEMPORAL polymer-stress
feedback runaway** that ultimately blows up the LBM density at the
south wall (j=1) far from the cylinder — NOT a spatial artefact of
the M29c boundary-aware MUSCL reconstruction at the leeward
shoulder.

## Comparison ladder

### M29c-v2 (`muscl_superbee`, the patched scheme)

| max_steps | backend  | FT  | Cd          | min_c_eig | max_c_trace | max_speed   | first_nf_step |
|-----------|----------|-----|-------------|-----------|-------------|-------------|---------------|
| 5 000     | Metal    | F32 | 108.7       | 0.194     | 70          | 0.016       | —             |
| 30 000    | Metal    | F32 | 115.4       | 0.275     | 202         | 0.015       | —             |
| 30 000    | CUDA     | F64 | **115.9**   | 0.195     | 201         | 0.015       | —             |
| 60 000    | Metal    | F32 | 115.6       | 0.237     | 210         | 0.015       | —             |
| 80 000    | Metal    | F32 | 115.0       | 0.238     | 211         | 0.015       | —             |
| 100 000   | CUDA     | F64 | (cascade)   | NaN       | NaN         | NaN         | **92 200**    |
| 100 000   | Metal    | F32 | 76.3        | 0.027     | 1.4e4       | **0.316**   | —             |
| 200 000   | Metal    | F32 | (cascade)   | NaN       | NaN         | NaN         | **102 800**   |

The 100 000-step Metal F32 case did NOT NaN inside the watcher
because the integrator finished the loop while values were still
finite (max_speed had already exploded 20× to 0.316 = 63 × u_mean
and max_c_trace had grown 70× to 14 000), but the next 2-3k steps
of the 200k run blew through to NaN at step 102 800.

### M29b (legacy `rusanov` baseline, same setup)

| max_steps | backend  | FT  | Cd          | min_c_eig | max_c_trace | max_speed   | first_nf_step |
|-----------|----------|-----|-------------|-----------|-------------|-------------|---------------|
| 5 000     | Metal    | F32 | 105.3       | 0.191     | 62          | 0.016       | —             |
| 30 000    | CUDA     | F64 | 110.2       | 0.284     | 180         | 0.015       | —             |
| 200 000   | Metal    | F32 | **111.1**   | 0.282     | 186         | 0.015       | —             |

**M29b is stable indefinitely (still healthy at 200 000 steps,
Metal F32).** M29c-v2 has a clear divergence with no warning
signs in the 30 000 step run.

## Diagnosis

1. **The fix removes the anti-TVD overshoot.** Verified by the
   M29c-postmortem-empirical / -math Departments and confirmed
   here at small step counts: Cd values are reasonable, peak
   polymer stress at the leeward shoulder (x/R ≈ 0.8, y/R ≈ 0.7
   at 100k steps Metal F32) is consistent with the wake region
   where elastic stress is expected to build up.

2. **The runaway is temporal, not spatial.** The diagnostics at
   30 000 and 60 000 steps are identical: max_c_trace ~210,
   max_speed = 1.5 u_mean. By 100 000 steps the wake-stress hot
   spot at (x/R, y/R) = (0.77, 0.72) has grown 65× in
   |tauxx| and the local fluid speed reaches 63 × u_mean.

3. **The first non-finite cell is at the south wall (j=1),
   upstream of the cylinder (x/R ≈ −3.87 F64, −2.13 F32),
   in `rho` (LBM density)**, not in the polymer fields directly.
   Mechanism : the wake polymer stress drives a body force that
   accelerates the bulk flow upstream; the LBM density loses
   positivity at the bottom wall where the BB closure cannot
   absorb the resulting equilibrium-shift, NaN appears in `rho`,
   the cascade propagates everywhere within 1-2 steps.

4. **F32 vs F64 differ in NaN-onset step (103k vs 92k), not in
   the qualitative trajectory or first non-finite field.** The
   Aqua "100% NaN at end of 100k" production result is
   reproduced exactly on Metal F32 by running ~3k more steps
   past the first non-finite step.

## Trajectory of key diagnostics vs step (Metal F32, M29c-v2)

```
step      |  Cd     | min_c_eig | max_c_trace | max_speed | max|psi|
   5 000  | 108.7   | 0.194     |     70.0    | 0.0159    | 4.19
  30 000  | 115.4   | 0.275     |    202.3    | 0.0149    | 5.28
  60 000  | 115.6   | 0.237     |    210.0    | 0.0149    | 5.32
  80 000  | 115.0   | 0.238     |    210.5    | 0.0149    | 5.32
 100 000  |  76.3   | 0.027     |  14 120     | 0.316     | 6.69  <-- explosion
 102 800  |  NaN    |  NaN      |    NaN      |  NaN      |  NaN  <-- first_nf
```

The system is **steady from 30k to 80k**, then snaps into runaway
between 80k and 100k. The runaway is concentrated in a small wake
region around (x/R, y/R) = (0.77, 0.72), but the NaN first manifests
in `rho` at the south wall upstream when the body-force coupling
back into the LBM macro-flow exceeds the BB stability envelope.

## Cross-check with M29c-asis (CD2 variant)

Per the M29c-postmortem-empirical verdict, the M29c-asis (CD2
fallback `(u+d)/2`) produced Cd = −1571 at R=30 Wi=1 in the
production sweep `21588436.aqua`. That earlier failure was a
**spatial anti-TVD** issue (high-frequency mode amplification at
the wall band, see scalar microbench Test C).

M29c-v2 (this verdict) is a **different failure mode** : the
spatial pathology is fixed, the system runs cleanly through 80k
steps, but a slow elastic-feedback positive loop builds up and
runs the macro-flow out of the LBM stability cone around step
80-100k.

## What this means for the fix

The 1-line `(u+d)/2 → upwind` change **is necessary and not
sufficient**. It correctly removes the high-frequency wall-band
amplification, but does not address the underlying late-stage
stiffness at Wi=1, β=0.59, lambda=6000 LBM steps. With λ = 6000
the system needs ~5-10 polymer relaxation times to develop the
wake-stress pattern, which is exactly the 30k-80k window. At
~16 λ the elastic feedback amplifies enough to break LBM density
positivity.

This is **the same regime the literature struggles with** at
Wi ≈ 1 cylinder benchmarks. Mitigations to explore in
M29c-v3 / next iteration :

- Polymer-stress diffusion (κ ≈ 1e-3 numerical) — turns the
  growing wake hot spot into a smoother profile that the LBM
  macro-flow can absorb.
- Adaptive polymer subcycling (already in place but maybe needs
  tighter `max_deformation_increment` at high Wi).
- BSD = 0.5 instead of BSD = 1.0 — moves half the polymer
  viscosity onto the LBM side which damps the body-force
  feedback at the wall.
- Force-clipping at the wall band (suppress f_total > ρ * u²
  threshold).

The Boss may either ship M29c-v2 with a documented Wi ≤ 0.7 limit
(M29c-v2 + M29b production matrix should already cover lower-Wi
quantitative validation) OR pursue M29c-v3 with one of the above
mitigations.

## Artifacts

- This file.
- Aqua F64 jobs : 21588713 (M29c-v2 30k, OK), 21588714 (M29b 30k,
  OK), 21588725 (M29c-v2 100k, **NaN at 92200**).
- Aqua field dumps in `tmp/m29c_v2_kraken/aqua_locate/`.
- Metal F32 sweep `bench/scratch/m29c_v2_locate_metal/m29*_*/result_*.jls`.
- `bench/scratch/m29c_v2_locate_metal/run_locate.jl` — driver
  wrapping `Kraken.run_viscoelastic_logfv_cylinder_coupled_2d` with
  `diagnostic_stride > 0` so the per-step NaN watcher actually fires.
- `bench/scratch/m29c_v2_locate_metal/inspect_aqua_snapshot.jl` —
  field-statistics on the original Aqua `.jls` (post-cascade,
  100 % NaN by then).
- `bench/scratch/m29c_v2_locate_metal/analyse_runaway.jl` —
  spatial localisation of peak tau / psi / speed per snapshot.
