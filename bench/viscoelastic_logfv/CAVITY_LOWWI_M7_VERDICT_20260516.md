# Cavity low-Wi sanity — M7 verdict (2026-05-16)

## TL;DR

**M7 as run is INCONCLUSIVE due to a design flaw in the Boss brief:**
the `polymer_on` and `nu_p_zero` cases have different total LBM
viscosities (`ν_total = ν_s + ν_p = 0.2` vs `0.1`), hence different
`Re_LU` (1.6 vs 3.2). The 3.4 % Kraken-vs-Kraken centerline L2 delta
is therefore confounded — it is plausibly explained by the Re factor
change alone, not necessarily by a polymer-coupling bug. **A follow-up
M7b with matched effective viscosity is required.**

## Setup

- Branch: `dev-viscoelastic` HEAD `9d66b2c2`.
- Aqua job `21405281.aqua`, walltime 00:04:19 (much shorter than the
  ~2h estimate — at `lambda_phys=0.001` the polymer substep cap
  binds quickly and the cases run fast).
- PBS: `bench/viscoelastic_logfv/run_cavity_lowwi_sanity.pbs`.
- Common parameters: `N=64, end_time=8.0, u_max=0.005,
  bsd_fraction=0.75, lambda_phys=0.001 (Wi≈0.001)`.

## Observable

Kraken-vs-Kraken comparison of the centerline `u(0.5, y)` profile
(extracted directly from the two output CSVs, sidestepping the
rheoTool reference which is at `lambda=1.0` and meaningless at
`Wi=0.001`):

| Case | ν_s | ν_p | ν_total | Re_LU | centerline u L2 norm |
|------|-----|-----|---------|-------|----------------------|
| polymer_on | 0.1 | 0.1 | 0.2 | 1.6 | 3.10 |
| nu_p_zero | 0.1 | 0.0 | **0.1** | **3.2** | 3.20 |

`rel_L2(polymer_on − nu_p_zero) / ‖polymer_on‖ = 3.41 %`
`max |polymer_on − nu_p_zero| = 4.22e-2`

(For the record, the relative L2 vs the rheoTool De=1 reference is
33.0 % and 36.7 % respectively for the two cases; these numbers
reflect the physics mismatch between Wi≈0.001 and the Wi=1 reference,
not a Kraken defect.)

## Design flaw and proposed M7b

The original M7 brief simply zeroed `nu_p` to disable the polymer,
without compensating in `ν_s`. The resulting cases have different
effective viscosities and therefore different Re_LU. The 3.4 % delta
between them is therefore not a clean signal of "polymer coupling at
Wi=0".

**M7b** (recommended) should run three cases at `lambda_phys=0.001`:
- A: `ν_s=0.1, ν_p=0.1` (current polymer_on, Re_LU=1.6)
- B: `ν_s=0.2, ν_p=0.0` (matched total viscosity, Re_LU=1.6, no
  polymer code active)
- C: `ν_s=0.1, ν_p=0.0` (current nu_p_zero, Re_LU=3.2 — kept as
  Re-doubling reference)

Decision rule:
- If `‖A − B‖ ≈ 0`: polymer machinery is silent at Wi=0, the cavity
  gap is NOT a Wi-independent coupling bug. We need to look at
  finite-Wi-specific terms.
- If `‖A − B‖ ≫ noise`: polymer machinery does something even at
  Wi=0; the coupling has a Wi-independent bug. Smoking gun.
- The A-vs-C delta should be similar to the current 3.4 % (purely
  Re effect) — sanity check that the new B case is correctly set up.

## What this means for the broader mission

M7 has not advanced the diagnostic. M8's ratchet of the polymer
pipeline still stands; the bug is still locus'd to the LBM ↔
polymer coupling. M7b needs to be run to either confirm or refute
the "Wi-independent coupling bug" hypothesis.

Meanwhile M9 is in flight with a first data point:
- N=32: L2_u = 31.4 %, L2_psi = 35.2 %
- N=64 (M1 baseline): L2_u = 18.0 %, L2_psi = 24.4 %

The 45 % drop from N=32 to N=64 is consistent with the "partial
discretization-floor" hypothesis; if the trend continues
(first-order in 1/N), N=128 may land near 10 %, and the residual
gap above any rheoTool-discretization floor would be smaller than
the headline 18 %. M9 N=96 and N=128 will refine this estimate.
