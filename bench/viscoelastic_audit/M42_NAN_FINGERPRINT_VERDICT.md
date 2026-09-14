# M42 NaN FINGERPRINT verdict

Date: 2026-05-24
Branch: dev-viscoelastic
Mission: M42-NAN-FINGERPRINT

Static audit only. Inputs were the four serialized M42 G5 v3 field
snapshots under `tmp/m42_g5_v3_results/`; no Julia simulation was
rerun. The R=30 Wi=0.1 reference snapshot was used as a loader sanity
check and had zero NaNs in `rho`, `ux`, `uy`, `tauxx`, `tauxy`, and
`tauyy`.

Scratch artefacts:
- `bench/scratch/m42_nan_fingerprint/00_plan.md`
- `bench/scratch/m42_nan_fingerprint/01_nan_counts.jl`
- `bench/scratch/m42_nan_fingerprint/02_spatial_fingerprint.jl`
- `bench/scratch/m42_nan_fingerprint/nan_counts.csv`
- `bench/scratch/m42_nan_fingerprint/spatial_fingerprint.csv`

Schema note: the snapshots do not contain `psi_xx` or velocity-gradient
buffers. Per mission instructions, `tauxx` is used as the `psi_xx` proxy,
and first-field inference is best-effort from post-run relative NaN
counts.

## Case R=30, Wi=1

### (a) NaN fraction

count(isnan, rho) / length(rho) = 0.973888888889
count(isnan, psi_xx proxy=tauxx) / length(psi_xx proxy=tauxx) = 0.973888888889
count(isnan, tau_xx) / length(tau_xx) = 0.973888888889
count(isnan, ux) / length(ux) = 0.973888888889
count(isnan, uy) / length(uy) = 0.973888888889
count(isnan, tauxy) / length(tauxy) = 0.973888888889
count(isnan, tauyy) / length(tauyy) = 0.973888888889

Fluid-cell check: 105180 / 105180 fluid cells are NaN in every inspected
field. The 2820 non-NaN cells are the solid mask.

### (b) Spatial classification

- Uniform sheet: >= 90 % NaN over all cells and 100 % NaN over fluid
  cells. The NaN field spans theta = -179.936 to +179.936 deg, all 36
  ten-degree azimuth bins are occupied, and r-R spans 0.004166 to
  423.917 LU. The front-shoulder arc window contains 246 cells only
  because the entire fluid domain is NaN, not because the pattern is
  localized.

### (c) First-field-to-NaN (best-effort from post-run snapshot)

- The snapshot has no `dudx/dudy` gradient buffers. `rho`, `ux`, `uy`,
  `tauxx`, `tauxy`, and `tauyy` all have identical NaN counts and are
  saturated over all fluid cells, so the post-run dump cannot identify
  a strict first field. The equal all-field saturation supports a
  catastrophic envelope / BC-overbounce propagation pattern rather than
  a still-localized polymer constitutive arc.

### (d) Fingerprint classification verdict

**FINGERPRINT**: uniform-BC-overbounce

### (e) Confidence (low / medium / high) and 1-sentence rationale

High. The stored NaNs occupy every fluid cell and every azimuth bin,
which matches the uniform catastrophic family and contradicts a
localized bilateral front-shoulder arc.

## Case R=60, Wi=0.1

### (a) NaN fraction

count(isnan, rho) / length(rho) = 0.973805555556
count(isnan, psi_xx proxy=tauxx) / length(psi_xx proxy=tauxx) = 0.973805555556
count(isnan, tau_xx) / length(tau_xx) = 0.973805555556
count(isnan, ux) / length(ux) = 0.973805555556
count(isnan, uy) / length(uy) = 0.973805555556
count(isnan, tauxy) / length(tauxy) = 0.973805555556
count(isnan, tauyy) / length(tauyy) = 0.973805555556

Fluid-cell check: 420684 / 420684 fluid cells are NaN in every inspected
field. The 11316 non-NaN cells are the solid mask.

### (b) Spatial classification

- Uniform sheet: >= 90 % NaN over all cells and 100 % NaN over fluid
  cells. The NaN field spans theta = -179.968 to +179.968 deg, all 36
  ten-degree azimuth bins are occupied, and r-R spans 0.002083 to
  847.899 LU. The front-shoulder arc window contains 466 cells only
  because the entire fluid domain is NaN.

### (c) First-field-to-NaN (best-effort from post-run snapshot)

- The snapshot has no `dudx/dudy` gradient buffers. `rho`, velocity, and
  tau fields have identical NaN counts, with no tau-vs-rho separation
  left in the post-run state. The best-effort inference is catastrophic
  all-field propagation consistent with BC over-bounce; the dump is too
  saturated to distinguish whether `rho` or constitutive stress became
  nonfinite first.

### (d) Fingerprint classification verdict

**FINGERPRINT**: uniform-BC-overbounce

### (e) Confidence (low / medium / high) and 1-sentence rationale

High. The low-Wi R=60 failure is domain-wide over the full fluid mask,
not a near-cylinder bilateral arc in r-R = 0..7 LU.

## Case R=60, Wi=1

### (a) NaN fraction

count(isnan, rho) / length(rho) = 0.973805555556
count(isnan, psi_xx proxy=tauxx) / length(psi_xx proxy=tauxx) = 0.973805555556
count(isnan, tau_xx) / length(tau_xx) = 0.973805555556
count(isnan, ux) / length(ux) = 0.973805555556
count(isnan, uy) / length(uy) = 0.973805555556
count(isnan, tauxy) / length(tauxy) = 0.973805555556
count(isnan, tauyy) / length(tauyy) = 0.973805555556

Fluid-cell check: 420684 / 420684 fluid cells are NaN in every inspected
field. The 11316 non-NaN cells are the solid mask.

### (b) Spatial classification

- Uniform sheet: >= 90 % NaN over all cells and 100 % NaN over fluid
  cells. The NaN field spans theta = -179.968 to +179.968 deg, all 36
  ten-degree azimuth bins are occupied, and r-R spans 0.002083 to
  847.899 LU. The front-shoulder arc window contains 466 cells only
  because the entire fluid domain is NaN.

### (c) First-field-to-NaN (best-effort from post-run snapshot)

- The snapshot has no `dudx/dudy` gradient buffers. All inspected fields
  have identical NaN counts and are saturated across all fluid cells.
  Compared with the D2bis polymer-arc signature, the saved state has no
  surviving localized tau/rho subset relationship; first-field ordering
  is therefore undetermined from this post-run dump.

### (d) Fingerprint classification verdict

**FINGERPRINT**: uniform-BC-overbounce

### (e) Confidence (low / medium / high) and 1-sentence rationale

High. Although this high-Wi/R case is physically capable of the D2bis
polymer-arc mechanism, the persisted M42 G5 v3 fingerprint is a complete
fluid-domain NaN sheet rather than bilateral front-shoulder arcs.

## Synthesis & M42-v2 recommendation

All 3 / 3 NaN cases classify as `uniform-BC-overbounce` by the mission
rubric: each failed snapshot has > 97 % total-cell NaNs and exactly
100 % fluid-cell NaNs in `rho`, velocity, and tau fields. This is the
catastrophic uniform-sheet family, not the localized polymer-arc family
that would support a narrower relaxation band.

- Majority = uniform-BC-overbounce, so M42-v2 should NOT be
  narrower-band. Recommend re-investigating the boundary spec and the
  pass-2 overwrite interaction; the Boss may revert to the halfwayBB
  pass-2 branch while isolating the unintended double-BC interaction.

**Decision call**: M42-v2 = revert

**Rationale**: The branch decision should follow the majority
fingerprint, and the majority is unanimous. A narrower relaxation band
targets a localized polymer-coupled instability margin, but these
post-run snapshots show no localized shoulder arcs at all: every fluid
cell is nonfinite in every persisted field. A one-sided minmod branch is
also less directly supported because there is no mixed population of
uniform and arc cases. The lowest-risk M42-v2 action is therefore to
step back to the known boundary behaviour, re-audit the relax pass-2
boundary interaction, and only then reintroduce a modified MUSCL
relaxation if the uniform sheet is eliminated.
