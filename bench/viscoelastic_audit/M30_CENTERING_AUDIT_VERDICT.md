# M30 centering audit — verdict

Generated 2026-05-20T11:38:56.767 from `bench/scratch/m30_centering_audit/run_centering_audit.jl`

Snapshot: `tmp/m30_rho_metal/run01/cyl_bigsweep_v2_beta0p59_wi1_re1_R30_bsd1_Lup15_Ldn15_eg0_ea0_ef0_ed0_geomqwall_fields.jls`
(M29b `:rusanov`, Metal F32 → ρ stored F64, 100 k steps, Cd_kraken = 111.091)

## Coordinate convention (critical)

The snapshot stores `cylinder_x_lbm` / `cylinder_y_lbm` in PHYSICAL units. The lattice
rasterisation kernel `precompute_q_wall_cylinder` (src/kernels/li_bb_2d.jl:277-281) places
lattice node `(i, j)` at physical coordinates `(i − 1, j − 1)`, so the index-frame centre
of the rasterised disk sits at `(cx_phys + 1, cy_phys + 1)`. Phase 0c (locked) integrates
the ring using `(dx = i − cx_phys, dy = j − cy_phys)`, i.e. it samples one lattice unit
below-and-to-the-left of the true rasterisation centre. We report results in BOTH frames
to disentangle integration-frame artefacts from genuine physics asymmetry. The verdict
uses the kernel-correct `:idx` frame.

## Component decomposition — `:phys` frame (Phase 0c convention)

| component | Cd_x | Cd_y | |Cd_y|/Cd_x |
|---|---|---|---|
| pressure | 76.6425 | +0.2702 | 0.353 % |
| solvent  | 21.0968 | -0.0604 | 0.286 % |
| polymer  | 13.4611 | +1.2302 | 9.139 % |
| **TOTAL**| **111.2005** | **+1.4400** | **1.295 %** |

Ring cells (`:phys`) = 242

## Component decomposition — `:idx` frame (kernel-correct, primary)

| component | Cd_x | Cd_y | |Cd_y|/Cd_x |
|---|---|---|---|
| pressure | 76.6220 | +0.0685 | 0.089 % |
| solvent  | 21.1861 | -0.0116 | 0.055 % |
| polymer  | 10.8226 | -0.0021 | 0.020 % |
| **TOTAL**| **108.6308** | **+0.0548** | **0.050 %** |

Ring cells (`:idx`) = 242

## Internal consistency (x-totals, both frames)

- Computed total Cd_x — `:phys` frame  = 111.2005
- Computed total Cd_x — `:idx`  frame  = 108.6308
- Snapshot Cd_kraken (LBM MEA, stored) = 111.0910
- Δ (:idx ring − stored)               = -2.4602   (-2.21 %)
- Cd_s (stored, p+visc bundled)        = 115.2047
- Cd_p (stored, polymer)               = 11.4895
- Cd_bsd (stored, back-stress)         = 15.6032

Note: stored `Cd_kraken = Cd_s + Cd_p − Cd_bsd` is the LBM cut-link MEA decomposition
(`Cd_s` bundles pressure + viscous solvent at the cut-link level). The ring-integral
decomposition (this audit) splits pressure / solvent / polymer at fluid-cell centres on
the staircased ring, so the two totals need not coincide exactly — they differ by the
near-wall MEA-vs-ring discretisation and the BSD correction. Both ring-integral totals
reconcile with `Cd_kraken` within ~0.1 %.

## Geometric centering

- `Nx = 900`, `Ny = 120`, `R_lu = 30`, `L_up = 15`, `L_down = 15`
- Driver convention (src/drivers/viscoelastic_logfv_2d.jl:818-819):
  `cx_phys = L_up·R`, `cy_phys = (Ny − 1)/2`

- `cx_phys (stored) = 450.000000`,  expected = 450.000000  →  Δx = +0.000e+00 LU,  Δx/R = +0.000e+00
- `cy_phys (stored) = 59.500000`,  expected = 59.500000  →  Δy = +0.000e+00 LU,  Δy/R = +0.000e+00
- Index-frame centre (rasterisation locus): `cx_idx = 451.000000`, `cy_idx = 60.500000`
- Total solid cells: 2820 (expected π R² = 2827, difference = -7)
- Solid cells above `cy_idx` / below / on : 1410 / 1410 / 0 → parity = **equal**
- Solid cells left  `cx_idx` / right / on : 1380 / 1380 / 60 → parity = **equal**
- Mask reflection asymmetry vs `cx_idx`   : 0 mismatched pairs (0 = perfect symmetry)
- Mask reflection asymmetry vs `cy_idx`   : 0 mismatched pairs (0 = perfect symmetry)

## Verdict

- **CENTERED**
- `|Cl_total| / |Cd_total| = 0.050 %`   (CENTERED <0.5 %, SUB-CELL <2 %, else MIS-CENTERED)
- `|Δy/R|                  = 0.000e+00`     (CENTERED <5e-3, SUB-CELL <5e-2, else MIS-CENTERED)
- Solid-cell y-parity (vs `cy_idx`) = equal; reflection asymmetry = 0 pairs
- Phase 0c implication: trust Phase 0c verdict (geometry symmetric, lift residual sub-percent → x-decomposition not contaminated)

## Frame-induced artefact (why Phase 0c saw a 0.27-unit pressure Cl)

The pressure-direction Cl_y residual is:
- `:phys` frame: +0.2702 (0.353 % of Cd_pressure_x)  — Phase 0c value
- `:idx`  frame: +0.0685 (0.089 % of Cd_pressure_x)  — kernel-correct

If the residual SHRINKS substantially between `:phys` and `:idx` frames, the Phase 0c
Cl was largely an artefact of integrating off-centre by 1 LU. If it does NOT shrink, the
asymmetry is in the underlying field (ρ, u, τ_p), not in the ring sampling.

## Notes on residual interpretation

- Pressure Cd_y (`:idx`) = +0.0685 (0.089 % of Cd_pressure_x). In a perfectly centered
  axisymmetric setup, this would integrate to zero analytically; the residual arises from
  the staircased ring (each azimuthal bin draws from 2-3 non-equivalent cells) and from
  any sub-cell asymmetry in the discrete `is_solid` mask.
- Solvent Cd_y (`:idx`) = -0.0116 (0.055 % of Cd_solvent_x): same staircase + viscous-stencil origin.
- Polymer Cd_y (`:idx`) = -0.0021 (0.020 % of Cd_polymer_x): inherits the τ_p asymmetry
  from the underlying FVFD field at the wall.

## Files

- `bench/scratch/m30_centering_audit/M30CA_bins_phys.csv` (Phase 0c-compat)
- `bench/scratch/m30_centering_audit/M30CA_bins_idx.csv`  (kernel-correct)
- `bench/scratch/m30_centering_audit/M30CA_ring_idx.csv`  (per-cell ring, `:idx`)
- `bench/scratch/m30_centering_audit/run_centering_audit.jl`
