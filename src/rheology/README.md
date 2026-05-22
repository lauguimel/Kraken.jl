# rheology/

Constitutive models — Newtonian (the default, baked into the BGK
relaxation time τ) plus Generalised Newtonian and viscoelastic
extensions. The non-Newtonian path replaces τ with a strain-rate-dependent
viscosity computed per-cell each timestep.

## Key entry points

| File | Symbol | Purpose |
|---|---|---|
| `models.jl` | `PowerLaw`, `Carreau`, `Bingham`, `Casson`, `OldroydB` | Constitutive law types + parameters |
| `viscosity.jl` | `viscosity(model, γ̇, …)` | Maps strain-rate magnitude → apparent viscosity |
| `strain_rate.jl` | `strain_rate_2d`, `strain_rate_3d` | Compute γ̇ tensor from off-equilibrium populations (Π neq) |
| `linalg.jl` | small symmetric matrix ops | Invariants, determinants, Cholesky for stress tensor work |

## Critical invariants

- **Positive viscosity**: every model must return η > 0 for all γ̇ ≥ 0.
  Newton's iteration on Bingham yield surface MUST converge to the
  yielded branch.
- **Strain-rate from Π_neq** (not from finite differences): the LBM
  off-equilibrium gives γ̇ directly with second-order accuracy, no extra
  stencil needed.
- **Viscoelastic**: the polymer stress tensor (τ_p) is advected by a
  SEPARATE field and coupled back into the momentum equation via the
  divergence in `kernels/viscoelastic_2d.jl`; this module owns only the
  constitutive update (`∂τ/∂t = ...`), not the streaming.
- **Mass / momentum conservation are unaffected** by the constitutive
  choice — the τ swap happens inside the collision operator only.

## Cross-module dependencies

Reads from: `lattice` (weights), `kernels` (called from
`collide_rheology_2d!` and `viscoelastic_2d.jl`).
Provides to: `kernels/collide_rheology_2d.jl`, `kernels/viscoelastic_2d.jl`,
ultimately `drivers/rheology.jl` and `drivers/viscoelastic.jl`.

## Status / scope notes

- v0.1.0 ships Newtonian only; GNF and viscoelastic ship on `dev/*`
  branches.
- Viscoelastic 2D cylinder Cd_s 5% err vs Liu 2025 reference; channel
  case validated; MEA double-counts τ_p (open issue, see project memory
  `project_viscoelastic_audit`).
