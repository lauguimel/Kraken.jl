# M45 — post-M44 mesh-Wi residual VERDICT

Date: 2026-05-25
Branch: dev-viscoelastic
Status: **UNDETERMINED, leaning mixed β/γ** — residual is NOT a Guo-half-step-class bug; M28-M42 cluster remains CLOSED.

---

## TL;DR

M44 sweep (`21827394.aqua`) showed Cd at Wi=1 β=0.59 decreases
monotonically with R (118.10 → 113.23 from R=30 → R=60). M45 ran
B (Boss-direct per-θ decomposition) and C (Codex adversarial audit
α/β/γ) in parallel.

**Combined verdict**:
- The residual is localized in **Cd_solvent × shoulder** (−1.08 from
  R=30→60) and **Cd_pressure × wake** (−1.61). NOT in Cd_polymer.
- The dominant M32-original bucket **Cd_pressure × front_pole** is
  **62 % closed** and CONTINUES converging with R toward the rT target
  (+1.81 from R=30→60).
- No Guo-half-step-class double-count found anywhere in pressure
  readouts or coupling timing (Codex γ audit was negative).
- rT reference R-coverage (α): **only R=30 published**, cannot
  discriminate.
- Most likely explanation: combined (β) lattice-distance to outlet
  pressure boundary growing 420→840 LU + TRT s_plus shift across R
  + (γ) default non-embedded FVFD polymer-force stencil at the q_wall
  cylinder. None is a M44-class bug.

**Decision**: ship M44 + M45 documentation as-is. Residual is a
research-grade open question, not a blocker.

---

## (a) B — Boss-direct per-θ decomposition (M32 P4 template)

`bench/scratch/m44_postfix_walldecomp/run_walldecomp_postfix.jl`
loaded the 4 post-fix .jls (R=30/40/50/60 Wi=1 β=0.59) and computed
the `:idx` frame Cd decomposition (36 azimuthal bins, 3-region split
front_pole / shoulder / wake).

### Mesh trend per (component, region)

| Component × Region | R=30 | R=60 | ΔR=30→60 | vs rT R=30 |
|---|---:|---:|---:|---:|
| Cd_pres × front_pole | 76.65 | **78.46** | **+1.81** ✓ converge | rT=79.86 |
| Cd_pres × shoulder | 19.30 | 17.99 | −1.31 | rT=20.07 |
| Cd_pres × wake | −14.19 | **−15.80** | **−1.61** ✗ diverge | rT=−14.14 |
| Cd_solv × front_pole | 1.93 | 2.06 | +0.13 | rT=1.56 |
| Cd_solv × shoulder | 17.16 | **16.08** | **−1.08** ✗ diverge | rT=17.12 |
| Cd_solv × wake | 0.97 | 1.07 | +0.09 | rT=1.18 |
| Cd_poly × front_pole | 0.25 | 0.77 | +0.52 ✓ converge | rT=0.95 |
| Cd_poly × shoulder | 11.99 | 11.86 | −0.13 | rT=11.65 |
| Cd_poly × wake | 1.65 | 1.10 | −0.55 | rT=0.84 |

### M44 fix closure at R=30 (vs pre-fix M32 P4 D1)

| Bucket | Pre-fix gap (M32) | Post-fix gap (M44+M45) | Closure |
|---|---:|---:|---:|
| Cd_pres × front_pole (the 80% bucket) | +8.34 | +3.21 | **62%** |
| Cd_pres × shoulder | +1.67 | +0.77 | 54% |
| Cd_pres × wake | −1.10 | −0.05 | ~100% |
| Cd_polymer × shoulder | +3.14 | −0.34 | over by 0.34 |
| Cd_polymer × wake | −1.05 | +0.81 | over by 0.81 |
| **Total** | **+10.37** | **+2.28** | **78%** |

**Key observations**:
- **Cd_polymer per-region is now nearly correct** at R=30 vs rT
  (gaps ≤ 1 Cd in absolute, with some sign flips around 0). The
  polymer-pressure coupling is fixed.
- The original M32 80%-bucket (Cd_pres × front_pole +8.34) is 62%
  closed (now +3.21) AND **continues converging with R** (+1.81 from
  R=30→60). Healthy mesh convergence.
- The new mesh-R residual is dominated by **Cd_solv × shoulder
  diverging** (−1.08 from R=30→60) and **Cd_pres × wake getting more
  negative** (−1.61). These are Newtonian-like channels (viscous
  shear + downstream pressure), NOT polymer-specific.

Artifacts:
- `bench/scratch/m44_postfix_walldecomp/M44P_kraken_R{30,40,50,60}_bins_idx.csv`
  (per-bin ring decomp, 36 bins each)
- `bench/scratch/m44_postfix_walldecomp/M44P_decomp_RvsR.csv`
  (aggregated 3-region per R)
- `bench/scratch/m44_postfix_walldecomp/run_walldecomp_postfix.jl`
  (driver)

---

## (b) C — Codex adversarial audit (α / β / γ)

Full report: `bench/viscoelastic_audit/M45_RESIDUAL_AUDIT_CODEX.md`.

### (α) rT reference R-coverage — **untestable**

`bench/rheotool/` only contains the R=30 case
(`cylinder_wi1.0_shrunk15R`, Cd.txt = 120.382717 at t=20). The
larger-domain rT variant (`cylinder_wi1.0`) differs by only 0.018 Cd.
**No rT mesh-refinement data exists locally**; cannot promote α
without generating new rT runs.

### (β) Domain-size effect — **mostly refuted (nondimensional) but partly live (lattice/TRT)**

- Nondimensional blockage D/H = 0.5 stays constant across all R (R is
  the lattice resolution of the same nondimensional cylinder, not
  the cylinder size in physical units).
- BUT lattice-distance to outlet ZouHe pressure boundary grows
  proportionally: 420 LU at R=30 → 840 LU at R=60. The fixed-density
  outlet imprint decays with R.
- nu_total scales with R (0.15 at R=30 → 0.30 at R=60), and TRT
  s_plus shifts (1.0526 → 0.7143). This couples another numerical
  variable with R.

**β verdict**: the literal "physical channel grows with R"
explanation is REFUTED. But lattice-distance + TRT-relaxation
scaling remain plausible mechanisms for a small mesh-dependent
residual on Cd_pres × wake (downstream pressure boundary) and
possibly Cd_solv × shoulder (viscosity coupling).

### (γ) Residual coupling bug audit — **no smoking gun**

Codex checked:
- Pressure readout `+F/2` patterns: NONE found. All ρ computations
  are raw `sum(f)`, no half-step bias on pressure side.
- `WriteMoments` ρ_out chain: clean. Two-pass Bouzidi has cut-link
  rho recompute, but the sweep used halfwayBB (not Bouzidi).
- Operator timing: `Cd_p` uses latest reconstructed τ before solvent
  step; `Cd_bsd` uses current-step ∇u; no lag-1 bug observed.
- **NEW candidate**: default `logfv_polymer_force_bc_aware_2d!` →
  `fvfd_tensor_divergence_2d!` uses BC-aware compact stencils but
  is NOT q_wall cut-cell aware (the sweep ran with
  `embedded_force=0`, `embedded_gradient=0`). At staircase wall
  cells, the polymer-force stencil + BSD-force stencil could
  produce a small Wi-amplified bias on the solvent pressure path.
  The Wi=0.1 row flatness (close to rT) argues this is a small
  effect; high Wi amplifies it.

**γ verdict**: no Guo-class double-count; only candidate is a
default non-embedded FVFD stencil near q_wall, testable by
re-running with `embedded_force=1` and/or `embedded_gradient=1`.

### Codex recommended next missions

1. Per-θ decomposition (DONE here in B — confirms residual locus
   on Cd_solv × shoulder + Cd_pres × wake, NOT polymer).
2. Controlled run with `embedded_force=1`, `embedded_gradient=1` at
   R=30 and R=60. If Cd_s moves toward flatness across R, the γ
   FVFD/q_wall candidate is live.
3. L_up/L_down discriminant at fixed R (R=60 with L=10, 15, 30 R) to
   separate pressure-boundary distance from mesh refinement.
4. Generate rT mesh-refinement (R=40, 50, 60) reference to close α.

---

## (c) Combined verdict

The post-M44 mesh-Wi residual is:
- **NOT** a Guo-half-step-class bug (γ negative on pressure readouts
  + B confirms Cd_polymer is fine).
- **Localized** in Cd_solv × shoulder + Cd_pres × wake (Newtonian-like
  channels, not polymer-specific).
- **Most likely** a combination of (β) lattice-distance to outlet
  pressure boundary + TRT relaxation scaling, possibly plus (γ)
  default non-embedded FVFD stencil near q_wall at high Wi.
- **α not testable** without generating new rT R=40/50/60 references.

The M44 fix closes 78 % of the M28-M42 gap and continues converging
on the dominant bucket (front_pole pressure) with R. The residual
is real but **NOT in the same class as the bug M44 fixed**, and the
sweep is **stable across the entire (R, Wi, β) envelope** (0/48 NaN).
**M28-M42 cluster remains CLOSED**; M45 documents an open
research-grade question for future work.

---

## (d) Recommended follow-up (NOT urgent, NOT blocking)

If we want to close the residual:

1. **`embedded_force=1` + `embedded_gradient=1` controlled run** at
   R=30 and R=60 (~6 min Aqua, 2 cases). If Cd_s flattens across
   R, γ confirmed and FVFD cut-cell awareness is the next fix.
2. **L_up/L_down discriminant** at fixed R=60 with L=10 R, 15 R,
   30 R (~10 min Aqua, 3 cases). If Cd_pres × wake stabilizes for
   L≥20R, the lattice-pressure-boundary distance is the dominant β
   effect.
3. **Generate rheoTool R=40/50/60 references** via OpenFOAM mesh
   refinement of the existing `cylinder_wi1.0_shrunk15R` case. Solid
   week of work; would close α definitively.

None of (1)-(3) is blocking M28-M42 closure or M44 ship.

---

## (e) Artifacts

- `bench/viscoelastic_audit/M44_GUO_FIX_VERDICT.md` (M44 root cause + fix)
- `bench/viscoelastic_audit/M44_GUO_AUDIT_CODEX.md` (M44 Codex G1-G7)
- `bench/viscoelastic_audit/M44_SWEEP_VERDICT.md` (M44 sweep 48 cases)
- `bench/viscoelastic_audit/M45_RESIDUAL_AUDIT_CODEX.md` (M45 Codex α/β/γ)
- `bench/scratch/m44_postfix_walldecomp/` (B per-θ decomp)
- `bench/scratch/m45_residual_audit/00_plan.md` (Codex plan)
- `tmp/m44_postfix_sweep/21827394.aqua/` (48 .jls field snapshots)

End of M45 verdict.
