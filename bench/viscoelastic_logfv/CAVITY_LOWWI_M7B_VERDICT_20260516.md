# Cavity low-Wi matched-viscosity sanity — M7b verdict (2026-05-16)

## TL;DR

**SMOKING GUN: the polymer coupling has a Wi-independent bug.** At
`lambda_phys=0.001` (Wi ≈ 0.001) with matched total LBM viscosity
between two cases (`A`: ν_s=0.1, ν_p=0.1, ν_total=0.2 vs `B`: ν_s=0.2,
ν_p=0, ν_total=0.2), the centerline u relative L2 difference is
**3.42 %** — far above the numerical noise floor demonstrated by the
B vs C comparison (Re-doubling at Newtonian, **0.014 %**). The
polymer machinery introduces a perturbation on `u` that does NOT
vanish at vanishing Weissenberg, contradicting the design intent of
the BSD/Guo split.

## Setup

- Branch: `dev-viscoelastic` HEAD `a4fffb2d`.
- Aqua job `21406676.aqua`, walltime 03:11, Exit_status 0.
- PBS: `bench/viscoelastic_logfv/run_cavity_lowwi_matched_visc.pbs`.
- Common parameters: `N=64, end_time=8.0, u_max=0.005,
  bsd_fraction=0.75, lambda_phys=0.001 (Wi ≈ 0.001)`.

## Three-case design and pairwise comparison

| Case | ν_s | ν_p | ν_total | Re_LU | Description |
|------|-----|-----|---------|-------|-------------|
| A | 0.1 | 0.1 | 0.2 | 1.6 | polymer-on, low Wi (M7 polymer_on) |
| B | 0.2 | 0.0 | 0.2 | 1.6 | matched ν_total, polymer silent |
| C | 0.1 | 0.0 | 0.1 | 3.2 | Re-doubling reference (M7 nu_p_zero) |

Centerline u relative L2:

| Comparison | rel L2 | What it isolates |
|------------|--------|------------------|
| **A vs B** | **3.42 %** | Polymer-coupling delta at matched ν_total + matched Re |
| A vs C | 3.41 % | Polymer-coupling delta at unmatched ν_total |
| **B vs C** | **0.014 %** | Pure Re-doubling at Newtonian — negligible |

Absolute max-diff:
- max\|A − B\| = 4.22e-2
- max\|A − C\| = 4.22e-2

## Interpretation

The B-vs-C comparison is the critical control: both are Newtonian
(`nu_p=0`), differing only in ν_s (0.2 vs 0.1) and therefore in
Re_LU (1.6 vs 3.2). Their delta is **0.014 %**, essentially machine
noise. **Doubling Re in a pure Newtonian cavity at this scale produces
no observable effect on the centerline u profile.**

Hence, the **3.42 % A-vs-B delta is NOT the Re factor 2** I
previously attributed it to in M7. It is **entirely the polymer-
coupling contribution**: A and B share the same total viscosity and
the same Re, but A runs the polymer code path with `ν_p=0.1` while B
disables it (`ν_p=0`). At Wi=0.001 the polymer stress should be
essentially Newtonian-additive (τ_p ≈ 2·ν_p·D), and the BSD/Guo
split is supposed to absorb this into the LBM solvent viscosity via
`ν_LBM = ν_s + ζ·ν_p`. The fact that the velocity field still differs
by 3.4 % from a true matched-viscosity Newtonian baseline proves the
absorption is INCOMPLETE.

This is the first concrete localisation of the cavity-gap bug since
the M1 audit started.

## Likely sources

The bug must live in the LBM ↔ polymer coupling, in a term that does
NOT vanish as Wi → 0:

1. **BSD correction magnitude or sign**: `−ζ·ν_p·∇²u` should remove
   the Newtonian portion of the polymer stress from the body force.
   A miscomputed factor (sign, ζ, ν_p, or operator stencil) would
   leave a Wi-independent residual.
2. **Guo body-force prefactor**: `(1 − s_plus/2)` (from
   `src/kernels/dsl/bricks.jl:168-171`) might not be applied
   consistently to all branches of the body force assembly.
3. **Polymer stress assembly at the Newtonian limit**: at Wi=0,
   τ_p = 2·η_p·D analytically; any discretization difference between
   this and what the LBM consumes after Guo injection becomes a
   non-vanishing residual.
4. **Order of operations / staggering**: BSD applied before/after
   the polymer stress reconstruction, vs the order rheoTool uses.

## Decision

The bug is now localised. The smallest-scope next mission is an
**audit of the BSD/Guo coupling math** specifically at the Wi → 0
limit:

- Algebraically write out F_Guo applied to the LBM = `div(τ_p) +
  BSD_correction` in the Wi=0 limit (τ_p = 2·ν_p·D, BSD term
  = −ζ·ν_p·∇²u). For the LBM to see the correct Newtonian body force
  contribution, these terms must cancel in such a way that the LBM
  effective viscosity is exactly `ν_s + ν_p`. Check whether the
  Kraken implementation actually achieves this.
- The kinetic-moment kernel from M5-B can be used as a diagnostic
  here: compare `F_kinetic` (computed from `Π^{neq}`) against the
  FD-laplacian path at Wi=0 — at smooth bulk they should agree
  per M5-B's 5.85e-16 result, but at the walls or near corners
  they may diverge by a Wi-independent magnitude.

## Memory candidate for the Boss

The B-vs-C control (0.014 %) is the new noise floor for any future
Kraken-vs-Kraken cavity comparison at low velocity. Any future delta
above ~0.1 % is meaningful, not Re-confounded.

## Artefacts

- Raw Aqua results:
  `tmp/cavity_lowwi_matched_visc/{A_polymer_on_nus0.1_nup0.1,
  B_matched_nus0.2_nup0, C_re_ref_nus0.1_nup0}/kraken_N64_*/`.
- The original M7 verdict
  (`CAVITY_LOWWI_M7_VERDICT_20260516.md`) is now superseded by this
  one: its Re-confounder reasoning was wrong — the Re effect alone
  is 0.014 %, not 3.4 %.
