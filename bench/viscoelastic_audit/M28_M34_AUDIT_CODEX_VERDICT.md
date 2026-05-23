# M28-M34 audit — Codex verdict — 2026-05-23

## Q1: Cd_pressure +10 pts deficit at Wi=1
### Top hypothesis (HIGH)
- Statement: The R=30 Wi=1 deficit is a near-wall solvent-pressure/stagnation error in the halfwayBB moment/pressure field, concentrated at the front pole after high-Wi polymer forcing, not a polymer wall-traction accounting error.
- Supporting Facts: F2 isolates +10.13 Cd_pressure with Cd_polymer wall integral matched to 0.05, and the 3x3 bucket puts +8.34/80.4% of the total gap in pressure x front_pole; F1 shows Newtonian and Wi=0.1 are close while Wi=1 loses about 9 Cd, and R=30/40 are R-invariant; F4 shows halfwayBB writes rho/ux/uy that vel_grad reads next step; F8 removes accounting, setup, L_down, resolution, and frame explanations.
- Contradicting Facts: F1 shows bouzidi_fl_twopass overpredicts Cd at Wi=0.1, so interpolated wall geometry can change pressure sign at low Wi; no stable bouzidi Wi=1 pressure decomposition exists.
- Discriminator: For one halfwayBB diagnostic build, recompute rho_out/ux_out/uy_out from post-collision f_out before the next vel_grad reads them, then submit R=30 Wi=1; if Cd_pressure rises by about 8-10 while Cd_polymer stays matched, the wall moment/pressure chain is implicated, otherwise this hypothesis is falsified.

### Runner-up 2 (MED)
- Statement: The remaining deficit is caused by under-resolved near-wall log-conformation advection, because MUSCL improves the bulk solution but still falls back to first-order Rusanov in the two-cell solid halo where the decisive front-pole pressure is set.
- Supporting Facts: F1 shows muscl_superbee improves R=30 Wi=1 by about +5 Cd but remains 3.27% low; F6 says MUSCL falls back to first-order Rusanov within +/-2 cells of any solid; F7 says rT uses CUBISTA NVD on a body-fitted O-grid; F2 localizes the deficit at the front pole and shoulder, exactly inside the fallback halo.
- Contradicting Facts: F2 says the polymer wall integral is already matched in M29b, and M29c-v2 no-fallback becomes NaN at rho on the south wall rather than cleanly closing Cd.
- Discriminator: Change only the MUSCL fallback width from 2 cells to 1 cell for a diagnostic build and submit R=30 Wi=1 halfwayBB; if Cd_pressure closes materially without NaN, near-wall advection order is causal, while unchanged pressure or early rho NaN falsifies it.

### Runner-up 3 (MED)
- Statement: A time-centering/order mismatch between polymer force, velocity-gradient reads, and halfwayBB moment writes depresses the stagnation pressure at Wi=1 even when the final polymer wall integral is correct.
- Supporting Facts: F4 gives the sequence psi_advect -> vel_grad -> poly_force -> lbm_step -> lbm_step_halfwayBB, with halfwayBB(n) WriteMoments feeding vel_grad(n+1); F1 shows the effect is high-Wi specific and absent in Newtonian/Wi=0.1; F2 shows pressure, not viscous or polymer wall drag, owns the deficit.
- Contradicting Facts: F2's matched polymer wall integral weakens a pure polymer-force explanation, and F8 rules out simple accounting/frame mistakes.
- Discriminator: Add a diagnostic one-step lag to the polymer force field used by lbm_step, leaving the reported wall integral unchanged, then submit R=30 Wi=1 halfwayBB; a large Cd_pressure shift supports force/moment time-centering, while bit-level Cd stability falsifies it.

## Q2: NaN under :bouzidi_fl_twopass
### Top hypothesis (HIGH)
- Statement: :bouzidi_fl_twopass diverges because pass-2 overwrites cut-link populations after pass-1 WriteMoments, leaving rho/ux/uy inconsistent with the f_out state that is streamed and forced on subsequent steps; M34v3 ruled out rho-only recompute as sufficient.
- Supporting Facts: F3 says all three bouzidi NaN cases have rho NaNs with psi_nan_frac=0.0 and a uniform 97.4% rho_nan_frac spanning wake, bulk, full inlet, and full outlet; F5 proves cut-link rho recompute fires and mutates rho but leaves Cd bit-identical; F4 shows moments written by the LBM step are read by vel_grad next step; F1 shows the same scheme is finite but pressure-biased at R=30/40 Wi=0.1 and NaNs at higher load.
- Contradicting Facts: F1 finite R=30/40 Wi=0.1 means the inconsistency is not instantly fatal at low load, and F5 was verified on the finite low-Wi case rather than a NaN case.
- Discriminator: Extend pass-3 to recompute ux_out and uy_out, not just rho_out, from post-pass-2 f_out on cut-link cells and submit R=30 Wi=1 bouzidi_fl_twopass; stability plus changed Cd supports stale full moments, while unchanged NaN falsifies it.

### Runner-up 2 (MED)
- Statement: The two-pass Bouzidi post-collision overwrite is incompatible with the open inlet/outlet treatment, so a wall-local f overwrite seeds a global density-mode instability through the x-boundaries.
- Supporting Facts: F3 reports FULL inlet and FULL outlet columns in the bouzidi NaN footprint, not just cylinder-adjacent cut links; F1 shows the NaN set includes R=30/40 Wi=1 and R=60 Wi=0.1, consistent with a load-amplified density mode; F3 also says psi is not the NaN field.
- Contradicting Facts: F3 includes wake and bulk too, so inlet/outlet may be an amplifier rather than the source; F1's R=30/40 Wi=0.1 cases are finite.
- Discriminator: Add a diagnostic guard that skips pass-2 Bouzidi overwrites for i==1 or i==Nx cut-link cells, then submit R=30 Wi=1 bouzidi_fl_twopass; disappearance or strong delay of rho NaN supports x-boundary coupling, while the same NaN timing/footprint falsifies it.

### Runner-up 3 (LOW)
- Statement: Bouzidi's post-collision interpolation over-amplifies polymer-forced momentum at cut links, causing a fluid-only rho blow-up once Wi or R crosses a load threshold.
- Supporting Facts: F1 shows bouzidi R=30/40 Wi=0.1 overpredicts Cd by +1.6% to +2.32% before NaNs appear at R=60 Wi=0.1 and Wi=1; F3's bouzidi NaNs are rho-only with psi_nan_frac=0.0; F5 says rho recompute does not change the finite low-Wi result, pointing beyond a scalar rho bookkeeping fix.
- Contradicting Facts: F1 shows halfwayBB R=30/40 Wi=1 remains finite under the same polymer model, so the forcing amplitude alone is not sufficient; F3's full inlet/outlet footprint is broader than a local cut-link overshoot.
- Discriminator: Multiply Fx_field and Fy_field by 0.5 only inside the bouzidi_fl_twopass LBM call and submit R=30 Wi=1; if NaN disappears but Cd and pressure bias scale with the force, this load-threshold hypothesis survives, while rho NaN at similar time falsifies it.

## Q3: NaN under :halfwayBB at R=60
### Top hypothesis (HIGH)
- Statement: R=60 halfwayBB NaN is the nonlinear version of the high-Wi near-wall force/moment coupling: polymer force spikes near the shoulder feed the LBM density field until rho, not psi, is the first failing field.
- Supporting Facts: F1 says R=60 Newtonian halfwayBB is finite but R=60 Wi>=0.1 is NaN, while R=30/40 Wi=1 are finite but pressure-deficient; F3 localizes the first halfwayBB NaN to bilateral near-wall arcs at theta about +/-38 to 48 degrees with pre-NaN |u|=554 and |F|=320, and first-NaN field rho; F4 links halfwayBB moment writes to next-step vel_grad/poly_force.
- Contradicting Facts: F3 also reports pre-NaN max |Psi_xx|=15.86, so polymer-state growth is part of the chain even though psi is not the first NaN field.
- Discriminator: Submit R=60 Wi=1 halfwayBB with polymer force clipped or scaled by 0.5 for the first 200 steps only; if rho NaN vanishes or moves far later while psi remains finite, force-to-momentum blow-up is causal, while identical step-29 rho NaN falsifies it.

### Runner-up 2 (MED)
- Statement: The velocity-gradient operator near the curved solid shoulder generates excessive local stretching from one-sided solid-aware derivatives, which then reconstructs a polymer force large enough to crash rho.
- Supporting Facts: F3's arcs sit at shoulder angles and within r-R in [0,7] LU, and pre-NaN |Psi_xx|=15.86, |F|=320; F4 identifies vel_grad as the immediate reader of halfwayBB moments before poly_force; the required source path shows the gradient kernel uses solid-aware one-sided derivative helpers near solids.
- Contradicting Facts: F3 says the first NaN field is rho, not psi, and F1 shows the same path survives R=30/40 Wi=1, so the gradient issue needs R-dependent amplification.
- Discriminator: In the R=60 diagnostic only, zero dudx/dudy/dvdx/dvdy for cells within two lattice cells of solid shoulder arcs before poly_force and submit one Aqua run; removal of the early rho NaN supports gradient-origin forcing, while unchanged rho NaN falsifies it.

### Runner-up 3 (LOW)
- Statement: First-order near-wall log-conformation advection around solids creates a shoulder stress overshoot at R=60 that is tolerated at R=30/40 but unstable at higher resolution/load.
- Supporting Facts: F6 says MUSCL falls back to first-order Rusanov within +/-2 cells of solids, so the shoulder halo remains low order; F3 places the halfwayBB NaN arcs in the near-wall halo; F1 shows R=60 fails for viscoelastic cases while Newtonian R=60 is finite.
- Contradicting Facts: F3's first NaN is rho and pre-NaN |u| and |F| are already enormous, so advection is at most upstream of a momentum blow-up; M29c-v2 no-fallback at R=30 Wi=1 also NaNs on rho at a south wall rather than cleanly stabilizing.
- Discriminator: Run R=60 Wi=0.1 halfwayBB with Psi advection locally frozen only inside the two-cell solid halo and BSD/poly_force unchanged; if the shoulder rho NaN timing is unchanged, near-wall advection is falsified, while large delay plus lower |Psi_xx| supports it.

## Relationship (Q1, Q2, Q3): chained
- Evidence: Q1 and Q3 are chained symptoms on the halfwayBB path: F1 shows finite but pressure-deficient R=30/40 Wi=1 and NaN R=60 Wi>=0.1, F2 locates the finite error in front/shoulder pressure, and F3 locates the R=60 failure in near-wall shoulder arcs with rho first. Q2 is a distinct branch in the same rho/moment family: F3's bouzidi footprint is uniform with full inlet/outlet columns and psi_nan_frac=0.0, unlike the halfwayBB shoulder arcs, while F5 rules out the attempted rho-only pass-3 fix.

## Recommended next mission
- Mission: Test full post-Bouzidi moment consistency by recomputing rho, ux, and uy after pass-2 cut-link overwrites.
- Why: F5 proves rho-only pass-3 is real but inert, while F3 says bouzidi NaNs are rho/momentum-field failures with psi untouched and F4 says next-step vel_grad consumes LBM-written moments.
- Concrete change: <=50 LOC in `src/kernels/dsl/bricks.jl` and `src/kernels/li_bb_2d_v2.jl`: replace/extend `ApplyCutLinkRhoRecompute` so its required args include `:ux_out, :uy_out`, compute rho plus velocity sums from post-pass-2 `f_out` on cut-link cells, and pass `ux, uy` into pass-3.
- Aqua submission: R=30 Wi=1 `:bouzidi_fl_twopass` F64; acceptance is no rho NaN through the previous failure window and a non-bit-identical Cd trace relative to M34v3, with R=30 Wi=0.1 retained finite as a smoke check.
