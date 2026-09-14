# M56 V&V Ladder Verdict

## L2 — static polymer source
- Definition: Static 32x32, R=4 cylinder-mask patch; hand-fed analytic velocity gradients at C=I are evaluated by the Oldroyd-B source algebra.
- Pass criterion: max absolute source-component error < 1e-12.
- Measured: max_abs_error = 0 over 972 fluid cells; CSV: `scratch/M56_vv_ladder/PT_polymer_source_static.csv`.
- Verdict: PASS
- Interpretation: The local Oldroyd-B source algebra reads the gradient tensor components correctly at C=I. The first failing rung is not polymer-internal source indexing or relaxation algebra.

## L3 — frozen-U cylinder
- Definition: R=10 Newtonian qwall cylinder warmup, then frozen-velocity log-FV polymer replay using the embedded qwall velocity-gradient path.
- Pass criterion: trace_C_max < 1e4 after 5000 polymer-only steps, with no first NaN cell.
- Measured: DID-NOT-RUN. Blocked by the L5 go/no-go gate after the planar coupled case failed.
- Verdict: DID-NOT-RUN
- Interpretation: No read-path conclusion can be drawn for the cylinder qwall gradient replay in this mission because the lower dynamic coupled planar rung is red.

## L5 — coupled Poiseuille M55-active
- Definition: 60x32 coupled Oldroyd-B Poiseuille channel on the current M55-active worktree, beta=0.59, lambda=2000, wall-cell Wi=1, 5000 steps, Metal Float32.
- Pass criterion: max velocity relative error < 0.02 and max tau_xx relative error < 0.05.
- Measured: u_rel = 0.029441078251935569; tau_xx_rel = 0.49594615704365863; tau_xy_rel = 0.18796148459732911; min_c_eig = 0.54708081483840942; polymer_substeps = 15; CSV: `scratch/M56_vv_ladder/PT_poiseuille_coupled_m55_profiles.csv`.
- Verdict: FAIL
- Interpretation: The planar coupled LBM/log-FV loop is red before the cylinder frozen-U replay can be trusted. Provenance caveat: this planar helper uses the bc-aware channel gradient path, not the embedded qwall M55 branch, so this result blocks the ladder but does not prove the qwall helper itself is the failing path.

## Localisation
The smallest failing rung reached is L5: L2 passes exactly, then coupled Poiseuille fails at the required 5000-step scale. This discriminates away from local Oldroyd-B source tensor indexing, but it does not yet discriminate M55 qwall read-path versus generic planar coupled-loop/transient behavior because the existing Poiseuille helper does not fire `_fvfd_apply_qw_quadratic_embedded_wall_gradient_2d`.

Per the mission gate, L3 was not run. The current evidence says the dynamic coupled planar case must be resolved or re-qualified before attributing the M48 cylinder NaN to cut-cell curvature.

## Next-mission candidates
- Add a L5 time-to-steady discriminator at 5k/10k/20k steps to separate true wrong steady state from slow lambda=2000 polymer relaxation.
- Add a provenance-counted planar embedded-halfplane/qwall Poiseuille rung if the Boss wants L5 to exercise the M55 qwall branch explicitly.
- Add an analytic-initialized coupled Poiseuille replay so the coupled force loop is tested without a long polymer relaxation transient.
- Only after L5 is green or reclassified, run the existing L3 frozen-U cylinder qwall replay to isolate read-path stability.
