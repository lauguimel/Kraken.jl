# M54 Symptom Audit (Codex)

## Symptom matrix

| ID | Symptom (1 line) | Fixture | Quantity measured | Wi | Mesh R | BC | Backend | Source audit | Suspected mechanism | Severity (LOW/MED/HIGH) |
|---|---|---|---|---|---|---|---|---|---|---|
| S01 | Guo half-step fix closes most original R=30 gap: Cd 111.09 -> 118.10, 78% closure. | Cylinder anchor | Cd vs rT 120.38 | 1 | 30 | halfwayBB qwall | Aqua A100 F64 | "Post-fix Aqua F64 100k" (M44_GUO_MEM:40-44) | Closed Convention-I readout double-count; background, not current U-shape. | LOW |
| S02 | Post-M44 Wi=1 residual was monotone R drop: 118.10 -> 116.99 -> 115.68 -> 113.23. | Cylinder sweep | Cd by R | 1 | 30-60 | halfwayBB qwall | Aqua A100 F64 | "Cd decreases monotonically with R" (M45_MEM:34-36) | Older mesh residual in Cd_s / pressure wake may share curved-wall FVFD/qwall bias. | MED |
| S03 | M48 halfwayBB Cd is U-shaped: R=10 114.48, R=30 117.62, R=50 114.26. | Cylinder R-sweep | Plateau Cd | 1 | 10,30,50 | halfwayBB qwall | Metal M3 Max F32 | "U-shape with the best value at R=30" (M48_HW:8-10) | Coupled wall/polymer bias grows nonlinearly with curved-wall/cut-link exposure. | HIGH |
| S04 | All M48 R values plateau within 1 flow-through; R=50 wrong value is not just transient drift. | Cylinder R-sweep | Last-quarter Cd drift | 1 | 10,30,50 | halfwayBB qwall | Metal M3 Max F32 | "All three are at genuine steady-state" (M48_HW:42-49) | Falsifies under-sampling/wake-transient explanation for the R=50 deficit. | MED |
| S05 | M48 trace_C rises strongly with R: 106, 199, 225, with R=50 still slowly growing. | Cylinder R-sweep | trace_C plateau | 1 | 10,30,50 | halfwayBB qwall | Metal M3 Max F32 | table rows (M48_HW:36-40) | Polymer amplification may convert local near-wall gradient bias into Cd bias. | MED |
| S06 | M51 flat-wall helper does not flatten M48: R=30 stays 117.62 -> 117.54; R=50 drops 114.26 -> 113.93. | Cylinder post-fix R-sweep | Plateau Cd pre/post | 1 | 10,30,50 | halfwayBB qwall | Metal M3 Max F32 | "does NOT flatten" (M48_POSTFIX:8-19) | Axis-aligned outer-wall bug is real but not dominant for cylinder Cd. | HIGH |
| S07 | Raw axis-aligned wall stencil fails P2/P3: P2 max err 1, P3 max err 2. | M49 wall canary | Max abs derivative error | n/a | n/a | halfway wall rows | CPU Float64 raw arrays | "fails P2/P3" (M49:10-12) | Stencil returns first-fluid-center derivative, not halfway-wall derivative. | HIGH |
| S08 | M51 helper fixes S07 in isolation: P1/P2/P3 max abs error 4.97e-14 and FVFD tests pass. | M49 extended canary | Max abs derivative error; test count | n/a | n/a | axis-aligned halfway walls | CPU Float64 | "abs_error = 4.97e-14" (M51:11-14) | Correct second-order flat-wall formula exists, but only for wall-position consumers. | MED |
| S09 | Caller audit splits direct stencil use into 4 cell-center correct, 4 wall-intent buggy, 4 embedded ambiguous. | Static call-site audit | Call-site classification | mixed | mixed | mixed | static | "Intent (b) wall: 4" (M50:46-50) | Same helper has overloaded cell-center vs wall-gradient semantics. | MED |
| S10 | `wall_bc=:halfwayBB` cut-cell LBM path honors q_wall via LI-BB; q_w=0.5 fallback hypothesis is false. | Static cut-cell audit | Dispatch / q_wall use | 1 target | cylinder | halfwayBB qwall | static | "does not force q_w=0.5" (M52a:4-5) | LBM-side cut-cell BC is likely not the M48 root; naming is misleading. | MED |
| S11 | Default cylinder-adjacent FVFD velocity gradient is not q_w-aware when `embedded_gradient=false`. | Static cut-cell audit | Gradient API inputs | 1 target | cylinder | halfwayBB qwall | static | "no q_wall, wall normal, or wall distance" (M52a:31-47) | Curved-wall gradient uses solid-mask cell-center derivative class near cut links. | HIGH |
| S12 | Cylinder-adjacent canary fails: mean abs_err 0.0711, max 0.1413, corr(q_w, err)=0.7758. | M52b cut-link canary | du/dn error | n/a | R=4 test cylinder | qwall cut links | CPU Float64 raw arrays | "mean abs_err 0.0711" (M52b:10-12) | Standalone q_w-dependent curved-wall gradient bug. | HIGH |
| S13 | M52b error rises sharply at high q_w: mean 0.1165 in 0.7-0.9 bin vs about 0.037 below 0.5. | M52b cut-link canary | Error by q_w bin | n/a | R=4 test cylinder | qwall cut links | CPU Float64 raw arrays | "errors are lowest... rise sharply" (M52b:50-58) | Variable-distance cut-link geometry, not a uniform factor-only error. | HIGH |
| S14 | Existing embedded path only partially helps M52b: mean 0.0711 -> 0.0521, still RED; small q_w bins get worse. | M53a embedded canary | mean/max err; q_w bins | n/a | R=4 test cylinder | embedded qwall | CPU Float64 raw arrays | "far below the YELLOW criterion" (M53a:36-44) | Embedded normal correction is not sufficient and under-corrects/over-noises low-q_w links. | HIGH |
| S15 | Embedded helper distance bug found: wall_inv_distance stored centroid distance but helper multiplies cell-center phi. | Static embedded audit | Distance convention | n/a | cut cells | embedded qwall | static | "wall_inv_distance is ... centroid distance" (M53b:4-5) | Gradient target used wrong distance convention, especially at low q_w. | HIGH |
| S16 | M53c bifurcation improves embedded canary mean to 0.0231 but script stays RED and patch ladder has 12 failures. | M53c validation | canary error; test failures | mixed | mixed | embedded/non-embedded | direct Julia CPU | "embedded mean abs_err 0.0231" (M53c:51-54) | Plane-vs-centroid split helps but first-order cut-cell helper remains inadequate and regressions remain. | HIGH |
| S17 | `embedded_gradient=true` on coupled M48 cylinder NaNs at R>=30 after bifurcation. | Coupled cylinder follow-up | NaN divergence | 1 | >=30 | embedded qwall | local coupled run | "NaN divergence at R=30+" (POSTMORTEM:49-52) | First-order embedded correction is too noisy/strong for coupled polymer cylinder. | HIGH |
| S18 | M2c diagonal failures are fixture fallout: `-0.7347401208725196` vs `-sqrt(2)/2` after plane-distance switch. | Log-FV patch ladder | dudx/dudy expected values | n/a | embedded fixture | embedded | CPU tests | per-failure table (M53d:9-12) | Test fixture seeded centroid distance while operator now uses plane distance. | MED |
| S19 | Ten M53c patch-ladder failures are physics regressions on non-embedded or embedded-disabled paths, not wall-distance consumers. | Log-FV patch ladder | C/tau/force/ux/rho failures | mixed | channel/square/BFS | non-embedded default | CPU tests/static triage | "2 (R), 0 (C), 10 (P)" (M53d:4-5) | Wall-position helper contamination or non-embedded source/force path regression, separate from embedded field bifurcation. | HIGH |
| S20 | M51 over-application broke polymer-chain tests because wall-position gradients were consumed as cell-center gradients. | M51-M53 session | Patch-ladder regressions | mixed | mixed | mixed | CPU tests | "wall-position gradient injection broke M5e" (POSTMORTEM:41-48) | Consumer semantic mismatch: wall gradient arrays cannot be blindly fed to FVFD volume/polymer source loops. | HIGH |
| S21 | Bouzidi-FL anomaly persists in background: Newtonian trace_C 209 -> 1.4e7 between R=30 and R=60. | M46/Bouzidi context | trace_C blow-up | Newt context | 30-60 | Bouzidi-FL twopass | prior sweep | "trace_C 209 -> 1.4e7" (NEXT_SESSION:117-120) | Possible BC-side instability independent of halfwayBB, but still useful discriminator. | MED |

## Clustering verdict

**CLUSTER-0: closed Guo forcing/readout bug**
- Members: S01.
- Shared candidate root cause: Guo Convention-I collision was paired with a second post-collision `+F/2` readout. The M44 fix closed the large M28-M42 gap and should be treated as a closed background cluster.
- Confidence: HIGH.

**CLUSTER-A: coupled cylinder mesh-convergence residual**
- Members: S02, S03, S04, S05, S06.
- Shared candidate root cause: a curved-wall / cut-link polymer-LBM coupling bias survives the Guo fix and the flat outer-wall helper. The M48 signal is steady-state and integration-level, not a transient artifact.
- Confidence: MED that these symptoms share a root; HIGH that S03/S04/S06 are the same observed M48 failure.

**CLUSTER-B: axis-aligned wall-position stencil bug**
- Members: S07, S08, S09.
- Shared candidate root cause: `_fvfd_solid_bc_derivative_*_2d` can answer a wall-gradient request with a first-fluid-center derivative. M51 fixes this for flat wall-position consumers, but S06 shows it is not sufficient for cylinder Cd.
- Confidence: HIGH inside the cluster; LOW as the dominant M48 root.

**CLUSTER-C: curved cut-cell / embedded-gradient geometry bug**
- Members: S10, S11, S12, S13, S14, S15, S16, S17.
- Shared candidate root cause: LBM cut-cell q_w is honored, but the FVFD velocity-gradient side either ignores q_w (default) or applies a first-order embedded normal correction that was distance-bugged and remains too noisy after bifurcation.
- Confidence: HIGH for standalone canary/root-cause linkage; MED-HIGH that it explains a large part of CLUSTER-A.

**CLUSTER-D: polymer consumer semantic regressions**
- Members: S18, S19, S20.
- Shared candidate root cause: wall-position gradient corrections were inserted into arrays consumed by cell-center / volume polymer-chain paths. This is a contract error rather than a missing distance-field consumer.
- Confidence: HIGH.

**CLUSTER-E: Bouzidi-FL BC-side anomaly**
- Members: S21.
- Shared candidate root cause: a separate Bouzidi-FL instability or polymer-chain BC interaction remains unresolved from M46/M47. It is not proven to be the halfwayBB U-shape root.
- Confidence: LOW-MED.

Inter-cluster:
- B and D are likely the same semantic bug family at different layers: wall-position gradients were not separated from cell-center gradient consumers.
- B and C share the geometric-offset theme, but S06 separates flat outer-wall correction from curved cylinder cut-link bias.
- A and C are plausibly the same bug expressed at integration scale, but only a q_w-aware second-order cut-cell helper or discriminator can prove it.
- C and D interact operationally but are not the same root: M53d shows the ten physics regressions are on non-embedded or embedded-disabled paths.
- E and C should be treated as separate until Bouzidi-FL changes M48 under the same FVFD gradient path.
- CLUSTER-0 is independent of A-E except as the closed baseline that exposed the smaller residual.

## Gaps / not-yet-measured discriminators

1. Measure M53a canary after a true second-order q_w-aware cut-cell helper.
   Why: directly tests CLUSTER-C without a 30-minute coupled run. Cost: <2 s CPU plus implementation.

2. Re-run M48 R=30/R=50 only after the M53a second-order canary is GREEN.
   Why: tests whether CLUSTER-C collapses CLUSTER-A. Cost: about 20-30 min Metal F32.

3. Run M48 with `wall_bc=:bouzidi_fl_twopass` at R=30/R=50 using otherwise identical defaults.
   Why: separates a BC-class contribution from the FVFD cut-cell gradient class. Cost: about 30 min Metal, risk of Bouzidi instability.

4. Log coupled cylinder gradient/Cd residuals by q_w bin and polar sector for R=30 and R=50.
   Why: checks whether the integration-scale Cd drop localizes to the same high-q_w links as M52b. Cost: one instrumented Metal sweep.

5. Trace M5e Couette steady-state fixed point after any gradient-array wiring change.
   Why: prevents another CLUSTER-D wall-vs-cell-center semantic regression. Cost: <1 min CPU.

## Recommended next mission

**Option A: cut-cell second-order helper.**

The strongest live cluster is CLUSTER-C: S11-S16 show a real q_w-dependent cylinder-adjacent FVFD gradient bug, while S10 refutes the simpler LBM-side q_w=0.5 hypothesis. S06 says the flat-wall M51 fix is not the cylinder fix, and S17 says promoting the current first-order embedded path is unsafe. Option B remains useful, but it mostly discriminates against a BC-side cluster (S21) that M52a has already weakened for halfwayBB. Scope Option A narrowly: make M53a GREEN first, do not overwrite shared polymer arrays blindly, and only then spend Metal time on M48.

## Boss-relevant flags

- `wall_bc=:halfwayBB` is a misleading name in this driver path: M52a says the selected LI-BB pre-phase reads true q_wall, so do not keep chasing a fixed-q_w halfway LBM bug without new evidence.
- The M51 helper is a real flat-wall fix but not a cylinder fix; broad application already produced CLUSTER-D regressions.
- The current embedded path is not a safe default for M48: post-bifurcation canary is improved but RED, and coupled runs NaN at R>=30.
- The session state is uncommitted per NEXT_SESSION, but tests were reported clean after cleanup; this is a process checkpoint, not by itself the best scientific next mission.
- Future audits should keep canary -> integration order: M49/M53a first, then M48; avoid launching expensive Metal sweeps against an ungreen local proxy.
