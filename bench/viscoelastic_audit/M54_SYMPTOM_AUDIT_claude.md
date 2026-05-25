# M54 Symptom Audit — Claude variant

Adversarial dual-spawn synthesis of M46→M53 viscoelastic cylinder audit
arc. Independent of `M54_SYMPTOM_AUDIT_codex.md` (not read). Pure
synthesis from existing audits + memory; no source-code reads, no
measurements. Cite format `(file:line)` inline.

## 1. Symptom matrix

| ID | Symptom (1 line) | Fixture | Quantity | Wi | Mesh R | BC | Backend | Source audit | Suspected mechanism | Severity |
|----|------------------|---------|----------|----|--------|----|---------|--------------|---------------------|----------|
| S01 | R-sweep plateau Cd non-monotone: 114.48 / 117.62 / 114.26 at R=10/30/50 (U-shape) | cylinder coupled | Cd plateau (last 50% FT) | 1.0 | 10/30/50 | halfwayBB | Metal F32 | M48_HW_MESHCONV:36-40 | wall-coupling bias scaling with cut-link cell count | HIGH |
| S02 | All three R values reach plateau within 1 FT (drift <0.05 Cd in last 30%) | cylinder coupled | Cd(t) drift | 1.0 | 10/30/50 | halfwayBB | Metal F32 | M48_HW_MESHCONV:42-49 | falsifies "M46-B wake-not-established" hypothesis | INFO |
| S03 | trace_C plateau scales with R: 106/199/225 (still growing at R=50) | cylinder coupled | trace_C plateau | 1.0 | 10/30/50 | halfwayBB | Metal F32 | M48_HW_MESHCONV:36-40 | polymer chain amplifies wall bias as resolution increases | MED |
| S04 | M44 anchor R=30 Wi=1 = 118.10 (Aqua F64), gap -1.89% vs rT 120.38 | cylinder coupled | Cd | 1.0 | 30 | halfwayBB | Aqua F64 | m44-guo-halfstep:40-44 | post-Guo-half-step fix baseline | INFO |
| S05 | M45 sweep 0/48 NaN incl. R=60 Wi=1 (pre-fix both NaN) | cylinder sweep | NaN rate | 0.1-1.0 | 30-60 | halfwayBB | Aqua F64 | m44-m45-sweep:21-26 | Guo velocity bias was instability trigger | INFO |
| S06 | Cd_pres × wake DIVERGES from rT as R grows (-1.61 over R=30->60) | cylinder coupled | Cd per region | 1.0 | 30-60 | halfwayBB | Aqua F64 | m44-m45-sweep:48-50 | residual is Newtonian-like (solv shoulder + pres wake) | MED |
| S07 | FVFD quadratic wall stencil P2/P3 ERR=1,2 vs analytic at halfway wall | PT_halfway_wall_stencil | abs_err vs analytic | – | – | halfwayBB | CPU F64 | M49_WALL_STENCIL:22-26 | returns first-fluid CENTER derivative not wall derivative | HIGH |
| S08 | FVFD stencil callsites split (a) 4 cell-center (correct) (b) 4 wall (BUG) (c) 4 ambiguous embedded | code audit | classification | – | – | – | – | M50_STENCIL_CALLER:46-50 | wall-row consumers semantically wrong | MED |
| S09 | M51 helper PASS canary @ 4.97e-14 abs_err P1/P2/P3 | PT_halfway_wall_stencil | max abs_err | – | – | halfwayBB | CPU F64 | M51_VERDICT:108-110 | derived quadratic `(3u1 - u2/3 - (8/3)u_wall)/dy` works | INFO |
| S10 | Post-M51 cylinder R-sweep: Cd 114.87 / 117.54 / 113.93 (deltas +0.39 / -0.08 / -0.33) | cylinder coupled | Cd plateau | 1.0 | 10/30/50 | halfwayBB | Metal F32 | M48_POSTFIX:14-19 | U-shape unchanged; outer walls not dominant source | HIGH |
| S11 | 12 patch_ladder tests RED after broad M51 application (M5d/M5e/M7d/M8h) | patch ladder | various tolerances | – | – | – | CPU F64 | M53c_BIFURCATION:56-62 | polymer chain at cell-center consumed wall-position grad | HIGH |
| S12 | M5e frozen Couette steady NOT preserved: max_c_error 0.67 (was machine eps) | M5e Couette frozen | max_c_error | – | – | – | CPU F64 | M53d_POLYMER:41-43 | non-embedded channel-gradient/source regression | HIGH |
| S13 | halfwayBB at cylinder cut-cells: q_w IS honored (dispatches `ApplyLiBBPrePhase`) | code audit | LBM-side BC | 1.0 | – | halfwayBB | – | M52a_CUTCELL:8-29 | LBM-side BC OK; FVFD-gradient side is the bug | INFO |
| S14 | FVFD velocity-gradient at cylinder-adjacent cells NOT q_w-aware (default path) | code audit | gradient mode | 1.0 | – | halfwayBB | – | M52a_CUTCELL:31-46 | identical M49 bug class at cut-cells | HIGH |
| S15 | M52b cyl-adj canary: mean abs_err 0.071, max 0.141, corr(q_w,err)=+0.78 | PT_cylinder_adjacent | abs_err vs Stokes | – | R=4 | – | CPU F64 | M52b_CYL_ADJ:30-36 | bias scales positively with q_w | HIGH |
| S16 | M53a embedded toggle: mean abs_err 0.052 (only 1.37x better, sign flips on q_w corr) | PT_cylinder_adjacent | abs_err split path | – | R=4 | – | CPU F64 | M53a_EMBEDDED:18-21 | embedded helper insufficient + breaks small q_w | HIGH |
| S17 | wall_inv_distance convention mismatch: stored is fluid-centroid not plane-distance | code audit | helper math | – | – | – | – | M53b_EMBEDDED:5,77-81 | small-q_w under-corrects, high-q_w fine | HIGH |
| S18 | Post-bifurcation embedded improves to 3.07x (mean 0.023) but RED | PT_cylinder_adjacent | abs_err | – | R=4 | – | CPU F64 | M53c_BIFURCATION:52 | first-order limit; cannot reach <1e-3 | MED |
| S19 | `embedded_gradient=true` -> NaN divergence cylinder R>=30 coupled | cylinder coupled | divergence | 1.0 | >=30 | halfwayBB | Metal F32 | postmortem:49-52 | inv_distance large at low q_w -> strong correction -> polymer instability | HIGH |
| S20 | M46 sweep Bouzidi-FL Newt trace_C blows up 209 -> 1.4e7 between R=30 and R=60 | cylinder Newt | trace_C | 0 | 30/60 | bouzidi_fl | Aqua F64 | NEXT_SESSION:118-120 | unresolved Bouzidi-side anomaly (M47 H1 parked) | MED |
| S21 | M45 Codex audit: residual cannot be attributed to Guo-class bug; mixed lattice/TRT scaling + non-embedded FVFD at q_wall | static + run | per-theta decomp | 0.1-1 | 30-60 | halfwayBB | Aqua F64 | m44-m45-sweep:62-79 | β (TRT s+ ∝ 1/R) + γ (FVFD non-qwall-aware) | MED |
| S22 | M44 fix closes 78% of original Cd gap at R=30 (Cd_pres×front_pole 62%) | cylinder coupled | per-region closure | 1.0 | 30 | halfwayBB | Aqua F64 | m44-m45-sweep:54-60 | Guo half-step double-count was the dominant 80% bucket | INFO |
| S23 | M2c diagonal fixture RED after bifurcation: dudx[3,3]=-0.7347 vs -sqrt(2)/2 | M2c patch | gradient value | – | – | – | CPU F64 | M53d_POLYMER:27-30 | fixture pre-seeded with centroid wall_distance; needs rebaseline | LOW |
| S24 | Cylinder cut-link count scales ~2πR; outer wall count ~2Nx; M51 closed outer only | geometric | cell-count | – | – | halfwayBB | – | M48_POSTFIX:48-58 | cut-link bias dominates; outer-wall bias is sub-dominant | MED |

## 2. Clustering verdict

**Cluster A — Wall-gradient stencil bug family (axis-aligned + cut-cell)**
Members: S07, S08, S09, S10, S14, S15, S16, S17, S18, S24.
Shared candidate root cause: `_fvfd_solid_bc_derivative_*_2d` returns
a derivative evaluated at the first-fluid CELL CENTER, never at the
wall position. M49 catches the axis-aligned version (S07); M52b/M53a
catch the cut-link generalization (S15-S16); M53b localises the
embedded-helper distance convention bug (S17). M51 fixes only the
axis-aligned outer walls (S09) and confirms cut-link is the dominant
contributor (S10, S24).
Confidence: **HIGH** that these share the same code-path family.

**Cluster B — Cylinder Cd U-shape mesh anomaly**
Members: S01, S03, S06, S10, S19, S21.
Shared candidate root cause: combination of cluster A residual at
cut-cells + a NEWTONIAN-like driver scaling (M45 β: TRT s+ ∝ 1/R,
lattice-distance to outlet doubles 420→840 LU at constant nondim
geometry). S10 proves cluster A axis-aligned-only fix is NOT enough.
S19 proves naive embedded toggle is unstable.
Confidence: **MED** that A fully explains B; the residual Cd_pres×wake
divergence (S06) and Newt-side decomposition (S21) suggest a second
independent term.

**Cluster C — M51 over-application regression**
Members: S11, S12, S23.
Shared candidate root cause: the M51 helper overwrote velocity-gradient
arrays at wall rows, but downstream polymer-chain consumers in
non-embedded drivers (M5d/M5e/M7d/M8h) read those arrays at
**cell-center** semantics (for the FVFD divergence convention).
S23 is a fixture rebase artifact (R-class), the rest are physics
regressions (P-class per M53d). RESOLVED by reverting M51 application
from shared step + 4 drivers (postmortem:31-35).
Confidence: **HIGH** that members share root cause; status is FIXED
in the worktree.

**Cluster D — Bouzidi-FL parked anomaly (orthogonal to A/B)**
Members: S20.
Shared candidate root cause: unresolved trace_C blowup specific to the
Bouzidi-FL BC code path (M47 H1 parked). Not addressed by any M48-M53
work. Not relevant to halfwayBB U-shape.
Confidence: **HIGH** (single member, isolated by M52a finding that
halfwayBB and Bouzidi-FL dispatch different bricks).

**Cluster E — M44 baseline / closure facts**
Members: S04, S05, S22.
Shared candidate root cause: NOT a bug; these are the M44 anchor
facts framing the residual.
Confidence: **HIGH** (factual).

**Inter-cluster relationships**:
- A ↔ B: A is the **subcomponent** mechanism, B is the **integrated**
  symptom. Very likely SAME underlying bug at different layers (canary
  vs coupled benchmark). The fact that M51 closed the canary (S09) but
  not the benchmark (S10) means A-axis-aligned is NOT the dominant
  layer; A-cut-cell (S15-S17) is the load-bearing one.
- A ↔ C: Same array (`dudx/dudy/dvdx/dvdy`), different consumer
  contracts. C is what happens when you fix A wrongly. They share the
  array, NOT the root mechanism.
- B ↔ D: Different BC dispatch paths per M52a; no expected coupling.
- A ↔ E: M44 fix is orthogonal to the wall stencil (it was a Guo
  readout double-count). E established the new baseline that exposed B.

## 3. Gaps / not-yet-measured discriminators

1. **q_w-aware quadratic cut-link canary on M53a fixture** (`u = a + b·s + c·s²` through `u(0)=0`, `u(q_w·dx)=u₁`, `u((q_w+1)·dx)=u₂`). Target: mean abs_err < 1e-3 vs current first-order 0.023. Would prove whether second-order cut-cell stencil exists and works BEFORE any coupled cylinder run. ~1 h Codex, <2s CPU validation.
2. **M48 R-sweep with `wall_bc=:bouzidi_fl_twopass`** but `embedded_gradient=false`. Discriminates whether the U-shape is BC-class (LBM-side) or stencil-class (FVFD-side). M52a predicts it should NOT change much since M52a found halfwayBB IS already q_w-aware at LBM-side. A negative result would confirm cluster A as the load-bearing path. ~30 min Metal.
3. **Static stability analysis of `_fvfd_apply_embedded_wall_gradient_2d` post-bifurcation under low q_w**. Specifically: what is `inv_distance` at q_w=0.05 for cylinder R=30? Then estimate amplification on the polymer-chain feedback loop. Would explain S19 (NaN at R≥30 with embedded toggle). ~30 min Codex static.
4. **Newt-only R-sweep at R={30,40,50,60} with the same halfwayBB + qwall + L=15R setup**. M45 says Cd_pres×wake diverges Newtonian-like (S06, S21); a pure Newtonian sweep would isolate whether the wake-divergence term is polymer-dependent at all.
5. **rheoTool reference at R={40,50,60}**. Closes M45's α option. Without external reference at R>30, "U-shape" is unattributable: it could be a Kraken bug, a rheoTool R=30-specific lucky number, or a real physical effect.

## 4. Recommended next mission

**Option A** — build the q_w-aware second-order cut-cell helper.

Justification (citing S-IDs):
1. S10 proves M51's axis-aligned-only fix did NOT flatten the U-shape, AND that the dominant contributor is the cylinder cut-link cells (S24).
2. S14 + S15 + S17 collectively localize the bug in a SINGLE family of code (FVFD stencil at variable q_w cut-cells, first-order at best per S18).
3. S16 + S19 show that the existing embedded helper is BOTH insufficient (only 1.37x improvement) AND unstable when toggled on coupled runs — a NEW helper is needed, not a flag flip.
4. Option B (Bouzidi-FL) is a DISCRIMINATOR not a fix and is dominated by gap #2 above (cheaper to run as a side-check anyway).
5. Option C (publication pivot) is premature — M48 U-shape is mechanistically explainable but not yet explained; A is the smallest fix that could close it.
6. Option D (commit + new session) should be done OPPORTUNISTICALLY (it costs nothing) but is not itself the next-mission decision; A and D are orthogonal.
7. Cluster A confidence is HIGH and the mathematical sketch (NEXT_SESSION:107-108) is straightforward — derivable in <1h Codex.

Recommendation: do Option A, gated on the canary in §3 gap #1 (target <1e-3), then run M48 R-sweep. Commit before-A (D) as a checkpoint. Skip B unless A fails the canary.

## 5. Boss-relevant flags

- **Contradiction-class flag**: NEXT_SESSION:39 says `_fvfd_solid_bc_derivative_*_2d` "returns derivative at first-fluid CENTER, not at wall" while M50:7 calls intent (a) cell-center "Correct usage" for tensor-divergence. Both true — the stencil semantics ARE cell-center-correct for FVFD divergence consumers but WRONG for wall-row gradient consumers. M50 already classifies callsites (a)/(b)/(c). Boss must not conflate the two contracts when reviewing any fix patch (this is exactly what cluster C taught).
- **Dead-toggle warning**: `embedded_gradient=true` is a kwarg that, per S19, produces NaN at R≥30 on the load-bearing cylinder fixture. It is preserved as a knob but is not safe to recommend as a workaround. M53d explicitly warns: "do not blame `wall_inv_distance_to_center` without enabling that path" (M53d:80-82).
- **Uncommitted infrastructure** (postmortem:108-111): M51 helper + M53c bifurcation field + cavity refactor are ALL on disk but NOT committed. Tests are 18213/953 GREEN. Recommend Option D (commit) BEFORE starting any new fix (independent of the A vs B vs C choice in §4).
- **Reference gap** (S22 + M45:62-65): rheoTool 120.38 is at R=30 ONLY. Without R=40/50/60 reference values, the U-shape interpretation depends on assuming rT extrapolates linearly. M45 already flagged this as "untestable α".
- **Process win to preserve**: M49 and M52b/M53a canaries are <2s and machine-precision discriminators (M48_POSTFIX:90-100, M52b:67). Per `[[feedback_small_tests_first]]`, every future fix MUST pass these BEFORE any Metal/Aqua R-sweep is launched. The 28-min M48 R-sweep iteration cost was the dominant time sink of the M48-M53 session.
