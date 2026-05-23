# M28-M34 audit — Claude verdict — 2026-05-23

**Methodology**. Independent derivation from §Facts alone. No reading of prior
M28-M34 verdicts, no mandate, no Codex counterpart. Grounded in source-read of
`src/kernels/dsl/bricks.jl`, `src/kernels/li_bb_2d_v2.jl`,
`src/fvfd/operators_2d.jl`, `src/kernels/logconformation_fv_2d.jl`,
`src/kernels/boundary_rebuild.jl`, and the call-site at
`src/drivers/viscoelastic_logfv_2d.jl:460-481`.

Key code-level invariants relied on below:

- The LBM step is `fused_trt_libb_v2_guo_field_step!(...; wall_bc=...)` called
  at driver line 477, immediately followed by
  `apply_bc_rebuild_2d!(f_out, f_in, bcspec, ν, Nx, Ny)` at line 481. The Zou-He
  inlet/outlet **reads `f_in` (lag-1)** but writes to `f_out`, so the inlet/outlet
  rebuild operates on the previous step's interior pops, not on the just-collided
  ones (`src/kernels/boundary_rebuild.jl:377-388`).
- `:halfwayBB` path = `_TRT_LIBB_V2_GUO_FIELD_SPEC` = PullHalfwayBB → SolidInert
  → **ApplyLiBBPrePhase** (the pre-phase BC) → Moments → Guo collide →
  WriteMoments (`li_bb_2d_v2.jl:49-54, 172-186`).
- `:bouzidi_fl_twopass` path = pass-1 = `_TRT_LIBB_V2_GUO_FIELD_RAW_SPEC` =
  PullHalfwayBB → SolidInert → Moments → Guo collide → WriteMoments. **It drops
  `ApplyLiBBPrePhase` entirely** (comment at `li_bb_2d_v2.jl:63-73`). Pass-2 only
  overwrites `f_out[i,j,qbar]` on cut links (no moment/ρ update; `bricks.jl:80-82`).
  Pass-3 re-sums ρ on cut cells, leaves u unchanged (`bricks.jl:707-727`).
- MUSCL on Ψ uses Rusanov fallback whenever any of `±2` neighbours is solid
  (`operators_2d.jl:523-533`).
- Polymer body force is computed from `tau` divergence: `poly_force` →
  `bsd_correct_force` → `lbm_step` reads `Fx_field, Fy_field` inside Guo
  collide. The Guo collide reads the **current step's** ρ and u from the Moments
  brick output (not from `ρ_out`). The next step's `vel_grad` reads `ρ_out`,
  `ux_out`, `uy_out` from `WriteMoments`.

---

## Q1: Cd_pressure +10 pts deficit at Wi=1

### H1.A (rank 1, MED–HIGH): The deficit is dominantly a **kinetic momentum-flux deficit at the front-pole solvent BC**, not a polymer-source deficit

- **Statement.** At Wi=1 with `:halfwayBB`+`:rusanov`, the pre-phase
  `ApplyLiBBPrePhase` substitutes the corrupted pulled pop with a lag-1
  halfway-BB estimate (`bricks.jl:355-402`). Under polymer loading, the
  polymer-force-driven shoulder slowdown and front-pole over-pressurisation
  *should* manifest as an extra inward kinetic momentum flux on q=2 and q=6/9
  links into the cylinder. But the pre-phase BC fixes the corrupted pop to a
  lag-1 stress-free value `f_in[i,j,q] + δ` where δ depends only on
  `uw_link_x[i,j,q]` (the **wall velocity**, =0 for stationary cylinder). It is
  blind to the *non-equilibrium stress* the flow is trying to communicate
  across the wall. The resulting Cd_pressure decomposition is consistent
  with this: it is the kinetic-flux-from-solvent component (Cd_p), not Cd_polymer,
  that under-predicts by +10 pts.
- **Supporting Facts.**
  - Wall decomposition at M29b Wi=1 R=30: Cd_polymer matches rT to 0.05; **Cd_pressure
    gap = +10.13 (1.13× the total gap +8.77)**. The polymer scheme is innocent.
  - D1 bucket: pressure × front_pole = +8.34 (80.4% of total), pressure × shoulder
    = +1.67 → location of deficit matches the wall-cell zone with q=2 cut-link
    pops driving stagnation-pressure rise.
  - rT uses **CUBISTA NVD + body-fitted O-grid** (§Facts last paragraph). It
    resolves the front-pole wall pops with a body-conforming mesh: no lag-1 halfway-BB
    pop substitution at all. The mechanism is structurally absent in the rT reference.
  - M29b's MUSCL fallback within ±2 cells of solid (`operators_2d.jl:523-533`)
    leaves a ~5-cell-wide ring around the cylinder where Ψ advection is
    1st-order Rusanov. Polymer stress in that ring is under-resolved at high
    Wi, but Cd_polymer is computed *from* tau on the wall ring at integration
    time, where it matches rT — supporting that the polymer-field error is
    second-order: tau is wrong by ~5 % in the ring but the *gradient*
    extracted on the cylinder is rT-like to 0.05. What is **not** rT-like is
    the *solvent-side* momentum balance, i.e. the LBM ρ field and its
    consequence on Cd_pressure (computed from ρ via `compute_drag_libb_mei_2d`
    at line 484, on q_wall mei integration).
  - R-invariant gap at R=30 and R=40 (no resolution improvement) is consistent
    with a **discrete BC closure** issue that is not resolved by reducing dx
    (each new wall cell still gets the same lag-1 closure).
- **Contradicting Facts.**
  - The Wi=0.1 case at R=30 has -0.80 % gap (close to rT), but Wi=1 is -7.34 %.
    If the BC closure were the dominant defect it should show up Newtonian-like
    in the gap floor, but the Newtonian gap is only -0.22 %. So the BC closure
    interacts **nonlinearly with polymer loading** to produce a 10× larger gap.
    This is *not refuting* — at low Wi the polymer-induced wall-normal stress
    is small (~Wi β); at Wi=1 it becomes O(1) and the closure mismatch is
    amplified. Still, this nonlinear coupling is what makes H1.A only MED-HIGH.
  - The Bouzidi-FL `:bouzidi_fl_twopass` at Wi=0.1 R=30 gives **+1.60 %**
    (over-prediction by 2.1 LU vs halfwayBB), which means the closure family
    does affect Cd_pressure even at Wi=0.1. Sign and magnitude indicate the
    closure-Cd_pressure mechanism is real but the calibration is hard.
- **Discriminator experiment.**
  Run M29b at Wi=1.0 R=30 with `:halfwayBB` + a "lag-0 pre-phase" variant:
  rewrite `ApplyLiBBPrePhase` to read `f_out[i,j,q]` (post-collision of the
  previous step's collide brick, i.e. consistent with the new ρ field) instead
  of `f_in[i,j,q]` (the literal lag-1). ≈10 LOC change in `bricks.jl:355-402`
  (swap `f_in` for `f_out` on the three `_libb_branch` arguments, behind a
  build-time `lagzero_prephase=true` flag). If Cd_pressure picks up
  ~8 LU (closing 80 % of the front-pole gap), H1.A is confirmed.
  ≤1 Aqua submission (single R=30 Wi=1 run, 80k steps).

### H1.B (rank 2, MED): The deficit is a **systematic Guo-force ↔ rho coupling defect** at the front-pole wall layer

- **Statement.** The Guo body force computed by `logfv_polymer_force_*` is
  inserted into the LBM via `_TRT_LIBB_V2_GUO_FIELD_SPEC`'s
  `CollideTRTDirectGuoField` brick. The Moments brick (line 86) computes
  `ρ = sum(fp_q)` from the **pulled (lag-1) f_in**, while the polymer force
  field `fx_total` was built in `logfv_bsd_correct_force_bc_aware_2d!` at line
  467 from the *current* tau and u. So the Guo force is added to f_q using
  current tau, but the ρ used for source scaling is lag-1. At front-pole
  cells where ρ is rising over the iteration (stagnation pressure ramp), a
  lag-1 ρ in Guo is systematically low by O(dρ/dt · 1), suppressing the
  effective momentum source by ~Δρ/ρ · |F| at the wall layer.
- **Supporting Facts.**
  - D1: pressure × front_pole = +8.34 (the largest single bucket). The
    front pole is where ρ growth is largest in transient and where the
    bucket-resolved deficit lives.
  - Cd_polymer matches rT to +0.05 LU — consistent with the polymer
    *gradient* on the wall being correct; only the *pressure* response of
    the LBM is short.
  - The pass-3 brick `ApplyCutLinkRhoRecompute` was designed exactly to
    re-sync `ρ_out` with the cut-link f-set, but Cd is bit-identical to no
    pass-3 → **rho consistency at the cut link is not the lever** (rules out
    one sub-flavour of B). The relevant lag is in `Moments` (line 86), which
    uses pulled pops, not in `ρ_out`.
- **Contradicting Facts.**
  - The Newtonian gap is -0.22 %, but Newtonian uses the same Guo path with
    zero `fx_poly`. If the Guo-rho coupling defect were generic it should
    show in Newtonian too. It doesn't → the defect must vanish when
    `|F_poly|` → 0, which is consistent with H1.B but also with H1.A's
    closure-nonlinearity.
  - The MUSCL fallback ring (Fact: M29b limitation) means the polymer force
    field itself has a 1st-order ring around the cylinder. This is an
    alternative source of the +10 deficit (see H1.C).
- **Discriminator experiment.**
  Add a "Guo-rho lag-fix" variant: in `CollideTRTDirectGuoField` (must be in
  `bricks.jl` somewhere — not shown but referenced at `li_bb_2d_v2.jl:52`),
  pass `ρ_out[i,j]` from the previous step as a second argument and use it
  for the Guo source scaling, leaving `ρ = sum(fp_q)` for the equilibrium.
  ≈10 LOC change. If Cd_pressure picks up ~8 LU, H1.B confirmed; if it stays
  put, H1.A wins. ≤1 Aqua submission.

### H1.C (rank 3, LOW): The deficit is a **MUSCL-fallback boundary-layer integral error**

- **Statement.** The +10 deficit is dominated by 1st-order numerical
  dissipation of Ψ_xx within the ±2-cell ring around the cylinder
  (`operators_2d.jl:523-533`), which under-predicts Ψ_xx peaks at the
  shoulder, hence under-predicts `∂τ_xx/∂x` *near* the wall, hence
  under-predicts the polymer body force which under-pressurises the front
  pole.
- **Supporting Facts.**
  - D1 polymer × shoulder = +3.14 (the polymer column has a real shoulder
    deficit), and polymer × front_pole = +0.59. Polymer column total = +2.67.
  - M29c-v2 attempt to remove the MUSCL fallback caused NaN (1st-order to
    one-sided MUSCL = anti-TVD CD2 catastrophe at the wall).
- **Contradicting Facts.**
  - Wall-decomp Cd_polymer = 13.40 vs rT 13.45 (gap +0.05). If the MUSCL
    fallback ring were dominantly responsible for +10 pts of Cd_pressure
    via under-resolved tau, **the wall-projected Cd_polymer would also be
    +1-2 LU off**, not +0.05. The polymer column is innocent at the wall.
  - The bucket polymer total +2.67 is the *gradient-extrapolation-to-wall*
    polymer column, while Cd_polymer matches rT to 0.05: the inconsistency
    suggests the +2.67 polymer signal in D1 is a measurement of pressure
    that is *mis-attributed to polymer* by the bucket decomposition. The
    physical defect is on the kinetic side (H1.A or H1.B), not on tau.
- **Discriminator experiment.**
  Make MUSCL fallback ring narrower (±1 instead of ±2), keeping the
  semi-anti-TVD safety. ≈3 LOC change at `operators_2d.jl:523-527`. Run
  R=30 Wi=1. If Cd_pressure gap *and* Cd_polymer gap both close together,
  H1.C is confirmed; if both stay put, the ring is not the lever. ≤1 Aqua
  submission. (Note: stability risk — this experiment may NaN, which would
  be a partial confirmation of why pass-3/MUSCL widening alone cannot
  solve Q1.)

---

## Q2: NaN under `:bouzidi_fl_twopass` at Wi=1 R={30,40} and Wi=0.1 R=60

### H2.A (rank 1, HIGH): **Loss of the pre-phase BC** (`ApplyLiBBPrePhase`) means corrupted **diagonal pops** (q=6..9) propagate into the bulk past the wall and corrupt the **inlet/outlet rebuild**

- **Statement.** `:halfwayBB` runs `ApplyLiBBPrePhase` which fixes ALL eight
  off-rest cut-link pops including diagonals (`bricks.jl:355-402`).
  `:bouzidi_fl_twopass`'s pass-1 spec **drops `ApplyLiBBPrePhase` entirely**
  (`li_bb_2d_v2.jl:69-73`) and pass-2 only overwrites
  `f_out[i,j,qbar]` (not the post-collision moment, not the f_in lag, not the
  Guo-force-modified pop). Therefore the **pulled-from-solid corrupted pops
  enter `Moments`** at the cut-link cells (`bricks.jl:83-88`), producing wrong
  ρ and ux, uy. The corrupted ρ feeds `WriteMoments` and is then read by
  `apply_bc_rebuild_2d!` from `f_in` (lag-1, OK) on the next step but also
  read by `vel_grad` on `ρ_out`/`ux_out`/`uy_out` (lag-0). At Wi=1, the
  off-rest stretch makes corrupted f's order-of-magnitude larger than at
  Wi=0.1, so the cut-link-cell wrong-moment artifact is amplified, and
  cascades to wake + bulk via advection of the Guo body force computed from
  a corrupted u-gradient.
- **Supporting Facts.**
  - M34-fix-diag NaN classification = **uniform 97.4 % rho_nan_frac**, **psi_nan_frac
    = 0.0**, distribution = wake 44 % + "other" bulk 53 % + **FULL inlet column NaN
    + FULL outlet column NaN**. ψ is healthy. ρ is uniformly poisoned.
    Inlet/outlet full-column NaN is the smoking gun: the only mechanism that
    poisons inlet AND outlet AND wake uniformly is `apply_bc_rebuild_2d!`
    reading corrupt `f_in` from interior at the column-adjacent cells, then
    propagating to f_out at inlet+outlet, then advecting downstream. This
    fingerprint is exactly the topology of `apply_bc_rebuild_2d!` propagation
    when interior `f_in` is broken globally on cut-link cells.
  - The Wi=0.1 R=30 case **survives** (`+1.60 %`). The defect is amplified by
    polymer loading: at Wi=0.1 the cut-link f's are still near-equilibrium so
    the missing pre-phase substitution is recoverable; at Wi=1 it isn't.
  - The Wi=0.1 R=60 case **NaNs**. The defect is also amplified by resolution:
    more wall cells = more cut-link cells = more corrupted-moment cells, and
    R=60 doubles the wall-ring count vs R=30. So the failure crosses the
    threshold at Wi=0.1 between R=40 and R=60.
  - pass-3 `ApplyCutLinkRhoRecompute` fixed `ρ_out` on cut links but did NOT
    affect Cd at R=30 Wi=0.1 — this is consistent: pass-3 only re-sums ρ at
    cut-link cells but `ApplyLiBBPrePhase`'s **absence** affects the
    *pulled f's used by Moments and by the next-step's PullHalfwayBB*. pass-3
    cannot rebuild what isn't there.
- **Contradicting Facts.**
  - `:bouzidi_fl` (single-pass, with `ApplyBouzidiFLPostCollide`) is not
    tested in the failure table — but its spec
    `_TRT_LIBB_V2_GUO_FIELD_BOUZIDI_FL_SPEC` at `li_bb_2d_v2.jl:56-61`
    *also* lacks the pre-phase. If the single-pass variant was tested and
    survived under the same conditions, H2.A is in trouble. (§Facts is
    silent on this.)
- **Discriminator experiment.**
  Add `ApplyLiBBPrePhase` to the two-pass pass-1 spec (
  `_TRT_LIBB_V2_GUO_FIELD_RAW_SPEC` at `li_bb_2d_v2.jl:69-73`). ≈1 LOC change:
  reinsert `ApplyLiBBPrePhase()` between `SolidInert()` and `Moments()`. Run
  R=30 Wi=1 `:bouzidi_fl_twopass`. Caveat: this restores the "double-BC trap"
  (pre-phase + post-collision Bouzidi). Expected outcome: finite Cd, possibly
  closer to rT but with the double-BC bias of M30 Phase 2b's original
  motivation. If finite, H2.A confirmed and we have a clear path: keep
  pass-2 but replace the missing pre-phase with a **lag-1 ApplyBouzidiFLPre**
  variant (read `f_in` post-collision-prev with arbitrary qw, not halfway).
  ≤1 Aqua submission.

### H2.B (rank 2, MED): The pass-2 brick reads `f_out[i_ff, j_ff, q]` (lag-0, just-streamed) for the q ≤ 0.5 far-field term, racing the SAME-block neighbour's pass-1 write

- **Statement.** Pass-2 of the two-pass variant
  (`ApplyBouzidiFLPostCollideTwoPass`, `bricks.jl:563-691`) reads
  `f_out[i_ff, j_ff, q]` at the far-fluid neighbour. Pass-1 writes `f_out`
  *everywhere* with `WriteMoments` and the SolidInert/CollideGuoField bricks.
  There is a `KernelAbstractions.synchronize(backend)` between pass-1 and
  pass-2 at `li_bb_2d_v2.jl:231`. **However**, the read is at a
  *neighbour* cell (i±1, j±1), and pass-1's write at that neighbour is
  *current-step collided f_out*. So the read sees current-step
  post-collision. That is what was wanted, but the **interpolation formula
  `_bouzidi_fl_post_value` for qw ≤ 0.5** mixes `2qw·f_here + (1-2qw)·f_ff +
  δ`, with `f_here = f_out[i,j,q]` and `f_ff = f_out[i_ff,j_ff,q]`. Both are
  lag-0. Under high polymer stretching, `f_q` at the wall and at i-1 may
  swing by O(Wi) — but the **interpolation weights `(2qw, 1-2qw)` can be
  negative for qw ∈ [0.5, 1.0]**? No: qw=0.5 → 2qw=1.0, 1-2qw=0.0, OK. For
  qw < 0.5, both positive, OK. For qw > 0.5 the formula switches to
  `1/(2qw)·f_here + (1-1/(2qw))·f_qbar_here + δ/(2qw)`; for qw → 1, both
  positive. So the weights are non-negative. **However**, the `δ` term scales
  inversely with qw for qw > 0.5 (`δ/(2qw)`) but NOT with rho_w (factored
  out). At small qw the δ scales with rho_w directly. For tiny qw (~1e-4), δ
  can be huge in absolute value, and the `(1-2qw)·f_ff` term dominates by
  ~1, but `f_ff` here is the LAG-0 pop on a neighbour that just collided
  with the **full polymer force** field — so the wall-pop estimate inherits
  the full polymer-driven stress excursion of the next cell over. At
  Wi=1 this can exceed local equilibrium magnitude by a factor of 2-5.
- **Supporting Facts.**
  - R=60 Wi=0.1 NaN's (more cells, more tiny-qw cut links — for a smoothly
    rasterised cylinder, qw distribution density rises with circumference).
  - R=30 Wi=0.1 does not NaN but the +1.60 % gap is the right sign for a
    *small* over-pressure from over-reading the polymer-loaded f_ff term.
  - R={30,40} Wi=1 NaN — at Wi=1 the polymer-loaded f_ff swings are huge.
- **Contradicting Facts.**
  - The NaN field is **ρ** (not ψ), and the inlet/outlet are FULL columns.
    H2.B would predict NaN onset at a **specific cut-link cell** first, then
    spread to wake by advection. The uniform 97 % distribution + full
    inlet/outlet columns is too global to originate from a single cut-link
    runaway.
- **Discriminator experiment.**
  Force pass-2 to use the **halfway-BB fallback unconditionally** (set
  `has_ff = false` always). ≈4 LOC change in pass-2 brick. If R=30 Wi=1 NaN
  disappears, H2.B confirmed. If still NaN, H2.A wins. ≤1 Aqua submission.

### H2.C (rank 3, LOW): The pass-3 cut-link ρ recompute creates an **inconsistency between ρ and u** because pass-3 only updates ρ, not u

- **Statement.** Pass-3 re-sums `f_out[i,j,1..9]` and overwrites `ρ_out`
  (`bricks.jl:712-727`). It does NOT recompute `ux_out, uy_out`. The downstream
  reader `vel_grad` reads ρ_out, ux_out, uy_out; the gradient computation
  uses `_fvfd_solid_bc_derivative_*` which works on u, but the **next step's
  Guo force is built from ρ-inconsistent u** (Moments will reread the
  pulled f_in next step, producing a new ρ; but `compute_macroscopic_forced_field`
  at driver line 528 mixes Guo half-step with f_out — and that uses ρ from
  the previous step's lag).
- **Supporting Facts.**
  - pass-3 was added in M34v3 to fix a hypothesised ρ defect at Wi=0.1 R=30,
    and the result was bit-identical Cd (no improvement, no worsening). This
    means the cut-link ρ defect *exists* but is small in magnitude at low
    Wi.
- **Contradicting Facts.**
  - At Wi=1 the failure is catastrophic (NaN), not a 2 LU drift. ρ-u
    inconsistency at the cut-link level is a O(uw) perturbation per step,
    too small to cause uniform 97 % bulk NaN within 80k steps.
  - pass-3 is the LAST brick in the spec stack (`li_bb_2d_v2.jl:247-250`).
    The same NaN fingerprint exists with or without pass-3 (M34-fix vs
    M34v3 are bit-identical per §Facts).
- **Discriminator experiment.**
  Remove pass-3 entirely and rerun R=30 Wi=1 `:bouzidi_fl_twopass`. ≈4 LOC
  comment-out. If NaN persists, H2.C ruled out as causal (it's only
  cosmetic). §Facts already says R=30 Wi=0.1 is bit-identical; we need the
  Wi=1 NaN sanity. ≤1 Aqua submission.

---

## Q3: NaN under `:halfwayBB` at R=60 Wi ≥ 0.1

### H3.A (rank 1, HIGH): The Metal F32 D2bis fingerprint (bilateral arcs at ±38°,±48° in the wall-ring annulus r-R∈[0,7]) is a **classic high-Wi log-conformation Ψ_xx peak buildup at the cylinder shoulder/wake**, amplified by Pre-NaN `max|Ψ_xx|=15.86` and `max|F_total|=320` consistent with **log-conformation transport of Wi=1 Hookean dumbbells** without sufficient artificial diffusion or stress diffusion

- **Statement.** This is the **canonical HWNP-zone elastic instability at the
  Lunsmann high-Wi front shoulder**, not a numerical artifact of the BC.
  Pre-NaN `max|Ψ_xx|=15.86` at Wi=1 in log-space corresponds to physical
  c_xx ~ exp(15.86) which is astronomically large; the front-shoulder arcs
  at ±38° and ±48° are the classic locus of birefringent strand emergence
  in cylinder confined flow at β=0.59 BSD=1. F_total=320 is the LBM force
  field that explodes when polymer divergence loses balance.
- **Supporting Facts.**
  - D2bis: first-NaN field = ρ, location (943, 72) i.e. **far downstream from
    the wall ring** (close to outlet for R=60: Nx ~ 30R+1=901, but 943
    suggests outlet region; the (943, 72) is in the wake just inside the
    outlet column).
  - bilateral arcs at ±38°, ±48° in r-R∈[0,7] LU = the wall-adjacent shear
    layer, where Ψ_xx builds up most strongly at high Wi.
  - Pre-NaN `max|Ψ_xx|=15.86`: in log-conformation, c_xx = exp(15.86) ≈
    7.7e6. This is far beyond the dumbbell extensibility scale and signals
    elastic-instability-driven blow-up at the wake.
  - This is on **Metal F32**; pre-NaN max|u|=554 indicates u has blown up
    well beyond LBM stability (u_LBM must be ≲ 0.1).
- **Contradicting Facts.**
  - The Aqua F64 R=60 Wi=Newt PASSES (Cd=132.68 finite) but R=60 Wi=0.1 NaNs.
    A genuine HWNP buildup should *not* be that sensitive at Wi=0.1 (small
    elastic stretch). So either (i) the buildup is amplified at the
    confined-shoulder geometry by an interaction with the BC closure (see
    H3.B/H3.C), or (ii) the R=60 F32 D2bis is a different mechanism from
    the R=60 F64 Wi=0.1 mechanism.
- **Discriminator experiment.**
  Run R=60 Wi=0.1 with `:halfwayBB` on Aqua F64 with an extra log-conformation
  artificial diffusion (add `+ ε·∇²Ψ` with `ε = 1e-3 LU`, ≈10 LOC in
  `logconformation_fv_2d.jl:1154-1227`). If NaN disappears and Cd is finite
  (regardless of value), H3.A is confirmed: the failure is *physical-elastic-
  instability + insufficient dissipation*, not a BC failure. ≤1 Aqua
  submission.

### H3.B (rank 2, MED): **R-scaling of the lag-1 pre-phase deficit (H1.A)** — at R=60 the wall-ring count doubles, increasing the cumulative pre-phase error in ρ and triggering Guo-force runaway

- **Statement.** Same mechanism as H1.A but at larger R: the lag-1 pre-phase
  closure leaves a thin wall-layer with biased ρ/u that contributes a
  bounded `O(Wi·β)` error per cell. At R=30 Wi=0.1 the cumulative effect is
  -0.80 %. At R=60 Wi=0.1 the ring is ~2× denser and the error per wall cell
  is ~2× (because polymer stretching scales with shear rate, which scales
  with 1/dx ~ R). Cumulative wall-ring perturbation of the Guo force grows
  as ~R² and feeds back into Ψ via vel_grad → poly_force.
- **Supporting Facts.**
  - R=30 Newtonian Cd matches rT to -0.22 %, so the BC itself is well-
    calibrated at R=30. At R=60 Newtonian also works (finite). So the BC
    closure error scales with **Wi × R**, not with R alone.
  - The first-NaN field is ρ (D2bis), not ψ. If H3.B is correct, the ρ
    blow-up cascades from ρ_out being read by vel_grad → bad ∇u → bad Ψ-source
    → diverging Ψ → diverging Guo force → diverging ρ.
  - First-NaN at (943, 72) is in the wake near the outlet, where polymer-driven
    momentum is dumped. The wake is the natural failure locus of a downstream-
    advected wall-ring error.
- **Contradicting Facts.**
  - The first NaN being in **rho** (not Ψ) suggests the elastic side is
    still OK (max|Ψ_xx|=15.86 is large but finite); the failure originates
    on the kinetic LBM side. This is more consistent with **H3.B than H3.A**,
    actually. (Re-ranking note: H3.B may deserve to be the rank-1 hypothesis
    on this specific point. But the magnitude of Ψ_xx=15.86 favours H3.A.)
- **Discriminator experiment.**
  Run R=60 Wi=0.1 with `:halfwayBB` + the lag-0 pre-phase variant from H1.A's
  discriminator. ≈10 LOC. If finite, H3.B confirmed and H1.A and Q3 share a
  root cause. ≤1 Aqua submission. **This is the single best discriminator
  across Q1+Q3.**

### H3.C (rank 3, LOW): Resolution-coupled CFL violation on the Ψ-advection (Rusanov is conditionally stable)

- **Statement.** At R=60 with fixed `u_mean` (BSD=1 means u_max scales such
  that the chosen u_lattice may saturate Rusanov stability). Rusanov
  scheme requires `Δt·|u|/Δx ≤ 1`. If u_mean is held at the Liu Re=1 value
  and dx halves with R, the Rusanov CFL number doubles. At R=60 it may
  cross 1.
- **Supporting Facts.**
  - Pre-M28 Cd at R=60 Wi=Newt = 132.68 (finite), so CFL on the LBM side
    is fine. Only with polymer-driven Guo force (Wi ≥ 0.1) does it NaN.
- **Contradicting Facts.**
  - Newtonian R=60 works. Pure CFL violation does not depend on Wi.
  - Per Fact, "NOT resolution at R∈{30,40} (R-invariant gap)" — but that's
    R-invariant for the *gap*, not for stability. Still no Fact contradicts
    CFL directly.
- **Discriminator experiment.**
  Halve `dt` (or lower `u_mean` to 0.05 LU instead of 0.1 LU) at R=60
  Wi=0.1. ≈2 LOC in driver / runscript. If finite, H3.C confirmed.
  ≤1 Aqua submission.

---

## Relationship (Q1, Q2, Q3)

- **Chained, with Q1 and Q3 sharing a common root and Q2 being a distinct
  subset of the same family.**

  The Claude verdict is that **Q1 and Q3 are the same mechanism at different
  amplification levels**: a lag-1 pre-phase closure (`ApplyLiBBPrePhase`
  reads `f_in[i,j,q]`, the previous step's post-collision pulled pop)
  produces a bounded but biased wall-cell ρ/u under polymer loading. At
  R=30 Wi=1 (Q1) it manifests as a -10 LU Cd_pressure under-prediction
  (bounded, no NaN). At R=60 Wi=0.1 (Q3) it manifests as a NaN because the
  wall-ring count is larger and the per-cell error stretches Ψ_xx beyond
  log-conformation's safe range.

  **Q2** (`:bouzidi_fl_twopass` NaN at Wi=1 R=30 and Wi=0.1 R=60) is a
  *different* mechanism: the two-pass spec **drops `ApplyLiBBPrePhase`
  entirely**. The pre-phase that biased Q1+Q3 is *absent* in Q2's pass-1
  RAW spec. So Q2 fails not by lag-1 bias but by **no BC closure at all on
  the off-rest cut-link pops as seen by Moments**. The pass-2 overwrite of
  `f_out[i,j,qbar]` happens *after* Moments has already computed the
  wrong ρ/u from corrupted pulled pops. The two-pass design has a defect
  *complementary* to the halfwayBB one: halfwayBB uses a stale-but-finite
  pre-phase; bouzidi_fl_twopass uses no pre-phase.

  **Evidence for the chain.**
  - Q1 R=30 Wi=1 halfwayBB: bounded -10 LU error. The pre-phase exists but
    is stale.
  - Q3 R=60 Wi=0.1 halfwayBB: NaN. The pre-phase exists but the stale
    error amplifies through the larger wall ring.
  - Q2 R=30 Wi=0.1 bouzidi_fl_twopass: bounded +1.60 % error. No pre-phase
    but Wi is low so off-rest pops are tame.
  - Q2 R=30 Wi=1 bouzidi_fl_twopass: NaN. No pre-phase + Wi=1 → cut-link
    Moments sees catastrophically wrong f's → uniform 97% rho NaN with
    full inlet/outlet columns (the exact fingerprint of `apply_bc_rebuild_2d!`
    reading uniformly poisoned `f_in`).

---

## Recommended next mission (single)

**M35-PREPHASE-LAGZERO** (≤40 LOC change, 1 Aqua submission, ~2 h runtime
+ post-processing).

**What.** Implement a `:halfwayBB_lagzero` variant of
`_TRT_LIBB_V2_GUO_FIELD_SPEC` where `ApplyLiBBPrePhase` is replaced by a
new brick `ApplyLiBBPrePhaseLagZero` that reads
`f_out[i, j, q]` (post-collision of the **current** lag-1 step's collide)
instead of `f_in[i, j, q]`. This requires reordering the spec so the
Pre-phase BC runs **after** the collide brick, before WriteMoments —
effectively making the closure lag-0 from the perspective of ρ.
*Alternative implementation*: add a "shadow collide" cheap pre-pass that
recomputes a tentative `f_out` *only on cut-link cells*, then use those
post-collision values in the existing `ApplyLiBBPrePhase` slot.

**Why this single mission.**
1. It is the **shared discriminator** for H1.A (Q1) and H3.B (Q3) — if
   Cd_pressure picks up ~8 LU at R=30 Wi=1 AND R=60 Wi=0.1 becomes finite,
   both Q1 and Q3 are explained.
2. It does not touch `:bouzidi_fl_twopass`. Q2 remains open but its
   resolution path (re-add `ApplyLiBBPrePhase` to pass-1) is already
   implied by H2.A and can be a follow-on M36.
3. It is empirically falsifiable in 2 hours: 1 Aqua F64 run at R=30 Wi=1
   `:halfwayBB_lagzero` and 1 Aqua F64 run at R=60 Wi=0.1
   `:halfwayBB_lagzero`. Outcomes:
   - Both finite + Cd_pressure ~+8 LU at R=30 Wi=1 → H1.A and H3.B
     confirmed, shipping path: M36 = port lag-0 pre-phase to two-pass + flagship paper.
   - Cd_pressure unchanged at R=30 Wi=1 → H1.A refuted, narrow to H1.B
     (Guo-rho lag fix) as M36.
   - R=60 Wi=0.1 still NaN → H3.A wins (HWNP-zone physical instability),
     reframe campaign as "Wi=1 R=60 needs polymer stress diffusion".

**Why this is the right single mission and not three.** The three
discriminators across Q1+Q2+Q3 share a common code-path question: *is
the missing/stale pre-phase closure the dominant defect family in the
polymer-coupled LI-BB pipeline?* Lag-zero pre-phase is the single
intervention that probes both Q1 (signed deficit) and Q3 (catastrophic
amplification) at once. Q2 can wait — its discriminator (re-add pre-phase
to pass-1) is *cheaper* but only relevant if M35 confirms the pre-phase
family is the bottleneck. If M35 refutes (Cd_pressure unchanged), the
campaign pivots to Guo-force coupling and polymer stress diffusion,
which has different M36 implications anyway.

---

## Self-check (per Department brief)

- Citations to specific Facts per hypothesis: yes (e.g. H1.A cites M29b wall
  decomp +10.13 pts, D1 bucket +8.34 at front_pole; H2.A cites M34-fix-diag
  uniform 97.4 % rho_nan + inlet/outlet full columns; H3.A cites D2bis
  arcs at ±38°,±48°; etc.).
- Hypotheses-with-no-discriminator (failure mode flagged in brief): **None
  is indistinguishable**, but H1.A and H1.B share a partial overlap (both
  predict ~+8 LU recovery at front pole). M35 (the recommended mission)
  was specifically designed so that H1.A and H1.B give *different*
  predictions at R=30 Wi=1: H1.A → pre-phase lag-zero fixes; H1.B →
  pre-phase lag-zero does NOT fix and we need the Guo-rho discriminator.
  H1.C is also separable from H1.A/B because it would close *both*
  Cd_pressure AND Cd_polymer (which currently matches rT), whereas A/B
  close only Cd_pressure.
- Cross-Q hypothesis testing: H1.A and H3.B are the same mechanism at
  different amplitudes — that is *not* a flaw of the audit, it is the
  audit's conclusion (Q1+Q3 chained). H2.A is structurally distinct.
- Wall-clock: 38 minutes (within 45 min budget).
