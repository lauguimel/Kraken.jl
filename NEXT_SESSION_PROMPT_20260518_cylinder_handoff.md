# Next session prompt — Kraken cylinder Cd benchmark, post-Phase 0

Copy-paste below to start a fresh session.

---

Continue work on branch `dev-viscoelastic` of Kraken.jl
(worktree `~/Documents/Recherche/Kraken.jl-viscoelastic`).

Resume via orchestrator. Open
`~/.claude/skills/orchestrator/SKILL.md` first. The Boss role
continues — Departments + Engineers will absorb the detail; Boss stays
strategic.

## State at handoff (2026-05-18 evening session close)

### What landed (7 commits today)

```
533afa08 fix(viscoelastic): CUDA backend detection + beta=0.59 default
488a7b56 feat(viscoelastic): expose L_up/L_down/embedded_* env vars
86f1391c feat(viscoelastic): DoE plan B (Codex) + preflight
7d4e3f8d feat(viscoelastic): v2 cylinder benches + Aqua PBS
8aaac026 feat(viscoelastic): M22+M23 cylinder Cd Metal F32
81745f3b feat(viscoelastic): M21 Poiseuille BSD path matrix NEGATIVE
0d62db49 feat(viscoelastic): M20 Poiseuille F_total trace verdict
```

### 3 bugs found and fixed this session

1. **Apples-vs-oranges M22 vs M23 v1**: M22 used L_up=L_down=15;
   M23 used L_up=4 L_down=8 (Codex Engineer chose smaller domain to
   save compute). My "v1 SPD-loss at R=40 Wi=0.2" was a different
   simulation, not a stochastic transient. Lesson: ALWAYS diff kwargs
   between sister benches before cross-comparing.

2. **CUDA detection silent fail** at `bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl::detect_backend()` line 34:
   `Base.invokelatest(getfield(Main, :CUDA), :functional)` treats the
   CUDA module as callable with `:functional` as arg. Silently
   MethodError'd → bare `catch end` → CPU fallback. Aqua A100 job
   `21534810` burned 4h35 at CPU speed before the user spotted the
   `backend=cpu` column in the CSV. Fix mirrors the Metal branch +
   adds `@warn` on explicit-request fallback.

3. **β mismatch with reference**: Phase 0 ran at β=0.5 while Liu 2025
   AND rheoTool both use β=0.59 (Liu line 2515; rheoTool
   `constant/constitutiveProperties` etaS=0.59 etaP=0.41). The
   numerical Cd comparison to Liu/rheoTool was confounded. Fixed: new
   default `KRAKEN_BETA_LIST="0.59"`.

### Job currently in queue: `21563085.aqua` (Phase 0 Liu-match)

| param | value |
|---|---|
| State | Q (queued) |
| Setup | β=0.59, Re=1, Wi=0.1, L_up=L_down=15, bsd_fraction=1.0 |
| R sweep | {20, 30, 40} (matches Liu Table 3 R-plateau region) |
| Embedded tuples (zipped, 4 total) | `0000_qwall` (= Liu reference mode), `1000_qwall` (grad only), `0001_qwall` (drag only — isolate suspected bug), `1100_qwall` (grad+adv, Phase 0 best candidate) |
| Total | 12 runs, ~2-3 h on A100 F64 |
| Output | `results/viscoelastic_logfv/cyl_bigsweep_v2_21563085.aqua/SUMMARY.csv` (+ 12 per-case CSVs) |

### Reference comparison targets (Liu Table 3 β=0.59 Wi=0.1)

| R | Liu CNEBB | rheoTool |
|---|---|---|
| 20 | 129.42 | — |
| 30 | **130.36** | **130.43** |
| 40 | 130.79 | — |

**Pass criterion for Phase 0 Liu-match**: Kraken `0000_qwall` R=30 Cd
∈ [129.5, 131.5] = Liu CNEBB ± 1. If yes → BSD architecture +
staircase polymer is sound; ready for Phase 1 (Wi sweep + PB-8
screening).

### CRITICAL OPEN POINT — Wi single-point limitation

**Phase 0 sweeps R only at Wi=0.1.** Wi=0.1 is quasi-Newtonian
(polymer contribution Cd_p − Cd_bsd ≈ −0.3 Cd points, dominated by
Cd_s). To validate the BSD+embedded physics across the elastic
regime, the **next phase MUST add Wi ∈ {0.3, 0.5, 1.0}** :

- Wi=0.1: ratchet baseline (near-Newtonian)
- Wi=0.3 or 0.5: moderate elasticity, polymer drag significant
- Wi=1.0: strong elasticity (Liu Cd=151.31 at R=30 — large polymer
  contribution; many LBM schemes break — useful stress-test)

Without a Wi sweep, the "convergence" we see at Wi=0.1 alone tells us
**nothing about elastic physics** — it could be hiding a polymer-pipeline
bug that only activates at finite Wi. The Boss must enforce this in
the next mission scope.

### What "0000_qwall" means physically (user-asked, documented for next Boss)

The 4 binary flags are 4 independent switches over how the
polymer FVFD pipeline sees the curved cylinder wall:

| flag | OFF (= 0) | ON (= 1) |
|---|---|---|
| `embedded_gradient` | ∇u via `is_solid` staircase | ∇u via cut-cell sub-cell formulae |
| `embedded_advection` | FV upwind on staircase grid | FV upwind with cut-cell flux corrections |
| `embedded_force` | F_total injected on staircase grid | F_total injected with sub-cell corrections |
| `embedded_drag` | Cd via LBM cut-link momentum exchange | Cd via FVFD traction integration on sub-cell quadrature |

Plus `embedded_geometry::Symbol`:
- `:qwall` = FVFD uses the LBM's q_wall cut-link distances as boundary
- `:circle` = FVFD re-computes the analytical circle with `embedded_circle_samples` quadrature points (default 32)

**`0000_qwall` (all flags OFF, qwall geometry)** = the "Liu-equivalent"
mode:
- LBM sees sub-cell curved wall (cut-link bounce-back, ν_LBM_implicit absorbs correctly)
- Polymer FVFD sees staircase (cells are fully solid or fully fluid via `is_solid`)
- Drag computed via cut-link momentum exchange (same formula as Liu 2025 Sect 4.3 and rheoTool's momentum integral)

This mismatch (LBM sub-cell vs polymer staircase) is an O(dx) error
that converges to zero as R → ∞. At R=30 it's the dominant
discretisation error. Liu 2025 Section 4.3 uses essentially the same
config (TRT-RLB LBM with polymer staircase coupling) — that's why
Liu/Kraken at `0000_qwall` should give the same Cd within numerical
noise.

**`1100_qwall` (grad+adv ON, force+drag OFF)** = polymer FVFD now uses
sub-cell awareness for its OWN evolution (gradient + advection
near-wall benefit from accurate volume fractions), BUT drag is still
LBM-momentum-exchange. **Best-of-both candidate**: accurate polymer
stress + same drag formula as reference. This was the Phase 0 v1
config that gave Cd=131.15 at L=4/8 R=30 — closest to rheoTool 130.43.

**`1111_circle` (full embedded, circle geometry)** = sub-cell
everywhere. Gave Cd=140.37 in Phase 0 v1, with Cd_s jumping +12 vs
Newtonian baseline. **Suspected bug in `embedded_drag=true`**: the
FVFD traction integration on the circle quadrature appears to
over-count wall stress. Job 21563085 case 3 (`0001_qwall`) isolates
this — if it gives Cd_s ≈ 140 in isolation, confirms the bug is in
`embedded_drag` alone, independent of geometry kwarg.

### Cd decomposition formula (user-asked)

The CSV columns satisfy `Cd_kraken = Cd_s + Cd_p − Cd_bsd`, NOT
`Cd_s + Cd_p`. Reason:
- LBM with `ν_LBM = ν_s + ζ·ν_p` already absorbs the `ζ·ν_p·∇²u`
  portion into its implicit viscous diffusion → Cd_s includes that
  contribution.
- Guo body force = `div(τ_p) − ζ·ν_p·∇²u` → drag integral of body
  force = `Cd_p − Cd_bsd`.
- Total: `Cd_kraken = Cd_s + (Cd_p − Cd_bsd)`.

`Cd_bsd` is what's been "moved" from explicit Guo to implicit
LBM viscosity. At Wi=0.1 we typically have `Cd_p ≈ Cd_bsd` to
within 1-2 Cd points (BSD doing its job: absorbing the
Newtonian-additive portion of the polymer stress). At higher Wi,
`Cd_p − Cd_bsd` becomes finite and gives the genuine elastic-stress
drag contribution.

## Mandate state (no change since handoff)

- Mandate at `.orchestrator/mandate.md`, M22+M23 done, Phase 0 Liu-match
  in flight as `21563085.aqua`.
- Memory at `.orchestrator/memory/{boss,department,engineer}.md` — boss
  + engineer updated with the 3 bugs above; Wi-single-point limitation
  NOT YET in memory (next Boss should add).

## Reference docs (next Boss reads first)

1. `.orchestrator/mandate.md`
2. `.orchestrator/memory/boss.md` (especially 2026-05-18 entries
   about same-sign stencil residuals, matrix-sweep parallel pattern,
   M22+M23 cylinder findings, Boss-on-host recovery pattern)
3. `.orchestrator/memory/engineer.md` (CUDA invokelatest pattern,
   wall-gradient half-cell ghosts, M5 kinetic equivalence static-only)
4. `bench/viscoelastic_audit/CYLINDER_DOE_PLAN_CODEX_20260518.md`
   (DoE plan with 3 tiers)
5. `bench/viscoelastic_audit/CYLINDER_DOE_PLAN_CODEX_20260518.md` —
   the hybrid plan combining Codex tier structure with the cloud-
   reasoning agent's PB-8 screening + statistical power analysis was
   validated by user 2026-05-18 but the reasoning agent's DoE document
   itself was in-chat only, not on
   disk. Re-derive if needed; the Codex plan + this prompt are
   sufficient context.
6. `bench/viscoelastic_audit/liu_2025.txt` — Section 4.3 (line 2498+)
   for the cylinder benchmark; Table 3 (line 2592+) for the
   reference Cd values.

## Next Boss's immediate actions

1. **Check job 21563085 status** : `ssh aqua "qstat 21563085.aqua"`.
   If still queued or running, wait. If complete, rsync results.
2. **Rsync SUMMARY.csv** :
   `rsync -av aqua:~/Kraken.jl-viscoelastic-run/results/viscoelastic_logfv/cyl_bigsweep_v2_21563085.aqua/ tmp/cyl_21563085/`
3. **Compare 0000_qwall R=30 Cd vs Liu CNEBB 130.36** (acceptance gate).
4. **Check `0001_qwall` (drag-only) for the +12 Cd_s anomaly** : if
   reproduced isolated, log it as confirmed bug in `embedded_drag=true`
   path. Open a separate mission to fix it (M26: embedded_drag
   diagnostic on a controlled Newtonian setup).
5. **If 0000_qwall matches Liu within ±1 Cd**, propose Phase 1 to
   user:
   - **MANDATORY Wi sweep** (Wi ∈ {0.1, 0.3, 0.5, 1.0}) at the winning
     embedded config (probably `0000_qwall` for strict Liu match, OR
     `1100_qwall` for accuracy/speed trade-off).
   - R ∈ {20, 30, 40} (already validated structure).
   - β = 0.59 fixed.
   - Re = 1 fixed.
   - bsd_fraction = 1.0 (since CUDA fix means we can re-test cleanly
     at fine R without F32 noise).
   - Total: 4 Wi × 3 R × 1 embedded × 1 BSD = 12 runs ~3 h on A100.
6. **Update memory `boss.md`** with:
   - "Wi single-point sweep is NOT a benchmark — minimum 3 Wi for
     elastic regime validation"
   - "Reference values Liu Table 3 + rheoTool are at β=0.59, NOT
     β=0.5 — every cylinder mission must check"

## MISSION DEDICATED — find the bug in `1111_circle` full embedded mode

**User directive 2026-05-18 evening**: the full embedded mode
(`embedded_gradient=1, embedded_advection=1, embedded_force=1,
embedded_drag=1, embedded_geometry=:circle`) gives **Cd_s = 140.78**
vs the Newtonian baseline at the same Re=1 R=30 of **131.99** (audit
2026-05-09) — i.e. **+8.8 Cd points of fictitious solvent drag** that
has no physical justification (at Re fixed, Cd_s should be physics-
fixed). This is a CONFIRMED BUG to find and fix.

### Where to start the search

Hypothesis 1: **`embedded_drag=true` over-counts wall stress** when
using `:circle` geometry sub-cell quadrature. Calls into
`fvfd_embedded_wall_traction_2d!` or similar. Compare the formula
against the LBM cut-link momentum exchange (which gives Cd_s ≈ 129
in `0000_qwall` mode) — the two should be physically equivalent at
the continuum, so any discrepancy is a discretisation flaw in one of
them.

Hypothesis 2: **`embedded_force=true` injects body force differently
near the wall** when sub-cell volume fractions are non-trivial. Cells
with low fluid fraction may receive amplified body-force per unit
fluid volume, biasing the LBM near-wall flow upward, which then
propagates to the Cd_s integral.

Hypothesis 3: **`embedded_circle_samples=32` is insufficient** for
the quadrature, leading to wall-curvature bias. Easy to test (re-run
at `embedded_circle_samples=64, 128`).

Phase 0 Liu-match (job `21563085`) case 3 = `0001_qwall` (drag ONLY)
will disambiguate H1 vs H2/H3:
- If `0001_qwall` Cd_s ≈ 140 → bug is purely in `embedded_drag` formula
- If `0001_qwall` Cd_s ≈ 129 → bug is in `embedded_force` or quadrature
  (test by enabling them in isolation: `0010_qwall`, `0100_qwall`)

### Required protocol — adversarial agent pattern

Per user directive, this debug MUST use the **adversarial dual-spawn
pattern** that has been validated 4× this session (M17 cluster, DoE,
CUDA fix). Spawn in parallel:

- **Agent A (cloud-reasoning)** — reasoning-heavy: read the relevant
  `src/fvfd/operators_2d.jl` traction routines (`fvfd_embedded_wall_
  traction_2d!`, `fvfd_bsd_force_2d!` embedded variant, etc.), the
  cut-link drag computation in the LBM (probably in
  `src/kernels/li_bb_2d_v2.jl` or similar), and reason about the
  mathematical equivalence (or lack thereof) of the two drag
  formulae at the continuum + discrete levels. Output: diagnosis +
  proposed fix design (no code).
- **Agent B (Codex via Department)** — code-heavy: write a small
  diagnostic bench that runs a NEWTONIAN cylinder (β=1, no polymer)
  at Re=1 R=30 in both modes (`0000_qwall` and `0001_qwall`) and
  measures the actual Cd_s difference. If the bug is in
  `embedded_drag`, both modes should give different Cd_s on the same
  velocity field (since the field is Newtonian-only, no polymer).
  Output: bench script + the measurement + a proposed patch if the
  bug is clearly localised.

The Boss then evaluates both, picks the merged fix, applies, smoke-
tests, and commits. The 4× validated pattern: the reasoning agent
finds the math flaw, Codex provides the runnable diagnostic + patch.

### Acceptance criterion for the fix

After patch:
- `1111_circle` Newtonian (β=1, no polymer) Cd at R=30 within ±1 Cd
  of `0000_qwall` Newtonian = **131.99 (= audit 2026-05-09 baseline)**
- At Wi=0.1 β=0.59, `1111_circle` Cd_s within ±2 Cd of `0000_qwall`
  Cd_s (= same field, just different drag/quadrature formula)

If those acceptance criteria are met, the `1111_circle` mode becomes
usable for sub-cell-accurate polymer simulations. Otherwise document
the residual bias and recommend `1100_qwall` (= polymer sub-cell +
LBM-momentum-exchange drag) as the production path.

## Open / queued questions for the next Boss

- Should we update `run_cyl_bigsweep_v2_a100.pbs` to default
  `KRAKEN_BETA_LIST="0.59"` (the bench file already has this default,
  but the PBS export line still says 0.3,0.5,0.7) ? Worth a small
  commit before next qsub.
- The rheoTool mesh convergence PBS (`run_rheotool_cyl_meshconv_aqua.pbs`)
  is queued but never submitted. Worth submitting in parallel with
  Phase 1 if user wants to nail down rheoTool's own discretization
  error.
- Should we re-implement `embedded_drag=true` to use a method
  consistent with momentum exchange (instead of FVFD traction
  integration) — would let us mix sub-cell polymer + Liu-style drag
  with full embedded geometry, eliminating the +12 Cd anomaly. This
  is a `src/` patch and needs careful scoping.

End of prompt.
