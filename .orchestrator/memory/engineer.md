# Engineer memory — Kraken.jl viscoelastic cavity spatial debug

Codebase do/don't and stack-specific traps. Read this before writing any
code. Initialised 2026-05-15.

## 2026-05-15 — GPU-only for non-trivial runs

Anything beyond a 50-line smoke test must run on GPU. CPU runs on the
log-FV cavity are pointless: substep cadence and BSD stencil are tuned
for GPU memory layout. Local: `KRAKEN_BACKEND=metal` (Float32, macOS
M-series). HPC: `KRAKEN_BACKEND=cuda` (Float64, Aqua A100/H100).

Tests under `test/` are allowed on CPU (<30 s budget).

**Why**: CPU log-FV cavity at N=64 takes hours and reveals nothing the
GPU version doesn't.

## 2026-05-15 — `tmp/` is .gitignored — do not commit results

Any CSV / .jls / fields snapshot from Aqua belongs under `tmp/` or
`results/`. Do not stage these. The verdict markdown that summarises
them goes under `bench/viscoelastic_logfv/` and IS committed.

**Why**: prior sessions cluttered the worktree with multi-MB binaries.

## 2026-05-15 — Confidentiality rules

- Never write `Claude`, `AI`, `LLM`, `Anthropic`, `assistant`,
  `Co-Authored-By` in source, comments, commits, scripts, or markdown.
- Conventional-commits message style (`feat:`, `fix:`, `refactor:`,
  `docs:`, `chore:`). No emojis.

**Why**: this repo will be made public; history will be audited.

## 2026-05-15 — Self-tests must isolate to `mktempdir()`

Bench-side self-tests (`--self-test` modes) must build all fixtures
inside `mktempdir() do dir ... end`. Never depend on `tmp/cavity_aqua_n64/`
or any other path under `tmp/` — that directory is `.gitignored` and
absent on a fresh clone or CI.

**Why**: ensures the self-test runs from a clean checkout; cleanup
guaranteed even on assertion failure. Found during M1.

## 2026-05-16 — Guo source / TRT collision entrypoints

The fused TRT + LI-BB + Guo step for the 2D log-FV cavity driver
lives at `src/kernels/li_bb_2d_v2.jl:115-127`
(`fused_trt_libb_v2_guo_field_step!`). The per-cell Guo source brick
is at `src/kernels/dsl/bricks.jl:145-199`, where the Guo prefactor is
`guo_pref = 1 − s_plus/2` (TRT plus-rate convention; not a
user-tunable knob). Any new BSD or body-force operator must reuse
this prefactor to stay consistent with the existing collision path.

**Why**: re-deriving the prefactor independently is a frequent
source of off-by-factor bugs. Found during M5-A.

## 2026-05-16 — No Π^{neq} accumulator exists yet in `src/`

As of 2026-05-16, no kernel under `src/` extracts the LBM
non-equilibrium momentum tensor `Π^{neq}_{αβ} = Σ_q c_qα c_qβ
(f_q − f_q^eq)`. Phase B of M5 must add it from scratch (proposed
location: `src/kernels/bsd_kinetic.jl` paired with the new BSD
kernel). Do not assume one exists — `grep` will return nothing.

**Why**: a future contributor browsing for "non-equilibrium" or
"Pi_neq" will find nothing; this note documents the absence so the
search is not repeated.

## 2026-05-16 — Kraken vs rheoTool wall BC on polymer fields

- `Ψ` (log-conformation): both Kraken and rheoTool use
  **zeroGradient** at all cavity walls. Kraken's is implicit via the
  upwind helper at `src/fvfd/operators_2d.jl:408-454` (returns
  `phi[i,j]` at walls).
- `τ` (polymer stress) at the wall row of the FD divergence used to
  build the body force: Kraken uses an **implicit one-sided quadratic
  3-point** extrapolation in `_fvfd_solid_bc_derivative_x_2d` and
  `_y` at `src/fvfd/operators_2d.jl:24-26 / 50-52`. rheoTool uses
  **linearExtrapolation** (2-point linear) on `τ` at the moving lid
  patch. The two conventions disagree only at the wall row; centred
  stencils in the bulk are identical.

**Why**: this is the most likely source of the M4 "54 % Guo vs FD"
discrepancy at cell (16, 63) (one row below the lid). Any future
polymer-stress wall BC work must keep these conventions straight.

## 2026-05-16 — `FVFDDomainBC2D` enum is trinary

`src/fvfd/specs.jl` defines `FVFDDomainBC2D` as exactly three cases:
`PERIODIC`, `OPEN`, `WALL`. The wall variant has no sub-flavours; the
extrapolation stencil is hard-coded inside the FD helpers. Adding a
new wall behaviour (e.g. linear vs quadratic extrapolation for the
polymer FD divergence) is therefore NOT done by adding an enum value;
it is done by adding a kwarg to the helper itself (e.g.
`polymer_wall_extrap::Symbol = :quadratic` defaulting to current
behaviour).

**Why**: avoids a future Engineer proposing an `enum` extension that
would be invasive across all drivers; the kwarg approach is the
proper surgical path.

## 2026-05-16 — Val-dispatched kwarg pattern for kernel constant-folding

`KernelAbstractions.@kernel` bodies cannot take kwargs natively. To
parameterise a kernel on a discrete choice (e.g. stencil variant)
without runtime branching cost, pass the parameter as
`Val{:tag}` rather than `Symbol`. The public Julia wrapper accepts a
`Symbol`, validates it (`x in (:a, :b)`), wraps it once via `Val(x)`,
and launches the kernel with the `Val`. Inside the kernel, branch
with `if param isa Val{:linear}` — this is constant-folded at
compile time per specialization.

Example deployed in `src/fvfd/operators_2d.jl` (dev/fvfd-core) for the
`polymer_wall_extrap::Val=Val(:quadratic)` kwarg on
`_fvfd_solid_bc_derivative_{x,y}_2d`. Reusable for any future
stencil-variant kwarg in `src/fvfd/`.

**Why**: keeps kernel hot-paths branch-free across the discrete
choice; documented so future contributors don't reach for a
`@generated` macro or duplicate the kernel.

## 2026-05-16 — `_fvfd_solid_bc_derivative_{x,y}_2d` are shared

These helpers are called from BOTH the polymer body-force divergence
path AND the velocity-gradient path (`fvfd_velocity_gradient_2d!`).
Any future change to them MUST keep the velocity-gradient default
behaviour byte-identical — otherwise channel, cylinder, and
contraction benchmarks silently regress. Pattern: kwarg with default
preserving the existing behaviour, threaded ONLY through the calling
path that needs the change (do not extend to all callers).

**Why**: surgical scope discipline; found during M6-B.

## 2026-05-16 — Production polymer-substep cadence is sound

At `n_substeps=4096` per LBM step (the production cavity cadence),
`dt_poly ≈ 2.4e-4` in LU. The Oldroyd-B source ODE integrator is
first-order in `dt_poly`; the per-step bias at production is ~4e-6
(verified empirically in M8 at four refinement levels with perfect
halving). Do NOT propose increasing `n_substeps` as a fix for any
profile gap — the cadence is already negligible-error.

**Why**: prevents wasted effort on substep-cadence "fixes".

## 2026-05-16 — `fvfd_velocity_gradient_2d!` wall stencil is bit-exact

`fvfd_velocity_gradient_2d!` reproduces `du/dy` at wall rows
(j=1, j=Ny) bit-exactly against an analytical Poiseuille profile
under `logfv_periodicx_wally_bcspec_2d()`. The wall-row velocity-
gradient extraction is sound — do NOT propose changes there as a
fix for the cavity gap.

**Why**: ratchets one more cavity suspect out; documented during M8.

## 2026-05-16 — BSD wide-vs-narrow laplacian stencil mismatch

The polymer body force `div(τ_p)` is built by chaining
`fvfd_velocity_gradient_2d!` then `fvfd_tensor_divergence_2d!` — two
central-difference passes that, in the Wi → 0 Newtonian-additive
limit, collapse to a **wide 3-point laplacian with 2dx spacing**
acting on `u`. The existing BSD correction in
`fvfd_bsd_force_2d_kernel!`
(`src/fvfd/operators_2d.jl:886-915`) uses a **narrow 3-point
laplacian** with the standard `dx` spacing. The two laplacians
converge to `∇²u` in the continuum but are NOT the same discrete
operator (wide has 4× the leading truncation error). Any future BSD
or polymer-force work must keep the SAME-stencil invariant on both
sides of the cancellation — otherwise the implicit `(1 − ζ)·ν_p·∇²u`
folding into the LBM viscosity leaves a Wi-independent residual.

**Why**: this stencil mismatch is the root cause of the 3.42 %
M7b smoking gun. Documented during M10.

## 2026-05-16 — `logfv_bsd_stress_from_gradient_2d!` is the latent same-stencil BSD

`src/kernels/logconformation_fv_2d.jl:678-708` already implements
`τ_BSD = 2·ζ·ν_p·D` cell-centered. Feeding this through
`fvfd_tensor_divergence_2d!` gives a BSD body force that uses the
SAME wide stencil as `div(τ_p)`, restoring the discrete
cancellation. The function existed for drag-reduction work but was
never wired into the cavity coupled driver body-force assembly.
M11 rewires it.

**Why**: avoids inventing a new BSD kernel — the right one is
already in the codebase, just not connected.

## 2026-05-16 — File-size architectural constraint: ≤500-700 LOC per file

LLM Departments cannot work effectively on files much larger than
500-700 LOC: Read tool calls become expensive, line numbers drift
between sessions, and each architectural fix requires re-grepping
the entire file context. As of 2026-05-16, the worst offender is
`src/drivers/viscoelastic_logfv_2d.jl` at **3429 LOC** (5× over).
Other big files: `src/kernels/logconformation_fv_2d.jl` (1278),
`src/fvfd/operators_2d.jl` (1084).

Any future driver-level refactor for cavity should include a SPLIT
of `viscoelastic_logfv_2d.jl` into ≤700-LOC modules. Suggested
split:
- `cavity_driver_2d.jl` — `run_viscoelastic_logfv_cavity_coupled_2d`
  itself (the timestep loop).
- `cavity_wall_correction_2d.jl` — `_logfv_cavity_apply_wall_gradient_correction!`
  and its kernel.
- `cavity_bsd_assembly_2d.jl` — BSD path selection (FD / kinetic /
  same-stencil Option 3) and the diagnostic dual-path machinery.
- `cavity_init_2d.jl` — buffer allocation / IC setup.
- `cavity_snapshot_2d.jl` — output / diagnostics writers.

**Why**: M11's small 5-LOC fix got lost in a 3429-LOC context;
M15's audit explicitly flagged line-anchor drift as a Department
hazard. Splitting is itself a load-bearing engineering choice for
LLM-driven debug efficiency, not optional cleanup.

## 2026-05-17 — Codex sandbox CAN run julia after host warm-up

Earlier department memory says the Codex `workspace-write` sandbox
blocks `juliaup`/`julia` from creating the launcher lockfile
(EPERM). M16 sub-mission B observed otherwise: once the host shell
has run `julia` at least once and `juliaup`'s launcher lockfile
already exists on disk, subsequent Codex invocations from the
sandbox CAN launch `julia` and run `Pkg.test`. M16A still hit EPERM
(host hadn't warmed up); M16B succeeded (host had warmed up between
the two). Practical rule: both `BLOCKED-BY-SANDBOX` (with the
Department revalidating on host) and full test-suite tail are valid
GREEN-pending shapes. Do NOT report RED purely because Codex
managed to run julia — that's a more recent (less-defensive) shape,
not a violation.

**Why**: prevents the next Engineer or Department from treating an
unexpected successful Codex `julia` run as a forbidden-action
warning. Both outcomes are legitimate.

## 2026-05-17 — Cavity wall-gradient correction writes half-cell ghosts

`_logfv_cavity_wall_gradient_correction_kernel!` in
`src/drivers/cavity_wall_correction_2d.jl` uses inverse spacings
`2/dx`, `2/dy` (i.e. half-cell one-sided FD against Dirichlet wall
velocity). At each wall cell it overwrites the relevant gradient
components; at the 4 corner cells (i=1,j=1 / i=1,j=Ny / i=Nx,j=1 /
i=Nx,j=Ny) TWO branches fire in cascade and overwrite all 4 of
`dudx, dudy, dvdx, dvdy`.

These half-cell values are **coherent for cell-local consumers**
(the source ODE in cavity step 4 reads them only at the local cell)
but are **NOT cell-centered gradients** in the FV sense. Routing
them through any divergence operator (e.g. `fvfd_tensor_divergence_2d!`
or `logfv_polymer_force_bc_aware_2d!` in M17 Option 3) amplifies the
corner artifact into a singular body force. Empirical confirmation:
M17-pre v1 NaN'd at SW corner (1,1) step 200 under the `:fd_v2`
path; the `:fd` baseline NaN'd at the same corner step 2200 under
the ADR-flagged `bsd_fraction=1.0` regime.

**Why**: any new design that routes `D_corrected` through a wide
neighbour-stencil operator MUST either (a) zero the BSD stress on
wall rows / corners, (b) re-derive cell-centered gradients before
routing, or (c) use the *uncorrected* (interior FD) `D` for BSD
specifically. Treat the half-cell `D` as a 1-cell ghost layer, not
a true gradient field.

## 2026-05-17 — Why wall correction breaks BSD wide-stencil divergence

The cavity wall-gradient correction writes half-cell ghosts at walls
(`2/dx`, `2/dy` inverse spacings) into `dudx, dudy, dvdx, dvdy`.
These values are coherent for cell-local consumers (the source ODE
reads them only at the cell where they live). But the FV divergence
kernel `logfv_polymer_force_bc_aware_2d!` uses a **wide stencil**
(5×3 pattern, reads `i±1` and effectively `i±2` through the velocity-
gradient chain). An interior cell at `j=2` queries the overwritten
`j=1` ghost as if it were a true cell-centred gradient, manufacturing
a fictitious one-cell-inside gradient.

The fix that M17-canary-A validated (commit `f60f5174`): give BSD
its **own** centred-FD `D_uncorrected` (re-call `fvfd_velocity_gradient_2d!`
into a separate buffer WITHOUT the wall-correction overlay), while
the source ODE keeps reading `D_corrected`. Two buffers, same memory
footprint as before counting the existing `D_corrected`. At walls,
BSD reads the "wrong" but bounded centred-FD value; the LBM bounce-
back already handles the viscous flux exactly, so no body-force
compensation is needed there.

**Why**: prevents the next M17-style attempt from reusing the
wall-corrected D as the BSD source — the L2b table (wall_drop 1860×)
makes this unambiguous. Any future kernel that takes a "gradient
field" argument needs the brief to specify which D it expects.

## 2026-05-17 — Discrete identity ≠ analytical identity for div(2·ν_p·D)

Analytically, `div(2·ν_p·D) = ν_p·∇²u + ν_p·∇(∇·u) = ν_p·∇²u` for an
incompressible flow. Discretely, the chain
`fvfd_velocity_gradient_2d! → logfv_polymer_force_bc_aware_2d!` on
`τ = 2·ν_p·D_cell` does NOT reduce to a pure wide Laplacian on `u`:
it leaves residual cross-derivative terms because the discrete
`∇·u_centered` is not bit-exactly zero (numerical drift).

Empirical evidence: M17-impl-v2 attempted to subtract τ_Newtonian =
2·ν_p·D_cell from τ_p cell-by-cell, then take the divergence of the
resulting "elastic" tensor. The numerical noise in the discrete
identity caused the constitutive ODE to diverge (log_spd negative)
at step ~120 on both Metal F32 AND CPU F64.

**Implication for any "extract Newtonian portion" trick**: do NOT
subtract tensors at cell-centers and then take divergence. Instead,
compute the same divergence operator on BOTH the full tensor and the
Newtonian asymptote SEPARATELY (same kernel, same arguments), then
subtract at the FORCE level. The discrete noise then cancels by
construction:

```
F_poly_full     = div_wide(τ_p)                  // kernel A
F_poly_newton   = div_wide(2·ν_p·D_cell)         // kernel A on different input
F_poly_elastic  = F_poly_full − F_poly_newton    // force-level subtract
```

`F_poly_full − F_poly_newton` = `div_wide(τ_p − 2·ν_p·D_cell)` analytically,
but discretely the operator's nonlinearities and BC handling on the
two inputs cancel in the force subtraction, whereas they accumulate
when subtracting at cell-tensor level.

**Why**: prevents the next Department from re-implementing the same
"τ_elastic = τ_p − 2·ν_p·D, then div_wide" pattern that M17-impl-v2
demonstrated to be unstable. The fix is mechanical (reorder
operations), not architectural.

## 2026-05-18 — Codex sandbox CAN run julia for short CPU smokes (M20 confirmation)

The earlier 2026-05-16 department memory note ("Codex sandbox cannot
run julia (lockfile EPERM)") is conditional, NOT absolute. The
2026-05-17 engineer memory note already softened this; M20 (2026-05-18)
provides one more datapoint: the M20 self-test (Ny=16, max_steps=1000,
~60 s CPU F64) ran INSIDE the Codex sandbox to exit 0. The lockfile
trap fires when `juliaup`/`julia` is invoked without a pre-existing
host warm-up; once warm, sandbox `julia` works. Practical rule for any
future Engineer brief: include `julia --project=. <self-test>` as the
exit criterion command without expecting it to BLOCK on sandbox; the
Department revalidates on host either way.

**Why**: prevents the next Engineer from adding "if Codex sandbox
blocks julia, the brief should stop here" defensive logic. Both
outcomes (sandbox runs julia / sandbox EPERMs) are legitimate; the
Department's host revalidation is the source of truth.

## 2026-05-18 — `logfv_velocity_gradient_bc_aware_2d!` is a trivial wrapper

`logfv_velocity_gradient_bc_aware_2d!`
(`src/kernels/logconformation_fv_2d.jl:918-926`) is literally:

```julia
function logfv_velocity_gradient_bc_aware_2d!(...)
    return fvfd_velocity_gradient_2d!(...)
end
```

Bit-identical to `fvfd_velocity_gradient_2d!`
(`src/fvfd/operators_2d.jl:1022 or 1062`). The Poiseuille driver
calls one, the cavity driver calls the other, but they execute the
SAME code. Do NOT propose any mission framed as "kernel difference
between cavity and Poiseuille drivers" — there IS no kernel
difference. Open Q5 closed negatively by M21 (commit `81745f3b`).

**Why**: would have saved M21 entirely if known before drafting. Future
Engineer briefs should check for thin wrapper relationships by
reading the function body before assuming two differently-named
kernels diverge in behaviour.

## 2026-05-18 — M5 kinetic-BSD equivalence is STATIC ONLY

The M5-B prototype (`src/kernels/bsd_kinetic.jl:89`
`compute_bsd_force_kinetic_2d!`) reports `5.85e-16` against the
FD-BSD path on smooth interior IN A STATIC TEST at t=2 (Taylor-Green
or similar smooth velocity field, no LBM evolution). **This static
equivalence does NOT survive the coupled steady state**. M21
(commit `81745f3b`) measured `:kinetic` on Poiseuille at 100k LBM
steps: F_BSD = -6.7e-8 vs target -3.75e-6 → 56× UNDER-shoots, leaving
F_total 30× wrong (1.86 = 186% rel residual).

Likely cause: the LI-BB wall pre-phase perturbs `f` at wall-adjacent
cells before Π^neq is read, so the kinetic moment carries non-Chapman
-Enskog content there. Over 100k steps this propagates into the
interior via streaming. The M5-A audit doc flagged this as the top
risk; M21 confirmed it empirically.

**Implication**: do NOT propose any future mission that uses
`:kinetic` as a production BSD path. It is useful only as an instantaneous
Π^neq accumulator for rheology diagnostics on smooth interior at
small evolution times.

**Why**: prevents the next Engineer from treating M5-B's "machine
epsilon" claim as a production-ready equivalence guarantee. The
static-vs-dynamic distinction is load-bearing.

## 2026-05-18 — `:fd_v2` wide-on-wide BSD: bulk OK, walls catastrophic

`logfv_bsd_stress_from_gradient_2d!` building `τ_BSD = 2·ζ·ν_p·D` and
routing through `fvfd_tensor_divergence_2d!` (the "wide-on-wide" BSD
path tried in M11 and M17-pre) is **structurally sound in the bulk**:
on Taylor-Green periodic it gives near-machine cancellation against
the wide F_poly (M17-canary L1 measurement). **It is catastrophic at
walls in the LBM-coupled steady state**. M21 (commit `81745f3b`)
measured on Poiseuille (no embedded obstacle, only halfway-bounce
walls): τ_yy abs max = **0.26**, vs target zero, vs τ_xy ≈ 2.5e-3.
That's 10⁵× the τ_xy amplitude in a quantity that must be analytically
zero. The wide div on τ_BSD at j ∈ {1, Ny} reads the one-sided
3-point velocity gradient that does NOT exactly cancel with the wide
div on τ_p (which was built from τ from the source ODE, consistent
with the same one-sided gradient — but the off-diagonal τ_BSD picks
up the antisymmetric cross-derivative that the source ODE doesn't).

The non-zero τ_yy loops back through the constitutive ODE at wall
cells; at Wi=1 the elastic response amplifies it until NaN.

**Implication**: any future "use the same kernel on both sides" BSD
fix MUST handle wall cells separately (mask the wide div there, or
use the M5 kinetic path interior + FD path at walls, etc.). The
half-cell wall stencil mismatch is the load-bearing issue, not the
bulk cancellation principle.

**Why**: prevents the next Engineer from implementing yet another
wide-on-wide variant and discovering the wall pathology only after
NaN. The bulk-OK / wall-broken split is structural.

## 2026-05-18 — Embedded force kernel: cell-fraction divisor overdose

`fvfd_tensor_divergence_embedded_2d_kernel!`
(`src/fvfd/operators_2d.jl:759-766`) outputs **force-per-fluid-volume**
(divides the integrated flux by `cell_fraction`), but the Guo source
consumer (`fused_trt_libb_v2_guo_field_step!` and the polymer
force-add path) expects **force-per-lattice-cell**. On cut cells with
small `cell_fraction` (e.g. 0.1-0.3 near the cylinder surface), this
overdoses the Guo body force by 3-10×, which biases `f` near the
wall, which inflates the LBM cut-link drag (MEA) integration.

**Implication**: any embedded driver routing polymer force through
this kernel MUST re-scale by `cell_fraction` on the consumer side
before adding to `fx_total/fy_total`. The kernel-side normalisation
is correct for cell-local consumers (constitutive ODE) but wrong for
field-level sources (Guo, LBM body force).

**Why**: this is the cell-fraction half of the +8.8 Cd_kraken
`1111_circle` ghost drag (M26-analysis 2026-05-18). The other half
is the half-cell-ghost coupling (see next entry).

## 2026-05-18 — Embedded gradient writes half-cell ghosts (cavity-family pattern)

`_fvfd_apply_embedded_wall_gradient_2d`
(`src/fvfd/operators_2d.jl:127-140`) writes **half-cell** normal
∂u/∂n into the cell-centered `dudx, dudy, dvdx, dvdy` buffers that
the source ODE AND the polymer-force divergence both consume. This
is **the same pathology family** as the cavity M17-canary-A bug
(`_logfv_cavity_wall_gradient_correction_kernel!`, see engineer.md
2026-05-17 entries).

When the force path queries the overwritten `i±1` cut-cell ghost
as if it were a true cell-centred gradient, it manufactures a
fictitious one-cell-inside gradient. Combined with the cell-fraction
overdose (previous entry), the cut cells receive an amplified-AND-
singular Guo force.

**Fix pattern (M17-canary-A applied to cylinder)**: give the FORCE
path its own `D_uncorrected` buffer (re-call `fvfd_velocity_gradient_2d!`
WITHOUT the embedded-wall correction overlay), keep the source ODE
on the corrected `D_corrected`. Two buffers, same memory footprint
counting the existing `D_corrected`. At cut cells, the FORCE reads
the bounded centred-FD value; the LBM cut-link bounce-back already
handles the viscous flux exactly there, so no body-force compensation
is needed.

**Why**: prevents the next embedded-mode debug from reusing the
wall-corrected D as the force source — same trap as cavity M17
attempts. Documented during M26-analysis 2026-05-18.

## 2026-05-18 — At nu_p=0, embedded_force/embedded_drag become NO-OPs

The polymer prefactor `nu_p/lambda` is multiplied into `tau` before
any embedded force/divergence/drag kernel sees it. When `nu_p = 0`
(β=1 strict-Newtonian), `tau ≡ 0` everywhere and `Cd_p = Cd_bsd = 0`
regardless of the embedded flag values. Bench `1111_circle` at β=1
gives bit-exact Cd_s to `0000_circle` at β=1; geometry kwarg
contributes a tiny +0.13 % from a quadrature artifact unrelated to
the polymer-coupling bug. Newtonian Cd_s baseline at R=20, 1000
steps, CPU F64 = **136.26 (qwall) / 136.44 (circle)** — useful as
a regression ratchet for any future embedded geometry work.

**Why**: a Newtonian probe is a clean way to RULE OUT polymer-side
hypotheses but CANNOT discriminate H1/H2/H3 inside the polymer
pipeline. Discrimination requires finite-Wi (and ideally
`bsd_fraction = 0` so `Cd_bsd = 0` separates from `Cd_p` cleanly).
Found during M26-impl 2026-05-18.

## 2026-05-18 evening — Cell-fraction divisor in embedded force kernel CONFIRMED bug

Phase 0b A100 F64 sweep (job `21572831.aqua`) gives empirical
confirmation of the M26-analysis hypothesis: at R=30 β=0.59 Wi=0.1
bsd_fraction=1.0, the flag-by-flag Δ vs `0000_circle` baseline 129.39 is:

| flag pattern | Cd_kraken | Δ |
|---|---|---|
| 0010_circle (force-only) | **137.49** | **+8.10** |
| 0010_qwall (force-only on qwall) | **138.10** | **+8.71** |
| 1111_circle (full) | 139.27 | +9.88 |

The bug is **entirely in `embedded_force=true`**: it gives the +8.1 Cd
ghost drag at R=30, with `:circle` and `:qwall` geometries producing
near-identical magnitudes (rules out quadrature-bias H3). The defect
lives in `fvfd_tensor_divergence_embedded_2d_kernel!`
(`src/fvfd/operators_2d.jl:759-766`) which divides the integrated
flux by `cell_fraction`. On cut cells (typical `cell_fraction ≈ 0.3`
at R=30), the output is force-per-fluid-volume but the Guo source
consumer expects force-per-lattice-cell → 3-10× overdose → biases `f`
near the wall → inflates the LBM cut-link MEA drag (Cd_s) by ~8 Cd.

**Cross-references**:
- M26-analysis (math) : `.orchestrator/M26_analysis_verdict.md`.
- M26-impl (Newtonian) : `bench/viscoelastic_audit/CYL_EMBEDDED_DRAG_DIAG_M26_VERDICT.md`.
- Joint Phase 0+0b verdict :
  `bench/viscoelastic_logfv/CYL_PHASE0_PHASE0B_VERDICT_20260518.md`.

**M26b fix recipe** (single-file patch in `src/fvfd/operators_2d.jl`):
- Option A (safest): multiply the output by `cell_fraction` AFTER the
  kernel returns, AT THE GUO CONSUMER SITE (cylinder driver line where
  `fx_total += embedded_polymer_force` happens). This preserves the
  kernel's interface (force-per-fluid-volume) for any other callers.
- Option B (intrusive): remove the `cell_fraction` divisor inside
  `fvfd_tensor_divergence_embedded_2d_kernel!`. Requires checking
  every caller for consistency (currently only the cylinder embedded
  path, but verify).
- Acceptance: `0010_qwall` Cd within ±0.3 of `0000_qwall` baseline at
  R=30 (= ratchet from +8.71 down to noise).

**Why**: definitive empirical pin of the bug location. Future M26b
brief should cite operators_2d.jl:759-766 and apply Option A (lower
blast radius).

## 2026-05-18 evening — Julia 1.12 world-age trap pattern (codebase-wide)

A Julia 1.12 world-age behavior change broke `detect_backend()` in
`bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl` and caused 4h35
of Aqua A100 time to silently run on CPU. Pattern that fails:

```julia
function f()
    try
        @eval using SomeModule          # advances world age in Main
        m = getfield(Main, :SomeModule)  # FAILS: function still in OLD world age
        ...
    catch e
        # bare `catch end` hides the UndefVarError
    end
end
```

The `@eval using SomeModule` advances the Main module's world age, but
the calling function `f` continues to execute at its compiled world
age, which doesn't see the new binding. `getfield(Main, :SomeModule)`
raises `UndefVarError: SomeModule not defined in Main` (hint mentions
"running in world age N, current world is M").

**Fix patterns** (in order of safety):
1. `Base.invokelatest(getfield, Main, :SomeModule)` — runs `getfield`
   in the latest world (minimal patch, used in commit `e602726f`).
2. Move `using SomeModule` to top-level (no `@eval`); the file load
   itself advances the world age before any function runs.
3. Wrap the whole "after using" block in `Base.invokelatest(() -> begin ... end)`.

**Anti-patterns**:
- `catch end` bare — always swallows the UndefVarError silently.
- Using `getfield(Module, :name)` directly after `@eval using ...`
  inside a function without `Base.invokelatest`.

**Detection**: any backend-detection / dynamic-import pattern inside
a function. Grep for `@eval using` + `getfield` pairs.

**Why**: this trap is now Julia 1.12 baseline; older code that worked
on 1.10 may fail silently. Any future bench script copying the
`detect_backend()` pattern MUST use `Base.invokelatest` wrapping.

## 2026-05-19 — `KRAKEN_MAX_STEPS_BASE` is straight assignment, NOT R²-scaled

The bench `bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl` has a
misleading comment near line 15 hinting at R²-scaling for max_steps.
The actual code is :

```julia
max_steps = parse(Int, get(ENV, "KRAKEN_MAX_STEPS_BASE", "100000"))
```

i.e. a **straight assignment**. No R² (or any other) scaling
applied. M28c (jobs 21579957/8/9) verified empirically that 100 000
steps at λ = 6000 LU R = 30 Wi = 1 is converged to F64 round-off vs
1 000 000 steps (Δ Cd = 3e-7). At λ ≤ 6000 LU and Wi ≤ 1, the M28c
configuration (`KRAKEN_MAX_STEPS_BASE = 100000`, `KRAKEN_AVG_WINDOW_FRAC
= 0.2` = last 20 000 steps averaged) is the validated production
operating point.

For higher λ (R ≳ 60 ⇒ λ ≳ 12 000 LU), an audit will be needed
because M28e showed step-0 NaN at R = 60 / 80 — likely an
initialisation / ψ→C exponentiation stability issue, NOT a
max_steps issue. Don't conflate.

**Why**: any future bench script copy-pasting from
`run_cyl_bigsweep_v2_2d.jl` will inherit this comment; do NOT
re-add R²-scaling without explicit calibration. The flat 100 k
default is the validated number.

## 2026-05-19 — M26b residual : wall-segment term is the dominant +Cd amplifier

M26b Option-A (consumer-side rescale by `cell_fraction` in the
cylinder `embedded_force` branch) closes only ~8 % of the +8 Cd
overdose at R=20 Wi=0.1 β=0.5 (delta +7.92 → +7.27). The
`1/cell_fraction` divisor inside
`fvfd_tensor_divergence_embedded_2d_kernel!`
(`src/fvfd/operators_2d.jl:759-766`) is **NOT** the dominant
amplifier. Per M26b verdict §"Recommended next mission" the
load-bearing residual is :

- The **wall-segment surface-flux term** `wall_x_length * tauxx[i, j] +
  wall_y_length * tauxy[i, j]` (and the analogous `_y_length * tauxy /
  tauyy` for the y-component) added at the cell center.

This term contributes a surface-delta of magnitude
`(west_fraction − east_fraction) * τ`, which is the **embedded
surface traction** at the cut segment. Geometrically that integral
belongs to the LBM cut-link MEA loop (`compute_drag_libb_mei_2d`),
NOT to the FVFD body-force source. **Hypothesis** : the
`fvfd_tensor_divergence_embedded_2d_kernel!` wall-segment term
double-counts the LBM-side cut-link traction. The Option-A
cell-fraction rescale does NOT fix this because the rescale acts on
the WHOLE divergence output `(F_volume + F_wall) / V_cell`, both
the body and the surface part ; only zeroing `wall_x_length /
wall_y_length` in the kernel would isolate the surface contribution.

**Recommended M26c kernel patch** : add a kwarg
`_drop_wall_segment_term :: Bool = false` to
`fvfd_tensor_divergence_embedded_2d_kernel!`. When `true`, set
`wall_x_length = wall_y_length = 0` in the per-cell sum. Re-run
the M26b smoke ; if Δ drops to < 1 Cd, the wall-segment term IS
the residual amplifier and the kernel should be split into
"body-only divergence" + "surface-only contribution" (the latter
disabled for LBM consumers that already integrate the surface via
MEA).

Production gate (per M26b verdict) : `0010_circle` and `1111_circle`
Cd within ±1 of `0000_circle` baseline at R=30 Wi=0.1 β=0.59.

**Phase 1 mesh-refinement evidence** (M28e, job 21580646) for the
`0000_qwall` Liu-mode path : Cd at Wi=1 plateaus at 111.4 across R ∈
{20, 30, 40}, NOT approaching rheoTool 120.40. The M28-cluster
synthesis (`bench/viscoelastic_logfv/CYL_SESSION_M28_SYNTHESIS_20260519.md`)
concludes the residual +7 % Wi=1 gap is in the polymer-coupling
internals (constitutive log-conf discretisation OR Guo source
ordering), NOT in the embedded force kernel (which is INACTIVE on
the `0000_qwall` Liu path). M26c remains a real bug but is
ORTHOGONAL to the M28 production gap.

**Why**: future engineer working on either M26c or M29-tau-compare
should not conflate the two threads. M26c fixes a `1111_circle`
embedded-mode bug ; M29 will localise the `0000_qwall` finite-Wi
gap. Both stay open ; the latter is the higher-priority production
blocker.

## 2026-05-19 evening — M29 closes the gap locus: Rusanov upwind on Ψ-advection is the defect

Field-level rheoTool vs Kraken at R=30 Wi=1.0 β=0.59 gives τ_xx
L2_rel = 0.93 (catastrophic) while u_x L2_rel = 0.17 (OK). Peak
τ_xx mis-prediction is **44 %** (Kraken 75.3 vs rheoTool 135.5)
at the leeward shoulder. The **first-order Rusanov upwind on
log-conformation Ψ advection** smears the wrap-around stress
feature (width O(1 LU at R=30)) over ~50 % of its magnitude.
rheoTool's `cubista` TVD preserves it.

**M29b fix recipe** (separate `src/` mission, not yet executed):
- Locate the Rusanov upwind in
  `src/kernels/logconformation_fv_2d.jl` (likely
  `logfv_advect_upwind_bc_aware_2d!` or a similar
  donor-cell-style flux assembly). The kernel is shared with
  cavity / channel benches — DO NOT branch ad-hoc per geometry.
- Implement a TVD HRS option behind a kwarg, e.g.
  `advection_scheme::Symbol = :rusanov` with `:cubista`,
  `:muscl_superbee`, or `:hrs_minmod` alternatives.
- CUBISTA (Alves & Pinho 2003) is the rheoTool default and the
  best-validated for log-conf advection. MUSCL-superbee is a
  simpler closed-form alternative that achieves similar peak
  preservation. Both are documented in
  `bench/viscoelastic_audit/EQUATION_AUDIT_LIU_RHEOTOOL.md`
  (if present) or in the rheoTool `fvSchemes` snippets across
  `bench/rheotool/cylinder_wi*/system/fvSchemes`.
- Acceptance criterion: at R=30 Wi=1.0 β=0.59 production setup,
  Kraken Cd within ±2 of rheoTool 120.40. Use the
  M29-tau-compare driver to cross-check τ_xx peak ≥ 130 (vs
  current 75.3). Bench:
  `bench/viscoelastic_audit/run_kraken_vs_rheotool_tau_compare.jl`.

**Snapshot infrastructure** (already committed bench-side, no `src/`
patch): `bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl` now
supports `KRAKEN_SAVE_FIELDS=1` env flag. Writes per-case `.jls`
with `(ux, uy, tauxx, tauxy, tauyy, is_solid, Nx, Ny, R, cx_lbm,
cy_lbm, u_mean, Cd_*)`, ~4 MB per case at R=30. Use this for any
future field-level cross-check without adding `src/` debt.

**Why**: M29b is the highest-leverage `src/` improvement on the
viscoelastic branch right now. The advection scheme upgrade is
**localised** (one kernel) but has **broad impact** (every viscoelastic
bench using log-conf advection — cylinder, cavity, channel,
contraction would all benefit). Should be next session's primary
deliverable.

## 2026-05-19 — CairoMakie gotcha: no :RdBu_r palette

`CairoMakie` does NOT support the `:RdBu_r` palette name (worked in
older Makie). Use `Reverse(:RdBu)` for diverging colormaps. Found
during M29-tau-compare plotting.

**Why**: a 1-minute fix that's easy to miss; documented to spare
the next plotting iteration.
