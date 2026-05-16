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
