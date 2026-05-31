---
module: physics-newtonian
path: src/kernels/
owner_concern: lbm-operator
status: implemented
last_verified: 2026-05-31
depends_on:
  - lbm
  - bc
  - backend
---

# physics-newtonian — module implication map

The Newtonian constitutive path is the **isothermal single-relaxation (BGK) and
two-relaxation-time (TRT) D2Q9/D3Q19 collision operator** — the bedrock of every
flow benchmark in Kraken. It owns ONE concern: the local collision that relaxes
populations toward the Maxwell-Boltzmann equilibrium `feq_2d` at a constant
kinematic viscosity (constant `ω`, no shear-rate dependence). It is consumed by
the cavity/Poiseuille/Couette/Taylor-Green/cylinder drivers in
`src/drivers/basic.jl` and by the `.krk` runner in `src/simulation_runner.jl`.
Validated M6: lid-driven cavity matches Ghia 1982 / icoFoam to <1%. Non-Newtonian
(GNF/viscoelastic) lives in sibling `physics-viscoelastic`; thermal coupling in
`physics-thermal`.

## Public surface

These are exported from `Kraken.jl` and callable directly (LBM operator API):

- `collide_2d!(f, is_solid, ω; sync=false)` — in-place BGK collision on `f`
  (`collide_stream_2d.jl`). Solid cells get a full bounce-back swap; fluid cells
  relax all 9 populations toward `feq_2d`. The canonical Newtonian collide.
- `collide_3d!(f, is_solid, ω)` — D3Q19 analogue (`collide_stream_3d.jl`).
- `stream_2d!(f_out, f_in, Nx, Ny; sync=false)` / `stream_3d!(...)` — pull-scheme
  streaming with halfway bounce-back at domain edges (clamped `ifelse` reads).
- `fused_bgk_step!(f_out, f_in, ρ, ux, uy, is_solid, Nx, Ny, ω)` — the
  PERFORMANCE path: one kernel launch fusing stream + bounce-back + BGK collide +
  macroscopic (`fused_bgk_2d.jl`). Reduces 3 launches → 1 per step.
- `fused_trt_step!(f_out, f_in, ρ, ux, uy, is_solid, Nx, Ny, ν; Λ=3/16)` — fused
  TRT step (`fused_trt_2d.jl`): symmetric mode at `s_plus`, antisymmetric at
  `s_minus`; `Λ=3/16` makes halfway-BB error viscosity-independent.
- `trt_rates(ν; Λ=3/16) -> (s_plus, s_minus)` — `s_plus = 1/(3ν+½)`,
  `s_minus = 1/(Λ/(3ν)+½)`. The conventioned rate pair the TRT kernel applies.
- `compute_macroscopic_2d!(ρ, ux, uy, f; sync=false)` / `compute_macroscopic_3d!`
  — standalone moment getters (`macroscopic.jl`), Integrated convention.
- `aa_even_step!` / `aa_odd_step!` (`aa_bgk_2d.jl`), `persistent_fused_bgk!`
  (`persistent_bgk_2d.jl`) — AA-pattern and persistent-kernel BGK variants.
- Driver entry points (`drivers/basic.jl`): `run_cavity_2d`, `run_cavity_3d`,
  `run_couette_2d`, `run_poiseuille_2d`, `run_taylor_green_2d`,
  `run_cylinder_2d`, plus `LBMConfig`, `omega(config)`, `reynolds(config)`,
  `initialize_2d`, `initialize_3d`.
- DSL bricks (`kernels/dsl/bricks.jl`): `CollideBGKDirect`, `CollideTRTDirect`,
  `CollideTRTLocalDirect` — `emit_code` fragments that reproduce the fused kernels.

## Reads from

The collision operator is nearly a leaf; it reads constants/types from:

- `lbm` (`src/lattice/`) — D2Q9 lattice topology: the inline `feq_2d(Val(q),…)`
  weights (4/9, 1/9, 1/36), the `moments_2d` reduction signs, and the
  bounce-back opposite-pair table baked into `bounce_back_2d!`
  (`equilibrium_helpers.jl`). Driver init also calls `weights(D2Q9())` and
  `equilibrium(D2Q9(),…)`.
- `bc` — read only at the driver/runner level, NOT inside the collide kernel:
  Zou-He velocity/pressure handlers (`apply_zou_he_north_2d!`,
  `apply_zou_he_south_2d!`, …) and halfway bounce-back walls are applied
  post-stream / pre-collide by the caller.
- `backend` — `KernelAbstractions.get_backend(f)` to launch the kernel on the
  array's device; `eltype(f)` to pin scalar precision. No other module state.

The kernel consumes its `f` / `is_solid` / `ω` arguments read-only-then-write;
it does not reach into any global registry.

## Writes to

- **Mutates the distribution array in place.** `collide_2d!` overwrites all 9
  populations of every fluid cell of `f`, and swaps the 4 opposite pairs of every
  solid cell (bounce-back). `fused_bgk_step!` / `fused_trt_step!` instead write
  `f_out` (double-buffer) and ALSO write the macroscopic arrays `ρ`, `ux`, `uy`
  in the same launch (solid cells forced to `ρ=1, u=0`).
- **Produces macroscopic fields.** `compute_macroscopic_2d!` writes `ρ`, `ux`,
  `uy`; the drivers return `(ρ, ux, uy[, uz], config)` NamedTuples copied to host.
- **No global state, no files.** The operator touches no registry and writes no
  output; VTK/PNG emission is the runner's job. Drivers double-buffer by
  rebinding `f_in, f_out = f_out, f_in` each step — the swap is in the caller,
  not the kernel.
- The blast radius of a bug here is the entire flow field: a wrong `ω`, sign, or
  weight silently corrupts every downstream moment, drag, and Nusselt number.

## Backend constraints

- **GPU-clean KernelAbstractions kernels.** `collide_2d_kernel!`,
  `fused_bgk_step_kernel!`, `fused_trt_step_kernel!` are `@kernel`/`@inbounds`,
  fully unrolled over the 9 directions, with NO dynamic allocation in the hot
  loop. Runs once per timestep per cell; backend-agnostic (CPU / CUDA / Metal).
- **Streaming uses clamped `ifelse`, not branches.** `ifelse` evaluates both
  arms, so the out-of-bounds "dead" read is given a clamped in-bounds index
  (`im=max(i-1,1)`, …). Removing the clamp → illegal GPU memory access.
- **The fused kernel is the production fast path** (1 launch vs 3); the discrete
  `stream_2d!` + `collide_2d!` pair is the readable/debuggable path used by the
  `basic.jl` drivers and the `.krk` runner. They must stay numerically identical
  (the DSL `CollideBGKDirect`/`CollideTRTDirect` bricks exist to oracle this).
- **Float32 caveat.** Scalars are coerced with `ET(ω)` / `ET(s_plus)` to the
  array eltype. On Metal F32 the collision rate loses precision as `ω→2` (high
  Re); see the Metal F32 R-drift finding (CUDA F32 ≡ F64, MPS quirk). The
  `units` module enforces a fatal `:tau_float32_floor` at `τ<0.6` to protect
  this path.
- `sync=false` by default — the redundant per-wrapper `KA.synchronize` was
  removed (commit f52335b04); callers synchronize once at loop end.

## Failure modes

This operator is simple but its constants are unforgiving. Receipts:

- **TRT rate-label swap (pre-2026-04-15 bug, fixed in `fused_trt_2d.jl`)** —
  earlier `trt_rates` returned `s_plus` from `Λ` and `s_minus` from `ν` (swapped).
  Because the collide applies the Λ-derived rate to the EVEN mode, effective
  viscosity became `ν_eff = Λ/(9ν)` ≈ 13×ν at Λ=3/16, ν=0.04 — inflating Re ~13×
  and silently over-dragging EVERY benchmark using `trt_rates`. The docstring now
  carries the full receipt; verify `ν = (1/s_plus − ½)/3` if drag looks wrong.
- **`Λ=Inf` does NOT give BGK.** `fused_trt_step!` with large Λ drives `s_plus→0`,
  yielding `a=s_minus/2` — not BGK. To recover BGK exactly call `fused_bgk_step!`
  or pass `Λ=(1/s−½)²`. A frequent "why doesn't TRT match BGK" trap.
- **`ω` floor for stability.** `omega = 1/(3ν+½)` → as ν→0, ω→2 (the LBM stability
  ceiling); above ~1.95 the BGK collide goes unstable / NaN. This is the LBM
  analogue of the units-module TRT magic-window guardrail (M59–M61): keep
  τ=1/ω in `[0.55, 1.5]`. Acoustic R-sweeps that drift τ produce the Cd "U-shape"
  artifact — fix τ, vary u (diffusive scaling).
- **Bounce-back at domain edges is implicit in the stream kernel**, NOT a
  separate BC. A cell flagged `is_solid` gets `bounce_back_2d!` inside the
  collide; if a driver ALSO applies an explicit wall BC on the same edge the
  population can bounce twice (cf. the multiblock corner double-application class,
  ebf0867). Order matters: stream → Zou-He BC → collide (see `run_cavity_2d`).
- **M6 cavity coordinate bug** — the lid-driven cavity ran with a mis-indexed
  coordinate frame; resolved at commit 6b1069283 (`axis_node_coords`), after
  which cavity matches Ghia/icoFoam <1%. If a NEW cavity regression appears,
  re-check the node-coordinate frame BEFORE suspecting the collide.
- **Solid-cell macroscopic forcing.** The fused kernels overwrite solid cells to
  `ρ=1, u=0`; the discrete `compute_macroscopic_2d!` does NOT mask solids, so its
  `ρ`/`u` inside obstacles are garbage divisions — never read them.
- **`inv_ρ = 1/ρ` with no floor** — a cell whose populations sum to ~0 (rare,
  post-instability) yields Inf/NaN velocity that then poisons neighbours via
  streaming. First-NaN is almost always `ρ`; check density before velocity.

## Touch order

For a suspected Newtonian-collide bug (wrong velocity profile, drag, NaN, Re
mismatch), inspect in this order:

1. `src/kernels/collide_stream_2d.jl` — the reference BGK collide + pull-stream.
   Check the `feq_2d` call signature, the `ω` relaxation, and the `is_solid`
   bounce-back branch. 80% of "wrong profile" bugs are a sign/weight here.
2. `src/kernels/equilibrium_helpers.jl` — the inline `feq_2d(Val(q),…)` weights
   and the `moments_2d` reduction signs + `bounce_back_2d!` opposite pairs. A
   single wrong sign here corrupts everything silently.
3. `src/kernels/fused_bgk_2d.jl` — if the GPU/production fast path diverges from
   the discrete path, diff the fused kernel against (1); they must match.
4. `src/kernels/fused_trt_2d.jl` — for TRT-specific drag/viscosity bugs: verify
   `trt_rates` label assignment (the swapped-rate receipt) and the `a/b`
   even/odd recombination.
5. `src/drivers/basic.jl` — the loop order (stream → BC → collide → macro →
   swap), `omega(config)=1/(3ν+½)`, `reynolds`, and `initialize_2d`. A wrong ν or
   a doubled BC lives here, not in the kernel.
6. `src/simulation_runner.jl` (collision dispatch ≈ the `collide_*_2d!` select
   block) — for `.krk`-driven runs: which collide is chosen (rheology vs body
   force vs plain BGK) and how `Fx/Fy` body force is wired.
7. `src/kernels/dsl/bricks.jl` — `CollideBGKDirect`/`CollideTRTDirect` emit_code
   if the DSL-generated kernel disagrees with the hand-written one.
