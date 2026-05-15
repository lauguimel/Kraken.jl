# Kraken-E S2 — D1 ownership + D2 uniform-block LBM update

Date: 2026-05-15
Branch: `dev/kraken-e-fvfd-blocks`
Worktree: `/Users/guillaume/Documents/Recherche/Kraken.jl-kraken-e-blocks`
Plan: [`kraken_e_fvfd_interface_plan_2026-05-15.md`](kraken_e_fvfd_interface_plan_2026-05-15.md) §3, §10, §13, §14 (the plan lives on `slbm-paper`; consult its copy at `/Users/guillaume/Documents/Recherche/Kraken.jl/docs/agent/`).
Roadmap: [`kraken_e_roadmap.md`](kraken_e_roadmap.md) row S2.
Branch contract: [`branch_contract_fvfd_blocks.md`](branch_contract_fvfd_blocks.md).

## 1. Purpose and scope

This document derives the two foundations of the Kraken-E block runtime:

- **D1**: state ownership for an AMR-tree leaf block. Defines every per-block
  field, who owns each (single-writer or deterministic reduction), and the
  ghost-layer convention. Includes paper-only fields that exist now as
  sentinel-valued placeholders so the same `LeafBlock` type carries the AMR
  metadata that S3..S7 will consume without a layout migration.
- **D2**: the same-level uniform-block LBM update on one leaf. Defines the
  per-step pipeline (`apply_bcs! → exchange_halo! → collide! → stream!`), the
  invariants the implementation must preserve, and the three analytical
  canaries that close S2.

Out of scope for S2 (see §5): coarse/fine geometry (D3..D4), interface
fluxes (D5), reflux/subcycling (D6), moment extraction and population
reconstruction (D7..D8), stress consistency at c/f interface (D9), wall +
interface + Guo composition (D10), viscoelastic coupling (D11), epoch
adaptation (D12), 3D, multi-leaf orchestration, GPU validation.

The SLBM-era `BlockState2D` (`src/multiblock/state.jl`) is **not reused**.
That struct is shaped by the gmsh-derived curvilinear multi-block path
and assumes a flat list of non-overlapping blocks linked by shared-face
records. AMR-tree leaves need: a tree level, parent and child pointers,
same-level neighbour ids drawn from a sibling pool, coarse/fine face
records, and reflux accumulators. Inheriting from `BlockState2D` would
either fork its fields (so it is not really inherited) or smuggle gmsh
assumptions into Kraken-E.  Fresh design.

## 2. D1 — State ownership for an AMR-tree leaf block

### 2.1 Per-block field inventory

Let `T <: AbstractFloat`, `AT2 = AbstractArray{T,2}`, `AT3 = AbstractArray{T,3}`,
`AT3i = AbstractArray{Int8,3}` (cell-flag arrays). All arrays are SoA in
`(i, j, q)` order on the extended grid, per the project memory rule
`feedback_soa_layout` (SoA `f[q,i,j]` was abandoned).

A leaf block is a struct `LeafBlock2D{T,AT2,AT3}` with the following fields,
grouped by purpose. The S2 implementation populates only the **active** groups;
the **paper-only** groups exist as struct fields with sentinel values
(`-1`, empty `Vector`, `nothing`) so S3..S7 can fill them without a layout
migration.

**Group A — Identity and AMR-tree placement (active in S2, sentinel-filled).**

| Field | Type | Meaning | S2 value |
|-------|------|---------|----------|
| `id` | `Int` | global unique leaf id within the tree | `1` for single-leaf canaries |
| `level` | `Int` | refinement level, `0` = root | `0` |
| `origin` | `SVector{2,T}` | world-coordinate corner of cell `(1,1)` (interior) | from canary config |
| `dx` | `T` | cell size at this level, `= dx_root / 2^level` | from canary config |
| `Nx, Ny` | `Int` | interior cell counts in `ξ, η` | `32, 32` for canaries |
| `ng` | `Int` | ghost-layer width | `1` (D2Q9) |

**Group B — LBM populations and macroscopic fields (active in S2).**

| Field | Type | Meaning |
|-------|------|---------|
| `f` | `AT3` size `(Nx+2ng, Ny+2ng, 9)` | populations, D2Q9 |
| `f_tmp` | `AT3` same shape | scratch for pull-streaming `f_tmp ← stream(f)`, then swap |
| `ρ` | `AT2` size `(Nx+2ng, Ny+2ng)` | density |
| `ux, uy` | `AT2` same shape | velocity components |

`f_tmp` is the only ping-pong buffer. After streaming, the runtime swaps
`f` and `f_tmp` by reassigning the (mutable) struct fields, **not** by
rebinding locals (see project memory `feedback_blockstate_swap`).

**Group C — Cell flags (active in S2 trivially; consumed by BCs).**

| Field | Type | Meaning |
|-------|------|---------|
| `cell_kind` | `AT3i`-equivalent `(Nx+2ng, Ny+2ng)` of `Int8` | enum: `INTERIOR=0`, `GHOST_HALO=1`, `GHOST_CF=2`, `WALL=3` |

In S2 only `INTERIOR` and `GHOST_HALO` are written. `GHOST_CF` and `WALL`
cells are not produced by the canaries (BCs are applied via ghost-layer
kernels writing reflected populations, not by marking cells as wall).

**Group D — Topology pointers (paper-only in S2; sentinel-filled).**

| Field | Type | S2 sentinel |
|-------|------|-------------|
| `parent_id::Int` | parent leaf id, or `-1` if root | `-1` |
| `child_ids::Vector{Int}` | 4 children if non-leaf, empty if leaf | `Int[]` |
| `same_level_neighbor_ids::NTuple{4,Int}` | E, N, W, S same-level neighbours, `-1` if none | `(-1,-1,-1,-1)` |
| `cf_face_records::Vector{CFFaceRecord}` | per-c/f-face metadata (placeholder type with empty defaults) | `CFFaceRecord[]` |
| `reflux_accumulators::Vector{T}` | accumulators for D6 conservative reflux | `T[]` |
| `epoch_remap_buffers::Vector{T}` | pre-allocated buffers for D12 epoch remap | `T[]` |

The `CFFaceRecord` type itself is declared in S2 as an empty `struct` (no
fields) or a struct with `Int` placeholders only. Its concrete schema is
fixed in S3 (D4). The point of declaring it now is to lock the field
into `LeafBlock2D`'s memory layout so S3 is a fields-of-a-record change,
not a fields-of-the-leaf-block change.

### 2.2 Ownership rules

Every value written by a kernel has exactly one owner, or a deterministic
reduction owner. The rules below cover all D1 fields; for S2 only the
rows tagged **active** are exercised by code.

| Field group / scope | Writer (owner) | Reader(s) | Reduction? |
|---|---|---|---|
| **active** Group A identity/grid | initializer (one-shot) | all kernels | no — immutable after init |
| **active** Group B `f` interior | `collide_*!`, `stream_*!` | macroscopic, BCs | no — single-writer per cell |
| **active** Group B `f` halo (ghost row) | `apply_bcs_*!`, `exchange_halo!` | `stream_*!` | no — last writer wins; BCs and exchange MUST NOT both target the same ghost cell on the same step (enforced by `cell_kind`) |
| **active** Group B `ρ, ux, uy` interior | `compute_macroscopic_2d!` | diagnostics, forcing, post-step | no — single-writer per cell |
| **active** Group B `ρ, ux, uy` halo | not written | not read by S2 kernels | n/a |
| **active** Group C `cell_kind` | initializer (one-shot) | all kernels | no — immutable after init for S2 |
| paper-only Group D `parent_id`, `child_ids` | AMR-tree manager (S3+) | tree traversal | no |
| paper-only Group D `same_level_neighbor_ids` | tree connectivity init (S3+) | `exchange_halo!` future variants | no |
| paper-only Group D `cf_face_records` | c/f geometry builder (S3, D4) | flux assembly (S4, D5) | no |
| paper-only Group D `reflux_accumulators` | flux assembly (S4) and time integrator | refluxing step | **yes** — coarse face value `= sum(fine subfaces)`; reduction owner is the coarse-side block |
| paper-only Group D `epoch_remap_buffers` | epoch manager (S8+, D12) | remap kernels | no |

S2 active code touches only: `f`, `f_tmp` (swap), `ρ`, `ux`, `uy`, `cell_kind`
(read-only after init). Halo cells of `f` are written by either
`apply_bcs_*!` (canary BC kernels) **or** `exchange_halo!` (no-op in S2),
never both. The `cell_kind` enum is the runtime guarantee that S5+ can
add ghost cells with mixed providers without races.

### 2.3 Layout choices justified

**`ng = 1` for D2Q9.** Streaming uses pull-scheme nearest-neighbour reads
along `c_q ∈ {(0,0), (±1,0), (0,±1), (±1,±1)}`. A single ghost row suffices
for one streaming step. The existing Kraken kernel
`stream_periodic_x_wall_y_2d_kernel!` (which clamps with `max(i-1,1)` etc.)
demonstrates that even bounce-back fallbacks work with one ghost row; here
we make the layer explicit so the streaming kernel is uniform and
BC-free.

**Extended grid `(Nx+2, Ny+2, 9)`.** Interior indices are
`(ng+1 .. ng+Nx, ng+1 .. ng+Ny, :)`. This matches `BlockState2D`'s
convention and lets the streaming kernel run a single `ndrange = (Nx, Ny)`
loop over interior cells with **no** boundary branching: every neighbour
read is in-bounds because the ghost layer is part of the same array.

**SoA `f[i,j,q]`.** Mandated by project memory `feedback_soa_layout` (the
inverse `f[q,i,j]` layout was deleted from Kraken). Coalesced reads in
collision: thread `(i,j)` reads `f[i,j,1..9]` from contiguous memory along
the q-axis stride-multiple. KernelAbstractions handles this layout
transparently on CPU and GPU; the AT3 type parameter lets the struct hold
`Array{T,3}`, `MtlArray{T,3}`, or `CuArray{T,3}` interchangeably.

**Pull-scheme streaming.** Reads neighbour populations into the local cell,
writes once. Compatible with `cell_kind`-aware BCs (the BC kernel can
overwrite the ghost-row source values before `stream_2d!` reads them).
The existing Kraken `stream_2d_kernel!` is pull-scheme.

### 2.4 Why fresh from SLBM `BlockState2D`

Three structural mismatches:

1. **AMR-tree topology vs flat block list.** `BlockState2D` carries
   `Nξ_phys, Nη_phys, n_ghost` and the populations, but its connectivity
   lives outside the struct (in `MultiBlock2D`, in the gmsh loader). An
   AMR leaf must carry `level`, `origin`, `dx`, `parent_id`, `child_ids`,
   and `same_level_neighbor_ids` so the tree manager can walk leaves
   without an external table.

2. **gmsh assumptions.** `BlockState2D` is consumed by the multi-block
   pipeline whose halo exchange logic assumes shared-node-stripped
   non-overlapping faces produced by Gmsh.jl (memory:
   `project_multiblock_v03`, `feedback_nonoverlap_preferred`). AMR leaves
   exchange same-level halos by sibling pointers, and coarse/fine ghosts
   by face records. Different abstraction.

3. **Coarse/fine ghost cells.** SLBM has no concept of two ghost-cell
   providers per face. AMR leaves need `cell_kind` to distinguish
   `GHOST_HALO` (same-level peer fills it) from `GHOST_CF`
   (coarse-side block fills it via D8 reconstruction).

The user explicitly rejected reuse for these reasons. Fresh design under
`src/kraken_e/` with public names prefixed `KrakenE*` / `kraken_e_*`.

## 3. D2 — Same-level uniform-block LBM update

### 3.1 Per-step pipeline

For a single leaf block at level 0 with periodic or simple physical BCs,
one step at time `t → t+1` (lattice units) is:

```text
1. apply_bcs!(block)         # fills f-halo cells with reflected/imposed populations
                             #   (no-op for fully periodic TG; mirror for Couette top wall;
                             #    bounce-back-half for Poiseuille walls)
2. exchange_halo!(block)     # in S2 single-leaf: no-op (or periodic-wrap for TG)
                             #   HOOK PRESENT for S3+ same-level peer exchange
3. compute_macroscopic_2d!   # writes ρ, ux, uy from f (interior cells only)
4. collide!(block)           # BGK in-place: f ← f - ω·(f - f^eq(ρ,u))
5. stream!(block)            # pull-scheme: f_tmp ← stream(f); swap f ↔ f_tmp
```

Order rationale:

- **BCs before exchange**: BCs write physical-boundary ghost cells.
  Same-level exchange writes interior-interface ghost cells. They target
  disjoint subsets of the ghost layer (separated by `cell_kind`), so the
  order between them is semantically free in S2; we fix BCs first as a
  convention so the exchange phase only ever overwrites halo cells with
  `cell_kind == GHOST_HALO`.

- **Exchange before macroscopic**: the moment computation is interior-only
  and does not depend on halo values; placing exchange before macroscopic
  keeps the invariant that all halos are up-to-date before any read that
  uses the wider stencil (relevant for c/f reconstruction in S5+).

- **Macroscopic before collide**: BGK reads `(ρ, ux, uy)` from arrays.
  An equivalent fused variant computes moments inline; we use the
  unfused variant in S2 because it matches existing Kraken kernels
  (`compute_macroscopic_2d!` + `collide_2d!` separately).

- **Collide before stream**: standard LBM. Collision is local; streaming
  pulls from neighbours. Order is fixed by the physics.

The **`exchange_halo!` hook is mandatory** even though it is a no-op in S2.
S3 implements same-level peer exchange (copy from neighbour's interior
border row into this block's halo row). S5 layers in coarse/fine ghost
reconstruction by reading `cf_face_records` and calling D8 routines.
If the pipeline does not declare this hook in S2, S3+ would have to
rewrite the pipeline; the hook is the architectural commitment.

### 3.2 Invariants

The S2 implementation must preserve, **bit-exact where applicable**:

1. **Uniform equilibrium is fixed.** Initialize `f = f^eq(ρ_0, 0, 0)` with
   constant `ρ_0`. After `N` steps of the full pipeline (with periodic
   BCs and no forcing), `max(|f - f^eq(ρ_0, 0, 0)|) ≤ 1e-12` for `T=Float64`.
   This catches: wrong streaming direction, wrong opposite-q for
   half-way bounce-back, wrong `f^eq` weights or `cs²`.

2. **Mass conservation in periodic Taylor-Green.** With fully periodic
   BCs (`apply_bcs!` no-op, `exchange_halo!` periodic wrap), total mass
   `∑_ij ρ` is conserved to `≤ 1e-12` relative drift over 200 steps.
   This catches: streaming losing populations to the void
   (forgotten ghost wrap), collide not being mass-conservative
   (a corrupted `f^eq` weights table).

3. **Momentum conservation in periodic Taylor-Green.** Same conditions:
   total momentum components `∑_ij ρuₓ, ∑_ij ρuᵧ` are conserved to
   `≤ 1e-12` (the analytical TG initial momentum is zero on a periodic
   square, so this is equivalent to bounding the absolute drift).

4. **Guo forcing convention pairing (D2 only if a body-force canary
   is used).** If `apply_bcs!` for Poiseuille is implemented by a constant
   body force `F = (g, 0)` rather than a pressure-gradient inlet/outlet,
   then the Guo correction `u_phys = (∑_q f_q c_q + F/2) / ρ` MUST be
   used both in the macroscopic getter and as the velocity argument
   to `f^eq` in collide. The pre-existing pair
   `collide_guo_2d!` + `compute_macroscopic_forced_2d!` (see fix in
   commit `1b8f8b94`) is the reference convention. Choice of canary
   driver (body-force vs pressure-grad) is the executor's call; whichever
   is picked must be stated in the test file's docstring.

### 3.3 Test plan

Three analytical canaries on a single leaf, `Nx = Ny = 32`, `τ = 0.8`
(so `ω = 1/τ = 1.25`, `ν = (τ - 0.5)/3 = 0.1`), Float64, CPU only.

**T1 — Poiseuille (steady).** Channel of `Ny = 32` fluid cells, periodic
in `x`, no-slip walls implemented as half-way bounce-back via ghost-layer
reflection. The walls are located at the mid-links between fluid cells
and ghost rows: the bottom wall sits at `y_wall_S = 0` (between ghost
`j = 0` and fluid `j = 1`), and the top wall sits at `y_wall_N = Ny`
(between fluid `j = Ny` and ghost `j = Ny + 1`). Fluid cell centres
are therefore at `y_j = j - 0.5` for `j = 1..Ny`, spanning
`y ∈ [0.5, Ny - 0.5]`, and the **physical channel height** is

```text
H_phys = Ny   (= 32 for the canary)
```

Either a constant body force `F = (g, 0)` or a pressure gradient via
inlet/outlet ghost cells; executor's call. Analytical steady velocity:

```text
u_x(y) = (g / (2ν)) · y · (H_phys - y),    y ∈ [0, H_phys]
```

evaluated at `y = y_j = j - 0.5`. The pressure-gradient analogue
replaces `g` with `(dp/dx)/ρ`.
Tune `g` so peak `u_max ≤ 0.05` (Mach ~ 0.087, safely subsonic).
Run until steady state (e.g. `8 · Ny² · τ ≈ 6500` steps, or until
the relative change between consecutive 100-step windows is `≤ 1e-8`).
**Pass criterion**: L2 error of `u_x` against the analytical profile
`≤ 1%` of `u_max`.

**T2 — Couette (steady).** Channel of width `H = Ny = 32`, periodic
in `x`, no-slip bottom wall at `j = 1`, moving top wall at `j = Ny`
with tangential velocity `U_w` imposed via a moving half-way BB
ghost-fill. The Ladd correction is (Krüger Eq. 5.27, with `c_q*`
the OUTGOING velocity at the fluid cell, `q*` = opp(q)):

```text
f_top_ghost,q = f_top,opp(q) - (2/c_s²)·w_q·ρ_w·(c_opp(q) · U_w)
              = f_top,opp(q) + 6·w_q·ρ_w·(c_q · U_w)
```

(using `c_opp(q) = -c_q` and `c_s² = 1/3`). Here `c_q` is the
velocity of the ghost-row population pointing into the fluid (i.e.
the incoming direction for the interior). `ρ_w` is the wall-adjacent
density.

Sanity check for the sign: Kraken's D2Q9 ordering has `q=8 = SW
(c = (-1,-1))` and `q=9 = SE (c = (+1,-1))` (see
`src/kernels/equilibrium_helpers.jl` `moments_2d`). For `U_w = (+U, 0)`
at the top wall:

```text
c_8 · U_w = -U  ⇒  f_top_ghost,8 = f_top,6 - 6·(1/36)·ρ·U = f_top,6 - ρU/6
c_9 · U_w = +U  ⇒  f_top_ghost,9 = f_top,7 + 6·(1/36)·ρ·U = f_top,7 + ρU/6
```

After streaming, the interior cell `(i, Ny)` sees `f8` decreased
by `ρU/6` and `f9` increased by `ρU/6`. Since
`ux = (... - f8 + f9)/ρ` (cf. `moments_2d`), the wall contributes
`+2·(ρU/6)/ρ = +U/3` per fluid cell per step in the right direction —
i.e. a positive top-wall velocity drives positive `u_x`, recovering
the analytical Couette profile.

Pick `U = 0.05`. Analytical steady velocity:
```text
u_x(y) = U · y / H_phys
```
Here `H_phys = Ny` and the analytical profile is evaluated at
`y = j - 0.5` for `j = 1..Ny`, matching the cell-centre convention
used in T1.

**Pass criterion**: L2 error of `u_x` against the analytical profile
`≤ 1%` of `U`.

**T3 — Taylor-Green (transient).** Square periodic domain
`L_x = L_y = Nx = 32`, fully periodic BCs (`apply_bcs!` no-op).
Initialize:
```text
u_x(x, y, 0) = U_0 · cos(2π x / L) · sin(2π y / L)
u_y(x, y, 0) = -U_0 · sin(2π x / L) · cos(2π y / L)
ρ(x, y, 0) = ρ_0 · (1 - (U_0² · M) · ... )    # or just ρ_0 and let it adjust
```
with `U_0 = 0.04`, `ρ_0 = 1`. The exact pressure field is not required
for the energy-decay slope test; initializing `ρ ≡ ρ_0` and
`f = f^eq(ρ_0, u_x, u_y)` is acceptable (a transient mass-coherent
acoustic perturbation will decay quickly under the BGK relaxation).

Analytical energy decay: with `k = 2π / L`, total kinetic energy
`E(t) = (1/2) · ∑_ij ρ (u_x² + u_y²)` satisfies
```text
E(t) = E(0) · exp(-2 · 2 · ν · k² · t) = E(0) · exp(-4 ν k² t)
```
(the factor 2 in `2νk²` from each Fourier mode times 2 for both
`u_x` and `u_y` components — equivalently, the velocity amplitude
decays as `exp(-2 ν k² t)` so energy decays as `exp(-4 ν k² t)`).

**Pass criterion (decay slope)**: fit `log(E(t))` against `t` over
steps `[1, 200]` and recover slope `s_fit`. The expected slope is
`s_exact = -4 ν k²`. Require `|s_fit - s_exact| / |s_exact| ≤ 1%`.

**Pass criterion (mass)**: `|∑_ij ρ(t) - ∑_ij ρ(0)| / (Nx · Ny · ρ_0) ≤ 1e-12`
at `t = 200`.

**Equilibrium-fixed sub-test (auxiliary).** Initialize `f = f^eq(ρ_0, 0, 0)`
with fully periodic BCs, run 10 steps. Require
`max |f - f^eq(ρ_0, 0, 0)| ≤ 1e-12`. This is the cheapest catch for
streaming-direction bugs; it should run **before** the three canaries
in the test file.

## 4. Exit criterion

A single shell command that the Boss will re-run:

```bash
cd /Users/guillaume/Documents/Recherche/Kraken.jl-kraken-e-blocks && \
  julia --project=. -e 'using Pkg; Pkg.test(test_args=["kraken_e_S2"])'
```

This must exit 0 with the test file `test/kraken_e/test_S2_uniform_block.jl`
reporting all of:

- equilibrium-fixed sub-test: `max |Δf| ≤ 1e-12`
- T1 Poiseuille: L2 error `≤ 1%` of `u_max`
- T2 Couette: L2 error `≤ 1%` of `U`
- T3 Taylor-Green: decay-slope error `≤ 1%`, mass drift `≤ 1e-12` relative

The test file MUST print the four numerical results in a single block
so the Boss can confirm by reading the test stdout without re-running.

## 5. Out of scope for S2

Carried by the `LeafBlock2D` struct as **paper-only** fields with sentinel
values (so S3..S7 are field-internals changes, not layout changes):

- AMR-tree management (multi-leaf orchestration, refinement, derefinement);
- coarse/fine face geometry (D4) — `cf_face_records` is an empty placeholder;
- conservative interface fluxes (D5) — `reflux_accumulators` is `T[]`;
- temporal subcycling (D6) — explicitly deferred to S8+;
- moment extraction (D7) and reconstruction (D8) — needed only at c/f ghosts;
- stress consistency (D9) — needed only at c/f;
- wall + interface + Guo composition (D10) — S7;
- viscoelastic coupling (D11) — phase 2;
- epoch remap (D12) — `epoch_remap_buffers` is `T[]`;
- 3D leaf blocks — separate validation ladder (10..11);
- GPU/Metal/CUDA validation — CPU only on S2, deferred to a later session;
- macro-flow benchmarks (Cd, MLUPS, drag) — forbidden until S7 green.

## 6. Failure modes the test must catch

The test plan above is designed so each canary catches at least one class of
implementation bug:

1. **Broken streaming direction** (e.g. swapped `c_q ↔ -c_q`): caught by
   the equilibrium-fixed sub-test (uniform `f^eq` would still be fixed
   under correct streaming, but a sign-flipped stream would write into
   the wrong neighbour and create density gradients) and by the
   Poiseuille profile (which would peak off-centre or be negative).

2. **Wrong `τ` or `ω` plumbing** (e.g. `ω = τ` instead of `1/τ`): caught
   by all three canaries via the wrong viscosity (Poiseuille peak
   off by a factor, TG decay slope off by a factor of `τ²` or so).

3. **Forgotten halo exchange hook** (skipping the no-op step in S2):
   technically passes S2 because the no-op cannot fail, but the test
   must assert (via an `@test` on the function being defined and
   callable) that `kraken_e_exchange_halo!` exists and runs. Otherwise
   S3 would discover the missing hook at the worst possible time.

4. **Ghost-layer not initialized** (uninitialized memory at first step):
   caught by the equilibrium-fixed sub-test if the ghost cells leak
   into interior reads, and by mass drift in TG.

5. **Bounce-back fallback firing on interior cells** (cell-kind logic
   inverted): caught by the Poiseuille profile (wrong slope at the
   wall) and by the Couette profile (top-wall velocity not transmitted).

6. **`f / f_tmp` swap done on a local binding instead of struct field**
   (pitfall recorded in `feedback_blockstate_swap`): caught by the
   equilibrium-fixed sub-test (because after the first step, the
   "post-stream" array would not be the one read next step) and
   loudly by all transient canaries.

7. **Guo forcing convention mismatch** (only relevant if T1 uses a
   body force): caught by the Poiseuille profile (peak off by `1+O(F)`
   if `F/2` correction is dropped) and by the steady mass-flux
   relation. The pair `collide_guo_2d!` + `compute_macroscopic_forced_2d!`
   is the reference.

8. **Equilibrium weights table corruption** (one weight off): caught
   trivially by the equilibrium-fixed sub-test.

9. **Wrong dimensions of the extended grid** (e.g. allocating
   `(Nx, Ny, 9)` instead of `(Nx+2, Ny+2, 9)`): caught by the
   streaming kernel reading out-of-bounds or by the BC kernel
   writing to invalid indices; on a managed Julia array, this manifests
   as an immediate exception, which the test must let propagate
   (no `try/catch` around the canary runs).

End of derivation.
