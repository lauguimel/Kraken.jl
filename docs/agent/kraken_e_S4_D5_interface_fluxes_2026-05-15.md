# Kraken-E S4 — D5 conservative FVFD interface fluxes (2-block patch)

Date: 2026-05-15
Branch: `dev/kraken-e-fvfd-blocks`
Worktree: `/Users/guillaume/Documents/Recherche/Kraken.jl-kraken-e-blocks`
Plan: [`kraken_e_fvfd_interface_plan_2026-05-15.md`](kraken_e_fvfd_interface_plan_2026-05-15.md) §10 D5, §13 ladder step 5, §14 stop criteria
Roadmap: [`kraken_e_roadmap.md`](kraken_e_roadmap.md) row S4
Predecessor: [`kraken_e_S3_D3_D4_block_fvfd_and_cf_faces_2026-05-15.md`](kraken_e_S3_D3_D4_block_fvfd_and_cf_faces_2026-05-15.md)

## 1. Purpose and scope

S4 delivers the first explicit conservative finite-volume update on a
two-block coarse/fine patch built from the S3 geometry record
(`CFFaceRecord2D`). The deliverable proves the plan §14 stop criterion:

```text
coarse/fine FVFD fluxes telescope to roundoff on a two-level patch.
```

Concretely, on a 2-block Cartesian-aligned patch (1 coarse block on the
left, 1 fine block on the right, sharing a single c/f interface that
consists of 1 coarse face = 2 fine subfaces in 2D), an explicit Euler
update of a scalar conserved variable `U` with a **prescribed analytic
flux velocity field** must satisfy:

1. **Algebraic telescoping** at every step:
   ```text
   F_coarse · A_coarse - Σ_k F_fine_k · A_fine_k  =  0   (to ≤ 1e-14)
   ```

2. **Global conservation** of `Σ_i U_i · V_i` on a fully-periodic or
   fully-closed patch:
   ```text
   |Σ_i U_i^{n} · V_i - Σ_i U_i^{0} · V_i|  ≤  1e-12 · |Σ_i U_i^0 · V_i|
   for n = 1, …, 100 explicit-Euler steps.
   ```

The scope is intentionally narrow:

- **scalar mass-like quantity only** (no LBM populations, no moments);
- **prescribed flux velocity** `v(x, y) = (v_x, v_y)`, analytic and
  constant in time (uniform along x in the canary);
- **coarse and fine advance with the same `dt`** (no subcycling — D6 is
  deferred to S8);
- **2-block patch** (1 coarse + 1 fine) sharing exactly one c/f face
  (which is geometrically 1 coarse face = 2 fine subfaces);
- **2D only** (3D is plan-deferred);
- **CPU only** (KernelAbstractions-backend-agnostic by construction;
  GPU port deferred until S7).

Out of scope (deferred):

- LBM coupling at the interface (S5, D7+D8);
- moment reconstruction (S5);
- subcycling and temporal reflux (D6, S8);
- wall + Guo composition (S7, D10);
- non-Cartesian-aligned interfaces, oblique blocks (deferred);
- viscoelastic (post-S7).

## 2. Geometric setup (2-block patch)

### 2.1 Block placement

The patch consists of two `LeafBlock2D` instances:

```text
              y ↑
                │
   ┌─────────────┬─────────────┐
   │             │             │
   │   COARSE    │    FINE     │
   │             │             │
   │  Nx_c × Ny_c│  Nx_f × Ny_f│
   │  dx_c = 2 h │  dx_f = h   │
   │             │             │
   └─────────────┴─────────────┘
   origin_c     origin_f         → x
   (0, 0)     (Lx_c, 0)
```

with

```text
Lx_c = Nx_c · dx_c
Ly_c = Ny_c · dx_c
Ly_f = Ny_f · dx_f = 2 · Ny_c · (dx_c / 2) = Ly_c     ← horizontal alignment
Nx_f and dx_f free; canary: Nx_f = 2 · Nx_c, dx_f = dx_c / 2.
```

Canary parameters: `Nx_c = Ny_c = 8`, `Nx_f = 16`, `Ny_f = 16`,
`dx_c = 1.0`, so `dx_f = 0.5`. Total cells = 64 + 256 = 320.

### 2.2 c/f interface enumeration

The shared interface is the east face of the coarse block at column
`I = Nx_c`. It contains `Ny_c` coarse faces, each splitting into 2 fine
subfaces (the west faces of fine column `i = 1`). Each coarse face
`(I, J)` is encoded by:

```text
record_J = kraken_e_build_cf_face_record_2d(T;
    coarse_block_id = id_c, fine_block_id = id_f,
    coarse_index = (Nx_c, J), fine_indices = ((1, 2J - 1), (1, 2J)),
    axis = KRAKEN_E_CF_FACE_X, side = KRAKEN_E_CF_FACE_HI,
    coarse_origin = (0, 0), coarse_dx = dx_c,
)
```

producing `coarse_area = dx_c`, `fine_areas = (dx_c/2, dx_c/2)`,
`fine_to_coarse_weights = (1/2, 1/2)`, `normal = (+1, 0)`. There are
`Ny_c` such records. The S3 D4 invariants guarantee `|C| = |F1| + |F2|`
exactly.

### 2.3 Outer boundary (canary choice)

The simplest configuration that proves telescoping without confounding
boundary effects is **fully periodic**:

- y-periodic on both blocks (top ↔ bottom of each block);
- x-periodic on the **outer pair**, i.e. the west face of the coarse
  block wraps to the east face of the fine block.

This makes the patch topologically a torus with one internal c/f
interface (the coarse–fine seam) and one outer same-level seam (the
fine–coarse wrap). All non-c/f faces are same-level and use the
standard FVFD same-level flux (computed from the cell-centred field and
upwinded by `v`).

Alternative: fully-closed (Neumann ∂U/∂n = 0 ⇔ zero flux). The
conservation invariant is identical and the implementation is simpler
because outer faces contribute `F_outer = 0`. The canary tests both
variants; the **default** is periodic because it exercises a non-trivial
same-level flux loop.

## 3. Flux scheme

### 3.1 Conservation law (scalar)

For a cell `i` with volume `V_i`, the explicit-Euler update of a
conserved scalar `U_i` under a prescribed velocity field `v(x, y)` is

```text
U_i^{n+1} = U_i^n - (dt / V_i) · Σ_{faces f of i} F_f · A_f · n_{f→outward}
```

where `F_f` is a face-centred numerical flux (face-averaged
`v · U`-product) and `A_f` is the face length in 2D. The
sign convention is: `n_{f→outward}` is the outward-pointing unit normal
from cell `i` across face `f`.

We pick the **first-order upwind face flux** (sufficient for an
algebraic telescoping proof; higher-order limiters do not change the
telescoping identity because they still produce a single face-averaged
value per face):

```text
F_f = v_f · U_upwind(f)
```

with `v_f` the face-normal component of `v` at the face centre and
`U_upwind(f) = U_{i_upwind}` where `i_upwind = i_left(f)` if `v_f > 0`,
else `i_right(f)` (one-sided index along the face normal).

### 3.2 Same-level faces

For two same-level cells sharing a face `f` with outward normals
`+n` and `-n` from the two sides, the flux contribution is
**equal and opposite**:

```text
contribution to cell L = -F_f · A_f       (n_outward(L) = +n, sign flips below)
contribution to cell R = +F_f · A_f       (n_outward(R) = -n)
```

When summed over both cells, the contributions cancel exactly:
`(-F_f · A_f) + (+F_f · A_f) = 0`. This is the standard FV
telescoping identity at same-level faces.

### 3.3 Coarse/fine faces — the telescoping identity

At a c/f face encoded by record `record_J`, the coarse cell
`(I_c, J_c) = (Nx_c, J)` and the two fine cells
`(i_f, j_f) ∈ {(1, 2J - 1), (1, 2J)}` share the interface.

Define:

- `F_coarse(J)` = numerical flux on the coarse face (a single value);
- `F_fine_k(J)` for `k = 1, 2` = numerical fluxes on each fine subface;
- `A_coarse(J) = dx_c`, `A_fine_k(J) = dx_c/2`.

The **telescoping closure** required for conservation is

```text
F_coarse(J) · A_coarse(J) = F_fine_1(J) · A_fine_1(J)
                          + F_fine_2(J) · A_fine_2(J).        (★)
```

Equivalently, factoring out `A_coarse = 2 · A_fine_k`:

```text
F_coarse(J) = (1/2) · F_fine_1(J) + (1/2) · F_fine_2(J)
            = w_1 · F_fine_1(J) + w_2 · F_fine_2(J),
```

where `(w_1, w_2) = record_J.fine_to_coarse_weights` (the S3 D4
quadrature). Because in D4 we proved
`w_1 + w_2 = 1` exactly and `w_k = 1/2` exactly,
the identity holds **algebraically** as long as we **construct**
`F_coarse(J)` by reduction from `F_fine_k(J)`.

This is the **single non-negotiable design choice** for S4:

```text
The coarse face flux is RECONSTRUCTED from the two fine subface
fluxes via the S3 D4 quadrature weights. It is never computed
independently from coarse-side data.
```

Any other choice (e.g. compute `F_coarse` from a coarse upwind state,
then compute `F_fine_k` from fine upwind state, then "correct" the
difference) reintroduces the AMR-D failure mode (per-orientation
ledger ≠ moment-mortar). Reconstruction is one-sided: fine wins.

### 3.4 Per-block update

Within the coarse block:

```text
U_c[I, J]^{n+1} = U_c[I, J]^n
   - (dt / V_c) · [
       F^E_same(I, J) · A_c  ·  +1       if I < Nx_c
     + F_coarse(J)    · A_c  ·  +1       if I == Nx_c        ← c/f east face
     - F^W_same(I, J) · A_c  ·  +1       if I > 1
     - F^W_outer(J)   · A_c  ·  +1       if I == 1            ← periodic wrap
     + F^N_same(I, J) · A_c  ·  +1       (y-periodic)
     - F^S_same(I, J) · A_c  ·  +1       (y-periodic)
   ]
```

with `V_c = dx_c^2` (2D cell volume = area).

Within the fine block:

```text
U_f[i, j]^{n+1} = U_f[i, j]^n
   - (dt / V_f) · [
       F^E_same(i, j) · A_f  ·  +1       if i < Nx_f
     + F^E_outer(j)   · A_f  ·  +1       if i == Nx_f         ← periodic wrap
     - F_fine_k(J)    · A_f  ·  +1       if i == 1            ← c/f west face
                                                                k chosen from j
     - F^W_same(i, j) · A_f  ·  +1       if i > 1
     + F^N_same(i, j) · A_f  ·  +1
     - F^S_same(i, j) · A_f  ·  +1
   ]
```

with `V_f = dx_f^2`.

### 3.5 Algebraic conservation proof

Sum `Σ_i U_i · V_i` over both blocks. Every same-level face inside a
block contributes `(±F_f · A_f)` to one cell and `(∓F_f · A_f)` to the
opposite cell of the same block (or across the outer periodic wrap
between blocks at matching `dx`). They cancel pairwise.

The c/f face contributions are:

```text
coarse side: + F_coarse(J) · A_coarse(J)         (one term, for cell (Nx_c, J))
fine side:   - F_fine_1(J) · A_fine_1(J)
             - F_fine_2(J) · A_fine_2(J)         (two terms, for cells (1, 2J-1) and (1, 2J))
```

Total c/f contribution per coarse face index `J`:

```text
F_coarse(J) · A_coarse(J) - F_fine_1(J) · A_fine_1(J) - F_fine_2(J) · A_fine_2(J)
```

By construction (§3.3, identity ★), this is **identically zero in
floating-point** (the right-hand side is computed first; the left-hand
side is then `(w_1 · F_fine_1) · (2 · A_fine_1) + (w_2 · F_fine_2) ·
(2 · A_fine_2) = F_fine_1 · A_fine_1 + F_fine_2 · A_fine_2` when
`w_k = 1/2` and `A_coarse = 2 · A_fine_k`). The factor-of-two relations
are exact in binary float (1/2 is representable, and doubling is exact
for any normal float). The arithmetic `(0.5 · F_1) · 1.0 + (0.5 · F_2)
· 1.0` versus `F_1 · 0.5 + F_2 · 0.5` differs by at most one
roundoff `ulp` per term, hence the algebraic identity holds **to a
single ulp**, well within the 1e-14 absolute threshold for canary
magnitudes (`F_k · A_f ~ O(1)` lattice units).

The outer periodic wrap between the fine east face and the coarse west
face is **not** a c/f interface: the wrap pairs blocks of different
`dx`, but since both blocks see the *same* face value through the wrap
and our scalar `U` is cell-centred, we treat the wrap as **two
independent same-level periodic faces with face areas equal to their
own dx**. This is geometrically inconsistent (the fine east edge is
twice the height it should be relative to the coarse west edge, except
the y-resolution matches by construction `Ny_f = 2 Ny_c`). The wrap is
handled at the **coarse y-resolution**: a coarse west face at `(1, J)`
pairs with the two fine east faces at `(Nx_f, 2J-1)` and
`(Nx_f, 2J)`, and the **same telescoping identity** as the inner
c/f face applies (this is just a second c/f interface, mirror-image).

To avoid two c/f interfaces in one canary (which would double the bug
surface), the **default** canary uses **fully-closed** outer BCs
(`F_outer = 0`); a secondary periodic-y + closed-x variant exercises
the same-level y-periodic loop without an outer x-c/f wrap.

We restrict the periodic canary to **closed-x + periodic-y** (the
simplest non-trivial outer BC). This leaves exactly one c/f interface
(the inner coarse-east ↔ fine-west seam) and same-level y-periodic
fluxes that cancel trivially.

### 3.6 Stability and timestep

The first-order upwind explicit-Euler scheme is stable under the CFL
condition

```text
dt · max(|v_x|/dx_f, |v_y|/dx_f) ≤ CFL_max = 1
```

(the fine block sets the constraint because `dx_f < dx_c`). For the
canary we pick `CFL = 0.5` to stay comfortably inside the stability
envelope.

```text
v = (0.1, 0.0)   (uniform along x, zero y-component)
dt = 0.5 · dx_f / |v_x| = 0.5 · 0.5 / 0.1 = 2.5
```

After 100 steps the wave travels `100 · dt · v_x = 25` lattice units,
i.e. through the patch once with the x-periodic wrap (`Lx_c + Lx_f =
8 + 8 = 16`). Since we use closed-x, the test cuts the velocity to
`v = (0.1, 0.0)` and uses a **periodic-y + closed-x** outer BC — there
is no x-wrap to worry about, the wave hits the fine east wall (closed)
and the coarse west wall (closed). Closed walls have `F = 0`, so no
flux is added there.

For closed-x, after 100 steps the wave has not reached steady state but
that does not matter for the conservation test: **total mass is
conserved regardless of the velocity field choice**, provided no fluxes
leave the domain.

## 4. Initial condition (canary)

A non-uniform but smooth initial field exercises both same-level and
c/f fluxes. We choose a 2D Gaussian bump centred in the coarse block:

```text
U(x, y, 0) = exp(-((x - x_c)^2 + (y - y_c)^2) / (2 σ^2))
```

with `(x_c, y_c) = (Lx_c / 2, Ly_c / 2) = (4, 4)` (centre of coarse
block), `σ = 1.5`. This ensures the initial field has gradients of
`O(1)` near the c/f interface, so the flux discrepancy (if any) would
be observable at `O(10^{-2})` magnitude — `1e-14` threshold is 12
orders below.

The initial field is sampled at cell centres in each block; the cell
volume integral of `U` is approximated as `Σ_i U_i · V_i`.

## 5. Tests (S4 canary)

File: `test/kraken_e/test_S4_telescoping_2d.jl`.

Single `@testset "Kraken-E S4 D5"`, CPU, Float64, three blocks of
assertions.

### 5.1 Algebraic telescoping (every face, every step)

After constructing the patch and the c/f records, fill `U_c, U_f` with
the Gaussian IC. Apply 100 explicit-Euler steps with the prescribed
velocity field. At every step `n` and every coarse face record
`record_J`:

1. compute `F_fine_1, F_fine_2` from the fine block's west boundary
   data;
2. reconstruct `F_coarse = w_1 · F_fine_1 + w_2 · F_fine_2`;
3. assert
   ```text
   abs(F_coarse · A_coarse - F_fine_1 · A_fine_1 - F_fine_2 · A_fine_2) ≤ 1e-14
   ```

Track the max of this absolute error over all steps and all faces;
report it.

### 5.2 Global conservation (100 steps)

Track `M(n) = Σ_i U_c[i,j]^n · V_c + Σ_i U_f[i,j]^n · V_f`.

Assert

```text
maximum_n abs(M(n) - M(0)) / abs(M(0)) ≤ 1e-12
```

(relative drift, since `M(0) ~ 1` for the chosen IC the absolute
threshold ~ 1e-12 is also satisfied).

### 5.3 Bit-identical telescoping with weights = (1/2, 1/2)

Verify that with the S3 weights stored as the literal `0.5, 0.5` in
the record, the reconstruction is exact at every face for the analytic
`F_fine_k` values used in the canary, to within a single ulp. The
test asserts the **absolute** identity:

```text
abs(F_coarse · A_coarse - sum_k F_fine_k · A_fine_k) == 0    (literal zero, FP)
```

is allowed to be `≤ 4 · eps(Float64) · max_k |F_fine_k · A_fine_k|`
(a single dot-product roundoff). The exit criterion requires `≤ 1e-14`,
which is `~50 · eps`; the test reports both the absolute error and
the unit-of-last-place count.

### 5.4 Regression for S2 + S3

The runner runs `test_args = ["kraken_e_S4", "kraken_e_S3",
"kraken_e_S2"]` in the same `Pkg.test` invocation; S2 and S3 canaries
remain green with their prior metrics.

## 6. Failure modes the S4 test must catch

1. **Coarse face flux computed independently of fine.** If
   `F_coarse(J)` is derived from a coarse-side upwind state rather than
   reduced from `Σ_k w_k · F_fine_k`, the telescoping error grows as
   `O(|grad U| · dx_c)` (~ 1e-1 for our IC), trivially failing 1e-14.
2. **Wrong weights.** If the implementation hardcodes `(1, 1)` (no
   half) or `(2/3, 1/3)`, the algebraic identity fails by `O(1)` or
   `O(1e-1)` respectively.
3. **Wrong face area.** Using `A_coarse = dx_f` (forgetting the c/f
   factor of 2) breaks both identities by `O(1)`.
4. **Sign error on the c/f side.** If the coarse-side normal sign is
   flipped, mass leaks out at the interface at `O(dt · F_coarse)` ~
   `0.25` per step — global drift catastrophic.
5. **Outer BC leak.** If closed-x walls accidentally apply a non-zero
   flux (e.g. periodic wrap reaches in), `M(n)` drifts; caught by
   §5.2.
6. **Indexing bug between fine subface index `k ∈ {1, 2}` and
   fine cell `j_f`.** If `k = 1 ↔ j_f = 2J` instead of `j_f = 2J - 1`,
   the algebraic identity §5.1 still holds (it sums over `k`), but
   the cell-by-cell flux update is wrong and the field develops
   asymmetric error visible in the conservation test at step ~10.
7. **`fine_to_coarse_weights` not retrieved from the record.** Hard-
   coding `0.5` in the reflux step rather than reading
   `record.fine_to_coarse_weights` defeats the purpose of S3; caught
   by introducing a synthetic mutated-weights test variant (not in
   §5.1 but documented for completeness — Codex may add a parametric
   test if budget permits).

## 7. Implementation files

### 7.1 `src/kraken_e/interface_flux_2d.jl` (new, ~150-200 lines)

Pure functions, no kernels for S4 (CPU loops; backend-agnostic
KernelAbstractions promotion is plan-deferred). Public API:

```julia
struct ScalarFluxField2D{T<:AbstractFloat,AT<:AbstractArray{T,2}}
    east :: AT     # shape (Nx+1, Ny) — face fluxes on x-normal faces
    north :: AT    # shape (Nx, Ny+1) — face fluxes on y-normal faces
end

function compute_same_level_upwind_fluxes_2d!(
    flux::ScalarFluxField2D, U::AbstractArray{T,2},
    vx::T, vy::T, dx::T;
    closed_x::Bool=true, periodic_y::Bool=true,
) where {T}
    ...
end

function reconstruct_coarse_flux_from_fine_2d(
    record::CFFaceRecord2D{T}, F_fine_1::T, F_fine_2::T,
)::T where {T}
    w1, w2 = record.fine_to_coarse_weights
    return w1 * F_fine_1 + w2 * F_fine_2
end

function apply_cf_flux_to_coarse_2d!(
    U_c::AbstractArray{T,2}, records::Vector{CFFaceRecord2D{T}},
    F_fine_per_record::Vector{NTuple{2,T}},
    dt::T, dx_c::T,
) where {T}
    # For each record: replace the coarse-side east-face flux by the
    # reconstructed F_coarse, accumulate dt * F * A / V into U_c.
    ...
end

function apply_cf_flux_to_fine_2d!(
    U_f::AbstractArray{T,2}, records::Vector{CFFaceRecord2D{T}},
    F_fine_per_record::Vector{NTuple{2,T}},
    dt::T, dx_f::T,
) where {T}
    # Symmetric for the fine-side west face.
    ...
end

# Telescoping diagnostic (used in tests):
function cf_flux_telescoping_error(
    record::CFFaceRecord2D{T}, F_fine_1::T, F_fine_2::T,
)::T where {T}
    F_c = reconstruct_coarse_flux_from_fine_2d(record, F_fine_1, F_fine_2)
    return abs(F_c * record.coarse_area
             - F_fine_1 * record.fine_areas[1]
             - F_fine_2 * record.fine_areas[2])
end
```

### 7.2 `src/kraken_e/two_block_patch_2d.jl` (new, ~150 lines)

```julia
struct TwoBlockPatch2D{T,B<:LeafBlock2D{T}}
    coarse :: B
    fine   :: B
    cf_records :: Vector{CFFaceRecord2D{T}}   # length Ny_c
end

function build_two_block_patch_2d(
    ::Type{T}=Float64;
    Nx_c::Int=8, Ny_c::Int=8, dx_c::T=one(T),
    Nx_f::Int=16, Ny_f::Int=16,
) where {T}
    @assert Ny_f == 2 * Ny_c  "fine y-resolution must double the coarse"
    coarse = allocate_leaf_block_2d(T; Nx=Nx_c, Ny=Ny_c, dx=dx_c,
                                    origin=(zero(T), zero(T)), id=1)
    dx_f = dx_c / T(2)
    fine = allocate_leaf_block_2d(T; Nx=Nx_f, Ny=Ny_f, dx=dx_f,
                                  origin=(T(Nx_c) * dx_c, zero(T)), id=2)
    records = [
        kraken_e_build_cf_face_record_2d(
            T; coarse_block_id=1, fine_block_id=2,
            coarse_index=(Nx_c, J),
            fine_indices=((1, 2J - 1), (1, 2J)),
            axis=KRAKEN_E_CF_FACE_X, side=KRAKEN_E_CF_FACE_HI,
            coarse_origin=(zero(T), zero(T)), coarse_dx=dx_c,
        )
        for J in 1:Ny_c
    ]
    return TwoBlockPatch2D(coarse, fine, records)
end

function patch_total_mass(patch::TwoBlockPatch2D{T}, U_c, U_f) where {T}
    Vc = patch.coarse.dx^2
    Vf = patch.fine.dx^2
    return sum(U_c) * Vc + sum(U_f) * Vf
end

function explicit_euler_step!(
    patch::TwoBlockPatch2D{T}, U_c, U_f, vx::T, vy::T, dt::T;
    closed_x::Bool=true, periodic_y::Bool=true,
) where {T}
    # 1. compute same-level fluxes inside each block
    # 2. compute fine-side west fluxes at the c/f interface (per record)
    # 3. reconstruct coarse east flux per record (telescoping)
    # 4. apply explicit-Euler update to each block
    # returns: telescoping_max_err :: T
    ...
end
```

The interior arrays `U_c, U_f` are plain Float64 matrices of shape
`(Nx_c, Ny_c)` and `(Nx_f, Ny_f)` (not the LeafBlock2D's `ρ` field —
S4 stays decoupled from LBM populations, that's S5's job).

### 7.3 `src/kraken_e/leaf_block.jl` (additive only)

No required change for S4. The patch struct owns its own scalar field
buffers; `LeafBlock2D` is used only for geometry (`dx`, `Nx`, `Ny`,
`origin`).

### 7.4 `src/kraken_e/KrakenE.jl` (additive include lines)

```julia
include("interface_flux_2d.jl")
include("two_block_patch_2d.jl")
```

### 7.5 `src/Kraken.jl` (additive exports)

```julia
# Kraken-E S4 exports
export ScalarFluxField2D, TwoBlockPatch2D
export compute_same_level_upwind_fluxes_2d!
export reconstruct_coarse_flux_from_fine_2d
export apply_cf_flux_to_coarse_2d!, apply_cf_flux_to_fine_2d!
export cf_flux_telescoping_error
export build_two_block_patch_2d, explicit_euler_step!, patch_total_mass
```

### 7.6 `test/kraken_e/test_S4_telescoping_2d.jl` (new)

Single `@testset` covering §5.1 telescoping, §5.2 conservation, §5.3
ulp-level identity.

### 7.7 `test/runtests.jl` (one new include line)

```julia
("kraken_e_S4" in ARGS || isempty(ARGS)) && include("kraken_e/test_S4_telescoping_2d.jl")
```

## 8. Allowed edit zones (S4)

- `docs/agent/kraken_e_S4_D5_interface_fluxes_2026-05-15.md` (this
  file; Department).
- `docs/agent/kraken_e_roadmap.md` (Department, at the very end of
  S4).
- `src/kraken_e/interface_flux_2d.jl` (Codex, new).
- `src/kraken_e/two_block_patch_2d.jl` (Codex, new).
- `src/kraken_e/leaf_block.jl` (Codex, additive only if any helper
  needed; expect zero change).
- `src/kraken_e/KrakenE.jl` (Codex, two new `include` lines).
- `src/Kraken.jl` (Codex, exports only).
- `test/kraken_e/test_S4_telescoping_2d.jl` (Codex, new).
- `test/runtests.jl` (Codex, one new `include` gated on
  `"kraken_e_S4" in ARGS || isempty(ARGS)`).

Read-only: `src/fvfd/`, `src/multiblock/`, `src/kernels/`,
`src/drivers/`, `src/lattice/`, `src/io/`, `Project.toml`,
`Manifest.toml`, every file in `src/kraken_e/` not listed above as
writeable.

## 9. Exit criterion

```bash
cd /Users/guillaume/Documents/Recherche/Kraken.jl-kraken-e-blocks && \
  julia --project=. -e 'using Pkg; Pkg.test(test_args=["kraken_e_S4","kraken_e_S3","kraken_e_S2"])'
```

Must exit 0 with all of:

- §5.1 max telescoping error over all faces × all steps ≤ 1e-14
  absolute (target: ≤ 4 · eps(Float64) ≈ 1e-16 per face, well below);
- §5.2 max relative conservation drift over 100 steps ≤ 1e-12;
- §5.3 ulp-level identity reports `≤ 4 · eps` per face;
- S2 regression: 396/396 (Poiseuille L2 = 0.05%, Couette L2 = 4e-6,
  TG slope = 0.08%, mass drift 1e-14 — unchanged);
- S3 regression: 25 assertions remain green at ≤ 1.5e-15.

## 10. Out of scope (deferred to later sessions)

- LBM moment extraction / population reconstruction at the c/f ghost
  (S5, D7+D8). The scalar field `U` in S4 is **not** an LBM moment;
  it is a plain advected scalar.
- Reflux accumulator in `LeafBlock2D.reflux_accumulators` (currently
  empty `Vector{T}()`); S4 stores fluxes in the local
  `ScalarFluxField2D` only. The accumulator field is reserved for D6
  (subcycling) at S8.
- Subcycling. Coarse and fine advance with the same `dt`.
- Wall + Guo composition (S7, D10).
- 3D telescoping (one coarse face = 4 fine faces). S4 is 2D only;
  the algebraic identity generalises but the indexing bookkeeping
  changes substantially.
- GPU validation. The arrays are plain `Matrix{Float64}`; promoting to
  `MtlArray` or `CuArray` is a mechanical change but defers until S7
  to keep the canary tractable.

## 11. Stop criteria (plan §14)

S4 immediately stops and Boss is notified if any of:

- the algebraic telescoping fails the 1e-14 absolute threshold on the
  canary (this is **the** plan §14 stop for D5);
- c/f corner ownership becomes ambiguous when fluxes are integrated
  (e.g. a corner cell receives two non-equal flux contributions from
  two records); → revisit D4 corner ownership (S3, §3.5);
- a two-block patch cannot be constructed because the S3 record API
  forces an ambiguous interface; → revisit D4 record API.

Threshold tuning is **forbidden**. The plan calls for redesign, not
workarounds. If §5.1 fails at 1e-13 or worse, the implementation must
be revised (likely a missing `fine_to_coarse_weights` retrieval or a
spurious independent coarse-side flux); the threshold stays at 1e-14.

End of derivation.
