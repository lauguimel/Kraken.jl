# Kraken-E S3 — D3 FVFD-on-block + D4 c/f Cartesian face geometry

Date: 2026-05-15
Branch: `dev/kraken-e-fvfd-blocks`
Worktree: `/Users/guillaume/Documents/Recherche/Kraken.jl-kraken-e-blocks`
Plan: [`kraken_e_fvfd_interface_plan_2026-05-15.md`](kraken_e_fvfd_interface_plan_2026-05-15.md) §10 D3+D4, §13, §14
Roadmap: [`kraken_e_roadmap.md`](kraken_e_roadmap.md) row S3
Branch contract: [`branch_contract_fvfd_blocks.md`](branch_contract_fvfd_blocks.md)
Predecessor derivation: [`kraken_e_S2_D1_D2_leaf_block_2026-05-15.md`](kraken_e_S2_D1_D2_leaf_block_2026-05-15.md)

## 1. Purpose and scope

S3 delivers two foundations needed before any coarse/fine flux work (S4) or
moment reconstruction (S5):

- **D3**: drive the FVFD operator library (owned by `dev/fvfd-core`, read-only
  here) from a `LeafBlock2D` instance. No FVFD reimplementation. A thin
  **adapter** module exposes block-shaped views (interior + uniform metric)
  to the operators that already live in `src/fvfd/operators_2d.jl`. Exactness
  invariants from the FVFD operator suite (constants → zero gradient, affine
  → exact gradient, constant tensors → zero divergence) are then re-tested
  via the adapter to prove the wiring is correct.

- **D4**: replace the S2 placeholder `CFFaceRecord` empty struct with a real
  Cartesian coarse/fine face record `CFFaceRecord2D` carrying face areas,
  centres, normals, fine-to-coarse quadrature weights, tangential
  interpolation stencils, and corner ownership. The record is the geometric
  contract every S4+ flux assembly will read from. No fluxes are computed in
  S3 — only the geometry and its quadrature.

Out of scope for S3 (deferred):

- conservative flux assembly across the c/f interface (S4, D5);
- reflux accumulators (S4, D6 deferred to S8);
- LBM moment extraction and population reconstruction (S5, D7+D8);
- stress consistency at the c/f interface (S5, D9);
- wall + interface composition (S7, D10);
- viscoelastic coupling (post-S7, D11);
- 3D (D4 c/f geometry is **2D only**; one coarse face splits into two fine
  faces; 3D's four-fine-face fan is deferred to a later session);
- GPU/Metal/CUDA validation (CPU correctness first; D3 adapter stays
  KernelAbstractions-backend-agnostic).

## 2. D3 — FVFD operators on one uniform LeafBlock2D

### 2.1 Source FVFD signatures (read-only)

The relevant FVFD entry points in `src/fvfd/operators_2d.jl` (canonical
signatures, do not edit):

```text
fvfd_velocity_gradient_2d!(
    dudx, dudy, dvdx, dvdy,
    ux, uy, is_solid, dx, dy, bc::FVFDDomainBC2D;
    sync::Bool=true,
)

fvfd_tensor_divergence_2d!(
    fx, fy, tauxx, tauxy, tauyy, is_solid, dx, dy, bc::FVFDDomainBC2D;
    sync::Bool=true,
)

fvfd_cell_velocity_to_faces_2d!(
    ux_face, uy_face, ux, uy, is_solid,
    ux_west, ux_east, uy_south, uy_north,
    bc::FVFDDomainBC2D;
    sync::Bool=true,
)
```

The operator suite assumes:

- **Interior-shaped arrays** sized `(Nx, Ny)`. Indices `(i, j) ∈ 1:Nx × 1:Ny`
  index physical cells. There is no ghost layer in the FVFD convention —
  boundaries are encoded in the `FVFDDomainBC2D` struct and in optional
  open-boundary field rows.
- **Uniform `dx, dy`** scalars (the FVFD `FVFDPatch2D` carries `dx, dy,
  level`). A `LeafBlock2D` carries a single `dx::T` (isotropic) and a
  `level::Int`. For the adapter we set `dy = dx` since Kraken-E leaves are
  isotropic.
- **`is_solid::Matrix{Bool}`** of shape `(Nx, Ny)` indicating obstacle
  cells. In S3 we restrict to fully fluid blocks; the canary builds
  `is_solid = falses(Nx, Ny)`.
- **`FVFDDomainBC2D`** with members `west, east, south, north ∈ {periodic,
  open, wall}`. For S3 canaries on a leaf, we pick periodic-x +
  periodic-y to obtain the strongest exactness invariant (no boundary
  truncation).

### 2.2 LeafBlock2D layout (predecessor S2)

A leaf block stores arrays with **ghost layers**:

```text
LeafBlock2D.f           :: AT3, shape (Nx+2, Ny+2, 9)
LeafBlock2D.ρ           :: AT2, shape (Nx+2, Ny+2)
LeafBlock2D.ux, .uy     :: AT2, shape (Nx+2, Ny+2)
LeafBlock2D.cell_kind   :: Matrix{Int8}, shape (Nx+2, Ny+2)
```

Interior cells live at `(i, j) ∈ (ng+1):(ng+Nx) × (ng+1):(ng+Ny)` with
`ng = 1` in S2. The FVFD operators expect interior-only arrays.

### 2.3 Adapter design — viewless, non-owning

We define a non-allocating adapter that produces **views** into the LBM
arrays sized for the FVFD operators:

```julia
# src/kraken_e/fvfd_block_adapters.jl

@inline kraken_e_interior_view_2d(A2::AbstractArray{T,2}, block::LeafBlock2D) =
    @view A2[(block.ng+1):(block.ng+block.Nx), (block.ng+1):(block.ng+block.Ny)]
```

The view returns an `Nx × Ny` SubArray, which is what the FVFD operators
expect. The adapter never copies. On GPU backends (Metal, CUDA), views of
backend arrays preserve the parent backend, so
`KernelAbstractions.get_backend(view) == get_backend(parent)`. The FVFD
operators dispatch on `get_backend(...)` and run on the correct device.

Three adapter wrappers, one per operator we expose in S3:

```julia
fvfd_velocity_gradient_block_2d!(
    dudx, dudy, dvdx, dvdy,
    block::LeafBlock2D, bc::FVFDDomainBC2D;
    is_solid::Union{Nothing,AbstractArray}=nothing,
    sync::Bool=true,
)

fvfd_tensor_divergence_block_2d!(
    fx, fy, tauxx, tauxy, tauyy,
    block::LeafBlock2D, bc::FVFDDomainBC2D;
    is_solid::Union{Nothing,AbstractArray}=nothing,
    sync::Bool=true,
)

fvfd_cell_velocity_to_faces_block_2d!(
    ux_face, uy_face,
    block::LeafBlock2D, bc::FVFDDomainBC2D;
    is_solid::Union{Nothing,AbstractArray}=nothing,
    ux_west=nothing, ux_east=nothing,
    uy_south=nothing, uy_north=nothing,
    sync::Bool=true,
)
```

Each:

1. Slices interior views of `block.ux`, `block.uy` (or user-supplied scalar
   fields living on the leaf's interior).
2. Defaults `is_solid` to a freshly-allocated `falses(Nx, Ny)` (CPU) — or,
   if Nothing is passed and the adapter is called repeatedly, expects the
   caller to provide a stable buffer. For the S3 canary we allocate once
   in the test.
3. Defaults open-boundary rows (`ux_west, ux_east, ...`) to empty vectors
   when the corresponding BC is not `:open`. The FVFD lowering accepts
   zero-length vectors when the BC is periodic or wall.
4. Forwards `block.dx, block.dx` (isotropic) as `dx, dy`.
5. Calls the underlying FVFD operator unchanged.

The output arrays (`dudx`, `dudy`, ..., `ux_face`, `uy_face`) live in the
caller's space. The adapter does **not** allocate output arrays. The
caller pre-allocates them with the appropriate shape:

- Cell-centred outputs (gradients, tensor divergence): `(Nx, Ny)`.
- x-face velocity output `ux_face`: `(Nx + 1, Ny)`.
- y-face velocity output `uy_face`: `(Nx, Ny + 1)`.

This matches the existing FVFD test fixtures.

### 2.4 D3 exactness invariants

The adapter inherits the FVFD operator exactness. The S3 canary
verifies:

1. **Constants → zero gradient.**
   Fill `block.ux ← α`, `block.uy ← β` on the interior with constants
   `α, β ∈ T`. Periodic-x + periodic-y BCs. Then
   `dudx, dudy, dvdx, dvdy` returned by the adapter must be `≤ 1e-12` in
   absolute value on every interior cell.

2. **Affine → exact gradient.**
   Fill `block.ux(i, j) = α + γₓ · x_c(i) + γᵧ · y_c(j)` and similarly
   `block.uy(i, j) = β + δₓ · x_c(i) + δᵧ · y_c(j)`, with cell-centre
   coordinates `x_c(i) = (i - 1/2) · dx`, `y_c(j) = (j - 1/2) · dx` (the
   FVFD convention; identical to the S2 Poiseuille canary `y = j - 0.5`).
   With periodic BCs the affine field is not globally periodic; we
   therefore use **wall BCs** for the affine test, which switches the FVFD
   stencil to one-sided at the boundary cells and still recovers the
   affine gradient exactly (the one-sided 3-point stencil
   `(3 φ_i - 4 φ_{i-1} + φ_{i-2})/(2 dx)` is exact on affine fields). The
   adapter must therefore return:

   ```text
   dudx == γₓ * ones(Nx, Ny)    (to ≤ 1e-12)
   dudy == γᵧ * ones(Nx, Ny)    (to ≤ 1e-12)
   dvdx == δₓ * ones(Nx, Ny)    (to ≤ 1e-12)
   dvdy == δᵧ * ones(Nx, Ny)    (to ≤ 1e-12)
   ```

3. **Constant tensor → zero divergence.**
   Fill `tauxx ← σ₁`, `tauxy ← σ₂`, `tauyy ← σ₃` on the interior with
   constants. Periodic-x + periodic-y. Then `fx, fy` returned by
   `fvfd_tensor_divergence_block_2d!` must be `≤ 1e-12` everywhere.

4. **Affine tensor → exact constant divergence.** (optional auxiliary)
   Fill `tauxx(i,j) = a₀ + aₓ·x_c + aᵧ·y_c`, etc. With wall BCs the
   adapter returns
   `fx(i,j) = aₓ + bᵧ`, `fy(i,j) = bₓ + cᵧ` where the coefficients are
   from `tauxy` and `tauyy`. Exact to `≤ 1e-12`.

The S3 canary tests (1), (2), (3). Test (4) is optional but included as
a regression hook because the existing FVFD test suite already covers it
(the adapter simply forwards).

### 2.5 GPU layout discipline (forward-compatibility note)

The adapter never allocates inside hot paths. The S3 test allocates the
output arrays and the `is_solid` buffer once. On a future GPU pass, the
caller will allocate backend arrays (`MtlArray`/`CuArray`) of the
appropriate shape, and the views into `block.ux` (which is itself a
`MtlArray` or `CuArray` on GPU) will preserve the backend. Nothing in the
adapter dispatches on `eltype(block.f)` or on the backend — that
dispatch lives inside `KernelAbstractions.get_backend(...)` called by
the underlying FVFD operator.

The adapter functions are non-`@kernel`; they are plain Julia. They do
not contribute to GPU hot-loop cost.

## 3. D4 — Coarse/fine Cartesian face geometry (2D)

### 3.1 Geometry of one c/f face in 2D

Consider a coarse leaf at level `ℓ_c` with cell size `dx_c` and a fine
neighbour at level `ℓ_f = ℓ_c + 1` with cell size `dx_f = dx_c / 2`.
The shared face is a 1D segment in 2D. Without loss of generality, take
the coarse east face of coarse cell `(I, J)`:

```text
coarse east face C_{I,J}^E :
  segment from (x_c^E, y_S) to (x_c^E, y_N)
  where x_c^E = x_origin_c + I · dx_c
        y_S  = y_origin_c + (J - 1) · dx_c
        y_N  = y_origin_c + J · dx_c
  length |C^E| = dx_c
  outward normal n^E = (+1, 0)
```

The fine neighbour's two west faces that share this segment are:

```text
fine west face F1^W : segment (x_f^W, y_S) → (x_f^W, y_S + dx_f),
                    length |F1^W| = dx_f = dx_c / 2
                    centre  c_{F1} = (x_f^W, y_S + dx_f/2)
fine west face F2^W : segment (x_f^W, y_S + dx_f) → (x_f^W, y_N),
                    length |F2^W| = dx_f
                    centre  c_{F2} = (x_f^W, y_S + 3·dx_f/2)
with x_f^W = x_c^E (same plane)
```

Identity (telescoping):

```text
|C^E| = |F1^W| + |F2^W|    (1 = 1/2 + 1/2 in units of dx_c)
```

The coarse face centre is `c_C = (x_c^E, y_S + dx_c/2) = (x_c^E, (y_S + y_N)/2)`.

### 3.2 CFFaceRecord2D struct

Replace the S2 placeholder `CFFaceRecord` (empty struct in
`leaf_block.jl`) with a typed record:

```julia
# src/kraken_e/cf_face_2d.jl

@enum KrakenECFFaceAxis KRAKEN_E_CF_FACE_X KRAKEN_E_CF_FACE_Y
@enum KrakenECFFaceSide KRAKEN_E_CF_FACE_LO KRAKEN_E_CF_FACE_HI

struct CFFaceRecord2D{T<:AbstractFloat}
    # Identity
    coarse_block_id  :: Int
    fine_block_id    :: Int       # neighbour id; -1 if domain boundary
    coarse_index     :: NTuple{2,Int}  # (I, J) of coarse cell whose face this is
    fine_indices     :: NTuple{2,NTuple{2,Int}}  # the two fine cells (i₁,j₁),(i₂,j₂)
    axis             :: KrakenECFFaceAxis        # X = vertical face, Y = horizontal face
    side             :: KrakenECFFaceSide        # LO = west/south, HI = east/north
    # Geometry (all in world units)
    coarse_area      :: T          # length of the coarse face in 2D (= dx_c)
    fine_areas       :: NTuple{2,T}  # lengths of the two fine subfaces (= dx_f each)
    coarse_center    :: SVector{2,T}
    fine_centers     :: NTuple{2,SVector{2,T}}
    normal           :: SVector{2,T}   # unit outward normal from coarse side
    # Quadrature (fine → coarse, conservative)
    fine_to_coarse_weights :: NTuple{2,T}  # w_k such that φ_C = Σ_k w_k · φ_F_k
    # Tangential interpolation (coarse value → fine subface centres)
    # In 2D one face has one tangential direction; the stencil is the two
    # adjacent coarse cells' face-centre values plus the central one.
    # Stored as (i_offset_k, j_offset_k, weight_k) per fine subface k = 1, 2.
    tangential_offsets_1 :: NTuple{3,NTuple{2,Int}}  # (i_off, j_off) for k = 1
    tangential_weights_1 :: NTuple{3,T}              # weights summing to 1
    tangential_offsets_2 :: NTuple{3,NTuple{2,Int}}
    tangential_weights_2 :: NTuple{3,T}
    # Corner ownership (two corners of the coarse face)
    corner_lo_owned  :: Bool   # true if the LO end-vertex belongs to this face
    corner_hi_owned  :: Bool   # true if the HI end-vertex belongs to this face
end
```

The record uses static-typed tuples and `SVector` to remain
GPU-blittable (when promoted to backend arrays later). All weights are
of type `T = eltype(leaf.f)`.

### 3.3 Quadrature weights (fine → coarse)

For a Cartesian coarse face split into two equal fine subfaces:

```text
|C| = dx_c,   |F_k| = dx_f = dx_c / 2,   k ∈ {1, 2}
```

The conservative quadrature for a face-averaged scalar `φ̄_C` from two
fine face-averaged values `φ̄_{F_1}, φ̄_{F_2}` is the **arithmetic mean
weighted by sub-area**:

```text
φ̄_C · |C| = φ̄_{F_1} · |F_1| + φ̄_{F_2} · |F_2|
⇒ φ̄_C = (|F_1| · φ̄_{F_1} + |F_2| · φ̄_{F_2}) / |C|
       = (1/2) · φ̄_{F_1} + (1/2) · φ̄_{F_2}
```

So `fine_to_coarse_weights = (1/2, 1/2)`. The construction is **exact**
on constants and on linear fields, because the 2-point trapezoidal rule
on equal subintervals integrates linear polynomials exactly.

D4 invariant — **fine-to-coarse exactness on constants + affine fields**:

Let `φ(x, y) = α + βₓ x + βᵧ y`. Define the face-averaged value as the
analytic line integral divided by face length. On a vertical face of
length `L`, centred at `(x_f, y_f)`, the face average is

```text
⟨φ⟩_face = (1/L) ∫_{y_f - L/2}^{y_f + L/2} φ(x_f, y) dy
         = α + βₓ x_f + βᵧ y_f
```

i.e. the face average of an affine field equals the field value at the
face centroid. Then:

- coarse face centroid: `(x_c, y_S + dx_c/2)`,
- fine subface centroids: `(x_c, y_S + dx_c/4)`, `(x_c, y_S + 3 dx_c/4)`.

```text
(1/2) · (α + βₓ x_c + βᵧ (y_S + dx_c/4))
+ (1/2) · (α + βₓ x_c + βᵧ (y_S + 3 dx_c/4))
= α + βₓ x_c + βᵧ (y_S + dx_c/2)
= ⟨φ⟩_coarse_face         ✓
```

Exact to machine precision (≤ 1e-12 in Float64).

### 3.4 Prolongation (coarse → fine) tangential interpolation stencil

Going the other way, given coarse face values `φ̄_C` we must produce
fine subface values `φ̄_{F_k}`. The conservative requirement is

```text
(1/2) φ̃_{F_1} + (1/2) φ̃_{F_2} = φ̄_C    (consistency with §3.3)
```

For S3 (no fluxes yet; only geometry + quadrature) the **prolongation
operator** must be exact on affine fields. The simplest stencil that
achieves this uses the coarse face value at the target index `(I, J)`
and the two tangential neighbours `(I, J-1)` and `(I, J+1)` along the
shared face's tangential axis (here, the y-axis):

```text
φ̃_{F_1} = (3/4) · φ̄_{C(I,J)} + (1/4) · φ̄_{C(I,J-1)}     ← lower fine subface
φ̃_{F_2} = (3/4) · φ̄_{C(I,J)} + (1/4) · φ̄_{C(I,J+1)}     ← upper fine subface
```

This is the standard 2-point central tangential interpolation that is
**exact on affine fields**: writing `φ̄_{C(I,J')} = α + βᵧ · y_C(J')`
with `y_C(J') = y_S + (J' - J + 1/2) dx_c`, the weighted combination
recovers the affine value at the fine subface centre to machine
precision.

Encoding in `CFFaceRecord2D`:

```text
tangential_offsets_1 = ((0, 0), (0, -1), (0, +1))   # (i_off, j_off)
tangential_weights_1 = (3/4, 1/4, 0)                # third slot unused for k=1
tangential_offsets_2 = ((0, 0), (0, -1), (0, +1))
tangential_weights_2 = (3/4, 0, 1/4)                # mirror; second slot unused
```

Storing three offsets per fine subface (even though only two are
non-zero) keeps the layout uniform across face axes (X vs Y) and across
sides (LO vs HI), at the cost of one redundant weight per stencil. The
adapter always sums three contributions; weights that should be zero
are stored as exactly `zero(T)`.

For horizontal faces (`axis = KRAKEN_E_CF_FACE_Y`) the tangential axis
becomes x; offsets are `((0, 0), (-1, 0), (+1, 0))` with the same weight
pattern.

D4 invariant — **prolongation exactness on constants + affine fields**:

For a constant `φ ≡ α`, both stencil rows reduce to `(3/4 + 1/4) · α =
α`. Both fine subface values equal `α`. Error `≤ 1e-12`.

For an affine `φ̄_{C(I,J')} = α + βᵧ · y_C(J')`,

```text
φ̃_{F_1} = (3/4)(α + βᵧ y_C(J)) + (1/4)(α + βᵧ y_C(J-1))
        = α + βᵧ · ((3/4) y_C(J) + (1/4) y_C(J-1))
        = α + βᵧ · (y_C(J) - dx_c/4)
        = α + βᵧ · y_{F_1}_centre        ✓
```

since `y_{F_1}_centre = y_S + dx_c/4 = y_C(J) - dx_c/4`. Symmetric for
`F_2`. Error `≤ 1e-12`.

### 3.5 Corner ownership rules

Each coarse face has two end-vertices (in 2D); each vertex is shared by
up to four faces (two coarse-axis faces and two tangential-axis faces).
A deterministic rule prevents two faces from writing the same corner with
different values during flux assembly (S4).

**Rule (deterministic, no ad-hoc tiebreak):**

A coarse face `(I, J, axis, side)` owns its endpoint `(x_v, y_v)`
**iff** the endpoint sits at the LO end of the face's tangential axis
**and** `side == LO`, OR the endpoint sits at the HI end of the face's
tangential axis **and** `side == HI`. Symbolically:

- a vertical (X-axis) face owns its **south** endpoint when `side ==
  LO`, and its **north** endpoint when `side == HI`;
- a horizontal (Y-axis) face owns its **west** endpoint when
  `side == LO`, and its **east** endpoint when `side == HI`.

This rule is the standard staggered-grid Cartesian convention: each
vertex is owned by exactly one face in each axis-side bucket, and the
ownership pattern is a pure function of `(axis, side)` — no neighbour
inspection, no global numbering tie-break.

Encoded as the two booleans `corner_lo_owned` and `corner_hi_owned` in
`CFFaceRecord2D`:

```text
axis = X, side = LO  ⇒  corner_lo_owned = true,  corner_hi_owned = false
axis = X, side = HI  ⇒  corner_lo_owned = false, corner_hi_owned = true
axis = Y, side = LO  ⇒  corner_lo_owned = true,  corner_hi_owned = false
axis = Y, side = HI  ⇒  corner_lo_owned = false, corner_hi_owned = true
```

(For S3 this rule applies to every face; the test verifies that on a
synthetic coarse-fine pair, no two `CFFaceRecord2D` instances claim the
same corner with `true`.)

### 3.6 Builder function

```julia
function kraken_e_build_cf_face_record_2d(
    ::Type{T};
    coarse_block_id::Int, fine_block_id::Int,
    coarse_index::NTuple{2,Int}, fine_indices::NTuple{2,NTuple{2,Int}},
    axis::KrakenECFFaceAxis, side::KrakenECFFaceSide,
    coarse_origin::NTuple{2,T}, coarse_dx::T,
) where {T<:AbstractFloat}
    ...
end
```

The builder fills every field deterministically from the inputs. It is
called once at AMR-tree construction time (S3 canary calls it directly
in tests; S4 will call it from a topology constructor). The builder is
pure: same inputs → same record, bitwise.

## 4. D3 + D4 — Updates to `LeafBlock2D`

The `leaf_block.jl` change is **minimal**: replace the empty
`struct CFFaceRecord end` with the typed `CFFaceRecord2D{T}` (renamed) and
update the parametrisation so a `LeafBlock2D{T,AT2,AT3}` stores
`cf_face_records::Vector{CFFaceRecord2D{T}}`. The S2 allocator continues
to return an empty `CFFaceRecord2D{T}[]` (no faces yet). The S2 sentinel
assertion `isempty(block.cf_face_records)` remains green.

To preserve the S2 export surface, the old name `CFFaceRecord` is kept
as a const alias to `CFFaceRecord2D{Float64}` (so user code importing
`CFFaceRecord` still resolves). The S2 test does
`isempty(block.cf_face_records)`; that still holds.

## 5. Test plan (S3 canaries)

File: `test/kraken_e/test_S3_cf_faces_2d.jl`.

Single `@testset`, CPU, Float64, three blocks of assertions.

### 5.1 D3 adapter exactness (one leaf, 32 × 32)

Allocate one `LeafBlock2D` `Nx = Ny = 32`, `dx = 1.0` (lattice units).
Build the FVFD output buffers (`dudx, dudy, dvdx, dvdy` of shape
`(32, 32)`) and `is_solid = falses(32, 32)`.

1. **Constant**: fill `block.ux ← 0.5`, `block.uy ← -0.25` on the interior;
   zero on ghosts. Build `bc = FVFDDomainBC2D(:periodic, :periodic,
   :periodic, :periodic)`. Call
   `fvfd_velocity_gradient_block_2d!(...)`.
   Assert `maximum(abs, dudx) ≤ 1e-12` and same for `dudy, dvdx, dvdy`.

2. **Affine**: fill `block.ux(i, j) = 0.1 + 0.03 * x_c(i) + 0.07 * y_c(j)`,
   `block.uy(i, j) = -0.2 + 0.05 * x_c(i) - 0.04 * y_c(j)`. Periodic
   BCs are not valid for affine; use `bc = FVFDDomainBC2D(:wall, :wall,
   :wall, :wall)`. Assert:
   - `maximum(abs, dudx .- 0.03) ≤ 1e-12`,
   - `maximum(abs, dudy .- 0.07) ≤ 1e-12`,
   - `maximum(abs, dvdx .- 0.05) ≤ 1e-12`,
   - `maximum(abs, dvdy .+ 0.04) ≤ 1e-12`.

3. **Constant tensor → zero divergence**: fill `tauxx ← 1.3`, `tauxy ←
   -0.7`, `tauyy ← 2.1` on the interior; allocate output `fx, fy`
   of shape `(32, 32)`. Periodic BCs. Call
   `fvfd_tensor_divergence_block_2d!(...)`. Assert
   `maximum(abs, fx) ≤ 1e-12`, `maximum(abs, fy) ≤ 1e-12`.

### 5.2 D4 c/f face geometry & quadrature (synthetic coarse face)

Construct a `CFFaceRecord2D{Float64}` with:

- coarse origin `(0.0, 0.0)`, coarse `dx_c = 1.0`,
- coarse index `(I, J) = (3, 5)`, axis = X, side = HI (east face of cell
  (3, 5)),
- fine indices = ((6, 9), (6, 10)) (illustrative; the two fine cells on
  the west of the fine block whose west face coincides with the coarse
  east face of (3, 5)),
- coarse block id 1, fine block id 2.

Then:

1. **Coarse area = sum of fine areas**:
   `record.coarse_area ≈ sum(record.fine_areas)` to `1e-12`.
2. **Coarse centre = average of fine centres** (linearity of segment
   midpoint):
   `record.coarse_center ≈ (record.fine_centers[1] + record.fine_centers[2]) / 2`
   to `1e-12`.
3. **Normal is unit and outward**:
   `norm(record.normal) ≈ 1` to `1e-12`, and `record.normal == (1, 0)`
   for axis=X, side=HI.
4. **Quadrature weights sum to 1 and are equal**:
   `sum(record.fine_to_coarse_weights) ≈ 1` and
   `abs(record.fine_to_coarse_weights[1] - 0.5) ≤ 1e-12`.

### 5.3 D4 exactness on constants + affine

Construct an analytical affine field
`φ(x, y) = α + βₓ · x + βᵧ · y` with `α = 0.3, βₓ = 0.02, βᵧ = -0.05`.

Compute the face-average values from §3.3 (analytical, using the field
value at the face centroid since affine averages equal centroid values):

```text
φ̄_C       = α + βₓ · cx + βᵧ · cy_C
φ̄_{F_1}   = α + βₓ · cx + βᵧ · cy_{F_1}
φ̄_{F_2}   = α + βₓ · cx + βᵧ · cy_{F_2}
```

Then:

1. **Fine-to-coarse quadrature exactness**:
   ```text
   abs(record.fine_to_coarse_weights[1] · φ̄_{F_1}
     + record.fine_to_coarse_weights[2] · φ̄_{F_2}
     - φ̄_C) ≤ 1e-12
   ```

2. **Prolongation exactness on constants**: take `α = 0.5, βₓ = βᵧ =
   0`. Construct a 3-cell coarse face value stencil `φ̄_C[J-1] =
   φ̄_C[J] = φ̄_C[J+1] = 0.5`. Apply the tangential interpolation
   stencil from `tangential_weights_1` and `tangential_offsets_1`
   (treating the `(0, j_off)` offsets as 1-D look-ups on a length-3
   tangential array). Assert the result equals `0.5` to `1e-12`.

3. **Prolongation exactness on affine**: take `α, βₓ, βᵧ` non-zero.
   Construct the 3-cell coarse face value stencil
   `φ̄_C[J + δ] = α + βₓ cx + βᵧ · (cy_C + δ · dx_c)` for `δ ∈ {-1, 0, 1}`.
   Apply the stencil for `F_1` and `F_2`. Assert each result matches
   the analytical face average at the fine subface centroid to
   `1e-12`.

### 5.4 D4 corner ownership uniqueness

Build four `CFFaceRecord2D` instances covering the four (axis, side)
combinations sharing one synthetic interior corner of a 2D coarse cell.
Assert that **exactly one** record's `corner_lo_owned` is `true` for
that vertex (and the other three have it `false`), and similarly for
the HI corner via a second four-record fan.

### 5.5 S2 regression

The S3 test file does not invoke any S2 canary directly; the runtests
runner runs both `kraken_e_S2` and `kraken_e_S3` test_args in the same
`Pkg.test` invocation, so S2 regression is automatic.

## 6. Exit criterion

A single shell command, re-runnable by the Boss:

```bash
cd /Users/guillaume/Documents/Recherche/Kraken.jl-kraken-e-blocks && \
  julia --project=. -e 'using Pkg; Pkg.test(test_args=["kraken_e_S3","kraken_e_S2"])'
```

This must exit 0 with:

- **D3 invariants** §5.1 (1)(2)(3): all errors `≤ 1e-12`.
- **D4 invariants** §5.2..§5.4: all errors `≤ 1e-12`; corner ownership
  uniqueness holds exactly (boolean, no tolerance).
- **S2 regression**: all 396 existing S2 tests remain green (the S2
  canary metrics from the previous session — Poiseuille L2 = 0.05%,
  Couette L2 = 4e-6, TG slope err = 0.08%, mass drift 1e-14 — must hold
  within their established thresholds).

The test file prints all D3 + D4 numerical residuals in a single block
so the Boss can confirm by reading the test stdout.

## 7. Allowed edit zones (S3)

- `docs/agent/kraken_e_S3_D3_D4_block_fvfd_and_cf_faces_2026-05-15.md`
  (this file; Department).
- `docs/agent/kraken_e_roadmap.md` (Department, at the very end of S3).
- `src/kraken_e/fvfd_block_adapters.jl` (Codex, new).
- `src/kraken_e/cf_face_2d.jl` (Codex, new).
- `src/kraken_e/leaf_block.jl` (Codex, additive only: replace `struct
  CFFaceRecord` placeholder body, add typed alias).
- `src/kraken_e/KrakenE.jl` (Codex, two new `include` lines).
- `src/Kraken.jl` (Codex, exports only).
- `test/kraken_e/test_S3_cf_faces_2d.jl` (Codex, new).
- `test/runtests.jl` (Codex, one new `include` gated on `"kraken_e_S3"
  in ARGS || isempty(ARGS)`).

Read-only: `src/fvfd/`, `src/multiblock/`, `src/kernels/`,
`src/drivers/`, `src/lattice/`, `src/io/`, `Project.toml`,
`Manifest.toml`, everything outside the above list.

## 8. Out of scope (deferred to later sessions)

- Conservative flux assembly across c/f faces (S4, D5).
- Reflux accumulator allocation/sizing (S4 / S8).
- LBM moment extraction at the c/f ghost (S5, D7).
- FVFD → LBM reconstruction (S5, D8).
- Stress consistency / Newtonian split at the c/f (S5, D9).
- Wall + interface + Guo composition (S7, D10).
- 3D c/f face geometry (one coarse face splits into four fine faces;
  algebra is analogous but the quadrature weights become `1/4 each`
  and the tangential interpolation uses a 2D bilinear stencil; deferred).
- GPU validation of adapter/cf face builder (CPU exactness suffices
  for S3; backend-agnostic by construction).

## 9. Failure modes the S3 test must catch

Each canary corresponds to a class of implementation defect:

1. **View vs copy bug in the adapter** — if the adapter copies LBM
   interior into a fresh buffer and forgets to write back, the FVFD
   operator runs on stale zeros and the constant-gradient test fails
   loudly with `dudx = 0` but `dudy = 0` mismatching the affine case.
   Caught by §5.1 (2).

2. **Wrong `(dx, dy)` plumbing** — if the adapter passes `dx = 1` when
   `block.dx = 0.5`, the gradient comes out off by `2×`. Caught by
   §5.1 (2): the expected coefficients `0.03, 0.07, 0.05, -0.04` would
   appear as `0.06, 0.14, 0.10, -0.08`.

3. **Wrong `is_solid` default** — if the default `is_solid` is
   `trues(Nx, Ny)`, the FVFD kernel zeros out every interior cell and
   `dudx ≡ 0`, which masks correctness on the constant test but fails
   the affine test.

4. **Wrong tangential stencil weights** — if `(3/4, 1/4)` is encoded as
   `(2/3, 1/3)` (an obvious near-miss), the affine prolongation
   recovers a value off by `βᵧ · dx_c / 12 ≠ 0`. Caught by §5.3 (3) at
   `~1e-3` level, well above `1e-12`.

5. **Mixed-up axis convention** — if the builder reverses `LO`/`HI` or
   `X`/`Y`, the normal direction is wrong (e.g. `(-1, 0)` instead of
   `(+1, 0)`). Caught by §5.2 (3).

6. **Non-deterministic corner ownership** — if two records claim the
   same corner with `corner_lo_owned = true` (e.g. axis-only rule
   ignoring side), the uniqueness assert §5.4 fails.

7. **Quadrature weights summing to ≠ 1** — if `fine_to_coarse_weights =
   (1.0, 1.0)` (forgot `dx_f / dx_c = 1/2`), the constant face-average
   doubles. Caught by §5.2 (4) and §5.3 (1).

8. **`LeafBlock2D` layout migration regression** — replacing the empty
   `CFFaceRecord` with the typed `CFFaceRecord2D{T}` must not break
   any S2 sentinel: `isempty(block.cf_face_records)` and the construct
   `allocate_leaf_block_2d(...)` must succeed. Caught by S2 regression
   (re-run of `test_S2_uniform_block.jl`).

9. **Affine test BC mismatch** — using periodic BCs on an affine field
   produces a discontinuity at the periodic seam, breaking the
   exactness invariant. The S3 canary mandates wall BCs for the affine
   gradient test specifically (the FVFD one-sided stencil at walls is
   exact on affine fields).

End of derivation.
