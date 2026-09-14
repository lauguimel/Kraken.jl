# M55 AUDIT — Claude variant

Adversarial dual-spawn audit of the M55 implementation diff (worktree,
uncommitted). Pure reading + Taylor algebra. No execution. Sibling
`M55_AUDIT_codex.md` not loaded (independence enforced).

## 1. Implementation reading

Codex added a new file `src/fvfd/qw_wall_gradient_2d.jl` (205 LOC) and
piped a new helper `_fvfd_apply_qw_quadratic_embedded_wall_gradient_2d`
through `fvfd_velocity_gradient_embedded_2d_kernel!`. The
`FVFDEmbeddedBoundary2D` struct gained a third type parameter `Q` and a
`wall_q::Array{T,3}` field of size `(Nx, Ny, 9)` storing the q-wall
fraction per D2Q9 lattice link.

`u2` access (`qw_wall_gradient_2d.jl:63-65`):
```julia
u1 = phi[i, j]
d2n = hxx * nx * nx + T(2) * hxy * nx * ny + hyy * ny * ny
u2 = u1 + h * (gx * nx + gy * ny) + (h * h / T(2)) * d2n
```
`u2` is a **truncated 2nd-order Taylor reconstruction** from the cell
value `phi[i,j]`, seed gradient `(gx,gy)`, and a finite-difference
Hessian `(hxx, hxy, hyy)` built from the wide stencil
(`_fvfd_solid_bc_second_derivative_x_2d`,
`_fvfd_solid_bc_mixed_derivative_xy_2d`). `n = -link/|link|` points
**outward** from the wall into the fluid. `h = min(|dx|,|dy|)`.

M55 application (`qw_wall_gradient_2d.jl:66-68`):
```julia
target_normal = (
    (distance + h) * (distance + h) * u1 - distance * distance * u2
) / (h * distance * (distance + h))
```
with `distance = qw·link_length`. Plugging `link_length = h` (axis-aligned)
recovers the M55 closed form with `u_wall = 0`. The path is invoked per
cut-link and combined via a least-squares (2×2 normal equations) plus a
best-pair candidate, blended 30/70.

`q_w_min` is hard-coded to `T(0.01)` at line 130 and checked at line 42
(`_fvfd_qw_wall_equation_2d`: `qw < q_w_min` → invalid). When **every**
cut-link is invalid, fallback at line 152 to the legacy first-order
`_fvfd_apply_embedded_wall_gradient_2d`.

## 2. Algebraic equivalence test

Take a smooth field along the outward wall normal `s`:

```
u(s) = u_w + c1·s + (c2/2)·s² + (c3/6)·s³ + (c4/24)·s⁴ + …
```
True values: `u1 = u(qw·h)`, `u2 = u((qw+1)·h)`,
`∂u/∂n|_wall = c1`.

| Quantity | True value | Code value (truncated Taylor) |
|---|---|---|
| `u1` | `u(qw·h)` | `u(qw·h)` (exact: cell value) |
| `gx·nx+gy·ny` at cell | `c1 + c2·qw·h + (c3/2)·(qw·h)² + …` | Seed FD; O(h²) accurate |
| `d2n = nᵀ·H·n` at cell | `c2 + c3·qw·h + (c4/2)·(qw·h)² + …` | FD Hessian; O(h²) accurate on cubics, contaminated on quartics |
| `u2` reconstructed | `u1 + h·∇u·n + (h²/2)·d2n` | Missing all terms `≥ s³` |
| `u2` exact | `u1 + h·u'+(h²/2)·u''+(h³/6)·u'''+…` | — |
| `δu2 ≡ u2_code − u2_true` | 0 (canary, c3=c4=0) | `−(h³/6)·c3 − (h⁴/24)·c4 + O(h²·c4)` (Hessian error) |

Substituting into the M55 formula with `d = qw·h`:

```
target − c1 = [(d+h)²·δu1 − d²·δu2] / (h·d·(d+h))
            = −qw² · δu2 / (qw·(qw+1)·h)
            = +(qw·h²)/(6·(qw+1))·c3 + O(h³·c4)
```

So **leading error in the recovered wall normal derivative is**
`qw·h²·c3 / [6·(qw+1)]`. This is **O(h²)** — second-order convergent in
mesh refinement, same order as a bilinear `u2` access would deliver.
**However** the Hessian itself contains additional O(h²·c4)
contamination, propagated linearly into the `(h²/2)·d2n` term.

| Field class | Code matches M55 exactly? |
|---|---|
| Constant | YES (trivially) |
| Linear (c2=c3=…=0) | YES (Hessian=0) |
| Quadratic (c3=c4=0) | **YES — bit-exact**. This is the canary class. |
| Cubic (c4=…=0) | NO; truncation error `qw·h²·c3/[6(qw+1)]` |
| Quartic+ | NO; truncation + Hessian-FD error |

**Verdict §2: GOALPOST-HIT for bit-exactness, EQUIVALENT in O(h²)
convergence order.** The canary mean-err = 8.32e-4 is **NOT** evidence
that the Taylor scheme reproduces M55; it is evidence that the
canary's exactly-quadratic field is in the kernel's null-error space.
A direct bilinear `u2` access would have produced the same answer to
machine precision. Both schemes are second-order on cylinder flows
where the velocity profile is smooth-but-not-quadratic.

## 3. q_w_min protection

| qw | Path taken | Behaviour |
|---|---|---|
| 0.50 | q-aware | numerator/denominator both O(1); clean |
| 0.10 | q-aware | `1/(qw·(qw+1)) ≈ 9.1`; manageable |
| 0.05 | q-aware (since 0.05 > 0.01) | `1/(qw·(qw+1)) ≈ 19`; smooth-field numerator scales as qw, finite |
| 0.01 (floor) | q-aware | `1/(qw·(qw+1)·h) ≈ 99/h`; numerator ≈ qw·h·c1 → target ≈ c1 finite |
| 0.005 (below floor) | **fallback to `_fvfd_apply_embedded_wall_gradient_2d`** (1st-order) | uses `wall_inv_distance_to_center`; **not** halfway-BB |
| 0.001 | fallback (1st-order) | same |

**Smooth-field algebra** at qw=0.01 confirms: `u1 ≈ u_w + 0.01·h·c1`,
`(qw+1)²·u1 − qw²·u2 ≈ 1.02·(u_w + 0.01·h·c1) − 0.0001·u2`. For
zero wall, target ≈ (0.0102·h·c1)/(0.0101·h) ≈ 1.01·c1. Finite and
correct in spirit — the 1/qw singularity is cancelled by qw·c1 in
the numerator for smooth fields. **Numerically OK at qw=0.01** in
Float64; in Float32 the cancellation loses ~2 digits of precision.

**Real concerns on cylinder R=30**:
1. The canary filters `0.1 ≤ qw ≤ 0.9` (line 31 of PT script). The
   q_w_min sweep `{0.01, 0.05, 0.1}` produces **identical** canary
   numbers because the floor is below the test's filter floor. The
   "smallest value retained" rationale (M55_IMPL_VERDICT §Design
   choices) is therefore **not empirically backed**.
2. On a uniform cylinder, qw ~ U[0,1] per cut-link → ~9% of cut-links
   have qw < 0.1, ~1% have qw < 0.01. For R=30 cylinder (perimeter
   ~188 cells, ~3-8 cut-links per cell → O(700-1500) cut-links),
   that means O(7-15) cells below the floor falling back to first-order.
3. Fallback is the **legacy first-order projection**, not halfway-BB
   as M55_DERIV §"Limiting behavior" recommended (`q_w < q_w_min` →
   "fallback to first-order **or halfway-BB**"). For polymer stress
   readout (sensitive to `du/dn` near front pole θ≈0°), a first-order
   fallback at a handful of cells injects an O(h) bias.

**Verdict §3: INADEQUATE q_w_min.** Floor `0.01` is below the canary's
own filter, so 0% empirical evidence on it. Recommend `0.05` with a
halfway-BB-style fallback. On R=30 the impact is small (~1% of
cut-links) but localised on front pole = exactly where M28-M34 found
the Cd_pressure gap.

## 4. New field on `FVFDEmbeddedBoundary2D`

`wall_q::Array{T,3}` of shape `(Nx, Ny, 9)` initialised in **all 6**
constructor return sites:

| Site | Line | Initialisation |
|---|---|---|
| `FVFDEmbeddedBoundary2D(wall_nx,...)` | 28 | `zeros(eltype, Nx, Ny, 9)` |
| `fvfd_empty_embedded_boundary_2d` | 49 | `zeros(FT, Nx, Ny, 9)` |
| `fvfd_embedded_boundary_from_halfplane_2d` | 218 | zeros (no per-link q) |
| `fvfd_embedded_boundary_from_circle_2d` | 384 | zeros (no per-link q) |
| `fvfd_embedded_boundary_from_qwall_2d` | 569 | populated at line 460 |
| `fvfd_transfer_embedded_boundary_2d` | 687 | `copyto!` from source |

**Concern**: only the `from_qwall_2d` constructor populates `wall_q`
with actual q values. The `from_halfplane_2d` and `from_circle_2d`
constructors leave `wall_q` as all-zeros. Downstream behaviour: when
the kernel reads `qw = wall_q[i,j,q]` (line 41), it sees `qw=0`, which
triggers the `qw < q_w_min` branch (line 42) → returns invalid → no
contribution. The cell then falls back ENTIRELY to the legacy
first-order projection via `valid_count == 0` branch (line 151).

**This is silent degradation**, not a crash. Production drivers using
`fvfd_embedded_boundary_from_circle_2d` (M28-M44 cylinder runs likely
use either this or `from_qwall_2d` — need code-path provenance check
per `feedback_code_path_provenance`) get **no benefit** from M55.
There is no log warning, no error. The kernel just silently uses the
old code on these inputs.

Type parameter `Q` added correctly; transfer path GPU-safe
(KernelAbstractions.allocate). Consumers (just the embedded-kernel
launcher in `operators_2d.jl:1166-1188`) updated consistently.

## 5. Verdict

**YELLOW — promote with caveats.**

Justification: the M55 closed-form is implemented faithfully, the new
struct field is wired through all constructors and the GPU transfer
path, and the canary GREEN is real (the implementation IS exact on the
canary's quadratic field). However:

| Caveat | Empirical signature on R-sweep |
|---|---|
| C1: Taylor `u2` is bit-exact on quadratic only; cubic/quartic content sees O(h²) extra term scaling as `qw·h²·c3/[6(qw+1)]` | Cd convergence will look 2nd-order but with a larger pre-factor than a bilinear-u2 implementation; no qualitative break |
| C2: `q_w_min=0.01` empirically untested (canary filters `qw≥0.1`) | If Float32 used on cylinder, near-floor cells lose ~2 digits — drift visible only at R≥60 fine mesh |
| C3: fallback below floor is 1st-order embedded, not halfway-BB as DERIV §Limiting recommended | Small bias at front pole (where Cd_pressure gap lives per M32) |
| C4: `from_halfplane_2d` and `from_circle_2d` constructors leave `wall_q=0`, silently degrading to legacy path | If R-sweep driver uses these (likely the cylinder driver does), **M55 may not actually fire**; need code-path provenance gate before A.3 (per `feedback_code_path_provenance`) |

**Smallest patch to close to GREEN before A.3 Metal sweep**:
1. **Verify code path**: instrument `_fvfd_apply_qw_quadratic_embedded_wall_gradient_2d` with a gated counter; confirm cylinder driver actually exercises it (not silently falling back via zero `wall_q`). If the driver builds embedded via `from_circle_2d`, either switch to `from_qwall_2d` or populate `wall_q` in `from_circle_2d` too.
2. **Raise `q_w_min` to 0.05** and extend the canary to include `qw ∈ [0.01, 0.10]` cut-links (drop the filter), measuring err vs qw.
3. **Switch fallback** to halfway-BB-equivalent at the cut-link (M51 formula at `qw=0.5`) instead of the 1st-order projection.
4. **Add a cubic-field PT** (`u = c3·s³/6`, c3 ≠ 0) to distinguish M55 from the Taylor scheme — would falsify or confirm goalpost-hit empirically.

Caveats are not blocking — convergence order is correct — but
C4 in particular could mean **A.3 measures legacy behaviour rebadged
as M55**. Recommend a 30-min code-path provenance check before
launching the Aqua R-sweep.
