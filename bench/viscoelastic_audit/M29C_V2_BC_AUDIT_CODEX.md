# M29C v2 BC Audit - Codex

Read-only audit of the working-tree M29c-v2 patch in
`src/fvfd/operators_2d.jl`, with the production cylinder call chain checked
through `src/kernels/logconformation_fv_2d.jl` and
`src/drivers/viscoelastic_logfv_2d.jl`.

## Q1. `:muscl_superbee` branch and cylinder entry point

The dispatch normalizer is `_fvfd_advection_scheme_val(advection_scheme::Symbol)`,
which accepts only `:rusanov` and `:muscl_superbee` and returns `Val(scheme)` in
`src/fvfd/operators_2d.jl:470-475`. The `:muscl_superbee` implementation itself
is the method `_fvfd_upwind_scalar_advective_rhs_2d(..., ::Val{:muscl_superbee})`
in `src/fvfd/operators_2d.jl:537-629`. That method computes `ue`, `uw`, `vn`,
`vs` in `src/fvfd/operators_2d.jl:543-546`, reads directional neighbor values
through the BC helpers in `src/fvfd/operators_2d.jl:548-551`, reconstructs
`phie`, `phiw`, `phin`, `phis` in `src/fvfd/operators_2d.jl:553-623`, and forms
the advective RHS in `src/fvfd/operators_2d.jl:625-628`.

The public cylinder entry point is
`run_viscoelastic_logfv_cylinder_coupled_2d` in
`src/drivers/viscoelastic_logfv_2d.jl:856-879`; it calls
`_run_viscoelastic_logfv_step_channel_coupled_2d` in
`src/drivers/viscoelastic_logfv_2d.jl:871-879`. The inner driver validates and
normalizes the advection scheme in `src/drivers/viscoelastic_logfv_2d.jl:223-225`.
Inside each step it calls `logfv_advect_upwind_bc_aware_2d!` in
`src/drivers/viscoelastic_logfv_2d.jl:402-412`. That wrapper routes to
`fvfd_sym2_advect_upwind_2d!` in
`src/kernels/logconformation_fv_2d.jl:1153-1172`, which then invokes the three
scalar FVFD advection calls in `src/fvfd/operators_2d.jl:711-731`.

## Q2. Scalar BC helpers

### `_fvfd_bc_east_scalar_2d` (`src/fvfd/operators_2d.jl:422-432`)

```julia
@inline function _fvfd_bc_east_scalar_2d(phi, east_phi, i, j, Nx, east_bc)
    if i < Nx
        return phi[i + 1, j]
    elseif east_bc == FVFD_BC_PERIODIC
        return phi[1, j]
    elseif east_bc == FVFD_BC_OPEN
        return east_phi[j]
    else
        return phi[i, j]
    end
end
```

It does not accept `is_solid`. Therefore it cannot test `is_solid` before
returning an interior east neighbor. If the east neighbor cell is solid and
`i < Nx`, the helper returns the stored value `phi[i + 1, j]`
(`src/fvfd/operators_2d.jl:422-424`).

### `_fvfd_bc_west_scalar_2d` (`src/fvfd/operators_2d.jl:434-444`)

```julia
@inline function _fvfd_bc_west_scalar_2d(phi, west_phi, i, j, Nx, west_bc)
    if i > 1
        return phi[i - 1, j]
    elseif west_bc == FVFD_BC_PERIODIC
        return phi[Nx, j]
    elseif west_bc == FVFD_BC_OPEN
        return west_phi[j]
    else
        return phi[i, j]
    end
end
```

It does not accept `is_solid`. Therefore it cannot test `is_solid` before
returning an interior west neighbor. If the west neighbor cell is solid and
`i > 1`, the helper returns the stored value `phi[i - 1, j]`
(`src/fvfd/operators_2d.jl:434-436`).

### `_fvfd_bc_north_scalar_2d` (`src/fvfd/operators_2d.jl:446-456`)

```julia
@inline function _fvfd_bc_north_scalar_2d(phi, north_phi, i, j, Ny, north_bc)
    if j < Ny
        return phi[i, j + 1]
    elseif north_bc == FVFD_BC_PERIODIC
        return phi[i, 1]
    elseif north_bc == FVFD_BC_OPEN
        return north_phi[i]
    else
        return phi[i, j]
    end
end
```

It does not accept `is_solid`. Therefore it cannot test `is_solid` before
returning an interior north neighbor. If the north neighbor cell is solid and
`j < Ny`, the helper returns the stored value `phi[i, j + 1]`
(`src/fvfd/operators_2d.jl:446-448`).

### `_fvfd_bc_south_scalar_2d` (`src/fvfd/operators_2d.jl:458-468`)

```julia
@inline function _fvfd_bc_south_scalar_2d(phi, south_phi, i, j, Ny, south_bc)
    if j > 1
        return phi[i, j - 1]
    elseif south_bc == FVFD_BC_PERIODIC
        return phi[i, Ny]
    elseif south_bc == FVFD_BC_OPEN
        return south_phi[i]
    else
        return phi[i, j]
    end
end
```

It does not accept `is_solid`. Therefore it cannot test `is_solid` before
returning an interior south neighbor. If the south neighbor cell is solid and
`j > 1`, the helper returns the stored value `phi[i, j - 1]`
(`src/fvfd/operators_2d.jl:458-460`).

In all four helpers, the value returned for an interior solid neighbor is not
created by the helper. It is whatever is already stored in `phi` at that solid
cell.

## Q3. Solid-cell zeroing and call ordering

The explicit solid-cell zeroing for scalar FVFD advection is in
`fvfd_advect_upwind_2d_kernel!`: if `is_solid[i, j]` is true, it writes
`phi_out[i, j] = zero(eltype(phi_out))` in
`src/fvfd/operators_2d.jl:641-642`. The symmetric-tensor wrapper applies this
same scalar kernel to `psixx`, `psixy`, and `psiyy` outputs through the calls in
`src/fvfd/operators_2d.jl:719-729`. The production cylinder driver also
initializes all three live and work `Psi` arrays with zeros in
`src/drivers/viscoelastic_logfv_2d.jl:292-300`.

The local constitutive substep does not explicitly enforce solid-cell zeroing:
`logfv_step_constitutive_log_2d_kernel!` has no `is_solid` argument and updates
every in-range cell in `src/kernels/logconformation_fv_2d.jl:417-455`. The
velocity-gradient kernel does zero gradients on solid cells in
`src/fvfd/operators_2d.jl:1123-1127`; with zero `Psi` and zero gradients, the
local constitutive formula maps the equilibrium conformation back to zero
log-conformation through `src/kernels/logconformation_fv_2d.jl:164-183` and
`src/kernels/logconformation_fv_2d.jl:128-155`. That preservation is algebraic,
not an explicit `is_solid` guard in the constitutive kernel.

The call order for the production cylinder advection path is:

1. `_run_viscoelastic_logfv_step_channel_coupled_2d` lowers cell velocity to
   faces in `src/drivers/viscoelastic_logfv_2d.jl:390-400`.
2. The driver calls `logfv_advect_upwind_bc_aware_2d!` in
   `src/drivers/viscoelastic_logfv_2d.jl:402-412`.
3. `logfv_advect_upwind_bc_aware_2d!` calls `fvfd_sym2_advect_upwind_2d!` in
   `src/kernels/logconformation_fv_2d.jl:1153-1172`.
4. `fvfd_sym2_advect_upwind_2d!` calls `fvfd_advect_upwind_2d!` once per tensor
   component in `src/fvfd/operators_2d.jl:719-729`.
5. `fvfd_advect_upwind_2d!` launches `fvfd_advect_upwind_2d_kernel!` in
   `src/fvfd/operators_2d.jl:655-675`.
6. For fluid cells, the kernel calls `_fvfd_upwind_scalar_advective_rhs_2d` in
   `src/fvfd/operators_2d.jl:644-648`, and the `:muscl_superbee` flux assembly is
   `src/fvfd/operators_2d.jl:537-629`.

The same-kernel solid zeroing does not happen before the MUSCL face flux for a
fluid neighbor in any ordering that the fluid calculation can observe. The
fluid cell RHS reads the input array `phi` through
`src/fvfd/operators_2d.jl:548-551`, and the east helper returns
`phi[i + 1, j]` in `src/fvfd/operators_2d.jl:422-424`. The solid-cell branch
writes `phi_out[i, j]` in `src/fvfd/operators_2d.jl:641-642`, which is a
different array from `phi`, and the kernel launch is parallel. Therefore the
fluid neighbor sees the pre-existing solid-cell value in `phi`, not a
same-substep write to `phi_out`.

## Q4. Algebraic trace for a fluid cell west of a solid cell

Assume cell `(i, j)` is fluid, `is_solid[i + 1, j] = true`, and the other
nearby stencil cells are fluid. The current `:muscl_superbee` branch first sets
`east_value = _fvfd_bc_east_scalar_2d(...)` in
`src/fvfd/operators_2d.jl:548`; for an interior east neighbor, that helper
returns `phi[i + 1, j]` in `src/fvfd/operators_2d.jl:422-424`. Thus the solid
neighbor's stored value enters the reconstruction as `east_value`.

Important production note: for the actual cylinder path, the east face between
this fluid cell and the solid cell has zero normal velocity. The driver uses
`logfv_cell_velocity_to_faces_bc_aware_2d!` by default in
`src/drivers/viscoelastic_logfv_2d.jl:396-400`; that wrapper calls
`fvfd_cell_velocity_to_faces_2d!` in
`src/kernels/logconformation_fv_2d.jl:1106-1117`. Interior x-faces are assigned
by `_fvfd_xface_average_or_zero_2d` in `src/fvfd/operators_2d.jl:214-215`, and
that helper returns `zero(T)` if either adjacent cell is solid in
`src/fvfd/operators_2d.jl:142-146`. If `embedded_advection=true`, the embedded
path also multiplies by a face fraction and the same average-or-zero helper in
`src/fvfd/operators_2d.jl:280-285`, with the fraction itself zero if either
adjacent cell is solid in `src/fvfd/operators_2d.jl:149-155`. Therefore the
`ue > 0` and `ue < 0` cases below are algebraic traces of the branch, not the
normal production value for the solid-fluid face.

Let `S(far, up, down)` denote the current guarded canonical Superbee face value
when the canonical stencil is usable:

```text
S(far, up, down) = up + 0.5 * superbee((up - far) / (down - up)) * (down - up)
```

with the zero-denominator case setting `r = 0`, from
`src/fvfd/operators_2d.jl:477-487`. If the canonical stencil is not usable, the
guarded helper returns `upwind` through `src/fvfd/operators_2d.jl:490-507`.

### Case A: `ue > 0`

The current code takes the `ue >= 0` branch in
`src/fvfd/operators_2d.jl:553-560`:

```text
upwind = phi[i, j]
downwind = east_value = phi[i + 1, j]
canonical_usable = i > 1 && !is_solid[i - 1, j]
far_upwind = canonical_usable ? phi[i - 1, j] : upwind
phie = guarded(far_upwind, upwind, downwind, canonical_usable)
```

With `Psi_xx[i, j] = 5`, `Psi_xx[i + 1, j] = 0`, and the west neighbor fluid
with value `Psi_xx[i - 1, j] = W`, the executed canonical formula is:

```text
phie = 5 + 0.5 * superbee((5 - W) / (0 - 5)) * (0 - 5)
```

For a locally flat fluid value `W = 5`, `r = 0`, the limiter is zero, and
`phie = 5`. Thus the solid value is present as the downwind value, but for this
flat local evaluation it does not change `phie` from the cell value. If `W > 5`,
the limiter can reduce `phie` below 5; if `W <= 5`, the limiter is zero for this
5-to-0 downwind jump.

M29b's all-or-nothing boundary band would have used the Rusanov branch for this
cell. The current Rusanov formula sets `phie = phi[i, j]` when `ue >= 0` in
`src/fvfd/operators_2d.jl:526`, so with `Psi_xx[i, j] = 5`, M29b would give
`phie = 5`.

### Case B: `ue < 0`

The current code takes the `else` branch in `src/fvfd/operators_2d.jl:561-568`:

```text
upwind = east_value = phi[i + 1, j]
downwind = phi[i, j]
canonical_usable = i + 2 <= Nx && !is_solid[i + 2, j]
far_upwind = canonical_usable ? phi[i + 2, j] : upwind
phie = guarded(far_upwind, upwind, downwind, canonical_usable)
```

With `Psi_xx[i, j] = 5`, `Psi_xx[i + 1, j] = 0`, and the next east neighbor
fluid with value `Psi_xx[i + 2, j] = E`, the executed canonical formula is:

```text
phie = 0 + 0.5 * superbee((0 - E) / (5 - 0)) * (5 - 0)
```

For a locally positive fluid value `E = 5`, `r = -1`, the limiter is zero, and
`phie = 0`. If the canonical stencil were blocked, the one-sided fallback would
also return the upwind value `0` through `src/fvfd/operators_2d.jl:490-507`.

M29b's all-or-nothing boundary band would have used the Rusanov branch. The
current Rusanov formula sets `phie = east_value` when `ue < 0` in
`src/fvfd/operators_2d.jl:526`, and `east_value` is `phi[i + 1, j]` from
`src/fvfd/operators_2d.jl:521-524` and `src/fvfd/operators_2d.jl:422-424`.
With the solid value zero, M29b would give `phie = 0`.

## Q5. Reconciliation

Verdict: partial.

The part of DIFF that holds is the structural read: the scalar BC helpers do
not accept `is_solid` and do return stored interior neighbor values, including
stored solid-cell values, as shown in `src/fvfd/operators_2d.jl:422-468`.
The current `:muscl_superbee` branch then feeds those helper values into the
face reconstruction for `phie`, `phiw`, `phin`, and `phis` in
`src/fvfd/operators_2d.jl:548-623`. For the west-of-solid example, the solid
east value enters `phie` as the downwind value for `ue >= 0` in
`src/fvfd/operators_2d.jl:553-560` and as the upwind value for `ue < 0` in
`src/fvfd/operators_2d.jl:561-568`.

The part of DIFF that does not hold for the production cylinder path is the
claim that this creates a nonzero through-wall advective mass flux into a
cylinder-adjacent fluid cell. The exact source lines that break that mechanism
are `_fvfd_xface_average_or_zero_2d`, which returns `zero(T)` if either side of
an x-face is solid in `src/fvfd/operators_2d.jl:142-146`, and the interior
x-face assignment that uses it in `src/fvfd/operators_2d.jl:214-215`. The
driver reaches that face lowering through `src/drivers/viscoelastic_logfv_2d.jl:396-400`
and `src/kernels/logconformation_fv_2d.jl:1106-1117`. Since the MUSCL flux uses
`ue = ux_face[i + 1, j]` in `src/fvfd/operators_2d.jl:543` and the flux term is
`ue * phie` in `src/fvfd/operators_2d.jl:625`, the per-step through-wall
advection from the solid value on that face is

```text
dt * ue * phie / dx = 0
```

for the actual solid-fluid face. The `divu` term also receives `ue = 0` from
that face in `src/fvfd/operators_2d.jl:627-628`.

If one ignores the face-velocity lowering and analyzes a hypothetical nonzero
normal solid-face velocity, the defect scale would be
`O(dt * abs(ue) * abs(Psi) / dx)`, i.e. `O(5 * abs(ue))` for `dt = dx = 1` and
`Psi_xx = 5`. That hypothetical is not the cylinder driver path cited above.

Therefore I do not find code-reading support for DIFF's direct "spurious mass
advected from `Psi_solid = 0` into a cylinder-adjacent cell" mechanism as the
cause of a slow buildup to a `rho` NaN at `j = 1` upstream. The helper/MUSCL
solid read is real, but for a cylinder-adjacent solid face it is dynamically
inert as a through-wall flux because the normal face velocity is zero. The NaN
root cause remains unknown from this audit.
