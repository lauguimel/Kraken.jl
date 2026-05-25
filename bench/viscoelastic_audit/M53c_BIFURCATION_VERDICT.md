# M53c - proper embedded-distance bifurcation verdict

Date: 2026-05-25

## Write-first plan

### Field contract

- `wall_distance`: centroid distance from embedded wall plane to the cut-cell fluid centroid. Kept for existing downstream cut-cell volume integration consumers.
- `wall_inv_distance`: inverse centroid distance, matching existing `wall_distance` semantics.
- `wall_inv_distance_to_center`: inverse plane distance from wall plane to the cell center, for `_fvfd_apply_embedded_wall_gradient_2d`.

### Four storage sites

1. Verify existing early lowering site keeps `wall_distance = max(centroid_distance, eps(FT))` and add/populate `wall_inv_distance_to_center = inv(max(distance, eps(FT)))`.
2. Restore the direct embedded averaging site from plane-distance overwrite back to centroid-distance storage, then populate `wall_inv_distance_to_center` with `distance`.
3. Restore the fallback/best-normal site from plane-distance overwrite back to centroid-distance storage, then populate `wall_inv_distance_to_center` with `best_distance`.
4. Restore the second fallback/best-normal site from plane-distance overwrite back to centroid-distance storage, then populate `wall_inv_distance_to_center` with `best_distance`.

### Test revert plan

- Revert the Boss rebaseline in `test/test_fvfd_operators_2d.jl` from `0.25` back to `0.375`.
- Revert the Boss rebaseline in `test/test_viscoelastic_logfv_patch_ladder.jl` from `0.25` back to `0.375`.

## Results

### Source changes

- Added `FVFDEmbeddedBoundary2D.wall_inv_distance_to_center`.
- Kept `wall_distance` and `wall_inv_distance` as centroid-distance fields.
- Populated `wall_inv_distance_to_center` with plane-distance inverse at the embedded-wall lowering sites.
- Updated `_fvfd_apply_embedded_wall_gradient_2d` and its kernel caller to consume `wall_inv_distance_to_center`.
- Reverted the explicit `0.25` wall-distance baselines to centroid `0.375`; the gradient fixtures now use the new plane-distance field.

Diffstat for edited tracked files:

```text
src/fvfd/lowering_2d.jl                      | 49 ++++++++++++++++++----------
src/fvfd/operators_2d.jl                     | 12 +++----
test/test_fvfd_operators_2d.jl               |  6 ++--
test/test_viscoelastic_logfv_patch_ladder.jl |  6 ++--
4 files changed, 45 insertions(+), 28 deletions(-)
```

### Validation

Commands were run with the direct Julia 1.12.5 binary and a writable depot prepended because the `julia` launcher and default compiled cache attempted to write outside the sandbox.

| Step | Log | Exit | Result |
|---|---|---:|---|
| M49 halfway wall stencil | `scratch/M53c_bifurcation/M49_halfway.log` | 0 | PASS; helper quadratic max abs error `4.9737991503207013e-14` |
| M53a cylinder-adjacent stencil | `scratch/M53c_bifurcation/M53a_cylinder_adjacent.log` | 1 | RED by script; embedded mean abs_err `0.023139961195471057`, default `0.071109500066542444`, improvement factor `3.0730172564187344` |
| FVFD operators 2D | `scratch/M53c_bifurcation/test_fvfd_operators_2d.log` | 0 | PASS `952/952` |
| Log-FV patch ladder | `scratch/M53c_bifurcation/test_viscoelastic_logfv_patch_ladder.log` | 1 | FAIL `18201/18213`, 12 failures |

Log-FV patch ladder failures:

- M2c embedded cut-link velocity gradient: 2 diagonal fixture failures at lines 639-640.
- M5d coupled Poiseuille source-force loop: 1 failure, `max_uy = 0.006706453084356583`.
- M5e frozen-channel CDE: 6 failures, including Couette `max_c_error = 0.6702410166183791` and Poiseuille `max_c_error = 0.003811814285554438`.
- M7d square channel near-Newtonian: 1 failure, max rho delta `0.0008146915798177279`.
- M8h BFS near-Newtonian: 2 failures, max ux delta `0.00023373470870723275`, max rho delta `0.0011236871631143952`.

Total validation wall time from `/usr/bin/time -p real`: `49.23 s`.

### Verdict

RED.

The bifurcation is implemented and the M53a embedded helper metric meets the target (`0.02314 <= 0.030`), but the script exits RED and the full Log-FV patch ladder remains RED. Per the M53c brief, no further fix was attempted after the failing validation.

### Anti-pattern flags

- Existing worktree was dirty before this mission, including files outside the allowed edit set.
- `julia` launcher/default depot were not sandbox-writable; validation required direct Julia binary plus writable depot prefix.
- M2c diagonal test still exposes a centroid-vs-plane fixture mismatch after the helper switch.
- Downstream M5d/M5e/M7d/M8h failures persist despite restored centroid storage semantics.

### Next mission recommendation

Run a read-only trace of the M5d/M5e consumers to identify whether they are reading `wall_inv_distance`, `wall_distance`, or the updated embedded velocity-gradient helper path; do not change tolerances.
