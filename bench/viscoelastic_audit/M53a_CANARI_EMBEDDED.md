# M53a - CANARI embedded-gradient cylinder-adjacent canary

Date: 2026-05-25
Script: `bench/viscoelastic_validation/patch_tests/PT_cylinder_adjacent_stencil.jl`
CSV: `scratch/M52b_canary/cyl_adj_canary.csv`
Backend: CPU Float64 raw arrays

## API check

- Embedded lowering found: `fvfd_embedded_boundary_from_qwall_2d(q_wall; FT::Type{<:AbstractFloat}=eltype(q_wall), include_axis_aligned::Bool=false, include_halfway::Bool=false)`.
- Embedded gradient found: `fvfd_velocity_gradient_embedded_2d!(dudx, dudy, dvdx, dvdy, ux, uy, is_solid, dx, dy, bc::FVFDDomainBC2D, embedded::FVFDEmbeddedBoundary2D; sync::Bool=true)`.
- There is no `embedded=` keyword on the raw `fvfd_velocity_gradient_2d!` signature inspected in `src/fvfd/operators_2d.jl`.

## Side-by-side stats

| path | n | mean abs_err | median abs_err | max abs_err | corr(q_w, abs_err) |
|---|---:|---:|---:|---:|---:|
| `:default` | 76 | 0.07110950006654243 | 0.0625 | 0.14125809929537925 | 0.77578440795516368 |
| `:embedded` | 76 | 0.052087719394423082 | 0.035474889577818747 | 0.10346596059635205 | -0.37408969583965512 |

## Per-q_w-bin error table

| path | q_w bin | n | mean abs_err | max abs_err |
|---|---|---:|---:|---:|
| `:default` | 0.1-0.3 | 8 | 0.037602909015314601 | 0.037602909015314601 |
| `:default` | 0.3-0.5 | 8 | 0.036835828037412366 | 0.036835828037412366 |
| `:default` | 0.5-0.7 | 44 | 0.066923537847296932 | 0.1187184335382292 |
| `:default` | 0.7-0.9 | 16 | 0.11651102770964653 | 0.14125809929537925 |
| `:embedded` | 0.1-0.3 | 8 | 0.10133305147265416 | 0.1013330514726542 |
| `:embedded` | 0.3-0.5 | 8 | 0.10346596059635205 | 0.10346596059635205 |
| `:embedded` | 0.5-0.7 | 44 | 0.033441887809546239 | 0.10155931772270049 |
| `:embedded` | 0.7-0.9 | 16 | 0.053051969612754213 | 0.070629049647689679 |

## Verdict

**RED**. Embedded improves the mean by only `1.3651874356041784x`, far below
the YELLOW criterion of `>=10x`, and it misses both GREEN thresholds:
mean `0.052087719394423082 > 1e-2`, max
`0.10346596059635205 > 5e-2`.

The q_w correlation flips sign because the embedded correction reduces the
large-q_w bins but makes the small-q_w bins worse. Toggling the existing
embedded path therefore does not flatten the C1 cylinder-adjacent normal
gradient error.

Measured script wall time reported by the canary: `1.255365 s` on the second
local run. The direct Julia process wall time was `2.70 s`; invoking the
`julia` launcher itself failed in this sandbox due to a juliaup lockfile
permission error, so the installed Julia binary was called directly.

## Recommendation to Boss

Do not promote `embedded_gradient=true` as the cylinder-driver default on this
evidence. The existing embedded path itself is insufficient for this canary;
next step is a focused q_w-aware cut-link/cylinder-adjacent gradient helper or
an audit of `_fvfd_apply_embedded_wall_gradient_2d` and the lowered
`wall_inv_distance` convention.
