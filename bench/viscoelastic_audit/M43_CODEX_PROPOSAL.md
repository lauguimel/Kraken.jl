# M43 Codex Proposal — Closing R=30 Wi=1 Cd_polymer shoulder deficit

## (A) Symptom restated (Codex's own words, ≤4 sentences)

The R=30 Wi=1 production run is numerically stable, but the polymer drag is under-resolved at the cylinder shoulder: Kraken reports Cd_polymer shoulder 8.51 versus rheoTool 11.65, a 3.14 Cd deficit. The same near-solid fallback band is only 344 cells, yet it carries the strongest polymer stresses and sits on the shoulder hot spot. The sign pattern is not uniform damping: the shoulder is low while the back pole is high, consistent with shoulder stress being smeared downstream rather than simply deleted.

## (B) Root-cause hypothesis

The current `:muscl_superbee` near-solid safety rule is directionally too coarse: a solid cell within ±2 lattice units on either Cartesian axis demotes the entire fluid cell to first-order Rusanov, so a normal-direction staircase hazard also suppresses tangential advection around the cylinder shoulder. At the top/bottom shoulders the high-gradient polymer streak is primarily transported tangentially, but the y-normal proximity to the solid forces x-advection to first order too; that loses the shoulder peak and feeds the observed smaller back-pole excess. M42 showed that restoring full canonical-fluid MUSCL on the whole cylinder band is too aggressive, so the patch should restore only the tangential face correction at one-axis near-solid cells while keeping Rusanov on normal, corner, and domain-edge faces.

Why this and not a generic boundary-condition leak: the source audit confirms solid-fluid face velocities are zero at the interface, while the measured deficit is concentrated in the Rusanov fallback band and has the shoulder-low/back-high signature of advective smearing, not a direct wall-value injection.

## (C) Surgical patch proposal

- Edit `src/fvfd/operators_2d.jl` only:
  - `_fvfd_advection_scheme_val` at `src/fvfd/operators_2d.jl:470-474` to accept a new experimental symbol, e.g. `:muscl_superbee_tangent`.
  - Existing scalar face helper `_fvfd_muscl_superbee_face_value_2d` at `src/fvfd/operators_2d.jl:483-487` remains reused.
  - Add a new `@inline _fvfd_upwind_scalar_advective_rhs_2d(..., ::Val{:muscl_superbee_tangent})` beside the current `:muscl_superbee` method at `src/fvfd/operators_2d.jl:517-565`.
  - No edit to the launch kernel shape at `src/fvfd/operators_2d.jl:567-589` or launcher at `src/fvfd/operators_2d.jl:591-620`, other than the scheme tag flowing through the existing `Val` dispatch.
- Static invariant audit: the current whole-cell demotion lives at `src/fvfd/operators_2d.jl:523-532`, where any cross-arm solid hit immediately recurses to `Val(:rusanov)`. The M42 dead-end is the two-pass overwrite in `src/fvfd/muscl_boundary.jl:141-157` and `src/fvfd/muscl_boundary.jl:190-209`; this proposal does not edit or reuse that path.
- Estimated diff size: 70-95 LOC in `src/fvfd/operators_2d.jl`.
- Pseudo-code:

```julia
if domain_edge_2
    return rhs_rusanov(...)
end

solid_x = solid at (i±1,j) or (i±2,j)
solid_y = solid at (i,j±1) or (i,j±2)

if !(solid_x || solid_y)
    return rhs_muscl_superbee(...)
elseif solid_x == solid_y
    return rhs_rusanov(...)   # corners / ambiguous staircase normal
end

phie, phiw, phin, phis = rusanov_face_values(...)

if solid_y && !solid_x
    # y is the broken/normal axis; x is tangential.
    # Replace only x-face values whose actual upwind stencil is fully fluid.
    phie = tangent_muscl_or_rusanov_xface(...)
    phiw = tangent_muscl_or_rusanov_xface(...)
elseif solid_x && !solid_y
    # x is the broken/normal axis; y is tangential.
    phin = tangent_muscl_or_rusanov_yface(...)
    phis = tangent_muscl_or_rusanov_yface(...)
end

return conservative_flux_rhs(phie, phiw, phin, phis)
```

- Why surgical: bulk `:muscl_superbee`, legacy `:rusanov`, domain-edge fallback, solid-cell zeroing, BC helper semantics, and the kernel launch topology are unchanged. Only the near-cylinder one-axis band gets a new face selection rule, and M42's two-pass overwrite remains untouched.

## (D) Dynamic test design

- Fixture path: `bench/scratch/m43_codex_adversarial/test_tangent_band_shoulder.jl`.
- Geometry / setup: reproduce M29b production with R=30, Wi=1, β=0.59, Re=1, `bsd_fraction=1`, q-wall/staircase cylinder, `L_up=L_down=15R`, and compare baseline `advection_scheme=:muscl_superbee` against candidate `:muscl_superbee_tangent`.
- Backend: Metal F32 local. Rationale: the empirical record says the case is NaN-free on Metal F32 and Aqua F64, and this mission is not allowed to launch Aqua jobs; any approved implementation should later receive an Aqua F64 confirmation.
- `max_steps`: 100_000 with diagnostics averaged over the last 20_000 steps. Walltime estimate: about 45-90 minutes on local Metal for the paired baseline/candidate fixture, plus a 10-20 minute 20k-step preflight.
- Diagnostics:
  - Cd(t), Cd_s, Cd_polymer, Cd_bsd, and total Cd over time.
  - Cd_polymer per-θ ring with the same front/shoulder/back aggregation and target shoulder rT value 11.65.
  - Fallback-band stats for `|τ_xx|`, `|τ_xy|`, `|τ_yy|`, and `|tr(τ_p)|`, split into near_solid, domain_edge_2, and bulk.
  - Stability proxy: `isfinite` on Ψ and τ, `rho_min`, `max(abs.(Ψ))`, `max(abs.(tr(τ_p)))`, and first step where any watched metric exceeds 2.5× the M41-bis band peak.
- PASS gate: candidate is NaN-free for 100k steps, Cd_polymer shoulder is within ±12% of 11.65 over the final 20k-step window, the back-pole Cd_polymer overshoot does not exceed +1.5 Cd points, and total Cd improves by at least +2.0 versus the same-script baseline without moving the Wi=0.1 R=30 sanity case outside ±1% of its current PASS.
- FAIL gate: any NaN, `rho_min <= 0`, `max(|tr(τ_p)|)` in the near-solid band above 0.004 for more than 100 consecutive steps, shoulder Cd_polymer below 10.25, or a back-pole overshoot above +2.0 Cd points.

## (E) kraken-trace plumbing

- Tool picked: Tool 4, `WatchedArray`. Static grep already resolved the dispatch location, and the M42 failure was a late first-spike/NaN ambiguity rather than a "which method ran" ambiguity, so a chronological watched-cell ledger is the useful trace.
- Decorated kernel(s): wrap the Ψ component arrays and τ diagnostic arrays flowing through `fvfd_sym2_advect_upwind_2d!` (`src/fvfd/operators_2d.jl:658-681`) and the inner `fvfd_advect_upwind_2d_kernel!` (`src/fvfd/operators_2d.jl:567-589`). The caller ledger should show writes from the advection kernel and the later source/τ materialisation path for the same cells.
- Decorated cells: the required `stats_summary.txt` contains the maxima but not their indices; recomputing the cells corresponding to those exact maxima from the M41-bis serialized probe arrays gives:
  - `(457,31)` peak near-solid `|τ_xx| = 1.584711e-3`.
  - `(457,90)` peak near-solid `|tr(τ_p)| = 1.607500e-3`.
  - `(432,36)` peak near-solid `|τ_xy| = 4.934843e-4`.
  - `(420,53)` peak near-solid `|τ_yy| = 6.988398e-4`.
- Short repro: CPU F64 only, because `WatchedArray` is CPU-first and intentionally falls through on Metal/CUDA. This is not the production validation; it is a 200-step ledger proving which writer touches the peak cells and whether the candidate writes an early spike.

```bash
KRAKEN_TRACE=1 \
KRAKEN_BACKEND=cpu \
KRAKEN_TRACE_FILE=tmp/m43_codex_adversarial/watch_cells.jsonl \
KRAKEN_TRACE_CELLS='457,31;457,90;432,36;420,53' \
julia --project=. bench/scratch/m43_codex_adversarial/test_tangent_band_shoulder.jl \
  --max-steps 200 --advection-scheme muscl_superbee_tangent --watch-cells
```

- Kernel-fired count:

```bash
jq -r 'select(.op=="write" and (.caller|test("fvfd_advect_upwind_2d_kernel!"))) | .idx' \
  tmp/m43_codex_adversarial/watch_cells.jsonl | sort | uniq -c
```

- First spike / first nonfinite:

```bash
jq -r 'select((.finite==false) or ((.abs_tr_tau // 0) > 0.004)) | [.step,.idx,.abs_tr_tau,.caller] | @tsv' \
  tmp/m43_codex_adversarial/watch_cells.jsonl | head -1
```

- Refutation gate: if Tool 4 shows that the candidate produces finite, materially different tangential Ψ writes at the four peak cells by step 200, but the 100k diagnostic still leaves shoulder Cd_polymer below 10.25 without a compensating band-stress increase, then the fallback-advection hypothesis is wrong; repivot to the alternative in §(F). If the first spike is written by the source/τ materialisation path before any changed advection write reaches the watched cells, this patch is also wrong because M42's NaN mechanism is not the tangential advection correction itself.

## (F) Risk + fallback

- Risks:
  - The tangential-only correction may still trigger the same high-Wi stiffness as M42, just later, if shoulder Ψ gradients are already at the stability margin.
  - The patch may under-correct because one-axis cells are too small a subset of the 344-cell band.
  - It may fix shoulder Cd_polymer but worsen the pressure/solvent balance or regress the Wi=0.1 R=30 PASS.
- Alternative root-cause hypothesis if §(E) refutes §(B): the shoulder stress exists in the fallback band, but the stress-to-drag projection under-integrates the staircase shoulder traction or couples it to the body force with the wrong local orientation; the next target would be polymer traction/ring integration and force projection rather than Ψ advection.
- Cost estimate for an implementation mission after approval: 2-3 hours for the `operators_2d.jl` patch and scratch fixture, 1 hour for the WatchedArray trace canary, 1-2 local Metal runs for the 100k validation, and one optional Aqua F64 confirmation job outside this mission.
