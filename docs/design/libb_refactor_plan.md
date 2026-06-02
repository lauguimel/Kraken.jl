# LI-BB refactor plan — resolve double-BC bug

*Status: 2026-04-15. Written after discovery that TRT+LI-BB fused kernel
applies bounce-back twice. Next session starting point.*

## Problem

`fused_trt_libb_step_kernel!` in [`src/kernels/li_bb_2d.jl`](../../src/kernels/li_bb_2d.jl)
produces wrong velocity fields on elementary test cases:

- **Planar Couette** (top wall at U=0.01, bottom stationary, TRT Λ=3/16,
  halfway BB at q_w=0.5 everywhere): u_x goes NEGATIVE in the bottom
  half of the channel. Expected linear profile u_x(y) = U·y/H.
- **Taylor-Couette annular** (concentric cylinders): L2 = 50 % on
  u_θ(r), independent of radial resolution.

Both cases share the pattern "fluid cell adjacent to a solid cell".

## Root cause

The kernel inherits Kraken's `fused_bgk_step!` pattern:

```julia
if is_solid[i, j]
    # Swap opposite populations (halfway bounce-back on stored f)
    f_out[i, j, 2] = fp4; f_out[i, j, 4] = fp2
    ...
end
```

When the next step's fluid cell pulls from this solid neighbour, it
reads populations that have already been bounced. Then the fluid cell
collides on them and the LI-BB overwrite applies Bouzidi's formula
again. Net result: the "bounce-back" momentum transfer is applied
twice, corrupting the wall-normal direction of the flow.

## Verification

On a planar Couette with q_wall[i, j, q] = 0.5 uniformly on fluid
cells adjacent to the walls, Bouzidi reduces to halfway BB + moving
wall correction. The expected behaviour is IDENTICAL to running
`fused_bgk_step!` with solid walls at top/bottom plus a moving-wall
correction kernel. In practice it gives a wrong profile: smoking gun.

## Refactor path A (recommended): solid-cells-inert kernel

Write a new fused kernel where solid cells do **not** swap populations.
Leave them at zero or equilibrium — they are INERT storage. All wall
physics goes through the LI-BB overwrite on fluid-cell populations.

### Implementation sketch

```julia
@kernel function fused_trt_libb_v2_step_kernel!(f_out, @Const(f_in), ...,
                                                 @Const(is_solid),
                                                 @Const(q_wall),
                                                 @Const(uw_link_x, uw_link_y),
                                                 Nx, Ny, s_plus, s_minus)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if is_solid[i, j]
            # INERT: write zeros (or feq(1, 0, 0)) — no swap
            for q in 1:9
                f_out[i, j, q] = zero(eltype(f_out))
            end
            ρ_out[i, j] = one(eltype(f_out))
            ux_out[i, j] = zero(eltype(f_out))
            uy_out[i, j] = zero(eltype(f_out))
        else
            # Pull (unchanged)
            fp1 = f_in[i, j, 1]
            fp2 = ifelse(i > 1,  f_in[i-1, j, 2], f_in[i, j, 4])
            # ... etc. Note: these values may be garbage when pulled
            # from solid neighbours, but they will be OVERWRITTEN by
            # LI-BB below for the affected populations.

            # TRT collision on the pulled values. For cut links the
            # pulled values are junk → post-collision values are also
            # junk → we will overwrite them. For other links, unchanged.
            ρ, ux, uy = moments_2d(fp1, ..., fp9)
            # ... fp_qc computed as before ...

            # LI-BB overwrite on cut links (unchanged from current kernel).
            # For q_w ≤ 1/2, formula uses fp_q (pull value from opposite
            # neighbour) — this MUST be from a fluid neighbour. Add a
            # guard or trust that the opposite neighbour of a wall-cut
            # link is always fluid in physically sensible geometries.
            # ...

            # Macroscopic moments computed AFTER LI-BB overwrite so
            # ρ_out/ux_out/uy_out reflect the corrected populations.
            # (Re-compute moments from the final fp_X_new if needed.)
            f_out[i, j, ...] = fp_X_new for each direction
        end
    end
end
```

### Pitfalls to watch

1. **Moments reported to user** (`ρ_out`, `ux_out`, `uy_out`) should
   come from the CORRECTED populations, not the pulled junk. Either
   (a) compute them twice (before and after LI-BB overwrite, use the
   second), or (b) accept that ρ/u reported on wall-adjacent cells are
   slightly off. Option (a) is cleaner.

2. **Second-fluid-neighbour lookup for q_w < 1/2** already reads the
   population OPPOSITE the wall, which is always fluid by construction
   (because the cut link is only flagged if the near-neighbour is
   solid; the opposite neighbour is by definition in the fluid
   direction). So fp_q used in Bouzidi's linear branch is valid.

3. **Cell with cut links in MULTIPLE directions** (e.g., cell inside a
   narrow gap, solid on two sides): rare in Schäfer-Turek but common
   in dense STL. Ensure each cut direction is overwritten independently
   — no coupling between (3, 5) and (6, 8) pairs in the formula.

4. **Regression tests**: `test_fused_trt_2d.jl` currently uses the
   CURRENT kernel. Update tests to use the new v2 kernel. The
   "reproduces BGK when s⁺=s⁻" test must still pass.

### Estimated effort

- Day 1: new kernel, refactor helpers if needed, unit tests identical
  to current.
- Day 2: planar Couette regression → expect L2 < 0.1 %. If this
  works, the fix is confirmed. Then planar Poiseuille (needs body
  force — small add), Couette at 30° (non-halfway q_w), Taylor-
  Couette (expect L2 < 2 %).
- Day 3: Schäfer-Turek 2D-1 (Cd within 1 %).

## Refactor path B (alternative): two-pass kernel

Split into two kernels per timestep:
1. `stream_collide!` — no BC, treats domain edges with clamp or
   periodic, solid cells inert.
2. `apply_li_bb!` — overwrites populations on flagged cells.

Simpler conceptually (mirrors Palabos/waLBerla/OpenLB) but loses the
single-kernel differentiator. +1 GPU dispatch per step. ~1 day.

## Recommendation

Path A. Keep single-kernel story intact. The "solid-cells-inert" change
is local and well-scoped. Path B is the safety net if Path A turns up
unexpected complications.

## References

- Bouzidi, Firdaouss, Lallemand 2001, *Phys. Fluids* 13, 3452
- Krüger et al. 2017, *The Lattice Boltzmann Method* §5.3.4
- Ginzburg, Verhaeghe, d'Humières 2008, *Commun. Comput. Phys.* 3
- Ginzburg, Silva, Marson et al. 2023, *J. Comput. Phys.* 473
