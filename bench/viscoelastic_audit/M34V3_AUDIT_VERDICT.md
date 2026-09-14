**Verdict**: FIRES

# M34v3 audit — pass-3 cut-link ρ recompute

## Mission

Empirically verify whether the `ApplyCutLinkRhoRecompute` brick added in
commit `e98b9687` (M34v3) actually fires at runtime AND writes a new value
into `ρ_out` at cut-link cells, OR is silently a no-op. The bit-exact Cd
identity (132.5100514137659 to 16 decimals on R=30 Wi=0.1) between
M34-fix and M34v3 could be explained by (H_A) silent no-op or (H_B/C) the
brick fires correctly but the ρ_w consistency hypothesis is sub-threshold.

## Method

1. **Static check**: read `src/kernels/dsl/bricks.jl` lines 708–727. The
   brick assigns `ρ_out[i, j] = f_out[i,j,1] + ... + f_out[i,j,9]` under
   guard `!is_solid[i,j] && any q_wall[i,j,2..9] > 0`. The kernel
   signature at call site
   `src/kernels/li_bb_2d_v2.jl:248 pass3!(f_out, ρ, is_solid, q_wall, Nx, Ny)`
   binds `ρ_out := ρ` (the master density array). So the assignment
   mutates the live driver state. Not a phantom.
2. **Dispatch trace**: added one line
   `@trace_enter :pass3_cutlink_rho` in
   `_fused_trt_libb_v2_guo_field_step!(::Val{:bouzidi_fl_twopass}, …)`
   immediately before the pass-3 launch (gated by `KRAKEN_TRACE`,
   default OFF). Re-ran the canonical R=30 Wi=0.1 repro with
   `wall_bc=:bouzidi_fl_twopass`, CPU Float64, intent 100 steps. The
   log-FV polymer pipe NaNed at step ~30 (unrelated, a `log(x<0)` in
   `logfv_log_spd_sym2_2d`), but 29 LBM steps did complete.
3. **Effect probe**: a 16×16 synthetic harness builds a state with
   exactly one cut-link cell (q_wall[8,8,2]=0.5, sum_pops=1.234567)
   and one no-cut-link control (ρ=1.5, sum_pops=0.99), then launches
   ONLY the pass-3 kernel (built via `build_lbm_kernel` from the public
   `_TRT_LIBB_V2_GUO_FIELD_BOUZIDI_FL_TWOPASS_PASS3_SPEC`).

## Results

### Trace token counts (`.engineer_logs/trace.jsonl`, 29 LBM steps)

```
  29 pass3_cutlink_rho            (NEW, added for this audit)
  29 lbm_step_bouzidiFL_twopass
  29 lbm_step
  29 poly_force
  30 vel_grad
  30 psi_advect
  30 psi_sym2_advect
  90 psi_advect_inner
   1 driver_step_entry
```

Pass-3 fires exactly 1× per LBM step (29 == 29 matches
`lbm_step_bouzidiFL_twopass`). No skipping.

### Effect probe (`tmp/m34v3_audit_effect.log`)

```
Flag cell (8, 8):
  ρ_before  = 1.0
  ρ_after   = 1.234567
  expected  = 1.234567
  Δ         = 0.23456699999999997
Control cell (4, 4):
  ρ_before  = 1.5
  ρ_after   = 1.5
  Δ         = 0.0
VERDICT: FIRES_AND_WRITES
```

- Cut-link cell: ρ_after equals `sum(f_out[i,j,1..9])` to machine
  precision. The brick writes.
- Control cell (no cut link, ρ ≠ sum_pops): ρ unchanged. The brick
  respects its gate.

## Verdict

**FIRES**. Pass-3 is dispatched, the brick body executes, and it
overwrites `ρ_out` with the post-pass-2 pop sum at cut-link cells. The
bit-exact M34v3 vs M34-fix Cd identity (132.5100514137659) is therefore
NOT a silent-no-op bug. The remaining hypotheses are:

- **H_B**: ρ_w-via-summed-pops differs from the pass-1 ρ_out by an
  amount too small to move Cd at the chosen precision and step count;
  the rho_w hypothesis is real but sub-threshold at R=30 Wi=0.1.
- **H_C**: ρ_w is consistent with pass-1 ρ_out at cut-link cells to
  machine precision in the converged state (i.e. the M34_FIX_DIAG
  "rho_w inconsistency" hypothesis was the wrong physics, and the
  +1.6% Cd residual at R=30 Wi=0.1 has a different cause).

Discriminator H_B vs H_C requires a second probe: in a converged
cylinder run, snapshot `ρ_out` immediately before and after pass-3,
compute `max(|Δρ|)` over cut-link cells. That probe is NOT part of
this mission's success criterion; flag for follow-up if H_B/C
distinction matters for the next iteration.

## Artifacts

- Code change: `src/kernels/li_bb_2d_v2.jl` +1 LOC at line 247
  (`@trace_enter :pass3_cutlink_rho`), gated by env var.
- Repro: `bench/scratch/m34v3_audit/run_repro.jl`.
- Effect probe: `bench/scratch/m34v3_audit/effect_probe.jl`.
- Run log: `tmp/m34v3_audit_run.log`.
- Effect log: `tmp/m34v3_audit_effect.log`.
- Trace: `.engineer_logs/trace.jsonl` (297 entries, 29 pass3 hits).

## Time

~25 min wall, within the 30 min ceiling.
