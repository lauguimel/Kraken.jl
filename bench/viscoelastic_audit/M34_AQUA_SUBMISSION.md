# M34 — Aqua F64 CUDA submission verdict

**Verdict**: GREEN. Both PBS submitted, both job IDs captured, both in queued state.

## PBS files

| Mission        | PBS path                                                                     | Walltime |
| -------------- | ---------------------------------------------------------------------------- | -------- |
| Matrix (4 cases) | `bench/viscoelastic_logfv/run_cyl_m34_bouzidi_fl_matrix_a100.pbs`            | 4 h      |
| R=60 NaN test    | `bench/viscoelastic_logfv/run_cyl_m34_bouzidi_fl_R60_Wi01_a100.pbs`          | 2 h      |

## Job IDs

| PBS         | Job ID            | Jobname              | Queue      | State |
| ----------- | ----------------- | -------------------- | ---------- | ----- |
| matrix      | `21654012.aqua`   | `M34_bouzidi_fl_*`   | gpu_batch  | Q     |
| R=60 Wi=0.1 | `21654013.aqua`   | `M34_bouzidi_fl_*`   | gpu_batch  | Q     |

**Submitted at**: 2026-05-22T16:15:29+1000 (Aqua local / Brisbane)
**Walltime cap**: 4 h (matrix) + 2 h (R60). Expected completion window 2026-05-22T18:15..20:15 local + queue wait.

## Runner plumbing change (ONE additive edit)

`bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl`:

1. Added `WALL_BC = Symbol(lowercase(strip(get(ENV, "KRAKEN_WALL_BC", "halfwayBB"))))` constant
   (validated against `:halfwayBB`, `:bouzidi_fl`, `:bouzidi_fl_twopass`; case-normalized).
2. Added `wall_bc=WALL_BC_NORMALIZED` kwarg to the
   `Kraken.run_viscoelastic_logfv_cylinder_coupled_2d(...)` call (line 554+).

Default value `halfwayBB` preserves bit-exact M32 behavior for runs that don't
set `KRAKEN_WALL_BC` — backward compatible. The change is uncommitted (per
mission directive). It's a 7-line additive patch — ready for caller to commit
when convenient.

## Check command for next session

```bash
# Job state + tail of stdout
ssh aqua 'qstat 21654012.aqua 21654013.aqua 2>&1; \
          ls -la ~/Kraken.jl-viscoelastic-run/M34_bouzidi_fl_*.o* 2>/dev/null; \
          tail -40 ~/Kraken.jl-viscoelastic-run/M34_bouzidi_fl_matrix.o21654012 2>/dev/null; \
          tail -40 ~/Kraken.jl-viscoelastic-run/M34_bouzidi_fl_R60_Wi01.o21654013 2>/dev/null'

# Pull results when done
rsync -az aqua:Kraken.jl-viscoelastic-run/tmp/m34_bouzidi_fl_matrix/ \
  ~/Documents/Recherche/Kraken.jl-viscoelastic/tmp/m34_bouzidi_fl_matrix/
rsync -az aqua:Kraken.jl-viscoelastic-run/tmp/m34_bouzidi_fl_R60_Wi01/ \
  ~/Documents/Recherche/Kraken.jl-viscoelastic/tmp/m34_bouzidi_fl_R60_Wi01/
```

## Acceptance interpretation (mandate §M34 G4 BC gate)

### Matrix (21654012)

| Case            | Pass if                                            |
| --------------- | -------------------------------------------------- |
| R=30 Wi=1.0     | `Cd_kraken ∈ [118, 122]` (closes the −7.3 % gap)   |
| R=30 Wi=0.1     | `Cd_kraken` within 1 % of rheoTool 130.43          |
| R=40 Wi=1.0     | reproduces R=30 Wi=1.0 within ±0.5 % (R-invariant) |
| R=40 Wi=0.1     | reproduces R=30 Wi=0.1 within ±0.5 % (R-invariant) |

### R=60 Wi=0.1 (21654013)

| Case        | Pass if                                                                   |
| ----------- | ------------------------------------------------------------------------- |
| R=60 Wi=0.1 | runs cleanly to 100k steps without NaN (vs `:halfwayBB` NaN at step ~36k) |

If all five pass → M34 G4 GREEN → empirical closure of M28-M32 cluster.

## Outputs to inspect

Per-case CSV: `tmp/m34_bouzidi_fl_*/<jobid>/cyl_bigsweep_v2_*.csv`
Summary:      `tmp/m34_bouzidi_fl_*/<jobid>/SUMMARY.csv`
Fields .jls:  `tmp/m34_bouzidi_fl_*/<jobid>/cyl_bigsweep_v2_*_fields.jls` (Cd_s, Cd_p, Cd_bsd, ux, uy, tau*, rho)

## Non-blocking submission pattern

Submission completed; the Department now exits without waiting. The Boss / next
session polls via the `qstat` + `tail` command above. No `Monitor`, no
sleep-loop, no run-engineer artifact gate (mission is submission-only).
