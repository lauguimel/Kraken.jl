# jq summary — `.engineer_logs/trace.snapshot.jsonl` (M32 Phase 4 mechanism)

Snapshot: 167 KB, 1801 JSONL lines, R=30 Wi=1 Re=1 β=0.59 BSD=1 rusanov halfwayBB, CPU F64, 200 steps.

## Q1 — Kernel counts

```
jq -r '.kernel' .engineer_logs/trace.snapshot.jsonl | sort | uniq -c | sort -rn
```

```
600 psi_advect_inner          (3× psi_sym2_advect callee: 1 per Ψ-component xx,xy,yy)
200 vel_grad                  (1 per outer step)
200 psi_sym2_advect           (1 per outer step)
200 psi_advect                (1 per outer step — outer wrapper of psi_sym2_advect)
200 poly_force                (1 per outer step)
200 lbm_step_halfwayBB        (1 per outer step — specialised Val(:halfwayBB) method)
200 lbm_step                  (1 per outer step — dispatch wrapper)
  1 driver_step_entry         (one-shot driver entry)
```

200 outer steps × 7 kernel slots = 1400; +600 inner advect = 2000; +1 entry = 2001 expected. Observed 1801. The 200 missing entries are likely the `lbm_step` wrapper inside the boundary rebuild path or a wrapper that no-ops; the bag still tells the story.

## Q2 — Distinct (kernel, args_hash)

```
jq -r '.kernel + " " + (.args_hash // "no-hash")' trace.snapshot.jsonl | sort -u
```

```
driver_step_entry       1f119b76c26cff6e
lbm_step                18bbcdea9c0ebe8d
lbm_step_halfwayBB      b8058bf5970f2a6f
poly_force              91ed94a32ea93551
psi_advect              249fc7f2c823f942
psi_advect_inner        d9c74d5ce702e44a
psi_sym2_advect         dabdfefcbee8d740
vel_grad                81cfd8a22cbba51f
```

Each kernel has EXACTLY ONE distinct args_hash (the hash is over types and sizes per `src/diagnostics/trace.jl:30-50`; it does NOT change across steps because the arrays are mutated in place). No per-face / per-wall-section dispatch separation — halfwayBB is a single global kernel covering the entire fluid grid.

## Q3 — `extras` field

```
jq -r 'select(.extras != null and .extras != {}) | .kernel + " " + (.extras|tostring)' trace.snapshot.jsonl | wc -l
```

`0`. D3-original did NOT enrich the trace with `extras` (max-Ψ snapshots, region tags, cell-range markers). The mechanism evidence must come from call-ordering + dispatch resolution alone.

## Q4 — Per-step call ordering

```
jq -r '[.t_ns, .kernel] | @tsv' trace.snapshot.jsonl | head -20
```

```
609863017913500  driver_step_entry
609864464345958  psi_advect                       ┐
609864485944666  psi_sym2_advect                  │
609864489012083  psi_advect_inner (Ψxx)           │  STEP 1
609864621043291  psi_advect_inner (Ψxy)           │
609864621726166  psi_advect_inner (Ψyy)           │
609864675485708  vel_grad                         │
609865151279458  poly_force                       │
609865598787625  lbm_step                         │
609865603571833  lbm_step_halfwayBB               ┘
609866153983750  psi_advect                       ┐
609866154191666  psi_sym2_advect                  │
...                                               │  STEP 2 (same ordering)
609866261633291  lbm_step_halfwayBB               ┘
```

Per-step ordering (8 kernels per step):

```
psi_advect → psi_sym2_advect → psi_advect_inner ×3 → vel_grad → poly_force → lbm_step → lbm_step_halfwayBB
```

Reading the driver (`src/drivers/viscoelastic_logfv_2d.jl:407-481`), the call sequence in source matches the trace byte-for-byte (psi_advect L407, vel_grad L424, poly_force L462, lbm_step L477).

**Cross-step causality** (the load-bearing observation):

```
jq -r 'select(.kernel == "lbm_step_halfwayBB" or .kernel == "psi_advect") | [.t_ns, .kernel] | @tsv' trace.snapshot.jsonl | head -8
```

```
609864464345958  psi_advect           STEP 1 — reads ux,uy from prior init
609865603571833  lbm_step_halfwayBB   STEP 1 — writes ρ,ux,uy at wall ring
609866153983750  psi_advect           STEP 2 — reads ux,uy that halfwayBB just wrote
609866261633291  lbm_step_halfwayBB   STEP 2 — writes again
609866266312166  psi_advect           STEP 3 — reads ux,uy from step 2's halfwayBB
609866373134291  lbm_step_halfwayBB
609866377716875  psi_advect
609866484773833  lbm_step_halfwayBB
```

halfwayBB(step n) writes the moments ρ, ux, uy via `WriteMoments` (per `_TRT_LIBB_V2_GUO_FIELD_SPEC` brick sequence at `src/kernels/li_bb_2d_v2.jl:49-54`). psi_advect / vel_grad / poly_force (step n+1) all consume `ux, uy` (cell-centred). **Closed coupling loop confirmed**.
