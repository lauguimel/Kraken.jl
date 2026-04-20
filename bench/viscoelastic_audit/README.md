# Viscoelastic audit — staircase convergence

Purpose : one isolated failure mode per step. Each script below adds
exactly ONE complication to the pipeline and measures its convergence
order against the analytic Poiseuille Oldroyd-B solution (Liu Eq 62,
see `../../REFERENCES.md`).

The goal is to find where order drops below ~2 and pin the responsible
component.

| Step | Collision | Source | Wall BC       | Geometry       | What it isolates |
|------|-----------|--------|---------------|----------------|------------------|
| 1    | BGK       | Guo+Hermite fused | HWBB periodic | flat channel | minimal baseline |
| 2    | TRT       | Hermite separate  | HWBB periodic | flat channel | TRT vs BGK |
| 3    | TRT       | Hermite separate  | LI-BB q_w=0.5 periodic | flat channel | LI-BB brick vs HWBB |
| 4    | TRT       | Hermite separate  | LI-BB + CNEBB + ZouHe in/out | confined cylinder | curved wall + inlet/outlet |
| 5    | TRT (3D)  | Hermite separate  | LI-BB + CNEBB + ZouHe | ducted sphere  | 3D port |

Pass criterion at each step : order p ≥ 1.8 on u_err, Cxy_err, N1_err
(measured by successive halving). Any step dropping below that pins the
culprit.

All steps share the analytic solution helpers in `common.jl` so the
reference values are defined once.

## Usage

```julia
julia --project=. bench/viscoelastic_audit/step1_bgk_guo.jl
julia --project=. bench/viscoelastic_audit/step2_trt_hermite.jl
# ...
```

Each script outputs its convergence table and writes a `.txt` file of
the same name in `results/`. The `MEMORY.md` at top level is updated
with the verdict of each step after completion.
