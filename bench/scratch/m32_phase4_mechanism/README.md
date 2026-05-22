# M32 Phase 4 Mechanism Trace

Use the concrete Julia binary if the sandbox `julia` launcher reports an EPERM lockfile error.

```bash
cd /Users/guillaume/Documents/Recherche/Kraken.jl-viscoelastic
JULIA=/Users/guillaume/.julia/juliaup/julia-1.12.5+0.aarch64.apple.darwin14/Julia-1.12.app/Contents/Resources/julia/bin/julia

$JULIA --project=. --compiled-modules=no bench/scratch/m32_phase4_mechanism/canary_contract_a.jl
$JULIA --project=. --compiled-modules=no bench/scratch/m32_phase4_mechanism/dispatch_probe.jl

mkdir -p tmp/m32_phase4_mechanism .engineer_logs
KRAKEN_TRACE=1 \
KRAKEN_TRACE_FILE=$(pwd)/.engineer_logs/trace.jsonl \
KRAKEN_R_LIST="30" \
KRAKEN_WI_LIST="1.0" \
KRAKEN_RE_LIST="1.0" \
KRAKEN_BETA_LIST="0.59" \
KRAKEN_BSD_LIST="1.0" \
KRAKEN_MAX_STEPS_BASE="200" \
KRAKEN_AVG_WINDOW_FRAC="0.5" \
KRAKEN_BACKEND="cpu" \
KRAKEN_FT="float64" \
KRAKEN_OUTPUT_DIR=$(pwd)/tmp/m32_phase4_mechanism \
KRAKEN_ADVECTION_SCHEME="rusanov" \
$JULIA --project=. --compiled-modules=no bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl

cp .engineer_logs/trace.jsonl .engineer_logs/trace.snapshot.jsonl
jq -r '.kernel' .engineer_logs/trace.snapshot.jsonl | sort | uniq -c
jq -r '[.kernel,.args_hash] | @tsv' .engineer_logs/trace.snapshot.jsonl | head -60
```

If the runner env plumbing changes, run:

```bash
KRAKEN_TRACE=1 \
KRAKEN_TRACE_FILE=$(pwd)/.engineer_logs/trace.jsonl \
KRAKEN_OUTPUT_DIR=$(pwd)/tmp/m32_phase4_mechanism \
$JULIA --project=. --compiled-modules=no bench/scratch/m32_phase4_mechanism/run_failing_repro.jl
```
