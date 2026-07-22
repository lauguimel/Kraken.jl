# Cavity GPU ablation — NVIDIA RTX A6000 48 GB (UGA), cross-arch data point

Run completed 2026-06-11 12:25 CEST (harvested 2026-07-22 from `~/kraken_ablation/ablation_a6000.log`,
raw log: [`ablation_a6000_raw.log`](ablation_a6000_raw.log)). Same C0→C4 ladder and driver as the
Aqua A100/H100 runs (`cavity_gpu_ablation_aqua_{a100,h100}.md`); lid-driven cavity, F64 base,
Ghia reference at 512².

Hardware context: RTX A6000 = GA102 (consumer/pro Ampere), FP64 ≈ 1/64 of FP32 (~0.6 TFLOPS F64)
vs A100 ~9.7 TFLOPS. Absolute F64 timings are therefore NOT comparable to the datacenter cards;
the relative ladder and the mixed-precision rung are the point of this run.

## Results (512², converged, parity vs CPU reference)

| config | delta | wall_s | iters | speedup_vs_cpu | parity_maxrel | ghia_err_pct | gpu_util_mean/peak % |
|--------|-------|--------|-------|----------------|---------------|--------------|----------------------|
| CPUref | legacy CPU reference | 2729.72 | 31218 | 1.00× | 0 | 2.308 | – |
| C0 | legacy (norm_stride=1, mg_cycles=0) | 454.99 | 31218 | 6.00× | 4.98e-16 | 2.308 | 40.1 / 48.0 |
| C1 | +norm_stride=25 | 437.73 | 31225 | 6.24× | 1.03e-05 | 2.307 | 40.7 / 47.0 |
| C2 | +fixed MG cycles (3/1) | 213.80 | 31225 | 12.77× | 1.03e-05 | 2.307 | 58.6 / 63.0 |
| C3 | +CUDA graph | 115.16 | 31225 | 23.70× | 1.03e-05 | 2.307 | 94.2 / 97.0 |
| C4 | +mixed precision (p+mom) | 75.21 | 31225 | **36.29×** | 1.03e-05 | 2.307 | 89.7 / 94.0 |

## 1024² status

C4 @ 1024²: NOT converged at maxiter=80 000 (Ghia err 5.52%, wall 389.4 s, util 95/99%).
Consistent with the A100/H100 finding that 1024² needs maxiter ≥ 150k — converged rerun pending
(NS-track issue #7/#8).

## Cross-arch takeaway

On A100/H100 the mixed-precision rung C4 did NOT pay off (F64 throughput is strong there:
C3 = 32.9× on A100, 44.1× on H100). On the A6000, where F64 is 1/64 of F32, **C4 is the winning
config (36.3× vs 23.7× for C3)** — mixed precision is the enabling rung for consumer/workstation
GPUs, with parity held at 1.03e-05 and identical Ghia error. This is the honest deployment story:
datacenter cards run pure F64 (stop at C3), consumer cards need the F32 pressure/momentum path (C4).
