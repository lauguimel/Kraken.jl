# SIMPLE cavity GPU-efficiency ablation — cross-arch copy (H100), Re=1000

Issues #7, #8. Same cumulative ablation as `cavity_gpu_ablation_aqua_a100.md` (the primary
artifact — reading and caveats live there), re-run unchanged on an H100 for cross-architecture
contrast: 512² Re=1000, Aqua H100 80GB HBM3 (sm_90), F64, job `22307305`, driver 580.159.03.
Driver: `benchmarks/krk/inc_ns/cavity_gpu_bench.{jl,pbs}` (H100 copy of the PBS). GPU utilization
sampled by `nvidia-smi` over the timed solve window only. CPU reference = same legacy solver on the
H100 node's CPU (2529.19 s vs 2704.13 s on the A100 node — different host CPU, so the ×-vs-CPU
columns of the two artifacts are not directly comparable; compare GPU walls instead).

## Results (512² Re=1000; CPU ref 2529.19 s, 31 218 iters, Ghia 2.308%)

| config | delta | wall (s) | iters | speed-up vs CPU | parity max-rel | Ghia err | GPU util mean/peak | step gain |
|--------|-------|----------|-------|-----------------|----------------|----------|--------------------|-----------|
| C0 | legacy kwargs (norm_stride=1, mg_cycles=0) | 351.52 | 31 218 | 7.19× | 5.0e-16 | 2.308% | 35.6% / 50% | — |
| C1 | +norm_stride=25 | 347.06 | 31 225 | 7.29× | 1.0e-05 | 2.307% | 36.0% / 42% | 1.01× |
| C2 | +fixed MG cycles (3/1) | 182.98 | 31 225 | 13.82× | 1.0e-05 | 2.307% | 53.3% / 55% | 1.90× |
| C3 | +CUDA graph | **57.35** | 31 225 | **44.10×** | 1.0e-05 | 2.307% | **93.8% / 96%** | **3.19×** |
| C4 | +mixed precision (p+mom) | 60.43 | 31 225 | 41.86× | 1.0e-05 | 2.307% | 94.3% / 96% | 0.95× |

1024² (GPU-only, pre-registered "best config" = C4): 245.0 s for 80 000 iters — maxiter hit, NOT
converged (Ghia 5.52% at stop), GPU util 96.9%/98%. Same budget shortfall as on the A100; a
converged 1024² number needs maxiter ≥150 k and should use C3.

## Cross-arch reading (vs the A100 artifact)

- **The ladder shape replicates exactly**: C1 ≈ noise, C2 ≈ 1.9×, C3 dominant (3.19× step here vs
  2.92× on A100), C4 a small net loss on top of graphs on this card too. The conclusions are
  architecture-stable, not A100 artifacts.
- **H100/A100 GPU-wall ratio at C3: 82.21/57.35 = 1.43×** (and 1.45× at 1024²) — consistent with a
  mostly bandwidth/latency-bound stencil ladder; nowhere near the F64-FLOPs ratio, confirming
  launch latency and memory traffic, not FLOPs, set the pace.
- **C4 mixed precision loses slightly on H100 as well (0.95×)** even with its faster F32 path —
  the F64↔F32 conversion and duplicate-hierarchy overhead dominates at this size, independent of
  the card's F32:F64 ratio.
