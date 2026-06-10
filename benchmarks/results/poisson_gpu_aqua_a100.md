# Pressure-Poisson GPU benchmark — cuDSS (A100) vs CHOLMOD (CPU)

Issue #8 GPU rung. Cut-cell pressure-Poisson (tilted embedded wall, Dirichlet MMS),
factorize-once linear-solve seam. Aqua A100-SXM4-40GB, F64, job `22299810`.
Driver: `benchmarks/krk/inc_ns/poisson_gpu_bench.jl` (run via `poisson_gpu_bench.pbs`).

## Results (same node, F64, parity enforced ‖x_gpu − x_cpu‖∞ < 1e-8)

| N | DOF | CPU factor (ms) | CPU solve (ms) | GPU factor (ms) | GPU solve (ms) | solve speed-up | parity ‖∞ |
|------|---------|-----------------|----------------|-----------------|----------------|----------------|-----------|
| 128  | 16 384  | 10.95           | 1.27           | 64.06           | 0.320          | 4.0×           | 6.1e-14   |
| 256  | 65 536  | 66.71           | 5.13           | 202.05          | 0.597          | 8.6×           | 2.7e-13   |
| 512  | 262 144 | 247.21          | 25.72          | 763.56          | 1.195          | 21×            | 1.1e-12   |
| 1024 | 1 048 576 | 1123.72       | 109.61         | 3439.25         | 3.620          | **30×**        | 4.3e-12   |

GPU solve throughput rises with N (5.1e7 → 2.9e8 cells/s) while CPU falls
(1.3e7 → 9.6e6) — the GPU advantage grows with problem size.

## Reading

- **Parity:** cuDSS F64 matches CHOLMOD to ~1e-12 — the GPU solver is numerically correct.
- **Solve (per-iteration cost):** ~30× faster on GPU at 1M DOF. This is the cost paid every
  SIMPLE outer iteration.
- **Factorize:** ~3× slower on GPU, but the Poisson matrix is geometry-only ⇒ constant across
  all outer iterations ⇒ factorized **once**.
- **Amortized over a steady cavity (3179 outer iters @ 1M DOF):**
  CPU ≈ 1.12 + 3179×0.110 ≈ **351 s** vs GPU ≈ 3.44 + 3179×0.0036 ≈ **15 s** → **~23×**.
  This is the payoff the factorize-once + assembled-sparse decision (design spike Decision 2)
  was chosen for.

## Next

- Push N to 2048/4096 (4M/16M DOF) to map the asymptotic GPU advantage.
- Port the full SIMPLE cavity solver to a CUDA backend end-to-end (operators are already KA-generic)
  and benchmark a complete GPU cavity solve vs CPU.
