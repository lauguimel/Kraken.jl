# End-to-end SIMPLE cavity — GPU vs CPU (A100), Re=1000

Issue #7. Full backend-parametric SIMPLE cavity (`src/methods/inc_ns/cavity_mg.jl`), matrix-free
multigrid for pressure + momentum, all KA kernels. Same source CPU/CUDA. Aqua A100-SXM4-40GB,
F64, job `22305186`. Driver: `benchmarks/krk/inc_ns/cavity_gpu_bench.{jl,pbs}`.

## Results (full SIMPLE solve wall time)

| N   | DOF     | CPU (s) | GPU (s) | speed-up | iters  | Ghia-err CPU | Ghia-err GPU | parity ‖Δ‖∞ (GPU↔CPU) |
|-----|---------|---------|---------|----------|--------|--------------|--------------|------------------------|
| 256 | 65 536  | 271.79  | 277.97  | 0.98×    | 9 432  | 4.127%       | 4.127%       | 2.2e-16                |
| 512 | 262 144 | 3686.92 | 865.03  | **4.26×**| 31 218 | **2.308%**   | 2.308%       | 1.1e-16                |

## Reading

- **Correctness — perfect.** GPU and CPU give **bit-identical** results (parity ~1e-16); both match
  Ghia 1982 Re=1000 (2.31% at 512², improving with grid). The backend-parametric port (same source,
  `backend_ka=CUDABackend(), atype=CuArray{Float64}`) is numerically flawless.
- **Speed-up modest:** 0.98× at 256² (too small — overhead dominates), **4.26× at 512²** (14 min vs
  61 min). Far below the Poisson-only MG GPU/CPU at the same size (~3.8× at 512² in the MG bench) is
  *consistent* — the cavity speed-up ≈ its dominant Poisson-solve speed-up, capped by SIMPLE overhead.
- **GPU utilization: mean 4.5%, peak 9%** — the A100 is ~95% idle during the GPU run.

## The structural lesson

The segregated SIMPLE loop is **intrinsically GPU-inefficient**: ~31k outer iterations, each with
several MG solves + advection/gradient/correction kernels + **global reductions** (residual /
velocity-change norms) + host-side orchestration. Kernel-launch latency and host round-trips
dominate; the GPU is starved (4.5% util). Unlike LBM — one fused, explicit, local kernel that
saturates the device — a steady segregated elliptic solver **cannot** saturate the GPU as written.

The 4.26× is "free" speed-up from offloading stencil work, but ~20× is left on the table (per the
4.5% utilization). Real GPU performance for the NS solver requires solver-architecture work, not
just "run it on the GPU":
- **kernel fusion** (collapse the many small per-iteration kernels);
- **batched / fewer global reductions** (the per-iteration norms force host sync);
- **fewer outer iterations** — a coupled / Newton–Krylov (JFNK) scheme instead of segregated SIMPLE;
- **larger grids** (1024²+) where per-iteration compute amortizes the launch overhead.

This is the full-solver confirmation of the earlier Poisson finding: the elliptic/segregated steady
solver's value vs LBM is **iteration count** (~1e3–1e4 vs 1e6–1e7 to steady) and robust convergence —
not raw GPU throughput.

## Next

- 1024² run (`CAVITY_BENCH_1024=1`) to see the speed-up grow with grid.
- Kernel-fusion + reduction-batching pass to lift GPU utilization.
- Unsteady driver (projection/PISO): per-step structure is similar, but a fixed-geometry Poisson
  amortizes a one-time factorization across thousands of steps (favours cuDSS there).
