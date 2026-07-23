# Matrix-free multigrid Poisson — GPU vs CPU vs cuDSS (A100)

Issue #8, GPU-performance path (design note Decision-2-reconsidered). Regular Cartesian
Poisson, Dirichlet MMS (`u=sin πx sin πy`), red-black Gauss-Seidel V-cycle (ν1=ν2=2),
tol=1e-8. Aqua A100-SXM4-40GB, F64, job `22299933`.
Driver: `benchmarks/krk/inc_ns/poisson_mg_gpu_bench.{jl,pbs}`.

## Results

| N    | DOF        | MG-CPU (ms) | MG-GPU (ms) | MG GPU/CPU | cuDSS solve (ms) | V-cycles | parity ‖∞ (MG gpu↔cpu) |
|------|------------|-------------|-------------|------------|------------------|----------|------------------------|
| 128  | 16 384     | 9.05        | 29.10       | 0.31×      | 0.66             | 10       | 3.2e-15                |
| 256  | 65 536     | 35.19       | 32.80       | 1.07×      | 0.74             | 10       | 9.7e-15                |
| 512  | 262 144    | 158.76      | 41.38       | 3.84×      | 1.83             | 11       | 1.7e-14                |
| 1024 | 1 048 576  | 750.07      | 59.14       | 12.68×     | 4.73             | 12       | 1.7e-14                |
| 2048 | 4 194 304  | 3399.62     | 108.74      | 31.26×     | 15.86            | 12       | 6.3e-14                |
| 4096 | 16 777 216 | 13886.99    | 320.64      | **43.31×** | 62.09            | 13       | 1.3e-13                |

MG-CPU analytic order = 2.000 at every N. V-cycle counts [10,10,11,12,12,13] — flat
(multigrid O(1)-cycles hallmark) even at 16M DOF.

## GPU utilization (nvidia-smi 1 Hz during the run)

- **mean 2.3% / peak 99%.** The low mean is a measurement artifact: the bench spends most
  wall-time in the slow MG-**CPU** reps (14 s × reps at 16M) + Julia/CUDA load, GPU idle in
  between. During the MG-GPU kernels the device hits **99%** (last samples 100/89/100%).
- **The matrix-free MG saturates the A100** — the red-black Gauss-Seidel smoother + stencil
  restriction/prolongation are LBM-like and keep the cores busy, unlike cuDSS's sequential
  triangular solves (its job peaked far lower).

## Reading

- **MG GPU/CPU speed-up grows with N** (0.31× → 43× at 16M, still rising) — the GPU-friendly
  scaling cuDSS's flat ~30× did not show.
- **MG scales to 16M DOF in O(N) memory**; cuDSS factorization at that size used ~15 GB
  resident / 118 GB virtual.
- **But for moderate-N 2D steady (factorization fits), cuDSS still wins per-solve:** its
  amortized back-substitution (4.73 ms at 1M) beats a full MG solve (59 ms at 1M), because it
  factorizes ONCE and reuses over ~3000 SIMPLE iterations. MG wins decisively at **large N,
  in 3D** (direct factorization explodes), and for **unsteady / changing matrices** (no
  factorize-once amortization). Different regimes — not a replacement.
- **Honest cap vs LBM:** even GPU-saturated, the elliptic MG does ~13 V-cycles × multi-level
  work per DOF (≫ one local LBM update) plus global reductions → it will not reach LBM's
  ~1000× CPU→GPU. The FVFD steady solver's real advantage over LBM is **iteration count**
  (~1e3 vs ~1e6–1e7 to steady), not per-solve GPU throughput.

## Next

- Clean GPU-only sustained-utilization micro-bench (drop the CPU/load phases) for a true
  saturation number.
- Port the full SIMPLE cavity to a CUDA backend end-to-end (MG for the pressure solve), bench
  a complete GPU cavity vs CPU.
