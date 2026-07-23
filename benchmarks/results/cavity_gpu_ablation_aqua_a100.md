# SIMPLE cavity GPU-efficiency ablation — per-change contribution (A100), Re=1000

Issues #7, #8. Cumulative ablation of the GPU-efficiency levers on the backend-parametric SIMPLE
cavity (`src/methods/inc_ns/cavity_mg.jl` + `cavity_mg_cuda.jl`), 512² Re=1000, Aqua A100-SXM4-40GB,
F64, job `22306791`. Driver: `benchmarks/krk/inc_ns/cavity_gpu_bench.{jl,pbs}`. GPU utilization is
sampled by `nvidia-smi` over the **timed solve window only** (not the whole job). The CPU reference
is the SAME legacy solver on the node's CPU — note it too got faster (2704 s vs 3687 s in job
`22305186`) because the kernel fusion and sync removal help the CPU path as well: every speed-up
below is measured against the *improved* CPU, i.e. conservatively.

## Results (512² Re=1000; CPU ref 2704.13 s, 31 218 iters, Ghia 2.308%)

| config | delta | wall (s) | iters | speed-up vs CPU | parity max-rel | Ghia err | GPU util mean/peak | step gain |
|--------|-------|----------|-------|-----------------|----------------|----------|--------------------|-----------|
| C0 | legacy kwargs (norm_stride=1, mg_cycles=0) | 466.22 | 31 218 | 5.80× | 3.3e-16 | 2.308% | 35.5% / 42% | — |
| C1 | +norm_stride=25 | 455.11 | 31 225 | 5.94× | 1.0e-05 | 2.307% | 36.1% / 44% | 1.02× |
| C2 | +fixed MG cycles (3/1) | 239.64 | 31 225 | 11.28× | 1.0e-05 | 2.307% | 43.9% / 51% | 1.90× |
| C3 | +CUDA graph | **82.21** | 31 225 | **32.89×** | 1.0e-05 | 2.307% | **88.8% / 96%** | **2.92×** |
| C4 | +mixed precision (p+mom) | 94.95 | 31 225 | 28.48× | 1.0e-05 | 2.307% | 81.7% / 96% | 0.87× |

1024² (GPU-only, pre-registered "best config" = C4): 355.8 s for 80 000 iters — **maxiter hit, NOT
converged** (Ghia 5.52% at stop), GPU util 97.5%/99%.

## Reading — where the 865 s actually went

- **Headline: 865 s → 82 s (10.5× on the GPU path itself), 4.26× → 32.9× vs CPU, util 20% → 89%.**
  The Mission-A target (~15×, ~70% util) is exceeded. Correctness intact end-to-end: parity vs CPU
  1.0e-5 ≪ 1e-3 gate (the deviation is the fast-path stopping 7 iters later, not numerics), Ghia
  unchanged at 2.307%.
- **C0 vs the old baseline (4.26× → 5.80×, util ~20% → 35.5%):** the *unconditional* changes —
  kernel fusion 18→7 and per-launch sync removal — are in every config including legacy kwargs.
  Fusion alone bought ~1.9× on the GPU wall and ~1.4× on the CPU.
- **C1 norm-stride is NOT the big unitary win we predicted** (1.02×): at 512² the outer-norm host
  syncs were already minor next to the inner MG-residual reductions. Its real value is enabling the
  static launch sequence (C2/C3 require it).
- **C2 fixed inner V-cycles, 1.90×:** eliminates ALL inner reductions and host syncs (3 pressure +
  1 momentum V-cycles per outer iteration, no per-cycle residual). This is where the
  "fewer/batched reductions" lever actually lived.
- **C3 CUDA graph, 2.92× — the dominant lever.** One graph replay per off-stride outer iteration
  (29 973 replays) amortizes ~800 kernel launches/iter into one. Launch latency, not bandwidth or
  FLOPs, was the bottleneck: util jumps 43.9% → 88.8%.
- **C4 mixed precision is a net LOSS on top of graphs (0.87×).** Once launches are amortized, the
  F32 V-cycle's bandwidth saving is outweighed by the F64↔F32 conversion kernels and duplicate
  hierarchy traffic at this size. Keep it default-OFF for the steady cavity; it may still pay in
  bandwidth-bound regimes (larger grids / 3D) **but that is unmeasured** — the only 1024² run used
  C4 by pre-registration, so C3-vs-C4 at 1024² is an open measurement.

## The honest caveats

- **1024² did not converge** in its 80 000-iteration budget (Ghia 5.52% at stop, still relaxing).
  SIMPLE outer-iteration count grows with grid (31 k at 512²); the budget, not the throughput, was
  short — util was 97.5%. A converged 1024² number needs a larger maxiter (≥150 k by extrapolation)
  and should use C3.
- **32.9× is vs one CPU job on the same node** (cpupercent ≈ 86, i.e. ~1 busy core) — the right
  comparison for "what does moving this run to the GPU buy", and the per-GPU order of magnitude now
  sits inside the Fluent-class 30–100×-per-GPU band quoted in the design note. It is NOT a
  multi-core-CPU comparison, and LBM's ~1000× per-kernel class remains out of reach for a
  segregated elliptic solver (global coupling ⇒ V-cycle ladders + occasional norms).
- The remaining ~11% idle is the coarse-MG levels (tiny grids can't fill an A100) + on-stride norm
  checks + host loop — diminishing returns from here; the next real lever is **iteration count**
  (coupled/JFNK), not kernel efficiency.

## Next

- 1024² converged run: C3, maxiter ≥150 k (walltime fine: ~6 min per 80 k iters).
- C3 as the documented production GPU path (flags: `norm_stride=25, mg_cycles=3, mom_mg_cycles=1`,
  graph executor from `cavity_mg_cuda.jl`); mixed precision stays opt-in/off.
- Coupled/JFNK (v2) to attack the 31 k outer iterations themselves.
- Unsteady projection driver reuses the same static-iteration structure — graph capture should
  transfer directly.
