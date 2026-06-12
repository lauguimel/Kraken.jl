# GPU-performance narrative — incompressible steady NS (FVFD/SIMPLE)

**Issues:** #7 (the `IncNS` method) · #8 (linear-solve service + GPU efficiency).
**Branch:** `dev/platform`. **Status:** measurement record — closes the GPU-efficiency
campaign (Mission A). Extends [inc_ns_design_spike.md](inc_ns_design_spike.md)
§"Decision 2 — RECONSIDERED" and §"Honest cap" with the measured end state.

This is the complete, honest story of what GPU acceleration buys this solver —
written for humans and for future sessions, so nobody re-litigates it from
partial numbers. Every figure below is from a recorded Aqua A100 artifact;
nothing is extrapolated.

## 1. Why elliptic is not LBM on a GPU (and why that's fine)

The steady incompressible problem is **elliptic**: infinite sound speed, every
pressure solve couples the whole domain. LBM is the opposite extreme — one
fused, explicit, *local* kernel that saturates a GPU by construction, but must
wait out the physical transient: **~1e6–1e7 time steps** to a steady state. The
elliptic solver relaxes in **~1e3–1e4 outer iterations**. That three-to-four
order-of-magnitude **iteration-count advantage is the real win** — not
per-kernel throughput, where LBM is unbeatable. Per-kernel, global coupling
costs V-cycle ladders plus occasional global norms; LBM's ~1000× CPU→GPU class
is structurally out of reach for a segregated elliptic solver. The right
expectation is the commercial-FV class instead (see §4).

## 2. The measurement ladder (what we measured, in order)

1. **cuDSS per-solve** ([poisson_gpu_aqua_a100.md](../../benchmarks/results/poisson_gpu_aqua_a100.md)):
   back-substitution **30× at 1 M DOF** (4.0× → 8.6× → 21× → 30×), factorize-once
   amortized ~23× over a steady cavity — but the job averaged only **~9 % GPU
   utilization**: triangular solves are sequentially dependent, the device is
   starved. This triggered the spike's "Decision 2 — RECONSIDERED".
2. **Matrix-free MG** ([poisson_mg_gpu_aqua_a100.md](../../benchmarks/results/poisson_mg_gpu_aqua_a100.md)):
   GPU/CPU **grows with N** — 0.31× at 128² to **43.31× at 16.8 M DOF**, still
   rising; V-cycles flat at 10–13 (O(N)); the stencil smoother hits **99 %**
   during GPU kernels. MG saturates the device where cuDSS cannot.
3. **End-to-end SIMPLE cavity** ([cavity_gpu_aqua_a100.md](../../benchmarks/results/cavity_gpu_aqua_a100.md)):
   **4.26× at 512²** (865 s vs 3 687 s), bit-identical fields (~1e-16), but GPU
   utilization mean **4.5 %** sampled over the whole job (~20 % over the timed
   solve window, the ablation's C-series baseline). Diagnosis — the
   **orchestration gap**: ~31 k outer iterations, each a chain of small kernels,
   inner MG solves with per-cycle reductions, outer norms forcing host syncs.
   Launch latency and host round-trips dominated; "~20× left on the table".

The ladder's lesson: the kernels were never the problem; the *loop* was.

## 3. The efficiency campaign and its ablation

Cumulative ablation, 512² Re=1000, same node, F64
([cavity_gpu_ablation_aqua_a100.md](../../benchmarks/results/cavity_gpu_ablation_aqua_a100.md),
job 22306791). CPU reference is the **improved** solver on CPU (2 704 s, down
from 3 687 s — fusion helped the CPU too), so the speed-ups are conservative.

| config | delta | wall (s) | speed-up | util mean/peak | step gain |
|---|---|---|---|---|---|
| C0 | legacy kwargs (fusion + sync removal only) | 466.22 | 5.80× | 35.5 % / 42 % | — |
| C1 | + norm_stride=25 | 455.11 | 5.94× | 36.1 % / 44 % | 1.02× |
| C2 | + fixed MG cycles (3/1) | 239.64 | 11.28× | 43.9 % / 51 % | 1.90× |
| C3 | + CUDA graph | **82.21** | **32.89×** | **88.8 % / 96 %** | **2.92×** |
| C4 | + mixed precision (p+mom) | 94.95 | 28.48× | 81.7 % / 96 % | 0.87× |

Headline: **865 s → 82 s (10.5× on the GPU path), 4.26× → 32.9× vs CPU, util
20 % → 88.8 %**, correctness intact (parity 1.0e-5 = stopping 7 iterations
later, not numerics; Ghia 2.307 % unchanged). Which lever paid, and why:

- **CUDA graph (C3, 2.92×) — the dominant lever.** One graph replay per
  off-stride iteration amortizes ~800 kernel launches into one. Launch latency,
  not bandwidth or FLOPs, was the bottleneck — util jumps 43.9 % → 88.8 %.
- **Fixed inner V-cycles (C2, 1.90×).** Eliminates ALL inner MG-residual
  reductions and host syncs. The "fewer/batched reductions" lever lived here,
  in the *inner* solves — not in the outer norms.
- **Norm-stride (C1, 1.02×) — our misprediction.** We expected the outer-norm
  host syncs to be the biggest unitary gain; measured, they were noise next to
  the inner reductions. Its real value is *enabling* the static launch sequence
  C2/C3 require. Lesson: a plausible bottleneck diagnosis (host syncs in the
  hot loop) is not a measurement — ablate before optimizing further.
- **Mixed precision (C4, 0.87×) — net loss once graphs amortize launches.**
  The F32 V-cycle bandwidth saving is outweighed by F64↔F32 conversion kernels
  and duplicate hierarchy traffic at this size. Default OFF; possibly useful in
  bandwidth-bound regimes (larger grids / 3D) but **unmeasured** — the only
  1024² run pre-registered C4, so C3-vs-C4 at 1024² is open.
- **C0 vs the old 4.26× baseline:** kernel fusion (18 → 7) and per-launch sync
  removal are unconditional, in every config — they alone gave 5.80×.

Caveat, stated plainly: **the 1024² run did not converge** in its 80 k-iteration
budget (Ghia 5.52 % at stop, util 97.5 %). The budget, not the throughput, was
short; a converged number needs maxiter ≥ 150 k and should use C3.

**Cross-architecture replication (H100, job 22307305,
`benchmarks/results/cavity_gpu_ablation_aqua_h100.md`):** the identical ablation
on an H100 reproduces the ladder shape (1.01× / 1.90× / 3.19× / 0.95×) — C3 at
**57.3 s, 44.1× vs its node's CPU, 93.8 % util**, mixed precision still a net
loss. The C3 H100/A100 wall ratio (~1.43×) tracks the HBM bandwidth ratio, not
the FLOPs ratio: independent confirmation that the converged solver is
latency/bandwidth-bound, and that the ablation's conclusions are not an
A100 artifact.

## 4. The honest cap, and the next lever

**32.9× sits inside the ~30–100×-per-GPU band commercial FV codes (Fluent
class) report vs a single CPU core** — that was the spike's calibration target,
now met with measurements rather than hope. Equally honestly:

- It is a **one-GPU vs one-busy-CPU-core** comparison (cpupercent ≈ 86), the
  right framing for "what does moving this run to the GPU buy" — not a
  multi-core CPU comparison.
- **Multi-GPU scaling and all-physics robustness** at that speed remain out of
  scope. **LBM's ~1000× class stays unreachable** for segregated elliptic
  (global coupling ⇒ V-cycle ladders + norms) — see §1; the iteration-count
  advantage is the compensating asset.
- The remaining ~11 % idle is coarse MG levels (tiny grids can't fill an A100),
  on-stride norm checks, and the host loop — **diminishing returns**. The next
  real lever is **iteration count itself**: a coupled / Newton–Krylov (JFNK)
  scheme attacking the 31 k outer iterations, not kernel efficiency.
- The **unsteady projection driver inherits the same static-iteration
  structure** (fixed inner cycles, off-stride steps), so the graph capture
  should transfer directly.

Production GPU path, as documented to users: **C3** — solver defaults
`norm_stride=25, mg_cycles=3, mom_mg_cycles=1` plus the CUDA-graph executor
(`src/methods/inc_ns/cavity_mg_cuda.jl`, manual-load under `using CUDA`);
mixed precision opt-in.

## Cross-references

- Ablation artifact: [benchmarks/results/cavity_gpu_ablation_aqua_a100.md](../../benchmarks/results/cavity_gpu_ablation_aqua_a100.md)
- Prior rungs: [poisson_gpu_aqua_a100.md](../../benchmarks/results/poisson_gpu_aqua_a100.md) ·
  [poisson_mg_gpu_aqua_a100.md](../../benchmarks/results/poisson_mg_gpu_aqua_a100.md) ·
  [cavity_gpu_aqua_a100.md](../../benchmarks/results/cavity_gpu_aqua_a100.md)
- User page (table + figure): [docs/src/users/incompressible-navier-stokes.md](../src/users/incompressible-navier-stokes.md)
- Decision record this extends: [inc_ns_design_spike.md](inc_ns_design_spike.md)
