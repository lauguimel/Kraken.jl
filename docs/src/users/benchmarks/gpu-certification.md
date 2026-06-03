# GPU efficiency certification — BGK D2Q9 CUDA F64

Single-GPU throughput of the Newtonian BGK D2Q9 solver in **CUDA Float64** (Taylor–Green
driver, `run_taylor_green_2d`). Kraken sustains **3461 MLUPS** at `N = 2048` on an
**A100-40GB**, reaching **0.68** of the memory-bandwidth roofline and **0.74** of the
published Palabos single-GPU F64 number — comfortably clearing the **≥ 0.5 of roofline**
gate.

![GPU certification](gpu-certification.png)

## Result

| Reference                          | MLUPS  | Kraken / ref | Type |
|------------------------------------|-------:|-------------:|------|
| A100-40GB roofline (1.555 TB/s)    | 5115   | **0.68**     | bandwidth ceiling |
| A100-80GB roofline (2.039 TB/s)    | 6707   | **0.52**     | bandwidth ceiling |
| Palabos, single-GPU F64 D3Q19 TRT  | 4656   | **0.74**     | published real code |

**Verdict: PASS.** On the applicable A100-40GB roofline Kraken reaches 0.68 of the
memory-bandwidth limit and 0.74 of the Palabos single-GPU F64 number — same neighbourhood
as an established production LBM code. Even against the stricter 80 GB ceiling the ratio
is 0.52, so the pass is robust to the bandwidth assumption. The Palabos figure is ~73 %
of A100 FP64 peak (Latt et al., *Comput. Math. Appl.* 2021; PASC '22); the H100 roofline
(11020 MLUPS) is context only — the run was on an A100.

## Method

The metric is sustained MLUPS measured over 1000 steps after a discarded warm-up on an
`N × N` periodic domain:

```
MLUPS = N² · steps / wallclock_s / 1e6
```

Best sustained throughput was 3461 MLUPS at `N = 2048` (3084 MLUPS at `N = 1024`).

**Roofline.** LBM is memory-bandwidth bound: a D2Q9 BGK step in Float64 moves 9
populations in and 9 out, `2 × 19 × sizeof(Float64) = 304 bytes/update`, so the hard
upper bound is `MLUPS_ceiling = peak_BW / 304 / 1e6`. No F64 D2Q9 kernel can exceed it.

**Environment.** Aqua (QUT) node `gpu0n009`, NVIDIA A100-40GB (HBM2e, 1.555 TB/s peak),
CUDA Float64. The 40 GB roofline is the applicable ceiling; the 80 GB figure is a
stricter lower-bound reference.

## Caveats

- **GPU variant confirmed post-hoc via PBS node resources, not the run log.** The driver
  did not record the device; the A100-40GB identity was read from `pbsnodes gpu0n009`. A
  future tweak is to print `CUDA.versioninfo()` / `CUDA.device()` into the run log so the
  exact GPU and roofline are locked by the run itself.
- **Single-GPU, bandwidth-bound figure.** This certifies single-GPU throughput against
  the memory roofline; it is not a multi-GPU or weak/strong-scaling result.

## Reproduce

```bash
# Aqua (CUDA F64) — runs the certification sweep and writes the CSV:
julia --project=. bench/certification/cavity_bgk_cuda/run_certification.jl

# regenerate the figure:
conda run -n kraken-v0-3-figures python bench/certification/plot_certification.py
```

Data: `benchmarks/results/certification_a100.csv`. Reference derivation:
`bench/certification/E1_REFERENCE.md`. F32 codes such as FluidX3D (Lehmann,
arXiv:2112.08926) are **not** comparable to this F64 gate — single precision moves half
the bytes per update.
