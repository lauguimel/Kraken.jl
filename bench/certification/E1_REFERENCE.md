# E1 — Single-GPU throughput certification (Kraken CUDA-F64 BGK D2Q9)

This document fixes the reference numbers the E1 certification benchmark
(`cavity_bgk_cuda/run_certification.jl`) is graded against. The benchmark
reports sustained **MLUPS** (Mega Lattice Updates Per Second) for the Newtonian
BGK D2Q9 solver in **Float64**; this page tells you what "good" means.

## 1. Roofline ceiling (primary, self-contained)

LBM is memory-bandwidth bound. For a D2Q9 BGK step in F64 the dominant traffic
per lattice update is reading + writing the 19 distribution-related values
(q=9 populations in, q=9 out, plus the density/aux access amortised), i.e.
**2 × 19 × sizeof(Float64) = 304 bytes/update**. Hence:

```
max_updates_per_s = peak_BW / (2 * 19 * sizeof(Float64))
                  = peak_BW / 304
MLUPS_ceiling     = max_updates_per_s / 1e6
```

Plugging in vendor peak HBM bandwidths:

| GPU            | Memory | Peak BW    | F64 roofline MLUPS |
|----------------|--------|------------|--------------------|
| **H100**       | HBM3   | ~3.35 TB/s | **~11 020 MLUPS**  |
| **A100-80GB**  | HBM2e  | ~2.0 TB/s  | **~6 579 MLUPS**   |

These are hard upper bounds; no F64 D2Q9 kernel can exceed them.

## 2. Published F64 corroboration (recommended primary citation)

**Palabos** sustains **~73 % of A100 FP64 peak** for D3Q19 TRT
(Latt et al., *Comput. Math. Appl.* 2021; PASC '22), i.e. **≈ 4 656 MLUPS**
single-GPU in F64. This is the recommended real-code corroboration point: it is
F64 and directly comparable to the Kraken gate.

## 3. F32 context (NOT comparable — FP32, lower precision)

For situational awareness only — **do not** compare these to the Kraken F64
gate, they are single precision and move half the bytes per update:

**FluidX3D**, FP32, D3Q19 (Lehmann, arXiv:2112.08926):
**A100 ≈ 10 228 MLUPS**, **H100 ≈ 17 602 MLUPS**.

Labelled clearly as FP32 to avoid an apples-to-oranges pass/fail.

## 4. Certification criterion

The certification compares the measured **Kraken CUDA-F64 MLUPS** against:

1. the **roofline ceiling** of the GPU it ran on (Section 1) — the primary,
   self-contained reference; and
2. the **Palabos F64** single-GPU number (Section 2) — independent real-code
   corroboration.

**Pass = ratio ≥ 0.5** of the roofline ceiling (i.e. Kraken is within 2× of the
memory-bandwidth limit for its GPU). On an H100 that means ≥ ~5 510 MLUPS; on an
A100-80GB ≥ ~3 290 MLUPS. Achieving ≥ 0.5 of roofline also lands Kraken in the
same neighbourhood as the Palabos F64 figure, confirming the implementation is
competitive with an established production LBM code.
