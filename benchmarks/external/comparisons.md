# External comparisons

Direct comparisons between Kraken.jl and established LBM / CFD codes.
All Kraken.jl numbers are from the AQUA H100 80 GB PCIe unless noted.

## Performance (MLUPS) — GPU

| Code | Language | Backend | Lattice | N | MLUPS | Source |
|------|----------|---------|---------|---|------:|--------|
| Kraken.jl | Julia | H100 | D2Q9 | 1024 | 24 041 | this work (AA+f32) |
| Kraken.jl | Julia | H100 | D2Q9 | 1024 | 12 840 | this work (fused, f64) |
| Kraken.jl | Julia | H100 | D2Q9 | 1024 | 5 631 | this work (baseline, f64) |
| waLBerla | C++ | A100 | D3Q19 | 256 | 4 700 | [Bauer 2021](https://doi.org/10.1016/j.cpc.2020.107746) |
| waLBerla | C++ | V100 | D3Q19 | 256 | 2 800 | [Bauer 2021](https://doi.org/10.1016/j.cpc.2020.107746) |
| waLBerla | C++ | H100 | D3Q19 | 1024 | ~7 000 | estimated (A100 × 1.5 BW ratio) |
| Sailfish | Python/CUDA | GTX Titan | D2Q9 | 1024 | 3 200 | [Januszewski 2014](https://doi.org/10.1016/j.cpc.2014.04.018) (f32) |
| Sailfish | Python/CUDA | GTX Titan | D3Q19 | 256 | 2 200 | [Januszewski 2014](https://doi.org/10.1016/j.cpc.2014.04.018) (f32) |
| OpenLB | C++ | V100 | D3Q19 | 256 | ~1 800 | estimated from [Krause 2021](https://doi.org/10.1016/j.camwa.2020.04.033) |

> **Caveat:** waLBerla and OpenLB numbers are for D3Q19 (19 distributions per node),
> while Kraken.jl numbers above are D2Q9 (9 distributions). D3Q19 moves ~2× more
> data per lattice update, so raw MLUPS are not directly comparable across lattice
> types. The comparison is included to show order-of-magnitude positioning.

## Performance (MLUPS) — CPU

| Code | Language | Backend | Lattice | N | MLUPS | Source |
|------|----------|---------|---------|---|------:|--------|
| Kraken.jl | Julia | Apple M2 (1 core) | D2Q9 | 256 | 27 | this work |
| Palabos | C++ | Xeon (1 core) | D3Q19 | 128 | 3.5 | [Latt 2021](https://doi.org/10.1016/j.camwa.2020.03.022) |
| Palabos | C++ | Xeon (24 cores) | D3Q19 | 128 | 55 | [Latt 2021](https://doi.org/10.1016/j.camwa.2020.03.022) |
| OpenLB | C++ | Xeon (1 core) | D3Q19 | 128 | 4.2 | [Krause 2021](https://doi.org/10.1016/j.camwa.2020.04.033) |
| OpenLB | C++ | Xeon (24 cores) | D3Q19 | 128 | 65 | [Krause 2021](https://doi.org/10.1016/j.camwa.2020.04.033) |
| waLBerla | C++ | Xeon 8280 (28 cores) | D3Q19 | 256 | 420 | [Bauer 2021](https://doi.org/10.1016/j.cpc.2020.107746) |

> **Caveat:** CPU comparisons mix D2Q9 (Kraken) and D3Q19 (others), and different
> architectures (M2 vs Xeon). Per-core throughput depends heavily on cache size,
> memory bandwidth, and SIMD width.

## Accuracy — lid-driven cavity Re = 100

| Code | N | Error vs Ghia | Source |
|------|---|---------------|--------|
| Kraken.jl | 64 | ~5% | this work |
| Kraken.jl | 128 | <1% | this work |
| Palabos | 128 | <1% | [Latt 2021](https://doi.org/10.1016/j.camwa.2020.03.022) |
| OpenLB | 128 | <1% | [Krause 2021](https://doi.org/10.1016/j.camwa.2020.04.033) |
| OpenFOAM | 128 | <1% | icoFoam standard tutorial (FVM, not LBM) |

> All proper LBM implementations converge to the Ghia reference at N >= 128.
> This is a necessary but not distinctive benchmark.

## Convergence order — Poiseuille flow

| Code | Observed order | Source |
|------|---------------|--------|
| Kraken.jl | 2.00 | this work |
| Palabos | 2.0 | [Latt 2021](https://doi.org/10.1016/j.camwa.2020.03.022) |
| OpenLB | 2.0 | [Krause 2021](https://doi.org/10.1016/j.camwa.2020.04.033) |

> Second-order spatial convergence is the theoretical expectation for BGK-LBM
> with bounce-back boundaries. All codes confirm this.

## Bandwidth efficiency

| Code | Backend | % of peak BW | Source |
|------|---------|-------------|--------|
| Kraken.jl (fused) | H100 | 65% | this work |
| Kraken.jl (AA+f32) | H100 | 52% | this work |
| waLBerla | A100 | 75% | [Bauer 2021](https://doi.org/10.1016/j.cpc.2020.107746) |

> waLBerla achieves higher bandwidth efficiency through hand-tuned C++/CUDA
> kernels with shared-memory tiling. Kraken.jl's 52-65% with pure Julia
> KernelAbstractions.jl is competitive for a high-level language implementation.

## Notes

- Direct comparisons are approximate: hardware generations, compiler versions,
  and exact test configurations differ across publications.
- Kraken.jl baseline numbers use the standard modular pipeline
  (separate stream, collide, macro kernels). Fused and AA numbers use
  optional optimized kernels.
- waLBerla, Palabos, and OpenLB are mature C++ frameworks with years of
  optimization. Kraken.jl is a single-developer Julia package.
- Sailfish numbers are from 2014 hardware (GTX Titan). Scaling to modern
  GPUs would likely yield comparable or higher MLUPS than Kraken.jl.
- OpenFOAM is included as a finite-volume reference, not an LBM code.
  MLUPS is not a standard metric for FVM solvers.
