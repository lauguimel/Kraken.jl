# External comparisons

Direct comparisons between Kraken.jl and established LBM / CFD codes.
All Kraken.jl numbers are from the AQUA H100 80 GB PCIe unless noted.

## Performance (MLUPS) — single GPU, D3Q19

| Code | Language | Backend | Lattice | Prec | MLUPS | Source |
|------|----------|---------|---------|------|------:|--------|
| FluidX3D | C++/OpenCL | H100 SXM | D3Q19 | f32 | 17,602 | [GitHub README](https://github.com/ProjectPhysX/FluidX3D) |
| FluidX3D | C++/OpenCL | H100 SXM | D3Q19 | f16s | 29,561 | [GitHub README](https://github.com/ProjectPhysX/FluidX3D) |
| FluidX3D | C++/OpenCL | A100 SXM | D3Q19 | f32 | 10,228 | [GitHub README](https://github.com/ProjectPhysX/FluidX3D) |
| Palabos | C++ stdpar | A100 SXM4 | D3Q19 | f32 | 9,481 | [GPU port, arXiv 2506.09242](https://arxiv.org/abs/2506.09242) |
| Palabos | C++ stdpar | A100 SXM4 | D3Q19 | f64 | 4,921 | [GPU port, arXiv 2506.09242](https://arxiv.org/abs/2506.09242) |
| OpenLB | C++ | 1× A100 | D3Q19 | f32 | 24,800 | [openlb.net/performance](https://www.openlb.net/performance/) |
| waLBerla | C++ | A100 | D3Q19 | f64 | 4,700 | [Bauer 2021](https://doi.org/10.1016/j.cpc.2020.107746) |
| waLBerla | C++ | V100 | D3Q19 | f64 | 2,800 | [Bauer 2021](https://doi.org/10.1016/j.cpc.2020.107746) |
| FluidX3D | C++/OpenCL | V100 SXM | D3Q19 | f32 | 4,471 | [GitHub README](https://github.com/ProjectPhysX/FluidX3D) |
| FluidX3D | C++/OpenCL | RTX 4090 | D3Q19 | f32 | 5,624 | [GitHub README](https://github.com/ProjectPhysX/FluidX3D) |

## Performance (MLUPS) — Kraken.jl (D2Q9)

| Variant | Backend | Lattice | Prec | MLUPS | Notes |
|---------|---------|---------|------|------:|-------|
| Baseline (3 kernels) | H100 PCIe | D2Q9 | f64 | 5,631 | |
| Fused stream+collide+macro | H100 PCIe | D2Q9 | f64 | 12,840 | |
| AA-pattern + Float32 | H100 PCIe | D2Q9 | f32 | 24,041 | |
| Single-thread | Apple M2 | D2Q9 | f64 | 27 | |

> **D2Q9 vs D3Q19 caveat:** Kraken.jl numbers are D2Q9 (9 distributions,
> 2×9×8 = 144 bytes/cell f64). The other codes benchmark D3Q19
> (19 distributions, 2×19×8 = 304 bytes/cell f64). Raw MLUPS are not
> comparable across lattice types. Bandwidth efficiency (below) is the
> fair metric.

## Bandwidth efficiency

| Code | Backend | Lattice | % of peak BW | Source |
|------|---------|---------|-------------|--------|
| Kraken.jl (fused, f64) | H100 PCIe | D2Q9 | 65% | this work |
| Kraken.jl (AA, f32) | H100 PCIe | D2Q9 | 52% | this work |
| waLBerla | A100 | D3Q19 | 75% | [Bauer 2021](https://doi.org/10.1016/j.cpc.2020.107746) |
| Palabos (GPU port) | A100 | D3Q19 | 73% | [arXiv 2506.09242](https://arxiv.org/abs/2506.09242) |

> Bandwidth efficiency is the meaningful metric for cross-lattice comparisons.
> waLBerla and Palabos achieve 73-75% via hand-tuned C++/CUDA or code generation.
> Kraken.jl's 65% with pure Julia KernelAbstractions.jl is competitive
> for a high-level language, with room to improve via shared-memory tiling.

## Performance (MLUPS) — CPU

| Code | Language | Backend | Lattice | N | MLUPS | Source |
|------|----------|---------|---------|---|------:|--------|
| Kraken.jl | Julia | Apple M2 (1 core) | D2Q9 | 256 | 27 | this work |
| Palabos | C++ | Xeon (1 core) | D3Q19 | 128 | 3.5 | [Latt 2021](https://doi.org/10.1016/j.camwa.2020.03.022) |
| Palabos | C++ | Xeon (24 cores) | D3Q19 | 128 | 55 | [Latt 2021](https://doi.org/10.1016/j.camwa.2020.03.022) |
| OpenLB | C++ | Xeon (1 core) | D3Q19 | 128 | 4.2 | [Krause 2021](https://doi.org/10.1016/j.camwa.2020.04.033) |
| OpenLB | C++ | Xeon (24 cores) | D3Q19 | 128 | 65 | [Krause 2021](https://doi.org/10.1016/j.camwa.2020.04.033) |
| waLBerla | C++ | Xeon 8280 (28 cores) | D3Q19 | 256 | 420 | [Bauer 2021](https://doi.org/10.1016/j.cpc.2020.107746) |

## Multi-GPU scaling (for reference)

| Code | Backend | GPUs | GLUPS | Source |
|------|---------|------|------:|--------|
| OpenLB | 4× A100 (HoreKa) | 4 | 24.8 | [openlb.net](https://www.openlb.net/performance/) |
| OpenLB | 512× A100 (HoreKa) | 512 | 1,330 | [openlb.net](https://www.openlb.net/performance/) |
| OpenLB | ~1000 nodes (Aurora) | ~6000 | 21,120 | [openlb.net](https://www.openlb.net/performance/) |

> Kraken.jl is single-GPU only in v0.1.0. Multi-GPU is planned for v0.2.0.

## Accuracy — lid-driven cavity

| Code | Re | N | Error vs Ghia | Source |
|------|-----|---|---------------|--------|
| Kraken.jl | 100 | 64 | ~5% | this work |
| Kraken.jl | 100 | 128 | <1% | this work |
| Palabos | 100 | 128 | <1% | [Latt 2021](https://doi.org/10.1016/j.camwa.2020.03.022) |
| OpenLB | 100 | 128 | <1% | [Krause 2021](https://doi.org/10.1016/j.camwa.2020.04.033) |
| OpenFOAM | 100 | 128 | <1% | icoFoam tutorial (FVM, not LBM) |

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

## Standard LBM validation cases (community)

Cases that the major codes all implement, useful for cross-code comparison:

| Case | Reference | Key metric | Kraken status |
|------|-----------|------------|---------------|
| Poiseuille 2D | Analytical | L2 error, order=2 | **done** |
| Taylor-Green 2D | Analytical | L2 decay, order=2 | **done** |
| Lid-driven cavity Re=100 | Ghia 1982 | Centerline u,v | **done** |
| Lid-driven cavity Re=1000 | Ghia 1982 | Stability + accuracy | **TODO** |
| Cylinder Re=20 (Schäfer-Turek) | Schäfer & Turek 1996 | Cd=5.58, Cl≈0 | **TODO** |
| Cylinder Re=100 (periodic) | Schäfer & Turek 1996 | Cd_max, Cl_max, St | **TODO** |
| Natural convection Ra=10^4 | De Vahl Davis 1983 | Nu=2.243 | partial (Ra=1k) |
| Natural convection Ra=10^6 | De Vahl Davis 1983 | Nu=8.8 | **TODO** |
| Static droplet (Laplace) | Analytical | ΔP/σ error | **done** (multiphase) |
| Double shear layer | -- | MRT stability | **TODO** |

## Notes

- Direct comparisons are approximate: hardware generations, compiler versions,
  and exact test configurations differ across publications.
- FluidX3D is the current single-GPU performance champion thanks to esoteric
  pull + FP16 memory compression. It is optimized for throughput, not physics
  flexibility.
- waLBerla achieves high bandwidth efficiency via lbmpy code generation with
  shared-memory tiling. Palabos GPU port achieves similar efficiency.
- OpenLB's strength is multi-GPU scaling (1.33 TLUPS on HoreKa) and the
  breadth of its physics models (130+ examples).
- Palabos is the most feature-rich open-source LBM code but was historically
  CPU-only; a GPU port appeared in 2025 (arXiv 2506.09242).
- Sailfish numbers are from 2014 hardware (GTX Titan); the project is no
  longer actively maintained.
- XLB (Autodesk, JAX-based) achieves 11,448 MLUPS on 8× A100 DGX
  (arXiv 2311.16080) — differentiable LBM, interesting for ML coupling.
- Kraken.jl is a single-developer Julia package targeting simplicity and
  GPU performance via KernelAbstractions.jl.

## Key references

- Wittmann et al. (2018). "LBM Benchmark Kernels as a Testbed for Performance Analysis". *Computers & Fluids*. [Code](https://github.com/RRZE-HPC/lbm-benchmark-kernels)
- Lehmann (2022). "Esoteric Pull and Esoteric Push: Two Simple In-Place Streaming Schemes for the LBM on GPUs". *Computation* 10(6), 92. DOI: 10.3390/computation10060092
- Bauer et al. (2021). "lbmpy: Automatic code generation for efficient parallel LBM". *J. Supercomputing*. DOI: 10.1016/j.cpc.2020.107746
- Latt et al. (2021). "Palabos: Parallel Lattice Boltzmann Solver". *Computers & Mathematics with Applications* 81, 334-350.
- Krause et al. (2021). "OpenLB — Open source lattice Boltzmann code". *Computers & Mathematics with Applications*.
- Januszewski & Kostur (2014). "Sailfish: A flexible multi-GPU implementation of the LBM". *CPC* 185(9).
- Ataei et al. (2024). "XLB: A Differentiable Massively Parallel LBM Library in Python". arXiv 2311.16080.
