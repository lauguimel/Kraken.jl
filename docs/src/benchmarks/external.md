# External comparisons

This page summarises how Kraken.jl compares to other LBM and CFD codes in
terms of performance and accuracy.

!!! note
    Direct performance comparisons across codes are inherently approximate:
    different hardware, compiler versions, data layouts, and collision
    operators all affect the numbers. The data below is meant to provide
    **context**, not a strict ranking.

## Literature comparison table

A detailed table comparing Kraken.jl MLUPs against published results from
Palabos, OpenLB, waLBerla, TCLB, and Sailfish is maintained in
[`benchmarks/external/comparisons.md`](https://github.com/lauguimel/Kraken.jl/blob/lbm/benchmarks/external/comparisons.md).

!!! warning "Work in progress"
    The external comparisons file is being assembled as part of Phase 5.4.
    Once available, a summary table will be inlined here.

## Key takeaways

1. **GPU throughput is competitive.** Kraken.jl reaches ~7 700 MLUPs (BGK
   D2Q9) on a single H100 with the baseline kernel, and ~24 000 MLUPs with
   the AA-pattern + Float32 variant. Published single-GPU numbers for mature
   C/C++ codes (Sailfish, TCLB) on comparable hardware range from
   5 000–30 000 MLUPs depending on collision model and precision.

2. **Code size.** Kraken.jl achieves these numbers in roughly 5 000 lines of
   Julia (solver core), compared to 50 000+ lines typical of C++ LBM
   frameworks. Julia's multiple dispatch and KernelAbstractions.jl remove
   the need for template metaprogramming or code generation.

3. **Accuracy.** On the canonical test cases (Poiseuille, Taylor-Green,
   lid-driven cavity), Kraken produces errors identical to or smaller than
   those reported for OpenFOAM's `icoFoam` at the same mesh resolution.
   Both codes are second-order; differences come from the discretisation
   details (LBM vs finite-volume) rather than from implementation bugs.

## OpenFOAM cross-validation

For single-phase laminar flows, Kraken results have been compared against
OpenFOAM 11 (`icoFoam`, second-order central, PISO) on matching structured
meshes. Both codes converge at ``\mathcal{O}(\Delta x^2)`` with Kraken
showing marginally smaller ``L_2`` errors on regular grids — a known
property of BGK on uniform lattices.

See the [Accuracy](accuracy.md) page for convergence tables.
