# AMR-D Complex Flow Debug and Backend Benchmark

Date: 2026-05-06

## Scope

This patch adds the reproducible local checks that should be inspected before
launching the Aqua H100 comparison.

- `benchmarks/amr_d_complex_flow_debug_plots_2d.jl` generates field, mesh and
  profile PNGs for BFS, square obstacle and cylinder obstacle.
- `benchmarks/amr_d_backend_complex_benchmark_2d.jl` writes a CSV with:
  AMR-D route-native CPU rows, leaf Cartesian CPU reference rows, and backend
  Cartesian rows for periodic-x wall-y solid flows.
- `hpc/amr_d_backend_complex_h100_aqua.pbs` runs the same benchmark on Aqua H100.

## Current GPU Boundary

The current D branch has CPU route-native AMR-D for BFS, square and cylinder.
It also has a GPU-ready route pack, but not yet production route-native GPU
kernels for the AMR schedule. Therefore the benchmark is intentionally honest:

- `amr_d_route_native_cpu`: refined D path, CPU only.
- `leaf_cartesian_reference_cpu`: current leaf-equivalent Cartesian reference,
  CPU.
- `leaf_cartesian_backend_periodic_solid`: KernelAbstractions backend kernel
  for square and cylinder on CPU, Metal or CUDA.
- BFS backend row is marked `unsupported_open_boundary_gpu_parity` until the
  Zou-He/open-channel GPU path is made equivalent to the CPU reference.

This means the first Aqua pass measures H100 vs CPU for the Cartesian reference
and keeps the refined AMR-D row as the CPU baseline. A later patch must add
device route kernels before claiming H100 AMR-D speedup.

## Local Commands

```bash
KRK_AMR_D_DEBUG_STEPS=240 \
  julia --project=. benchmarks/amr_d_complex_flow_debug_plots_2d.jl
```

Default plot output:

```text
benchmarks/results/figures/amr_d_complex_flow_debug_2d/
```

```bash
KRK_AMR_D_BACKEND=cpu KRK_AMR_D_BENCH_STEPS=240 \
  julia --project=. benchmarks/amr_d_backend_complex_benchmark_2d.jl
```

On Apple Metal, use Float32 unless a specific kernel has been validated in
Float64:

```bash
KRK_AMR_D_BACKEND=metal KRK_AMR_D_BENCH_T=float32 KRK_AMR_D_BENCH_STEPS=240 \
  julia --project=. benchmarks/amr_d_backend_complex_benchmark_2d.jl
```

## Aqua Command

After local plots have been inspected:

```bash
qsub hpc/amr_d_backend_complex_h100_aqua.pbs
```

The output CSV is:

```text
benchmarks/results/amr_d_backend_complex_benchmark_2d_<tag>.csv
```

## Next GPU Patch

To turn this into full AMR-D GPU, implement the route-native device path in this
order:

1. Transfer `ConservativeTreeGPURoutePack2D` arrays to device memory.
2. Store composite AMR populations in one packed cell-major array.
3. Launch direct-route streaming for one L/L+1 interface without boundaries.
4. Add periodic/wall solid boundary routes.
5. Add open-channel Zou-He routes for BFS.
6. Compare device route output against CPU route output on one step, then on
   10, 100 and production-step runs.
