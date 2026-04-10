# Performance: MLUPs throughput

The standard metric for LBM codes is **Mega Lattice Updates Per Second**
(MLUPs):

```math
\text{MLUPs} = \frac{N_x \times N_y \times N_{\text{steps}}}{t_{\text{wall}} \times 10^6}
```

We benchmark Kraken's BGK D2Q9 lid-driven cavity solver across grid sizes
on CPU (Apple M2, single-thread) and GPU (NVIDIA H100 80 GB PCIe on the
QUT AQUA cluster). Hardware details are on the [Hardware](@ref) page.

## Results

| Grid (N×N) | CPU MLUPs | GPU MLUPs | GPU / CPU speedup |
|:-----------|----------:|----------:|------------------:|
| 64×64      |        34 |       134 |               3×  |
| 128×128    |        36 |       534 |              15×  |
| 256×256    |        40 |     2 090 |              52×  |
| 512×512    |        36 |     5 810 |             164×  |
| 1024×1024  |        32 |     7 675 |             236×  |

CPU throughput stays in the 26--44 MLUPs range regardless of grid size,
consistent with the memory-bandwidth limit of a single M2 core.
On the H100, throughput climbs steeply up to N = 512 as the GPU fills its
streaming multiprocessors, then plateaus around 7 700 MLUPs for N = 1024.

## Discussion

**Scaling regime.** The GPU advantage is negligible at N = 64 (kernel launch
overhead dominates) and exceeds two orders of magnitude at N = 1024.
For production runs the grid should contain at least ~100 k nodes to amortise
launch costs.

**Optimisation headroom.** The numbers above use the baseline BGK kernel with
`Float64` arrays and a standard pull-stream pattern.  Prior experiments with
the AA-pattern and `Float32` storage reached ~24 000 MLUPs on the same H100
hardware (a 3× improvement), but those optimisations are not yet merged into
the default driver. A dedicated benchmark run will be published once the
AA-pattern is validated on all test cases.

**Memory.** Each D2Q9 node stores 9 distribution values plus macroscopic
fields.  At `Float64`, the 1024×1024 benchmark allocates ~75 MB of
distribution arrays — well within the H100's 80 GB HBM3.

## Reproduce this benchmark

```bash
# CPU only (any machine)
julia --project benchmarks/run_all.jl --suite=perf --hardware-id=my_machine

# GPU (requires CUDA.jl)
julia --project benchmarks/run_all.jl --suite=perf --gpu --hardware-id=my_gpu
```

Results are written to `benchmarks/results/` as CSV files tagged with the
hardware ID and timestamp. Add your hardware to
[`benchmarks/hardware.toml`](https://github.com/lauguimel/Kraken.jl/blob/lbm/benchmarks/hardware.toml)
before running so the results are traceable.
