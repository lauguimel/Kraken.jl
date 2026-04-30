# Performance

The standard throughput metric for LBM codes is **Mega Lattice Updates Per
Second** (MLUPs):

```math
\text{MLUPs} = \frac{N_x \times N_y \times N_{\text{steps}}}{t_{\text{wall}} \times 10^6}
```

This page only publishes numbers that are traceable to files in
`benchmarks/results/`. Older headline H100 throughput numbers were removed
from this branch's docs because the matching CSV artifact is not present.

## Current traceable throughput artifacts

| File | Hardware id | Backend | Precision | Status |
|---|---|---|---|---|
| `perf_mlups_cpu_apple_m2_20260410_115127.csv` | `apple_m2` | CPU | Float64 | legacy local CPU baseline |
| `perf_mlups_metal_apple_m3max_final.csv` | `apple_m3max` | Metal | Float32 | local artifact, not headline benchmark |

The public benchmark narrative should be CPU baseline plus H100 once a fresh
H100 throughput CSV is committed. Until then, H100 is only used here for the
Poiseuille convergence artifact, not for MLUPS claims.

## CPU baseline artifact

| Grid | MLUPs |
|---:|---:|
| 64 x 64 | 31.896 |
| 128 x 128 | 31.854 |
| 256 x 256 | 26.542 |

Source: `benchmarks/results/perf_mlups_cpu_apple_m2_20260410_115127.csv`.

## Local Metal artifact

| Grid | MLUPs |
|---:|---:|
| 64 x 64 | 6.8 |
| 128 x 128 | 20.7 |
| 256 x 256 | 81.8 |
| 512 x 512 | 257.3 |
| 1024 x 1024 | 572.7 |
| 2048 x 2048 | 838.4 |
| 4096 x 4096 | 920.5 |

Source: `benchmarks/results/perf_mlups_metal_apple_m3max_final.csv`.

This table is useful for development tracking but should not be presented as
the main performance story.

## Reproduce

```bash
# CPU only
julia --project=. benchmarks/perf_mlups.jl

# Full benchmark harness, tagged with a hardware id
julia --project=. benchmarks/run_all.jl --suite=perf --hardware-id=my_machine
```

Before publishing a new number, add the hardware entry to
`benchmarks/hardware.toml` and commit the CSV result with the docs update.
