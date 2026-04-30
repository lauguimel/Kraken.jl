# Hardware

Every benchmark CSV should carry a `hardware_id` that maps to
`benchmarks/hardware.toml`.

## Declared machines

### `apple_m3max`

| Property | Value |
|---|---|
| Kind | GPU |
| Model | Apple M3 Max |
| Backend | Metal |
| Notes | Local development artifact; not the public headline benchmark |

### `aqua_h100`

| Property | Value |
|---|---|
| Kind | GPU |
| Model | NVIDIA H100 80 GB PCIe |
| Host | QUT AQUA cluster |
| Scheduler | PBS Pro |
| Notes | Preferred GPU target for public benchmark reruns |

### `aqua_a100`

| Property | Value |
|---|---|
| Kind | GPU |
| Model | NVIDIA A100 |
| Host | QUT AQUA cluster |
| Notes | Secondary comparison target |

## Legacy artifacts

Some existing CSVs still carry the `apple_m2` hardware id. Keep them as raw
historical data if needed, but do not use Apple laptop labels as the public
performance story. The recommended public comparison is CPU baseline plus
H100, with committed CSV provenance.

## Adding hardware

Add a new table to `benchmarks/hardware.toml`, then run benchmarks with the
matching id:

```bash
julia --project=. benchmarks/run_all.jl --hardware-id=my_machine
```

Commit the generated CSV together with any documentation update that cites it.
