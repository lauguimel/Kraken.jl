# Streaming

Streaming propagates post-collision distributions to neighboring lattice
nodes. The public streaming API in this branch covers the standard 2D/3D
paths plus the periodic variants used by the canonical examples.

Axisymmetric streaming functions are not exported by this branch.

## Quick reference

| Symbol | Purpose |
|---|---|
| `stream_2d!` | Standard D2Q9 streaming |
| `stream_3d!` | Standard D3Q19 streaming |
| `stream_periodic_x_wall_y_2d!` | Periodic in x, wall in y; used by channel flows |
| `stream_fully_periodic_2d!` | Periodic in both 2D directions; used by Taylor-Green |

## Core signatures

```julia
stream_2d!(f_out, f_in, Nx, Ny; sync=true)
stream_3d!(f_out, f_in, Nx, Ny, Nz)
stream_periodic_x_wall_y_2d!(f_out, f_in, Nx, Ny)
stream_fully_periodic_2d!(f_out, f_in, Nx, Ny)
```

The `.krk` runner chooses the streaming path from the parsed boundaries:
fully periodic cases use the periodic kernel, channel-style periodic-x cases
use `stream_periodic_x_wall_y_2d!`, and closed-box cases use the standard
wall-aware path.
