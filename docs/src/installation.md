# Installation

## Prerequisites

### Julia

Install Julia via [juliaup](https://github.com/JuliaLang/juliaup) (recommended):

```bash
# macOS / Linux
curl -fsSL https://install.julialang.org | sh

# Windows
winget install julia -s msstore
```

Kraken.jl requires **Julia 1.10** or later.

## Installing Kraken.jl

```julia
using Pkg
Pkg.add("Kraken")
```

Or from the Pkg REPL (press `]`):

```
pkg> add Kraken
```

## GPU Setup

### Metal (Apple Silicon — Mac M1/M2/M3/M4)

Metal support is included by default. No additional setup is required on macOS with Apple Silicon.

Verify Metal is available:

```julia
using Metal
Metal.versioninfo()
```

### CUDA (NVIDIA GPUs)

Install the CUDA toolkit and the Julia CUDA package:

```julia
using Pkg
Pkg.add("CUDA")
```

CUDA.jl will automatically download the appropriate CUDA toolkit. Requires compute capability ≥ 7.5.

Verify CUDA is available:

```julia
using CUDA
CUDA.versioninfo()
```

## Verifying Installation

```julia
using Kraken

# Check the available compute backends (CPU + any detected GPU)
available_backends()

# Run a tiny lid-driven cavity as a smoke test
config = LBMConfig(D2Q9(); Nx=32, Ny=32, ν=0.1, u_lid=0.1, max_steps=100)
run_cavity_2d(config)
```

If the cavity simulation completes without errors, Kraken.jl is correctly
installed. From here, head to [Getting started](getting_started.md) for your
first real simulation, or to the [`.krk` overview](krk/overview.md) to describe
a run declaratively.
