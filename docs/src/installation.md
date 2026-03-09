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

# Check available backends
available_backends()

# Run a quick test
run_cavity(; N=32, Re=100.0, dt=0.001, nsteps=100)
```

If the cavity simulation completes without errors, Kraken.jl is correctly installed.
