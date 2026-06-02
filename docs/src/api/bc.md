# Boundary Condition API

Boundary-condition APIs include direct Zou-He and bounce-back kernels, spatially varying inlet/outlet wrappers, modular rebuild specs, and cut-link precomputation for interpolated bounce-back boundaries.

```@autodocs
Modules = [Kraken]
Pages = [
    "kernels/boundary_2d.jl",
    "kernels/boundary_3d.jl",
    "kernels/boundary_rebuild.jl",
    "kernels/boundary_spatial_2d.jl",
    "kernels/li_bb_2d.jl",
    "kernels/li_bb_2d_v2.jl",
    "kernels/li_bb_3d_v2.jl",
]
Order = [:constant, :type, :function]
```
