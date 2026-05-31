# Thermal Physics API

Thermal physics APIs provide the DDF temperature kernels, Boussinesq coupling, natural-convection drivers, and thermal refinement helpers for 2D and 3D workflows.

```@autodocs
Modules = [Kraken]
Pages = [
    "kernels/thermal_2d.jl",
    "kernels/thermal_3d.jl",
    "kernels/fused_thermal_2d.jl",
    "drivers/thermal.jl",
    "refinement/thermal_refinement.jl",
    "refinement/refinement_3d.jl",
]
Order = [:constant, :type, :function]
```
