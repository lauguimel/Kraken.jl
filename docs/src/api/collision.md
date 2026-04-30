# Collision operators

Collision kernels relax the distribution `f` toward local equilibrium. The
public collision model in this branch is BGK, with Guo forcing for body-force
terms and thermal/Boussinesq variants for the thermal DDF path.

MRT and axisymmetric collision kernels are not exported by this branch and are
not part of the v0.1.0 public API.

## Quick reference

| Symbol | Purpose |
|---|---|
| `collide_2d!` | Plain BGK collision, D2Q9 |
| `collide_3d!` | Plain BGK collision, D3Q19 |
| `collide_guo_2d!` | BGK with constant Guo body force, 2D |
| `collide_guo_field_2d!` | BGK with per-cell Guo force field, 2D |
| `collide_guo_3d!` | BGK with constant Guo body force, 3D |
| `collide_guo_field_3d!` | BGK with per-cell Guo force field, 3D |
| `collide_thermal_2d!` | Temperature DDF collision, 2D |
| `collide_thermal_3d!` | Temperature DDF collision, 3D |
| `collide_boussinesq_2d!` | Boussinesq force coupling, 2D |
| `collide_boussinesq_vt_2d!` | Boussinesq with temperature-dependent viscosity, 2D |
| `collide_boussinesq_vt_modified_2d!` | Modified Arrhenius Boussinesq path, 2D |
| `collide_boussinesq_3d!` | Boussinesq force coupling, 3D |

## Notes

- All public kernels are written with KernelAbstractions and can run on CPU or
  compatible GPU backends.
- `collide_boussinesq_vt_2d!` and
  `collide_boussinesq_vt_modified_2d!` are lower-level kernels. The usual
  user entry point is `run_natural_convection_2d(; Rc=...)`.
- The `.krk` runner uses the standard BGK or thermal paths. It does not
  dispatch MRT, axisymmetric or rheology kernels in this branch.

## Core signatures

```julia
collide_2d!(f, is_solid, omega; sync=true)
collide_3d!(f, is_solid, omega)

collide_guo_2d!(f, is_solid, omega, Fx, Fy)
collide_guo_3d!(f, is_solid, omega, Fx, Fy, Fz)

collide_thermal_2d!(g, ux, uy, omega_T)
collide_thermal_3d!(g, ux, uy, uz, omega_T)
```

See the [public API inventory](public_api.md) for the complete export list.
