# Face name aliases

Face names in `Boundary` directives depend on (a) the lattice and (b) the
active modules. This page is the authoritative table.

## Base 2D names (D2Q9)

| Face | Position | Comment |
|------|----------|---------|
| `west`  | `x = 0`  | |
| `east`  | `x = Lx` | |
| `south` | `y = 0`  | |
| `north` | `y = Ly` | |
| `front` / `back` | — | legacy aliases, kept for backward compatibility |

3D-only names (`top`, `bottom`) are **rejected** on D2Q9 by
`_validate_faces_vs_lattice`:

```julia
d2_faces = (:north, :south, :east, :west, :front, :back)
if setup.lattice == :D2Q9
    for b in setup.boundaries
        if !(b.face in d2_faces)
            throw(ArgumentError(
                "Boundary face ':$(b.face)' is not valid for D2Q9. " *
                "2D face names are: north/south/east/west."))
        end
    end
end
```

## Additional 3D names (D3Q19)

| Face | Position |
|------|----------|
| `bottom` | `z = 0`  |
| `top`    | `z = Lz` |

Plus all the 2D names: `west`/`east`/`south`/`north` still refer to the
x and y bounding planes.

## Axisymmetric aliases (`Module axisymmetric`)

When `Module axisymmetric` is active, the parser rewrites the first token
of every `Boundary` line using `_resolve_axisym_face` **before** the
generic face validator runs:

```julia
_resolve_axisym_face(face::AbstractString) =
    face == "z"    ? "x"     :
    face == "wall" ? "north" :
    face == "axis" ? "south" : String(face)
```

| User-facing alias | Mapped to | Physical meaning |
|-------------------|-----------|------------------|
| `z`    | `x`     | streamwise / axial direction — usually periodic |
| `wall` | `north` | outer radial wall at `r = R` (top of the 2D grid) |
| `axis` | `south` | axis of symmetry at `r = 0` (bottom of the 2D grid) |

This mapping only exists because an axisymmetric run is conceptually on an
`(r, z)` mesh but physically uses D2Q9 where `x` is streamwise and `y` is
radial. Writing `Boundary wall wall` is readable; writing
`Boundary north wall` requires the reader to remember the convention.

**Activation**: the alias is only applied if `is_axisym` is true, which is
set by the pre-scan over `Module` lines in `_parse_kraken_internal_single`:

```julia
for line in lines
    _first_word(line) == "Module" || continue
    push!(modules, _parse_module(line))
end
is_axisym = :axisymmetric in modules
```

**Without** `Module axisymmetric`, writing `Boundary wall ...` throws
`Unknown boundary face 'wall'` because `wall` is not in the allowed face
set.

## Summary table

| Name | D2Q9 plain | D3Q19 plain | D2Q9 + axisym |
|------|:----------:|:-----------:|:-------------:|
| `west`   | yes | yes | yes |
| `east`   | yes | yes | yes |
| `south`  | yes | yes | yes (mapped to by `axis`) |
| `north`  | yes | yes | yes (mapped to by `wall`) |
| `top`    | no  | yes | no |
| `bottom` | no  | yes | no |
| `front` / `back` | legacy | legacy | legacy |
| `z`      | no  | no  | yes → `x` |
| `wall`   | no  | no  | yes → `north` |
| `axis`   | no  | no  | yes → `south` |

## Axis shorthand

Independent of lattice, two extra forms exist via
`Boundary <axis> periodic`:

| Shorthand | Expands to |
|-----------|------------|
| `Boundary x periodic` | `Boundary west periodic` + `Boundary east periodic` |
| `Boundary y periodic` | `Boundary south periodic` + `Boundary north periodic` |

(`z periodic` is a future extension; use per-face for 3D periodic in z today.)

```julia
if axis == "x"
    return [BoundarySetup(:west, :periodic, ...),
            BoundarySetup(:east, :periodic, ...)]
elseif axis == "y"
    return [BoundarySetup(:south, :periodic, ...),
            BoundarySetup(:north, :periodic, ...)]
end
```

Source: `src/io/kraken_parser.jl`.
