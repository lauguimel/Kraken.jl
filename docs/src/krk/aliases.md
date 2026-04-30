# Face name aliases

Face names in `Boundary` directives depend on the lattice dimensionality.
This page documents the public names for this branch and the parser-only
aliases that should not be treated as supported runtime features.

## Base 2D names (D2Q9)

| Face | Position |
|---|---|
| `west` | `x = 0` |
| `east` | `x = Lx` |
| `south` | `y = 0` |
| `north` | `y = Ly` |
| `front` / `back` | legacy aliases |

3D-only names such as `top` and `bottom` are rejected on D2Q9 by
`_validate_faces_vs_lattice`.

## Additional 3D names (D3Q19)

| Face | Position |
|---|---|
| `bottom` | `z = 0` |
| `top` | `z = Lz` |

The 2D names `west`, `east`, `south` and `north` still refer to the x/y
bounding planes in 3D.

## Periodic axis shorthand

The parser supports:

| Shorthand | Expands to |
|---|---|
| `Boundary x periodic` | `west` + `east` |
| `Boundary y periodic` | `south` + `north` |

Use explicit face declarations for 3D z-periodicity in this branch.

## Parser-only axisymmetric aliases

When `Module axisymmetric` is present, the parser can rewrite:

| Alias | Mapped to |
|---|---|
| `z` | `x` |
| `wall` | `north` |
| `axis` | `south` |

However, `Module axisymmetric` is not a public runtime module in this branch:
`run_simulation` rejects it. These aliases are documented only so old or
development `.krk` files fail in an understandable way.

Source: `src/io/kraken_parser.jl`.
