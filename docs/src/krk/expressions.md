# Expressions — the KrakenExpr grammar

Many directives accept scalar expressions instead of bare numbers: body
forces, boundary-condition values, initial fields, geometry conditions,
domain sizes. These expressions are parsed by `parse_kraken_expr` in
`src/io/expression.jl`, validated against a whitelist, and compiled into a
sandboxed Julia anonymous function.

## Variables

Built-in variables (always available):

```julia
const EXPR_BUILTIN_VARS = Set{Symbol}([
    :x, :y, :z, :t,
    :Lx, :Ly, :Lz,
    :Nx, :Ny, :Nz,
    :dx, :dy, :dz,
])
```

- `x, y, z` — spatial coordinates (physical units).
- `t` — time (for time-dependent BCs).
- `Lx, Ly, Lz` — domain extents.
- `Nx, Ny, Nz` — grid resolution.
- `dx, dy, dz` — cell sizes.

User variables declared with `Define` are substituted at parse time:

```text
Define Re = 1000
Define U  = 0.1
Physics nu = U * 1.0 / Re           # = 1e-4, computed at parse time
```

## Constants

```julia
const EXPR_CONSTANTS = Dict{Symbol, Float64}(
    :pi => π, :π => π,
    :e  => ℯ,
    :Inf => Inf,
)
```

## Operators

Arithmetic: `+`, `-`, `*`, `/`, `^`, `mod`, `rem`, `div`

Comparison: `>`, `<`, `>=`, `<=`, `==`, `!=`

Logical: `!`, `&`, `|`

Grouping via parentheses.

## Built-in functions (whitelist)

From `src/io/expression.jl`:

```julia
const EXPR_WHITELIST = Set{Symbol}([
    :+, :-, :*, :/, :^, :mod, :rem, :div,
    :sin, :cos, :tan, :asin, :acos, :atan,
    :sinh, :cosh, :tanh,
    :exp, :log, :log2, :log10,
    :sqrt, :cbrt, :abs, :sign, :floor, :ceil, :round,
    :min, :max, :clamp, :ifelse,
    :>, :<, :>=, :<=, :(==), :(!=), :!, :(&), :(|),
    :one, :zero, :float,
])
```

Any call to a non-whitelisted function (`rand`, `randn`, `eval`, any
`Base.*`, …) throws at parse time:

```
ArgumentError: Function 'rand' is not allowed in expressions.
```

## Examples

### Parabolic Poiseuille profile

```text
Boundary west velocity(
    ux = 0.1 * (1 - ((y - Ly/2) / (Ly/2))^2),
    uy = 0
)
```

### Logarithmic ABL profile

```text
Define u_star = 0.02
Define z0     = 0.001
Boundary west velocity(
    ux = (u_star / 0.41) * log((y + z0) / z0),
    uy = 0
)
```

### Time-pulsed inlet

```text
Define Uavg = 0.05
Boundary west velocity(
    ux = Uavg * (1 + 0.1*sin(2*pi*t / 500)),
    uy = 0
)
```

### Geometry condition (obstacle)

```text
Obstacle disk wall { (x - 0.5)^2 + (y - 0.5)^2 < 0.01 }
Obstacle wedge    { (x > 0.4) & (x < 0.6) & (y < 0.2*(x - 0.4)) }
```

### Taylor–Green initial condition

```text
Initial {
    ux = 0.05 * sin(2*pi*x) * cos(2*pi*y)
    uy = -0.05 * cos(2*pi*x) * sin(2*pi*y)
}
```

## Validation

Expressions are validated by walking the parsed AST (`validate_ast!`) and
rejecting disallowed heads (`:using`, `:import`, `:module`, `:export`,
`:macrocall`, `:struct`, `:global`, `:const`, …) and non-whitelisted calls.
`ccall` is explicitly banned.

## Programmatic usage

```julia
expr = parse_kraken_expr("sin(2*pi*x/Lx) + U*y", Dict(:U => 0.1))
evaluate(expr; x=0.5, y=1.0, Lx=1.0)  # returns a Float64
```
