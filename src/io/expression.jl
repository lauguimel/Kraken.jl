# --- Sandboxed expression evaluation for .krk config files ---

"""
    KrakenExpr

A parsed and compiled math expression from a .krk config file.

# Fields
- `source::String`: original expression string.
- `func::Function`: compiled function `(; x, y, z, t, vars...) -> value`.
- `variables::Set{Symbol}`: variables referenced in the expression.
"""
struct KrakenExpr
    source::String
    func::Function
    variables::Set{Symbol}
end

# Whitelisted function calls allowed in expressions
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

# Built-in constants
const EXPR_CONSTANTS = Dict{Symbol, Float64}(
    :pi => Float64(π),
    :π  => Float64(π),
    :e  => Float64(ℯ),
    :Inf => Inf,
)

# Standard spatial/temporal variables available in all expressions
const EXPR_BUILTIN_VARS = Set{Symbol}([
    :x, :y, :z, :t,
    :Lx, :Ly, :Lz,
    :Nx, :Ny, :Nz,
    :dx, :dy, :dz,
])

"""
    validate_ast!(expr, user_var_names::Set{Symbol})

Walk the parsed AST and reject unsafe constructs. Throws `ArgumentError`
if the expression contains disallowed function calls, imports, or symbols.
"""
function validate_ast!(expr, allowed_vars::Set{Symbol})
    _validate_node!(expr, allowed_vars)
end

function _validate_node!(ex::Expr, allowed_vars::Set{Symbol})
    # Reject dangerous expression heads
    if ex.head in (:using, :import, :module, :export, :macrocall,
                    :struct, :mutable, :global, :local, :const)
        throw(ArgumentError("Disallowed expression type '$(ex.head)' in: $ex"))
    end

    # For function calls, check the function is whitelisted
    if ex.head == :call
        fn = ex.args[1]
        fn_sym = fn isa Symbol ? fn : fn isa Expr && fn.head == :. ? nothing : nothing
        if fn isa Symbol && !(fn in EXPR_WHITELIST)
            throw(ArgumentError("Function '$fn' is not allowed in expressions. " *
                "Allowed: $(join(sort(collect(EXPR_WHITELIST)), ", "))"))
        elseif fn isa Expr
            throw(ArgumentError("Qualified function call not allowed: $fn"))
        end
    end

    # Reject ccall, @eval, etc.
    if ex.head == :ccall
        throw(ArgumentError("ccall is not allowed in expressions"))
    end

    # Recurse into children
    for arg in ex.args
        _validate_node!(arg, allowed_vars)
    end
end

function _validate_node!(s::Symbol, allowed_vars::Set{Symbol})
    # Allow whitelisted functions (they appear as symbols in AST)
    s in EXPR_WHITELIST && return
    # Allow known constants
    s in keys(EXPR_CONSTANTS) && return
    # Allow user and builtin variables
    s in allowed_vars && return
    # Allow boolean literals
    s in (:true, :false) && return
    throw(ArgumentError("Unknown symbol '$s' in expression. " *
        "Available variables: $(join(sort(collect(allowed_vars)), ", "))"))
end

function _validate_node!(::Number, ::Set{Symbol})
    # Numbers are always safe
end

function _validate_node!(::AbstractString, ::Set{Symbol})
    # String literals (shouldn't appear but harmless)
end

function _validate_node!(::LineNumberNode, ::Set{Symbol}) end
function _validate_node!(::Nothing, ::Set{Symbol}) end
function _validate_node!(::Bool, ::Set{Symbol}) end
function _validate_node!(::QuoteNode, ::Set{Symbol}) end

"""
    collect_variables(expr) -> Set{Symbol}

Collect all variable symbols referenced in the expression
(excluding function names and constants).
"""
function collect_variables(expr)::Set{Symbol}
    vars = Set{Symbol}()
    _collect_vars!(expr, vars)
    return vars
end

function _collect_vars!(ex::Expr, vars::Set{Symbol})
    if ex.head == :call
        # Skip function name (first arg), collect rest
        for i in 2:length(ex.args)
            _collect_vars!(ex.args[i], vars)
        end
    elseif ex.head in (:comparison, :&&, :||)
        for arg in ex.args
            _collect_vars!(arg, vars)
        end
    else
        for arg in ex.args
            _collect_vars!(arg, vars)
        end
    end
end

function _collect_vars!(s::Symbol, vars::Set{Symbol})
    # Skip function names, constants, booleans
    s in EXPR_WHITELIST && return
    s in keys(EXPR_CONSTANTS) && return
    s in (:true, :false) && return
    push!(vars, s)
end

_collect_vars!(::Number, ::Set{Symbol}) = nothing
_collect_vars!(::AbstractString, ::Set{Symbol}) = nothing
_collect_vars!(::LineNumberNode, ::Set{Symbol}) = nothing
_collect_vars!(::Nothing, ::Set{Symbol}) = nothing
_collect_vars!(::Bool, ::Set{Symbol}) = nothing
_collect_vars!(::QuoteNode, ::Set{Symbol}) = nothing

"""
    parse_kraken_expr(source::String, user_vars::Dict{Symbol,Float64}=Dict{Symbol,Float64}()) -> KrakenExpr

Parse a math expression string from a .krk file, validate it against the
whitelist, and compile it to a Julia function.

The compiled function accepts keyword arguments for all builtin variables
(`x`, `y`, `z`, `t`, `Lx`, `Ly`, `Lz`, `Nx`, `Ny`, `Nz`, `dx`, `dy`, `dz`)
and any user-defined variables (`Define` in .krk).

# Example
```julia
expr = parse_kraken_expr("sin(2*pi*x/Lx) + U*y")
result = expr.func(x=0.5, y=1.0, Lx=1.0, U=0.1)
```
"""
function parse_kraken_expr(source::AbstractString,
                           user_vars::Dict{Symbol,Float64}=Dict{Symbol,Float64}())
    # Parse the string into a Julia AST
    parsed = Meta.parse(source)

    # Collect all user variable names
    user_var_names = Set{Symbol}(keys(user_vars))
    all_allowed = union(EXPR_BUILTIN_VARS, user_var_names)

    # Validate the AST
    validate_ast!(parsed, all_allowed)

    # Collect referenced variables
    all_vars = collect_variables(parsed)

    # Substitute constants and user variables with known values
    substituted = _substitute_constants(parsed, user_vars)

    # Build the list of free variables (those that need to be passed at eval time)
    free_vars = setdiff(all_vars, keys(EXPR_CONSTANTS), keys(user_vars))

    # Compile to a function with keyword arguments for free variables
    func = _compile_expr(substituted, free_vars)

    return KrakenExpr(source, func, all_vars)
end

"""
    _substitute_constants(expr, user_vars) -> Expr

Replace known constants (pi, e) and user-defined variables with their
numeric values in the AST.
"""
function _substitute_constants(ex::Expr, user_vars::Dict{Symbol,Float64})
    new_args = map(a -> _substitute_constants(a, user_vars), ex.args)
    # Don't substitute function names in :call expressions
    if ex.head == :call
        new_args[1] = ex.args[1]  # keep function symbol as-is
    end
    return Expr(ex.head, new_args...)
end

function _substitute_constants(s::Symbol, user_vars::Dict{Symbol,Float64})
    if haskey(EXPR_CONSTANTS, s)
        return EXPR_CONSTANTS[s]
    elseif haskey(user_vars, s)
        return user_vars[s]
    else
        return s
    end
end

_substitute_constants(x, ::Dict{Symbol,Float64}) = x

"""
    _compile_expr(expr, free_vars) -> Function

Compile the substituted AST into a function taking keyword arguments
for the remaining free variables.
"""
function _compile_expr(expr, free_vars::Set{Symbol})
    # Sort for deterministic ordering
    sorted_vars = sort(collect(free_vars))

    # Build keyword argument list with default value 0.0
    kw_args = [Expr(:kw, v, 0.0) for v in sorted_vars]

    # Build function: (;x=0.0, y=0.0, ...) -> expr
    if isempty(kw_args)
        # No free variables — return constant function
        val = Core.eval(Module(), expr)
        return (; kwargs...) -> val
    end

    # Use a fresh module for sandboxed evaluation
    sandbox = Module()

    # Import whitelisted math functions into the sandbox
    Core.eval(sandbox, :(using Base: sin, cos, tan, asin, acos, atan,
                                     sinh, cosh, tanh,
                                     exp, log, log2, log10,
                                     sqrt, cbrt, abs, sign, floor, ceil, round,
                                     min, max, clamp, ifelse,
                                     mod, rem, div, one, zero, float))

    # Build the function expression with kwargs... to accept extra keyword arguments
    # This allows callers to pass all variables without error
    splat = Expr(:..., :_extra_kwargs_)
    func_expr = Expr(:function,
        Expr(:tuple, Expr(:parameters, kw_args..., splat)),
        expr)

    return Core.eval(sandbox, func_expr)
end

"""
    has_variable(ke::KrakenExpr, sym::Symbol) -> Bool

Check if the expression references a specific variable.
"""
has_variable(ke::KrakenExpr, sym::Symbol) = sym in ke.variables

"""
    is_time_dependent(ke::KrakenExpr) -> Bool

Check if the expression depends on `t` (time).
"""
is_time_dependent(ke::KrakenExpr) = has_variable(ke, :t)

"""
    is_spatial(ke::KrakenExpr) -> Bool

Check if the expression depends on `x`, `y`, or `z`.
"""
is_spatial(ke::KrakenExpr) = has_variable(ke, :x) || has_variable(ke, :y) || has_variable(ke, :z)

"""
    evaluate(ke::KrakenExpr; kwargs...) -> Float64

Evaluate the expression with the given variable values.

# Example
```julia
expr = parse_kraken_expr("sin(x) + y")
evaluate(expr; x=1.0, y=2.0)  # ≈ 2.8414709848
```
"""
evaluate(ke::KrakenExpr; kwargs...) = Base.invokelatest(ke.func; kwargs...)
