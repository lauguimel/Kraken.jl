# CUDA-Graph executor for the backend-parametric SIMPLE cavity solver (M-A2).
#
# DO NOT LOAD THIS ON A CPU-ONLY MACHINE. It requires `using CUDA` to be in
# scope BEFORE it is included (the Aqua GPU job's project, not the main
# Project.toml) — exactly like src/solve/linear_solve_cuda.jl. The KA-only
# solver in cavity_mg.jl stays CUDA-free; this file only adds an opt-in
# OFF-STRIDE iteration executor that records the whole iteration as a CUDA
# graph once and replays the instantiated executable thereafter.
#
# WHY A GRAPH
# -----------
# With `mg_cycles > 0` the off-stride SIMPLE iteration is a STATIC sequence of
# ~hundreds of small kernel launches (dominated by the MG coarse-level RBGS
# sweeps, ncoarse=50 -> 100 launches per V-cycle at the coarsest level) with
# ZERO host syncs and — with `static_gauge=true` — ZERO device allocations.
# That is precisely the CUDA-graph sweet spot: one `cuGraphLaunch` replaces the
# entire per-iteration launch stream, removing the per-launch CPU cost that
# dominates this latency-bound solve on the A100.
#
# WHAT IS CAPTURED / HOW IT INTERACTS WITH norm_stride
# ----------------------------------------------------
#   * captured  : one full OFF-STRIDE outer iteration, i.e.
#                 `_cavity_mg_offstride_step!(S)` = phases 1-4 (fused physics
#                 kernels + 2 momentum MG solves + pressure MG solve + static
#                 gauge + corrections). Mixed-precision inner solves
#                 (`mg_mixed_precision` / `mom_mg_mixed_precision`) are also a
#                 static launch sequence and capture transparently.
#   * replayed  : every subsequent off-stride iteration is ONE
#                 `CUDA.launch(exec)` on the current stream.
#   * uncaptured: every ON-STRIDE (norm-check) iteration runs the regular
#                 phase-by-phase path inside the solver loop (it needs host
#                 reductions, which cannot live inside a capture). Graph
#                 launches and regular launches are issued on the SAME
#                 task-local stream, so ordering is automatic.
#
# REQUIREMENTS / GOTCHAS (checked or documented below)
# ----------------------------------------------------
#   * fixed inner cycles (`mg_cycles > 0`): the tolerance path has per-cycle
#     host reductions — not capturable. Enforced.
#   * `static_gauge=true`: the library `sum!` gauge reduction may allocate a
#     GPU temporary, and "graph capture does not support asynchronous memory
#     operations" (CUDA.jl). Forced ON by the wrapper.
#   * allocation-stable buffers: all per-iteration device buffers live in the
#     state NamedTuple `S` and the shared MG hierarchies, allocated once before
#     the loop (see cavity_mg.jl). The executor ASSERTS the buffers it first
#     saw are the ones it keeps seeing (graph kernel arguments are raw device
#     pointers — a reallocation would silently corrupt the replay).
#   * `CUDA.capture` RECORDS the work without EXECUTING it: after a successful
#     capture the executor must `launch` the instantiated graph once so the
#     capture call still advances the state by exactly one iteration.
#   * first-touch JIT: kernels compiled DURING capture invalidate it, so the
#     executor runs `warmup` uncaptured iterations first (>= 1; default 2) and,
#     mirroring CUDA.@captured, retries once after an uncaptured run if the
#     first capture attempt still fails. If the retry fails too it degrades
#     permanently to plain per-launch execution (correctness unchanged).
#   * inner-cycle counters: replayed iterations re-run only the recorded DEVICE
#     work, so `S.counters` (host-side) only reflects uncaptured iterations.
#     Fixed-cycles mode makes the true totals iteration-proportional anyway.
#   * world age: this file is `include`d at top level in the bench script AFTER
#     `using CUDA` and after cavity_mg.jl; the executor reaches the (possibly
#     freshly included) solver methods via `Base.invokelatest`. Call the
#     wrapper itself via `Base.invokelatest` if the bench script calls it from
#     the same top-level expression that did the includes.
#
# Usage from the Aqua bench driver:
#   using CUDA
#   include("src/methods/inc_ns/cavity_mg.jl")        # KA solver (CUDA-free)
#   include("src/methods/inc_ns/cavity_mg_cuda.jl")   # this file
#   res = Base.invokelatest(solve_incns_cavity_mg_cuda_graph;
#                           nx=512, ny=512, Re=1000.0,
#                           backend_ka=CUDABackend(), atype=CuArray{Float64},
#                           mg_cycles=3, mom_mg_cycles=1)

using CUDA

# The KA solver (state NamedTuple + `_cavity_mg_offstride_step!` + the
# `offstride_executor` seam) must be in scope; pull it in if not.
if !isdefined(@__MODULE__, :solve_incns_cavity_mg)
    include(joinpath(@__DIR__, "cavity_mg.jl"))
end

# Fail LOUDLY at include time if the CUDA.jl graph API moved (names verified
# against CUDA.jl lib/cudadrv/graph.jl: `capture(f; flags, throw_error)`,
# `instantiate(graph)`, `launch(exec, [stream])`).
if !(isdefined(CUDA, :capture) && isdefined(CUDA, :instantiate) &&
     isdefined(CUDA, :launch))
    error("cavity_mg_cuda.jl: this CUDA.jl version does not export the " *
          "graph API (capture/instantiate/launch) — check CUDA.jl >= 3.5 " *
          "and the current names in CUDA.jl lib/cudadrv/graph.jl.")
end

"""
    CavityCudaGraphExecutor(; warmup=2)

Callable off-stride iteration executor for `solve_incns_cavity_mg`'s
`offstride_executor` seam. Lifecycle per call (one call = one OFF-STRIDE
SIMPLE iteration):

  calls 1..warmup   run `_cavity_mg_offstride_step!` uncaptured (JIT-compiles
                    every kernel so the capture below is compilation-free)
  call warmup+1     `CUDA.capture` the step (records WITHOUT executing),
                    `instantiate`, then `launch` once (performs this call's
                    iteration); on capture failure: execute uncaptured, retry
                    the capture once, and degrade to `fallback` mode if it
                    still fails
  later calls       `CUDA.launch(exec)` — one graph launch per iteration

Fields `ncalls` / `graph_launches` / `fallback` expose what actually happened
for reporting; `buffer_witness` pins the first-seen state buffer to assert
allocation stability across iterations.
"""
mutable struct CavityCudaGraphExecutor
    exec::Any                 # CUDA.CuGraphExec once instantiated (Any: keep
                              # the struct definition robust to API renames)
    warmup::Int
    ncalls::Int
    graph_launches::Int
    fallback::Bool
    buffer_witness::Any       # S.u of the first call (allocation-stability assert)
end

CavityCudaGraphExecutor(; warmup::Integer=2) =
    CavityCudaGraphExecutor(nothing, max(Int(warmup), 1), 0, 0, false, nothing)

function _cavity_graph_assert_stable!(ge::CavityCudaGraphExecutor, S)
    if ge.buffer_witness === nothing
        ge.buffer_witness = S.u
    elseif !(ge.buffer_witness === S.u)
        error("CavityCudaGraphExecutor: the solver state buffers changed " *
              "identity across iterations — the captured graph holds stale " *
              "device pointers. This violates the allocation-stability " *
              "contract of the fixed-cycles fast path.")
    end
    return nothing
end

function _cavity_graph_capture!(ge::CavityCudaGraphExecutor, S)
    # Attempt 1. `throw_error=false` returns `nothing` on a benign capture
    # invalidation (e.g. an operation that cannot be captured); harder errors
    # still throw and are degraded to fallback mode.
    graph = try
        CUDA.capture(; throw_error=false) do
            Base.invokelatest(_cavity_mg_offstride_step!, S)
        end
    catch err
        @warn "cavity CUDA-graph capture threw; falling back to per-launch execution" exception=(err, catch_backtrace())
        ge.fallback = true
        nothing
    end

    if graph === nothing && !ge.fallback
        # Mirror CUDA.@captured: the failure was likely first-touch work
        # (JIT/module loading). Execute uncaptured once, then retry.
        Base.invokelatest(_cavity_mg_offstride_step!, S)
        graph = try
            CUDA.capture(; throw_error=false) do
                Base.invokelatest(_cavity_mg_offstride_step!, S)
            end
        catch err
            @warn "cavity CUDA-graph capture retry threw; falling back" exception=(err, catch_backtrace())
            nothing
        end
        if graph === nothing
            ge.fallback = true
            return nothing
        end
        # The retry capture RECORDED one iteration without executing it, and
        # the uncaptured run above already advanced this call's iteration:
        # instantiate but do NOT launch now.
        ge.exec = CUDA.instantiate(graph)
        return nothing
    elseif graph === nothing
        # fallback already set; this call's iteration still has to happen.
        Base.invokelatest(_cavity_mg_offstride_step!, S)
        return nothing
    end

    # First-try success: nothing has executed yet for this call. Instantiate
    # and launch once so the state advances by exactly one iteration.
    ge.exec = CUDA.instantiate(graph)
    CUDA.launch(ge.exec)
    ge.graph_launches += 1
    return nothing
end

function (ge::CavityCudaGraphExecutor)(S)
    ge.ncalls += 1
    _cavity_graph_assert_stable!(ge, S)
    if ge.fallback || ge.ncalls <= ge.warmup
        Base.invokelatest(_cavity_mg_offstride_step!, S)
        return nothing
    end
    if ge.exec === nothing
        _cavity_graph_capture!(ge, S)
        return nothing
    end
    CUDA.launch(ge.exec)
    ge.graph_launches += 1
    return nothing
end

"""
    solve_incns_cavity_mg_cuda_graph(; warmup=2, kwargs...) -> NamedTuple

Opt-in CUDA-graph front-end for `solve_incns_cavity_mg`: identical numerics
and keywords, but every OFF-STRIDE outer iteration is replayed from a captured
CUDA graph (one launch per iteration) instead of being re-launched kernel by
kernel. Requires `backend_ka=CUDABackend()`, a `CuArray` `atype`, and the
fixed-cycles fast path (`mg_cycles > 0`); forces `static_gauge=true` (the
allocation-free gauge reduction — last-ulp difference vs `sum!` only).

Returns the solver's NamedTuple augmented with `graph_captured` (whether a
graph executed at least once) and `graph_launches` (replay count). On any
capture failure it degrades to plain execution and reports
`graph_captured=false` — results are unaffected.
"""
function solve_incns_cavity_mg_cuda_graph(; warmup::Integer=2,
                                          mg_cycles::Integer=3,
                                          mom_mg_cycles::Integer=1,
                                          kwargs...)
    Int(mg_cycles) > 0 ||
        throw(ArgumentError("the CUDA-graph path requires fixed inner MG " *
                            "cycles (mg_cycles > 0): the tolerance-driven " *
                            "path performs host reductions inside the " *
                            "iteration, which cannot be stream-captured"))
    bk = get(kwargs, :backend_ka, nothing)
    bk isa CUDA.CUDABackend ||
        throw(ArgumentError("pass backend_ka=CUDABackend() (got $(typeof(bk))) — " *
                            "the graph executor records CUDA stream work"))
    at = get(kwargs, :atype, nothing)
    (at isa Type && at <: CuArray) ||
        throw(ArgumentError("pass atype=CuArray{Float64} (got $at)"))

    ge = CavityCudaGraphExecutor(; warmup=warmup)
    # kwargs FIRST so the graph-critical keywords below win over any
    # caller-supplied duplicates.
    res = Base.invokelatest(solve_incns_cavity_mg;
                            kwargs..., mg_cycles=Int(mg_cycles),
                            mom_mg_cycles=Int(mom_mg_cycles),
                            static_gauge=true, offstride_executor=ge)
    return (; res..., graph_captured=(ge.graph_launches > 0),
            graph_launches=ge.graph_launches, graph_fallback=ge.fallback)
end
