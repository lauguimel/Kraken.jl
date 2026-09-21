# H4: local copy with the east BC vector built by an explicit loop instead of psi[Nx, :].
include(joinpath(@__DIR__, "lib_kraken.jl"))
run_local_step_case("h4_eastloop", FAST, Val(:kraken), Val(:loop))
