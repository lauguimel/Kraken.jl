# H3: local copy with rusanov-only advect (no MUSCL). Bisection only, changes numerics.
include(joinpath(@__DIR__, "lib_kraken.jl"))
run_local_step_case("h3_pass1", FAST, Val(:pass1), Val(:slice))
