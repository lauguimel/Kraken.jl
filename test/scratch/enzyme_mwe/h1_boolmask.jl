# H1: local copy with Matrix{Bool} mask everywhere the mask is passed explicitly.
include(joinpath(@__DIR__, "lib_kraken.jl"))
run_local_step_case("h1_boolmask", FAST, Val(:kraken), Val(:slice); mask=Matrix{Bool})
