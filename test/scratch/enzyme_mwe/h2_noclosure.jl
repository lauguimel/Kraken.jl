# H2: local copy with the closure-free advect.
include(joinpath(@__DIR__, "lib_kraken.jl"))
run_local_step_case("h2_noclosure", FAST, Val(:noclosure), Val(:slice))
