# H0 (control): local copy of the coupled step, Kraken advect, BitMatrix mask, slice east BC.
include(joinpath(@__DIR__, "lib_kraken.jl"))
run_local_step_case("h0_local_verbatim", FAST, Val(:kraken), Val(:slice))
