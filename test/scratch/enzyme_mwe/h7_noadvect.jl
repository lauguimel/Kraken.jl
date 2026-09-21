# round 2: coupled step with advection removed (locates the second crash site)
include(joinpath(@__DIR__, "lib_kraken.jl"))
run_local_step_case("h7_noadvect", FAST, Val(:none), Val(:slice))
