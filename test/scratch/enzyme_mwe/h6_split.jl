# round 2: coupled step with the split-loop advect (workaround candidate, bit-identical primal)
include(joinpath(@__DIR__, "lib_kraken.jl"))
run_local_step_case("h6_split", FAST, Val(:split), Val(:slice))
