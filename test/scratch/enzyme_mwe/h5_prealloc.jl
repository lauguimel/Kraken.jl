# H5: local copy with the advect result array allocated by the caller.
include(joinpath(@__DIR__, "lib_kraken.jl"))
run_local_step_case("h5_prealloc", FAST, Val(:prealloc), Val(:slice))
