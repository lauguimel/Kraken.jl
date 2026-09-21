# Issue #41 probe, round 2: shrink the standalone reproduction (s0 = rusanov
# branch + MUSCL branch in one loop) and test the split-loop workaround inside
# the coupled step. See enzyme_mwe/harness.jl.
include(joinpath(@__DIR__, "enzyme_mwe", "harness.jl"))
run_matrix([
    "t1_nopass2.jl", "t2_indexband.jl", "t3_norus.jl", "t4_nomuscl.jl",
    "t5_split.jl", "t6_nosolid.jl", "t7_small.jl", "t8_ifelse_to_if.jl",
    "t9_s0_nocb.jl",
    "h6_split.jl", "h7_noadvect.jl",
]; label="enzyme_mwe2")
