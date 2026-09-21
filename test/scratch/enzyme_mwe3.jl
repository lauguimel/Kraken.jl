# Issue #41 probe, round 3: the reductions found on macOS (wrong or NaN reverse
# gradient), on Linux; and the suite's reverse call without bounds checks
# (the production setting). See enzyme_mwe/harness.jl.
include(joinpath(@__DIR__, "enzyme_mwe", "harness.jl"))
run_matrix([
    "u1_n2_shape.jl", "u1_n2_shape_nocb.jl", "u2_m7_nan.jl", "u0_single_file.jl",
    "k1_reverse_alone_nocb.jl",
]; label="enzyme_mwe3")
