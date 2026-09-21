# Issue #41 probe, round 1: is the crash in the advection operator alone, and
# which of its constructs matter. See enzyme_mwe/harness.jl.
include(joinpath(@__DIR__, "enzyme_mwe", "harness.jl"))
run_matrix([
    "s0_verbatim.jl", "s1_x3_slice.jl", "s2_boolmask.jl", "s3_noclosure.jl",
    "s4_pass1only.jl", "s5_musclonly.jl", "s6_noinbounds.jl", "s7_prealloc.jl",
    "s8_no_runtime_activity.jl",
    "k3_forward_alone.jl", "k1_reverse_alone.jl", "k2_forward_then_reverse.jl",
    "k4_advect_alone.jl", "k5_reverse_small.jl",
    "h0_local_verbatim.jl", "h1_boolmask.jl", "h2_noclosure.jl", "h3_pass1.jl",
    "h4_eastloop.jl", "h5_prealloc.jl",
]; label="enzyme_mwe")
