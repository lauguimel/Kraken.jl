# Issue #41 probe, round 4: does LLVM loop unrolling take part in the crash?
# Enzyme.jl #3605 (Bool mask in a loop, platform-dependent because LLVM unrolls
# differently on x86_64 and aarch64) was worked around by disabling runtime
# unrolling. Same three reproductions as round 3, each run three times in a
# child process: default LLVM options, runtime unrolling off, all unrolling
# off. JULIA_LLVM_ARGS is read by the child at startup and applies to Enzyme's
# pass pipeline too (Enzyme_jll links the same libLLVM). A crash that goes away
# under either setting ties #41 to #3605; one that stays separates them.
include(joinpath(@__DIR__, "enzyme_mwe", "harness.jl"))
const ROUND4 = ["u0_single_file.jl", "u1_n2_shape.jl", "k1_reverse_alone.jl"]
run_matrix(ROUND4; label="enzyme_mwe4_default")
run_matrix(ROUND4; label="enzyme_mwe4_no_runtime_unroll",
           env=["JULIA_LLVM_ARGS" => "-unroll-runtime=false"])
run_matrix(ROUND4; label="enzyme_mwe4_no_unroll",
           env=["JULIA_LLVM_ARGS" => "-unroll-threshold=0 -unroll-runtime=false"])
