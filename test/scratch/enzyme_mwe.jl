# Issue #41 probe harness. Runs every variant script in test/scratch/enzyme_mwe/
# in its own Julia process (same flags and project as this one), so a segfault
# in one variant is recorded as a signal instead of ending the run, and prints
# a result matrix at the end. The @testset never fails on a crash: the log is
# the deliverable. Not part of the suite; dispatched with the CI `only` input.
using Test

const MWE_DIR = joinpath(@__DIR__, "enzyme_mwe")
const VARIANTS = [
    "s0_verbatim.jl", "s1_x3_slice.jl", "s2_boolmask.jl", "s3_noclosure.jl",
    "s4_pass1only.jl", "s5_musclonly.jl", "s6_noinbounds.jl", "s7_prealloc.jl",
    "s8_no_runtime_activity.jl",
    "k3_forward_alone.jl", "k1_reverse_alone.jl", "k2_forward_then_reverse.jl",
    "k4_advect_alone.jl", "k5_reverse_small.jl",
    "h0_local_verbatim.jl", "h1_boolmask.jl", "h2_noclosure.jl", "h3_pass1.jl",
    "h4_eastloop.jl", "h5_prealloc.jl",
]
const TIMEOUT_S = 900
# ENZYME_MWE_FILTER=<regex> restricts the variants (local reruns).
const FILTER = Regex(get(ENV, "ENZYME_MWE_FILTER", ""))
filter!(f -> occursin(FILTER, f), VARIANTS)

function run_variant(file)
    script = joinpath(MWE_DIR, file)
    cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) $script`
    log = tempname()
    t0 = time()
    # one handle shared by stdout and stderr, so the two streams interleave
    # instead of overwriting each other from offset 0
    io = open(log, "w")
    proc = run(pipeline(ignorestatus(cmd); stdout=io, stderr=io); wait=false)
    while process_running(proc) && time() - t0 < TIMEOUT_S
        sleep(2)
    end
    timed_out = process_running(proc)
    timed_out && kill(proc)
    wait(proc)
    close(io)
    elapsed = round(time() - t0; digits=1)
    text = read(log, String)
    status = if timed_out
        "TIMEOUT"
    elseif proc.termsignal != 0
        "SIGNAL $(proc.termsignal)"
    elseif proc.exitcode != 0
        "EXIT $(proc.exitcode)"
    else
        "PASS"
    end
    markers = [m.match for m in eachmatch(r"(REVERSE_OK|FORWARD_OK|IDENTITY_OK|IDENTITY_FAIL)", text)]
    return (; file, status, elapsed, markers, text)
end

println("=== enzyme_mwe harness: julia $(VERSION) $(Sys.MACHINE) threads=$(Threads.nthreads())")
println("=== julia_cmd = $(Base.julia_cmd())")
println("=== project   = $(Base.active_project())")
results = []
for file in VARIANTS
    println("\n##### VARIANT $file"); flush(stdout)
    r = run_variant(file)
    lines = split(r.text, '\n')
    keep = length(lines) > 400 ? vcat(lines[1:150], ["... ($(length(lines) - 300) lines dropped) ..."], lines[end-149:end]) : lines
    println(join(keep, '\n'))
    println("##### RESULT $file -> $(r.status) in $(r.elapsed)s markers=$(join(r.markers, ","))")
    flush(stdout)
    push!(results, r)
end

println("\n=== enzyme_mwe MATRIX (julia $(VERSION), $(Sys.MACHINE))")
for r in results
    println(rpad(r.file, 30), rpad(r.status, 12), rpad("$(r.elapsed)s", 9), join(r.markers, ","))
end
println("=== end MATRIX")

@testset "enzyme_mwe harness ran" begin
    @test length(results) == length(VARIANTS)
end
