# Issue #41 probe harness (shared by test/scratch/enzyme_mwe*.jl). Runs each
# variant script in its own Julia process (same flags and project as this
# one), so a segfault in one variant is recorded as a signal instead of ending
# the run, and prints a result matrix at the end. Variants named *_nocb.jl run
# with --check-bounds=no appended. ENZYME_MWE_FILTER=<regex> restricts the
# list (local reruns). Not part of the suite; dispatched with the CI `only`
# input.
using Test

const MWE_DIR = @__DIR__
const TIMEOUT_S = 900

function run_variant(file; env=Pair{String,String}[])
    script = joinpath(MWE_DIR, file)
    flags = endswith(file, "_nocb.jl") ? `--check-bounds=no` : ``
    cmd = `$(Base.julia_cmd()) $flags --project=$(Base.active_project()) $script`
    # extra environment for the child only (e.g. JULIA_LLVM_ARGS, which the
    # child parses at startup; setting it in this process would be too late)
    isempty(env) || (cmd = addenv(cmd, env...))
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

function run_matrix(variants; label="enzyme_mwe", env=Pair{String,String}[])
    filt = Regex(get(ENV, "ENZYME_MWE_FILTER", ""))
    variants = filter(f -> occursin(filt, f), variants)
    println("=== $label harness: julia $(VERSION) $(Sys.MACHINE) threads=$(Threads.nthreads())")
    println("=== julia_cmd = $(Base.julia_cmd())")
    println("=== project   = $(Base.active_project())")
    isempty(env) || println("=== child env = $(env)")
    results = []
    for file in variants
        println("\n##### VARIANT $file"); flush(stdout)
        r = run_variant(file; env)
        lines = split(r.text, '\n')
        keep = length(lines) > 400 ? vcat(lines[1:150], ["... ($(length(lines) - 300) lines dropped) ..."], lines[end-149:end]) : lines
        println(join(keep, '\n'))
        println("##### RESULT $file -> $(r.status) in $(r.elapsed)s markers=$(join(r.markers, ","))")
        flush(stdout)
        push!(results, r)
    end
    println("\n=== $label MATRIX (julia $(VERSION), $(Sys.MACHINE))")
    for r in results
        println(rpad(r.file, 30), rpad(r.status, 12), rpad("$(r.elapsed)s", 9), join(r.markers, ","))
    end
    println("=== end MATRIX")
    @testset "$label harness ran" begin
        @test length(results) == length(variants)
    end
    return results
end
