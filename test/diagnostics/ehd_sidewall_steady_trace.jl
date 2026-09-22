# Diagnostic only for Issue #21 / PR #37. Not included in the default suite.
# KRAKEN_ONLY=diagnostics/ehd_sidewall_steady_trace.jl
# The original EC-SIDEWALLS.jl assertions, including failures, are unchanged.
module ECSidewallSteadyTrace
using Test
using LinearAlgebra
using Kraken

const BASELINE = "7ce7e11286110f153fb553c6a5d3965464fb8118"
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))

function initial_state(FT, scheme, sign)
    L, Nx, Ny = 16, 17, 5
    nu, force = FT(0.1), FT(sign)*FT(1e-4)
    f = zeros(FT, Nx, Ny, 9)
    weights = (4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36)
    for d in 1:9
        f[:,:,d] .= FT(weights[d])
    end
    return (; L, Nx, Ny, nu, force, scheme, f, out=similar(f),
            solid=zeros(Bool, Nx, Ny), Fx=fill(force, Nx, Ny),
            Fy=zeros(FT, Nx, Ny), rho=ones(FT, Nx, Ny),
            ux=zeros(FT, Nx, Ny), uy=zeros(FT, Nx, Ny))
end

function cycle!(s)
    # Identical production calls/order and halo copies to the failed fixture.
    FT = eltype(s.f)
    if s.scheme === :bgk
        Kraken.collide_guo_field_2d!(s.f, s.solid, s.Fx, s.Fy, inv(3*s.nu+FT(0.5)))
    else
        Kraken.ehd_collide_mrt_2d!(s.f, s.Fx, s.Fy, s.solid, s.nu)
    end
    Kraken.stream_fully_periodic_2d!(s.out, s.f, s.Nx, s.Ny)
    Kraken.apply_no_slip_sidewalls_2d!(s.out, s.Fx, s.Fy, s.Nx, s.Ny)
    s.out[:,1,:] .= s.out[:,4,:]
    s.out[:,5,:] .= s.out[:,2,:]
    return merge(s, (f=s.out, out=s.f))
end

function observe!(s)
    Kraken.compute_macroscopic_guo_field_2d!(s.rho, s.ux, s.uy,
        s.f, s.Fx, s.Fy, s.Nx, s.Ny)
    # Convert only diagnostics, never populations, to Float64.
    return (rho=Float64.(s.rho[:,2:4]), ux=Float64.(s.ux[:,2:4]),
            uy=Float64.(s.uy[:,2:4]))
end

velocity_difference(a, b) = maximum(hypot.(a.ux-b.ux, a.uy-b.uy))

function report(s, current, previous, lag, step, checkpoint)
    FT = eltype(s.f)
    mean_rho = sum(current.rho)/length(current.rho)
    expected = repeat(reshape(3Float64(s.force) .* (collect(0:s.L) .- s.L/2), s.Nx, 1), 1, 3)
    profile = norm(current.rho .- mean_rho .- expected)/norm(expected)
    gradient = maximum(abs, diff(current.rho; dims=1) ./ (3Float64(s.force)) .- 1)
    speed = maximum(hypot.(current.ux, current.uy))
    du1 = velocity_difference(current, previous)
    du100 = isnothing(lag) ? NaN : velocity_difference(current, lag)
    drho100 = isnothing(lag) ? NaN : maximum(abs, current.rho-lag.rho)/(3abs(Float64(s.force))*s.L)
    wall_speed = maximum(hypot.(current.ux[[1,end],:], current.uy[[1,end],:]))
    # CSV records in the persistent CI log; no new workflow or upload privilege.
    println("EC_TRACE,", join((FT, s.scheme, Float64(s.force), step,
        Float64(s.nu)*step/s.L^2, mean_rho, profile, gradient, speed,
        du1, du100, drho100, current.ux[9,2], current.uy[9,2], wall_speed), ','))
    if checkpoint
        gate = FT === Float64 ? 1e-4 : 5e-3
        speed_gate = FT === Float64 ? 1e-8 : 2e-6
        # Report the frozen acceptance gates; do not assert them here, loosen
        # them, or replace the original failing assertions with @test_broken.
        println("EC_GATE,", join((FT, s.scheme, Float64(s.force), step,
            profile <= gate, gradient <= gate, speed <= speed_gate,
            drho100 <= gate/10, du100 <= speed_gate/10), ','))
        for i in 1:s.Nx
            println("EC_PROFILE,", join((FT, s.scheme, Float64(s.force), step,
                i-1, current.rho[i,2], current.ux[i,2], current.uy[i,2]), ','))
        end
    end
end

function trace_case(FT, scheme, sign)
    s = initial_state(FT, scheme, sign)
    nbase = ceil(Int, 4*s.L^2/s.nu)
    checkpoints = (nbase, 2nbase, 4nbase)
    previous = observe!(s)
    history = Vector{typeof(previous)}(undef, 100)
    history[1] = previous # Slot 1 contains step 0, then 100, 200, ...
    report(s, previous, previous, nothing, 0, false)
    checkpoint_f = copy(s.f)
    for step in 1:4nbase
        s = cycle!(s)
        current = observe!(s)
        slot = mod(step, 100)+1
        lag = step >= 100 ? history[slot] : nothing
        dense = any(c -> c-127 <= step <= c, checkpoints)
        if step % 100 == 0 || dense
            report(s, current, previous, lag, step, step in checkpoints)
        end
        if step in checkpoints
            @test all(isfinite, s.f)
            @test all(isfinite, current.rho) && all(isfinite, current.ux) && all(isfinite, current.uy)
            @test minimum(current.rho) > 0
        end
        step == nbase && (checkpoint_f = copy(s.f))
        history[slot] = current
        previous = current
    end
    # Same loop without per-step macro observation: prove tracing did not
    # perturb the trajectory at the original failing horizon, for every case.
    replay = initial_state(FT, scheme, sign)
    for _ in 1:nbase
        replay = cycle!(replay)
    end
    @test replay.f == checkpoint_f
    println("EC_OBSERVER_PARITY,", join((FT, scheme, sign, nbase, replay.f == checkpoint_f), ','))
end

@testset "Sidewall trace integrity ONLY; not acceptance" begin
    @test realpath(dirname(dirname(pathof(Kraken)))) == realpath(ROOT)
    # workflow checkout is shallow; Git object access to BASELINE is not
    # required at runtime. Pre-publication diff verifies unchanged src/ext/gates.
    println("EC_IDENTITY,", readchomp(`git -C $ROOT rev-parse HEAD`), ",", VERSION, ",", pathof(Kraken))
    println("EC_BASELINE,", BASELINE)
    println("EC_DIAGNOSTIC_ONLY: successful execution does not accept the original failed gates")
    println("EC_TRACE_HEADER,type,scheme,force,step,Fo,mean_rho,profile_error,gradient_error,speed,du1,du100,drho100,ux_mid,uy_mid,wall_speed")
    println("EC_GATE_HEADER,type,scheme,force,step,profile_ok,gradient_ok,speed_ok,density_change_ok,velocity_change_ok")
    println("EC_PROFILE_HEADER,type,scheme,force,step,x,rho,ux,uy")
    for (FT, scheme) in ((Float32, :mrt), (Float64, :mrt), (Float32, :bgk)), sign in (-1, 1)
        @testset "$FT $scheme sign=$sign" begin
            trace_case(FT, scheme, sign)
        end
    end
end
end # module
