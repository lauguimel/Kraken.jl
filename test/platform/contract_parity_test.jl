using Test
using Kraken

# A method that declares nothing → exercises the default (fail-safe) capabilities().
# (structs are top-level only in Julia, so it lives outside the @testset.)
struct _DummyMethod <: Kraken.AbstractMethod end

@testset "platform contract (Phase 0/0b)" begin

    @testset "capabilities introspection" begin
        @test capabilities(LBM()) == Set((ForwardSolve, GPUExecution, SteadyAdjoint))
        @test capabilities(_DummyMethod()) == Set{Capability}()   # default = empty (fail-safe)
    end

    @testset "solve/sample parity with run_simulation" begin
        krk = """
            Simulation parity D2Q9
            Domain L = 0.25 x 1.0  N = 8 x 16
            Physics nu = 0.1  Fx = 1e-5
            Boundary x periodic
            Boundary south wall
            Boundary north wall
            Run 200 steps
        """
        # Separate setups so any setup mutation can't couple the two runs.
        ref = run_simulation(parse_kraken(krk))
        sol = solve(parse_kraken(krk), LBM())

        @test sol isa LBMSolution
        @test sol isa AbstractSolution

        # Behaviour-preserving: the wrapped result is bit-for-bit identical.
        @test sol.result.ρ  == ref.ρ
        @test sol.result.ux == ref.ux
        @test sol.result.uy == ref.uy

        # sample is a faithful pass-through (same array object; indexed access matches).
        @test sample(sol, :ux)         === sol.result.ux
        @test sample(sol, :ux, :)      === sol.result.ux
        @test sample(sol, :uy, (2, 2)) == ref.uy[2, 2]
    end
end
