using Test

# Use Kraken if available, otherwise include directly for standalone testing
if !isdefined(@__MODULE__, :KrakenExpr)
    include(joinpath(@__DIR__, "..", "src", "io", "expression.jl"))
    include(joinpath(@__DIR__, "..", "src", "io", "kraken_parser.jl"))
end

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")

@testset "Kraken Parser" begin

    @testset "Cavity example" begin
        setup = load_kraken(joinpath(EXAMPLES_DIR, "cavity.krk"))

        @test setup.name == "cavity"
        @test setup.lattice == :D2Q9
        @test setup.domain.Lx ≈ 1.0
        @test setup.domain.Ly ≈ 1.0
        @test setup.domain.Nx == 128
        @test setup.domain.Ny == 128
        @test setup.physics.params[:nu] ≈ 0.128
        @test setup.max_steps == 60000

        # Boundaries
        @test length(setup.boundaries) == 4
        faces = Dict(b.face => b for b in setup.boundaries)
        @test faces[:north].type == :velocity
        @test evaluate(faces[:north].values[:ux]) ≈ 0.1
        @test faces[:south].type == :wall
        @test faces[:east].type == :wall
        @test faces[:west].type == :wall

        # Output
        @test setup.output !== nothing
        @test setup.output.format == :vtk
        @test setup.output.interval == 10000
        @test :rho in setup.output.fields
        @test :ux in setup.output.fields

        # No obstacles, no modules
        @test isempty(setup.regions)
        @test isempty(setup.modules)
    end

    @testset "Poiseuille example" begin
        setup = load_kraken(joinpath(EXAMPLES_DIR, "poiseuille.krk"))

        @test setup.name == "poiseuille"
        @test setup.domain.Nx == 4
        @test setup.domain.Ny == 32
        @test setup.physics.params[:nu] ≈ 0.1

        # Body force from Physics line
        @test haskey(setup.physics.body_force, :Fx)
        @test evaluate(setup.physics.body_force[:Fx]) ≈ 1e-5

        # Periodic x → generates west + east periodic
        faces = Dict(b.face => b for b in setup.boundaries)
        @test faces[:west].type == :periodic
        @test faces[:east].type == :periodic
        @test faces[:south].type == :wall
        @test faces[:north].type == :wall
    end

    @testset "Cylinder example" begin
        setup = load_kraken(joinpath(EXAMPLES_DIR, "cylinder.krk"))

        @test setup.name == "cylinder"
        @test setup.domain.Lx ≈ 10.0
        @test setup.domain.Ly ≈ 2.5
        @test setup.domain.Nx == 200
        @test setup.domain.Ny == 50

        # User-defined variables
        @test setup.user_vars[:U] ≈ 0.05
        @test setup.user_vars[:H] ≈ 2.5
        @test setup.user_vars[:cx] ≈ 2.5
        @test setup.user_vars[:R] ≈ 0.5

        # Obstacle
        @test length(setup.regions) == 1
        obs = setup.regions[1]
        @test obs.name == "cylinder"
        @test obs.kind == :obstacle
        @test evaluate(obs.condition; x=2.5, y=1.25) == true  # center
        @test evaluate(obs.condition; x=0.0, y=0.0) == false  # far

        # Boundaries — west has spatial velocity
        faces = Dict(b.face => b for b in setup.boundaries)
        @test faces[:west].type == :velocity
        @test is_spatial(faces[:west].values[:ux])
        # Parabolic profile at y = H/2 → max = U
        @test evaluate(faces[:west].values[:ux]; y=1.25) ≈ 0.05 atol=1e-12
        @test faces[:east].type == :pressure

        # Diagnostics
        @test setup.diagnostics !== nothing
        @test setup.diagnostics.interval == 100
        @test :drag in setup.diagnostics.columns
    end

    @testset "Couette example" begin
        setup = load_kraken(joinpath(EXAMPLES_DIR, "couette.krk"))

        @test setup.name == "couette"
        @test setup.user_vars[:u_wall] ≈ 0.05

        faces = Dict(b.face => b for b in setup.boundaries)
        @test faces[:north].type == :velocity
        @test evaluate(faces[:north].values[:ux]) ≈ 0.05
        @test faces[:west].type == :periodic
        @test faces[:east].type == :periodic
    end

    @testset "Fluid region parsing" begin
        text = """
        Simulation channel D2Q9
        Domain L = 4 x 1  N = 80 x 20
        Physics nu = 0.1

        Fluid channel { y > 0.2 && y < 0.8 }

        Boundary west velocity(ux = 0.1)
        Boundary east pressure(rho = 1.0)
        Boundary south wall
        Boundary north wall

        Run 5000 steps
        """
        setup = parse_kraken(text)

        @test length(setup.regions) == 1
        @test setup.regions[1].kind == :fluid
        @test setup.regions[1].name == "channel"
        @test evaluate(setup.regions[1].condition; y=0.5) == true
        @test evaluate(setup.regions[1].condition; y=0.1) == false
    end

    @testset "Combined Fluid + Obstacle" begin
        text = """
        Simulation pipe_cylinder D2Q9
        Domain L = 10 x 2  N = 200 x 40
        Physics nu = 0.05

        Fluid pipe { y > 0.3 && y < 1.7 }
        Obstacle cylinder { (x - 5)^2 + (y - 1)^2 <= 0.3^2 }

        Boundary west velocity(ux = 0.05)
        Boundary east pressure(rho = 1.0)
        Boundary south wall
        Boundary north wall

        Run 10000 steps
        """
        setup = parse_kraken(text)

        @test length(setup.regions) == 2
        @test setup.regions[1].kind == :fluid
        @test setup.regions[2].kind == :obstacle
    end

    @testset "Initial conditions" begin
        text = """
        Simulation test D2Q9
        Domain L = 1 x 1  N = 32 x 32
        Physics nu = 0.1

        Boundary south wall
        Boundary north wall
        Boundary east wall
        Boundary west wall

        Initial { ux = 0.01*sin(2*pi*x/Lx) uy = 0 }

        Run 1000 steps
        """
        setup = parse_kraken(text)

        @test setup.initial !== nothing
        @test haskey(setup.initial.fields, :ux)
        @test is_spatial(setup.initial.fields[:ux])
    end

    @testset "Module parsing" begin
        text = """
        Simulation test D2Q9
        Domain L = 1 x 1  N = 32 x 32
        Physics nu = 0.1 Pr = 1.0 Ra = 2000

        Module thermal

        Boundary south wall
        Boundary north wall
        Boundary east wall
        Boundary west wall

        Run 1000 steps
        """
        setup = parse_kraken(text)
        @test :thermal in setup.modules
    end

    @testset "Error handling" begin
        # Missing Simulation line
        @test_throws ArgumentError parse_kraken("Domain L = 1 x 1 N = 32 x 32\nRun 100 steps")

        # Missing Domain
        @test_throws ArgumentError parse_kraken("Simulation test D2Q9\nRun 100 steps")

        # Missing Run
        @test_throws ArgumentError parse_kraken("Simulation test D2Q9\nDomain L = 1 x 1 N = 32 x 32")

        # Unknown keyword
        @test_throws ArgumentError parse_kraken("""
            Simulation test D2Q9
            Domain L = 1 x 1 N = 32 x 32
            FooBar baz
            Run 100 steps
        """)

        # Invalid lattice
        @test_throws ArgumentError parse_kraken("""
            Simulation test D2Q99
            Domain L = 1 x 1 N = 32 x 32
            Run 100 steps
        """)
    end

    @testset "Setup reynolds helper" begin
        text = """
        Simulation re_test D2Q9
        Domain L = 1.0 x 1.0  N = 128 x 128
        Setup reynolds = 1000 L_ref = 128 U_ref = 0.1
        Boundary north velocity(ux = 0.1, uy = 0)
        Boundary south wall
        Boundary east wall
        Boundary west wall
        Run 1000 steps
        """
        setup = parse_kraken(text)
        @test setup.physics.params[:nu] ≈ 0.0128 atol=1e-9
        @test setup.physics.params[:Re] ≈ 1000
    end

    @testset "Setup reynolds conflict with nu" begin
        text = """
        Simulation bad D2Q9
        Domain L = 1 x 1  N = 64 x 64
        Physics nu = 0.01
        Setup reynolds = 1000
        Boundary north wall
        Boundary south wall
        Boundary east wall
        Boundary west wall
        Run 100 steps
        """
        @test_throws ArgumentError parse_kraken(text)
    end

    @testset "Setup rayleigh helper" begin
        text = """
        Simulation ra_test D2Q9
        Domain L = 1.0 x 1.0  N = 128 x 128
        Setup rayleigh = 1e5 prandtl = 0.71 L_ref = 128 U_ref = 0.1
        Module thermal
        Boundary south wall T = 1.0
        Boundary north wall T = 0.0
        Boundary east wall
        Boundary west wall
        Run 100 steps
        """
        setup = parse_kraken(text)
        @test setup.physics.params[:Ra] ≈ 1e5
        @test setup.physics.params[:Pr] ≈ 0.71
        @test haskey(setup.physics.params, :nu)
        @test haskey(setup.physics.params, :alpha)
        @test haskey(setup.physics.params, :gbeta_DT)
        # Consistency: Pr = nu/alpha
        @test setup.physics.params[:nu] / setup.physics.params[:alpha] ≈ 0.71 atol=1e-9
        # Ra = gβΔT L^3 / (ν α)
        ν = setup.physics.params[:nu]
        α = setup.physics.params[:alpha]
        gβΔT = setup.physics.params[:gbeta_DT]
        @test gβΔT * 128.0^3 / (ν * α) ≈ 1e5 atol=1e-3
    end

    @testset "Preset cavity_2d" begin
        text = "Preset cavity_2d"
        setup = parse_kraken(text)
        @test setup.name == "cavity_2d"
        @test setup.domain.Nx == 128
        faces = Dict(b.face => b for b in setup.boundaries)
        @test faces[:north].type == :velocity
        @test faces[:south].type == :wall
    end

    @testset "Preset override" begin
        text = """
        Preset cavity_2d
        Setup reynolds = 5000 L_ref = 128 U_ref = 0.1
        """
        # Preset sets nu=0.01; reynolds would conflict. Use a preset without nu,
        # or override by manually clearing? For now test that Re-only overrides:
        @test_throws ArgumentError parse_kraken(text)

        # Override via Physics works:
        text2 = """
        Preset cavity_2d
        Physics nu = 0.005
        """
        setup = parse_kraken(text2)
        @test setup.physics.params[:nu] ≈ 0.005
    end

    @testset "Spell-correction for directives" begin
        text = """
        Simulation test D2Q9
        Domain L = 1 x 1 N = 32 x 32
        Physics nu = 0.1
        Boundayr north wall
        Run 100 steps
        """
        err = try
            parse_kraken(text)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("Boundary", err.msg)
    end

    @testset "Spell-correction for BC type" begin
        text = """
        Simulation test D2Q9
        Domain L = 1 x 1 N = 32 x 32
        Physics nu = 0.1
        Boundary north velocty(ux = 0.1)
        Boundary south wall
        Boundary east wall
        Boundary west wall
        Run 100 steps
        """
        err = try
            parse_kraken(text)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("velocity", err.msg)
    end

    @testset "Sanity check tau < 0.5" begin
        # nu = -0.1 would give tau = 0.2; use a valid-parseable Setup
        text = """
        Simulation unstable D2Q9
        Domain L = 1 x 1 N = 64 x 64
        Physics nu = -0.1
        Boundary north wall
        Boundary south wall
        Boundary east wall
        Boundary west wall
        Run 100 steps
        """
        err = try
            parse_kraken(text)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("increase", err.msg) && occursin("ν", err.msg)
    end

    @testset "Sanity check Mach warning" begin
        text = """
        Simulation fast D2Q9
        Domain L = 1 x 1 N = 64 x 64
        Physics nu = 0.1
        Boundary north velocity(ux = 0.5, uy = 0)
        Boundary south wall
        Boundary east wall
        Boundary west wall
        Run 100 steps
        """
        msgs = String[]
        logger = Base.CoreLogging.SimpleLogger(IOBuffer())
        setup = Base.CoreLogging.with_logger(logger) do
            parse_kraken(text)
        end
        # Capture: easier — just redirect stderr
        io = IOBuffer()
        Base.CoreLogging.with_logger(Base.CoreLogging.ConsoleLogger(io)) do
            parse_kraken(text)
        end
        out = String(take!(io))
        @test occursin("Mach", out)
    end

    @testset "Sweep directive" begin
        text = """
        Simulation sweep_test D2Q9
        Domain L = 1 x 1 N = 64 x 64
        Physics nu = 0.01
        Boundary north velocity(ux = 0.1, uy = 0)
        Boundary south wall
        Boundary east wall
        Boundary west wall
        Sweep Re = [100, 200, 500]
        Run 100 steps
        """
        setups = parse_kraken_sweep(text)
        @test length(setups) == 3
        # Re kwarg doesn't affect nu directly since no Setup reynolds present,
        # but it should be applied as kwarg override (no :Re param in physics).
        # Test with Setup reynolds instead:
        text2 = """
        Simulation sweep_re D2Q9
        Domain L = 1 x 1 N = 64 x 64
        Setup reynolds = Re L_ref = 64 U_ref = 0.1
        Boundary north velocity(ux = 0.1, uy = 0)
        Boundary south wall
        Boundary east wall
        Boundary west wall
        Sweep Re = [100, 200, 500]
        Run 100 steps
        """
        # This won't work since Setup parses literal numbers, not kwargs.
        # Simpler: check Sweep just produces 3 setups and kwargs are applied
        # by verifying any downstream effect. Use Physics nu override with Sweep:
        text3 = """
        Simulation sweep_nu D2Q9
        Domain L = 1 x 1 N = 64 x 64
        Physics nu = 0.01
        Boundary north velocity(ux = 0.1, uy = 0)
        Boundary south wall
        Boundary east wall
        Boundary west wall
        Sweep nu = [0.01, 0.02, 0.05]
        Run 100 steps
        """
        setups3 = parse_kraken_sweep(text3)
        @test length(setups3) == 3
        nus = sort([s.physics.params[:nu] for s in setups3])
        @test nus ≈ [0.01, 0.02, 0.05]
    end

    @testset "3D domain" begin
        text = """
        Simulation cavity3d D3Q19
        Domain L = 1 x 1 x 1  N = 32 x 32 x 32
        Physics nu = 0.1

        Boundary south wall
        Boundary north velocity(ux = 0.1)
        Boundary east wall
        Boundary west wall

        Run 1000 steps
        """
        setup = parse_kraken(text)
        @test setup.lattice == :D3Q19
        @test setup.domain.Lz ≈ 1.0
        @test setup.domain.Nz == 32
    end
end
