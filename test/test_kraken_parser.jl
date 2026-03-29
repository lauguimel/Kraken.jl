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
