using Test
using Kraken
using KernelAbstractions

@testset "Boussinesq" begin

    @testset "buoyancy_force! correctness" begin
        N = 16
        T_field = ones(N, N)
        # Set interior to known values
        for j in 2:N-1, i in 2:N-1
            T_field[i, j] = 0.5 + 0.1 * i
        end

        fu = zeros(N, N)
        fv = zeros(N, N)
        β = 2.0
        T_ref = 0.5
        gx = 0.0
        gy = 1.0

        buoyancy_force!(fu, fv, T_field, β, T_ref, gx, gy)

        # Check interior points
        for j in 2:N-1, i in 2:N-1
            dT = T_field[i, j] - T_ref
            @test fu[i, j] ≈ β * dT * gx
            @test fv[i, j] ≈ β * dT * gy
        end

        # Boundary should be untouched (zero)
        @test fu[1, 1] == 0.0
        @test fv[1, 1] == 0.0
    end

    @testset "advance_temperature! pure diffusion" begin
        # Pure diffusion: u=v=0, T₀ = sin(πx)sin(πy)
        # Exact: T(t) = exp(-2π²κt) * sin(πx)sin(πy)
        N = 33
        dx = 1.0 / (N - 1)
        κ = 0.01
        dt = 0.2 * dx^2 / κ  # stable explicit timestep
        n_steps = 50

        T_field = zeros(N, N)
        for j in 1:N, i in 1:N
            x = (i - 1) * dx
            y = (j - 1) * dx
            T_field[i, j] = sin(π * x) * sin(π * y)
        end

        u = zeros(N, N)
        v = zeros(N, N)
        adv_T = zeros(N, N)
        lap_T = zeros(N, N)

        t = 0.0
        for _ in 1:n_steps
            advance_temperature!(T_field, u, v, κ, dx, dt;
                                adv_T=adv_T, lap_T=lap_T)
            # Dirichlet BC: T=0 on boundaries (sin is 0 there)
            T_field[:, 1] .= 0.0
            T_field[:, N] .= 0.0
            T_field[1, :] .= 0.0
            T_field[N, :] .= 0.0
            t += dt
        end

        # Compute exact solution
        T_exact = zeros(N, N)
        for j in 1:N, i in 1:N
            x = (i - 1) * dx
            y = (j - 1) * dx
            T_exact[i, j] = exp(-2 * π^2 * κ * t) * sin(π * x) * sin(π * y)
        end

        # L2 error on interior
        err = 0.0
        count = 0
        for j in 2:N-1, i in 2:N-1
            err += (T_field[i, j] - T_exact[i, j])^2
            count += 1
        end
        L2 = sqrt(err / count)
        @test L2 < 1e-3
    end

    @testset "pure conduction (Ra=0 equivalent)" begin
        # No buoyancy: with T_hot=1 bottom, T_cold=0 top, steady state = linear profile
        # Run a few diffusion steps with no velocity, check T stays linear
        N = 17
        dx = 1.0 / (N - 1)
        κ = 0.1
        dt = 0.2 * dx^2 / κ

        T_field = zeros(N, N)
        # Initialize with linear profile: T = 1 - y
        for j in 1:N, i in 1:N
            y = (j - 1) * dx
            T_field[i, j] = 1.0 - y
        end

        u = zeros(N, N)
        v = zeros(N, N)
        adv_T = zeros(N, N)
        lap_T = zeros(N, N)

        T_hot = 1.0
        T_cold = 0.0

        # Run 100 steps — linear profile should be maintained (it's the steady state)
        for _ in 1:100
            advance_temperature!(T_field, u, v, κ, dx, dt;
                                adv_T=adv_T, lap_T=lap_T)
            # Apply RB BCs
            T_field[:, 1] .= T_hot
            T_field[:, N] .= T_cold
            T_field[1, :] .= @view T_field[2, :]
            T_field[N, :] .= @view T_field[N-1, :]
        end

        # Check interior matches linear profile within tolerance
        max_err = 0.0
        for j in 2:N-1, i in 2:N-1
            y = (j - 1) * dx
            T_exact = 1.0 - y
            max_err = max(max_err, abs(T_field[i, j] - T_exact))
        end
        @test max_err < 1e-6
    end

    @testset "Rayleigh-Bénard smoke test" begin
        # Ra=1e4, Pr=0.71: expect convection rolls and Nu ≈ 2.243 (±20%)
        u, v, p, T_field, converged = run_rayleigh_benard(
            N=32, Ra=1e4, Pr=0.71, max_steps=3000,
            tol=1e-7, cfl=0.3, verbose=false
        )

        N = 32
        dx = 1.0 / (N - 1)
        H = 1.0
        ΔT = 1.0

        # Compute Nusselt number at bottom wall: Nu = -H/ΔT * mean(∂T/∂y|_{y=0})
        # ∂T/∂y|_{y=0} ≈ (T[:,2] - T[:,1]) / dx
        T_arr = Array(T_field)
        dTdy_bottom = (T_arr[:, 2] .- T_arr[:, 1]) ./ dx
        Nu = -H / ΔT * sum(dTdy_bottom[2:N-1]) / (N - 2)

        # For Ra=1e4, expected Nu ≈ 2.243
        # Allow generous tolerance for coarse grid + explicit scheme
        @test Nu > 1.5   # must be > 1 (convection enhances heat transfer)
        @test Nu < 4.0   # should not be wildly high

        # Temperature should be bounded [0, 1]
        @test minimum(T_arr) >= -0.1
        @test maximum(T_arr) <= 1.1

        # Velocity should not blow up
        @test maximum(abs.(Array(u))) < 1000.0
        @test maximum(abs.(Array(v))) < 1000.0
    end

end
