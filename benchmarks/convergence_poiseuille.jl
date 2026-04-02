# Poiseuille flow convergence study
# Verifies 2nd-order spatial accuracy of the LBM BGK solver
using Kraken
using Printf

function run_poiseuille_convergence()
    Nys = [8, 16, 32, 64, 128, 256]
    ν = 0.1
    Fx = 1e-5
    errors = Float64[]

    for Ny in Nys
        max_steps = max(10000, 200 * Ny)
        result = run_poiseuille_2d(; Nx=4, Ny=Ny, ν=ν, Fx=Fx, max_steps=max_steps)

        # Analytical parabolic profile with half-way bounce-back
        # Walls at y=0.5 and y=Ny+0.5 (half-way BB convention)
        H = Ny
        u_analytical = [Fx / (2ν) * (j - 0.5) * (H + 0.5 - j) for j in 1:Ny]

        # L2 relative error
        u_num = result.ux[2, :]
        err = sqrt(sum((u_num .- u_analytical) .^ 2) / sum(u_analytical .^ 2))
        push!(errors, err)
    end

    # Print results
    println("\n=== Poiseuille Convergence ===")
    @printf("  %5s   %12s   %5s\n", "Ny", "L2 error", "Order")
    @printf("  %5s   %12s   %5s\n", "-----", "----------", "-----")
    for (i, Ny) in enumerate(Nys)
        order = i > 1 ? log2(errors[i-1] / errors[i]) : NaN
        @printf("  %5d   %12.4e   %5s\n", Ny, errors[i],
                isnan(order) ? "  -" : @sprintf("%.2f", order))
    end

    return Nys, errors
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_poiseuille_convergence()
end
