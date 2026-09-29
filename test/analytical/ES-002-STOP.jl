# Issue #23: potential convergence does not imply field-moment convergence.
# CPU Float64 regression; no GPU, charge-transport or onset qualification.
# Run: julia --project test/analytical/ES-002-STOP.jl
module ES002FieldStoppingRegression

using Test
using Kraken
using KernelAbstractions

const CPU_BACKEND = KernelAbstractions.CPU()
const H = 16
const FIELD_GATE = 1e-6
const WEIGHTS = (4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36)
const CY = (0, 0, 1, 0, -1, 1, 1, -1, -1)

function potential_step!(f, g, phi, q, tau, xbc)
    nx, ny = size(phi)
    Kraken.collide_electric_potential_2d!(f, q, 0.7, inv(tau), (tau-0.5)/3)
    if xbc === :neumann
        Kraken.stream_wall_x_wall_y_2d!(g, f, nx, ny)
    else
        Kraken.stream_periodic_x_wall_y_2d!(g, f, nx, ny)
    end
    Kraken.compute_ehd_scalar_2d!(phi, g)
    if xbc === :neumann
        Kraken.apply_phi_nee_box_2d!(g, phi, 1.0, 0.0, nx, ny)
    else
        Kraken.apply_phi_nee_walls_2d!(g, phi, 1.0, 0.0, nx, ny)
    end
    Kraken.compute_ehd_scalar_2d!(phi, g)
    KernelAbstractions.synchronize(CPU_BACKEND)
    return g, f
end

function capacitor(xbc, tau, initialization)
    nx = xbc === :neumann ? 11 : 10
    ny = H + 1
    reference = [1.0-(j-1)/H for i in 1:nx, j in 1:ny]
    f = zeros(nx, ny, 9)
    # Use the real driver initializer for the cold populations.
    Kraken._fill_phi_populations!(f, vec(reference[1, :]), nx, ny, Float64)
    if initialization === :consistent
        for k in 1:9
            f[:, :, k] .+= WEIGHTS[k] * tau * CY[k] / H
        end
    end
    g = copy(f)
    phi = copy(reference)
    previous = similar(phi)
    q = zeros(nx, ny)
    ex, ey = zeros(nx, ny), zeros(nx, ny)
    diag = zeros(2)
    accepted = false
    n = 0
    relative_change = Inf
    # Mirrors the CURRENT EC scalar-only stopping loop, using production
    # collision, streaming, boundary, reduction and field kernels. There is
    # no reusable potential-solve API accepting these states yet. When #23
    # introduces that seam, route this fixture through it before promoting
    # its broken assertions. A driver-only fix cannot change a mirrored loop.
    for iteration in 1:128
        copyto!(previous, phi)
        f, g = potential_step!(f, g, phi, q, tau, xbc)
        n = iteration
        if iteration % 8 == 0
            Kraken.ehd_rel_change_2d!(diag, phi, previous, nx, ny)
            KernelAbstractions.synchronize(CPU_BACKEND)
            relative_change = diag[1]
            if relative_change <= 1e-10
                accepted = true
                break
            end
        end
    end
    Kraken.compute_electric_field_2d!(ex, ey, f, tau)
    KernelAbstractions.synchronize(CPU_BACKEND)
    ex .*= H
    ey .*= H
    field_error = max(maximum(abs, ex), maximum(abs.(ey .- 1.0)))
    # Independent recurrence: a_(n+1)=(1-1/tau)*a_n+1/H, a_0=0.
    prediction = initialization === :cold ? 1-(1-1/tau)^n : 1.0
    prediction_error = max(maximum(abs, ex), maximum(abs.(ey .- prediction)))
    return (; accepted, n, relative_change, field_error, prediction_error,
            phi_error=maximum(abs.(phi .- reference)),
            finite=all(isfinite, f) && all(isfinite, ex) && all(isfinite, ey))
end

field_difference(a, b) = H * max(maximum(abs.(a.Ex .- b.Ex)),
                               maximum(abs.(a.Ey .- b.Ey)))

@testset "ES-002-STOP: known field-stopping defect (#23)" begin
    @testset "$xbc tau=$tau $initialization" for xbc in (:neumann, :periodic),
            tau in (0.8, 1.4, 2.0), initialization in (:cold, :consistent)
        result = capacitor(xbc, tau, initialization)
        @test result.accepted
        @test result.finite
        @test result.relative_change <= 1e-10
        # Exact capacitor phi*=1-y/H, E*=(0,1), ALL nodes including corners.
        # Budgets frozen in ES-002-STOP on 2026-09-15, not fitted to a repair.
        @test result.phi_error <= 1e-10
        @test result.prediction_error <= 1e-11
        if initialization === :cold
            # https://github.com/lauguimel/Kraken.jl/issues/23
            @test_broken result.field_error <= FIELD_GATE
        else
            @test result.field_error <= FIELD_GATE
            @test result.field_error <= 1e-10
        end
        @info "Capacitor stopping" xbc tau initialization result
    end

    @testset "Public EC driver: first-cycle field convergence" begin
        # The driver cannot accept q=0/custom potential populations. This
        # additional small charged case exercises its ACTUAL adaptive loop,
        # without copying a stop predicate or changing production code.
        # C=0.01, tau_U=2 and one cycle isolate cold-start potential iteration;
        # q is identical until that first inner solve finishes. Returned E
        # still belongs to that solve, before the subsequent charge update.
        parameters = (; Nx=11, Ny=H+1, C=0.01, M=10.0, T=175.0,
                      Ma_E=0.01, alpha=1e-4, gamma=0.5, max_cycles=1,
                      perturb_amplitude=0.0, phi_scheme=:lbm,
                      backend=CPU_BACKEND, FT=Float64)
        adaptive = Kraken.run_electroconvection_2d(; parameters..., phi_tol=1e-4)
        fixed = Kraken.run_electroconvection_2d(; parameters..., phi_substeps=1024)
        reference = Kraken.run_electroconvection_2d(; parameters..., phi_substeps=2048)
        # Fixed sweeps are a test-only iterative reference, NOT the proposed
        # production repair. Doubling must leave E* unchanged to 1e-8, two
        # orders tighter than the 1e-6 stopping-accuracy gate (set 2026-09-16).
        @test all(isfinite, adaptive.Ex) && all(isfinite, adaptive.Ey)
        @test all(isfinite, reference.Ex) && all(isfinite, reference.Ey)
        @test adaptive.steps == fixed.steps == reference.steps == 1
        @test field_difference(fixed, reference) <= 1e-8
        @test_broken field_difference(adaptive, reference) <= FIELD_GATE
        @info "Public driver field stopping" iterations=adaptive.phi_iters_last error=field_difference(adaptive, reference) reference_change=field_difference(fixed, reference)
        # This assertion can detect a driver-only repair. The capacitor
        # fixtures above additionally need rewiring to its new solver seam.
        # Code-to-code iterative convergence is not physical validation.
    end
end

end # module
