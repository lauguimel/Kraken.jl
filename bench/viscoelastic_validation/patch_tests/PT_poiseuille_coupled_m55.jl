#!/usr/bin/env julia

using Printf
using Kraken
using KernelAbstractions

try
    @eval using Metal
catch
end

const OUTDIR = joinpath("scratch", "M56_vv_ladder")
const CSV = joinpath(OUTDIR, "PT_poiseuille_coupled_m55_profiles.csv")

function backend_choice()
    requested = lowercase(get(ENV, "KRAKEN_BACKEND", "metal"))
    if requested == "metal" && isdefined(Main, :Metal)
        try
            metal = getfield(Main, :Metal)
            @eval KernelAbstractions.allocate(::Metal.MetalBackend, ::Type{T}, dims::Tuple;
                                              unified=nothing) where {T} =
                Metal.MtlArray{T}(undef, dims)
            @eval KernelAbstractions.zeros(::Metal.MetalBackend, ::Type{T}, dims::Tuple;
                                           unified=nothing) where {T} =
                Metal.zeros(T, dims)
            return metal.MetalBackend(), Float32, "metal"
        catch err
            @printf("M56_L5 backend_warning=metal_unavailable error=%s\n", sprint(showerror, err))
        end
    end
    return KernelAbstractions.CPU(), Float64, "cpu"
end

function gamma_from_u(u)
    Ny = length(u)
    gamma = zeros(Float64, Ny)
    for j in 1:Ny
        up = j == Ny ? 0.0 : u[j + 1]
        um = j == 1 ? 0.0 : u[j - 1]
        gamma[j] = 0.5 * (up - um)
    end
    return gamma
end

function tau_profiles(result)
    Nx, Ny = size(result.psixx)
    prefactor = result.nu_p / result.lambda
    tauxx = zeros(Float64, Ny)
    tauxy = zeros(Float64, Ny)
    tauyy = zeros(Float64, Ny)
    for j in 1:Ny
        sx = sy = sz = 0.0
        for i in 1:Nx
            cxx, cxy, cyy = Kraken.logfv_exp_sym2_2d(
                Float64(result.psixx[i, j]),
                Float64(result.psixy[i, j]),
                Float64(result.psiyy[i, j]),
            )
            sx += prefactor * (cxx - 1.0)
            sy += prefactor * cxy
            sz += prefactor * (cyy - 1.0)
        end
        tauxx[j] = sx / Nx
        tauxy[j] = sy / Nx
        tauyy[j] = sz / Nx
    end
    return tauxx, tauxy, tauyy
end

function max_relative_error(values, reference, js)
    denom = maximum(abs.(reference[js]))
    return maximum(abs.(values[js] .- reference[js])) / max(denom, eps(Float64))
end

function write_profiles(result, tauxx, tauxy, tauyy, tau_xx_ref, tau_xy_ref, tau_yy_ref)
    mkpath(OUTDIR)
    open(CSV, "w") do io
        println(io, "j,u_mean,u_ref,tauxx,tauxx_ref,tauxy,tauxy_ref,tauyy,tauyy_ref")
        for j in 1:result.Ny
            @printf(io, "%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                    j, Float64(result.ux_mean[j]), Float64(result.reference_ux[j]),
                    tauxx[j], tau_xx_ref[j], tauxy[j], tau_xy_ref[j], tauyy[j], tau_yy_ref[j])
        end
    end
end

function run_l5()
    backend, FT, backend_name = backend_choice()
    Nx = parse(Int, get(ENV, "M56_L5_NX", "60"))
    Ny = parse(Int, get(ENV, "M56_L5_NY", "32"))
    max_steps = parse(Int, get(ENV, "M56_L5_STEPS", "5000"))
    nu_total = 0.05
    beta = 0.59
    nu_s = beta * nu_total
    nu_p = (1.0 - beta) * nu_total
    lambda = 2000.0
    Fx_body = nu_total / (lambda * ((Ny - 1) / 2))

    t0 = time()
    result = Kraken.run_viscoelastic_logfv_poiseuille_coupled_2d(;
        Nx, Ny, nu_s=FT(nu_s), nu_p=FT(nu_p), Fx_body=FT(Fx_body),
        lambda=FT(lambda), bsd_fraction=FT(1.0), polymer_substeps=:auto,
        max_steps, backend, T=FT,
    )
    tauxx, tauxy, tauyy = tau_profiles(result)
    gamma = gamma_from_u(result.reference_ux)
    tau_xy_ref = nu_p .* gamma
    tau_xx_ref = 2.0 .* lambda .* nu_p .* gamma .^ 2
    tau_yy_ref = zeros(Float64, Ny)
    interior = collect(3:(Ny - 2))
    u_rel = result.max_rel_error
    tauxx_rel = max_relative_error(tauxx, tau_xx_ref, interior)
    tauxy_rel = max_relative_error(tauxy, tau_xy_ref, interior)
    write_profiles(result, tauxx, tauxy, tauyy, tau_xx_ref, tau_xy_ref, tau_yy_ref)

    pass = u_rel < 0.02 && tauxx_rel < 0.05
    @printf("M56_L5 backend=%s steps=%d polymer_substeps=%d walltime_s=%.3f\n",
            backend_name, max_steps, result.polymer_substeps, time() - t0)
    @printf("M56_L5 qwall_provenance=planar_channel_uses_bc_aware_gradient_no_embedded_qwall\n")
    @printf("M56_L5 verdict=%s u_rel=%.17g tauxx_rel=%.17g tauxy_rel=%.17g min_c_eig=%.17g csv=%s\n",
            pass ? "PASS" : "FAIL", u_rel, tauxx_rel, tauxy_rel, result.min_c_eig, CSV)
    return pass ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(run_l5())
end
