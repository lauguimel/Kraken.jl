#!/usr/bin/env julia
using Kraken
using Printf
using Dates
using KernelAbstractions
using Metal
const OUTDIR = joinpath("scratch", "M48_hw_meshconv")
const UMEAN = 0.005
const BETA = 0.59
const WI = 1.0
const RE = 1.0
const BSD = 1.0

function metal_backend()
    lowercase(get(ENV, "KRAKEN_BACKEND", "metal")) == "metal" ||
        error("M48 requires KRAKEN_BACKEND=metal; CPU fallback is forbidden")
    return Metal.MetalBackend(), Float32
end

KernelAbstractions.allocate(::Metal.MetalBackend, ::Type{T}, dims::Tuple; unified=nothing) where {T} =
    Metal.MtlArray{T}(undef, dims)
KernelAbstractions.zeros(::Metal.MetalBackend, ::Type{T}, dims::Tuple; unified=nothing) where {T} =
    Metal.zeros(T, dims)

function csv_cell(x)
    x isa AbstractFloat && !isfinite(x) && return string(x)
    x isa AbstractFloat && return @sprintf("%.16g", Float64(x))
    return string(x)
end

function write_row(io, vals)
    println(io, join(csv_cell.(vals), ","))
    flush(io)
end

function trace_c_max(psixx, psixy, psiyy, is_solid)
    axx = Array(psixx); axy = Array(psixy); ayy = Array(psiyy)
    best = -Inf
    @inbounds for j in axes(is_solid, 2), i in axes(is_solid, 1)
        is_solid[i, j] && continue
        m = 0.5 * (Float64(axx[i, j]) + Float64(ayy[i, j]))
        d = 0.5 * (Float64(axx[i, j]) - Float64(ayy[i, j]))
        r = hypot(d, Float64(axy[i, j]))
        trc = exp(m + r) + exp(m - r)
        isfinite(trc) || return NaN
        best = max(best, trc)
    end
    return best
end

function bsd_drag(dudx, dudy, dvdx, dvdy, q_wall, nx, ny; cx, cy, radius, zeta_nu_p)
    dux = Array(dudx); duy = Array(dudy); dvx = Array(dvdx); dvy = Array(dvdy)
    txx = Matrix{Float64}(undef, nx, ny)
    txy = similar(txx)
    tyy = similar(txx)
    @inbounds for j in 1:ny, i in 1:nx
        txx[i, j] = 2.0 * zeta_nu_p * Float64(dux[i, j])
        txy[i, j] = zeta_nu_p * (Float64(duy[i, j]) + Float64(dvx[i, j]))
        tyy[i, j] = 2.0 * zeta_nu_p * Float64(dvy[i, j])
    end
    return Kraken.compute_polymeric_drag_2d(
        txx, txy, tyy, q_wall, nx, ny;
        cx, cy, radius, extrapolate=true, reconstruction_order=2,
    )
end

function run_R(backend, FT, R::Int, max_steps::Int, log_every::Int)
    mkpath(OUTDIR)
    path = joinpath(OUTDIR, "cdtraj_R$(R)_wi1_halfway.csv")
    log_path = joinpath(OUTDIR, "run_R$(R).log")
    L_up = 15.0
    L_down = 15.0
    H = 4 * R
    nx = ceil(Int, (L_up + L_down) * R)
    ny = H
    nu_total = UMEAN * R / RE
    nu_s = BETA * nu_total
    nu_p = (1.0 - BETA) * nu_total
    lambda = WI * R / UMEAN
    cx = L_up * R
    cy = (ny - 1) / 2
    diameter = 2.0 * R
    t0 = time()

    open(path, "w") do io
        write_row(io, [
            "step", "t_phys", "flow_through", "Cd_kraken", "Cd_s", "Cd_p",
            "Cd_bsd", "trace_C_max",
        ])
        callback = function (step, s)
            (step == 1 || step % log_every == 0 || step == max_steps) || return nothing
            drag_s = Kraken.compute_drag_libb_mei_2d(s.f_out, s.q_wall, s.uwx, s.uwy, nx, ny)
            drag_p = Kraken.compute_polymeric_drag_2d(
                s.tauxx, s.tauxy, s.tauyy, s.q_wall, nx, ny;
                cx, cy, radius=R, extrapolate=true, reconstruction_order=2,
            )
            drag_bsd = bsd_drag(
                s.dudx, s.dudy, s.dvdx, s.dvdy, s.q_wall, nx, ny;
                cx, cy, radius=R, zeta_nu_p=BSD * nu_p,
            )
            cd_s = 2.0 * drag_s.Fx / (UMEAN^2 * diameter)
            cd_p = 2.0 * drag_p.Fx / (UMEAN^2 * diameter)
            cd_bsd = 2.0 * drag_bsd.Fx / (UMEAN^2 * diameter)
            cd = cd_s + cd_p - cd_bsd
            t_phys = step * UMEAN / R
            write_row(io, [step, t_phys, step / max_steps, cd, cd_s, cd_p, cd_bsd,
                           trace_c_max(s.psixx, s.psixy, s.psiyy, s.is_solid_h)])
            return nothing
        end
        result = Kraken.run_viscoelastic_logfv_cylinder_coupled_2d(;
            radius=R, H, L_up, L_down,
            nu_s=FT(nu_s), nu_p=FT(nu_p), lambda=FT(lambda),
            polymer_model=:oldroydb, L_max=FT(10.0),
            u_mean=FT(UMEAN), Fx_body=FT(0.0),
            bsd_fraction=FT(BSD), polymer_substeps=:auto,
            subcycle_relative_tolerance=FT(0.01),
            max_deformation_increment=FT(0.05),
            max_memory_deformation_increment=FT(0.07),
            max_polymer_substeps=64, max_steps, avg_window=max_steps,
            drag_stride=log_every, diagnostic_stride=0,
            embedded_geometry=:qwall, embedded_gradient=false,  # historical default — see m48-toggle-flip-postmortem
            embedded_advection=false, embedded_force=false, embedded_drag=false,
            advection_scheme=:muscl_superbee, wall_bc=:halfwayBB,
            embedded_circle_samples=32, force_boundary_fill=:bc_aware,
            step_callback=callback, backend, T=FT,
        )
        open(log_path, "w") do logio
            @printf(logio, "R=%d completed_steps=%d Cd_final_avg=%.16g Cd_s=%.16g Cd_p=%.16g Cd_bsd=%.16g walltime_s=%.3f\n",
                    R, result.completed_steps, result.Cd, result.Cd_s, result.Cd_p,
                    result.Cd_bsd, time() - t0)
        end
    end
    return time() - t0
end

function main()
    backend, FT = metal_backend()
    cases = [(10, 60_000, 500), (30, 180_000, 1000), (50, 300_000, 2000)]
    timings = Dict{Int,Float64}()
    for (R, max_steps, log_every) in cases
        @printf("[%s] M48 R=%d max_steps=%d log_every=%d\n",
                Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"), R, max_steps, log_every)
        timings[R] = run_R(backend, FT, R, max_steps, log_every)
    end
    @printf("M48 timings: R10=%.3f R30=%.3f R50=%.3f total=%.3f\n",
            timings[10], timings[30], timings[50], sum(values(timings)))
end

main()
