# M47 discriminator: one real LBM wall-BC step -> log-FV wall residual.
using DelimitedFiles
using KernelAbstractions
using Kraken

const OUTDIR = joinpath("scratch", "M47_runverdict")
const CSV = joinpath(OUTDIR, "PT_M47_bouzidi_logfv_wall_residual.csv")
const Nx = 64
const Ny = 40
const R = 6.0
const CX = 22.5
const CY = 19.5
const U0 = 0.01
const NU = 0.1
const LAMBDA = R / U0
const NSTEPS = 200
const LOG_EVERY = 20
const W = Float64[4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]
const CXV = Int[0, 1, 0, -1, 0, 1, -1, -1, 1]
const CYV = Int[0, 0, 1, 0, -1, 1, 1, -1, -1]

function feq(rho, ux, uy, q)
    cu = 3.0 * (CXV[q] * ux + CYV[q] * uy)
    usq = ux * ux + uy * uy
    return W[q] * rho * (1 + cu + 0.5 * cu * cu - 1.5 * usq)
end

function trace_from_log_fields(psixx, psixy, psiyy)
    trace_c = similar(psixx)
    for idx in eachindex(psixx)
        cxx, _, cyy = Kraken.logfv_exp_sym2_2d(psixx[idx], psixy[idx], psiyy[idx])
        trace_c[idx] = cxx + cyy
    end
    return trace_c
end

function initial_equilibrium(q_wall, is_solid)
    f = zeros(Float64, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx
        x = (i - 1) - CX
        y = (j - 1) - CY
        r2 = max(x * x + y * y, R * R)
        ux = is_solid[i, j] ? 0.0 : U0 * (1 - R^2 / r2)
        uy = is_solid[i, j] ? 0.0 : -U0 * R^2 * x * y / (r2 * r2)
        for q in 1:9
            f[i, j, q] = feq(1.0, ux, uy, q)
        end
    end
    return f
end

function wall_band_mask(q_wall, is_solid)
    mask = falses(Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        mask[i, j] = !is_solid[i, j] && any(q_wall[i, j, q] > 0 for q in 2:9)
    end
    return mask
end

function run_case(wall_bc::Symbol, q_wall, is_solid, wall_mask, bulk_mask)
    f_in = initial_equilibrium(q_wall, is_solid)
    f_out = similar(f_in)
    rho = ones(Float64, Nx, Ny)
    ux = zeros(Float64, Nx, Ny); uy = zeros(Float64, Nx, Ny)
    uwx = zeros(Float64, Nx, Ny, 9); uwy = zeros(Float64, Nx, Ny, 9)
    fx = zeros(Float64, Nx, Ny); fy = zeros(Float64, Nx, Ny)
    Kraken.fused_trt_libb_v2_guo_field_step!(
        f_out, f_in, rho, ux, uy, is_solid, q_wall, uwx, uwy, fx, fy, Nx, Ny, NU;
        wall_bc=wall_bc,
    )
    KernelAbstractions.synchronize(KernelAbstractions.CPU())
    Kraken.logfv_compute_macroscopic_forced_field_2d!(rho, ux, uy, f_out, fx, fy)

    bc = Kraken.fvfd_openx_wally_bcspec_2d()
    dudx, dudy, dvdx, dvdy = (zeros(Float64, Nx, Ny) for _ in 1:4)
    Kraken.fvfd_velocity_gradient_2d!(dudx, dudy, dvdx, dvdy, ux, uy, is_solid, 1.0, 1.0, bc)

    psixx = zeros(Float64, Nx, Ny); psixy = zeros(Float64, Nx, Ny); psiyy = zeros(Float64, Nx, Ny)
    nxtx = similar(psixx); nxty = similar(psixy); ntyy = similar(psiyy)
    model_code = Kraken.logfv_constitutive_model_code(:oldroydb)
    rows = Vector{NTuple{5, Float64}}()
    for step in 0:NSTEPS
        if step % LOG_EVERY == 0
            tc = trace_from_log_fields(psixx, psixy, psiyy)
            wall_raw = maximum(tc[wall_mask])
            bulk_raw = maximum(tc[bulk_mask])
            resid = maximum(abs.(tc[wall_mask] .- 2.0)) - maximum(abs.(tc[bulk_mask] .- 2.0))
            grad_wall = maximum(abs.(dudy[wall_mask])) + maximum(abs.(dvdx[wall_mask]))
            push!(rows, (Float64(step), wall_raw, bulk_raw, resid, grad_wall))
        end
        step == NSTEPS && break
        Kraken.logfv_step_constitutive_log_2d!(
            nxtx, nxty, ntyy, psixx, psixy, psiyy,
            dudx, dudy, dvdx, dvdy, LAMBDA, 1.0, model_code, Inf,
        )
        psixx, nxtx = nxtx, psixx
        psixy, nxty = nxty, psixy
        psiyy, ntyy = ntyy, psiyy
    end
    return rows
end

function run_PT_M47_bouzidi_logfv_wall_residual()::Bool
    mkpath(OUTDIR)
    q_bouzidi, is_solid = Kraken.precompute_q_wall_cylinder(Nx, Ny, CX, CY, R)
    q_half = ifelse.(q_bouzidi .> 0.0, 0.5, 0.0)
    wall_mask = wall_band_mask(q_bouzidi, is_solid)
    bulk_mask = .!is_solid .& .!wall_mask
    rows_h = run_case(:halfwayBB, q_half, is_solid, wall_mask, bulk_mask)
    rows_b = run_case(:bouzidi_fl_twopass, q_bouzidi, is_solid, wall_mask, bulk_mask)
    open(CSV, "w") do io
        println(io, "case,step,wall_max_trace_C,bulk_max_trace_C,wall_minus_bulk_excess,wall_grad_proxy")
        for row in rows_h
            println(io, join(("halfwayBB_q05", row...), ","))
        end
        for row in rows_b
            println(io, join(("bouzidi_fl_twopass", row...), ","))
        end
    end
    final_h = rows_h[end][4]
    final_b = rows_b[end][4]
    delta = final_b - final_h
    ratio = final_b / max(abs(final_h), 1e-12)
    criterion = delta > 1e-4 && ratio > 10
    println("PT_M47_bouzidi_logfv_wall_residual: residual_half=$(final_h), residual_bouzidi=$(final_b), delta=$(delta), ratio=$(ratio), pass=$(criterion)")
    return criterion
end

const PT_RESULT = run_PT_M47_bouzidi_logfv_wall_residual()
exit(PT_RESULT ? 0 : 1)
