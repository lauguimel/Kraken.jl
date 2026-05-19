#!/usr/bin/env julia

using Kraken
using KernelAbstractions
using Printf

# Top-level conditional Metal load: avoids the world-age trap that hits when
# `MetalBackend()` is constructed inside a function via invokelatest. The
# package is only attached on macOS; on linux/HPC we skip and try CUDA below.
const _LOAD_METAL = Sys.isapple()
if _LOAD_METAL
    try
        @eval using Metal
    catch
    end
end
const _LOAD_CUDA = !Sys.isapple()
if _LOAD_CUDA
    try
        @eval using CUDA
    catch
    end
end

function detect_backend()
    req = lowercase(get(ENV, "KRAKEN_BACKEND", "auto"))
    ft = lowercase(get(ENV, "KRAKEN_FT", "float64"))
    FT = ft in ("f32", "float32", "single") ? Float32 : Float64

    if (req == "auto" || req == "cuda") && isdefined(Main, :CUDA)
        try
            CUDAMod = getfield(Main, :CUDA)
            if CUDAMod.functional()
                return CUDAMod.CUDABackend(), "cuda", FT
            end
        catch
        end
    end

    if (req == "auto" || req == "metal") && isdefined(Main, :Metal)
        try
            MetalMod = getfield(Main, :Metal)
            if MetalMod.functional()
                return MetalMod.MetalBackend(), "metal", FT == Float64 ? Float32 : FT
            end
        catch
        end
    end

    if req != "auto" && req != "cpu"
        @warn "Requested KRAKEN_BACKEND=$req but detection failed; falling back to CPU."
    end
    return KernelAbstractions.CPU(), "cpu", FT
end

const BACKEND, BACKEND_LABEL, FT = detect_backend()

const RADIUS = 20
const H = 4 * RADIUS
const L_UP = 4.0
const L_DOWN = 4.0
const U_MEAN = 0.005
const NU_S = 0.05
const NU_P = 0.05
const LAMBDA = 400.0
const MAX_STEPS = 2000
const AVG_WINDOW = 400
const DRAG_STRIDE = 200
const TOL = 0.5

const CASES = (
    (label="A_baseline", embedded_force=false),
    (label="B_force_only", embedded_force=true),
)

function run_case(case_cfg)
    @printf("[M26b] starting %s backend=%s FT=%s\n", case_cfg.label, BACKEND_LABEL, string(FT))
    flush(stdout)

    result = Kraken.run_viscoelastic_logfv_cylinder_coupled_2d(;
        radius=RADIUS,
        H=H,
        L_up=L_UP,
        L_down=L_DOWN,
        u_mean=FT(U_MEAN),
        nu_s=FT(NU_S),
        nu_p=FT(NU_P),
        lambda=FT(LAMBDA),
        bsd_fraction=FT(1.0),
        polymer_model=:oldroydb,
        Fx_body=FT(0.0),
        polymer_substeps=:auto,
        max_steps=MAX_STEPS,
        avg_window=AVG_WINDOW,
        drag_stride=DRAG_STRIDE,
        diagnostic_stride=0,
        embedded_gradient=false,
        embedded_advection=false,
        embedded_force=case_cfg.embedded_force,
        embedded_drag=false,
        embedded_geometry=:circle,
        backend=BACKEND,
        T=FT,
    )

    @printf(
        "[M26b] %s Cd_kraken=%.8g Cd_s=%.8g completed=%d nan_step=%d\n",
        case_cfg.label,
        Float64(result.Cd),
        Float64(result.Cd_s),
        result.completed_steps,
        result.first_nonfinite_step,
    )
    flush(stdout)
    return result
end

function main()
    println("=== M26b embedded-force cell-fraction smoke ===")
    @printf(
        "backend=%s FT=%s radius=%d H=%d L_up=%.1f L_down=%.1f max_steps=%d avg_window=%d\n",
        BACKEND_LABEL, string(FT), RADIUS, H, L_UP, L_DOWN, MAX_STEPS, AVG_WINDOW,
    )
    @printf(
        "u_mean=%.6g Re=%.6g beta=%.6g Wi=%.6g lambda=%.6g\n",
        U_MEAN,
        U_MEAN * RADIUS / (NU_S + NU_P),
        NU_S / (NU_S + NU_P),
        U_MEAN * LAMBDA / RADIUS,
        LAMBDA,
    )
    flush(stdout)

    baseline = run_case(CASES[1])
    force_only = run_case(CASES[2])
    delta = Float64(force_only.Cd_s) - Float64(baseline.Cd_s)
    completed = baseline.completed_steps == MAX_STEPS && force_only.completed_steps == MAX_STEPS
    finite = isfinite(Float64(baseline.Cd_s)) && isfinite(Float64(force_only.Cd_s))
    pass = finite && completed && abs(delta) <= TOL

    @printf("[M26b] delta_Cd_s(B-A)=%.8g tolerance=%.3g\n", delta, TOL)
    println(pass ? "PASS M26b embedded_force delta within tolerance" :
                   "FAIL M26b embedded_force delta outside tolerance")
    flush(stdout)
    exit(pass ? 0 : 1)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
