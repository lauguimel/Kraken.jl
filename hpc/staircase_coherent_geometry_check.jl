using Kraken
using Printf
using Dates
using KernelAbstractions

const _METAL_MOD = if Sys.isapple()
    try
        @eval using Metal
        getfield(Main, :Metal)
    catch
        nothing
    end
else
    nothing
end

function _select_backend()
    requested = lowercase(get(ENV, "KRAKEN_BACKEND", "cpu"))
    if requested == "metal" && _METAL_MOD !== nothing
        if Base.invokelatest(getfield(_METAL_MOD, :functional))
            return Base.invokelatest(getfield(_METAL_MOD, :MetalBackend)), Float32, "Metal Float32"
        end
        error("Metal requested but not functional")
    end
    requested == "cpu" || @warn "Unsupported backend requested; falling back to CPU" requested
    return CPU(), Float64, "CPU Float64"
end

function _min_eig_c(Cxx, Cxy, Cyy)
    mineig = Inf
    maxtr = -Inf
    @inbounds for k in eachindex(Cxx)
        tr = Cxx[k] + Cyy[k]
        diff = Cxx[k] - Cyy[k]
        disc = sqrt(diff * diff + 4 * Cxy[k] * Cxy[k])
        mineig = min(mineig, 0.5 * (tr - disc))
        maxtr = max(maxtr, tr)
    end
    return (; mineig, maxtr)
end

function run_visco(; backend, FT, R, Nx, Ny, cx, cy, u_mean, beta, Wi,
                   tau_plus, max_steps, avg_window, drag_stride,
                   wall_geometry, conformation_collision)
    ν_total = u_mean * R
    ν_s = beta * ν_total
    ν_p = (1 - beta) * ν_total
    λ = Wi * R / u_mean
    result = run_conformation_cylinder_libb_2d(;
        Nx, Ny, radius=R, cx, cy,
        u_mean=FT(u_mean), ν_s=FT(ν_s), ν_p=FT(ν_p), lambda=FT(λ),
        tau_plus=FT(tau_plus), max_steps, avg_window, drag_stride,
        wall_geometry, conformation_collision,
        allow_diagnostic_conformation_collision=true,
        backend, FT)
    eig = _min_eig_c(result.C_xx, result.C_xy, result.C_yy)
    return merge(result, eig)
end

function run_modern_newtonian(; backend, FT, R, Nx, Ny, cx, cy, u_mean,
                              max_steps, avg_window, drag_stride,
                              wall_geometry, conformation_collision)
    ν_total = u_mean * R
    result = run_conformation_cylinder_libb_2d(;
        Nx, Ny, radius=R, cx, cy,
        u_mean=FT(u_mean), ν_s=FT(ν_total), ν_p=FT(0), lambda=FT(1),
        tau_plus=FT(1), max_steps, avg_window, drag_stride,
        wall_geometry, conformation_collision,
        allow_diagnostic_conformation_collision=true,
        backend, FT)
    return result
end

backend, FT, backend_label = _select_backend()
R = parse(Int, get(ENV, "KRAKEN_R", "10"))
Nx = parse(Int, get(ENV, "KRAKEN_NX", string(30R)))
Ny = parse(Int, get(ENV, "KRAKEN_NY", string(4R)))
cx = parse(Float64, get(ENV, "KRAKEN_CX", string(15R)))
cy = parse(Float64, get(ENV, "KRAKEN_CY", string(2R)))
u_mean = parse(Float64, get(ENV, "KRAKEN_U_MEAN", "0.005"))
beta = parse(Float64, get(ENV, "KRAKEN_BETA", "0.59"))
Wi = parse(Float64, get(ENV, "KRAKEN_WI", "0.1"))
Sc = parse(Float64, get(ENV, "KRAKEN_SC", "10000"))
steps_per_R = parse(Int, get(ENV, "KRAKEN_STEPS_PER_R", "300"))
avg_divisor = parse(Int, get(ENV, "KRAKEN_AVG_DIVISOR", "5"))
drag_stride = parse(Int, get(ENV, "KRAKEN_DRAG_STRIDE", "50"))
conformation_collision = Symbol(get(ENV, "KRAKEN_CONFORMATION_COLLISION", "trt"))
wall_geometries = Symbol.(split(get(ENV, "KRAKEN_WALL_GEOMETRIES", "cutlink,staircase"), ","))

max_steps = parse(Int, get(ENV, "KRAKEN_STEPS", string(steps_per_R * R)))
avg_window = max(1, max_steps ÷ avg_divisor)
ν_total = u_mean * R
ν_s = beta * ν_total
tau_sc = 0.5 + 3ν_s / Sc

println("="^132)
println("Coherent staircase geometry check")
println("Date/time: $(Dates.now())")
@printf("backend=%s FT=%s R=%d Nx=%d Ny=%d cx=%.6g cy=%.6g steps=%d avg=%d\n",
        backend_label, FT, R, Nx, Ny, cx, cy, max_steps, avg_window)
@printf("u=%.6g beta=%.6g Wi=%.6g Re_R=%.6g Re_D=%.6g Sc=%.6g tau_sc=%.9g\n",
        u_mean, beta, Wi, u_mean * R / ν_total, 2u_mean * R / ν_total, Sc, tau_sc)
println("wall geometries: $(join(string.(wall_geometries), ", "))")
println("conformation collision: $conformation_collision")
println("="^132)
@printf("%-11s %-14s %-12s %-13s %-13s %-13s %-13s %-13s %-13s\n",
        "wall", "case", "tau_plus", "Cd", "Cd/Cd_newt", "Cd_s", "Cd_p", "min_eig", "max_tr")
println("-"^132)

for wall_geometry in wall_geometries
    newt = run_modern_newtonian(;
        backend, FT, R, Nx, Ny, cx, cy, u_mean,
        max_steps, avg_window, drag_stride,
        wall_geometry, conformation_collision)
    @printf("%-11s %-14s %-12s %-13.6f %-13.6f %-13.6f %-13.6f %-13s %-13s\n",
            string(wall_geometry), "newtonian", "-", newt.Cd, 1.0,
            newt.Cd_s, newt.Cd_p, "-", "-")

    for (label, tau_plus) in (("visco_tau1", 1.0), ("visco_sc", tau_sc))
        result = try
            run_visco(; backend, FT, R, Nx, Ny, cx, cy, u_mean, beta, Wi,
                      tau_plus, max_steps, avg_window, drag_stride,
                      wall_geometry, conformation_collision)
        catch err
            @warn "visco coherent geometry case failed" wall_geometry label tau_plus err
            nothing
        end
        if result === nothing
            @printf("%-11s %-14s %-12.9g %-13s %-13s %-13s %-13s %-13s %-13s\n",
                    string(wall_geometry), label, tau_plus, "NaN", "NaN", "NaN", "NaN", "NaN", "NaN")
            continue
        end
        @printf("%-11s %-14s %-12.9g %-13.6f %-13.6f %-13.6f %-13.6f %-13.6g %-13.6g\n",
                string(wall_geometry), label, tau_plus, result.Cd, result.Cd / newt.Cd,
                result.Cd_s, result.Cd_p, result.mineig, result.maxtr)
    end
end
println("="^132)
println("Interpretation: only q_wall is changed between cutlink and staircase; the modern source/drag path is otherwise identical.")
