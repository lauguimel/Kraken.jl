include(joinpath(@__DIR__, "..", "src", "Kraken.jl"))
using .Kraken
using Printf
using Dates
using KernelAbstractions

const _CUDA_MOD = try
    @eval using CUDA
    getfield(Main, :CUDA)
catch
    nothing
end

function _select_backend()
    if _CUDA_MOD !== nothing
        try
            if Base.invokelatest(getfield(_CUDA_MOD, :functional))
                return getfield(_CUDA_MOD, :CUDABackend)(), Float64,
                       "CUDA $(getfield(_CUDA_MOD, :name)(getfield(_CUDA_MOD, :device)()))"
            end
        catch
        end
    end
    return KernelAbstractions.CPU(), Float64, "CPU"
end

function _parse_list(::Type{T}, value::AbstractString) where {T}
    return [parse(T, strip(x)) for x in split(value, ',') if !isempty(strip(x))]
end

function _bilinear(field, x, y)
    nx, ny = size(field)
    x = clamp(x, 0.0, nx - 1.0)
    y = clamp(y, 0.0, ny - 1.0)
    i0 = clamp(floor(Int, x) + 1, 1, nx)
    j0 = clamp(floor(Int, y) + 1, 1, ny)
    i1 = min(i0 + 1, nx)
    j1 = min(j0 + 1, ny)
    ax = x - (i0 - 1)
    ay = y - (j0 - 1)
    return (1 - ax) * (1 - ay) * field[i0, j0] +
           ax * (1 - ay) * field[i1, j0] +
           (1 - ax) * ay * field[i0, j1] +
           ax * ay * field[i1, j1]
end

function polymer_drag_circle(txx, txy, tyy; cx, cy, radius,
                             u_ref, nθ=8192, offset=0.5)
    Fx = 0.0
    Fy = 0.0
    dθ = 2π / nθ
    for k in 0:nθ-1
        θ = (k + 0.5) * dθ
        nx = cos(θ)
        ny = sin(θ)
        xw = cx + radius * nx
        yw = cy + radius * ny
        xs = xw + offset * nx
        ys = yw + offset * ny
        τxx = _bilinear(txx, xs, ys)
        τxy = _bilinear(txy, xs, ys)
        τyy = _bilinear(tyy, xs, ys)
        ds = radius * dθ
        Fx += (τxx * nx + τxy * ny) * ds
        Fy += (τxy * nx + τyy * ny) * ds
    end
    D = 2radius
    return (Fx=Fx, Fy=Fy, Cd=2Fx / (u_ref^2 * D))
end

function run_case(; backend, FT, R, formulation,
                  u_mean=0.005, beta=0.59, Wi=0.1,
                  steps_per_R=4000, avg_divisor=10, drag_stride=200,
                  drag_mode=:post_source_mea,
                  hermite_source_mode=:liu_direct,
                  solvent_source_mode=:post_collision,
                  conformation_magic=0.25,
                  momentum_exchange_mode=:mei_reconstruct)
    nu_total = u_mean * R
    nu_s = beta * nu_total
    nu_p = (1 - beta) * nu_total
    λ = Wi * R / u_mean
    G = nu_p / λ
    model = formulation == "logconf" ?
        LogConfOldroydB(G=FT(G), λ=FT(λ)) :
        OldroydB(G=FT(G), λ=FT(λ))
    Nx = 30R
    Ny = 4R
    max_steps = steps_per_R * R
    avg_window = max(1, max_steps ÷ avg_divisor)
    return run_conformation_cylinder_libb_2d(;
        Nx, Ny, radius=R, u_mean=FT(u_mean),
        ν_s=FT(nu_s), polymer_model=model,
        inlet=:parabolic, ρ_out=one(FT), tau_plus=one(FT),
        max_steps, avg_window, drag_stride, drag_mode, hermite_source_mode,
        solvent_source_mode, conformation_magic, momentum_exchange_mode,
        allow_diagnostic_log_wall_bc = formulation == "logconf",
        backend, FT)
end

backend, FT, backend_label = _select_backend()
R_values = _parse_list(Int, get(ENV, "KRAKEN_R_LIST", "20"))
formulations = split(get(ENV, "KRAKEN_FORMULATIONS", "direct"), ',')
steps_per_R = parse(Int, get(ENV, "KRAKEN_STEPS_PER_R", "4000"))
avg_divisor = parse(Int, get(ENV, "KRAKEN_AVG_DIVISOR", "10"))
drag_stride = parse(Int, get(ENV, "KRAKEN_DRAG_STRIDE", "200"))
drag_mode = Symbol(get(ENV, "KRAKEN_DRAG_MODE", "post_source_mea"))
hermite_source_mode = Symbol(get(ENV, "KRAKEN_HERMITE_SOURCE_MODE", "liu_direct"))
solvent_source_mode = Symbol(get(ENV, "KRAKEN_SOLVENT_SOURCE_MODE", "post_collision"))
conformation_magic = parse(Float64, get(ENV, "KRAKEN_CONFORMATION_MAGIC", "1e-6"))
momentum_exchange_mode = Symbol(get(ENV, "KRAKEN_MOMENTUM_EXCHANGE_MODE", "mei_reconstruct"))

liu_visco = Dict(20 => 129.42, 25 => 129.61, 30 => 130.36,
                 35 => 130.77, 40 => 130.79, 48 => 130.83)
rheotool_visco = 130.428774404

println("="^132)
println("Viscoelastic Cd decomposition")
println("Date/time: $(Dates.now())")
println("Backend: $backend_label, FT=$FT")
println("R values: $(join(R_values, ", "))")
println("Formulations: $(join(formulations, ", "))")
println("Reported Cd mode: $(String(drag_mode))")
println("Hermite source mode: $(String(hermite_source_mode))")
println("Solvent source mode: $(String(solvent_source_mode))")
println("Conformation magic: $(conformation_magic)")
println("Momentum exchange mode: $(String(momentum_exchange_mode))")
println("="^132)
@printf("%-8s %-4s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n",
        "case", "R", "Cd_ref", "Cd", "Cd_s", "need_p", "link_p",
        "circle0.5", "circle1.0", "circle1.5", "postMEA", "scaled", "err_link%", "err_c1%")
println("-"^132)

for R in R_values
    for formulation in formulations
        result = run_case(; backend, FT, R, formulation=strip(formulation),
                          steps_per_R, avg_divisor, drag_stride,
                          drag_mode, hermite_source_mode, solvent_source_mode,
                          conformation_magic,
                          momentum_exchange_mode)
        ref = get(liu_visco, R, rheotool_visco)
        cx = 30R / 4
        cy = 4R / 2
        c05 = polymer_drag_circle(result.tau_p_xx, result.tau_p_xy, result.tau_p_yy;
                                  cx, cy, radius=R, u_ref=result.u_ref, offset=0.5)
        c10 = polymer_drag_circle(result.tau_p_xx, result.tau_p_xy, result.tau_p_yy;
                                  cx, cy, radius=R, u_ref=result.u_ref, offset=1.0)
        c15 = polymer_drag_circle(result.tau_p_xx, result.tau_p_xy, result.tau_p_yy;
                                  cx, cy, radius=R, u_ref=result.u_ref, offset=1.5)
        need_p = ref - result.Cd_s
        err_link = (result.Cd_s + result.Cd_p - ref) / ref * 100
        err_c1 = (result.Cd_s + c10.Cd - ref) / ref * 100
        @printf("%-8s %-4d %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f %-10.3f %-10.3f\n",
                strip(formulation), R, ref, result.Cd, result.Cd_s, need_p, result.Cd_p,
                c05.Cd, c10.Cd, c15.Cd, result.Cd_mea_post_source,
                result.Cd_mea_source_scaled,
                err_link, err_c1)
        flush(stdout)
    end
end

println("="^132)
println("Done.")
