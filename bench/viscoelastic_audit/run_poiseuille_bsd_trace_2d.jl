using Kraken
using KernelAbstractions
using Printf

const OUT_DIR = joinpath(@__DIR__, "..", "scratch")
const CSV_COLUMNS = [
    "y_idx",
    "y",
    "ux_cell_mean",
    "uy_cell_mean",
    "F_poly_wide",
    "F_BSD_narrow",
    "F_total_nb",
    "F_poly_target",
    "F_BSD_target",
    "F_total_target",
    "rel_resid_F_poly",
    "rel_resid_F_total",
]

struct Case
    label::String
    nu_s::Float64
    nu_p::Float64
    zeta::Float64
    lambda::Float64
end

function high_wi_lambda(nu_s::Float64, nu_p::Float64, Fx_body::Float64, Ny::Int)
    nu_total = nu_s + nu_p
    gamma_dot_max = abs(Fx_body) * Ny / (2.0 * nu_total)
    gamma_dot_max > 0.0 || throw(ArgumentError("gamma_dot_max must be positive"))
    return 1.0 / gamma_dot_max
end

function build_cases(self_test::Bool; Ny::Int, Fx_body::Float64)
    if self_test
        return [Case("A_selftest", 0.1, 0.1, 0.75, 1.0)]
    end

    nu_s = 0.1
    nu_p = 0.1
    lambda_high_wi = high_wi_lambda(nu_s, nu_p, Fx_body, Ny)
    return [
        Case("A_no_BSD",  nu_s, nu_p, 0.00, 1.0),
        Case("A",         nu_s, nu_p, 0.75, 1.0),
        Case("A_high_Wi", nu_s, nu_p, 0.75, lambda_high_wi),
    ]
end

function row_means(a)
    Nx, Ny = size(a)
    out = Vector{Float64}(undef, Ny)
    @inbounds for j in 1:Ny
        s = 0.0
        for i in 1:Nx
            s += Float64(a[i, j])
        end
        out[j] = s / Nx
    end
    return out
end

function reconstruct_forces(res, c::Case, Fx_body::Float64)
    Nx = Int(res.Nx)
    Ny = Int(res.Ny)
    dx = 1.0
    dy = 1.0
    prefactor = c.nu_p / c.lambda

    tauxx = zeros(Float64, Nx, Ny)
    tauxy = zeros(Float64, Nx, Ny)
    tauyy = zeros(Float64, Nx, Ny)
    Kraken.logfv_stress_from_log_2d!(
        tauxx, tauxy, tauyy, res.psixx, res.psixy, res.psiyy, prefactor; sync=true,
    )

    fx_poly = zeros(Float64, Nx, Ny)
    fy_poly = zeros(Float64, Nx, Ny)
    is_solid = falses(Nx, Ny)
    Kraken.logfv_polymer_force_bc_aware_2d!(
        fx_poly, fy_poly, tauxx, tauxy, tauyy,
        is_solid, dx, dy, Kraken.logfv_periodicx_wally_bcspec_2d();
        sync=true, polymer_wall_extrap=:quadratic,
    )

    fx_total_nb = res.fx_total .- Fx_body
    fy_total_nb = res.fy_total
    fx_bsd = fx_poly .- fx_total_nb

    return (
        F_poly_x_mean=row_means(fx_poly),
        F_total_nb_x_mean=row_means(fx_total_nb),
        F_BSD_narrow_x_mean=row_means(fx_bsd),
        ux_mean=row_means(res.ux),
        uy_mean=row_means(res.uy),
        F_total_nb_y_mean=row_means(fy_total_nb),
    )
end

function run_case(c::Case; Nx::Int, Ny::Int, Fx_body::Float64, max_steps::Int)
    res = Kraken.run_viscoelastic_logfv_poiseuille_coupled_2d(;
        Nx=Nx,
        Ny=Ny,
        nu_s=c.nu_s,
        nu_p=c.nu_p,
        Fx_body=Fx_body,
        lambda=c.lambda,
        bsd_fraction=c.zeta,
        max_steps=max_steps,
        force_boundary_fill=:bc_aware,
        backend=KernelAbstractions.CPU(),
        T=Float64,
    )
    return res, reconstruct_forces(res, c, Fx_body)
end

rel_resid(value::Float64, target::Float64) =
    abs(value - target) / max(abs(target), eps(Float64))

function write_csv(
    path::AbstractString,
    Ny::Int,
    F_poly,
    F_BSD,
    F_total_nb,
    ux_mean,
    uy_mean,
    F_poly_tgt::Float64,
    F_BSD_tgt::Float64,
    F_total_tgt::Float64,
)
    open(path, "w") do io
        println(io, join(CSV_COLUMNS, ","))
        for j in 1:Ny
            y = (j - 0.5)
            rp = rel_resid(Float64(F_poly[j]), F_poly_tgt)
            rt = rel_resid(Float64(F_total_nb[j]), F_total_tgt)
            println(
                io,
                @sprintf(
                    "%d,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e",
                    j,
                    y,
                    ux_mean[j],
                    uy_mean[j],
                    F_poly[j],
                    F_BSD[j],
                    F_total_nb[j],
                    F_poly_tgt,
                    F_BSD_tgt,
                    F_total_tgt,
                    rp,
                    rt,
                ),
            )
        end
    end
    return nothing
end

function residual_vectors(Ny::Int, F_poly, F_total_nb, F_poly_tgt::Float64, F_total_tgt::Float64)
    rp = Vector{Float64}(undef, Ny)
    rt = Vector{Float64}(undef, Ny)
    @inbounds for j in 1:Ny
        rp[j] = rel_resid(Float64(F_poly[j]), F_poly_tgt)
        rt[j] = rel_resid(Float64(F_total_nb[j]), F_total_tgt)
    end
    return rp, rt
end

function summarize_case(
    label::String,
    Ny::Int,
    F_poly,
    F_total_nb,
    F_poly_tgt::Float64,
    F_total_tgt::Float64,
)
    interior = 3:(Ny - 2)
    walls = (1, 2, Ny - 1, Ny)
    rp, rt = residual_vectors(Ny, F_poly, F_total_nb, F_poly_tgt, F_total_tgt)
    rp_int = sqrt(sum(rp[j]^2 for j in interior) / length(interior))
    rt_int = sqrt(sum(rt[j]^2 for j in interior) / length(interior))
    rp_wall = maximum(rp[j] for j in walls)
    rt_wall = maximum(rt[j] for j in walls)
    line = @sprintf(
        "M20 %s: interior F_poly rel_resid_L2 = %.2e, wall F_poly rel_resid = %.2e, interior F_total rel_resid_L2 = %.2e, wall F_total rel_resid = %.2e",
        label,
        rp_int,
        rp_wall,
        rt_int,
        rt_wall,
    )
    println(line)
    return line
end

function targets(c::Case, Fx_body::Float64)
    nu_total = c.nu_s + c.nu_p
    F_poly_tgt = -c.nu_p * Fx_body / nu_total
    F_BSD_tgt = -c.zeta * c.nu_p * Fx_body / nu_total
    F_total_tgt = -(1.0 - c.zeta) * c.nu_p * Fx_body / nu_total
    return F_poly_tgt, F_BSD_tgt, F_total_tgt
end

function assert_selftest_csv(path::AbstractString)
    @assert isfile(path) "self-test CSV was not written at $(path)"
    lines = readlines(path)
    @assert !isempty(lines) "self-test CSV is empty at $(path)"

    header = split(lines[1], ",")
    @assert header == CSV_COLUMNS "self-test CSV header mismatch: got $(join(header, ","))"

    target_values = Float64[]
    for line in lines[2:end]
        cols = split(line, ",")
        @assert length(cols) == length(CSV_COLUMNS) "self-test CSV row has $(length(cols)) columns"
        y_idx = parse(Int, cols[1])
        if 3 <= y_idx <= 14
            push!(target_values, parse(Float64, cols[8]))
        end
    end
    @assert length(target_values) == 12 "self-test expected 12 interior rows in y_idx 3:14"

    mean_target = sum(target_values) / length(target_values)
    denom = max(abs(mean_target), eps(Float64))
    max_rel_dev = maximum(abs(v - mean_target) / denom for v in target_values)
    @assert max_rel_dev <= 0.05 "self-test F_poly_target max rel deviation $(max_rel_dev) > 0.05"
    return nothing
end

function parse_self_test(args)
    allowed = Set(["--full", "--self-test"])
    unknown = setdiff(args, allowed)
    isempty(unknown) || throw(ArgumentError("unknown argument(s): $(join(unknown, ", "))"))
    return !("--full" in args)
end

function main(args)
    self_test = parse_self_test(args)
    Nx, Ny, Fx_body, max_steps = if self_test
        (8, 16, 1.0e-5, 1000)
    else
        (8, 32, 1.0e-5, 100_000)
    end

    mkpath(OUT_DIR)
    summaries = String[]
    for c in build_cases(self_test; Ny=Ny, Fx_body=Fx_body)
        _, rec = run_case(c; Nx=Nx, Ny=Ny, Fx_body=Fx_body, max_steps=max_steps)
        F_poly_tgt, F_BSD_tgt, F_total_tgt = targets(c, Fx_body)
        path = joinpath(OUT_DIR, "poiseuille_bsd_trace_$(c.label).csv")
        write_csv(
            path,
            Ny,
            rec.F_poly_x_mean,
            rec.F_BSD_narrow_x_mean,
            rec.F_total_nb_x_mean,
            rec.ux_mean,
            rec.uy_mean,
            F_poly_tgt,
            F_BSD_tgt,
            F_total_tgt,
        )
        push!(
            summaries,
            summarize_case(
                c.label,
                Ny,
                rec.F_poly_x_mean,
                rec.F_total_nb_x_mean,
                F_poly_tgt,
                F_total_tgt,
            ),
        )
        self_test && assert_selftest_csv(path)
    end
    return summaries
end

main(ARGS)
