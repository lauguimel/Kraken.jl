using Kraken
using Printf

const CX2 = Int.(Kraken.velocities_x(D2Q9()))
const CY2 = Int.(Kraken.velocities_y(D2Q9()))
const OPP2 = Int.(Kraken.opposite(D2Q9()))

function _fill_equilibrium!(g, φ, ux, uy)
    Nx, Ny = size(φ)
    @inbounds for q in 1:9, j in 1:Ny, i in 1:Nx
        g[i, j, q] = Kraken.equilibrium(D2Q9(), φ[i, j], ux[i, j], uy[i, j], q)
    end
    return g
end

function _field_value(profile::Symbol, x, y, amp, kx, ky)
    if profile === :sin
        return 1.0 + amp * sin(kx * x + ky * y)
    elseif profile === :linear
        return 1.0 + amp * (x + 0.37y)
    elseif profile === :quadratic
        return 1.0 + amp * (x + 0.37y + 0.5x^2 - 0.2x*y)
    else
        error("unknown profile $(profile); expected :sin, :linear, or :quadratic")
    end
end

function _parse_q_values(raw)
    vals = strip(raw)
    isempty(vals) && return Float64[]
    return [parse(Float64, strip(x)) for x in split(vals, ",") if !isempty(strip(x))]
end

function _run_case(; qw, profile, mode, q_out, amp, kx, ky)
    Nx, Ny = 7, 7
    i, j = 4, 4
    q_missing = OPP2[q_out]
    sx = CX2[q_out]
    sy = CY2[q_out]

    is_solid = falses(Nx, Ny)
    is_solid[i + sx, j + sy] = true

    q_wall = zeros(Float64, Nx, Ny, 9)
    q_wall[i, j, q_out] = qw

    ux = zeros(Float64, Nx, Ny)
    uy = zeros(Float64, Nx, Ny)
    φ = zeros(Float64, Nx, Ny)
    @inbounds for jj in 1:Ny, ii in 1:Nx
        x = ii - i
        y = jj - j
        φ[ii, jj] = _field_value(profile, x, y, amp, kx, ky)
    end

    g_pre = _fill_equilibrium!(zeros(Float64, Nx, Ny, 9), φ, ux, uy)
    g_post = similar(g_pre)
    stream_2d!(g_post, g_pre, Nx, Ny; sync=true)

    C_after = copy(φ)
    if mode === :raw_libb_sum
        q_missing = OPP2[q_out]
        reconstructed = if qw <= 0.5
            2qw * g_pre[i, j, q_out] + (1 - 2qw) * g_post[i, j, q_out]
        else
            inv_two_q = 1 / (2qw)
            inv_two_q * g_pre[i, j, q_out] +
            (1 - inv_two_q) * g_pre[i, j, q_missing]
        end
        C_after[i, j] = g_post[i, j, 1] + reconstructed +
                        sum(g_post[i, j, q] for q in 2:9 if q != q_missing)
        g_post[i, j, q_missing] = reconstructed
    elseif mode === :legacy_pre_opp
        apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, C_after, ux, uy;
                                     phi_mode=:pre_opp)
    elseif mode === :qaware_pre_opp
        apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, q_wall, C_after,
                                     ux, uy; phi_mode=:pre_opp)
    elseif mode === :qaware_eq_gradient
        apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, q_wall, C_after,
                                     ux, uy; phi_mode=:eq_gradient)
    elseif mode === :qaware_field
        apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, q_wall, C_after,
                                     ux, uy; phi_mode=:field)
    else
        error("unknown mode $(mode)")
    end

    exact = φ[i, j]
    conservation = sum(g_post[i, j, q] for q in 1:9) - C_after[i, j]
    normal_moment = sum(CX2[q] * sx * g_post[i, j, q] +
                        CY2[q] * sy * g_post[i, j, q] for q in 1:9)
    return (;
        qw,
        q_minus_half = qw - 0.5,
        profile,
        mode,
        exact,
        after = C_after[i, j],
        err = C_after[i, j] - exact,
        abs_err = abs(C_after[i, j] - exact),
        conservation,
        normal_moment,
        q_out,
        q_missing,
    )
end

q_values = _parse_q_values(get(ENV, "KRAKEN_Q_VALUES", "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"))
profile = Symbol(get(ENV, "KRAKEN_PROFILE", "sin"))
amp = parse(Float64, get(ENV, "KRAKEN_AMP", "0.1"))
kx = parse(Float64, get(ENV, "KRAKEN_KX", "0.7"))
ky = parse(Float64, get(ENV, "KRAKEN_KY", "0.23"))
q_out = parse(Int, get(ENV, "KRAKEN_Q_OUT", "2"))
modes = [Symbol(strip(x)) for x in split(get(ENV, "KRAKEN_MODES",
    "legacy_pre_opp,qaware_pre_opp,raw_libb_sum,qaware_eq_gradient,qaware_field"), ",")]
results_dir = get(ENV, "KRAKEN_RESULTS_DIR", "tmp/cnebb_single_link_q_sweep")
mkpath(results_dir)
csv_path = joinpath(results_dir, "cnebb_single_link_q_sweep.csv")

rows = NamedTuple[]
for mode in modes, qw in q_values
    push!(rows, _run_case(; qw, profile, mode, q_out, amp, kx, ky))
end

open(csv_path, "w") do io
    println(io, "mode,profile,qw,q_minus_half,exact,after,err,abs_err,conservation,normal_moment,q_out,q_missing")
    for r in rows
        println(io, join((r.mode, r.profile, r.qw, r.q_minus_half, r.exact,
                          r.after, r.err, r.abs_err, r.conservation,
                          r.normal_moment, r.q_out, r.q_missing), ","))
    end
end

println("="^128)
println("CNEBB single-link q sweep")
println("profile=$(profile) amp=$(amp) kx=$(kx) ky=$(ky) q_out=$(q_out) q_missing=$(OPP2[q_out])")
println("CSV: $(csv_path)")
println("="^128)
@printf("%-18s %-8s %-12s %-12s %-12s %-12s %-12s\n",
        "mode", "q", "q-0.5", "err", "abs_err", "conserv", "normal_mom")
println("-"^128)
for r in rows
    @printf("%-18s %-8.3f %-12.3f %-12.4e %-12.4e %-12.4e %-12.4e\n",
            string(r.mode), r.qw, r.q_minus_half, r.err, r.abs_err,
            r.conservation, r.normal_moment)
end

println("-"^128)
for mode in modes
    mode_rows = [r for r in rows if r.mode === mode]
    half_row = only([r for r in mode_rows if abs(r.qw - 0.5) < 1e-12])
    max_abs = maximum(r.abs_err for r in mode_rows)
    @printf("%-18s err(q=.5)=% .4e max_abs=% .4e\n",
            string(mode), half_row.err, max_abs)
end
println("="^128)
