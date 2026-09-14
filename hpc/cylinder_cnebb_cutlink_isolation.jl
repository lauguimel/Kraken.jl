using Kraken
using Printf
using Statistics

function _parse_symbols(s::AbstractString)
    vals = strip(s)
    isempty(vals) && return Symbol[]
    return [Symbol(strip(x)) for x in split(vals, ",") if !isempty(strip(x))]
end

function _profile_value(profile::Symbol, x, y, cx, cy, R, amp)
    X = (x - cx) / R
    Y = (y - cy) / R
    if profile === :linear
        return 1.0 + amp * (0.8X - 0.6Y)
    elseif profile === :radial_quadratic
        r = sqrt((x - cx)^2 + (y - cy)^2)
        d = (r - R) / R
        return 1.0 + amp * d + 0.5amp * d^2
    else
        error("unknown profile $profile")
    end
end

function _cut_mask(q_wall)
    Nx, Ny, _ = size(q_wall)
    mask = falses(Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx, q in 2:9
        if q_wall[i, j, q] > 0
            mask[i, j] = true
        end
    end
    return mask
end

function _cut_links(q_wall)
    links = Tuple{Int,Int,Int,Float64}[]
    Nx, Ny, _ = size(q_wall)
    @inbounds for j in 1:Ny, i in 1:Nx, q in 2:9
        qw = q_wall[i, j, q]
        qw > 0 && push!(links, (i, j, q, Float64(qw)))
    end
    return links
end

function _residual_metrics(values, reference, mask)
    idx = findall(mask)
    err = abs.(values[idx] .- reference[idx])
    ref_scale = max(maximum(abs.(reference[idx])), eps(Float64))
    return (;
        l2 = sqrt(sum(abs2, values[idx] .- reference[idx]) / length(idx)) / ref_scale,
        max_abs = maximum(err),
        mean_abs = mean(err),
    )
end

function _population_error(g_post, C_ref, ux, uy, links)
    opp = Kraken.opposite(D2Q9())
    errors = Float64[]
    qws = Float64[]
    @inbounds for (i, j, q_into_wall, qw) in links
        q_missing = Int(opp[q_into_wall])
        expected = Kraken.equilibrium(D2Q9(), C_ref[i, j], ux[i, j], uy[i, j],
                                      q_missing)
        push!(errors, abs(g_post[i, j, q_missing] - expected))
        push!(qws, qw)
    end
    return (; errors, qws)
end

function _libb_branch(qw, f_here, f_back, f_opp_here)
    if qw <= 0.5
        return 2qw * f_here + (1 - 2qw) * f_back
    end
    inv_two_q = 1 / (2qw)
    return inv_two_q * f_here + (1 - inv_two_q) * f_opp_here
end

function _apply_cutlink_libb_field!(g_post, g_pre, q_wall, C_field)
    opp = Kraken.opposite(D2Q9())
    Nx, Ny, _ = size(g_post)
    @inbounds for j in 1:Ny, i in 1:Nx
        touched = false
        for q_into_wall in 2:9
            qw = q_wall[i, j, q_into_wall]
            qw == 0 && continue
            q_missing = Int(opp[q_into_wall])
            g_post[i, j, q_missing] = _libb_branch(qw,
                                                    g_pre[i, j, q_into_wall],
                                                    g_post[i, j, q_into_wall],
                                                    g_pre[i, j, q_missing])
            touched = true
        end
        if touched
            nonrest = 0.0
            for q in 2:9
                nonrest += g_post[i, j, q]
            end
            g_post[i, j, 1] = C_field[i, j] - nonrest
        end
    end
    return nothing
end

function _corr(x, y)
    length(x) < 2 && return NaN
    sx = std(x)
    sy = std(y)
    (sx == 0 || sy == 0) && return NaN
    return cor(x, y)
end

function run_case(; R::Int, Nx::Int, Ny::Int, cx::Float64, cy::Float64,
                  profile::Symbol, amp::Float64, phi_mode::Symbol)
    q_wall, is_solid = precompute_q_wall_cylinder(Nx, Ny, cx, cy, R)
    cut_mask = _cut_mask(q_wall)
    links = _cut_links(q_wall)

    C0 = zeros(Float64, Nx, Ny)
    ux = fill(0.02, Nx, Ny)
    uy = zeros(Float64, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        x = i - 1
        y = j - 1
        C0[i, j] = _profile_value(profile, x, y, cx, cy, R, amp)
    end

    g_pre = zeros(Float64, Nx, Ny, 9)
    @inbounds for q in 1:9, j in 1:Ny, i in 1:Nx
        g_pre[i, j, q] = Kraken.equilibrium(D2Q9(), C0[i, j], ux[i, j], uy[i, j], q)
    end
    g_post = similar(g_pre)
    stream_2d!(g_post, g_pre, Nx, Ny; sync=true)

    C_after = copy(C0)
    if phi_mode === :cutlink_libb_field
        _apply_cutlink_libb_field!(g_post, g_pre, q_wall, C_after)
    else
        apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, C_after, ux, uy;
                                     phi_mode)
    end

    macro_metrics = _residual_metrics(C_after, C0, cut_mask)
    pop = _population_error(g_post, C0, ux, uy, links)
    return (;
        profile, phi_mode,
        n_cells = count(cut_mask),
        n_links = length(links),
        qw_min = minimum(qw for (_, _, _, qw) in links),
        qw_max = maximum(qw for (_, _, _, qw) in links),
        qw_mean = mean(qw for (_, _, _, qw) in links),
        qw_far_from_half = count(abs(qw - 0.5) > 0.25 for (_, _, _, qw) in links),
        macro_l2 = macro_metrics.l2,
        macro_max_abs = macro_metrics.max_abs,
        macro_mean_abs = macro_metrics.mean_abs,
        pop_mean_abs = mean(pop.errors),
        pop_max_abs = maximum(pop.errors),
        pop_corr_qw = _corr(pop.qws, pop.errors),
        pop_corr_half = _corr(abs.(pop.qws .- 0.5), pop.errors),
    )
end

R = parse(Int, get(ENV, "KRAKEN_R", "30"))
Ny = parse(Int, get(ENV, "KRAKEN_NY", string(4R)))
Nx = parse(Int, get(ENV, "KRAKEN_NX", string(8R)))
cx = parse(Float64, get(ENV, "KRAKEN_CX", string(Nx / 4)))
cy = parse(Float64, get(ENV, "KRAKEN_CY", string(Ny / 2)))
amp = parse(Float64, get(ENV, "KRAKEN_PROFILE_AMP", "0.05"))
profiles = _parse_symbols(get(ENV, "KRAKEN_PROFILES", "linear,radial_quadratic"))
phi_modes = _parse_symbols(get(ENV, "KRAKEN_CNEBB_PHI_MODES", "pre_opp,field,eq_gradient,cutlink_libb_field"))

println("="^152)
println("Cylinder CNEBB cut-link isolation")
@printf("Nx=%d Ny=%d R=%d cx=%.6g cy=%.6g amp=%.6g\n", Nx, Ny, R, cx, cy, amp)
println("Profiles: $(join(string.(profiles), ", "))")
println("CNEBB phi modes: $(join(string.(phi_modes), ", "))")
println("="^152)
@printf("%-17s %-11s %-8s %-8s %-9s %-9s %-9s %-10s %-12s %-12s %-12s %-12s %-12s %-11s %-11s\n",
        "profile", "phi", "cells", "links", "qw_min", "qw_mean", "qw_max",
        "|qw-.5|>.25", "macro_l2", "macro_max", "pop_mean", "pop_max",
        "corr_qw", "corr_half", "note")
println("-"^152)

for profile in profiles
    for phi_mode in phi_modes
        result = run_case(; R, Nx, Ny, cx, cy, profile, amp, phi_mode)
        note = result.qw_far_from_half > 0 ? "cut-link" : "halfway"
        @printf("%-17s %-11s %-8d %-8d %-9.4f %-9.4f %-9.4f %-10d %-12.4e %-12.4e %-12.4e %-12.4e %-12.4f %-11.4f %-11s\n",
                string(result.profile), string(result.phi_mode),
                result.n_cells, result.n_links,
                result.qw_min, result.qw_mean, result.qw_max,
                result.qw_far_from_half,
                result.macro_l2, result.macro_max_abs,
                result.pop_mean_abs, result.pop_max_abs,
                result.pop_corr_qw, result.pop_corr_half, note)
    end
end

println("="^152)
println("Interpretation: production CNEBB sees only is_solid. q_wall enters the solvent LI-BB, not the polymer wall update.")
