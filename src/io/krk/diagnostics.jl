"""
    SanityIssue

A single validation issue found by `sanity_check`.
`level` is `:error`, `:warn`, or `:info`.
"""
struct SanityIssue
    level::Symbol      # :error, :warn, :info
    category::Symbol   # :relaxation, :compressibility, :resolution, :thermal, :twophase, :rheology, :refinement
    message::String
end

function _push_issue!(issues, level, category, msg)
    push!(issues, SanityIssue(level, category, msg))
end

# ── 1. Relaxation checks (τ, ω) ──────────────────────────────────────

function _check_relaxation!(issues, setup)
    ν = get(setup.physics.params, :nu, NaN)
    isnan(ν) && return
    τ = 3ν + 0.5
    ω = 1.0 / τ
    if τ < 0.5
        _push_issue!(issues, :error, :relaxation,
            "tau = $(round(τ, digits=4)) < 0.5 (unstable, ν = $(round(ν, digits=6)) is negative or zero). " *
            "Fix: increase ν, or use Setup reynolds with larger L_ref / smaller Re.")
    elseif τ < 0.51
        _push_issue!(issues, :warn, :relaxation,
            "tau = $(round(τ, digits=4)) is very close to 0.5 (marginally stable). " *
            "MRT collision recommended. Fix: increase ν from $(round(ν, digits=6)), " *
            "or reduce Re / increase N from $(setup.domain.Nx).")
    end
    if τ > 100.0
        _push_issue!(issues, :error, :relaxation,
            "tau = $(round(τ, digits=1)) is absurdly large (ω = $(round(ω, digits=6))). " *
            "Advective physics is dead — collision does almost nothing. " *
            "Fix: decrease ν (=$(round(ν, digits=4))), e.g. increase Re or decrease N.")
    elseif τ > 10.0
        _push_issue!(issues, :warn, :relaxation,
            "tau = $(round(τ, digits=2)) is very large (ω = $(round(ω, digits=4))). " *
            "Collision relaxes only $(round(ω*100, digits=1))% per step — numerical diffusion dominates. " *
            "Fix: decrease ν (=$(round(ν, digits=4))), e.g. increase Re or decrease N.")
    end
end

# ── 2. Compressibility checks (Ma, U_ref) ────────────────────────────

function _check_compressibility!(issues, setup)
    U_ref = _probe_U_ref(setup.boundaries)
    cs = 1.0 / sqrt(3.0)
    if U_ref < 1e-6 && U_ref > 0.0
        _push_issue!(issues, :warn, :compressibility,
            "U_ref = $(U_ref) is near zero — round-off errors will dominate in Float32. " *
            "Fix: increase U_ref or use Float64.")
    end
    # Textbook bound: U_ref ≤ 0.1 (Krüger et al. 2017)
    if U_ref > 0.3
        Ma = U_ref / cs
        _push_issue!(issues, :error, :compressibility,
            "U_ref = $(round(U_ref, digits=4)) > 0.3, Mach = $(round(Ma, digits=3)) (CFL-critical, will diverge). " *
            "Fix: decrease U_ref or increase N from $(setup.domain.Nx).")
    elseif U_ref > 0.1
        Ma = U_ref / cs
        _push_issue!(issues, :warn, :compressibility,
            "U_ref = $(round(U_ref, digits=4)) > 0.1, Mach = $(round(Ma, digits=3)) (compressibility errors likely). " *
            "Fix: decrease U_ref to ≤ 0.1 or increase grid N.")
    end
end

# ── 3. Spatial resolution checks ─────────────────────────────────────

function _check_resolution!(issues, setup)
    N = min(setup.domain.Nx, setup.domain.Ny)
    if N < 10
        _push_issue!(issues, :warn, :resolution,
            "N = $N is very coarse — insufficient for quantitative results.")
    end
    Re = get(setup.physics.params, :Re, NaN)
    if !isnan(Re) && Re > 0 && N / Re < 1.0
        _push_issue!(issues, :warn, :resolution,
            "N/Re = $(round(N/Re, digits=2)) < 1 — boundary layer under-resolved. " *
            "Fix: increase N (currently $N) or reduce Re (currently $(round(Re, digits=1))).")
    end
end

# ── 4. Thermal checks (if :thermal module) ───────────────────────────

function _check_thermal!(issues, setup)
    :thermal in setup.modules || return
    params = setup.physics.params
    α = get(params, :alpha, NaN)
    if isnan(α)
        _push_issue!(issues, :warn, :thermal,
            "Module thermal is active but no thermal diffusivity (alpha) found. " *
            "It may be computed internally, but verify your setup.")
        return
    end
    τ_α = 3α + 0.5
    if τ_α < 0.5
        _push_issue!(issues, :error, :thermal,
            "Thermal tau = $(round(τ_α, digits=4)) < 0.5 (unstable). " *
            "Fix: increase alpha (=$(round(α, digits=6))).")
    elseif τ_α < 0.51
        _push_issue!(issues, :warn, :thermal,
            "Thermal tau = $(round(τ_α, digits=4)) is marginally stable.")
    end
    if τ_α > 10.0
        _push_issue!(issues, :warn, :thermal,
            "Thermal tau = $(round(τ_α, digits=2)) is very large — thermal diffusion too fast for this grid.")
    end
    Pr = get(params, :Pr, NaN)
    if !isnan(Pr) && (Pr < 0.1 || Pr > 10.0)
        _push_issue!(issues, :warn, :thermal,
            "Pr = $(round(Pr, digits=3)) is extreme for SRT thermal LBM. " *
            "MRT collision recommended for accuracy.")
    end

    # --- Thermal boundary layer resolution check ---
    # For natural convection, δ_T ~ L / Ra^(1/4) (Pr ~ 1).
    # We want ≥ 3 cells across the BL → N ≥ 3 · Ra^(1/4).
    Ra = get(params, :Ra, NaN)
    if !isnan(Ra) && Ra > 0
        dom = setup.domain
        N_min = min(dom.Nx, dom.Ny, dom.Nz > 1 ? dom.Nz : typemax(Int))
        N_req = 3 * Ra^(0.25)
        # Account for refinement near walls: if a patch touches a thermal wall,
        # its effective N is N_base * max_ratio.
        max_ratio = isempty(setup.refinements) ? 1 : maximum(r.ratio for r in setup.refinements)
        N_eff = N_min * max_ratio
        if N_eff < N_req
            _push_issue!(issues, :warn, :thermal,
                "Thermal BL under-resolved: N_eff = $N_eff < 3·Ra^(1/4) = " *
                "$(round(N_req, digits=1)) for Ra = $(round(Ra, sigdigits=3)). " *
                "Fix: increase N, or add Refine near thermal walls.")
        end
    end
end

# ── 5. Two-phase checks (if :twophase_vof module) ────────────────────

function _check_twophase!(issues, setup)
    :twophase_vof in setup.modules || return
    params = setup.physics.params
    ν   = get(params, :nu, NaN)
    ν_l = get(params, :nu_l, ν)
    ν_g = get(params, :nu_g, ν)
    if !isnan(ν_l) && !isnan(ν_g) && ν_g > 0
        ratio_ν = ν_l / ν_g
        if ratio_ν > 100.0 || ratio_ν < 0.01
            _push_issue!(issues, :warn, :twophase,
                "Viscosity ratio ν_l/ν_g = $(round(ratio_ν, digits=1)) is extreme. " *
                "MRT collision strongly recommended.")
        end
        τ_g = 3ν_g + 0.5
        if τ_g < 0.51
            _push_issue!(issues, :warn, :twophase,
                "Gas-phase tau = $(round(τ_g, digits=4)) is marginally stable.")
        end
        if τ_g > 10.0
            _push_issue!(issues, :warn, :twophase,
                "Gas-phase tau = $(round(τ_g, digits=2)) is very large — over-diffusive gas phase.")
        end
    end
    ρ_l = get(params, :rho_l, NaN)
    ρ_g = get(params, :rho_g, NaN)
    if !isnan(ρ_l) && !isnan(ρ_g) && ρ_g > 0
        ratio_ρ = ρ_l / ρ_g
        if ratio_ρ > 100.0
            _push_issue!(issues, :warn, :twophase,
                "Density ratio ρ_l/ρ_g = $(round(ratio_ρ, digits=0)) > 100. " *
                "Pressure-based model (phase-field) recommended.")
        end
    end
end

# ── 6. Rheology checks ───────────────────────────────────────────────

function _check_rheology!(issues, setup)
    isempty(setup.rheology) && return
    for rs in setup.rheology
        # Estimate minimum viscosity from model bounds
        ν_min = get(rs.params, :nu_min, NaN)
        if !isnan(ν_min)
            τ_min = 3ν_min + 0.5
            if τ_min < 0.51
                _push_issue!(issues, :warn, :rheology,
                    "Rheology model $(rs.model) (phase=$(rs.phase)) has nu_min=$(round(ν_min, digits=6)) " *
                    "→ tau_min=$(round(τ_min, digits=4)). Local instability possible. " *
                    "Fix: increase nu_min or use MRT.")
            end
        end
    end
end

# ── 7. Refinement checks ─────────────────────────────────────────────

function _check_refinement!(issues, setup)
    isempty(setup.refinements) && return
    ν = get(setup.physics.params, :nu, NaN)
    isnan(ν) && return
    τ_base = 3ν + 0.5

    # Probe reference velocity for Ma check on fine grid
    U_ref = _probe_U_ref(setup.boundaries)
    cs = 1.0 / sqrt(3.0)

    # Base grid resolution
    dom = setup.domain
    N_base = max(dom.Nx, dom.Ny, dom.Nz > 1 ? dom.Nz : 0)

    # Thermal parameters (if thermal module active)
    is_thermal = :thermal in setup.modules
    params = setup.physics.params
    α_thermal = NaN
    τ_T_base = NaN
    if is_thermal
        Pr = get(params, :Pr, 0.71)
        α_thermal = haskey(params, :alpha) ? params[:alpha] : ν / Pr
        τ_T_base = 3 * α_thermal + 0.5
    end

    for ref in setup.refinements
        ratio = ref.ratio
        name = ref.name

        # --- Flow τ on fine grid ---
        τ_fine = ratio * (τ_base - 0.5) + 0.5
        if τ_fine < 0.51
            _push_issue!(issues, :warn, :refinement,
                "Patch '$name' (ratio=$ratio): fine-grid τ = $(round(τ_fine, digits=4)) " *
                "is marginally stable after Filippova-Hanel rescaling.")
        end
        if τ_fine > 10.0
            _push_issue!(issues, :warn, :refinement,
                "Patch '$name' (ratio=$ratio): fine-grid τ = $(round(τ_fine, digits=2)) " *
                "is very large — over-diffusive fine grid.")
        end

        # --- Thermal τ on fine grid ---
        if is_thermal && !isnan(τ_T_base)
            τ_T_fine = ratio * (τ_T_base - 0.5) + 0.5
            if τ_T_fine < 0.51
                _push_issue!(issues, :warn, :refinement,
                    "Patch '$name' (ratio=$ratio): fine-grid thermal τ_α = $(round(τ_T_fine, digits=4)) " *
                    "is marginally stable.")
            end
            if τ_T_fine > 10.0
                _push_issue!(issues, :warn, :refinement,
                    "Patch '$name' (ratio=$ratio): fine-grid thermal τ_α = $(round(τ_T_fine, digits=2)) " *
                    "is very large — over-diffusive thermal fine grid.")
            end
        end

        # --- Fine-grid resolution check ---
        N_fine = N_base * ratio
        Re = get(params, :Re, NaN)
        if !isnan(Re) && Re > 0 && N_fine / Re < 1.0
            _push_issue!(issues, :warn, :refinement,
                "Patch '$name' (ratio=$ratio): fine-grid N/Re = $(round(N_fine/Re, digits=2)) < 1 " *
                "— boundary layer may be under-resolved even on fine grid.")
        end

        # --- Compressibility on fine grid (Ma is preserved, dt_fine = dt/ratio) ---
        # Ma is invariant across refinement levels (same U_ref in lattice units)
        # but U_ref on fine grid = U_ref_coarse (acoustic scaling)
        # No additional check needed — Ma is already validated at coarse level
    end
end

# ── Parameter summary ─────────────────────────────────────────────────

function _print_parameter_summary(setup)
    dom = setup.domain
    params = setup.physics.params
    ν = get(params, :nu, NaN)
    τ = isnan(ν) ? NaN : 3ν + 0.5
    ω = isnan(τ) ? NaN : 1.0 / τ
    Re = get(params, :Re, NaN)
    U_ref = _probe_U_ref(setup.boundaries)
    cs = 1.0 / sqrt(3.0)
    Ma = U_ref / cs

    grid = dom.Nz > 1 ? "$(dom.Nx)×$(dom.Ny)×$(dom.Nz)" : "$(dom.Nx)×$(dom.Ny)"
    mods = isempty(setup.modules) ? "none" : join(string.(setup.modules), ", ")

    lines = String[]
    push!(lines, "N = $grid, lattice = $(setup.lattice)")
    if !isnan(ν)
        push!(lines, "ν = $(round(ν, digits=6)), τ = $(round(τ, digits=4)), ω = $(round(ω, digits=4))")
    end
    re_str = isnan(Re) ? "" : "Re = $(round(Re, digits=2)), "
    push!(lines, "$(re_str)Ma = $(round(Ma, digits=4)), U_ref = $(round(U_ref, digits=4))")
    push!(lines, "Modules: $mods")
    push!(lines, "Steps: $(setup.max_steps)")

    # Refinement patches
    if !isempty(setup.refinements)
        is_thermal = :thermal in setup.modules
        α_thermal = NaN
        if is_thermal && !isnan(ν)
            Pr = get(params, :Pr, 0.71)
            α_thermal = haskey(params, :alpha) ? params[:alpha] : ν / Pr
        end
        for ref in setup.refinements
            τ_fine = isnan(ν) ? NaN : ref.ratio * (τ - 0.5) + 0.5
            info = "  Refine '$(ref.name)': ratio=$(ref.ratio), τ_fine=$(round(τ_fine, digits=4))"
            if is_thermal && !isnan(α_thermal)
                τ_T_base = 3 * α_thermal + 0.5
                τ_T_fine = ref.ratio * (τ_T_base - 0.5) + 0.5
                info *= ", τ_T_fine=$(round(τ_T_fine, digits=4))"
            end
            dim = ref.is_3d ? "3D" : "2D"
            info *= " [$dim]"
            push!(lines, info)
        end
    end

    @info "LBM parameters\n  " * join(lines, "\n  ")
end

# ── Issue emission ────────────────────────────────────────────────────

function _emit_issues(issues)
    errors = String[]
    for issue in issues
        if issue.level === :error
            push!(errors, "[$(issue.category)] $(issue.message)")
            @error "Sanity check [$(issue.category)]: $(issue.message)"
        elseif issue.level === :warn
            @warn "Sanity check [$(issue.category)]: $(issue.message)"
        else
            @info "Sanity check [$(issue.category)]: $(issue.message)"
        end
    end
    if !isempty(errors)
        error("Sanity check failed with $(length(errors)) error(s):\n" *
              join(["  • " * e for e in errors], "\n"))
    end
end

# ── Main entry point ──────────────────────────────────────────────────

"""
    sanity_check(setup::SimulationSetup; verbose=true) -> Vector{SanityIssue}

Validate LBM parameters for a parsed setup.

Runs 7 families of checks:
1. **Relaxation** — τ too low (unstable) or too high (diffusion-dominated)
2. **Compressibility** — Mach number / CFL bounds
3. **Resolution** — grid points vs Reynolds number
4. **Thermal** — thermal τ and Prandtl range (if `:thermal` module)
5. **Two-phase** — viscosity/density ratios (if `:twophase_vof` module)
6. **Rheology** — local τ bounds from non-Newtonian models
7. **Refinement** — fine-grid τ after Filippova-Hanel rescaling

Returns a `Vector{SanityIssue}` for programmatic inspection.
Emits `@warn` for soft issues, throws `ErrorException` for critical ones,
and prints a parameter summary when `verbose=true`.
"""
function sanity_check(setup::SimulationSetup; verbose::Bool=true)
    issues = SanityIssue[]

    _check_relaxation!(issues, setup)
    _check_compressibility!(issues, setup)
    _check_resolution!(issues, setup)
    _check_thermal!(issues, setup)
    _check_twophase!(issues, setup)
    _check_rheology!(issues, setup)
    _check_refinement!(issues, setup)

    verbose && _print_parameter_summary(setup)
    _emit_issues(issues)

    return issues
end

"""
    sanity_check_sweep(setups::Vector{SimulationSetup}; verbose=true) -> Vector{Vector{SanityIssue}}

Validate all setups in a sweep. Prints a compact summary table and returns
per-setup issues. Does NOT throw on :error — instead marks them so the caller
can decide whether to skip or abort.
"""
function sanity_check_sweep(setups::Vector{SimulationSetup}; verbose::Bool=true)
    all_issues = Vector{SanityIssue}[]
    rows = String[]

    for (i, setup) in enumerate(setups)
        issues = SanityIssue[]
        _check_relaxation!(issues, setup)
        _check_compressibility!(issues, setup)
        _check_resolution!(issues, setup)
        _check_thermal!(issues, setup)
        _check_twophase!(issues, setup)
        _check_rheology!(issues, setup)
        _check_refinement!(issues, setup)
        push!(all_issues, issues)

        # Build compact row
        params = setup.physics.params
        ν = get(params, :nu, NaN)
        τ = isnan(ν) ? NaN : 3ν + 0.5
        Re = get(params, :Re, NaN)
        U_ref = _probe_U_ref(setup.boundaries)
        n_err = count(i -> i.level == :error, issues)
        n_warn = count(i -> i.level == :warn, issues)
        status = n_err > 0 ? "✗" : n_warn > 0 ? "⚠" : "✓"
        re_str = isnan(Re) ? "—" : string(round(Re, digits=2))
        push!(rows, "$status  #$i  Re=$re_str  τ=$(round(τ, digits=3))  U=$(round(U_ref, digits=4))  " *
                     "err=$n_err warn=$n_warn")
    end

    if verbose
        @info "Sweep sanity check ($(length(setups)) cases)\n  " * join(rows, "\n  ")
        # Emit warnings for problematic cases
        for (i, issues) in enumerate(all_issues)
            for issue in issues
                if issue.level == :error
                    @warn "Case #$i [$(issue.category)]: $(issue.message)"
                end
            end
        end
    end

    return all_issues
end

"""
    _parse_sweep(line) -> Pair{Symbol, Vector{Float64}}

Parse `Sweep param = [a, b, c]` into a (name, values) pair.
"""
function _parse_sweep(line::String)
    m = match(r"^Sweep\s+(\w+)\s*=\s*\[([^\]]+)\]", line)
    m === nothing && throw(ArgumentError("Cannot parse Sweep: $line"))
    key = Symbol(m.captures[1])
    vals = Float64[]
    for s in split(m.captures[2], ",")
        t = strip(s)
        isempty(t) && continue
        push!(vals, parse(Float64, t))
    end
    return key => vals
end

"""
    _expand_preset(line) -> Vector{String}

Expand a `Preset <name>` directive into a list of .krk lines.
Known presets: `cavity_2d`, `poiseuille_2d`, `couette_2d`,
`taylor_green_2d`, `rayleigh_benard_2d`, `natural_convection_2d`.
"""
function _expand_preset(line::String)
    tokens = split(line)
    length(tokens) >= 2 || throw(ArgumentError("Preset needs a name: $line"))
    name = tokens[2]
    known = ("cavity_2d", "poiseuille_2d", "couette_2d",
             "taylor_green_2d", "rayleigh_benard_2d",
             "natural_convection_2d")
    if name ∉ known
        sug = _suggest_name(name, known)
        msg = "Unknown Preset '$name'"
        sug !== nothing && (msg *= " (did you mean: $sug?)")
        throw(ArgumentError(msg))
    end
    return _preset_lines(name)
end

function _preset_lines(name::AbstractString)
    if name == "cavity_2d"
        return [
            "Simulation cavity_2d D2Q9",
            "Domain L = 1.0 x 1.0  N = 128 x 128",
            "Physics nu = 0.01",
            "Boundary north velocity(ux = 0.1, uy = 0)",
            "Boundary south wall",
            "Boundary east wall",
            "Boundary west wall",
            "Run 10000 steps",
        ]
    elseif name == "poiseuille_2d"
        return [
            "Simulation poiseuille_2d D2Q9",
            "Domain L = 4.0 x 1.0  N = 64 x 32",
            "Physics nu = 0.1 Fx = 1e-5",
            "Boundary x periodic",
            "Boundary south wall",
            "Boundary north wall",
            "Run 10000 steps",
        ]
    elseif name == "couette_2d"
        return [
            "Simulation couette_2d D2Q9",
            "Domain L = 1.0 x 1.0  N = 32 x 64",
            "Physics nu = 0.1",
            "Boundary x periodic",
            "Boundary south wall",
            "Boundary north velocity(ux = 0.05, uy = 0)",
            "Run 5000 steps",
        ]
    elseif name == "taylor_green_2d"
        return [
            "Simulation taylor_green_2d D2Q9",
            "Domain L = 1.0 x 1.0  N = 64 x 64",
            "Physics nu = 0.01",
            "Boundary x periodic",
            "Boundary y periodic",
            "Initial { ux = 0.05*sin(2*pi*x)*cos(2*pi*y) uy = -0.05*cos(2*pi*x)*sin(2*pi*y) }",
            "Run 5000 steps",
        ]
    elseif name == "rayleigh_benard_2d"
        return [
            "Simulation rayleigh_benard_2d D2Q9",
            "Domain L = 2.0 x 1.0  N = 128 x 64",
            "Physics nu = 0.02 Pr = 0.71 Ra = 1e5",
            "Module thermal",
            "Boundary x periodic",
            "Boundary south wall T = 1.0",
            "Boundary north wall T = 0.0",
            "Run 20000 steps",
        ]
    elseif name == "natural_convection_2d"
        return [
            "Simulation natural_convection_2d D2Q9",
            "Domain L = 1.0 x 1.0  N = 64 x 64",
            "Physics nu = 0.05 Pr = 0.71 Ra = 1e3",
            "Module thermal",
            "Boundary west wall T = 1.0",
            "Boundary east wall T = 0.0",
            "Boundary south wall",
            "Boundary north wall",
            "Run 10000 steps",
        ]
    end
    return String[]
end
