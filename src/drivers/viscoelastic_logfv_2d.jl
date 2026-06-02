using KernelAbstractions

@inline function _logfv_channel_shear(flow::Symbol, y, height, umax, uwall)
    if flow === :poiseuille
        return 4 * umax / height * (1 - 2 * y / height)
    elseif flow === :couette
        return uwall / height
    else
        error("unsupported log-FV channel flow $(flow); expected :poiseuille or :couette")
    end
end

function _logfv_compute_bsd_drag_2d(
    dudx, dudy, dvdx, dvdy, q_wall, Nx::Integer, Ny::Integer;
    cx::Real,
    cy::Real,
    radius::Real,
    zeta_nu_p::Real,
    reconstruction_order::Integer=2,
)
    zeta_nu_p_f = Float64(zeta_nu_p)
    zeta_nu_p_f == 0.0 && return (Fx=0.0, Fy=0.0)

    dudx_h = Array(dudx)
    dudy_h = Array(dudy)
    dvdx_h = Array(dvdx)
    dvdy_h = Array(dvdy)
    Nx_i = Int(Nx)
    Ny_i = Int(Ny)
    tau_bsd_xx = Matrix{Float64}(undef, Nx_i, Ny_i)
    tau_bsd_xy = Matrix{Float64}(undef, Nx_i, Ny_i)
    tau_bsd_yy = Matrix{Float64}(undef, Nx_i, Ny_i)
    @inbounds for j in 1:Ny_i, i in 1:Nx_i
        tau_bsd_xx[i, j] = 2.0 * zeta_nu_p_f * Float64(dudx_h[i, j])
        tau_bsd_xy[i, j] = zeta_nu_p_f * (Float64(dudy_h[i, j]) + Float64(dvdx_h[i, j]))
        tau_bsd_yy[i, j] = 2.0 * zeta_nu_p_f * Float64(dvdy_h[i, j])
    end

    return compute_polymeric_drag_2d(
        tau_bsd_xx, tau_bsd_xy, tau_bsd_yy, q_wall, Nx_i, Ny_i;
        cx=Float64(cx),
        cy=Float64(cy),
        radius=Float64(radius),
        extrapolate=true,
        reconstruction_order,
    )
end

function _logfv_first_nonfinite_field_2d(is_solid_h, fields::Pair{Symbol,<:Any}...)
    for pair in fields
        name = pair.first
        values = Array(pair.second)
        @inbounds for j in axes(values, 2), i in axes(values, 1)
            is_solid_h[i, j] && continue
            value = Float64(values[i, j])
            if !isfinite(value)
                return (finite=false, field=name, i=i, j=j)
            end
        end
    end
    return (finite=true, field=:none, i=0, j=0)
end

function _logfv_bsd_dual_path_relative_l2_2d(
    fx_active, fy_active, fx_alt, fy_alt, is_solid_h, backend,
)
    KernelAbstractions.synchronize(backend)
    fx_active_h = Array(fx_active)
    fy_active_h = Array(fy_active)
    fx_alt_h = Array(fx_alt)
    fy_alt_h = Array(fy_alt)
    Nx, Ny = size(fx_active_h)
    active_sum = 0.0
    delta_sum = 0.0
    @inbounds for j in 2:(Ny - 1), i in 2:(Nx - 1)
        is_solid_h[i, j] && continue
        ax = Float64(fx_active_h[i, j])
        ay = Float64(fy_active_h[i, j])
        dx = ax - Float64(fx_alt_h[i, j])
        dy = ay - Float64(fy_alt_h[i, j])
        active_sum += ax * ax + ay * ay
        delta_sum += dx * dx + dy * dy
    end
    active_l2 = sqrt(active_sum)
    delta_l2 = sqrt(delta_sum)
    return active_l2 > 0.0 ? delta_l2 / active_l2 : (delta_l2 == 0.0 ? 0.0 : Inf)
end

function _logfv_normalize_polymer_symbol(polymer_model)
    raw = lowercase(String(polymer_model))
    normalized = Symbol(replace(raw, '-' => '_'))
    normalized in (:oldroydb, :oldroyd_b, :oldroyd_benchmark, :ob) && return :oldroydb
    normalized in (:fenep, :fene_p, :fene_peterlin) && return :fenep
    throw(ArgumentError("unsupported log-FV polymer_model=$(polymer_model); expected :oldroydb or :fenep"))
end

function _logfv_polymer_model_config(polymer_model, L_max, ::Type{T}) where {T<:AbstractFloat}
    model_symbol = if polymer_model isa Symbol || polymer_model isa AbstractString
        _logfv_normalize_polymer_symbol(polymer_model)
    elseif polymer_model isa FENEPPolymer
        L_max = polymer_model.L_max
        :fenep
    elseif polymer_model isa AbstractPolymerModel
        :oldroydb
    elseif hasproperty(polymer_model, :L_max)
        L_max = getproperty(polymer_model, :L_max)
        :fenep
    elseif hasproperty(polymer_model, :lambda)
        :oldroydb
    else
        throw(ArgumentError("unsupported log-FV polymer_model=$(polymer_model); expected Symbol/String or polymer model object"))
    end

    model_code = logfv_constitutive_model_code(model_symbol)
    if model_symbol === :fenep
        L_max_t = T(L_max)
        isfinite(Float64(L_max_t)) || throw(ArgumentError("FENE-P requires finite L_max"))
        L_max_t > zero(T) || throw(ArgumentError("FENE-P requires positive L_max"))
        L2_t = L_max_t * L_max_t
        L2_t > T(2) || throw(ArgumentError("FENE-P requires L_max^2 > 2 in 2D"))
        return (; polymer_model=model_symbol, model_code, L_max=Float64(L_max_t), L2=L2_t)
    end
    return (; polymer_model=model_symbol, model_code, L_max=0.0, L2=zero(T))
end

function _logfv_conformation_diagnostics_2d(psixx, psixy, psiyy, is_solid_h, model_code, L2)
    Nx, Ny = size(psixx)
    min_c_eig = Inf
    max_c_trace = 0.0
    min_fene_denom = model_code == LOGFV_MODEL_FENEP ? Inf : NaN
    max_fene_factor = model_code == LOGFV_MODEL_FENEP ? -Inf : 1.0
    for j in 1:Ny, i in 1:Nx
        if !is_solid_h[i, j]
            cxx, cxy, cyy = logfv_exp_sym2_2d(psixx[i, j], psixy[i, j], psiyy[i, j])
            trc = Float64(cxx + cyy)
            min_c_eig = min(min_c_eig, logfv_min_eig_sym2_2d(cxx, cxy, cyy))
            max_c_trace = max(max_c_trace, trc)
            if model_code == LOGFV_MODEL_FENEP
                denom = Float64(L2) - trc
                min_fene_denom = min(min_fene_denom, denom)
                max_fene_factor = max(max_fene_factor, (Float64(L2) - 2.0) / denom)
            end
        end
    end
    return (; min_c_eig, max_c_trace, min_fene_denom, max_fene_factor)
end

function _logfv_embedded_circle_normal_alignment_2d(embedded, cx::Real, cy::Real)
    min_alignment = Inf
    sum_alignment = 0.0
    samples = 0
    cx_f = Float64(cx)
    cy_f = Float64(cy)
    @inbounds for idx in CartesianIndices(embedded.cut_count)
        embedded.cut_count[idx] > 0 || continue
        i, j = Tuple(idx)
        x = (Float64(i) - 0.5) - cx_f
        y = (Float64(j) - 0.5) - cy_f
        r = hypot(x, y)
        r > 0 || continue
        alignment = Float64(embedded.wall_nx[idx]) * x / r +
                    Float64(embedded.wall_ny[idx]) * y / r
        min_alignment = min(min_alignment, alignment)
        sum_alignment += alignment
        samples += 1
    end
    if samples == 0
        return (min=NaN, mean=NaN, samples=0)
    end
    return (min=min_alignment, mean=sum_alignment / samples, samples)
end

include("viscoelastic_logfv_coupled_step_2d.jl")
include("viscoelastic_logfv_step_wrappers_2d.jl")
include("viscoelastic_logfv_channel_2d.jl")
include("viscoelastic_logfv_frozen_circle_shear_2d.jl")
include("viscoelastic_logfv_frozen_circle_tangential_2d.jl")
include("viscoelastic_logfv_poiseuille_2d.jl")
include("viscoelastic_logfv_obstacle_bfs_2d.jl")
