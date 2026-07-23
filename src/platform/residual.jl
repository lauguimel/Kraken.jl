"""
    LBMGeomParams

Parameter bundle for the CPU-Float64 LBM Newtonian AD path. Wraps the geometry
(q_wall, is_solid), BC scalars, and TRT rates from `ad_forward_solve`. This is the
geometry-only rung; M-P2b extends to material parameters.
"""
struct LBMGeomParams
    q_wall::Array{Float64,3}
    is_solid::BitMatrix
    u_profile::Vector{Float64}
    rho_out::Float64
    s_plus::Float64
    s_minus::Float64
    Nx::Int
    Ny::Int
end

"""
    LBMThermalParams

Parameter bundle for the CPU-Float64 LBM thermal (Boussinesq) AD path.
Wraps the shared q_wall and ADNatconvParams from `ad_thermal_forward_solve`.
Both q_flow and q_therm in `ad_thermal_cut_step!` are this struct's `q_wall`.
"""
struct LBMThermalParams
    q_wall::Array{Float64,3}
    params::ADNatconvParams
    Nx::Int
    Ny::Int
end

"""
    LBMVEParams

Parameter bundle for the CPU-Float64 LBM viscoelastic (Oldroyd-B) AD path.
Wraps the ADVEEmbeddedGeom, q_wall, u_profile, and ADVECoupledParams from
`ad_ve_forward_solve` / `ad_ve_build_geom`.
"""
struct LBMVEParams
    g::ADVEEmbeddedGeom
    q_wall::Array{Float64,3}
    u_profile::Vector{Float64}
    p::ADVECoupledParams
end

"""
    LBMScalarParams

Parameter bundle for the Newtonian LBM AD path with FREE scalar viscosity ν.
Geometry fields are identical to `LBMGeomParams`; ν is the free DOF.
`s_plus` and `s_minus` are DERIVED from ν via `ad_trt_rates_inline` at construction —
never stored stale. Construct from `LBMGeomParams` + candidate ν:
  `LBMScalarParams(geom_p, ν)`.
"""
struct LBMScalarParams
    q_wall::Array{Float64,3}
    is_solid::BitMatrix
    u_profile::Vector{Float64}
    rho_out::Float64
    ν::Float64
    s_plus::Float64
    s_minus::Float64
    Nx::Int
    Ny::Int
end

function LBMScalarParams(geom::LBMGeomParams, ν::Float64)
    s_plus, s_minus = ad_trt_rates_inline(ν)
    LBMScalarParams(geom.q_wall, geom.is_solid, geom.u_profile,
                    geom.rho_out, ν, s_plus, s_minus, geom.Nx, geom.Ny)
end

function residual(problem, method::AbstractMethod, u, p)
    error("residual not implemented for $(typeof(method)) with p=$(typeof(p))")
end

function adjoint_vjp(problem, method::AbstractMethod, u_star, p, v)
    error("adjoint_vjp not implemented for $(typeof(method)) with p=$(typeof(p))")
end

"""
    residual(problem, ::LBM, f::Array{Float64,3}, p::LBMGeomParams) -> Array{Float64,3}

Newtonian LBM steady residual R(f,p) = f - G(f,p) on the CPU Float64 path.
At the converged fixed point, norm(residual)/norm(f) < forward convergence tol.
"""
function residual(::Any, ::LBM, f::Array{Float64,3}, p::LBMGeomParams)
    out = similar(f)
    ad_step!(out, f, p.q_wall, p.is_solid, p.u_profile, p.rho_out,
             p.s_plus, p.s_minus, p.Nx, p.Ny)
    return f .- out
end

"""
    adjoint_vjp(problem, ::LBM, f_star::Array{Float64,3}, p::LBMGeomParams, v::Array{Float64,3}) -> Array{Float64,3}

Steady-adjoint VJP for the Newtonian LBM: (I - ∂G/∂u)^T v.
Requires Enzyme loaded (delegates to `_ad_vjp_GtT`; throws without Enzyme).
"""
function adjoint_vjp(::Any, ::LBM, f_star::Array{Float64,3},
                     p::LBMGeomParams, v::Array{Float64,3})
    GtT_v = _ad_vjp_GtT(f_star, v, p.q_wall, p.is_solid, p.u_profile,
                         p.rho_out, p.s_plus, p.s_minus, p.Nx, p.Ny)
    return v .- GtT_v
end

"""
    residual(problem, ::LBM, f::Array{Float64,3}, p::LBMScalarParams) -> Array{Float64,3}

Newtonian LBM steady residual R(f,p) = f - G(f,p) on the CPU Float64 path.
Uses the s_plus/s_minus derived from p.ν. At the converged fixed point,
norm(residual)/norm(f) < forward convergence tol.
"""
function residual(::Any, ::LBM, f::Array{Float64,3}, p::LBMScalarParams)
    out = similar(f)
    ad_step!(out, f, p.q_wall, p.is_solid, p.u_profile, p.rho_out,
             p.s_plus, p.s_minus, p.Nx, p.Ny)
    return f .- out
end

"""
    adjoint_vjp(problem, ::LBM, f_star::Array{Float64,3}, p::LBMScalarParams, v::Array{Float64,3}) -> Array{Float64,3}

Steady-adjoint VJP for the Newtonian LBM with free ν: (I - ∂G/∂u)^T v.
Requires Enzyme loaded (delegates to `_ad_vjp_GtT`; throws without Enzyme).
"""
function adjoint_vjp(::Any, ::LBM, f_star::Array{Float64,3},
                     p::LBMScalarParams, v::Array{Float64,3})
    GtT_v = _ad_vjp_GtT(f_star, v, p.q_wall, p.is_solid, p.u_profile,
                         p.rho_out, p.s_plus, p.s_minus, p.Nx, p.Ny)
    return v .- GtT_v
end

"""
    LBMFieldParams

Parameter bundle for the Newtonian LBM AD path with FREE per-row viscosity field ν(y).
`nu_field` is a `Vector{Float64}` of length `Ny`; `s_plus_field` and `s_minus_field` are
derived per-row rate vectors from `ad_trt_rates_inline`. Construct from `LBMGeomParams` +
candidate ν vector: `LBMFieldParams(geom_p, nu_field)`.
Relationship to `LBMScalarParams`: analogous but free DOF is Ny scalars instead of 1.
"""
struct LBMFieldParams
    q_wall::Array{Float64,3}
    is_solid::BitMatrix
    u_profile::Vector{Float64}
    rho_out::Float64
    nu_field::Vector{Float64}
    s_plus_field::Vector{Float64}
    s_minus_field::Vector{Float64}
    Nx::Int
    Ny::Int
end

function LBMFieldParams(geom::LBMGeomParams, nu_field::Vector{Float64})
    Ny = geom.Ny
    length(nu_field) == Ny ||
        throw(ArgumentError("nu_field length $(length(nu_field)) must be Ny=$Ny"))
    s_plus_field  = [ad_trt_rates_inline(nu_field[j])[1] for j in 1:Ny]
    s_minus_field = [ad_trt_rates_inline(nu_field[j])[2] for j in 1:Ny]
    LBMFieldParams(geom.q_wall, geom.is_solid, geom.u_profile, geom.rho_out,
                   copy(nu_field), s_plus_field, s_minus_field, geom.Nx, geom.Ny)
end

"""
    residual(problem, ::LBM, f::Array{Float64,3}, p::LBMFieldParams) -> Array{Float64,3}

Newtonian LBM steady residual R(f,p) = f - G(f,p) using the per-row ν(y) step.
"""
function residual(::Any, ::LBM, f::Array{Float64,3}, p::LBMFieldParams)
    out = similar(f)
    ad_step_nufield!(out, f, p.q_wall, p.is_solid, p.u_profile, p.rho_out,
                     p.nu_field, p.Nx, p.Ny)
    return f .- out
end

"""
    adjoint_vjp(problem, ::LBM, f_star::Array{Float64,3}, p::LBMFieldParams, v::Array{Float64,3}) -> Array{Float64,3}

Steady-adjoint VJP for the Newtonian LBM with free per-row ν(y): (I - ∂G/∂f)^T v.
Uses `_ad_vjp_GtT_nufield` (Const nu_field — state VJP, exact for the field rates).
Requires Enzyme loaded.
"""
function adjoint_vjp(::Any, ::LBM, f_star::Array{Float64,3},
                     p::LBMFieldParams, v::Array{Float64,3})
    GtT_v = _ad_vjp_GtT_nufield(f_star, v, p.q_wall, p.is_solid, p.u_profile,
                                p.rho_out, p.nu_field, p.Nx, p.Ny)
    return v .- GtT_v
end

"""
    residual(problem, ::LBM, w::Vector{Float64}, p::LBMThermalParams) -> Vector{Float64}

Thermal LBM steady residual R(w,p) = w - G(w,p) (Boussinesq coupled f+g).
At the converged fixed point, norm(residual)/norm(w) < forward convergence tol.
"""
function residual(::Any, ::LBM, w::Vector{Float64}, p::LBMThermalParams)
    out = similar(w)
    ad_thermal_cut_step!(out, w, p.q_wall, p.q_wall, p.params)
    return w .- out
end

"""
    adjoint_vjp(problem, ::LBM, w_star::Vector{Float64}, p::LBMThermalParams, v::Vector{Float64}) -> Vector{Float64}

Steady-adjoint VJP for the thermal LBM: (I - ∂G/∂w)^T v.
Requires Enzyme loaded (delegates to `_ad_thermal_vjp_GtT`; throws without Enzyme).
"""
function adjoint_vjp(::Any, ::LBM, w_star::Vector{Float64},
                     p::LBMThermalParams, v::Vector{Float64})
    GtT_v = _ad_thermal_vjp_GtT(w_star, v, p.q_wall, p.q_wall, p.params)
    return v .- GtT_v
end

"""
    residual(problem, ::LBM, w::Vector{Float64}, p::LBMVEParams) -> Vector{Float64}

Viscoelastic LBM steady residual R(w,p) = w - G(w,p) (Oldroyd-B coupled f+ψ).
At the converged fixed point, norm(residual)/norm(w) < forward convergence tol.
"""
function residual(::Any, ::LBM, w::Vector{Float64}, p::LBMVEParams)
    out = similar(w)
    ad_ve_coupled_step!(out, w, p.g, p.q_wall, p.p, p.u_profile, 1.0, nothing)
    return w .- out
end

"""
    adjoint_vjp(problem, ::LBM, w_star::Vector{Float64}, p::LBMVEParams, v::Vector{Float64}) -> Vector{Float64}

Steady-adjoint VJP for the viscoelastic LBM: (I - ∂G/∂w)^T v.
Requires Enzyme loaded (delegates to `_ad_ve_vjp_GtT`; throws without Enzyme).
"""
function adjoint_vjp(::Any, ::LBM, w_star::Vector{Float64},
                     p::LBMVEParams, v::Vector{Float64})
    GtT_v = _ad_ve_vjp_GtT(w_star, v, p.g, p.q_wall, p.u_profile, p.p)
    return v .- GtT_v
end
