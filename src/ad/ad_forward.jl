# Residual-converged steady forward solve for the AD cylinder path.
# CPU Float64 only; this iterates the inline AD step, not the fused kernel.

function ad_inlet_profile(Ny::Int, u_in::Real, inlet::Symbol)
    if inlet === :parabolic
        return [4.0 * Float64(u_in) * Float64(j - 1) * Float64(Ny - j) /
                Float64(Ny - 1)^2 for j in 1:Ny]
    elseif inlet === :uniform
        return fill(Float64(u_in), Ny)
    end
    throw(ArgumentError("unknown inlet $(inlet); expected :parabolic or :uniform"))
end

function ad_initial_cylinder_equilibrium(Nx::Int, Ny::Int, u_profile)
    f = zeros(Float64, Nx, Ny, 9)
    @inbounds for j in 1:Ny, i in 1:Nx
        rho = 1.0
        ux = u_profile[j]
        uy = 0.0
        usq = ux * ux
        f[i, j, 1] = ad_feq(Val(1), rho, ux, uy, usq)
        f[i, j, 2] = ad_feq(Val(2), rho, ux, uy, usq)
        f[i, j, 3] = ad_feq(Val(3), rho, ux, uy, usq)
        f[i, j, 4] = ad_feq(Val(4), rho, ux, uy, usq)
        f[i, j, 5] = ad_feq(Val(5), rho, ux, uy, usq)
        f[i, j, 6] = ad_feq(Val(6), rho, ux, uy, usq)
        f[i, j, 7] = ad_feq(Val(7), rho, ux, uy, usq)
        f[i, j, 8] = ad_feq(Val(8), rho, ux, uy, usq)
        f[i, j, 9] = ad_feq(Val(9), rho, ux, uy, usq)
    end
    return f
end

function ad_relative_step_residual(f_out, f_in)
    num = 0.0
    den = 0.0
    @inbounds for idx in eachindex(f_in)
        d = f_out[idx] - f_in[idx]
        num += d * d
        den += f_in[idx] * f_in[idx]
    end
    return sqrt(num) / sqrt(den)
end

function ad_geometry_cylinder(Nx::Int, Ny::Int, cx::Real, cy::Real, radius::Real)
    q_wall, is_solid =
        precompute_q_wall_cylinder(Nx, Ny, Float64(cx), Float64(cy),
                                   Float64(radius); FT=Float64)
    return (; q_wall=q_wall, is_solid=is_solid)
end

function ad_forward_solve(; Nx::Int, Ny::Int,
                            cx::Union{Nothing,Real}=nothing,
                            cy::Union{Nothing,Real}=nothing,
                            radius::Real,
                            u_in::Real,
                            nu::Real,
                            inlet::Symbol=:parabolic,
                            rho_out::Real=1.0,
                            tol::Real=1e-12,
                            max_steps::Int=120_000,
                            f_init=nothing)
    cx_f = Float64(isnothing(cx) ? Nx ÷ 4 : cx)
    cy_f = Float64(isnothing(cy) ? Ny ÷ 2 : cy)
    radius_f = Float64(radius)
    nu_f = Float64(nu)
    rho_out_f = Float64(rho_out)

    geom = ad_geometry_cylinder(Nx, Ny, cx_f, cy_f, radius_f)
    u_profile = ad_inlet_profile(Ny, u_in, inlet)
    u_ref = inlet === :parabolic ? (2.0 / 3.0) * Float64(u_in) : Float64(u_in)
    D = 2.0 * radius_f
    s_plus, s_minus = ad_trt_rates_inline(nu_f)

    f_in = f_init === nothing ?
        ad_initial_cylinder_equilibrium(Nx, Ny, u_profile) : copy(f_init)
    f_out = similar(f_in)
    residual = Inf
    n_iter = 0
    converged = false

    for step in 1:max_steps
        ad_step!(f_out, f_in, geom.q_wall, geom.is_solid, u_profile,
                 rho_out_f, s_plus, s_minus, Nx, Ny)
        residual = ad_relative_step_residual(f_out, f_in)
        n_iter = step
        if residual < Float64(tol)
            converged = true
            break
        end
        f_in, f_out = f_out, f_in
    end

    f_star = copy(f_out)
    cd = cd_pure(f_star, geom.q_wall, u_ref, D, Nx, Ny)

    return (;
        f_star=f_star,
        q_wall=geom.q_wall,
        is_solid=geom.is_solid,
        u_profile=u_profile,
        rho_out=rho_out_f,
        nu=nu_f,
        s_plus=s_plus,
        s_minus=s_minus,
        Nx=Nx,
        Ny=Ny,
        cx=cx_f,
        cy=cy_f,
        radius=radius_f,
        u_in=Float64(u_in),
        u_ref=u_ref,
        D=D,
        Cd=cd,
        n_iter=n_iter,
        residual=Float64(residual),
        converged=converged,
        inlet=inlet,
    )
end

