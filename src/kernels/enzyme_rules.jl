# Analytic geometry derivatives for shape optimization via chain rule.
#
# Enzyme differentiates the time-stepping loop but cannot handle the
# discontinuous geometry precomputation (ray-circle intersection with
# if/sqrt/ternary). Instead we provide analytic dq_wall/dR and chain:
#
#   dFx/dR = Σ_{i,j,q} (dFx/dq_wall[i,j,q]) · (dq_wall[i,j,q]/dR)
#
# where dFx/dq_wall comes from Enzyme reverse-mode on the time loop,
# and dq_wall/dR is the closed-form derivative of the ray-circle
# intersection parameter.

"""
    dq_wall_dR_cylinder(Nx, Ny, cx, cy, R; FT=Float64)
        -> dq_dR::Array{FT, 3}

Analytic derivative of `q_wall[i,j,q]` with respect to cylinder radius
`R`. For a cut link with quadratic `a·t²+b·t+c=0` where `c = |P-C|²-R²`:

    dt/dR = ∓ 2R / √(b²-4ac)

Sign depends on which root was selected (t1 vs t2). Returns zero for
non-cut links and for links where the discriminant is non-positive.
"""
function dq_wall_dR_cylinder(Nx::Int, Ny::Int,
                              cx::Real, cy::Real, R::Real;
                              FT::Type{<:AbstractFloat}=Float64)
    cxT, cyT, RT = FT(cx), FT(cy), FT(R)
    R² = RT * RT
    cxs = velocities_x(D2Q9())
    cys = velocities_y(D2Q9())

    dq_dR = zeros(FT, Nx, Ny, 9)
    @inbounds for j in 1:Ny, i in 1:Nx
        xf = FT(i - 1); yf = FT(j - 1)
        dx_f = xf - cxT; dy_f = yf - cyT
        dx_f * dx_f + dy_f * dy_f ≤ R² && continue
        for q in 2:9
            cqx = FT(cxs[q]); cqy = FT(cys[q])
            xn = xf + cqx; yn = yf + cqy
            dx_n = xn - cxT; dy_n = yn - cyT
            dx_n * dx_n + dy_n * dy_n > R² && continue

            a = cqx * cqx + cqy * cqy
            b_coeff = FT(2) * (dx_f * cqx + dy_f * cqy)
            c_coeff = dx_f * dx_f + dy_f * dy_f - R²
            disc = b_coeff * b_coeff - FT(4) * a * c_coeff
            disc ≤ zero(FT) && continue

            sd = sqrt(disc)
            t1 = (-b_coeff - sd) / (FT(2) * a)
            if t1 > zero(FT)
                dq_dR[i,j,q] = -FT(2) * RT / sd
            else
                dq_dR[i,j,q] =  FT(2) * RT / sd
            end
        end
    end
    return dq_dR
end
