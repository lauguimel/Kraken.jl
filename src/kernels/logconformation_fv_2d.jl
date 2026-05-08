# Cell-centered log-conformation FV/FD helpers for the production polymer
# backend. These functions are scalar, allocation-free, and GPU-compatible.

@inline function logfv_min_eig_sym2_2d(a, b, d)
    m = (a + d) / 2
    h = (a - d) / 2
    return m - hypot(h, b)
end

@inline function logfv_exp_sym2_2d(a, b, d)
    T = typeof(a + b + d)
    m = (a + d) / T(2)
    h = (a - d) / T(2)
    delta = hypot(h, b)
    em = exp(m)
    delta2 = delta * delta
    scale = ifelse(delta < sqrt(eps(T)), one(T) + delta2 / T(6), sinh(delta) / delta)
    ch = cosh(delta)
    return (
        em * (ch + scale * h),
        em * scale * b,
        em * (ch - scale * h),
    )
end

@inline function logfv_log_spd_sym2_2d(a, b, d)
    T = typeof(a + b + d)
    m = (a + d) / T(2)
    h = (a - d) / T(2)
    delta = hypot(h, b)
    lp = log(m + delta)
    lm = log(m - delta)
    alpha = (lp + lm) / T(2)
    beta = ifelse(
        delta < sqrt(eps(T)) * max(one(T), abs(m)),
        inv(m) + delta * delta / (T(3) * m * m * m),
        (lp - lm) / (T(2) * delta),
    )
    return (
        alpha + beta * h,
        beta * b,
        alpha - beta * h,
    )
end

@inline function logfv_oldroydb_relax_c_2d(cxx, cxy, cyy, lambda, dt)
    decay = exp(-dt / lambda)
    return (
        one(cxx) + (cxx - one(cxx)) * decay,
        cxy * decay,
        one(cyy) + (cyy - one(cyy)) * decay,
    )
end

@inline function logfv_oldroydb_relax_log_2d(psixx, psixy, psiyy, lambda, dt)
    cxx, cxy, cyy = logfv_exp_sym2_2d(psixx, psixy, psiyy)
    rxx, rxy, ryy = logfv_oldroydb_relax_c_2d(cxx, cxy, cyy, lambda, dt)
    return logfv_log_spd_sym2_2d(rxx, rxy, ryy)
end

@inline function logfv_oldroydb_source_c_2d(cxx, cxy, cyy, dudx, dudy, dvdx, dvdy, lambda)
    inv_lambda = inv(lambda)
    return (
        2 * (cxx * dudx + cxy * dudy) - inv_lambda * (cxx - one(cxx)),
        cxx * dvdx + cyy * dudy + cxy * (dudx + dvdy) - inv_lambda * cxy,
        2 * (cxy * dvdx + cyy * dvdy) - inv_lambda * (cyy - one(cyy)),
    )
end

@inline function logfv_upwind_scalar_advective_rhs_2d(phi, ux_face, uy_face, i, j)
    ue = ux_face[i + 1, j]
    uw = ux_face[i, j]
    vn = uy_face[i, j + 1]
    vs = uy_face[i, j]

    phie = ifelse(ue >= 0, phi[i, j], phi[i + 1, j])
    phiw = ifelse(uw >= 0, phi[i - 1, j], phi[i, j])
    phin = ifelse(vn >= 0, phi[i, j], phi[i, j + 1])
    phis = ifelse(vs >= 0, phi[i, j - 1], phi[i, j])

    flux_div = ue * phie - uw * phiw + vn * phin - vs * phis
    divu = ue - uw + vn - vs
    return -(flux_div - phi[i, j] * divu)
end

@inline function logfv_upwind_tensor_advective_rhs_2d(psixx, psixy, psiyy, ux_face, uy_face, i, j)
    return (
        logfv_upwind_scalar_advective_rhs_2d(psixx, ux_face, uy_face, i, j),
        logfv_upwind_scalar_advective_rhs_2d(psixy, ux_face, uy_face, i, j),
        logfv_upwind_scalar_advective_rhs_2d(psiyy, ux_face, uy_face, i, j),
    )
end
