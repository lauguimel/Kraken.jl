"""
    ehd_hydrostatic_profiles(C, Ny; FT=Float64)

Analytic strong-injection hydrostatic EHD base state from Luo, Wu, Yi & Tan
(PRE 93, 023309, 2016), evaluated on node-on-wall coordinates
`y* = (j - 1) / (Ny - 1)`.
"""
function ehd_hydrostatic_profiles(C, Ny; FT=Float64)
    CT = FT(C)
    froot(b) = (FT(4) * CT / FT(3)) * sqrt(b) * ((one(FT) + b)^(FT(1.5)) - b^(FT(1.5))) - one(FT)
    lo = FT(1e-12)
    hi = one(FT)
    flo = froot(lo)
    fhi = froot(hi)
    while sign(flo) == sign(fhi) && hi < FT(1e12)
        hi *= FT(2)
        fhi = froot(hi)
    end
    sign(flo) == sign(fhi) && error("Unable to bracket hydrostatic EHD root for C=$(C).")
    for _ in 1:200
        mid = (lo + hi) / FT(2)
        fm = froot(mid)
        if abs(fm) <= FT(10) * eps(FT)
            lo = mid
            hi = mid
            break
        elseif sign(flo) == sign(fm)
            lo = mid
            flo = fm
        else
            hi = mid
        end
    end
    b = (lo + hi) / FT(2)
    a = FT(2) * CT * sqrt(b)
    y = [FT(j - 1) / FT(Ny - 1) for j in 1:Ny]
    q_star = [a / (FT(2) * CT * sqrt(yj + b)) for yj in y]
    E_star = [a * sqrt(yj + b) for yj in y]
    phi = [one(FT) - (FT(2) * a / FT(3)) * ((yj + b)^(FT(1.5)) - b^(FT(1.5))) for yj in y]
    return (y=y, b=b, a=a, q_star=q_star, E_star=E_star, phi=phi)
end

function _ehd_lattice_params(Ny, C, M, Ma_E, alpha, delta_U; FT)
    H = FT(Ny - 1)
    cs = inv(sqrt(FT(3)))
    K = FT(Ma_E) * H * cs / FT(delta_U)
    eps_e = (FT(M) * K)^2
    q_inj = FT(C) * eps_e * FT(delta_U) / H^2
    D = FT(alpha) * K * FT(delta_U)
    gamma = FT(0.3)
    tau_U = FT(3) * gamma + FT(0.5)
    nu_U = (tau_U - FT(0.5)) / FT(3)
    tau_q = FT(3) * D + FT(0.5)
    return (H=H, cs=cs, K=K, eps=eps_e, q_inj=q_inj, D=D,
            tau_U=tau_U, nu_U=nu_U, tau_q=tau_q,
            omega_U=inv(tau_U), omega_q=inv(tau_q))
end

function _fill_phi_populations!(f_cpu, phi, Nx, Ny, FT)
    w = (FT(4)/FT(9), FT(1)/FT(9), FT(1)/FT(9), FT(1)/FT(9), FT(1)/FT(9),
         FT(1)/FT(36), FT(1)/FT(36), FT(1)/FT(36), FT(1)/FT(36))
    for j in 1:Ny, i in 1:Nx, q in 1:9
        f_cpu[i, j, q] = w[q] * phi[j]
    end
    return f_cpu
end

function _charge_feq_host(q, ux, uy, qdir, FT)
    w = qdir == 1 ? FT(4)/FT(9) : (qdir <= 5 ? FT(1)/FT(9) : FT(1)/FT(36))
    cx = (FT(0), FT(1), FT(0), FT(-1), FT(0), FT(1), FT(-1), FT(-1), FT(1))[qdir]
    cy = (FT(0), FT(0), FT(1), FT(0), FT(-1), FT(1), FT(1), FT(-1), FT(-1))[qdir]
    cu3 = FT(3) * (cx * ux + cy * uy)
    return q * w * (one(FT) + cu3 + FT(0.5) * cu3^2 - FT(1.5) * (ux^2 + uy^2))
end

function _fill_charge_populations!(f_cpu, q_profile, E_profile, K, Nx, Ny, FT)
    for j in 1:Ny, i in 1:Nx, qdir in 1:9
        f_cpu[i, j, qdir] = _charge_feq_host(q_profile[j], zero(FT), K * E_profile[j], qdir, FT)
    end
    return f_cpu
end

function _rel_l2(a, b)
    num = sqrt(sum(abs2, a .- b))
    den = max(sqrt(sum(abs2, b)), floatmin(eltype(float.(b))))
    return num / den
end

"""
    run_ehd_hydrostatic_2d(; Nx=8, Ny=96, C=10.0, M=10.0, Ma_E=1e-2,
                            alpha=1e-4, charge_scheme=:srt, ...)

Run the uncoupled D2Q9/D2Q9 EHD hydrostatic base-state solve with `u=0`.
The electric potential uses the Jiachen pseudo-time Poisson DDF and wall-node
non-equilibrium extrapolation BCs (`phi_bottom=1`, `phi_top=0`). The charge
field uses full second-order drift equilibrium with either `:srt` or the
regularized collision. The hydrostatic validation uses SRT; regularized is kept
available because the default `alpha=1e-4` gives `tau_q - 0.5 = O(1e-4)`, where
the faithful MATLAB driver commonly uses regularization.

Steady state is declared when
`max(abs(q^{n+1}-q^n)) / max(abs(q^{n+1})) <= charge_tol` after a converged
inner phi solve. Wall values are node-on-wall; interior DDF profiles are compared
at the effective half-link samples `y*=(j-3/2)/(Ny-1)`, which is the grid where
the faithful wall-node NEE charge update matches the diffusion-free base state.
"""
function run_ehd_hydrostatic_2d(; Nx=8, Ny=96, C=10.0, M=10.0, Ma_E=1e-2,
                                  alpha=1e-4, delta_U=1.0,
                                  charge_scheme=:srt,
                                  max_steps=100000, charge_tol=1e-8,
                                  phi_tol=1e-4, phi_max_iter=10000,
                                  backend=KernelAbstractions.CPU(), FT=Float64)
    Nx < 3 && throw(ArgumentError("Nx must be at least 3."))
    Ny < 8 && throw(ArgumentError("Ny must be at least 8."))
    charge_scheme in (:srt, :regularized) ||
        throw(ArgumentError("charge_scheme must be :srt or :regularized."))

    p = _ehd_lattice_params(Ny, C, M, Ma_E, alpha, delta_U; FT=FT)
    p.tau_q <= FT(0.5) && error("Charge relaxation time must be greater than 0.5.")
    analytic = ehd_hydrostatic_profiles(C, Ny; FT=FT)
    q_init_profile = FT(p.q_inj) .* analytic.q_star
    Ey_init_profile = FT(delta_U) .* analytic.E_star ./ FT(p.H)
    y_compare = copy(analytic.y)
    for j in 2:(Ny - 1)
        y_compare[j] = FT(j) - FT(1.5)
        y_compare[j] /= FT(p.H)
    end
    q_star_compare = [analytic.a / (FT(2) * FT(C) * sqrt(yj + analytic.b)) for yj in y_compare]
    E_star_compare = [analytic.a * sqrt(yj + analytic.b) for yj in y_compare]
    phi_compare = [one(FT) - (FT(2) * analytic.a / FT(3)) *
                   ((yj + analytic.b)^(FT(1.5)) - analytic.b^(FT(1.5))) for yj in y_compare]
    q_analytic = FT(p.q_inj) .* q_star_compare
    Ey_analytic = FT(delta_U) .* E_star_compare ./ FT(p.H)

    phi_f_in = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    phi_f_out = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    q_f_in = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    q_f_out = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    phi = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    qfield = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Ex = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Ey = KernelAbstractions.zeros(backend, FT, Nx, Ny)

    phi_init = _fill_phi_populations!(zeros(FT, Nx, Ny, 9), analytic.phi, Nx, Ny, FT)
    q_init = _fill_charge_populations!(zeros(FT, Nx, Ny, 9), q_init_profile, Ey_init_profile, FT(p.K), Nx, Ny, FT)
    copyto!(phi_f_in, phi_init)
    copyto!(phi_f_out, phi_init)
    copyto!(q_f_in, q_init)
    copyto!(q_f_out, q_init)
    compute_ehd_scalar_2d!(phi, phi_f_in)
    compute_ehd_scalar_2d!(qfield, q_f_in)
    phi_cpu = Array(phi)
    q_cpu = Array(qfield)

    phi_iters_last = 0
    phi_rel_last = Inf
    q_rel = Inf
    steps_done = 0

    for step in 1:max_steps
        steps_done = step
        q_before = q_cpu

        for iter in 1:phi_max_iter
            phi_old = phi_cpu
            collide_electric_potential_2d!(phi_f_in, qfield, p.eps, p.omega_U, p.nu_U)
            stream_periodic_x_wall_y_2d!(phi_f_out, phi_f_in, Nx, Ny)
            compute_ehd_scalar_2d!(phi, phi_f_out)
            apply_phi_nee_walls_2d!(phi_f_out, phi, one(FT), zero(FT), Nx, Ny)
            compute_ehd_scalar_2d!(phi, phi_f_out)
            phi_cpu = Array(phi)
            denom = max(maximum(abs, phi_cpu), floatmin(FT))
            phi_rel_last = maximum(abs.(phi_cpu .- phi_old)) / denom
            phi_f_in, phi_f_out = phi_f_out, phi_f_in
            phi_iters_last = iter
            phi_rel_last <= phi_tol && break
            iter == phi_max_iter &&
                error("Electric potential solve did not converge within $(phi_max_iter) iterations. Last relative change: $(phi_rel_last).")
        end

        compute_electric_field_2d!(Ex, Ey, phi_f_in, p.tau_U)

        if charge_scheme == :srt
            collide_electric_charge_srt_2d!(q_f_in, Ex, Ey, p.tau_q, p.K)
        else
            collide_electric_charge_regularized_2d!(q_f_in, Ex, Ey, p.tau_q, p.K)
        end
        stream_periodic_x_wall_y_2d!(q_f_out, q_f_in, Nx, Ny)
        compute_ehd_scalar_2d!(qfield, q_f_out)
        apply_charge_nee_walls_2d!(q_f_out, qfield, Ex, Ey, p.q_inj, zero(FT), p.K, Nx, Ny)
        compute_ehd_scalar_2d!(qfield, q_f_out)
        q_cpu = Array(qfield)
        q_rel = maximum(abs.(q_cpu .- q_before)) / max(maximum(abs, q_cpu), floatmin(FT))
        q_f_in, q_f_out = q_f_out, q_f_in
        all(isfinite, q_cpu) || error("Charge field became non-finite at step $(step).")
        q_rel <= charge_tol && break
        step == max_steps &&
            error("EHD hydrostatic charge solve did not converge within $(max_steps) steps. Last relative change: $(q_rel).")
    end

    compute_electric_field_2d!(Ex, Ey, phi_f_in, p.tau_U)
    compute_ehd_scalar_2d!(phi, phi_f_in)
    compute_ehd_scalar_2d!(qfield, q_f_in)
    phi_cpu = Array(phi)
    q_cpu = Array(qfield)
    Ex_cpu = Array(Ex)
    Ey_cpu = Array(Ey)
    q_profile = [sum(@view q_cpu[:, j]) / FT(Nx) for j in 1:Ny]
    phi_profile = [sum(@view phi_cpu[:, j]) / FT(Nx) for j in 1:Ny]
    Ex_profile = [sum(@view Ex_cpu[:, j]) / FT(Nx) for j in 1:Ny]
    Ey_profile = [sum(@view Ey_cpu[:, j]) / FT(Nx) for j in 1:Ny]
    interior = 2:(Ny - 1)
    err_q = _rel_l2(q_profile[interior], q_analytic[interior])
    err_E = _rel_l2(Ey_profile[interior], Ey_analytic[interior])
    xvar_q = maximum(abs.(q_cpu .- reshape(q_profile, 1, Ny)))
    xvar_phi = maximum(abs.(phi_cpu .- reshape(phi_profile, 1, Ny)))

    return (q=q_cpu, phi=phi_cpu, Ex=Ex_cpu, Ey=Ey_cpu,
            q_profile=q_profile, phi_profile=phi_profile,
            Ex_profile=Ex_profile, Ey_profile=Ey_profile,
            analytic=(y=y_compare, b=analytic.b, a=analytic.a,
                      q_star=q_star_compare, E_star=E_star_compare,
                      phi=phi_compare, q=q_analytic, Ey=Ey_analytic,
                      y_node=analytic.y, phi_node=analytic.phi),
            err_q=err_q, err_E=err_E, xvar_q=xvar_q, xvar_phi=xvar_phi,
            steps=steps_done, q_rel_change=q_rel,
            phi_iters_last=phi_iters_last, phi_rel_last=phi_rel_last,
            Nx=Nx, Ny=Ny, C=C, M=M, Ma_E=Ma_E, alpha=alpha,
            charge_scheme=charge_scheme, bc=:non_equilibrium_extrapolation,
            y_mapping=:wall_nodes_interior_half_link, params=p)
end
