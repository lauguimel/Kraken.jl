function _build_spec(::Type{EHDSpec}, kw)
    FT = _get(kw, :FT, Float64)
    T_ehd = FT(_required(kw, :T_ehd))
    C = FT(_required(kw, :C))
    M = FT(_required(kw, :M))
    alpha = FT(_required(kw, :alpha))
    Ma_E = FT(_required(kw, :Ma_E))
    T_ehd > zero(FT) || throw(ArgumentError("T_ehd must be positive"))
    C > zero(FT) || throw(ArgumentError("C must be positive"))
    M > zero(FT) || throw(ArgumentError("M must be positive"))
    alpha > zero(FT) || throw(ArgumentError("alpha must be positive"))
    Ma_E > zero(FT) || throw(ArgumentError("Ma_E must be positive"))
    return EHDSpec{FT}(T_ehd, C, M, alpha, Ma_E)
end

_compile_with_spec(::EHDSpec, args...) = throw(phase_stub_error(:ehd_ec))
_audit_with_spec_type(::Type{EHDSpec}, args...) = throw(phase_stub_error(:ehd_ec))

function ehd_ec_lattice_params(spec::EHDSpec, Ny, delta_U, gamma; FT=Float64)
    H = FT(Ny - 1)
    cs = inv(sqrt(FT(3)))
    K = FT(spec.Ma_E) * H * cs / FT(delta_U)
    nu = FT(spec.M)^2 * K * FT(delta_U) / FT(spec.T_ehd)
    tau = FT(0.5) + FT(3) * nu
    eps_e = (FT(spec.M) * K)^2
    q_inj = FT(spec.C) * eps_e * FT(delta_U) / H^2
    D = FT(spec.alpha) * K * FT(delta_U)
    tau_U = FT(0.5) + FT(3) * FT(gamma)
    tau_q = FT(0.5) + FT(3) * D
    dt_star = K * FT(delta_U) / H^2
    return (H=H, cs=cs, K=K, nu=nu, tau=tau, omega=inv(tau),
            eps=eps_e, q_inj=q_inj, D=D, tau_U=tau_U, nu_U=FT(gamma),
            omega_U=inv(tau_U), tau_q=tau_q, omega_q=inv(tau_q),
            dt_star=dt_star, T_check=eps_e * FT(delta_U) / (nu * K),
            C_check=q_inj * H^2 / (eps_e * FT(delta_U)),
            M_check=sqrt(eps_e) / K, alpha_check=D / (K * FT(delta_U)))
end

_build_spec(::Type{MHDSpec}, kw) = throw(phase_stub_error(:mhd))
_compile_with_spec(::MHDSpec, args...) = throw(phase_stub_error(:mhd))
_audit_with_spec_type(::Type{MHDSpec}, args...) = throw(phase_stub_error(:mhd))

register_physics!(:ehd_ec, EHDSpec)
