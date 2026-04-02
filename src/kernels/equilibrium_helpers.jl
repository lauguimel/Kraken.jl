"""
Shared inline helpers for D2Q9 equilibrium and moment computations.
Used across all collision kernels to avoid code duplication.
"""

# --- D2Q9 equilibrium distribution for each population ---

@inline function feq_2d(::Val{1}, ρ::T, ux::T, uy::T, usq::T) where T
    T(4.0/9.0) * ρ * (one(T) - T(1.5) * usq)
end

@inline function feq_2d(::Val{2}, ρ::T, ux::T, uy::T, usq::T) where T
    cu = ux
    T(1.0/9.0) * ρ * (one(T) + T(3) * cu + T(4.5) * cu * cu - T(1.5) * usq)
end

@inline function feq_2d(::Val{3}, ρ::T, ux::T, uy::T, usq::T) where T
    cu = uy
    T(1.0/9.0) * ρ * (one(T) + T(3) * cu + T(4.5) * cu * cu - T(1.5) * usq)
end

@inline function feq_2d(::Val{4}, ρ::T, ux::T, uy::T, usq::T) where T
    cu = -ux
    T(1.0/9.0) * ρ * (one(T) + T(3) * cu + T(4.5) * cu * cu - T(1.5) * usq)
end

@inline function feq_2d(::Val{5}, ρ::T, ux::T, uy::T, usq::T) where T
    cu = -uy
    T(1.0/9.0) * ρ * (one(T) + T(3) * cu + T(4.5) * cu * cu - T(1.5) * usq)
end

@inline function feq_2d(::Val{6}, ρ::T, ux::T, uy::T, usq::T) where T
    cu = ux + uy
    T(1.0/36.0) * ρ * (one(T) + T(3) * cu + T(4.5) * cu * cu - T(1.5) * usq)
end

@inline function feq_2d(::Val{7}, ρ::T, ux::T, uy::T, usq::T) where T
    cu = -ux + uy
    T(1.0/36.0) * ρ * (one(T) + T(3) * cu + T(4.5) * cu * cu - T(1.5) * usq)
end

@inline function feq_2d(::Val{8}, ρ::T, ux::T, uy::T, usq::T) where T
    cu = -ux - uy
    T(1.0/36.0) * ρ * (one(T) + T(3) * cu + T(4.5) * cu * cu - T(1.5) * usq)
end

@inline function feq_2d(::Val{9}, ρ::T, ux::T, uy::T, usq::T) where T
    cu = ux - uy
    T(1.0/36.0) * ρ * (one(T) + T(3) * cu + T(4.5) * cu * cu - T(1.5) * usq)
end

# --- Macroscopic moments from D2Q9 populations ---

@inline function moments_2d(f1::T, f2::T, f3::T, f4::T, f5::T,
                            f6::T, f7::T, f8::T, f9::T) where T
    ρ = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
    inv_ρ = one(T) / ρ
    ux = (f2 - f4 + f6 - f7 - f8 + f9) * inv_ρ
    uy = (f3 - f5 + f6 + f7 - f8 - f9) * inv_ρ
    return ρ, ux, uy
end

# --- Bounce-back swap for solid nodes ---

@inline function bounce_back_2d!(f, i, j)
    tmp2 = f[i,j,2]; f[i,j,2] = f[i,j,4]; f[i,j,4] = tmp2
    tmp3 = f[i,j,3]; f[i,j,3] = f[i,j,5]; f[i,j,5] = tmp3
    tmp6 = f[i,j,6]; f[i,j,6] = f[i,j,8]; f[i,j,8] = tmp6
    tmp7 = f[i,j,7]; f[i,j,7] = f[i,j,9]; f[i,j,9] = tmp7
end
