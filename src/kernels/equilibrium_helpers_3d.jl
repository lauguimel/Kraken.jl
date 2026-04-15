# =====================================================================
# D3Q19 equilibrium + moments helpers (inline @kernel-friendly).
#
# D3Q19 indexing (Julia 1-based) matches src/lattice/d3q19.jl:
#   q=1:  rest        c=(0, 0, 0)    w=1/3
#   q=2:  +x          c=(1, 0, 0)    w=1/18
#   q=3:  -x          c=(-1, 0, 0)   w=1/18
#   q=4:  +y          c=(0, 1, 0)    w=1/18
#   q=5:  -y          c=(0, -1, 0)   w=1/18
#   q=6:  +z          c=(0, 0, 1)    w=1/18
#   q=7:  -z          c=(0, 0, -1)   w=1/18
#   q=8:  +x+y        c=(1, 1, 0)    w=1/36
#   q=9:  -x+y        c=(-1, 1, 0)   w=1/36
#   q=10: +x-y        c=(1, -1, 0)   w=1/36
#   q=11: -x-y        c=(-1, -1, 0)  w=1/36
#   q=12: +x+z        c=(1, 0, 1)    w=1/36
#   q=13: -x+z        c=(-1, 0, 1)   w=1/36
#   q=14: +x-z        c=(1, 0, -1)   w=1/36
#   q=15: -x-z        c=(-1, 0, -1)  w=1/36
#   q=16: +y+z        c=(0, 1, 1)    w=1/36
#   q=17: -y+z        c=(0, -1, 1)   w=1/36
#   q=18: +y-z        c=(0, 1, -1)   w=1/36
#   q=19: -y-z        c=(0, -1, -1)  w=1/36
# =====================================================================

@inline function feq_3d(::Val{1}, ρ::T, ux::T, uy::T, uz::T, usq::T) where T
    T(1.0/3.0) * ρ * (one(T) - T(1.5) * usq)
end
@inline function feq_3d(::Val{2}, ρ::T, ux::T, uy::T, uz::T, usq::T) where T
    cu = ux
    T(1.0/18.0) * ρ * (one(T) + T(3) * cu + T(4.5) * cu * cu - T(1.5) * usq)
end
@inline function feq_3d(::Val{3}, ρ::T, ux::T, uy::T, uz::T, usq::T) where T
    cu = -ux
    T(1.0/18.0) * ρ * (one(T) + T(3) * cu + T(4.5) * cu * cu - T(1.5) * usq)
end
@inline function feq_3d(::Val{4}, ρ::T, ux::T, uy::T, uz::T, usq::T) where T
    cu = uy
    T(1.0/18.0) * ρ * (one(T) + T(3) * cu + T(4.5) * cu * cu - T(1.5) * usq)
end
@inline function feq_3d(::Val{5}, ρ::T, ux::T, uy::T, uz::T, usq::T) where T
    cu = -uy
    T(1.0/18.0) * ρ * (one(T) + T(3) * cu + T(4.5) * cu * cu - T(1.5) * usq)
end
@inline function feq_3d(::Val{6}, ρ::T, ux::T, uy::T, uz::T, usq::T) where T
    cu = uz
    T(1.0/18.0) * ρ * (one(T) + T(3) * cu + T(4.5) * cu * cu - T(1.5) * usq)
end
@inline function feq_3d(::Val{7}, ρ::T, ux::T, uy::T, uz::T, usq::T) where T
    cu = -uz
    T(1.0/18.0) * ρ * (one(T) + T(3) * cu + T(4.5) * cu * cu - T(1.5) * usq)
end
@inline function feq_3d(::Val{8}, ρ::T, ux::T, uy::T, uz::T, usq::T) where T
    cu = ux + uy
    T(1.0/36.0) * ρ * (one(T) + T(3) * cu + T(4.5) * cu * cu - T(1.5) * usq)
end
@inline function feq_3d(::Val{9}, ρ::T, ux::T, uy::T, uz::T, usq::T) where T
    cu = -ux + uy
    T(1.0/36.0) * ρ * (one(T) + T(3) * cu + T(4.5) * cu * cu - T(1.5) * usq)
end
@inline function feq_3d(::Val{10}, ρ::T, ux::T, uy::T, uz::T, usq::T) where T
    cu = ux - uy
    T(1.0/36.0) * ρ * (one(T) + T(3) * cu + T(4.5) * cu * cu - T(1.5) * usq)
end
@inline function feq_3d(::Val{11}, ρ::T, ux::T, uy::T, uz::T, usq::T) where T
    cu = -ux - uy
    T(1.0/36.0) * ρ * (one(T) + T(3) * cu + T(4.5) * cu * cu - T(1.5) * usq)
end
@inline function feq_3d(::Val{12}, ρ::T, ux::T, uy::T, uz::T, usq::T) where T
    cu = ux + uz
    T(1.0/36.0) * ρ * (one(T) + T(3) * cu + T(4.5) * cu * cu - T(1.5) * usq)
end
@inline function feq_3d(::Val{13}, ρ::T, ux::T, uy::T, uz::T, usq::T) where T
    cu = -ux + uz
    T(1.0/36.0) * ρ * (one(T) + T(3) * cu + T(4.5) * cu * cu - T(1.5) * usq)
end
@inline function feq_3d(::Val{14}, ρ::T, ux::T, uy::T, uz::T, usq::T) where T
    cu = ux - uz
    T(1.0/36.0) * ρ * (one(T) + T(3) * cu + T(4.5) * cu * cu - T(1.5) * usq)
end
@inline function feq_3d(::Val{15}, ρ::T, ux::T, uy::T, uz::T, usq::T) where T
    cu = -ux - uz
    T(1.0/36.0) * ρ * (one(T) + T(3) * cu + T(4.5) * cu * cu - T(1.5) * usq)
end
@inline function feq_3d(::Val{16}, ρ::T, ux::T, uy::T, uz::T, usq::T) where T
    cu = uy + uz
    T(1.0/36.0) * ρ * (one(T) + T(3) * cu + T(4.5) * cu * cu - T(1.5) * usq)
end
@inline function feq_3d(::Val{17}, ρ::T, ux::T, uy::T, uz::T, usq::T) where T
    cu = -uy + uz
    T(1.0/36.0) * ρ * (one(T) + T(3) * cu + T(4.5) * cu * cu - T(1.5) * usq)
end
@inline function feq_3d(::Val{18}, ρ::T, ux::T, uy::T, uz::T, usq::T) where T
    cu = uy - uz
    T(1.0/36.0) * ρ * (one(T) + T(3) * cu + T(4.5) * cu * cu - T(1.5) * usq)
end
@inline function feq_3d(::Val{19}, ρ::T, ux::T, uy::T, uz::T, usq::T) where T
    cu = -uy - uz
    T(1.0/36.0) * ρ * (one(T) + T(3) * cu + T(4.5) * cu * cu - T(1.5) * usq)
end

@inline function moments_3d(f1::T, f2::T, f3::T, f4::T, f5::T, f6::T, f7::T,
                             f8::T, f9::T, f10::T, f11::T, f12::T, f13::T,
                             f14::T, f15::T, f16::T, f17::T, f18::T, f19::T) where T
    ρ = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 + f10 +
        f11 + f12 + f13 + f14 + f15 + f16 + f17 + f18 + f19
    inv_ρ = one(T) / ρ
    ux = (f2 - f3 + f8 - f9 + f10 - f11 + f12 - f13 + f14 - f15) * inv_ρ
    uy = (f4 - f5 + f8 + f9 - f10 - f11 + f16 - f17 + f18 - f19) * inv_ρ
    uz = (f6 - f7 + f12 + f13 - f14 - f15 + f16 + f17 - f18 - f19) * inv_ρ
    return ρ, ux, uy, uz
end
