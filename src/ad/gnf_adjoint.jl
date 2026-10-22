# --- Steady adjoint of the GNF step (inverse rheology, milestone J2) ---
#
# State: the augmented post-collision state w = (f, τ) of `gnf_step!`, stored
# as one flat vector of length n = 10 N (N = Nx Ny): component (i, j, q) at
# i + Nx (j − 1) + N (q − 1), q = 1..9 the populations, q = 10 the relaxation
# time τ (the vec of an (Nx, Ny, 10) array). One step is w' = G(w):
#
#   arrivals fa = S f (pull stream, a permutation; τ untouched)
#   fluid node c:  (f'_c, τ'_c) = H(fa_c, τ_c),  H = BGK + Guo at ν(I1(fa_c, τ_c))
#   solid node c:  f'_c = swap(fa_c) (bounce-back), τ'_c = τ_c (never written)
#
# so ∂G/∂w = A = Bdiag · S, with Bdiag block diagonal: per fluid node the 10×10
# Jacobian of H, per solid node the swap (9×9) and 1 for τ.
#
# Local blocks. ν depends on the state through I1 only. At the linearisation
# point (I1_c, ν_c) the closure is replaced by its tangent,
# ν = ν_c + (dν/dI1)_c (I1 − I1_c) + δν, which is exact to first order; the
# block of node c is the ForwardDiff Jacobian of that cell map in
# (fa_c, τ_c, δν): 10 × 11, the 11th column ∂(f', τ')/∂ν. (dν/dI1)_c is the
# total derivative of the closure (ForwardDiff on `gnf_viscosity` in I1), so a
# learned closure only has to provide that scalar per node. Blocks are built
# on the host, once per adjoint solve, and copied to the device (10 × 10 per
# node; the ν column stays on the host for the parameter gradient).
#
# Active space. Components of w that never interact with the fluid are left
# out (they carry eigenvalues ±1 of A and would make I − Aᵀ singular): solid
# rest populations, populations of solid–solid links, τ at solid nodes. Active
# = every component of a fluid node + the solid-side population of each
# fluid–solid link (stored solid component whose arrival slot is at a fluid
# node; it is also the one fed by the fluid). A and Aᵀ map the active space
# into itself and the inactive space into itself (test T4b). P = the 0/1 mask.
#
# Gauge. Mass is conserved: c = 1 on active populations, 0 on τ, satisfies
# Aᵀ c = c, so I − Aᵀ has c in its kernel on the active space. The adjoint
# system is bordered with ĉ = c/‖c‖ (as `ad_gauge_augmented_adjoint`, written
# here for any array type):
#
#   [ I − P Aᵀ P   ĉ ] [λ]   [P ∂J/∂w]
#   [ ĉᵀ           0 ] [η] = [  0    ]
#
# The parameter gradient λᵀ ∂G/∂θ does not depend on the gauge because the
# collision conserves mass for every ν (Σ_q ∂f'_q/∂ν = 0).
#
# Gradient: dJ/dθ = Σ_c s_c ∂ν_c/∂θ, s_c = (∂(f', τ')_c/∂ν)ᵀ λ_c, with ∂ν_c/∂θ
# by ForwardDiff of `gnf_viscosity(closure_of(θ), I1_c)` (lattice units: for
# the starred units of the obstacle channel this is ν0,lat ∂η*/∂θ(I1*_c)).
#
# Krylov solver: restarted GMRES written for any AbstractVector (Array or
# CuArray): Gram–Schmidt with reorthogonalisation (modified, `orth = :mgs2`,
# default; or classical in two passes with matrix–vector products,
# `orth = :cgs2`), Givens rotations, stop on ‖r‖/‖b‖ ≤ tol checked on the
# explicit residual at the end of each cycle, optional deflated restarting
# (GMRES-DR, Morgan 2002: `deflate = k` harmonic Ritz vectors kept).
#
# No automatic differentiation runs inside a device kernel: JVP/VJP kernels
# are hand-written (blocks + `_spxwy_src`, the stream map of
# `kernels/stream_transpose_2d.jl`). Enzyme only appears in test T4.

using LinearAlgebra
import ForwardDiff

const _GNF_CX = (0, 1, 0, -1, 0, 1, -1, -1, 1)
const _GNF_CY = (0, 0, 1, 0, -1, 1, 1, -1, -1)

# ============================================================
# State packing
# ============================================================

"""
    gnf_pack_state(f, tau) -> w

Flat state vector `[vec(f); vec(tau)]` (length 10 Nx Ny) on the backend of `f`.
"""
gnf_pack_state(f, tau) = vcat(vec(f), vec(tau))

"""
    gnf_unpack_state(w, Nx, Ny) -> (f, tau)

Copies of the populations (Nx, Ny, 9) and τ (Nx, Ny) held in a flat state vector.
"""
function gnf_unpack_state(w, Nx::Int, Ny::Int)
    N = Nx * Ny
    return reshape(w[1:9N], Nx, Ny, 9), reshape(w[9N+1:10N], Nx, Ny)
end

_gnf_host_field(v::Number, Nx, Ny) = fill(Float64(v), Nx, Ny)
_gnf_host_field(v::AbstractMatrix, Nx, Ny) =
    (size(v) == (Nx, Ny) || throw(DimensionMismatch("force field must be $(Nx)×$(Ny)"));
     Float64.(Array(v)))

# ============================================================
# Linearisation (host)
# ============================================================

# Cell map of the linearised step in x = (fa_1..fa_9, τ_prev, δν).
@inline function _gnf_lin_cell(x::SVector{11}, Fx, Fy, ν_c, κ_c, I1_c)
    fp = SVector{9}(x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9])
    ρ, ux, uy = gnf_moments_cell(fp, Fx, Fy)
    I1 = gnf_strain_invariant_cell(fp, ρ, ux, uy, x[10])
    ν = ν_c + κ_c * (I1 - I1_c) + x[11]
    fo, τn = gnf_collide_cell(fp, Fx, Fy, ν)
    return SVector{10}(fo[1], fo[2], fo[3], fo[4], fo[5], fo[6], fo[7], fo[8], fo[9], τn)
end

"""
    gnf_active_mask(is_solid) -> Vector{Bool}

Active components of the flat state (length 10 Nx Ny): all ten components of
fluid nodes, and the populations of solid nodes whose arrival slot is at a fluid
node (solid side of the fluid–solid links).
"""
function gnf_active_mask(is_solid::AbstractMatrix{Bool})
    Nx, Ny = size(is_solid)
    N = Nx * Ny
    act = fill(false, 10N)
    for j in 1:Ny, i in 1:Nx
        if !is_solid[i, j]
            for q in 1:10
                act[_spxwy_lin(i, j, q, Nx, Ny)] = true
            end
        else
            for q in 1:9
                d = _spxwy_dest(i, j, q, Nx, Ny)
                act[_spxwy_lin(i, j, q, Nx, Ny)] = !is_solid[(d - 1) % N + 1]
            end
        end
    end
    return act
end

"""
    GNFLinearisation

Host data of the step linearised at a state w = (f, τ) (`gnf_linearise`):
blocks `B` (Nx, Ny, 10, 11) (zero at solid nodes), the strain invariant `I1`,
viscosity `nu` and closure slope `dnu_dI1` used by the step from w, the
post-collision moments `rho`, `ux`, `uy` of w (velocity 0 at solid nodes), the
active mask, and counts (`n_small_I1` = fluid nodes with I1 < 1e-12, lattice units).
"""
struct GNFLinearisation
    Nx::Int
    Ny::Int
    is_solid::Matrix{Bool}
    B::Array{Float64,4}
    I1::Matrix{Float64}
    nu::Matrix{Float64}
    dnu_dI1::Matrix{Float64}
    rho::Matrix{Float64}
    ux::Matrix{Float64}
    uy::Matrix{Float64}
    active::Vector{Bool}
    n_fluid::Int
    n_active::Int
    n_small_I1::Int
    I1_min::Float64
    time::Float64
end

"""
    gnf_post_moments(f, is_solid, Fx, Fy) -> (rho, ux, uy)

Host density and velocity of post-collision populations (as
`gnf_macroscopic_post_2d!`: u = (Σ c f − F/2)/ρ, velocity 0 at solid nodes).
"""
function gnf_post_moments(f, is_solid::AbstractMatrix{Bool}, Fx, Fy=0)
    fh = Float64.(Array(f))
    Nx, Ny = size(is_solid)
    Fxh = _gnf_host_field(Fx, Nx, Ny); Fyh = _gnf_host_field(Fy, Nx, Ny)
    rho = zeros(Nx, Ny); ux = zeros(Nx, Ny); uy = zeros(Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        f1=fh[i,j,1]; f2=fh[i,j,2]; f3=fh[i,j,3]; f4=fh[i,j,4]; f5=fh[i,j,5]
        f6=fh[i,j,6]; f7=fh[i,j,7]; f8=fh[i,j,8]; f9=fh[i,j,9]
        ρ = f1+f2+f3+f4+f5+f6+f7+f8+f9
        rho[i, j] = ρ
        if !is_solid[i, j]
            ux[i, j] = ((f2-f4+f6-f7-f8+f9) - Fxh[i, j]/2) / ρ
            uy[i, j] = ((f3-f5+f6+f7-f8-f9) - Fyh[i, j]/2) / ρ
        end
    end
    return rho, ux, uy
end

"""
    gnf_linearise(f, tau, is_solid, closure, Fx, Fy=0) -> GNFLinearisation

Local blocks of the GNF step at the post-collision state (f, τ) (any backend;
computed on the host in Float64). For every fluid node: arrived populations
fa = S f, (I1_c, ν_c) of the step from w (`gnf_cell_update`), closure slope
κ_c = dν/dI1 (ForwardDiff in I1), and the 10 × 11 Jacobian of the linearised
cell map in (fa_c, τ_c, δν).
"""
function gnf_linearise(f, tau, is_solid::AbstractMatrix{Bool}, closure, Fx, Fy=0)
    t0 = time()
    fh = Float64.(Array(f)); th = Float64.(Array(tau)); solid = Array{Bool}(is_solid)
    Nx, Ny = size(solid)
    size(fh) == (Nx, Ny, 9) || throw(DimensionMismatch("f must be $(Nx)×$(Ny)×9"))
    Fxh = _gnf_host_field(Fx, Nx, Ny); Fyh = _gnf_host_field(Fy, Nx, Ny)
    cl = gnf_convert_closure(Float64, closure)
    fa = similar(fh)
    stream_periodic_x_wall_y_2d!(fa, fh, Nx, Ny)
    B = zeros(Nx, Ny, 10, 11)
    I1 = zeros(Nx, Ny); nu = zeros(Nx, Ny); dnu = zeros(Nx, Ny)
    Threads.@threads for j in 1:Ny
        for i in 1:Nx
            solid[i, j] && continue
            fp = SVector{9,Float64}(fa[i,j,1], fa[i,j,2], fa[i,j,3], fa[i,j,4], fa[i,j,5],
                                    fa[i,j,6], fa[i,j,7], fa[i,j,8], fa[i,j,9])
            τp = th[i, j]; Fxc = Fxh[i, j]; Fyc = Fyh[i, j]
            _, _, s, ν = gnf_cell_update(fp, τp, Fxc, Fyc, cl)
            κ = ForwardDiff.derivative(t -> gnf_viscosity(cl, t), s)
            x0 = SVector{11,Float64}(fp[1], fp[2], fp[3], fp[4], fp[5], fp[6], fp[7], fp[8],
                                     fp[9], τp, 0.0)
            Jc = ForwardDiff.jacobian(x -> _gnf_lin_cell(x, Fxc, Fyc, ν, κ, s), x0)
            for col in 1:11, r in 1:10
                B[i, j, r, col] = Jc[r, col]
            end
            I1[i, j] = s; nu[i, j] = ν; dnu[i, j] = κ
        end
    end
    rho, ux, uy = gnf_post_moments(fh, solid, Fxh, Fyh)
    active = gnf_active_mask(solid)
    fl = .!solid
    n_small = count(fl .& (I1 .< 1e-12))
    I1min = any(fl) ? minimum(I1[fl]) : NaN
    return GNFLinearisation(Nx, Ny, solid, B, I1, nu, dnu, rho, ux, uy, active,
                            count(fl), count(active), n_small, I1min, time() - t0)
end

# ============================================================
# Device operator: JVP / VJP kernels
# ============================================================

"""
    GNFStepOperator(lin::GNFLinearisation; backend=CPU(), T=Float64)

Device copy of the 10 × 10 blocks, solid mask, active mask (`active`, in T),
all-ones mask (`ones`, for unprojected products) and gauge vector `chat`
(Float64, ĉ = c/‖c‖). `T` is the arithmetic type of the products
(Float32 only for the fallback (iv) of the J2 gate).
"""
struct GNFStepOperator{T, TB<:AbstractArray{T,4}, TS<:AbstractMatrix{Bool},
                       TV<:AbstractVector{T}, TC<:AbstractVector{Float64}}
    Nx::Int
    Ny::Int
    n::Int
    B::TB
    solid::TS
    active::TV
    ones::TV
    chat::TC
end

function GNFStepOperator(lin::GNFLinearisation; backend=KernelAbstractions.CPU(), T::Type=Float64)
    Nx, Ny = lin.Nx, lin.Ny
    N = Nx * Ny; n = 10N
    B = KernelAbstractions.allocate(backend, T, Nx, Ny, 10, 10)
    copyto!(B, T.(lin.B[:, :, :, 1:10]))
    solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny)
    copyto!(solid, lin.is_solid)
    act = KernelAbstractions.allocate(backend, T, n)
    copyto!(act, T.(lin.active))
    on = KernelAbstractions.allocate(backend, T, n)
    fill!(on, one(T))
    c = Float64.(lin.active); c[9N+1:10N] .= 0
    c ./= norm(c)
    chat = KernelAbstractions.allocate(backend, Float64, n)
    copyto!(chat, c)
    return GNFStepOperator(Nx, Ny, n, B, solid, act, on, chat)
end

Base.eltype(::GNFStepOperator{T}) where {T} = T

# Per-node pieces of the products. The arithmetic type T is a static parameter
# of these helpers (never a captured local), so the kernels stay type-stable on
# every backend.

# Projected values of the ten components that node (i, j) reads in the JVP:
# the nine arrival slots (pull sources) and its own τ.
@inline function _gnf_gather_arrivals(x, pin, ::Type{T}, i, j, Nx, Ny) where {T}
    return ntuple(Val(10)) do s
        k = s <= 9 ? _spxwy_src(i, j, s, Nx, Ny) : _spxwy_lin(i, j, 10, Nx, Ny)
        @inbounds T(pin[k]) * T(x[k])
    end
end

# Projected values of the ten stored components of node (i, j).
@inline function _gnf_gather_local(x, pin, ::Type{T}, i, j, Nx, Ny) where {T}
    return ntuple(Val(10)) do r
        k = _spxwy_lin(i, j, r, Nx, Ny)
        @inbounds T(pin[k]) * T(x[k])
    end
end

# Block product B_c a (JVP) and its transpose B_cᵀ λ (VJP) at node (i, j).
@inline function _gnf_block_mul(B, a::NTuple{10,T}, i, j) where {T}
    return ntuple(Val(10)) do r
        acc = zero(T)
        for s in 1:10
            @inbounds acc += B[i, j, r, s] * a[s]
        end
        acc
    end
end

@inline function _gnf_block_mul_t(B, λ::NTuple{10,T}, i, j) where {T}
    return ntuple(Val(10)) do s
        acc = zero(T)
        for r in 1:10
            @inbounds acc += B[i, j, r, s] * λ[r]
        end
        acc
    end
end

# Solid node: half-way bounce-back swap of the populations, τ unchanged (the
# swap is its own transpose).
@inline _gnf_swap(a::NTuple{10,T}) where {T} =
    (a[1], a[4], a[5], a[2], a[3], a[8], a[9], a[6], a[7], a[10])

@kernel function gnf_jvp_2d_kernel!(y, @Const(x), @Const(B), @Const(solid),
                                    @Const(pin), @Const(pout), Nx, Ny)
    i, j = @index(Global, NTuple)
    a = _gnf_gather_arrivals(x, pin, eltype(B), i, j, Nx, Ny)
    issolid = @inbounds solid[i, j]
    out = issolid ? _gnf_swap(a) : _gnf_block_mul(B, a, i, j)
    for r in 1:10
        k = _spxwy_lin(i, j, r, Nx, Ny)
        @inbounds y[k] = pout[k] * out[r]
    end
end

@kernel function gnf_vjp_2d_kernel!(y, @Const(x), @Const(B), @Const(solid),
                                    @Const(pin), @Const(pout), Nx, Ny)
    i, j = @index(Global, NTuple)
    λ = _gnf_gather_local(x, pin, eltype(B), i, j, Nx, Ny)
    issolid = @inbounds solid[i, j]
    z = issolid ? _gnf_swap(λ) : _gnf_block_mul_t(B, λ, i, j)
    # transposed stream: arrival slot (i, j, s) goes back to its source component
    for s in 1:9
        k = _spxwy_src(i, j, s, Nx, Ny)
        @inbounds y[k] = pout[k] * z[s]
    end
    k = _spxwy_lin(i, j, 10, Nx, Ny)
    @inbounds y[k] = pout[k] * z[10]
end

"""
    gnf_jvp!(y, op, x; project=true) -> y

Tangent product y = P A P x of the linearised step (A = ∂G/∂w). `x`, `y`: flat
vectors of length 10 Nx Ny on the device of `op`, distinct. `project=false`
gives the unprojected A x.
"""
function gnf_jvp!(y, op::GNFStepOperator, x; project::Bool=true)
    m = project ? op.active : op.ones
    backend = KernelAbstractions.get_backend(op.B)
    gnf_jvp_2d_kernel!(backend)(y, x, op.B, op.solid, m, m, op.Nx, op.Ny; ndrange=(op.Nx, op.Ny))
    return y
end

"""
    gnf_vjp!(y, op, x; project=true) -> y

Adjoint product y = P Aᵀ P x (transposed blocks, then the transposed stream).
`project=false` gives the unprojected Aᵀ x.
"""
function gnf_vjp!(y, op::GNFStepOperator, x; project::Bool=true)
    m = project ? op.active : op.ones
    backend = KernelAbstractions.get_backend(op.B)
    gnf_vjp_2d_kernel!(backend)(y, x, op.B, op.solid, m, m, op.Nx, op.Ny; ndrange=(op.Nx, op.Ny))
    return y
end

"""
    GNFBorderedAdjoint(op)

Callable `A(y, x)` applying the gauge-bordered adjoint operator to vectors of
length n + 1: y[1:n] = x − P Aᵀ P x + ĉ x[n+1] (identity on inactive
components), y[n+1] = ĉᵀ x[1:n].
"""
struct GNFBorderedAdjoint{TO<:GNFStepOperator, TV<:AbstractVector{Float64}}
    op::TO
    tmp::TV
end
GNFBorderedAdjoint(op::GNFStepOperator) = GNFBorderedAdjoint(op, similar(op.chat))

function (A::GNFBorderedAdjoint)(y, x)
    n = A.op.n
    xn = view(x, 1:n); yn = view(y, 1:n); xs = view(x, n+1:n+1)
    gnf_vjp!(A.tmp, A.op, xn)
    yn .= xn .- A.tmp .+ A.op.chat .* xs
    fill!(view(y, n+1:n+1), dot(A.op.chat, xn))
    return y
end

# ============================================================
# GMRES (restarted, optional deflated restarting), any AbstractVector
# ============================================================

_gnf_sync(x) = KernelAbstractions.synchronize(KernelAbstractions.get_backend(x))

# Orthogonalise w against V[:, 1:j]; coefficients added to h[1:j]; returns ‖w‖.
function _gnf_orthogonalise!(h, V, w, j, orth::Symbol, hbuf)
    if orth === :mgs2
        for pass in 1:2
            for i in 1:j
                vi = view(V, :, i)
                hij = dot(vi, w)
                axpy!(-hij, vi, w)
                h[i] += hij
            end
        end
    elseif orth === :cgs2
        Vj = view(V, :, 1:j)
        hj = view(hbuf, 1:j)
        for pass in 1:2
            mul!(hj, transpose(Vj), w)
            mul!(w, Vj, hj, -one(eltype(w)), one(eltype(w)))
            h[1:j] .+= Array(hj)
        end
    else
        throw(ArgumentError("orth must be :mgs2 or :cgs2, got :$orth"))
    end
    return norm(w)
end

# Real basis of the k harmonic Ritz vectors of smallest |θ| (complex pairs kept whole).
function _gnf_harmonic_ritz(Hbar::AbstractMatrix{Float64}, m::Int, k::Int)
    Hm = Hbar[1:m, 1:m]; hm1 = Hbar[m+1, m]
    em = zeros(m); em[m] = 1
    fvec = transpose(Hm) \ em
    M = Hm + hm1^2 * fvec * transpose(em)
    E = eigen(M)
    order = sortperm(abs.(E.values))
    cols = Vector{Vector{Float64}}()
    vals = ComplexF64[]
    used = falses(m)
    for idx in order
        length(cols) >= k && break
        used[idx] && continue
        θ = E.values[idx]; g = E.vectors[:, idx]
        used[idx] = true
        if abs(imag(θ)) <= 1e-12 * max(abs(θ), 1e-300)
            push!(cols, real.(g)); push!(vals, θ)
        else
            push!(cols, real.(g)); push!(cols, imag.(g)); push!(vals, θ); push!(vals, conj(θ))
            # mark the conjugate as used
            for jdx in order
                if !used[jdx] && abs(E.values[jdx] - conj(θ)) <= 1e-10 * max(abs(θ), 1e-300)
                    used[jdx] = true; break
                end
            end
        end
    end
    return reduce(hcat, cols), vals
end

"""
    gnf_gmres!(x, apply!, b; tol=1e-6, restart=1000, maxiter=100_000, orth=:mgs2,
               deflate=0, max_wall=Inf, ritz=false, verbose=false) -> info

Restarted GMRES for `apply!(y, x)` (y = M x), in place on the initial guess `x`.
Works on any AbstractVector type (Array, CuArray): Krylov basis
`similar(b, length(b), restart + 1)`, small dense algebra on the host.

- `orth = :mgs2`: modified Gram–Schmidt, two passes (reorthogonalisation);
  `:cgs2`: classical Gram–Schmidt, two passes, matrix–vector products.
- Givens rotations on the Hessenberg matrix; the implicit residual decides when
  a cycle ends, the explicit residual ‖b − M x‖/‖b‖ ≤ `tol` decides convergence.
- `deflate = k > 0`: GMRES-DR (Morgan 2002), each restart keeps the k harmonic
  Ritz vectors of smallest |θ| of the last cycle.
- Stops at `maxiter` iterations or after `max_wall` seconds.

- `snapshots`: tolerances above `tol`; when the implicit residual first falls
  below one, the iterate at that point is formed (one extra product with the
  basis) and kept, with the iteration count and the elapsed time, so one solve
  gives the cost and the solution at several tolerances.
- `ritz = true`: harmonic Ritz values of every full cycle (the last is kept).

Times exclude the diagnostics (snapshots, Ritz values), reported as `t_diag`.
Returns `(; converged, iters, cycles, relres, relres_cycles, relres_iter, t_iter,
t_total, t_op, t_orth, t_diag, n_apply, restart, orth, deflate, ritz, snaps)`;
`ritz` is empty unless `ritz = true`; `snaps` = vector of `(; tol, iters, t, x)`.
"""
function gnf_gmres!(x, apply!, b; tol::Real=1e-6, restart::Integer=1000,
                    maxiter::Integer=100_000, orth::Symbol=:mgs2, deflate::Integer=0,
                    max_wall::Real=Inf, ritz::Bool=false, verbose::Bool=false,
                    snapshots=())
    Tv = eltype(b)
    Nv = length(b)
    m = Int(restart)
    k = Int(deflate)
    0 <= k < m || throw(ArgumentError("need 0 ≤ deflate < restart"))
    t0 = time(); t_op = 0.0; t_orth = 0.0; t_diag = 0.0; n_apply = 0
    V = similar(b, Nv, m + 1)
    hbuf = similar(b, m + 1)
    r = similar(b)
    bnorm = norm(b)
    relres_cycles = Float64[]; relres_iter = Float64[]; t_iter = Float64[]
    ritz_vals = ComplexF64[]
    snap_tols = sort(collect(Float64, snapshots); rev=true)
    snaps = NamedTuple[]; next_snap = 1
    iters = 0; cycles = 0
    if bnorm == 0
        fill!(x, zero(Tv))
        return (; converged=true, iters=0, cycles=0, relres=0.0, relres_cycles, relres_iter, t_iter,
                t_total=0.0, t_op, t_orth, t_diag, n_apply, restart=m, orth, deflate=k,
                ritz=ritz_vals, snaps)
    end
    ta = time(); apply!(r, x); _gnf_sync(r); t_op += time() - ta; n_apply += 1
    r .= b .- r
    β = norm(r)
    push!(relres_cycles, β / bnorm)
    converged = β <= tol * bnorm

    Hbar = zeros(m + 1, m)       # unrotated (block) Hessenberg of the cycle
    R = zeros(m + 1, m)          # rotated columns
    g = zeros(m + 1)
    cvec = zeros(m + 1)          # residual coordinates in V at the cycle start
    cs = zeros(m); sn = zeros(m)
    kk = 0                       # deflated columns carried into the cycle
    Q1 = ones(1, 1)
    first_cycle = true
    while !converged && iters < maxiter && time() - t0 < max_wall
        cycles += 1
        if first_cycle || k == 0
            kk = 0
            view(V, :, 1) .= r ./ β
            fill!(Hbar, 0); fill!(cvec, 0); cvec[1] = β
            Q1 = ones(1, 1)
        end
        first_cycle = false
        fill!(R, 0); fill!(g, 0)
        g[1:kk+1] .= transpose(Q1) * cvec[1:kk+1]
        if kk > 0
            F = qr(Hbar[1:kk+1, 1:kk])
            R[1:kk, 1:kk] .= F.R
        end
        jlast = kk
        for j in kk+1:m
            vj = view(V, :, j); w = view(V, :, j + 1)
            ta = time(); apply!(w, vj); _gnf_sync(w); t_op += time() - ta; n_apply += 1
            ta = time()
            hcol = zeros(j + 1)
            hn = _gnf_orthogonalise!(hcol, V, w, j, orth, hbuf)
            hcol[j+1] = hn
            hn > 0 && (w ./= hn)
            _gnf_sync(w); t_orth += time() - ta
            Hbar[1:j+1, j] .= hcol
            # rotate: Q1ᵀ on rows 1..kk+1, then the Givens of columns kk+1..j-1
            rc = copy(hcol)
            rc[1:kk+1] .= transpose(Q1) * rc[1:kk+1]
            for i in kk+1:j-1
                tmp = cs[i] * rc[i] + sn[i] * rc[i+1]
                rc[i+1] = -sn[i] * rc[i] + cs[i] * rc[i+1]
                rc[i] = tmp
            end
            ρh = hypot(rc[j], rc[j+1])
            if ρh == 0
                cs[j] = 1.0; sn[j] = 0.0
            else
                cs[j] = rc[j] / ρh; sn[j] = rc[j+1] / ρh
            end
            rc[j] = ρh; rc[j+1] = 0.0
            R[1:j+1, j] .= rc
            g[j+1] = -sn[j] * g[j]; g[j] = cs[j] * g[j]
            iters += 1; jlast = j
            rel_now = abs(g[j+1]) / bnorm
            push!(relres_iter, rel_now); push!(t_iter, time() - t0 - t_diag)
            while next_snap <= length(snap_tols) && rel_now <= snap_tols[next_snap]
                # iterate at this point of the cycle (does not change the iteration;
                # its cost goes to t_diag, not to the solve times)
                td = time()
                ds = similar(b, j); copyto!(ds, UpperTriangular(R[1:j, 1:j]) \ g[1:j])
                xs = copy(x); mul!(xs, view(V, :, 1:j), ds, one(Tv), one(Tv)); _gnf_sync(xs)
                push!(snaps, (; tol=snap_tols[next_snap], iters, t=td - t0 - t_diag, x=xs))
                next_snap += 1
                t_diag += time() - td
            end
            if verbose && iters % 100 == 0
                @info "gnf_gmres" iters relres = abs(g[j+1]) / bnorm t = time() - t0
            end
            (abs(g[j+1]) <= tol * bnorm || iters >= maxiter || hn == 0 ||
             time() - t0 > max_wall) && break
        end
        d = UpperTriangular(R[1:jlast, 1:jlast]) \ g[1:jlast]
        ddev = similar(b, jlast); copyto!(ddev, d)
        mul!(x, view(V, :, 1:jlast), ddev, one(Tv), one(Tv))
        ta = time(); apply!(r, x); _gnf_sync(r); t_op += time() - ta; n_apply += 1
        r .= b .- r
        β = norm(r)
        push!(relres_cycles, β / bnorm)
        converged = β <= tol * bnorm
        if ritz && jlast == m
            td = time()
            ritz_vals = eigvals(Hbar[1:m, 1:m] + Hbar[m+1, m]^2 *
                                (transpose(Hbar[1:m, 1:m]) \ [zeros(m - 1); 1.0]) *
                                transpose([zeros(m - 1); 1.0]))
            t_diag += time() - td
        end
        if k > 0 && !converged && jlast == m
            # deflated restart: keep k harmonic Ritz vectors + the residual direction
            G, _ = _gnf_harmonic_ritz(Hbar, m, k)
            kd = size(G, 2)
            kd + 1 <= m || (kd = m - 1; G = G[:, 1:kd])
            Pk = zeros(m + 1, kd); Pk[1:m, :] .= G
            Pk = Matrix(qr(Pk).Q)[:, 1:kd]
            resv = cvec - Hbar * d
            pr = resv - Pk * (transpose(Pk) * resv)
            pr = pr - Pk * (transpose(Pk) * pr)
            pr ./= norm(pr)
            P = hcat(Pk, pr)
            Hnew = transpose(P) * (Hbar * Pk[1:m, :])
            cnew = transpose(P) * resv
            Pdev = similar(b, m + 1, kd + 1); copyto!(Pdev, P)
            Vn = similar(b, Nv, kd + 1)
            mul!(Vn, view(V, :, 1:m+1), Pdev)
            view(V, :, 1:kd+1) .= Vn
            fill!(Hbar, 0); Hbar[1:kd+1, 1:kd] .= Hnew
            fill!(cvec, 0); cvec[1:kd+1] .= cnew
            kk = kd
            Q1 = Matrix(qr(Hnew).Q * Matrix{Float64}(I, kd + 1, kd + 1))
        elseif k > 0
            first_cycle = true      # stopped early: plain restart from the residual
        end
    end
    t_total = time() - t0 - t_diag
    return (; converged, iters, cycles, relres=relres_cycles[end], relres_cycles, relres_iter,
            t_iter, t_total, t_op, t_orth, t_diag, n_apply, restart=m, orth, deflate=k,
            ritz=ritz_vals, snaps)
end

# ============================================================
# Observation, adjoint solve, parameter gradient
# ============================================================

"""
    GNFVelocityObs(node, weight, comp, target, scale)

Velocity observations: value m is `scale · Σ_k weight[k, m] · u_comp[node[k, m]]`
(`comp[m]` = 1 for ux, 2 for uy; `node` = linear node index, 0 = unused slot),
with u the post-collision lattice velocity (0 at solid nodes). Objective
J = ½ Σ_m (value_m − target_m)².
"""
struct GNFVelocityObs
    node::Matrix{Int}
    weight::Matrix{Float64}
    comp::Vector{Int}
    target::Vector{Float64}
    scale::Float64
end

function gnf_obs_values(obs::GNFVelocityObs, ux::AbstractMatrix, uy::AbstractMatrix)
    M = length(obs.target)
    v = zeros(M)
    for mi in 1:M
        s = 0.0
        for kk in axes(obs.node, 1)
            nd = obs.node[kk, mi]
            nd == 0 && continue
            s += obs.weight[kk, mi] * (obs.comp[mi] == 1 ? ux[nd] : uy[nd])
        end
        v[mi] = obs.scale * s
    end
    return v
end

"""
    gnf_obs_objective(obs, ux, uy) -> J

J = ½ Σ (value − target)² from host velocity fields (post-collision, lattice units).
"""
gnf_obs_objective(obs::GNFVelocityObs, ux, uy) =
    0.5 * sum(abs2, gnf_obs_values(obs, ux, uy) .- obs.target)

"""
    gnf_obs_grad_u(obs, lin) -> (J, gux, guy)

Objective J = ½ Σ (value − target)² at the linearisation state and its
cotangent on the lattice velocity, gux = ∂J/∂u_x and guy = ∂J/∂u_y (host
Nx × Ny, zero where no observation reads).
"""
function gnf_obs_grad_u(obs::GNFVelocityObs, lin::GNFLinearisation)
    Nx, Ny = lin.Nx, lin.Ny
    v = gnf_obs_values(obs, lin.ux, lin.uy)
    res = v .- obs.target
    J = 0.5 * sum(abs2, res)
    gx = zeros(Nx, Ny); gy = zeros(Nx, Ny)
    for mi in eachindex(res)
        for kk in axes(obs.node, 1)
            nd = obs.node[kk, mi]
            nd == 0 && continue
            gval = res[mi] * obs.scale * obs.weight[kk, mi]
            obs.comp[mi] == 1 ? (gx[nd] += gval) : (gy[nd] += gval)
        end
    end
    return J, gx, gy
end

"""
    gnf_velocity_rhs(lin, gux, guy) -> b

Adjoint right-hand side b = (∂u/∂w)ᵀ g_u for a cotangent (gux, guy) on the
post-collision lattice velocity of the state (as `gnf_post_moments`; host
vector of length 10 N, nonzero on fluid populations only):
∂u_x/∂f_q = (c_xq − u_x)/ρ. Values of g_u at solid nodes are ignored (u ≡ 0 there).
"""
function gnf_velocity_rhs(lin::GNFLinearisation, gux::AbstractMatrix, guy::AbstractMatrix)
    Nx, Ny = lin.Nx, lin.Ny; N = Nx * Ny
    (size(gux) == (Nx, Ny) && size(guy) == (Nx, Ny)) ||
        throw(DimensionMismatch("g_u components must be $(Nx)×$(Ny)"))
    gx = Float64.(Array(gux)); gy = Float64.(Array(guy))
    b = zeros(10N)
    for c in 1:N
        lin.is_solid[c] && continue
        (gx[c] == 0 && gy[c] == 0) && continue
        ρ = lin.rho[c]; ux = lin.ux[c]; uy = lin.uy[c]
        for q in 1:9
            b[c + N * (q - 1)] = (gx[c] * (_GNF_CX[q] - ux) + gy[c] * (_GNF_CY[q] - uy)) / ρ
        end
    end
    return b
end

"""
    gnf_obs_rhs(obs, lin) -> (J, b)

Objective at the linearisation state and its gradient b = ∂J/∂w:
`gnf_obs_grad_u` then `gnf_velocity_rhs`.
"""
function gnf_obs_rhs(obs::GNFVelocityObs, lin::GNFLinearisation)
    J, gx, gy = gnf_obs_grad_u(obs, lin)
    return J, gnf_velocity_rhs(lin, gx, gy)
end

"""
    gnf_adjoint_solve(op, b; tol=1e-6, restart=1000, maxiter=100_000, orth=:mgs2,
                      deflate=0, λ0=nothing, max_wall=Inf, ritz=false, verbose=false)
        -> (λ, η, info)

Solves the gauge-bordered adjoint system for the right-hand side `b` = ∂J/∂w
(device vector, length n; projected on the active space here). `λ0`: warm
start (projected on the active space and on ĉ⊥). Returns λ (length n, λ ⊥ ĉ up
to the solver tolerance), the border unknown η and the `gnf_gmres!` info.
"""
function gnf_adjoint_solve(op::GNFStepOperator, b; tol::Real=1e-6, restart::Integer=1000,
                           maxiter::Integer=100_000, orth::Symbol=:mgs2, deflate::Integer=0,
                           λ0=nothing, max_wall::Real=Inf, ritz::Bool=false,
                           verbose::Bool=false, snapshots=())
    n = op.n
    bb = similar(op.chat, n + 1)
    view(bb, 1:n) .= op.active .* b
    fill!(view(bb, n+1:n+1), 0.0)
    x = similar(bb); fill!(x, 0.0)
    if λ0 !== nothing
        xn = view(x, 1:n)
        xn .= op.active .* λ0
        s = dot(op.chat, xn)
        xn .-= s .* op.chat
    end
    A = GNFBorderedAdjoint(op)
    info = gnf_gmres!(x, A, bb; tol, restart, maxiter, orth, deflate, max_wall, ritz, verbose,
                      snapshots)
    λ = x[1:n]
    η = Array(x[n+1:n+1])[1]
    return λ, η, info
end

"""
    gnf_parameter_gradient(lin, λ, θ, closure_of) -> (grad, s)

dJ/dθ = Σ_c s_c ∂ν_c/∂θ with s_c = (∂(f', τ')_c/∂ν)ᵀ λ_c (ν column of the
blocks) and ∂ν_c/∂θ = ForwardDiff gradient of `gnf_viscosity(closure_of(θ), I1_c)`.
`closure_of(θ)` returns the closure in lattice units; `λ` any backend.
"""
function gnf_parameter_gradient(lin::GNFLinearisation, λ, θ, closure_of)
    s = gnf_nu_sensitivity(lin, λ)
    return _gnf_theta_gradient(lin, s, θ, closure_of), s
end

"""
    gnf_nu_sensitivity(lin, λ) -> s

Per-cell cotangent of ν_lat, s_c = (∂(f', τ')_c/∂ν)ᵀ λ_c (ν column of the
blocks), host Nx × Ny, zero at solid nodes; `λ` any backend.
"""
function gnf_nu_sensitivity(lin::GNFLinearisation, λ)
    λh = Float64.(Array(λ))
    Nx, Ny = lin.Nx, lin.Ny
    s = zeros(Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        lin.is_solid[i, j] && continue
        acc = 0.0
        for r in 1:10
            acc += lin.B[i, j, r, 11] * λh[_spxwy_lin(i, j, r, Nx, Ny)]
        end
        s[i, j] = acc
    end
    return s
end

# Σ_c s_c ∂ν_c/∂θ, ∂ν_c/∂θ by ForwardDiff through `gnf_viscosity(closure_of(θ), I1_c)`.
function _gnf_theta_gradient(lin::GNFLinearisation, s, θ, closure_of)
    Nx, Ny = lin.Nx, lin.Ny
    P = length(θ)
    θs = SVector{P,Float64}(Tuple(θ))
    grad = zeros(P)
    for j in 1:Ny, i in 1:Nx
        lin.is_solid[i, j] && continue
        I1c = lin.I1[i, j]
        gθ = ForwardDiff.gradient(t -> gnf_viscosity(closure_of(t), I1c), θs)
        for p in 1:P
            grad[p] += s[i, j] * gθ[p]
        end
    end
    return grad
end

_gnf_split_gu(gu::Tuple{<:AbstractMatrix,<:AbstractMatrix}) = gu
_gnf_split_gu(gu::NamedTuple{(:ux, :uy)}) = (gu.ux, gu.uy)
function _gnf_split_gu(gu::AbstractArray{<:Real,3})
    size(gu, 3) == 2 || throw(DimensionMismatch("g_u must be Nx×Ny×2, got $(size(gu))"))
    g = Array(gu)
    return (g[:, :, 1], g[:, :, 2])
end

"""
    gnf_steady_adjoint(lin, op, gu; m=nothing, nu0_lat=nothing, solver=:kstep, kwargs...) -> (s, stats)
    gnf_steady_adjoint(f, tau, gu; is_solid, closure, Fx, Fy=0, backend=CPU(),
                       T=Float64, m=nothing, nu0_lat=nothing, kwargs...) -> (s, stats)

Generic steady adjoint (gradient contract J2 → J3a). For any loss L of the
lattice velocity at the fixed point w* = (f, tau) of the GNF step with
`closure` (lattice units), `gu` = ∂L/∂u_lat: an Nx × Ny × 2 array (component 1
= ∂L/∂u_x, 2 = ∂L/∂u_y), a tuple `(gux, guy)` or a named tuple `(; ux, uy)`
of Nx × Ny arrays, u = post-collision lattice velocity (`gnf_post_moments`,
0 at solid nodes, where `gu` is ignored). Builds b = (∂u/∂w)ᵀ g_u
(`gnf_velocity_rhs`), solves the adjoint system with `solver` = `:kstep`
(default: GMRES on the K-step system, `gnf_adjoint_kstep_solve`, defaults
K=1001 (must be odd), restart=50, orth=:cgs2), `:gmres` (gauge-bordered GMRES,
`gnf_adjoint_solve`) or `:fixed_point` (`gnf_adjoint_fixed_point`), `kwargs`
passed on (tol, restart, orth, deflate, max_wall, K, …; `λ0` = initial adjoint
state for a warm start, a device vector of length `op.n` such as `stats.λ` of
the previous call),
and returns

- `s`: per-cell cotangent of ν_lat (Nx × Ny host, zero at solid nodes,
  `gnf_nu_sensitivity`), so that dL/dθ = Σ_c s_c ∂ν_lat,c/∂θ
  = Σ_c s_c ν0,lat ∂η*_c/∂θ, plus any explicit θ dependence of L;
- `stats`: `(; converged, iters, relres, t_rhs, t_solve, t_s, t_lin, t_copy,
  λ, eta, info, I1_lat, I1star, lin, op)`; `I1star` = I1_lat m⁴/ν0,lat² when
  `m` and `nu0_lat` are given (units of plan §4.4), else `nothing`. The second
  form linearises first (`gnf_linearise`, `GNFStepOperator` on `backend`).
"""
function gnf_steady_adjoint(lin::GNFLinearisation, op::GNFStepOperator, gu;
                            m=nothing, nu0_lat=nothing, solver::Symbol=:kstep,
                            t_lin=0.0, t_copy=0.0, kwargs...)
    t0 = time()
    gux, guy = _gnf_split_gu(gu)
    bh = gnf_velocity_rhs(lin, gux, guy)
    b = similar(op.chat, op.n); copyto!(b, bh); _gnf_sync(b)
    t1 = time()
    λ, η, info = _gnf_solve_adjoint(op, b, solver; kwargs...)
    t2 = time()
    s = gnf_nu_sensitivity(lin, λ)
    t3 = time()
    I1star = (m === nothing || nu0_lat === nothing) ? nothing : lin.I1 .* (m^4 / nu0_lat^2)
    stats = (; converged=info.converged, iters=info.iters, relres=info.relres,
             t_rhs=t1 - t0, t_solve=t2 - t1, t_s=t3 - t2, t_lin, t_copy,
             λ, eta=η, info, I1_lat=lin.I1, I1star, lin, op)
    return s, stats
end

function gnf_steady_adjoint(f::AbstractArray, tau::AbstractArray, gu; is_solid, closure, Fx, Fy=0,
                            backend=KernelAbstractions.CPU(), T::Type=Float64, kwargs...)
    t0 = time()
    lin = gnf_linearise(f, tau, is_solid, closure, Fx, Fy)
    t1 = time()
    op = GNFStepOperator(lin; backend, T)
    _gnf_sync(op.chat)
    t2 = time()
    return gnf_steady_adjoint(lin, op, gu; t_lin=t1 - t0, t_copy=t2 - t1, kwargs...)
end

"""
    gnf_steady_gradient(f, tau, θ, obs; is_solid, closure_of, Fx, Fy=0,
                        backend=CPU(), T=Float64, kwargs...) -> NamedTuple

Value and parameter gradient of J = ½ Σ (u_obs − target)² at the fixed point
w* = (f, tau) of the GNF step with closure `closure_of(θ)` (lattice units):
thin wrapper of `gnf_steady_adjoint` with g_u = ∂J/∂u (`gnf_obs_grad_u`),
then dJ/dθ = Σ_c s_c ∂ν_c/∂θ; `kwargs` go to `gnf_steady_adjoint` (default
`solver=:kstep`; `λ0` for a warm start). Returns `(; J, grad, λ, η, info, lin,
op, s, t_lin, t_copy, t_solve, t_grad, t_total)`.
"""
function gnf_steady_gradient(f, tau, θ, obs::GNFVelocityObs; is_solid, closure_of, Fx, Fy=0,
                             backend=KernelAbstractions.CPU(), T::Type=Float64, kwargs...)
    t0 = time()
    lin = gnf_linearise(f, tau, is_solid, closure_of(θ), Fx, Fy)
    t1 = time()
    op = GNFStepOperator(lin; backend, T)
    J, gux, guy = gnf_obs_grad_u(obs, lin)
    t2 = time()
    s, st = gnf_steady_adjoint(lin, op, (gux, guy); kwargs...)
    t3 = time()
    grad = _gnf_theta_gradient(lin, s, θ, closure_of)
    t4 = time()
    return (; J, grad, λ=st.λ, η=st.eta, info=st.info, lin, op, s, t_lin=t1 - t0,
            t_copy=t2 - t1 + st.t_rhs, t_solve=st.t_solve + st.t_s, t_grad=t4 - t3,
            t_total=t4 - t0)
end

# ============================================================
# Diagnostics: power iteration, truncated adjoint series, real step
# ============================================================

"""
    gnf_power_iteration(op; iters=50, seed=1) -> (; mu, ratio, history)

Power iteration on Q P Aᵀ P (Q = I − ĉĉᵀ removes the mass eigenvalue 1):
`mu` = last Rayleigh quotient, `ratio` = last ‖y‖/‖x‖, `history` of both.
"""
function gnf_power_iteration(op::GNFStepOperator; iters::Integer=50, seed::Integer=1)
    n = op.n
    xh = _gnf_pseudo_randn(seed, n)
    x = similar(op.chat); copyto!(x, xh)
    x .*= op.active
    x .-= dot(op.chat, x) .* op.chat
    x ./= norm(x)
    y = similar(x)
    hist = Tuple{Float64,Float64}[]
    mu = NaN; ratio = NaN
    for it in 1:iters
        gnf_vjp!(y, op, x)
        y .-= dot(op.chat, y) .* op.chat
        mu = dot(x, y); ratio = norm(y)
        push!(hist, (mu, ratio))
        x .= y ./ ratio
    end
    return (; mu, ratio, history=hist)
end

# Deterministic pseudo-normal numbers (xorshift, sum of four uniforms) to seed
# the power iteration without a Random dependency in src.
function _gnf_pseudo_randn(seed::Integer, n::Integer)
    s = UInt64(seed) * 0x9E3779B97F4A7C15 + 0x0000000001234567
    out = zeros(n)
    for i in 1:n
        acc = 0.0
        for _ in 1:4
            s ⊻= s << 13; s ⊻= s >> 7; s ⊻= s << 17
            acc += (s >> 11) * (1.0 / 9007199254740992.0)
        end
        out[i] = (acc - 2.0) * sqrt(3.0)
    end
    return out
end

"""
    gnf_adjoint_series(op, b, Ns) -> Dict(N => λ_N)

Truncated Neumann series of the adjoint, λ_N = Q Σ_{n<N} (P Aᵀ P)ⁿ P b with
Q = I − ĉĉᵀ (computed as Σ (Q P Aᵀ P)ⁿ Q P b), for every N in `Ns`.
"""
function gnf_adjoint_series(op::GNFStepOperator, b, Ns)
    Nmax = maximum(Ns)
    x = similar(op.chat); x .= op.active .* b
    x .-= dot(op.chat, x) .* op.chat
    acc = copy(x)
    y = similar(x)
    out = Dict{Int,typeof(x)}()
    1 in Ns && (out[1] = copy(acc))
    for it in 1:Nmax-1
        gnf_vjp!(y, op, x)
        y .-= dot(op.chat, y) .* op.chat
        x, y = y, x
        acc .+= x
        (it + 1) in Ns && (out[it+1] = copy(acc))
    end
    return out
end

# ============================================================
# Fixed-point and K-step adjoint solvers (fallbacks v and vi of the J2 gate)
# ============================================================

# y = M^K x with M = Q P Aᵀ P (adjoint iteration operator, Q = I − ĉĉᵀ), or with
# `transposed = true` y = (Mᵀ)^K x, Mᵀ = P A P Q. Buffers u, v distinct from x, y.
function _gnf_power!(y, op::GNFStepOperator, x, K::Integer, u, v; transposed::Bool=false)
    copyto!(u, x)
    for _ in 1:K
        if transposed
            u .-= dot(op.chat, u) .* op.chat
            gnf_jvp!(v, op, u)
        else
            gnf_vjp!(v, op, u)
            v .-= dot(op.chat, v) .* op.chat
        end
        u, v = v, u
    end
    copyto!(y, u)
    return y
end

"""
    gnf_adjoint_power!(y, op, x, K) -> y      y = M^K x,     M  = Q P Aᵀ P
    gnf_tangent_power!(y, op, x, K) -> y      y = (Mᵀ)^K x,  Mᵀ = P A P Q

K products of the adjoint iteration operator M (Q = I − ĉĉᵀ removes the mass
mode) or of its transpose; ⟨M^K l, v⟩ = ⟨l, (Mᵀ)^K v⟩ is the K-step form of T4c.
"""
gnf_adjoint_power!(y, op::GNFStepOperator, x, K::Integer) =
    _gnf_power!(y, op, x, K, similar(x), similar(x))
gnf_tangent_power!(y, op::GNFStepOperator, x, K::Integer) =
    _gnf_power!(y, op, x, K, similar(x), similar(x); transposed=true)

"""
    GNFKStepAdjoint(op, K)

Callable `A(y, x)`: y = x − M^K x (K VJPs, M = Q P Aᵀ P), the operator of the
K-step adjoint system of `gnf_adjoint_kstep_solve`.
"""
struct GNFKStepAdjoint{TO<:GNFStepOperator, TV<:AbstractVector{Float64}}
    op::TO
    K::Int
    u::TV
    v::TV
    w::TV
end
GNFKStepAdjoint(op::GNFStepOperator, K::Integer) =
    GNFKStepAdjoint(op, Int(K), similar(op.chat), similar(op.chat), similar(op.chat))

function (A::GNFKStepAdjoint)(y, x)
    _gnf_power!(A.w, A.op, x, A.K, A.u, A.v)
    y .= x .- A.w
    return y
end

# b' = Q P b and the projected warm start (zero if λ0 === nothing).
function _gnf_projected_rhs_start(op::GNFStepOperator, b, λ0)
    bp = similar(op.chat); bp .= op.active .* b
    bp .-= dot(op.chat, bp) .* op.chat
    λ = similar(bp)
    if λ0 === nothing
        fill!(λ, 0.0)
    else
        λ .= op.active .* λ0
        λ .-= dot(op.chat, λ) .* op.chat
    end
    return bp, λ
end

# Explicit residual ‖b' − (I − M) λ‖ / ‖b'‖ (one VJP).
function _gnf_true_relres(op::GNFStepOperator, bp, λ)
    y = similar(bp)
    gnf_vjp!(y, op, λ)
    y .-= dot(op.chat, y) .* op.chat
    y .= bp .- λ .+ y
    return norm(y) / max(norm(bp), 1e-300)
end

"""
    gnf_adjoint_fixed_point(op, b; tol=1e-6, check_every=1000, fit_checks=20,
                            maxiter=10^7, λ0=nothing, max_wall=Inf, snapshots=())
        -> (λ, η, info)

Adjoint by the plain fixed-point iteration λ ← M λ + b' (M = Q P Aᵀ P,
b' = Q P b), i.e. the truncated series continued to convergence, with the stop
rule of the forward driver (`run_gnf_periodic_steady`, `stop = :distance`):
every `check_every` iterations r2 = ‖λ_k − λ_(k−2)‖/‖λ_k‖ (the period-2 mode
μ ≈ −1 cancels in it), contraction ρ = exp of the slope of log r2 over the last
`fit_checks` checks. Converged when the distance estimate r2 ρ²/(1 − ρ²) < `tol`
and the explicit residual ‖b' − (I − M)λ̄‖/‖b'‖ of the mean λ̄ of the last two
iterates is ≤ `tol` (as `gnf_adjoint_kstep_solve`); the explicit residual (one
VJP, counted in `t_total`) is evaluated only at checks where the distance
estimate is below `tol`, and the iteration goes on while it is above `tol`.
Returns λ̄ (removes the period-2 oscillation), η = 0 and info
`(; converged, iters, relres, dist_est, contraction, ...)` in the layout of
`gnf_gmres!`, with `relres` the explicit residual of the returned λ,
`relres_true` the explicit residuals evaluated at checks and `relres_iter` the
distance estimate at each check. Stops only at a check (or after `maxiter` or
`max_wall`, unconverged). `snapshots`: distance tolerances above `tol` at which
the mean iterate is kept as `(; tol, iters, t, x)`.
"""
function gnf_adjoint_fixed_point(op::GNFStepOperator, b; tol::Real=1e-6, check_every::Integer=1000,
                                 fit_checks::Integer=20, maxiter::Integer=10^7, λ0=nothing,
                                 max_wall::Real=Inf, snapshots=())
    check_every >= 2 || throw(ArgumentError("check_every must be ≥ 2 (two-step change)"))
    t0 = time(); t_diag = 0.0
    bp, λ = _gnf_projected_rhs_start(op, b, λ0)
    y = similar(bp); λp = similar(bp); λp2 = similar(bp); xm = similar(bp)
    history = Tuple{Int,Float64,Float64,Float64,Float64}[]
    relres_iter = Float64[]; t_iter = Float64[]; relres_true = Tuple{Int,Float64}[]
    snap_tols = sort(collect(Float64, snapshots); rev=true)
    snaps = NamedTuple[]; next_snap = 1
    ρ = NaN; dist = Inf; converged = false; k = 0; relres = NaN
    while true
        k += 1
        (k + 1) % check_every == 0 && copyto!(λp2, λ)      # λ_(k−2) at the next check
        check = k % check_every == 0
        check && copyto!(λp, λ)                            # λ_(k−1)
        gnf_vjp!(y, op, λ)
        s = dot(op.chat, y)
        λ .= y .- s .* op.chat .+ bp
        check || continue
        nl = max(norm(λ), 1e-300)
        y .= λ .- λp; r1 = norm(y) / nl
        y .= λ .- λp2; r2 = norm(y) / nl
        push!(history, (k, r1, NaN, NaN, r2))
        ρ, dist = _gnf_distance_estimate(history, Int(fit_checks))
        push!(relres_iter, dist); push!(t_iter, time() - t0 - t_diag)
        while next_snap <= length(snap_tols) && dist <= snap_tols[next_snap]
            td = time()
            xs = similar(λ); xs .= 0.5 .* (λ .+ λp); _gnf_sync(xs)
            push!(snaps, (; tol=snap_tols[next_snap], iters=k, t=td - t0 - t_diag, x=xs))
            next_snap += 1
            t_diag += time() - td
        end
        isfinite(r2) || break
        if dist < tol
            # explicit one-step residual of the mean iterate (one VJP)
            xm .= 0.5 .* (λ .+ λp)
            relres = _gnf_true_relres(op, bp, xm)
            push!(relres_true, (k, relres))
            relres <= tol && (converged = true; break)
        end
        (k >= maxiter || time() - t0 - t_diag > max_wall) && break
    end
    λ .= 0.5 .* (λ .+ λp)
    _gnf_sync(λ)
    t_total = time() - t0 - t_diag
    if !converged
        td = time()
        relres = _gnf_true_relres(op, bp, λ)
        t_diag += time() - td
    end
    info = (; converged, iters=k, cycles=0, relres, relres_cycles=[relres], relres_true, relres_iter, t_iter,
            t_total, t_op=t_total, t_orth=0.0, t_diag, n_apply=k, restart=0, orth=:none,
            deflate=0, ritz=ComplexF64[], snaps, dist_est=dist, contraction=ρ, history,
            check_every=Int(check_every))
    return λ, 0.0, info
end

"""
    gnf_adjoint_kstep_solve(op, b; K=1001, tol=1e-6, restart=50, orth=:cgs2,
                            maxiter=10_000, λ0=nothing, max_wall=Inf, snapshots=())
        -> (λ, η, info)

Adjoint by GMRES on the K-step system (I − M^K) λ = Σ_(j<K) M^j b'
(M = Q P Aᵀ P, b' = Q P b): the system (I − M)λ = b' left-multiplied by the
K-term truncated series Σ_(j<K) M^j, so it has the same solution. One GMRES
iteration costs K VJPs (`GNFKStepAdjoint`), which amortises the
orthogonalisation. K must be odd: the period-2 mode μ ≈ −1 gives 1 − μ^K ≈ 0
for even K. Convergence: explicit residual ‖b' − (I − M)λ‖/‖b'‖ ≤ `tol`
(`info.relres`, as for `gnf_adjoint_solve`); GMRES first runs to `tol` on the
K-step residual, then, while the explicit one is above `tol`, continues from
its iterate with the K-step tolerance scaled down by the measured ratio
(`refine` rounds at most; the explicit residual, one VJP, counts in
`t_total`). `snapshots` apply to the K-step residual of the first round;
`info.t_rhs` = cost of the right-hand side (K − 1 VJPs, included in `t_total`
and in the snapshot times). Returns η = 0.
"""
function gnf_adjoint_kstep_solve(op::GNFStepOperator, b; K::Integer=1001, tol::Real=1e-6,
                                 restart::Integer=50, orth::Symbol=:cgs2, maxiter::Integer=10_000,
                                 λ0=nothing, max_wall::Real=Inf, snapshots=(), verbose::Bool=false,
                                 refine::Integer=3)
    isodd(K) || throw(ArgumentError("K must be odd (period-2 mode μ ≈ −1), got $K"))
    t0 = time()
    bp, x = _gnf_projected_rhs_start(op, b, λ0)
    rhs = copy(bp); u = copy(bp); v = similar(bp)
    for _ in 1:K-1
        gnf_vjp!(v, op, u)
        v .-= dot(op.chat, v) .* op.chat
        u, v = v, u
        rhs .+= u
    end
    _gnf_sync(rhs)
    t_rhs = time() - t0
    A = GNFKStepAdjoint(op, K)
    # GMRES on the K-step residual, then the explicit residual of (I − M)λ = b';
    # if it is above `tol`, continue from x with the K-step tolerance scaled down
    # by the measured ratio (at most `refine` extra rounds)
    tol_k = Float64(tol); rounds = 0; t_true = 0.0
    gi = nothing; iters = 0; n_apply = 0; t_gm = 0.0; t_op = 0.0; t_orth = 0.0; t_gdiag = 0.0
    snaps = NamedTuple[]; relres = Inf; relres_hist = Float64[]
    while true
        gr = gnf_gmres!(x, A, rhs; tol=tol_k, restart, maxiter=maxiter - iters, orth,
                        max_wall=max_wall - t_rhs - t_gm - t_true, verbose,
                        snapshots=rounds == 0 ? snapshots : ())
        for s in gr.snaps
            push!(snaps, (; tol=s.tol, iters=s.iters + iters, t=s.t + t_rhs + t_gm + t_true, x=s.x))
        end
        iters += gr.iters; n_apply += gr.n_apply; t_gm += gr.t_total; t_op += gr.t_op
        t_orth += gr.t_orth; t_gdiag += gr.t_diag
        gi = gr
        td = time()
        relres = _gnf_true_relres(op, bp, x)
        t_true += time() - td
        push!(relres_hist, relres)
        (relres <= tol || !gr.converged || rounds >= refine || iters >= maxiter ||
         t_rhs + t_gm + t_true > max_wall) && break
        tol_k *= 0.5 * tol / relres
        rounds += 1
    end
    converged = gi.converged && relres <= tol
    info = merge(gi, (; converged, iters, n_apply, t_op, t_orth, relres, relres_kstep=gi.relres,
                      relres_true_rounds=relres_hist, tol_kstep=tol_k, rounds,
                      t_total=t_rhs + t_gm + t_true, t_rhs, t_diag=t_gdiag, K=Int(K),
                      n_vjp=(K - 1) + K * n_apply + length(relres_hist), snaps))
    return x, 0.0, info
end

# Solver selection for `gnf_steady_adjoint` (`solver` keyword).
function _gnf_solve_adjoint(op::GNFStepOperator, b, solver::Symbol; kwargs...)
    solver === :gmres && return gnf_adjoint_solve(op, b; kwargs...)
    solver === :fixed_point && return gnf_adjoint_fixed_point(op, b; kwargs...)
    solver === :kstep && return gnf_adjoint_kstep_solve(op, b; kwargs...)
    throw(ArgumentError("solver must be :gmres, :fixed_point or :kstep, got :$solver"))
end

"""
    gnf_step_state(w, is_solid, closure, Fx, Fy) -> w'

One real GNF step (`gnf_step!`, split path) applied to a flat state vector, on
the backend of `w`; `is_solid`, `Fx`, `Fy` device arrays (Nx, Ny). The closure
should hold numbers of `eltype(w)`.
"""
function gnf_step_state(w, is_solid, closure, Fx, Fy)
    Nx, Ny = size(is_solid)
    f, tau = gnf_unpack_state(w, Nx, Ny)
    fo = similar(f)
    I1 = similar(tau); nu = similar(tau)
    gnf_step!(fo, f, tau, I1, nu, is_solid, closure, Fx, Fy)
    return gnf_pack_state(fo, tau)
end
