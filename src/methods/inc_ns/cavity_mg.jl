# Backend-parametric steady SIMPLE incompressible solver for the 2D lid-driven
# cavity, built so the SAME source runs on CPU now and CUDA later.
#
# WHAT IS DIFFERENT FROM cavity.jl
# --------------------------------
# cavity.jl allocates plain CPU `Array`s, assembles sparse `-Laplacian`s and
# factorizes them with CHOLMOD (via the linear-solve seam). That is fast on CPU
# but (a) under-uses the GPU and (b) is not the matrix-free MG path. This file:
#
#   * allocates EVERY field through a backend array type `atype` (Array on CPU,
#     CuArray later) — no plain `zeros`,
#   * does EVERY elementwise op as a KernelAbstractions `@kernel` launched on the
#     KA backend `backend_ka` — no host scalar-indexing loops over device arrays,
#   * solves BOTH the pressure-correction Poisson AND the viscous/momentum
#     predictor with the matrix-free multigrid `solve_poisson_mg`,
#   * reductions (residual norms, gauge means) go through `sum`/`norm`, which
#     dispatch to the device implementation on GPU.
#
# So a CUDA run is `backend_ka = CUDABackend()`, `atype = CuArray{Float64}` with
# NO other change. We validate on the CPU backend here (no GPU locally).
#
# MOMENTUM OPERATOR (Laplace vs Helmholtz)
# ----------------------------------------
# The momentum predictor is solved by MG, which inverts `(-∇² + σ)` (the optional
# Helmholtz shift added to poisson_mg.jl). We keep advection in the RHS (deferred
# correction, first-order upwind on Rhie-Chow face fluxes) and fold the SIMPLE
# under-relaxation into a pseudo-transient diagonal shift `τ = ρ V_cell / (αu Δt*)`
# implemented as the Helmholtz σ. Concretely, dividing the viscous momentum eqn by
# the kinematic viscosity ν=μ (ρ=1), the per-cell predictor is
#
#     (-∇² + σ) u* = ( -conv_u - ∂p/∂x + src_lid )/ν + σ u_old
#
# with σ = (1/αu - 1) * a_diag / ν, a_diag a representative central diagonal of
# the viscous operator. The +σ u_old term makes the fixed point converge to the
# under-relaxed steady solution while keeping the operator strongly diagonally
# dominant (textbook MG convergence) — exactly the SIMPLE momentum under-relax,
# but expressed so a PLAIN-shift MG can solve it.
#
# DIRICHLET CONVENTION (must match poisson_mg.jl)
# -----------------------------------------------
# The MG Dirichlet operator uses the GHOST-0 ("+1/h² per missing face") boundary,
# i.e. the wall value lives at the ghost CELL CENTRE one half-cell outside the
# boundary. A non-homogeneous wall value u_w is folded into the RHS as +u_w/h²
# at the boundary row (the missing face contributes (u_w - u_c)/h²). The lid
# (u = U_lid on north) is injected this way. All velocity gradient/divergence
# wall treatments below use the SAME ghost-0 convention so the discrete operators
# are mutually consistent.
#
# Public entry point:
#   solve_incns_cavity_mg(; nx, ny, U_lid, Re, relax, tol, maxiter,
#                         backend_ka=CPU(), atype=Array{Float64}, ...)
#     -> NamedTuple(u, v, p, residual_history, iters, converged, ...)
#
# KA + stdlib only; no `using Kraken`, no AbstractMethod.

using KernelAbstractions
using LinearAlgebra: norm

# Matrix-free multigrid (pressure + momentum) and its backend tags.
if !isdefined(@__MODULE__, :solve_poisson_mg)
    include(joinpath(@__DIR__, "..", "..", "solve", "poisson_mg.jl"))
end

# ---------------------------------------------------------------------------
# KA kernels. Square grid (nx == ny == N), cell-centred on [0,L]^2, h = L/N.
# Field layout u[i,j], i in 1:N (x), j in 1:N (y). North = j=N (lid), south = j=1,
# west = i=1, east = i=N. Faces: uf[i,j] = east face of (i,j) (i in 1:N-1 active),
# vf[i,j] = north face of (i,j) (j in 1:N-1 active); wall faces carry 0.
# ---------------------------------------------------------------------------

# Compact cell-centred pressure gradient (gx=+dp/dx, gy=+dp/dy) with homogeneous
# Neumann at the walls (zero normal pressure gradient), consistent with the
# all-Neumann pressure Laplacian. Backend-generic @kernel.
@kernel function cav_compact_gradient_kernel!(gx, gy, @Const(p), invdx, invdy, N)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= N && j <= N
            pe = (i < N) ? (p[i, j] + p[i + 1, j]) * eltype(p)(0.5) : p[i, j]
            pw = (i > 1) ? (p[i - 1, j] + p[i, j]) * eltype(p)(0.5) : p[i, j]
            gx[i, j] = (pe - pw) * invdx
            pn = (j < N) ? (p[i, j] + p[i, j + 1]) * eltype(p)(0.5) : p[i, j]
            ps = (j > 1) ? (p[i, j - 1] + p[i, j]) * eltype(p)(0.5) : p[i, j]
            gy[i, j] = (pn - ps) * invdy
        end
    end
end

# Rhie-Chow face velocities. uf east faces (i<N), vf north faces (j<N). Wall
# faces set to 0 (no-slip; lid face vf at north is 0 since v_lid=0).
@kernel function cav_rhie_chow_kernel!(
    uf, vf, @Const(u), @Const(v), @Const(p), @Const(gpx), @Const(gpy),
    @Const(d_u), @Const(d_v), invdx, invdy, N,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= N && j <= N
            T = eltype(uf)
            half = T(0.5)
            # east face of (i,j)
            if i < N
                ubar = half * (u[i, j] + u[i + 1, j])
                dbar = half * (d_u[i, j] + d_u[i + 1, j])
                gp_face = (p[i + 1, j] - p[i, j]) * invdx
                gp_cell = half * (gpx[i, j] + gpx[i + 1, j])
                uf[i, j] = ubar - dbar * (gp_face - gp_cell)
            else
                uf[i, j] = zero(T)
            end
            # north face of (i,j)
            if j < N
                vbar = half * (v[i, j] + v[i, j + 1])
                dbar = half * (d_v[i, j] + d_v[i, j + 1])
                gp_face = (p[i, j + 1] - p[i, j]) * invdy
                gp_cell = half * (gpy[i, j] + gpy[i, j + 1])
                vf[i, j] = vbar - dbar * (gp_face - gp_cell)
            else
                vf[i, j] = zero(T)
            end
        end
    end
end

# Deferred-correction convection: conv = +div(u_face * phi) for phi = u and v,
# first-order upwind on the Rhie-Chow face fluxes. Wall fluxes vanish (no-slip);
# the lid carries U_lid for the advected u value at the north wall but vf=0 there
# so it does not contribute a flux — matching cavity.jl exactly.
@kernel function cav_convection_kernel!(
    conv_u, conv_v, @Const(u), @Const(v), @Const(uf), @Const(vf),
    invdx, invdy, U_lid, N,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= N && j <= N
            T = eltype(conv_u)
            Fe = (i < N) ? uf[i, j] : zero(T)
            Fw = (i > 1) ? uf[i - 1, j] : zero(T)
            Fn = (j < N) ? vf[i, j] : zero(T)
            Fs = (j > 1) ? vf[i, j - 1] : zero(T)

            uE = (i < N) ? (Fe >= 0 ? u[i, j] : u[i + 1, j]) : zero(T)
            uW = (i > 1) ? (Fw >= 0 ? u[i - 1, j] : u[i, j]) : zero(T)
            uN = (j < N) ? (Fn >= 0 ? u[i, j] : u[i, j + 1]) : T(U_lid)
            uS = (j > 1) ? (Fs >= 0 ? u[i, j - 1] : u[i, j]) : zero(T)

            vE = (i < N) ? (Fe >= 0 ? v[i, j] : v[i + 1, j]) : zero(T)
            vW = (i > 1) ? (Fw >= 0 ? v[i - 1, j] : v[i, j]) : zero(T)
            vN = (j < N) ? (Fn >= 0 ? v[i, j] : v[i, j + 1]) : zero(T)
            vS = (j > 1) ? (Fs >= 0 ? v[i, j - 1] : v[i, j]) : zero(T)

            conv_u[i, j] = (Fe * uE - Fw * uW) * invdx + (Fn * uN - Fs * uS) * invdy
            conv_v[i, j] = (Fe * vE - Fw * vW) * invdx + (Fn * vN - Fs * vS) * invdy
        end
    end
end

# Face divergence of the Rhie-Chow field (cell-centred). Wall faces are 0.
@kernel function cav_face_divergence_kernel!(divu, @Const(uf), @Const(vf), invdx, invdy, N)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= N && j <= N
            T = eltype(divu)
            ue = (i < N) ? uf[i, j] : zero(T)
            uw = (i > 1) ? uf[i - 1, j] : zero(T)
            vn = (j < N) ? vf[i, j] : zero(T)
            vs = (j > 1) ? vf[i, j - 1] : zero(T)
            divu[i, j] = (ue - uw) * invdx + (vn - vs) * invdy
        end
    end
end

# Correct FACE velocities from the pressure-correction `pcorr` (exact projection).
# The MG pressure operator solves (-∇²)pcorr = div(u*)/dscalar, so adding
# d^f * grad_face(pcorr) annihilates the face divergence in one shot.
@kernel function cav_correct_faces_kernel!(
    uf, vf, @Const(pcorr), @Const(d_u), @Const(d_v), invdx, invdy, N,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= N && j <= N
            T = eltype(uf)
            half = T(0.5)
            if i < N
                dbar = half * (d_u[i, j] + d_u[i + 1, j])
                uf[i, j] += dbar * (pcorr[i + 1, j] - pcorr[i, j]) * invdx
            end
            if j < N
                dbar = half * (d_v[i, j] + d_v[i, j + 1])
                vf[i, j] += dbar * (pcorr[i, j + 1] - pcorr[i, j]) * invdy
            end
        end
    end
end

# Build the momentum RHS for the MG solve of (-∇² + σ) phi* = rhs.
#
# The PHYSICAL viscous momentum equation is  μ(-∇²)φ + conv = -∂p/∂x  with
# Dirichlet walls. Dividing by ν=μ (ρ=1) gives the MG-form operator (-∇²)φ, so
# the convective and pressure-gradient terms divide by ν. The Dirichlet wall
# SOURCE pairs with the MG operator's own boundary diagonal (already the post-÷ν
# viscous coefficient) and is therefore NOT divided by ν.
#
# WALL TREATMENT (verified against the assembled +2/h² operator).
# On a cell-centred grid the wall sits at the FACE (half a cell from the centre),
# so the Dirichlet ghost is u_g = 2 u_w - u_c. The resulting OPERATOR is bit-for-
# bit identical to the MG ghost-0 Dirichlet operator: BOTH have boundary diagonal
# (4 + n_wall)/h². The "+1/h² vs +2/h²" distinction lives in the SOURCE only, not
# the stencil. A non-homogeneous wall value u_w therefore needs source +2 u_w/h²
# at the boundary row; homogeneous walls (u_w=0) need NO source. Only the moving
# lid (north, u-momentum) carries a source, +2 U_lid/h². (Verified empirically:
# the ghost-0 assembled operator == cavity.jl's +2/h² assembled operator,
# bit-identical, so NO diagonal/deferred correction is needed.)
#
# An optional pseudo-transient shift σ (operator + σ φ_old on the RHS) provides
# diagonal dominance/stability for the deferred-correction advection; it cancels
# at the fixed point (φ=φ_old), so the converged answer is σ-independent.
#   rhs = ( -conv - gp )/ν + src_lid + σ φ_old
@kernel function cav_momentum_rhs_kernel!(
    rhs, @Const(conv), @Const(gp), @Const(phi_old), invnu, sigma,
    invh2, wall_lid, U_lid, N,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= N && j <= N
            T = eltype(rhs)
            src = zero(T)
            # Lid source on the north row for the u-component only (wall_lid=1):
            # +2 U_lid/h² (the half-spacing ghost 2 u_w - u_c). All other walls
            # are homogeneous and need no source.
            if wall_lid == 1 && j == N
                src += T(2) * T(U_lid) * invh2
            end
            rhs[i, j] = (-conv[i, j] - gp[i, j]) * invnu + src + sigma * phi_old[i, j]
        end
    end
end

# Cell-velocity correction: phi += d * grad(pcorr)_component.
@kernel function cav_cell_correct_kernel!(phi, @Const(d), @Const(gcorr), N)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= N && j <= N
            phi[i, j] += d[i, j] * gcorr[i, j]
        end
    end
end

# Under-relaxed pressure update: p -= alpha_p * pcorr.
@kernel function cav_pressure_update_kernel!(p, @Const(pcorr), alpha_p, N)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= N && j <= N
            p[i, j] -= alpha_p * pcorr[i, j]
        end
    end
end

# Velocity-settle metric (max abs change), written without host scalar indexing:
# delta = phi_new - phi_old computed elementwise; the host calls maximum(abs, .).
@kernel function cav_delta_kernel!(du, @Const(unew), @Const(uold), @Const(vnew), @Const(vold), N)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= N && j <= N
            a = abs(unew[i, j] - uold[i, j])
            b = abs(vnew[i, j] - vold[i, j])
            du[i, j] = a > b ? a : b
        end
    end
end

# Subtract a scalar (gauge / zero-mean) from a field, elementwise.
@kernel function cav_shift_kernel!(a, s, N)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= N && j <= N
            a[i, j] -= s
        end
    end
end

# ---------------------------------------------------------------------------
# Host-side launch helpers (sync after each launch; backend-generic).
# ---------------------------------------------------------------------------
_cav_sync(kab) = KernelAbstractions.synchronize(kab)

# Checkerboard metric (high-frequency pressure energy / variance). Uses a kernel
# to form the 5-point high-pass and the squared deviation, then device reductions.
@kernel function cav_checker_kernel!(osc, dev2, @Const(p), pbar, N)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= N && j <= N
            T = eltype(p)
            if 2 <= i <= N - 1 && 2 <= j <= N - 1
                lap = p[i + 1, j] + p[i - 1, j] + p[i, j + 1] + p[i, j - 1] - T(4) * p[i, j]
                osc[i, j] = lap * lap
                d = p[i, j] - pbar
                dev2[i, j] = d * d
            else
                osc[i, j] = zero(T)
                dev2[i, j] = zero(T)
            end
        end
    end
end

"""
    solve_incns_cavity_mg(; nx=128, ny=128, U_lid=1.0, Re=100.0,
                          relax=(u=0.7, p=0.3), tol=1e-7, vel_tol=1e-6,
                          maxiter=8000, L=1.0, backend_ka=CPU(),
                          atype=Array{Float64}, mg_tol=1e-3, mg_maxcycles=20,
                          mom_mg_tol=1e-3, mom_mg_maxcycles=20, verbose=false)

Backend-parametric steady SIMPLE incompressible solver for the 2D lid-driven
cavity on `[0,L]^2` (square grid, `nx == ny`). Top wall moves at `u=U_lid, v=0`;
the other three walls are no-slip. `Re = U_lid*L/nu`, `rho=1` so `mu=nu`.

Both the viscous/momentum predictor AND the pressure-correction Poisson are
solved with the matrix-free multigrid `solve_poisson_mg`: the momentum solve uses
a Helmholtz shift `σ` (under-relaxation/pseudo-transient) and the lid Dirichlet
value folded into the RHS (ghost-0 convention); the pressure solve is all-Neumann
(`bc=:neumann`, σ=0). Every field is allocated via `atype` and every elementwise
op is a KA `@kernel` launched on `backend_ka`, so passing `backend_ka=CUDABackend()`
and `atype=CuArray{Float64}` runs the identical source on the GPU.

Returns a NamedTuple with `u, v, p` (nx x ny host `Array`s for convenience),
`residual_history`, `iters`, `converged`, grid metrics, and `checkerboard`.
"""
function solve_incns_cavity_mg(; nx::Integer=128, ny::Integer=128,
                               U_lid::Real=1.0, Re::Real=100.0,
                               relax=(u=0.7, p=0.3),
                               tol::Real=1e-7, vel_tol::Real=1e-6,
                               maxiter::Integer=8000,
                               L::Real=1.0,
                               backend_ka=KernelAbstractions.CPU(),
                               atype::Type{<:AbstractArray}=Array{Float64},
                               mg_tol::Real=1e-3, mg_maxcycles::Integer=20,
                               mom_mg_tol::Real=1e-3, mom_mg_maxcycles::Integer=20,
                               verbose::Bool=false)
    nx == ny || throw(ArgumentError("cavity_mg requires a square grid (nx == ny)"))
    N = Int(nx)
    U_lid = Float64(U_lid); Re = Float64(Re); L = Float64(L)
    αu = Float64(relax.u); αp = Float64(relax.p)
    nu = U_lid * L / Re                 # rho = 1 -> mu = nu
    dx = L / N; dy = L / N
    h = dx
    invdx = 1.0 / dx; invdy = 1.0 / dy
    invh2 = 1.0 / (h * h)
    kab = backend_ka

    # ----- SIMPLE response coefficient d = αu / a_p (relaxed) -----
    # Momentum under-relaxation is folded into the operator DIAGONAL via the
    # Helmholtz shift σ (below), so the effective momentum diagonal is the relaxed
    # a_p = a_visc/αu with a_visc = ν·4/h². The SIMPLE velocity-pressure response
    # is then d = αu / a_p = αu² / a_visc — identical to cavity.jl, which
    # reproduces Ghia's v-profile. Uniform d (collocated regular grid).
    a_visc = nu * 4.0 * invh2
    a_p = a_visc / αu
    dval = αu / a_p                     # = αu² / a_visc
    # Pressure-correction operator is dscalar*(-∇²); on the unit-square-scaled MG
    # the d-coefficient is folded into the RHS scaling (see below), so MG solves
    # the PLAIN Neumann Laplacian and we rescale.

    # ----- momentum under-relaxation as a Helmholtz shift (matches cavity.jl) ----
    # cavity.jl folds SIMPLE momentum under-relaxation into the operator diagonal:
    #   A_relaxed = A_visc + (1/αu - 1) Diag(a_visc),  rhs += (1/αu-1) a_visc φ_old.
    # Dividing by ν gives exactly the MG Helmholtz form (-∇² + σ) with
    #   σ = (1/αu - 1) · 4/h²   and   rhs += σ φ_old   (interior a_visc/ν = 4/h²).
    # The predictor IS the under-relaxed velocity, so we copy it straight to u,v
    # (no second, explicit relaxation). At the fixed point φ=φ_old the σ term
    # cancels and the converged field satisfies the un-relaxed steady momentum
    # equation — reproducing Ghia. σ also gives strong diagonal dominance, which
    # stabilises the deferred-correction advection at high Re and makes the MG
    # momentum solve converge in very few V-cycles.
    sigma_mom = (1.0 / αu - 1.0) * 4.0 * invh2

    # ----- backend fields (all via atype) -----
    mk() = (a = atype(undef, N, N); fill!(a, 0.0); a)
    u = mk(); v = mk(); p = mk()
    u_old = mk(); v_old = mk()
    gpx = mk(); gpy = mk()
    uf = mk(); vf = mk()
    conv_u = mk(); conv_v = mk()
    divstar = mk(); pcorr = mk()
    rhs_u = mk(); rhs_v = mk()
    d_u = mk(); d_v = mk()
    delta = mk()
    osc = mk(); dev2 = mk()
    # per-cell d = αu² h² / (ν (4 + n_wall)); dnum = αu² h² / ν = dval * 4
    # (dval = αu²/(ν·4/h²) is the interior value).
    dnum = αu * αu / (nu * invh2)
    cav_fill_d_kernel!(kab)(d_u, dnum, N; ndrange=(N, N))
    cav_fill_d_kernel!(kab)(d_v, dnum, N; ndrange=(N, N))
    _cav_sync(kab)

    xcenters = [(i - 0.5) * dx for i in 1:N]
    ycenters = [(j - 0.5) * dy for j in 1:N]

    residual_history = Float64[]
    converged = false
    iters = 0
    vel_change = Inf
    ref_flux = U_lid

    for it in 1:Int(maxiter)
        iters = it

        # 1. cell pressure gradient (for momentum source + Rhie-Chow cell term)
        cav_compact_gradient_kernel!(kab)(gpx, gpy, p, invdx, invdy, N; ndrange=(N, N))
        _cav_sync(kab)

        # 2. Rhie-Chow face velocities from current u,v,p
        cav_rhie_chow_kernel!(kab)(uf, vf, u, v, p, gpx, gpy, d_u, d_v,
                                   invdx, invdy, N; ndrange=(N, N))
        _cav_sync(kab)

        # 3. deferred-correction convection on the current field
        cav_convection_kernel!(kab)(conv_u, conv_v, u, v, uf, vf,
                                    invdx, invdy, U_lid, N; ndrange=(N, N))
        _cav_sync(kab)

        # snapshot old velocity for the settle metric and the σ*phi_old term
        copyto!(u_old, u); copyto!(v_old, v)

        # 4. momentum predictor: pure-viscous MG solve, then EXPLICIT under-relax.
        #    Lid + wall-at-face Dirichlet folded into the RHS (deferred). u carries
        #    the lid source on north (wall_lid=1); v has homogeneous walls (=0).
        cav_momentum_rhs_kernel!(kab)(rhs_u, conv_u, gpx, u_old, 1.0 / nu, sigma_mom,
                                      invh2, 1, U_lid, N; ndrange=(N, N))
        _cav_sync(kab)
        cav_momentum_rhs_kernel!(kab)(rhs_v, conv_v, gpy, v_old, 1.0 / nu, sigma_mom,
                                      invh2, 0, U_lid, N; ndrange=(N, N))
        _cav_sync(kab)

        # MG solves on the unit square (spacing 1/N). The physical operator uses
        # spacing h=L/N; on a square domain L scales (-∇²) by 1/L², so we pass the
        # RHS scaled by L² (and σ by L²). With L=1 these are identities. We warm-
        # start from the current velocity to cut V-cycles in the SIMPLE loop.
        L2 = L * L
        upred, _, _ = solve_poisson_mg(_scaled_rhs(rhs_u, L2, atype, kab, N), N;
                                       bc=:dirichlet, backend_ka=kab, atype=atype,
                                       tol=mom_mg_tol, maxcycles=Int(mom_mg_maxcycles),
                                       sigma=sigma_mom * L2, u0=u)
        vpred, _, _ = solve_poisson_mg(_scaled_rhs(rhs_v, L2, atype, kab, N), N;
                                       bc=:dirichlet, backend_ka=kab, atype=atype,
                                       tol=mom_mg_tol, maxcycles=Int(mom_mg_maxcycles),
                                       sigma=sigma_mom * L2, u0=v)
        # The σ-shifted predictor IS the under-relaxed velocity (relaxation is in
        # the operator/RHS, not applied again here): copy it straight in.
        copyto!(u, upred); copyto!(v, vpred)
        _cav_sync(kab)

        # settle metric (device reduction; no scalar host indexing)
        cav_delta_kernel!(kab)(delta, u, u_old, v, v_old, N; ndrange=(N, N))
        _cav_sync(kab)
        umax_prev = max(_field_absmax(u_old), _field_absmax(v_old), U_lid * eps())
        vel_change = _field_absmax(delta) / umax_prev

        # 5. Rhie-Chow faces from the predictor (continuity RHS)
        cav_compact_gradient_kernel!(kab)(gpx, gpy, p, invdx, invdy, N; ndrange=(N, N))
        _cav_sync(kab)
        cav_rhie_chow_kernel!(kab)(uf, vf, u, v, p, gpx, gpy, d_u, d_v,
                                   invdx, invdy, N; ndrange=(N, N))
        _cav_sync(kab)

        # 6. continuity residual = div(u*_face)
        cav_face_divergence_kernel!(kab)(divstar, uf, vf, invdx, invdy, N; ndrange=(N, N))
        _cav_sync(kab)
        res = sqrt(sum(abs2, divstar) / (N * N)) * dx / max(ref_flux, eps())
        push!(residual_history, res)

        if verbose && (it <= 5 || it % 100 == 0)
            @info "cavity_mg SIMPLE" it res vel_change
        end
        # Convergence: continuity residual below `tol` AND the velocity field
        # essentially steady (settle below `vel_tol`). The velocity-settle gate is
        # looser because explicit momentum under-relaxation settles more slowly
        # than the continuity residual drops; the physical solution is converged
        # once continuity is satisfied and the field stops moving at `vel_tol`.
        if res < tol && vel_change < vel_tol
            converged = true
            break
        end

        # 7. pressure-correction Poisson: dscalar*(-∇²) pcorr = div(u*).
        #    MG solves the PLAIN Neumann (-∇²) on the unit square: pass the RHS
        #    scaled so that (-∇²_unit) pcorr = div(u*) * L² / dval. Then pcorr is
        #    MINUS the physical pressure correction (sign matched to the projection).
        rhs_p = _scaled_rhs(divstar, L2 / dval, atype, kab, N)
        pc, _, _ = solve_poisson_mg(rhs_p, N; bc=:neumann, backend_ka=kab, atype=atype,
                                    tol=mg_tol, maxcycles=Int(mg_maxcycles), sigma=0.0)
        copyto!(pcorr, pc)
        # Neumann gauge: zero-mean.
        m = sum(pcorr) / (N * N)
        cav_shift_kernel!(kab)(pcorr, m, N; ndrange=(N, N))
        _cav_sync(kab)

        # 8. correct FACE velocities directly (exact projection)
        cav_correct_faces_kernel!(kab)(uf, vf, pcorr, d_u, d_v, invdx, invdy, N; ndrange=(N, N))
        _cav_sync(kab)

        # 9. correct CELL velocities with the pressure-correction cell gradient
        cav_compact_gradient_kernel!(kab)(gpx, gpy, pcorr, invdx, invdy, N; ndrange=(N, N))
        _cav_sync(kab)
        cav_cell_correct_kernel!(kab)(u, d_u, gpx, N; ndrange=(N, N))
        cav_cell_correct_kernel!(kab)(v, d_v, gpy, N; ndrange=(N, N))
        _cav_sync(kab)

        # 10. under-relaxed pressure update
        cav_pressure_update_kernel!(kab)(p, pcorr, αp, N; ndrange=(N, N))
        _cav_sync(kab)
    end

    # checkerboard metric via device reductions
    pbar = sum(p) / (N * N)
    cav_checker_kernel!(kab)(osc, dev2, p, pbar, N; ndrange=(N, N))
    _cav_sync(kab)
    checkerboard = sqrt(sum(osc) / max(sum(dev2), eps()))

    # Return host copies for convenience (interpolation/plotting on the host).
    return (; u=Array(u), v=Array(v), p=Array(p),
            residual_history, iters, converged, vel_change,
            dx, dy, xcenters, ycenters, nx=N, ny=N, U_lid, Re, mu=nu, L,
            checkerboard, sigma_mom)
end

# Device reduction for the max abs value (no host scalar indexing of device arrays).
_field_absmax(a) = maximum(abs, a)

# Return a NEW scaled RHS array (device) = src .* s. solve_poisson_mg copies the
# RHS into its hierarchy, so a fresh scaled array per call is fine and keeps every
# op on the backend (broadcast dispatches to the device).
function _scaled_rhs(src, s::Real, atype, kab, N)
    out = atype(undef, N, N)
    cav_scale_kernel!(kab)(out, src, eltype(out)(s), N; ndrange=(N, N))
    _cav_sync(kab)
    return out
end

@kernel function cav_scale_kernel!(out, @Const(src), s, N)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= N && j <= N
            out[i, j] = src[i, j] * s
        end
    end
end

# Per-cell SIMPLE response coefficient d = αu / a_p, with the boundary-aware
# relaxed diagonal a_p = ν·(4 + n_wall)/h² / αu (n_wall = number of Dirichlet wall
# faces the cell touches). So d = αu² h² / (ν (4 + n_wall)). Using the per-cell
# (not uniform) diagonal — matching cavity.jl — keeps the Rhie-Chow correction
# from over-shooting at boundary/corner cells (which otherwise spikes v in the
# corners and weakens the interior centreline). `dnum = αu² h² / ν`.
@kernel function cav_fill_d_kernel!(d, dnum, N)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= N && j <= N
            nwall = 0
            if i == 1; nwall += 1; end
            if i == N; nwall += 1; end
            if j == 1; nwall += 1; end
            if j == N; nwall += 1; end
            d[i, j] = eltype(d)(dnum) / (4 + nwall)
        end
    end
end
