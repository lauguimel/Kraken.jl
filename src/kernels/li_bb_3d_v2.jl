# =====================================================================
# TRT + LI-BB 3D kernel (D3Q19), Ginzburg-exact via DSL.
#
# Same recipe as li_bb_2d_v2.jl, ported to D3Q19:
#   PullHalfwayBB_3D → SolidInert_3D | ApplyLiBBPrePhase_3D →
#                      Moments_3D → CollideTRTDirect_3D →
#                      WriteMoments_3D
#
# Reuses `_libb_branch` from li_bb_2d.jl (the Bouzidi formula is
# stencil-independent).
# =====================================================================

const _TRT_LIBB_V2_SPEC_3D = LBMSpec(
    PullHalfwayBB_3D(), SolidInert_3D(),
    ApplyLiBBPrePhase_3D(),
    Moments_3D(), CollideTRTDirect_3D(),
    WriteMoments_3D();
    stencil = :D3Q19,
)

"""
    fused_trt_libb_v2_step_3d!(f_out, f_in, ρ, ux, uy, uz, is_solid,
                                q_wall, uw_x, uw_y, uw_z,
                                Nx, Ny, Nz, ν; Λ=3/16)

Single D3Q19 step: pull-stream + SolidInert on solid cells, or
ApplyLiBBPrePhase + TRT collision on fluid cells. Ginzburg-exact for
halfway-BB + TRT Λ = 3/16 on Couette; handles arbitrary q_w ∈ (0, 1]
via the full Bouzidi formula.
"""
function fused_trt_libb_v2_step_3d!(f_out, f_in, ρ, ux, uy, uz, is_solid,
                                      q_wall, uw_link_x, uw_link_y, uw_link_z,
                                      Nx, Ny, Nz, ν; Λ::Real=3/16)
    backend = KernelAbstractions.get_backend(f_in)
    ET = eltype(f_in)
    s_plus, s_minus = trt_rates(ν; Λ=Λ)
    kernel! = build_lbm_kernel(backend, _TRT_LIBB_V2_SPEC_3D)
    kernel!(f_out, ρ, ux, uy, uz, f_in, is_solid,
            q_wall, uw_link_x, uw_link_y, uw_link_z,
            Nx, Ny, Nz, ET(s_plus), ET(s_minus);
            ndrange=(Nx, Ny, Nz))
end

"""
    precompute_q_wall_sphere_3d(Nx, Ny, Nz, cx, cy, cz, R; FT=Float64)
        -> (q_wall, is_solid)

Analytical sub-cell `q_w` for a sphere of radius `R` centred at
`(cx, cy, cz)` on an `Nx×Ny×Nz` D3Q19 grid with unit lattice spacing.
For every fluid node whose link `q` crosses the sphere, solves the
quadratic `|x_f + t · c_q − c|² = R²` for the smallest positive
`t ∈ (0, 1]` and stores that as `q_wall[i,j,k,q]`.
"""
function precompute_q_wall_sphere_3d(Nx::Int, Ny::Int, Nz::Int,
                                      cx::Real, cy::Real, cz::Real,
                                      R::Real;
                                      FT::Type{<:AbstractFloat}=Float64)
    cxT, cyT, czT, RT = FT(cx), FT(cy), FT(cz), FT(R)
    R² = RT * RT
    is_solid = zeros(Bool, Nx, Ny, Nz)
    q_wall = zeros(FT, Nx, Ny, Nz, 19)
    cxs = velocities_x(D3Q19())
    cys = velocities_y(D3Q19())
    czs = velocities_z(D3Q19())

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        xf = FT(i - 1); yf = FT(j - 1); zf = FT(k - 1)
        dx_f = xf - cxT; dy_f = yf - cyT; dz_f = zf - czT
        if dx_f * dx_f + dy_f * dy_f + dz_f * dz_f ≤ R²
            is_solid[i, j, k] = true
            continue
        end
        for q in 2:19
            cqx = FT(cxs[q]); cqy = FT(cys[q]); cqz = FT(czs[q])
            xn = xf + cqx; yn = yf + cqy; zn = zf + cqz
            dx_n = xn - cxT; dy_n = yn - cyT; dz_n = zn - czT
            if dx_n * dx_n + dy_n * dy_n + dz_n * dz_n > R²
                continue
            end
            a = cqx * cqx + cqy * cqy + cqz * cqz
            b = FT(2) * (dx_f * cqx + dy_f * cqy + dz_f * cqz)
            c = dx_f * dx_f + dy_f * dy_f + dz_f * dz_f - R²
            disc = b * b - FT(4) * a * c
            disc < zero(FT) && continue
            sd = sqrt(disc)
            t1 = (-b - sd) / (FT(2) * a)
            t2 = (-b + sd) / (FT(2) * a)
            t = t1 > zero(FT) ? t1 : t2
            if t > zero(FT) && t ≤ one(FT)
                q_wall[i, j, k, q] = t
            end
        end
    end
    return q_wall, is_solid
end

"""
    compute_drag_libb_3d(f_post, q_wall, Nx, Ny, Nz)
        -> (Fx, Fy, Fz)

D3Q19 MEA drag/lift for a stationary LI-BB wall. Same halfway-BB
convention as the 2D version (`F = 2·c_q·f_q(post-coll)`).
"""
function compute_drag_libb_3d(f_post, q_wall, Nx::Int, Ny::Int, Nz::Int)
    f = Array(f_post)
    qw = Array(q_wall)
    cxs = velocities_x(D3Q19())
    cys = velocities_y(D3Q19())
    czs = velocities_z(D3Q19())
    Fx = 0.0; Fy = 0.0; Fz = 0.0
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        for q in 2:19
            if qw[i, j, k, q] > 0
                fv = Float64(f[i, j, k, q])
                Fx += 2.0 * Float64(cxs[q]) * fv
                Fy += 2.0 * Float64(cys[q]) * fv
                Fz += 2.0 * Float64(czs[q]) * fv
            end
        end
    end
    return (Fx = Fx, Fy = Fy, Fz = Fz)
end

"""
    run_sphere_libb_3d(; Nx=120, Ny=60, Nz=60, radius=8, u_in=0.04,
                        ν=…, max_steps=20000, avg_window=5000)

3D sphere in uniform x-flow. Halfway-BB on y/z walls (kernel
fallback). Equilibrium velocity inlet (i=1), Neumann outlet (i=Nx).
Drag integrated via `compute_drag_libb_3d` over `avg_window` final
steps.

Returns a NamedTuple with (; ρ, ux, uy, uz, Cd, Fx, Fy, Fz, ...).
Reference: 3D sphere in uniform flow at Re=20, Cd ≈ 2.6 (Clift et al.
1978; confinement-adjusted).
"""
function run_sphere_libb_3d(; Nx::Int=120, Ny::Int=60, Nz::Int=60,
                              cx::Union{Nothing,Real}=nothing,
                              cy::Union{Nothing,Real}=nothing,
                              cz::Union{Nothing,Real}=nothing,
                              radius::Real=8,
                              u_in::Real=0.04, ν::Real=0.04,
                              max_steps::Int=20_000,
                              avg_window::Int=5_000,
                              backend=KernelAbstractions.CPU(),
                              T::Type{<:AbstractFloat}=Float64)
    cx = isnothing(cx) ? Nx ÷ 4 : Float64(cx)
    cy = isnothing(cy) ? Ny ÷ 2 : Float64(cy)
    cz = isnothing(cz) ? Nz ÷ 2 : Float64(cz)

    # Precompute on CPU (analytic geometry, not kernel-launched)
    q_wall_h, is_solid_h = precompute_q_wall_sphere_3d(Nx, Ny, Nz, cx, cy, cz,
                                                        radius; FT=T)
    f_in_h = zeros(T, Nx, Ny, Nz, 19)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx, q in 1:19
        f_in_h[i, j, k, q] = Kraken.equilibrium(D3Q19(), one(T), T(u_in),
                                                 zero(T), zero(T), q)
    end

    # Allocate on the target backend
    q_wall   = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19)
    is_solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny, Nz)
    uw_x     = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19)
    uw_y     = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19)
    uw_z     = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19)
    f_in     = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19)
    f_out    = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19)
    ρ        = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz)
    ux       = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz)
    uy       = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz)
    uz       = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz)

    copyto!(q_wall, q_wall_h)
    copyto!(is_solid, is_solid_h)
    fill!(uw_x, zero(T)); fill!(uw_y, zero(T)); fill!(uw_z, zero(T))
    copyto!(f_in, f_in_h)
    fill!(ρ, one(T));  fill!(ux, zero(T))
    fill!(uy, zero(T)); fill!(uz, zero(T))

    # Equilibrium-inlet slab and Neumann-outlet slab staged on host for
    # the per-step CPU→device patch.
    f_inlet_slab_h = zeros(T, Ny, Nz, 19)
    for k in 1:Nz, j in 1:Ny, q in 1:19
        f_inlet_slab_h[j, k, q] = Kraken.equilibrium(D3Q19(), one(T),
                                                      T(u_in),
                                                      zero(T), zero(T), q)
    end
    f_inlet_slab = KernelAbstractions.allocate(backend, T, Ny, Nz, 19)
    copyto!(f_inlet_slab, f_inlet_slab_h)

    Fx_sum = 0.0; Fy_sum = 0.0; Fz_sum = 0.0; n_avg = 0

    for step in 1:max_steps
        fused_trt_libb_v2_step_3d!(f_out, f_in, ρ, ux, uy, uz, is_solid,
                                    q_wall, uw_x, uw_y, uw_z,
                                    Nx, Ny, Nz, T(ν))

        # Inlet / outlet slab patches via views (GPU-safe).
        @views f_out[1,  :, :, :] .= f_inlet_slab
        @views f_out[Nx, :, :, :] .= f_out[Nx-1, :, :, :]

        if step > max_steps - avg_window
            drag = compute_drag_libb_3d(f_out, q_wall, Nx, Ny, Nz)
            Fx_sum += drag.Fx; Fy_sum += drag.Fy; Fz_sum += drag.Fz
            n_avg += 1
        end

        f_in, f_out = f_out, f_in
    end
    KernelAbstractions.synchronize(backend)

    Fx_avg = Fx_sum / n_avg
    Fy_avg = Fy_sum / n_avg
    Fz_avg = Fz_sum / n_avg
    D = 2 * Float64(radius)
    A = π * Float64(radius)^2        # cross-section area
    u_ref = Float64(u_in)
    Cd = 2.0 * Fx_avg / (u_ref^2 * A)

    return (; ρ = Array(ρ), ux = Array(ux), uy = Array(uy), uz = Array(uz),
             Cd, Fx = Fx_avg, Fy = Fy_avg, Fz = Fz_avg,
             q_wall = Array(q_wall), is_solid = Array(is_solid),
             u_ref, D, A)
end
