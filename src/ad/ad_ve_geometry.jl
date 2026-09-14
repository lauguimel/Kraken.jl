# Analytic embedded-g geometry derivatives d(geom)/dR for the VE shape-adjoint
# AD track + the assembly of the analytic dG/dR (forward-JVP of the coupled VE
# one-step operator seeded by d(geom)/dR) + the UNGAUGED open-BC adjoint solve.
# Companion to ad_ve_step.jl / ad_ve_forward.jl. Plain-Julia, Enzyme-FREE: the
# Enzyme forward-JVP goes through the `_ad_ve_dGdR_jvp` ext seam (src stub).
#
# Ported verbatim (preserving every formula) from the validated scratch chain
# (bench/scratch/ve_ad_c3_analytic.jl build_dcircle_geom_dR + _d_*_frac_dR +
# _d_cell_fraction_dR ; ve_ad_c3_finalclose.jl plain_adjoint), namespaced ad_ve_*.
#
# The net d(Cd_polymer)/dR is a ~20x catastrophic cancellation (explicit + state-
# response), so dG/dR MUST be exact (no central-FD truncation): these analytic
# field derivatives feed the production gradient. The sampled fields
# (cell_fraction, wall_inv_distance) use the chord / projected-offset closed forms
# whose field-level small-h FD is staircase noise; they are validated through the
# operator (per-channel lambda' dG/dR) in C2/A2, not by a field-FD.
#
# BIT-MIRROR of ad_ve_build_circle_geom (ad_ve_ops.jl) differentiated wrt R within
# FIXED topology (is_solid frozen, cut set frozen). The circle center is the
# control-volume frame (cx, cy) directly; q_wall uses the node frame (cx-.5, cy-.5).

# ----------------------------------------------------------------------------
# d(_circle_vface_frac)/dR.  frac = clamp(1 - inside/(y1-y0)).
#   half = sqrt(R^2 - dx^2),  d(half)/dR = R/half  (half>0 on the cut segment).
#   inside = min(y1, cy+half) - max(y0, cy-half).
#   d(frac)/dR = -d(inside)/dR / (y1-y0)  (0 where frac saturated at 0 or 1).
@inline function ad_ve_d_vface_frac_dR(x_face, y0, y1, cx, cy, R)
    dx = x_face - cx
    abs(dx) >= R && return 0.0                 # outside reach: frac==1 const
    half = sqrt(max(0.0, R * R - dx * dx))
    half <= 0.0 && return 0.0
    top = cy + half; bot = cy - half
    inside = min(y1, top) - max(y0, bot)
    (inside <= 0.0 || inside >= (y1 - y0)) && return 0.0   # saturated frac: d=0
    dhalf = R / half
    dinside = 0.0
    (top < y1) && (dinside += dhalf)           # top edge strictly inside cell -> moves
    (bot > y0) && (dinside += dhalf)           # bottom edge strictly inside cell -> moves
    return -dinside / (y1 - y0)
end

# d(_circle_hface_frac)/dR  (x<->y swap).
@inline function ad_ve_d_hface_frac_dR(y_face, x0, x1, cx, cy, R)
    dy = y_face - cy
    abs(dy) >= R && return 0.0
    half = sqrt(max(0.0, R * R - dy * dy))
    half <= 0.0 && return 0.0
    right = cx + half; left = cx - half
    inside = min(x1, right) - max(x0, left)
    (inside <= 0.0 || inside >= (x1 - x0)) && return 0.0
    dhalf = R / half
    dinside = 0.0
    (right < x1) && (dinside += dhalf)
    (left > x0) && (dinside += dhalf)
    return -dinside / (x1 - x0)
end

# d(cell_fraction)/dR. cell_fraction = (outside-fluid area)/(cell area). As R
# grows the circle sweeps outward, the fluid (outside) area shrinks by the in-cell
# circle-arc length per unit R: d(cf)/dR = -(arc length inside cell)/Acell. Integrate
# ds = R dtheta over the arc inside [x0,x1]x[y0,y1] (matches the sampled cf continuum
# limit; exact in R). Acell = (x1-x0)*(y1-y0) = 1.
function ad_ve_d_cell_fraction_dR(x0, y0, x1, y1, cx, cy, R; ns::Int=4096)
    inside_len = 0.0
    dtheta = 2pi / ns
    @inbounds for k in 0:(ns - 1)
        th = (k + 0.5) * dtheta
        x = cx + R * cos(th); y = cy + R * sin(th)
        if (x0 <= x <= x1) && (y0 <= y <= y1)
            inside_len += R * dtheta
        end
    end
    Acell = (x1 - x0) * (y1 - y0)
    return -inside_len / Acell
end

# ----------------------------------------------------------------------------
"""
    ad_ve_build_dcircle_geom_dR(Nx, Ny, cx, cy, R, base; samples=16, ns=8192)
        -> ADVEEmbeddedGeom

Analytic d(geom)/dR shadow as an `ADVEEmbeddedGeom`-shaped struct of derivatives
(is_solid/cut frozen; their derivatives are 0). Mirrors `ad_ve_build_circle_geom`'s
wall_n / wall_inv_distance assembly with the quotient/normalization derivatives.
The sampled centroid is treated R-frozen; the centroid-offset is carried along the
wall normal (proj), validated against the per-channel frozen-solid FD in C2/A2.
`base` is the geometry at R (for the saturated-frac mask + wall_n / dist values).
Mirror of scratch `build_dcircle_geom_dR`.
"""
function ad_ve_build_dcircle_geom_dR(Nx, Ny, cx, cy, R, base::ADVEEmbeddedGeom;
                                     samples=16, ns::Int=8192)
    dcf = zeros(Float64, Nx, Ny)
    dwf = zeros(Float64, Nx, Ny); def = zeros(Float64, Nx, Ny)
    dsf = zeros(Float64, Nx, Ny); dnf = zeros(Float64, Nx, Ny)
    dwall = zeros(Float64, Nx, Ny)
    dwnx = zeros(Float64, Nx, Ny); dwny = zeros(Float64, Nx, Ny)
    dwidc = zeros(Float64, Nx, Ny)
    tol = sqrt(eps(Float64))
    @inbounds for j in 1:Ny, i in 1:Nx
        x0 = Float64(i - 1); x1 = Float64(i)
        y0 = Float64(j - 1); y1 = Float64(j)
        dwest  = ad_ve_d_vface_frac_dR(x0, y0, y1, cx, cy, R)
        deast  = ad_ve_d_vface_frac_dR(x1, y0, y1, cx, cy, R)
        dsouth = ad_ve_d_hface_frac_dR(y0, x0, x1, cx, cy, R)
        dnorth = ad_ve_d_hface_frac_dR(y1, x0, x1, cx, cy, R)
        dwf[i, j] = dwest; def[i, j] = deast; dsf[i, j] = dsouth; dnf[i, j] = dnorth

        # cell_fraction: only nonzero where cut (interior frac==1 / solid frac==0:
        # both d=0 within frozen topology). Use the chord formula.
        c = base.cell_fraction[i, j]
        if tol < c < 1.0 - tol
            dcf[i, j] = ad_ve_d_cell_fraction_dR(x0, y0, x1, y1, cx, cy, R; ns=ns)
        end

        # wall_n = -(area_x, area_y)/len ; area_x = west-east, area_y = south-north.
        west  = base.west_fraction[i, j]; east  = base.east_fraction[i, j]
        south = base.south_fraction[i, j]; north = base.north_fraction[i, j]
        area_x = west - east
        area_y = south - north
        len = hypot(area_x, area_y)
        d_area_x = dwest - deast
        d_area_y = dsouth - dnorth
        dlen = (len > tol) ? (area_x * d_area_x + area_y * d_area_y) / len : 0.0
        dwall[i, j] = dlen
        if len > tol && tol < c < 1.0 - tol
            # wnx = -area_x/len ; d(wnx)/dR = -(d_area_x*len - area_x*dlen)/len^2
            dwnx[i, j] = -(d_area_x * len - area_x * dlen) / (len * len)
            dwny[i, j] = -(d_area_y * len - area_y * dlen) / (len * len)

            # wall_inv_distance = 1/dist ; dist = scd + nx*(cenx-xc)+ny*(ceny-yc).
            # scd = hypot(xc-cx, yc-cy) - R ; d(scd)/dR = -1 (center frozen).
            xc = (x0 + x1) / 2; yc = (y0 + y1) / 2
            nx = -area_x / len; ny = -area_y / len
            inv_d = base.wall_inv_distance_to_center[i, j]
            if inv_d > 0.0
                dist = 1.0 / inv_d
                scd = hypot(xc - cx, yc - cy) - R
                proj = dist - scd                          # = nx*ox + ny*oy (offset along n)
                # centroid frozen, n moves: approximate offset parallel to n (dominant).
                dproj = (dwnx[i, j] * nx + dwny[i, j] * ny) * proj
                ddist = -1.0 + dproj                       # d(scd)/dR = -1
                dwidc[i, j] = -ddist / (dist * dist)
            end
        end
    end
    return ADVEEmbeddedGeom(dcf, dwf, def, dsf, dnf, dwall, dwnx, dwny, dwidc,
                            falses(Nx, Ny), zeros(UInt8, Nx, Ny))
end

# ----------------------------------------------------------------------------
"""
    ad_ve_assemble_dGdR(w_star, geom, p; cx, cy, samples=16, ns=8192) -> Vector

Assemble the analytic dG/dR: build the d(geom)/dR shadow + the analytic dq_wall/dR
(node frame, reusing the production `dq_wall_dR_cylinder`), then one Enzyme
forward-JVP of `ad_ve_coupled_step!` seeded by both shadows (through the
`_ad_ve_dGdR_jvp` ext seam). FD-free; this is the chain that feeds the production
gradient (the net is a ~20x cancellation, so no central-FD of dG/dR is admissible).
Mirror of scratch `dGdR_analytic_jvp` (with the analytic geometry seed).
"""
function ad_ve_assemble_dGdR(w_star, geom::ADVEGeom, p::ADVECoupledParams;
                             cx, cy, samples=16, ns::Int=8192)
    Nx, Ny = p.Nx, p.Ny
    R = geom.radius
    dg = ad_ve_build_dcircle_geom_dR(Nx, Ny, cx, cy, R, geom.g; samples=samples, ns=ns)
    dq_wall = dq_wall_dR_cylinder(Nx, Ny, cx - 0.5, cy - 0.5, R; FT=Float64)
    return _ad_ve_dGdR_jvp(w_star, geom.g, dg, geom.q_wall, dq_wall, geom.u_profile, p)
end

# ----------------------------------------------------------------------------
"""
    ad_ve_mass_gradient(p) -> Vector{Float64}

The conserved-mass null mode of the coupled VE state for the gauge-augmented
adjoint (closed/periodic BC only). The f-block (1:9n) carries the rho=1 mass
mode (1.0 on every population); the psi-block (9n+1:12n) carries no mass mode (0).
The open-BC path does NOT use this (its ZouHe pins the mass mode).
"""
function ad_ve_mass_gradient(p::ADVECoupledParams)
    n = ad_ve_n(p)
    m = zeros(Float64, 12n)
    @inbounds for idx in 1:9n
        m[idx] = 1.0
    end
    return m
end

"""
    ad_ve_ungauged_adjoint(w_star, geom, p, rhs; gmres_tol=1e-11, restart, max_restarts)
        -> NamedTuple

Solve the UNGAUGED open-BC adjoint `(I - dG^T) lambda = dJ/dw` via the existing
`ad_gmres_givens`, with the matvec `v -> v - _ad_ve_vjp_GtT(...)`. The cylinder's
open BC (west ZouHe velocity inlet + east ZouHe pressure outlet) pins the rho=1
mass mode, so `(I - dG^T)` is non-singular — NO mass-gauge augmentation (the
gauged path for closed/periodic BC stays in `ad_gauge_augmented_adjoint`).
Returns `(; lambda, n_iter, linres, converged, original_linres)`. Mirror of
scratch `plain_adjoint`.
"""
function ad_ve_ungauged_adjoint(w_star, geom::ADVEGeom, p::ADVECoupledParams, rhs;
                                gmres_tol::Real=1e-11, restart::Int=640,
                                max_restarts::Int=30)
    apply_VJP = v -> _ad_ve_vjp_GtT(w_star, v, geom.g, geom.q_wall, geom.u_profile, p)
    apply_A = x -> x .- apply_VJP(x)
    sol = ad_gmres_givens(apply_A, vec(rhs); tol=gmres_tol, restart=restart,
                          max_restarts=max_restarts)
    lambda = sol.x
    # honest residual ||(I - dG^T) lambda - rhs|| / ||rhs|| (re-applies the VJP)
    gt = apply_VJP(lambda)
    rnorm = 0.0; bnorm = 0.0
    @inbounds for idx in eachindex(lambda)
        r = lambda[idx] - gt[idx] - rhs[idx]
        rnorm += r * r; bnorm += rhs[idx] * rhs[idx]
    end
    original_linres = sqrt(rnorm) / max(sqrt(bnorm), eps(Float64))
    return (; lambda=lambda, n_iter=sol.n_iter, linres=sol.rel,
            converged=sol.converged, original_linres=original_linres)
end
