# Standalone (no Kraken) copies of the constructs in src/ad/ad_ve_ops.jl:455-541
# (`ad_ve_advect_prodbc` and its helpers), one variant per suspect, plus the
# synthetic data and the reverse/forward driver shared by every S-variant.
# Issue #41 probe. Not part of the test suite.
using Enzyme, LinearAlgebra

const RT_REV = Enzyme.set_runtime_activity(Enzyme.Reverse)
const RT_FWD = Enzyme.set_runtime_activity(Enzyme.Forward)

# ---- helpers, verbatim from ad_ve_ops.jl -----------------------------------
@inline function superbee_limiter(r)
    one_r = one(r)
    two_r = one_r + one_r
    return max(zero(r), max(min(two_r * r, one_r), min(r, two_r)))
end

@inline function muscl_superbee_face_value(far_upwind, upwind, downwind)
    d_up = upwind - far_upwind
    d_down = downwind - upwind
    r = ifelse(d_down == zero(d_down), zero(d_down), d_up / d_down)
    return upwind + (one(r) / (one(r) + one(r))) * superbee_limiter(r) * d_down
end

@inline function is_cylinder_band(is_solid, i, j, Nx, Ny)
    if i <= 2 || i >= Nx - 1 || j <= 2 || j >= Ny - 1
        return false
    end
    return is_solid[i - 2, j] | is_solid[i - 1, j] |
           is_solid[i + 1, j] | is_solid[i + 2, j] |
           is_solid[i, j - 2] | is_solid[i, j - 1] |
           is_solid[i, j + 1] | is_solid[i, j + 2]
end

@inline function muscl_relax_rhs(phi, ux_face, uy_face, is_solid, i, j, Nx, Ny,
                                 inv_dx, inv_dy)
    ue = ux_face[i + 1, j]; uw = ux_face[i, j]
    vn = uy_face[i, j + 1]; vs = uy_face[i, j]
    phie = if ue >= 0.0
        (i > 1 && !is_solid[i - 1, j] && !is_solid[i + 1, j]) ?
            muscl_superbee_face_value(phi[i - 1, j], phi[i, j], phi[i + 1, j]) : phi[i, j]
    else
        (i + 2 <= Nx && !is_solid[i + 2, j] && !is_solid[i + 1, j]) ?
            muscl_superbee_face_value(phi[i + 2, j], phi[i + 1, j], phi[i, j]) : phi[i + 1, j]
    end
    phiw = if uw >= 0.0
        (i - 2 >= 1 && !is_solid[i - 2, j] && !is_solid[i - 1, j]) ?
            muscl_superbee_face_value(phi[i - 2, j], phi[i - 1, j], phi[i, j]) : phi[i - 1, j]
    else
        (i + 1 <= Nx && !is_solid[i + 1, j] && !is_solid[i - 1, j]) ?
            muscl_superbee_face_value(phi[i + 1, j], phi[i, j], phi[i - 1, j]) : phi[i, j]
    end
    phin = if vn >= 0.0
        (j > 1 && !is_solid[i, j - 1] && !is_solid[i, j + 1]) ?
            muscl_superbee_face_value(phi[i, j - 1], phi[i, j], phi[i, j + 1]) : phi[i, j]
    else
        (j + 2 <= Ny && !is_solid[i, j + 2] && !is_solid[i, j + 1]) ?
            muscl_superbee_face_value(phi[i, j + 2], phi[i, j + 1], phi[i, j]) : phi[i, j + 1]
    end
    phis = if vs >= 0.0
        (j - 2 >= 1 && !is_solid[i, j - 2] && !is_solid[i, j - 1]) ?
            muscl_superbee_face_value(phi[i, j - 2], phi[i, j - 1], phi[i, j]) : phi[i, j - 1]
    else
        (j + 1 <= Ny && !is_solid[i, j + 1] && !is_solid[i, j - 1]) ?
            muscl_superbee_face_value(phi[i, j + 1], phi[i, j], phi[i, j - 1]) : phi[i, j]
    end
    flux_div = (ue * phie - uw * phiw) * inv_dx + (vn * phin - vs * phis) * inv_dy
    divu = (ue - uw) * inv_dx + (vn - vs) * inv_dy
    return -(flux_div - phi[i, j] * divu)
end

# ---- V0: verbatim ad_ve_advect_prodbc ---------------------------------------
function advect_v0(phi, ux_face, uy_face, is_solid, Nx, Ny, east_phi)
    adv = zeros(Nx, Ny)
    wbc(i, j) = i > 1 ? phi[i - 1, j] : 0.0
    ebc(i, j) = i < Nx ? phi[i + 1, j] : east_phi[j]
    sbc(i, j) = j > 1 ? phi[i, j - 1] : phi[i, j]
    nbc(i, j) = j < Ny ? phi[i, j + 1] : phi[i, j]
    rus(i, j) = begin
        ue = ux_face[i + 1, j]; uw = ux_face[i, j]; vn = uy_face[i, j + 1]; vs = uy_face[i, j]
        phie = ue >= 0 ? phi[i, j] : ebc(i, j)
        phiw = uw >= 0 ? wbc(i, j) : phi[i, j]
        phin = vn >= 0 ? phi[i, j] : nbc(i, j)
        phis = vs >= 0 ? sbc(i, j) : phi[i, j]
        fl = (ue * phie - uw * phiw) + (vn * phin - vs * phis); du = (ue - uw) + (vn - vs)
        -(fl - phi[i, j] * du)
    end
    @inbounds for j in 1:Ny, i in 1:Nx
        if is_solid[i, j]
            adv[i, j] = 0.0; continue
        end
        if i <= 2 || i >= Nx - 1 || j <= 2 || j >= Ny - 1 ||
           is_solid[i - 2, j] || is_solid[i - 1, j] || is_solid[i + 1, j] || is_solid[i + 2, j] ||
           is_solid[i, j - 2] || is_solid[i, j - 1] || is_solid[i, j + 1] || is_solid[i, j + 2]
            adv[i, j] = phi[i, j] + rus(i, j)
        else
            adv[i, j] = phi[i, j] + muscl_relax_rhs(phi, ux_face, uy_face, is_solid, i, j, Nx, Ny, 1.0, 1.0)
        end
    end
    @inbounds for j in 1:Ny, i in 1:Nx
        if !is_solid[i, j] && is_cylinder_band(is_solid, i, j, Nx, Ny)
            adv[i, j] = phi[i, j] + muscl_relax_rhs(phi, ux_face, uy_face, is_solid, i, j, Nx, Ny, 1.0, 1.0)
        end
    end
    return adv
end

# ---- variant: the five closures replaced by plain functions -----------------
@inline wbc_nc(phi, i, j) = i > 1 ? phi[i - 1, j] : 0.0
@inline ebc_nc(phi, east_phi, i, j, Nx) = i < Nx ? phi[i + 1, j] : east_phi[j]
@inline sbc_nc(phi, i, j) = j > 1 ? phi[i, j - 1] : phi[i, j]
@inline nbc_nc(phi, i, j, Ny) = j < Ny ? phi[i, j + 1] : phi[i, j]
@inline function rus_nc(phi, ux_face, uy_face, east_phi, Nx, Ny, i, j)
    ue = ux_face[i + 1, j]; uw = ux_face[i, j]; vn = uy_face[i, j + 1]; vs = uy_face[i, j]
    phie = ue >= 0 ? phi[i, j] : ebc_nc(phi, east_phi, i, j, Nx)
    phiw = uw >= 0 ? wbc_nc(phi, i, j) : phi[i, j]
    phin = vn >= 0 ? phi[i, j] : nbc_nc(phi, i, j, Ny)
    phis = vs >= 0 ? sbc_nc(phi, i, j) : phi[i, j]
    fl = (ue * phie - uw * phiw) + (vn * phin - vs * phis); du = (ue - uw) + (vn - vs)
    return -(fl - phi[i, j] * du)
end

function advect_noclosure(phi, ux_face, uy_face, is_solid, Nx, Ny, east_phi)
    adv = zeros(Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        if is_solid[i, j]
            adv[i, j] = 0.0; continue
        end
        if i <= 2 || i >= Nx - 1 || j <= 2 || j >= Ny - 1 ||
           is_solid[i - 2, j] || is_solid[i - 1, j] || is_solid[i + 1, j] || is_solid[i + 2, j] ||
           is_solid[i, j - 2] || is_solid[i, j - 1] || is_solid[i, j + 1] || is_solid[i, j + 2]
            adv[i, j] = phi[i, j] + rus_nc(phi, ux_face, uy_face, east_phi, Nx, Ny, i, j)
        else
            adv[i, j] = phi[i, j] + muscl_relax_rhs(phi, ux_face, uy_face, is_solid, i, j, Nx, Ny, 1.0, 1.0)
        end
    end
    @inbounds for j in 1:Ny, i in 1:Nx
        if !is_solid[i, j] && is_cylinder_band(is_solid, i, j, Nx, Ny)
            adv[i, j] = phi[i, j] + muscl_relax_rhs(phi, ux_face, uy_face, is_solid, i, j, Nx, Ny, 1.0, 1.0)
        end
    end
    return adv
end

# ---- variant: pass 1 with rusanov everywhere, no MUSCL, no pass 2 -----------
function advect_pass1(phi, ux_face, uy_face, is_solid, Nx, Ny, east_phi)
    adv = zeros(Nx, Ny)
    wbc(i, j) = i > 1 ? phi[i - 1, j] : 0.0
    ebc(i, j) = i < Nx ? phi[i + 1, j] : east_phi[j]
    sbc(i, j) = j > 1 ? phi[i, j - 1] : phi[i, j]
    nbc(i, j) = j < Ny ? phi[i, j + 1] : phi[i, j]
    rus(i, j) = begin
        ue = ux_face[i + 1, j]; uw = ux_face[i, j]; vn = uy_face[i, j + 1]; vs = uy_face[i, j]
        phie = ue >= 0 ? phi[i, j] : ebc(i, j)
        phiw = uw >= 0 ? wbc(i, j) : phi[i, j]
        phin = vn >= 0 ? phi[i, j] : nbc(i, j)
        phis = vs >= 0 ? sbc(i, j) : phi[i, j]
        fl = (ue * phie - uw * phiw) + (vn * phin - vs * phis); du = (ue - uw) + (vn - vs)
        -(fl - phi[i, j] * du)
    end
    @inbounds for j in 1:Ny, i in 1:Nx
        if is_solid[i, j]
            adv[i, j] = 0.0; continue
        end
        adv[i, j] = phi[i, j] + rus(i, j)
    end
    return adv
end

# ---- variant: MUSCL only, interior cells, no closures, no band logic --------
function advect_musclonly(phi, ux_face, uy_face, is_solid, Nx, Ny, east_phi)
    adv = zeros(Nx, Ny)
    @inbounds for j in 3:(Ny - 2), i in 3:(Nx - 2)
        if is_solid[i, j]
            adv[i, j] = 0.0; continue
        end
        adv[i, j] = phi[i, j] + muscl_relax_rhs(phi, ux_face, uy_face, is_solid, i, j, Nx, Ny, 1.0, 1.0)
    end
    return adv
end

# ---- variant: no @inbounds ---------------------------------------------------
function advect_noinbounds(phi, ux_face, uy_face, is_solid, Nx, Ny, east_phi)
    adv = zeros(Nx, Ny)
    wbc(i, j) = i > 1 ? phi[i - 1, j] : 0.0
    ebc(i, j) = i < Nx ? phi[i + 1, j] : east_phi[j]
    sbc(i, j) = j > 1 ? phi[i, j - 1] : phi[i, j]
    nbc(i, j) = j < Ny ? phi[i, j + 1] : phi[i, j]
    rus(i, j) = begin
        ue = ux_face[i + 1, j]; uw = ux_face[i, j]; vn = uy_face[i, j + 1]; vs = uy_face[i, j]
        phie = ue >= 0 ? phi[i, j] : ebc(i, j)
        phiw = uw >= 0 ? wbc(i, j) : phi[i, j]
        phin = vn >= 0 ? phi[i, j] : nbc(i, j)
        phis = vs >= 0 ? sbc(i, j) : phi[i, j]
        fl = (ue * phie - uw * phiw) + (vn * phin - vs * phis); du = (ue - uw) + (vn - vs)
        -(fl - phi[i, j] * du)
    end
    for j in 1:Ny, i in 1:Nx
        if is_solid[i, j]
            adv[i, j] = 0.0; continue
        end
        if i <= 2 || i >= Nx - 1 || j <= 2 || j >= Ny - 1 ||
           is_solid[i - 2, j] || is_solid[i - 1, j] || is_solid[i + 1, j] || is_solid[i + 2, j] ||
           is_solid[i, j - 2] || is_solid[i, j - 1] || is_solid[i, j + 1] || is_solid[i, j + 2]
            adv[i, j] = phi[i, j] + rus(i, j)
        else
            adv[i, j] = phi[i, j] + muscl_relax_rhs(phi, ux_face, uy_face, is_solid, i, j, Nx, Ny, 1.0, 1.0)
        end
    end
    for j in 1:Ny, i in 1:Nx
        if !is_solid[i, j] && is_cylinder_band(is_solid, i, j, Nx, Ny)
            adv[i, j] = phi[i, j] + muscl_relax_rhs(phi, ux_face, uy_face, is_solid, i, j, Nx, Ny, 1.0, 1.0)
        end
    end
    return adv
end

# ---- variant: result array preallocated by the caller ------------------------
function advect_prealloc!(adv, phi, ux_face, uy_face, is_solid, Nx, Ny, east_phi)
    wbc(i, j) = i > 1 ? phi[i - 1, j] : 0.0
    ebc(i, j) = i < Nx ? phi[i + 1, j] : east_phi[j]
    sbc(i, j) = j > 1 ? phi[i, j - 1] : phi[i, j]
    nbc(i, j) = j < Ny ? phi[i, j + 1] : phi[i, j]
    rus(i, j) = begin
        ue = ux_face[i + 1, j]; uw = ux_face[i, j]; vn = uy_face[i, j + 1]; vs = uy_face[i, j]
        phie = ue >= 0 ? phi[i, j] : ebc(i, j)
        phiw = uw >= 0 ? wbc(i, j) : phi[i, j]
        phin = vn >= 0 ? phi[i, j] : nbc(i, j)
        phis = vs >= 0 ? sbc(i, j) : phi[i, j]
        fl = (ue * phie - uw * phiw) + (vn * phin - vs * phis); du = (ue - uw) + (vn - vs)
        -(fl - phi[i, j] * du)
    end
    @inbounds for j in 1:Ny, i in 1:Nx
        if is_solid[i, j]
            adv[i, j] = 0.0; continue
        end
        if i <= 2 || i >= Nx - 1 || j <= 2 || j >= Ny - 1 ||
           is_solid[i - 2, j] || is_solid[i - 1, j] || is_solid[i + 1, j] || is_solid[i + 2, j] ||
           is_solid[i, j - 2] || is_solid[i, j - 1] || is_solid[i, j + 1] || is_solid[i, j + 2]
            adv[i, j] = phi[i, j] + rus(i, j)
        else
            adv[i, j] = phi[i, j] + muscl_relax_rhs(phi, ux_face, uy_face, is_solid, i, j, Nx, Ny, 1.0, 1.0)
        end
    end
    @inbounds for j in 1:Ny, i in 1:Nx
        if !is_solid[i, j] && is_cylinder_band(is_solid, i, j, Nx, Ny)
            adv[i, j] = phi[i, j] + muscl_relax_rhs(phi, ux_face, uy_face, is_solid, i, j, Nx, Ny, 1.0, 1.0)
        end
    end
    return nothing
end

# ---- synthetic data: the FAST 24x24 cut cylinder of the test, node frame -----
function standalone_data(; Nx=24, Ny=24, cx=12.35, cy=11.65, R=5.13, mask=BitMatrix)
    solid = falses(Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        dx = (i - 1) - (cx - 0.5); dy = (j - 1) - (cy - 0.5)
        solid[i, j] = dx * dx + dy * dy <= R * R
    end
    is_solid = mask(solid)
    phi = zeros(Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        is_solid[i, j] && continue
        x = (i - 1) / Nx; y = (j - 1) / Ny
        phi[i, j] = 0.05 * sin(3pi * x) * cos(2pi * y)
    end
    ux_face = zeros(Nx + 1, Ny); uy_face = zeros(Nx, Ny + 1)
    for J in 1:Ny, I in 1:(Nx + 1)
        il = max(I - 1, 1); ir = min(I, Nx)
        (is_solid[il, J] || is_solid[ir, J]) && continue
        ux_face[I, J] = 2e-4 * (0.5 + sin(2pi * (J - 1) / Ny) + 0.3 * cos(2pi * (I - 1) / Nx))
    end
    for J in 1:(Ny + 1), I in 1:Nx
        (J == 1 || J == Ny + 1) && continue
        (is_solid[I, J - 1] || is_solid[I, J]) && continue
        uy_face[I, J] = 1e-4 * sin(2pi * (I - 1) / Nx) * cos(2pi * (J - 1) / Ny)
    end
    east_phi = phi[Nx, :]
    return (; Nx, Ny, is_solid, phi, ux_face, uy_face, east_phi)
end

_seed(len, a, b) = [sin(a * idx + b) for idx in 1:len]

# Runs reverse then forward (or the opposite) on `wrap!(out, phi, ux_face,
# uy_face, is_solid, Nx, Ny, east_phi)`, prints a marker after each call and
# the transpose identity v.(J u) == (J^T v).u at the end.
function run_standalone_case(name, wrap!, d; forward_first=false,
                             revmode=RT_REV, fwdmode=RT_FWD)
    (; Nx, Ny, is_solid, phi, ux_face, uy_face, east_phi) = d
    println("[$name] primal"); flush(stdout)
    out = zeros(Nx, Ny)
    wrap!(out, phi, ux_face, uy_face, is_solid, Nx, Ny, east_phi)
    v = reshape(_seed(Nx * Ny, 0.211, 0.7), Nx, Ny)
    u_phi = reshape(_seed(Nx * Ny, 0.137, 0.3), Nx, Ny)
    u_ux = reshape(_seed((Nx + 1) * Ny, 0.091, 1.1), Nx + 1, Ny)
    u_uy = reshape(_seed(Nx * (Ny + 1), 0.173, 0.2), Nx, Ny + 1)
    u_east = _seed(Ny, 0.31, 0.9)
    function reverse()
        println("[$name] reverse: compile+run"); flush(stdout)
        o = zeros(Nx, Ny); dout = copy(v)
        dphi = zeros(Nx, Ny); dux = zeros(Nx + 1, Ny); duy = zeros(Nx, Ny + 1)
        deast = zeros(Ny)
        Enzyme.autodiff(revmode, wrap!, Enzyme.Duplicated(o, dout),
                        Enzyme.Duplicated(copy(phi), dphi),
                        Enzyme.Duplicated(copy(ux_face), dux),
                        Enzyme.Duplicated(copy(uy_face), duy),
                        Enzyme.Const(is_solid), Enzyme.Const(Nx), Enzyme.Const(Ny),
                        Enzyme.Duplicated(copy(east_phi), deast))
        println("[$name] REVERSE_OK"); flush(stdout)
        return dot(dphi, u_phi) + dot(dux, u_ux) + dot(duy, u_uy) + dot(deast, u_east)
    end
    function forward()
        println("[$name] forward: compile+run"); flush(stdout)
        o = zeros(Nx, Ny); dout = zeros(Nx, Ny)
        Enzyme.autodiff(fwdmode, wrap!, Enzyme.Duplicated(o, dout),
                        Enzyme.Duplicated(copy(phi), copy(u_phi)),
                        Enzyme.Duplicated(copy(ux_face), copy(u_ux)),
                        Enzyme.Duplicated(copy(uy_face), copy(u_uy)),
                        Enzyme.Const(is_solid), Enzyme.Const(Nx), Enzyme.Const(Ny),
                        Enzyme.Duplicated(copy(east_phi), copy(u_east)))
        println("[$name] FORWARD_OK"); flush(stdout)
        return dot(v, dout)
    end
    if forward_first
        vJu = forward(); Jtvu = reverse()
    else
        Jtvu = reverse(); vJu = forward()
    end
    rel = abs(vJu - Jtvu) / max(abs(vJu), eps(Float64))
    println("[$name] transpose_rel = $rel ", rel < 1e-10 ? "IDENTITY_OK" : "IDENTITY_FAIL")
    flush(stdout)
    return rel
end

# ============================================================================
# Round 2: shrink s0 (rus branch + MUSCL branch in one loop) further.
# ============================================================================

# rusanov as a plain function, pass-1 body reused by the round-2 variants
@inline function rus_plain(phi, ux_face, uy_face, east_phi, Nx, Ny, i, j)
    ue = ux_face[i + 1, j]; uw = ux_face[i, j]; vn = uy_face[i, j + 1]; vs = uy_face[i, j]
    phie = ue >= 0 ? phi[i, j] : (i < Nx ? phi[i + 1, j] : east_phi[j])
    phiw = uw >= 0 ? (i > 1 ? phi[i - 1, j] : 0.0) : phi[i, j]
    phin = vn >= 0 ? phi[i, j] : (j < Ny ? phi[i, j + 1] : phi[i, j])
    phis = vs >= 0 ? (j > 1 ? phi[i, j - 1] : phi[i, j]) : phi[i, j]
    fl = (ue * phie - uw * phiw) + (vn * phin - vs * phis); du = (ue - uw) + (vn - vs)
    return -(fl - phi[i, j] * du)
end

@inline function in_band(is_solid, i, j, Nx, Ny)
    return i <= 2 || i >= Nx - 1 || j <= 2 || j >= Ny - 1 ||
           is_solid[i - 2, j] || is_solid[i - 1, j] || is_solid[i + 1, j] || is_solid[i + 2, j] ||
           is_solid[i, j - 2] || is_solid[i, j - 1] || is_solid[i, j + 1] || is_solid[i, j + 2]
end

# t1: pass 1 only (no band pass 2), rus + MUSCL branches, no closures
function advect_t1_nopass2(phi, ux_face, uy_face, is_solid, Nx, Ny, east_phi)
    adv = zeros(Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        if is_solid[i, j]
            adv[i, j] = 0.0; continue
        end
        if in_band(is_solid, i, j, Nx, Ny)
            adv[i, j] = phi[i, j] + rus_plain(phi, ux_face, uy_face, east_phi, Nx, Ny, i, j)
        else
            adv[i, j] = phi[i, j] + muscl_relax_rhs(phi, ux_face, uy_face, is_solid, i, j, Nx, Ny, 1.0, 1.0)
        end
    end
    return adv
end

# t2: band condition on indices only (no mask reads in the condition)
function advect_t2_indexband(phi, ux_face, uy_face, is_solid, Nx, Ny, east_phi)
    adv = zeros(Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        if is_solid[i, j]
            adv[i, j] = 0.0; continue
        end
        if i <= 2 || i >= Nx - 1 || j <= 2 || j >= Ny - 1
            adv[i, j] = phi[i, j] + rus_plain(phi, ux_face, uy_face, east_phi, Nx, Ny, i, j)
        else
            adv[i, j] = phi[i, j] + muscl_relax_rhs(phi, ux_face, uy_face, is_solid, i, j, Nx, Ny, 1.0, 1.0)
        end
    end
    return adv
end

# t3: band branch is a plain copy (no rusanov), MUSCL elsewhere
function advect_t3_norus(phi, ux_face, uy_face, is_solid, Nx, Ny, east_phi)
    adv = zeros(Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        if is_solid[i, j]
            adv[i, j] = 0.0; continue
        end
        if in_band(is_solid, i, j, Nx, Ny)
            adv[i, j] = phi[i, j]
        else
            adv[i, j] = phi[i, j] + muscl_relax_rhs(phi, ux_face, uy_face, is_solid, i, j, Nx, Ny, 1.0, 1.0)
        end
    end
    return adv
end

# t4: rusanov in the band, a trivial central stencil (no MUSCL) elsewhere
function advect_t4_nomuscl(phi, ux_face, uy_face, is_solid, Nx, Ny, east_phi)
    adv = zeros(Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        if is_solid[i, j]
            adv[i, j] = 0.0; continue
        end
        if in_band(is_solid, i, j, Nx, Ny)
            adv[i, j] = phi[i, j] + rus_plain(phi, ux_face, uy_face, east_phi, Nx, Ny, i, j)
        else
            adv[i, j] = phi[i, j] - 0.5 * ux_face[i, j] * (phi[i + 1, j] - phi[i - 1, j]) -
                        0.5 * uy_face[i, j] * (phi[i, j + 1] - phi[i, j - 1])
        end
    end
    return adv
end

# t5 (workaround candidate): pass 1 as two loops over disjoint cell sets
# (band cells with rusanov, the rest with MUSCL); pass 2 verbatim. Every
# cell is written once in pass 1 by the same formula as advect_v0, so the
# result is bit-identical.
function advect_split(phi, ux_face, uy_face, is_solid, Nx, Ny, east_phi)
    adv = zeros(Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        if is_solid[i, j]
            adv[i, j] = 0.0
        elseif in_band(is_solid, i, j, Nx, Ny)
            adv[i, j] = phi[i, j] + rus_plain(phi, ux_face, uy_face, east_phi, Nx, Ny, i, j)
        end
    end
    @inbounds for j in 1:Ny, i in 1:Nx
        if !is_solid[i, j] && !in_band(is_solid, i, j, Nx, Ny)
            adv[i, j] = phi[i, j] + muscl_relax_rhs(phi, ux_face, uy_face, is_solid, i, j, Nx, Ny, 1.0, 1.0)
        end
    end
    @inbounds for j in 1:Ny, i in 1:Nx
        if !is_solid[i, j] && is_cylinder_band(is_solid, i, j, Nx, Ny)
            adv[i, j] = phi[i, j] + muscl_relax_rhs(phi, ux_face, uy_face, is_solid, i, j, Nx, Ny, 1.0, 1.0)
        end
    end
    return adv
end

# t8: superbee's ifelse replaced by an if/else (a select vs a branch)
@inline function muscl_face_if(far_upwind, upwind, downwind)
    d_up = upwind - far_upwind
    d_down = downwind - upwind
    r = 0.0
    if d_down != 0.0
        r = d_up / d_down
    end
    return upwind + 0.5 * superbee_limiter(r) * d_down
end
@inline function muscl_relax_rhs_if(phi, ux_face, uy_face, is_solid, i, j, Nx, Ny, inv_dx, inv_dy)
    ue = ux_face[i + 1, j]; uw = ux_face[i, j]
    vn = uy_face[i, j + 1]; vs = uy_face[i, j]
    phie = if ue >= 0.0
        (i > 1 && !is_solid[i - 1, j] && !is_solid[i + 1, j]) ?
            muscl_face_if(phi[i - 1, j], phi[i, j], phi[i + 1, j]) : phi[i, j]
    else
        (i + 2 <= Nx && !is_solid[i + 2, j] && !is_solid[i + 1, j]) ?
            muscl_face_if(phi[i + 2, j], phi[i + 1, j], phi[i, j]) : phi[i + 1, j]
    end
    phiw = if uw >= 0.0
        (i - 2 >= 1 && !is_solid[i - 2, j] && !is_solid[i - 1, j]) ?
            muscl_face_if(phi[i - 2, j], phi[i - 1, j], phi[i, j]) : phi[i - 1, j]
    else
        (i + 1 <= Nx && !is_solid[i + 1, j] && !is_solid[i - 1, j]) ?
            muscl_face_if(phi[i + 1, j], phi[i, j], phi[i - 1, j]) : phi[i, j]
    end
    phin = if vn >= 0.0
        (j > 1 && !is_solid[i, j - 1] && !is_solid[i, j + 1]) ?
            muscl_face_if(phi[i, j - 1], phi[i, j], phi[i, j + 1]) : phi[i, j]
    else
        (j + 2 <= Ny && !is_solid[i, j + 2] && !is_solid[i, j + 1]) ?
            muscl_face_if(phi[i, j + 2], phi[i, j + 1], phi[i, j]) : phi[i, j + 1]
    end
    phis = if vs >= 0.0
        (j - 2 >= 1 && !is_solid[i, j - 2] && !is_solid[i, j - 1]) ?
            muscl_face_if(phi[i, j - 2], phi[i, j - 1], phi[i, j]) : phi[i, j - 1]
    else
        (j + 1 <= Ny && !is_solid[i, j + 1] && !is_solid[i, j - 1]) ?
            muscl_face_if(phi[i, j + 1], phi[i, j], phi[i, j - 1]) : phi[i, j]
    end
    flux_div = (ue * phie - uw * phiw) * inv_dx + (vn * phin - vs * phis) * inv_dy
    divu = (ue - uw) * inv_dx + (vn - vs) * inv_dy
    return -(flux_div - phi[i, j] * divu)
end
function advect_t8_if(phi, ux_face, uy_face, is_solid, Nx, Ny, east_phi)
    adv = zeros(Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        if is_solid[i, j]
            adv[i, j] = 0.0; continue
        end
        if in_band(is_solid, i, j, Nx, Ny)
            adv[i, j] = phi[i, j] + rus_plain(phi, ux_face, uy_face, east_phi, Nx, Ny, i, j)
        else
            adv[i, j] = phi[i, j] + muscl_relax_rhs_if(phi, ux_face, uy_face, is_solid, i, j, Nx, Ny, 1.0, 1.0)
        end
    end
    return adv
end
