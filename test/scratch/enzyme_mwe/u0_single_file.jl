# Self-contained reproduction for the upstream report (no Kraken dependency).
# Reverse-mode Enzyme over a 2-pass finite-volume advection step: cells in a
# band around solid cells / domain edges use a first-order upwind (rusanov)
# flux, the other cells a MUSCL/superbee limited flux. Reverse mode segfaults
# inside the generated thunk on x86_64 Linux (Julia 1.11.9 / 1.12.7, Enzyme
# 0.13.204 and 0.13.138); passes on aarch64 macOS with the same versions.
# Forward mode passes everywhere. Each branch alone (rusanov only, or MUSCL
# only) also passes on Linux.
using Enzyme, LinearAlgebra

@inline function superbee(r)
    return max(0.0, max(min(2.0 * r, 1.0), min(r, 2.0)))
end

@inline function muscl_face(far_upwind, upwind, downwind)
    d_up = upwind - far_upwind
    d_down = downwind - upwind
    r = ifelse(d_down == 0.0, 0.0, d_up / d_down)
    return upwind + 0.5 * superbee(r) * d_down
end

@inline function muscl_rhs(phi, ux_face, uy_face, is_solid, i, j, Nx, Ny)
    ue = ux_face[i + 1, j]; uw = ux_face[i, j]
    vn = uy_face[i, j + 1]; vs = uy_face[i, j]
    phie = if ue >= 0.0
        (i > 1 && !is_solid[i - 1, j] && !is_solid[i + 1, j]) ?
            muscl_face(phi[i - 1, j], phi[i, j], phi[i + 1, j]) : phi[i, j]
    else
        (i + 2 <= Nx && !is_solid[i + 2, j] && !is_solid[i + 1, j]) ?
            muscl_face(phi[i + 2, j], phi[i + 1, j], phi[i, j]) : phi[i + 1, j]
    end
    phiw = if uw >= 0.0
        (i - 2 >= 1 && !is_solid[i - 2, j] && !is_solid[i - 1, j]) ?
            muscl_face(phi[i - 2, j], phi[i - 1, j], phi[i, j]) : phi[i - 1, j]
    else
        (i + 1 <= Nx && !is_solid[i + 1, j] && !is_solid[i - 1, j]) ?
            muscl_face(phi[i + 1, j], phi[i, j], phi[i - 1, j]) : phi[i, j]
    end
    phin = if vn >= 0.0
        (j > 1 && !is_solid[i, j - 1] && !is_solid[i, j + 1]) ?
            muscl_face(phi[i, j - 1], phi[i, j], phi[i, j + 1]) : phi[i, j]
    else
        (j + 2 <= Ny && !is_solid[i, j + 2] && !is_solid[i, j + 1]) ?
            muscl_face(phi[i, j + 2], phi[i, j + 1], phi[i, j]) : phi[i, j + 1]
    end
    phis = if vs >= 0.0
        (j - 2 >= 1 && !is_solid[i, j - 2] && !is_solid[i, j - 1]) ?
            muscl_face(phi[i, j - 2], phi[i, j - 1], phi[i, j]) : phi[i, j - 1]
    else
        (j + 1 <= Ny && !is_solid[i, j + 1] && !is_solid[i, j - 1]) ?
            muscl_face(phi[i, j + 1], phi[i, j], phi[i, j - 1]) : phi[i, j]
    end
    flux_div = (ue * phie - uw * phiw) + (vn * phin - vs * phis)
    divu = (ue - uw) + (vn - vs)
    return -(flux_div - phi[i, j] * divu)
end

@inline function rusanov_rhs(phi, ux_face, uy_face, east_phi, Nx, Ny, i, j)
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

# The crashing operator: one loop whose body takes either the rusanov branch
# or the MUSCL branch. Removing either branch makes reverse mode pass.
function advect!(out, phi, ux_face, uy_face, is_solid, Nx, Ny, east_phi)
    @inbounds for j in 1:Ny, i in 1:Nx
        if is_solid[i, j]
            out[i, j] = 0.0
        elseif in_band(is_solid, i, j, Nx, Ny)
            out[i, j] = phi[i, j] + rusanov_rhs(phi, ux_face, uy_face, east_phi, Nx, Ny, i, j)
        else
            out[i, j] = phi[i, j] + muscl_rhs(phi, ux_face, uy_face, is_solid, i, j, Nx, Ny)
        end
    end
    return nothing
end

# ---- data: 24x24 grid, cut cylinder, smooth fields ---------------------------
Nx = 24; Ny = 24; cx = 12.35; cy = 11.65; R = 5.13
is_solid = falses(Nx, Ny)
for j in 1:Ny, i in 1:Nx
    is_solid[i, j] = ((i - 1) - (cx - 0.5))^2 + ((j - 1) - (cy - 0.5))^2 <= R^2
end
phi = zeros(Nx, Ny)
for j in 1:Ny, i in 1:Nx
    is_solid[i, j] || (phi[i, j] = 0.05 * sin(3pi * (i - 1) / Nx) * cos(2pi * (j - 1) / Ny))
end
ux_face = zeros(Nx + 1, Ny); uy_face = zeros(Nx, Ny + 1)
for J in 1:Ny, I in 1:(Nx + 1)
    il = max(I - 1, 1); ir = min(I, Nx)
    (is_solid[il, J] || is_solid[ir, J]) && continue
    ux_face[I, J] = 2e-4 * (0.5 + sin(2pi * (J - 1) / Ny) + 0.3 * cos(2pi * (I - 1) / Nx))
end
for J in 2:Ny, I in 1:Nx
    (is_solid[I, J - 1] || is_solid[I, J]) && continue
    uy_face[I, J] = 1e-4 * sin(2pi * (I - 1) / Nx) * cos(2pi * (J - 1) / Ny)
end
east_phi = phi[Nx, :]

out = zeros(Nx, Ny)
advect!(out, phi, ux_face, uy_face, is_solid, Nx, Ny, east_phi)
println("primal ok")

dout = [cos(0.211 * k + 0.7) for k in 1:(Nx * Ny)]
dout = reshape(dout, Nx, Ny)
dphi = zeros(Nx, Ny); dux = zeros(Nx + 1, Ny); duy = zeros(Nx, Ny + 1); deast = zeros(Ny)
println("reverse ..."); flush(stdout)
Enzyme.autodiff(Enzyme.set_runtime_activity(Enzyme.Reverse), advect!,
                Enzyme.Duplicated(zeros(Nx, Ny), dout),
                Enzyme.Duplicated(copy(phi), dphi),
                Enzyme.Duplicated(copy(ux_face), dux),
                Enzyme.Duplicated(copy(uy_face), duy),
                Enzyme.Const(is_solid), Enzyme.Const(Nx), Enzyme.Const(Ny),
                Enzyme.Duplicated(copy(east_phi), deast))
println("REVERSE_OK  norm(dphi) = ", norm(dphi))
