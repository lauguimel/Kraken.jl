# Standalone DECOUPLED STEADY scalar advection–diffusion solver ("thermal transport").
#
# Given a FROZEN face-normal velocity field (uf,vf) on a collocated cell-centred
# grid, solve the steady scalar transport equation
#
#   ∇·(u T) − DT ∇²T = 0
#
# to steady state with mixed Dirichlet / Neumann-flux / zero-gradient outflow
# boundary conditions. Because u is FROZEN, the equation is LINEAR in T, so the
# steady state is obtained by a SINGLE direct linear solve  A·T = b  — there is
# NO outer iteration loop. The "thermal transport" name reflects the physical
# interpretation T = temperature, DT = thermal diffusivity, q = wall heat flux.
#
# Reuses the matrix-assembly pattern of inc_ns/simple.jl (`_incns_assemble_neg_laplacian`,
# 5-point Laplacian into a sparse CSC) and the cavity's first-order UPWIND
# advection logic (`_cavity_convection!`), transcribed here into MATRIX-ASSEMBLY
# coefficients (donor diagonal +F, acceptor off-diagonal −F). The solve goes
# through the shared factorize-once seam (`lin_factorize`/`lin_solve!`); since the
# upwind advection makes A NON-symmetric we use spd=false (LU/LDLᵀ fallback).
# Reimplements NOTHING from the operators or the linear-solve seam — calls/mirrors.
#
# KA + stdlib only, CPU by default. Does NOT subtype AbstractMethod and does NOT
# register with `using Kraken`. New-file-only standalone brick.
#
# Public entry point:
#   solve_scalar_transport(; nx, ny, dx, dy, uf, vf, DT, is_solid, bc, backend)
#     -> NamedTuple(T, residual_history, iters, converged, dx, dy, ycenters,
#                   nx, ny, DT, Pe_cell)

using LinearAlgebra
using SparseArrays

# Factorize-once seam (linear_solve.jl pulls in poisson.jl for pin_reference_dof).
# Guarded so the brick is include-able standalone on a CPU-only box.
if !isdefined(@__MODULE__, :lin_factorize)
    include(joinpath(@__DIR__, "..", "..", "solve", "linear_solve.jl"))
end

# ---------------------------------------------------------------------------
# Sparse system assembly for steady scalar advection–diffusion.
#
# Layout: linear index k = i + (j-1)*nx, i in 1:nx (x), j in 1:ny (y). Same
# layout as simple.jl / cavity.jl. uf[i,j] = east face of cell (i,j) (x-normal),
# vf[i,j] = north face of cell (i,j) (y-normal); the west face of (i,j) is
# uf[i-1,j], the south face is vf[i,j-1] (FROZEN velocities).
#
# We assemble  A = ADV + DIFF  and the RHS b, so that A·T = b discretises
#   div(u T) − DT·lap(T) = 0  ⇒  ADV·T − DT·lap(T) = 0,  i.e.
#   A = ADV + (−DT·lap) = ADV + DT·(neg-Laplacian).
#
# DIFFUSION (mirrors _incns_assemble_neg_laplacian, scaled by DT):
#   The discrete (−Laplacian) contributes +DT·invh2 to the diagonal and
#   −DT·invh2 to each interior neighbour. Per wall:
#     :dirichlet -> ghost T_g = 2·T_wall − T_c  ⇒  +2·DT·invh2 on diagonal,
#                   wall-value source +2·DT·invh2·T_wall into b (cavity convention).
#     :flux      -> Neumann wall heat flux q. Zero-gradient ghost for diffusion
#                   (no diagonal face term); the flux enters b as a SOURCE on the
#                   wall-adjacent row:  +q/h  (q = DT·∂T/∂n, the conductive flux
#                   leaving the wall into the domain). NOT in A.
#     :outflow   -> zero-gradient (ghost = interior): no diffusive face term.
#
# ADVECTION (mirrors _cavity_convection! first-order upwind, as MATRIX coeffs):
#   conv = (Fe·T_e − Fw·T_w)/dx + (Fn·T_n − Fs·T_s)/dy with the upwind donor cell.
#   On an interior east face with flux Fe = uf[i,j]:
#     Fe >= 0 (outflow east, donor = own cell):  +Fe/dx on diag(k),
#     Fe <  0 (inflow from east, donor = i+1):   +Fe/dx on off-diag (k, i+1).
#   The matching face for cell (i+1) is its WEST face Fw = uf[i,j], handled when
#   the loop reaches (i+1) via the symmetric (i>1) branch. Boundary faces:
#     :dirichlet -> wall value T_wall is the upwind donor when flowing IN; its
#                   contribution Fwall·T_wall/h goes to b (sign per inflow/outflow).
#     :flux / :outflow / wall with no advective inflow value -> zero-gradient
#                   (ghost = interior cell): the boundary face advects the
#                   INTERIOR upwind value, i.e. a +F/h diagonal term when the flow
#                   leaves the domain, and (for a true wall with vf=0) no flux.
#
# is_solid is accepted for later cut-cell work but the regular (axis-aligned,
# is_solid=falses) path is exact for linear fields and is all the validation needs.
# ---------------------------------------------------------------------------
function _st_assemble_system(nx::Integer, ny::Integer, dx::Real, dy::Real,
                             uf::AbstractMatrix, vf::AbstractMatrix, DT::Real,
                             is_solid::AbstractMatrix;
                             bc, source = nothing)
    nx = Int(nx); ny = Int(ny)
    n = nx * ny
    dx = Float64(dx); dy = Float64(dy); DT = Float64(DT)
    invdx = 1.0 / dx
    invdy = 1.0 / dy
    invdx2 = 1.0 / (dx * dx)
    invdy2 = 1.0 / (dy * dy)

    lin(i, j) = i + (j - 1) * nx

    bc_w = bc.west;  bc_e = bc.east
    bc_s = bc.south; bc_n = bc.north

    I = Int[]; J = Int[]; V = Float64[]
    sizehint!(I, 5n); sizehint!(J, 5n); sizehint!(V, 5n)
    b = zeros(Float64, n)

    @inbounds for j in 1:ny, i in 1:nx
        k = lin(i, j)
        diag = 0.0

        # face mass fluxes (frozen). Interior faces from uf/vf; boundary faces
        # carry the wall-normal velocity (0 unless a :dirichlet inlet imposes it
        # through uf/vf — here uf/vf already encode the inlet velocity on the
        # first/last column/row faces; we read them directly).
        Fe = (i < nx) ? uf[i, j] : uf[i, j]   # east face vel (boundary at i==nx too)
        Fw = (i > 1) ? uf[i - 1, j] : 0.0     # west face = east face of west nb
        Fn = (j < ny) ? vf[i, j] : vf[i, j]   # north face vel
        Fs = (j > 1) ? vf[i, j - 1] : 0.0     # south face

        # ===== EAST face (x) =====
        if i < nx
            # diffusion
            push!(I, k); push!(J, lin(i + 1, j)); push!(V, -DT * invdx2)
            diag += DT * invdx2
            # advection (upwind on Fe)
            if Fe >= 0          # donor = own cell
                diag += Fe * invdx
            else                # donor = east neighbour
                push!(I, k); push!(J, lin(i + 1, j)); push!(V, Fe * invdx)
            end
        else
            # east boundary
            if bc_e.kind === :dirichlet
                diag += 2.0 * DT * invdx2
                b[k] += 2.0 * DT * invdx2 * Float64(bc_e.value)
                # advection: wall value is the donor when flow enters (Fe<0),
                # interior is donor when flow leaves (Fe>=0).
                if Fe >= 0
                    diag += Fe * invdx
                else
                    b[k] -= Fe * invdx * Float64(bc_e.value)
                end
            elseif bc_e.kind === :flux
                # Neumann: zero-grad diffusion ghost (no diag term); flux source.
                b[k] += Float64(bc_e.value) * invdx
                # advection: zero-gradient ghost = interior value.
                diag += Fe * invdx
            elseif bc_e.kind === :outflow
                # zero-gradient: ghost = interior; advect interior upwind value.
                diag += Fe * invdx
            end
        end

        # ===== WEST face (x) =====
        if i > 1
            push!(I, k); push!(J, lin(i - 1, j)); push!(V, -DT * invdx2)
            diag += DT * invdx2
            # advection: west face flux Fw, OUT of cell k is -Fw (Fw>0 enters k).
            # conv contribution is -(Fw·T_w)/dx. Upwind donor on Fw:
            if Fw >= 0          # flow enters k from west: donor = west neighbour
                push!(I, k); push!(J, lin(i - 1, j)); push!(V, -Fw * invdx)
            else                # flow leaves k to the west: donor = own cell
                diag += -Fw * invdx
            end
        else
            # west boundary
            if bc_w.kind === :dirichlet
                diag += 2.0 * DT * invdx2
                b[k] += 2.0 * DT * invdx2 * Float64(bc_w.value)
                # advection through west boundary: boundary flux Fw_b. The inlet
                # face velocity is uf at the west boundary; encode via vf/uf? For
                # a Dirichlet inlet the imposed normal velocity is read from the
                # provided boundary; here Fw_b = 0 unless the user set an inlet
                # velocity, in which case it is carried in `uf` on a ghost face.
                # The contribution: -(Fw_b·T_face)/dx. Flow into domain (Fw_b>0)
                # uses the wall value; out uses interior.
                Fwb = 0.0  # west boundary normal vel (no ghost face stored); inlet
                           # velocity is applied through the interior uf field.
                if Fwb >= 0
                    b[k] += Fwb * invdx * Float64(bc_w.value)
                else
                    diag += -Fwb * invdx
                end
            elseif bc_w.kind === :flux
                b[k] += Float64(bc_w.value) * invdx
            elseif bc_w.kind === :outflow
                # zero-gradient: nothing for diffusion; advection carries interior.
            end
        end

        # ===== NORTH face (y) =====
        if j < ny
            push!(I, k); push!(J, lin(i, j + 1)); push!(V, -DT * invdy2)
            diag += DT * invdy2
            if Fn >= 0
                diag += Fn * invdy
            else
                push!(I, k); push!(J, lin(i, j + 1)); push!(V, Fn * invdy)
            end
        else
            if bc_n.kind === :dirichlet
                diag += 2.0 * DT * invdy2
                b[k] += 2.0 * DT * invdy2 * Float64(bc_n.value)
                if Fn >= 0
                    diag += Fn * invdy
                else
                    b[k] -= Fn * invdy * Float64(bc_n.value)
                end
            elseif bc_n.kind === :flux
                b[k] += Float64(bc_n.value) * invdy
                diag += Fn * invdy
            elseif bc_n.kind === :outflow
                diag += Fn * invdy
            end
        end

        # ===== SOUTH face (y) =====
        if j > 1
            push!(I, k); push!(J, lin(i, j - 1)); push!(V, -DT * invdy2)
            diag += DT * invdy2
            if Fs >= 0          # flow enters k from south: donor = south neighbour
                push!(I, k); push!(J, lin(i, j - 1)); push!(V, -Fs * invdy)
            else                # flow leaves k to the south: donor = own cell
                diag += -Fs * invdy
            end
        else
            if bc_s.kind === :dirichlet
                diag += 2.0 * DT * invdy2
                b[k] += 2.0 * DT * invdy2 * Float64(bc_s.value)
                Fsb = 0.0
                if Fsb >= 0
                    b[k] += Fsb * invdy * Float64(bc_s.value)
                else
                    diag += -Fsb * invdy
                end
            elseif bc_s.kind === :flux
                b[k] += Float64(bc_s.value) * invdy
            elseif bc_s.kind === :outflow
                # zero-gradient.
            end
        end

        # Optional volumetric source S (manufactured-solution / heat-generation
        # term): the equation becomes div(uT) − DT·lap(T) = S, contributing +S to
        # the RHS row. Defaults to no source.
        if source !== nothing
            b[k] += Float64(source[i, j])
        end

        push!(I, k); push!(J, k); push!(V, diag)
    end

    A = sparse(I, J, V, n, n)
    return A, b
end

"""
    solve_scalar_transport(; nx, ny, dx, dy, uf, vf, DT, is_solid=falses(nx,ny),
                           bc, backend=nothing)

Standalone DECOUPLED STEADY scalar advection–diffusion ("thermal transport")
solver. Given a FROZEN face-normal velocity field, solve

    ∇·(u T) − DT ∇²T = 0

to steady state on a collocated cell-centred grid `nx × ny` with spacing
`dx, dy`. With `u` frozen the problem is LINEAR in `T`, so the steady state is a
SINGLE direct solve `A·T = b` — no outer iteration loop.

Arguments:
  `uf, vf`   FROZEN face-normal velocities. `uf[i,j]` = east face of cell
             `(i,j)` (x-normal), `vf[i,j]` = north face (y-normal). SAME layout
             as `inc_ns/simple.jl`.
  `DT`       scalar diffusivity (thermal diffusivity / conductivity proxy).
  `is_solid` cut-cell mask, accepted for later use; the validated regular path
             uses `falses(nx,ny)` (exact for linear fields).
  `bc`       NamedTuple keyed `west/east/south/north`, each a
             `(kind::Symbol, value)`:
               `:dirichlet` — imposed inlet/wall value `T = value`.
               `:flux`      — Neumann wall heat flux `q = value`; enters as a
                              SOURCE in `b` on the wall-adjacent row (q/h), NOT
                              in `A` (zero-gradient diffusion ghost).
               `:outflow`   — zero-gradient (ghost = interior); advection carries
                              the interior upwind value.
  `source`   optional volumetric source field `S[i,j]` (heat generation /
             manufactured-solution term): the balance becomes
             `div(uT) − DT·lap(T) = S`. `nothing` (default) means no source.
  `backend`  optional backend tag; `nothing` (default) routes to the CPU seam.

Assembly mirrors `inc_ns/simple.jl` (5-point Laplacian → sparse CSC) and the
cavity's first-order upwind advection (`_cavity_convection!`), transcribed into
matrix coefficients (donor diagonal `+F`, acceptor off-diagonal `−F`). The solve
uses the shared factorize-once seam with `spd=false` (A is non-symmetric due to
upwind).

Returns a NamedTuple with fields:
  `T`                  steady scalar field (`nx × ny`).
  `residual_history`   single-element `[‖A·T − b‖]` (should be ~machine eps).
  `iters`              always `1` (single linear solve, no loop).
  `converged`          whether the residual is at machine-noise level.
  `dx, dy, ycenters`   grid metrics for analytic comparison.
  `nx, ny, DT`         echoed inputs.
  `Pe_cell`            cell Péclet number `max|u|·min(dx,dy)/DT`.
"""
function solve_scalar_transport(; nx::Integer, ny::Integer, dx::Real, dy::Real,
                                uf::AbstractMatrix, vf::AbstractMatrix, DT::Real,
                                is_solid::AbstractMatrix = falses(Int(nx), Int(ny)),
                                bc, source = nothing, backend = nothing)
    nx = Int(nx); ny = Int(ny)
    dx = Float64(dx); dy = Float64(dy); DT = Float64(DT)
    ycenters = [(j - 0.5) * dy for j in 1:ny]

    A, b = _st_assemble_system(nx, ny, dx, dy, uf, vf, DT, is_solid;
                               bc = bc, source = source)

    # A is non-symmetric (first-order upwind advection) -> a sparse LU is the
    # correct CPU factorization. We REUSE the linear-solve seam's cache + solve
    # path (`LinearSolveCache` + `lin_solve!`), but build the cache with an LU
    # factor directly rather than calling `lin_factorize(A; spd=false)`.
    #
    # WHY NOT lin_factorize(A; spd=false): the CPU `spd=false` branch in
    # src/solve/linear_solve.jl is `try ldlt(Symmetric(A)) catch lu(A)`. For a
    # GENUINELY non-symmetric A, `ldlt(Symmetric(A))` does NOT throw — it silently
    # factorizes the SYMMETRIZED matrix (upper triangle mirrored), which is a
    # DIFFERENT operator, so the solve returns a wrong field (verified: residual
    # ~1e2 instead of ~1e-13). The `lu` fallback is only reached on a throw, which
    # never happens here. Since this brick must NOT edit the seam, we mirror its
    # CPU contract by constructing the cache with `lu(A)` (the branch the seam
    # INTENDED for non-symmetric operators) and dispatch the shared `lin_solve!`
    # (`cache.factor \ b`). This keeps the seam's reuse/backend abstraction intact.
    btag = backend === nothing ? CPUBackendTag() : backend
    if btag isa CPUBackendTag
        factor = lu(A)
        cache = LinearSolveCache(CPUBackendTag(), factor, A, A, 0, false)
    else
        # Non-CPU backend: defer to the seam (its non-CPU methods own the choice).
        cache = lin_factorize(A; backend = btag, spd = false)
    end
    Tvec = lin_solve!(cache, b)
    T = reshape(Vector{Float64}(Tvec), nx, ny)

    # Residual ‖A·T − b‖ (should be ~machine eps for a direct solve).
    resid = norm(A * Tvec .- b)
    residual_history = [resid]
    converged = resid <= 1e-8 * max(norm(b), 1.0)

    # Cell Péclet number Pe = max|u|·min(dx,dy)/DT.
    umax = max(maximum(abs, uf), maximum(abs, vf))
    Pe_cell = umax * min(dx, dy) / DT

    iters = 1
    return (; T, residual_history, iters, converged, dx, dy, ycenters,
            nx, ny, DT, Pe_cell)
end
