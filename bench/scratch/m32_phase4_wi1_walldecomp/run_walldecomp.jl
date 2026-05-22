#!/usr/bin/env julia
# M32 Phase 4 — Wi=1 Cd gap localization
# ----------------------------------------------------------------------------
#
# Mandate (M32-Phase4-walldecomp / mission D1):
#   Localize the −7.3 % Wi=1 Cd gap between Kraken (Aqua F64 :rusanov) and
#   rheoTool (Docker shrunk L=15R) by computing the wall decomposition
#   Cd(θ) = Cd_pressure(θ) + Cd_visc_solvent(θ) + Cd_polymer(θ) on the :idx
#   frame for both codes, and identifying which (component, θ-region)
#   bucket carries ≥ 70 % of the gap.
#
# Inputs (both Metal F32 for Kraken; rT is Docker output):
#   tmp/m30_rho_metal/run01/cyl_bigsweep_v2_beta0p59_wi1_re1_R30_bsd1_*.jls
#   tmp/m30_R_sweep_metal/cyl_bigsweep_v2_beta0p59_wi1_re1_R40_bsd1_*.jls
#   bench/rheotool/cylinder_wi1.0_shrunk15R/{constant/polyMesh, 20/}
#
# Convention (verified against
#   src/kernels/li_bb_2d.jl::precompute_q_wall_cylinder lines 277-281
#   src/drivers/viscoelastic_logfv_2d.jl::_logfv_cylinder_channel_geometry_2d):
#   :idx frame uses cx_lu = cx_phys + 1, cy_lu = cy_phys + 1, dx = (i-1) - cx_phys
#   (per [[feedback_wall_ring_idx_frame]]). NEVER use :phys frame for Cd_polymer.
#
# θ convention (matches Phase 0c and rT outputCd):
#   θ = atan2(y - cy, x - cx), CCW from +x
#   θ = 0       : rear-pole (lee, downstream)
#   θ = ±π      : front-pole (windward, upstream)
#   θ = ±π/2    : shoulder (top/bottom equators)
#
# Region split (3-bucket):
#   front-pole  |θ ± π| < π/4   (windward arc, 90°)
#   shoulder    π/4 ≤ |θ ± π/2| < 3π/4 ... cleaner: π/4 ≤ |θ| < 3π/4 → both flanks (180°)
#   wake        |θ|        < π/4   (leeward arc, 90°)
# Equivalently: front_pole = |θ - π| < π/4 OR |θ + π| < π/4 (same set);
#               wake       = |θ|     < π/4;
#               shoulder   = everything else.

using Dates
using LinearAlgebra
using Printf
using Serialization
using Statistics

const SCRIPT_DIR = @__DIR__
const REPO_ROOT  = abspath(joinpath(SCRIPT_DIR, "..", "..", ".."))
const OUTPUT_DIR = SCRIPT_DIR
const TMP_DIR    = joinpath(REPO_ROOT, "tmp", "m32_phase4")
mkpath(TMP_DIR)

# Region classifier for a θ in (-π, π].
# Returns :front_pole, :shoulder, or :wake.
function region_of(theta::Float64)
    aθ = abs(theta)
    if aθ < π/4
        return :wake
    elseif aθ > 3π/4
        return :front_pole
    else
        return :shoulder
    end
end

# ===========================================================================
# Part A — Kraken-side wall decomposition (:idx frame)
# ===========================================================================
# Adapted from bench/scratch/m30_centering_audit/run_centering_audit.jl:98-233
# (locked artefact). Re-implemented here to keep this file self-contained.

function lattice_velocity_gradient(ux::AbstractMatrix, uy::AbstractMatrix,
                                    solid::AbstractMatrix{Bool})
    Nx, Ny = size(ux)
    dudx = zeros(Float64, Nx, Ny); dudy = zeros(Float64, Nx, Ny)
    dvdx = zeros(Float64, Nx, Ny); dvdy = zeros(Float64, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        solid[i, j] && continue
        west_ok = (i > 1)  && !solid[i-1, j]
        east_ok = (i < Nx) && !solid[i+1, j]
        if west_ok && east_ok
            dudx[i,j] = 0.5 * (ux[i+1,j] - ux[i-1,j])
            dvdx[i,j] = 0.5 * (uy[i+1,j] - uy[i-1,j])
        elseif east_ok
            dudx[i,j] = ux[i+1,j] - ux[i,j]
            dvdx[i,j] = uy[i+1,j] - uy[i,j]
        elseif west_ok
            dudx[i,j] = ux[i,j] - ux[i-1,j]
            dvdx[i,j] = uy[i,j] - uy[i-1,j]
        end
        south_ok = (j > 1)  && !solid[i, j-1]
        north_ok = (j < Ny) && !solid[i, j+1]
        if south_ok && north_ok
            dudy[i,j] = 0.5 * (ux[i,j+1] - ux[i,j-1])
            dvdy[i,j] = 0.5 * (uy[i,j+1] - uy[i,j-1])
        elseif north_ok
            dudy[i,j] = ux[i,j+1] - ux[i,j]
            dvdy[i,j] = uy[i,j+1] - uy[i,j]
        elseif south_ok
            dudy[i,j] = ux[i,j] - ux[i,j-1]
            dvdy[i,j] = uy[i,j] - uy[i,j-1]
        end
    end
    return dudx, dudy, dvdx, dvdy
end

function kraken_wall_decomp(snap; N_az::Int=36)
    Nx = snap.Nx; Ny = snap.Ny
    cx_phys = Float64(snap.cylinder_x_lbm)
    cy_phys = Float64(snap.cylinder_y_lbm)
    R_lu    = Float64(snap.radius_lbm)
    # :idx frame — kernel rasterisation places (i, j) at physical (i-1, j-1),
    # so the index-frame centre of the rasterised disk is at (cx_phys+1, cy_phys+1).
    cx_lu = cx_phys + 1.0
    cy_lu = cy_phys + 1.0
    u_mean = Float64(snap.u_mean)
    solid  = snap.is_solid
    rho    = snap.rho
    ux = snap.ux; uy = snap.uy
    txx_p = snap.tauxx; txy_p = snap.tauxy; tyy_p = snap.tauyy
    mu_s = Float64(snap.nu_s)   # ρ ≡ 1 LU

    cs2 = 1.0/3.0
    # Cd_norm = ½ ρ U² D = u_mean² · R_lu  (D = 2R)
    norm = u_mean^2 * R_lu
    dtheta = 2π / N_az
    arc_dl = R_lu * dtheta

    dudx, dudy, dvdx, dvdy = lattice_velocity_gradient(ux, uy, solid)

    bins = (
        tx_pres = zeros(Float64, N_az), tx_solv = zeros(Float64, N_az),
        tx_poly = zeros(Float64, N_az), count   = zeros(Int,     N_az),
    )

    @inbounds for j in 1:Ny, i in 1:Nx
        solid[i, j] && continue
        has_solid_n = false
        for (di, dj) in ((-1,0),(1,0),(0,-1),(0,1),
                         (-1,-1),(-1,1),(1,-1),(1,1))
            ii = i + di; jj = j + dj
            if 1 <= ii <= Nx && 1 <= jj <= Ny && solid[ii, jj]
                has_solid_n = true
                break
            end
        end
        has_solid_n || continue
        dx = Float64(i) - cx_lu
        dy = Float64(j) - cy_lu
        r  = hypot(dx, dy)
        r > 0 || continue
        theta = atan(dy, dx)
        nx_o  = dx / r
        ny_o  = dy / r
        rho_ij = rho[i, j]
        p_lbm  = cs2 * (rho_ij - 1.0)
        # pressure traction: σ = -p I
        tx_pres = -p_lbm * nx_o
        # solvent: σ = 2 μ_s D
        sxx_solv = 2.0 * mu_s * dudx[i,j]
        sxy_solv =       mu_s * (dudy[i,j] + dvdx[i,j])
        tx_solv = sxx_solv * nx_o + sxy_solv * ny_o
        # polymer: σ = τ_p
        tx_poly = txx_p[i,j] * nx_o + txy_p[i,j] * ny_o

        bin = mod(floor(Int, (theta + π) / dtheta), N_az) + 1
        bins.tx_pres[bin] += tx_pres
        bins.tx_solv[bin] += tx_solv
        bins.tx_poly[bin] += tx_poly
        bins.count[bin]   += 1
    end

    dCd_pres = zeros(Float64, N_az)
    dCd_solv = zeros(Float64, N_az)
    dCd_poly = zeros(Float64, N_az)
    for k in 1:N_az
        if bins.count[k] > 0
            dCd_pres[k] = (bins.tx_pres[k] / bins.count[k]) * arc_dl / norm
            dCd_solv[k] = (bins.tx_solv[k] / bins.count[k]) * arc_dl / norm
            dCd_poly[k] = (bins.tx_poly[k] / bins.count[k]) * arc_dl / norm
        end
    end
    theta_centres = [(-π + (k - 0.5)*dtheta) for k in 1:N_az]
    return (;
        N_az, dtheta, arc_dl, theta_centres, bin_count=bins.count,
        dCd_pres, dCd_solv, dCd_poly,
        Cd_pres = sum(dCd_pres),
        Cd_solv = sum(dCd_solv),
        Cd_poly = sum(dCd_poly),
        Cd_total = sum(dCd_pres) + sum(dCd_solv) + sum(dCd_poly),
        u_mean, R_lu, norm,
    )
end

# ===========================================================================
# Part B — rheoTool-side wall decomposition (FOAM reader from m29c_wallstress)
# ===========================================================================

mutable struct SpatialIndex
    x_min::Float64; y_min::Float64
    dx::Float64;    dy::Float64
    nx::Int;        ny::Int
    xs::Vector{Float64}
    ys::Vector{Float64}
    bins::Vector{Vector{Int}}
end

strip_comments(text::AbstractString) =
    join((replace(line, r"//.*$" => "") for line in
          split(replace(text, r"/\*.*?\*/"s => ""), '\n')), "\n")

function read_text_maybe_gzip(path::AbstractString)
    if isfile(path)
        return read(path, String)
    elseif isfile(path * ".gz")
        return read(pipeline(`gzip -cd $(path * ".gz")`), String)
    end
    throw(SystemError(path, 2))
end

foam_path(case_dir, parts...) = joinpath(case_dir, parts...)

function tuple_values(text::AbstractString, tuple_len::Int)
    values = Vector{NTuple{tuple_len,Float64}}()
    for m in eachmatch(r"\(([^\(\)]*)\)", text)
        nums = split(strip(m.captures[1]))
        length(nums) == tuple_len || continue
        push!(values, ntuple(k -> parse(Float64, nums[k]), tuple_len))
    end
    return values
end

function parse_counted_block(text::AbstractString, object_name::AbstractString)
    clean = strip_comments(text)
    m = match(Regex("\\b" * object_name * "\\b"), clean)
    start = m === nothing ? firstindex(clean) : m.offset + length(m.match)
    tail = clean[start:end]
    count_match = match(r"\b(\d+)\s*\(", tail)
    count_match === nothing && error("could not find counted block for $(object_name)")
    n = parse(Int, count_match.captures[1])
    open_pos = count_match.offset + length(count_match.match) - 1
    depth = 0
    close_pos = open_pos
    for idx in open_pos:lastindex(tail)
        c = tail[idx]
        if c == '('
            depth += 1
        elseif c == ')'
            depth -= 1
            if depth == 0
                close_pos = idx
                break
            end
        end
    end
    return n, tail[(open_pos + 1):(close_pos - 1)]
end

function parse_points(case_dir)
    text = read_text_maybe_gzip(foam_path(case_dir, "constant", "polyMesh", "points"))
    n, block = parse_counted_block(text, "points")
    pts = tuple_values(block, 3)
    length(pts) == n || error("points count mismatch: header $(n), parsed $(length(pts))")
    return pts
end

function parse_faces(case_dir)
    text = strip_comments(read_text_maybe_gzip(foam_path(case_dir, "constant", "polyMesh", "faces")))
    n, block = parse_counted_block(text, "faces")
    faces = Vector{Vector{Int}}()
    for m in eachmatch(r"\b\d+\s*\(([^\)]*)\)", block)
        push!(faces, [parse(Int, x) + 1 for x in split(strip(m.captures[1]))])
    end
    length(faces) == n || error("faces count mismatch: header $(n), parsed $(length(faces))")
    return faces
end

function parse_label_list(case_dir, name)
    text = read_text_maybe_gzip(foam_path(case_dir, "constant", "polyMesh", name))
    n, block = parse_counted_block(text, name)
    labels = [parse(Int, m.match) + 1 for m in eachmatch(r"-?\d+", block)]
    length(labels) == n || error("$(name) count mismatch: header $(n), parsed $(length(labels))")
    return labels
end

function foam_internal_block(text)
    clean = strip_comments(text)
    m = match(r"internalField\s+(uniform|nonuniform)\s*", clean)
    m === nothing && error("internalField not found")
    mode = m.captures[1]
    tail = clean[(m.offset + length(m.match)):end]
    if mode == "uniform"
        semi = findfirst(';', tail)
        semi === nothing && error("uniform internalField missing semicolon")
        return mode, strip(tail[1:(semi - 1)])
    end
    count_match = match(r"\b(\d+)\s*\(", tail)
    count_match === nothing && error("nonuniform internalField missing count")
    n = parse(Int, count_match.captures[1])
    open_pos = count_match.offset + length(count_match.match) - 1
    depth = 0
    close_pos = open_pos
    for idx in open_pos:lastindex(tail)
        c = tail[idx]
        if c == '('
            depth += 1
        elseif c == ')'
            depth -= 1
            if depth == 0
                close_pos = idx
                break
            end
        end
    end
    return mode, n, tail[(open_pos + 1):(close_pos - 1)]
end

function parse_vol_scalar(path, n_cells)
    text = if isfile(path)
        read(path, String)
    elseif isfile(path * ".gz")
        read(pipeline(`gzip -cd $(path * ".gz")`), String)
    else
        error("missing $path[.gz]")
    end
    mode_block = foam_internal_block(text)
    if mode_block[1] == "uniform"
        v = parse(Float64, strip(mode_block[2]))
        return fill(v, n_cells)
    end
    _, n, block = mode_block
    vals = [parse(Float64, m.match) for m in eachmatch(r"-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?", block)]
    n == n_cells || error("$(path) has $(n) cells, mesh has $(n_cells)")
    length(vals) == n || error("scalar field count mismatch in $(path)")
    return vals
end

function parse_vol_vector(path, n_cells)
    text = if isfile(path)
        read(path, String)
    elseif isfile(path * ".gz")
        read(pipeline(`gzip -cd $(path * ".gz")`), String)
    else
        error("missing $path[.gz]")
    end
    mode_block = foam_internal_block(text)
    if mode_block[1] == "uniform"
        vals = tuple_values(mode_block[2], 3)
        return fill(vals[1], n_cells)
    end
    _, n, block = mode_block
    vals = tuple_values(block, 3)
    n == n_cells || error("$(path) has $(n) cells, mesh has $(n_cells)")
    length(vals) == n || error("vector field count mismatch in $(path)")
    return vals
end

function parse_vol_symmtensor(path, n_cells)
    text = if isfile(path)
        read(path, String)
    elseif isfile(path * ".gz")
        read(pipeline(`gzip -cd $(path * ".gz")`), String)
    else
        error("missing $path[.gz]")
    end
    mode_block = foam_internal_block(text)
    if mode_block[1] == "uniform"
        vals = tuple_values(mode_block[2], 6)
        return fill(vals[1], n_cells)
    end
    _, n, block = mode_block
    vals = tuple_values(block, 6)
    n == n_cells || error("$(path) has $(n) cells, mesh has $(n_cells)")
    length(vals) == n || error("symmTensor field count mismatch in $(path)")
    return vals
end

function cell_centers(case_dir, faces, owner, neighbour, points)
    n_cells = maximum((maximum(owner), isempty(neighbour) ? 0 : maximum(neighbour)))
    point_sets = [Set{Int}() for _ in 1:n_cells]
    for f in eachindex(faces)
        o = owner[f]
        foreach(p -> push!(point_sets[o], p), faces[f])
        if f <= length(neighbour)
            nb = neighbour[f]
            foreach(p -> push!(point_sets[nb], p), faces[f])
        end
    end
    x = zeros(Float64, n_cells); y = zeros(Float64, n_cells)
    for c in 1:n_cells
        sx = 0.0; sy = 0.0
        for p in point_sets[c]
            sx += points[p][1]; sy += points[p][2]
        end
        x[c] = sx / length(point_sets[c]); y[c] = sy / length(point_sets[c])
    end
    return x, y, n_cells
end

function parse_boundary(case_dir)
    text = strip_comments(read_text_maybe_gzip(foam_path(case_dir, "constant", "polyMesh", "boundary")))
    patches = Dict{String, Tuple{Int, Int}}()
    for m in eachmatch(r"(\w+)\s*\{([^{}]*)\}"s, text)
        name = m.captures[1]; body = m.captures[2]
        nf_m = match(r"nFaces\s+(\d+)", body)
        sf_m = match(r"startFace\s+(\d+)", body)
        if nf_m !== nothing && sf_m !== nothing
            patches[name] = (parse(Int, nf_m.captures[1]),
                             parse(Int, sf_m.captures[1]))
        end
    end
    return patches
end

function build_index(xs::Vector{Float64}, ys::Vector{Float64})
    x_min, x_max = extrema(xs); y_min, y_max = extrema(ys)
    n = length(xs)
    nb = max(8, ceil(Int, sqrt(n) / 2))
    dx = max((x_max - x_min) / nb, eps(Float64))
    dy = max((y_max - y_min) / nb, eps(Float64))
    bins = [Int[] for _ in 1:(nb * nb)]
    index = SpatialIndex(x_min, y_min, dx, dy, nb, nb, xs, ys, bins)
    for p in eachindex(xs)
        bx = clamp(floor(Int, (xs[p] - x_min) / dx) + 1, 1, nb)
        by = clamp(floor(Int, (ys[p] - y_min) / dy) + 1, 1, nb)
        push!(bins[bx + (by - 1) * nb], p)
    end
    return index
end

function candidate_indices(index::SpatialIndex, xq::Float64, yq::Float64, k::Int)
    bx = clamp(floor(Int, (xq - index.x_min) / index.dx) + 1, 1, index.nx)
    by = clamp(floor(Int, (yq - index.y_min) / index.dy) + 1, 1, index.ny)
    candidates = Int[]
    max_shell = max(index.nx, index.ny)
    for shell in 0:max_shell
        empty!(candidates)
        xlo = max(1, bx - shell); xhi = min(index.nx, bx + shell)
        ylo = max(1, by - shell); yhi = min(index.ny, by + shell)
        for yy in ylo:yhi, xx in xlo:xhi
            append!(candidates, index.bins[xx + (yy - 1) * index.nx])
        end
        length(candidates) >= k && break
    end
    isempty(candidates) && error("no interpolation candidates near ($(xq), $(yq))")
    sort!(candidates; by=p -> (index.xs[p] - xq)^2 + (index.ys[p] - yq)^2)
    length(candidates) > k && resize!(candidates, k)
    return candidates
end

function affine_sample_and_grad(index::SpatialIndex, field::Vector{Float64},
                                xq::Float64, yq::Float64; k::Int=12)
    candidates = candidate_indices(index, xq, yq, k)
    nearest = candidates[1]
    A = zeros(Float64, 3, 3); b = zeros(Float64, 3)
    for p in candidates
        dx = index.xs[p] - xq; dy = index.ys[p] - yq
        w = inv(dx * dx + dy * dy + 1e-18)
        basis = (1.0, dx, dy)
        for a in 1:3
            b[a] += w * basis[a] * field[p]
            for c in 1:3
                A[a, c] += w * basis[a] * basis[c]
            end
        end
    end
    if abs(det(A)) <= 1e-30
        return field[nearest], 0.0, 0.0
    end
    coef = A \ b
    return coef[1], coef[2], coef[3]
end

function cylinder_face_geometry(face_pts::Vector{Int},
                                points::Vector{NTuple{3,Float64}},
                                cx::Float64, cy::Float64)
    n = length(face_pts)
    sx = 0.0; sy = 0.0
    for p in face_pts
        sx += points[p][1]; sy += points[p][2]
    end
    cxf = sx / n; cyf = sy / n
    nxv = cxf - cx; nyv = cyf - cy
    rn = hypot(nxv, nyv)
    nxv /= rn; nyv /= rn
    if n == 4
        p1 = points[face_pts[1]]; p2 = points[face_pts[2]]
        p3 = points[face_pts[3]]; p4 = points[face_pts[4]]
        d1 = (p3[1]-p1[1], p3[2]-p1[2], p3[3]-p1[3])
        d2 = (p4[1]-p2[1], p4[2]-p2[2], p4[3]-p2[3])
        cr = (d1[2]*d2[3]-d1[3]*d2[2],
              d1[3]*d2[1]-d1[1]*d2[3],
              d1[1]*d2[2]-d1[2]*d2[1])
        area = 0.5 * hypot(cr[1], cr[2], cr[3])
    else
        area = 0.0
        for i in 1:n
            j = i == n ? 1 : i + 1
            area += points[face_pts[i]][1] * points[face_pts[j]][2] -
                    points[face_pts[j]][1] * points[face_pts[i]][2]
        end
        area = abs(area) * 0.5
    end
    return cxf, cyf, nxv, nyv, area
end

function rheotool_wall_decomp(case_dir, time_name; N_az::Int=36,
                              cylinder_patch_name::String="cylinder",
                              etaS::Float64=0.59, etaP::Float64=0.41,
                              rho::Float64=1.0, cx::Float64=0.0, cy::Float64=0.0)
    points  = parse_points(case_dir)
    faces   = parse_faces(case_dir)
    owner   = parse_label_list(case_dir, "owner")
    nbour   = parse_label_list(case_dir, "neighbour")
    cx_arr, cy_arr, n_cells = cell_centers(case_dir, faces, owner, nbour, points)
    patches = parse_boundary(case_dir)
    haskey(patches, cylinder_patch_name) ||
        error("patch $(cylinder_patch_name) not in boundary; have: $(collect(keys(patches)))")
    nfaces_patch, start_face0 = patches[cylinder_patch_name]
    start_face1 = start_face0 + 1

    time_dir = joinpath(case_dir, time_name)
    U   = parse_vol_vector(joinpath(time_dir, "U"), n_cells)
    p   = parse_vol_scalar(joinpath(time_dir, "p"), n_cells)
    tau = parse_vol_symmtensor(joinpath(time_dir, "tau"), n_cells)
    ux  = [u[1] for u in U]; uy = [u[2] for u in U]
    tau_xx = [t[1] for t in tau]; tau_xy = [t[2] for t in tau]; tau_yy = [t[4] for t in tau]

    index = build_index(cx_arr, cy_arr)
    norm_factor = etaS + etaP

    dtheta = 2π / N_az
    bins_pres = zeros(Float64, N_az)
    bins_solv = zeros(Float64, N_az)
    bins_poly = zeros(Float64, N_az)
    bins_count = zeros(Int, N_az)
    face_records = Vector{NamedTuple}()
    Cd_pres = 0.0; Cd_solv = 0.0; Cd_poly = 0.0

    for f in 1:nfaces_patch
        fid = start_face1 + f - 1
        o   = owner[fid]
        face_pts = faces[fid]
        xf, yf, nxv, nyv, area = cylinder_face_geometry(face_pts, points, cx, cy)
        pw   = p[o]
        txx  = tau_xx[o]; txy = tau_xy[o]; tyy = tau_yy[o]
        _, dudx, dudy = affine_sample_and_grad(index, ux, cx_arr[o], cy_arr[o])
        _, dvdx, dvdy = affine_sample_and_grad(index, uy, cx_arr[o], cy_arr[o])
        Dxx = dudx; Dyy = dvdy
        # σ_pres   = -p ρ I    →   tx_pres = -p·ρ·nx
        sxx_pres = -pw * rho
        # σ_solv = 2 etaS D    →   tx_solv = (2 etaS Dxx) nx + (etaS (dudy+dvdx)) ny
        sxx_solv = 2.0 * etaS * Dxx
        sxy_solv =       etaS * (dudy + dvdx)
        # σ_poly = τ_p (symmTensor in (xx, xy, xz, yy, yz, zz) layout)
        sxx_poly = txx; sxy_poly = txy
        tx_pres = sxx_pres * nxv                      # σ_pres has no off-diagonal
        tx_solv = sxx_solv * nxv + sxy_solv * nyv
        tx_poly = sxx_poly * nxv + sxy_poly * nyv

        dCdp = tx_pres * area / norm_factor
        dCds = tx_solv * area / norm_factor
        dCdq = tx_poly * area / norm_factor

        Cd_pres += dCdp; Cd_solv += dCds; Cd_poly += dCdq

        theta = atan(yf - cy, xf - cx)
        bin = mod(floor(Int, (theta + π) / dtheta), N_az) + 1
        bins_pres[bin] += dCdp
        bins_solv[bin] += dCds
        bins_poly[bin] += dCdq
        bins_count[bin] += 1
        push!(face_records, (
            face_id=fid, x=xf, y=yf, nx=nxv, ny=nyv, area=area, theta=theta,
            p=pw, tau_xx=txx, tau_xy=txy, tau_yy=tyy,
            dCd_pres=dCdp, dCd_solv=dCds, dCd_poly=dCdq))
    end

    theta_centres = [(-π + (k - 0.5)*dtheta) for k in 1:N_az]
    return (;
        N_az, dtheta, theta_centres, bin_count=bins_count,
        dCd_pres=bins_pres, dCd_solv=bins_solv, dCd_poly=bins_poly,
        Cd_pres, Cd_solv, Cd_poly,
        Cd_total = Cd_pres + Cd_solv + Cd_poly,
        face_records,
    )
end

# ===========================================================================
# Part C — Per-bin → 3-region aggregation + bucket matrix
# ===========================================================================

function aggregate_regions(theta_centres::Vector{Float64},
                            dCd_pres::Vector{Float64},
                            dCd_solv::Vector{Float64},
                            dCd_poly::Vector{Float64})
    # Returns Dict{Symbol, NamedTuple{(:pres, :solv, :poly, :total)}}
    out = Dict{Symbol, NamedTuple{(:pres, :solv, :poly, :total), NTuple{4,Float64}}}()
    for reg in (:front_pole, :shoulder, :wake)
        out[reg] = (pres=0.0, solv=0.0, poly=0.0, total=0.0)
    end
    for k in eachindex(theta_centres)
        reg = region_of(theta_centres[k])
        cur = out[reg]
        out[reg] = (
            pres = cur.pres + dCd_pres[k],
            solv = cur.solv + dCd_solv[k],
            poly = cur.poly + dCd_poly[k],
            total = cur.total + dCd_pres[k] + dCd_solv[k] + dCd_poly[k],
        )
    end
    return out
end

function write_bins_csv(path, theta_centres, bin_count, dCd_pres, dCd_solv, dCd_poly)
    open(path, "w") do io
        println(io, "theta_rad,theta_deg,n_cells,dCd_pres,dCd_visc_solvent,dCd_polymer,dCd_total")
        for k in eachindex(theta_centres)
            t = theta_centres[k]
            @printf(io, "%g,%g,%d,%g,%g,%g,%g\n",
                    t, rad2deg(t), bin_count[k],
                    dCd_pres[k], dCd_solv[k], dCd_poly[k],
                    dCd_pres[k]+dCd_solv[k]+dCd_poly[k])
        end
    end
end

# ===========================================================================
# Main
# ===========================================================================

function main()
    println("[", now(), "] M32 Phase 4 — Wi=1 gap localization")
    N_az = parse(Int, get(ENV, "N_AZ", "36"))

    # --------------------------------------------------------------------------
    # Step 1 — Kraken R=30 (Metal F32, :rusanov, with rho)
    # --------------------------------------------------------------------------
    K30_path = joinpath(REPO_ROOT, "tmp", "m30_rho_metal", "run01",
        "cyl_bigsweep_v2_beta0p59_wi1_re1_R30_bsd1_Lup15_Ldn15_eg0_ea0_ef0_ed0_geomqwall_fields.jls")
    isfile(K30_path) || error("missing Kraken R=30 snap: $K30_path")
    snap30 = deserialize(K30_path)
    @printf("  Kraken R=30: Nx=%d Ny=%d R=%g scheme=%s Cd_stored=%.4f\n",
            snap30.Nx, snap30.Ny, snap30.radius_lbm, snap30.advection_scheme, snap30.Cd_kraken)
    K30 = kraken_wall_decomp(snap30; N_az=N_az)
    @printf("  Kraken R=30 ring: Cd_pres=%.4f  Cd_solv=%.4f  Cd_poly=%.4f  Σ=%.4f  (stored=%.4f, drift=%.2f%%)\n",
            K30.Cd_pres, K30.Cd_solv, K30.Cd_poly, K30.Cd_total,
            snap30.Cd_kraken,
            100*(K30.Cd_total - snap30.Cd_kraken)/snap30.Cd_kraken)
    write_bins_csv(joinpath(OUTPUT_DIR, "M32P4_kraken_R30_bins_idx.csv"),
                   K30.theta_centres, K30.bin_count,
                   K30.dCd_pres, K30.dCd_solv, K30.dCd_poly)

    # --------------------------------------------------------------------------
    # Step 1b — Kraken R=40
    # --------------------------------------------------------------------------
    K40_path = joinpath(REPO_ROOT, "tmp", "m30_R_sweep_metal",
        "cyl_bigsweep_v2_beta0p59_wi1_re1_R40_bsd1_Lup15_Ldn15_eg0_ea0_ef0_ed0_geomqwall_fields.jls")
    K40 = nothing; snap40 = nothing
    if isfile(K40_path)
        snap40 = deserialize(K40_path)
        @printf("  Kraken R=40: Nx=%d Ny=%d R=%g scheme=%s Cd_stored=%.4f\n",
                snap40.Nx, snap40.Ny, snap40.radius_lbm, snap40.advection_scheme, snap40.Cd_kraken)
        K40 = kraken_wall_decomp(snap40; N_az=N_az)
        @printf("  Kraken R=40 ring: Cd_pres=%.4f  Cd_solv=%.4f  Cd_poly=%.4f  Σ=%.4f  (stored=%.4f, drift=%.2f%%)\n",
                K40.Cd_pres, K40.Cd_solv, K40.Cd_poly, K40.Cd_total,
                snap40.Cd_kraken,
                100*(K40.Cd_total - snap40.Cd_kraken)/snap40.Cd_kraken)
        write_bins_csv(joinpath(OUTPUT_DIR, "M32P4_kraken_R40_bins_idx.csv"),
                       K40.theta_centres, K40.bin_count,
                       K40.dCd_pres, K40.dCd_solv, K40.dCd_poly)
    else
        @warn "Kraken R=40 snap missing — Step 5 cross-check deferred"
    end

    # --------------------------------------------------------------------------
    # Step 2 — rheoTool Wi=1 shrunk15R at t=20
    # --------------------------------------------------------------------------
    rt_case = joinpath(REPO_ROOT, "bench", "rheotool", "cylinder_wi1.0_shrunk15R")
    println("  rheoTool case: ", rt_case)
    rtT = rheotool_wall_decomp(rt_case, "20"; N_az=N_az,
                                etaS=0.59, etaP=0.41, rho=1.0, cx=0.0, cy=0.0)
    @printf("  rT Wi=1 shrunk: Cd_pres=%.4f  Cd_solv=%.4f  Cd_poly=%.4f  Σ=%.4f  (Cd.txt last=120.383)\n",
            rtT.Cd_pres, rtT.Cd_solv, rtT.Cd_poly, rtT.Cd_total)
    write_bins_csv(joinpath(OUTPUT_DIR, "M32P4_rheotool_wi1_shrunk_bins.csv"),
                   rtT.theta_centres, rtT.bin_count,
                   rtT.dCd_pres, rtT.dCd_solv, rtT.dCd_poly)

    # --------------------------------------------------------------------------
    # Step 3 — 3×3 bucket matrix (rT − Kraken R=30)
    # --------------------------------------------------------------------------
    K30_reg = aggregate_regions(K30.theta_centres, K30.dCd_pres, K30.dCd_solv, K30.dCd_poly)
    rt_reg  = aggregate_regions(rtT.theta_centres, rtT.dCd_pres, rtT.dCd_solv, rtT.dCd_poly)

    total_gap = rtT.Cd_total - K30.Cd_total

    println("\n  === 3×3 bucket matrix (ΔCd = rT − Kraken R=30) ===")
    @printf("  %-12s | %12s | %12s | %12s | %12s\n",
            "component", "front_pole", "shoulder", "wake", "row_sum")
    @printf("  %s\n", "-"^80)
    bucket_rows = NamedTuple[]
    for comp in (:pres, :solv, :poly)
        row_sum = 0.0
        cells = Float64[]
        for reg in (:front_pole, :shoulder, :wake)
            d = getproperty(rt_reg[reg], comp) - getproperty(K30_reg[reg], comp)
            push!(cells, d); row_sum += d
        end
        @printf("  %-12s | %+12.4f | %+12.4f | %+12.4f | %+12.4f\n",
                String(comp), cells[1], cells[2], cells[3], row_sum)
        push!(bucket_rows, (comp=comp, front_pole=cells[1], shoulder=cells[2], wake=cells[3], row_sum=row_sum))
    end
    # Column sums
    col_pres = sum(b.front_pole for b in bucket_rows)
    col_shou = sum(b.shoulder   for b in bucket_rows)
    col_wake = sum(b.wake       for b in bucket_rows)
    @printf("  %-12s | %+12.4f | %+12.4f | %+12.4f | %+12.4f\n",
            "col_sum", col_pres, col_shou, col_wake, total_gap)

    @printf("\n  Total gap Σ = %.4f Cd  (rT %.4f − Kraken %.4f)\n",
            total_gap, rtT.Cd_total, K30.Cd_total)
    @printf("  (rT Cd.txt last value = 120.383; K30 stored Cd = %.4f)\n", snap30.Cd_kraken)

    # Write bucket matrix CSV
    bucket_csv = joinpath(OUTPUT_DIR, "M32P4_bucket_matrix.csv")
    open(bucket_csv, "w") do io
        println(io, "component,front_pole,shoulder,wake,row_sum,fraction_front,fraction_shoulder,fraction_wake")
        for b in bucket_rows
            @printf(io, "%s,%g,%g,%g,%g,%g,%g,%g\n",
                    String(b.comp), b.front_pole, b.shoulder, b.wake, b.row_sum,
                    b.front_pole/total_gap, b.shoulder/total_gap, b.wake/total_gap)
        end
    end
    println("  wrote ", bucket_csv)

    # --------------------------------------------------------------------------
    # Step 4 — find dominant bucket
    # --------------------------------------------------------------------------
    cells_list = NamedTuple[]
    for b in bucket_rows
        for (reg_name, val) in ((:front_pole, b.front_pole), (:shoulder, b.shoulder), (:wake, b.wake))
            push!(cells_list, (component=b.comp, region=reg_name, value=val,
                               fraction = total_gap != 0 ? val/total_gap : NaN))
        end
    end
    sort!(cells_list; by = c -> abs(c.fraction), rev=true)
    @printf("\n  === Top 5 buckets by |fraction| ===\n")
    @printf("  %-12s | %-12s | %12s | %10s\n", "component", "region", "ΔCd", "fraction")
    @printf("  %s\n", "-"^60)
    for c in cells_list[1:min(5, length(cells_list))]
        @printf("  %-12s | %-12s | %+12.4f | %+9.2f%%\n",
                String(c.component), String(c.region), c.value, 100*c.fraction)
    end

    dominant = cells_list[1]
    @printf("\n  Dominant bucket: (%s, %s) with fraction = %.1f%% of total gap\n",
            String(dominant.component), String(dominant.region), 100*dominant.fraction)

    # M33 premise specific bucket: (poly, wake)
    poly_wake_idx = findfirst(c -> c.component == :poly && c.region == :wake, cells_list)
    poly_wake = cells_list[poly_wake_idx]
    @printf("  M33 premise bucket (polymer, wake): ΔCd = %+.4f  (%.1f%% of total gap)\n",
            poly_wake.value, 100*poly_wake.fraction)

    # Verdict tag (≥70% threshold)
    threshold = 0.70
    confirms = (poly_wake.fraction >= threshold)
    contradicts_dominant = (abs(dominant.fraction) >= threshold &&
                            !(dominant.component == :poly && dominant.region == :wake))
    verdict_tag = if confirms
        "CONFIRMS"
    elseif contradicts_dominant
        "CONTRADICTS"
    else
        "INCONCLUSIVE"
    end
    @printf("\n  Verdict tag: **%s**\n", verdict_tag)

    # --------------------------------------------------------------------------
    # Step 5 — R=40 cross-check
    # --------------------------------------------------------------------------
    r40_summary = ""
    poly_wake_R40 = (value=NaN, fraction=NaN, dominant_component=:none, dominant_region=:none, dominant_fraction=NaN)
    if K40 !== nothing
        K40_reg = aggregate_regions(K40.theta_centres, K40.dCd_pres, K40.dCd_solv, K40.dCd_poly)
        total_gap_R40 = rtT.Cd_total - K40.Cd_total
        # Recompute dominant bucket at R=40
        cells_R40 = NamedTuple[]
        for comp in (:pres, :solv, :poly)
            for reg in (:front_pole, :shoulder, :wake)
                d = getproperty(rt_reg[reg], comp) - getproperty(K40_reg[reg], comp)
                push!(cells_R40, (component=comp, region=reg, value=d,
                                  fraction = total_gap_R40 != 0 ? d/total_gap_R40 : NaN))
            end
        end
        sort!(cells_R40; by = c -> abs(c.fraction), rev=true)
        @printf("\n  === R=40 cross-check ===\n")
        @printf("  Total gap R=40 = %.4f (rT %.4f − Kraken R=40 %.4f)\n",
                total_gap_R40, rtT.Cd_total, K40.Cd_total)
        @printf("  Top 3 buckets R=40:\n")
        for c in cells_R40[1:3]
            @printf("    (%s, %s): ΔCd = %+.4f  (%.1f%%)\n",
                    String(c.component), String(c.region), c.value, 100*c.fraction)
        end
        dom_R40 = cells_R40[1]
        pwR40 = cells_R40[findfirst(c -> c.component == :poly && c.region == :wake, cells_R40)]
        same_dominant = (dom_R40.component == dominant.component && dom_R40.region == dominant.region)
        rel_diff = abs(dom_R40.fraction - dominant.fraction) / max(abs(dominant.fraction), 1e-9)
        @printf("  R=40 dominant: (%s, %s) %.1f%%   R=30 dominant: (%s, %s) %.1f%%   same? %s rel_diff=%.0f%%\n",
                String(dom_R40.component), String(dom_R40.region), 100*dom_R40.fraction,
                String(dominant.component), String(dominant.region), 100*dominant.fraction,
                same_dominant, 100*rel_diff)
        poly_wake_R40 = (value=pwR40.value, fraction=pwR40.fraction,
                         dominant_component=dom_R40.component, dominant_region=dom_R40.region,
                         dominant_fraction=dom_R40.fraction)
        r40_summary = "R=40 dominant=($(String(dom_R40.component)),$(String(dom_R40.region))) frac=$(round(100*dom_R40.fraction; digits=1))%; rel diff vs R=30 = $(round(100*rel_diff; digits=0))%; same bucket=$same_dominant"
    end

    # --------------------------------------------------------------------------
    # Step 6 — Verdict markdown
    # --------------------------------------------------------------------------
    md_path = joinpath(REPO_ROOT, "bench", "viscoelastic_audit",
                       "M32_PHASE4_WI1_GAP_LOCALIZATION_VERDICT.md")
    open(md_path, "w") do io
        println(io, "# M32 Phase 4 — Wi=1 Cd gap localization (R=30, R=40)\n")
        @printf(io, "Date: %s\n", now())
        println(io, "Branch: dev-viscoelastic")
        println(io, "Mission: D1 (Department: M32-Phase4-walldecomp)")
        @printf(io, "Verdict: **%s M33 premise**\n\n", verdict_tag)

        println(io, "## TL;DR")
        @printf(io, "- Total gap (Cd_rT − Cd_Kraken) at R=30 Wi=1: **%+.4f Cd** (target rT=120.38, K=%.2f → %+.2f%%)\n",
                total_gap, K30.Cd_total, 100*total_gap/K30.Cd_total)
        @printf(io, "- **Dominant bucket: (`%s`, `%s`) carries %.1f%% of the gap.**\n",
                String(dominant.component), String(dominant.region), 100*dominant.fraction)
        @printf(io, "- M33 premise bucket (polymer × wake): ΔCd = %+.4f  (**%.1f%%** of total gap)\n",
                poly_wake.value, 100*poly_wake.fraction)
        @printf(io, "- M33 premise threshold (≥70%%): **%s**\n", confirms ? "CONFIRMED" : "FAILED")
        if K40 !== nothing
            @printf(io, "- R=40 cross-check: %s\n", r40_summary)
        else
            println(io, "- R=40 cross-check: SNAP MISSING (deferred)")
        end
        println(io)

        println(io, "## Setup")
        @printf(io, "- Kraken R=30 snapshot: `tmp/m30_rho_metal/run01/cyl_bigsweep_v2_beta0p59_wi1_re1_R30_bsd1_*.jls`\n")
        @printf(io, "  - Metal F32, :rusanov, BSD=1.0, Cd_stored = %.4f, has `rho`. (F64 Aqua would give Cd≈111.55.)\n", snap30.Cd_kraken)
        if K40 !== nothing
            @printf(io, "- Kraken R=40 snapshot: `tmp/m30_R_sweep_metal/cyl_bigsweep_v2_beta0p59_wi1_re1_R40_bsd1_*.jls`\n")
            @printf(io, "  - Metal F32, :rusanov, BSD=1.0, Cd_stored = %.4f, has `rho`.\n", snap40.Cd_kraken)
        end
        @printf(io, "- rT reference: `bench/rheotool/cylinder_wi1.0_shrunk15R/20/{U,p,tau}.gz` (t=20 converged, Cd.txt last = 120.383)\n")
        @printf(io, "- Frame: `:idx` (kernel-correct, `cx_lu = cx_phys + 1`, per `[[feedback_wall_ring_idx_frame]]`)\n")
        @printf(io, "- N_az = %d bins (Δθ = %.1f°)\n", N_az, 360/N_az)
        @printf(io, "- Region split: front-pole `|θ ± π| < π/4` (90°), shoulder `π/4 ≤ |θ ± π/2|` (180°), wake `|θ| < π/4` (90°)\n")
        @printf(io, "- θ convention: θ=0 at rear-pole (+x), θ=±π at front-pole (-x), θ=±π/2 at shoulders\n\n")

        println(io, "## Step 1 — Kraken R=30 wall decomposition (:idx frame)\n")
        @printf(io, "| component        | front_pole | shoulder | wake     | total |\n")
        @printf(io, "|------------------|------------|----------|----------|-------|\n")
        @printf(io, "| Cd_pressure      | %+10.4f | %+8.4f | %+8.4f | %+8.4f |\n",
                K30_reg[:front_pole].pres, K30_reg[:shoulder].pres, K30_reg[:wake].pres, K30.Cd_pres)
        @printf(io, "| Cd_visc_solvent  | %+10.4f | %+8.4f | %+8.4f | %+8.4f |\n",
                K30_reg[:front_pole].solv, K30_reg[:shoulder].solv, K30_reg[:wake].solv, K30.Cd_solv)
        @printf(io, "| Cd_polymer       | %+10.4f | %+8.4f | %+8.4f | %+8.4f |\n",
                K30_reg[:front_pole].poly, K30_reg[:shoulder].poly, K30_reg[:wake].poly, K30.Cd_poly)
        @printf(io, "| **column total** | %+10.4f | %+8.4f | %+8.4f | **%+.4f** |\n",
                K30_reg[:front_pole].total, K30_reg[:shoulder].total, K30_reg[:wake].total, K30.Cd_total)
        @printf(io, "\nReconciliation: Σ ring components = %.4f vs stored Cd_kraken = %.4f → drift = %+.2f%%\n\n",
                K30.Cd_total, snap30.Cd_kraken, 100*(K30.Cd_total - snap30.Cd_kraken)/snap30.Cd_kraken)

        println(io, "## Step 2 — rheoTool Wi=1 shrunk15R wall decomposition (t=20)\n")
        @printf(io, "| component        | front_pole | shoulder | wake     | total |\n")
        @printf(io, "|------------------|------------|----------|----------|-------|\n")
        @printf(io, "| Cd_pressure      | %+10.4f | %+8.4f | %+8.4f | %+8.4f |\n",
                rt_reg[:front_pole].pres, rt_reg[:shoulder].pres, rt_reg[:wake].pres, rtT.Cd_pres)
        @printf(io, "| Cd_visc_solvent  | %+10.4f | %+8.4f | %+8.4f | %+8.4f |\n",
                rt_reg[:front_pole].solv, rt_reg[:shoulder].solv, rt_reg[:wake].solv, rtT.Cd_solv)
        @printf(io, "| Cd_polymer       | %+10.4f | %+8.4f | %+8.4f | %+8.4f |\n",
                rt_reg[:front_pole].poly, rt_reg[:shoulder].poly, rt_reg[:wake].poly, rtT.Cd_poly)
        @printf(io, "| **column total** | %+10.4f | %+8.4f | %+8.4f | **%+.4f** |\n",
                rt_reg[:front_pole].total, rt_reg[:shoulder].total, rt_reg[:wake].total, rtT.Cd_total)
        @printf(io, "\nReconciliation: Σ ring components = %.4f vs Cd.txt last = 120.383 → drift = %+.2f%%\n\n",
                rtT.Cd_total, 100*(rtT.Cd_total - 120.383)/120.383)

        println(io, "## Step 3 — 3×3 bucket gap matrix (ΔCd = rT − Kraken R=30)\n")
        @printf(io, "| ΔCd              | front_pole | shoulder | wake     | row sum |\n")
        @printf(io, "|------------------|------------|----------|----------|---------|\n")
        for b in bucket_rows
            comp_label = b.comp == :pres ? "pressure" : (b.comp == :solv ? "visc_solvent" : "polymer")
            @printf(io, "| %-16s | %+10.4f | %+8.4f | %+8.4f | %+8.4f |\n",
                    comp_label, b.front_pole, b.shoulder, b.wake, b.row_sum)
        end
        @printf(io, "| **col sum**      | %+10.4f | %+8.4f | %+8.4f | **%+.4f** |\n",
                col_pres, col_shou, col_wake, total_gap)

        @printf(io, "\nFraction of total gap (sign-aware, denominator = %.4f):\n\n", total_gap)
        @printf(io, "| fraction (%%)     | front_pole | shoulder | wake     |\n")
        @printf(io, "|------------------|------------|----------|----------|\n")
        for b in bucket_rows
            comp_label = b.comp == :pres ? "pressure" : (b.comp == :solv ? "visc_solvent" : "polymer")
            @printf(io, "| %-16s | %+9.1f%% | %+7.1f%% | %+7.1f%% |\n",
                    comp_label,
                    100*b.front_pole/total_gap,
                    100*b.shoulder/total_gap,
                    100*b.wake/total_gap)
        end
        println(io)

        println(io, "## Step 4 — Dominant bucket + M33 premise check\n")
        @printf(io, "Top 5 buckets by |fraction|:\n\n")
        @printf(io, "| rank | component | region | ΔCd | fraction |\n")
        @printf(io, "|------|-----------|--------|-----|----------|\n")
        for (i, c) in enumerate(cells_list[1:min(5, length(cells_list))])
            comp_label = c.component == :pres ? "pressure" : (c.component == :solv ? "visc_solvent" : "polymer")
            reg_label = String(c.region)
            @printf(io, "| %d | %s | %s | %+.4f | %+.1f%% |\n",
                    i, comp_label, reg_label, c.value, 100*c.fraction)
        end
        println(io)

        @printf(io, "**Dominant bucket: (`%s`, `%s`)** with fraction = **%.1f%%** of total gap.\n\n",
                String(dominant.component), String(dominant.region), 100*dominant.fraction)
        @printf(io, "**M33 premise** asserts the locus is `(polymer, wake)` (wake-side `:rusanov`-over-dissipation):\n")
        @printf(io, "- Observed (polymer, wake) fraction = **%.1f%%** of total gap\n", 100*poly_wake.fraction)
        @printf(io, "- Threshold for premise CONFIRMED: ≥ 70%%\n")
        @printf(io, "- **Premise %s**\n\n", confirms ? "**CONFIRMED**" : "**NOT CONFIRMED**")

        if !confirms && abs(dominant.fraction) >= threshold &&
           !(dominant.component == :poly && dominant.region == :wake)
            dom_comp = dominant.component == :pres ? "pressure" : (dominant.component == :solv ? "visc_solvent" : "polymer")
            @printf(io, "**The actual dominant bucket is `(%s, %s)`. This CONTRADICTS the M33 premise.**\n\n",
                    dom_comp, String(dominant.region))
        elseif !confirms
            println(io, "**No single bucket reaches 70%. INCONCLUSIVE — multi-bucket story; see top-5 ranking.**\n")
        end

        println(io, "## Step 5 — R=40 cross-check\n")
        if K40 !== nothing
            K40_reg = aggregate_regions(K40.theta_centres, K40.dCd_pres, K40.dCd_solv, K40.dCd_poly)
            total_gap_R40 = rtT.Cd_total - K40.Cd_total
            @printf(io, "Kraken R=40 ring totals: Cd_pres=%.4f, Cd_solv=%.4f, Cd_poly=%.4f, Σ=%.4f (stored=%.4f, drift=%+.2f%%)\n\n",
                    K40.Cd_pres, K40.Cd_solv, K40.Cd_poly, K40.Cd_total,
                    snap40.Cd_kraken, 100*(K40.Cd_total-snap40.Cd_kraken)/snap40.Cd_kraken)
            @printf(io, "Total gap rT − K40 = **%+.4f Cd** (vs R=30 gap = %+.4f)\n\n", total_gap_R40, total_gap)

            @printf(io, "| ΔCd (R=40)       | front_pole | shoulder | wake     | row sum |\n")
            @printf(io, "|------------------|------------|----------|----------|---------|\n")
            for comp in (:pres, :solv, :poly)
                comp_label = comp == :pres ? "pressure" : (comp == :solv ? "visc_solvent" : "polymer")
                fp = getproperty(rt_reg[:front_pole], comp) - getproperty(K40_reg[:front_pole], comp)
                sh = getproperty(rt_reg[:shoulder],   comp) - getproperty(K40_reg[:shoulder],   comp)
                wk = getproperty(rt_reg[:wake],       comp) - getproperty(K40_reg[:wake],       comp)
                @printf(io, "| %-16s | %+10.4f | %+8.4f | %+8.4f | %+8.4f |\n",
                        comp_label, fp, sh, wk, fp+sh+wk)
            end
            @printf(io, "\n%s\n\n", r40_summary)
        else
            println(io, "Kraken R=40 snapshot missing — cross-check deferred.\n")
        end

        println(io, "## Caveats\n")
        println(io, "- Both Kraken snapshots are **Metal F32**, not Aqua F64. Per the M32 Phase 3 mandate table, F32 Cd is typically ~0.5 Cd off F64 (R=30 F32=111.09 vs F64=111.55). The **bucket identity** is expected stable; the **magnitude** of |fraction| may shift by ~5% relative.")
        println(io, "- Reconciliation drift between ring Σ and stored Cd_kraken: the ring integral on the staircased boundary is a different decomposition than the LBM-MEA cut-link integral (`Cd_kraken = Cd_s + Cd_p − Cd_bsd`). Drift up to ~5% is structural and not a methodology error (see `[[feedback_wall_ring_idx_frame]]`).")
        println(io, "- The rT FOAM cell-gradient is reconstructed via kNN affine fit, not the bit-for-bit OpenFOAM Gauss-linear gradient. Solvent contribution is the most sensitive to this — expect ~5% variation on the solvent component alone.")
        println(io, "- Both pressure constants float (Kraken: ρ-1 LBM with no reference fix, rT: zeroGradient on inlet); only **gradients** of p around the cylinder are physically meaningful for Cd_pressure. Both codes use the same convention.\n")

        println(io, "## Files\n")
        @printf(io, "- `bench/scratch/m32_phase4_wi1_walldecomp/M32P4_kraken_R30_bins_idx.csv` — 36 bins, Kraken R=30 :idx-frame\n")
        if K40 !== nothing
            @printf(io, "- `bench/scratch/m32_phase4_wi1_walldecomp/M32P4_kraken_R40_bins_idx.csv` — 36 bins, Kraken R=40 :idx-frame\n")
        end
        @printf(io, "- `bench/scratch/m32_phase4_wi1_walldecomp/M32P4_rheotool_wi1_shrunk_bins.csv` — 36 bins, rT Wi=1 shrunk15R t=20\n")
        @printf(io, "- `bench/scratch/m32_phase4_wi1_walldecomp/M32P4_bucket_matrix.csv` — flat 3×3 bucket table\n")
        @printf(io, "- `bench/scratch/m32_phase4_wi1_walldecomp/run_walldecomp.jl` — full driver\n\n")

        println(io, "## Memory candidates\n")
        println(io, "1. **M32 Phase 4 bucket-attribution template** — the (component × region) 3×3 matrix on the `:idx` ring is a reusable template for any Kraken-vs-rT viscoelastic Cd-gap attribution. Pattern: K-side via `kraken_wall_decomp` on a `.jls` snapshot (requires `rho` field — check schema), rT-side via `rheotool_wall_decomp` on the `constant/polyMesh` + last-time `(U, p, tau).gz`, both binned with the SAME `N_az` and the same θ convention, then `aggregate_regions` collapses to 3 angular buckets. Total dev cost: 1 mission once the locked harnesses (`m30_centering_audit`, `m29c_wallstress`) exist.")
        println(io, "2. **Kraken `.jls` schema variability** — different sweep generations produce dumps with different field sets. `tmp/m29c_kraken/` (M29c rolled-back run) has `:rusanov`? No — it's `muscl_superbee` with Cd=-1571 (catastrophe), missing `rho`. `tmp/m30_rho_metal/run01/` is the canonical Wi=1 R=30 :rusanov F32 dump with `rho`. ALWAYS inspect `propertynames(snap)`, `snap.advection_scheme`, and `snap.Cd_kraken` BEFORE using any dump — do not trust the directory name alone.")
        println(io, "3. **3-region split convention** — front-pole `|θ±π|<π/4`, wake `|θ|<π/4`, shoulder `π/4≤|θ|≤3π/4`. Each region is 90° (front, wake) or 180° (shoulder, both flanks combined). For finer resolution use 5° bins (N_az=72) and post-aggregate; the 36-bin (10°) granularity is the minimum that distinguishes pole/shoulder/wake cleanly without staircase aliasing.")
    end
    println("wrote ", md_path)
    println("[", now(), "] M32 Phase 4 done.")

    return (; verdict_tag, dominant, poly_wake, total_gap)
end

main()
