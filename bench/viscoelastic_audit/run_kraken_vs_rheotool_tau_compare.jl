#!/usr/bin/env julia
#
# M29-tau-compare — direct field-level comparison of (u, tau) between Kraken
# (LBM, Cartesian) and rheoTool (FVM, body-fitted O-grid) at the steady-state
# operating point β=0.59, Re=1, R=30, Wi=1.0.
#
# Inputs (env-overridable):
#   KRAKEN_FIELDS_JLS    Path to Kraken field snapshot (.jls produced by the
#                        patched run_cyl_bigsweep_v2_2d.jl with
#                        KRAKEN_SAVE_FIELDS=1).
#   RHEOTOOL_CASE_DIR    Path to rheoTool case (default
#                        bench/rheotool/cylinder_wi1.0).
#   RHEOTOOL_TIME        Time directory to load (default "10").
#   OUTPUT_DIR           Where plots + numeric residuals land
#                        (default bench/scratch/m29_tau_compare).
#
# Mesh / unit conventions
# -----------------------
# rheoTool : x ∈ [-20, 60], y ∈ [-2, 2] (mirrored to full y from half [0, 2]),
#            cylinder at origin, R=1, U_mean=1 (inlet parabolic).
# Kraken   : Cartesian LBM grid in lattice units. Cylinder center at
#            (cx, cy) = (L_up*R, (Ny-1)/2) in cell-index units (1-based).
#            Cells are 1..Nx × 1..Ny, dx = dy = 1 LU. u_mean = 0.005 LU/step.
#
# To compare we map a Cartesian ROI in *physical* (rheoTool) units onto BOTH
# datasets:
#   physical -> Kraken cell index : i_phys(x_p) = cx + x_p * R + 0.5
#                                   j_phys(y_p) = cy + y_p * R + 0.5
#   physical -> rheoTool          : direct kNN+affine interpolation
#
# Velocity unit conversion: Kraken velocity is in LU/step, rheoTool in
# physical m/s. The natural non-dimensionalisation is u / U_mean — we
# compare velocities as **u/U_mean**, dimensionless. Stress (tau) in
# rheoTool is **viscous polymer extra stress** (units = ρ·U_mean²·1)
# whereas Kraken's tauxx is the LBM polymer extra stress in LU². To
# compare apples to apples we report tau / (ρ·U_mean²) on both sides
# (Kraken: ρ=1 LU, U_mean=0.005; rheoTool: ρ=1, U_mean=1).

using Dates
using LinearAlgebra
using Printf
using Serialization
using Statistics

using CairoMakie

const SCRIPT_DIR = @__DIR__
const REPO_ROOT  = abspath(joinpath(SCRIPT_DIR, "..", ".."))

# Reuse the FOAM I/O helpers from the cavity frozen-replay script. They're
# stand-alone functions (parse_points / parse_faces / parse_label_list /
# parse_vol_vector / parse_vol_symmtensor / cell_centers / SpatialIndex /
# build_index / candidate_indices / affine_sample). Including the file
# evaluates them at top-level here (no side-effects beyond function defs
# because the cavity driver wraps its run in main()).

# We replicate only the helpers we need, to avoid pulling Kraken / CUDA
# transitive deps from the cavity script.

# ---------------------------------------------------------------------------
# Minimal FOAM ASCII reader (copied from
# bench/viscoelastic_logfv/run_rheotool_frozen_replay_cavity_2d.jl, MIT-
# style: same author, same project).
# ---------------------------------------------------------------------------

mutable struct SpatialIndex
    x_min::Float64
    y_min::Float64
    dx::Float64
    dy::Float64
    nx::Int
    ny::Int
    xs::Vector{Float64}
    ys::Vector{Float64}
    bins::Vector{Vector{Int}}
end

function strip_comments(text::AbstractString)
    text = replace(text, r"/\*.*?\*/"s => "")
    return join((replace(line, r"//.*$" => "") for line in split(text, '\n')), "\n")
end

function read_text_maybe_gzip(path::AbstractString)
    if isfile(path)
        return read(path, String)
    elseif isfile(path * ".gz")
        return read(pipeline(`gzip -cd $(path * ".gz")`), String)
    end
    throw(SystemError(path, 2))
end

function foam_path(case_dir, parts...)
    plain = joinpath(case_dir, parts...)
    return plain
end

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
    open_pos = start + count_match.offset + length(count_match.match) - 2
    depth = 0
    close_pos = open_pos
    for idx in open_pos:lastindex(clean)
        c = clean[idx]
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
    return n, clean[(open_pos + 1):(close_pos - 1)]
end

function parse_points(case_dir::AbstractString)
    text = read_text_maybe_gzip(foam_path(case_dir, "constant", "polyMesh", "points"))
    n, block = parse_counted_block(text, "points")
    pts = tuple_values(block, 3)
    length(pts) == n || error("points count mismatch: header $(n), parsed $(length(pts))")
    return pts
end

function parse_faces(case_dir::AbstractString)
    text = strip_comments(read_text_maybe_gzip(foam_path(case_dir, "constant", "polyMesh", "faces")))
    n, block = parse_counted_block(text, "faces")
    faces = Vector{Vector{Int}}()
    for m in eachmatch(r"\b\d+\s*\(([^\)]*)\)", block)
        push!(faces, [parse(Int, x) + 1 for x in split(strip(m.captures[1]))])
    end
    length(faces) == n || error("faces count mismatch: header $(n), parsed $(length(faces))")
    return faces
end

function parse_label_list(case_dir::AbstractString, name::AbstractString)
    text = read_text_maybe_gzip(foam_path(case_dir, "constant", "polyMesh", name))
    n, block = parse_counted_block(text, name)
    labels = [parse(Int, m.match) + 1 for m in eachmatch(r"-?\d+", block)]
    length(labels) == n || error("$(name) count mismatch: header $(n), parsed $(length(labels))")
    return labels
end

function foam_internal_block(text::AbstractString)
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

function parse_vol_vector(path::AbstractString, n_cells::Int)
    # Try with .gz fallback
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
        length(vals) == 1 || error("uniform vector field in $(path) did not parse")
        return fill(vals[1], n_cells)
    end
    _, n, block = mode_block
    vals = tuple_values(block, 3)
    n == n_cells || error("$(path) has $(n) cells, mesh has $(n_cells)")
    length(vals) == n || error("vector field count mismatch in $(path)")
    return vals
end

function parse_vol_symmtensor(path::AbstractString, n_cells::Int)
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
        length(vals) == 1 || error("uniform symmTensor field in $(path) did not parse")
        return fill(vals[1], n_cells)
    end
    _, n, block = mode_block
    vals = tuple_values(block, 6)
    n == n_cells || error("$(path) has $(n) cells, mesh has $(n_cells)")
    length(vals) == n || error("symmTensor field count mismatch in $(path)")
    return vals
end

function cell_centers(case_dir::AbstractString)
    points = parse_points(case_dir)
    faces = parse_faces(case_dir)
    owner = parse_label_list(case_dir, "owner")
    neighbour = parse_label_list(case_dir, "neighbour")
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
    x = zeros(Float64, n_cells)
    y = zeros(Float64, n_cells)
    for c in 1:n_cells
        isempty(point_sets[c]) && error("cell $(c) has no points")
        sx = 0.0
        sy = 0.0
        for p in point_sets[c]
            sx += points[p][1]
            sy += points[p][2]
        end
        x[c] = sx / length(point_sets[c])
        y[c] = sy / length(point_sets[c])
    end
    return x, y
end

function build_index(xs::Vector{Float64}, ys::Vector{Float64})
    x_min, x_max = extrema(xs)
    y_min, y_max = extrema(ys)
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
        xlo = max(1, bx - shell)
        xhi = min(index.nx, bx + shell)
        ylo = max(1, by - shell)
        yhi = min(index.ny, by + shell)
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

function affine_sample(index::SpatialIndex, field::Vector{Float64}, xq::Float64, yq::Float64; k::Int=12)
    candidates = candidate_indices(index, xq, yq, k)
    nearest = candidates[1]
    nearest_r2 = (index.xs[nearest] - xq)^2 + (index.ys[nearest] - yq)^2
    nearest_r2 <= 1e-24 && return field[nearest]
    A = zeros(Float64, 3, 3)
    b = zeros(Float64, 3)
    for p in candidates
        dx = index.xs[p] - xq
        dy = index.ys[p] - yq
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
        return field[nearest]
    end
    return (A \ b)[1]
end

# ---------------------------------------------------------------------------
# rheoTool snapshot loader for the cylinder case (tau is gzipped here).
# ---------------------------------------------------------------------------

function load_rheotool_snapshot(case_dir::AbstractString, time_name::AbstractString)
    time_dir = joinpath(case_dir, time_name)
    isdir(time_dir) || error("rheoTool time directory $(time_dir) does not exist")
    x, y = cell_centers(case_dir)
    n_cells = length(x)
    U = parse_vol_vector(joinpath(time_dir, "U"), n_cells)
    tau = parse_vol_symmtensor(joinpath(time_dir, "tau"), n_cells)
    return (;
        x, y, n_cells,
        ux=[u[1] for u in U],
        uy=[u[2] for u in U],
        tau_xx=[t[1] for t in tau],
        tau_xy=[t[2] for t in tau],
        tau_yy=[t[4] for t in tau],
    )
end

# ---------------------------------------------------------------------------
# Kraken snapshot loader from .jls
# ---------------------------------------------------------------------------

function load_kraken_snapshot(jls_path::AbstractString)
    snap = deserialize(jls_path)
    # snap is a NamedTuple with fields: ux, uy, tauxx, tauxy, tauyy, is_solid,
    # Nx, Ny, R, dx, dy, cylinder_x_lbm, cylinder_y_lbm, u_mean, ...
    return snap
end

# ---------------------------------------------------------------------------
# Co-registration: pick a ROI in physical units, sample BOTH datasets there.
# ---------------------------------------------------------------------------

function build_kraken_phys_grid(kraken)
    Nx = kraken.Nx; Ny = kraken.Ny
    R = Float64(kraken.radius_lbm)
    cx = Float64(kraken.cylinder_x_lbm)
    cy = Float64(kraken.cylinder_y_lbm)
    # Cell (i, j) is at lattice coordinate (i, j) (1-based node centers).
    # Physical position (cylinder at origin in rheoTool conventions):
    #   x_phys = (i - cx) / R    y_phys = (j - cy) / R
    x_phys = [(i - cx) / R for i in 1:Nx]
    y_phys = [(j - cy) / R for j in 1:Ny]
    return (; x_phys, y_phys, R, cx, cy, Nx, Ny)
end

function sample_kraken_on_phys_grid(kraken, kphys, x_targets, y_targets)
    # Bilinear in the lattice grid; out-of-range or solid -> NaN.
    Nx = kraken.Nx; Ny = kraken.Ny
    R = kphys.R; cx = kphys.cx; cy = kphys.cy
    ux = kraken.ux; uy = kraken.uy
    txx = kraken.tauxx; txy = kraken.tauxy; tyy = kraken.tauyy
    solid = kraken.is_solid
    Nout_x = length(x_targets); Nout_y = length(y_targets)
    out = (
        ux  = fill(NaN, Nout_x, Nout_y),
        uy  = fill(NaN, Nout_x, Nout_y),
        txx = fill(NaN, Nout_x, Nout_y),
        txy = fill(NaN, Nout_x, Nout_y),
        tyy = fill(NaN, Nout_x, Nout_y),
        solid_mask = falses(Nout_x, Nout_y),
    )
    for (jj, yp) in enumerate(y_targets), (ii, xp) in enumerate(x_targets)
        i_lu = cx + xp * R
        j_lu = cy + yp * R
        (i_lu < 1 || i_lu > Nx || j_lu < 1 || j_lu > Ny) && continue
        i0 = clamp(floor(Int, i_lu), 1, Nx - 1)
        j0 = clamp(floor(Int, j_lu), 1, Ny - 1)
        fx = i_lu - i0
        fy = j_lu - j0
        # Solidity test : conservative — any of the 4 corners solid => mask.
        any_solid = solid[i0,j0] | solid[i0+1,j0] | solid[i0,j0+1] | solid[i0+1,j0+1]
        if any_solid
            out.solid_mask[ii, jj] = true
            continue
        end
        function bilin(F)
            return (1-fx)*(1-fy)*F[i0,j0] + fx*(1-fy)*F[i0+1,j0] +
                   (1-fx)*fy*F[i0,j0+1] + fx*fy*F[i0+1,j0+1]
        end
        out.ux[ii, jj]  = bilin(ux)
        out.uy[ii, jj]  = bilin(uy)
        out.txx[ii, jj] = bilin(txx)
        out.txy[ii, jj] = bilin(txy)
        out.tyy[ii, jj] = bilin(tyy)
    end
    return out
end

function sample_rheotool_on_phys_grid(rheo, index, x_targets, y_targets;
                                     cylinder_x=0.0, cylinder_y=0.0, radius=1.0)
    Nx_out = length(x_targets); Ny_out = length(y_targets)
    out = (
        ux  = fill(NaN, Nx_out, Ny_out),
        uy  = fill(NaN, Nx_out, Ny_out),
        txx = fill(NaN, Nx_out, Ny_out),
        txy = fill(NaN, Nx_out, Ny_out),
        tyy = fill(NaN, Nx_out, Ny_out),
        solid_mask = falses(Nx_out, Ny_out),
    )
    for (jj, yp) in enumerate(y_targets), (ii, xp) in enumerate(x_targets)
        if hypot(xp - cylinder_x, yp - cylinder_y) <= radius
            out.solid_mask[ii, jj] = true
            continue
        end
        out.ux[ii, jj]  = affine_sample(index, rheo.ux, xp, yp)
        out.uy[ii, jj]  = affine_sample(index, rheo.uy, xp, yp)
        out.txx[ii, jj] = affine_sample(index, rheo.tau_xx, xp, yp)
        out.txy[ii, jj] = affine_sample(index, rheo.tau_xy, xp, yp)
        out.tyy[ii, jj] = affine_sample(index, rheo.tau_yy, xp, yp)
    end
    return out
end

# ---------------------------------------------------------------------------
# Residual metrics + spatial localisation
# ---------------------------------------------------------------------------

function l2_max(a::Matrix{Float64}, b::Matrix{Float64}, valid::BitMatrix)
    s2_diff = 0.0; s2_ref = 0.0; max_abs = 0.0
    n = 0
    for idx in eachindex(a)
        if valid[idx] && isfinite(a[idx]) && isfinite(b[idx])
            d = a[idx] - b[idx]
            s2_diff += d * d
            s2_ref  += b[idx] * b[idx]
            max_abs = max(max_abs, abs(d))
            n += 1
        end
    end
    n == 0 && return (l2_rel=NaN, max_abs=NaN, n=0, ref_l2=0.0)
    ref_l2 = sqrt(s2_ref / n)
    l2_diff = sqrt(s2_diff / n)
    l2_rel = ref_l2 > 0 ? l2_diff / ref_l2 : NaN
    return (l2_rel=l2_rel, max_abs=max_abs, n=n, ref_l2=ref_l2)
end

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

function plot_field_pair_and_diff(x, y, kraken_field, rheo_field, label;
                                  outpath, valid::BitMatrix, cmap=:RdBu)
    diff_field = kraken_field .- rheo_field
    # mask invalid -> NaN already in input

    fig = Figure(size=(1700, 480))
    # Common color range for the two raw fields
    finite_k = filter(isfinite, kraken_field[valid])
    finite_r = filter(isfinite, rheo_field[valid])
    if isempty(finite_k) || isempty(finite_r)
        @warn "no finite samples for $label"
        return nothing
    end
    lo = min(minimum(finite_k), minimum(finite_r))
    hi = max(maximum(finite_k), maximum(finite_r))
    if lo == hi
        lo -= 1; hi += 1
    end

    ax1 = Axis(fig[1, 1]; title="Kraken $label", xlabel="x/R", ylabel="y/R",
               aspect=DataAspect())
    hm1 = heatmap!(ax1, x, y, kraken_field; colormap=cmap, colorrange=(lo, hi),
                   nan_color=:gray80)
    Colorbar(fig[1, 2], hm1)

    ax2 = Axis(fig[1, 3]; title="rheoTool $label", xlabel="x/R", ylabel="y/R",
               aspect=DataAspect())
    hm2 = heatmap!(ax2, x, y, rheo_field; colormap=cmap, colorrange=(lo, hi),
                   nan_color=:gray80)
    Colorbar(fig[1, 4], hm2)

    # Diff with symmetric color range
    finite_d = filter(isfinite, diff_field[valid])
    if isempty(finite_d)
        save(outpath, fig)
        return nothing
    end
    d_abs = maximum(abs, finite_d)
    d_abs == 0 && (d_abs = 1e-12)
    ax3 = Axis(fig[1, 5]; title="diff (Kraken - rheo) $label", xlabel="x/R",
               ylabel="y/R", aspect=DataAspect())
    hm3 = heatmap!(ax3, x, y, diff_field; colormap=Reverse(:RdBu),
                   colorrange=(-d_abs, d_abs), nan_color=:gray80)
    Colorbar(fig[1, 6], hm3)

    save(outpath, fig)
    return outpath
end

# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

function main()
    kraken_jls = get(ENV, "KRAKEN_FIELDS_JLS", "")
    isempty(kraken_jls) && error(
        "KRAKEN_FIELDS_JLS must be set to a path to the Kraken field .jls " *
        "(produced by run_cyl_bigsweep_v2_2d.jl with KRAKEN_SAVE_FIELDS=1)",
    )
    isfile(kraken_jls) || error("missing Kraken snapshot: $kraken_jls")
    rheotool_case = get(ENV, "RHEOTOOL_CASE_DIR",
                        joinpath(REPO_ROOT, "bench", "rheotool", "cylinder_wi1.0"))
    rheotool_time = get(ENV, "RHEOTOOL_TIME", "10")
    output_dir = get(ENV, "OUTPUT_DIR",
                     joinpath(REPO_ROOT, "bench", "scratch", "m29_tau_compare"))
    mkpath(output_dir)

    println("[", now(), "] M29 tau-compare driver")
    println("  Kraken snapshot : $kraken_jls")
    println("  rheoTool case   : $rheotool_case (time=$rheotool_time)")
    println("  Output dir      : $output_dir")
    flush(stdout)

    kraken = load_kraken_snapshot(kraken_jls)
    @printf("Kraken: Nx=%d Ny=%d R=%g cx=%g cy=%g Cd=%.3f Cd_s=%.3f Cd_p=%.3f Cd_bsd=%.3f\n",
        kraken.Nx, kraken.Ny, kraken.radius_lbm,
        kraken.cylinder_x_lbm, kraken.cylinder_y_lbm,
        kraken.Cd_kraken, kraken.Cd_s, kraken.Cd_p, kraken.Cd_bsd)

    rheo = load_rheotool_snapshot(rheotool_case, rheotool_time)
    @printf("rheoTool: %d cells, x∈[%g,%g], y∈[%g,%g]\n",
        rheo.n_cells, extrema(rheo.x)..., extrema(rheo.y)...)

    # Build ROI in physical units (cylinder centered at origin, R=1 phys).
    # Target the region where the wake stagnation + wall-row sit:
    #   x ∈ [-3, 8],  y ∈ [-2, 2]
    # Sample grid : 256 × 128 (rough, but enough for spatial localisation).
    x_phys_min = parse(Float64, get(ENV, "ROI_X_MIN", "-3.0"))
    x_phys_max = parse(Float64, get(ENV, "ROI_X_MAX",  "8.0"))
    y_phys_min = parse(Float64, get(ENV, "ROI_Y_MIN", "-2.0"))
    y_phys_max = parse(Float64, get(ENV, "ROI_Y_MAX",  "2.0"))
    Nsx = parse(Int, get(ENV, "ROI_NSX", "256"))
    Nsy = parse(Int, get(ENV, "ROI_NSY", "128"))
    x_targets = collect(range(x_phys_min, x_phys_max; length=Nsx))
    y_targets = collect(range(y_phys_min, y_phys_max; length=Nsy))

    println("ROI: x∈[$x_phys_min, $x_phys_max], y∈[$y_phys_min, $y_phys_max], grid=$(Nsx)×$(Nsy)")
    flush(stdout)

    # rheoTool kNN index
    println("[", now(), "] building rheoTool kNN index ...")
    flush(stdout)
    index = build_index(rheo.x, rheo.y)
    println("  index built ($(length(index.bins)) bins).")
    flush(stdout)

    # Kraken phys grid info
    kphys = build_kraken_phys_grid(kraken)
    @printf("Kraken phys range: x∈[%g, %g], y∈[%g, %g]\n",
        first(kphys.x_phys), last(kphys.x_phys),
        first(kphys.y_phys), last(kphys.y_phys))

    println("[", now(), "] sampling Kraken on phys grid ...")
    flush(stdout)
    k_sampled = sample_kraken_on_phys_grid(kraken, kphys, x_targets, y_targets)

    println("[", now(), "] sampling rheoTool on phys grid ...")
    flush(stdout)
    r_sampled = sample_rheotool_on_phys_grid(rheo, index, x_targets, y_targets;
                                             cylinder_x=0.0, cylinder_y=0.0,
                                             radius=1.0)

    # Non-dimensionalise velocities by U_mean. rheoTool U_mean = 1 phys,
    # Kraken U_mean = u_mean LU/step (snap.u_mean).
    u_mean_k = kraken.u_mean
    u_mean_r = 1.0
    ux_k = k_sampled.ux ./ u_mean_k
    uy_k = k_sampled.uy ./ u_mean_k
    ux_r = r_sampled.ux ./ u_mean_r
    uy_r = r_sampled.uy ./ u_mean_r

    # Tau non-dim : Kraken in LU² (rho=1 LU), rheoTool in phys (rho=1).
    # Both use rho=1, so divide by U_mean² of each side.
    txx_k = k_sampled.txx ./ (u_mean_k^2)
    txy_k = k_sampled.txy ./ (u_mean_k^2)
    tyy_k = k_sampled.tyy ./ (u_mean_k^2)
    txx_r = r_sampled.txx ./ (u_mean_r^2)
    txy_r = r_sampled.txy ./ (u_mean_r^2)
    tyy_r = r_sampled.tyy ./ (u_mean_r^2)

    # "valid" = fluid in both AND finite both sides
    valid = .!(k_sampled.solid_mask .| r_sampled.solid_mask)
    for arr in (ux_k, uy_k, txx_k, txy_k, tyy_k,
                ux_r, uy_r, txx_r, txy_r, tyy_r)
        for idx in eachindex(arr)
            if !isfinite(arr[idx])
                valid[idx] = false
            end
        end
    end
    n_valid = count(valid)
    println("Valid samples: $n_valid / $(length(valid)) ($(round(100*n_valid/length(valid); digits=1))%)")
    flush(stdout)

    # ---- L2 + max
    println("\n== L2 / max residuals (non-dim) ==")
    metrics = Dict{Symbol,NamedTuple}()
    for (name, k, r) in (
        (:ux,  ux_k,  ux_r),
        (:uy,  uy_k,  uy_r),
        (:tau_xx, txx_k, txx_r),
        (:tau_xy, txy_k, txy_r),
        (:tau_yy, tyy_k, tyy_r),
    )
        m = l2_max(k, r, valid)
        metrics[name] = m
        @printf("%-8s : L2_rel=%.4f  max_abs=%.4g  ref_L2=%.4g  n=%d\n",
                String(name), m.l2_rel, m.max_abs, m.ref_l2, m.n)
    end

    # ---- Spatial localisation: |diff| as f(x) and f(y) — wake vs wall vs bulk
    println("\n== Spatial localisation ==")
    # Strip x band aggregates
    function band_stats(diff::Matrix{Float64}, valid::BitMatrix, x_targets::Vector{Float64})
        nx = length(x_targets)
        per_band_l2 = fill(NaN, nx)
        per_band_max = fill(NaN, nx)
        for ii in 1:nx
            s2 = 0.0; mx = 0.0; n = 0
            for jj in 1:size(diff, 2)
                if valid[ii, jj]
                    d = diff[ii, jj]
                    if isfinite(d)
                        s2 += d * d
                        mx = max(mx, abs(d))
                        n += 1
                    end
                end
            end
            n > 0 && (per_band_l2[ii] = sqrt(s2 / n); per_band_max[ii] = mx)
        end
        return per_band_l2, per_band_max
    end
    diff_txx = txx_k .- txx_r
    diff_ux = ux_k .- ux_r
    band_txx_l2, band_txx_mx = band_stats(diff_txx, valid, x_targets)
    band_ux_l2,  band_ux_mx  = band_stats(diff_ux,  valid, x_targets)

    # Save numerics
    residual_csv = joinpath(output_dir, "M29_residuals.csv")
    open(residual_csv, "w") do io
        println(io, "field,L2_rel,max_abs,ref_L2,n_valid")
        for (k, v) in metrics
            @printf(io, "%s,%.10g,%.10g,%.10g,%d\n",
                    String(k), v.l2_rel, v.max_abs, v.ref_l2, v.n)
        end
    end
    println("Wrote $residual_csv")

    band_csv = joinpath(output_dir, "M29_band_stats_x.csv")
    open(band_csv, "w") do io
        println(io, "x_R,L2_diff_tau_xx,max_diff_tau_xx,L2_diff_ux,max_diff_ux")
        for ii in eachindex(x_targets)
            @printf(io, "%.6g,%.6g,%.6g,%.6g,%.6g\n",
                    x_targets[ii], band_txx_l2[ii], band_txx_mx[ii],
                    band_ux_l2[ii], band_ux_mx[ii])
        end
    end
    println("Wrote $band_csv")

    # ---- Plots
    println("\n== Plotting ==")
    valid_bm = BitMatrix(valid)
    plot_field_pair_and_diff(
        x_targets, y_targets, ux_k, ux_r, "u_x / U_mean";
        outpath=joinpath(output_dir, "M29_field_ux.png"), valid=valid_bm)
    plot_field_pair_and_diff(
        x_targets, y_targets, uy_k, uy_r, "u_y / U_mean";
        outpath=joinpath(output_dir, "M29_field_uy.png"), valid=valid_bm)
    plot_field_pair_and_diff(
        x_targets, y_targets, txx_k, txx_r, "tau_xx / (rho U_mean^2)";
        outpath=joinpath(output_dir, "M29_field_tau_xx.png"), valid=valid_bm)
    plot_field_pair_and_diff(
        x_targets, y_targets, txy_k, txy_r, "tau_xy / (rho U_mean^2)";
        outpath=joinpath(output_dir, "M29_field_tau_xy.png"), valid=valid_bm)
    plot_field_pair_and_diff(
        x_targets, y_targets, tyy_k, tyy_r, "tau_yy / (rho U_mean^2)";
        outpath=joinpath(output_dir, "M29_field_tau_yy.png"), valid=valid_bm)

    # Band plot
    fig = Figure(size=(1100, 420))
    ax1 = Axis(fig[1, 1]; title="Per-x-band L2(diff) — wall row vs wake stagnation",
               xlabel="x/R", ylabel="L2(field_diff)")
    lines!(ax1, x_targets, band_txx_l2; label="tau_xx", color=:firebrick)
    lines!(ax1, x_targets, band_ux_l2;  label="u_x",    color=:steelblue)
    vlines!(ax1, [-1.0, 1.0]; color=:black, linestyle=:dash, label="cylinder edges")
    axislegend(ax1; position=:rt)
    ax2 = Axis(fig[1, 2]; title="Per-x-band max|diff|",
               xlabel="x/R", ylabel="max|field_diff|")
    lines!(ax2, x_targets, band_txx_mx; label="tau_xx", color=:firebrick)
    lines!(ax2, x_targets, band_ux_mx;  label="u_x",    color=:steelblue)
    vlines!(ax2, [-1.0, 1.0]; color=:black, linestyle=:dash, label="cylinder edges")
    axislegend(ax2; position=:rt)
    save(joinpath(output_dir, "M29_band_diffs.png"), fig)

    # Final summary print so the calling Department can grep
    println("\n=== M29 tau-compare summary ===")
    @printf("Cd_kraken=%.3f Cd_rheo=120.40 dCd=%.3f (%.2f%%)\n",
            kraken.Cd_kraken, kraken.Cd_kraken - 120.40,
            (kraken.Cd_kraken - 120.40) / 120.40 * 100)
    for k in (:ux, :uy, :tau_xx, :tau_xy, :tau_yy)
        m = metrics[k]
        @printf("L2_rel[%s] = %.4f   max|diff|[%s] = %.4g\n",
                String(k), m.l2_rel, String(k), m.max_abs)
    end
    println("Done at $(now())")
end

main()
