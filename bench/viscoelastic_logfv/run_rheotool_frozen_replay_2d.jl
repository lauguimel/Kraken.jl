#!/usr/bin/env julia

# Frozen-flow replay of a rheoTool cylinder field through Kraken's 2D log-FV
# polymer pipeline.  This is a diagnostic harness: it does not update LBM
# populations or add another embedded-geometry production slice.

using Dates
using KernelAbstractions
using LinearAlgebra
using Printf
using Serialization

using Kraken

struct ReplayConfig
    case_dir::String
    foam_time::String
    r_values::Vector{Int}
    x_min::Float64
    x_max::Float64
    y_min::Float64
    y_max::Float64
    cylinder_x::Float64
    cylinder_y::Float64
    radius::Float64
    beta::Float64
    wi::Float64
    u_scale::Float64
    physical_time::Float64
    cfl::Float64
    max_steps::Int
    source_substeps::Int
    samples::Int
    initial::Symbol
    cache_dir::String
    output_dir::String
    prepare_only::Bool
    force_prepare::Bool
end

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

function env_list(::Type{T}, name::AbstractString, default::AbstractString) where {T}
    raw = get(ENV, name, default)
    values = T[]
    for part in split(replace(raw, ';' => ','), ',')
        text = strip(part)
        isempty(text) && continue
        push!(values, parse(T, text))
    end
    isempty(values) && error("$(name) did not contain any values")
    return values
end

function env_bool(name::AbstractString, default::Bool)
    raw = lowercase(strip(get(ENV, name, default ? "1" : "0")))
    raw in ("1", "true", "yes", "on") && return true
    raw in ("0", "false", "no", "off") && return false
    error("$(name) must be boolean-like, got $(raw)")
end

function parse_config()
    case_dir = get(
        ENV,
        "KRAKEN_RHEOTOOL_CASE",
        joinpath("bench", "rheotool", "cylinder_oldroydb_log_re1_wi01"),
    )
    initial = Symbol(lowercase(get(ENV, "KRAKEN_REPLAY_INITIAL", "identity")))
    initial in (:identity, :rheotool) ||
        error("KRAKEN_REPLAY_INITIAL must be identity or rheotool")
    return ReplayConfig(
        case_dir,
        get(ENV, "KRAKEN_RHEOTOOL_TIME", "0.8"),
        env_list(Int, "KRAKEN_R_LIST", "10,20,30"),
        parse(Float64, get(ENV, "KRAKEN_REPLAY_X_MIN", "-5")),
        parse(Float64, get(ENV, "KRAKEN_REPLAY_X_MAX", "15")),
        parse(Float64, get(ENV, "KRAKEN_REPLAY_Y_MIN", "-2")),
        parse(Float64, get(ENV, "KRAKEN_REPLAY_Y_MAX", "2")),
        parse(Float64, get(ENV, "KRAKEN_REPLAY_CYLINDER_X", "0")),
        parse(Float64, get(ENV, "KRAKEN_REPLAY_CYLINDER_Y", "0")),
        parse(Float64, get(ENV, "KRAKEN_REPLAY_RADIUS", "1")),
        parse(Float64, get(ENV, "KRAKEN_BETA", "0.59")),
        parse(Float64, get(ENV, "KRAKEN_WI", "0.1")),
        parse(Float64, get(ENV, "KRAKEN_REPLAY_U_SCALE", "1.0")),
        parse(Float64, get(ENV, "KRAKEN_REPLAY_PHYSICAL_TIME", "2.0")),
        parse(Float64, get(ENV, "KRAKEN_REPLAY_CFL", "0.25")),
        parse(Int, get(ENV, "KRAKEN_REPLAY_MAX_STEPS", "0")),
        parse(Int, get(ENV, "KRAKEN_REPLAY_SOURCE_SUBSTEPS", "1")),
        parse(Int, get(ENV, "KRAKEN_LOGFV_EMBEDDED_CIRCLE_SAMPLES", "32")),
        initial,
        get(ENV, "KRAKEN_REPLAY_CACHE_DIR", joinpath("tmp", "rheotool_frozen_replay", "cache")),
        get(
            ENV,
            "KRAKEN_OUTPUT_DIR",
            joinpath("tmp", "rheotool_frozen_replay", Dates.format(now(), "yyyymmdd_HHMMSS")),
        ),
        env_bool("KRAKEN_REPLAY_PREPARE_ONLY", false),
        env_bool("KRAKEN_REPLAY_FORCE_PREPARE", false),
    )
end

csv_cell(x) = x isa AbstractFloat ? @sprintf("%.16g", x) : string(x)

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
    isfile(plain) && return plain
    isfile(plain * ".gz") && return plain
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
    mode_block = foam_internal_block(read_text_maybe_gzip(path))
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
    mode_block = foam_internal_block(read_text_maybe_gzip(path))
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

function load_foam_snapshot(case_dir::AbstractString, foam_time::AbstractString)
    time_dir = joinpath(case_dir, foam_time)
    isdir(time_dir) || error(
        "RheoTool time directory $(time_dir) does not exist. " *
        "The checked-in case currently has no saved nonzero U/tau fields; " *
        "rerun the case with writeInterval covering $(foam_time), or point " *
        "KRAKEN_RHEOTOOL_TIME at an existing saved time.",
    )
    x, y = cell_centers(case_dir)
    n_cells = length(x)
    U = parse_vol_vector(joinpath(time_dir, "U"), n_cells)
    tau = parse_vol_symmtensor(joinpath(time_dir, "tau"), n_cells)
    ux = [u[1] for u in U]
    uy = [u[2] for u in U]
    tauxx = [t[1] for t in tau]
    tauxy = [t[2] for t in tau]
    tauyy = [t[4] for t in tau]
    return (; x, y, ux, uy, tauxx, tauxy, tauyy, n_cells)
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
        bx = clamp(fld(Int(floor((xs[p] - x_min) / dx)), 1) + 1, 1, nb)
        by = clamp(fld(Int(floor((ys[p] - y_min) / dy)), 1) + 1, 1, nb)
        push!(bins[bx + (by - 1) * nb], p)
    end
    return index
end

function candidate_indices(index::SpatialIndex, xq::Float64, yq::Float64, k::Int)
    bx = clamp(fld(Int(floor((xq - index.x_min) / index.dx)), 1) + 1, 1, index.nx)
    by = clamp(fld(Int(floor((yq - index.y_min) / index.dy)), 1) + 1, 1, index.ny)
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

function target_grid(cfg::ReplayConfig, R::Int)
    Nx = round(Int, (cfg.x_max - cfg.x_min) * R)
    Ny = round(Int, (cfg.y_max - cfg.y_min) * R)
    dx = (cfg.x_max - cfg.x_min) / Nx
    dy = (cfg.y_max - cfg.y_min) / Ny
    x = [cfg.x_min + (i - 0.5) * dx for i in 1:Nx]
    y = [cfg.y_min + (j - 0.5) * dy for j in 1:Ny]
    return (; Nx, Ny, dx, dy, x, y)
end

function resample_snapshot(snapshot, cfg::ReplayConfig, R::Int)
    grid = target_grid(cfg, R)
    index = build_index(snapshot.x, snapshot.y)
    fields = Dict{Symbol,Matrix{Float64}}(
        :ux => zeros(grid.Nx, grid.Ny),
        :uy => zeros(grid.Nx, grid.Ny),
        :tauxx_ref => zeros(grid.Nx, grid.Ny),
        :tauxy_ref => zeros(grid.Nx, grid.Ny),
        :tauyy_ref => zeros(grid.Nx, grid.Ny),
    )
    solid = falses(grid.Nx, grid.Ny)
    for j in 1:grid.Ny, i in 1:grid.Nx
        xq = grid.x[i]
        yq = grid.y[j]
        solid[i, j] = hypot(xq - cfg.cylinder_x, yq - cfg.cylinder_y) <= cfg.radius
        if !solid[i, j]
            fields[:ux][i, j] = cfg.u_scale * affine_sample(index, snapshot.ux, xq, yq)
            fields[:uy][i, j] = cfg.u_scale * affine_sample(index, snapshot.uy, xq, yq)
            fields[:tauxx_ref][i, j] = affine_sample(index, snapshot.tauxx, xq, yq)
            fields[:tauxy_ref][i, j] = affine_sample(index, snapshot.tauxy, xq, yq)
            fields[:tauyy_ref][i, j] = affine_sample(index, snapshot.tauyy, xq, yq)
        end
    end
    return (; R, grid..., solid, fields...)
end

function cache_path(cfg::ReplayConfig, R::Int)
    safe_time = replace(cfg.foam_time, '/' => '_')
    return joinpath(cfg.cache_dir, "rheotool_replay_R$(R)_t$(safe_time).jls")
end

function prepare_or_load(cfg::ReplayConfig, R::Int)
    path = cache_path(cfg, R)
    if !cfg.force_prepare && isfile(path)
        return deserialize(path)
    end
    snapshot = load_foam_snapshot(cfg.case_dir, cfg.foam_time)
    replay_input = resample_snapshot(snapshot, cfg, R)
    mkpath(dirname(path))
    serialize(path, replay_input)
    return replay_input
end

function boundary_bc(field::AbstractMatrix)
    Nx, Ny = size(field)
    return Kraken.FVFDFieldBC2D(
        copy(@view field[1, :]),
        copy(@view field[Nx, :]),
        copy(@view field[:, 1]),
        copy(@view field[:, Ny]),
    )
end

function psi_from_tau(tauxx, tauxy, tauyy, prefactor, solid)
    Nx, Ny = size(tauxx)
    psixx = zeros(Float64, Nx, Ny)
    psixy = zeros(Float64, Nx, Ny)
    psiyy = zeros(Float64, Nx, Ny)
    bad = 0
    for j in 1:Ny, i in 1:Nx
        solid[i, j] && continue
        cxx = 1.0 + tauxx[i, j] / prefactor
        cxy = tauxy[i, j] / prefactor
        cyy = 1.0 + tauyy[i, j] / prefactor
        if Kraken.logfv_min_eig_sym2_2d(cxx, cxy, cyy) <= 0
            bad += 1
            continue
        end
        psixx[i, j], psixy[i, j], psiyy[i, j] = Kraken.logfv_log_spd_sym2_2d(cxx, cxy, cyy)
    end
    return psixx, psixy, psiyy, bad
end

function replay_logfv(input, cfg::ReplayConfig)
    backend = KernelAbstractions.CPU()
    T = Float64
    Nx, Ny = input.Nx, input.Ny
    dx, dy = input.dx, input.dy
    prefactor = (1.0 - cfg.beta) / cfg.wi
    lambda = cfg.wi
    max_speed = maximum(hypot.(input.ux[.!input.solid], input.uy[.!input.solid]))
    dt_cfl = max_speed > 0 ? cfg.cfl * min(dx, dy) / max_speed : cfg.physical_time
    dt = min(cfg.physical_time, dt_cfl)
    nsteps = cfg.max_steps > 0 ? cfg.max_steps :
        (cfg.physical_time <= 0 || dt <= 0 ? 0 : ceil(Int, cfg.physical_time / dt))
    nsteps > 0 && (dt = cfg.physical_time / nsteps)
    source_substeps = max(1, cfg.source_substeps)
    dt_source = dt / source_substeps

    cx_idx = (cfg.cylinder_x - cfg.x_min) / dx
    cy_idx = (cfg.cylinder_y - cfg.y_min) / dy
    radius_idx = cfg.radius / dx
    bc = Kraken.FVFDDomainBC2D(west=:open, east=:open, south=:wall, north=:wall)
    geometry_h = Kraken.fvfd_geometry_from_circle_2d(
        Nx, Ny, dx, dy, bc, cx_idx, cy_idx, radius_idx;
        FT=T, samples=cfg.samples,
    )
    geometry = Kraken.fvfd_transfer_geometry_2d(geometry_h, backend, T)

    ux = KernelAbstractions.allocate(backend, T, Nx, Ny)
    uy = KernelAbstractions.allocate(backend, T, Nx, Ny)
    copyto!(ux, input.ux)
    copyto!(uy, input.uy)
    fill_solid_velocity!(ux, uy, geometry_h.is_solid)

    psixx0, psixy0, psiyy0, bad_ref_spd = if cfg.initial === :rheotool
        psi_from_tau(input.tauxx_ref, input.tauxy_ref, input.tauyy_ref, prefactor, geometry_h.is_solid)
    else
        (zeros(T, Nx, Ny), zeros(T, Nx, Ny), zeros(T, Nx, Ny), 0)
    end

    psixx = KernelAbstractions.allocate(backend, T, Nx, Ny)
    psixy = KernelAbstractions.allocate(backend, T, Nx, Ny)
    psiyy = KernelAbstractions.allocate(backend, T, Nx, Ny)
    copyto!(psixx, psixx0)
    copyto!(psixy, psixy0)
    copyto!(psiyy, psiyy0)

    psixx_adv = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixy_adv = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psiyy_adv = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixx_next = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixy_next = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psiyy_next = KernelAbstractions.zeros(backend, T, Nx, Ny)
    ux_face = KernelAbstractions.zeros(backend, T, Nx + 1, Ny)
    uy_face = KernelAbstractions.zeros(backend, T, Nx, Ny + 1)
    dudx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    dudy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    dvdx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    dvdy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tauxx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tauxy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tauyy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    fx_poly = KernelAbstractions.zeros(backend, T, Nx, Ny)
    fy_poly = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    ty = KernelAbstractions.zeros(backend, T, Nx, Ny)

    ux_bc = Kraken.fvfd_transfer_field_bc_2d(boundary_bc(input.ux), backend, T, Nx, Ny, bc; name=:ux_bc)
    uy_bc = Kraken.fvfd_transfer_field_bc_2d(boundary_bc(input.uy), backend, T, Nx, Ny, bc; name=:uy_bc)
    psixx_bc_h, psixy_bc_h, psiyy_bc_h, _ = psi_from_tau(
        input.tauxx_ref, input.tauxy_ref, input.tauyy_ref, prefactor, geometry_h.is_solid,
    )
    psixx_bc = Kraken.fvfd_transfer_field_bc_2d(boundary_bc(psixx_bc_h), backend, T, Nx, Ny, bc; name=:psixx_bc)
    psixy_bc = Kraken.fvfd_transfer_field_bc_2d(boundary_bc(psixy_bc_h), backend, T, Nx, Ny, bc; name=:psixy_bc)
    psiyy_bc = Kraken.fvfd_transfer_field_bc_2d(boundary_bc(psiyy_bc_h), backend, T, Nx, Ny, bc; name=:psiyy_bc)

    Kraken.fvfd_velocity_gradient_embedded_2d!(dudx, dudy, dvdx, dvdy, ux, uy, geometry; sync=false)
    for step in 1:nsteps
        Kraken.logfv_advect_upwind_embedded_2d!(
            psixx_adv, psixy_adv, psiyy_adv,
            psixx, psixy, psiyy,
            psixx_bc, psixy_bc, psiyy_bc,
            ux_face, uy_face, ux, uy,
            geometry, ux_bc, uy_bc, dt; sync=false,
        )
        psixx_work, psixy_work, psiyy_work = psixx_adv, psixy_adv, psiyy_adv
        for _ in 1:source_substeps
            Kraken.logfv_step_constitutive_log_2d!(
                psixx_next, psixy_next, psiyy_next,
                psixx_work, psixy_work, psiyy_work,
                dudx, dudy, dvdx, dvdy,
                lambda, dt_source, Kraken.LOGFV_MODEL_OLDROYDB, 0.0; sync=false,
            )
            psixx_work, psixx_next = psixx_next, psixx_work
            psixy_work, psixy_next = psixy_next, psixy_work
            psiyy_work, psiyy_next = psiyy_next, psiyy_work
        end
        psixx, psixx_adv = psixx_work, psixx
        psixy, psixy_adv = psixy_work, psixy
        psiyy, psiyy_adv = psiyy_work, psiyy
        step % 50 == 0 && @printf("  R=%d replay step %d/%d\n", input.R, step, nsteps)
    end

    Kraken.logfv_stress_from_log_2d!(tauxx, tauxy, tauyy, psixx, psixy, psiyy, prefactor; sync=false)
    Kraken.fvfd_tensor_divergence_embedded_2d!(fx_poly, fy_poly, tauxx, tauxy, tauyy, geometry; sync=false)
    Kraken.fvfd_embedded_wall_traction_2d!(tx, ty, tauxx, tauxy, tauyy, geometry; sync=true)

    return (;
        R=input.R,
        Nx,
        Ny,
        dx,
        dy,
        dt,
        nsteps,
        source_substeps,
        physical_time=cfg.physical_time,
        prefactor,
        lambda,
        max_speed,
        bad_ref_spd,
        geometry=geometry_h,
        ux=Array(ux),
        uy=Array(uy),
        dudx=Array(dudx),
        dudy=Array(dudy),
        dvdx=Array(dvdx),
        dvdy=Array(dvdy),
        psixx=Array(psixx),
        psixy=Array(psixy),
        psiyy=Array(psiyy),
        tauxx=Array(tauxx),
        tauxy=Array(tauxy),
        tauyy=Array(tauyy),
        fx_poly=Array(fx_poly),
        fy_poly=Array(fy_poly),
        wall_tx=Array(tx),
        wall_ty=Array(ty),
        Fx_poly_drag=dx * sum(Array(tx)),
        Fy_poly_drag=dy * sum(Array(ty)),
        input,
    )
end

function fill_solid_velocity!(ux, uy, is_solid)
    ux_h = Array(ux)
    uy_h = Array(uy)
    ux_h[is_solid] .= 0.0
    uy_h[is_solid] .= 0.0
    copyto!(ux, ux_h)
    copyto!(uy, uy_h)
    return nothing
end

function rel_metrics(field, ref, solid)
    mask = (.!solid) .& isfinite.(field) .& isfinite.(ref)
    if !any(mask)
        return (; linf=NaN, rel_linf=NaN, rel_l2=NaN)
    end
    diff = field[mask] .- ref[mask]
    refv = ref[mask]
    linf = maximum(abs.(diff))
    denom_inf = max(maximum(abs.(refv)), eps(Float64))
    denom_l2 = max(norm(refv), eps(Float64))
    return (; linf, rel_linf=linf / denom_inf, rel_l2=norm(diff) / denom_l2)
end

function summarize(result)
    solid = result.geometry.is_solid
    input = result.input
    mx = rel_metrics(result.tauxx, input.tauxx_ref, solid)
    mxy = rel_metrics(result.tauxy, input.tauxy_ref, solid)
    my = rel_metrics(result.tauyy, input.tauyy_ref, solid)
    cd_poly = result.Fx_poly_drag / (0.5 * 1.0^2 * 2.0)
    cd_ref = embedded_drag_from_ref(input, result.geometry)
    return (;
        R=result.R,
        Nx=result.Nx,
        Ny=result.Ny,
        dx=result.dx,
        dt=result.dt,
        nsteps=result.nsteps,
        max_speed=result.max_speed,
        bad_ref_spd=result.bad_ref_spd,
        cut_cells=count(result.geometry.embedded.cut_count .> 0),
        wall_length=result.dx * sum(result.geometry.embedded.wall_fraction),
        tau_xx_rel_linf=mx.rel_linf,
        tau_xx_rel_l2=mx.rel_l2,
        tau_xy_rel_linf=mxy.rel_linf,
        tau_xy_rel_l2=mxy.rel_l2,
        tau_yy_rel_linf=my.rel_linf,
        tau_yy_rel_l2=my.rel_l2,
        Fx_poly_drag=result.Fx_poly_drag,
        Cd_polymer=cd_poly,
        Fx_poly_drag_ref=cd_ref.Fx,
        Cd_polymer_ref=cd_ref.Fx / (0.5 * 1.0^2 * 2.0),
        Cd_polymer_error=cd_poly - cd_ref.Fx / (0.5 * 1.0^2 * 2.0),
    )
end

function embedded_drag_from_ref(input, geometry)
    backend = KernelAbstractions.CPU()
    Nx, Ny = input.Nx, input.Ny
    tx = KernelAbstractions.zeros(backend, Float64, Nx, Ny)
    ty = KernelAbstractions.zeros(backend, Float64, Nx, Ny)
    tauxx = KernelAbstractions.allocate(backend, Float64, Nx, Ny)
    tauxy = KernelAbstractions.allocate(backend, Float64, Nx, Ny)
    tauyy = KernelAbstractions.allocate(backend, Float64, Nx, Ny)
    copyto!(tauxx, input.tauxx_ref)
    copyto!(tauxy, input.tauxy_ref)
    copyto!(tauyy, input.tauyy_ref)
    Kraken.fvfd_embedded_wall_traction_2d!(tx, ty, tauxx, tauxy, tauyy, geometry; sync=true)
    return (; Fx=input.dx * sum(Array(tx)), Fy=input.dy * sum(Array(ty)))
end

function write_summary_csv(path, rows)
    cols = (
        :R, :Nx, :Ny, :dx, :dt, :nsteps, :max_speed, :bad_ref_spd,
        :cut_cells, :wall_length,
        :tau_xx_rel_linf, :tau_xx_rel_l2,
        :tau_xy_rel_linf, :tau_xy_rel_l2,
        :tau_yy_rel_linf, :tau_yy_rel_l2,
        :Fx_poly_drag, :Cd_polymer,
        :Fx_poly_drag_ref, :Cd_polymer_ref, :Cd_polymer_error,
    )
    open(path, "w") do io
        println(io, join(string.(cols), ","))
        for row in rows
            println(io, join((csv_cell(getproperty(row, c)) for c in cols), ","))
        end
    end
end

function write_profile_csv(path, result)
    input = result.input
    i_near = argmin(abs.(input.x .- (1.0 + input.dx)))
    i_wake = argmin(abs.(input.x .- 3.0))
    open(path, "w") do io
        println(io, "slice,x,y,ux,uy,tauxx,tauxy,tauyy,tauxx_ref,tauxy_ref,tauyy_ref,fx_poly,fy_poly")
        for (label, i) in (("near_wall", i_near), ("wake", i_wake))
            for j in 1:input.Ny
                vals = (
                    label, input.x[i], input.y[j],
                    result.ux[i, j], result.uy[i, j],
                    result.tauxx[i, j], result.tauxy[i, j], result.tauyy[i, j],
                    input.tauxx_ref[i, j], input.tauxy_ref[i, j], input.tauyy_ref[i, j],
                    result.fx_poly[i, j], result.fy_poly[i, j],
                )
                println(io, join(csv_cell.(vals), ","))
            end
        end
    end
end

function color_map(v, vmin, vmax)
    if !isfinite(v)
        return "#555555"
    end
    t = vmax <= vmin ? 0.5 : clamp((v - vmin) / (vmax - vmin), 0.0, 1.0)
    r = round(Int, 255 * t)
    b = round(Int, 255 * (1 - t))
    g = round(Int, 90 * (1 - abs(2t - 1)))
    return @sprintf("#%02x%02x%02x", r, g, b)
end

function svg_heatmap(field, solid; width=280, height=150)
    Nx, Ny = size(field)
    sx = max(1, ceil(Int, Nx / 140))
    sy = max(1, ceil(Int, Ny / 72))
    vals = field[(.!solid) .& isfinite.(field)]
    vmin, vmax = isempty(vals) ? (0.0, 1.0) : extrema(vals)
    rects = IOBuffer()
    for j0 in 1:sy:Ny, i0 in 1:sx:Nx
        i1 = min(Nx, i0 + sx - 1)
        j1 = min(Ny, j0 + sy - 1)
        block = @view field[i0:i1, j0:j1]
        solid_block = @view solid[i0:i1, j0:j1]
        finite_values = block[isfinite.(block)]
        mean_value = isempty(finite_values) ? NaN : sum(finite_values) / length(finite_values)
        color = all(solid_block) ? "#111111" : color_map(mean_value, vmin, vmax)
        x = width * (i0 - 1) / Nx
        y = height * (Ny - j1) / Ny
        w = width * (i1 - i0 + 1) / Nx
        h = height * (j1 - j0 + 1) / Ny
        @printf(rects, "<rect x='%.2f' y='%.2f' width='%.2f' height='%.2f' fill='%s'/>", x, y, w, h, color)
    end
    return String(take!(rects)), vmin, vmax
end

function write_dashboard(path, rows, results)
    open(path, "w") do io
        println(io, "<!doctype html><meta charset='utf-8'><title>RheoTool Frozen Replay</title>")
        println(io, "<style>body{font-family:system-ui,sans-serif;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:4px 8px;text-align:right}td:first-child,th:first-child{text-align:left}.grid{display:grid;grid-template-columns:repeat(3,minmax(240px,1fr));gap:14px;margin:12px 0 28px}.panel{border:1px solid #ccc;padding:8px}.panel h4{margin:0 0 6px;font-size:14px}svg{width:100%;height:auto;background:#111}</style>")
        println(io, "<h1>RheoTool frozen-flow replay</h1>")
        println(io, "<table><tr><th>R</th><th>Nx</th><th>Ny</th><th>steps</th><th>tau_xx rel L∞</th><th>tau_xy rel L∞</th><th>tau_yy rel L∞</th><th>Cd polymer</th><th>Cd ref traction</th><th>Cd error</th></tr>")
        for r in rows
            @printf(io, "<tr><td>%d</td><td>%d</td><td>%d</td><td>%d</td><td>%.4e</td><td>%.4e</td><td>%.4e</td><td>%.8g</td><td>%.8g</td><td>%.4e</td></tr>\n",
                r.R, r.Nx, r.Ny, r.nsteps, r.tau_xx_rel_linf, r.tau_xy_rel_linf,
                r.tau_yy_rel_linf, r.Cd_polymer, r.Cd_polymer_ref, r.Cd_polymer_error)
        end
        println(io, "</table>")
        for result in results
            solid = result.geometry.is_solid
            println(io, "<h2>R=$(result.R)</h2>")
            println(io, "<div class='grid'>")
            for (label, field) in (
                ("Kraken tau_xx", result.tauxx),
                ("RheoTool tau_xx", result.input.tauxx_ref),
                ("relative error tau_xx", abs.(result.tauxx .- result.input.tauxx_ref) ./ max.(abs.(result.input.tauxx_ref), eps(Float64))),
                ("Kraken tau_xy", result.tauxy),
                ("RheoTool tau_xy", result.input.tauxy_ref),
                ("relative error tau_xy", abs.(result.tauxy .- result.input.tauxy_ref) ./ max.(abs.(result.input.tauxy_ref), eps(Float64))),
                ("Kraken tau_yy", result.tauyy),
                ("RheoTool tau_yy", result.input.tauyy_ref),
                ("relative error tau_yy", abs.(result.tauyy .- result.input.tauyy_ref) ./ max.(abs.(result.input.tauyy_ref), eps(Float64))),
            )
                rects, vmin, vmax = svg_heatmap(field, solid)
                @printf(io, "<div class='panel'><h4>%s [%.3g, %.3g]</h4><svg viewBox='0 0 280 150'>%s</svg></div>\n",
                    label, vmin, vmax, rects)
            end
            println(io, "</div>")
        end
    end
end

function write_verdict(path, cfg, rows)
    open(path, "w") do io
        println(io, "# RheoTool Frozen Replay — $(Dates.format(today(), "yyyy-mm-dd"))")
        println(io)
        println(io, "- case: `$(cfg.case_dir)`")
        println(io, "- time: `$(cfg.foam_time)`")
        println(io, "- initial Psi: `$(cfg.initial)`")
        println(io, "- replay physical time: `$(cfg.physical_time)`")
        println(io)
        println(io, "## Metrics")
        println(io)
        println(io, "| R | tau_xx rel L∞ | tau_xy rel L∞ | tau_yy rel L∞ | Cd polymer | Cd ref traction | Cd error |")
        println(io, "|---:|---:|---:|---:|---:|---:|---:|")
        for r in rows
            @printf(io, "| %d | %.4e | %.4e | %.4e | %.8g | %.8g | %.4e |\n",
                r.R, r.tau_xx_rel_linf, r.tau_xy_rel_linf, r.tau_yy_rel_linf,
                r.Cd_polymer, r.Cd_polymer_ref, r.Cd_polymer_error)
        end
        println(io)
        println(io, "## Verdict")
        println(io)
        if isempty(rows)
            println(io, "No replay rows were produced.")
        elseif all(r.tau_xx_rel_linf < 0.01 && r.tau_xy_rel_linf < 0.01 &&
                   r.tau_yy_rel_linf < 0.01 for r in rows)
            if any(abs(r.Cd_polymer_error) > 0.01 * max(abs(r.Cd_polymer_ref), eps()) for r in rows)
                println(io, "Tau fields match within 1%, but polymer drag does not. This rules in the curved-wall traction integration branch of the decision tree.")
            else
                println(io, "Tau fields and polymer drag match within the 1% threshold. The next target is LBM solvent or tau-to-momentum coupling.")
            end
        else
            println(io, "Tau fields do not match within 1%. The next target is the polymer CDE pipeline on cut cells, starting with advection/source/wall-closure toggles under frozen U.")
        end
    end
end

function main()
    cfg = parse_config()
    mkpath(cfg.output_dir)
    @printf("RheoTool frozen replay case=%s time=%s R=%s output=%s\n",
        cfg.case_dir, cfg.foam_time, join(cfg.r_values, ","), cfg.output_dir)
    rows = NamedTuple[]
    results = NamedTuple[]
    for R in cfg.r_values
        @printf("Preparing R=%d ...\n", R)
        input = prepare_or_load(cfg, R)
        @printf("  grid Nx=%d Ny=%d dx=%.6g\n", input.Nx, input.Ny, input.dx)
        cfg.prepare_only && continue
        result = replay_logfv(input, cfg)
        row = summarize(result)
        push!(rows, row)
        push!(results, result)
        serialize(joinpath(cfg.output_dir, "fields_R$(R).jls"), result)
        write_profile_csv(joinpath(cfg.output_dir, "profiles_R$(R).csv"), result)
        @printf("  R=%d tau_xy_rel_linf=%.4e Cd=%.8g Cd_ref=%.8g\n",
            R, row.tau_xy_rel_linf, row.Cd_polymer, row.Cd_polymer_ref)
    end
    cfg.prepare_only && return nothing
    write_summary_csv(joinpath(cfg.output_dir, "summary.csv"), rows)
    write_dashboard(joinpath(cfg.output_dir, "dashboard.html"), rows, results)
    write_verdict(
        joinpath(cfg.output_dir, "RHEOTOOL_FROZEN_REPLAY_$(Dates.format(today(), "yyyymmdd")).md"),
        cfg,
        rows,
    )
    println("Summary: ", joinpath(cfg.output_dir, "summary.csv"))
    println("Dashboard: ", joinpath(cfg.output_dir, "dashboard.html"))
end

main()
