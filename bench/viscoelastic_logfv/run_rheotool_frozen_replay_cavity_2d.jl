#!/usr/bin/env julia

# Closed-cavity frozen-flow replay of a rheoTool field through Kraken's 2D
# log-FV polymer pipeline. This diagnostic freezes U and advances only the
# polymer advection, constitutive source, and stress reconstruction.

using Dates
using DelimitedFiles
using KernelAbstractions
using LinearAlgebra
using Printf
using Serialization

using Kraken

const DEFAULT_RHEOTOOL_CASE = joinpath("bench", "rheotool", "cavity_oldroydb_log_re001_de1_b05")
const DEFAULT_OUTPUT_ROOT = joinpath("tmp", "cavity_frozen_replay")

const CUDA_MOD = try
    @eval using CUDA
    getfield(Main, :CUDA)
catch
    nothing
end

const METAL_MOD = if Sys.isapple()
    try
        @eval using Metal
        getfield(Main, :Metal)
    catch
        nothing
    end
else
    nothing
end

struct ReplayConfig
    N::Int
    replay_physical_time::Float64
    initial::Symbol
    case_dir::String
    output_dir::String
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

csv_cell(x) = x isa AbstractFloat ? @sprintf("%.16g", x) : string(x)

function parse_args(args)
    mode_seen = false
    N = 32
    replay_time = 0.25
    initial = :rheotool
    for arg in args
        if arg == "--self-test"
            N = 32
            replay_time = 0.25
            mode_seen = true
        elseif arg == "--full"
            N = 64
            replay_time = 1.0
            mode_seen = true
        elseif startswith(arg, "--initial=")
            raw = lowercase(last(split(arg, "=", limit=2)))
            raw in ("rheotool", "identity") ||
                error("--initial must be rheotool or identity, got $(raw)")
            initial = Symbol(raw)
        else
            error("unrecognised argument: $(arg)")
        end
    end
    if !mode_seen
        N = 32
        replay_time = 0.25
    end
    stamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    case_dir = get(ENV, "KRAKEN_RHEOTOOL_CASE", DEFAULT_RHEOTOOL_CASE)
    output_dir = get(ENV, "KRAKEN_OUTPUT_DIR", joinpath(DEFAULT_OUTPUT_ROOT, stamp))
    return ReplayConfig(N, replay_time, initial, case_dir, output_dir)
end

function pick_backend()
    requested = lowercase(get(ENV, "KRAKEN_BACKEND", "auto"))
    if requested in ("metal", "mtl") && METAL_MOD !== nothing
        try
            return :metal, METAL_MOD.MetalBackend(), Float32
        catch err
            @warn "Metal backend requested but unavailable; falling back to CPU" exception=(err, catch_backtrace())
            return :cpu, KernelAbstractions.CPU(), Float64
        end
    elseif requested in ("cuda", "gpu") && CUDA_MOD !== nothing
        return :cuda, CUDA_MOD.CUDABackend(), Float64
    elseif requested == "cpu"
        return :cpu, KernelAbstractions.CPU(), Float64
    end
    if METAL_MOD !== nothing && Sys.isapple()
        try
            return :metal, METAL_MOD.MetalBackend(), Float32
        catch
        end
    elseif CUDA_MOD !== nothing && CUDA_MOD.functional()
        return :cuda, CUDA_MOD.CUDABackend(), Float64
    end
    return :cpu, KernelAbstractions.CPU(), Float64
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
    text = read_text_maybe_gzip(foam_path(case_dir, "constant", "polyMesh", "faces"))
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

function pick_rheotool_sample_dir(case_dir::AbstractString, target_t::Float64)
    sample_root = joinpath(case_dir, "postProcessing", "sampleDict")
    isdir(sample_root) || error("missing $(sample_root); did rheoTool run?")
    dirs = filter(isdir, [joinpath(sample_root, d) for d in readdir(sample_root)])
    candidates = Tuple{Float64,String}[]
    for d in dirs
        t = tryparse(Float64, basename(d))
        t === nothing && continue
        t > 0 || continue
        push!(candidates, (t, d))
    end
    isempty(candidates) && error("no non-zero sample times found in $(sample_root)")
    _, best = findmin(t -> abs(t[1] - target_t), candidates)
    return candidates[best]
end

function read_rheotool_horizontal_tautheta(path::AbstractString)
    raw = readdlm(path)
    x = Vector{Float64}(raw[:, 1])
    tau_xx = Vector{Float64}(raw[:, 2])
    tau_xy = Vector{Float64}(raw[:, 3])
    tau_yy = Vector{Float64}(raw[:, 5])
    theta_xx = Vector{Float64}(raw[:, 8])
    theta_xy = Vector{Float64}(raw[:, 9])
    theta_yy = Vector{Float64}(raw[:, 11])
    return (;
        x,
        tau_xx,
        tau_xy,
        tau_yy,
        theta_xx,
        theta_xy,
        theta_yy,
    )
end

function load_foam_snapshot(case_dir::AbstractString, time_name::AbstractString)
    time_dir = joinpath(case_dir, time_name)
    isdir(time_dir) || error("RheoTool time directory $(time_dir) does not exist")
    x, y = cell_centers(case_dir)
    n_cells = length(x)
    U = parse_vol_vector(joinpath(time_dir, "U"), n_cells)
    theta = parse_vol_symmtensor(joinpath(time_dir, "theta"), n_cells)
    tau = isfile(joinpath(time_dir, "tau")) || isfile(joinpath(time_dir, "tau.gz")) ?
        parse_vol_symmtensor(joinpath(time_dir, "tau"), n_cells) :
        fill((0.0, 0.0, 0.0, 0.0, 0.0, 0.0), n_cells)
    return (;
        x,
        y,
        ux=[u[1] for u in U],
        uy=[u[2] for u in U],
        theta_xx=[p[1] for p in theta],
        theta_xy=[p[2] for p in theta],
        theta_yy=[p[4] for p in theta],
        tau_xx=[p[1] for p in tau],
        tau_xy=[p[2] for p in tau],
        tau_yy=[p[4] for p in tau],
        n_cells,
    )
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

function resample_snapshot(snapshot, N::Int)
    Nx = N
    Ny = N
    dx_phys = 1.0 / Nx
    dy_phys = 1.0 / Ny
    x_phys = [(i - 0.5) / Nx for i in 1:Nx]
    y_phys = [(j - 0.5) / Ny for j in 1:Ny]
    index = build_index(snapshot.x, snapshot.y)
    fields = Dict{Symbol,Matrix{Float64}}(
        :ux => zeros(Nx, Ny),
        :uy => zeros(Nx, Ny),
        :theta_xx => zeros(Nx, Ny),
        :theta_xy => zeros(Nx, Ny),
        :theta_yy => zeros(Nx, Ny),
        :tau_xx_ref => zeros(Nx, Ny),
        :tau_xy_ref => zeros(Nx, Ny),
        :tau_yy_ref => zeros(Nx, Ny),
    )
    for j in 1:Ny, i in 1:Nx
        xq = x_phys[i]
        yq = y_phys[j]
        fields[:ux][i, j] = affine_sample(index, snapshot.ux, xq, yq)
        fields[:uy][i, j] = affine_sample(index, snapshot.uy, xq, yq)
        fields[:theta_xx][i, j] = affine_sample(index, snapshot.theta_xx, xq, yq)
        fields[:theta_xy][i, j] = affine_sample(index, snapshot.theta_xy, xq, yq)
        fields[:theta_yy][i, j] = affine_sample(index, snapshot.theta_yy, xq, yq)
        fields[:tau_xx_ref][i, j] = affine_sample(index, snapshot.tau_xx, xq, yq)
        fields[:tau_xy_ref][i, j] = affine_sample(index, snapshot.tau_xy, xq, yq)
        fields[:tau_yy_ref][i, j] = affine_sample(index, snapshot.tau_yy, xq, yq)
    end
    for (name, field) in fields
        all(isfinite, field) || error("interpolated $(name) contains non-finite values")
    end
    return (; Nx, Ny, dx_phys, dy_phys, x_phys, y_phys, solid=fill(false, Nx, Ny), fields...)
end

function sample_horizontal_kraken(
    field::AbstractMatrix, Nx::Integer, Ny::Integer, y_frac::Real,
    x_target::AbstractVector,
)
    j_real = y_frac * Ny + 0.5
    j_lo = clamp(floor(Int, j_real), 1, Ny - 1)
    j_hi = j_lo + 1
    wy = clamp(j_real - j_lo, 0.0, 1.0)
    row = [(1 - wy) * field[i, j_lo] + wy * field[i, j_hi] for i in 1:Nx]
    x_phys = [(i - 0.5) / Nx for i in 1:Nx]
    out = similar(x_target, Float64)
    for (k, x) in enumerate(x_target)
        if x <= x_phys[1]
            out[k] = row[1]
        elseif x >= x_phys[end]
            out[k] = row[end]
        else
            i_lo = searchsortedlast(x_phys, x)
            i_hi = i_lo + 1
            wx = (x - x_phys[i_lo]) / (x_phys[i_hi] - x_phys[i_lo])
            out[k] = (1 - wx) * row[i_lo] + wx * row[i_hi]
        end
    end
    return (x=collect(x_target), values=out)
end

function rel_l2_error(kraken::AbstractVector, ref::AbstractVector)
    @assert length(kraken) == length(ref)
    denom = sqrt(sum(abs2, ref))
    denom < eps(Float64) && return NaN
    return sqrt(sum(abs2.(kraken .- ref))) / denom
end

function rel_linf_error(kraken::AbstractVector, ref::AbstractVector)
    @assert length(kraken) == length(ref)
    scale = maximum(abs, ref)
    scale < eps(Float64) && return NaN
    return maximum(abs.(kraken .- ref)) / scale
end

function run_frozen_replay(input, cfg::ReplayConfig; backend, T)
    Nx = input.Nx
    Ny = input.Ny
    u_max = 1.0
    lambda_phys = 1.0
    nu_p = 0.1
    dt_phys = u_max / Nx
    nsteps = max(1, ceil(Int, cfg.replay_physical_time / dt_phys))
    lambda_lu = T(lambda_phys) / T(dt_phys)
    selected_polymer_substeps = 8
    dt_poly = one(T) / T(selected_polymer_substeps)
    prefactor = T(nu_p) / lambda_lu
    dx_lu = one(T)
    dy_lu = one(T)

    is_solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny)
    copyto!(is_solid, input.solid)
    ux = KernelAbstractions.allocate(backend, T, Nx, Ny)
    uy = KernelAbstractions.allocate(backend, T, Nx, Ny)
    copyto!(ux, T.(input.ux))
    copyto!(uy, T.(input.uy))

    psixx0 = cfg.initial === :rheotool ? T.(input.theta_xx) : zeros(T, Nx, Ny)
    psixy0 = cfg.initial === :rheotool ? T.(input.theta_xy) : zeros(T, Nx, Ny)
    psiyy0 = cfg.initial === :rheotool ? T.(input.theta_yy) : zeros(T, Nx, Ny)
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
    dummy_y = KernelAbstractions.zeros(backend, T, Ny)
    dummy_x = KernelAbstractions.zeros(backend, T, Nx)
    logfv_bc = Kraken.logfv_wallxwally_bcspec_2d()

    t_start = time()
    for step in 1:nsteps
        Kraken.logfv_cell_velocity_to_faces_bc_aware_2d!(
            ux_face, uy_face, ux, uy, is_solid,
            dummy_y, dummy_y, dummy_x, dummy_x, logfv_bc; sync=false,
        )
        Kraken.logfv_advect_upwind_bc_aware_2d!(
            psixx_adv, psixy_adv, psiyy_adv,
            psixx, psixy, psiyy,
            dummy_y, dummy_y, dummy_y, dummy_y, dummy_y, dummy_y,
            dummy_x, dummy_x, dummy_x, dummy_x, dummy_x, dummy_x,
            ux_face, uy_face, is_solid, dx_lu, dy_lu, logfv_bc, one(T); sync=false,
        )
        Kraken.fvfd_velocity_gradient_2d!(
            dudx, dudy, dvdx, dvdy, ux, uy, is_solid, dx_lu, dy_lu, logfv_bc; sync=false,
        )
        psixx_work, psixy_work, psiyy_work = psixx_adv, psixy_adv, psiyy_adv
        for _ in 1:selected_polymer_substeps
            Kraken.logfv_step_constitutive_log_2d!(
                psixx_next, psixy_next, psiyy_next,
                psixx_work, psixy_work, psiyy_work,
                dudx, dudy, dvdx, dvdy,
                lambda_lu, dt_poly, Kraken.LOGFV_MODEL_OLDROYDB, T(0.0); sync=false,
            )
            psixx_work, psixx_next = psixx_next, psixx_work
            psixy_work, psixy_next = psixy_next, psixy_work
            psiyy_work, psiyy_next = psiyy_next, psiyy_work
        end
        psixx, psixx_adv = psixx_work, psixx
        psixy, psixy_adv = psixy_work, psixy
        psiyy, psiyy_adv = psiyy_work, psiyy
        Kraken.logfv_stress_from_log_2d!(
            tauxx, tauxy, tauyy, psixx, psixy, psiyy, prefactor;
            model_code=Kraken.LOGFV_MODEL_OLDROYDB, L2=T(0.0), sync=false,
        )
        if step == 1 || step == nsteps || step % 10 == 0
            @printf("  replay step %d/%d\n", step, nsteps)
        end
    end
    KernelAbstractions.synchronize(backend)
    wallclock_s = time() - t_start

    return (;
        Nx,
        Ny,
        dt_phys,
        nsteps,
        lambda_lu=Float64(lambda_lu),
        prefactor=Float64(prefactor),
        selected_polymer_substeps,
        initial=cfg.initial,
        replay_physical_time=cfg.replay_physical_time,
        wallclock_s,
        ux=Array(ux),
        uy=Array(uy),
        psixx=Array(psixx),
        psixy=Array(psixy),
        psiyy=Array(psiyy),
        tauxx=Array(tauxx),
        tauxy=Array(tauxy),
        tauyy=Array(tauyy),
    )
end

function write_outputs(cfg, result, ref_tt, rel_l2_psixy, rel_linf_psixy)
    mkpath(cfg.output_dir)
    Nx, Ny = result.Nx, result.Ny
    kr_psixy = sample_horizontal_kraken(result.psixy, Nx, Ny, 0.75, ref_tt.x)
    kr_tauxy = sample_horizontal_kraken(result.tauxy, Nx, Ny, 0.75, ref_tt.x)
    profile_path = joinpath(cfg.output_dir, "profile_horizontal_y0.75.csv")
    open(profile_path, "w") do io
        println(io, "x,kraken_psixy,kraken_tauxy,rheotool_thetaxy,rheotool_tauxy")
        for k in eachindex(ref_tt.x)
            println(io, join(csv_cell.((ref_tt.x[k], kr_psixy.values[k], kr_tauxy.values[k],
                                        ref_tt.theta_xy[k], ref_tt.tau_xy[k])), ","))
        end
    end

    nan_count_psixy = count(x -> !isfinite(x), result.psixy)
    keys = (
        "N", "dt_phys", "nsteps", "lambda_lu", "prefactor", "initial",
        "replay_physical_time", "kraken_psixy_max", "rheotool_thetaxy_max",
        "rel_l2_psixy_y075", "rel_linf_psixy_y075", "nan_count_psixy",
        "wallclock_s",
    )
    values = (
        result.Nx,
        result.dt_phys,
        result.nsteps,
        result.lambda_lu,
        result.prefactor,
        string(result.initial),
        result.replay_physical_time,
        maximum(abs, kr_psixy.values),
        maximum(abs, ref_tt.theta_xy),
        rel_l2_psixy,
        rel_linf_psixy,
        nan_count_psixy,
        result.wallclock_s,
    )
    open(joinpath(cfg.output_dir, "summary.csv"), "w") do io
        println(io, join(keys, ","))
        println(io, join(csv_cell.(values), ","))
    end
    open(joinpath(cfg.output_dir, "fields.jls"), "w") do io
        serialize(io, (;
            ux=result.ux,
            uy=result.uy,
            psixx=result.psixx,
            psixy=result.psixy,
            psiyy=result.psiyy,
            tauxx=result.tauxx,
            tauxy=result.tauxy,
            tauyy=result.tauyy,
        ))
    end
    return profile_path
end

function main()
    cfg = parse_args(ARGS)
    backend_name, backend, T = pick_backend()
    println("Backend = $(backend_name), float = $(T)")
    println("Output directory = $(cfg.output_dir)")
    println("Mode: N=$(cfg.N), replay_physical_time=$(cfg.replay_physical_time), initial=$(cfg.initial)")

    rheotool_t, rheotool_sample_dir = pick_rheotool_sample_dir(cfg.case_dir, 8.0)
    foam_time = basename(rheotool_sample_dir)
    println("Reading rheoTool fields from $(joinpath(cfg.case_dir, foam_time)) (t = $(rheotool_t))")
    snapshot = load_foam_snapshot(cfg.case_dir, foam_time)
    input = resample_snapshot(snapshot, cfg.N)
    ref_tt = read_rheotool_horizontal_tautheta(joinpath(rheotool_sample_dir, "lineHorz_y0.75_tau_theta.xy"))

    result = run_frozen_replay(input, cfg; backend=backend, T=T)
    kr_psixy = sample_horizontal_kraken(result.psixy, result.Nx, result.Ny, 0.75, ref_tt.x)
    rel_l2_psixy = rel_l2_error(kr_psixy.values, ref_tt.theta_xy)
    rel_linf_psixy = rel_linf_error(kr_psixy.values, ref_tt.theta_xy)
    profile_path = write_outputs(cfg, result, ref_tt, rel_l2_psixy, rel_linf_psixy)

    nan_count_psixy = count(x -> !isfinite(x), result.psixy)
    nan_count_tauxy = count(x -> !isfinite(x), result.tauxy)
    println()
    println("psi_xy(x,y=0.75) rel L2   = $(rel_l2_psixy)")
    println("psi_xy(x,y=0.75) rel Linf = $(rel_linf_psixy)")
    println("nan_count_psixy = $(nan_count_psixy)")
    println("nan_count_tauxy = $(nan_count_tauxy)")
    println("profile = $(profile_path)")
    println("summary = $(joinpath(cfg.output_dir, "summary.csv"))")
    println("fields  = $(joinpath(cfg.output_dir, "fields.jls"))")

    if nan_count_psixy != 0 || nan_count_tauxy != 0
        error("non-finite polymer values detected")
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        main()
    catch err
        println(stderr, "ERROR: ", err)
        for frame in stacktrace(catch_backtrace())[1:min(end, 8)]
            println(stderr, "  ", frame)
        end
        exit(1)
    end
end
