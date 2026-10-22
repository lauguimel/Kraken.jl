# Periodic obstacle channel for the Kraken GNF inverse-rheology track:
# geometry, unit conversions (plan §4.4), reference .npy I/O, comparison metrics.
#
# Physical case: domain (0,8)×(0,4), density 1, η0 = 1, walls at y = 0 and
# y = 4, periodic in x, two disks of radius 1.5 centred at (4, 0) and (4, 4)
# (throat gap y ∈ [1.5, 2.5] at x = 4), pressure-gradient body force G* = 5.
# Observation (reference) grid: 256×128, dx = 1/32, staggered velocity (ux at
# x = (i+1)dx, y = (j+½)dx; uy at x = (i+½)dx, y = (j+1)dx, 0-based i, j).
#
# Kraken: length unit ↔ m nodes, node (i, j) at ((i−½)/m, (j−½)/m), half-way
# walls, solid = nodes inside the disks.
#
# Optional reference data (not distributed), under the root given by the
# environment variable KRAKEN_REFERENCE_DATA (`Kraken.reference_data_dir()`):
#   REF_FIELDS_DIR    reference fields     env KRAKEN_REFERENCE_FIELDS, default <root>/reference_fields
#   REF_MASK_FROZEN   frozen 256×128 mask  env KRAKEN_REFERENCE_MASK, default <root>/masks/mask_frozen_256x128.npy
#   <root>/benchmarks/inverse_rheology/reference_runs.jl defines
#   REFERENCE_RUNS = Dict(run => (; folder, eta_inf, lam, n)), merged into CASE_RUNS.
#
# Grid rule (decided 2026-10-19), see `case_grid`: (m; ν0,lat) = (64; 0.34) by
# default, (72; 0.34) for any fluid whose Mach number at (64; 0.34) exceeds 0.1.

using Kraken
using KernelAbstractions
using Printf
using TOML

const CASE_L = 8.0
const CASE_H = 4.0
const CASE_R = 1.5
const CASE_G = 5.0
const REF_DX = 1 / 32            # reference grid spacing
const REF_NX, REF_NY = 256, 128
const _REF_ROOT = something(Kraken.reference_data_dir(), "")
const REF_FIELDS_DIR = get(ENV, "KRAKEN_REFERENCE_FIELDS",
                           isempty(_REF_ROOT) ? "" : joinpath(_REF_ROOT, "reference_fields"))
const REF_MASK_FROZEN = get(ENV, "KRAKEN_REFERENCE_MASK",
                            isempty(_REF_ROOT) ? "" : joinpath(_REF_ROOT, "masks", "mask_frozen_256x128.npy"))

# Truth CY parameters (η0 = 1, a = 2) per run and the folder of each run in the
# reference fields. Run 1 is the case definition; the other runs and the
# folders come from the reference data when present.
const CASE_RUNS = let runs = Dict{Int,NamedTuple{(:folder, :eta_inf, :lam, :n),Tuple{String,Float64,Float64,Float64}}}(
        1 => (folder = "", eta_inf = 0.02, lam = 5.0, n = 0.7))
    f = isempty(_REF_ROOT) ? "" : joinpath(_REF_ROOT, "benchmarks", "inverse_rheology", "reference_runs.jl")
    if isfile(f)
        mref = Module(:ReferenceRuns)
        Base.include(mref, f)
        for (k, v) in Base.invokelatest(getglobal, mref, :REFERENCE_RUNS)
            runs[k] = (folder = String(v.folder), eta_inf = Float64(v.eta_inf), lam = Float64(v.lam), n = Float64(v.n))
        end
    end
    runs
end

# ------------------------------------------------------------ grid rule ---
# (m; ν0,lat) per run. Default (64; 0.34). A fluid whose Mach number
# Ma = √3 u_lat,max exceeds 0.1 at (64; 0.34) runs at (72; 0.34): same ν0, so
# the same τ range, and u_lat = ν0/m · u* drops by 64/72. Measured (J1, J1b,
# steady solves on A100): run 1 Ma 0.074, run 4 Ma 0.106 → 0.094 at (72; 0.34),
# run 7 Ma 0.061. Runs 2, 3, 5, 6 are not measured yet and use the default
# until a run shows Ma > 0.1 (results.toml records `mach` and `mach_pass`).
const CASE_GRID_DEFAULT = (m = 64, nu0 = 0.34)
const CASE_GRID_OVERRIDE = Dict(4 => (m = 72, nu0 = 0.34))
case_grid(run::Int) = get(CASE_GRID_OVERRIDE, run, CASE_GRID_DEFAULT)

# ------------------------------------------------------------------ units ---
struct CaseUnits
    m::Int          # nodes per unit length
    nu0::Float64    # lattice kinematic viscosity at zero shear (η* = 1)
end
u_ref_lat(U::CaseUnits) = U.nu0 / U.m                    # lattice velocity of u* = 1
g_lat(U::CaseUnits) = CASE_G * U.nu0^2 / U.m^3          # body force per unit mass
dt_star(U::CaseUnits) = U.nu0 / U.m^2                     # physical time per step
k_lat(U::CaseUnits, k) = k * U.m^2 / U.nu0
nu_star(U::CaseUnits, nu_lat) = nu_lat / U.nu0

function case_closure(U::CaseUnits, run::Int)
    p = CASE_RUNS[run]
    return CarreauYasuda(U.nu0, p.eta_inf * U.nu0, k_lat(U, p.lam), 2.0, p.n)
end

grid_size(U::CaseUnits) = (round(Int, CASE_L * U.m), round(Int, CASE_H * U.m))

function case_solid_mask(U::CaseUnits)
    Nx, Ny = grid_size(U)
    s = fill(false, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        x = (i - 0.5) / U.m; y = (j - 0.5) / U.m
        if (x - 4)^2 + y^2 < CASE_R^2 || (x - 4)^2 + (y - CASE_H)^2 < CASE_R^2
            s[i, j] = true
        end
    end
    return s
end

# Distance (physical) to the nearest wall: channel walls and disk surfaces.
function wall_distance(x, y)
    r1 = hypot(x - 4, y); r2 = hypot(x - 4, y - CASE_H)
    return min(y, CASE_H - y, abs(r1 - CASE_R), abs(r2 - CASE_R))
end

# ------------------------------------------------------------------- npy ---
function read_npy(path::AbstractString)
    open(path, "r") do io
        magic = read(io, 6)
        magic == UInt8[0x93, UInt8('N'), UInt8('U'), UInt8('M'), UInt8('P'), UInt8('Y')] ||
            error("$path: not an .npy file")
        major = read(io, UInt8); _ = read(io, UInt8)
        hlen = major == 1 ? Int(ltoh(read(io, UInt16))) : Int(ltoh(read(io, UInt32)))
        header = String(read(io, hlen))
        descr = match(r"'descr':\s*'([^']*)'", header)[1]
        fortran = occursin(r"'fortran_order':\s*True", header)
        shp_s = match(r"'shape':\s*\(([^)]*)\)", header)[1]
        shp = Tuple(parse(Int, strip(s)) for s in split(shp_s, ',') if !isempty(strip(s)))
        T = descr == "<f4" ? Float32 : descr == "<f8" ? Float64 :
            descr == "|u1" ? UInt8 : descr == "|b1" ? Bool :
            error("$path: unsupported dtype $descr")
        data = Vector{T}(undef, prod(shp))
        read!(io, data)
        if fortran || length(shp) == 1
            return reshape(data, shp)
        else
            return permutedims(reshape(data, reverse(shp)), reverse(1:length(shp)))
        end
    end
end

function write_npy(path::AbstractString, A::AbstractArray)
    T = eltype(A)
    descr = T == Float64 ? "<f8" : T == Float32 ? "<f4" : T == UInt8 ? "|u1" :
            T == Bool ? "|b1" : error("unsupported eltype $T")
    shp = size(A)
    shape_s = length(shp) == 1 ? "($(shp[1]),)" : "(" * join(shp, ", ") * ")"
    header = "{'descr': '$descr', 'fortran_order': True, 'shape': $shape_s, }"
    pad = 64 - mod(10 + length(header) + 1, 64)
    header = header * " "^pad * "\n"
    open(path, "w") do io
        write(io, UInt8[0x93]); write(io, "NUMPY"); write(io, UInt8(1)); write(io, UInt8(0))
        write(io, htol(UInt16(length(header)))); write(io, header)
        write(io, collect(A))
    end
    return path
end

load_reference_fields(run::Int) = (folder = joinpath(REF_FIELDS_DIR, CASE_RUNS[run].folder);
                         (Float64.(read_npy(joinpath(folder, "reference_velocity_x.npy"))),
                          Float64.(read_npy(joinpath(folder, "reference_velocity_y.npy")))))

# Observation positions of the reference grid (0-based i, j stored at 1-based [i+1, j+1]).
ux_pos(i, j) = ((i - 1 + 1) * REF_DX, (j - 1 + 0.5) * REF_DX)
uy_pos(i, j) = ((i - 1 + 0.5) * REF_DX, (j - 1 + 1) * REF_DX)

# ---------------------------------------------------------- interpolation ---
# Bilinear interpolation of a Kraken node field at physical (x, y): lattice
# coordinate ξ = x m + ½ (node i at ξ = i), periodic in x; in y the half-way
# walls at η = ½ and η = Ny + ½ carry u = 0. Solid nodes carry u = 0.
function interp_node(u::AbstractMatrix, m::Int, x, y)
    Nx, Ny = size(u)
    ξ = x * m + 0.5; η = y * m + 0.5
    i0 = floor(Int, ξ); tx = ξ - i0
    ia = mod1(i0, Nx); ib = mod1(i0 + 1, Nx)
    if η <= 1
        w = clamp((η - 0.5) / 0.5, 0.0, 1.0)
        va = w * u[ia, 1]; vb = w * u[ib, 1]
    elseif η >= Ny
        w = clamp((Ny + 0.5 - η) / 0.5, 0.0, 1.0)
        va = w * u[ia, Ny]; vb = w * u[ib, Ny]
    else
        j0 = floor(Int, η); ty = η - j0
        va = (1 - ty) * u[ia, j0] + ty * u[ia, j0 + 1]
        vb = (1 - ty) * u[ib, j0] + ty * u[ib, j0 + 1]
    end
    return (1 - tx) * va + tx * vb
end

# Kraken physical velocity at the reference staggered positions, (256, 128) each.
function kraken_at_reference(uxs::AbstractMatrix, uys::AbstractMatrix, m::Int)
    kx = zeros(REF_NX, REF_NY); ky = zeros(REF_NX, REF_NY)
    for j in 1:REF_NY, i in 1:REF_NX
        x, y = ux_pos(i, j); kx[i, j] = interp_node(uxs, m, x, y)
        x, y = uy_pos(i, j); ky[i, j] = interp_node(uys, m, x, y)
    end
    return kx, ky
end

# ---------------------------------------------------------------- metrics ---
# RRMSE over the frozen observation mask (1 = kept), vector and ux only; share
# of the squared error carried by points within `near` cells (dx = 1/32) of a
# wall, and the RRMSE of the remaining (far) points.
function compare_fields(kx, ky, zx, zy, mask; near = 2.0)
    M = mask .!= 0
    dx_cells = zeros(size(zx)); dy_cells = zeros(size(zy))
    for j in 1:REF_NY, i in 1:REF_NX
        dx_cells[i, j] = wall_distance(ux_pos(i, j)...) / REF_DX
        dy_cells[i, j] = wall_distance(uy_pos(i, j)...) / REF_DX
    end
    ex = kx .- zx; ey = ky .- zy
    num = sum(ex[M] .^ 2) + sum(ey[M] .^ 2)
    den = sum(zx[M] .^ 2) + sum(zy[M] .^ 2)
    rrmse_u = sqrt(num / den)
    rrmse_ux = sqrt(sum(ex[M] .^ 2) / sum(zx[M] .^ 2))
    nx = M .& (dx_cells .<= near); ny = M .& (dy_cells .<= near)
    fx = M .& .!nx; fy = M .& .!ny
    near_share = (sum(ex[nx] .^ 2) + sum(ey[ny] .^ 2)) / num
    rrmse_far = sqrt((sum(ex[fx] .^ 2) + sum(ey[fy] .^ 2)) / (sum(zx[fx] .^ 2) + sum(zy[fy] .^ 2)))
    rrmse_all = sqrt((sum(ex .^ 2) + sum(ey .^ 2)) / (sum(zx .^ 2) + sum(zy .^ 2)))
    return (; rrmse_u, rrmse_ux, near_share, rrmse_far, rrmse_all,
            n_mask = count(M), ex, ey, dx_cells, dy_cells)
end

# Throat profile: reference ux column at x = 4 (0-based i = 127), gap y ∈ [1.5, 2.5].
function throat_profile(kx, zx)
    i = 128
    rows = Tuple{Float64,Float64,Float64}[]
    for j in 1:REF_NY
        _, y = ux_pos(i, j)
        (1.5 <= y <= 2.5) && push!(rows, (y, zx[i, j], kx[i, j]))
    end
    return rows
end

function git_state(repo)
    sha = try readchomp(`git -C $repo rev-parse HEAD`) catch; "unknown" end
    dirty = try !isempty(readchomp(`git -C $repo status --porcelain`)) catch; true end
    return sha, dirty
end
