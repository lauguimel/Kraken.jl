# Pseudo-PIV: micro-PIV-like observations generated from a converged Kraken
# velocity field (inverse-rheology track, periodic obstacle channel).
#
# Pipeline (order fixed, each step is a separate function):
#   1. conversion lattice -> u* units       u* = u_lat * m / nu0_lat
#   2. window average   `piv_window_average`  uniform mean of the FLUID nodes whose
#                                             centre lies in the square window
#   3. lost band        `piv_lost_band`       window closer than `lost_band` to a
#                                             solid node or a wall -> vector removed
#   4. Gaussian noise   `piv_add_noise!`      relative (x u*_max) then absolute
#   5. outliers         `piv_add_outliers!`   norm x `outlier_scale`, random direction
#   6. dropout          `piv_drop!`           uniform random removal
#
# Units. Input: lattice velocities on an (Nx, Ny) node grid, node (i, j) at
# ((i - 1/2)/m, (j - 1/2)/m), domain Lx = Nx/m by Ly = Ny/m (length unit = m
# nodes), periodic in x, walls at y = 0 and y = Ly (half-way, as in J1).
# Output: velocities in starred units u*, positions in domain length units.
#
# All settings live in `PseudoPIVSettings` (one place, defaults documented
# there). They are generic placeholders, to be recalibrated on the micro-PIV
# of the experiment.
#
# No dependency on the `Random` stdlib (not a dependency of Kraken): the caller
# passes an `AbstractRNG` (e.g. `Random.Xoshiro(settings.seed)`); `randn`/`rand`
# are taken from the module that defines the RNG type. Only RNGs defined in
# Random (Xoshiro, MersenneTwister, TaskLocalRNG) are supported.
#
# Not exported: use `Kraken.PseudoPIVSettings`, `Kraken.pseudo_piv`, ...

"""
    PseudoPIVSettings(; window, overlap, lost_band, y_walls, sigma_rel, sigma_abs,
                        dropout, outlier_frac, outlier_scale, seed)

Every parameter of the pseudo-PIV generator, in domain length units (obstacle channel:
domain 8 x 4) unless noted. Defaults are generic, not calibrated.

| field | default | meaning |
|---|---|---|
| `window` | 0.125 | side of the square interrogation window (= 4 nodes at m = 32) |
| `overlap` | 0.5 | window overlap fraction; vector spacing = `window * (1 - overlap)` |
| `lost_band` | `nothing` (= `window`) | vector removed if its window is closer than this to a solid node/wall (distance window-to-solid-cell, strict `<`) |
| `y_walls` | `true` | walls at y = 0 and y = Ly count as solid for the lost band |
| `sigma_rel` | 0.02 | Gaussian noise std, fraction of u*_max (max speed over fluid nodes), per component |
| `sigma_abs` | 0.0 | Gaussian noise std, absolute, in u* units, per component (independent of the relative one) |
| `dropout` | 0.03 | probability that a still-valid vector is missing (uniform, Bernoulli) |
| `outlier_frac` | 0.0 | probability that a vector is an outlier (Bernoulli) |
| `outlier_scale` | 3.0 | outlier norm = `outlier_scale` x local vector norm, random direction |
| `seed` | 20261020 | convention for the caller: `rng = Random.Xoshiro(seed)` |
"""
Base.@kwdef struct PseudoPIVSettings
    window::Float64 = 0.125
    overlap::Float64 = 0.5
    lost_band::Union{Nothing,Float64} = nothing
    y_walls::Bool = true
    sigma_rel::Float64 = 0.02
    sigma_abs::Float64 = 0.0
    dropout::Float64 = 0.03
    outlier_frac::Float64 = 0.0
    outlier_scale::Float64 = 3.0
    seed::Int = 20261020
end

piv_lost_band_value(s::PseudoPIVSettings) = s.lost_band === nothing ? s.window : s.lost_band
piv_spacing(s::PseudoPIVSettings) = s.window * (1 - s.overlap)

function piv_check(s::PseudoPIVSettings)
    s.window > 0 || throw(ArgumentError("window must be > 0"))
    0 <= s.overlap < 1 || throw(ArgumentError("overlap must be in [0, 1)"))
    piv_lost_band_value(s) >= 0 || throw(ArgumentError("lost_band must be >= 0"))
    (s.sigma_rel >= 0 && s.sigma_abs >= 0) || throw(ArgumentError("sigmas must be >= 0"))
    (0 <= s.dropout <= 1 && 0 <= s.outlier_frac <= 1) || throw(ArgumentError("fractions must be in [0, 1]"))
    s.outlier_scale >= 0 || throw(ArgumentError("outlier_scale must be >= 0"))
    return s
end

"""
    PseudoPIVField

Result of [`pseudo_piv`](@ref). `x`, `y`: vector positions (domain units);
`ux`, `uy`: observed u* (NaN where not valid), matrices `(length(x), length(y))`;
`ux_clean`, `uy_clean`: window means before noise/outliers/dropout (NaN where
removed by the lost band); `valid`: vector present; `outlier`: vector replaced by
an outlier; `counts`: `(total, lost_band, no_fluid, dropout, valid, outliers)`;
`u_ref`: u*_max used for the relative noise; `sigma_eff`: measured std of
(observed - clean) on valid non-outlier vectors, both components pooled.
"""
struct PseudoPIVField
    x::Vector{Float64}
    y::Vector{Float64}
    ux::Matrix{Float64}
    uy::Matrix{Float64}
    ux_clean::Matrix{Float64}
    uy_clean::Matrix{Float64}
    valid::BitMatrix
    outlier::BitMatrix
    counts::NamedTuple{(:total, :lost_band, :no_fluid, :dropout, :valid, :outliers),NTuple{6,Int}}
    u_ref::Float64
    sigma_eff::Float64
    settings::PseudoPIVSettings
    m::Int
    Lx::Float64
    Ly::Float64
end

# ------------------------------------------------------------ random draws ---
# randn/rand from the module that defines the RNG type (supports RNGs defined in
# Random only: Xoshiro, MersenneTwister, TaskLocalRNG): avoids a `Random` dependency in Project.toml.
piv_randn_fn(rng) = getfield(parentmodule(typeof(rng)), :randn)
piv_rand_fn(rng) = getfield(parentmodule(typeof(rng)), :rand)

# ------------------------------------------------------------ window layout ---
"""
    piv_window_centers(Lx, Ly, window, spacing) -> (xc, yc)

x: `(k - 1/2) * spacing`, k = 1 .. floor(Lx/spacing) (periodic). y: the largest
symmetric lattice of centres whose windows lie inside `[0, Ly]`.
"""
function piv_window_centers(Lx, Ly, window, spacing)
    nvx = floor(Int, Lx / spacing + 1e-9)
    nvy = floor(Int, (Ly - window) / spacing + 1e-9) + 1
    nvx >= 1 && nvy >= 1 || throw(ArgumentError("window/spacing larger than the domain"))
    xc = [(k - 0.5) * spacing for k in 1:nvx]
    y0 = (Ly - (nvy - 1) * spacing) / 2
    yc = [y0 + (l - 1) * spacing for l in 1:nvy]
    return xc, yc
end

# Node index range (1-based, possibly outside 1:N) of the nodes whose centre
# (i - 1/2)/m lies in [lo, hi) (lower edge inclusive up to 1e-9 node).
@inline function piv_index_range(lo, hi, m)
    i_lo = ceil(Int, lo * m + 0.5 - 1e-9)
    i_hi = ceil(Int, hi * m + 0.5 - 1e-9) - 1
    return i_lo, i_hi
end

# -------------------------------------------------------- 2. window average ---
"""
    piv_window_average(u, solid, m, xc, yc, window) -> (mean, nfluid)

Uniform mean of `u` over the fluid nodes (`solid == false`) whose centre lies in
the square window of side `window` centred on `(xc[a], yc[b])`; x wraps
periodically. `mean` is NaN where the window holds no fluid node.
"""
function piv_window_average(u::AbstractMatrix, solid::AbstractMatrix{Bool}, m::Integer,
                            xc::AbstractVector, yc::AbstractVector, window::Real)
    Nx, Ny = size(u)
    size(solid) == (Nx, Ny) || throw(DimensionMismatch("u and solid sizes differ"))
    out = fill(NaN, length(xc), length(yc))
    cnt = zeros(Int, length(xc), length(yc))
    h = window / 2
    for b in eachindex(yc)
        j_lo, j_hi = piv_index_range(yc[b] - h, yc[b] + h, m)
        j_lo = max(j_lo, 1); j_hi = min(j_hi, Ny)
        for a in eachindex(xc)
            i_lo, i_hi = piv_index_range(xc[a] - h, xc[a] + h, m)
            s = 0.0; n = 0
            for j in j_lo:j_hi, i in i_lo:i_hi
                ii = mod1(i, Nx)
                if !solid[ii, j]
                    s += u[ii, j]; n += 1
                end
            end
            cnt[a, b] = n
            n > 0 && (out[a, b] = s / n)
        end
    end
    return out, cnt
end

# ---------------------------------------------------------- 3. lost band ---
"""
    piv_lost_band(solid, m, xc, yc, window, band; y_walls=true) -> BitMatrix

`true` where the window (square of side `window`) is closer than `band` to a
solid node (the solid node is its square cell of side 1/m) or, if `y_walls`,
to the walls y = 0 / y = Ly. Distance 0 if they overlap; strict test with a
1e-9 tolerance (a window exactly `band` away is kept). x is periodic.
"""
function piv_lost_band(solid::AbstractMatrix{Bool}, m::Integer, xc::AbstractVector,
                       yc::AbstractVector, window::Real, band::Real; y_walls::Bool = true)
    Nx, Ny = size(solid)
    Lx = Nx / m; Ly = Ny / m
    tol = 1e-9
    lost = falses(length(xc), length(yc))
    h = window / 2
    hc = 0.5 / m
    reach = band + hc
    for b in eachindex(yc), a in eachindex(xc)
        yb = yc[b]; xa = xc[a]
        if y_walls && (yb - h < band - tol || Ly - (yb + h) < band - tol)
            lost[a, b] = true
            continue
        end
        j_lo = max(1, floor(Int, (yb - h - reach) * m + 0.5) )
        j_hi = min(Ny, ceil(Int, (yb + h + reach) * m + 0.5))
        i_lo = floor(Int, (xa - h - reach) * m + 0.5)
        i_hi = ceil(Int, (xa + h + reach) * m + 0.5)
        found = false
        for j in j_lo:j_hi
            cy = (j - 0.5) / m
            dy = max(0.0, abs(cy - yb) - (h + hc))
            for i in i_lo:i_hi
                solid[mod1(i, Nx), j] || continue
                cx = (i - 0.5) / m
                dxr = abs(cx - xa)
                dxr = min(dxr, abs(dxr - Lx))     # periodic image
                dx = max(0.0, dxr - (h + hc))
                if hypot(dx, dy) < band - tol
                    found = true; break
                end
            end
            found && break
        end
        lost[a, b] = found
    end
    return lost
end

# ------------------------------------------------------------- 4. noise ---
"""
    piv_add_noise!(rng, ux, uy, sigma_rel, sigma_abs, u_ref)

Adds independent zero-mean Gaussian noise to every entry (NaN stay NaN): first
std `sigma_rel * u_ref`, then std `sigma_abs`. Draw order is entry-by-entry in
column-major order, ux then uy, relative then absolute, so the stream does not
depend on which vectors are valid.
"""
function piv_add_noise!(rng, ux::AbstractMatrix, uy::AbstractMatrix, sigma_rel::Real,
                        sigma_abs::Real, u_ref::Real)
    size(ux) == size(uy) || throw(DimensionMismatch("ux and uy sizes differ"))
    rn = piv_randn_fn(rng)
    s1 = sigma_rel * u_ref; s2 = sigma_abs
    for k in eachindex(ux)
        if s1 > 0
            ux[k] += s1 * rn(rng); uy[k] += s1 * rn(rng)
        end
        if s2 > 0
            ux[k] += s2 * rn(rng); uy[k] += s2 * rn(rng)
        end
    end
    return ux, uy
end

# ---------------------------------------------------------- 5. outliers ---
"""
    piv_add_outliers!(rng, ux, uy, valid, frac, scale) -> BitMatrix

Each entry is an outlier with probability `frac` (one uniform draw per entry,
valid or not); an outlier valid vector is replaced by a vector of norm
`scale * |u|` (its own norm before replacement) in a uniformly random direction
(a second draw, only for selected entries). Returns the outlier flags.
"""
function piv_add_outliers!(rng, ux::AbstractMatrix, uy::AbstractMatrix, valid::AbstractMatrix{Bool},
                           frac::Real, scale::Real)
    rd = piv_rand_fn(rng)
    flag = falses(size(ux))
    for k in eachindex(ux)
        r = rd(rng)
        r < frac || continue
        θ = 2π * rd(rng)
        valid[k] || continue
        n = scale * hypot(ux[k], uy[k])
        ux[k] = n * cos(θ); uy[k] = n * sin(θ)
        flag[k] = true
    end
    return flag
end

# ------------------------------------------------------------ 6. dropout ---
"""
    piv_drop!(rng, valid, frac) -> Int

Marks each currently valid vector as missing with probability `frac` (one
uniform draw per entry). Returns the number removed.
"""
function piv_drop!(rng, valid::AbstractMatrix{Bool}, frac::Real)
    rd = piv_rand_fn(rng)
    n = 0
    for k in eachindex(valid)
        r = rd(rng)
        if valid[k] && r < frac
            valid[k] = false; n += 1
        end
    end
    return n
end

# ----------------------------------------------------------------- driver ---
"""
    pseudo_piv(ux_lat, uy_lat, solid, m, nu0_lat, settings = PseudoPIVSettings(); rng) -> PseudoPIVField

Pseudo micro-PIV observation of a converged Kraken field. Inputs in lattice
units on an (Nx, Ny) node grid (see file header); `solid[i, j]` marks solid
nodes; `m` nodes per unit length; `nu0_lat` zero-shear lattice viscosity.
Conversion: `u* = u_lat * m / nu0_lat` (starred units, eta0 = 1). Pipeline order:
window average, lost band, noise, outliers, dropout. `rng` is required
(`Random.Xoshiro(settings.seed)`); same RNG state => same output bit for bit.
"""
function pseudo_piv(ux_lat::AbstractMatrix, uy_lat::AbstractMatrix, solid::AbstractMatrix{Bool},
                    m::Integer, nu0_lat::Real, settings::PseudoPIVSettings = PseudoPIVSettings();
                    rng)
    piv_check(settings)
    size(ux_lat) == size(uy_lat) == size(solid) || throw(DimensionMismatch("field sizes differ"))
    Nx, Ny = size(ux_lat)
    Lx = Nx / m; Ly = Ny / m
    conv = m / nu0_lat
    uxs = ux_lat .* conv; uys = uy_lat .* conv
    u_ref = 0.0
    for k in eachindex(uxs)
        solid[k] || (u_ref = max(u_ref, hypot(uxs[k], uys[k])))
    end
    xc, yc = piv_window_centers(Lx, Ly, settings.window, piv_spacing(settings))
    mx, nfl = piv_window_average(uxs, solid, m, xc, yc, settings.window)
    my, _ = piv_window_average(uys, solid, m, xc, yc, settings.window)
    band = piv_lost_band_value(settings)
    lost = piv_lost_band(solid, m, xc, yc, settings.window, band; y_walls = settings.y_walls)
    nofluid = (nfl .== 0) .& .!lost
    valid = .!(lost .| nofluid)
    clean_x = copy(mx); clean_y = copy(my)
    clean_x[.!valid] .= NaN; clean_y[.!valid] .= NaN
    ux = copy(clean_x); uy = copy(clean_y)
    piv_add_noise!(rng, ux, uy, settings.sigma_rel, settings.sigma_abs, u_ref)
    outl = piv_add_outliers!(rng, ux, uy, valid, settings.outlier_frac, settings.outlier_scale)
    ndrop = piv_drop!(rng, valid, settings.dropout)
    ux[.!valid] .= NaN; uy[.!valid] .= NaN
    outl .&= valid
    # measured noise (non-outlier valid vectors)
    ss = 0.0; nn = 0
    for k in eachindex(ux)
        if valid[k] && !outl[k]
            ss += (ux[k] - clean_x[k])^2 + (uy[k] - clean_y[k])^2; nn += 2
        end
    end
    sig = nn > 0 ? sqrt(ss / nn) : NaN
    counts = (total = length(valid), lost_band = count(lost), no_fluid = count(nofluid),
              dropout = ndrop, valid = count(valid), outliers = count(outl))
    return PseudoPIVField(xc, yc, ux, uy, clean_x, clean_y, valid, outl, counts, u_ref, sig,
                          settings, Int(m), Lx, Ly)
end

# --------------------------------------------------- reference grid output ---
# Staggered offsets (in cells, position = (i - 1 + ax, j - 1 + bx) * d, 1-based i, j).
# :mac (assumed, hypothesis H17 not verified): ux at (1, 1/2), uy at (1/2, 1).
# :center: both components at cell centres (1/2, 1/2).
function piv_stagger_offsets(stagger::Symbol)
    stagger === :mac && return ((1.0, 0.5), (0.5, 1.0))
    stagger === :center && return ((0.5, 0.5), (0.5, 0.5))
    throw(ArgumentError("unknown stagger $(repr(stagger)); use :mac or :center"))
end

"""
    to_reference_grid(field::PseudoPIVField; nx=256, ny=128, Lx=field.Lx, Ly=field.Ly, stagger=:mac)
        -> (ux, uy, valid)

Projects the valid vectors onto the reference grid by assigning each component to
the nearest staggered node of that component (ties: upper index). `ux`, `uy`:
`Float32` `(nx, ny)` arrays in (x, y) order, `NaN` where no vector. Vectors
landing on the same node are averaged. `valid` is `(x = BitMatrix, y = BitMatrix)`:
the nodes holding an observation of ux / uy (they differ under a staggered
convention). x is periodic: index `mod1(i, nx)`; with `:mac` a vector at
x = Lx - eps goes to column `nx` for both components, one at x = eps goes to
column `nx` for ux (ux sits at x = i*dx, column `nx` is x = Lx = 0) and column 1
for uy. y is not periodic: vectors outside `1:ny` are dropped. With `:center`
the x = Lx - eps vector goes to column `nx` and x = eps to column 1.
`Lx`/`Ly` default to the field's own domain and must match it (error otherwise).
If H17 is false, only `stagger` changes (offsets in `piv_stagger_offsets`).
"""
function to_reference_grid(f::PseudoPIVField; nx::Integer = 256, ny::Integer = 128,
                        Lx::Real = f.Lx, Ly::Real = f.Ly, stagger::Symbol = :mac)
    (isapprox(Lx, f.Lx; rtol = 1e-9) && isapprox(Ly, f.Ly; rtol = 1e-9)) ||
        throw(ArgumentError("Lx, Ly = ($Lx, $Ly) disagree with the field's domain ($(f.Lx), $(f.Ly))"))
    offs = piv_stagger_offsets(stagger)
    dx = Lx / nx; dy = Ly / ny
    out = Matrix{Float32}[]; masks = BitMatrix[]
    for (c, src) in enumerate((f.ux, f.uy))
        ax, bx = offs[c]
        S = zeros(Float64, nx, ny); N = zeros(Int, nx, ny)
        for b in eachindex(f.y), a in eachindex(f.x)
            f.valid[a, b] || continue
            i = floor(Int, f.x[a] / dx - ax + 1 + 0.5)
            j = floor(Int, f.y[b] / dy - bx + 1 + 0.5)
            (1 <= j <= ny) || continue
            i = mod1(i, nx)
            S[i, j] += src[a, b]; N[i, j] += 1
        end
        G = fill(Float32(NaN), nx, ny)
        for k in eachindex(G)
            N[k] > 0 && (G[k] = Float32(S[k] / N[k]))
        end
        push!(out, G); push!(masks, N .> 0)
    end
    return out[1], out[2], (x = masks[1], y = masks[2])
end

"""
    piv_mask(valid) -> BitMatrix

Observation mask usable as a loss mask. For a `BitMatrix` (vector validity of a
`PseudoPIVField`): a copy. For a `PseudoPIVField`: its `valid`. For the
`(x, y)` pair returned by `to_reference_grid`: the union of both components.
"""
piv_mask(v::AbstractMatrix{Bool}) = BitMatrix(v)
piv_mask(f::PseudoPIVField) = copy(f.valid)
piv_mask(v::NamedTuple{(:x, :y)}) = v.x .| v.y

# ------------------------------------------------------------------ .npy ---
"""
    write_npy(path, A)

Writes a 2-D `Float32`, `Float64`, `UInt8` or `Bool` array as an `.npy` v1.0
file in C order (numpy `a[i, j] == A[i+1, j+1]`, shape `(size(A,1), size(A,2))`,
same layout as the reference fields).
"""
function write_npy(path::AbstractString, A::AbstractMatrix)
    T = eltype(A)
    descr = T == Float64 ? "<f8" : T == Float32 ? "<f4" : T == UInt8 ? "|u1" :
            T == Bool ? "|b1" : throw(ArgumentError("unsupported eltype $T"))
    hdr = "{'descr': '$descr', 'fortran_order': False, 'shape': ($(size(A, 1)), $(size(A, 2))), }"
    pad = mod(-(10 + length(hdr) + 1), 64)
    hdr = hdr * " "^pad * "\n"
    open(path, "w") do io
        write(io, UInt8[0x93, UInt8('N'), UInt8('U'), UInt8('M'), UInt8('P'), UInt8('Y'), 0x01, 0x00])
        write(io, htol(UInt16(length(hdr))))
        write(io, hdr)
        write(io, collect(permutedims(A)))      # row-major order of A
    end
    return path
end

"""
    read_npy(path) -> Array

Reads `.npy` (v1/v2, `<f4`, `<f8`, `|u1`, `|b1`, C or Fortran order, 1-D or
N-D) into a Julia array with the same index order as numpy.
"""
function read_npy(path::AbstractString)
    open(path, "r") do io
        magic = read(io, 6)
        magic == UInt8[0x93, UInt8('N'), UInt8('U'), UInt8('M'), UInt8('P'), UInt8('Y')] ||
            error("$path: not an .npy file")
        major = read(io, UInt8); read(io, UInt8)
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

"""
    write_reference_npy(dir, ux, uy, valid)

Writes `reference_velocity_x.npy` / `reference_velocity_y.npy` (Float32,
`(nx, ny)`, NaN where unobserved) and `observation_mask_x.npy` /
`observation_mask_y.npy` (UInt8 0/1) in `dir` (created if needed).
"""
function write_reference_npy(dir::AbstractString, ux::AbstractMatrix{Float32}, uy::AbstractMatrix{Float32},
                          valid::NamedTuple{(:x, :y)})
    mkpath(dir)
    write_npy(joinpath(dir, "reference_velocity_x.npy"), ux)
    write_npy(joinpath(dir, "reference_velocity_y.npy"), uy)
    write_npy(joinpath(dir, "observation_mask_x.npy"), UInt8.(valid.x))
    write_npy(joinpath(dir, "observation_mask_y.npy"), UInt8.(valid.y))
    return dir
end
