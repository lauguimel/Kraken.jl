#!/usr/bin/env julia

using KernelAbstractions
using Kraken
using Printf
using Serialization

if isempty(ARGS)
    println(stderr, "usage: julia --project=. bench/viscoelastic_logfv/analyze_logfv_gradient_dump.jl FIELD_DUMP.jls")
    exit(2)
end

dump_path = ARGS[1]
payload = open(deserialize, dump_path)
metadata = payload.metadata
fields = payload.fields

ux_h = Float64.(fields["ux"])
uy_h = Float64.(fields["uy"])
solid_h = fields["solid"] .> 0.5
Nx, Ny = size(ux_h)
R = Float64(get(metadata, :R, NaN))
L_up = Float64(get(metadata, :L_up, NaN))
lambda = Float64(get(metadata, :lambda, NaN))
cx = L_up * R
cy = (Ny - 1) / 2

q_wall, solid_from_q = Kraken.precompute_q_wall_cylinder(Nx, Ny, cx, cy, R; FT=Float64)
embedded_h = Kraken.fvfd_embedded_boundary_from_qwall_2d(q_wall; FT=Float64)
bc = Kraken.fvfd_openx_wally_bcspec_2d()
backend = KernelAbstractions.CPU()

ux = KernelAbstractions.allocate(backend, Float64, Nx, Ny)
uy = KernelAbstractions.allocate(backend, Float64, Nx, Ny)
solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny)
copyto!(ux, ux_h)
copyto!(uy, uy_h)
copyto!(solid, solid_h)
embedded = Kraken.fvfd_transfer_embedded_boundary_2d(embedded_h, backend, Float64)

function gradient_arrays()
    return (
        zeros(Float64, Nx, Ny),
        zeros(Float64, Nx, Ny),
        zeros(Float64, Nx, Ny),
        zeros(Float64, Nx, Ny),
    )
end

plain = gradient_arrays()
embedded_out = gradient_arrays()
Kraken.fvfd_velocity_gradient_2d!(plain..., ux, uy, solid, 1.0, 1.0, bc)
Kraken.fvfd_velocity_gradient_embedded_2d!(
    embedded_out..., ux, uy, solid, 1.0, 1.0, bc, embedded,
)

function surface_distance(i::Integer, j::Integer)
    x = Float64(i - 1)
    y = Float64(j - 1)
    return hypot(x - cx, y - cy) - R
end

function gradient_stats(grads)
    dudx, dudy, dvdx, dvdy = grads
    max_norm = -Inf
    max_i = 0
    max_j = 0
    n_nonfinite = 0
    @inbounds for j in 1:Ny, i in 1:Nx
        solid_h[i, j] && continue
        vals = (dudx[i, j], dudy[i, j], dvdx[i, j], dvdy[i, j])
        if any(v -> !isfinite(v), vals)
            n_nonfinite += 1
            continue
        end
        norm = sqrt(sum(v -> v * v, vals))
        if norm > max_norm
            max_norm = norm
            max_i = i
            max_j = j
        end
    end
    return (; max_norm, max_i, max_j,
            surface_distance=surface_distance(max_i, max_j),
            lambda_norm=isfinite(lambda) ? lambda * max_norm : NaN,
            n_nonfinite)
end

plain_stats = gradient_stats(plain)
embedded_stats = gradient_stats(embedded_out)

max_inv = maximum(embedded_h.wall_inv_distance)
max_inv_idx = argmax(embedded_h.wall_inv_distance)
max_inv_i, max_inv_j = Tuple(max_inv_idx)
active_embedded = count(>(0), embedded_h.wall_inv_distance)

println("="^78)
println("Log-FV velocity-gradient dump diagnostic")
println("dump=$(dump_path)")
println("R=$(get(metadata, :R, "")) Wi=$(get(metadata, :Wi, "")) lambda=$(lambda) completed_steps=$(get(metadata, :completed_steps, ""))")
println("solid_mask_mismatch=$(count(solid_h .!= solid_from_q)) active_embedded_cells=$(active_embedded)")
@printf("embedded max wall_inv_distance=%.9g distance=%.9g at=(%d,%d) d_wall=%.9g\n",
        max_inv, inv(max_inv), max_inv_i, max_inv_j,
        surface_distance(max_inv_i, max_inv_j))
@printf("plain    max||grad u||=%.9g lambda*norm=%.9g at=(%d,%d) d_wall=%.9g nonfinite=%d\n",
        plain_stats.max_norm, plain_stats.lambda_norm,
        plain_stats.max_i, plain_stats.max_j,
        plain_stats.surface_distance, plain_stats.n_nonfinite)
@printf("embedded max||grad u||=%.9g lambda*norm=%.9g at=(%d,%d) d_wall=%.9g nonfinite=%d\n",
        embedded_stats.max_norm, embedded_stats.lambda_norm,
        embedded_stats.max_i, embedded_stats.max_j,
        embedded_stats.surface_distance, embedded_stats.n_nonfinite)
