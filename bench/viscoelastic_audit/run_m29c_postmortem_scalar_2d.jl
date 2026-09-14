#!/usr/bin/env julia

using Kraken

struct ScalarCase
    label::String
    kind::Symbol
    n_steps::Int
    west_value::Float64
    east_value::Float64
end

function _local_superbee_limiter(r)
    return max(zero(r), max(min(2 * r, one(r)), min(r, 2 * one(r))))
end

function _local_muscl_superbee_face_value(far_upwind, upwind, downwind)
    d_up = upwind - far_upwind
    d_down = downwind - upwind
    r = d_down == zero(d_down) ? zero(d_down) : d_up / d_down
    return upwind + (one(r) / (one(r) + one(r))) * _local_superbee_limiter(r) * d_down
end

function _local_guarded_face_value(far_upwind, upwind, downwind, canonical_usable)
    if canonical_usable
        return _local_muscl_superbee_face_value(far_upwind, upwind, downwind)
    end
    return upwind
end

function _local_bc_east(phi, east_value_arr, i, j, Nx)
    return i < Nx ? phi[i + 1, j] : east_value_arr[j]
end

function _local_bc_west(phi, west_value_arr, i, j)
    return i > 1 ? phi[i - 1, j] : west_value_arr[j]
end

function _local_bc_north(phi, i, j, Ny)
    return j < Ny ? phi[i, j + 1] : phi[i, 1]
end

function _local_bc_south(phi, i, j, Ny)
    return j > 1 ? phi[i, j - 1] : phi[i, Ny]
end

function _local_muscl_superbee_1sided_upwind_step!(
    phi_out, phi, ux_face, uy_face, is_solid,
    Nx, Ny, dx, dy, dt,
    west_value_arr, east_value_arr,
    south_value_arr, north_value_arr,
)
    inv_dx = inv(dx)
    inv_dy = inv(dy)
    @inbounds for j in 1:Ny, i in 1:Nx
        if is_solid[i, j]
            phi_out[i, j] = zero(eltype(phi_out))
        else
            ue = ux_face[i + 1, j]
            uw = ux_face[i, j]
            vn = uy_face[i, j + 1]
            vs = uy_face[i, j]

            east_value = _local_bc_east(phi, east_value_arr, i, j, Nx)
            west_value = _local_bc_west(phi, west_value_arr, i, j)
            north_value = _local_bc_north(phi, i, j, Ny)
            south_value = _local_bc_south(phi, i, j, Ny)

            phie = if ue >= 0
                upwind = phi[i, j]
                downwind = east_value
                canonical_usable = i > 1 && !is_solid[i - 1, j]
                far_upwind = canonical_usable ? phi[i - 1, j] : upwind
                _local_guarded_face_value(far_upwind, upwind, downwind, canonical_usable)
            else
                upwind = east_value
                downwind = phi[i, j]
                canonical_usable = i + 2 <= Nx && !is_solid[i + 2, j]
                far_upwind = canonical_usable ? phi[i + 2, j] : upwind
                _local_guarded_face_value(far_upwind, upwind, downwind, canonical_usable)
            end

            phiw = if uw >= 0
                upwind = west_value
                downwind = phi[i, j]
                canonical_usable = i > 2 && !is_solid[i - 2, j]
                far_upwind = canonical_usable ? phi[i - 2, j] : upwind
                _local_guarded_face_value(far_upwind, upwind, downwind, canonical_usable)
            else
                upwind = phi[i, j]
                downwind = west_value
                canonical_usable = i < Nx && !is_solid[i + 1, j]
                far_upwind = canonical_usable ? phi[i + 1, j] : upwind
                _local_guarded_face_value(far_upwind, upwind, downwind, canonical_usable)
            end

            phin = if vn >= 0
                upwind = phi[i, j]
                downwind = north_value
                canonical_usable = j > 1 && !is_solid[i, j - 1]
                far_upwind = canonical_usable ? phi[i, j - 1] : upwind
                _local_guarded_face_value(far_upwind, upwind, downwind, canonical_usable)
            else
                upwind = north_value
                downwind = phi[i, j]
                canonical_usable = j + 2 <= Ny && !is_solid[i, j + 2]
                far_upwind = canonical_usable ? phi[i, j + 2] : upwind
                _local_guarded_face_value(far_upwind, upwind, downwind, canonical_usable)
            end

            phis = if vs >= 0
                upwind = south_value
                downwind = phi[i, j]
                canonical_usable = j > 2 && !is_solid[i, j - 2]
                far_upwind = canonical_usable ? phi[i, j - 2] : upwind
                _local_guarded_face_value(far_upwind, upwind, downwind, canonical_usable)
            else
                upwind = phi[i, j]
                downwind = south_value
                canonical_usable = j < Ny && !is_solid[i, j + 1]
                far_upwind = canonical_usable ? phi[i, j + 1] : upwind
                _local_guarded_face_value(far_upwind, upwind, downwind, canonical_usable)
            end

            flux_div = (ue * phie - uw * phiw) * inv_dx +
                       (vn * phin - vs * phis) * inv_dy
            divu = (ue - uw) * inv_dx + (vn - vs) * inv_dy
            rhs = -(flux_div - phi[i, j] * divu)
            phi_out[i, j] = phi[i, j] + dt * rhs
        end
    end
    return nothing
end

function zero_solid!(phi, is_solid)
    Nx, Ny = size(phi)
    @inbounds for j in 1:Ny, i in 1:Nx
        if is_solid[i, j]
            phi[i, j] = zero(eltype(phi))
        end
    end
    return phi
end

function initial_phi(case::ScalarCase, ::Type{T}, Nx, Ny, is_solid) where {T}
    phi = zeros(T, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        if case.kind == :step
            phi[i, j] = i >= 8 ? one(T) : zero(T)
        elseif case.kind == :gaussian
            phi[i, j] = exp(-((T(i) - T(12)) / T(3))^2)
        elseif case.kind == :checkerboard
            in_band = 3 <= i <= Nx - 2
            phi[i, j] = in_band && isodd(i) ? one(T) : zero(T)
        else
            error("unsupported test case $(case.kind)")
        end
    end
    return zero_solid!(phi, is_solid)
end

function exact_value(case::ScalarCase, i, step, dx, dt)
    shift = step * dt / dx
    if case.kind == :step
        return i - shift >= 8 ? 1.0 : 0.0
    elseif case.kind == :gaussian
        return exp(-(((i - shift) - 12.0) / 3.0)^2)
    end
    return 0.0
end

function extrema_fluid(phi, is_solid)
    Nx, Ny = size(phi)
    max_phi = -Inf
    min_phi = Inf
    @inbounds for j in 1:Ny, i in 1:Nx
        if !is_solid[i, j]
            value = phi[i, j]
            max_phi = max(max_phi, value)
            min_phi = min(min_phi, value)
        end
    end
    return max_phi, min_phi
end

function exact_errors(phi, case::ScalarCase, is_solid, step, dx, dt)
    Nx, Ny = size(phi)
    linf = 0.0
    sum2 = 0.0
    count = 0
    @inbounds for j in 1:Ny, i in 1:Nx
        if !is_solid[i, j]
            err = phi[i, j] - exact_value(case, i, step, dx, dt)
            linf = max(linf, abs(err))
            sum2 += err * err
            count += 1
        end
    end
    return linf, sqrt(sum2 / count)
end

function reference_errors(phi, reference, i_lo, i_hi)
    Nx, Ny = size(phi)
    i_start = max(1, i_lo)
    i_stop = min(Nx, i_hi)
    linf = 0.0
    sum2 = 0.0
    count = 0
    @inbounds for j in 1:Ny, i in i_start:i_stop
        err = phi[i, j] - reference[i, j]
        linf = max(linf, abs(err))
        sum2 += err * err
        count += 1
    end
    return linf, sqrt(sum2 / count)
end

function mode1_amp(phi, initial_max, Nx)
    final_max = -Inf
    @inbounds for i in 3:(Nx - 2)
        final_max = max(final_max, phi[i, 2])
    end
    return initial_max == 0.0 ? 0.0 : final_max / initial_max
end

function metric_values(phi, case::ScalarCase, is_solid, step, dx, dt, step1, initial_max)
    max_phi, min_phi = extrema_fluid(phi, is_solid)
    if case.kind == :checkerboard
        linf, l2 = reference_errors(phi, step1, 3, size(phi, 1) - 2)
        amp = mode1_amp(phi, initial_max, size(phi, 1))
        return max_phi, min_phi, linf, l2, amp
    end
    linf, l2 = exact_errors(phi, case, is_solid, step, dx, dt)
    return max_phi, min_phi, linf, l2, 0.0
end

function trajectory_error(phi, case::ScalarCase, is_solid, step, dx, dt, step1)
    if case.kind == :checkerboard
        step1 === nothing && return 0.0
        _, l2 = reference_errors(phi, step1, 3, size(phi, 1) - 2)
        return l2
    end
    _, l2 = exact_errors(phi, case, is_solid, step, dx, dt)
    return l2
end

function write_trajectory(path, rows)
    open(path, "w") do io
        println(io, "step,max_phi,min_phi,L2_err")
        for row in rows
            println(io, join((row.step, row.max_phi, row.min_phi, row.l2_err), ","))
        end
    end
    return nothing
end

function write_metrics(path, rows)
    header = "scheme,test,N_steps,max_phi,min_phi,Linf_err,L2_err,mode1_amp"
    open(path, "w") do io
        println(io, header)
        for row in rows
            println(io, join((
                row.scheme,
                row.test,
                row.N_steps,
                row.max_phi,
                row.min_phi,
                row.Linf_err,
                row.L2_err,
                row.mode1_amp,
            ), ","))
        end
    end
    return nothing
end

function write_plot_data(path, trajectory_rows)
    open(path, "w") do io
        println(io, "test,scheme,step,max_phi,min_phi,L2_err")
        for row in trajectory_rows
            println(io, join((
                row.test,
                row.scheme,
                row.step,
                row.max_phi,
                row.min_phi,
                row.l2_err,
            ), ","))
        end
    end
    return nothing
end

function print_summary(rows)
    println("scheme,test,N_steps,max_phi,min_phi,Linf_err,L2_err,mode1_amp")
    for row in rows
        println(join((
            row.scheme,
            row.test,
            row.N_steps,
            row.max_phi,
            row.min_phi,
            row.Linf_err,
            row.L2_err,
            row.mode1_amp,
        ), ","))
    end
    return nothing
end

function assert_finite_metrics(rows)
    for row in rows
        for value in (row.max_phi, row.min_phi, row.Linf_err, row.L2_err, row.mode1_amp)
            isfinite(value) || error("non-finite metric in $(row.scheme), $(row.test)")
        end
    end
    return nothing
end

function run_case!(
    scheme, case::ScalarCase, phi, phi_out, phi_bc,
    ux_face, uy_face, is_solid, dx, dy, dt, bc, scratch_dir,
)
    Nx, Ny = size(phi)
    scheme_label = String(scheme)
    initial_max = case.kind == :checkerboard ? mode1_amp(phi, 1.0, Nx) : 0.0
    step1 = nothing
    trajectory = NamedTuple[]
    all_trajectory_rows = NamedTuple[]

    max_phi, min_phi = extrema_fluid(phi, is_solid)
    l2_0 = trajectory_error(phi, case, is_solid, 0, dx, dt, step1)
    push!(trajectory, (step=0, max_phi=max_phi, min_phi=min_phi, l2_err=l2_0))
    push!(all_trajectory_rows, (
        test=case.label, scheme=scheme_label, step=0,
        max_phi=max_phi, min_phi=min_phi, l2_err=l2_0,
    ))

    for step in 1:case.n_steps
        if scheme == :muscl_superbee_1sided_upwind
            _local_muscl_superbee_1sided_upwind_step!(
                phi_out, phi, ux_face, uy_face, is_solid,
                Nx, Ny, dx, dy, dt,
                phi_bc.west, phi_bc.east, phi_bc.south, phi_bc.north,
            )
        else
            fvfd_advect_upwind_2d!(
                phi_out, phi, phi_bc, ux_face, uy_face, is_solid,
                dx, dy, bc, dt; advection_scheme=scheme,
            )
        end
        phi, phi_out = phi_out, phi
        if step == 1
            step1 = copy(phi)
        end
        max_phi, min_phi = extrema_fluid(phi, is_solid)
        l2_err = trajectory_error(phi, case, is_solid, step, dx, dt, step1)
        push!(trajectory, (step=step, max_phi=max_phi, min_phi=min_phi, l2_err=l2_err))
        push!(all_trajectory_rows, (
            test=case.label, scheme=scheme_label, step=step,
            max_phi=max_phi, min_phi=min_phi, l2_err=l2_err,
        ))
    end

    max_phi, min_phi, linf, l2, amp = metric_values(
        phi, case, is_solid, case.n_steps, dx, dt, step1, initial_max,
    )
    trajectory_path = joinpath(scratch_dir, "trajectory_$(scheme_label)_$(case.label).csv")
    write_trajectory(trajectory_path, trajectory)
    return (
        result=(
            scheme=scheme_label,
            test=case.label,
            N_steps=case.n_steps,
            max_phi=max_phi,
            min_phi=min_phi,
            Linf_err=linf,
            L2_err=l2,
            mode1_amp=amp,
        ),
        trajectory_rows=all_trajectory_rows,
    )
end

function main()
    T = Float64
    Nx = 64
    Ny = 4
    dx = T(1)
    dy = T(1)
    dt = T(0.4)
    repo_root = abspath(joinpath(@__DIR__, "..", ".."))
    scratch_dir = joinpath(repo_root, "bench", "scratch", "m29c_postmortem_empirical")
    tmp_dir = joinpath(repo_root, "tmp", "m29c_postmortem")
    mkpath(scratch_dir)
    mkpath(tmp_dir)

    bc = FVFDDomainBC2D(; west=:open, east=:open, south=:periodic, north=:periodic)
    is_solid = falses(Nx, Ny)
    is_solid[1, :] .= true
    is_solid[2, :] .= true
    ux_face = ones(T, Nx + 1, Ny)
    uy_face = zeros(T, Nx, Ny + 1)
    schemes = (:rusanov, :muscl_superbee, :muscl_superbee_1sided_upwind)
    cases = (
        ScalarCase("test_a_step", :step, 30, 0.0, 1.0),
        ScalarCase("test_b_gaussian", :gaussian, 30, 0.0, 0.0),
        ScalarCase("test_c_checkerboard", :checkerboard, 20, 0.0, 0.0),
    )

    metric_rows = NamedTuple[]
    plot_rows = NamedTuple[]
    for scheme in schemes
        for case in cases
            phi = initial_phi(case, T, Nx, Ny, is_solid)
            phi_out = similar(phi)
            west = fill(T(case.west_value), Ny)
            east = fill(T(case.east_value), Ny)
            south = zeros(T, Nx)
            north = zeros(T, Nx)
            phi_bc = FVFDFieldBC2D(; west=west, east=east, south=south, north=north)
            outcome = run_case!(
                scheme, case, phi, phi_out, phi_bc,
                ux_face, uy_face, is_solid, dx, dy, dt, bc, scratch_dir,
            )
            push!(metric_rows, outcome.result)
            append!(plot_rows, outcome.trajectory_rows)
        end
    end

    length(metric_rows) == 9 || error("expected 9 metric rows, got $(length(metric_rows))")
    assert_finite_metrics(metric_rows)
    write_metrics(joinpath(scratch_dir, "scalar_metrics.csv"), metric_rows)
    write_plot_data(joinpath(scratch_dir, "plot_data.csv"), plot_rows)
    print_summary(metric_rows)
    println("rows=$(length(metric_rows))")
    println("all_numeric_finite=true")
    return nothing
end

main()
