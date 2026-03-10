using KernelAbstractions

@kernel function advect_kernel!(out, @Const(u), @Const(v), @Const(f), inv_dx)
    i, j = @index(Global, NTuple)
    # Interior point indices (offset by 1 for boundary padding)
    ii = i + 1
    jj = j + 1

    @inbounds begin
        u_loc = u[ii, jj]
        v_loc = v[ii, jj]

        # Upwind scheme for ∂(u*f)/∂x
        if u_loc > 0
            dfdx = (f[ii, jj] - f[ii-1, jj]) * inv_dx
        else
            dfdx = (f[ii+1, jj] - f[ii, jj]) * inv_dx
        end

        # Upwind scheme for ∂(v*f)/∂y
        if v_loc > 0
            dfdy = (f[ii, jj] - f[ii, jj-1]) * inv_dx
        else
            dfdy = (f[ii, jj+1] - f[ii, jj]) * inv_dx
        end

        # Advection: u·∇f
        out[ii, jj] = u_loc * dfdx + v_loc * dfdy
    end
end

@kernel function advect_tvd_kernel!(out, @Const(u), @Const(v), @Const(f), inv_dx)
    i, j = @index(Global, NTuple)
    ii = i + 1
    jj = j + 1

    @inbounds begin
        u_loc = u[ii, jj]
        v_loc = v[ii, jj]
        nx = size(f, 1)
        ny = size(f, 2)

        # --- x-direction with MUSCL + Van Leer limiter ---
        fim1 = f[ii-1, jj]
        fi   = f[ii, jj]
        fip1 = f[ii+1, jj]
        fim2 = ii > 2    ? f[ii-2, jj] : fim1
        fip2 = ii+2 <= nx ? f[ii+2, jj] : fip1

        if u_loc > 0
            df_down = fi - fim1       # upwind gradient
            df_up   = fip1 - fi       # downwind gradient
            r = df_down / (df_up + copysign(eltype(f)(1e-30), df_up))
            psi = (r + abs(r)) / (one(eltype(f)) + abs(r))  # Van Leer
            dfdx = (df_down + eltype(f)(0.5) * psi * (df_up - df_down)) * inv_dx
        else
            df_down = fi - fip1       # upwind gradient (from right)
            df_up   = fim1 - fi       # downwind gradient (from right)
            r = df_down / (df_up + copysign(eltype(f)(1e-30), df_up))
            psi = (r + abs(r)) / (one(eltype(f)) + abs(r))
            dfdx = -(df_down + eltype(f)(0.5) * psi * (df_up - df_down)) * inv_dx
        end

        # --- y-direction with MUSCL + Van Leer limiter ---
        fjm1 = f[ii, jj-1]
        fj   = f[ii, jj]
        fjp1 = f[ii, jj+1]
        fjm2 = jj > 2    ? f[ii, jj-2] : fjm1
        fjp2 = jj+2 <= ny ? f[ii, jj+2] : fjp1

        if v_loc > 0
            df_down = fj - fjm1
            df_up   = fjp1 - fj
            r = df_down / (df_up + copysign(eltype(f)(1e-30), df_up))
            psi = (r + abs(r)) / (one(eltype(f)) + abs(r))
            dfdy = (df_down + eltype(f)(0.5) * psi * (df_up - df_down)) * inv_dx
        else
            df_down = fj - fjp1
            df_up   = fjm1 - fj
            r = df_down / (df_up + copysign(eltype(f)(1e-30), df_up))
            psi = (r + abs(r)) / (one(eltype(f)) + abs(r))
            dfdy = -(df_down + eltype(f)(0.5) * psi * (df_up - df_down)) * inv_dx
        end

        out[ii, jj] = u_loc * dfdx + v_loc * dfdy
    end
end

"""
    advect!(out, u, v, f, dx; scheme=:upwind)

Compute the advection of scalar field `f` by velocity `(u, v)` on a uniform grid
with spacing `dx`. Supports first-order `:upwind` (default) and second-order
`:tvd` (MUSCL + Van Leer limiter) schemes.

    out = u * ∂f/∂x + v * ∂f/∂y

The advection is computed for interior points `(2:N-1, 2:N-1)` only; boundary
values in `out` are left untouched.

The compute backend (CPU, CUDA, Metal) is selected automatically from the
array type via `KernelAbstractions.get_backend`.

# Arguments
- `out`: output field (N × N), advection result
- `u`: x-component of velocity (N × N)
- `v`: y-component of velocity (N × N)
- `f`: scalar field to advect (N × N)
- `dx`: uniform grid spacing

# Returns
- `out`: the modified output array.

# Example
```julia
N = 64; dx = 1.0 / (N - 1)
u, v, f = zeros(N, N), zeros(N, N), zeros(N, N)
out = zeros(N, N)
advect!(out, u, v, f, dx)
```

See also: [`laplacian!`](@ref), [`projection_step!`](@ref)
"""
function advect!(out, u, v, f, dx; ndrange=nothing, scheme=:upwind)
    backend = KernelAbstractions.get_backend(f)
    inv_dx = one(eltype(f)) / dx
    n = size(f, 1) - 2
    if ndrange === nothing
        ndrange = (n, n)
    end
    if scheme == :tvd
        kernel! = advect_tvd_kernel!(backend)
    else
        kernel! = advect_kernel!(backend)
    end
    kernel!(out, u, v, f, inv_dx; ndrange=ndrange)
    KernelAbstractions.synchronize(backend)
    return out
end
