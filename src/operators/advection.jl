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

"""
    advect!(out, u, v, f, dx)

Compute the advection of scalar field `f` by velocity `(u, v)` using first-order
upwind differencing on a uniform grid with spacing `dx`.

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
"""
function advect!(out, u, v, f, dx; ndrange=nothing)
    backend = KernelAbstractions.get_backend(f)
    inv_dx = one(eltype(f)) / dx
    n = size(f, 1) - 2
    if ndrange === nothing
        ndrange = (n, n)
    end
    kernel! = advect_kernel!(backend)
    kernel!(out, u, v, f, inv_dx; ndrange=ndrange)
    KernelAbstractions.synchronize(backend)
    return out
end
