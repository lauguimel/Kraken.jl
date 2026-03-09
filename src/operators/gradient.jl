using KernelAbstractions

@kernel function gradient_kernel!(gx, gy, @Const(p), inv_2dx)
    i, j = @index(Global, NTuple)
    # Central difference on interior points (2:N-1, 2:N-1)
    @inbounds gx[i+1, j+1] = (p[i+2, j+1] - p[i, j+1]) * inv_2dx
    @inbounds gy[i+1, j+1] = (p[i+1, j+2] - p[i+1, j]) * inv_2dx
end

"""
    gradient!(gx, gy, p, dx; ndrange=nothing)

Compute the gradient of a 2D scalar field `p` using central differences on a
uniform grid with spacing `dx`.

The gradient is computed for interior points `(2:N-1, 2:N-1)` only; boundary
values in `gx` and `gy` are left untouched.

The compute backend (CPU, CUDA, Metal) is selected automatically from the
array type via `KernelAbstractions.get_backend`.

# Arguments
- `gx`: output array for ∂p/∂x, same size as `p`
- `gy`: output array for ∂p/∂y, same size as `p`
- `p`: input scalar field (N × N)
- `dx`: uniform grid spacing
"""
function gradient!(gx, gy, p, dx; ndrange=nothing)
    backend = KernelAbstractions.get_backend(p)
    inv_2dx = one(eltype(p)) / (2 * dx)
    n = size(p, 1) - 2
    if ndrange === nothing
        ndrange = (n, n)
    end
    kernel! = gradient_kernel!(backend)
    kernel!(gx, gy, p, inv_2dx; ndrange=ndrange)
    KernelAbstractions.synchronize(backend)
    return gx, gy
end
