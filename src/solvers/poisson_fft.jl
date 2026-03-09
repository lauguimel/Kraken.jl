using FFTW

"""
    solve_poisson_fft!(phi, f, dx)

Solve the 2D Poisson equation nabla^2 phi = f with periodic boundary conditions
using the FFT spectral method.

The solution is computed by transforming `f` to spectral space, dividing by the
Laplacian eigenvalues, and transforming back. The zero mode is set to zero to
fix the gauge (mean of phi = 0).

CPU-only (uses FFTW). For GPU, a CUFFT-based implementation would be needed.

# Arguments
- `phi`: output array (N x N), will be overwritten with the solution
- `f`: right-hand side array (N x N)
- `dx`: uniform grid spacing

# Example
```julia
N = 64; dx = 1.0 / N
phi = zeros(N, N)
f = zeros(N, N)
solve_poisson_fft!(phi, f, dx)
```
"""
function solve_poisson_fft!(phi, f, dx)
    N = size(f, 1)
    T = eltype(f)

    # Forward FFT of RHS
    f_hat = fft(f)

    # Build Laplacian eigenvalues: lambda_{k,l} = (-2 + 2cos(2*pi*k/N))/dx^2 + same for l
    # = (2cos(2*pi*k/N) - 2)/dx^2 + (2cos(2*pi*l/N) - 2)/dx^2
    inv_dx2 = one(T) / (dx * dx)

    @inbounds for l in 1:N, k in 1:N
        # Wave numbers (0-indexed)
        kk = k - 1
        ll = l - 1
        eigenval = (2 * cos(2 * T(pi) * kk / N) - 2) * inv_dx2 +
                   (2 * cos(2 * T(pi) * ll / N) - 2) * inv_dx2

        if k == 1 && l == 1
            # Zero mode: set to zero to fix gauge
            f_hat[k, l] = zero(eltype(f_hat))
        else
            f_hat[k, l] /= eigenval
        end
    end

    # Inverse FFT to get solution
    phi .= real.(ifft(f_hat))
    return phi
end
