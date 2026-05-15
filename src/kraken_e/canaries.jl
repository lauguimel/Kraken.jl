const KRAKEN_E_CX = (0, 1, 0, -1, 0, 1, -1, -1, 1)
const KRAKEN_E_CY = (0, 0, 1, 0, -1, 1, 1, -1, -1)
const KRAKEN_E_W = (4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
                    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0)

@inline function kraken_e_feq(q::Int, ρ::T, ux::T, uy::T) where {T}
    cu = T(KRAKEN_E_CX[q]) * ux + T(KRAKEN_E_CY[q]) * uy
    usq = ux * ux + uy * uy
    return T(KRAKEN_E_W[q]) * ρ * (one(T) + T(3) * cu + T(4.5) * cu * cu - T(1.5) * usq)
end

function kraken_e_initialize_equilibrium_2d!(block::LeafBlock2D; ρ0=1.0,
                                             ux0=0.0, uy0=0.0)
    T = eltype(block.f)
    ρ = T(ρ0)
    ux = T(ux0)
    uy = T(uy0)
    for j in kraken_e_j_range(block), i in kraken_e_i_range(block)
        block.ρ[i,j] = ρ
        block.ux[i,j] = ux
        block.uy[i,j] = uy
        for q in 1:9
            block.f[i,j,q] = kraken_e_feq(q, ρ, ux, uy)
        end
    end
    kraken_e_exchange_halo_periodic_xy!(block)
    return block
end

function kraken_e_initialize_taylor_green_2d!(block::LeafBlock2D; ρ0=1.0, U0=0.04)
    T = eltype(block.f)
    ρ_base = T(ρ0)
    U = T(U0)
    L = T(block.Nx)
    k = T(2π) / L
    for jj in 1:block.Ny, ii in 1:block.Nx
        i = ii + 1
        j = jj + 1
        x = T(ii) - T(0.5)
        y = T(jj) - T(0.5)
        ux = U * cos(k * x) * sin(k * y)
        uy = -U * sin(k * x) * cos(k * y)
        ρ = ρ_base * (one(T) - T(0.75) * U * U * (cos(T(2) * k * x) + cos(T(2) * k * y)))
        block.ρ[i,j] = ρ
        block.ux[i,j] = ux
        block.uy[i,j] = uy
        for q in 1:9
            block.f[i,j,q] = kraken_e_feq(q, ρ, ux, uy)
        end
    end
    kraken_e_exchange_halo_periodic_xy!(block)
    return block
end

function kraken_e_poiseuille_reference(block::LeafBlock2D, g, ν)
    T = eltype(block.f)
    H = T(block.Ny)
    out = zeros(T, block.Ny)
    for jj in 1:block.Ny
        y = T(jj) - T(0.5)
        out[jj] = T(g) / (T(2) * T(ν)) * y * (H - y)
    end
    return out
end

function kraken_e_couette_reference(block::LeafBlock2D, U)
    T = eltype(block.f)
    H = T(block.Ny)
    out = zeros(T, block.Ny)
    for jj in 1:block.Ny
        y = T(jj) - T(0.5)
        out[jj] = T(U) * y / H
    end
    return out
end

function kraken_e_mean_ux_by_y(block::LeafBlock2D)
    T = eltype(block.f)
    out = zeros(T, block.Ny)
    for jj in 1:block.Ny
        j = jj + 1
        s = zero(T)
        for i in kraken_e_i_range(block)
            s += block.ux[i,j]
        end
        out[jj] = s / T(block.Nx)
    end
    return out
end

function kraken_e_l2_over_scale(numerical, reference, scale)
    T = promote_type(eltype(numerical), eltype(reference), typeof(scale))
    err2 = zero(T)
    for idx in eachindex(numerical, reference)
        d = T(numerical[idx]) - T(reference[idx])
        err2 += d * d
    end
    return sqrt(err2 / T(length(numerical))) / T(scale)
end

function kraken_e_kinetic_energy(block::LeafBlock2D)
    T = eltype(block.f)
    e = zero(T)
    for j in kraken_e_j_range(block), i in kraken_e_i_range(block)
        e += T(0.5) * block.ρ[i,j] * (block.ux[i,j]^2 + block.uy[i,j]^2)
    end
    return e
end

function kraken_e_mass(block::LeafBlock2D)
    T = eltype(block.f)
    m = zero(T)
    for j in kraken_e_j_range(block), i in kraken_e_i_range(block)
        m += block.ρ[i,j]
    end
    return m
end
