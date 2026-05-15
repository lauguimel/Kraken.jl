function kraken_e_exchange_halo_periodic_x!(block::LeafBlock2D)
    f = block.f
    west = 1
    east = block.Nx + 2
    west_src = block.Nx + 1
    east_src = 2
    for q in 1:9, j in 1:(block.Ny + 2)
        f[west, j, q] = f[west_src, j, q]
        f[east, j, q] = f[east_src, j, q]
    end
    return block
end

function kraken_e_exchange_halo_periodic_xy!(block::LeafBlock2D)
    kraken_e_exchange_halo_periodic_x!(block)
    f = block.f
    south = 1
    north = block.Ny + 2
    south_src = block.Ny + 1
    north_src = 2
    for q in 1:9, i in 1:(block.Nx + 2)
        f[i, south, q] = f[i, south_src, q]
        f[i, north, q] = f[i, north_src, q]
    end
    return block
end

function kraken_e_halfway_bounce_back_y!(block::LeafBlock2D)
    f = block.f
    south = 1
    north = block.Ny + 2
    bottom = 2
    top = block.Ny + 1
    for i in kraken_e_i_range(block)
        west = kraken_e_west_i(block, i)
        east = kraken_e_east_i(block, i)

        f[i, south, 3] = f[i, bottom, 5]
        f[i, south, 6] = f[east, bottom, 8]
        f[i, south, 7] = f[west, bottom, 9]

        f[i, north, 5] = f[i, top, 3]
        f[i, north, 8] = f[west, top, 6]
        f[i, north, 9] = f[east, top, 7]
    end
    return block
end

function kraken_e_moving_wall_bounce_back_y!(block::LeafBlock2D, u_top)
    f = block.f
    T = eltype(f)
    U = T(u_top)
    south = 1
    north = block.Ny + 2
    bottom = 2
    top = block.Ny + 1
    for i in kraken_e_i_range(block)
        west = kraken_e_west_i(block, i)
        east = kraken_e_east_i(block, i)

        f[i, south, 3] = f[i, bottom, 5]
        f[i, south, 6] = f[east, bottom, 8]
        f[i, south, 7] = f[west, bottom, 9]

        ρ_west = kraken_e_density_at(f, west, top)
        ρ_mid = kraken_e_density_at(f, i, top)
        ρ_east = kraken_e_density_at(f, east, top)
        f[i, north, 5] = f[i, top, 3]
        f[i, north, 8] = f[west, top, 6] - ρ_west * U / T(6)
        f[i, north, 9] = f[east, top, 7] + ρ_east * U / T(6)
        block.ρ[i, top] = ρ_mid
    end
    return block
end

function kraken_e_apply_bcs!(block::LeafBlock2D; kind::Symbol=:none, u_top=0.0)
    if kind === :none || kind === :periodic_xy
        return block
    elseif kind === :bounce_back_y || kind === :poiseuille
        return kraken_e_halfway_bounce_back_y!(block)
    elseif kind === :couette
        return kraken_e_moving_wall_bounce_back_y!(block, u_top)
    else
        throw(ArgumentError("unknown Kraken-E S2 BC kind: $(kind)"))
    end
end

function kraken_e_exchange_halo!(block::LeafBlock2D; kind::Symbol=:none)
    if kind === :none
        return block
    elseif kind === :periodic_x
        return kraken_e_exchange_halo_periodic_x!(block)
    elseif kind === :periodic_xy
        return kraken_e_exchange_halo_periodic_xy!(block)
    else
        throw(ArgumentError("unknown Kraken-E S2 halo exchange kind: $(kind)"))
    end
end
