struct TwoBlockPatch2D{T<:AbstractFloat,B<:LeafBlock2D{T}}
    coarse::B
    fine::B
    cf_records::Vector{CFFaceRecord2D{T}}
end

function build_two_block_patch_2d(
    ::Type{T}=Float64;
    Nx_c::Int=8, Ny_c::Int=8, dx_c::T=one(T),
    Nx_f::Int=16, Ny_f::Int=16,
) where {T<:AbstractFloat}
    Nx_c > 0 || throw(ArgumentError("Nx_c must be positive"))
    Ny_c > 0 || throw(ArgumentError("Ny_c must be positive"))
    Nx_f > 0 || throw(ArgumentError("Nx_f must be positive"))
    Ny_f == 2 * Ny_c || throw(ArgumentError("Ny_f must equal 2 * Ny_c"))

    coarse = allocate_leaf_block_2d(
        T; Nx=Nx_c, Ny=Ny_c, dx=dx_c, origin=(zero(T), zero(T)), id=1,
    )
    dx_f = dx_c / T(2)
    fine = allocate_leaf_block_2d(
        T; Nx=Nx_f, Ny=Ny_f, dx=dx_f, origin=(T(Nx_c) * dx_c, zero(T)), id=2,
    )

    records = Vector{CFFaceRecord2D{T}}(undef, Ny_c)
    for J in 1:Ny_c
        records[J] = kraken_e_build_cf_face_record_2d(
            T;
            coarse_block_id=1,
            fine_block_id=2,
            coarse_index=(Nx_c, J),
            fine_indices=((1, 2J - 1), (1, 2J)),
            axis=KRAKEN_E_CF_FACE_X,
            side=KRAKEN_E_CF_FACE_HI,
            coarse_origin=(zero(T), zero(T)),
            coarse_dx=dx_c,
        )
    end

    return TwoBlockPatch2D{T,typeof(coarse)}(coarse, fine, records)
end

function patch_total_mass(
    patch::TwoBlockPatch2D{T}, U_c::AbstractArray{T,2}, U_f::AbstractArray{T,2},
) where {T<:AbstractFloat}
    return sum(U_c) * patch.coarse.dx^2 + sum(U_f) * patch.fine.dx^2
end

@inline function _cf_fine_subface_flux_2d(
    U_c::AbstractArray{T,2}, U_f::AbstractArray{T,2},
    coarse_index::NTuple{2,Int}, fine_index::NTuple{2,Int}, vx::T,
) where {T<:AbstractFloat}
    ic, jc = coarse_index
    ifine, jfine = fine_index
    return vx > zero(T) ? vx * U_c[ic, jc] : vx * U_f[ifine, jfine]
end

function explicit_euler_step!(
    patch::TwoBlockPatch2D{T},
    U_c::AbstractArray{T,2}, U_f::AbstractArray{T,2},
    flux_c::ScalarFluxField2D{T}, flux_f::ScalarFluxField2D{T},
    vx::T, vy::T, dt::T,
)::T where {T<:AbstractFloat}
    Nx_c, Ny_c = size(U_c)
    Nx_f, Ny_f = size(U_f)
    Nx_c == patch.coarse.Nx && Ny_c == patch.coarse.Ny ||
        throw(DimensionMismatch("coarse scalar field does not match patch coarse block"))
    Nx_f == patch.fine.Nx && Ny_f == patch.fine.Ny ||
        throw(DimensionMismatch("fine scalar field does not match patch fine block"))

    compute_same_level_upwind_fluxes_2d!(
        flux_c, U_c, vx, vy; skip_east=true, closed_x=true, periodic_y=true,
    )
    compute_same_level_upwind_fluxes_2d!(
        flux_f, U_f, vx, vy; skip_west=true, closed_x=true, periodic_y=true,
    )

    max_telescope_err = zero(T)
    for record in patch.cf_records
        F_fine_1 = _cf_fine_subface_flux_2d(
            U_c, U_f, record.coarse_index, record.fine_indices[1], vx,
        )
        F_fine_2 = _cf_fine_subface_flux_2d(
            U_c, U_f, record.coarse_index, record.fine_indices[2], vx,
        )
        F_coarse = reconstruct_coarse_flux_from_fine_2d(record, F_fine_1, F_fine_2)

        ic, jc = record.coarse_index
        if1, jf1 = record.fine_indices[1]
        if2, jf2 = record.fine_indices[2]
        flux_c.east[ic + 1, jc] = F_coarse
        flux_f.east[if1, jf1] = F_fine_1
        flux_f.east[if2, jf2] = F_fine_2

        max_telescope_err = max(
            max_telescope_err,
            cf_flux_telescoping_error(record, F_fine_1, F_fine_2),
        )
    end

    dx_c = patch.coarse.dx
    dx_f = patch.fine.dx
    inv_dx_c = one(T) / dx_c
    inv_dx_f = one(T) / dx_f

    for j in 1:Ny_c, i in 1:Nx_c
        U_c[i, j] -= dt * (
            (flux_c.east[i + 1, j] - flux_c.east[i, j]) +
            (flux_c.north[i, j + 1] - flux_c.north[i, j])
        ) * inv_dx_c
    end

    for j in 1:Ny_f, i in 1:Nx_f
        U_f[i, j] -= dt * (
            (flux_f.east[i + 1, j] - flux_f.east[i, j]) +
            (flux_f.north[i, j + 1] - flux_f.north[i, j])
        ) * inv_dx_f
    end

    return max_telescope_err
end
