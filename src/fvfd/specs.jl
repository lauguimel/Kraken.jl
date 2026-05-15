const FVFD_BC_PERIODIC = UInt8(1)
const FVFD_BC_OPEN = UInt8(2)
const FVFD_BC_WALL = UInt8(3)

struct FVFDDomainBC2D
    west::UInt8
    east::UInt8
    south::UInt8
    north::UInt8
end

struct FVFDFieldBC2D{W,E,S,N}
    west::W
    east::E
    south::S
    north::N
end

struct FVFDPatch2D{T}
    dx::T
    dy::T
    level::Int
end

struct FVFDGeometry2D{M,E,P,B}
    is_solid::M
    embedded::E
    patch::P
    bc::B
end

function fvfd_domain_bc_code(bc)
    if bc isa Integer
        code = UInt8(bc)
        if code == FVFD_BC_PERIODIC || code == FVFD_BC_OPEN || code == FVFD_BC_WALL
            return code
        end
        throw(ArgumentError("unsupported FVFD BC code $(bc)"))
    end
    normalized = Symbol(replace(lowercase(String(bc)), '-' => '_'))
    normalized in (:periodic, :wrap) && return FVFD_BC_PERIODIC
    normalized in (:open, :inlet, :outlet, :inflow, :outflow, :neumann, :zero_gradient) &&
        return FVFD_BC_OPEN
    normalized in (:wall, :wally, :no_normal_flux, :symmetry, :closed) && return FVFD_BC_WALL
    throw(ArgumentError("unsupported FVFD BC $(bc); expected :periodic, :open, or :wall"))
end

function FVFDDomainBC2D(; west=:periodic, east=west, south=:wall, north=south)
    return FVFDDomainBC2D(
        fvfd_domain_bc_code(west),
        fvfd_domain_bc_code(east),
        fvfd_domain_bc_code(south),
        fvfd_domain_bc_code(north),
    )
end

FVFDFieldBC2D(; west, east, south, north) = FVFDFieldBC2D(west, east, south, north)

function _fvfd_check_boundary_length(name::Symbol, values, expected::Integer)
    observed = try
        length(values)
    catch
        throw(DimensionMismatch(
            "$(name) boundary does not provide a length; expected $(expected)",
        ))
    end
    observed == expected || throw(DimensionMismatch(
        "$(name) boundary length $(observed) does not match expected $(expected)",
    ))
    return nothing
end

function fvfd_validate_field_bc_2d(
    field_bc::FVFDFieldBC2D, Nx::Integer, Ny::Integer, bc::FVFDDomainBC2D;
    name::Symbol=:field_bc,
)
    bc.west == FVFD_BC_OPEN &&
        _fvfd_check_boundary_length(Symbol(name, :_west), field_bc.west, Ny)
    bc.east == FVFD_BC_OPEN &&
        _fvfd_check_boundary_length(Symbol(name, :_east), field_bc.east, Ny)
    bc.south == FVFD_BC_OPEN &&
        _fvfd_check_boundary_length(Symbol(name, :_south), field_bc.south, Nx)
    bc.north == FVFD_BC_OPEN &&
        _fvfd_check_boundary_length(Symbol(name, :_north), field_bc.north, Nx)
    return nothing
end

fvfd_periodicx_wally_bcspec_2d() =
    FVFDDomainBC2D(; west=:periodic, east=:periodic, south=:wall, north=:wall)

fvfd_openx_wally_bcspec_2d() =
    FVFDDomainBC2D(; west=:open, east=:open, south=:wall, north=:wall)

fvfd_wallxwally_bcspec_2d() =
    FVFDDomainBC2D(; west=:wall, east=:wall, south=:wall, north=:wall)

FVFDPatch2D(dx::Real, dy::Real; level::Integer=0) =
    FVFDPatch2D(dx, dy, Int(level))
