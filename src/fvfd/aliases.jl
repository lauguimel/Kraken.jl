const LOGFV_BC_PERIODIC = FVFD_BC_PERIODIC
const LOGFV_BC_OPEN = FVFD_BC_OPEN
const LOGFV_BC_WALL = FVFD_BC_WALL
const LogFVDomainBC2D = FVFDDomainBC2D
const LogFVFieldBC2D = FVFDFieldBC2D
const LogFVEmbeddedBoundary2D = FVFDEmbeddedBoundary2D

const logfv_domain_bc_code = fvfd_domain_bc_code
logfv_periodicx_wally_bcspec_2d() = fvfd_periodicx_wally_bcspec_2d()
logfv_openx_wally_bcspec_2d() = fvfd_openx_wally_bcspec_2d()
logfv_wallxwally_bcspec_2d() = fvfd_wallxwally_bcspec_2d()
logfv_empty_embedded_boundary_2d(args...; kwargs...) =
    fvfd_empty_embedded_boundary_2d(args...; kwargs...)
logfv_embedded_boundary_from_qwall_2d(args...; kwargs...) =
    fvfd_embedded_boundary_from_qwall_2d(args...; kwargs...)
logfv_transfer_embedded_boundary_2d(args...; kwargs...) =
    fvfd_transfer_embedded_boundary_2d(args...; kwargs...)
logfv_transfer_field_bc_2d(args...; kwargs...) =
    fvfd_transfer_field_bc_2d(args...; kwargs...)
