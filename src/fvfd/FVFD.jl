using KernelAbstractions

include("specs.jl")
include("lowering_2d.jl")
include("operators_2d.jl")
include("operators_3d.jl")
include("operators_3d_openbc.jl")
include("operators_3d_velocity_gradient.jl")
include("halfway_wall_gradient_correction_2d.jl")
include("muscl_boundary.jl")
