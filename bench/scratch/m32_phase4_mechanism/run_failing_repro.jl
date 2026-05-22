using Kraken
using KernelAbstractions

mkpath(get(ENV, "KRAKEN_OUTPUT_DIR", joinpath(pwd(), "tmp", "m32_phase4_mechanism")))

result = Kraken.run_viscoelastic_logfv_cylinder_coupled_2d(;
    radius=30,
    H=120,
    L_up=15,
    L_down=15,
    nu_s=0.177,
    nu_p=0.123,
    lambda=6000.0,
    u_mean=0.005,
    Fx_body=0.0,
    bsd_fraction=1.0,
    polymer_substeps=:auto,
    max_steps=200,
    avg_window=100,
    drag_stride=1,
    diagnostic_stride=0,
    embedded_gradient=false,
    embedded_advection=false,
    embedded_force=false,
    embedded_drag=false,
    embedded_geometry=:qwall,
    force_boundary_fill=:bc_aware,
    advection_scheme=:rusanov,
    wall_bc=:halfwayBB,
    backend=KernelAbstractions.CPU(),
    T=Float64,
)

@show result.Cd result.Cd_s result.Cd_p result.Cd_bsd
