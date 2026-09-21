# K5: reverse alone on ad_ve_coupled_step!, 12x12 grid, warm state (no solve).
include(joinpath(@__DIR__, "lib_kraken.jl"))
c = SMALL
p, geom = build_case(c)
w_star = base_state(c, p, geom; converge=false)
println("[k5] grid=$(c.Nx)x$(c.Ny) solid=$(count(geom.g.is_solid)) cut=$(count(>(0.0), geom.q_wall))")
u, v = seeds(length(w_star))
println("[k5] reverse: compile+run (_ad_ve_vjp_GtT)"); flush(stdout)
Jtv = K._ad_ve_vjp_GtT(w_star, v, geom.g, geom.q_wall, geom.u_profile, p)
println("[k5] REVERSE_OK"); flush(stdout)
Ju = suite_jvp(w_star, u, geom, p)
println("[k5] FORWARD_OK"); flush(stdout)
rel = abs(dot(v, Ju) - dot(Jtv, u)) / max(abs(dot(v, Ju)), eps(Float64))
println("[k5] transpose_rel = $rel ", rel < 1e-10 ? "IDENTITY_OK" : "IDENTITY_FAIL")
