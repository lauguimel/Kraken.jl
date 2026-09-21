# K1 (probe A): the suite's C0 path with the reverse call FIRST, forward after.
include(joinpath(@__DIR__, "lib_kraken.jl"))
c = FAST
p, geom = build_case(c)
w_star = base_state(c, p, geom; converge=true)
u, v = seeds(length(w_star))
println("[k1] reverse: compile+run (_ad_ve_vjp_GtT)"); flush(stdout)
Jtv = K._ad_ve_vjp_GtT(w_star, v, geom.g, geom.q_wall, geom.u_profile, p)
println("[k1] REVERSE_OK"); flush(stdout)
println("[k1] forward: compile+run (suite _jvp)"); flush(stdout)
Ju = suite_jvp(w_star, u, geom, p)
println("[k1] FORWARD_OK"); flush(stdout)
rel = abs(dot(v, Ju) - dot(Jtv, u)) / max(abs(dot(v, Ju)), eps(Float64))
println("[k1] transpose_rel = $rel ", rel < 1e-10 ? "IDENTITY_OK" : "IDENTITY_FAIL")
