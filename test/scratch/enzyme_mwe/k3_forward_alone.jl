# K3: forward JVP only (separates the two thunks on the Julia 1.12 compile-time crash).
include(joinpath(@__DIR__, "lib_kraken.jl"))
c = FAST
p, geom = build_case(c)
w_star = base_state(c, p, geom; converge=true)
u, v = seeds(length(w_star))
println("[k3] forward: compile+run (suite _jvp)"); flush(stdout)
Ju = suite_jvp(w_star, u, geom, p)
println("[k3] FORWARD_OK  norm(Ju)=$(norm(Ju))"); flush(stdout)
