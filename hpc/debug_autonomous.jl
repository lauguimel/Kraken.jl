# Autonomous debug on Aqua H100 — systematic narrowing of the Cd_VE/Cd_Newt
# ratio deviation at Wi → 0 (should be 1.000, measured 0.934).
#
# Hypotheses to test:
#   H1: Resolution (R) — does the gap close at higher R?
#   H2: LI-BB V2 vs halfway-BB cylinder solver
#   H3: Polymer wall BC (CNEBB vs None)
#   H4: Inlet profile (Poiseuille vs plug)
#   H5: β (viscosity ratio) — does the gap scale with ν_p?
#   H6: The flow-field time-to-steady (insufficient convergence)
#   H7: log-conformation vs direct-C

using Kraken, Printf, CUDA
const backend = CUDABackend()
const FT = Float64

println("="^72)
println("AUTONOMOUS DEBUG — Cd_VE/Cd_Newt at Wi→0 should be 1.0")
println("GPU: ", CUDA.name(CUDA.device()))
println("="^72)

# Reference Newtonian Cd at matching setup (parabolic inlet, LI-BB V2)
function ref_newt(R, u_mean, ν_total, steps)
    r = run_cylinder_libb_2d(; Nx=30R, Ny=4R, radius=R,
            u_in=FT(1.5*u_mean), ν=FT(ν_total),
            max_steps=steps, avg_window=steps÷5,
            inlet=:parabolic, backend=backend, T=FT)
    r.Cd
end

function ref_newt_hwBB(R, u_mean, ν_total, steps)
    r = run_cylinder_2d(; Nx=30R, Ny=4R, radius=R, u_in=u_mean, ν=ν_total,
            max_steps=steps, avg_window=steps÷5, backend=backend, T=FT)
    r.Cd
end

function ve_libb(R, u_mean, ν_s, ν_p, λ, steps; pbc=CNEBB(), inlet=:parabolic)
    r = run_conformation_cylinder_libb_2d(;
            Nx=30R, Ny=4R, radius=R, cx=15R, cy=2R,
            u_mean=u_mean, ν_s=ν_s, ν_p=ν_p, lambda=λ, tau_plus=1.0,
            polymer_bc=pbc, inlet=inlet,
            max_steps=steps, avg_window=steps÷5,
            backend=backend, FT=FT)
    r.Cd
end

function ve_hwBB(R, u_mean, ν_s, ν_p, λ, steps)
    r = run_conformation_cylinder_2d(; Nx=30R, Ny=4R, radius=R,
            u_in=u_mean, ν_s=ν_s, ν_p=ν_p, lambda=λ, tau_plus=1.0,
            max_steps=steps, avg_window=steps÷5, backend=backend, FT=FT)
    r.Cd
end

# ---------------------------------------------------------------------
# H1: Resolution sweep (Wi=0.001, β=0.59 fixed)
# ---------------------------------------------------------------------
println("\n### H1: Resolution sweep (Wi=0.001, β=0.59, LI-BB V2)")
@printf("%-5s %-10s %-10s %-8s\n", "R", "Cd_Newt", "Cd_VE", "ratio")
for R in [16, 24, 32, 48]
    u_mean = 0.02; ν_total = u_mean * R / 1.0
    ν_s = 0.59 * ν_total; ν_p = ν_total - ν_s
    λ = 0.001 * R / u_mean
    steps = round(Int, 100_000 * (R/20)^1.5); steps = min(steps, 400_000)
    cN = ref_newt(R, u_mean, ν_total, steps)
    cV = ve_libb(R, u_mean, ν_s, ν_p, λ, steps)
    @printf("%-5d %-10.3f %-10.3f %-8.4f\n", R, cN, cV, cV/cN)
end

# ---------------------------------------------------------------------
# H2: LI-BB V2 vs halfway-BB (same setup at R=20)
# ---------------------------------------------------------------------
println("\n### H2: LI-BB V2 vs halfway-BB @ Wi=0.001, R=20, β=0.59")
R = 20; u_mean = 0.02; ν_total = 0.4; ν_s = 0.236; ν_p = 0.164; λ = 1.0
steps = 100_000
cN_libb = ref_newt(R, u_mean, ν_total, steps)
cN_hwbb = ref_newt_hwBB(R, u_mean, ν_total, steps)
cV_libb = ve_libb(R, u_mean, ν_s, ν_p, λ, steps)
cV_hwbb = ve_hwBB(R, u_mean, ν_s, ν_p, λ, steps)
@printf("LI-BB V2:   Cd_Newt=%.3f  Cd_VE=%.3f  ratio=%.4f\n", cN_libb, cV_libb, cV_libb/cN_libb)
@printf("halfway-BB: Cd_Newt=%.3f  Cd_VE=%.3f  ratio=%.4f\n", cN_hwbb, cV_hwbb, cV_hwbb/cN_hwbb)

# ---------------------------------------------------------------------
# H3: Polymer wall BC (CNEBB vs None) @ R=30
# ---------------------------------------------------------------------
println("\n### H3: Polymer wall BC @ Wi=0.001, R=30, β=0.59")
R = 30; u_mean = 0.02; ν_total = u_mean * R; ν_s = 0.59*ν_total; ν_p = ν_total-ν_s
λ = 0.001 * R / u_mean; steps = 200_000
cN = ref_newt(R, u_mean, ν_total, steps)
for (label, pbc) in [("CNEBB", CNEBB()), ("None", NoPolymerWallBC())]
    cV = ve_libb(R, u_mean, ν_s, ν_p, λ, steps; pbc=pbc)
    @printf("%-8s Cd_VE=%.3f  ratio=%.4f\n", label, cV, cV/cN)
end

# ---------------------------------------------------------------------
# H4: Inlet profile (parabolic vs uniform)
# ---------------------------------------------------------------------
println("\n### H4: Inlet profile @ Wi=0.001, R=30, β=0.59")
for inlet in [:parabolic, :uniform]
    cV = ve_libb(R, u_mean, ν_s, ν_p, λ, steps; inlet=inlet)
    cN_i = if inlet === :parabolic
        ref_newt(R, u_mean, ν_total, steps)
    else
        ref_newt_hwBB(R, u_mean, ν_total, steps)
    end
    @printf("%-10s Cd_Newt=%.3f  Cd_VE=%.3f  ratio=%.4f\n",
            string(inlet), cN_i, cV, cV/cN_i)
end

# ---------------------------------------------------------------------
# H5: β (viscosity ratio) sweep @ R=30
# ---------------------------------------------------------------------
println("\n### H5: β sweep @ Wi=0.001, R=30")
R = 30; u_mean = 0.02; ν_total = u_mean * R; λ = 0.001 * R / u_mean
steps = 200_000
cN = ref_newt(R, u_mean, ν_total, steps)
@printf("%-5s %-8s %-8s %-10s %-8s\n", "β", "ν_s", "ν_p", "Cd_VE", "ratio")
for β in [0.9, 0.7, 0.59, 0.4, 0.2]
    ν_s = β * ν_total; ν_p = ν_total - ν_s
    cV = ve_libb(R, u_mean, ν_s, ν_p, λ, steps)
    @printf("%-5.2f %-8.3f %-8.3f %-10.3f %-8.4f\n", β, ν_s, ν_p, cV, cV/cN)
end

# ---------------------------------------------------------------------
# H6: Steady-state convergence (step count sweep @ R=30, β=0.59)
# ---------------------------------------------------------------------
println("\n### H6: Step count sweep @ Wi=0.001, R=30, β=0.59")
R = 30; u_mean = 0.02; ν_total = u_mean * R; ν_s = 0.59*ν_total; ν_p = ν_total-ν_s
λ = 0.001 * R / u_mean
@printf("%-10s %-10s %-8s\n", "steps", "Cd_VE", "ratio")
for st in [50_000, 100_000, 200_000, 400_000]
    cN_h6 = ref_newt(R, u_mean, ν_total, st)
    cV = ve_libb(R, u_mean, ν_s, ν_p, λ, st)
    @printf("%-10d %-10.3f %-8.4f\n", st, cV, cV/cN_h6)
end

# ---------------------------------------------------------------------
# H7: log-conf vs direct-C @ R=30, Wi=0.001
# ---------------------------------------------------------------------
println("\n### H7: log-conf vs direct-C @ Wi=0.001, R=30, β=0.59")
R = 30; u_mean = 0.02; ν_total = u_mean * R; ν_s = 0.59*ν_total; ν_p = ν_total-ν_s
λ = 0.001 * R / u_mean; steps = 200_000
cN = ref_newt(R, u_mean, ν_total, steps)
for (label, model) in [("direct", OldroydB(G=ν_p/λ, λ=λ)),
                         ("logconf", LogConfOldroydB(G=ν_p/λ, λ=λ))]
    r = run_conformation_cylinder_libb_2d(;
            Nx=30R, Ny=4R, radius=R, cx=15R, cy=2R,
            u_mean=u_mean, ν_s=ν_s,
            polymer_model=model, polymer_bc=CNEBB(),
            inlet=:parabolic, tau_plus=1.0,
            max_steps=steps, avg_window=steps÷5,
            backend=backend, FT=FT)
    @printf("%-10s Cd_VE=%.3f  ratio=%.4f\n", label, r.Cd, r.Cd/cN)
end

println("\n", "="^72)
println("AUTONOMOUS DEBUG COMPLETE")
println("="^72)
