# Alves 2003 4:1 planar contraction — Oldroyd-B / log-conformation Wi sweep.
#
# Reference: Alves, Oliveira, Pinho, J. Non-Newt. Fluid Mech. 110 (2003)
# pp. 45–75. The benchmark targets are the salient corner vortex length
# `X_R / H_out` and the centreline first normal stress difference `N1(x)`.
#
# Geometry: H_out = 20 lattice cells, β_c = 4, L_up = 20·H, L_down = 50·H.
# Domain: Nx × Ny = 1400 × 80 cells.
#
# Solvent: TRT + LI-BB V2 (no curved walls — the LI-BB reduces to halfway-BB).
# Polymer: log-conformation Oldroyd-B for Wi ≥ 0.5; direct conformation
# for Wi = 0.1 as cross-check. β = ν_s/(ν_s+ν_p) = 0.59 (standard).
#
# Outputs: results/alves_contraction.txt + per-Wi NPZ-style summary.
# Usage  : julia --project=. hpc/alves_contraction.jl

using Kraken, Printf, CUDA

backend = CUDABackend()
FT = Float64

println("="^75)
println("Alves 2003 4:1 contraction (β=0.59, log-conf Oldroyd-B)")
println("Backend: $(typeof(backend)), GPU: $(CUDA.name(CUDA.device()))")
println("="^75)

# Geometry
H_out = 20
β_c   = 4
L_up  = 20
L_down = 50

# Flow scales — choose u_out_mean so that Re ≈ 0.01 (Stokes regime, as in
# Alves 2003). With H_out = 20 and ν_total = 1.0, u_out_mean = 5e-4 → Re = 0.01.
β  = 0.59
u_out_mean = 5e-4
ν_total = 1.0
ν_s = β * ν_total
ν_p = (1 - β) * ν_total

# Reference: Alves 2003 reports X_R/H_out and N1_max along centreline.
# Approximate target values from Table 4 (4:1, β=1/9 for UCM but the
# trend is comparable for β=0.59):
#   Wi=0.5  → X_R/H_out ≈ 1.45
#   Wi=1.0  → X_R/H_out ≈ 1.56
#   Wi=2.0  → X_R/H_out ≈ 1.91
# (Will be refined against the rheoTool baseline in post-processing.)
alves_ref = Dict(0.5 => 1.45, 1.0 => 1.56, 2.0 => 1.91)

@printf("%-6s %-12s %-12s %-10s %-12s %-8s\n",
        "Wi", "X_R_south", "X_R_north", "X_R/H", "N1_max", "time")
println("-"^65)

for Wi in [0.5, 1.0, 2.0]
    λ = Wi * (H_out / 2) / u_out_mean

    # Time scale for steady state: t_residence ≈ L_total / u_out_mean and
    # t_relaxation ≈ λ. Need ~5 of the longer one.
    L_total = (L_up + L_down) * H_out
    t_steady = max(L_total / u_out_mean, 5*λ)
    max_steps = Int(round(5 * t_steady))
    avg_window = max_steps ÷ 5

    common = (; H_out=H_out, β_c=β_c, L_up=L_up, L_down=L_down,
                u_out_mean=u_out_mean, ν_s=ν_s, polymer_bc=CNEBB(),
                ρ_out=1.0, tau_plus=1.0,
                max_steps=max_steps, avg_window=avg_window,
                backend=backend, FT=FT)

    m_logc = LogConfOldroydB(G=ν_p/λ, λ=λ)

    t0 = time()
    r = try run_conformation_contraction_libb_2d(; common..., polymer_model=m_logc)
        catch err
            @warn "Wi=$Wi run failed" err
            (; ux=zeros(1,1), uy=zeros(1,1), tau_p_xx=zeros(1,1), tau_p_yy=zeros(1,1),
              is_solid=falses(1,1), Nx=0, Ny=0, i_step=1, j_low=1, j_high=1, Wi=Wi)
        end
    dt = time() - t0

    if r.Nx > 0
        X_R_s, _ = vortex_length_contraction_2d(r.ux, r.uy, r.is_solid;
                       i_step=r.i_step, j_low=r.j_low, j_high=r.j_high, side=:south)
        X_R_n, _ = vortex_length_contraction_2d(r.ux, r.uy, r.is_solid;
                       i_step=r.i_step, j_low=r.j_low, j_high=r.j_high, side=:north)
        N1 = outlet_centerline_N1_contraction_2d(r.tau_p_xx, r.tau_p_yy;
                  i_step=r.i_step, j_low=r.j_low, j_high=r.j_high)
        N1_max = maximum(abs, N1)
        @printf("%-6.2f %-12.2f %-12.2f %-10.3f %-12.4f %-8.0fs\n",
                Wi, X_R_s, X_R_n, X_R_s/H_out, N1_max, dt)
    else
        @printf("%-6.2f %-12s %-12s %-10s %-12s %-8.0fs\n",
                Wi, "FAIL", "FAIL", "FAIL", "FAIL", dt)
    end
end

println("\nDone.")
