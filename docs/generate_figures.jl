#!/usr/bin/env julia
# Generate all SVG figures for the multiphase examples documentation
# Run with: julia --project docs/generate_figures.jl

using Kraken
using CairoMakie
const KA = Kraken.KernelAbstractions

const OUTDIR = joinpath(@__DIR__, "src", "examples")

# ============================================================================
# 11. Zalesak Disk
# ============================================================================
println("=== 11. Zalesak Disk ===")

N = 100; R = 15.0; cx = 50.0; cy = 75.0; slot_w = 5.0

# Initial geometry
C0 = zeros(N, N)
for j in 1:N, i in 1:N
    x = i - 0.5; y = j - 0.5
    r = sqrt((x - cx)^2 + (y - cy)^2)
    disk = 0.5 * (1 - tanh((r - R) / 2))
    in_slot = abs(x - cx) < slot_w / 2 && y < cy && y > cy - R
    C0[i, j] = in_slot ? 0.0 : disk
end

fig = Figure(size=(480, 480))
ax = Axis(fig[1, 1]; title="Initial geometry — Zalesak disk",
          xlabel="x", ylabel="y", aspect=DataAspect())
hm = heatmap!(ax, 1:N, 1:N, C0'; colormap=:blues, colorrange=(0, 1))
contour!(ax, 1:N, 1:N, C0'; levels=[0.5], color=:red, linewidth=2)
Colorbar(fig[1, 2], hm; label="C")
save(joinpath(OUTDIR, "zalesak_geometry.svg"), fig)
println("  ✓ geometry")

# Run simulation
angular_vel = 2π / (N * π)
max_steps = round(Int, 2π / angular_vel)

function zalesak_init(x, y)
    r = sqrt((x - cx)^2 + (y - cy)^2)
    disk = 0.5 * (1 - tanh((r - R) / 2))
    in_slot = abs(x - cx) < slot_w / 2 && y < cy && y > cy - R
    return in_slot ? 0.0 : disk
end
vel_z(x, y, t) = (-(y - 50.0) * angular_vel, (x - 50.0) * angular_vel)

result = run_advection_2d(; Nx=N, Ny=N, max_steps=max_steps,
                           velocity_fn=vel_z, init_C_fn=zalesak_init)

# Before/after
fig2 = Figure(size=(960, 440))
ax1 = Axis(fig2[1, 1]; title="t = 0", xlabel="x", ylabel="y", aspect=DataAspect())
heatmap!(ax1, 1:N, 1:N, result.C0'; colormap=:blues, colorrange=(0, 1))
contour!(ax1, 1:N, 1:N, result.C0'; levels=[0.5], color=:red, linewidth=2)
ax2 = Axis(fig2[1, 2]; title="t = T (1 rotation)", xlabel="x", ylabel="y", aspect=DataAspect())
heatmap!(ax2, 1:N, 1:N, result.C'; colormap=:blues, colorrange=(0, 1))
contour!(ax2, 1:N, 1:N, result.C'; levels=[0.5], color=:red, linewidth=2)
save(joinpath(OUTDIR, "zalesak_before_after.svg"), fig2)
println("  ✓ before/after")

# Error map
fig3 = Figure(size=(480, 440))
ax3 = Axis(fig3[1, 1]; title="Local error |C(T) - C(0)|",
           xlabel="x", ylabel="y", aspect=DataAspect())
hm3 = heatmap!(ax3, 1:N, 1:N, abs.(result.C .- result.C0)';
               colormap=:inferno, colorrange=(0, 0.5))
Colorbar(fig3[1, 2], hm3; label="|ΔC|")
save(joinpath(OUTDIR, "zalesak_error_map.svg"), fig3)
println("  ✓ error map")

# Convergence
N_list = [50, 100, 200]
errors_z = Float64[]
for Ni in N_list
    ω_i = 2π / (Ni * π); steps_i = round(Int, 2π / ω_i)
    cx_i = Ni / 2; cy_i = 3Ni / 4; R_i = 0.15 * Ni; w_i = 0.05 * Ni
    init_i(x, y) = begin
        r = sqrt((x - cx_i)^2 + (y - cy_i)^2)
        disk = 0.5 * (1 - tanh((r - R_i) / 2))
        in_slot = abs(x - cx_i) < w_i / 2 && y < cy_i && y > cy_i - R_i
        in_slot ? 0.0 : disk
    end
    vel_i(x, y, t) = (-(y - Ni / 2) * ω_i, (x - Ni / 2) * ω_i)
    res = run_advection_2d(; Nx=Ni, Ny=Ni, max_steps=steps_i,
                            velocity_fn=vel_i, init_C_fn=init_i)
    push!(errors_z, sum(abs.(res.C .- res.C0)) / sum(res.C0))
end

fig4 = Figure(size=(520, 420))
ax4 = Axis(fig4[1, 1]; xlabel="N", ylabel="L₁ shape error",
           title="Convergence — Zalesak disk", xscale=log10, yscale=log10)
scatterlines!(ax4, Float64.(N_list), errors_z;
              linewidth=2, markersize=10, label="VOF (1st order)")
ref1 = errors_z[1] .* (N_list[1] ./ N_list) .^ 1
lines!(ax4, Float64.(N_list), ref1; linestyle=:dash, color=:gray, label="slope 1")
axislegend(ax4; position=:rt)
save(joinpath(OUTDIR, "zalesak_convergence.svg"), fig4)
println("  ✓ convergence — DONE")

# ============================================================================
# 12. Reversed Vortex
# ============================================================================
println("=== 12. Reversed Vortex ===")

N = 128; R_v = 0.15 * N; cx_v = 0.5 * N; cy_v = 0.75 * N
T_period = 8.0 * N

# Geometry with velocity arrows
C0_v = zeros(N, N)
for j in 1:N, i in 1:N
    x = i - 0.5; y = j - 0.5
    C0_v[i, j] = 0.5 * (1 - tanh((sqrt((x - cx_v)^2 + (y - cy_v)^2) - R_v) / 2))
end

fig_g = Figure(size=(500, 480))
ax_g = Axis(fig_g[1, 1]; title="Initial circle + velocity field",
            xlabel="x", ylabel="y", aspect=DataAspect())
heatmap!(ax_g, 1:N, 1:N, C0_v'; colormap=:blues, colorrange=(0, 1))
xs = 8:16:N; ys = 8:16:N
ux_arr = [-sin(π * x / N) * cos(π * y / N) for x in xs, y in ys]
uy_arr = [ cos(π * x / N) * sin(π * y / N) for x in xs, y in ys]
arrows!(ax_g, repeat(Float64.(collect(xs)), outer=length(ys)),
        repeat(Float64.(collect(ys)), inner=length(xs)),
        vec(ux_arr) .* 10, vec(uy_arr) .* 10;
        color=:white, linewidth=1.5, arrowsize=6)
save(joinpath(OUTDIR, "reversed_vortex_geometry.svg"), fig_g)
println("  ✓ geometry")

function vortex_velocity(x, y, t)
    xn = x / N; yn = y / N
    scale = cos(π * t / T_period) * 0.5
    return (-sin(π * xn) * cos(π * yn) * scale,
             cos(π * xn) * sin(π * yn) * scale)
end
init_v(x, y) = 0.5 * (1 - tanh((sqrt((x - cx_v)^2 + (y - cy_v)^2) - R_v) / 2))

max_steps_v = round(Int, T_period)
result_v = run_advection_2d(; Nx=N, Ny=N, max_steps=max_steps_v,
                             velocity_fn=vortex_velocity, init_C_fn=init_v)
result_half = run_advection_2d(; Nx=N, Ny=N, max_steps=max_steps_v ÷ 2,
                                velocity_fn=vortex_velocity, init_C_fn=init_v)

# 3 snapshots
fig_s = Figure(size=(1300, 440))
for (idx, (data, title)) in enumerate([(result_v.C0, "t = 0"),
                                        (result_half.C, "t = T/2 (max deformation)"),
                                        (result_v.C, "t = T (recovered)")])
    ax_s = Axis(fig_s[1, idx]; title=title, aspect=DataAspect())
    heatmap!(ax_s, 1:N, 1:N, data'; colormap=:blues, colorrange=(0, 1))
    contour!(ax_s, 1:N, 1:N, data'; levels=[0.5], color=:red, linewidth=2)
end
save(joinpath(OUTDIR, "reversed_vortex_snapshots.svg"), fig_s)
println("  ✓ snapshots")

# Error map
fig_e = Figure(size=(480, 440))
ax_e = Axis(fig_e[1, 1]; title="Recovery error |C(T) - C(0)|",
            xlabel="x", ylabel="y", aspect=DataAspect())
hm_e = heatmap!(ax_e, 1:N, 1:N, abs.(result_v.C .- result_v.C0)';
                colormap=:inferno, colorrange=(0, 0.5))
Colorbar(fig_e[1, 2], hm_e; label="|ΔC|")
save(joinpath(OUTDIR, "reversed_vortex_error.svg"), fig_e)
println("  ✓ error map")

# VOF vs CLSVOF
result_cls = run_advection_2d(; Nx=N, Ny=N, max_steps=max_steps_v,
                                velocity_fn=vortex_velocity, init_C_fn=init_v,
                                use_clsvof=true)
L1_vof = sum(abs.(result_v.C .- result_v.C0)) / sum(result_v.C0)
L1_cls = sum(abs.(result_cls.C .- result_cls.C0)) / sum(result_cls.C0)

fig_cmp = Figure(size=(520, 360))
ax_cmp = Axis(fig_cmp[1, 1]; xlabel="Method", ylabel="L₁ recovery error",
              title="Shape error after full reversal",
              xticks=(1:2, ["VOF", "CLSVOF"]))
barplot!(ax_cmp, [1, 2], [L1_vof, L1_cls]; color=[:steelblue, :coral])
save(joinpath(OUTDIR, "reversed_vortex_comparison.svg"), fig_cmp)
println("  ✓ comparison")

# Mass conservation
fig_m = Figure(size=(520, 360))
ax_m = Axis(fig_m[1, 1]; xlabel="Step", ylabel="Total C", title="Mass conservation")
lines!(ax_m, 0:length(result_v.mass_history)-1, result_v.mass_history;
       label="VOF", linewidth=2)
lines!(ax_m, 0:length(result_cls.mass_history)-1, result_cls.mass_history;
       label="CLSVOF", linewidth=2, color=:coral)
axislegend(ax_m; position=:rt)
save(joinpath(OUTDIR, "reversed_vortex_mass.svg"), fig_m)
println("  ✓ mass — DONE")

# ============================================================================
# 13. Capillary Wave
# ============================================================================
println("=== 13. Capillary Wave ===")

Nx_c = 128; Ny_c = 256
λ_c = Float64(Nx_c); H_c = Float64(Ny_c)
σ_c = 0.01; ν_c = 0.1; ρ_l_c = 1.0; ρ_g_c = 0.1; a0_c = 1.0
k_c = 2π / λ_c
ω_ana = sqrt(σ_c * k_c^3 / (ρ_l_c + ρ_g_c))
γ_ana = 2ν_c * k_c^2
T_osc = 2π / ω_ana
max_steps_c = round(Int, 3 * T_osc)

# Geometry
C0_c = zeros(Nx_c, Ny_c)
for j in 1:Ny_c, i in 1:Nx_c
    x = i - 0.5; y = j - 0.5
    y_int = H_c / 2 + a0_c * cos(k_c * x)
    C0_c[i, j] = 0.5 * (1 - tanh((y - y_int) / 2))
end

fig_cg = Figure(size=(360, 600))
ax_cg = Axis(fig_cg[1, 1]; title="Initial interface", xlabel="x", ylabel="y", aspect=DataAspect())
hm_cg = heatmap!(ax_cg, 1:Nx_c, 1:Ny_c, C0_c; colormap=:blues, colorrange=(0, 1))
Colorbar(fig_cg[1, 2], hm_cg; label="C")
save(joinpath(OUTDIR, "capwave_geometry.svg"), fig_cg)
println("  ✓ geometry")

# Run simulation
config_c = LBMConfig(D2Q9(); Nx=Nx_c, Ny=Ny_c, ν=ν_c, u_lid=0.0, max_steps=max_steps_c)
state_c = initialize_2d(config_c, Float64; backend=KA.CPU())
f_in_c, f_out_c = state_c.f_in, state_c.f_out
ρ_c, ux_c, uy_c = state_c.ρ, state_c.ux, state_c.uy
is_solid_c = state_c.is_solid

C_c = zeros(Float64, Nx_c, Ny_c); C_new_c = zeros(Float64, Nx_c, Ny_c)
nx_c = zeros(Float64, Nx_c, Ny_c); ny_c = zeros(Float64, Nx_c, Ny_c)
κ_c = zeros(Float64, Nx_c, Ny_c)
Fx_c = zeros(Float64, Nx_c, Ny_c); Fy_c = zeros(Float64, Nx_c, Ny_c)

for j in 1:Ny_c, i in 1:Nx_c
    x = i - 0.5; y = j - 0.5
    C_c[i, j] = 0.5 * (1 - tanh((y - H_c / 2 - a0_c * cos(k_c * x)) / 2))
end

w_c = weights(D2Q9())
f_cpu_c = zeros(Float64, Nx_c, Ny_c, 9)
for j in 1:Ny_c, i in 1:Nx_c
    ρ_init = C_c[i, j] * ρ_l_c + (1 - C_c[i, j]) * ρ_g_c
    for q in 1:9; f_cpu_c[i, j, q] = w_c[q] * ρ_init; end
end
copyto!(f_in_c, f_cpu_c); copyto!(f_out_c, f_cpu_c)

times_c = Float64[]; amps_c = Float64[]
for step in 1:max_steps_c
    stream_fully_periodic_2d!(f_out_c, f_in_c, Nx_c, Ny_c)
    compute_macroscopic_2d!(ρ_c, ux_c, uy_c, f_out_c)
    advect_vof_step!(C_c, C_new_c, ux_c, uy_c, Nx_c, Ny_c)
    copyto!(C_c, C_new_c)
    compute_vof_normal_2d!(nx_c, ny_c, C_c, Nx_c, Ny_c)
    compute_hf_curvature_2d!(κ_c, C_c, nx_c, ny_c, Nx_c, Ny_c)
    compute_surface_tension_2d!(Fx_c, Fy_c, κ_c, C_c, σ_c, Nx_c, Ny_c)
    collide_twophase_2d!(f_out_c, C_c, Fx_c, Fy_c, is_solid_c;
                         ρ_l=ρ_l_c, ρ_g=ρ_g_c, ν_l=ν_c, ν_g=ν_c)
    global f_in_c, f_out_c = f_out_c, f_in_c
    if step % 5 == 0
        i_probe = Nx_c ÷ 2; y_int = 0.0
        for j in 1:Ny_c-1
            if C_c[i_probe, j] > 0.5 && C_c[i_probe, j+1] <= 0.5
                y_int = (j - 0.5) + (C_c[i_probe, j] - 0.5) / (C_c[i_probe, j] - C_c[i_probe, j+1])
                break
            end
        end
        push!(times_c, Float64(step)); push!(amps_c, y_int - H_c / 2)
    end
end
println("  ✓ simulation")

# Oscillation plot
t_ana = range(0, maximum(times_c); length=500)
a_ana = a0_c .* exp.(-γ_ana .* t_ana) .* cos.(ω_ana .* t_ana)
fig_osc = Figure(size=(720, 440))
ax_osc = Axis(fig_osc[1, 1]; xlabel="Time (steps)", ylabel="Interface displacement a(t)",
              title="Capillary wave — Prosperetti (1981)")
lines!(ax_osc, collect(t_ana), collect(a_ana); label="Analytical", linewidth=2, color=:black)
scatter!(ax_osc, times_c, amps_c; label="LBM-VOF", markersize=4, color=:steelblue)
axislegend(ax_osc; position=:rt)
save(joinpath(OUTDIR, "capwave_oscillation.svg"), fig_osc)
println("  ✓ oscillation")

# Final fields
compute_macroscopic_2d!(ρ_c, ux_c, uy_c, f_in_c)
fig_cf = Figure(size=(720, 400))
ax_cc = Axis(fig_cf[1, 1]; title="C field at t = $max_steps_c", aspect=DataAspect())
heatmap!(ax_cc, 1:Nx_c, 1:Ny_c, C_c; colormap=:blues, colorrange=(0, 1))
umag_c = @. sqrt(ux_c^2 + uy_c^2)
umag_c[isnan.(umag_c)] .= 0.0
ax_cu = Axis(fig_cf[1, 2]; title="Velocity magnitude", aspect=DataAspect())
hm_cu = heatmap!(ax_cu, 1:Nx_c, 1:Ny_c, umag_c; colormap=:viridis)
Colorbar(fig_cf[1, 3], hm_cu; label="|u|")
save(joinpath(OUTDIR, "capwave_final.svg"), fig_cf)
println("  ✓ final fields — DONE")

# ============================================================================
# 14. Static Droplet
# ============================================================================
println("=== 14. Static Droplet ===")

N_d = 128; R_d = N_d ÷ 4

# Geometry
C0_d = zeros(N_d, N_d)
for j in 1:N_d, i in 1:N_d
    r = sqrt(Float64((i - N_d÷2)^2 + (j - N_d÷2)^2))
    C0_d[i, j] = 0.5 * (1 - tanh((r - R_d) / 2))
end
fig_dg = Figure(size=(500, 480))
ax_dg = Axis(fig_dg[1, 1]; title="Initial droplet — R = $R_d",
             xlabel="x", ylabel="y", aspect=DataAspect())
hm_dg = heatmap!(ax_dg, 1:N_d, 1:N_d, C0_d'; colormap=:blues, colorrange=(0, 1))
contour!(ax_dg, 1:N_d, 1:N_d, C0_d'; levels=[0.5], color=:red, linewidth=2)
Colorbar(fig_dg[1, 2], hm_dg; label="C")
save(joinpath(OUTDIR, "droplet_geometry.svg"), fig_dg)
println("  ✓ geometry")

# VOF run
rv = run_static_droplet_2d(; N=N_d, R=R_d, σ=0.01, ν=0.1, ρ_l=1.0, ρ_g=0.001, max_steps=5000)

# Results
fig_dr = Figure(size=(1050, 440))
ax_d1 = Axis(fig_dr[1, 1]; title="Volume fraction C", aspect=DataAspect())
hm_d1 = heatmap!(ax_d1, 1:N_d, 1:N_d, rv.C'; colormap=:blues, colorrange=(0, 1))
contour!(ax_d1, 1:N_d, 1:N_d, rv.C'; levels=[0.5], color=:red, linewidth=2)

p_line = rv.ρ[N_d÷2, :] ./ 3
p_line[isnan.(p_line)] .= 0.0
ax_d2 = Axis(fig_dr[1, 2]; title="Pressure (y-centreline)", xlabel="y", ylabel="p = ρ/3")
lines!(ax_d2, 1:N_d, p_line; linewidth=2, color=:steelblue)
vlines!(ax_d2, [N_d÷2 - R_d, N_d÷2 + R_d]; color=:red, linestyle=:dash, label="Interface")
axislegend(ax_d2; position=:rt)

umag_d = @. sqrt(rv.ux^2 + rv.uy^2)
umag_d[isnan.(umag_d)] .= 0.0
ax_d3 = Axis(fig_dr[1, 3]; title="Spurious currents |u|", aspect=DataAspect())
hm_d3 = heatmap!(ax_d3, 1:N_d, 1:N_d, umag_d'; colormap=:inferno)
Colorbar(fig_dr[1, 4], hm_d3; label="|u|")
save(joinpath(OUTDIR, "droplet_vof_results.svg"), fig_dr)
println("  ✓ VOF results")

# Convergence
N_list_d = [64, 128, 256]
sp_vof = Float64[]; sp_cls = Float64[]
for Ni in N_list_d
    Ri = Ni ÷ 4
    r1 = run_static_droplet_2d(; N=Ni, R=Ri, σ=0.01, ν=0.1, ρ_l=1.0, ρ_g=0.001, max_steps=5000)
    push!(sp_vof, r1.max_u_spurious)
    r2 = run_static_droplet_clsvof_2d(; N=Ni, R=Ri, σ=0.01, ν=0.1, ρ_l=1.0, ρ_g=0.001, max_steps=5000)
    push!(sp_cls, r2.max_u_spurious)
end

fig_dc = Figure(size=(560, 440))
ax_dc = Axis(fig_dc[1, 1]; xlabel="N", ylabel="max |u| (spurious)",
             title="Spurious currents — VOF vs CLSVOF", xscale=log10, yscale=log10)
scatterlines!(ax_dc, Float64.(N_list_d), sp_vof;
              label="VOF (HF curvature)", linewidth=2, markersize=10)
scatterlines!(ax_dc, Float64.(N_list_d), sp_cls;
              label="CLSVOF (LS curvature)", linewidth=2, markersize=10, color=:coral)
axislegend(ax_dc; position=:rt)
save(joinpath(OUTDIR, "droplet_convergence.svg"), fig_dc)
println("  ✓ convergence — DONE")

# ============================================================================
# 15. Rayleigh-Plateau
# ============================================================================
println("=== 15. Rayleigh-Plateau ===")

R0_rp = 20; λ_ratio_rp = 4.5; ε_rp = 0.05
σ_rp = 0.01; ν_rp = 0.05; ρ_l_rp = 1.0; ρ_g_rp = 0.01
λ_rp = λ_ratio_rp * R0_rp
Nz_rp = round(Int, λ_rp); Nr_rp = 3 * R0_rp

# Geometry
C0_rp = zeros(Nz_rp, Nr_rp)
for j in 1:Nr_rp, i in 1:Nz_rp
    z = i - 0.5; r = j - 0.5
    R_local = R0_rp * (1 - ε_rp * cos(2π * z / λ_rp))
    C0_rp[i, j] = 0.5 * (1 - tanh((r - R_local) / 2))
end

fig_rg = Figure(size=(700, 340))
ax_rg = Axis(fig_rg[1, 1]; title="Initial jet — R₀=$R0_rp, λ/R₀=$λ_ratio_rp, ε=$ε_rp",
             xlabel="z (axial)", ylabel="r (radial)", aspect=DataAspect())
hm_rg = heatmap!(ax_rg, 1:Nz_rp, 1:Nr_rp, C0_rp; colormap=:blues, colorrange=(0, 1))
Colorbar(fig_rg[1, 2], hm_rg; label="C")
save(joinpath(OUTDIR, "rp_geometry.svg"), fig_rg)
println("  ✓ geometry")

# Run
result_rp = run_rp_clsvof_2d(; R0=R0_rp, λ_ratio=λ_ratio_rp, ε=ε_rp,
                               σ=σ_rp, ν=ν_rp, ρ_l=ρ_l_rp, ρ_g=ρ_g_rp,
                               max_steps=15000, output_interval=500)
println("  ✓ simulation")

# Final shape
Nz_r = result_rp.config.Nx; Nr_r = result_rp.config.Ny
fig_rs = Figure(size=(700, 340))
ax_rs = Axis(fig_rs[1, 1]; title="Jet at t = 15000 (CLSVOF axisym)",
             xlabel="z", ylabel="r", aspect=DataAspect())
hm_rs = heatmap!(ax_rs, 1:Nz_r, 1:Nr_r, result_rp.C; colormap=:blues, colorrange=(0, 1))
Colorbar(fig_rs[1, 2], hm_rs; label="C")
save(joinpath(OUTDIR, "rp_final_shape.svg"), fig_rs)
println("  ✓ final shape")

# Thinning
fig_rt = Figure(size=(600, 420))
ax_rt = Axis(fig_rt[1, 1]; xlabel="Time (steps)", ylabel="r_min / R₀",
             title="Jet thinning — Rayleigh-Plateau")
t_rp = Float64.(result_rp.times); r_rp = Float64.(result_rp.r_min) ./ R0_rp
valid_rp = r_rp .< Inf
lines!(ax_rt, t_rp[valid_rp], r_rp[valid_rp]; linewidth=2, color=:steelblue, label="CLSVOF")
hlines!(ax_rt, [1 - ε_rp]; color=:gray, linestyle=:dash, label="Initial (1-ε)")
hlines!(ax_rt, [1.0]; color=:black, linestyle=:dot, label="Unperturbed R₀")
axislegend(ax_rt; position=:rt)
save(joinpath(OUTDIR, "rp_thinning.svg"), fig_rt)
println("  ✓ thinning")

# Fields
fig_rf = Figure(size=(1050, 340))
ax_rp1 = Axis(fig_rf[1, 1]; title="Level-set φ", aspect=DataAspect())
heatmap!(ax_rp1, 1:Nz_r, 1:Nr_r, result_rp.phi; colormap=:RdBu, colorrange=(-10, 10))
umag_rp = @. sqrt(result_rp.uz^2 + result_rp.ur^2)
ax_rp2 = Axis(fig_rf[1, 2]; title="Velocity |u|", aspect=DataAspect())
heatmap!(ax_rp2, 1:Nz_r, 1:Nr_r, umag_rp; colormap=:viridis)
ax_rp3 = Axis(fig_rf[1, 3]; title="Density ρ", aspect=DataAspect())
heatmap!(ax_rp3, 1:Nz_r, 1:Nr_r, result_rp.ρ; colormap=:thermal)
save(joinpath(OUTDIR, "rp_fields.svg"), fig_rf)
println("  ✓ fields — DONE")

println("\n✅ All figures generated!")
