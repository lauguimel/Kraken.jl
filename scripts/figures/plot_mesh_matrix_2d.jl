#!/usr/bin/env julia --project=docs
# WP-MESH-4 figure: bar chart of Cd error (%) and MLUPS for the three
# baselines on Schäfer-Turek 2D-1, plus a small table-style annotation.
using CairoMakie

const RUNS = [
    (label = "(A) Cartesian + halfway-BB",      err = 3.55, mlups = 37.0),
    (label = "(B) Cartesian + LI-BB v2",         err = 2.12, mlups = 29.0),
    (label = "(C) gmsh + SLBM + LI-BB v2",       err = 2.37, mlups = 11.0),
]

fig = Figure(; size = (980, 380))

ax_err = Axis(fig[1, 1];
    title  = "Schäfer-Turek 2D-1, Re=20 — Cd accuracy",
    xticks = (1:length(RUNS), [r.label for r in RUNS]),
    xticklabelrotation = π/8,
    ylabel = "|Cd − 5.58| / 5.58 [%]")
barplot!(ax_err, 1:length(RUNS), [r.err for r in RUNS];
         color = [:steelblue, :firebrick, :darkorange],
         strokecolor = :black, strokewidth = 0.6)
for (k, r) in enumerate(RUNS)
    text!(ax_err, k, r.err + 0.1; text = string(round(r.err, digits=2), "%"),
          align = (:center, :bottom), fontsize = 14)
end
ylims!(ax_err, 0, 5)

ax_mlups = Axis(fig[1, 2];
    title  = "Throughput (CPU, 36 603 cells × 30 000 steps)",
    xticks = (1:length(RUNS), [r.label for r in RUNS]),
    xticklabelrotation = π/8,
    ylabel = "MLUPS")
barplot!(ax_mlups, 1:length(RUNS), [r.mlups for r in RUNS];
         color = [:steelblue, :firebrick, :darkorange],
         strokecolor = :black, strokewidth = 0.6)
for (k, r) in enumerate(RUNS)
    text!(ax_mlups, k, r.mlups + 0.5; text = string(round(Int, r.mlups)),
          align = (:center, :bottom), fontsize = 14)
end
ylims!(ax_mlups, 0, 45)

mkpath("paper/figures")
save("paper/figures/mesh_matrix_2d.pdf", fig)
save("paper/figures/mesh_matrix_2d.png", fig; px_per_unit = 2)
@info "wrote" pdf="paper/figures/mesh_matrix_2d.pdf"
