# M57 Beefed Ladder Verdict

## L3b — per-cut-cell M55 vs first-order
- Definition: Frozen R=10 Newtonian cylinder field; compare embedded M55 normal-gradient magnitude to first-order FVFD on the same cut links, binned by q_w.
- CSV: scratch/M57/L3b_per_qw_bin.csv
- Verdict: GREEN
- Measured: max |delta| = 1.7469401085757359e-3; max |FO| = 5.7871269500186949e-3; worst ratio = 0.3018665606722335 in q_w bin 0.80-0.90.
- Interpretation: M55b perturbs the frozen Newtonian cut-cell gradient, but the perturbation is below the 0.5x FO gate and is not by itself a dominant frozen-field gradient error.

## L3-recompute — time-marching stability on steady field
- Definition: Frozen R=10 cylinder replay; recompute embedded M55 gradient every polymer step from identical ux/uy and track gradient drift.
- CSV: scratch/M57/L3_recompute_trajectory.csv
- Verdict: GREEN
- Measured: grad_drift_at_5000 / max_grad_at_0 = 0 / 0.007109771016985178 = 0; trace_C_max_at_5000 = 100.75484920170352; F_poly_max_at_5000 = 1.8286361071871067e-4.
- Interpretation: Recomputing M55b on an unchanged velocity field is deterministic on this Metal run. No stateful gradient drift is implicated.

## L3-Guo — Guo back-coupling with u pinned
- Definition: Frozen R=10 cylinder polymer loop with embedded M55 gradient and one Guo LBM step per iteration, then ux/uy reset to the frozen Newtonian field.
- CSV: scratch/M57/L3_guo_trajectory.csv
- Verdict: GREEN
- Measured: no NaN through 5000 steps; trace_C_max_at_5000 = 100.75484920170352; F_poly_max_at_5000 = 1.8286361071871067e-4.
- Interpretation: Guo injection plus the M55 source is stable when ux/uy is pinned. The M48 NaN requires the velocity field to evolve under feedback.

## L5b — coupled Poiseuille with M55 fired
- Definition: Periodic-x channel with qwall-populated halfway walls; coupled Oldroyd-B/Guo loop uses embedded qwall M55 gradient and compares steady ux to Poiseuille.
- CSV: scratch/M57/L5b_poiseuille_steady.csv
- Verdict: GREEN
- Measured: max |u_err| / max(u_analytic) = 0.0014114339948680175; max |uy| = 2.793966302760964e-7; qwall wall_q entries = 360; no NaN.
- Interpretation: M55b is stable in a planar coupled qwall channel. The M55b NaN is not a general planar M55 instability.

## Forensic M48 — first-NaN cell at R=30
- Definition: M48 R=30/M55b smoke with diagnostic stride 1 and rolling pre-NaN field snapshot; post-process first nonfinite cell plus 8 neighbors.
- First-NaN cell: (439, 32) at step 2145; first nonfinite field = rho; pre-NaN snapshot = step 2144; theta = -112.83365417791755 deg.
- CSV: scratch/M57/M48_forensic_R30_first_nan_cell.csv
- Worst q_w near cell: no q_w in [0.1, 0.2]; neighboring cell (439,33) has below-threshold q_w = 0.003164-0.010421 links that fall back to FO. Worst active cut link is (438,33), q=3, q_w=0.46298789978027344.
- Δ(du_dn_M55 − du_dn_FO) at NaN cell: max active-link delta at (439,32) = 0.06504013769572065 for q=6/q=8; worst neighbor delta = 0.14885757633793142 at (438,33), q=3.
- τ_p / F_poly magnitudes at NaN-1 step: NaN cell tau_mag = 9.71141607524549e-5 and F_poly_mag = 0.09035804918466184; neighbor (438,33) tau_mag = 1.6743256587560036e-4 and F_poly_mag = 0.10728830328872811; neighbor (439,33) tau_mag = 0.244365500635004 and F_poly_mag = 0.0508961208118846.
- Interpretation: The pre-NaN local force and velocity are already enormous relative to L3, while frozen-field and pinned-u tests stay green. The failure localizes to evolved cylinder cut-cell feedback, with active M55/FO differences near the NaN cell and a nearby q_w_min discontinuity cluster.

| cell | theta_deg | ux | uy | tau_mag | F_poly_mag |
| --- | ---: | ---: | ---: | ---: | ---: |
| (438,31) | -113.8 | 0.00247182 | -0.00367161 | 5.16785e-5 | 3.43704e-6 |
| (439,31) | -112.1 | 0.00350045 | 0.00176316 | 6.36348e-5 | 6.48816e-6 |
| (440,31) | -110.4 | -0.000369337 | 0.00195948 | 7.24443e-5 | 9.63865e-6 |
| (438,32) | -114.5 | -0.00585207 | -0.00597291 | 6.42165e-5 | 1.98513e-5 |
| (439,32) | -112.8 | -0.0712312 | 0.0553995 | 9.71142e-5 | 0.090358 |
| (440,32) | -111.1 | -0.00135549 | -0.00161515 | 1.16998e-4 | 2.47354e-5 |
| (438,33) | -115.3 | 0.0835778 | -0.0650016 | 1.67433e-4 | 0.107288 |
| (439,33) | -113.6 | 0.037529 | -0.031882 | 0.244366 | 0.0508961 |
| (440,33) | -111.8 | 0 | 0 | 0 | 0 |

## Localisation
The top suspect is the evolved-u cylinder cut-cell feedback loop, not standalone M55b, not stateful recomputation, and not Guo+M55 with pinned velocity. L3b/L3-recompute/L3-Guo/L5b are all GREEN, while M48 fails at a cylinder cut-cell after ux/uy and F_poly have locally blown up. The forensic rows point to a qwall/stencil-local mechanism: active M55-vs-FO differences near the NaN cell plus adjacent below-q_w_min fallback links create a sharp stencil transition that only becomes unstable when Guo feedback is allowed to move u.

## Next-mission candidates
- Add a 2100-2145 M48 microtrace that logs ux/uy, F_poly, tau, and M55-vs-FO deltas every 5 steps for cells (438:440,31:33).
- Run M48 R=30 with embedded_gradient=true but force_boundary_fill=:none/:nearest matrix to separate force fill from M55 gradient.
- Run a q_w_min threshold sweep as an observational mission only: q_w_min 0.05, 0.1, 0.2, recording first-NaN step and the same forensic table.
- Build a two-cell local replay from the step-2100 M48 snapshot that freezes all cells except the 3x3 NaN neighborhood and applies the Guo force update.
- Compare M55 active vs forced-FO fallback on the same M48 R=30 forensic window without changing any other ingredient.
