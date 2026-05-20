# M30 Phase 2a — Claude step 1: standalone interpBB on analytical Couette

Author engine : Claude (Anthropic Opus 4.7, 1M).
Date         : 2026-05-20.
Patch        : `bench/scratch/m30_phase2a_interpBB_claude/m30_phase2a.jl` (pure D2Q9 SRT, no Kraken/src dependency, ~520 LOC).

Step 1 of the adversarial protocol — Claude implements and measures
**before** Codex is spawned with the same problem statement.

## Setup chosen

- 2D D2Q9 SRT, Float64 CPU, ~520 LOC standalone Julia.
- Domain `64 × 64` LU, cell centres `(1..64, 1..64)`. Centre `(32.5, 32.5)`.
- Inner rotor `R_in = 10` LU at angular velocity `ω` ; outer stationary wall
  `R_out = 25` LU.
- Both walls use the same BC scheme per run (halfway-BB or Bouzidi interpBB on
  inner ; outer stays halfway-BB for both runs so the BC effect is isolated to
  the inner rotor). `τ = 1.0` → `ν = 1/6` LU. 5 000 steps. Initialised at the
  equilibrium of the analytical `(u_θ, p)` profile.
- Two configurations measured:
  - **Config 1** — `ω = 0.001` (Re_in ≈ 0.6, max u ≈ 1e-2 LU, low Mach).
  - **Config 2** — `ω = 0.005` (Re_in ≈ 3, max u ≈ 5e-2 LU, more pressure signal).

Analytical Couette closed form:
- `u_θ(r) = A·r + B/r`, with `B = ω·R_in²·R_out²/(R_out²−R_in²)`, `A = −B/R_out²`.
- `p(r) = ρ₀·[A²(r²−R_in²)/2 + 2AB·log(r/R_in) − B²(1/r² − 1/R_in²)/2]`,
  obtained by integrating the centripetal radial momentum balance `dp/dr = ρ·u_θ²/r`.
- Torque on inner cylinder: `T_an = 4πρν·ω·R_in²·R_out² / (R_out²−R_in²)`.
- Drag on inner cylinder: identically zero by symmetry.

LBM pressure is recovered by `p_LBM = (ρ − ρ₀)·cs²`. Because LBM
pressure is defined up to a constant, all `p_max_rel` figures below are
measured **after removing the spatial mean offset** `<p_LBM − p_an>` across
the fluid annulus (otherwise slow rho-drift over 5000 steps would mask the
profile error).

## Bouzidi-FL formula

Convention: `q = |x_b − x_w| / |x_b − x_f|` ∈ (0, 1] where `x_b` is the solid
neighbour beyond `x_f` along link direction `c_i`. `q = 0.5` is halfway-BB.

- For `q ≤ 0.5`:
  `f_ī(x_f, t+dt) = 2q·f_i(x_f, t*) + (1−2q)·f_i(x_ff, t*) − 2·w_i·ρ·(c_i·u_w)/cs²`,
  where `x_ff = x_f − c_i` is the second fluid cell back along `−c_i`. If
  `x_ff` is not fluid (rare for `R_in/R_out` chosen here), fall back to halfway-BB.
- For `q > 0.5`:
  `f_ī(x_f, t+dt) = (1/2q)·f_i(x_f, t*) + ((2q−1)/2q)·f_ī(x_f, t*) − (1/q)·w_i·ρ·(c_i·u_w)/cs²`.

Both halfway-BB and Bouzidi-FL include the moving-wall correction
`±2·w_i·ρ·(c_i·u_w)/cs²` evaluated at the **link's actual wall hit point**
`x_w = x_f + q·c_i`, not at the lattice centre. The local fluid density
`ρ(x_f)` is used as `ρ_w` (Ladd convention).

## Force/torque measurement

Ladd MEA: for each fluid–inner-solid link in direction `c_q`,
`F_link = c_q · (f_post(x_f, q) + f_after_BB(x_f, q̄))`. Sum gives total force
on the rotor by the fluid. Torque adds `(x_w × F_link)·ẑ` using the wall hit
point.

Sign: Ladd MEA gives momentum from solid **to** fluid, so the torque the
**fluid** exerts on the rotor is `−T_LBM`. We compare magnitudes
(`abs(|T_LBM| − |T_an|) / |T_an|`) to be sign-convention-independent.

## Results — Config 1 (ω = 0.001)

| metric            | halfway-BB    | interpBB      | Δ improvement   |
|-------------------|--------------:|--------------:|----------------:|
| u_max_rel         | 3.900e-02     | 6.632e-03     | **−83.0 %**     |
| u_l2_rel          | 8.18e-03      | 2.04e-03      | −75.1 %         |
| p_max_rel (wall)  | 5.19          | 0.873         | **−83.2 %**     |
| p_l2_rel          | 0.514         | 0.146         | −71.6 %         |
| drag F=|F_x,F_y|  | 7.33e-16      | 8.96e-16      | machine-eps     |
| torque rel err    | 1.65e-02      | 3.47e-03      | **−78.9 %**     |

NB. The large absolute `p_max_rel ≈ 0.87` for interpBB in Config 1 reflects
that the analytical pressure scale itself is tiny (`p_ref ≈ 2.75e-5` LU) at
ω=0.001. The compressible-LBM floor `O(Ma²·cs⁻²) ≈ 3e-5` becomes
comparable to the signal. Config 2 with ω=0.005 has a 25× stronger
pressure scale, exposing the true wall-BC profile error.

## Results — Config 2 (ω = 0.005)

| metric            | halfway-BB    | interpBB      | Δ improvement   |
|-------------------|--------------:|--------------:|----------------:|
| u_max_rel         | 3.88e-02      | 6.63e-03      | **−82.9 %**     |
| u_l2_rel          | 7.84e-03      | 2.41e-03      | −69.3 %         |
| p_max_rel (wall)  | 1.087         | 0.286         | **−73.7 %**     |
| p_l2_rel          | 0.106         | 0.0342        | −67.7 %         |
| drag F=|F_x,F_y|  | 4.60e-16      | 3.60e-15      | machine-eps     |
| torque rel err    | 1.85e-02      | 1.01e-04      | **−99.45 %**    |

The torque accuracy gain is striking: interpBB delivers `|T| − |T_an|`
within `0.01 %` of analytical, while halfway-BB carries a systematic
`1.85 %` excess — exactly the kind of structural bias that the Phase 1
R-sweep flagged as `K/rT` plateau on the full cylinder benchmark.

## Q1 — implementation correctness

- Q1 (Claude): max `u_θ` rel err = **6.63e-03** (interpBB) / **3.90e-02**
  (halfway-BB). Max `p` rel err (offset-removed) = **0.87** (interpBB) /
  **5.19** (halfway-BB) in Config 1.
- Sanity: `Cd_drag = 0 ± 1e-15` for both (analytical drag is 0; LBM machine
  precision OK).
- Steady-state check: `u_max_rel` plateaus from step 500 onward (no
  amplification, no NaN). Both schemes preserve the analytical state under
  evolution.

## Q2 — wall-pressure error at r = R_in

Config 2 (the better-conditioned probe):

- halfway-BB: max `|p − p_an|` (offset-removed) at wall = `1.087 × p_ref`.
- interpBB:   max `|p − p_an|` (offset-removed) at wall = `0.286 × p_ref`.
- **Δ improvement = −73.7 %.**

## Q3 — drag and torque on inner cylinder

Drag (analytical = 0): both schemes recover 0 ± 1e-15 (machine eps).
Couette has no asymmetric stagnation, so drag is not a discriminator here.

Torque (analytical 1.2467 LU in Config 2):

| BC          | T_LBM      | abs rel err |
|-------------|----------:|---:|
| halfway-BB  | -1.26976  | **1.85 %** |
| interpBB    | -1.24679  | **0.01 %** |

The torque is **180× better** with interpBB. This is the analog of the
front-pole pressure deficit on the full cylinder benchmark — a BC that
mis-allocates wall traction here will mis-allocate stagnation pressure on
the cylinder.

## Q4 — GO/NO-GO

**GO.** Bouzidi interpBB reduces:
- wall-pressure max error by **73.7 %** (Config 2),
- torque error by **99.45 %** (Config 2),
- velocity max error by **83 %** (both configs).

All three improvements are far above the 30 % threshold set by the brief.
The implementation runs stably for 5000 steps at two ω values, with no
sign of instability or amplification. Drag stays machine-eps in both
schemes (the Couette test cannot distinguish drag artefacts; interpretation
must rely on torque + wall-pressure).

**Recommendation**: proceed to Phase 2b — port Bouzidi-FL interpBB to
`src/kernels/li_bb_2d.jl` (or equivalent) as an alternative to the existing
halfway-BB in Kraken's cylinder driver, behind a kwarg like
`wall_bc::Symbol = :halfwayBB` with `:bouzidi_fl` as the new option.
Cross-check on the full Wi=1 β=0.59 R=30 cylinder run against rheoTool
reference.

## Caveats / open questions for Codex's review

1. The standalone implementation uses Ladd's wall density convention
   (`ρ_w = ρ(x_f)`). An alternative (Ladd 1994, Section 4.2) uses
   `ρ_w = (ρ(x_f) + ρ(x_w_extrap))/2` — second-order in the local rho
   field. Worth a quick test if Codex can include it; the present
   implementation may slightly over-state the halfway-BB error at the wall.
2. The fall-back for `q ≤ 0.5` when `x_ff` is solid is "use halfway-BB" —
   for the present geometry no link falls into that case, but for the
   real cylinder driver this fall-back will need attention.
3. The Bouzidi-FL formula for `q > 0.5` uses `f_ī(x_f, t*)` (the
   post-collision opposite pop at the same fluid cell), which means the
   BC must be applied **before** the opposite pop is overwritten by
   streaming from a neighbour. The current implementation respects this
   by streaming into `s.f` first (only fluid→fluid) and then doing all BC
   writes second.
4. The torque sign flip (LBM −0.249 vs analytical +0.249) is the Ladd
   MEA "fluid receives momentum from solid" convention; the magnitude
   match is what matters for the verdict.

## Files

- `bench/scratch/m30_phase2a_interpBB_claude/m30_phase2a.jl`        (standalone LBM)
- `bench/scratch/m30_phase2a_interpBB_claude/diag_halfwayBB.csv`    (Config 1 diag)
- `bench/scratch/m30_phase2a_interpBB_claude/diag_interpBB.csv`     (Config 1 diag)
- `bench/scratch/m30_phase2a_interpBB_claude/wall_ring_*.csv`       (Config 1 wall band)
- `bench/scratch/m30_phase2a_interpBB_claude/test2_Rin10p5/`        (Config 2 results)
- `bench/scratch/m30_phase2a_interpBB_claude/summary.txt`           (Config 1 summary)

Awaiting Codex's independent implementation for cross-check.
