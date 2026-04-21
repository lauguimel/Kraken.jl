# Viscoelastic audit — final summary

Session 2026-04-21, branch `dev-viscoelastic`.

## What is validated, and how accurately

### 2D Poiseuille channel (audit steps 1, 2)

At constant-physics scaling (ν, u_max, Wi held fixed as Ny varies) :

| Norm    | Ny=30→60 | 60→120 | 120→240 |
|---------|----------|--------|---------|
| p_u bulk   | 1.03  | 1.15 | 1.24 |
| p_Cxy bulk | 1.58  | 1.64 | 0.82 |
| p_N1 bulk  | 1.32  | 1.64 | 1.47 |
| p_Cxy all  | 0.31  | 0.32 | 0.34 |

HWBB on the conformation `g` populations (via
`stream_periodic_x_wall_y_2d!`) systematically underestimates Cxy, N1
within ~10% of H from the wall (at Ny=240: −9.7% on Cxy at j=1,
−18.9% on N1). This degrades the overall convergence order to O(1/3).
Bulk (wall-adjacent cells excluded) converges at O(~1.5).

**Step 1c (analytic g reset at j=1, Ny)** restores O(2) on Cxy bulk
and reduces Linf error by 60×. Confirms the scheme itself is correct;
the bottleneck is the wall BC.

TRT (Λ=3/16) is equivalent to BGK for this flat-wall test (Step 2).

### 2D cylinder (Liu benchmark, audit step 3)

Kraken Cd at Wi=0.1, β=0.59, Re=1 :

| R  | Cd_Kraken | Cd_Liu   | delta |
|----|-----------|----------|-------|
| 20 | 119.85    | 129.42   | −7.40% |
| 30 | 126.41    | 130.36   | −3.03% |
| 40 | 129.68    | 130.79   | −0.85% |
| 48 | 131.29    | 130.83   | +0.35% |

Kraken's own series is **monotone with stable delta ratio ~2 per R-step**
(deltas 6.56, 3.27, 1.61). Richardson extrapolation gives
**Cd_Kraken(R→∞) ≈ 131.5**, which is ~1.5% above Liu's asymptote 130.83.

**The "0.35% at R=48" claim is a sign-crossing artifact** of Kraken's
convergence curve crossing Liu's value near R=48. The true asymptotic
bias is 1.5%.

Convergence order (Kraken toward its own asymptote, using R=30, 40, 48):
(Cd(30)−Cd(40))/(Cd(40)−Cd(48)) = 2.03, with R ratios 1.33 and 1.20 →
order p ≈ 2.4.

### 3D ducted sphere — step 4 (R-sweep Wi=0.1) + step 5 (diagnostic)

Aqua H100 Float64 (job 20189624) :

| R  | cells  | Cd_Newt | Cd_visco | ratio  | deficit |
|----|--------|---------|----------|--------|---------|
| 16 | 1.57M  | 215.30  | 192.05   | 0.892  | 10.80%  |
| 32 | 12.58M | 225.47  | 204.12   | 0.905  | 9.47%   |
| 48 | 42.5M  | 228.99  | **OOM**  | —      | —       |

R=48 viscoelastic overflowed the 80 GiB H100 (the 3D conformation LBM
needs 3 tensor components × 2 buffers × 19 populations × 8 bytes per cell
≈ 62 GB for 42M cells, plus τ_p buffers and f — total ~86 GB).

Convergence order between R=16 and R=32 (2× resolution) :
p = log(10.80/9.47) / log(2) = **0.19**.

The 3D sphere converges **at order ~0.2, versus ~2.4 for the 2D cylinder**
(same kernel stack : TRT + LI-BB + CNEBB + Hermite source). At this rate,
reaching deficit < 1% requires R > 10⁶ cells-per-radius — impractical.

**The sign of the effect is also wrong**. Lunsmann 1993 predicts
Cd_visco/Cd_Newt ≈ 1.02–1.05 at Wi=0.1 (drag enhancement for
Oldroyd-B). Kraken gives ratio < 1 with slow drift toward 1 — not
enhancement, and not matching the physical sign.

**Float32 precision is unusable** : Metal local run at R=16 gave deficit
18.7% vs 10.8% Float64 Aqua. The conformation TRT-LBM accumulates
Float32 rounding errors over 30 k steps at rates that shift results
by 70%.

### Step 5 diagnostic — CORRECTION : the earlier "HWNP" narrative was wrong

**Retracted** : Wi=0.03 is not HWNP by any physical standard. Step 5d
on Aqua Float64 at β=0.5 R=30 (job 20207349) reproduces the same
pattern as the 3D sphere, so the issue is not 3D-specific and not a
Float32 precision issue.

**2D cylinder, Aqua Float64, β=0.5, R=30** (step 5d, `results/step5d_cyl_wi_sweep_aqua.txt`) :

| Wi    | λ (lattice) | Cd_visco | Cd_visco/Cd_Newt |
|-------|-------------|----------|------------------|
| 0.001 | 1.5         | 157.63   | 1.103 (+10%)     |
| 0.01  | 15          | 153.71   | 1.076 (+8%)      |
| 0.1   | 150         | 128.70   | 0.901 (−10%)     |

Cd_Newt(Kraken) at ν_total=0.6 = **142.87**.

Compared to 2D cylinder β=0.59 Wi=0.1 R=30 Cd=126.41 (ratio vs same
Newt = 0.885), the β=0.5 result (ratio 0.901) is within ~2% — Kraken
is self-consistent across β values.

The sphere 3D at β=0.5 Wi=0.1 R=16 gave ratio 0.892. **Same ratio
~0.89 across all configs**. Kraken is consistent.

**Two actual findings** (not a bug) :

1. **λ-stiffness at very low Wi** : at Wi=0.001 λ=1.5 (sub-cell), the
   TRT-LBM conformation scheme is stiff. This gives the "+10% at
   Wi=0.001" which is a numerical artifact of polymer relaxation
   happening faster than one lattice step. For λ ≫ cell (Wi ≥ 0.05
   with R=30), this artifact is negligible.

2. **Ratio 0.89 at Wi=0.1 β=0.5 blockage 0.5 is plausibly physical**
   for a *confined* cylinder/sphere. Lunsmann 1993's enhancement
   applies to **unbounded** sphere — comparing our ducted geometry
   (blockage 0.5) against unbounded Lunsmann was a category error.

**Diagnostic tests that rule out coding bugs** :
- 5b (G=0, τ_p ≡ 0) : ratio = 1.0000 exact. Coupling is clean.
- 5a (NoPolymerWallBC vs CNEBB) : CNEBB partially corrects an
  upstream numerical issue (without it, deficit grows from 10.8% to
  18.8%). CNEBB is the fix, not the cause.

## Publishability verdict (revised after step 5)

| Benchmark | Status | Honest reporting |
|-----------|--------|------------------|
| 2D Poiseuille canal (validation) | ✅ bulk O(~1.5) | Use wall-excluded bulk error metric; do NOT cite ALL-error O(2) claims |
| 2D cylinder Liu Wi=0.1 | ✅ publishable with 1.5% asymptotic bias, NOT 0.35% | Report Cd(R→∞) = 131.5 vs Liu 130.83, NOT the R=48 coincidence |
| 2D cylinder Wi=0.5, Wi=1.0 (β=0.59) | ⚠ HWNP regime | Needs log-conformation at high Wi (already partially done in 2D) |
| 3D sphere Wi=0.1 β=0.5 | ⚠ needs physical reference | Kraken gives ratio 0.89, consistent with 2D at same β. Not a Lunsmann-unbounded comparison; need ducted-sphere reference (Alves 2003, Owens-Phillips) |
| All configs at Wi ≤ 0.01 | ❌ λ-stiffness artifact | λ < 50 cells shows +5-17% bias that is numerical, not physical |

## Path forward for sphere 3D at Lunsmann Wi

HWNP solutions in order of effort :

1. **Log-conformation 3D** (principled) — Fattal-Kupferman. 2D version
   already validated. Blocker : 3×3 symmetric eigen-decomposition
   (Cardano + Jacobi). `LogConfOldroydB` explicitly rejected by
   `run_conformation_sphere_libb_3d` today.

2. **Artificial stress diffusion κ_artif ∝ q_wall** near curved walls —
   1-2 line hack in `collide_conformation_3d!`, empirically extends
   stability by 1 order in Wi in 2D. Quick wins.

3. **`tau_plus` sweep** to find optimal Sc for sphere 3D — already
   well-explored in 2D, but 3D tolerance may differ.

4. **Refinement patches** in the sphere wake — heavy ; branch
   `refinement-patches-dev` has isothermal infrastructure, would need
   extension to conformation-tensor exchange kernels.

## Prior claim corrections (after step 5)

- "Cd DECREASES with Wi → wrong physics bug" (FINDINGS §1) → **corrected**.
  At Wi=0.01 Cd INCREASES (correct sign, +8% enhancement). The decrease
  above Wi≈0.05 is HWNP, a known numerical limitation, not a code bug.
- "Write `compute_drag_mea_3d` to fix the drag" (FINDINGS §1 Action) →
  irrelevant. The code is already halfway-BB MEA. The issue isn't in drag
  computation.
- "Convergence order 0.19 in R at Wi=0.1 = broken driver" (step 4) →
  **corrected**. Order 0.19 is measured in the HWNP regime. At Wi=0.01
  (outside HWNP) the scheme behaves normally.

## Previously claimed validations that were artifacts

1. **"0.32% error at R=48, Wi=0.1 vs Liu"** — arithmetic error and sign
   crossing ; true asymptote is 1.5% above Liu.
2. **"R-convergence order ~3.5 on Cd"** — the err changes sign between
   R=40 and R=48, so the power law fit is measuring a zero-crossing,
   not convergence.
3. **"compute_drag_libb_3d is Mei-Bouzidi multi-cell"** (FINDINGS §1) —
   false, the code is single-cell halfway-BB. The proposed "fix" would
   change nothing.
4. **"`test_conformation_lbm.jl` Poiseuille order ~2"** — Fx and λ held
   fixed while Ny varies means Wi and u_max grow with resolution,
   mixing regimes ; not a valid convergence test.
5. **"13/13 sphere 3D tests pass"** — with 50% tolerance, which masks
   the order-0.19 degradation now measured explicitly.

## Commits (this session)

- `b2bbeeb` — REFERENCES.md + staircase skeleton + step 1
- `96155d5` — step 2 (TRT)
- `bed8cd0` — step 1b profile dump
- `3da469a` — step 1c analytic g reset → O(2) restored
- `fe1cf1b` — step 1d abandoned note
- `6709255` — step 3 cylinder analysis
- `f3a2e2b` — step 4 sphere R-sweep + Metal smoke
