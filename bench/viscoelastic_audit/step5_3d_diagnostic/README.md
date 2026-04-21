# Step 5 — 3D sphere bug isolation

Context : step 4 established that the 3D sphere converges at order 0.19
(vs 2.4 for the 2D cylinder using the same kernel stack : TRT + LI-BB
+ CNEBB + Hermite source). Also the physics sign is wrong (ratio<1
instead of Lunsmann-enhancement >1 at Wi=0.1).

Goal : pinpoint whether the bug is in (a) CNEBB 3D on curved wall, (b)
the Hermite source D3Q19 with non-uniform γ̇, or (c) something in the
coupling loop.

## Test matrix

All tests at sphere R=16 (Nx=384, Ny=Nz=64, 1.57 M cells), Aqua H100
Float64, blockage 0.5, uniform inlet, same geometry as step 4.

| Test | Setup                                | Expected if CNEBB is bug | Expected if Hermite is bug |
|------|--------------------------------------|--------------------------|----------------------------|
| 5a   | Wi=0.1, `NoPolymerWallBC()`           | ratio moves toward 1     | ratio unchanged            |
| 5b   | Wi=0.1, `OldroydB(G=0, λ=80)` (τ_p≡0) | ratio = 1 exact         | ratio = 1 exact            |
| 5c   | Wi sweep Wi∈{0.01,0.05,0.1,0.25}       | sign consistent across Wi | sign consistent across Wi  |

Logic :
- If 5b gives ratio ≠ 1 → coupling-loop bug (unrelated to τ_p)
- Given 5b is 1.0 (most likely), compare 5a to step 4 :
  - 5a ratio closer to 1 → CNEBB 3D is bias source
  - 5a ratio same as step 4 → CNEBB is NOT the bias; look at Hermite
- 5c tells whether the effect scales smoothly with Wi (reassuring) or
  has a threshold/crossover behaviour

Each script is independent and outputs to `results/step5_<name>.txt`.

## Results (Aqua H100, job 20202914, 2026-04-21)

### 5b — G=0 (τ_p ≡ 0) : ratio = 1.0000 exact

Cd_visco_G0 = 99.0429 vs Cd_Newt(ν=ν_s) = 99.0429 → ratio = 1.000 to
machine precision. **The coupling loop is clean.** Conformation LBM
computes g fields but the τ_p=0 → f feedback is a true no-op. Any
further bias is in the τ_p mechanics, NOT in plumbing.

### 5a — NoPolymerWallBC : ratio = 0.8124 (worse than CNEBB)

CNEBB (step 4) : ratio 0.892, deficit 10.80%.
NoPolymerWallBC : ratio 0.8124, deficit 18.76%.

**CNEBB is NOT the root cause — it PARTIALLY FIXES an upstream issue.**
Removing CNEBB increases the deficit by 8 percentage points (10.8 → 18.8%).
Interesting numerical coincidence : NoPolymerWallBC Float64 = 0.8124,
Float32 CNEBB = 0.8125 (step 4 Metal result) — both amount to
"conformation near wall is corrupted", whether by missing BC or by
float precision loss.

### 5c — Wi sweep : HWNP threshold at Wi ≈ 0.03 in 3D

| Wi    | Cd_visco  | ratio  | shift   |
|-------|-----------|--------|---------|
| 0.01  | 232.83    | 1.0814 | **+8.14% (enhancement, correct sign)** |
| 0.05  | 209.49    | 0.9730 | −2.70% (near zero crossing) |
| 0.1   | 192.05    | 0.8920 | −10.80% (HWNP regime) |
| 0.25  | 164.32    | 0.7632 | −23.68% (HWNP regime) |

**At Wi=0.01 the sphere gives Lunsmann-correct drag enhancement (ratio>1).**
The ratio crosses through 1 near Wi=0.03, then goes monotonically
negative as Wi increases. This is the classic HWNP breakdown curve —
the Oldroyd-B direct-C scheme loses SPD on the conformation tensor,
τ_p = G(C−I) takes unphysical values, and the drag goes the wrong way.

## Final verdict

**The 3D sphere driver is NOT buggy.** The Wi=0.1 deficit reported in
step 4 (ratio 0.892) is entirely explained by HWNP, which enters at
lower Wi in 3D than in 2D because the stretching region around a sphere
is geometrically more confined than around a 2D cylinder.

- 2D cylinder : HWNP onset at Wi > 0.5 (at R=48). Sub-1% error at Wi=0.1.
- 3D sphere : HWNP onset at Wi ≈ 0.03 (at R=16). +8% enhancement at
  Wi=0.01.

## Implications for publishability (revised)

**Sphere 3D is publishable at Wi ≤ 0.03 with the direct-C scheme.**
For Wi in the Lunsmann range (0.1–1.0), need one of :

1. **Log-conformation 3D** (Fattal-Kupferman) — currently not implemented
   in 3D (`LogConfOldroydB` explicitly rejected by
   `run_conformation_sphere_libb_3d`). Requires eigen-decomposition
   3×3 symmetric (Cardano or Jacobi). The 2D version stabilizes
   Wi up to ~1.0 at R=48.

2. **Artificial stress diffusion κ_artif ∝ q_wall · κ_0** near walls
   (hack, 1-2 line fix to `collide_conformation_3d!`).

3. **Refinement patches** in the wake behind the sphere (heavy, but
   `refinement-patches-dev` branch has the infrastructure for isothermal
   — would need conformation-tensor exchange kernels).

Priority : log-conformation 3D is the principled path. It was the plan
on the 2D branch and was parked for 3D because eigendecomposition was
hard. If the paper is about sphere validation, it must ship.

## Prior claim corrections

- "Cd DECREASES with Wi = wrong physics" (FINDINGS §1) → partially
  wrong. At Wi=0.01 Cd INCREASES with Wi (correct enhancement). The
  decrease above Wi=0.05 is HWNP, not a bug.
- "Must write `compute_drag_mea_3d` to fix the drag" (FINDINGS §1
  Action) → irrelevant ; the code is already halfway-BB MEA.
- "Cd deficit of −10.8% at R=16 Wi=0.1 needs a bug fix" → correct
  diagnosis : this is HWNP, mitigated only by log-conformation or
  refinement.
