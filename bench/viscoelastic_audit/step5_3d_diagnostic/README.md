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
