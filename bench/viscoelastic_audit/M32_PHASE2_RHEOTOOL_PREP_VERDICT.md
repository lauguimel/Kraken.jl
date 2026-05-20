# M32 Phase 2 — rheoTool-side shrink-to-Kraken prep verdict

**Date:** 2026-05-21
**Department:** M32-Phase2-rheotool-prep
**Mandate:** Build modified rheoTool case directories whose geometry matches
Kraken (L_up=15R, L_down=15R, blockage 0.5). Preparation only — no execution.

---

## 1. Original rT setup (recap from Phase 1)

| Parameter             | Value                                                        |
|-----------------------|--------------------------------------------------------------|
| L_up / R              | 20.0 (vertex (-20, …) in `blockMeshDict`)                    |
| L_down / R            | 60.0 (vertex (60, …) in `blockMeshDict`)                     |
| Channel half-height / R | 2.0 (post-mirror full channel y∈[-2, 2])                   |
| Blockage R/halfH      | 0.5                                                          |
| Cells (blockMesh)     | 12,447                                                       |
| Cells (post-mirror)   | 24,894                                                       |
| endTime (Wi=1.0)      | 10 (Cd not converged, drift 3.4 units between t=8 and t=10)  |
| endTime (Wi=0.1)      | 6                                                            |
| endTime (Newtonian)   | 1                                                            |

*Citations:*
* `bench/rheotool/cylinder_wi1.0/system/blockMeshDict:20,30`
* `bench/rheotool/cylinder_wi1.0/log.blockMesh:94` (`nCells: 12447`)
* `bench/rheotool/cylinder_wi1.0/log.mirrorMesh:1` (`New cells: 24894`)
* `bench/rheotool/cylinder_{wi1.0,wi0.1,newtonian_re1}/system/controlDict:24`

---

## 2. Shrunk rT setup (this prep)

Target Kraken geometry from `run_cyl_bigsweep_v2_2d.jl:101-102`
(`KRAKEN_L_UP_LIST=15.0`, `KRAKEN_L_DOWN_LIST=15.0`).

| Parameter             | Original                | Shrunk (new)            |
|-----------------------|-------------------------|-------------------------|
| L_up / R              | 20.0                    | **15.0**                |
| L_down / R            | 60.0                    | **15.0**                |
| Channel half-height / R | 2.0                   | 2.0 (unchanged)         |
| Block 0 cells (i × j) | 33 × 40                 | **23 × 40**             |
| Block 7 cells (i × j) | 50 × 60                 | **20 × 60**             |
| Block 7 grading       | 20 (axial)              | **8 (axial)**           |
| Cells (blockMesh)     | 12,447                  | **10,247** (–17.7 %)    |
| Cells (post-mirror)   | 24,894                  | **20,494** (–17.7 %)    |
| endTime (Wi=1.0)      | 10 (non-converged)      | **20** (G1 mitigation)  |
| endTime (Wi=0.1)      | 6                       | **10** (sweep consistency) |
| endTime (Newtonian)   | 1                       | **10** (safety margin)  |
| dt (Wi=1.0)           | 1e-2                    | 1e-2 (unchanged)        |
| dt (Wi=0.1, Newtonian) | 2e-2                   | 2e-2 (unchanged)        |
| BCs (U, p, tau)       | parabolic / zeroGrad / linearExtrapolation | unchanged |
| Stabilisation         | `coupling` (no BSD)     | unchanged               |
| CUBISTA               | yes                     | unchanged               |
| Near-cylinder cell size (block 1 j-min) | 0.0049 R | unchanged (0.0049 R) |
| Sample line probe end | (60, 0, 0.5)            | **(15, 0, 0.5)**        |

**Block 0 (upstream) mesh delta:** Original `simpleGrading 0.12` over length
17.17 with N=33 gives last cell ~0.149 (near cylinder). Shrunk length is 12.17
and N=23, same grading 0.12 → last cell ≈ 0.150 (matched to block 1 face within
1%). Near-wall resolution **preserved**.

**Block 7 (downstream wake) mesh delta:** Original `simpleGrading 20` over
length 57.17 with N=50 gives first cell ~0.178 (near cylinder), last cell
~3.56 (coarse outlet). Shrunk length is 12.17 with N=20 and grading 8 → first
cell ≈ 0.178 (matched), last cell ≈ 1.42 (still expanded but less aggressive
because wake is 4× shorter). Near-wall resolution **preserved**; the only
resolution loss is in the far-wake region we have just removed.

**Blocks 1–6 (cylinder O-grid + immediate vicinity): all unchanged.** No
near-body resolution change.

### Diff summary per case

All three shrunk cases share identical `blockMeshDict` deltas (the original
files are byte-identical across Wi=0.1 / Wi=1.0 / Newtonian):

```
< vertex 0:    (-20 0 0)    -> (-15 0 0)
< vertex 10:   ( 60 0 0)    -> ( 15 0 0)
< vertex 11:   ( 60 2 0)    -> ( 15 2 0)
< vertex 17:   (-20 2 0)    -> (-15 2 0)
< vertex 18:   (-20 0 1)    -> (-15 0 1)
< vertex 28:   ( 60 0 1)    -> ( 15 0 1)
< vertex 29:   ( 60 2 1)    -> ( 15 2 1)
< vertex 35:   (-20 2 1)    -> (-15 2 1)
< block 0:     (33 40 1)    -> (23 40 1)    grading unchanged 0.12
< block 7:     (50 60 1) g=20  -> (20 60 1) g=8
```

`controlDict` deltas:
* `cylinder_wi1.0_shrunk15R/system/controlDict:24` `endTime 10 -> 20`
* `cylinder_wi0.1_shrunk15R/system/controlDict:24` `endTime 6 -> 10`
* `cylinder_newtonian_re1_shrunk15R/system/controlDict:24` `endTime 1 -> 10`

`sampleDict` delta (all three cases):
* `system/sampleDict:45` `end (60 0 0.5) -> end (15 0 0.5)`

**No other dict modified.** `fvSchemes` (CUBISTA + Gauss linear corrected),
`fvSolution` (SIMPLE + PETSc), `constitutiveProperties` (Oldroyd-BLog or
Newtonian, etaS=0.59, etaP=0.41 for VE / etaS=1, etaP=0 for Newtonian),
`mirrorMeshDict` (y-plane mirror at y=0), `petscDict`, `decomposeParDict` are
all identical to originals.

---

## 3. Newtonian existing case audit (per brief step 3)

The existing `bench/rheotool/cylinder_newtonian_re1/system/blockMeshDict` is
**byte-identical** to `cylinder_wi1.0/system/blockMeshDict` (verified via
`diff`: zero output). It therefore has `L_up=20R`, `L_down=60R` — the **same
oversize geometry** flagged by Phase 1. A shrunk copy was therefore created
(`cylinder_newtonian_re1_shrunk15R`) following the brief's traceability
preference (dedicated copy even if it had matched).

The original `cylinder_newtonian_re1` also has `endTime=1` (too short for
robust Cd average even at Re=1), which the shrunk copy lifts to 10.

---

## 4. Run script

**Path:** `bench/rheotool/run_shrunk_matrix.sh`

```bash
bash bench/rheotool/run_shrunk_matrix.sh           # all 3 cases sequential
bash bench/rheotool/run_shrunk_matrix.sh newtonian # only Newtonian
bash bench/rheotool/run_shrunk_matrix.sh wi0.1     # only Wi=0.1
bash bench/rheotool/run_shrunk_matrix.sh wi1.0     # only Wi=1.0
```

Each case invokes its local `run_docker.sh`, which pulls
`guiguitcho/openfoam9-rheotool:v1.2` and runs `./Allclean && ./Allrun`
(`blockMesh` → `mirrorMesh -overwrite` → `rheoFoam`).

### Expected walltime per case (workstation, single core)

| Case                | dt    | Nsteps | Per-step cost vs orig | Walltime (est.) |
|---------------------|-------|--------|-----------------------|-----------------|
| Newtonian (shrunk)  | 2e-2  | 500    | 0.82 × (10247/12447)  | ~5–15 min       |
| Wi=0.1 (shrunk)     | 2e-2  | 500    | 0.82 ×                | ~15–40 min      |
| Wi=1.0 (shrunk)     | 1e-2  | 2000   | 0.82 ×                | ~1.5–2.5 h      |
| **Matrix total**    | —     | —      | —                     | **~2–3 h**      |

Reference: original Wi=1.0 (endTime=10, 1000 steps, 12447 cells, no
near-cylinder resolution change) reportedly took ~2–3 h. The shrunk case
doubles `endTime` but loses ~18 % of cells, so per-step cost ↓ ~18 %, total
≈ 2 × 0.82 ≈ 1.64 × the original Wi=1.0 walltime ≈ 3–5 h worst case if
docker overhead dominates. Wi=0.1 and Newtonian are at half the step count
and lower polymer cost → considerably faster.

### Aqua / Apptainer

**rheoTool is NOT available as an Apptainer SIF on Aqua.** Searched
`~/Documents/Clouds/UGA/Recherche/HPC/aqua/` recursively for
`*rheotool*`/`*rheofoam*` references — zero hits. The Aqua OpenFOAM module
ships vanilla OpenFOAM (no rheoTool extension). Building a SIF from the
existing Docker image `guiguitcho/openfoam9-rheotool:v1.2` is feasible via
`docker-to-sif.sh` if HPC throughput becomes necessary, but not pursued in
this Phase 2 prep. **Decision:** local Docker only.

---

## 5. Files created

* `bench/rheotool/cylinder_wi1.0_shrunk15R/`
  * `system/blockMeshDict` (edited: vertices + block 0 + block 7)
  * `system/controlDict` (edited: `endTime 10 -> 20`)
  * `system/sampleDict` (edited: probe end `60 -> 15`)
  * `system/{fvSchemes,fvSolution,mirrorMeshDict,petscDict,decomposeParDict}` (copies)
  * `constant/constitutiveProperties` (copy: Oldroyd-BLog, λ=1.0, etaS=0.59, etaP=0.41)
  * `0/{U,p,tau,theta}` (copies)
  * `Allrun`, `Allclean`, `writeData`, `summarize_cd.sh`, `run_docker.sh`

* `bench/rheotool/cylinder_wi0.1_shrunk15R/`
  * `system/blockMeshDict` (edited)
  * `system/controlDict` (edited: `endTime 6 -> 10`)
  * `system/sampleDict` (edited)
  * other dicts as copies
  * `constant/constitutiveProperties` (Oldroyd-BLog, λ=0.1)
  * `0/{U,p,tau,theta}` (copies)
  * run scripts as above

* `bench/rheotool/cylinder_newtonian_re1_shrunk15R/`
  * `system/blockMeshDict` (edited)
  * `system/controlDict` (edited: `endTime 1 -> 10`)
  * `system/sampleDict` (edited)
  * other dicts as copies
  * `constant/constitutiveProperties` (Newtonian)
  * `0/{U,p,tau,theta}` (copies)
  * run scripts as above

* `bench/rheotool/run_shrunk_matrix.sh` (new matrix driver, executable)
* `bench/viscoelastic_audit/M32_PHASE2_RHEOTOOL_PREP_VERDICT.md` (this file)

`constant/polyMesh/`, time-output dirs (`5/`, `10/`), `dynamicCode/`,
`log.*`, `Cd.txt` are **deliberately not copied** — they are regenerated by
the run pipeline. No commits, no execution performed.

---

## 6. Sanity flags and risks

* **F1. Block 7 grading 8 vs 20 → wake mesh stretched less aggressively.**
  This may slightly oversample the very far wake (no impact on Cd) at the
  cost of 2 extra cells per row vs strict proportional reduction. Acceptable.

* **F2. The `parabolicMeanU1` codedFixedValue inlet in `0/U` is mesh-agnostic
  and recompiles on first run** — no edit needed.

* **F3. `mirrorMesh` operates on y-plane (point (0,0,2.5), normal (0,-1,0))
  in `system/mirrorMeshDict`** and is independent of x-domain length.
  No edit needed.

* **F4. The original Wi=1.0 case's outlet x=60R is far enough that polymer
  wake fully relaxes before reaching it; the new x=15R may bias outlet
  stress.** This is the **whole point of M32 Phase 2** — to expose whether
  Kraken's L_down=15R is or is not adequate for Wi=1.0. Wi=0.1 is much
  less sensitive (relaxation distance ≪ L_down).

* **F5. Cell-count balance:** the shrunk mesh keeps the near-cylinder
  resolution and only sheds cells from the upstream and far-wake regions.
  This means the Newtonian shrunk case will have **drag** comparable to the
  original Newtonian case (drag is dominated by boundary-layer shear, which
  is preserved). If Newtonian Cd_shrunk and Cd_original differ by >2 %, the
  shrink is biased — but this is **exactly the G3 sanity test** the Boss
  intends to run.

---

## 7. Report format (for Boss)

```
## M32 Phase 2 rheoTool-side prep — verdict

### Original rT setup (per audit)
- L_up=20R, L_down=60R, blockage 0.5, 24,894 cells post-mirror

### Shrunk rT setup (this prep)
- L_up=15R, L_down=15R, blockage 0.5 (all 3 cases identical geometry)
- Cells before: 24,894 ; after: 20,494 (-17.7 %)
- endTime: 10 -> 20 (Wi=1.0, per Phase 1 G1 flag); 6 -> 10 (Wi=0.1); 1 -> 10 (Newtonian)

### Newtonian existing case audit
- `cylinder_newtonian_re1/`: identical blockMeshDict to wi1.0 (L_up=20R, L_down=60R)
- Shrunk copy created (`cylinder_newtonian_re1_shrunk15R`)

### Run script
- `bench/rheotool/run_shrunk_matrix.sh [all|newtonian|wi0.1|wi1.0]`
- Expected walltime per case: Newtonian ~5-15 min, Wi=0.1 ~15-40 min, Wi=1.0 ~1.5-2.5h
- Target: local Docker (Apptainer SIF for rT not available on Aqua)

### Files
- bench/rheotool/cylinder_wi1.0_shrunk15R/
- bench/rheotool/cylinder_wi0.1_shrunk15R/
- bench/rheotool/cylinder_newtonian_re1_shrunk15R/
- bench/rheotool/run_shrunk_matrix.sh
- bench/viscoelastic_audit/M32_PHASE2_RHEOTOOL_PREP_VERDICT.md
```
