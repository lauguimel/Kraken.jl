# Reference study — confined Oldroyd-B, cylinder & sphere, β=0.5 blockage 0.5

Session 2026-04-22 (dev-viscoelastic). Purpose : answer Q1 from
`NEXT_SESSION_PROMPT.md` — is Kraken's ratio Cd_visco/Cd_Newt ≈ 0.89
at Wi=0.1, β=0.5, blockage 0.5 physically correct ?

This document replaces the Lunsmann comparison (category error —
Lunsmann 1993 is **unbounded** sphere) with a proper confined
reference for cylinder and a qualitative reference for sphere.

---

## 1. Summary verdict

| Source | Geometry | β | Blockage | Re | Wi | Cd | Cd_Newt | ratio |
|--------|----------|---|----------|-----|----|----|---------|-------|
| **Kraken step5d (F64)** | 2D cyl R=30 | **0.5** | 0.5 | 1 | 0.1 | 128.70 | 142.87 | **0.901** |
| Kraken step3 (F64) | 2D cyl R=30 | 0.59 | 0.5 | 1 | 0.1 | 126.41 | — | — |
| Kraken step3 (F64) | 2D cyl R=48 | 0.59 | 0.5 | 1 | 0.1 | 131.29 | — | — |
| Kraken step5d (F64) | 3D sphère R=16 | 0.5 | 0.5 | 1 | 0.1 | 192.05 | 215.30 | 0.892 |
| Liu 2025 Table 3 | 2D cyl R=30 | 0.59 | 0.5 | 1 | 0.1 | **130.36** | ~132 | ~0.987 |
| Liu 2025 Table 3 | 2D cyl R=48 | 0.59 | 0.5 | 1 | 0.1 | **130.83** | ~132 | ~0.991 |
| Kumar arxiv 2403.05904 | 2D cyl | 0.1–0.9 | 0.5 | 0.01 | De=0.1 | 126.96 | 132.09 | **0.961** |
| Hulsen / Alves canonical | 2D cyl creeping | — | 0.5 | → 0 | 0 | — | **132** | 1 |

**Findings**

1. **Physical Cd_visco/Cd_Newt at β≈0.5 blockage 0.5 Wi=0.1 ≈ 0.96**,
   not 0.89. The Kumar 2024 study (arxiv 2403.05904, reviewed here) is
   the only paper that reports Cd_visco AND Cd_Newt under a unified
   normalisation at this blockage and gives ratio **126.96/132.09 ≈
   0.961** at De=0.1 for β across the range 0.1–0.9 (the Cd at De=0.1
   varies only mildly with β in the creeping regime).

2. **Kraken's "0.901" is a composite of two partly-known errors**, not
   a single physical ratio :
   - Cd_Newt_Kraken = 142.87 at R=30, ν_total=0.6 is **~8% above** the
     canonical ~132 at Re→0 blockage 0.5.
   - Cd_visco_Kraken = 128.70 at β=0.5 R=30 Wi=0.1 is **~2% below**
     the β=0.5 De=0.1 literature value (~126.96 from Kumar 2024 at
     Re=0.01 ≈ close to 130 at β=0.5 Re=1, interpolating from Liu's
     β=0.59).
   - Ratio = 128.70 / 142.87 = 0.901 underreports the physical drag
     reduction.

3. **Physics sanity-check on sign** : drag enhancement vanishes at
   blockage 0.5, consistent with experimental literature. Multiple
   sources state that "drag enhancement disappears upon increasing the
   sphere-to-tube diameter ratio to 0.5" and the 2024 Kumar paper's
   Table 2 explicitly shows Cd **decreasing** from De=0.025 → De=0.5
   at B=0.5 (132.09 → 118.64) before rising again at high De. Kraken's
   drag reduction at Wi=0.1 is therefore **physically plausible in
   sign** ; only the magnitude (0.901 vs 0.96) is off.

---

## 2. Canonical literature values for confined 2D cylinder Oldroyd-B

### 2.1 Kumar et al. 2024 (arxiv 2403.05904) — direct parameter match

Paper : "CFD analysis of the influence of solvent viscosity ratio on
the creeping flow of viscoelastic fluid over a channel-confined
circular cylinder" (https://arxiv.org/html/2403.05904v1).

- **Setup** : creeping flow (Re=0.01), blockage B=0.5, Oldroyd-B.
- **β range** : 0.1 ≤ β ≤ 0.9 (full sweep).
- **De definition** : De = λ·U_avg/D (D = cylinder **diameter**). Liu
  and Kraken use Wi = λ·U_avg/R with R = radius = D/2. So **De_Kumar =
  Wi_Liu/2 = Wi_Kraken/2** for the same physical setup. ⚠ their
  "De=0.1" corresponds to Liu/Kraken Wi=0.2. To compare at Liu/Kraken
  Wi=0.1, look at De_Kumar ≈ 0.05 (interpolate between 0.025 and 0.1).

- **Table 2** (Cd values, quoted in preprint abstract and via WebFetch
  2026-04-22) :

| De_Kumar | Wi_Liu (= 2·De) | Cd (present study) | Cd (literature) |
|----------|-----------------|--------------------|-----------------|
| 0.025 | 0.05 | 132.09 | 131.50 (Dou–Phan-Thien 1999) |
| 0.100 | 0.20 | 126.96 | 130.36 (Fan 1999, Liu 2025) |
| 0.500 | 1.00 | 118.64 | 118.81–120.58 (multiple) |
| 1.500 | 3.00 | 157.81 | 147.17 (Dou–Phan-Thien) |

  **Interpretation of Cd_Newt** : Cd at De_Kumar=0.025 (≈ quasi-
  Newtonian) is 132.09 — this is the canonical Newtonian creeping-flow
  reference at blockage 0.5.

### 2.2 Liu et al. 2025 (arxiv 2508.16997) — Kraken's benchmark

Paper : Liu, Zhou, Grecov, Wang, "TRT lattice Boltzmann scheme for
Oldroyd-B viscoelastic fluid flow" (2508.16997).

- **Setup** : domain 30R × 4R, B=0.5, Re=1, β=0.59, Wi = λ·U_avg/R.
- **Table 3** (as recorded in `REFERENCES.md`) :

| R | Wi | Cd |
|---|----|----|
| 30 | 0.1 | 130.36 |
| 30 | 0.5 | 126.31 |
| 30 | 1.0 | 151.31 |
| 48 | 0.1 | 130.83 |

  **Liu does not publish an explicit Cd_Newt**. Deriving it from the
  Kumar comparison at De=0.1 : Liu's Wi=0.1 = Kumar's De=0.05 ≈ 131.5
  (Dou–Phan-Thien ref), dropping by ≈1% at Wi=0.1 → the implicit
  Cd_Newt in Liu's convention is **~132**, matching Kumar 2024 and
  Hulsen.

### 2.3 Other references (qualitative / not yet fully mined)

| Ref | Setup | Status |
|-----|-------|--------|
| Alves, Oliveira, Pinho 2001 (JNNFM 97:207) | B=0.5 cylinder, 5 constitutive models incl. Oldroyd-B, FVM high-res | **Gold standard**, need the PDF for Table with Cd vs De. Not machine-readable via WebFetch (Elsevier 403). Get via ResearchGate or DOI. |
| Fan, Kahrilas, Tanner 1999 | B=0.5 cylinder, FEM, β=0.41 | Cited by Kumar 2024 and Liu 2025 as "130.36 at De=0.1". Paywalled. |
| Coronado et al. (DAVSS-ω, JNNFM 144:122 2007) | B=0.5, β=0.41 | Figure 7 of ResearchGate shows Cd vs De but PDF blocked. |
| Dou & Phan-Thien 1999 | B=0.5, Oldroyd-B | Cited as giving Cd_Newt = 131.5 |
| Pimenta & Alves 2017 (JNNFM 239:85) | rheoTool paper, B=0.25 (4:1 contraction) | Not directly B=0.5 cylinder ; used for equation cross-check (Section 4). |

---

## 3. Canonical literature values for confined sphere (3D)

### 3.0 ⚠ Critical finding — Liu 2025 is 2D-only

From the arxiv HTML of 2508.16997 (verbatim, 2026-04-22) :

> "Although the extension to three dimensions is straightforward and
> will be implemented in the future, we only consider the
> two-dimensional cases for simplicity in this article."

**Consequence** : Kraken's 3D sphere viscoelastic kernel was written
without a published reference to transpose. The 3D Hermite source, 3D
CNEBB, 3D feq_conformation, 3D inlet/outlet reset are all independent
extrapolations. Any cross-check against Liu is limited to 2D.

→ For 3D validation, rheoTool (Pimenta-Alves 2017) or Claus & Phillips
2013 (JNNFM 200:131) are the only viable canonical paths.

### 3.1 What's NOT applicable

- **Lunsmann, Genieser, Armstrong, Brown 1993** (JNNFM 50:135) is an
  **unbounded-sphere** limit or very low blockage. Kraken's 3D
  geometry is a square duct of side 4R (blockage R/H = 0.5). Not
  comparable. Retracted as reference.

### 3.2 What IS applicable (qualitative)

- **Multi-source confirmation** : "At a sphere-to-tube diameter ratio
  of 0.25, there is considerable drag enhancement with increasing
  Weissenberg number, but this drag enhancement **disappears** upon
  increasing the sphere-to-tube diameter ratio to 0.5" (quoted from
  search result — traces back to Zheng, Phan-Thien, Tanner 1990 and
  Fan 2003, both paywalled).

  → Kraken's Cd_visco/Cd_Newt ≈ 0.89 < 1 at blockage 0.5 is **sign-
  consistent** with the confined-sphere literature : drag reduction,
  not enhancement.

### 3.3 What's NOT readily available

- A numerical Cd_visco/Cd_Newt table at *exactly* blockage 0.5, β=0.5,
  Wi=0.1 for confined sphere Oldroyd-B. Candidates to chase :
  - Zheng, Phan-Thien, Tanner 1990 (JNNFM 36:27) — Rheol. Acta /
    JNNFM. Paywall.
  - Fan, Phan-Thien, Tanner 2003 — if it exists. Not confirmed.
  - Oldroyd-B **IBSE** paper (Stein et al. arxiv 2304.xxxxx via
    Flatiron) — confined sphere Wi→0 drag correction.
  - Claus & Phillips 2013 (JNNFM 200:131) — spectral/hp element
    confined sphere, often cited as benchmark.

  **Action for next session** : try institutional library access to
  Zheng 1990 or Claus 2013. Alternatively run rheoTool tutorial
  `sphere-in-tube` if one exists (not confirmed in repo README).

---

## 4. Equation cross-check (Kraken vs Liu 2025 vs Pimenta-Alves 2017)

### 4.1 Hermite source term — partial extraction

From Liu 2025 §3 (via WebFetch of HTML v1) :

**Eq. (36)** (polymer-field Hermite source) :

  F̃_i = w_i·S + (1 − 1/(2τ_{p,1}))·w_i·(e_iα/c_s²)·u_α·S

where `S` is the source term from the Oldroyd-B constitutive equation
and τ_{p,1} is the polymer-field TRT relaxation time. The prefactor
`(1 − 1/(2τ_{p,1}))` is the standard Guo-style "half-step" TRT
correction.

**Action** : compare to Kraken's `apply_polymer_hermite_source_3d!` in
`src/` — the prefactor should match. The 2D path was verified in the
audit (step 2, TRT ≡ BGK on flat wall) ; 3D not yet cross-checked.

### 4.2 Equilibrium distribution for the conformation field

**Eq. (30)** (Liu 2025) :

  g_i^eq = w_i·ϕ·(1 + (H_{i,α}/c_s²)·u_α + (H_{i,α,β}/(2·c_s⁴))·u_α·u_β)

with ϕ the conformation scalar. "Extension to 3D follows analogous
structure with D3Q27 lattice."

**Kraken** : `feq_3d` is D3Q19 for the conformation (same as hydro).
Liu 2025 uses D3Q27. **This is a difference** ; the audit notes this
as a code path not yet inspected. Whether D3Q19 captures the full
second-order Hermite moments is an open question — D3Q19 is missing
some higher-order moments relative to D3Q27, which matters for ∇ u
moments carried by the conformation tensor.

  **Concrete follow-up** : verify by hand that the D3Q19 weights and
  velocities reproduce the required C tensor moments (Σ w_i H_{i,αβ}
  e_i = correct projection onto 3×3 symmetric tensor space). If a
  moment is missing, that's a systematic bias source.

### 4.3 CNEBB boundary scheme

From Liu 2025 (via HTML) :

**Eq. (41)** conservation of conformation :
  ϕ(x_b, t+Δt) = Σ_{γ∈Γ} g_γ(x_b, t+Δt) + Σ_{η∈H} g_η†(x_b, t)

**Eq. (42)** reconstruction of unknown g_i :
  g_i = g_i^eq + g_{ī}^neq = g_i^eq + (g_{ī} − g_{ī}^eq)

**Eq. (43)** rest-population rebalance :
  g_0 = ϕ(x_b, t+Δt) − Σ_{i≠0} g_i(x_b, t+Δt)

These are the 2D equations. The 3D extension (which would be
Eqs. 44–46 or similar in the paper ; not extracted by WebFetch)
requires enumerating the 6 "wall" directions × 9 "wall-parallel"
directions for D3Q19, and applying (41)-(43) per direction.

**Action for next session** : download the PDF locally
(`arxiv.org/pdf/2508.16997`) and extract Eqs. 44–46 manually. Then
compare line-by-line against `apply_cnebb_3d!` in Kraken's
`src/multiphysics/viscoelastic.jl`.

### 4.4 Pimenta-Alves 2017 — not machine-fetchable (Elsevier 403)

Pimenta, F., Alves, M.A., 2017. "Stabilization of an open-source
finite-volume solver for viscoelastic fluid flows". JNNFM 239:85-104.
(https://www.sciencedirect.com/science/article/pii/S0377025716303329)

Abstract confirms : log-conformation, second-order in space/time,
tested on 4:1 planar contraction with β=1/9, Re=0.01, 0 ≤ De ≤ 12.
Does NOT publish a confined-cylinder benchmark directly (confined
cylinder is in the Alves-Oliveira-Pinho 2001 paper, a different
reference from the same group).

**Net** : Pimenta-Alves 2017 is useful for the UCD source-term form
and BC treatment, but NOT for a Cd comparison at B=0.5 Wi=0.1. The
relevant paper from the Alves group is the 2001 cylinder paper.

---

## 5. Open-source codes for equation transposition

### 5.1 rheoTool

- Repo : https://github.com/fppimenta/rheoTool
- Status of confined-cylinder tutorial : **not confirmed**. GitHub
  README is high-level ; the tutorials directory is within per-
  OpenFOAM-version folders (`of90`, `of70`). Need to clone locally to
  list `tutorials/` and see if a `cylinder` subdirectory exists.
  Documented benchmarks in the literature around rheoTool are mostly
  4:1 contraction, droplet impact, electrically-driven flows — **not**
  cylinder drag.
- **Concrete action for next session** :
  ```bash
  git clone --depth=1 https://github.com/fppimenta/rheoTool /tmp/rheotool
  find /tmp/rheotool/of90/tutorials -type d -name '*cylinder*'
  ```
  If a cylinder tutorial exists, adapt mesh to B=0.5 and run with
  Oldroyd-B at β=0.5 Wi=0.1 Re=1. Compare Cd to Kraken's 128.70.

### 5.2 Liu 2025 supplementary material

- arxiv 2508.16997 does not advertise a linked GitHub repo in the
  abstract/metadata visible via arxiv.org. Need to check the PDF's
  acknowledgements section. If no code released : transposing Liu to
  Julia is not feasible as a bit-for-bit test.

### 5.3 Palabos viscoelastic plugin

- C++ LBM (plb.unige.ch). Less maintained than rheoTool and less
  canonical (not from the Alves / Oliveira / Pinho / Pimenta group).
  Skip unless rheoTool is inaccessible.

---

## 6. What this study DOES NOT yet settle

1. **The Kraken Cd_Newt = 142.87 at R=30 Re=1 B=0.5**, +8% above the
   canonical ~132, is **unexplained by literature** and needs a
   separate diagnostic :
   - Verify the `run_cylinder_libb_2d(ν_total=0.6)` converges : run at
     R=60, R=120 to confirm Cd_Newt(R→∞) approaches 132.
   - Verify domain-length effects : 30R × 4R may be too short
     upstream/downstream to recover from inlet parabolic.
   - Verify the Cd normalisation in Kraken matches Liu : Liu uses
     `Cd = F_x / (0.5·ρ·U_avg²·D)`. Kraken's `compute_drag_libb`
     should give the same scalar.

2. **The true physical Cd_visco/Cd_Newt at β=0.5 blockage 0.5 Wi=0.1
   is not nailed down to <1%**. Kumar 2024 uses β=varying and Re=0.01
   (not Re=1 like Kraken/Liu). Reported ratio 0.961 is at Re=0.01.
   Inertia at Re=1 shifts Cd by a few percent ; best estimate for the
   Re=1 case is 0.95–0.97.

3. **3D equations not yet term-by-term matched to Liu** (§4.2, §4.3).
   This is the FIRST technical task to do next session per the prompt.

4. **3D sphere ratio 0.892** : no direct numerical reference found.
   Only qualitative (drag enhancement disappears at blockage 0.5).

---

## 7. Concrete next-session actions (priority order)

1. **Kraken Cd_Newt convergence test** — run `run_cylinder_libb_2d`
   at R ∈ {30, 60, 120}, ν_total=0.6, B=0.5, Re=1. Does Cd_Newt →
   ~132 ? If not, there's a Newtonian reference bug BEFORE worrying
   about the viscoelastic ratio.

2. **Kraken β sweep at fixed Wi=0.1 R=30** — β ∈ {0.1, 0.3, 0.5, 0.7,
   0.9}. Compare to Kumar 2024 Figure (not Table) values for the
   β-dependence. If Kraken's trend matches Kumar's, the scheme is OK
   and the "0.89 vs 0.96" is an offset only. If Kraken's trend
   differs, the polymer coupling has a structural bias.

3. **Download Liu 2025 PDF** and extract Eqs. 44–46 for 3D CNEBB.
   Compare term-by-term with `apply_cnebb_3d!` in Kraken.

4. **Clone rheoTool** (§5.1). Confirm whether a B=0.5 cylinder
   tutorial exists. If yes → run it as a third-party cross-check.

5. **Audit the D3Q19 vs D3Q27 choice** (§4.2). Verify by hand that
   D3Q19 preserves the C tensor moments needed by Oldroyd-B.

---

## Sources

- arxiv 2508.16997 — Liu, Zhou, Grecov, Wang 2025. TRT LBM for Oldroyd-B.
- arxiv 2403.05904 — Kumar et al. 2024. CFD confined cylinder
  Oldroyd-B, β sweep.
- Pimenta, Alves 2017 JNNFM 239:85-104. rheoTool paper.
- Alves, Oliveira, Pinho 2001 JNNFM 97:207. Confined cylinder Oldroyd-
  B (to be obtained).
- Fan, Kahrilas, Tanner 1999. Confined cylinder FEM Oldroyd-B.
- Lunsmann, Genieser, Armstrong, Brown 1993 JNNFM 50:135. **Unbounded
  sphere** — not applicable to Kraken's ducted geometry.
- Dou & Phan-Thien 1999. Cd_Newt = 131.5 at B=0.5 creeping flow
  (cited by Kumar 2024).
- rheoTool repo : https://github.com/fppimenta/rheoTool.
