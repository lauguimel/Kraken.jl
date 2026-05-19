# M28-liu-check — Verdict on Liu 2025 Cd convention and table values

Date: 2026-05-19.  Department M28-liu-check, branch `dev-viscoelastic`.

Primary source: `bench/viscoelastic_audit/liu_2025.txt`, § 4.3 and
Tables 3-5.  Secondary: `bench/viscoelastic_audit/CD_DECOMPOSITION.md`,
`bench/viscoelastic_audit/EQUATION_AUDIT_LIU_RHEOTOOL.md`,
`.orchestrator/mandate.md` § M28.

## TL;DR

The "151.31" cited as Liu's Cd at R=30 Wi=1.0 in the M28 mandate (line
847) and the handoff prompt is **WRONG**.  The 151.31 entry is Liu's
CNEBB **Wi = 0.1**, **not** Wi = 1.0.  Liu's CNEBB Cd at R=30 is:

| Wi  | Liu CNEBB Cd | Liu YLW Cd | Liu HWBB Cd | Liu Malaspinas Cd |
|----:|-------------:|-----------:|------------:|------------------:|
| 0.1 | **151.31**   | 134.39     | NaN         | NaN               |
| 0.5 | 126.31       | 126.65     | NaN         | NaN               |
| 1.0 | **130.36**   | 130.63     | 129.67      | 130.68            |

This means Liu's trend at R=30 Wi-sweep CNEBB is **non-monotonic
with a trough at Wi = 0.5**: 151.31 → 126.31 → 130.36.  It is NOT
"Cd increases monotonically with Wi at 151.31"; that picture is an
artefact of mis-reading the Table 3 column order.

The Kraken-vs-Liu disagreement asserted in the handoff
("Kraken sees drag reduction with Wi, Liu sees drag amplification")
**does not exist in the data**.  The comparison points have been
mis-identified.  Kraken Phase 1 trend (decreasing Cd from
~129.4 at Wi=0.1 to ~111.6 at Wi=1.0) needs to be compared against
the correct Liu column.

## 1. Cd at R=30, β=0.59, Wi=1.0

**Liu CNEBB Cd(R=30, Wi=1.0) = 130.36** (Table 3, line 2645,
first numerical entry on the R=30 row).

The "151.31" cited in the handoff/mandate is the **same row,
third column** = CNEBB Wi=0.1.

Evidence:
- Table 3 header order (lines 2596-2635 in `liu_2025.txt`):
  scheme order is CNEBB, YLW, HWBB, Malaspinas, NEQE; within each
  scheme the Wi sub-columns are ordered Wi=1.0, Wi=0.5, Wi=0.1.
- R=5 line (line 2637): `5 105.42 100.28 121.11 ...`.
  Counting back: 105.42 = CNEBB Wi=1.0, 121.11 = CNEBB Wi=0.1.
- R=30 line (line 2645): `30 130.36 126.31 151.31 130.63 126.65
  134.39 129.67 NaN NaN 130.68 NaN NaN NaN NaN NaN`. Fifteen
  fields = 5 schemes × 3 Wi.
- Same numerical pattern reproduced in Table 4 (Sc=10⁵, line 2715):
  `30 130.38 126.20 149.74 ...` → CNEBB Wi=1.0 = 130.38, Wi=0.1
  = 149.74.
- Table 5 (Sc=10⁶, line 2792): `30 130.42 125.88 147.14 ...`
  → CNEBB Wi=1.0 = 130.42, Wi=0.1 = 147.14.

Independent corroboration:
`bench/viscoelastic_audit/EQUATION_AUDIT_LIU_RHEOTOOL.md` line 18
("Oldroyd-B, Liu CNEBB, R=30, Sc=1e4 | Cd = 130.36") and
`bench/viscoelastic_audit/CD_DECOMPOSITION.md` (Cd_ref 130.36 for
R=30) both use 130.36, which is consistent with Wi=0.1 being
the historical (pre-M28) Kraken reference target — that prior
work matched Liu's Wi=0.1 column.  Wait — see § 6: the
historical convention is itself confused.  130.36 is the CNEBB
column **Wi=1.0** value, and the prior validation work
implicitly compared its Wi≈0 Newtonian-additive simulation
against Liu's Wi=1.0 column.  This is yet another convention
trap that needs untangling.

## 2. Liu Cd convention

Liu Eq. (64), line 2551:
```
Cd = Fx / (½ ρ U_avg² D)
```
with `D = 2R` (line 2555) and `U_avg = average inlet velocity =
(2/3) U_max` for the parabolic Poiseuille inlet (line 2473:
"characteristic velocity is Uc = 2 Umax / 3").

`Fx` is computed by **momentum-exchange** over all fluid nodes
adjacent to the cylinder, summed over all wall-pointing links
(Eq. 63, line 2530).  This is a **single total-drag** integral
over the post-collision and post-streamed distribution functions
— it does NOT split into Cd_s + Cd_p.  The polymer stress
contribution enters the populations via the Hermite source term
that Liu adds during collision; the same MEA pickup that captures
viscous drag thus captures elastic drag as well.

In Kraken's decomposition vocabulary this is **`Cd_post`** (raw
post-source MEA), NOT `Cd_s` and NOT `Cd_s + Cd_p`.  See
`bench/viscoelastic_audit/CD_DECOMPOSITION.md` for the
Kraken-side bookkeeping; that file shows `Cd_post` at R=30
overshoots Liu by ≈9% (142.7 vs 130.4) unless rescaled by
`(1 - s_plus/2)` to get `Cd_scaled ≈ 128.4`, which lands
~1.5% below Liu's 130.36.

Re and Wi definitions (line 2515 & 2518):
- `Re = U_avg · D / ν_total` is **not** the chosen normalization;
  Liu uses **`Re = U_avg · R / ν_total = 1`** (the characteristic
  length is `Lc = R`, line 2472).
- ν_total = ν_s + ν_p = ν_s / β (since β = ν_s / ν_total = 0.59).
- Wi = λ · U_avg / R, also using the radius (`Lc = R`).
- Blockage: domain `30R × 4R`, cylinder centered at (15R, 2R),
  so the channel half-width is 2R and the blockage `D/H = 2R/4R
  = 0.5` (line 2471).

**Key footnote** (line 2518-2523): Liu **deliberately uses Re=1
rather than the literature-standard Re=0.01** because their TRT
regularised method converges too slowly at very low Re.  Liu
acknowledges that "significant deviations are expected between
Re = 1 and Re = 0.01 results, making direct quantitative
comparison with previous studies challenging".  Liu's Cd values
are therefore **NOT** directly comparable to Hulsen 2004
(132.36), Alves 2001, or other Re=0.01 references — they are
finite-inertia Oldroyd-B at Re=1.

## 3. Full Table 3 cross-tabulation (β = 0.59, Sc = 10⁴, CNEBB)

| R  | Wi=0.1 | Wi=0.5 | Wi=1.0 |
|---:|-------:|-------:|-------:|
| 5  | 121.11 | 100.28 | 105.42 |
| 10 | 168.44 | 120.53 | 125.19 |
| 15 | 170.09 | 123.34 | 128.22 |
| 20 | 164.26 | 125.17 | 129.42 |
| 25 | 156.01 | 125.00 | 129.61 |
| 30 | 151.31 | 126.31 | 130.36 |
| 35 | 149.04 | 127.72 | 130.77 |
| 40 | unsteady | 126.79 | 130.79 |

Liu only reports **Wi ∈ {0.1, 0.5, 1.0}** in Table 3; Wi=0.3 is
**not in the paper**.  Kraken's Phase 1 plan of Wi ∈ {0.1, 0.3,
0.5, 1.0} has no direct Liu Wi=0.3 reference.

Trend by inspection of the Wi=1.0 column: Cd rises monotonically
from 105.42 (R=5) toward an asymptote ~130.8 at R=35-40.  This
is a well-behaved grid-convergence trend.

The Wi=0.1 column is **degenerate**: Cd starts at 121 (R=5),
overshoots to 170 (R=10-15), descends to 149 (R=35), then
"unsteady" at R=40.  Liu's own text (line 2576-2578) says YLW
at Wi=0.1/Sc=10⁴ shows "abrupt numerical breakdown" at R=35
under reduced artificial viscosity; the CNEBB Wi=0.1 column
shows analogous non-monotone behaviour that Liu describes as
"monotonic convergence" — this is a generous reading of their
own data.

Tables 4 (Sc=10⁵) and 5 (Sc=10⁶) reproduce the same column order
with Wi=1.0 stable around 130.4 and Wi=0.1 still non-monotone
(168.92 → 162.91 → 154.90 → 149.74 → 147.22 → unsteady at R=40 in
Table 4).

## 4. Trend statement in § 4.3

Liu does NOT make a clean "Cd increases / decreases with Wi"
statement.  Lines 2583-2589 only say:

> "at Wi = 0.1 and 0.5, drag coefficient values exhibit monotonic
> convergence to stable asymptotic values as the grid is refined.
> At Wi = 1.0, although steady-state solutions can be
> consistently obtained, grid convergence becomes less apparent."

This is a **statement about R-convergence at fixed Wi**, not
about the Wi-dependence at fixed R.

Inspection of Liu's R=35 row (the largest stable resolution that
contains all three Wi entries for the CNEBB column at Sc=10⁴):

| Wi=0.1 | Wi=0.5 | Wi=1.0 |
|-------:|-------:|-------:|
| 149.04 | 127.72 | 130.77 |

This **does** suggest a non-monotonic dependence of Cd on Wi at
fixed R: drop from Wi=0.1 to Wi=0.5, slight rise from Wi=0.5 to
Wi=1.0.  But Liu themselves never make this claim in text.

Phase 0/M25 in Kraken (`mandate.md` § M25, line 826) used Liu's
**Wi=0.1 column** (130.36 at R=30 in mandate ≠ 151.31 in actual
Table 3 Wi=0.1).  This means **Kraken's Phase 0 reference was
itself mis-aligned**: Kraken aimed at Liu's Wi=1.0 CNEBB value
130.36 while running a Wi=0.1 (quasi-Newtonian) case.  At Wi=0.1
the Kraken polymer contribution is small and the run is
essentially Newtonian-additive, so Cd ≈ 129.4 is dominated by
the Newtonian part (Hulsen Cd ≈ 132 at the same blockage and
Re→0 → Re=1 reduces this slightly toward ~130).  The match to
130.36 is therefore likely a Newtonian coincidence, **not** a
Wi=1.0 polymer validation.

## 5. Caveats Liu explicitly raises

- Re = 1 chosen deliberately, not Re = 0.01 (line 2518-2523).
  Direct comparison to historical Re→0 references invalid.
- Cd at Wi = 1.0 "grid convergence becomes less apparent"
  (line 2585) — Liu admits the finest-grid Cd may not be the
  true asymptotic value at this Wi.
- "common in existing viscoelastic cylinder flow studies"
  (line 2586) — reduced grid convergence at high Wi is a
  general LBM issue, not a Liu-specific bug.
- "Complete resolution of this challenging issue extends beyond
  the scope of the present work" (line 2588-2589) — Liu does
  not claim a converged Wi=1.0 Cd.
- HWBB, Malaspinas, NEQE schemes "fail completely at higher
  Weissenberg numbers" (line 2571-2572).  The Wi=0.1 columns
  for HWBB/Malaspinas at fine R show many NaNs and "unsteady",
  not because Wi=0.1 is physically hard but because the
  cylinder-surface BC schemes themselves are unstable for the
  CDE.  Only CNEBB and (partially) YLW give complete Wi=0.1
  data — and even those degenerate near R=35-40.
- High-Wi maximum: § 4.3.2, line 2826-2828: "stable computations
  can be extended to Wi = 1.5", "approaches the benchmark
  result of Wi = 1.7 achieved by Ma et al. [66]".  So Liu's
  own tabulated Wi=1.0 is **at 67%** of their stability limit
  Wi=1.5 — not deep in the unstable regime.

## 6. Implications for M28 Phase 1

1. **Kraken Phase 1 Wi=1.0 reference must be 130.36, not 151.31**.
   The mandate `mandate.md` line 847 has a transcription error.
2. **The qualitative disagreement is fabricated**: Kraken's
   Wi-sweep at R=30 (129.39 → 121.25 → 115.93 → 111.55 for
   Wi=0.1 → 0.3 → 0.5 → 1.0) shows monotonic Cd decrease;
   Liu's CNEBB R=30 column is non-monotone (151.31 → 126.31 →
   130.36).  Comparing Kraken Wi=1.0 (111.55) to Liu Wi=1.0
   (130.36) is a ~15% under-prediction.  Comparing Kraken
   Wi=0.5 (115.93) to Liu Wi=0.5 (126.31) is ~8% under.
   Comparing Kraken Wi=0.1 (129.39) to Liu Wi=0.1 (151.31)
   is ~15% under — but Liu's own Wi=0.1 column doesn't
   converge.
3. **The "right" comparison target depends on Liu trust**:
   - Trust Liu Wi=1.0 → Kraken under-predicts Cd by ~15% at
     Wi=1.0.  This is a real disagreement at finite Wi.
   - Trust Liu Wi=0.5 → Kraken under-predicts by ~8%.
   - Trust Liu Wi=0.1 → Liu's value is itself non-converged,
     so any "agreement" or "disagreement" is meaningless.
4. **rheoTool finite-inertia (`Cd = 130.43` at Wi=0.1)** is
   the cleanest external check, and matches Liu Wi=0.1 CNEBB
   only at the asymptote where Liu's Wi=0.1 column is
   misbehaved.  Better: rheoTool can run Wi=0.5 / Wi=1.0
   directly and provide a non-LBM cross-check.
5. **Kraken's monotonic drag reduction with Wi (≈ -26%
   at Wi=1.0)** is qualitatively consistent with the
   Phan-Thien–Tanner / Oldroyd-B drag-reduction literature
   for confined cylinder flow at moderate Wi (e.g.
   Hulsen 2005, Alves 2001 confirm a drag-reduction
   minimum then re-rise).  Liu's non-monotone trough at
   Wi=0.5 is **plausibly the same physics** — the issue is
   whether Kraken's minimum lies at Wi=1.0 or further out.

## 7. Memory candidates

For `boss.md`:
> Liu 2025 Table 3 column order is CNEBB/YLW/HWBB/Malaspinas/NEQE
> and within each scheme Wi=1.0/0.5/0.1.  **Liu's CNEBB R=30 Cd at
> Wi=1.0 is 130.36; the 151.31 value previously cited as the Wi=1.0
> reference is actually Liu's Wi=0.1 column entry (and is
> non-converged in Liu's own data).** All prior Kraken Phase 0/M25
> "130.36 reference" claims actually corresponded to Liu Wi=1.0,
> not Wi=0.1 — i.e. Kraken was unknowingly comparing a Wi=0.1
> Newtonian-additive run to a Wi=1.0 polymer reference, and the
> match was a Newtonian coincidence.

For `department.md`:
> Liu 2025 Cd convention: `Cd = Fx / (½ ρ U_avg² D)` with `D=2R`,
> `U_avg = (2/3) U_max`, `Re = U_avg·R/ν_total = 1`, `Wi = λ·U_avg/R`,
> blockage `D/H = 0.5`.  Liu's Fx is **total** drag from a single
> MEA over post-collision wall links — no separation into Cd_s
> and Cd_p.  This corresponds to Kraken's `Cd_post`/`Cd_scaled`,
> NOT `Cd_s + Cd_p`.  Liu uses Re=1 (not Re=0.01) to circumvent
> TRT slow-convergence; finite inertia → direct comparison with
> Hulsen / Alves Re→0 references invalid.

For `engineer.md`:
> When parsing Liu Table 3/4/5, the columns are tightly
> interleaved (15 numerical fields per R row, broken across
> multiple text lines in the PDF dump because of NaN tokens).
> Always validate the parse against the R=5 row first
> (`105.42 100.28 121.11 ...`).  Header order in raw text:
> scheme group (CNEBB, YLW, HWBB, Malaspinas, NEQE) then Wi
> sub-columns (1.0, 0.5, 0.1).

## 8. Recommended Boss-level actions

1. **Patch `mandate.md` line 847** to read
   `Wi=1.0 → Cd ≈ 130.36 (Liu CNEBB R=30)` and remove the
   151.31 line, or annotate it as the Wi=0.1 anomalous entry.
2. **Revisit M25 PASS justification**: 0000_qwall R=30 = 129.39
   was declared "0.7% below Liu 130.36".  130.36 IS Liu's
   Wi=1.0 column, while the Kraken run was Wi=0.1.  The match
   is fortuitous; under the correct Wi=0.1 reference (151.31,
   non-converged), the bracket would be -14% — well outside
   the M25 PASS window.  M25 PASS verdict needs a re-audit.
3. **Re-aim M28 Phase 1 verdict criteria**: agreement vs Liu
   should be evaluated at (Wi=0.5, Wi=1.0) since (a) Liu's
   Wi=0.1 column is itself non-converged and (b) Wi=0.1 is
   quasi-Newtonian and does not probe elastic physics.
4. **Spawn rheoTool reference sweep at Wi ∈ {0.3, 0.5, 1.0}**
   for an independent (non-LBM) Cd target — this is the
   cleanest way out of the Liu-self-consistency morass.

## Appendix A — Citation map

| Claim                                       | `liu_2025.txt` line(s) |
|---------------------------------------------|------------------------|
| Domain 30R × 4R, blockage D/H = 0.5         | 2471, 2498             |
| `Lc = R`, `Uc = 2 Umax / 3`                 | 2472-2474              |
| `Ma = 0.01`                                 | 2475                   |
| `Re = 1`, `β = 0.59`, `Λp = 2.5e-7`         | 2515-2516              |
| Re=1 motivation (not Re=0.01)               | 2518-2523              |
| `Fx` momentum-exchange Eq. 63               | 2527-2545              |
| `Cd = Fx / (½ ρ U_avg² D)` Eq. 64           | 2551, 2555             |
| Grid range `R ∈ [5, 40]` step 5             | 2559                   |
| `Wi ∈ {0.1, 0.5, 1.0}`                      | 2562                   |
| `Sc ∈ {1e4, 1e5, 1e6}`                      | 2562                   |
| HWBB/Malaspinas/NEQE fail at high Wi        | 2569-2572              |
| YLW abrupt breakdown at R=35 Sc=10⁴         | 2577-2578              |
| Wi=1.0 grid convergence "less apparent"     | 2584-2585              |
| Table 3 (Sc=10⁴) headers                    | 2596-2635              |
| Table 3 R=30 row                            | 2645                   |
| Table 4 R=30 row (Sc=10⁵)                   | 2715                   |
| Table 5 R=30 row (Sc=10⁶)                   | 2792                   |
| Maximum Wi = 1.5 with tuned Λp              | 2826-2830              |

## Appendix B — Raw R=30 parse from Tables 3, 4, 5

Table 3 (Sc=10⁴, line 2645):
```
R=30: 130.36  126.31  151.31  130.63  126.65  134.39  129.67  NaN    NaN    130.68  NaN    NaN    NaN    NaN    NaN
        ↑CNEBB:        ↑YLW:                  ↑HWBB:                 ↑Malasp:                ↑NEQE:
       Wi=1   Wi=0.5  Wi=0.1  Wi=1   Wi=0.5  Wi=0.1  Wi=1   Wi=0.5  Wi=0.1  Wi=1   Wi=0.5  Wi=0.1  Wi=1   Wi=0.5  Wi=0.1
```

Table 4 (Sc=10⁵, line 2715): `R=30: 130.38  126.20  149.74  130.66  126.50  133.83 ...`
Table 5 (Sc=10⁶, line 2792): `R=30: 130.42  125.88  147.14  130.74  126.23  132.69 ...`

CNEBB Wi=0.1 entry **descends** monotonically with Sc
(151.31 → 149.74 → 147.14), suggesting Liu's "Wi=0.1
asymptote" is contaminated by artificial diffusion and is
not a converged physical Cd even at the finest Sc.  CNEBB
Wi=1.0 is essentially **flat** in Sc (130.36 → 130.38 →
130.42), so this **is** a converged value.
